"""
run_nsfnet.py
=============
Benchmark 案例的三階段 pipeline 執行器（wiring）。支援兩個真實拓撲：
  - nsfnet：14 節點 21 弧，用於 K=5（有 ground truth）
  - usnet ：24 節點 43 弧，用於 K=10（NSFNET 21 弧結構上撐不住 10 對）

  - Stage 1：直接呼叫 data_generator.generate_dataset()，注入 benchmark 網路。
  - Stage 2：以 subprocess 呼叫原始 train_surrogate.py（網路無關，可原樣重用）。
  - Stage 3：import 原始 ga_optimizer 的 run_ga / mc_verify，注入「正確的」網路做 MC 驗證，
             繞過 ga_optimizer.main() 對 K>3 用 generate_random_pairs 重建錯網路、
             以及 float(mean_processing_time) 對 per-arc 陣列會 crash 的兩個問題。

用法：
  python run_nsfnet.py --K 5  --stage all               # 預設 nsfnet
  python run_nsfnet.py --K 5  --ground_truth
  python run_nsfnet.py --K 10 --stage all               # 預設 usnet
  （亦可顯式 --topology nsfnet|usnet 覆蓋預設）
"""

# --- path bootstrap (restructured package) ---
import os as _os, sys as _sys
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('', 'core', 'models', 'instances', 'baselines', 'experiments', 'figures_code'):
    _p = _os.path.join(ROOT, _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
HERE = ROOT   # data/ and figures/ live at the package root after restructuring
CODE = ROOT


import os
# Windows 上 conda 的 MKL 與 PyTorch 各帶一份 OpenMP runtime（libiomp5md.dll），
# 會觸發 "OMP: Error #15 ... already initialized" 而中止。必須在 import numpy/torch 之前設定。
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import sys
import json
import time
import argparse
import subprocess
import itertools
import numpy as np


from benchmark_network import get_network
from data_generator import generate_dataset, presample_network_states
from MC_QP_multi_st import greedy_allocation

DATA_ROOT = os.path.join(HERE, 'data')

# ── 各 K 的預設設定（對照原論文 Table 4 / 6）────────────────────
CONFIG = {
    5:  dict(psi=10000, N_perm=120,   N_d=50,
             d_model=64,  nhead=4, num_layers=3, batch_size=64,  epochs=200, patience=30,
             pop_size=200, n_generations=500,  top_n=10, psi_eval=10000),
    10: dict(psi=5000,  N_perm=30000, N_d=50,
             d_model=64,  nhead=4, num_layers=3, batch_size=256, epochs=200, patience=30,
             pop_size=500, n_generations=1000, top_n=10, psi_eval=10000),
}


def default_topology(K):
    """K=5 -> NSFNET（有 ground truth）；K=10 -> USNET（足以支撐 10 對）。"""
    return 'nsfnet' if K == 5 else 'usnet'


def data_dir_for(topology, K):
    return os.path.join(DATA_ROOT, f'{topology}_K{K}')


# ════════════════════════════════════════════════════════════
# Stage 1: 資料生成（注入 benchmark 網路）
# ════════════════════════════════════════════════════════════
def run_stage1(topology, K, cfg, seed, n_workers=None):
    out = data_dir_for(topology, K)
    os.makedirs(out, exist_ok=True)
    MPs_list, T_list, binom, Cmax, mpt = get_network(topology, K, verbose=True)
    print(f"\n[{topology} Stage 1] K={K} -> {out}")
    t0 = time.perf_counter()
    generate_dataset(
        MPs_list, T_list, binom, Cmax, mpt,
        psi=cfg['psi'], N_perm=cfg['N_perm'], N_d=cfg['N_d'],
        n_workers=n_workers, seed=seed, output_dir=out,
    )
    with open(os.path.join(out, 'timing_stage1.json'), 'w') as f:
        json.dump({'stage1_sec': time.perf_counter() - t0, 'K': K,
                   'topology': topology, 'seed': seed}, f, indent=2)
    return out


# ════════════════════════════════════════════════════════════
# Stage 2: 訓練（subprocess 呼叫原始 train_surrogate.py）
# ════════════════════════════════════════════════════════════
def run_stage2(topology, K, cfg, model_type, device, seed):
    out = data_dir_for(topology, K)
    model_dir = os.path.join(out, 'models')
    os.makedirs(model_dir, exist_ok=True)
    cmd = [
        sys.executable, os.path.join(ROOT, 'core', 'train_surrogate.py'),
        '--data_dir', out, '--model', model_type,
        '--epochs', str(cfg['epochs']), '--batch_size', str(cfg['batch_size']),
        '--lr', '1e-3', '--d_model', str(cfg['d_model']),
        '--nhead', str(cfg['nhead']), '--num_layers', str(cfg['num_layers']),
        '--patience', str(cfg['patience']), '--save_dir', model_dir,
    ]
    if device:
        cmd += ['--device', device]
    print(f"\n[{topology} Stage 2] K={K} model={model_type}\n  {' '.join(cmd)}")
    env = os.environ.copy()
    env['KMP_DUPLICATE_LIB_OK'] = 'TRUE'   # 傳給子行程（train_surrogate.py）
    subprocess.run(cmd, check=True, cwd=CODE, env=env)
    return os.path.join(model_dir, f'{model_type}_best.pt')


# ════════════════════════════════════════════════════════════
# Stage 3: GA + MC 驗證（注入正確 benchmark 網路）
# ════════════════════════════════════════════════════════════
def run_stage3(topology, K, cfg, model_type, device, seed, demand=None):
    import torch
    from ga_optimizer import run_ga, mc_verify
    from models.transformer_surrogate import PermutationTransformer
    from models.mlp_baseline import PermutationMLP

    out = data_dir_for(topology, K)
    model_path = os.path.join(out, 'models', f'{model_type}_best.pt')

    if demand is None:
        meta = np.load(os.path.join(out, 'metadata.npz'), allow_pickle=True)
        d_ranges = meta['d_ranges']
        demand = ((d_ranges[:, 0] + d_ranges[:, 1]) / 2).astype(int)
    demand = np.array(demand, dtype=np.float32)

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    d_max = ckpt['d_max']
    m_args = ckpt.get('args', {})
    if model_type == 'transformer':
        model = PermutationTransformer(
            K=K, d_model=m_args.get('d_model', 64), nhead=m_args.get('nhead', 4),
            num_layers=m_args.get('num_layers', 3), dim_feedforward=m_args.get('d_model', 64) * 2)
    else:
        model = PermutationMLP(K=K)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device).eval()

    # ── GA 搜索 ──
    t_ga = time.perf_counter()
    best_perm, best_fit, _, final_pop, final_fit = run_ga(
        model, K, demand, d_max, device,
        pop_size=cfg['pop_size'], n_generations=cfg['n_generations'], seed=seed)
    ga_time = time.perf_counter() - t_ga

    # ── 取 Top-N 去重 ──
    seen, top = set(), []
    for idx in np.argsort(final_fit)[::-1]:
        pt = tuple(final_pop[idx])
        if pt not in seen:
            seen.add(pt); top.append(final_pop[idx])
            if len(top) >= cfg['top_n']:
                break
    if tuple(best_perm) not in seen:
        top.insert(0, best_perm)
    top = np.array(top)

    # ── MC 驗證：注入正確 benchmark 網路 ──
    MPs_list, T_list, binom, Cmax, mpt = get_network(topology, K)
    t_mc = time.perf_counter()
    mc_results = mc_verify(top, demand.astype(int), MPs_list, T_list,
                           binom, Cmax, mpt, psi=cfg['psi_eval'], seed=seed + 100)
    mc_time = time.perf_counter() - t_mc

    mc_best = max(mc_results, key=lambda r: r['R_joint'])
    print(f"\n[{topology} Stage 3] K={K} model={model_type}")
    print(f"  GA best π={best_perm.tolist()} predicted={best_fit:.4f}")
    print(f"  MC best π={mc_best['perm']} R_joint={mc_best['R_joint']:.4f} CI={mc_best['CI_95']}")

    res_dir = os.path.join(out, 'ga_results')
    os.makedirs(res_dir, exist_ok=True)
    results = dict(topology=topology, K=K, model_type=model_type,
                   demand=demand.astype(int).tolist(),
                   ga_best_perm=best_perm.tolist(), ga_best_predicted=float(best_fit),
                   ga_time_sec=ga_time, mc_verification=mc_results,
                   mc_best_perm=mc_best['perm'], mc_best_R_joint=mc_best['R_joint'],
                   mc_time_sec=mc_time, seed=seed)
    path = os.path.join(res_dir, f'ga_{model_type}.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  [存檔] {path}")
    return results


# ════════════════════════════════════════════════════════════
# Ground truth（K<=7 全枚舉，驗證 GA 最優性；對應論文 Q3）
# ════════════════════════════════════════════════════════════
def run_ground_truth(topology, K, cfg, seed, demand=None):
    assert K <= 7, "全枚舉僅適用於小 K"
    out = data_dir_for(topology, K)
    MPs_list, T_list, binom, Cmax, mpt = get_network(topology, K)
    caps, lts = presample_network_states(binom, Cmax, mpt, cfg['psi_eval'], seed=seed + 100)
    if demand is None:
        meta = np.load(os.path.join(out, 'metadata.npz'), allow_pickle=True)
        d_ranges = meta['d_ranges']
        demand = ((d_ranges[:, 0] + d_ranges[:, 1]) / 2).astype(int)
    demand = np.array(demand)

    perms = list(itertools.permutations(range(K)))
    R = np.zeros(len(perms))
    for pi, perm in enumerate(perms):
        D = np.stack([greedy_allocation(caps[t], lts[t], MPs_list, T_list, list(perm))
                      for t in range(cfg['psi_eval'])])
        R[pi] = np.mean(np.all(D >= demand, axis=1))
    order = np.argsort(R)[::-1]
    gt = dict(topology=topology, K=K, demand=demand.tolist(), psi=cfg['psi_eval'],
              best_perm=list(perms[order[0]]), best_R=float(R[order[0]]),
              worst_perm=list(perms[order[-1]]), worst_R=float(R[order[-1]]),
              ratio=float(R[order[0]] / R[order[-1]]) if R[order[-1]] > 0 else None,
              all_R=R.tolist(), all_perms=[list(p) for p in perms])
    path = os.path.join(out, 'ground_truth.json')
    os.makedirs(out, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(gt, f, indent=2)
    print(f"[Ground truth] best={gt['best_R']:.4f}@{gt['best_perm']} "
          f"worst={gt['worst_R']:.4f} ratio={gt['ratio']:.3f} -> {path}")
    return gt


# ════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description='Benchmark (NSFNET/USNET) pipeline')
    ap.add_argument('--K', type=int, required=True, choices=[5, 10])
    ap.add_argument('--topology', choices=['nsfnet', 'usnet'], default=None,
                    help='預設：K=5->nsfnet, K=10->usnet')
    ap.add_argument('--stage', choices=['1', '2', '3', 'all'], default='all')
    ap.add_argument('--ground_truth', action='store_true', help='K<=7 全枚舉 ground truth')
    ap.add_argument('--models', nargs='+', default=['transformer', 'mlp'])
    ap.add_argument('--device', default=None, help='cuda/mps/cpu（預設自動）')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--n_workers', type=int, default=None)
    for k in ('psi', 'N_perm', 'N_d', 'epochs', 'pop_size', 'n_generations', 'psi_eval'):
        ap.add_argument(f'--{k}', type=int, default=None)
    args = ap.parse_args()

    topology = args.topology or default_topology(args.K)
    cfg = dict(CONFIG[args.K])
    for k in ('psi', 'N_perm', 'N_d', 'epochs', 'pop_size', 'n_generations', 'psi_eval'):
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)

    print(f"### topology={topology}, K={args.K}, stage={args.stage} ###")
    if args.ground_truth:
        run_ground_truth(topology, args.K, cfg, args.seed)
        return
    if args.stage in ('1', 'all'):
        run_stage1(topology, args.K, cfg, args.seed, n_workers=args.n_workers)
    if args.stage in ('2', 'all'):
        for m in args.models:
            run_stage2(topology, args.K, cfg, m, args.device, args.seed)
    if args.stage in ('3', 'all'):
        for m in args.models:
            run_stage3(topology, args.K, cfg, m, args.device, args.seed)


if __name__ == '__main__':
    main()
