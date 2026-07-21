"""
run_ablation.py
===============
隔離主要設計選擇貢獻的消融實驗。
全部在合成 K=10 主實例（data/K10_psi5000，論文主規模）上做，one-at-a-time 掃描。

三組（用 --groups 選，預設 ga arch）：
  ga   ── GA 超參數敏感度。重用既有 transformer，零重訓：
           pop_size / n_generations / crossover_rate / mutation_rate / N_top 各自掃描，
           每點 GA→top-N→MC(ψ=10000) 驗證，記錄 MC 最佳 R_joint 與決策 wall-clock。
           → 展示所選配置在寬範圍內穩健（「adequate / not brittle」）。
  arch ── Transformer 架構。用原 train_surrogate.py 重訓，one-at-a-time：
           num_layers ∈ {2,3,4}、nhead ∈ {2,4,8}、d_model ∈ {32,64,128}。
           每點記 test MAE/R²/Spearman（+ 選配 GA+MC）。
  data ── 資料生成。縮scale 重生訓練集 + 重訓：
           CRN on/off（跨排列共用 vs 各自重抽 MC 狀態）、demand LHS vs uniform。
           成本最高，預設不含；要跑加 --groups data。

用法（GPU）：
  python run_ablation.py --groups ga            # 便宜，~30-60 min
  python run_ablation.py --groups arch --device cuda   # 重訓，~4-6 hr
  python run_ablation.py --groups ga arch data --device cuda   # 全套
輸出：data/ablation/ablation_{group}.json（可續跑，已完成點自動 skip）。
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
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import sys
import json
import time
import argparse
import subprocess
import numpy as np
from scipy.stats import spearmanr


DATA = os.path.join(CODE, 'data', 'K10_psi5000')
OUT = os.path.join(HERE, 'data', 'ablation')
K = 10
PSI_EVAL = 10000
SEED = 42

# GA 基準配置（掃描時固定其餘）
GA_BASE = dict(pop_size=500, n_generations=1000, crossover_rate=0.8,
               mutation_rate=0.3, N_top=10)
GA_SWEEP = {
    'pop_size':       [100, 200, 500, 1000],
    'n_generations':  [250, 500, 1000, 2000],
    'crossover_rate': [0.6, 0.8, 1.0],
    'mutation_rate':  [0.1, 0.2, 0.3, 0.5],
    'N_top':          [1, 5, 10, 20, 50],
}
ARCH_BASE = dict(num_layers=3, nhead=4, d_model=64)
ARCH_SWEEP = {
    'num_layers': [2, 3, 4],
    'nhead':      [2, 4, 8],
    'd_model':    [32, 64, 128],
}


# ─────────────────────────────────────────────────────────
def load_network():
    """從 metadata 建網路（binom/Cmax/mpt 直接存；MPs/T 用同 seed 重建）。"""
    from data_generator import generate_random_pairs
    m = np.load(os.path.join(DATA, 'metadata.npz'), allow_pickle=True)
    n_arcs = int(m['n_arcs'])
    binom = np.asarray(m['binom'])
    Cmax = np.asarray(m['Cmax'])
    mpt = float(m['mean_processing_time'])
    MPs, T = generate_random_pairs(n_arcs, K, m_per_pair=10,
                                   min_arcs_per_path=2, max_arcs_per_path=4, seed=SEED)
    demand = ((m['d_ranges'][:, 0] + m['d_ranges'][:, 1]) / 2).astype(int)
    return MPs, T, binom, Cmax, mpt, demand


def load_transformer(ckpt_path, device):
    import torch
    from models.transformer_surrogate import PermutationTransformer
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ck.get('args', {})
    model = PermutationTransformer(K=K, d_model=a.get('d_model', 64), nhead=a.get('nhead', 4),
                                   num_layers=a.get('num_layers', 3),
                                   dim_feedforward=a.get('d_model', 64) * 2)
    model.load_state_dict(ck['model_state_dict'])
    return model.to(device).eval(), np.asarray(ck['d_max'], np.float32)


def eval_on_test(model, d_max, device):
    import torch
    z = np.load(os.path.join(DATA, 'test.npz'))
    P = torch.as_tensor(z['perms'].astype(np.int64))
    D = torch.as_tensor(z['demands'].astype(np.float32) / d_max)
    Y = z['R_joint'].astype(np.float32)
    preds = []
    with torch.no_grad():
        for i in range(0, len(Y), 8192):
            preds.append(model(P[i:i+8192].to(device), D[i:i+8192].to(device)).cpu().numpy())
    yp = np.concatenate(preds)
    return dict(MAE=float(np.mean(np.abs(yp - Y))),
                R2=float(1 - np.sum((yp - Y)**2) / np.sum((Y - Y.mean())**2)),
                Spearman=float(spearmanr(yp, Y).correlation))


def ga_and_mc(model, d_max, net, device, pop_size, n_generations,
              crossover_rate, mutation_rate, N_top, seed=SEED):
    from ga_optimizer import run_ga, mc_verify
    MPs, T, binom, Cmax, mpt, demand = net
    t0 = time.perf_counter()
    best_perm, best_fit, _, pop, fit = run_ga(
        model, K, demand.astype(np.float32), d_max, device,
        pop_size=pop_size, n_generations=n_generations,
        crossover_rate=crossover_rate, mutation_rate=mutation_rate, seed=seed)
    seen, top = set(), []
    for i in np.argsort(fit)[::-1]:
        tup = tuple(pop[i])
        if tup not in seen:
            seen.add(tup); top.append(pop[i])
            if len(top) >= N_top:
                break
    if tuple(best_perm) not in seen:
        top.insert(0, best_perm)
    mc = mc_verify(np.array(top), demand, MPs, T, binom, Cmax, mpt,
                   psi=PSI_EVAL, seed=seed + 100)
    best = max(mc, key=lambda r: r['R_joint'])
    return dict(mc_best_R=best['R_joint'], CI=best['CI_95'],
                predicted=float(best_fit), time_sec=time.perf_counter() - t0)


# ─────────────────────────────────────────────────────────
def run_group_ga(device):
    import torch
    os.makedirs(OUT, exist_ok=True)
    # K=10 沿用舊檔名 ablation_ga.json；其餘 K 加後綴，避免覆蓋
    fname = 'ablation_ga.json' if K == 10 else f'ablation_ga_K{K}.json'
    path = os.path.join(OUT, fname)
    res = json.load(open(path)) if os.path.exists(path) else {'base': GA_BASE, 'sweeps': {}}
    model, d_max = load_transformer(os.path.join(DATA, 'models', 'transformer_best.pt'), device)
    net = load_network()
    for knob, values in GA_SWEEP.items():
        res['sweeps'].setdefault(knob, {})
        for v in values:
            key = str(v)
            if key in res['sweeps'][knob]:
                print(f"  [skip] ga {knob}={v}"); continue
            cfg = dict(GA_BASE); cfg[knob] = v
            print(f"  [ga] {knob}={v} ...", flush=True)
            r = ga_and_mc(model, d_max, net, device, **cfg)
            res['sweeps'][knob][key] = r
            json.dump(res, open(path, 'w'), indent=2)
            print(f"      MC R={r['mc_best_R']:.4f} pred={r['predicted']:.4f} t={r['time_sec']:.0f}s")
    print(f"[存檔] {path}")


def run_group_arch(device):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, 'ablation_arch.json')
    res = json.load(open(path)) if os.path.exists(path) else {'base': ARCH_BASE, 'sweeps': {}}
    net = load_network()
    # 收集 unique 配置（one-at-a-time，baseline 只算一次）
    configs = {}
    for knob, values in ARCH_SWEEP.items():
        for v in values:
            cfg = dict(ARCH_BASE); cfg[knob] = v
            configs[(cfg['num_layers'], cfg['nhead'], cfg['d_model'])] = (knob, v, cfg)
    for (nl, nh, dm), (knob, v, cfg) in configs.items():
        tag = f"L{nl}_H{nh}_D{dm}"
        res['sweeps'].setdefault(knob, {})
        if str(v) in res['sweeps'][knob]:
            print(f"  [skip] arch {knob}={v}"); continue
        save_dir = os.path.join(OUT, 'models', tag)
        os.makedirs(save_dir, exist_ok=True)
        print(f"  [arch] {tag} (train) ...", flush=True)
        cmd = [sys.executable, os.path.join(ROOT, 'core', 'train_surrogate.py'),
               '--data_dir', DATA, '--model', 'transformer',
               '--num_layers', str(nl), '--nhead', str(nh), '--d_model', str(dm),
               '--save_dir', save_dir]
        if device:
            cmd += ['--device', device]
        env = dict(os.environ); env['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        t0 = time.perf_counter()
        subprocess.run(cmd, check=True, env=env)
        train_time = time.perf_counter() - t0
        import torch
        model, d_max = load_transformer(os.path.join(save_dir, 'transformer_best.pt'), device)
        acc = eval_on_test(model, d_max, device)
        ga = ga_and_mc(model, d_max, net, device, **{**GA_BASE})
        n_params = sum(p.numel() for p in model.parameters())
        res['sweeps'][knob][str(v)] = dict(tag=tag, n_params=n_params,
                                           train_time_sec=train_time, **acc,
                                           mc_best_R=ga['mc_best_R'])
        json.dump(res, open(path, 'w'), indent=2)
        print(f"      {tag}: {n_params/1e3:.0f}K MAE={acc['MAE']:.4f} "
              f"rho={acc['Spearman']:.4f} MC_R={ga['mc_best_R']:.4f}")
    print(f"[存檔] {path}")


def run_group_data(device):
    """CRN on/off、demand LHS/uniform：縮scale 重生 + 重訓。成本最高。"""
    from data_generator import (presample_network_states, sample_demands_lhs,
                                sample_permutations, _init_worker, _compute_dmax_for_perm)
    from multiprocessing import Pool
    import torch
    from models.transformer_surrogate import PermutationTransformer
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, 'ablation_data.json')
    res = json.load(open(path)) if os.path.exists(path) else {'configs': {}}
    MPs, T, binom, Cmax, mpt, demand_mid = load_network()
    m = np.load(os.path.join(DATA, 'metadata.npz'), allow_pickle=True)
    d_ranges = m['d_ranges']
    # 縮scale 以控成本
    PSI, N_PERM, N_D = 3000, 400, 20

    def build_labels(crn, demand_mode, seed):
        rng = np.random.default_rng(seed)
        perms = sample_permutations(K, N_PERM, seed=seed)
        if demand_mode == 'lhs':
            demands = sample_demands_lhs(d_ranges, N_D, seed=seed)
        else:  # uniform
            demands = np.stack([rng.integers(d_ranges[:, 0], d_ranges[:, 1] + 1)
                                for _ in range(N_D)])
        # CRN on：所有排列共用一組預抽樣 MC 狀態；off：每排列各自重抽
        rows_P, rows_D, rows_Y = [], [], []
        if crn:
            caps, lts = presample_network_states(binom, Cmax, mpt, PSI, seed=seed + 7)
        from MC_QP_multi_st import greedy_allocation
        for pi, perm in enumerate(perms):
            if not crn:
                caps, lts = presample_network_states(binom, Cmax, mpt, PSI, seed=seed + 7 + pi)
            Dmax = np.stack([greedy_allocation(caps[t], lts[t], MPs, T, list(perm))
                             for t in range(PSI)])
            for d in demands:
                rows_P.append(perm); rows_D.append(d)
                rows_Y.append(float(np.mean(np.all(Dmax >= d, axis=1))))
        return (np.array(rows_P), np.array(rows_D, np.float32), np.array(rows_Y, np.float32))

    def train_eval(P, D, Y, device, seed=SEED):
        d_max = np.maximum(D.max(0), 1.0)
        n = len(Y); idx = np.random.default_rng(seed).permutation(n)
        ntr = int(n * 0.8); nva = int(n * 0.1)
        tr, va, te = idx[:ntr], idx[ntr:ntr+nva], idx[ntr+nva:]
        Pt = torch.as_tensor(P.astype(np.int64)); Dt = torch.as_tensor(D / d_max)
        Yt = torch.as_tensor(Y)
        model = PermutationTransformer(K=K, d_model=64, nhead=4, num_layers=3,
                                       dim_feedforward=128).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        lossf = torch.nn.MSELoss()
        best, best_state, patience = 1e9, None, 30
        for ep in range(200):
            model.train(); perm = tr[np.random.default_rng(ep).permutation(len(tr))]
            for i in range(0, len(perm), 256):
                b = perm[i:i+256]
                opt.zero_grad()
                out = model(Pt[b].to(device), Dt[b].to(device))
                loss = lossf(out, Yt[b].to(device)); loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                vp = model(Pt[va].to(device), Dt[va].to(device)).cpu().numpy()
            vmae = np.mean(np.abs(vp - Y[va]))
            if vmae < best:
                best, patience = vmae, 30
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience -= 1
                if patience <= 0:
                    break
        model.load_state_dict(best_state); model.eval()
        with torch.no_grad():
            tp = model(Pt[te].to(device), Dt[te].to(device)).cpu().numpy()
        return dict(MAE=float(np.mean(np.abs(tp - Y[te]))),
                    Spearman=float(spearmanr(tp, Y[te]).correlation),
                    R2=float(1 - np.sum((tp - Y[te])**2) / np.sum((Y[te] - Y[te].mean())**2)),
                    n_samples=n)

    for crn, dm in [(True, 'lhs'), (False, 'lhs'), (True, 'uniform')]:
        key = f"CRN{'on' if crn else 'off'}_{dm}"
        if key in res['configs']:
            print(f"  [skip] data {key}"); continue
        print(f"  [data] {key} (regen+train) ...", flush=True)
        t0 = time.perf_counter()
        P, D, Y = build_labels(crn, dm, SEED)
        r = train_eval(P, D, Y, device)
        r['wall_sec'] = time.perf_counter() - t0
        res['configs'][key] = r
        json.dump(res, open(path, 'w'), indent=2)
        print(f"      {key}: MAE={r['MAE']:.4f} rho={r['Spearman']:.4f} ({r['wall_sec']:.0f}s)")
    print(f"[存檔] {path}  (基準=CRNon_lhs；比較 CRN、抽樣的影響)")


def main():
    import torch
    ap = argparse.ArgumentParser(description='ablations')
    ap.add_argument('--groups', nargs='+', default=['ga', 'arch'],
                    choices=['ga', 'arch', 'data'])
    ap.add_argument('--K', type=int, default=10,
                    help='目標實例（ga 組可指定 15 做參數敏感度；arch/data 建議留 10）')
    ap.add_argument('--device', default=None)
    args = ap.parse_args()
    global K, DATA
    K = args.K
    DATA = os.path.join(CODE, 'data', f'K{K}_psi5000')
    device = args.device or ('cuda' if torch.cuda.is_available() else
                             ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"device={device}  K={K}  DATA={DATA}  groups={args.groups}")
    if 'ga' in args.groups:
        print("\n===== GROUP ga (GA 超參數敏感度) =====")
        run_group_ga(device)
    if 'arch' in args.groups:
        print("\n===== GROUP arch (Transformer 架構) =====")
        run_group_arch(device)
    if 'data' in args.groups:
        print("\n===== GROUP data (CRN / LHS) =====")
        run_group_data(device)


if __name__ == '__main__':
    main()
