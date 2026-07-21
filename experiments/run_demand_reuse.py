"""
run_demand_reuse.py
===================
同一代理模型跨多需求向量「重用（不重訓）」的量化驗證。

兩個實驗（對每個需求尺度 α，demand = clip(round(α × mid), 1..)）：
  A. 精度探測：200 個隨機排列 × 共享 MC 狀態算真值（D_max 只算一次、對所有需求免費），
     比較代理預測 vs MC 真值的 MAE / Spearman —— 含「訓練分布外（OOD）」的大需求，
     評估套用到與訓練分布顯著不同的需求向量時的準確度。
  B. 最佳化重用：對每個需求向量跑 GA（同一代理、零重訓）→ top-10 → 獨立 MC 驗證，
     記錄 MC 最佳 R、代理-MC 落差、決策 wall-clock —— 檢驗單一固定需求
     與「攤銷」主張。

用法（本機即可）：
  python run_demand_reuse.py --data nsfnet_K5  --topology nsfnet --K 5
  python run_demand_reuse.py --data usnet_K10  --topology usnet  --K 10
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
import numpy as np
from scipy.stats import spearmanr


from benchmark_network import get_network
from data_generator import presample_network_states
from MC_QP_multi_st import greedy_allocation
from ga_optimizer import run_ga, mc_verify, surrogate_fitness_batch

ALPHAS = [0.6, 1.0, 1.4, 1.8, 2.2, 2.6]   # 相對校準中點的需求尺度；尾端為 OOD
N_PROBE_PERMS = 200
PSI_PROBE = 3000
PSI_EVAL = 10000
GA_CFG = {5: dict(pop_size=200, n_generations=500),
          10: dict(pop_size=500, n_generations=1000)}


def main():
    import torch
    from models.transformer_surrogate import PermutationTransformer

    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--topology', choices=['nsfnet', 'usnet'], required=True)
    ap.add_argument('--K', type=int, required=True)
    ap.add_argument('--device', default=None)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    K, seed = args.K, args.seed

    data_dir = os.path.join(HERE, 'data', args.data)
    meta = np.load(os.path.join(data_dir, 'metadata.npz'), allow_pickle=True)
    d_ranges = meta['d_ranges']
    mid = ((d_ranges[:, 0] + d_ranges[:, 1]) / 2)
    hi = d_ranges[:, 1].astype(float)

    device = args.device or ('cuda' if torch.cuda.is_available() else
                             ('mps' if torch.backends.mps.is_available() else 'cpu'))
    ckpt = torch.load(os.path.join(data_dir, 'models', 'transformer_best.pt'),
                      map_location=device, weights_only=False)
    a = ckpt.get('args', {})
    model = PermutationTransformer(K=K, d_model=a.get('d_model', 64), nhead=a.get('nhead', 4),
                                   num_layers=a.get('num_layers', 3),
                                   dim_feedforward=a.get('d_model', 64) * 2)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device).eval()
    d_max = np.asarray(ckpt['d_max'], dtype=np.float32)

    net = get_network(args.topology, K)
    MPs, T_list, binom, Cmax, mpt = net

    # ── 實驗 A：精度探測（D_max 只算一次，跨需求免費）──
    print(f"[A] 精度探測：{N_PROBE_PERMS} 排列 × ψ={PSI_PROBE}（D_max 共用）...")
    rng = np.random.default_rng(seed + 5)
    probe_perms = np.array([rng.permutation(K) for _ in range(N_PROBE_PERMS)])
    caps, lts = presample_network_states(binom, Cmax, mpt, PSI_PROBE, seed=seed + 55)
    t0 = time.perf_counter()
    D = np.zeros((N_PROBE_PERMS, PSI_PROBE, K), dtype=int)
    for pi in range(N_PROBE_PERMS):
        for t in range(PSI_PROBE):
            D[pi, t] = greedy_allocation(caps[t], lts[t], MPs, T_list, list(probe_perms[pi]))
    print(f"    D_max 完成（{time.perf_counter()-t0:.0f}s）")

    results = dict(K=K, data=args.data, alphas=ALPHAS, probe=[], reuse=[])
    for alpha in ALPHAS:
        dem = np.maximum(1, np.round(alpha * mid)).astype(int)
        ood = bool(np.any(dem > hi + 1e-9))
        true_R = np.all(D >= dem, axis=2).mean(axis=1)
        pred = surrogate_fitness_batch(model, probe_perms,
                                       (dem / d_max).astype(np.float32), device)
        mae = float(np.mean(np.abs(pred - true_R)))
        rho_v = spearmanr(pred, true_R).correlation
        rho = float(rho_v) if np.isfinite(rho_v) else float('nan')
        results['probe'].append(dict(alpha=alpha, demand=dem.tolist(), OOD=ood,
                                     mean_true_R=float(true_R.mean()), MAE=mae, Spearman=rho))
        print(f"    α={alpha:>3}: mean R={true_R.mean():.3f} MAE={mae:.4f} "
              f"Spearman={rho:.4f} {'[OOD]' if ood else ''}")

    # ── 實驗 B：GA 重用（每需求：搜索+MC 驗證；零重訓）──
    print(f"[B] GA 跨需求重用（零重訓）...")
    cfg = GA_CFG[K]
    for alpha in ALPHAS:
        dem = np.maximum(1, np.round(alpha * mid)).astype(int)
        ood = bool(np.any(dem > hi + 1e-9))
        t0 = time.perf_counter()
        best_perm, best_fit, _, pop, fit = run_ga(
            model, K, dem.astype(np.float32), d_max, device,
            pop_size=cfg['pop_size'], n_generations=cfg['n_generations'], seed=seed)
        seen, top = set(), []
        for idx in np.argsort(fit)[::-1]:
            pt = tuple(pop[idx])
            if pt not in seen:
                seen.add(pt); top.append(pop[idx])
                if len(top) >= 10:
                    break
        if tuple(best_perm) not in seen:
            top.insert(0, best_perm)
        mc = mc_verify(np.array(top), dem, MPs, T_list, binom, Cmax, mpt,
                       psi=PSI_EVAL, seed=seed + 100)
        best = max(mc, key=lambda r: r['R_joint'])
        wall = time.perf_counter() - t0
        gap = float(best_fit) - best['R_joint']
        results['reuse'].append(dict(alpha=alpha, demand=dem.tolist(), OOD=ood,
                                     mc_best_R=best['R_joint'], CI=best['CI_95'],
                                     pred_at_best=float(best_fit), pred_mc_gap=gap,
                                     decision_time_sec=wall))
        print(f"    α={alpha:>3}: MC best={best['R_joint']:.4f} "
              f"pred-MC gap={gap:+.4f} time={wall:.0f}s {'[OOD]' if ood else ''}")

    out = os.path.join(data_dir, 'demand_reuse_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[存檔] {out}")


if __name__ == '__main__':
    main()
