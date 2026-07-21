"""
run_independent_items.py
========================
獨立項小實驗（本機即可）：
  A. R3-C8 標籤雜訊：NSFNET K=5，兩組獨立 ψ=5000 標籤 vs 一組 ψ=10000，
     量測經驗標籤雜訊 std / 排序穩定度，對照理論上界 0.5/√ψ。
  B. R3-C9 Stage-1 分解與平行效率：NSFNET K=5（ψ=3000、120 排列）分解
     預抽樣/校準外的主成本（D_max、標籤），並量 n_workers ∈ {1,2,4,8} 的加速比。
輸出：data/independent_items_results.json
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
import numpy as np
from multiprocessing import Pool
from scipy.stats import spearmanr


from benchmark_network import get_network_nsfnet
from data_generator import presample_network_states, _init_worker, _compute_dmax_for_perm
from MC_QP_multi_st import greedy_allocation
import itertools

SEED = 42


def labels_for(MPs, T, binom, Cmax, mpt, perms, demand, psi, seed):
    caps, lts = presample_network_states(binom, Cmax, mpt, psi, seed=seed)
    R = np.zeros(len(perms))
    for pi, perm in enumerate(perms):
        D = np.stack([greedy_allocation(caps[t], lts[t], MPs, T, list(perm))
                      for t in range(psi)])
        R[pi] = np.mean(np.all(D >= demand, axis=1))
    return R


def main():
    out = {}
    MPs, T, binom, Cmax, mpt = get_network_nsfnet(5)
    perms = [list(p) for p in itertools.permutations(range(5))]
    demand = np.array([19, 18, 15, 18, 23])

    # ── A. 標籤雜訊 ──
    print("[A] 標籤雜訊（120 排列 × ψ=5000 ×2 + ψ=10000）...")
    R5a = labels_for(MPs, T, binom, Cmax, mpt, perms, demand, 5000, SEED + 1)
    R5b = labels_for(MPs, T, binom, Cmax, mpt, perms, demand, 5000, SEED + 2)
    R10 = labels_for(MPs, T, binom, Cmax, mpt, perms, demand, 10000, SEED + 3)
    diff = R5a - R5b
    noise_std = float(diff.std(ddof=1) / np.sqrt(2))      # 單組標籤的經驗雜訊
    theory = 0.5 / np.sqrt(5000)
    out['label_noise'] = dict(
        empirical_std_psi5000=noise_std,
        theory_upper_psi5000=theory,
        spearman_5000_vs_5000=float(spearmanr(R5a, R5b).correlation),
        spearman_5000_vs_10000=float(spearmanr(R5a, R10).correlation),
        mean_R=float(R10.mean()))
    print(f"    經驗雜訊 std={noise_std:.5f}（理論上界 {theory:.5f}）")
    print(f"    排序穩定：ρ(5k,5k)={out['label_noise']['spearman_5000_vs_5000']:.4f} "
          f"ρ(5k,10k)={out['label_noise']['spearman_5000_vs_10000']:.4f}")

    # ── B. Stage-1 分解 + 平行效率 ──
    print("[B] Stage-1 分解與平行效率（ψ=3000、120 排列）...")
    psi = 3000
    t0 = time.perf_counter()
    caps, lts = presample_network_states(binom, Cmax, mpt, psi, seed=SEED)
    t_presample = time.perf_counter() - t0

    speedup = {}
    perm_tuples = [tuple(p) for p in perms]
    for nw in (1, 2, 4, 8):
        t0 = time.perf_counter()
        with Pool(processes=nw, initializer=_init_worker,
                  initargs=(caps, lts, MPs, T)) as pool:
            Dlist = pool.map(_compute_dmax_for_perm, perm_tuples)
        speedup[nw] = time.perf_counter() - t0
        print(f"    D_max n_workers={nw}: {speedup[nw]:.1f}s")
    D = np.stack(Dlist)   # (120, psi, K)

    t0 = time.perf_counter()
    _ = np.all(D >= demand, axis=2).mean(axis=1)   # 標籤計算（單一需求代表）
    t_label_one = time.perf_counter() - t0
    out['stage1_breakdown'] = dict(
        psi=psi, n_perms=len(perms),
        presample_sec=t_presample,
        dmax_sec_by_workers={str(k): v for k, v in speedup.items()},
        speedup_8w=speedup[1] / speedup[8],
        parallel_efficiency_8w=speedup[1] / speedup[8] / 8,
        label_per_demand_sec=t_label_one,
        dmax_share_1w=speedup[1] / (speedup[1] + t_presample + t_label_one))
    print(f"    presample={t_presample:.2f}s, label/需求={t_label_one:.4f}s")
    print(f"    8-worker 加速比={out['stage1_breakdown']['speedup_8w']:.2f}x "
          f"(效率 {out['stage1_breakdown']['parallel_efficiency_8w']*100:.0f}%)")
    print(f"    D_max 佔 Stage-1（單核）比例 ≈ {out['stage1_breakdown']['dmax_share_1w']*100:.1f}%")

    p = os.path.join(HERE, 'data', 'independent_items_results.json')
    with open(p, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"[存檔] {p}")


if __name__ == '__main__':
    main()
