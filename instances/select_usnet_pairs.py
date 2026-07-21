"""
select_usnet_pairs.py
=====================
在 USNET 圖上自動挑選 K=10 的 s-t 對（短路徑、多候選），
並直接跑 D_max + 需求校準，確認 K=10 能校準到平均 R_joint ~= 0.5 且有排序效應。
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

import os, sys, itertools
import numpy as np
import networkx as nx

sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from data_generator import presample_network_states
from MC_QP_multi_st import greedy_allocation
from usnet_data import NODES, LINKS, CMAX, REL_HIGH, REL_LOW, LT_BASE, LT_SPAN, T_BUDGET
from math import radians, sin, cos, asin, sqrt

MAX_ARC = 5      # 候選路徑最長弧數
M = 10           # 每對候選數上限
PSI = 1500
SEED = 42


def hav(a, b):
    lo1, la1 = radians(NODES[a][1]), radians(NODES[a][2])
    lo2, la2 = radians(NODES[b][1]), radians(NODES[b][2])
    h = sin((la2-la1)/2)**2 + cos(la1)*cos(la2)*sin((lo2-lo1)/2)**2
    return 2*6371*asin(sqrt(h))


def build():
    G = nx.Graph(); G.add_nodes_from(NODES); G.add_edges_from(LINKS)
    arc = {}
    for i, (u, v) in enumerate(LINKS):
        arc[(u, v)] = i; arc[(v, u)] = i
    lengths = np.array([hav(u, v) for u, v in LINKS])
    frac = lengths/lengths.max()
    binom = REL_HIGH - (REL_HIGH-REL_LOW)*frac
    mpt = LT_BASE + LT_SPAN*frac
    Cmax = np.full(len(LINKS), CMAX, int)
    return G, arc, binom, Cmax, mpt


def paths_for(G, arc, s, t, n):
    rows, lens = [], []
    for node_path in nx.shortest_simple_paths(G, s, t):
        L = len(node_path)-1
        if L > MAX_ARC: break
        m = np.zeros(n, bool)
        for a, b in zip(node_path[:-1], node_path[1:]):
            m[arc[(a, b)]] = True
        rows.append(m); lens.append(L)
        if len(rows) >= M: break
    return np.array(rows, bool), lens


def main():
    G, arc, binom, Cmax, mpt = build()
    n = len(LINKS)
    # 候選對：最短路 3-4 弧、候選數 >= 8
    cand = []
    for s, t in itertools.combinations(NODES, 2):
        sp = nx.shortest_path_length(G, s, t)
        if sp in (3, 4):
            rows, lens = paths_for(G, arc, s, t, n)
            if len(rows) >= 8:
                cand.append((s, t, len(rows), min(lens)))
    # 貪婪挑 10 對，盡量讓端點分散
    cand.sort(key=lambda x: (-x[2], x[3]))
    chosen, used = [], set()
    for s, t, npath, mn in cand:
        if s in used or t in used:
            continue
        chosen.append((s, t)); used.add(s); used.add(t)
        if len(chosen) >= 10:
            break
    # 若不足 10（端點用完），放寬「端點不重用」補齊
    if len(chosen) < 10:
        for s, t, npath, mn in cand:
            if (s, t) in chosen:
                continue
            chosen.append((s, t))
            if len(chosen) >= 10:
                break
    print("選出的 K=10 對:")
    for s, t in chosen:
        print(f"  ({s:>2},{t:>2})  {NODES[s][0].split(',')[0]:>14} -> {NODES[t][0].split(',')[0]}")

    # ---- 校準測試 ----
    K = len(chosen)
    MPs = [paths_for(G, arc, s, t, n)[0] for s, t in chosen]
    T_list = [T_BUDGET]*K
    caps, lts = presample_network_states(binom, Cmax, mpt, PSI, seed=SEED)
    rng = np.random.default_rng(SEED)
    perms = [tuple(rng.permutation(K)) for _ in range(60)]
    D = np.zeros((len(perms), PSI, K), int)
    for pi, perm in enumerate(perms):
        for ti in range(PSI):
            D[pi, ti] = greedy_allocation(caps[ti], lts[ti], MPs, T_list, list(perm))
    ref = np.array([np.median(D[..., k][D[..., k] > 0]) if np.any(D[..., k] > 0) else 1.0
                    for k in range(K)])

    def mR(a):
        dem = np.maximum(2, np.round(a*ref)).astype(int)
        return np.all(D >= dem, axis=2).mean(), dem
    lo, hi = 0.005, 3.0
    for _ in range(22):
        m = (lo+hi)/2; r, _ = mR(m)
        if r > 0.5: lo = m
        else: hi = m
    a = (lo+hi)/2; r, dem = mR(a)
    pr = np.all(D >= dem, axis=2).mean(axis=1)
    ratio = pr.max()/pr.min() if pr.min() > 0 else float('inf')
    print(f"\n校準: alpha*={a:.3f} meanR={r:.3f} bestR={pr.max():.3f} worstR={pr.min():.3f} "
          f"ratio={ratio:.2f}")
    print("ST_PAIRS_K10 =", [list(p) for p in chosen])


if __name__ == "__main__":
    main()
