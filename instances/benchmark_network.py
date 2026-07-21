"""
benchmark_network.py
====================
把 benchmark 拓撲規格（nsfnet_data / usnet_data）轉成原始 pipeline 可用的網路物件。

提供
----
get_network_nsfnet(K) -> (MPs_list, T_list, binom, Cmax, mpt)   # NSFNET，K<=10（實用 5）
get_network_usnet(K)  -> (MPs_list, T_list, binom, Cmax, mpt)   # USNET 24 節點，K<=10（用於 K=10）
get_network(topology, K)                                        # 統一入口："nsfnet" / "usnet"

回傳格式與 code/data_generator.py 的 get_network_K3() 完全一致。

設計要點
--------
1. 每條實體鏈路 = 一條「共享容量 arc」（無向）。arc 索引 i 對應 data.LINKS[i]。
2. 每 arc 的 p_i、mu_i 依鏈路大圓長度異質（公式與常數在各 data 檔）。
3. 每個 s-t 對用 Yen's algorithm 枚舉「至多 M_PER_PAIR 條、長度 <= MAX_PATH_ARCS」
   的最短簡單路徑；各對路徑數/長度自然不同（異質）。
4. mpt 回傳為 per-arc 陣列；原始 presample_network_states 內
   rng.exponential(mpt, size=(psi, n)) 會自動對 (n,) 廣播成 (psi, n)。
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


import numpy as np
import networkx as nx
from math import radians, sin, cos, asin, sqrt

import nsfnet_data
import usnet_data


def _haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, (lon1, lat1, lon2, lat2))
    a = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * asin(sqrt(a))


def _edge_attributes(data):
    nodes, links = data.NODES, data.LINKS
    lengths = np.array([
        _haversine_km(nodes[u][1], nodes[u][2], nodes[v][1], nodes[v][2])
        for (u, v) in links
    ])
    frac = lengths / lengths.max()
    binom = data.REL_HIGH - (data.REL_HIGH - data.REL_LOW) * frac
    mpt = data.LT_BASE + data.LT_SPAN * frac
    Cmax = np.full(len(links), data.CMAX, dtype=int)
    return binom, Cmax, mpt, lengths


def _arc_index_map(data):
    idx = {}
    for i, (u, v) in enumerate(data.LINKS):
        idx[(u, v)] = i
        idx[(v, u)] = i
    return idx


def _enumerate_paths(G, arc_idx, s, t, n_arcs, m_per_pair, max_arcs):
    rows, lens = [], []
    for node_path in nx.shortest_simple_paths(G, s, t):
        L = len(node_path) - 1
        if L > max_arcs:
            break
        mask = np.zeros(n_arcs, dtype=bool)
        for a, b in zip(node_path[:-1], node_path[1:]):
            mask[arc_idx[(a, b)]] = True
        rows.append(mask)
        lens.append(L)
        if len(rows) >= m_per_pair:
            break
    return np.array(rows, dtype=bool), lens


def _select_pairs(data, K):
    pairs = data.ST_PAIRS_K10
    if K == len(pairs):
        return pairs
    if K <= len(pairs):
        return pairs[:K]
    raise ValueError(f"K={K} 超過可用 s-t 對數 {len(pairs)}")


def _build(data, K, verbose=False):
    G = nx.Graph()
    G.add_nodes_from(data.NODES.keys())
    G.add_edges_from(data.LINKS)
    arc_idx = _arc_index_map(data)
    n_arcs = len(data.LINKS)
    binom, Cmax, mpt, lengths = _edge_attributes(data)

    MPs_list, T_list = [], []
    for (s, t) in _select_pairs(data, K):
        MPs_k, path_lens = _enumerate_paths(
            G, arc_idx, s, t, n_arcs, data.M_PER_PAIR, data.MAX_PATH_ARCS)
        if len(MPs_k) == 0:
            raise RuntimeError(f"s-t 對 ({s},{t}) 找不到符合條件的路徑")
        MPs_list.append(MPs_k)
        T_list.append(float(data.T_BUDGET))
        if verbose:
            print(f"  pair ({s:>2},{t:>2}): m_k={len(MPs_k):>2}, path arcs={sorted(path_lens)}")

    if verbose:
        print(f"  n_arcs={n_arcs}, arc 長度 km: min={lengths.min():.0f}, max={lengths.max():.0f}")
        print(f"  p_i 範圍: [{binom.min():.3f}, {binom.max():.3f}], "
              f"mu_i 範圍: [{mpt.min():.2f}, {mpt.max():.2f}]")

    return MPs_list, T_list, binom, Cmax, mpt


def get_network_nsfnet(K, verbose=False):
    return _build(nsfnet_data, K, verbose)


def get_network_usnet(K, verbose=False):
    return _build(usnet_data, K, verbose)


def get_network(topology, K, verbose=False):
    if topology == "nsfnet":
        return get_network_nsfnet(K, verbose)
    if topology == "usnet":
        return get_network_usnet(K, verbose)
    raise ValueError(f"未知拓撲: {topology}")


if __name__ == "__main__":
    for topo, K in (("nsfnet", 5), ("usnet", 10)):
        print(f"\n=== {topo}, K={K} ===")
        get_network(topo, K, verbose=True)
