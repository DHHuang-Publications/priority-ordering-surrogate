"""
usnet_data.py
=============
較大的真實骨幹 benchmark：24 節點 US 骨幹（USNET），供 K=10 案例使用。
NSFNET(21 弧) 結構上只撐得住 ~5 對；本拓撲約 43 弧，足以支撐 10 對的容量競爭。

與 nsfnet_data.py 同結構、同邊屬性協議（容量 Binomial(9,p_i)、
前置時間 Exponential(mean=mu_i)，p_i/mu_i 依鏈路長度異質），確保兩個真實拓撲
的建模一致、可直接對照。

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


# ============================================================
# 1. 節點：id -> (城市, 經度, 緯度)
# ============================================================
NODES = {
    1:  ("Seattle, WA",        -122.3, 47.6),
    2:  ("Portland, OR",       -122.7, 45.5),
    3:  ("San Francisco, CA",  -122.4, 37.8),
    4:  ("Los Angeles, CA",    -118.2, 34.1),
    5:  ("San Diego, CA",      -117.2, 32.7),
    6:  ("Salt Lake City, UT", -111.9, 40.8),
    7:  ("Las Vegas, NV",      -115.1, 36.2),
    8:  ("Phoenix, AZ",        -112.1, 33.4),
    9:  ("Denver, CO",         -105.0, 39.7),
    10: ("Albuquerque, NM",    -106.6, 35.1),
    11: ("El Paso, TX",        -106.5, 31.8),
    12: ("Dallas, TX",          -96.8, 32.8),
    13: ("Houston, TX",         -95.4, 29.8),
    14: ("Kansas City, MO",     -94.6, 39.1),
    15: ("Minneapolis, MN",     -93.3, 45.0),
    16: ("Chicago, IL",         -87.6, 41.9),
    17: ("St. Louis, MO",       -90.2, 38.6),
    18: ("Atlanta, GA",         -84.4, 33.7),
    19: ("Nashville, TN",       -86.8, 36.2),
    20: ("Detroit, MI",         -83.0, 42.3),
    21: ("Washington, DC",      -77.0, 38.9),
    22: ("New York, NY",        -74.0, 40.7),
    23: ("Boston, MA",          -71.1, 42.4),
    24: ("Miami, FL",           -80.2, 25.8),
}

# ============================================================
# 2. 鏈路：43 條雙向實體鏈路（無向；容量兩方向共享）
# ============================================================
LINKS = [
    (1, 2), (1, 6),
    (2, 3), (2, 6),
    (3, 4), (3, 6),
    (4, 5), (4, 7),
    (5, 7), (5, 8),
    (6, 7), (6, 9),
    (7, 8),
    (8, 10), (8, 11),
    (9, 10), (9, 14), (9, 15), (9, 17),
    (10, 11),
    (11, 12), (11, 13),
    (12, 13), (12, 14),
    (13, 18),
    (14, 15), (14, 16), (14, 17),
    (15, 16),
    (16, 17), (16, 19), (16, 20),
    (17, 18), (17, 19),
    (18, 19), (18, 21), (18, 24),
    (19, 20),
    (20, 21), (20, 22),
    (21, 22), (21, 24),
    (22, 23),
]

# ============================================================
# 3. Source-Terminal 對（K=10；K<=10 取前 K 個）
# ============================================================
# 由 select_usnet_pairs.py 自動挑選：最短路 3-4 弧、候選 >=8、端點分散。
ST_PAIRS_K10 = [
    (1, 4),    # Seattle       -> Los Angeles
    (2, 5),    # Portland      -> San Diego
    (3, 8),    # San Francisco -> Phoenix
    (6, 11),   # Salt Lake City-> El Paso
    (7, 12),   # Las Vegas     -> Dallas
    (9, 13),   # Denver        -> Houston
    (10, 16),  # Albuquerque   -> Chicago
    (14, 21),  # Kansas City   -> Washington
    (15, 18),  # Minneapolis   -> Atlanta
    (17, 22),  # St. Louis     -> New York
]

# ============================================================
# 4. 邊屬性參數協議（與 nsfnet_data.py 一致）
# ============================================================
# 每弧元件數 = 30：較 NSFNET(9) 高，因 10 對競爭 43 弧需更高每弧容量才能
# 讓 10-way 聯合可靠度落在資訊區（平均 ~0.5）。此為「每 arc 容量隨競爭對數調整」，
# 對應合成實驗「網路大小隨 K 縮放(n=5K)」的作法——固定真實拓撲無法加弧，故調容量。
# 經驗校準：Cmax=9→mean~0.30、15→0.41、25→0.49、30→0.51（LHS 校準到目標 0.5）。
CMAX = 30
REL_HIGH = 0.95
REL_LOW  = 0.85
LT_BASE = 3.0
LT_SPAN = 7.0
T_BUDGET = 50.0
M_PER_PAIR = 10
MAX_PATH_ARCS = 6   # 拓撲較大、路徑較長，放寬到 6 弧

N_NODES = len(NODES)
N_LINKS = len(LINKS)

if __name__ == "__main__":
    from collections import Counter
    deg = Counter()
    for u, v in LINKS:
        deg[u] += 1; deg[v] += 1
    print(f"USNET: 節點={N_NODES}, 鏈路={N_LINKS}, 度數總和={sum(deg.values())} (應=2L={2*N_LINKS})")
    print(f"  各節點度數: {dict(sorted(deg.items()))}")
    print(f"  孤立節點: {[n for n in NODES if deg[n]==0]}")
    print(f"  重複鏈路: {[e for e,c in Counter(tuple(sorted(l)) for l in LINKS).items() if c>1]}")
