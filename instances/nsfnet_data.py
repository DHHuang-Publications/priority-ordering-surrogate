"""
nsfnet_data.py
==============
NSFNET benchmark 拓撲的「所需資料」單一來源（single source of truth）。

用途
----
真實 benchmark 案例研究。
本檔只放「資料與參數規格」，不含任何 pipeline 邏輯；
路徑枚舉與 MPs_list 建構由 benchmark_network.py 負責（Phase 1 下一步）。


拓撲說明
--------
採用文獻中廣泛使用的 14 節點、21 條鏈路 NSFNET 骨幹網路。
- 節點 = 骨幹路由節點（附常用城市標籤與近似經緯度，供繪圖與距離計算用）
- 鏈路 = 雙向共享容量的實體鏈路（一條光纖鏈路的容量在兩個方向間共享）
- 完整鏈路清單見 LINKS（21 條），可完全重現。

邊屬性協議（documented benchmark-generation protocol）
----------------------------------------------------
容量可靠度與前置時間不採預先設定的同質值，而是設為
「隨鏈路實體長度變化」的異質參數，公式明列於下方常數區，
並由 benchmark_network.py 依 NODES 座標計算鏈路長度後套用。

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
# 1. 節點：id -> (城市標籤, 經度 lon, 緯度 lat)
#    經緯度僅供拓撲繪圖與鏈路長度（→ 異質參數）計算，
#    不影響可靠度演算法本身。
# ============================================================
NODES = {
    1:  ("Palo Alto, CA",   -122.1, 37.4),
    2:  ("Seattle, WA",     -122.3, 47.6),
    3:  ("San Diego, CA",   -117.2, 32.7),
    4:  ("Salt Lake City, UT", -111.9, 40.8),
    5:  ("Boulder, CO",     -105.3, 40.0),
    6:  ("Houston, TX",      -95.4, 29.8),
    7:  ("Lincoln, NE",      -96.7, 40.8),
    8:  ("Champaign, IL",    -88.2, 40.1),
    9:  ("Pittsburgh, PA",   -79.9, 40.4),
    10: ("Atlanta, GA",      -84.4, 33.7),
    11: ("College Park, MD", -76.9, 39.0),
    12: ("Ithaca, NY",       -76.5, 42.4),
    13: ("Princeton, NJ",    -74.7, 40.3),
    14: ("Ann Arbor, MI",    -83.7, 42.3),
}

# ============================================================
# 2. 鏈路：21 條雙向實體鏈路（無向；容量在兩方向共享）
#    每條鏈路在 benchmark_network.py 中對應一條「共享容量 arc」。
#    (u, v) 以節點 id 表示，u < v。
# ============================================================
LINKS = [
    (1, 2), (1, 3), (1, 8),
    (2, 3), (2, 4),
    (3, 6),
    (4, 5), (4, 11),
    (5, 6), (5, 7),
    (6, 10), (6, 14),
    (7, 8),
    (8, 9),
    (9, 10), (9, 12), (9, 13),
    (10, 14),
    (11, 12), (11, 13),
    (12, 14),
]
# 註：其中 (1,8) 與 (4,11) 為兩條長距骨幹幹線（transcontinental trunk），
#     此為國家級骨幹網路的常見特徵，已納入文件化協議。

# ============================================================
# 3. Source-Terminal 對（優先順序最佳化的決策對象）
#    K=5 為 K=10 的前 5 對，確保兩個實驗規模一致可比。
#    選取原則：跨網路、彼此有足夠路徑多樣性以造成容量競爭。
# ============================================================
ST_PAIRS_K10 = [
    (1, 10),   # P1  Palo Alto  -> Atlanta
    (2, 13),   # P2  Seattle    -> Princeton
    (3, 9),    # P3  San Diego  -> Pittsburgh
    (7, 14),   # P4  Lincoln    -> Ann Arbor
    (4, 6),    # P5  Salt Lake  -> Houston
    (5, 12),   # P6  Boulder    -> Ithaca
    (8, 11),   # P7  Champaign  -> College Park
    (1, 13),   # P8  Palo Alto  -> Princeton
    (3, 14),   # P9  San Diego  -> Ann Arbor
    (2, 10),   # P10 Seattle    -> Atlanta
]
ST_PAIRS_K5 = ST_PAIRS_K10[:5]

# ============================================================
# 4. 邊屬性參數協議（documented, reproducible）
# ============================================================
# --- 容量（多狀態）---
# 每 arc 有 CMAX 個獨立元件，每元件以機率 p_i 正常運作，
# 故 arc 容量 ~ Binomial(CMAX, p_i)，狀態空間 {0,1,...,CMAX}。
CMAX = 9

# 每 arc 元件可靠度 p_i 隨鏈路長度線性遞減（長鏈路較不可靠）：
#     p_i = REL_HIGH - (REL_HIGH - REL_LOW) * (len_i / max_len)
# len_i = 兩端點大圓距離；max_len = 所有鏈路中最長者。
REL_HIGH = 0.95   # 最短鏈路的元件可靠度
REL_LOW  = 0.85   # 最長鏈路的元件可靠度

# --- 前置時間（lead time）---
# 各 arc 前置時間獨立服從指數分布 Exponential(mean = mu_i)（明確指定 f_i(.)）：
#     mu_i = LT_BASE + LT_SPAN * (len_i / max_len)
# 長鏈路傳輸/處理延遲較長；平均約落在 LT_BASE..(LT_BASE+LT_SPAN)。
# 尺度刻意設為讓「多弧路徑（至多 MAX_PATH_ARCS 條弧）仍能在共用時間預算 T_BUDGET 內可行」，
# 使 benchmark 落在與合成實驗相同的資訊可靠度區間（平均聯合可靠度可校準至 ~0.5）。
LT_BASE = 3.0    # 最短鏈路的前置時間期望值
LT_SPAN = 7.0    # 長度造成的前置時間增量上限（最長鏈路 mu ~= 10）

# --- 時間預算 ---
# 各 pair 共用相同時間限制 T（與合成實驗一致，便於對照）。
# 異質 T_k 之敏感度分析另列。
T_BUDGET = 50.0

# --- 每 pair 候選路徑數上限 ---
# 對每個 s-t 對枚舉「至多」 M_PER_PAIR 條最短簡單路徑。
# 真實拓撲下各 pair 實際取得的路徑數與長度自然不同（異質），
# 直接展示對路徑結構變異的穩健性。
M_PER_PAIR = 10
MAX_PATH_ARCS = 5   # 只保留 arc 數 <= 此值的簡單路徑（控制前置時間在預算內）

# ============================================================
# 5. 便捷檢視
# ============================================================
N_NODES = len(NODES)
N_LINKS = len(LINKS)

if __name__ == "__main__":
    # 基本自我檢查：節點涵蓋、鏈路數、度數
    from collections import Counter
    deg = Counter()
    for u, v in LINKS:
        deg[u] += 1
        deg[v] += 1
    print(f"NSFNET benchmark 規格")
    print(f"  節點數 = {N_NODES}, 鏈路數 = {N_LINKS}")
    print(f"  度數總和 = {sum(deg.values())} (應為 2*L = {2*N_LINKS})")
    print(f"  各節點度數: {dict(sorted(deg.items()))}")
    print(f"  孤立節點: {[n for n in NODES if deg[n] == 0]}")
    print(f"  K=5  對: {ST_PAIRS_K5}")
    print(f"  K=10 對: {ST_PAIRS_K10}")
