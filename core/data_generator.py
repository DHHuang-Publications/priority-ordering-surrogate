"""
data_generator.py
=================
Stage 1: MC Data Generation for Surrogate-Assisted Optimization

核心流程
--------
1. 預抽樣 ψ 組網路狀態 (capacity, lead_time)，所有排列共用
2. 抽樣 N_perm 個不重複排列 π（K≤7 全列舉，K>7 隨機抽樣）
3. 對每個 π，用共用網路狀態跑 greedy_allocation → D_max_all (ψ×K)
4. 探索 D_max 分布 → 決定 d 的合理範圍
5. 用 LHS 在 d 空間撒 N_d 組需求向量
6. 對每個 (π, d) 組合，算 R_joint = mean(∀k: D_max[k] ≥ d_k)
7. 儲存 (π, d, R_joint) 為 .npz

關鍵設計
--------
- D_max 的計算不依賴 d → 同一個 π 可配多組 d，幾乎免費
- 預抽樣共用 → 不同 π 的 R_joint 差異純粹來自排列順序，signal 更乾淨
- multiprocessing 加速 greedy_allocation（CPU-bound）

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
import itertools
import time
import os
import argparse
from multiprocessing import Pool, cpu_count
from scipy.stats.qmc import LatinHypercube

# 匯入核心函式
from MC_QP_multi_st import greedy_allocation


# ============================================================
# 網路定義
# ============================================================

def get_network_K3():
    """
    K=3 的示範網路（同 Paper 1）。
    20 條弧、3 組 source-terminal pair，每組 10 條候選路徑。

    回傳
    ----
    MPs_list : list of np.ndarray (bool)
    T_list : list of float
    binom : np.ndarray
    Cmax : np.ndarray
    mean_processing_time : float
    """
    binom = np.array([
        0.887563353, 0.862754756, 0.875297853, 0.884953836,
        0.894545163, 0.897964571, 0.877360776, 0.856931222,
        0.857464700, 0.862875413, 0.892035863, 0.862714109,
        0.890714241, 0.862176248, 0.896463181, 0.867499188,
        0.859829763, 0.862554193, 0.880802234, 0.873664442
    ])
    n_arcs = len(binom)
    Cmax = np.full(n_arcs, 9, dtype=int)
    mean_processing_time = 10.0

    # Pair 1: 10 條候選路徑（bool matrix, shape=(10, 20)）
    MPs_pair1_raw = [
        [1,0,3,4,5,6,7,8,9,10,11,0,0,0,0,0,0,0,0,0],
        [1,0,3,4,5,6,7,8,9,0,11,0,0,0,0,0,0,18,19,20],
        [1,0,3,4,5,6,0,0,0,10,11,0,0,14,0,16,17,18,19,0],
        [1,0,3,4,5,6,0,0,0,0,11,0,0,14,0,16,17,0,0,20],
        [1,0,3,4,0,0,7,8,9,10,11,0,13,14,0,0,0,0,0,0],
        [1,0,3,4,0,0,7,8,9,0,11,0,13,14,0,0,0,18,19,20],
        [1,0,3,4,0,0,0,0,0,10,11,0,13,0,0,16,17,18,19,0],
        [1,0,3,4,0,0,0,0,0,0,11,0,13,0,0,16,17,0,0,20],
        [1,0,0,0,5,6,7,8,9,10,11,12,13,0,15,16,0,0,0,0],
        [1,0,0,0,5,6,7,8,9,0,11,12,13,0,15,16,0,18,19,20],
    ]
    # Pair 2: 10 條候選路徑
    MPs_pair2_raw = [
        [1,0,0,0,0,0,7,8,9,10,11,12,0,14,15,16,0,0,0,0],
        [1,0,0,0,0,0,7,8,9,0,11,12,0,14,15,16,0,18,19,20],
        [1,0,0,0,0,0,0,0,0,10,11,12,0,0,15,0,17,18,19,0],
        [1,0,0,0,0,0,0,0,0,0,11,12,0,0,15,0,17,0,0,20],
        [0,2,3,4,0,0,7,8,9,10,11,12,0,14,15,16,0,0,0,0],
        [0,2,3,4,0,0,7,8,9,0,11,12,0,14,15,16,0,18,19,20],
        [0,2,3,4,0,0,0,0,0,10,11,12,0,0,15,0,17,18,19,0],
        [0,2,3,4,0,0,0,0,0,0,11,12,0,0,15,0,17,0,0,20],
        [0,2,3,0,5,0,7,8,9,10,11,12,0,14,15,16,0,0,0,0],
        [0,2,3,0,5,0,7,8,9,0,11,12,0,14,15,16,0,18,19,20],
    ]
    # Pair 3: 10 條候選路徑
    MPs_pair3_raw = [
        [0,2,3,0,5,0,0,0,0,10,11,12,0,0,15,0,17,18,19,0],
        [0,2,3,0,5,0,0,0,0,0,11,12,0,0,15,0,17,0,0,20],
        [0,2,0,0,5,6,7,8,9,10,11,0,0,0,0,0,0,0,0,0],
        [0,2,0,0,5,6,7,8,9,0,11,0,0,0,0,0,0,18,19,20],
        [0,2,0,0,5,6,0,0,0,10,11,0,0,14,0,16,17,18,19,0],
        [0,2,0,0,5,6,0,0,0,0,11,0,0,14,0,16,17,0,0,20],
        [0,2,0,0,0,0,7,8,9,10,11,0,13,14,0,0,0,0,0,0],
        [0,2,0,0,0,0,7,8,9,0,11,0,13,14,0,0,0,18,19,20],
        [0,2,0,0,0,0,0,0,0,10,11,0,13,0,0,16,17,18,19,0],
        [0,2,0,0,0,0,0,0,0,0,11,0,13,0,0,16,17,0,0,20],
    ]

    MPs_list = [
        np.array(MPs_pair1_raw, dtype=bool),
        np.array(MPs_pair2_raw, dtype=bool),
        np.array(MPs_pair3_raw, dtype=bool),
    ]
    T_list = [50.0, 50.0, 50.0]

    return MPs_list, T_list, binom, Cmax, mean_processing_time


def generate_random_pairs(n_arcs, K, m_per_pair=10,
                          min_arcs_per_path=4, max_arcs_per_path=10,
                          seed=None):
    """
    為固定的 n_arcs 條弧網路，自動生成 K 組 source-terminal pair 的候選路徑。
    每組 pair 有 m_per_pair 條 minimal path。

    生成策略：
    - 每條路徑隨機選 [min_arcs, max_arcs] 條弧
    - 確保同一組 pair 內路徑不完全重複
    - 不同 pair 之間自然會有弧重疊（越多 pair，重疊越多）

    參數
    ----
    n_arcs : int — 弧的總數
    K : int — pair 數
    m_per_pair : int — 每組 pair 的候選路徑數
    min_arcs_per_path, max_arcs_per_path : int — 每條路徑的弧數範圍
    seed : int or None

    回傳
    ----
    MPs_list : list of np.ndarray (bool), length K
    T_list : list of float, length K
    """
    rng = np.random.default_rng(seed)
    MPs_list = []

    for k in range(K):
        paths = set()
        MPs_k = np.zeros((m_per_pair, n_arcs), dtype=bool)

        for j in range(m_per_pair):
            # 隨機路徑長度
            n_path_arcs = rng.integers(min_arcs_per_path, max_arcs_per_path + 1)
            # 隨機選弧，確保不重複
            max_attempts = 100
            for _ in range(max_attempts):
                arc_indices = tuple(sorted(rng.choice(n_arcs, size=n_path_arcs, replace=False)))
                if arc_indices not in paths:
                    paths.add(arc_indices)
                    break
            MPs_k[j, list(arc_indices)] = True

        MPs_list.append(MPs_k)

    # 統一時間限制
    T_list = [50.0] * K

    return MPs_list, T_list


# ============================================================
# 預抽樣網路狀態
# ============================================================

def presample_network_states(binom, Cmax, mean_processing_time, psi, seed=None):
    """
    預抽樣 ψ 組網路狀態（capacity + lead_time），所有排列共用。

    參數
    ----
    binom : np.ndarray, shape (n,) — 各弧元件可靠度
    Cmax : np.ndarray, shape (n,) — 各弧最大容量
    mean_processing_time : float — lead time 期望值（exponential）
    psi : int — MC trial 數
    seed : int or None

    回傳
    ----
    capacities : np.ndarray, shape (ψ, n), int — 各 trial 的弧容量
    lead_times : np.ndarray, shape (ψ, n), float — 各 trial 的弧 lead time
    """
    rng = np.random.default_rng(seed)
    n = len(binom)

    # 容量：每條弧有 Cmax[i] 個元件，每個元件以機率 binom[i] 正常運作
    # capacities[t, i] = sum of Bernoulli(binom[i]) trials, count = Cmax[i]
    capacities = np.zeros((psi, n), dtype=int)
    for i in range(n):
        capacities[:, i] = rng.binomial(Cmax[i], binom[i], size=psi)

    # Lead time：各弧獨立 Exponential(mean_processing_time)
    lead_times = rng.exponential(mean_processing_time, size=(psi, n))

    return capacities, lead_times


# ============================================================
# 排列抽樣
# ============================================================

def sample_permutations(K, N_perm, seed=None):
    """
    抽樣 N_perm 個不重複排列。
    K≤7 時全列舉（K! ≤ 5040）；K>7 時隨機抽樣。

    參數
    ----
    K : int — pair 數
    N_perm : int — 要抽的排列數（全列舉時會自動調整為 K!）
    seed : int or None

    回傳
    ----
    perms : np.ndarray, shape (N_actual, K), int — 排列矩陣
    """
    import math
    K_fact = math.factorial(K)

    if K <= 7 or N_perm >= K_fact:
        # 全列舉
        perms = np.array(list(itertools.permutations(range(K))), dtype=int)
        print(f"  [排列] K={K}, K!={K_fact}, 全列舉 {len(perms)} 個排列")
        return perms

    # 隨機抽樣（去重）
    rng = np.random.default_rng(seed)
    perm_set = set()
    perms_list = []

    while len(perms_list) < N_perm:
        # 批次生成，加速去重
        batch_size = min(N_perm * 2, N_perm - len(perms_list) + 1000)
        for _ in range(batch_size):
            p = tuple(rng.permutation(K))
            if p not in perm_set:
                perm_set.add(p)
                perms_list.append(p)
                if len(perms_list) >= N_perm:
                    break

    perms = np.array(perms_list[:N_perm], dtype=int)
    print(f"  [排列] K={K}, K!={K_fact:.2e}, 隨機抽樣 {len(perms)} 個排列（去重）")
    return perms


# ============================================================
# D_max 計算（單個排列，供 multiprocessing 使用）
# ============================================================

# 全域變數：供 worker 讀取（避免 pickle 大型 array）
_shared_data = {}


def _init_worker(capacities, lead_times, MPs_list, T_list):
    """multiprocessing worker 初始化：將共用資料存入全域變數。"""
    _shared_data['capacities'] = capacities
    _shared_data['lead_times'] = lead_times
    _shared_data['MPs_list'] = MPs_list
    _shared_data['T_list'] = T_list


def _compute_dmax_for_perm(perm):
    """
    對單一排列 π，用所有預抽樣的網路狀態計算 D_max。

    參數
    ----
    perm : tuple/list of int — 排列

    回傳
    ----
    D_max_all : np.ndarray, shape (ψ, K) — 各 trial 各 pair 的 D_max
    """
    caps = _shared_data['capacities']
    lts = _shared_data['lead_times']
    MPs_list = _shared_data['MPs_list']
    T_list = _shared_data['T_list']
    psi = caps.shape[0]
    K = len(MPs_list)

    D_max_all = np.zeros((psi, K), dtype=int)
    order = list(perm)

    for t in range(psi):
        D_max_all[t] = greedy_allocation(caps[t], lts[t], MPs_list, T_list, order)

    return D_max_all


# ============================================================
# D_max 分布探索 → 以聯合成功率校準 d 範圍
# ============================================================

def calibrate_d_ranges_joint(capacities, lead_times, MPs_list, T_list,
                             n_sample_perms=200, target_mean_rjoint=0.50,
                             seed=None):
    """
    以 R_joint mean 為目標，校準 d 的 LHS 範圍。

    核心思路：
    1. 收集大量 D_max 向量 (n_sample_perms * psi, K)
    2. 對每個 pair k，取 D_max 非零值的中位數作為 ref_k
    3. 模擬 LHS 抽樣 d in [1, alpha * ref_k] 的 R_joint mean
    4. 二分搜尋 alpha 使得 LHS 的 R_joint mean ~ target_mean_rjoint

    參數
    ----
    capacities : np.ndarray, shape (psi, n)
    lead_times : np.ndarray, shape (psi, n)
    MPs_list : list of np.ndarray
    T_list : list of float
    n_sample_perms : int — 用來探索的隨機排列數
    target_mean_rjoint : float — 目標 R_joint 平均值（如 0.50）
    seed : int or None

    回傳
    ----
    d_ranges : np.ndarray, shape (K, 2) — [[d_min_k, d_max_k], ...]
    dmax_stats : dict — 校準過程的資訊
    """
    rng = np.random.default_rng(seed)
    K = len(MPs_list)
    psi = capacities.shape[0]

    # --- Step 1: 收集 D_max ---
    print(f"  [d 校準] 收集 D_max（{n_sample_perms} 排列 x {psi} trials）...")
    all_dmax = []
    for _ in range(n_sample_perms):
        perm = rng.permutation(K)
        D_max_trial = np.zeros((psi, K), dtype=int)
        for t in range(psi):
            D_max_trial[t] = greedy_allocation(
                capacities[t], lead_times[t], MPs_list, T_list, list(perm)
            )
        all_dmax.append(D_max_trial)

    all_dmax = np.concatenate(all_dmax, axis=0)
    N_total = all_dmax.shape[0]

    all_positive_rate = np.all(all_dmax > 0, axis=1).mean()
    print(f"  [d 校準] D_max 全 pair 非零率: {all_positive_rate:.4f}")

    # --- Step 2: 計算各 pair 的參考值 ref_k（非零中位數） ---
    ref = np.zeros(K, dtype=float)
    for k in range(K):
        vals = all_dmax[:, k]
        nonzero = vals[vals > 0]
        if len(nonzero) > 10:
            ref[k] = np.median(nonzero)
        else:
            ref[k] = max(1.0, np.percentile(vals, 95))
    print(f"  [d 校準] ref (各 pair 非零中位數): {ref.astype(int)}")

    # --- Step 3: 二分搜尋 alpha ---
    # d_ranges = [1, alpha * ref_k]
    # 模擬 LHS 的 R_joint mean，調 alpha 使其 ~ target
    N_d_cal = 200
    N_lhs_repeats = 3

    def simulate_lhs_mean_rjoint(alpha):
        """給定 alpha，用多組 LHS seed 平均估計 R_joint mean。"""
        d_max_arr = np.maximum(2, np.round(alpha * ref)).astype(int)
        all_rjoints = []
        for rep in range(N_lhs_repeats):
            sampler = LatinHypercube(d=K, seed=12345 + rep * 1000)
            samples_01 = sampler.random(n=N_d_cal)
            demands = np.zeros((N_d_cal, K), dtype=int)
            for kk in range(K):
                demands[:, kk] = np.round(
                    samples_01[:, kk] * (d_max_arr[kk] - 1) + 1
                ).astype(int)
            for di in range(N_d_cal):
                d_vec = demands[di]
                success = np.all(all_dmax >= d_vec[None, :], axis=1)
                all_rjoints.append(np.mean(success))
        return np.mean(all_rjoints)

    a_lo, a_hi = 0.01, 5.0
    r_lo = simulate_lhs_mean_rjoint(a_lo)
    r_hi = simulate_lhs_mean_rjoint(a_hi)
    print(f"  [d 校準] alpha={a_lo:.2f} -> R_mean={r_lo:.4f}")
    print(f"  [d 校準] alpha={a_hi:.2f} -> R_mean={r_hi:.4f}")

    # R_joint mean 隨 alpha 遞減（alpha 大 → d 大 → R 小）
    if r_lo <= target_mean_rjoint:
        alpha_opt = a_lo
    elif r_hi >= target_mean_rjoint:
        alpha_opt = a_hi
    else:
        for iteration in range(50):
            a_mid = (a_lo + a_hi) / 2
            r_mid = simulate_lhs_mean_rjoint(a_mid)
            if r_mid > target_mean_rjoint:
                a_lo = a_mid  # R 太高 → alpha 太小 → 增大
            else:
                a_hi = a_mid  # R 太低 → alpha 太大 → 減小
            if (a_hi - a_lo) < 0.01:
                break
        alpha_opt = (a_lo + a_hi) / 2

    final_mean = simulate_lhs_mean_rjoint(alpha_opt)
    print(f"  [d 校準] alpha_opt={alpha_opt:.4f}, R_joint mean={final_mean:.4f}")

    # --- Step 4: 設定 d_ranges ---
    d_max = np.maximum(2, np.round(alpha_opt * ref)).astype(int)
    d_ranges = np.column_stack([np.ones(K, dtype=int), d_max])

    print(f"  [d 校準] d_ranges:")
    for k in range(K):
        print(f"    Pair {k}: d in [1, {d_max[k]}]")

    dmax_stats = {
        'ref': ref,
        'alpha_opt': alpha_opt,
        'lhs_mean_rjoint': final_mean,
        'all_positive_rate': all_positive_rate,
        'medians': np.median(all_dmax, axis=0),
        'nonzero_rates': np.array([np.mean(all_dmax[:, k] > 0) for k in range(K)]),
    }

    return d_ranges, dmax_stats


# ============================================================
# LHS 生成需求向量 d
# ============================================================

def sample_demands_lhs(d_ranges, N_d, seed=None):
    """
    用 Latin Hypercube Sampling 在 d 空間均勻撒 N_d 組需求向量。

    參數
    ----
    d_ranges : np.ndarray, shape (K, 2) — 各 pair 的 d 範圍
    N_d : int — 要生成的 d 向量數
    seed : int or None

    回傳
    ----
    demands : np.ndarray, shape (N_d, K), int — 需求向量矩陣
    """
    K = d_ranges.shape[0]
    sampler = LatinHypercube(d=K, seed=seed)
    # LHS 產生 [0, 1]^K 的均勻樣本
    samples_01 = sampler.random(n=N_d)

    # 映射到各 pair 的 d 範圍，取整數
    demands = np.zeros((N_d, K), dtype=int)
    for k in range(K):
        d_lo, d_hi = d_ranges[k]
        demands[:, k] = np.round(samples_01[:, k] * (d_hi - d_lo) + d_lo).astype(int)

    print(f"  [LHS] 生成 {N_d} 組需求向量，K={K} 維")
    for k in range(K):
        print(f"    Pair {k+1}: d in [{demands[:, k].min()}, {demands[:, k].max()}]")

    return demands


# ============================================================
# 主流程：生成 training data
# ============================================================

def generate_dataset(MPs_list, T_list, binom, Cmax, mean_processing_time,
                     psi, N_perm, N_d,
                     val_ratio=0.125, test_ratio=0.125,
                     n_workers=None, seed=42,
                     output_dir=None):
    """
    完整的 data generation pipeline。

    參數
    ----
    psi : int — MC trial 數（預抽樣）
    N_perm : int — 排列抽樣數
    N_d : int — LHS 需求向量數
    val_ratio, test_ratio : float — validation / test 的比例
    n_workers : int or None — multiprocessing worker 數（None = cpu_count）
    seed : int
    output_dir : str or None — 輸出資料夾

    回傳
    ----
    dataset : dict — 包含 train/val/test splits
    """
    K = len(MPs_list)
    n_arcs = len(binom)
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    print("=" * 60)
    print(f"  Stage 1: MC Data Generation")
    print(f"  K={K}, n={n_arcs}, ψ={psi}, N_perm={N_perm}, N_d={N_d}")
    print(f"  Workers: {n_workers}")
    print("=" * 60)

    t_start = time.perf_counter()

    # --- Step 1: 預抽樣網路狀態 ---
    print("\n[Step 1] 預抽樣網路狀態...")
    capacities, lead_times = presample_network_states(
        binom, Cmax, mean_processing_time, psi, seed=seed
    )
    print(f"  capacities: {capacities.shape}, lead_times: {lead_times.shape}")

    # --- Step 2: 以 R_joint mean 為目標校準 d 範圍（目標 ~0.5） ---
    n_cal_perms = min(200, max(50, K * 10))
    print(f"\n[Step 2] d 校準（{n_cal_perms} 排列，目標 R_joint mean ~0.5）...")
    d_ranges, dmax_stats = calibrate_d_ranges_joint(
        capacities, lead_times, MPs_list, T_list,
        n_sample_perms=n_cal_perms, target_mean_rjoint=0.50, seed=seed
    )

    # --- Step 3: LHS 生成需求向量 ---
    print("\n[Step 3] LHS 生成需求向量...")
    demands = sample_demands_lhs(d_ranges, N_d, seed=seed)

    # --- Step 4: 抽樣排列 ---
    print("\n[Step 4] 抽樣排列...")
    perms = sample_permutations(K, N_perm, seed=seed)
    actual_N_perm = len(perms)

    # --- Step 5: 對每個排列計算 D_max（multiprocessing） ---
    print(f"\n[Step 5] 計算 D_max（{actual_N_perm} 個排列 × {psi} trials）...")
    t_mc = time.perf_counter()

    # 用 multiprocessing Pool
    perm_list = [tuple(p) for p in perms]

    with Pool(processes=n_workers,
              initializer=_init_worker,
              initargs=(capacities, lead_times, MPs_list, T_list)) as pool:
        # 分批處理，顯示進度
        batch_size = max(1, actual_N_perm // 20)
        D_max_results = []

        for i in range(0, actual_N_perm, batch_size):
            batch = perm_list[i:i + batch_size]
            batch_results = pool.map(_compute_dmax_for_perm, batch)
            D_max_results.extend(batch_results)
            done = min(i + batch_size, actual_N_perm)
            elapsed = time.perf_counter() - t_mc
            eta = elapsed / done * (actual_N_perm - done) if done > 0 else 0
            print(f"    進度: {done}/{actual_N_perm} "
                  f"({done/actual_N_perm*100:.1f}%, "
                  f"elapsed={elapsed:.1f}s, ETA={eta:.1f}s)")

    print(f"  D_max 計算完成，耗時 {time.perf_counter() - t_mc:.1f}s")

    # --- Step 6: 對每個 (π, d) 計算 R_joint ---
    print(f"\n[Step 6] 計算 R_joint（{actual_N_perm} × {N_d} = {actual_N_perm * N_d} 筆）...")
    t_rj = time.perf_counter()

    # 預分配結果 arrays
    total_samples = actual_N_perm * N_d
    all_perms = np.zeros((total_samples, K), dtype=int)
    all_demands = np.zeros((total_samples, K), dtype=int)
    all_R_joint = np.zeros(total_samples, dtype=float)

    idx = 0
    for p_idx in range(actual_N_perm):
        D_max_all = D_max_results[p_idx]  # shape (ψ, K)

        for d_idx in range(N_d):
            d = demands[d_idx]
            # R_joint = Pr(∀k: D_max^(k) ≥ d_k)
            joint_success = np.all(D_max_all >= d[None, :], axis=1)
            R_joint = np.mean(joint_success)

            all_perms[idx] = perms[p_idx]
            all_demands[idx] = d
            all_R_joint[idx] = R_joint
            idx += 1

    print(f"  R_joint 計算完成，耗時 {time.perf_counter() - t_rj:.1f}s")
    print(f"  R_joint 統計: mean={all_R_joint.mean():.4f}, "
          f"std={all_R_joint.std():.4f}, "
          f"min={all_R_joint.min():.4f}, max={all_R_joint.max():.4f}")

    # --- Step 7: Train / Val / Test split ---
    print(f"\n[Step 7] 資料分割...")
    rng = np.random.default_rng(seed + 999)
    indices = rng.permutation(total_samples)

    n_test = int(total_samples * test_ratio)
    n_val = int(total_samples * val_ratio)
    n_train = total_samples - n_val - n_test

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    dataset = {
        'train': {
            'perms': all_perms[train_idx],
            'demands': all_demands[train_idx],
            'R_joint': all_R_joint[train_idx],
        },
        'val': {
            'perms': all_perms[val_idx],
            'demands': all_demands[val_idx],
            'R_joint': all_R_joint[val_idx],
        },
        'test': {
            'perms': all_perms[test_idx],
            'demands': all_demands[test_idx],
            'R_joint': all_R_joint[test_idx],
        },
    }

    print(f"  Train: {n_train}, Val: {n_val}, Test: {n_test}")

    # --- Step 8: 儲存 ---
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        for split_name, split_data in dataset.items():
            path = os.path.join(output_dir, f'{split_name}.npz')
            np.savez_compressed(path,
                                perms=split_data['perms'],
                                demands=split_data['demands'],
                                R_joint=split_data['R_joint'])
            print(f"  [存檔] {path} ({len(split_data['R_joint'])} 筆)")

        # 存 metadata
        meta_path = os.path.join(output_dir, 'metadata.npz')
        np.savez(meta_path,
                 K=K, n_arcs=n_arcs, psi=psi,
                 N_perm=actual_N_perm, N_d=N_d,
                 d_ranges=d_ranges,
                 binom=binom, Cmax=Cmax,
                 mean_processing_time=mean_processing_time)
        print(f"  [存檔] {meta_path}")

    total_time = time.perf_counter() - t_start
    print(f"\n{'='*60}")
    print(f"  完成！總耗時: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  總樣本數: {total_samples}")
    print(f"{'='*60}")

    return dataset


# ============================================================
# CLI 入口
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stage 1: MC Data Generation')
    parser.add_argument('--K', type=int, default=3, help='Pair 數（3=使用 Paper 1 網路，>3=自動生成）')
    parser.add_argument('--psi', type=int, default=5000, help='MC trial 數')
    parser.add_argument('--N_perm', type=int, default=None, help='排列抽樣數（None=自動決定）')
    parser.add_argument('--N_d', type=int, default=50, help='LHS 需求向量數')
    parser.add_argument('--n_workers', type=int, default=None, help='CPU 核心數')
    parser.add_argument('--seed', type=int, default=42, help='隨機種子')
    parser.add_argument('--output_dir', type=str, default=None, help='輸出資料夾')
    # --percentile_hi 已移除，改用 calibrate_d_ranges_joint 自動校準
    args = parser.parse_args()

    # 網路設定
    if args.K == 3:
        MPs_list, T_list, binom, Cmax, mpt = get_network_K3()
        print("[網路] 使用 Paper 1 的 K=3 示範網路")
    else:
        # K>3: 放大弧數 + 縮短路徑長度 [2,4]
        # 原因：20 弧 + 長路徑(4~10) → lead time 超 T 且弧重疊嚴重 → D_max 幾乎全 0
        _, _, binom_base, Cmax_base, mpt = get_network_K3()
        n_arcs = max(30, args.K * 5)  # 弧數隨 K 放大
        binom = np.tile(binom_base, (n_arcs // len(binom_base)) + 1)[:n_arcs]
        Cmax = np.full(n_arcs, Cmax_base[0], dtype=int)
        MPs_list, T_list = generate_random_pairs(
            n_arcs, args.K, m_per_pair=10,
            min_arcs_per_path=2, max_arcs_per_path=4,
            seed=args.seed
        )
        print(f"[網路] 自動生成 K={args.K} 組 pair（n={n_arcs} 弧，路徑 2~4 弧，每組 10 條候選路徑）")

    # 自動決定 N_perm
    import math
    K = args.K
    if args.N_perm is None:
        K_fact = math.factorial(K)
        if K <= 7:
            N_perm = K_fact  # 全列舉
        elif K <= 10:
            N_perm = 30000
        elif K <= 15:
            N_perm = 50000
        else:
            N_perm = 80000
    else:
        N_perm = args.N_perm

    # 自動決定 output_dir
    if args.output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.output_dir = os.path.join(script_dir, 'data', f'K{K}_psi{args.psi}')

    generate_dataset(
        MPs_list, T_list, binom, Cmax, mpt,
        psi=args.psi,
        N_perm=N_perm,
        N_d=args.N_d,
        n_workers=args.n_workers,
        seed=args.seed,
        output_dir=args.output_dir,
    )
