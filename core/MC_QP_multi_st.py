"""
MC_QP_multi_st.py
=================
QP Flow Network Reliability — 多源匯對 (Multiple Source-Terminal Pairs)
Greedy Capacity Allocation + Average over Permutations

【論文對應】Paper 1 核心模擬程式；Paper 2 的 Stage 1 & Stage 3 會 import greedy_allocation()

模型說明（對應論文 §2.1 — QP flow network reliability）
--------
K 組源匯對 (s_k, t_k) 共享同一組弧（edges），容量為有限資源。
當多組源匯對同時使用網路時，容量分配的優先順序 π 會影響結果。

本程式實作三種分析：
  1. Fixed π (Greedy)        — 已知優先順序 π_0 = (1,2,...,K) 的確定性估計
  2. Average over all π      — 未知優先順序時的期望估計
  3. Sensitivity (Variance)  — 量化「排序影響有多大」

演算法（對應論文 Algorithm 1: Greedy Capacity Allocation）：
  - 對每次 MC trial，先抽樣網路狀態 (capacity ξ, lead time ℓ)
  - Fixed π: 按預設順序 greedy 分配容量
  - Average π: 對所有（或隨機抽樣的）排列跑 greedy，取平均

使用範例
--------
直接執行本檔即可跑 K=3 的示範網路（全列舉 3!=6 種排列）。

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
import matplotlib.pyplot as plt
import time
import csv
import os

# ============================================================
# 核心函式
# ============================================================

def greedy_allocation(capacity, lead_time, MPs_list, T_list, order):
    """
    Greedy 容量分配（對應論文 Algorithm 1: Greedy Capacity Allocation）

    【演算法概述】
    按照 order（排列 π）中的優先順序，依序為每組源匯對分配容量。
    每組 pair k 從其 m_k 條候選最小路徑中，選出 D_max 最大的那一條，
    然後扣除該路徑的 bottleneck 容量 ω。

    【論文符號對照】
    - π (order)          : 分配優先順序，π ∈ S_K（K 的所有排列）
    - ξ_i (capacity[i])  : 弧 e_i 的隨機容量（二項分布抽樣結果）
    - ℓ_i (lead_time[i]) : 弧 e_i 的隨機 lead time（指數分布抽樣結果）
    - P_{k,r} (MPs_list[k][r]) : 第 k 組 pair 的第 r 條最小路徑
    - T_k (T_list[k])    : 第 k 組 pair 的時間限制
    - ω_j (omega_j)      : 路徑 j 的 bottleneck 容量 = min(remaining_cap[arc_idx])
    - τ_j (R_j)          : 路徑 j 的剩餘時間 = T_k - Σ ℓ_i (for e_i ∈ path j)
    - d̄_j (d_j_max)      : 路徑級最大可達需求 = ω_j × ⌊τ_j⌋
    - D_max^(k)          : 第 k 組 pair 的最大可達需求 = max_j d̄_j

    參數
    ----
    capacity : np.ndarray, shape (n,), int
        當前 trial 的弧容量向量 ξ = (ξ_1, ..., ξ_n)（原始，未消耗）。
    lead_time : np.ndarray, shape (n,), float
        當前 trial 的弧 lead time 向量 ℓ = (ℓ_1, ..., ℓ_n)。
    MPs_list : list of np.ndarray
        MPs_list[k] 為第 k 組源匯對的最小路徑矩陣，shape (m_k, n), bool。
        MPs_list[k][j, i] = True 表示第 k 組的第 j 條路徑經過弧 e_i。
    T_list : list of float
        T_list[k] 為第 k 組源匯對的時間限制 T_k。
    order : list/tuple of int
        分配優先順序 π，例如 [0, 1, 2] 表示先分配第 0 組。

    回傳
    ----
    D_max_per_pair : np.ndarray, shape (K,), int
        各組源匯對的 D_max^(k)（在此排列 π 下）。
    """
    K = len(MPs_list)       # K: 源匯對數量
    n = len(capacity)       # n: 弧的數量
    remaining_cap = capacity.copy()  # 剩餘容量（會被逐步消耗，模擬共享弧競爭）
    D_max_per_pair = np.zeros(K, dtype=int)  # 儲存各 pair 的 D_max^(k)

    # --- Algorithm 1 主迴圈：按排列 π 的順序逐一處理每組 pair ---
    for k in order:
        # 取得第 k 組 pair 的最小路徑矩陣和時間限制
        MPs_k = MPs_list[k]   # shape (m_k, n)，m_k 條候選路徑的弧使用矩陣
        T_k = T_list[k]       # 第 k 組的時間限制 T_k
        m_k = MPs_k.shape[0]  # m_k: 第 k 組的候選路徑數（目前 m_k=10）

        # 初始化：追蹤最佳路徑（D_max 最大的那條）
        best_d_max = 0        # 目前找到的最大 d̄_j
        best_path_idx = -1    # 最佳路徑的 index
        best_omega = 0        # 最佳路徑的 bottleneck 容量 ω

        # --- 對第 k 組的每條候選路徑 P_{k,j}，用「剩餘容量」計算可達需求 ---
        for j in range(m_k):
            # arc_idx: 路徑 P_{k,j} 經過的弧的 boolean mask
            arc_idx = MPs_k[j, :]

            # ω_j = min(remaining_cap[e_i]) for e_i ∈ P_{k,j}
            # 論文符號：bottleneck 容量，路徑上剩餘容量最小的弧決定了流量上限
            omega_j = np.min(remaining_cap[arc_idx])
            if omega_j == 0:
                # 路徑已被之前的 pair 完全佔用（某弧容量歸零），跳過
                continue

            # τ_j = T_k - Σ ℓ_i (for e_i ∈ P_{k,j})
            # 論文符號：剩餘時間 = 時間限制減去路徑上所有弧的 lead time 總和
            R_j = T_k - np.sum(lead_time[arc_idx])
            if R_j < 1:
                # 剩餘時間不足一個時間單位，無法傳送任何需求，跳過
                continue

            # d̄_j = ω_j × ⌊τ_j⌋
            # 論文符號：路徑級最大可達需求 = bottleneck 容量 × 可用時間槽數
            d_j_max = omega_j * int(np.floor(R_j))

            # 更新最佳路徑（選 d̄_j 最大的）
            if d_j_max > best_d_max:
                best_d_max = d_j_max
                best_path_idx = j
                best_omega = omega_j

        # D_max^(k) = max_j d̄_j（第 k 組 pair 在此排列 π 下的最大可達需求）
        D_max_per_pair[k] = best_d_max

        # --- 扣除容量：最佳路徑上的弧各減去 ω（bottleneck 容量）---
        # 這模擬了「先到先用」的資源競爭：
        #   例如弧容量 [5, 3, 7] → bottleneck ω=3 → 扣完變 [2, 0, 4]
        # 注意：共享弧的衝突是「漸進消耗」，不是全有全無
        if best_path_idx >= 0 and best_omega > 0:
            arc_idx = MPs_list[k][best_path_idx, :]
            remaining_cap[arc_idx] = np.maximum(remaining_cap[arc_idx] - best_omega, 0)

    return D_max_per_pair


def simulate_multi_st(MPs_list, T_list, Cmax, binom, mean_processing_time,
                      psi, n_perm='all', fixed_order=None, seed=None):
    """
    多源匯對 MC 模擬（對應論文 §2.2 — Multi-pair QP reliability evaluation）

    【演算法概述】
    對每次 MC trial：
      1. 抽樣網路狀態：容量 ξ ~ Binomial(Cmax, p)，lead time ℓ ~ Exp(μ)
      2. Fixed π: 按預設順序跑一次 greedy_allocation
      3. Average π: 對所有（或隨機抽樣的）排列各跑一次 greedy_allocation，取平均

    【論文符號對照】
    - ψ (psi)            : MC 模擬次數（論文中的 ψ）
    - Cmax_i (Cmax[i])   : 弧 e_i 的最大容量（二項分布的 n 參數）
    - p_i (binom[i])     : 弧 e_i 各元件的可靠度（二項分布的 p 參數）
    - μ (mean_processing_time) : lead time 的期望值（指數分布的 mean）
    - S_K               : K 的所有排列集合，|S_K| = K!

    參數
    ----
    MPs_list : list of np.ndarray
        K 組源匯對各自的最小路徑矩陣。
    T_list : list of float
        K 組的時間限制 T_k。
    Cmax : np.ndarray, shape (n,), int
        各弧最大容量 Cmax_i。
    binom : np.ndarray, shape (n,), float
        各弧元件可靠度 p_i。
    mean_processing_time : float
        lead time 期望值 μ。
    psi : int
        MC 模擬次數 ψ。
    n_perm : int or 'all'
        排列數。'all' 表示全列舉 K!；整數表示隨機抽樣。
    fixed_order : list/tuple of int or None
        Fixed π 的順序，None 則用 (0,1,...,K-1)。
    seed : int or None
        隨機種子。

    回傳
    ----
    results : dict，包含：
        'D_max_fixed'  : np.ndarray, shape (psi, K)  — Fixed π 的 D_max
        'D_max_avg'    : np.ndarray, shape (psi, K)  — Average over π 的 D_max
        'D_max_worst'  : np.ndarray, shape (psi, K)  — Worst-case（最悲觀排列）
        'D_max_var'    : np.ndarray, shape (psi, K)  — Variance across π（排序敏感度）
        'perms_used'   : int  — 實際使用的排列數
    """
    if seed is not None:
        np.random.seed(seed)

    K = len(MPs_list)  # K: 源匯對數量
    n = len(Cmax)      # n: 弧的數量

    # 預設排列 π_0 = (0, 1, ..., K-1)
    if fixed_order is None:
        fixed_order = list(range(K))

    # --- 決定排列集合 S_K（或其子集）---
    # 計算 K! = K 的階乘
    K_factorial = 1
    for i in range(1, K + 1):
        K_factorial *= i

    if n_perm == 'all' or (isinstance(n_perm, int) and n_perm >= K_factorial):
        # 全列舉：K! 個排列（K≤7 時可行）
        all_perms = list(itertools.permutations(range(K)))
        actual_n_perm = len(all_perms)
        use_all = True
    else:
        # 隨機抽樣：K>7 時 K! 太大，改用抽樣
        actual_n_perm = int(n_perm)
        use_all = False

    # --- 儲存空間（ψ × K 矩陣）---
    D_max_fixed = np.zeros((psi, K), dtype=int)    # Fixed π 的 D_max^(k)
    D_max_avg = np.zeros((psi, K), dtype=float)    # Average over π 的 D_max^(k)（小數）
    D_max_worst = np.zeros((psi, K), dtype=int)    # Worst-case D_max^(k)
    D_max_var = np.zeros((psi, K), dtype=float)    # Variance of D_max^(k) across π

    # --- MC 主迴圈：ψ 次 trial ---
    for t in range(psi):
        # === Step 1: 抽樣網路狀態 ===
        # ξ_i ~ Binomial(Cmax_i, p_i)：每條弧的容量 = 可用元件數
        # 物理意義：弧有 Cmax_i 個平行元件，每個獨立以機率 p_i 運作
        capacity = np.array([
            np.sum(np.random.rand(Cmax[i]) <= binom[i])
            for i in range(n)
        ], dtype=int)
        # ℓ_i ~ Exp(μ)：每條弧的 lead time（傳輸延遲）
        lead_time = np.random.exponential(mean_processing_time, size=n)

        # === Step 2: Fixed π Greedy（用預設排列 π_0）===
        # 只跑一次 Algorithm 1，得到 D_max^(k) for all k
        D_max_fixed[t, :] = greedy_allocation(
            capacity, lead_time, MPs_list, T_list, fixed_order
        )

        # === Step 3: Average over Permutations（對所有排列取平均）===
        if use_all:
            # 全列舉模式：使用所有 K! 個排列
            perms_to_run = all_perms
        else:
            # 隨機抽樣模式：抽 n_perm 個不重複排列
            perms_set = set()
            perms_set.add(tuple(fixed_order))  # 確保 fixed order 被包含
            while len(perms_set) < actual_n_perm:
                perms_set.add(tuple(np.random.permutation(K)))
            perms_to_run = list(perms_set)

        # 收集所有排列的 D_max 結果：shape (|perms|, K)
        perm_results = np.zeros((len(perms_to_run), K), dtype=int)
        for p_idx, perm in enumerate(perms_to_run):
            # 每個排列 π 共用同一組 (capacity, lead_time)，只有分配順序不同
            perm_results[p_idx, :] = greedy_allocation(
                capacity, lead_time, MPs_list, T_list, perm
            )

        # 彙整統計量（across all permutations）
        D_max_avg[t, :] = np.mean(perm_results, axis=0)    # E_π[D_max^(k)]
        D_max_worst[t, :] = np.min(perm_results, axis=0)   # min_π D_max^(k)
        D_max_var[t, :] = np.var(perm_results, axis=0)     # Var_π[D_max^(k)]

    return {
        'D_max_fixed': D_max_fixed,
        'D_max_avg': D_max_avg,
        'D_max_worst': D_max_worst,
        'D_max_var': D_max_var,
        'perms_used': len(perms_to_run) if use_all else actual_n_perm,
    }


def reliability_curve(D_max_col):
    """
    從一組 D_max 樣本計算可靠度曲線 R(d)。
    （對應論文 §2.1 — 可靠度定義 R(d) = Pr(D_max ≥ d)）

    【論文符號】
    R(d) = Pr(D_max ≥ d) ≈ (1/ψ) Σ_{t=1}^{ψ} I(D_max^(t) ≥ d)

    D_max_col 可以是 int（Fixed π）或 float（Average π 時為小數）。

    回傳
    ----
    d_values : np.ndarray — d = 0, 1, 2, ..., max(D_max)
    R_hat    : np.ndarray — 對應的估計可靠度 R̂(d)
    """
    psi = len(D_max_col)  # MC trial 數 ψ
    d_max_upper = int(np.ceil(np.max(D_max_col)))  # D_max 的上界
    d_values = np.arange(0, d_max_upper + 1)
    # R̂(d) = (1/ψ) × #{trials where D_max ≥ d}
    R_hat = np.array([np.mean(D_max_col >= d) for d in d_values])
    return d_values, R_hat


def joint_reliability_curve(D_max_matrix):
    """
    聯合可靠度曲線（對應論文 §2.2 — Joint reliability）

    【論文符號】
    R_joint(d) = Pr(∀k: D_max^(k) ≥ d)
              = Pr(min_k D_max^(k) ≥ d)

    D_max_matrix : shape (psi, K)，每個 trial 每組 pair 的 D_max

    回傳
    ----
    d_values, R_hat — 同 reliability_curve
    """
    psi, K = D_max_matrix.shape
    # 每個 trial 取「最差 pair」的 D_max：min_k D_max^(k)
    D_max_joint = np.min(D_max_matrix, axis=1)
    return reliability_curve(D_max_joint)


# ============================================================
# 輸出與繪圖（結果呈現用，非論文核心演算法）
# ============================================================

def save_pair_csv(filepath, d_values, R_fixed, R_avg, R_worst):
    """儲存單組源匯對的三種可靠度（Fixed / Average / Worst π）到 CSV。"""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['d', 'R_fixed', 'R_avg', 'R_worst'])
        max_len = max(len(d_values), len(R_fixed), len(R_avg))
        for i in range(len(d_values)):
            row = [d_values[i]]
            row.append(f'{R_fixed[i]:.6f}' if i < len(R_fixed) else '')
            row.append(f'{R_avg[i]:.6f}' if i < len(R_avg) else '')
            row.append(f'{R_worst[i]:.6f}' if i < len(R_worst) else '')
            writer.writerow(row)
    print(f"  [已儲存] {filepath}")


def save_joint_csv(filepath, d_fixed, R_j_fixed, d_avg, R_j_avg, d_worst, R_j_worst):
    """儲存聯合可靠度 R_joint 到 CSV（Fixed / Average / Worst π 三種）。"""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    max_d = max(d_fixed[-1] if len(d_fixed) > 0 else 0,
                d_avg[-1] if len(d_avg) > 0 else 0,
                d_worst[-1] if len(d_worst) > 0 else 0)
    d_all = np.arange(0, max_d + 1)

    def lookup(d_vals, R_vals, d):
        """查表：若 d 在 R_vals 範圍內則回傳 R(d)，否則回傳 0"""
        if d < len(R_vals):
            return R_vals[d]
        return 0.0

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['d', 'R_joint_fixed', 'R_joint_avg', 'R_joint_worst'])
        for d in d_all:
            writer.writerow([
                d,
                f'{lookup(d_fixed, R_j_fixed, d):.6f}',
                f'{lookup(d_avg, R_j_avg, d):.6f}',
                f'{lookup(d_worst, R_j_worst, d):.6f}'
            ])
    print(f"  [已儲存] {filepath}")


def plot_multi_st_results(pair_curves, joint_curves, sensitivity_data, save_path=None):
    """
    繪製多源匯對結果圖表：
    - 上排：各 pair 的 R(d)，比較 Fixed π / Average π / Worst π
    - 下排左：聯合可靠度 R_joint(d)
    - 下排右：Ordering Sensitivity（排列變異性）
    """
    K = len(pair_curves)
    n_cols = max(K, 2)
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 8))

    # --- 上排：各 pair 的可靠度曲線 R^(k)(d) ---
    for k in range(K):
        ax = axes[0, k] if n_cols > 1 else axes[0]
        d_f, R_f = pair_curves[k]['fixed']
        d_a, R_a = pair_curves[k]['avg']
        d_w, R_w = pair_curves[k]['worst']
        ax.plot(d_f, R_f, 'b-', linewidth=1.5, label=f'Fixed $\\pi$')
        ax.plot(d_a, R_a, 'r--', linewidth=1.5, label=f'Average $\\pi$')
        ax.plot(d_w, R_w, 'g:', linewidth=1.2, label=f'Worst $\\pi$')
        ax.set_title(f'Pair {k+1}', fontsize=11)
        ax.set_xlabel('Demand $d$')
        ax.set_ylabel('$R(d)$')
        ax.legend(fontsize=8)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.3)
    # 隱藏多餘子圖
    for k in range(K, n_cols):
        axes[0, k].set_visible(False)

    # --- 下排左：聯合可靠度 R_joint(d) ---
    ax = axes[1, 0]
    d_jf, R_jf = joint_curves['fixed']
    d_ja, R_ja = joint_curves['avg']
    d_jw, R_jw = joint_curves['worst']
    ax.plot(d_jf, R_jf, 'b-', linewidth=1.5, label='Joint Fixed $\\pi$')
    ax.plot(d_ja, R_ja, 'r--', linewidth=1.5, label='Joint Average $\\pi$')
    ax.plot(d_jw, R_jw, 'g:', linewidth=1.2, label='Joint Worst $\\pi$')
    ax.set_title('Joint Reliability', fontsize=11)
    ax.set_xlabel('Demand $d$')
    ax.set_ylabel('$R_{joint}(d)$')
    ax.legend(fontsize=8)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)

    # --- 下排右：Sensitivity（排列造成的 D_max 變異性）---
    ax = axes[1, 1]
    for k in range(K):
        d_vals, mean_var = sensitivity_data[k]
        ax.plot(d_vals, mean_var, linewidth=1.5, label=f'Pair {k+1}')
    ax.set_title('Ordering Sensitivity (Mean Var across $\\pi$)', fontsize=11)
    ax.set_xlabel('Demand $d$')
    ax.set_ylabel('Avg Variance of $D_{max}$')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 隱藏多餘子圖
    for k in range(2, n_cols):
        axes[1, k].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [已儲存] 圖表 → {save_path}")
    plt.close()


# ============================================================
# 主程式：K=3 示範網路（Paper 1 原版網路，用於驗證 pipeline）
# ============================================================

if __name__ == '__main__':

    # --- 共享弧參數（20 條弧）---
    # binom[i] = p_i：弧 e_i 各元件的可靠度（Binomial 的 p 參數）
    binom = np.array([
        0.887563353, 0.862754756, 0.875297853, 0.884953836,
        0.894545163, 0.897964571, 0.877360776, 0.856931222,
        0.857464700, 0.862875413, 0.892035863, 0.862714109,
        0.890714241, 0.862176248, 0.896463181, 0.867499188,
        0.859829763, 0.862554193, 0.880802234, 0.873664442
    ])
    n_arcs = len(binom)         # n = 20 條弧
    Cmax_val = 9                # Cmax_i = 9：每條弧最多 9 個平行元件
    Cmax = np.full(n_arcs, Cmax_val, dtype=int)
    mean_processing_time = 10   # μ = 10：lead time 期望值

    # --- 定義 K=3 組源匯對（各自的最小路徑集合）---
    # 每組 pair 有 10 條候選最小路徑 (m_k = 10)
    # 矩陣中非零值表示該路徑經過該弧（轉為 bool 後 True = 經過）
    # Pair 1: 使用原始 30 條路徑的前 10 條（路徑 1~10）
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
    # Pair 2: 路徑 11~20
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
    # Pair 3: 路徑 21~30
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

    # 轉為 bool 矩陣：MPs_list[k][j, i] = True 表示 pair k 的路徑 j 經過弧 i
    MPs_list = [
        np.array(MPs_pair1_raw, dtype=bool),
        np.array(MPs_pair2_raw, dtype=bool),
        np.array(MPs_pair3_raw, dtype=bool),
    ]
    T_list = [50, 50, 50]  # 各組時間限制 T_k = 50
    K = len(MPs_list)      # K = 3 組源匯對

    # --- 模擬參數 ---
    psi = 10000             # MC 模擬次數 ψ = 10,000
    n_perm = 'all'          # K=3 → 全列舉 3! = 6 種排列（可改為整數如 200）
    fixed_order = [0, 1, 2] # 預設排列 π_0 = (0, 1, 2)

    # --- 執行模擬 ---
    print("=" * 60)
    print("QP Flow Network Reliability — Multi Source-Terminal Pairs")
    print(f"Greedy + Average over Permutations (n_perm={n_perm})")
    print("=" * 60)
    print(f"K = {K} 組源匯對, 弧數 n = {n_arcs}, Cmax = {Cmax_val}")
    print(f"psi = {psi}, T = {T_list}")
    print()

    t0 = time.perf_counter()
    results = simulate_multi_st(
        MPs_list, T_list, Cmax, binom, mean_processing_time,
        psi=psi, n_perm=n_perm, fixed_order=fixed_order, seed=42
    )
    sim_time = time.perf_counter() - t0
    print(f"[模擬完成] 耗時 {sim_time:.2f} 秒, 使用 {results['perms_used']} 種排列")

    # --- 計算各 pair 的可靠度曲線 R^(k)(d) ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pair_curves = []
    sensitivity_data = []

    for k in range(K):
        # R^(k)_fixed(d): Fixed π 的可靠度曲線
        d_f, R_f = reliability_curve(results['D_max_fixed'][:, k])
        # R^(k)_avg(d): Average over π 的可靠度曲線
        d_a, R_a = reliability_curve(results['D_max_avg'][:, k])
        # R^(k)_worst(d): Worst-case π 的可靠度曲線
        d_w, R_w = reliability_curve(results['D_max_worst'][:, k])

        # 確保長度一致（padding to common d range）
        max_d = max(d_f[-1], d_a[-1], d_w[-1])
        d_common = np.arange(0, max_d + 1)

        def extend_R(d_orig, R_orig, d_target):
            """將 R 曲線延伸到 d_target 的長度（超出範圍填 0）"""
            R_ext = np.zeros(len(d_target))
            for i, d in enumerate(d_target):
                if d < len(R_orig):
                    R_ext[i] = R_orig[d]
                else:
                    R_ext[i] = 0.0
            return R_ext

        R_f_ext = extend_R(d_f, R_f, d_common)
        R_a_ext = extend_R(d_a, R_a, d_common)
        R_w_ext = extend_R(d_w, R_w, d_common)

        pair_curves.append({
            'fixed': (d_common, R_f_ext),
            'avg': (d_common, R_a_ext),
            'worst': (d_common, R_w_ext),
        })

        # 儲存 CSV
        csv_path = os.path.join(script_dir, f'reliability_multi_st_pair{k+1}.csv')
        save_pair_csv(csv_path, d_common, R_f_ext, R_a_ext, R_w_ext)

        # Sensitivity: 每個 d 的平均 variance（Var_π[D_max^(k)]）
        mean_var_per_d = np.zeros(len(d_common))
        for i, d in enumerate(d_common):
            # D_max_var[:, k] 儲存的是每個 trial 的跨排列 variance
            # 取所有 trial 的平均 variance
            mean_var_per_d[i] = np.mean(results['D_max_var'][:, k])
        sensitivity_data.append((d_common, np.full(len(d_common), np.mean(results['D_max_var'][:, k]))))

    # --- 聯合可靠度 R_joint(d) = Pr(∀k: D_max^(k) ≥ d) ---
    d_jf, R_jf = joint_reliability_curve(results['D_max_fixed'])
    d_ja, R_ja = joint_reliability_curve(results['D_max_avg'])
    d_jw, R_jw = joint_reliability_curve(results['D_max_worst'])

    joint_csv_path = os.path.join(script_dir, 'reliability_multi_st_joint.csv')
    save_joint_csv(joint_csv_path, d_jf, R_jf, d_ja, R_ja, d_jw, R_jw)

    joint_curves = {
        'fixed': (d_jf, R_jf),
        'avg': (d_ja, R_ja),
        'worst': (d_jw, R_jw),
    }

    # --- 繪圖 ---
    png_path = os.path.join(script_dir, 'reliability_multi_st.png')
    plot_multi_st_results(pair_curves, joint_curves, sensitivity_data, save_path=png_path)

    # --- 摘要輸出 ---
    print("\n--- 各 Pair 可靠度摘要 (d=0, 20, 40, ...) ---")
    for k in range(K):
        d_c, R_f = pair_curves[k]['fixed']
        _, R_a = pair_curves[k]['avg']
        print(f"\nPair {k+1}:")
        print(f"  {'d':>6} {'R_fixed':>10} {'R_avg':>10} {'ΔR':>10}")
        for d in range(0, min(121, len(d_c)), 20):
            delta = R_f[d] - R_a[d]
            print(f"  {d:>6} {R_f[d]:>10.4f} {R_a[d]:>10.4f} {delta:>+10.4f}")

    print(f"\n--- 聯合可靠度摘要 ---")
    print(f"  {'d':>6} {'R_joint_fixed':>14} {'R_joint_avg':>14}")
    for d in range(0, min(81, len(R_jf)), 10):
        r_f = R_jf[d] if d < len(R_jf) else 0
        r_a = R_ja[d] if d < len(R_ja) else 0
        print(f"  {d:>6} {r_f:>14.4f} {r_a:>14.4f}")

    print(f"\n[完成] 總耗時 {time.perf_counter() - t0:.2f} 秒")
