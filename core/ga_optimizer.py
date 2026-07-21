"""
ga_optimizer.py
===============
Stage 3: Surrogate-Assisted Genetic Algorithm for Priority Ordering Optimization
（對應論文 §3.4 — Surrogate-assisted genetic algorithm）

用 GA 搜索最佳排列 π*，fitness = Transformer surrogate 預測的 R_joint。
最終 Top-N 解用 MC (ψ=10,000) 精確驗證。

【論文問題定義】（§2.2）
  π* = argmax_{π ∈ S_K} R_joint^(π)(d_1, ..., d_K)
  其中 R_joint 為 expensive black-box function（MC 估計需數分鐘），
  S_K 為 K 的所有排列空間（K! 種），窮舉不可行。

【GA 設計】（§3.4）
- Chromosome = permutation（整數編碼，§3.4.1）
- Crossover: PMX — Partially Mapped Crossover（§3.4.2）
- Mutation: swap / insert（§3.4.2）
- Selection: tournament selection
- Fitness = surrogate predicted R_joint（毫秒級，取代 MC 的分鐘級）
- 最終 Top-N 解用 MC (ψ=10,000) 精確驗證（§3.4.3）

用法
----
python ga_optimizer.py \\
    --model_path data/K10_psi5000/models/transformer_best.pt \\
    --demand 10 10 10 \\
    --psi_verify 10000

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
import torch
import time
import os
import argparse
import json

# 載入模型定義
from models.transformer_surrogate import PermutationTransformer
from models.mlp_baseline import PermutationMLP
# 載入 greedy_allocation（Paper 1 的 Algorithm 1，用於 MC 驗證）
from MC_QP_multi_st import greedy_allocation


# ============================================================
# GA 運算子（對應論文 §3.4.2 — Crossover and mutation operators）
# ============================================================

def pmx_crossover(parent1, parent2, rng):
    """
    Partially Mapped Crossover (PMX)（對應論文 §3.4.2）

    【演算法概述】
    PMX 是排列型染色體專用的交叉運算子，保證子代仍為合法排列。
    1. 隨機選兩個切點 cx1, cx2
    2. 將 parent1[cx1:cx2+1] 的中間段直接複製到 child
    3. 用 PMX mapping 解決 parent2 中的衝突值
    4. 剩餘空位填 parent2 的值

    【為什麼用 PMX】
    - 排列編碼不能用傳統交叉（會產生重複/遺漏）
    - PMX 保留了兩個 parent 的「相對順序資訊」
    - 是排列型 GA 的標準做法

    參數
    ----
    parent1, parent2 : np.ndarray, shape (K,) — 兩個排列 π₁, π₂
    rng : np.random.Generator — 隨機數產生器

    回傳
    ----
    child : np.ndarray, shape (K,) — 子代排列（保證是合法排列）
    """
    K = len(parent1)
    child = np.full(K, -1, dtype=int)  # -1 表示尚未填入

    # Step 1: 隨機選兩個切點
    cx1, cx2 = sorted(rng.choice(K, size=2, replace=False))

    # Step 2: 複製 parent1 的中間段 [cx1, cx2] 到 child
    child[cx1:cx2+1] = parent1[cx1:cx2+1]

    # Step 3: PMX mapping — 處理 parent2 中被中間段「擠掉」的值
    for i in range(cx1, cx2+1):
        if parent2[i] not in child:
            # parent2[i] 還沒出現在 child 中，需要找個位置放
            val = parent2[i]
            pos = i
            # 沿著 mapping chain 追蹤，直到找到中間段以外的空位
            while cx1 <= pos <= cx2:
                # parent1[pos] 佔了位置 pos，去找 parent1[pos] 在 parent2 中的位置
                mapped_val = parent1[pos]
                pos = np.where(parent2 == mapped_val)[0][0]
            child[pos] = val

    # Step 4: 剩餘空位直接填 parent2 的值
    for i in range(K):
        if child[i] == -1:
            child[i] = parent2[i]

    return child


def swap_mutation(perm, rng):
    """
    Swap Mutation（對應論文 §3.4.2 — Mutation operators）

    隨機交換排列中兩個位置的值。
    例：[0,1,2,3,4] → swap(1,3) → [0,3,2,1,4]

    這是最簡單的排列突變，改變量小（局部搜索）。
    """
    child = perm.copy()
    K = len(child)
    # 隨機選兩個不同位置
    i, j = rng.choice(K, size=2, replace=False)
    # 交換
    child[i], child[j] = child[j], child[i]
    return child


def insert_mutation(perm, rng):
    """
    Insert Mutation（對應論文 §3.4.2 — Mutation operators）

    隨機取一個元素，插入到另一個位置。
    例：[0,1,2,3,4] → 取位置1的'1'插到位置3 → [0,2,3,1,4]

    改變量比 swap 大（影響多個位置的相對順序），適合跳出局部最優。
    GA 中 swap 和 insert 各以 50% 機率使用。
    """
    child = perm.copy()
    K = len(child)
    # 隨機選「取出位置 i」和「插入位置 j」
    i, j = rng.choice(K, size=2, replace=False)
    val = child[i]
    child = np.delete(child, i)                       # 先刪除
    child = np.insert(child, j if j < i else j, val)  # 再插入
    return child


def tournament_selection(population, fitness, tournament_size, rng):
    """
    Tournament Selection（錦標賽選擇）

    從 population 中隨機抽 tournament_size 個個體，
    選 fitness 最高的作為 parent。

    【設計考量】
    - tournament_size 越大 → 選擇壓力越大 → 收斂越快但多樣性降低
    - 預設 tournament_size=5，在 pop_size=200 時是合理的平衡

    回傳
    ----
    selected_idx : int — 被選中的個體在 population 中的 index
    """
    # 隨機抽取 tournament_size 個候選人
    candidates = rng.choice(len(population), size=tournament_size, replace=False)
    # 選 fitness 最高的
    best = candidates[np.argmax(fitness[candidates])]
    return best


# ============================================================
# Surrogate Fitness Evaluation（批量）
# （對應論文 §3.4.3 — Surrogate fitness evaluation）
# ============================================================

def surrogate_fitness_batch(model, population, demand_normalized, device):
    """
    用 surrogate model 批量評估 population 的 fitness。

    【關鍵優勢】
    - MC 評估一個排列需數秒（ψ=10,000 次模擬）
    - Surrogate 評估整個族群（200 個排列）只需毫秒
    - 加速比 ≈ 10,000×，使 GA 在合理時間內完成

    參數
    ----
    model : nn.Module — 訓練好的 surrogate（Transformer 或 MLP）
    population : np.ndarray, shape (pop_size, K) — 排列族群
    demand_normalized : np.ndarray, shape (K,) — 正規化後的需求向量 d/d_max
    device : torch.device

    回傳
    ----
    fitness : np.ndarray, shape (pop_size,) — predicted R_joint（越高越好）
    """
    model.eval()
    with torch.no_grad():
        # 將排列轉為 tensor：shape (pop_size, K)
        perm_tensor = torch.from_numpy(population).long().to(device)
        # 需求向量複製成 batch：shape (pop_size, K)
        # 同一次 GA 搜索中，所有排列共用同一組需求 d
        demand_tensor = torch.from_numpy(
            np.tile(demand_normalized, (len(population), 1))
        ).float().to(device)
        # 前向傳播：model(π, d/d_max) → predicted R_joint
        pred = model(perm_tensor, demand_tensor)
    return pred.cpu().numpy()


# ============================================================
# MC 精確驗證
# （對應論文 §3.4.3 — MC verification of top-N solutions）
# ============================================================

def mc_verify(perms_to_verify, demand, MPs_list, T_list, binom, Cmax,
              mean_processing_time, psi, seed=None):
    """
    用 MC 模擬精確驗證一組排列的 R_joint。

    【為什麼需要 MC 驗證】
    - Surrogate 的預測可能有誤差（尤其在訓練資料稀疏的區域）
    - GA 找到的 Top-N 解需要用「真實」MC 模擬驗證
    - 只驗證少數解（Top-N ≈ 10），計算量可接受

    【MC 估計】
    R_joint^(π)(d) = (1/ψ) Σ_{t=1}^{ψ} I(∀k: D_max^(k,t) ≥ d_k)
    其中 D_max^(k,t) 由 Algorithm 1 (greedy_allocation) 計算

    參數
    ----
    perms_to_verify : np.ndarray, shape (N, K) — 要驗證的排列
    demand : np.ndarray, shape (K,) — 需求向量（原始值，非正規化）
    psi : int — MC trial 數（ψ=10,000 提供約 ±0.01 的 95% CI）

    回傳
    ----
    results : list of dict — 每個排列的驗證結果
        - 'perm': 排列
        - 'R_joint': MC 估計的聯合可靠度
        - 'CI_95': 95% 信賴區間 [lo, hi]
        - 'D_max_mean': 各 pair 的平均 D_max
    """
    rng = np.random.default_rng(seed)
    n = len(binom)       # 弧數
    K = len(MPs_list)    # pair 數

    # --- 預抽樣：所有排列共用同一批網路狀態 ---
    # 這是 Paper 2 的「預抽樣共用策略」（§3.1），確保公平比較
    # ξ_{i,t} ~ Binomial(Cmax_i, p_i) for t = 1,...,ψ
    capacities = np.zeros((psi, n), dtype=int)
    for i in range(n):
        capacities[:, i] = rng.binomial(Cmax[i], binom[i], size=psi)
    # ℓ_{i,t} ~ Exp(μ) for t = 1,...,ψ
    lead_times = rng.exponential(mean_processing_time, size=(psi, n))

    results = []
    for idx, perm in enumerate(perms_to_verify):
        # 對每個排列 π，跑 ψ 次 greedy_allocation
        D_max_all = np.zeros((psi, K), dtype=int)  # shape (ψ, K)
        for t in range(psi):
            # Algorithm 1: 在排列 π 下計算各 pair 的 D_max^(k)
            D_max_all[t] = greedy_allocation(
                capacities[t], lead_times[t], MPs_list, T_list, list(perm)
            )

        # --- 計算 R_joint ---
        # R_joint^(π)(d) = Pr(∀k: D_max^(k) ≥ d_k)
        # 等價於：每個 trial 檢查「所有 pair 都滿足需求」
        joint_success = np.all(D_max_all >= demand[None, :], axis=1)  # shape (ψ,), bool
        R_joint = np.mean(joint_success)  # MC 估計值

        # --- 95% 信賴區間（Wald interval）---
        # SE = sqrt(p(1-p)/n)，CI = p ± 1.96 × SE
        se = np.sqrt(R_joint * (1 - R_joint) / psi) if psi > 0 else 0
        ci_lo = max(0, R_joint - 1.96 * se)
        ci_hi = min(1, R_joint + 1.96 * se)

        results.append({
            'perm': perm.tolist(),                       # 排列 π
            'R_joint': float(R_joint),                   # MC 估計 R_joint
            'CI_95': [float(ci_lo), float(ci_hi)],       # 95% CI
            'D_max_mean': D_max_all.mean(axis=0).tolist(),  # 各 pair 平均 D_max
        })

    return results


# ============================================================
# GA 主流程（對應論文 §3.4 — Surrogate-Assisted GA）
# ============================================================

def run_ga(model, K, demand, d_max, device,
           pop_size=200, n_generations=500,
           crossover_rate=0.8, mutation_rate=0.3,
           tournament_size=5, elitism=5,
           seed=42):
    """
    Surrogate-Assisted GA 搜索最佳排列 π*。

    【GA 流程概述】
    1. 初始化：隨機產生 pop_size 個排列
    2. 重複 n_generations 代：
       a. 評估 fitness = surrogate(π, d)
       b. 菁英保留（elitism 個最佳直接進入下一代）
       c. Tournament selection → PMX crossover → swap/insert mutation
    3. 回傳歷史最佳排列 π*

    【超參數設定】（論文 §4.1）
    - pop_size=200: 族群大小
    - n_generations=500: 演化代數
    - crossover_rate=0.8: 80% 機率進行交叉
    - mutation_rate=0.3: 30% 機率進行突變
    - tournament_size=5: 錦標賽大小
    - elitism=5: 菁英保留數

    參數
    ----
    model : nn.Module — 訓練好的 surrogate
    K : int — pair 數
    demand : np.ndarray, shape (K,) — 需求向量（原始值）
    d_max : np.ndarray, shape (K,) — 正規化用的最大值
    pop_size : int — 族群大小（population size）
    n_generations : int — 世代數
    crossover_rate : float — 交叉機率 p_c
    mutation_rate : float — 突變機率 p_m
    tournament_size : int — 錦標賽大小
    elitism : int — 菁英保留數（直接進入下一代，不經選擇/交叉/突變）

    回傳
    ----
    best_perm : np.ndarray, shape (K,) — 歷史最佳排列 π*
    best_fitness : float — 最佳 predicted R_joint
    history : dict — 每代的 best/mean fitness 及 best perm
    final_population : np.ndarray — 最終代的族群
    final_fitness : np.ndarray — 最終代的 fitness
    """
    rng = np.random.default_rng(seed)

    # 需求正規化（跟 Stage 2 training 時一致）
    demand_normalized = (demand / d_max).astype(np.float32)

    # --- 初始化族群：pop_size 個隨機排列 ---
    population = np.array([rng.permutation(K) for _ in range(pop_size)])

    # --- 評估初始 fitness：surrogate(π, d) ---
    fitness = surrogate_fitness_batch(model, population, demand_normalized, device)

    # --- 紀錄 ---
    history = {
        'best_fitness': [],   # 每代最佳 fitness
        'mean_fitness': [],   # 每代平均 fitness
        'best_perm': [],      # 每代最佳排列
    }

    # 追蹤歷史最佳（across all generations）
    best_ever_fitness = -np.inf
    best_ever_perm = None

    print(f"\n  [GA] pop={pop_size}, gen={n_generations}, K={K}")
    print(f"  [GA] demand={demand.tolist()}")

    # --- GA 主迴圈：n_generations 代演化 ---
    for gen in range(n_generations):
        # === 紀錄當代統計 ===
        gen_best_idx = np.argmax(fitness)
        gen_best_fitness = fitness[gen_best_idx]
        gen_mean_fitness = fitness.mean()

        # 更新歷史最佳
        if gen_best_fitness > best_ever_fitness:
            best_ever_fitness = gen_best_fitness
            best_ever_perm = population[gen_best_idx].copy()

        history['best_fitness'].append(float(gen_best_fitness))
        history['mean_fitness'].append(float(gen_mean_fitness))
        history['best_perm'].append(population[gen_best_idx].tolist())

        # 每 50 代印一次進度
        if (gen + 1) % 50 == 0 or gen == 0:
            print(f"    Gen {gen+1:>4d}/{n_generations} | "
                  f"Best={gen_best_fitness:.6f} | "
                  f"Mean={gen_mean_fitness:.6f} | "
                  f"Best π={population[gen_best_idx].tolist()}")

        # === 菁英保留（Elitism）===
        # 將當代 fitness 最高的 elitism 個個體直接保留到下一代
        # 確保最佳解不會因隨機性而遺失
        elite_idx = np.argsort(fitness)[-elitism:]
        new_population = [population[i].copy() for i in elite_idx]

        # === 產生下一代（填滿剩餘 pop_size - elitism 個位置）===
        while len(new_population) < pop_size:
            # --- Selection: Tournament（選兩個 parent）---
            p1_idx = tournament_selection(population, fitness, tournament_size, rng)
            p2_idx = tournament_selection(population, fitness, tournament_size, rng)
            parent1 = population[p1_idx]
            parent2 = population[p2_idx]

            # --- Crossover: PMX（以機率 p_c 進行交叉）---
            if rng.random() < crossover_rate:
                child = pmx_crossover(parent1, parent2, rng)
            else:
                # 不交叉：直接複製 parent1
                child = parent1.copy()

            # --- Mutation: swap 或 insert（以機率 p_m 進行突變）---
            if rng.random() < mutation_rate:
                if rng.random() < 0.5:
                    child = swap_mutation(child, rng)     # 50% swap
                else:
                    child = insert_mutation(child, rng)   # 50% insert

            new_population.append(child)

        # 更新族群
        population = np.array(new_population[:pop_size])

        # === 評估新族群 fitness（毫秒級 surrogate 推論）===
        fitness = surrogate_fitness_batch(model, population, demand_normalized, device)

    # --- 最終結果 ---
    final_best_idx = np.argmax(fitness)
    if fitness[final_best_idx] > best_ever_fitness:
        best_ever_fitness = fitness[final_best_idx]
        best_ever_perm = population[final_best_idx].copy()

    print(f"\n  [GA 完成] Best π={best_ever_perm.tolist()}, "
          f"Predicted R_joint={best_ever_fitness:.6f}")

    return best_ever_perm, best_ever_fitness, history, population, fitness


# ============================================================
# CLI 主程式（串接 Stage 2 → Stage 3 的完整流程）
# ============================================================

def main():
    # --- 命令列參數 ---
    parser = argparse.ArgumentParser(description='Stage 3: Surrogate-Assisted GA')
    parser.add_argument('--model_path', type=str, required=True,
                        help='訓練好的 surrogate model (.pt)（Stage 2 產出）')
    parser.add_argument('--demand', type=int, nargs='+', required=True,
                        help='需求向量 d = (d_1, ..., d_K)，K 個整數')
    parser.add_argument('--pop_size', type=int, default=200,
                        help='GA 族群大小（§4.1）')
    parser.add_argument('--n_generations', type=int, default=500,
                        help='GA 世代數（§4.1）')
    parser.add_argument('--top_n', type=int, default=10,
                        help='MC 驗證的 Top-N 解（§3.4.3）')
    parser.add_argument('--psi_verify', type=int, default=10000,
                        help='MC 驗證的 trial 數 ψ（§3.4.3）')
    parser.add_argument('--device', type=str, default=None, help='Device')
    parser.add_argument('--seed', type=int, default=42, help='隨機種子')
    parser.add_argument('--output_dir', type=str, default=None, help='結果儲存路徑')
    args = parser.parse_args()

    # --- Device 自動偵測 ---
    if args.device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    # --- 載入 Stage 2 訓練好的 surrogate model ---
    print("=" * 60)
    print("  Stage 3: Surrogate-Assisted GA")
    print(f"  Model: {args.model_path}")
    print(f"  Device: {device}")
    print("=" * 60)

    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    K = checkpoint['K']             # pair 數
    d_max = checkpoint['d_max']     # demand 正規化基準
    model_type = checkpoint['model_type']  # 'transformer' 或 'mlp'

    # 需求向量（CLI 輸入）
    demand = np.array(args.demand, dtype=np.float32)
    assert len(demand) == K, f"需求向量長度 {len(demand)} ≠ K={K}"

    print(f"\n  K={K}, model={model_type}")
    print(f"  demand={demand.tolist()}")
    print(f"  d_max={d_max.tolist()}")

    # --- 重建模型架構並載入權重 ---
    if model_type == 'transformer':
        model_args = checkpoint.get('args', {})
        model = PermutationTransformer(
            K=K,
            d_model=model_args.get('d_model', 64),
            nhead=model_args.get('nhead', 4),
            num_layers=model_args.get('num_layers', 3),
            dim_feedforward=model_args.get('d_model', 64) * 2,
        )
    else:
        model = PermutationMLP(K=K)

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()  # 推論模式

    # ============================================================
    # Phase 1: GA 搜索（surrogate-assisted，毫秒級 fitness）
    # ============================================================
    print("\n[Phase 1: GA 搜索]")
    t_ga = time.perf_counter()

    best_perm, best_fitness, history, final_pop, final_fit = run_ga(
        model, K, demand, d_max, device,
        pop_size=args.pop_size,
        n_generations=args.n_generations,
        seed=args.seed,
    )

    ga_time = time.perf_counter() - t_ga
    print(f"  GA 耗時: {ga_time:.2f}s")

    # --- 從最終族群取 Top-N 解（去重）---
    top_idx = np.argsort(final_fit)[::-1]  # fitness 由高到低排序
    seen = set()
    top_perms = []
    for idx in top_idx:
        p_tuple = tuple(final_pop[idx])
        if p_tuple not in seen:
            seen.add(p_tuple)
            top_perms.append(final_pop[idx])
            if len(top_perms) >= args.top_n:
                break

    # 確保 GA 歷史最佳也在 Top-N 中
    best_tuple = tuple(best_perm)
    if best_tuple not in seen:
        top_perms.insert(0, best_perm)
    top_perms = np.array(top_perms)

    # 列出 Top-N 解及其 surrogate 預測
    print(f"\n  Top-{len(top_perms)} unique solutions for MC verification:")
    for i, p in enumerate(top_perms):
        fit = surrogate_fitness_batch(model, p[None], (demand / d_max).astype(np.float32), device)[0]
        print(f"    #{i+1}: π={p.tolist()}, predicted R_joint={fit:.6f}")

    # ============================================================
    # Phase 2: MC 精確驗證（§3.4.3）
    # ============================================================
    print(f"\n[Phase 2: MC 驗證] ψ={args.psi_verify}")

    # --- 載入網路參數（弧可靠度、容量等）---
    # 從 model_path 的上層目錄找 metadata.npz
    data_dir = os.path.dirname(os.path.dirname(args.model_path))
    meta_path = os.path.join(data_dir, 'metadata.npz')

    if os.path.exists(meta_path):
        meta = np.load(meta_path, allow_pickle=True)
        binom = meta['binom']                      # p_i: 弧元件可靠度
        Cmax = meta['Cmax']                        # Cmax_i: 弧最大容量
        mpt = float(meta['mean_processing_time'])  # μ: lead time 期望值
        print(f"  [載入網路參數] {meta_path}")
    else:
        # fallback: 用 K=3 的預設網路（Paper 1 原版）
        from data_generator import get_network_K3
        _, _, binom, Cmax, mpt = get_network_K3()
        print("  [警告] 找不到 metadata.npz，使用預設 K=3 網路參數")

    # --- 載入最小路徑集合 MPs_list ---
    # metadata 沒存 MPs_list（因為矩陣太大），需要重建
    if K == 3:
        # K=3: 使用 Paper 1 的原版網路
        from data_generator import get_network_K3
        MPs_list, T_list, _, _, _ = get_network_K3()
    else:
        # K>3: 使用 data_generator 的隨機生成函式重建
        # 參數與 Stage 1 一致：路徑 2~4 弧，m_per_pair=10
        from data_generator import generate_random_pairs
        n_arcs = len(binom)
        MPs_list, T_list = generate_random_pairs(
            n_arcs, K, m_per_pair=10,
            min_arcs_per_path=2, max_arcs_per_path=4,
            seed=args.seed
        )

    # --- 執行 MC 驗證 ---
    t_mc = time.perf_counter()
    mc_results = mc_verify(
        top_perms, demand.astype(int), MPs_list, T_list,
        binom, Cmax, mpt, psi=args.psi_verify, seed=args.seed + 100
    )
    mc_time = time.perf_counter() - t_mc

    # --- 輸出 MC 驗證結果 ---
    print(f"\n  MC 驗證結果（ψ={args.psi_verify}）:")
    print(f"  {'Rank':>4} {'π':>30} {'Predicted':>10} {'MC R_joint':>10} {'95% CI':>20}")
    for i, res in enumerate(mc_results):
        # 同時顯示 surrogate 預測值和 MC 真值，方便比較
        pred = surrogate_fitness_batch(
            model, np.array([res['perm']]),
            (demand / d_max).astype(np.float32), device
        )[0]
        ci_str = f"[{res['CI_95'][0]:.4f}, {res['CI_95'][1]:.4f}]"
        print(f"  {i+1:>4} {str(res['perm']):>30} {pred:>10.6f} {res['R_joint']:>10.6f} {ci_str:>20}")

    # 找 MC 驗證後的真正最佳解（可能跟 surrogate 排名不同）
    mc_best_idx = np.argmax([r['R_joint'] for r in mc_results])
    mc_best = mc_results[mc_best_idx]

    print(f"\n  MC 驗證最佳: π={mc_best['perm']}, R_joint={mc_best['R_joint']:.6f}")
    print(f"  MC 驗證耗時: {mc_time:.2f}s")

    # --- 儲存完整結果（JSON）---
    if args.output_dir is None:
        args.output_dir = os.path.join(data_dir, 'ga_results')

    os.makedirs(args.output_dir, exist_ok=True)
    results = {
        'demand': demand.tolist(),
        'K': K,
        'model_type': model_type,
        'ga_params': {
            'pop_size': args.pop_size,
            'n_generations': args.n_generations,
        },
        'ga_best_perm': best_perm.tolist(),                  # GA 找到的最佳 π
        'ga_best_predicted_R_joint': float(best_fitness),    # surrogate 預測值
        'ga_time_sec': ga_time,
        'mc_verification': mc_results,                       # 所有 Top-N 的 MC 結果
        'mc_best_perm': mc_best['perm'],                     # MC 驗證後的最佳 π*
        'mc_best_R_joint': mc_best['R_joint'],               # MC 驗證的 R_joint
        'mc_psi': args.psi_verify,
        'mc_time_sec': mc_time,
    }

    d_str = '_'.join(map(str, args.demand))
    result_path = os.path.join(args.output_dir, f'ga_d{d_str}.json')
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  [存檔] {result_path}")

    print(f"\n{'='*60}")
    print(f"  完成！Total time: GA={ga_time:.1f}s + MC={mc_time:.1f}s = {ga_time+mc_time:.1f}s")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
