"""
run_experiments.py
==================
一鍵執行 Paper 2 全部實驗：Stage 1 → Stage 2 → Stage 3

支援兩種模式：
  1. 全自動（預設）：依序跑 K = 5, 10, 15, 20
  2. 指定 K：只跑某個 K 值

用法
----
# 全自動（推薦，約一個下午）
python run_experiments.py --all

# 只跑 K=5（快速驗證）
python run_experiments.py --K 5

# 只跑 Stage 1（data generation）
python run_experiments.py --K 10 --stage 1

# 只跑 Stage 2（training），前提：Stage 1 已完成
python run_experiments.py --K 10 --stage 2

# 只跑 Stage 3（GA optimization），前提：Stage 2 已完成
python run_experiments.py --K 10 --stage 3

# 自訂參數
python run_experiments.py --K 10 --psi 10000 --N_d 100 --epochs 300

硬體需求
--------
- Stage 1 (MC data gen): CPU-bound, 建議 i9 (16+ cores)
- Stage 2 (Training): GPU-bound, 建議 RTX 4090
- Stage 3 (GA + MC verify): GPU inference + CPU MC

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
import sys
import time
import argparse
import json
import subprocess
import numpy as np

# ============================================================
# 實驗參數配置
# ============================================================

# 各 K 值的預設參數
EXPERIMENT_CONFIGS = {
    5: {
        'psi_train': 10000,     # Stage 1: MC trial 數
        'N_perm': None,         # None = 全列舉 (K! = 120)
        'N_d': 50,              # LHS demand 向量數
        'psi_eval': 10000,      # Stage 3: MC 驗證 trial 數
        'epochs': 200,          # Stage 2: 最大 epoch 數
        'batch_size': 64,
        'd_model': 64,
        'nhead': 4,
        'num_layers': 3,
        'patience': 30,
        'pop_size': 200,        # Stage 3: GA 族群大小
        'n_generations': 500,   # Stage 3: GA 世代數
        'top_n': 10,            # Stage 3: MC 驗證的 Top-N
    },
    10: {
        'psi_train': 5000,
        'N_perm': 30000,
        'N_d': 50,
        'psi_eval': 10000,
        'epochs': 200,
        'batch_size': 256,
        'd_model': 64,
        'nhead': 4,
        'num_layers': 3,
        'patience': 30,
        'pop_size': 500,
        'n_generations': 1000,
        'top_n': 10,
    },
    15: {
        'psi_train': 5000,
        'N_perm': 50000,
        'N_d': 50,
        'psi_eval': 10000,
        'epochs': 300,
        'batch_size': 512,
        'd_model': 128,
        'nhead': 8,
        'num_layers': 4,
        'patience': 40,
        'pop_size': 500,
        'n_generations': 1500,
        'top_n': 10,
    },
    20: {
        'psi_train': 5000,
        'N_perm': 80000,
        'N_d': 50,
        'psi_eval': 10000,
        'epochs': 300,
        'batch_size': 512,
        'd_model': 128,
        'nhead': 8,
        'num_layers': 4,
        'patience': 40,
        'pop_size': 1000,
        'n_generations': 2000,
        'top_n': 10,
    },
}

# Stage 3 用的需求向量（會在 Stage 1 完成後自動從 d_ranges 中間值生成）
# 也可以手動指定
DEFAULT_DEMANDS = None  # None = 從 metadata.npz 的 d_ranges 取中位數


def get_script_dir():
    """取得本腳本所在目錄。"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_command(cmd, description=""):
    """執行子程序指令，即時顯示輸出。"""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    # 設定環境變數：解決 OMP 重複連結 crash + Windows cp950 編碼問題
    import os as _os
    env = _os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["PYTHONIOENCODING"] = "utf-8"

    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=get_script_dir(), env=env)
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"\n  [FAIL] 失敗！returncode={result.returncode}，耗時 {elapsed:.1f}s")
        return False, elapsed
    else:
        print(f"\n  [OK] 完成！耗時 {elapsed:.1f}s ({elapsed/60:.1f} min)")
        return True, elapsed


# ============================================================
# Stage 1: MC Data Generation
# ============================================================

def run_stage1(K, config, n_workers=None, seed=42):
    """
    執行 Stage 1: 產生 MC training data。

    回傳
    ----
    success : bool
    data_dir : str — 資料存放路徑
    elapsed : float — 耗時（秒）
    """
    script_dir = get_script_dir()
    data_dir = os.path.join(script_dir, 'data', f'K{K}_psi{config["psi_train"]}')

    # 檢查是否已存在
    if os.path.exists(os.path.join(data_dir, 'train.npz')):
        print(f"\n  [WARN]  Stage 1 資料已存在: {data_dir}")
        print(f"      若要重新生成，請先手動刪除該資料夾。")
        return True, data_dir, 0.0

    cmd = [
        sys.executable, os.path.join(get_script_dir(), 'core', 'data_generator.py'),
        '--K', str(K),
        '--psi', str(config['psi_train']),
        '--N_d', str(config['N_d']),
        '--seed', str(seed),
        '--output_dir', data_dir,
    ]
    if config['N_perm'] is not None:
        cmd += ['--N_perm', str(config['N_perm'])]
    if n_workers is not None:
        cmd += ['--n_workers', str(n_workers)]

    success, elapsed = run_command(cmd, f"Stage 1: Data Generation (K={K})")
    return success, data_dir, elapsed


# ============================================================
# Stage 2: Surrogate Model Training
# ============================================================

def run_stage2(K, config, data_dir, model_type='transformer', device=None, seed=42):
    """
    執行 Stage 2: 訓練 surrogate model。

    回傳
    ----
    success : bool
    model_path : str — 模型存放路徑
    elapsed : float
    """
    model_dir = os.path.join(data_dir, 'models')
    model_path = os.path.join(model_dir, f'{model_type}_best.pt')

    # 檢查是否已存在
    if os.path.exists(model_path):
        print(f"\n  [WARN]  Stage 2 模型已存在: {model_path}")
        print(f"      若要重新訓練，請先手動刪除。")
        return True, model_path, 0.0

    cmd = [
        sys.executable, os.path.join(get_script_dir(), 'core', 'train_surrogate.py'),
        '--data_dir', data_dir,
        '--model', model_type,
        '--epochs', str(config['epochs']),
        '--batch_size', str(config['batch_size']),
        '--lr', '1e-3',
        '--d_model', str(config['d_model']),
        '--nhead', str(config['nhead']),
        '--num_layers', str(config['num_layers']),
        '--patience', str(config['patience']),
        '--save_dir', model_dir,
    ]
    if device is not None:
        cmd += ['--device', device]

    model_display = 'Transformer+MLP' if model_type == 'transformer' else 'MLP'
    success, elapsed = run_command(cmd, f"Stage 2: Train {model_display} (K={K})")
    return success, model_path, elapsed


# ============================================================
# Stage 3: GA Optimization
# ============================================================

def run_stage3(K, config, data_dir, model_path, demand=None, device=None, seed=42):
    """
    執行 Stage 3: GA 搜索 + MC 驗證。

    回傳
    ----
    success : bool
    elapsed : float
    """
    # 決定 demand 向量
    if demand is None:
        # 從 metadata.npz 的 d_ranges 取中位數
        meta_path = os.path.join(data_dir, 'metadata.npz')
        if os.path.exists(meta_path):
            meta = np.load(meta_path, allow_pickle=True)
            d_ranges = meta['d_ranges']
            demand = ((d_ranges[:, 0] + d_ranges[:, 1]) / 2).astype(int).tolist()
            print(f"\n  [自動決定 demand] 從 d_ranges 中位數: {demand}")
        else:
            # fallback: 全部設為 10
            demand = [10] * K
            print(f"\n  [警告] 找不到 metadata，使用 demand={demand}")

    demand_str = [str(d) for d in demand]

    cmd = [
        sys.executable, os.path.join(get_script_dir(), 'core', 'ga_optimizer.py'),
        '--model_path', model_path,
        '--demand', *demand_str,
        '--pop_size', str(config['pop_size']),
        '--n_generations', str(config['n_generations']),
        '--top_n', str(config['top_n']),
        '--psi_verify', str(config['psi_eval']),
        '--seed', str(seed),
    ]
    if device is not None:
        cmd += ['--device', device]

    success, elapsed = run_command(cmd, f"Stage 3: GA Optimization (K={K}, d={demand})")
    return success, elapsed


# ============================================================
# Exhaustive Ground Truth（僅限 K <= 7）
# ============================================================

def run_exhaustive_ground_truth(K, data_dir, demand=None, psi=10000, seed=123):
    """
    對 K <= 7，全列舉所有排列計算 MC ground truth。
    用來評估 surrogate + GA 的優化品質。
    """
    import math
    if K > 7:
        print(f"  [跳過] K={K} > 7，排列空間 {math.factorial(K)} 太大，不做全列舉")
        return None

    print(f"\n{'='*60}")
    print(f"  Exhaustive Ground Truth (K={K}, ψ={psi})")
    print(f"{'='*60}")

    # 匯入必要模組
    sys.path.insert(0, get_script_dir())
    from data_generator import get_network_K3, generate_random_pairs, presample_network_states
    from MC_QP_multi_st import greedy_allocation
    import itertools

    # 載入網路（與 data_generator.py CLI 一致）
    if K == 3:
        MPs_list, T_list, binom, Cmax, mpt = get_network_K3()
    else:
        _, _, binom_base, Cmax_base, mpt = get_network_K3()
        n_arcs = max(30, K * 5)
        binom = np.tile(binom_base, (n_arcs // len(binom_base)) + 1)[:n_arcs]
        Cmax = np.full(n_arcs, Cmax_base[0], dtype=int)
        MPs_list, T_list = generate_random_pairs(
            n_arcs, K, m_per_pair=10,
            min_arcs_per_path=2, max_arcs_per_path=4, seed=42
        )

    # 決定 demand
    if demand is None:
        meta_path = os.path.join(data_dir, 'metadata.npz')
        if os.path.exists(meta_path):
            meta = np.load(meta_path, allow_pickle=True)
            d_ranges = meta['d_ranges']
            demand = ((d_ranges[:, 0] + d_ranges[:, 1]) / 2).astype(int)
        else:
            demand = np.array([10] * K)

    demand = np.array(demand)
    print(f"  demand = {demand.tolist()}")

    # 預抽樣
    caps, lts = presample_network_states(binom, Cmax, mpt, psi, seed=seed)

    # 全列舉
    all_perms = list(itertools.permutations(range(K)))
    results = {}
    t0 = time.perf_counter()

    for pi, perm in enumerate(all_perms):
        D_max_all = np.zeros((psi, K), dtype=int)
        for t in range(psi):
            D_max_all[t] = greedy_allocation(caps[t], lts[t], MPs_list, T_list, list(perm))
        joint = np.all(D_max_all >= demand, axis=1)
        R = float(np.mean(joint))
        results[perm] = R

        if (pi + 1) % max(1, len(all_perms) // 10) == 0:
            print(f"    進度: {pi+1}/{len(all_perms)}")

    elapsed = time.perf_counter() - t0

    # 排序輸出
    sorted_results = sorted(results.items(), key=lambda x: -x[1])
    print(f"\n  全列舉結果（Top 10 / {len(all_perms)}）:")
    for i, (perm, R) in enumerate(sorted_results[:10]):
        print(f"    #{i+1}: π={list(perm)}, R_joint={R:.6f}")

    print(f"  最差: π={list(sorted_results[-1][0])}, R_joint={sorted_results[-1][1]:.6f}")
    print(f"  Best/Worst ratio: {sorted_results[0][1] / max(sorted_results[-1][1], 1e-10):.1f}x")
    print(f"  耗時: {elapsed:.1f}s")

    # 存檔
    gt_path = os.path.join(data_dir, 'ground_truth.json')
    gt_data = {
        'K': K,
        'demand': demand.tolist(),
        'psi': psi,
        'best_perm': list(sorted_results[0][0]),
        'best_R_joint': sorted_results[0][1],
        'worst_perm': list(sorted_results[-1][0]),
        'worst_R_joint': sorted_results[-1][1],
        'all_results': [{'perm': list(p), 'R_joint': r} for p, r in sorted_results],
    }
    with open(gt_path, 'w') as f:
        json.dump(gt_data, f, indent=2)
    print(f"  [存檔] {gt_path}")

    return gt_data


# ============================================================
# 單個 K 值的完整實驗
# ============================================================

def run_single_K(K, config, stage=None, model_types=None,
                 n_workers=None, device=None, demand=None, seed=42):
    """
    執行單個 K 值的完整實驗流程。

    參數
    ----
    stage : int or None — 只跑某個 stage (1, 2, 3)，None = 全部
    model_types : list of str — 要訓練的模型 ['transformer', 'mlp']
    """
    if model_types is None:
        model_types = ['transformer', 'mlp']

    print(f"\n{'#'*60}")
    print(f"  實驗: K = {K}")
    print(f"  Config: {json.dumps(config, indent=2)}")
    print(f"{'#'*60}")

    timing = {}
    data_dir = None

    # --- Stage 1 ---
    if stage is None or stage == 1:
        success, data_dir, elapsed = run_stage1(K, config, n_workers=n_workers, seed=seed)
        timing['stage1'] = elapsed
        if not success:
            print(f"\n  [FAIL] Stage 1 失敗，中止 K={K} 實驗")
            return False, timing
    else:
        # 推算 data_dir
        data_dir = os.path.join(get_script_dir(), 'data', f'K{K}_psi{config["psi_train"]}')

    # --- Stage 2 ---
    if stage is None or stage == 2:
        for model_type in model_types:
            success, model_path, elapsed = run_stage2(
                K, config, data_dir, model_type=model_type, device=device, seed=seed
            )
            timing[f'stage2_{model_type}'] = elapsed
            if not success:
                print(f"\n  [FAIL] Stage 2 ({model_type}) 失敗")
                return False, timing

    # --- Stage 3 ---
    if stage is None or stage == 3:
        for model_type in model_types:
            model_path = os.path.join(data_dir, 'models', f'{model_type}_best.pt')
            if not os.path.exists(model_path):
                print(f"\n  [WARN]  跳過 Stage 3 ({model_type}): 模型不存在 {model_path}")
                continue

            success, elapsed = run_stage3(
                K, config, data_dir, model_path, demand=demand, device=device, seed=seed
            )
            timing[f'stage3_{model_type}'] = elapsed
            if not success:
                print(f"\n  [FAIL] Stage 3 ({model_type}) 失敗")

    # --- Exhaustive Ground Truth（K <= 7）---
    if (stage is None or stage == 3) and K <= 7:
        gt = run_exhaustive_ground_truth(K, data_dir, demand=demand)
        if gt:
            timing['ground_truth'] = 0  # 已包含在 print 中

    # --- 總結 ---
    print(f"\n{'='*60}")
    print(f"  K={K} 實驗完成！耗時摘要:")
    for stage_name, t in timing.items():
        if t > 0:
            print(f"    {stage_name}: {t:.1f}s ({t/60:.1f} min)")
    total = sum(timing.values())
    print(f"    合計: {total:.1f}s ({total/60:.1f} min)")
    print(f"{'='*60}")

    # 存 timing
    if data_dir:
        timing_path = os.path.join(data_dir, 'timing.json')
        with open(timing_path, 'w') as f:
            json.dump(timing, f, indent=2)

    return True, timing


# ============================================================
# CLI 主程式
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Paper 2: 一鍵執行全部實驗',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  python run_experiments.py --all                  # 全自動跑 K=5,10,15,20
  python run_experiments.py --K 5                  # 只跑 K=5
  python run_experiments.py --K 10 --stage 1       # 只跑 K=10 的 Stage 1
  python run_experiments.py --K 10 --stage 2       # 只跑 K=10 的 Stage 2
  python run_experiments.py --K 10 --stage 3       # 只跑 K=10 的 Stage 3
  python run_experiments.py --K 10 --transformer_only  # 只訓練 Transformer
  python run_experiments.py --test                 # 快速測試（K=3, 小參數）
        """
    )
    parser.add_argument('--all', action='store_true',
                        help='跑所有 K 值 (5, 10, 15, 20)')
    parser.add_argument('--K', type=int, default=None,
                        help='指定 K 值')
    parser.add_argument('--stage', type=int, default=None, choices=[1, 2, 3],
                        help='只跑某個 stage')
    parser.add_argument('--test', action='store_true',
                        help='快速測試模式（K=3, 小參數）')

    # 可覆蓋的參數
    parser.add_argument('--psi', type=int, default=None, help='覆蓋 psi_train')
    parser.add_argument('--N_perm', type=int, default=None, help='覆蓋 N_perm')
    parser.add_argument('--N_d', type=int, default=None, help='覆蓋 N_d')
    parser.add_argument('--epochs', type=int, default=None, help='覆蓋 epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='覆蓋 batch_size')
    parser.add_argument('--demand', type=int, nargs='+', default=None,
                        help='指定 demand 向量')
    parser.add_argument('--n_workers', type=int, default=None,
                        help='CPU 核心數（Stage 1）')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu/mps)')
    parser.add_argument('--seed', type=int, default=42, help='隨機種子')
    parser.add_argument('--transformer_only', action='store_true',
                        help='只訓練 Transformer（不跑 MLP）')

    args = parser.parse_args()

    # 決定模型列表
    model_types = ['transformer'] if args.transformer_only else ['transformer', 'mlp']

    # --- 快速測試模式 ---
    if args.test:
        print("\n[TEST] 快速測試模式 (K=3, 小參數)")
        config = {
            'psi_train': 1000, 'N_perm': None, 'N_d': 10, 'psi_eval': 5000,
            'epochs': 50, 'batch_size': 32, 'd_model': 32, 'nhead': 2,
            'num_layers': 2, 'patience': 20, 'pop_size': 20,
            'n_generations': 50, 'top_n': 5,
        }
        run_single_K(3, config, stage=args.stage, model_types=model_types,
                     n_workers=args.n_workers, device=args.device,
                     demand=args.demand, seed=args.seed)
        return

    # --- 單一 K ---
    if args.K is not None:
        K = args.K
        if K in EXPERIMENT_CONFIGS:
            config = EXPERIMENT_CONFIGS[K].copy()
        else:
            # 自動內插參數
            print(f"\n  [WARN]  K={K} 不在預設配置中，使用 K=10 的參數")
            config = EXPERIMENT_CONFIGS[10].copy()

        # 覆蓋使用者指定的參數
        if args.psi is not None: config['psi_train'] = args.psi
        if args.N_perm is not None: config['N_perm'] = args.N_perm
        if args.N_d is not None: config['N_d'] = args.N_d
        if args.epochs is not None: config['epochs'] = args.epochs
        if args.batch_size is not None: config['batch_size'] = args.batch_size

        run_single_K(K, config, stage=args.stage, model_types=model_types,
                     n_workers=args.n_workers, device=args.device,
                     demand=args.demand, seed=args.seed)
        return

    # --- 全自動 ---
    if args.all:
        print("\n[RUN] 全自動模式: K = 5, 10, 15, 20")
        all_timing = {}
        t_total = time.perf_counter()

        for K in [5, 10, 15, 20]:
            config = EXPERIMENT_CONFIGS[K].copy()
            if args.psi is not None: config['psi_train'] = args.psi
            if args.N_d is not None: config['N_d'] = args.N_d
            if args.epochs is not None: config['epochs'] = args.epochs

            success, timing = run_single_K(
                K, config, stage=args.stage, model_types=model_types,
                n_workers=args.n_workers, device=args.device,
                demand=args.demand, seed=args.seed
            )
            all_timing[f'K{K}'] = timing

            if not success:
                print(f"\n  [WARN]  K={K} 失敗，繼續下一個...")

        total_elapsed = time.perf_counter() - t_total
        print(f"\n{'#'*60}")
        print(f"  全部實驗完成！總耗時: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
        print(f"{'#'*60}")

        # 存總結
        summary_path = os.path.join(get_script_dir(), 'data', 'experiment_summary.json')
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, 'w') as f:
            json.dump({
                'total_time_sec': total_elapsed,
                'timings': all_timing,
                'model_types': model_types,
            }, f, indent=2)
        print(f"  [存檔] {summary_path}")
        return

    # 沒指定任何模式
    parser.print_help()
    print("\n請指定 --all、--K <value> 或 --test")


if __name__ == '__main__':
    main()
