"""
run_multiseed.py
================
合成實驗多 seed 重跑 + 統計檢定。

每個 seed 生成「獨立的合成網路實例」（generate_random_pairs 用該 seed）＋獨立 MC 抽樣，
跑完整三階段，記錄 Transformer 與 MLP 的 test MAE/R²/Spearman 與 GA 的 MC 驗證 R_joint。
--aggregate 再算跨 seed 的 mean±std，並對 Transformer vs MLP 做配對顯著性檢定
（paired t-test + Wilcoxon）。


K 選擇：預設 --Ks 5 10
  - K=5 ：便宜（~7min/seed）、有 ground truth → 展示 GA 穩定找回最優。
  - K=10：合成實驗中 Transformer 最佳化優勢首次出現的規模（原論文 +4.4%）→
          多 seed 提供跨獨立實例的統計顯著性檢定。
  - K=15/20 每 seed ~10–12hr，多 seed 過貴 → 維持單次，於文中註明計算限制。

用法：
  python run_multiseed.py --stage run --Ks 5 10 --seeds 42 43 44 45 46 --device cuda
  python run_multiseed.py --stage aggregate --Ks 5 10 --seeds 42 43 44 45 46
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
import subprocess
import numpy as np


from data_generator import get_network_K3, generate_random_pairs, generate_dataset
from MC_QP_multi_st import greedy_allocation

DATA_ROOT = os.path.join(HERE, 'data', 'multiseed')

CONFIG = {
    5:  dict(psi=10000, N_perm=120,   N_d=50, batch_size=64,  epochs=200, patience=30,
             pop_size=200, n_generations=500,  top_n=10, psi_eval=10000),
    10: dict(psi=5000,  N_perm=30000, N_d=50, batch_size=256, epochs=200, patience=30,
             pop_size=500, n_generations=1000, top_n=10, psi_eval=10000),
    15: dict(psi=5000,  N_perm=50000, N_d=50, batch_size=512, epochs=300, patience=40,
             pop_size=500, n_generations=1500, top_n=10, psi_eval=10000),
}


def build_synth_network(K, seed):
    """與原 run_experiments.py 一致的合成網路建法（每 seed 一個獨立實例）。"""
    _, _, binom_base, Cmax_base, mpt = get_network_K3()
    n_arcs = max(30, K * 5)
    binom = np.tile(binom_base, (n_arcs // len(binom_base)) + 1)[:n_arcs]
    Cmax = np.full(n_arcs, int(Cmax_base[0]), dtype=int)
    MPs_list, T_list = generate_random_pairs(
        n_arcs, K, m_per_pair=10, min_arcs_per_path=2, max_arcs_per_path=4, seed=seed)
    return MPs_list, T_list, binom, Cmax, mpt


def run_dir(K, seed):
    return os.path.join(DATA_ROOT, f'synth_K{K}_seed{seed}')


def run_one(K, seed, device):
    out = run_dir(K, seed)
    res_path = os.path.join(out, 'result.json')
    if os.path.exists(res_path):
        print(f"  [skip] K={K} seed={seed} 已完成")
        return
    os.makedirs(out, exist_ok=True)
    cfg = CONFIG[K]
    MPs, T, binom, Cmax, mpt = build_synth_network(K, seed)

    # ── Stage 1 ──
    generate_dataset(MPs, T, binom, Cmax, mpt,
                     psi=cfg['psi'], N_perm=cfg['N_perm'], N_d=cfg['N_d'],
                     seed=seed, output_dir=out)

    # ── Stage 2（subprocess，兩模型）──
    env = os.environ.copy(); env['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    model_dir = os.path.join(out, 'models'); os.makedirs(model_dir, exist_ok=True)
    for m in ('transformer', 'mlp'):
        cmd = [sys.executable, os.path.join(ROOT, 'core', 'train_surrogate.py'),
               '--data_dir', out, '--model', m,
               '--epochs', str(cfg['epochs']), '--batch_size', str(cfg['batch_size']),
               '--lr', '1e-3', '--patience', str(cfg['patience']), '--save_dir', model_dir]
        if device:
            cmd += ['--device', device]
        subprocess.run(cmd, check=True, cwd=CODE, env=env)

    # ── Stage 3（GA + MC 驗證，注入正確合成網路）──
    import torch
    from ga_optimizer import run_ga, mc_verify
    from models.transformer_surrogate import PermutationTransformer
    from models.mlp_baseline import PermutationMLP
    meta = np.load(os.path.join(out, 'metadata.npz'), allow_pickle=True)
    d_ranges = meta['d_ranges']
    demand = ((d_ranges[:, 0] + d_ranges[:, 1]) / 2).astype(int)
    dev = device or ('cuda' if torch.cuda.is_available() else
                     ('mps' if torch.backends.mps.is_available() else 'cpu'))

    result = dict(K=K, seed=seed)
    for m in ('transformer', 'mlp'):
        acc = json.load(open(os.path.join(model_dir, f'{m}_results.json')))
        ckpt = torch.load(os.path.join(model_dir, f'{m}_best.pt'),
                          map_location=dev, weights_only=False)
        d_max = ckpt['d_max']; a = ckpt.get('args', {})
        if m == 'transformer':
            model = PermutationTransformer(
                K=K, d_model=a.get('d_model', 64), nhead=a.get('nhead', 4),
                num_layers=a.get('num_layers', 3), dim_feedforward=a.get('d_model', 64) * 2)
        else:
            model = PermutationMLP(K=K)
        model.load_state_dict(ckpt['model_state_dict']); model = model.to(dev).eval()
        best_perm, _, _, final_pop, final_fit = run_ga(
            model, K, demand.astype(np.float32), d_max, dev,
            pop_size=cfg['pop_size'], n_generations=cfg['n_generations'], seed=seed)
        # top-N 去重
        seen, top = set(), []
        for idx in np.argsort(final_fit)[::-1]:
            pt = tuple(final_pop[idx])
            if pt not in seen:
                seen.add(pt); top.append(final_pop[idx])
                if len(top) >= cfg['top_n']:
                    break
        if tuple(best_perm) not in seen:
            top.insert(0, best_perm)
        mc = mc_verify(np.array(top), demand, MPs, T, binom, Cmax, mpt,
                       psi=cfg['psi_eval'], seed=seed + 100)
        mc_best = max(mc, key=lambda r: r['R_joint'])
        result[m] = dict(
            MAE=acc['test_MAE'], R2=acc['test_R2'], Spearman=acc['test_Spearman'],
            mc_R=mc_best['R_joint'], mc_CI=mc_best['CI_95'])

    result['gap_mc_R'] = result['transformer']['mc_R'] - result['mlp']['mc_R']
    result['gap_spearman'] = result['transformer']['Spearman'] - result['mlp']['Spearman']
    with open(res_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  [done] K={K} seed={seed}: "
          f"Trans mc_R={result['transformer']['mc_R']:.4f} vs "
          f"MLP {result['mlp']['mc_R']:.4f} (gap {result['gap_mc_R']:+.4f})")


def aggregate(Ks, seeds):
    from scipy import stats
    print(f"\n{'='*72}\n多 seed 統計（{len(seeds)} seeds: {seeds}）\n{'='*72}")
    for K in Ks:
        rows = [json.load(open(os.path.join(run_dir(K, s), 'result.json')))
                for s in seeds if os.path.exists(os.path.join(run_dir(K, s), 'result.json'))]
        if not rows:
            print(f"\nK={K}: 尚無結果"); continue
        n = len(rows)
        print(f"\n--- K={K} ({n} seeds) ---")
        for metric, key in (('Spearman', 'Spearman'), ('MAE', 'MAE'), ('MC R_joint', 'mc_R')):
            tv = np.array([r['transformer'][key] for r in rows])
            mv = np.array([r['mlp'][key] for r in rows])
            print(f"  {metric:12s} Trans {tv.mean():.4f}±{tv.std(ddof=1):.4f} | "
                  f"MLP {mv.mean():.4f}±{mv.std(ddof=1):.4f}")
        # 配對顯著性檢定（Transformer vs MLP）
        for metric, key in (('Spearman', 'Spearman'), ('MC R_joint', 'mc_R')):
            tv = np.array([r['transformer'][key] for r in rows])
            mv = np.array([r['mlp'][key] for r in rows])
            if n >= 2:
                t_p = stats.ttest_rel(tv, mv).pvalue
                try:
                    w_p = stats.wilcoxon(tv, mv).pvalue
                except ValueError:
                    w_p = float('nan')
                print(f"  paired test [{metric}]: Δ={ (tv-mv).mean():+.4f}, "
                      f"t-test p={t_p:.4f}, Wilcoxon p={w_p:.4f}")


def main():
    ap = argparse.ArgumentParser(description='multi-seed synthetic runs + stats')
    ap.add_argument('--stage', choices=['run', 'aggregate'], default='run')
    ap.add_argument('--Ks', type=int, nargs='+', default=[5, 10])
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    if args.stage == 'aggregate':
        aggregate(args.Ks, args.seeds)
        return
    print(f"### multi-seed run: Ks={args.Ks}, seeds={args.seeds} ###")
    for K in args.Ks:
        for s in args.seeds:
            t0 = time.perf_counter()
            print(f"\n>>> K={K} seed={s}")
            run_one(K, s, args.device)
            print(f"    ({(time.perf_counter()-t0)/60:.1f} min)")
    aggregate(args.Ks, args.seeds)


if __name__ == '__main__':
    main()
