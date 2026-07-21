"""
run_grouped_split.py
====================
以「grouped split by permutation」重訓 Transformer / MLP，並與樹模型同協議比較。

背景：已證實 random split 高估泛化——USNET K=10 上 RF 的 Spearman 由
0.9997（random）崩到 0.845（grouped）。本腳本補上 NN 的 grouped-split 版本：
若 Transformer 在「未見過的排列」上仍保持高精度，即完成
「樹模型=記憶、序列感知架構=真泛化」的證據鏈。

流程：
  1. split：合併 train/val/test → 依排列分組重切 80/10/10（同一排列的所有樣本
     只進同一子集）→ 寫入 data/<src>_grouped/（含 metadata 複製）
  2. train：subprocess 原始 train_surrogate.py 訓練 transformer + mlp
  3. eval ：NN 在「未見排列」測試集上的 MAE/R²/Spearman；並以同一切分重跑
     RF/XGBoost（同協議 apples-to-apples）→ grouped_results.json

用法（GPU）：
  python run_grouped_split.py --data usnet_K10 --stage all --device cuda
  python run_grouped_split.py --data nsfnet_K5 --stage all --device cuda   # 選配（K=5 排列少）
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
import shutil
import argparse
import subprocess
import numpy as np
from scipy.stats import spearmanr


# 訓練超參（與原 benchmark 跑法一致）
TRAIN_CFG = {
    'nsfnet_K5':  dict(batch_size=64,  epochs=200, patience=30),
    'usnet_K10':  dict(batch_size=256, epochs=200, patience=30),
}


def load_all(src_dir):
    Ps, Ds, Ys = [], [], []
    for name in ('train', 'val', 'test'):
        z = np.load(os.path.join(src_dir, f'{name}.npz'))
        Ps.append(z['perms'].astype(int))
        Ds.append(z['demands'].astype(np.float32))
        Ys.append(z['R_joint'].astype(np.float32))
    return np.vstack(Ps), np.vstack(Ds), np.concatenate(Ys)


def make_grouped_split(src_dir, dst_dir, seed=42, val_frac=0.10, test_frac=0.10):
    P, D, Y = load_all(src_dir)
    keys = np.array([hash(tuple(p)) for p in P])
    uniq = np.unique(keys)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_test = max(1, int(len(uniq) * test_frac))
    n_val = max(1, int(len(uniq) * val_frac))
    test_g = set(uniq[:n_test].tolist())
    val_g = set(uniq[n_test:n_test + n_val].tolist())

    masks = {
        'test':  np.array([k in test_g for k in keys]),
        'val':   np.array([k in val_g for k in keys]),
    }
    masks['train'] = ~(masks['test'] | masks['val'])

    os.makedirs(dst_dir, exist_ok=True)
    for name, m in masks.items():
        np.savez_compressed(os.path.join(dst_dir, f'{name}.npz'),
                            perms=P[m], demands=D[m], R_joint=Y[m])
        print(f"  [{name}] 樣本={m.sum()}（排列數={len(np.unique(keys[m]))}）")
    shutil.copy(os.path.join(src_dir, 'metadata.npz'), dst_dir)
    print(f"  [存檔] {dst_dir}（grouped split by permutation, seed={seed}）")


def train_nn(dst_dir, cfg, device):
    env = os.environ.copy()
    env['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    model_dir = os.path.join(dst_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    for m in ('transformer', 'mlp'):
        cmd = [sys.executable, os.path.join(ROOT, 'core', 'train_surrogate.py'),
               '--data_dir', dst_dir, '--model', m,
               '--epochs', str(cfg['epochs']), '--batch_size', str(cfg['batch_size']),
               '--lr', '1e-3', '--patience', str(cfg['patience']), '--save_dir', model_dir]
        if device:
            cmd += ['--device', device]
        print(f"\n[train {m}] {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=CODE, env=env)


def eval_all(dst_dir, seed=42, rf_subsample=300000):
    """NN（讀 train_surrogate 產生的 *_results.json）+ 樹模型（同切分重訓）。"""
    from baseline_surrogates import featurize, evaluate
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor

    results = {}
    for m in ('transformer', 'mlp'):
        p = os.path.join(dst_dir, 'models', f'{m}_results.json')
        if os.path.exists(p):
            r = json.load(open(p))
            results[m] = dict(MAE=r['test_MAE'], R2=r['test_R2'], Spearman=r['test_Spearman'])

    tr = np.load(os.path.join(dst_dir, 'train.npz'))
    te = np.load(os.path.join(dst_dir, 'test.npz'))
    d_max = np.maximum(tr['demands'].astype(np.float32).max(axis=0), 1.0)
    rng = np.random.default_rng(seed)
    for mode in ('flat', 'onehot'):
        Xtr = featurize(tr['perms'].astype(int), tr['demands'].astype(np.float32), d_max, mode)
        ytr = tr['R_joint'].astype(np.float32)
        Xte = featurize(te['perms'].astype(int), te['demands'].astype(np.float32), d_max, mode)
        yte = te['R_joint'].astype(np.float32)
        if len(ytr) > rf_subsample:
            idx = rng.choice(len(ytr), rf_subsample, replace=False)
            Xrf, yrf = Xtr[idx], ytr[idx]
        else:
            Xrf, yrf = Xtr, ytr
        rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                   n_jobs=-1, random_state=seed).fit(Xrf, yrf)
        results[f'RF_{mode}'] = evaluate(yte, rf.predict(Xte))
        xgb = XGBRegressor(n_estimators=600, max_depth=8, learning_rate=0.08,
                           subsample=0.8, colsample_bytree=0.8, tree_method='hist',
                           n_jobs=-1, random_state=seed, verbosity=0).fit(Xtr, ytr)
        results[f'XGB_{mode}'] = evaluate(yte, xgb.predict(Xte))

    out = os.path.join(dst_dir, 'grouped_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n=== grouped split（未見排列）測試結果 ===")
    for k, v in results.items():
        print(f"  {k:12s} MAE={v['MAE']:.4f} R2={v['R2']:.4f} Spearman={v['Spearman']:.4f}")
    print(f"[存檔] {out}")


def main():
    ap = argparse.ArgumentParser(description='grouped split by permutation, NN retrain')
    ap.add_argument('--data', required=True, help='data/ 下的來源資料夾（如 usnet_K10）')
    ap.add_argument('--stage', choices=['split', 'train', 'eval', 'all'], default='all')
    ap.add_argument('--device', default=None)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    src = os.path.join(HERE, 'data', args.data)
    dst = os.path.join(HERE, 'data', f'{args.data}_grouped')
    cfg = TRAIN_CFG.get(args.data, dict(batch_size=256, epochs=200, patience=30))

    if args.stage in ('split', 'all'):
        print(f"[split] {src} -> {dst}")
        make_grouped_split(src, dst, seed=args.seed)
    if args.stage in ('train', 'all'):
        train_nn(dst, cfg, args.device)
    if args.stage in ('eval', 'all'):
        eval_all(dst, seed=args.seed)


if __name__ == '__main__':
    main()
