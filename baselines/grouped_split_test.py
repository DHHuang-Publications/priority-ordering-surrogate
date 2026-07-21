"""
grouped_split_test.py
=====================
驗證以下假設：
「隨機切分下 RF 這類記憶型模型看起來很強，是因為測試排列在訓練中已見過（同排列不同需求）；
 改用 grouped split by permutation（測試集只含未見過的排列）後，記憶失效，
 序列感知模型（Transformer）的泛化優勢才顯現。」

作法：合併 train/val/test → 依「排列」分組重切（10% 排列為測試組，其所有樣本進測試集）
→ 重訓 RF/XGBoost 比較 random split vs grouped split 的表現落差。
（NN 的 grouped split 重訓成本高，若樹模型假設成立。）

用法：
  python grouped_split_test.py --data_dir data/usnet_K10 --subsample 300000
  python grouped_split_test.py --data_dir data/nsfnet_K5
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

import json
import time
import argparse
import numpy as np
from scipy.stats import spearmanr

from baseline_surrogates import featurize, evaluate  # 重用同一特徵化/指標


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--subsample', type=int, default=None)
    ap.add_argument('--test_frac', type=float, default=0.10)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor

    data_dir = args.data_dir if os.path.isabs(args.data_dir) else os.path.join(HERE, args.data_dir)

    # 合併全部樣本
    Ps, Ds, Ys = [], [], []
    for name in ('train', 'val', 'test'):
        z = np.load(os.path.join(data_dir, f'{name}.npz'))
        Ps.append(z['perms'].astype(int))
        Ds.append(z['demands'].astype(np.float32))
        Ys.append(z['R_joint'].astype(np.float32))
    P, D, Y = np.vstack(Ps), np.vstack(Ds), np.concatenate(Ys)
    K = P.shape[1]
    d_max = np.maximum(D.max(axis=0), 1.0)

    # 依排列分組
    keys = np.array([hash(tuple(p)) for p in P])
    uniq = np.unique(keys)
    rng = np.random.default_rng(args.seed)
    test_groups = set(rng.choice(uniq, size=max(1, int(len(uniq) * args.test_frac)),
                                 replace=False).tolist())
    te_mask = np.array([k in test_groups for k in keys])
    tr_mask = ~te_mask
    print(f"data={os.path.basename(data_dir)} K={K} 全樣本={len(Y)} "
          f"排列數={len(uniq)}（測試組 {len(test_groups)} 個排列，樣本 {te_mask.sum()}）")

    rng2 = np.random.default_rng(args.seed + 1)
    results = {}
    for mode in ('flat', 'onehot'):
        Xtr = featurize(P[tr_mask], D[tr_mask], d_max, mode)
        ytr = Y[tr_mask]
        Xte = featurize(P[te_mask], D[te_mask], d_max, mode)
        yte = Y[te_mask]

        if args.subsample and len(ytr) > args.subsample:
            idx = rng2.choice(len(ytr), args.subsample, replace=False)
            Xtr_rf, ytr_rf = Xtr[idx], ytr[idx]
        else:
            Xtr_rf, ytr_rf = Xtr, ytr

        t0 = time.perf_counter()
        rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                   n_jobs=-1, random_state=args.seed)
        rf.fit(Xtr_rf, ytr_rf)
        res_rf = evaluate(yte, rf.predict(Xte))
        res_rf['train_time_sec'] = time.perf_counter() - t0
        results[f'RF_{mode}'] = res_rf
        print(f"  [grouped] RF  [{mode:6s}] MAE={res_rf['MAE']:.4f} "
              f"R2={res_rf['R2']:.4f} Spearman={res_rf['Spearman']:.4f}")

        t0 = time.perf_counter()
        xgb = XGBRegressor(n_estimators=600, max_depth=8, learning_rate=0.08,
                           subsample=0.8, colsample_bytree=0.8,
                           tree_method='hist', n_jobs=-1, random_state=args.seed,
                           verbosity=0)
        xgb.fit(Xtr, ytr)
        res_xgb = evaluate(yte, xgb.predict(Xte))
        res_xgb['train_time_sec'] = time.perf_counter() - t0
        results[f'XGB_{mode}'] = res_xgb
        print(f"  [grouped] XGB [{mode:6s}] MAE={res_xgb['MAE']:.4f} "
              f"R2={res_xgb['R2']:.4f} Spearman={res_xgb['Spearman']:.4f}")

    out = os.path.join(data_dir, 'models', 'grouped_split_tree_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[存檔] {out}")


if __name__ == '__main__':
    main()
