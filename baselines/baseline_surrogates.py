"""
baseline_surrogates.py
======================
樹模型代理 baseline —— Random Forest 與 XGBoost。

與神經網路代理使用「完全相同的資料切分」（train/val/test.npz），
兩種特徵表示皆測，避免輸入表示造成不公平：
  - flat   ：與 MLP 相同的 [perm/(K-1), demand/d_max] ∈ R^{2K}
  - onehot ：pair-at-position one-hot ∈ {0,1}^{K^2} + demand/d_max ∈ R^K

輸出 test MAE / R² / Spearman 與訓練時間，結果存 <data_dir>/models/baseline_results.json。

用法：
  python baseline_surrogates.py --data_dir data/nsfnet_K5
  python baseline_surrogates.py --data_dir data/usnet_K10 --subsample 300000
（大資料集可 --subsample 控制 RF 訓練時間；XGBoost 用全量。）
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



def load_split(data_dir, name):
    z = np.load(os.path.join(data_dir, f'{name}.npz'))
    return z['perms'].astype(int), z['demands'].astype(np.float32), z['R_joint'].astype(np.float32)


def featurize(perms, demands, d_max, mode):
    K = perms.shape[1]
    dn = demands / d_max
    if mode == 'flat':
        return np.hstack([perms / (K - 1), dn]).astype(np.float32)
    if mode == 'onehot':
        N = perms.shape[0]
        oh = np.zeros((N, K * K), dtype=np.float32)
        rows = np.repeat(np.arange(N), K)
        cols = (np.tile(np.arange(K), N) * K + perms.ravel())
        oh[rows, cols] = 1.0
        return np.hstack([oh, dn]).astype(np.float32)
    raise ValueError(mode)


def evaluate(y_true, y_pred):
    mae = float(np.mean(np.abs(y_pred - y_true)))
    r2 = float(1 - np.sum((y_pred - y_true) ** 2) / np.sum((y_true - y_true.mean()) ** 2))
    rho = float(spearmanr(y_pred, y_true).correlation)
    return dict(MAE=mae, R2=r2, Spearman=rho)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--subsample', type=int, default=None,
                    help='RF 訓練子抽樣數（None=全量；大資料集建議 30 萬）')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor

    data_dir = args.data_dir if os.path.isabs(args.data_dir) else os.path.join(HERE, args.data_dir)
    Xp_tr, Xd_tr, y_tr = load_split(data_dir, 'train')
    Xp_te, Xd_te, y_te = load_split(data_dir, 'test')
    K = Xp_tr.shape[1]
    d_max = np.maximum(Xd_tr.max(axis=0), 1.0)
    print(f"data={os.path.basename(data_dir)}  K={K}  train={len(y_tr)}  test={len(y_te)}")

    rng = np.random.default_rng(args.seed)
    results = {}
    for mode in ('flat', 'onehot'):
        Xtr = featurize(Xp_tr, Xd_tr, d_max, mode)
        Xte = featurize(Xp_te, Xd_te, d_max, mode)

        # --- Random Forest（可子抽樣控制時間）---
        if args.subsample and len(y_tr) > args.subsample:
            idx = rng.choice(len(y_tr), args.subsample, replace=False)
            Xtr_rf, ytr_rf = Xtr[idx], y_tr[idx]
            note = f'subsample={args.subsample}'
        else:
            Xtr_rf, ytr_rf = Xtr, y_tr
            note = 'full'
        t0 = time.perf_counter()
        rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                   n_jobs=-1, random_state=args.seed)
        rf.fit(Xtr_rf, ytr_rf)
        rf_time = time.perf_counter() - t0
        res_rf = evaluate(y_te, rf.predict(Xte))
        res_rf.update(train_time_sec=rf_time, train_size=len(ytr_rf), note=note)
        results[f'RF_{mode}'] = res_rf
        print(f"  RF  [{mode:6s}] MAE={res_rf['MAE']:.4f} R2={res_rf['R2']:.4f} "
              f"Spearman={res_rf['Spearman']:.4f} ({rf_time:.0f}s, {note})")

        # --- XGBoost（全量，hist）---
        t0 = time.perf_counter()
        xgb = XGBRegressor(n_estimators=600, max_depth=8, learning_rate=0.08,
                           subsample=0.8, colsample_bytree=0.8,
                           tree_method='hist', n_jobs=-1, random_state=args.seed,
                           verbosity=0)
        xgb.fit(Xtr, y_tr)
        xgb_time = time.perf_counter() - t0
        res_xgb = evaluate(y_te, xgb.predict(Xte))
        res_xgb.update(train_time_sec=xgb_time, train_size=len(y_tr), note='full')
        results[f'XGB_{mode}'] = res_xgb
        print(f"  XGB [{mode:6s}] MAE={res_xgb['MAE']:.4f} R2={res_xgb['R2']:.4f} "
              f"Spearman={res_xgb['Spearman']:.4f} ({xgb_time:.0f}s)")

    out = os.path.join(data_dir, 'models', 'baseline_results.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[存檔] {out}")


if __name__ == '__main__':
    main()
