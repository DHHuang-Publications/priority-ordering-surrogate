"""
plot_diagnostics.py
===================
從已訓練的 K=5 NSFNET 模型 + test.npz 產生代理模型診斷圖。
- (a) Predicted vs MC 散佈圖（Transformer 與 MLP）
- (b) 校準圖（分箱：平均預測 vs 平均實際）
- (c) 絕對誤差分布

輸出：nsfnet_K5_diagnostics.pdf / .png
※ 這是資料圖（非拓撲示意圖），matplotlib 為適用工具。
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
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from models.transformer_surrogate import PermutationTransformer
from models.mlp_baseline import PermutationMLP

TF_BLUE, MLP_ORANGE = '#1f77b4', '#e8820e'


def load_model(mtype):
    ckpt = torch.load(os.path.join(DATA, 'models', f'{mtype}_best.pt'),
                      map_location='cpu', weights_only=False)
    a = ckpt.get('args', {})
    if mtype == 'transformer':
        model = PermutationTransformer(
            K=ckpt['K'], d_model=a.get('d_model', 64), nhead=a.get('nhead', 4),
            num_layers=a.get('num_layers', 3), dim_feedforward=a.get('d_model', 64) * 2)
    else:
        model = PermutationMLP(K=ckpt['K'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, np.asarray(ckpt['d_max'], dtype=np.float32)


def predict(model, perms, demands, d_max):
    with torch.no_grad():
        p = torch.as_tensor(perms, dtype=torch.long)
        d = torch.as_tensor(demands / d_max, dtype=torch.float32)
        return model(p, d).cpu().numpy().ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='nsfnet_K5', help='data/ 下的資料夾名')
    ap.add_argument('--title', default=None)
    ap.add_argument('--out', default=None, help='輸出檔名 stem（不含副檔名）')
    args = ap.parse_args()
    global DATA
    DATA = os.path.join(HERE, 'data', args.data)
    title = args.title or f'{args.data}: surrogate diagnostics on held-out test set'
    out_stem = args.out or f'{args.data}_diagnostics'
    test = np.load(os.path.join(DATA, 'test.npz'))
    perms, demands, y = test['perms'], test['demands'].astype(np.float32), test['R_joint']

    preds, stats = {}, {}
    for mtype, color in (('transformer', TF_BLUE), ('mlp', MLP_ORANGE)):
        model, d_max = load_model(mtype)
        yp = predict(model, perms, demands, d_max)
        preds[mtype] = yp
        mae = np.mean(np.abs(yp - y))
        r2 = 1 - np.sum((yp - y) ** 2) / np.sum((y - y.mean()) ** 2)
        rho = spearmanr(yp, y).correlation
        stats[mtype] = dict(mae=mae, r2=r2, rho=rho, color=color)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8))
    label = {'transformer': 'Transformer+MLP', 'mlp': 'MLP'}

    # (a) Predicted vs MC
    ax = axes[0]
    lim = [min(y.min(), *[preds[m].min() for m in preds]) - 0.02,
           max(y.max(), *[preds[m].max() for m in preds]) + 0.02]
    ax.plot(lim, lim, 'k--', lw=1, alpha=0.6, label='ideal ($y=x$)')
    for m in preds:
        ax.scatter(y, preds[m], s=9, alpha=0.35, color=stats[m]['color'],
                   label=f"{label[m]} ($R^2$={stats[m]['r2']:.3f})")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('MC-estimated $R_{\\mathrm{joint}}$')
    ax.set_ylabel('Surrogate prediction')
    ax.set_title('(a) Predicted vs. MC reliability')
    ax.legend(fontsize=9, loc='upper left'); ax.grid(ls=':', alpha=0.4)

    # (b) Calibration (binned)
    ax = axes[1]
    bins = np.linspace(y.min(), y.max(), 11)
    idx = np.digitize(y, bins) - 1
    centers = 0.5 * (bins[:-1] + bins[1:])
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=1, alpha=0.6)
    for m in preds:
        means = [preds[m][idx == b].mean() if np.any(idx == b) else np.nan
                 for b in range(len(bins) - 1)]
        ax.plot(centers, means, 'o-', ms=5, color=stats[m]['color'], label=label[m])
    ax.set_xlabel('MC-estimated $R_{\\mathrm{joint}}$ (bin)')
    ax.set_ylabel('Mean surrogate prediction')
    ax.set_title('(b) Calibration')
    ax.legend(fontsize=9); ax.grid(ls=':', alpha=0.4)

    # (c) Absolute error distribution
    ax = axes[2]
    for m in preds:
        ax.hist(np.abs(preds[m] - y), bins=40, alpha=0.55, color=stats[m]['color'],
                label=f"{label[m]} (MAE={stats[m]['mae']:.4f})")
    ax.set_xlabel('Absolute error $|\\hat{R}-R|$')
    ax.set_ylabel('Count')
    ax.set_title('(c) Error distribution')
    ax.legend(fontsize=9); ax.grid(ls=':', alpha=0.4)

    fig.suptitle(title, fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ('pdf', 'png'):
        out = os.path.join(HERE, 'figures', f'{out_stem}.{ext}')
        fig.savefig(out, dpi=200, bbox_inches='tight')
        print(f'[存檔] {out}')
    for m in stats:
        print(f"  {label[m]:16s} MAE={stats[m]['mae']:.4f} "
              f"R2={stats[m]['r2']:.4f} Spearman={stats[m]['rho']:.4f}")


if __name__ == '__main__':
    main()
