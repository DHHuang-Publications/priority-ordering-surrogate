"""
plot_attention.py — Transformer self-attention 熱圖。
在 test 排列上平均每層 position→position 注意力（average over heads & samples），
畫 2 拓撲 × 3 層熱圖。列=query 位置（優先權槽，0=最高優先），行=key 位置。
揭示模型關注哪些優先權位置——反映聯合可靠度的瓶頸結構。

輸出：figures/attention_heatmap.pdf / .png
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
import sys, numpy as np, torch
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); CODE = os.path.dirname(HERE)
sys.path.insert(0, CODE); sys.path.insert(0, HERE)
from models.transformer_surrogate import PermutationTransformer

CASES = [('nsfnet', 5, 'NSFNET ($K=5$)'), ('usnet', 10, 'USNET ($K=10$)')]


def load(topo, K):
    d = os.path.join(HERE, 'data', f'{topo}_K{K}')
    ck = torch.load(os.path.join(d, 'models', 'transformer_best.pt'),
                    map_location='cpu', weights_only=False)
    a = ck.get('args', {})
    m = PermutationTransformer(K=K, d_model=a.get('d_model', 64), nhead=a.get('nhead', 4),
                               num_layers=a.get('num_layers', 3),
                               dim_feedforward=a.get('d_model', 64) * 2)
    m.load_state_dict(ck['model_state_dict']); m.eval()
    z = np.load(os.path.join(d, 'test.npz'))
    return m, z, np.asarray(ck['d_max'], np.float32)


def embed(m, perm, demand):
    pe = m.pair_embedding(perm)
    pos = m.pos_embedding(torch.arange(m.K)).unsqueeze(0).expand(perm.shape[0], -1, -1)
    dat = torch.gather(demand, 1, perm).unsqueeze(-1)
    return pe + pos + m.demand_proj(dat)


def layer_attn(layer, x):
    out, w = layer.self_attn(x, x, x, need_weights=True, average_attn_weights=True)
    x = layer.norm1(x + out)
    ff = layer.linear2(layer.activation(layer.linear1(x)))
    x = layer.norm2(x + ff)
    return x, w


def mean_attention(topo, K):
    m, z, dmax = load(topo, K)
    n = min(3000, len(z['perms']))
    perm = torch.as_tensor(z['perms'][:n].astype(np.int64))
    dem = torch.as_tensor(z['demands'][:n].astype(np.float32) / dmax)
    with torch.no_grad():
        x = embed(m, perm, dem)
        mats = []
        for layer in m.transformer_encoder.layers:
            x, w = layer_attn(layer, x)
            mats.append(w.mean(0).numpy())
    return mats  # list of (K,K)


def main():
    nlayers = 3
    fig, axes = plt.subplots(2, nlayers, figsize=(10.5, 6.6))
    for r, (topo, K, title) in enumerate(CASES):
        mats = mean_attention(topo, K)
        vmax = max(A.max() for A in mats)
        for c in range(nlayers):
            ax = axes[r, c]
            im = ax.imshow(mats[c], cmap='viridis', vmin=0, vmax=vmax, aspect='equal')
            ax.set_title(f'{title} — layer {c+1}', fontsize=9)
            ax.set_xlabel('key position (priority slot)', fontsize=8)
            if c == 0:
                ax.set_ylabel('query position', fontsize=8)
            ax.set_xticks(range(K)); ax.set_yticks(range(K))
            ax.tick_params(labelsize=6)
        fig.colorbar(im, ax=axes[r, :].tolist(), fraction=0.018, pad=0.02,
                     label='mean attention')
    fig.subplots_adjust(hspace=0.45, wspace=0.25)
    out = os.path.join(HERE, 'figures', 'attention_heatmap')
    fig.savefig(out + '.pdf', bbox_inches='tight')
    fig.savefig(out + '.png', dpi=150, bbox_inches='tight')
    print(f"[存檔] {out}.pdf/.png")


if __name__ == '__main__':
    main()
