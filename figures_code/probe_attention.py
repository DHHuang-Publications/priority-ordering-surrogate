"""
probe_attention.py — 驗證 Transformer 的 self-attention 是否可解釋。
手動複現 post-norm TransformerEncoderLayer forward 以擷取每層 attention 權重，
在一批 test 排列上平均，檢查是否偏離均勻(1/K)。若可解釋→畫熱圖；否則改替代方向。
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
import sys, numpy as np, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); CODE = os.path.dirname(HERE)
sys.path.insert(0, CODE); sys.path.insert(0, HERE)
from models.transformer_surrogate import PermutationTransformer


def load(topo, K, device='cpu'):
    d = os.path.join(HERE, 'data', f'{topo}_K{K}')
    ck = torch.load(os.path.join(d, 'models', 'transformer_best.pt'),
                    map_location=device, weights_only=False)
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
    """複現 post-norm forward，回 (下一層輸入 x, attn 權重 avg over heads)。"""
    attn_out, w = layer.self_attn(x, x, x, need_weights=True, average_attn_weights=True)
    x = layer.norm1(x + attn_out)
    ff = layer.linear2(layer.activation(layer.linear1(x)))
    x = layer.norm2(x + ff)
    return x, w  # w: (B, K, K)


def main():
    for topo, K in [('nsfnet', 5), ('usnet', 10)]:
        m, z, dmax = load(topo, K)
        n = min(2000, len(z['perms']))
        perm = torch.as_tensor(z['perms'][:n].astype(np.int64))
        dem = torch.as_tensor((z['demands'][:n].astype(np.float32)) / dmax)
        with torch.no_grad():
            x = embed(m, perm, dem)
            per_layer = []
            for layer in m.transformer_encoder.layers:
                x, w = layer_attn(layer, x)
                per_layer.append(w.mean(0).numpy())  # (K,K) 平均 over batch
        print(f"\n=== {topo} K={K}  (uniform={1/K:.3f}) ===")
        for li, A in enumerate(per_layer):
            row = A.mean(0)  # 平均每個 key position 收到的注意力
            print(f" layer{li}: attn range [{A.min():.3f},{A.max():.3f}] "
                  f"std={A.std():.3f}  per-key mean={np.round(row,3)}")
        # 對角/下三角/上三角能量（看是否有優先順序結構）
        A = np.mean(per_layer, 0)
        iu = np.triu_indices(K, 1); il = np.tril_indices(K, -1)
        print(f" MEAN over layers: diag={np.mean(np.diag(A)):.3f} "
              f"lower(attend earlier)={A[il].mean():.3f} upper(attend later)={A[iu].mean():.3f}")


if __name__ == '__main__':
    main()
