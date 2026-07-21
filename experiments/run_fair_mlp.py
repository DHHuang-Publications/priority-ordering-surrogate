"""
run_fair_mlp.py
===============
K=15/20 的公平比較 MLP 變體。

變體（皆用原論文 K=15/20 合成資料 code/data/K{K}_psi5000）：
  - matched：hidden=[736,512,288]，參數 ~551K ≈ Transformer
  - large  ：hidden=[1024,704,448]，參數 ~1.07M ≈ 2×Transformer（檢驗容量能否替代自注意力架構）
  - onehot ：pair-at-position one-hot 輸入（K²+K 維），hidden=[256,256,128]


訓練配方與原 train_surrogate.py 完全一致：AdamW(lr 1e-3, wd 1e-4)、cosine 到 0、
MSE、grad clip 1.0、early stop on val MAE（patience 40）、batch 512、demands/d_max 正規化。

每變體輸出 <data>/models/fair_{variant}_results.json（+ checkpoint），
訓練後跑 GA+MC 驗證（同原配置），與既有 transformer/mlp 結果並列印出。

用法（GPU）：
  python run_fair_mlp.py --Ks 15 20 --variants matched large onehot --device cuda
估時：每變體 1–3 hr，全套（2K × 3 變體）約 8–16 hr。已完成的變體會自動 skip（可續跑）。
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
import numpy as np
from scipy.stats import spearmanr


HIDDEN = {
    'matched': [736, 512, 288],    # ~551K（對齊 K=15/20 Transformer 550.7K/551.9K）
    'large':   [1024, 704, 448],   # ~1.07M（≈2×Transformer）
    'onehot':  [256, 256, 128],    # 表示變體：容量沿用預設，換輸入編碼
}
TRAIN_CFG = {15: dict(batch=512, epochs=300, patience=40),
             20: dict(batch=512, epochs=300, patience=40)}
GA_CFG = {15: dict(pop_size=500, n_generations=1500),
          20: dict(pop_size=1000, n_generations=2000)}
PSI_EVAL = 10000


# ──────────────────────────────────────────────────────────────
def build_model(variant, K):
    import torch.nn as nn
    from models.mlp_baseline import PermutationMLP

    if variant in ('matched', 'large'):
        return PermutationMLP(K=K, hidden_dims=HIDDEN[variant])

    class OneHotMLP(nn.Module):
        """pair-at-position one-hot（K²）+ demand（K）輸入；其餘與 PermutationMLP 同構。"""

        def __init__(self, K, hidden_dims, dropout=0.1):
            super().__init__()
            self.K = K
            layers, prev = [], K * K + K
            for h in hidden_dims:
                layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)]
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.network = nn.Sequential(*layers)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

        def forward(self, perm, demand):
            import torch
            oh = torch.nn.functional.one_hot(perm, num_classes=self.K)  # (B,K,K)
            x = torch.cat([oh.float().flatten(1), demand], dim=-1)
            return self.network(x).squeeze(-1)

    return OneHotMLP(K, HIDDEN['onehot'])


def load_split(data_dir, name, d_max=None):
    import torch
    z = np.load(os.path.join(data_dir, f'{name}.npz'))
    P = torch.as_tensor(z['perms'].astype(np.int64))
    D = z['demands'].astype(np.float32)
    if d_max is None:
        d_max = np.maximum(D.max(axis=0), 1.0)
    D = torch.as_tensor(D / d_max)
    Y = torch.as_tensor(z['R_joint'].astype(np.float32))
    return P, D, Y, d_max


def evaluate(model, P, D, Y, device, bs=8192):
    import torch
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(Y), bs):
            preds.append(model(P[i:i+bs].to(device), D[i:i+bs].to(device)).cpu())
    yp = torch.cat(preds).numpy()
    y = Y.numpy()
    return dict(MAE=float(np.mean(np.abs(yp - y))),
                R2=float(1 - np.sum((yp-y)**2)/np.sum((y-y.mean())**2)),
                Spearman=float(spearmanr(yp, y).correlation))


def train_variant(variant, K, data_dir, device, seed=42, epochs_override=None,
                  max_train=None):
    import torch
    cfg = TRAIN_CFG[K]
    epochs = epochs_override or cfg['epochs']
    torch.manual_seed(seed)
    np.random.seed(seed)

    Ptr, Dtr, Ytr, d_max = load_split(data_dir, 'train')
    Pva, Dva, Yva, _ = load_split(data_dir, 'val', d_max)
    Pte, Dte, Yte, _ = load_split(data_dir, 'test', d_max)
    if max_train:
        Ptr, Dtr, Ytr = Ptr[:max_train], Dtr[:max_train], Ytr[:max_train]

    model = build_model(variant, K).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  [{variant}] params={n_params:,} train={len(Ytr):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = torch.nn.MSELoss()
    bs = cfg['batch']
    best_mae, best_state, patience_left = float('inf'), None, cfg['patience']
    t0 = time.perf_counter()
    n = len(Ytr)
    for ep in range(1, epochs + 1):
        model.train()
        idx = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            b = idx[i:i+bs]
            opt.zero_grad()
            out = model(Ptr[b].to(device), Dtr[b].to(device))
            loss = lossf(out, Ytr[b].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(b)
        sched.step()
        vmae = evaluate(model, Pva, Dva, Yva, device)['MAE']
        star = ''
        if vmae < best_mae:
            best_mae, patience_left = vmae, cfg['patience']
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            star = '★'
        else:
            patience_left -= 1
        if ep % 10 == 0 or star:
            print(f"    ep {ep:3d}/{epochs} loss={tot/n:.6f} valMAE={vmae:.5f} {star}")
        if patience_left <= 0:
            print(f"    early stop @ ep {ep}")
            break
    train_time = time.perf_counter() - t0
    model.load_state_dict(best_state)
    res = evaluate(model, Pte, Dte, Yte, device)
    res.update(variant=variant, K=K, n_params=n_params,
               epochs_trained=ep, train_time_sec=train_time)
    return model, res, d_max


def ga_verify(model, K, data_dir, d_max, device, seed=42):
    """GA（原配置）+ MC 驗證（合成網路照 run_experiments 重建，seed=42）。"""
    from ga_optimizer import run_ga, mc_verify
    from data_generator import get_network_K3, generate_random_pairs
    meta = np.load(os.path.join(data_dir, 'metadata.npz'), allow_pickle=True)
    demand = ((meta['d_ranges'][:, 0] + meta['d_ranges'][:, 1]) / 2).astype(int)
    _, _, binom_base, Cmax_base, mpt = get_network_K3()
    n_arcs = int(meta['n_arcs'])
    binom = np.tile(binom_base, (n_arcs // len(binom_base)) + 1)[:n_arcs]
    Cmax = np.full(n_arcs, int(Cmax_base[0]), dtype=int)
    MPs, T = generate_random_pairs(n_arcs, K, m_per_pair=10,
                                   min_arcs_per_path=2, max_arcs_per_path=4, seed=42)
    cfg = GA_CFG[K]
    best_perm, best_fit, _, pop, fit = run_ga(
        model, K, demand.astype(np.float32), d_max, device,
        pop_size=cfg['pop_size'], n_generations=cfg['n_generations'], seed=seed)
    seen, top = set(), []
    for i in np.argsort(fit)[::-1]:
        t = tuple(pop[i])
        if t not in seen:
            seen.add(t); top.append(pop[i])
            if len(top) >= 10:
                break
    if tuple(best_perm) not in seen:
        top.insert(0, best_perm)
    mc = mc_verify(np.array(top), demand, MPs, T, binom, Cmax, mpt,
                   psi=PSI_EVAL, seed=seed + 100)
    best = max(mc, key=lambda r: r['R_joint'])
    return dict(mc_best_R=best['R_joint'], CI=best['CI_95'],
                predicted=float(best_fit))


def main():
    import torch
    ap = argparse.ArgumentParser(description='fair MLP variants (K=15/20)')
    ap.add_argument('--Ks', type=int, nargs='+', default=[15, 20])
    ap.add_argument('--variants', nargs='+', default=['matched', 'large', 'onehot'],
                    choices=['matched', 'large', 'onehot'])
    ap.add_argument('--device', default=None)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--no_ga', action='store_true', help='只訓練+測試，不跑 GA 驗證')
    ap.add_argument('--epochs', type=int, default=None, help='覆蓋（tiny 測試用）')
    ap.add_argument('--max_train', type=int, default=None, help='覆蓋（tiny 測試用）')
    args = ap.parse_args()
    device = args.device or ('cuda' if torch.cuda.is_available() else
                             ('mps' if torch.backends.mps.is_available() else 'cpu'))

    for K in args.Ks:
        data_dir = os.path.join(CODE, 'data', f'K{K}_psi5000')
        model_dir = os.path.join(data_dir, 'models')
        # 既有結果對照
        ref = {}
        for m in ('transformer', 'mlp'):
            p = os.path.join(model_dir, f'{m}_results.json')
            if os.path.exists(p):
                r = json.load(open(p))
                ref[m] = (r['n_params'], r['test_MAE'], r['test_Spearman'])
        print(f"\n===== K={K} =====  既有: " + " | ".join(
            f"{m} {v[0]/1e3:.0f}K MAE={v[1]:.4f} ρ={v[2]:.4f}" for m, v in ref.items()))

        for variant in args.variants:
            out_path = os.path.join(model_dir, f'fair_{variant}_results.json')
            if os.path.exists(out_path) and not args.epochs:
                print(f"  [skip] {variant} 已完成")
                continue
            model, res, d_max = train_variant(variant, K, data_dir, device,
                                              seed=args.seed,
                                              epochs_override=args.epochs,
                                              max_train=args.max_train)
            if not args.no_ga:
                res['ga'] = ga_verify(model, K, data_dir, d_max, device, seed=args.seed)
            import torch as _t
            _t.save({'model_state_dict': model.state_dict(), 'K': K,
                     'd_max': d_max, 'variant': variant,
                     'hidden_dims': HIDDEN[variant]},
                    os.path.join(model_dir, f'fair_{variant}_best.pt'))
            with open(out_path, 'w') as f:
                json.dump(res, f, indent=2)
            ga_str = (f" | GA mc_R={res['ga']['mc_best_R']:.4f}" if 'ga' in res else "")
            print(f"  [done] {variant}: {res['n_params']/1e3:.0f}K "
                  f"MAE={res['MAE']:.4f} ρ={res['Spearman']:.4f}{ga_str}")


if __name__ == '__main__':
    main()
