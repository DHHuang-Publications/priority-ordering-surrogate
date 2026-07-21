"""
plot_ga_convergence.py
======================
GA 收斂圖。從已訓練代理模型同 seed 重跑 GA（純代理前向、
不做 MC），擷取 run_ga 回傳的 history，畫「best-so-far」與「族群平均」fitness
對世代的曲線，證明 GA 在所選世代預算內收斂並趨於平台。

兩面板：左 NSFNET (K=5, pop=200, gen=500)、右 USNET (K=10, pop=500, gen=1000)，
每面板疊 Transformer 與 MLP。配置/seed/demand 與 run_nsfnet.py 正式跑完全一致，
故終點與論文 GA 結果（NSFNET 0.6658、USNET 0.8561）重現。

輸出：figures/ga_convergence.pdf / .png
用法：python plot_ga_convergence.py
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
import numpy as np
import torch
import matplotlib.pyplot as plt


from ga_optimizer import run_ga
from models.transformer_surrogate import PermutationTransformer
from models.mlp_baseline import PermutationMLP

TF_BLUE, MLP_ORANGE = '#1f77b4', '#e8820e'
SEED = 42

# (topology, K, pop_size, n_generations, demand) —— 同 run_nsfnet.py 正式配置
CASES = [
    dict(topo='nsfnet', K=5,  pop=200, gens=500,  demand=[19, 18, 15, 18, 23],
         title='NSFNET ($K=5$)'),
    dict(topo='usnet',  K=10, pop=500, gens=1000, demand=[4, 4, 3, 4, 4, 4, 4, 5, 5, 5],
         title='USNET ($K=10$)'),
]


def data_dir(topo, K):
    return os.path.join(HERE, 'data', f'{topo}_K{K}')


def load_model(topo, K, mtype, device):
    ckpt = torch.load(os.path.join(data_dir(topo, K), 'models', f'{mtype}_best.pt'),
                      map_location=device, weights_only=False)
    a = ckpt.get('args', {})
    if mtype == 'transformer':
        model = PermutationTransformer(
            K=K, d_model=a.get('d_model', 64), nhead=a.get('nhead', 4),
            num_layers=a.get('num_layers', 3), dim_feedforward=a.get('d_model', 64) * 2)
    else:
        model = PermutationMLP(K=K)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device).eval()
    return model, np.asarray(ckpt['d_max'], dtype=np.float32)


def ga_history(topo, K, mtype, pop, gens, demand, device):
    model, d_max = load_model(topo, K, mtype, device)
    _, best_fit, hist, _, _ = run_ga(
        model, K, np.array(demand, dtype=np.float32), d_max, device,
        pop_size=pop, n_generations=gens, seed=SEED)
    best = np.maximum.accumulate(np.asarray(hist['best_fitness']))  # best-so-far
    mean = np.asarray(hist['mean_fitness'])
    return best, mean, float(best_fit)


def main():
    device = 'cuda' if torch.cuda.is_available() else (
        'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"device={device}")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, c in zip(axes, CASES):
        for mtype, color in (('transformer', TF_BLUE), ('mlp', MLP_ORANGE)):
            best, mean, bf = ga_history(c['topo'], c['K'], mtype,
                                        c['pop'], c['gens'], c['demand'], device)
            g = np.arange(1, len(best) + 1)
            label = 'Transformer' if mtype == 'transformer' else 'MLP'
            ax.plot(g, best, color=color, lw=2.0, label=f'{label} (best-so-far)')
            ax.plot(g, mean, color=color, lw=1.0, ls='--', alpha=0.6,
                    label=f'{label} (population mean)')
            print(f"  {c['topo']} {mtype}: final best={bf:.4f} "
                  f"(gen1 best-so-far={best[0]:.4f})")
        ax.set_title(c['title'])
        ax.set_xlabel('Generation (log scale)')
        ax.set_ylabel('Predicted joint reliability $R_{\\mathrm{joint}}$')
        ax.set_xscale('log')
        ax.set_xlim(1, c['gens'])
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    out = os.path.join(HERE, 'figures', 'ga_convergence')
    fig.savefig(out + '.pdf', bbox_inches='tight')
    fig.savefig(out + '.png', dpi=150, bbox_inches='tight')
    print(f"[存檔] {out}.pdf / .png")


if __name__ == '__main__':
    main()
