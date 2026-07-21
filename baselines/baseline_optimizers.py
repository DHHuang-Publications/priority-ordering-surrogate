"""
baseline_optimizers.py
======================
最佳化端 baseline，與 GA 公平比較。

方法（皆用「同一個訓練好的 Transformer 代理」當適應度、同一 eval 預算 = GA 的 pop×gens）：
  - ga            ：原 run_ga（對照組，本機重跑以取同機 wall-clock）
  - random_search ：均勻隨機抽排列，批次評估
  - hill_climb    ：best-improvement swap 鄰域爬山 + 隨機重啟
  - sim_anneal    ：64 條平行退火鏈（swap/insert 提案、幾何降溫），批次評估
  - greedy        ：逐位置貪婪建構（每候選以 32 次隨機補全的平均分數評分）——
                    surrogate-guided greedy / ranking-based strategy
  - mc_ga（僅 K<=7）：GA 直接用 MC 當適應度（快取重複排列）——MC-guided GA
                    for smaller problem sizes

每方法取 top-N 去重 → 以獨立 MC（psi_eval）驗證 → 報告最佳 MC R_joint、95% CI、
surrogate-eval 數與 wall-clock。K=5 另附 ground-truth 名次。

用法：
  python baseline_optimizers.py --topology nsfnet --K 5
  python baseline_optimizers.py --topology usnet  --K 10
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


from benchmark_network import get_network
from data_generator import presample_network_states
from MC_QP_multi_st import greedy_allocation
from ga_optimizer import run_ga, mc_verify, surrogate_fitness_batch

BUDGET = {5: 200 * 500, 10: 500 * 1000,
          15: 500 * 1500, 20: 1000 * 2000}   # 與 GA pop×gens 一致（原論文 Table 6）
GA_CFG = {5: dict(pop_size=200, n_generations=500),
          10: dict(pop_size=500, n_generations=1000),
          15: dict(pop_size=500, n_generations=1500),
          20: dict(pop_size=1000, n_generations=2000)}
TOP_N = 10
PSI_EVAL = 10000


# ──────────────────────────────────────────────────────────────
class SurrogateEvaluator:
    """批次代理評估器，記錄 eval 數。"""

    def __init__(self, model, demand_norm, device):
        self.model, self.dn, self.device = model, demand_norm, device
        self.n_evals = 0

    def __call__(self, perms):
        perms = np.asarray(perms)
        self.n_evals += len(perms)
        return surrogate_fitness_batch(self.model, perms, self.dn, self.device)


def topn_unique(perms, scores, n=TOP_N):
    order = np.argsort(scores)[::-1]
    seen, out = set(), []
    for i in order:
        t = tuple(perms[i])
        if t not in seen:
            seen.add(t)
            out.append(perms[i])
            if len(out) >= n:
                break
    return np.array(out)


# ──────────────────────────────────────────────────────────────
def random_search(K, budget, ev, rng):
    best_p, best_s = [], []
    for i in range(0, budget, 8192):
        m = min(8192, budget - i)
        perms = np.array([rng.permutation(K) for _ in range(m)])
        s = ev(perms)
        keep = np.argsort(s)[::-1][:TOP_N]
        best_p.extend(perms[keep]); best_s.extend(s[keep])
    return topn_unique(np.array(best_p), np.array(best_s))


def hill_climb(K, budget, ev, rng):
    pairs = [(i, j) for i in range(K) for j in range(i + 1, K)]
    best_p, best_s = [], []
    while ev.n_evals < budget:
        cur = rng.permutation(K)
        cur_s = float(ev(cur[None])[0])
        improved = True
        while improved and ev.n_evals < budget:
            neigh = []
            for (i, j) in pairs:
                p = cur.copy(); p[i], p[j] = p[j], p[i]
                neigh.append(p)
            s = ev(np.array(neigh))
            b = int(np.argmax(s))
            if s[b] > cur_s:
                cur, cur_s = neigh[b], float(s[b])
            else:
                improved = False
        best_p.append(cur); best_s.append(cur_s)
    return topn_unique(np.array(best_p), np.array(best_s))


def sim_anneal(K, budget, ev, rng, n_chains=64, T0=0.02, Tend=1e-4):
    steps = max(1, budget // n_chains) - 1
    cur = np.array([rng.permutation(K) for _ in range(n_chains)])
    cur_s = ev(cur)
    best_p, best_s = cur.copy(), cur_s.copy()
    alpha = (Tend / T0) ** (1 / max(steps, 1))
    T = T0
    for _ in range(steps):
        prop = cur.copy()
        for c in range(n_chains):
            if rng.random() < 0.5:   # swap
                i, j = rng.choice(K, 2, replace=False)
                prop[c, i], prop[c, j] = prop[c, j], prop[c, i]
            else:                    # insert
                i, j = rng.choice(K, 2, replace=False)
                v = prop[c, i]
                p = np.delete(prop[c], i)
                prop[c] = np.insert(p, j, v)
        s = ev(prop)
        dE = s - cur_s
        acc = (dE > 0) | (rng.random(n_chains) < np.exp(np.minimum(dE / T, 0)))
        cur[acc] = prop[acc]; cur_s[acc] = s[acc]
        upd = cur_s > best_s
        best_p[upd] = cur[upd]; best_s[upd] = cur_s[upd]
        T *= alpha
    return topn_unique(best_p, best_s)


def greedy_construct(K, ev, rng, n_rollouts=32, n_restarts=8):
    """逐位置貪婪：每步對每個剩餘候選做 n_rollouts 次隨機補全取平均分。"""
    best_p, best_s = [], []
    for _ in range(n_restarts):
        prefix = []
        remaining = list(range(K))
        while remaining:
            cand_perms, cand_ids = [], []
            for c in remaining:
                rest = [x for x in remaining if x != c]
                for _ in range(n_rollouts):
                    tail = rng.permutation(rest).tolist() if rest else []
                    cand_perms.append(prefix + [c] + tail)
                    cand_ids.append(c)
            s = ev(np.array(cand_perms))
            s = s.reshape(len(remaining), n_rollouts).mean(axis=1)
            pick = remaining[int(np.argmax(s))]
            prefix.append(pick); remaining.remove(pick)
        p = np.array(prefix)
        best_p.append(p); best_s.append(float(ev(p[None])[0]))
    return topn_unique(np.array(best_p), np.array(best_s))


def mc_ga(K, net, demand, cfg, seed):
    """GA 直接用 MC 當適應度（快取），僅小 K 可行。回傳 (top_perms, n_mc_evals)。"""
    MPs, T_list, binom, Cmax, mpt = net
    caps, lts = presample_network_states(binom, Cmax, mpt, 2000, seed=seed + 7)
    cache = {}

    def mc_fit(perm):
        t = tuple(perm)
        if t not in cache:
            D = np.stack([greedy_allocation(caps[i], lts[i], MPs, T_list, list(perm))
                          for i in range(2000)])
            cache[t] = float(np.mean(np.all(D >= demand, axis=1)))
        return cache[t]

    rng = np.random.default_rng(seed)
    pop = np.array([rng.permutation(K) for _ in range(30)])
    from ga_optimizer import pmx_crossover, swap_mutation, insert_mutation, tournament_selection
    fit = np.array([mc_fit(p) for p in pop])
    for _ in range(20):
        elite = pop[np.argsort(fit)[::-1][:3]]
        children = list(elite)
        while len(children) < len(pop):
            a = pop[tournament_selection(pop, fit, 3, rng)]   # 回傳為索引
            b = pop[tournament_selection(pop, fit, 3, rng)]
            c = pmx_crossover(a, b, rng) if rng.random() < 0.8 else a.copy()
            if rng.random() < 0.3:
                c = swap_mutation(c, rng) if rng.random() < 0.5 else insert_mutation(c, rng)
            children.append(c)
        pop = np.array(children)
        fit = np.array([mc_fit(p) for p in pop])
    scores = np.array([cache[tuple(p)] for p in pop])
    return topn_unique(pop, scores), len(cache)


# ──────────────────────────────────────────────────────────────
def main():
    import torch
    from models.transformer_surrogate import PermutationTransformer

    ap = argparse.ArgumentParser()
    ap.add_argument('--topology', choices=['nsfnet', 'usnet', 'synth'], required=True)
    ap.add_argument('--K', type=int, required=True)
    ap.add_argument('--device', default=None)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--methods', nargs='+',
                    default=['ga', 'random_search', 'hill_climb', 'sim_anneal', 'greedy', 'mc_ga'])
    args = ap.parse_args()

    K, seed = args.K, args.seed
    if args.topology == 'synth':
        # 原論文合成實驗的資料/模型（code/data/K{K}_psi*），網路照 run_experiments 重建（seed=42）
        psi_tag = 'psi10000' if K == 5 else 'psi5000'
        data_dir = os.path.join(CODE, 'data', f'K{K}_{psi_tag}')
    else:
        data_dir = os.path.join(HERE, 'data', f'{args.topology}_K{K}')
    device = args.device or ('cuda' if torch.cuda.is_available() else
                             ('mps' if torch.backends.mps.is_available() else 'cpu'))

    # demand 與模型
    meta = np.load(os.path.join(data_dir, 'metadata.npz'), allow_pickle=True)
    demand = ((meta['d_ranges'][:, 0] + meta['d_ranges'][:, 1]) / 2).astype(int)
    ckpt = torch.load(os.path.join(data_dir, 'models', 'transformer_best.pt'),
                      map_location=device, weights_only=False)
    a = ckpt.get('args', {})
    model = PermutationTransformer(K=K, d_model=a.get('d_model', 64), nhead=a.get('nhead', 4),
                                   num_layers=a.get('num_layers', 3),
                                   dim_feedforward=a.get('d_model', 64) * 2)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device).eval()
    dn = (demand / ckpt['d_max']).astype(np.float32)

    if args.topology == 'synth':
        # 與原 run_experiments.py / ga_optimizer.py 相同的合成網路重建（seed=42）
        from data_generator import get_network_K3, generate_random_pairs
        _, _, binom_base, Cmax_base, mpt_s = get_network_K3()
        n_arcs = int(meta['n_arcs'])
        binom_s = np.tile(binom_base, (n_arcs // len(binom_base)) + 1)[:n_arcs]
        Cmax_s = np.full(n_arcs, int(Cmax_base[0]), dtype=int)
        MPs_s, T_s = generate_random_pairs(n_arcs, K, m_per_pair=10,
                                           min_arcs_per_path=2, max_arcs_per_path=4, seed=42)
        net = (MPs_s, T_s, binom_s, Cmax_s, mpt_s)
    else:
        net = get_network(args.topology, K)
    budget = BUDGET[K]
    gt = None
    gt_path = os.path.join(data_dir, 'ground_truth.json')
    if os.path.exists(gt_path):
        gt = json.load(open(gt_path))

    print(f"### optimizer baselines: {args.topology} K={K} budget={budget} device={device} ###")
    results = {}
    for method in args.methods:
        if method == 'mc_ga' and K > 7:
            continue
        rng = np.random.default_rng(seed)
        ev = SurrogateEvaluator(model, dn, device)
        t0 = time.perf_counter()
        n_evals = None
        if method == 'ga':
            best_perm, _, _, pop, fit = run_ga(model, K, demand.astype(np.float32),
                                               ckpt['d_max'], device,
                                               pop_size=GA_CFG[K]['pop_size'],
                                               n_generations=GA_CFG[K]['n_generations'],
                                               seed=seed)
            top = topn_unique(np.vstack([pop, best_perm[None]]),
                              np.concatenate([fit, [fit.max() + 1e-9]]))
            n_evals = budget
        elif method == 'random_search':
            top = random_search(K, budget, ev, rng); n_evals = ev.n_evals
        elif method == 'hill_climb':
            top = hill_climb(K, budget, ev, rng); n_evals = ev.n_evals
        elif method == 'sim_anneal':
            top = sim_anneal(K, budget, ev, rng); n_evals = ev.n_evals
        elif method == 'greedy':
            top = greedy_construct(K, ev, rng); n_evals = ev.n_evals
        elif method == 'mc_ga':
            top, n_evals = mc_ga(K, net, demand, GA_CFG[K], seed)
        search_time = time.perf_counter() - t0

        mc = mc_verify(top, demand, *net, psi=PSI_EVAL, seed=seed + 100)
        best = max(mc, key=lambda r: r['R_joint'])
        row = dict(mc_R=best['R_joint'], CI=best['CI_95'], perm=best['perm'],
                   n_evals=int(n_evals), search_time_sec=search_time)
        if gt is not None:
            all_R = np.array(gt['all_R'])
            rank = int(np.sum(all_R > best['R_joint'] + 1e-12)) + 1
            row['gt_rank'] = rank
            row['is_gt_best'] = (list(best['perm']) == list(gt['best_perm'])
                                 or abs(best['R_joint'] - gt['best_R']) < 1e-9)
        results[method] = row
        extra = f" gt_rank={row.get('gt_rank')}" if gt else ""
        print(f"  {method:14s} MC_R={best['R_joint']:.4f} CI=[{best['CI_95'][0]:.3f},{best['CI_95'][1]:.3f}] "
              f"evals={n_evals:>7} time={search_time:6.1f}s{extra}")

    out = os.path.join(data_dir, 'ga_results', 'optimizer_baselines.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[存檔] {out}")


if __name__ == '__main__':
    main()
