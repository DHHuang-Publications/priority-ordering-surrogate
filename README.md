# Surrogate-Assisted Priority Ordering Optimization for Joint Reliability in Multi-State Flow Networks

This repository contains the code, network instances, trained surrogate models,
and benchmark protocol needed to reproduce the results reported in the paper.

The framework optimizes the **priority ordering** of *K* source–terminal pairs
that compete for shared edge capacity in a multi-state flow network, so as to
maximize the **joint reliability** (the probability that all pairs simultaneously
meet their demand targets under a sequential capacity-assignment strategy). It
follows a three-stage design:

1. **Stage 1 — Monte Carlo data generation.** For sampled priority orderings,
   network states are drawn and a sequential capacity-assignment procedure
   computes the joint-reliability label.
2. **Stage 2 — surrogate training.** A Transformer encoder (with an MLP head)
   learns to map an ordering, together with its demand vector, to a predicted
   joint reliability; parameter-matched MLP and tree-based baselines are provided
   for comparison.
3. **Stage 3 — surrogate-assisted search.** A permutation genetic algorithm uses
   the trained surrogate as a fast fitness function and verifies only its top
   candidates by full Monte Carlo simulation.

---

## Repository structure

The code is organized by function. Each script adds the package directories to
`sys.path` at import time, so scripts may be run directly from the repository
root (for example, `python experiments/run_nsfnet.py --K 5`).

```
.
├── README.md  BENCHMARK.md  requirements.txt  LICENSE  .gitignore
│
├── core/              Stages 1–3 engine
│   ├── MC_QP_multi_st.py     Multi-state quickest-path model and sequential assignment
│   ├── data_generator.py     Stage 1: MC data generation, CRN pre-sampling, LHS demand sampling, calibration
│   ├── train_surrogate.py    Stage 2: Transformer / MLP surrogate training
│   └── ga_optimizer.py       Stage 3: permutation GA (PMX crossover, swap/insert mutation) + MC verification
│
├── models/            Surrogate model definitions (transformer_surrogate.py, mlp_baseline.py)
│
├── instances/         Network instances and benchmark construction
│   ├── nsfnet_data.py        NSFNET instance (14 nodes, 21 links, K = 5)
│   ├── usnet_data.py         USNET instance (24 nodes, 43 links, K = 10)
│   ├── benchmark_network.py  Builds the real-topology networks for the pipeline
│   └── select_usnet_pairs.py Automatic selection of the USNET source–terminal pairs
│
├── experiments/       Experiment drivers
│   ├── run_experiments.py    Synthetic experiments (K = 3, 5, 10, 15, 20)
│   ├── run_nsfnet.py         Real-topology case studies: NSFNET (K = 5), USNET (K = 10)
│   ├── run_multiseed.py      Multi-seed robustness (5 independent instances/seeds)
│   ├── run_grouped_split.py  Grouped-split retraining of the neural surrogates
│   ├── run_fair_mlp.py       Parameter-matched / high-capacity / one-hot MLP variants
│   ├── run_ablation.py       GA-hyperparameter, architecture, and data-generation ablations
│   ├── run_demand_reuse.py   Surrogate reuse across demand vectors (amortization, OOD probe)
│   └── run_independent_items.py  Label-noise and Stage-1 cost analysis
│
├── baselines/         Baseline surrogates and search strategies
│   ├── baseline_surrogates.py    Random Forest / XGBoost surrogate baselines
│   ├── baseline_optimizers.py    Random / hill climbing / simulated annealing / greedy search
│   └── grouped_split_test.py     Grouped-split-by-permutation evaluation (tree models)
│
├── figures_code/      Plotting scripts (plot_diagnostics, plot_ga_convergence, plot_attention, probe_attention, make_drawio)
│
├── data/              Result JSON files and trained checkpoints (per instance)
└── figures/           Generated figures used in the paper
```

### A note on the datasets

The large per-split Monte Carlo training datasets (`train.npz`, `val.npz`,
`test.npz`, ~180 MB in total) are **not** included in the git repository because
they are fully regenerable from the provided scripts and fixed random seeds. To
recreate them, run the data-generation step of the corresponding driver. The small
`metadata.npz` files (network parameters and demand ranges), the **trained
surrogate checkpoints** (`*.pt`), and the **result files** (`*.json`) *are*
included, so the reported numbers and figures can be reproduced directly without
retraining. Because the random seeds are fixed, re-running the generation step
reproduces those datasets exactly.

---

## Installation

```bash
python -m venv venv && source venv/bin/activate     # optional
pip install -r requirements.txt
```

On some Windows / Intel-OpenMP setups, `numpy`/`torch` may raise an OpenMP
duplicate-runtime error. The scripts already set `KMP_DUPLICATE_LIB_OK=TRUE`
before importing `numpy`; if you invoke functions directly, set the same
environment variable.

---

## Reproducing the paper

All commands are run from the repository root. Each command regenerates the data
it needs (or reuses the included checkpoints) and prints or writes the reported
quantities. GPU (`--device cuda`) is recommended for the larger scales.

| Paper item | Command |
|---|---|
| Synthetic experiments (K = 3, 5, 10, 15, 20): accuracy, GA, timing | `python experiments/run_experiments.py` |
| Real-topology case studies — NSFNET (K = 5), USNET (K = 10) | `python experiments/run_nsfnet.py --K 5` and `--K 10` |
| Demand reuse / amortization and OOD probe | `python experiments/run_demand_reuse.py --data nsfnet_K5 --topology nsfnet --K 5` (and USNET) |
| Tree-based surrogate baselines | `python baselines/baseline_surrogates.py` |
| Grouped split by permutation | `python experiments/run_grouped_split.py --data usnet_K10 --stage all` |
| Search-strategy ablation | `python baselines/baseline_optimizers.py` |
| Multi-seed robustness | `python experiments/run_multiseed.py --stage run --Ks 5 10 --seeds 42 43 44 45 46` then `--stage aggregate` |
| Fair MLP comparison | `python experiments/run_fair_mlp.py --Ks 15 20` |
| GA-hyperparameter and architecture ablations | `python experiments/run_ablation.py --groups ga arch` (and `--groups ga --K 15`) |
| Data-generation ablation (CRN, LHS) | `python experiments/run_ablation.py --groups data` |
| Label-noise and Stage-1 cost analysis | `python experiments/run_independent_items.py` |
| Diagnostic figures | `python figures_code/plot_diagnostics.py` |
| GA convergence figure | `python figures_code/plot_ga_convergence.py` |
| Self-attention heatmap | `python figures_code/plot_attention.py` |

The full generation protocol for the real-topology benchmarks (node coordinates,
link lists, source–terminal pairs, edge-attribute distributions, path
enumeration, and demand calibration) is documented in
[`BENCHMARK.md`](BENCHMARK.md), with the machine-readable source in
[`instances/`](instances/).

---

## Citation

If you use this code, please cite:

> Ding-Hsiang Huang (2026). Surrogate-assisted priority ordering optimization
> for joint reliability in multi-state flow networks. *Engineering Applications
> of Artificial Intelligence*, 116689. https://doi.org/10.1016/j.engappai.2026.116689

```bibtex
@article{huang2026priority,
  author  = {Huang, Ding-Hsiang},
  title   = {Surrogate-assisted priority ordering optimization for joint
             reliability in multi-state flow networks},
  journal = {Engineering Applications of Artificial Intelligence},
  pages   = {116689},
  year    = {2026},
  doi     = {10.1016/j.engappai.2026.116689}
}
```

## License

Released under the MIT License; see [`LICENSE`](LICENSE).
