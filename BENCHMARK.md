# Benchmark-Generation Protocol: Real-Topology Instances

This document specifies the two real-topology benchmark instances (NSFNET and
USNET) completely, so that they can be reconstructed or extended independently of
this code base. The machine-readable source of every quantity below is
[`instances/nsfnet_data.py`](instances/nsfnet_data.py),
[`instances/usnet_data.py`](instances/usnet_data.py), and
[`instances/benchmark_network.py`](instances/benchmark_network.py); the values in
this document are copied from those files.

Both instances follow the same protocol: a recognized backbone topology with
geographic node coordinates, undirected physical links whose capacity is shared
by all traffic crossing them, *K* source–terminal (s–t) pairs competing for that
shared capacity, heterogeneous per-link stochastic attributes derived from link
length, candidate quickest paths enumerated directly on the topology, and a
demand calibration that places the instance in the informative reliability
regime.

---

## 1. Instance summary

| | NSFNET | USNET |
|---|---|---|
| Nodes | 14 (US cities, lon/lat coordinates) | 24 (US cities, lon/lat coordinates) |
| Undirected links (shared-capacity arcs) | 21 | 43 |
| s–t pairs | K = 5 (first five of a documented list of 10) | K = 10 |
| Components per arc, `CMAX` | 9 | 30 |
| Component reliability range, `REL_HIGH`–`REL_LOW` | 0.95–0.85 | 0.95–0.85 |
| Lead-time scale, `LT_BASE` / `LT_SPAN` | 3.0 / 7.0 | 3.0 / 7.0 |
| Common time budget, `T_BUDGET` | 50.0 | 50.0 |
| Candidate paths per pair, `M_PER_PAIR` (upper bound) | 10 | 10 |
| Maximum arcs per candidate path, `MAX_PATH_ARCS` | 5 | 6 |

The NSFNET topology is the classical 14-node U.S. National Science Foundation
backbone (including the two transcontinental trunks Palo Alto–Champaign and Salt
Lake City–College Park). The USNET topology is a 24-node, 43-link U.S. continental
backbone. Node identities, coordinates, and the full link lists are given in the
instance files.

`CMAX` differs between the instances by design: with 10 pairs competing on 43
arcs, a larger per-arc component count (30 versus 9) is required for the 10-way
joint reliability to fall in the informative regime (mean near 0.5) — the
real-topology analogue of scaling network size with *K* in synthetic experiments,
where a fixed real topology cannot gain additional arcs.

## 2. Source–terminal pairs

Pairs are chosen to span the network and to overlap on shared links, so that
priority ordering creates genuine capacity competition. NSFNET (K = 5 uses the
first five):

| # | NSFNET pair | # | USNET pair |
|---|---|---|---|
| P1 | Palo Alto → Atlanta | P1 | Seattle → Los Angeles |
| P2 | Seattle → Princeton | P2 | Portland → San Diego |
| P3 | San Diego → Pittsburgh | P3 | San Francisco → Phoenix |
| P4 | Lincoln → Ann Arbor | P4 | Salt Lake City → El Paso |
| P5 | Salt Lake City → Houston | P5 | Las Vegas → Dallas |
| P6 | Boulder → Ithaca | P6 | Denver → Houston |
| P7 | Champaign → College Park | P7 | Albuquerque → Chicago |
| P8 | Palo Alto → Princeton | P8 | Kansas City → Washington |
| P9 | San Diego → Ann Arbor | P9 | Minneapolis → Atlanta |
| P10 | Seattle → Atlanta | P10 | St. Louis → New York |

The USNET pairs were selected automatically
([`instances/select_usnet_pairs.py`](instances/select_usnet_pairs.py)) under the
criteria: shortest-path length of 3–4 hops, at least 8 candidate paths within the
length cap, and geographically dispersed endpoints.

## 3. Link attributes (heterogeneous, length-derived)

Let `len_i` be the great-circle (haversine) length of link *i* computed from the
node coordinates, and `frac_i = len_i / max_j len_j` its normalized length.

**Capacity.** Each arc consists of `CMAX` independent binary components, each
operating with probability `p_i`, so the arc capacity is a random variable

```
X_i ~ Binomial(CMAX, p_i),      state space {0, 1, ..., CMAX}
p_i = REL_HIGH - (REL_HIGH - REL_LOW) * frac_i
```

Shorter links are more reliable (`p_i` up to 0.95); the longest link has
`p_i = 0.85`.

**Lead time.** Each arc's lead time is exponentially distributed,

```
L_i ~ Exponential(mean = mu_i)
mu_i = LT_BASE + LT_SPAN * frac_i
```

so mean lead times range from 3.0 (shortest link) to about 10.0 (longest link).
The scale is chosen such that a candidate path of up to `MAX_PATH_ARCS` arcs
remains feasible within the common time budget `T_BUDGET = 50` for every pair.

## 4. Candidate-path enumeration

For each s–t pair, candidate paths are enumerated directly on the topology as the
shortest simple paths in hop count (`networkx.shortest_simple_paths`, i.e. Yen's
algorithm), keeping only paths with at most `MAX_PATH_ARCS` arcs, up to
`M_PER_PAIR = 10` paths per pair. Because enumeration is performed on the real
topology rather than prescribed, the resulting path sets are heterogeneous across
pairs — different pairs naturally obtain different numbers of candidate paths
(roughly 5–10) and different path lengths — and overlapping pairs share arcs,
which is precisely what makes the priority ordering consequential.

## 5. Demand calibration

Demand ranges are calibrated per instance so that the mean joint reliability over
sampled orderings and demand vectors is approximately 0.5, which maximizes the
information content of the training labels (labels saturated near 0 or 1 are
uninformative for ranking orderings). Calibration performs a binary search on a
scaling factor applied to per-pair reference demands derived from the observed
maximum deliverable flows, using Latin Hypercube samples of the demand space (see
`calibrate_d_ranges_joint` in [`core/data_generator.py`](core/data_generator.py)).
The calibrated demand ranges for each instance are stored in that instance's
`data/<instance>/metadata.npz`.

## 6. Reconstruction

Running

```bash
python experiments/run_nsfnet.py --K 5              # NSFNET, ground truth available
python experiments/run_nsfnet.py --K 10             # USNET (selected automatically for K = 10)
```

rebuilds each instance from the specification above (topology → attributes →
path enumeration → calibration), regenerates the Monte Carlo training data,
trains the surrogates, and runs the surrogate-assisted search, using the fixed
random seeds recorded in the scripts.
