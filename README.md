# Hodge VNext · real48

Hodge network dissimilarity based on **independent exclusive q-layers**, with manual alpha sweeps, learned alpha, and learned q-layer weights on 48 real networks. Here q is maximal-clique size minus one. Each layer has its own downward closure; no global complex is constructed.

## Files and environment

| Path | Contents |
| --- | --- |
| `hodge_vnext/*.py` | Maximal cliques, simplices, Hodge operators, Exact/SLQ, heat traces, distances, and staged caching |
| `benchmarks/baselines/*.py` | DMeasure, NetLSD, Portrait, Count, and other baselines with provenance |
| `validation/*.py` | Parameter learning, train/test splits, retrieval metrics, and plotting |
| `scripts/*.py` / `experiments/real/*.json` | Run entry points / parameters, seeds, and experiment protocols |
| `data/real_domains/` | `.edgelist` edge lists, `.nodes` node lists, and a CSV provenance manifest for 48 networks |
| `tests/` | Core, gradient, split, and cache regression tests |
| `pyproject.toml` / `requirements.txt` | Package dependencies / pinned versions of the validation environment |

CPU implementation; Python >=3.12. Validated on Windows with Python 3.12.7. The benchmark contains five domains with 10/9/10/10/9 networks. Keep the node lists: they preserve isolated vertices. Data and baseline sources are recorded in the manifest and `UPSTREAM_PROVENANCE`. Respect the original source licenses when publishing; this repository does not relicense third-party data.

```bash
python -m venv .venv
# Activate on Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Activate on Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m pytest -q
```

## Run experiments

Run from the repository root. Relative paths in JSON configurations are resolved against that root.

```bash
# Five-method comparison with a fixed configuration
python scripts/run_real_domain_validation.py --config experiments/real/comparison.json

# Manual alpha sweeps: maximal cliques up to sizes 4, 8, and 10
# alpha = -1, -0.75, ..., 1
python scripts/run_real_alpha_sweep.py --config experiments/real/manual_s4.json
python scripts/run_real_alpha_sweep.py --config experiments/real/manual_s8.json
python scripts/run_real_alpha_sweep.py --config experiments/real/manual_s10.json

# Learned alpha: training-set NCA loss + Adam; 5 repeats x 4 folds
python scripts/run_real_alpha_gradient.py --config experiments/real/alpha_adam.json

# Free q-layer weights: q=0..9, training-set NCA + Adam; 5 repeats x 4 folds
python scripts/run_real_q_weight_learning.py --config experiments/real/q_weights_cv.json

# Per-q RMS scaling x Uniform / Single-alpha / Free-q; 10 repeats x 5 folds
python scripts/run_real_q_weight_scale_cv.py --config experiments/real/q_scale_cv.json
```

Additional configurations: `alpha_holdout.json` (one 80/20 split) and `q_weights_holdout10.json` (ten 80/20 splits), using the corresponding alpha/q entry points. Run `nested_cv.json` with `run_real_learning_nested.py` to select S/Q/qmin/normalization/alpha through inner CV; candidate values are in the JSON. The original learned-alpha configuration uses S=all and Q=all. Large-clique networks may require substantial memory; this configuration has not been replaced with a truncated experiment.

## Method, evaluation, and caching

- Heat trace: `h(t)=tr(exp(-tL))`. `per_simplex` divides by the dimension of the current `L_k^(q)`; `per_node` divides by the original network's node count. Neither rescales L or its eigenvalues.
- S retains only **globally maximal cliques** of size <=S in the original graph; it does not split larger cliques. Original alpha model: `A_q=min(q,Q)+1` (A_q=q+1 for Q=all), `W_q=A_q^alpha / sum_r(A_r^alpha)`, with matrix weight `W_q/A_q`. Free-q weights use softmax. Both are renormalized over q<=q* for each pair, where q* is the larger of the two retained maximum q values. Missing channels are zero-filled.
- Single-alpha in `q_scale_cv` uses a different parameterization: `beta_q=alpha*q/9`. RMS scales use training pairs only: `s_q=sqrt(mean_train(C_q))`, `C_scaled_q=C_q/(s_q+eps)^2`, where C_q is the mean squared channel distance within layer q.
- Automated tuning never uses outer-test labels. Reports include AUROC, 1-NN, P@k, mAP, and NDCG. Full-benchmark manual sweeps are descriptive, not unbiased test scores.
- `cache/` stores graphs, maximal cliques, simplices, boundaries, Hodge operators, spectral objects, heat traces, profiles, and baselines. Changing alpha reuses profiles; increasing SLQ probes computes only missing probes. Load only trusted pickle caches you generated yourself.
- `results/` stores configurations/splits, raw arrays (`.npy/.npz`), metrics (`.csv`), and figures (`.png/.pdf`). Caches and results are excluded from Git. `--force` regenerates run reports while reusing valid caches. See each entry point's `--help` for more options.
