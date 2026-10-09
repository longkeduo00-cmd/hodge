from __future__ import annotations

"""Supervised learning of exclusive-q Hodge layer weights on real networks.

First-stage scientific contract
-------------------------------
* Hodge construction is unchanged and each exclusive q-layer remains independent.
* Compute at most maximal K10, i.e. q=0..9. K11+ Hodge layers are not built.
* Q=all, qmin=0, normalization=per_simplex are fixed.
* The learned parameters are ten q-layer logits beta_q.  Their softmax gives
  nonnegative weights summing to one on the full q=0..9 schema.
* For each graph pair, logits are masked to its original pair-specific q* and
  renormalized; the existing pair-adaptive schema is preserved.
* Training objective is NCA/soft-nearest-neighbour loss on outer-training labels.
* No smoothness, entropy, L1, L2, sparsity, or other regularization coefficient
  is introduced in this first version.
* Outer-test labels are never used for fitting weights.
"""

from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import json
import time

import numpy as np
import pandas as pd

from hodge_vnext.cache import CACHE_SCHEMA
from hodge_vnext.graph_io import GraphRecord, create_real_record
from hodge_vnext.profile import GraphProfile, compute_graph_profile
from hodge_vnext.distance import profile_distance
from hodge_vnext.utils import safe_token, stable_hash, write_json

from .alpha_gradient import build_pair_layer_energies
from .config import method_from_payload, resolve_path
from .methods import MethodManager, PRIMARY_METHOD
from .q_weight_learning import (
    QWeightFitResult,
    average_effective_pair_weights,
    beta_from_scalar_alpha,
    distance_matrix_from_beta,
    fit_q_weights_adam,
    softmax_weights,
    weight_diagnostics,
)
from .real_domains import _prepare_manifest
from .real_learning import (
    _domain_metric_rows,
    _plot_square_method,
    _square_metrics,
    cross_partition_metrics,
)
from .real_learning_nested import (
    _aggregate_metric_frame,
    _compute_or_load_baseline_matrix,
    _coverage_table,
    _outer_split_membership_table,
    make_repeated_outer_splits,
)


LEARNED_METHOD = "HodgeLearnedQWeights"
FROZEN_METHOD = "HodgeFrozenAlpha0"
MAX_CLIQUE_SIZE = 10
NUM_Q = 10  # q=0..9 corresponds maximal K1..K10


def _progress(msg: str) -> None:
    print(msg, flush=True)


def _validate_scientific_fixed_config(cfg) -> None:
    if cfg.max_clique_size != MAX_CLIQUE_SIZE:
        raise ValueError(
            f"q-weight experiment requires method.max_clique_size={MAX_CLIQUE_SIZE} "
            "(compute maximal K10 and below only)"
        )
    if cfg.max_hodge_order is not None:
        raise ValueError("q-weight experiment requires method.max_hodge_order=null (Q=all)")
    if cfg.min_layer_q != 0:
        raise ValueError("q-weight experiment requires method.min_layer_q=0")
    if cfg.normalization != "per_simplex":
        raise ValueError("q-weight experiment requires method.normalization='per_simplex'")


def _optimizer_kwargs(learning: dict[str, Any]) -> dict[str, Any]:
    opt = dict(learning.get("optimizer", {}))
    return {
        "learning_rate": float(opt.get("learning_rate", 0.05)),
        "max_epochs": int(opt.get("max_epochs", 2000)),
        "beta1": float(opt.get("beta1", 0.9)),
        "beta2": float(opt.get("beta2", 0.999)),
        "adam_eps": float(opt.get("adam_eps", 1e-8)),
        "grad_tolerance": float(opt.get("grad_tolerance", 1e-8)),
        "loss_tolerance": float(opt.get("loss_tolerance", 1e-10)),
        "patience": int(opt.get("patience", 100)),
    }


def _run_stem(n: int, benchmark_id: str, payload: dict[str, Any], config_path: Path) -> str:
    cfg = method_from_payload(payload, config_path)
    learning = dict(payload.get("learning", {}))
    token = stable_hash({
        "benchmark_id": benchmark_id,
        "n": n,
        "method": {
            "max_clique_size": cfg.max_clique_size,
            "Q": cfg.max_hodge_order,
            "qmin": cfg.min_layer_q,
            "normalization": cfg.normalization,
            "solver": cfg.solver,
            "slq": cfg.slq.__dict__,
            "times": [cfg.time_min, cfg.time_max, cfg.time_points],
        },
        "model": "free_q_softmax_pair_adaptive",
        "regularization": "none",
        "optimizer": _optimizer_kwargs(learning),
        "outer": learning.get("outer_cv", {}),
    }, length=12)
    return f"real{n}_{safe_token(benchmark_id)}_learned-q-weights_{token}"


def _compute_profile_worker(index: int, rec: GraphRecord, cfg) -> tuple[int, GraphProfile, float]:
    t0 = time.perf_counter()
    profile = compute_graph_profile(rec, cfg, progress=None)
    return index, profile, float(time.perf_counter() - t0)


def _pad_energies(energies: np.ndarray, num_q: int) -> np.ndarray:
    if energies.shape[2] > num_q:
        raise AssertionError(f"energy q dimension {energies.shape[2]} exceeds fixed budget {num_q}")
    if energies.shape[2] == num_q:
        return energies
    out = np.zeros((energies.shape[0], energies.shape[1], num_q), dtype=float)
    out[:, :, : energies.shape[2]] = energies
    return out


def _profiles_metadata(profiles: Sequence[GraphProfile], num_q: int) -> tuple[np.ndarray, np.ndarray]:
    qmax = np.asarray([(-1 if p.q_max is None else int(p.q_max)) for p in profiles], dtype=np.int32)
    has_layer = np.zeros((len(profiles), num_q), dtype=np.uint8)
    for i, p in enumerate(profiles):
        for q, _k in p.channels:
            if 0 <= int(q) < num_q:
                has_layer[i, int(q)] = 1
    return qmax, has_layer


def _load_or_compute_layer_energies(
    records: Sequence[GraphRecord],
    cfg,
    run: Path,
    runtime_rows: list[dict[str, Any]],
    *,
    recompute: bool,
    profile_workers: int,
    progress: Callable[[str], None],
) -> tuple[list[GraphProfile] | None, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load cheap run artifact or reconstruct it from staged Q7 profile cache."""
    path = run / "arrays" / "hodge_pair_layer_energies_q0_q9.npz"
    if path.exists() and not recompute:
        obj = np.load(path)
        energies = np.asarray(obj["energies"], dtype=float)
        q_star = np.asarray(obj["q_star"], dtype=np.int32)
        profile_qmax = np.asarray(obj["profile_qmax"], dtype=np.int32)
        has_layer = np.asarray(obj["has_layer"], dtype=np.uint8)
        if (
            energies.shape == (len(records), len(records), NUM_Q)
            and q_star.shape == (len(records), len(records))
            and profile_qmax.shape == (len(records),)
            and has_layer.shape == (len(records), NUM_Q)
        ):
            progress("[HODGE CACHE] q0..q9 pair-layer energy tensor hit; profile/Hodge recomputation skipped")
            return None, energies, q_star, profile_qmax, has_layer

    workers = max(1, min(int(profile_workers), len(records))) if records else 1
    profiles_tmp: list[GraphProfile | None] = [None] * len(records)
    if workers == 1:
        for i, rec in enumerate(records):
            idx, profile, elapsed = _compute_profile_worker(i, rec, cfg)
            profiles_tmp[idx] = profile
            runtime_rows.append({
                "stage": "qweight_profile", "label": rec.source.get("label", str(i)),
                "elapsed_sec": elapsed, "solver": cfg.solver,
                "max_clique_size": cfg.max_clique_size,
                "max_hodge_order": cfg.max_hodge_order,
                "profile_workers": workers,
            })
            progress(f"[PROFILE] {i + 1}/{len(records)} {rec.source.get('label', str(i))} completed")
    else:
        progress(f"[PROFILE PARALLEL] workers={workers}; one network per process")
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as executor:
            future_map = {executor.submit(_compute_profile_worker, i, rec, cfg): (i, rec) for i, rec in enumerate(records)}
            completed = 0
            for future in as_completed(future_map):
                i, rec = future_map[future]
                idx, profile, elapsed = future.result()
                profiles_tmp[idx] = profile
                completed += 1
                runtime_rows.append({
                    "stage": "qweight_profile", "label": rec.source.get("label", str(i)),
                    "elapsed_sec": elapsed, "solver": cfg.solver,
                    "max_clique_size": cfg.max_clique_size,
                    "max_hodge_order": cfg.max_hodge_order,
                    "profile_workers": workers,
                })
                progress(f"[PROFILE PARALLEL] {completed}/{len(records)} {rec.source.get('label', str(i))} completed")
    if any(p is None for p in profiles_tmp):
        raise RuntimeError("profile stage completed with missing profiles")
    profiles = [p for p in profiles_tmp if p is not None]
    raw_energies, q_star = build_pair_layer_energies(
        profiles, min_layer_q=0, max_hodge_order=None
    )
    energies = _pad_energies(raw_energies, NUM_Q)
    if np.any(q_star >= NUM_Q):
        raise AssertionError("pair q* exceeds q=9 despite max_clique_size=10")
    profile_qmax, has_layer = _profiles_metadata(profiles, NUM_Q)
    np.savez_compressed(
        path,
        energies=energies,
        q_star=q_star,
        profile_qmax=profile_qmax,
        has_layer=has_layer,
    )
    progress(f"[HODGE CACHE] saved q0..q9 pair-layer energy tensor shape={energies.shape}")
    return profiles, energies, q_star, profile_qmax, has_layer


def _audit_scalar_family_embedding(
    profiles: Sequence[GraphProfile], energies: np.ndarray, q_star: np.ndarray, cfg
) -> pd.DataFrame:
    """Free-q logits must exactly contain the original scalar-alpha family."""
    rows: list[dict[str, Any]] = []
    n = len(profiles)
    pairs = [(i, j) for i in range(min(n, 4)) for j in range(i + 1, min(n, 4))]
    for alpha in (-16.0, 0.0, 2.0):
        beta = beta_from_scalar_alpha(alpha, NUM_Q, max_hodge_order=None)
        D = distance_matrix_from_beta(beta, energies, q_star)
        cfg_a = replace(cfg, alpha=float(alpha))
        for i, j in pairs:
            core = profile_distance(profiles[i], profiles[j], cfg_a).total_distance
            recon = float(D[i, j])
            err = abs(core - recon)
            rows.append({
                "alpha": alpha, "i": i, "j": j,
                "core_distance": core, "qweight_embedded_distance": recon,
                "abs_error": err,
            })
            if err > 1e-10 * max(1.0, abs(core)):
                raise AssertionError(f"scalar-family embedding mismatch alpha={alpha} pair=({i},{j}) err={err}")
    return pd.DataFrame(rows)


def _layer_availability_table(
    profile_qmax: np.ndarray,
    has_layer: np.ndarray,
    energies: np.ndarray,
    q_star: np.ndarray,
) -> pd.DataFrame:
    n = len(profile_qmax)
    rows: list[dict[str, Any]] = []
    iu = np.triu_indices(n, k=1)
    for q in range(NUM_Q):
        active = q_star[iu] >= q
        e = energies[:, :, q][iu]
        rows.append({
            "q": q,
            "clique_size": q + 1,
            "graphs_with_exclusive_layer": int(np.sum(has_layer[:, q] > 0)),
            "graphs_with_qmax_at_least_q": int(np.sum(profile_qmax >= q)),
            "active_pair_schema_count": int(np.sum(active)),
            "nonzero_energy_pair_count": int(np.sum(np.logical_and(active, e > 0))),
        })
    return pd.DataFrame(rows)


def _split_inventory_tables(
    splits, labels: Sequence[str], domains: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Describe every repeated holdout and pairwise test-set overlap.

    Repeated holdouts are correlated resamples, not independent datasets.  These
    tables make that dependence inspectable and also guard against accidentally
    evaluating the exact same test set twice.
    """
    labels_arr = np.asarray(labels, dtype=str)
    domains_arr = np.asarray(domains, dtype=str)
    summary_rows: list[dict[str, Any]] = []
    test_sets: dict[str, set[int]] = {}
    for split in splits:
        te = np.asarray(split.test_idx, dtype=int)
        tr = np.asarray(split.train_idx, dtype=int)
        test_set = set(int(x) for x in te)
        test_sets[split.outer_id] = test_set
        dcounts = pd.Series(domains_arr[te]).value_counts().to_dict()
        summary_rows.append({
            "outer_id": split.outer_id,
            "repeat": split.repeat,
            "fold": split.fold,
            "outer_seed": split.seed,
            "train_size": int(len(tr)),
            "test_size": int(len(te)),
            "test_indices": ";".join(str(int(x)) for x in te),
            "test_labels": ";".join(labels_arr[te].tolist()),
            "test_domains": ";".join(domains_arr[te].tolist()),
            "test_domain_counts": ";".join(f"{k}:{int(v)}" for k, v in sorted(dcounts.items())),
        })

    overlap_rows: list[dict[str, Any]] = []
    ids = [split.outer_id for split in splits]
    for ai in range(len(ids)):
        for bi in range(ai + 1, len(ids)):
            a = test_sets[ids[ai]]
            b = test_sets[ids[bi]]
            inter = len(a & b)
            union = len(a | b)
            overlap_rows.append({
                "outer_id_a": ids[ai],
                "outer_id_b": ids[bi],
                "test_intersection": int(inter),
                "test_union": int(union),
                "test_jaccard": float(inter / union) if union else 0.0,
                "identical_test_set": bool(a == b),
            })
    return pd.DataFrame(summary_rows), pd.DataFrame(overlap_rows)


def _fit_weights_table(
    outer,
    fit: QWeightFitResult,
    q_star: np.ndarray,
) -> pd.DataFrame:
    train_eff = average_effective_pair_weights(fit.beta, q_star, outer.train_idx)
    test_train_eff = average_effective_pair_weights(fit.beta, q_star, outer.test_idx, outer.train_idx)
    train_sub = q_star[np.ix_(outer.train_idx, outer.train_idx)]
    train_qmax = int(np.max(train_sub)) if train_sub.size else -1
    train_schema = np.full(NUM_Q, np.nan, dtype=float)
    if train_qmax >= 0:
        train_schema[: train_qmax + 1] = softmax_weights(fit.beta[: train_qmax + 1])
    rows = []
    for q in range(NUM_Q):
        rows.append({
            "outer_id": outer.outer_id,
            "repeat": outer.repeat,
            "fold": outer.fold,
            "outer_seed": outer.seed,
            "q": q,
            "clique_size": q + 1,
            "beta": float(fit.beta[q]),
            "train_identifiable": bool(q <= train_qmax),
            "train_schema_weight": float(train_schema[q]) if np.isfinite(train_schema[q]) else np.nan,
            "reference_weight": float(fit.reference_weights[q]),
            "mean_train_pair_effective_weight": float(train_eff[q]),
            "mean_test_train_pair_effective_weight": float(test_train_eff[q]),
        })
    return pd.DataFrame(rows)


def run_real_q_weight_learning(
    payload: dict[str, Any],
    config_path: Path,
    *,
    force: bool = False,
    recompute_hodge: bool = False,
    recompute_baselines: bool = False,
    progress: Callable[[str], None] | None = None,
) -> Path:
    progress = progress or _progress
    cfg = method_from_payload(payload, config_path)
    _validate_scientific_fixed_config(cfg)
    exp = dict(payload["real_validation"])
    learning = dict(payload.get("learning", {}))
    execution = dict(payload.get("execution", {}))
    outer_cfg = dict(learning.get("outer_cv", {}))
    opt_kwargs = _optimizer_kwargs(learning)
    profile_workers = max(1, int(execution.get("profile_workers", 1)))

    df = _prepare_manifest(payload, config_path)
    labels = df["label"].astype(str).tolist()
    domains = df["domain"].astype(str).tolist()
    unique_domains = list(dict.fromkeys(domains))
    if len(unique_domains) < 2:
        raise ValueError("q-weight learning requires at least two domains")

    strategy = str(outer_cfg.get("strategy", "repeated_stratified_holdout"))
    repeats = int(outer_cfg.get("repeats", 1))
    folds = int(outer_cfg.get("folds", 4))
    outer_seed = int(outer_cfg.get("seed", 20260930))
    test_fraction = float(outer_cfg.get("test_fraction", 0.2))
    min_test = int(outer_cfg.get("require_min_test_per_domain", 2))
    outer_splits = make_repeated_outer_splits(
        domains,
        strategy=strategy,
        repeats=repeats,
        seed=outer_seed,
        outer_folds=folds,
        test_fraction=test_fraction,
        require_min_test_per_domain=min_test,
    )
    membership = _outer_split_membership_table(df, outer_splits)
    coverage = _coverage_table(membership)
    split_inventory, split_overlap = _split_inventory_tables(outer_splits, labels, domains)
    if "holdout" in strategy.lower() and not split_overlap.empty and bool(split_overlap["identical_test_set"].any()):
        dup = split_overlap.loc[split_overlap["identical_test_set"], ["outer_id_a", "outer_id_b"]]
        raise RuntimeError(f"repeated holdout generated duplicate test sets; inspect seeds: {dup.to_dict(orient='records')}")

    benchmark_id = str(exp.get("benchmark_id", "real48-learned-q-weights"))
    results_root = resolve_path(learning.get("results_root", "results/validation/real_q_weight_learning"), config_path)
    run = results_root / _run_stem(len(df), benchmark_id, payload, config_path)
    complete = run / "RUN_COMPLETE.json"
    if complete.exists() and not force:
        progress(f"[SKIP] completed q-weight run exists: {run}")
        return run
    run.mkdir(parents=True, exist_ok=True)
    for d in ("arrays", "tables", "plots", "outer"):
        (run / d).mkdir(parents=True, exist_ok=True)
    if force:
        for p in (run / "plots").glob("**/*"):
            if p.is_file():
                p.unlink()
        complete.unlink(missing_ok=True)

    membership.to_csv(run / "tables" / "outer_split_membership.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(run / "tables" / "outer_test_coverage.csv", index=False, encoding="utf-8-sig")
    split_inventory.to_csv(run / "tables" / "outer_split_inventory.csv", index=False, encoding="utf-8-sig")
    split_overlap.to_csv(run / "tables" / "outer_split_overlap.csv", index=False, encoding="utf-8-sig")

    effective = json.loads(json.dumps(payload))
    effective["cache_schema"] = CACHE_SCHEMA
    effective["learned_q_weight_contract"] = {
        "computed_maximal_clique_sizes": "K1..K10 only; K11+ Hodge layers not built",
        "q_layers": list(range(NUM_Q)),
        "learned_parameterization": "beta_q logits with pair-adaptive masked softmax",
        "regularization": "none in this first-stage experiment",
        "training_loss": "NCA / soft-nearest-neighbour differentiable surrogate",
        "qmin_fixed": 0,
        "normalization_fixed": "per_simplex",
        "Q": "all",
        "final_metrics_not_optimized_directly": ["1-NN Macro-F1", "AUROC", "mAP", "NDCG", "ARI", "NMI", "MST"],
    }
    write_json(run / "effective_config.json", effective)

    records: list[GraphRecord] = []
    for _, row in df.iterrows():
        rec = create_real_record(row.path, cfg.cache.root, str(row.label), row.nodes_path or None)
        records.append(rec)
        progress(f"[REAL] {row.domain}/{row.label}: N={rec.graph.number_of_nodes()} E={rec.graph.number_of_edges()}")

    runtime_rows: list[dict[str, Any]] = []
    profiles, energies, q_star, profile_qmax, has_layer = _load_or_compute_layer_energies(
        records, cfg, run, runtime_rows,
        recompute=recompute_hodge,
        profile_workers=profile_workers,
        progress=progress,
    )
    layer_availability = _layer_availability_table(profile_qmax, has_layer, energies, q_star)
    layer_availability.to_csv(run / "tables" / "q_layer_availability.csv", index=False, encoding="utf-8-sig")

    audit_path = run / "tables" / "q_weight_scalar_family_embedding_audit.csv"
    if profiles is not None:
        audit_df = _audit_scalar_family_embedding(profiles, energies, q_star, cfg)
        audit_df.to_csv(audit_path, index=False, encoding="utf-8-sig")
    elif not audit_path.exists():
        progress("[AUDIT] energy tensor was loaded without profiles; reconstructing cached profiles once for scalar-family audit")
        audit_profiles = [compute_graph_profile(rec, cfg, progress=None) for rec in records]
        audit_df = _audit_scalar_family_embedding(audit_profiles, energies, q_star, cfg)
        audit_df.to_csv(audit_path, index=False, encoding="utf-8-sig")

    uniform_beta = np.zeros(NUM_Q, dtype=float)
    frozen_D = distance_matrix_from_beta(uniform_beta, energies, q_star)
    np.save(run / "arrays" / f"distance_matrix_{FROZEN_METHOD}_full.npy", frozen_D)
    pd.DataFrame(frozen_D, index=labels, columns=labels).to_csv(
        run / "tables" / f"distance_matrix_{FROZEN_METHOD}_full.csv", encoding="utf-8-sig"
    )

    requested_baselines = [m for m in exp.get("methods", []) if m != PRIMARY_METHOD]
    baseline_methods = MethodManager.validate_methods(requested_baselines) if requested_baselines else []
    manager = MethodManager.from_config(cfg, payload.get("baselines", {}))
    baseline_matrices: dict[str, np.ndarray] = {}
    for method in baseline_methods:
        progress(f"[BASELINE] {method}")
        baseline_matrices[method] = _compute_or_load_baseline_matrix(
            method, records, labels, manager, run, runtime_rows,
            recompute=recompute_baselines, progress=progress,
        )

    linkage_method = str(exp.get("linkage", "average"))
    mst_permutations = int(outer_cfg.get("mst_permutations", exp.get("mst_permutations", 5000)))
    eval_seed = int(exp.get("seed", 20260826))

    fit_rows: list[dict[str, Any]] = []
    weight_frames: list[pd.DataFrame] = []
    traj_frames: list[pd.DataFrame] = []
    inductive_rows: list[dict[str, Any]] = []
    inductive_qframes: list[pd.DataFrame] = []
    inductive_rframes: list[pd.DataFrame] = []
    square_rows: list[dict[str, Any]] = []
    square_qframes: list[pd.DataFrame] = []
    square_rframes: list[pd.DataFrame] = []
    square_domain_rows: list[dict[str, Any]] = []

    for oi, outer in enumerate(outer_splits, start=1):
        progress(f"[OUTER] {oi}/{len(outer_splits)} {outer.outer_id}: train={len(outer.train_idx)} test={len(outer.test_idx)}")
        outer_dir = run / "outer" / outer.outer_id
        outer_dir.mkdir(parents=True, exist_ok=True)

        fit = fit_q_weights_adam(
            energies, q_star, outer.train_idx, domains, **opt_kwargs
        )
        diag = weight_diagnostics(fit.reference_weights)
        fit_row = {
            "outer_id": outer.outer_id,
            "repeat": outer.repeat,
            "fold": outer.fold,
            "outer_seed": outer.seed,
            "train_size": len(outer.train_idx),
            "test_size": len(outer.test_idx),
            "train_loss": fit.best_loss,
            "final_grad_norm": fit.final_grad_norm,
            "epochs": fit.epochs,
            "converged": fit.converged,
            "stop_reason": fit.stop_reason,
            "temperature": fit.temperature,
            **diag,
        }
        fit_rows.append(fit_row)
        weights_df = _fit_weights_table(outer, fit, q_star)
        weight_frames.append(weights_df)
        weights_df.to_csv(outer_dir / "learned_q_weights.csv", index=False, encoding="utf-8-sig")
        write_json(outer_dir / "learned_q_weight_summary.json", fit_row)

        traj = fit.trajectory.copy()
        traj.insert(0, "outer_id", outer.outer_id)
        traj_frames.append(traj)
        traj.to_csv(outer_dir / "q_weight_optimization_trajectory.csv", index=False, encoding="utf-8-sig")

        learned_D = distance_matrix_from_beta(fit.beta, energies, q_star)
        np.save(outer_dir / f"distance_matrix_{LEARNED_METHOD}_full.npy", learned_D)
        matrices: dict[str, np.ndarray] = {
            LEARNED_METHOD: learned_D,
            FROZEN_METHOD: frozen_D,
            **baseline_matrices,
        }
        test_set = set(outer.test_idx.tolist())
        pd.DataFrame({
            "index": np.arange(len(df), dtype=int),
            "label": labels,
            "domain": domains,
            "partition": ["test" if i in test_set else "train" for i in range(len(df))],
        }).to_csv(outer_dir / "split.csv", index=False, encoding="utf-8-sig")

        per_outer_inductive: list[dict[str, Any]] = []
        per_outer_square: list[dict[str, Any]] = []
        per_outer_inductive_qframes: list[pd.DataFrame] = []
        per_outer_inductive_rframes: list[pd.DataFrame] = []
        per_outer_square_qframes: list[pd.DataFrame] = []
        per_outer_square_rframes: list[pd.DataFrame] = []
        per_outer_domain_rows: list[dict[str, Any]] = []
        for method, D in matrices.items():
            met, qdf, rdf = cross_partition_metrics(D, outer.test_idx, outer.train_idx, domains, labels)
            meta = {
                "outer_id": outer.outer_id,
                "repeat": outer.repeat,
                "fold": outer.fold,
                "outer_seed": outer.seed,
                "train_size": len(outer.train_idx),
                "test_size": len(outer.test_idx),
                "method": method,
            }
            row = {**meta, **met}
            inductive_rows.append(row)
            per_outer_inductive.append(row)
            for col, val in reversed(list(meta.items())):
                qdf.insert(0, col, val)
                rdf.insert(0, col, val)
            inductive_qframes.append(qdf)
            inductive_rframes.append(rdf)
            per_outer_inductive_qframes.append(qdf.copy())
            per_outer_inductive_rframes.append(rdf.copy())

            sq, sq_qdf, sq_rdf, clustering = _square_metrics(
                D, outer.test_idx, labels, domains, method, linkage_method,
                mst_permutations, eval_seed + oi - 1,
            )
            sqrow = {**meta, **{k: v for k, v in sq.items() if k != "method"}}
            square_rows.append(sqrow)
            per_outer_square.append(sqrow)
            for col, val in reversed(list(meta.items())):
                sq_qdf.insert(0, col, val)
                sq_rdf.insert(0, col, val)
            square_qframes.append(sq_qdf)
            square_rframes.append(sq_rdf)
            per_outer_square_qframes.append(sq_qdf.copy())
            per_outer_square_rframes.append(sq_rdf.copy())
            for drow in _domain_metric_rows(method, sq_qdf):
                row_domain = {
                    "outer_id": outer.outer_id,
                    "repeat": outer.repeat,
                    "fold": outer.fold,
                    "outer_seed": outer.seed,
                    **drow,
                }
                square_domain_rows.append(row_domain)
                per_outer_domain_rows.append(row_domain)
            idx = np.asarray(outer.test_idx, dtype=int)
            pd.DataFrame({
                "label": [labels[i] for i in idx],
                "domain": [domains[i] for i in idx],
                "cluster": clustering.cluster_labels,
                "leaf_order_position": np.argsort(clustering.leaf_order),
            }).to_csv(outer_dir / f"cluster_assignments_{method}.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(
                D[np.ix_(idx, idx)],
                index=[labels[i] for i in idx], columns=[labels[i] for i in idx],
            ).to_csv(outer_dir / f"test_distance_matrix_{method}.csv", encoding="utf-8-sig")

        pd.DataFrame(per_outer_inductive).to_csv(outer_dir / "inductive_metrics.csv", index=False, encoding="utf-8-sig")
        pd.concat(per_outer_inductive_qframes, ignore_index=True).to_csv(outer_dir / "generalization_per_query.csv", index=False, encoding="utf-8-sig")
        pd.concat(per_outer_inductive_rframes, ignore_index=True).to_csv(outer_dir / "generalization_rankings.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(per_outer_square).to_csv(outer_dir / "square_test_metrics.csv", index=False, encoding="utf-8-sig")
        pd.concat(per_outer_square_qframes, ignore_index=True).to_csv(outer_dir / "square_test_dataset_metrics.csv", index=False, encoding="utf-8-sig")
        pd.concat(per_outer_square_rframes, ignore_index=True).to_csv(outer_dir / "square_test_rankings.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(per_outer_domain_rows).to_csv(outer_dir / "square_test_domain_metrics.csv", index=False, encoding="utf-8-sig")

        split_frame = pd.read_csv(outer_dir / "split.csv")
        split_counts = (
            split_frame.groupby(["domain", "partition"], sort=False).size()
            .unstack(fill_value=0).reset_index()
        )
        for col in ("train", "test"):
            if col not in split_counts:
                split_counts[col] = 0
        split_counts[["domain", "train", "test"]].to_csv(outer_dir / "split_domain_counts.csv", index=False, encoding="utf-8-sig")

        if bool(learning.get("generate_plots", True)):
            from .real_q_weight_learning_plotting import (
                plot_effective_vs_reference_weights,
                plot_learned_q_weights,
                plot_q_weight_loss,
                plot_q_weight_trajectories,
            )
            split_plot_dir = outer_dir / "plots"
            plot_learned_q_weights(weights_df, split_plot_dir / "learned_q_weights", title=f"{outer.outer_id}: learned exclusive-q weights")
            plot_effective_vs_reference_weights(weights_df, split_plot_dir / "reference_vs_effective_q_weights")
            plot_q_weight_trajectories(traj, split_plot_dir / "q_weight_optimization_trajectory")
            plot_q_weight_loss(traj, split_plot_dir / "q_weight_nca_loss")

        write_json(outer_dir / "OUTER_COMPLETE.json", {
            "status": "complete",
            "outer_id": outer.outer_id,
            "outer_seed": outer.seed,
            "individual_artifacts_preserved": True,
            **fit_row,
        })

        # Checkpoints for long repeated runs.
        pd.DataFrame(fit_rows).to_csv(run / "tables" / "q_weight_fit_by_outer_split.csv", index=False, encoding="utf-8-sig")
        pd.concat(weight_frames, ignore_index=True).to_csv(run / "tables" / "learned_q_weights_by_outer_split.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(inductive_rows).to_csv(run / "tables" / "generalization_metrics_by_outer_split.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(square_rows).to_csv(run / "tables" / "square_test_metrics_by_outer_split.csv", index=False, encoding="utf-8-sig")

    fit_df = pd.DataFrame(fit_rows)
    weights_all = pd.concat(weight_frames, ignore_index=True)
    traj_df = pd.concat(traj_frames, ignore_index=True)
    inductive_df = pd.DataFrame(inductive_rows)
    square_df = pd.DataFrame(square_rows)
    inductive_qdf = pd.concat(inductive_qframes, ignore_index=True)
    inductive_rdf = pd.concat(inductive_rframes, ignore_index=True)
    square_qdf = pd.concat(square_qframes, ignore_index=True)
    square_rdf = pd.concat(square_rframes, ignore_index=True)
    square_domain_df = pd.DataFrame(square_domain_rows)

    fit_df.to_csv(run / "tables" / "q_weight_fit_by_outer_split.csv", index=False, encoding="utf-8-sig")
    weights_all.to_csv(run / "tables" / "learned_q_weights_by_outer_split.csv", index=False, encoding="utf-8-sig")
    traj_df.to_csv(run / "tables" / "q_weight_optimization_trajectories.csv", index=False, encoding="utf-8-sig")
    inductive_df.to_csv(run / "tables" / "generalization_metrics_by_outer_split.csv", index=False, encoding="utf-8-sig")
    inductive_qdf.to_csv(run / "tables" / "generalization_per_query.csv", index=False, encoding="utf-8-sig")
    inductive_rdf.to_csv(run / "tables" / "generalization_rankings.csv", index=False, encoding="utf-8-sig")
    square_df.to_csv(run / "tables" / "square_test_metrics_by_outer_split.csv", index=False, encoding="utf-8-sig")
    square_qdf.to_csv(run / "tables" / "square_test_dataset_metrics.csv", index=False, encoding="utf-8-sig")
    square_rdf.to_csv(run / "tables" / "square_test_rankings.csv", index=False, encoding="utf-8-sig")
    square_domain_df.to_csv(run / "tables" / "square_test_domain_metrics.csv", index=False, encoding="utf-8-sig")

    inductive_summary = _aggregate_metric_frame(inductive_df)
    square_summary = _aggregate_metric_frame(square_df)
    inductive_summary.to_csv(run / "tables" / "generalization_metrics_summary.csv", index=False, encoding="utf-8-sig")
    square_summary.to_csv(run / "tables" / "square_test_metrics_summary.csv", index=False, encoding="utf-8-sig")

    weight_summary = weights_all.groupby(["q", "clique_size"], sort=True).agg(
        identifiable_outer_count=("train_identifiable", "sum"),
        mean_train_schema_weight=("train_schema_weight", "mean"),
        std_train_schema_weight=("train_schema_weight", "std"),
        mean_reference_weight=("reference_weight", "mean"),
        std_reference_weight=("reference_weight", "std"),
        median_reference_weight=("reference_weight", "median"),
        mean_train_pair_effective_weight=("mean_train_pair_effective_weight", "mean"),
        mean_test_train_pair_effective_weight=("mean_test_train_pair_effective_weight", "mean"),
    ).reset_index()
    weight_summary["std_reference_weight"] = weight_summary["std_reference_weight"].fillna(0.0)
    weight_summary["std_train_schema_weight"] = weight_summary["std_train_schema_weight"].fillna(0.0)
    weight_summary.to_csv(run / "tables" / "learned_q_weights_summary.csv", index=False, encoding="utf-8-sig")

    summary = {
        "n_outer_evaluations": int(len(fit_df)),
        "mean_effective_num_layers": float(fit_df["effective_num_layers"].mean()),
        "mean_max_weight": float(fit_df["max_weight"].mean()),
        "most_frequent_max_weight_q": int(fit_df["max_weight_q"].mode().iloc[0]),
        "converged_fraction": float(fit_df["converged"].mean()),
        "regularization": "none",
    }
    write_json(run / "learned_q_weight_summary.json", summary)

    if bool(learning.get("generate_plots", True)):
        from .real_learning_nested_plotting import (
            plot_nested_confusion,
            plot_nested_metric_summary,
            plot_outer_metric_distributions,
            plot_outer_test_coverage,
        )
        from .real_q_weight_learning_plotting import (
            plot_effective_vs_reference_weights,
            plot_learned_q_weights,
            plot_outer_weight_heatmap,
            plot_q_weight_loss,
            plot_q_weight_trajectories,
        )
        plot_nested_metric_summary(inductive_summary, run / "plots" / "generalization_mean_std", title="Learned q weights: outer-test → outer-train generalization")
        plot_outer_metric_distributions(inductive_df, run / "plots" / "generalization_distributions")
        plot_outer_test_coverage(coverage, run / "plots" / "outer_test_coverage")
        learned_queries = inductive_qdf[inductive_qdf["method"] == LEARNED_METHOD].copy()
        plot_nested_confusion(learned_queries, unique_domains, run / "plots" / "HodgeLearnedQWeights_aggregate_outer_confusion")
        plot_outer_weight_heatmap(weights_all, run / "plots" / "learned_q_weight_outer_heatmap")

        if len(outer_splits) == 1:
            one = weights_all.copy()
            plot_learned_q_weights(one, run / "plots" / "learned_q_weights")
            plot_effective_vs_reference_weights(one, run / "plots" / "reference_vs_effective_q_weights")
            plot_q_weight_trajectories(traj_df, run / "plots" / "q_weight_optimization_trajectory")
            plot_q_weight_loss(traj_df, run / "plots" / "q_weight_nca_loss")

            outer = outer_splits[0]
            learned_D = np.load(run / "outer" / outer.outer_id / f"distance_matrix_{LEARNED_METHOD}_full.npy")
            matrices = {LEARNED_METHOD: learned_D, FROZEN_METHOD: frozen_D, **baseline_matrices}
            plot_options = dict(exp.get("plotting", {})); plot_options.update(payload.get("plotting", {}))
            for method, D in matrices.items():
                _, _, _, clustering = _square_metrics(
                    D, outer.test_idx, labels, domains, method, linkage_method,
                    mst_permutations, eval_seed,
                )
                _plot_square_method(
                    run, "test_holdout", method, D, outer.test_idx, labels, domains,
                    clustering, plot_options, eval_seed, skip_existing=not force,
                )

    pd.DataFrame(runtime_rows).to_csv(run / "tables" / "runtime_rows.csv", index=False, encoding="utf-8-sig")
    write_json(run / "run_manifest.json", {
        "architecture": "supervised free q-layer weight learning over frozen independent-q Hodge VNext core",
        "method_core_changed": False,
        "cache_schema": CACHE_SCHEMA,
        "benchmark_id": benchmark_id,
        "num_networks": len(df),
        "domains": unique_domains,
        "fixed_scientific_settings": {
            "max_clique_size": MAX_CLIQUE_SIZE,
            "computed_q": "0..9",
            "max_hodge_order": "all",
            "min_layer_q": 0,
            "normalization": "per_simplex",
        },
        "learned_parameters": {
            "name": "beta_q / W_q for q=0..9",
            "parameterization": "W=softmax(beta), masked and renormalized to each pair-specific q*",
            "optimizer": "Adam",
            "objective": "NCA/soft-nearest-neighbour loss",
            "initialization": "beta_q=0 (uniform q weights)",
            "regularization": "none",
            "optimizer_settings": opt_kwargs,
        },
        "outer_evaluation": {
            "strategy": strategy,
            "repeats": repeats,
            "folds": folds,
            "seed": outer_seed,
            "num_outer_evaluations": len(outer_splits),
            "individual_split_artifacts_preserved": True,
            "split_inventory_table": "tables/outer_split_inventory.csv",
            "split_overlap_table": "tables/outer_split_overlap.csv",
        },
        "methods_reported": [LEARNED_METHOD, FROZEN_METHOD, *baseline_methods],
        "execution": {
            "profile_workers": profile_workers,
            "gpu_backend": "none; Hodge/SLQ remains NumPy/SciPy CPU",
        },
        "scientific_boundaries": [
            "K11+ Hodge layers are not computed; K10 is the computational ceiling, not a learned truncation",
            "outer-test labels are never used to fit q weights",
            "no q-weight regularization is used in this first-stage experiment",
            "pair-specific q* masking/renormalization preserves the original pair-adaptive q schema",
            "1-NN Macro-F1/AUROC/mAP and other discrete/ranking metrics are evaluation metrics, not gradient objectives",
            "domain labels are external annotations, not unique structural ground truth",
            "if weights collapse or vary strongly across resamples, regularization is a future hypothesis, not assumed here",
            "repeated holdout evaluations overlap and are correlated resamples; aggregate SD/quantiles describe split-to-split variability rather than iid uncertainty",
        ],
        "effects": {
            "method_math_definition": "new supervised aggregation extension only; Hodge core unchanged",
            "cache_identity": "existing staged Hodge cache unchanged; q0..q9 pair-layer energies are a new run artifact",
            "experiment_statistics": "new supervised q-layer metric-learning protocol",
            "random_graph_seed": "not applicable / unchanged",
            "baseline_semantics": "unchanged",
            "existing_result_comparability": "report separately from original unsupervised Hodge",
        },
    })
    write_json(complete, {"status": "complete", "run_dir": str(run), **summary})
    progress(f"[COMPLETE] {run}")
    return run
