from __future__ import annotations

"""Repeated 5-fold diagnosis: train-only q-scale normalization × 3 weighting strategies.

The independent-q Hodge core is unchanged. Compare on identical repeated
stratified folds:

    Raw    × {Uniform, Single-alpha, Free-q}
    Scaled × {Uniform, Single-alpha, Free-q}

where Scaled uses one fold-specific RMS scale per q estimated from train-train
pairs only.  Final statistics are computed at the *repeat* level: the five
out-of-fold test partitions are merged so each of the 48 networks contributes
exactly one OOF prediction per repeat.  The 50 folds are retained as diagnostic
artifacts but are not treated as 50 independent samples.
"""

from pathlib import Path
from typing import Any, Callable, Sequence
import json
import time

import numpy as np
import pandas as pd

from hodge_vnext.cache import CACHE_SCHEMA
from hodge_vnext.graph_io import GraphRecord, create_real_record
from hodge_vnext.utils import safe_token, stable_hash, write_json

from .config import method_from_payload, resolve_path
from .methods import MethodManager, PRIMARY_METHOD
from .q_weight_learning import (
    average_effective_pair_weights,
    distance_matrix_from_beta,
    fit_q_weights_adam,
    softmax_weights,
    weight_diagnostics,
)
from .q_weight_scale_cv import (
    LayerScaleResult,
    apply_layer_scales_to_squared_distances,
    estimate_train_only_layer_scales,
    fit_single_alpha_train_only,
    layer_scale_table,
    single_alpha_beta,
)
from .real_domains import _prepare_manifest
from .real_learning import (
    _macro_f1,
    _rectangular_auroc,
    cross_partition_metrics,
)
from .real_learning_nested import (
    _aggregate_metric_frame,
    _compute_or_load_baseline_matrix,
    _coverage_table,
    _outer_split_membership_table,
    make_repeated_outer_splits,
)
from .real_q_weight_learning import (
    MAX_CLIQUE_SIZE,
    NUM_Q,
    _load_or_compute_layer_energies,
    _profiles_metadata,
    _validate_scientific_fixed_config,
)


METHOD_SPECS = [
    ("raw", "uniform", "HodgeRawUniform"),
    ("raw", "single_alpha", "HodgeRawSingleAlpha"),
    ("raw", "free_q", "HodgeRawFreeQ"),
    ("scaled", "uniform", "HodgeScaledUniform"),
    ("scaled", "single_alpha", "HodgeScaledSingleAlpha"),
    ("scaled", "free_q", "HodgeScaledFreeQ"),
]
METHOD_META = {name: (scaling, weighting) for scaling, weighting, name in METHOD_SPECS}
DISPLAY_NAMES = {
    "HodgeRawUniform": "Raw + Uniform",
    "HodgeRawSingleAlpha": "Raw + Single-α",
    "HodgeRawFreeQ": "Raw + Free-q",
    "HodgeScaledUniform": "Scaled + Uniform",
    "HodgeScaledSingleAlpha": "Scaled + Single-α",
    "HodgeScaledFreeQ": "Scaled + Free-q",
    "DMeasure": "DMeasure",
    "NetLSD": "NetLSD",
    "Portrait": "Portrait",
    "Count": "Count",
}


def _progress(msg: str) -> None:
    print(msg, flush=True)


def _optimizer_kwargs(learning: dict[str, Any]) -> dict[str, Any]:
    opt = dict(learning.get("free_q_optimizer", learning.get("optimizer", {})))
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


def _alpha_kwargs(learning: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(learning.get("single_alpha", {}))
    return {
        "coarse_min": float(cfg.get("coarse_min", -10.0)),
        "coarse_max": float(cfg.get("coarse_max", 10.0)),
        "coarse_step": float(cfg.get("coarse_step", 1.0)),
        "fine_half_width": float(cfg.get("fine_half_width", 1.0)),
        "fine_step": float(cfg.get("fine_step", 0.05)),
    }


def _run_stem(n: int, benchmark_id: str, payload: dict[str, Any], config_path: Path) -> str:
    cfg = method_from_payload(payload, config_path)
    learning = dict(payload.get("learning", {}))
    token = stable_hash({
        "benchmark_id": benchmark_id,
        "n": int(n),
        "method": {
            "max_clique_size": cfg.max_clique_size,
            "max_hodge_order": cfg.max_hodge_order,
            "min_layer_q": cfg.min_layer_q,
            "normalization": cfg.normalization,
            "solver": cfg.solver,
            "slq": cfg.slq.__dict__,
            "time_grid": [cfg.time_min, cfg.time_max, cfg.time_points],
        },
        "protocol": "train_only_q_rms_scale__repeated_stratified_5fold__raw_scaled_x_uniform_singlealpha_freeq",
        "outer_cv": learning.get("outer_cv", {}),
        "scale_normalization": learning.get("scale_normalization", {}),
        "single_alpha": _alpha_kwargs(learning),
        "free_q_optimizer": _optimizer_kwargs(learning),
    }, length=12)
    return f"real{n}_{safe_token(benchmark_id)}_qscale-rskf_{token}"


def _load_precomputed_tensor(path: Path, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    obj = np.load(path)
    # Backward-compatible field name from the prior q-weight experiment.
    key = "squared_layers" if "squared_layers" in obj.files else "energies"
    C = np.asarray(obj[key], dtype=float)
    q_star = np.asarray(obj["q_star"], dtype=np.int32)
    profile_qmax = np.asarray(obj["profile_qmax"], dtype=np.int32)
    has_layer = np.asarray(obj["has_layer"], dtype=np.uint8)
    if C.shape != (n, n, NUM_Q):
        raise ValueError(f"precomputed layer tensor shape {C.shape} != {(n, n, NUM_Q)}")
    if q_star.shape != (n, n) or profile_qmax.shape != (n,) or has_layer.shape != (n, NUM_Q):
        raise ValueError("precomputed layer tensor metadata has incompatible shape")
    return C, q_star, profile_qmax, has_layer


def _save_layer_tensor(path: Path, C: np.ndarray, q_star: np.ndarray, profile_qmax: np.ndarray, has_layer: np.ndarray) -> None:
    np.savez_compressed(
        path,
        squared_layers=np.asarray(C, dtype=float),
        q_star=np.asarray(q_star, dtype=np.int32),
        profile_qmax=np.asarray(profile_qmax, dtype=np.int32),
        has_layer=np.asarray(has_layer, dtype=np.uint8),
    )


def _repeat_split_audit(membership: pd.DataFrame, n_networks: int, repeats: int, folds: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for repeat in range(1, repeats + 1):
        sub = membership[membership["repeat"] == repeat]
        counts = (
            sub.assign(is_test=(sub["partition"] == "test").astype(int))
            .groupby("index", sort=False)["is_test"].sum()
        )
        ok = bool(len(counts) == n_networks and (counts == 1).all())
        rows.append({
            "repeat": repeat,
            "folds": folds,
            "n_networks": n_networks,
            "min_test_appearances_within_repeat": int(counts.min()) if len(counts) else 0,
            "max_test_appearances_within_repeat": int(counts.max()) if len(counts) else 0,
            "all_networks_tested_exactly_once": ok,
        })
        if not ok:
            raise RuntimeError(f"repeat={repeat} is not a complete out-of-fold partition")
    return pd.DataFrame(rows)


def _weight_table_for_fold(outer, fit, q_star: np.ndarray, scaling: str) -> pd.DataFrame:
    train_eff = average_effective_pair_weights(fit.beta, q_star, outer.train_idx)
    test_train_eff = average_effective_pair_weights(fit.beta, q_star, outer.test_idx, outer.train_idx)
    train_sub = q_star[np.ix_(outer.train_idx, outer.train_idx)]
    train_qmax = int(np.max(train_sub)) if train_sub.size else -1
    schema = np.full(NUM_Q, np.nan, dtype=float)
    if train_qmax >= 0:
        schema[: train_qmax + 1] = softmax_weights(fit.beta[: train_qmax + 1])
    rows = []
    for q in range(NUM_Q):
        rows.append({
            "outer_id": outer.outer_id,
            "repeat": outer.repeat,
            "fold": outer.fold,
            "scaling": scaling,
            "q": q,
            "clique_size": q + 1,
            "beta": float(fit.beta[q]),
            "train_identifiable": bool(q <= train_qmax),
            "train_schema_weight": float(schema[q]) if np.isfinite(schema[q]) else np.nan,
            "reference_weight": float(fit.reference_weights[q]),
            "mean_train_pair_effective_weight": float(train_eff[q]),
            "mean_test_train_pair_effective_weight": float(test_train_eff[q]),
        })
    return pd.DataFrame(rows)


def _repeat_oof_metrics(qdf: pd.DataFrame, rdf: pd.DataFrame, n_expected: int) -> dict[str, float]:
    if len(qdf) != n_expected or qdf["query"].nunique() != n_expected:
        raise RuntimeError(
            f"OOF repeat must contain exactly {n_expected} unique queries; got rows={len(qdf)} unique={qdf['query'].nunique()}"
        )
    if bool(qdf["query"].duplicated().any()):
        raise RuntimeError("OOF repeat contains duplicate query predictions")
    pair_scores = [(float(d), int(r)) for d, r in zip(rdf["distance"], rdf["relevant"])]
    true = qdf["domain"].astype(str).tolist()
    pred = qdf["nearest_neighbor_domain"].astype(str).tolist()
    within = float(qdf["mean_same_domain_distance"].mean())
    between = float(qdf["mean_other_domain_distance"].mean())
    out: dict[str, float] = {
        "auroc_same_domain": float(_rectangular_auroc(pair_scores)),
        "nn1_accuracy": float(qdf["nearest_neighbor_correct"].mean()),
        "nn1_macro_f1": float(_macro_f1(true, pred)),
        "mAP": float(qdf["average_precision"].mean()),
        "ndcg_at_7": float(qdf["ndcg_at_7"].mean()),
        "within_domain_mean_distance": within,
        "between_domain_mean_distance": between,
        "between_within_ratio": float(between / within) if within > 0 else float("inf"),
    }
    for k in (1, 3, 5, 7):
        out[f"precision_at_{k}"] = float(qdf[f"precision_at_{k}"].mean())
        out[f"recall_at_{k}"] = float(qdf[f"recall_at_{k}"].mean())
    return out


def _free_q_cosine_tables(weights: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for scaling, sub in weights.groupby("scaling", sort=False):
        pivot = sub.pivot(index="outer_id", columns="q", values="reference_weight").sort_index()
        W = pivot.to_numpy(dtype=float)
        norms = np.linalg.norm(W, axis=1)
        vals: list[float] = []
        ids = pivot.index.tolist()
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                denom = float(norms[i] * norms[j])
                cosine = float(np.dot(W[i], W[j]) / denom) if denom > 0 else np.nan
                pair_rows.append({"scaling": scaling, "outer_id_a": ids[i], "outer_id_b": ids[j], "cosine_similarity": cosine})
                if np.isfinite(cosine):
                    vals.append(cosine)
        a = np.asarray(vals, dtype=float)
        summary_rows.append({
            "scaling": scaling,
            "n_fold_pairs": int(len(a)),
            "mean_cosine_similarity": float(np.mean(a)) if len(a) else np.nan,
            "std_cosine_similarity": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
            "median_cosine_similarity": float(np.median(a)) if len(a) else np.nan,
            "min_cosine_similarity": float(np.min(a)) if len(a) else np.nan,
            "max_cosine_similarity": float(np.max(a)) if len(a) else np.nan,
        })
    return pd.DataFrame(pair_rows), pd.DataFrame(summary_rows)


def _alpha_summary(alpha_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scaling, sub in alpha_df.groupby("scaling", sort=False):
        a = sub["alpha"].to_numpy(dtype=float)
        rows.append({
            "scaling": scaling,
            "n_folds": int(len(a)),
            "mean_alpha": float(np.mean(a)),
            "std_alpha": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
            "median_alpha": float(np.median(a)),
            "negative_fraction": float(np.mean(a < -1e-12)),
            "near_zero_fraction": float(np.mean(np.abs(a) <= 1e-12)),
            "positive_fraction": float(np.mean(a > 1e-12)),
            "boundary_hit_fraction": float(sub["boundary_hit"].mean()),
        })
    return pd.DataFrame(rows)


def _paper_main_table(summary: pd.DataFrame) -> pd.DataFrame:
    wanted = [
        "HodgeRawUniform", "HodgeRawSingleAlpha", "HodgeRawFreeQ",
        "HodgeScaledUniform", "HodgeScaledSingleAlpha", "HodgeScaledFreeQ",
        "DMeasure", "NetLSD", "Portrait", "Count",
    ]
    rows = []
    by = {str(r.method): r for r in summary.itertuples(index=False)}
    for method in wanted:
        if method not in by:
            continue
        r = by[method]
        def fmt(mean_name: str, std_name: str) -> str:
            return f"{float(getattr(r, mean_name)):.3f} ± {float(getattr(r, std_name)):.3f}"
        rows.append({
            "Method": DISPLAY_NAMES.get(method, method),
            "AUROC": fmt("mean_auroc_same_domain", "std_auroc_same_domain"),
            "1-NN": fmt("mean_nn1_accuracy", "std_nn1_accuracy"),
            "Macro-F1": fmt("mean_nn1_macro_f1", "std_nn1_macro_f1"),
            "mAP": fmt("mean_mAP", "std_mAP"),
            "NDCG@7": fmt("mean_ndcg_at_7", "std_ndcg_at_7"),
        })
    return pd.DataFrame(rows)


def run_real_q_weight_scale_cv(
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
    scale_cfg = dict(learning.get("scale_normalization", {}))
    eps = float(scale_cfg.get("eps", 1e-8))
    opt_kwargs = _optimizer_kwargs(learning)
    alpha_kwargs = _alpha_kwargs(learning)
    profile_workers = max(1, int(execution.get("profile_workers", 1)))

    df = _prepare_manifest(payload, config_path)
    labels = df["label"].astype(str).tolist()
    domains = df["domain"].astype(str).tolist()
    unique_domains = list(dict.fromkeys(domains))
    if len(df) != 48 or len(unique_domains) != 5:
        progress(f"[WARN] formal protocol was designed for 48 networks / 5 domains; got {len(df)} / {len(unique_domains)}")

    strategy = str(outer_cfg.get("strategy", "repeated_stratified_kfold"))
    repeats = int(outer_cfg.get("repeats", 10))
    folds = int(outer_cfg.get("folds", 5))
    outer_seed = int(outer_cfg.get("seed", 42))
    min_test = int(outer_cfg.get("require_min_test_per_domain", 1))
    if strategy not in {"repeated_stratified_kfold", "stratified_kfold", "kfold"}:
        raise ValueError("this diagnosis requires repeated_stratified_kfold")
    outer_splits = make_repeated_outer_splits(
        domains,
        strategy="repeated_stratified_kfold",
        repeats=repeats,
        seed=outer_seed,
        outer_folds=folds,
        require_min_test_per_domain=min_test,
    )
    membership = _outer_split_membership_table(df, outer_splits)
    coverage = _coverage_table(membership)
    repeat_audit = _repeat_split_audit(membership, len(df), repeats, folds)
    if not bool((coverage["test_appearances"] == repeats).all()):
        raise RuntimeError("repeated K-fold coverage audit failed: every network must be test exactly once per repeat")

    benchmark_id = str(exp.get("benchmark_id", "real48-qscale-rskf10x5"))
    results_root = resolve_path(learning.get("results_root", "results/validation/real_q_weight_scale_cv"), config_path)
    run = results_root / _run_stem(len(df), benchmark_id, payload, config_path)
    complete = run / "RUN_COMPLETE.json"
    if complete.exists() and not force:
        progress(f"[SKIP] completed q-scale CV run exists: {run}")
        return run
    run.mkdir(parents=True, exist_ok=True)
    for d in ("arrays", "tables", "plots", "outer", "repeat"):
        (run / d).mkdir(parents=True, exist_ok=True)
    if force:
        complete.unlink(missing_ok=True)

    membership.to_csv(run / "tables" / "outer_split_membership.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(run / "tables" / "outer_test_coverage.csv", index=False, encoding="utf-8-sig")
    repeat_audit.to_csv(run / "tables" / "repeat_split_audit.csv", index=False, encoding="utf-8-sig")

    effective = json.loads(json.dumps(payload))
    effective["cache_schema"] = CACHE_SCHEMA
    effective["diagnostic_contract"] = {
        "outer_protocol": f"{repeats} repeats × {folds}-fold repeated stratified CV",
        "final_statistics_unit": "repeat-level OOF metrics; 50 folds are diagnostics, not iid samples",
        "distance_scaling": "Raw and train-only per-q RMS-scaled",
        "scale_estimation": "train-train active pairs only",
        "weighting_strategies": ["Uniform", "Single-alpha exp(alpha*q/9)", "Free-q softmax"],
        "regularization": "none",
        "computed_q": "0..9 / maximal K1..K10 computational ceiling",
        "Hodge_Q": "all",
        "qmin": 0,
        "profile_normalization": "per_simplex",
    }
    write_json(run / "effective_config.json", effective)

    records: list[GraphRecord] = []
    for _, row in df.iterrows():
        records.append(create_real_record(row.path, cfg.cache.root, str(row.label), row.nodes_path or None))

    runtime_rows: list[dict[str, Any]] = []
    precomputed = learning.get("precomputed_layer_tensor")
    profiles = None
    if precomputed:
        p = resolve_path(precomputed, config_path)
        progress(f"[LAYER TENSOR] loading precomputed squared layer distances: {p}")
        C, q_star, profile_qmax, has_layer = _load_precomputed_tensor(p, len(df))
    else:
        profiles, C, q_star, profile_qmax, has_layer = _load_or_compute_layer_energies(
            records, cfg, run, runtime_rows,
            recompute=recompute_hodge,
            profile_workers=profile_workers,
            progress=progress,
        )
    _save_layer_tensor(run / "arrays" / "hodge_layer_squared_distances_q0_q9.npz", C, q_star, profile_qmax, has_layer)

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

    fold_metric_rows: list[dict[str, Any]] = []
    all_query_frames: list[pd.DataFrame] = []
    all_rank_frames: list[pd.DataFrame] = []
    scale_frames: list[pd.DataFrame] = []
    alpha_rows: list[dict[str, Any]] = []
    alpha_trace_frames: list[pd.DataFrame] = []
    free_weight_frames: list[pd.DataFrame] = []
    free_fit_rows: list[dict[str, Any]] = []
    free_traj_frames: list[pd.DataFrame] = []

    for oi, outer in enumerate(outer_splits, start=1):
        progress(f"[FOLD] {oi}/{len(outer_splits)} {outer.outer_id}: train={len(outer.train_idx)} test={len(outer.test_idx)}")
        odir = run / "outer" / outer.outer_id
        odir.mkdir(parents=True, exist_ok=True)
        test_set = set(int(x) for x in outer.test_idx)
        pd.DataFrame({
            "index": np.arange(len(df), dtype=int),
            "label": labels,
            "domain": domains,
            "partition": ["test" if i in test_set else "train" for i in range(len(df))],
        }).to_csv(odir / "split.csv", index=False, encoding="utf-8-sig")

        scale = estimate_train_only_layer_scales(C, q_star, outer.train_idx, eps=eps, require_identifiable=True)
        sdf = layer_scale_table(scale)
        sdf.insert(0, "outer_id", outer.outer_id)
        sdf.insert(1, "repeat", outer.repeat)
        sdf.insert(2, "fold", outer.fold)
        scale_frames.append(sdf)
        sdf.to_csv(odir / "train_only_q_scales.csv", index=False, encoding="utf-8-sig")
        C_scaled = apply_layer_scales_to_squared_distances(C, scale)

        method_matrices: dict[str, np.ndarray] = {}
        for scaling, Cmode in (("raw", C), ("scaled", C_scaled)):
            # Uniform: exact pair-adaptive uniform q weights.
            method_name = "HodgeRawUniform" if scaling == "raw" else "HodgeScaledUniform"
            method_matrices[method_name] = distance_matrix_from_beta(np.zeros(NUM_Q), Cmode, q_star)

            # Single-alpha: deterministic train-only coarse+fine NCA search.
            afit = fit_single_alpha_train_only(Cmode, q_star, outer.train_idx, domains, **alpha_kwargs)
            method_name = "HodgeRawSingleAlpha" if scaling == "raw" else "HodgeScaledSingleAlpha"
            method_matrices[method_name] = distance_matrix_from_beta(afit.beta, Cmode, q_star)
            arow = {
                "outer_id": outer.outer_id,
                "repeat": outer.repeat,
                "fold": outer.fold,
                "scaling": scaling,
                "alpha": afit.alpha,
                "best_loss": afit.best_loss,
                "temperature": afit.temperature,
                "boundary_hit": afit.boundary_hit,
            }
            alpha_rows.append(arow)
            write_json(odir / f"{scaling}_single_alpha_summary.json", arow)
            atr = afit.trajectory.copy()
            atr.insert(0, "outer_id", outer.outer_id)
            atr.insert(1, "repeat", outer.repeat)
            atr.insert(2, "fold", outer.fold)
            atr.insert(3, "scaling", scaling)
            alpha_trace_frames.append(atr)
            atr.to_csv(odir / f"{scaling}_single_alpha_search.csv", index=False, encoding="utf-8-sig")

            # Free-q: same unregularized Adam model as the current experiment.
            ffit = fit_q_weights_adam(Cmode, q_star, outer.train_idx, domains, **opt_kwargs)
            method_name = "HodgeRawFreeQ" if scaling == "raw" else "HodgeScaledFreeQ"
            method_matrices[method_name] = distance_matrix_from_beta(ffit.beta, Cmode, q_star)
            wdf = _weight_table_for_fold(outer, ffit, q_star, scaling)
            free_weight_frames.append(wdf)
            wdf.to_csv(odir / f"{scaling}_free_q_weights.csv", index=False, encoding="utf-8-sig")
            diag = weight_diagnostics(ffit.reference_weights)
            frow = {
                "outer_id": outer.outer_id,
                "repeat": outer.repeat,
                "fold": outer.fold,
                "scaling": scaling,
                "train_loss": ffit.best_loss,
                "final_grad_norm": ffit.final_grad_norm,
                "epochs": ffit.epochs,
                "converged": ffit.converged,
                "stop_reason": ffit.stop_reason,
                "temperature": ffit.temperature,
                **diag,
            }
            free_fit_rows.append(frow)
            write_json(odir / f"{scaling}_free_q_summary.json", frow)
            tr = ffit.trajectory.copy()
            tr.insert(0, "outer_id", outer.outer_id)
            tr.insert(1, "repeat", outer.repeat)
            tr.insert(2, "fold", outer.fold)
            tr.insert(3, "scaling", scaling)
            free_traj_frames.append(tr)
            tr.to_csv(odir / f"{scaling}_free_q_trajectory.csv", index=False, encoding="utf-8-sig")

        # Hodge matrices are fold-specific; baselines are fixed but evaluated on the same folds.
        for method, D in method_matrices.items():
            np.save(odir / f"distance_matrix_{method}_full.npy", D)
        matrices = {**method_matrices, **baseline_matrices}

        fold_rows: list[dict[str, Any]] = []
        fold_qframes: list[pd.DataFrame] = []
        fold_rframes: list[pd.DataFrame] = []
        for method, D in matrices.items():
            metrics, qdf, rdf = cross_partition_metrics(D, outer.test_idx, outer.train_idx, domains, labels)
            scaling, weighting = METHOD_META.get(method, ("baseline", "baseline"))
            meta = {
                "outer_id": outer.outer_id,
                "repeat": outer.repeat,
                "fold": outer.fold,
                "outer_seed": outer.seed,
                "train_size": len(outer.train_idx),
                "test_size": len(outer.test_idx),
                "method": method,
                "display_method": DISPLAY_NAMES.get(method, method),
                "distance_scaling": scaling,
                "weighting": weighting,
            }
            row = {**meta, **metrics}
            fold_metric_rows.append(row)
            fold_rows.append(row)
            for k, v in reversed(list(meta.items())):
                qdf.insert(0, k, v)
                rdf.insert(0, k, v)
            all_query_frames.append(qdf.copy())
            all_rank_frames.append(rdf.copy())
            fold_qframes.append(qdf)
            fold_rframes.append(rdf)

        pd.DataFrame(fold_rows).to_csv(odir / "fold_metrics.csv", index=False, encoding="utf-8-sig")
        pd.concat(fold_qframes, ignore_index=True).to_csv(odir / "generalization_per_query.csv", index=False, encoding="utf-8-sig")
        pd.concat(fold_rframes, ignore_index=True).to_csv(odir / "generalization_rankings.csv", index=False, encoding="utf-8-sig")
        write_json(odir / "OUTER_COMPLETE.json", {
            "status": "complete", "outer_id": outer.outer_id, "repeat": outer.repeat, "fold": outer.fold,
            "train_size": len(outer.train_idx), "test_size": len(outer.test_idx),
            "train_only_scales": True, "individual_artifacts_preserved": True,
        })

        # Restart-safe checkpoints.
        pd.DataFrame(fold_metric_rows).to_csv(run / "tables" / "fold_metrics.csv", index=False, encoding="utf-8-sig")
        pd.concat(scale_frames, ignore_index=True).to_csv(run / "tables" / "train_only_q_scales_by_fold.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(alpha_rows).to_csv(run / "tables" / "single_alpha_by_fold.csv", index=False, encoding="utf-8-sig")
        pd.concat(free_weight_frames, ignore_index=True).to_csv(run / "tables" / "free_q_weights_by_fold.csv", index=False, encoding="utf-8-sig")

    fold_metrics = pd.DataFrame(fold_metric_rows)
    query_all = pd.concat(all_query_frames, ignore_index=True)
    rank_all = pd.concat(all_rank_frames, ignore_index=True)
    scales_all = pd.concat(scale_frames, ignore_index=True)
    alpha_df = pd.DataFrame(alpha_rows)
    alpha_trace = pd.concat(alpha_trace_frames, ignore_index=True)
    free_weights = pd.concat(free_weight_frames, ignore_index=True)
    free_fit = pd.DataFrame(free_fit_rows)
    free_traj = pd.concat(free_traj_frames, ignore_index=True)

    fold_metrics.to_csv(run / "tables" / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    query_all.to_csv(run / "tables" / "fold_generalization_per_query.csv", index=False, encoding="utf-8-sig")
    rank_all.to_csv(run / "tables" / "fold_generalization_rankings.csv", index=False, encoding="utf-8-sig")
    scales_all.to_csv(run / "tables" / "train_only_q_scales_by_fold.csv", index=False, encoding="utf-8-sig")
    alpha_df.to_csv(run / "tables" / "single_alpha_by_fold.csv", index=False, encoding="utf-8-sig")
    alpha_trace.to_csv(run / "tables" / "single_alpha_search_trace.csv", index=False, encoding="utf-8-sig")
    free_weights.to_csv(run / "tables" / "free_q_weights_by_fold.csv", index=False, encoding="utf-8-sig")
    free_fit.to_csv(run / "tables" / "free_q_fit_by_fold.csv", index=False, encoding="utf-8-sig")
    free_traj.to_csv(run / "tables" / "free_q_optimization_trajectories.csv", index=False, encoding="utf-8-sig")

    # Merge five folds inside each repeat before computing final metrics.
    repeat_metric_rows: list[dict[str, Any]] = []
    repeat_query_frames: list[pd.DataFrame] = []
    repeat_rank_frames: list[pd.DataFrame] = []
    method_order = [name for _, _, name in METHOD_SPECS] + baseline_methods
    for repeat in range(1, repeats + 1):
        rdir = run / "repeat" / f"r{repeat:02d}"
        rdir.mkdir(parents=True, exist_ok=True)
        r_qframes: list[pd.DataFrame] = []
        r_rframes: list[pd.DataFrame] = []
        r_rows: list[dict[str, Any]] = []
        for method in method_order:
            qsub = query_all[(query_all["repeat"] == repeat) & (query_all["method"] == method)].copy()
            rsub = rank_all[(rank_all["repeat"] == repeat) & (rank_all["method"] == method)].copy()
            metrics = _repeat_oof_metrics(qsub, rsub, len(df))
            scaling, weighting = METHOD_META.get(method, ("baseline", "baseline"))
            row = {
                "repeat": repeat,
                "method": method,
                "display_method": DISPLAY_NAMES.get(method, method),
                "distance_scaling": scaling,
                "weighting": weighting,
                "n_oof_queries": len(qsub),
                **metrics,
            }
            repeat_metric_rows.append(row)
            r_rows.append(row)
            repeat_query_frames.append(qsub)
            repeat_rank_frames.append(rsub)
            r_qframes.append(qsub)
            r_rframes.append(rsub)
        pd.DataFrame(r_rows).to_csv(rdir / "oof_metrics.csv", index=False, encoding="utf-8-sig")
        pd.concat(r_qframes, ignore_index=True).to_csv(rdir / "oof_per_query.csv", index=False, encoding="utf-8-sig")
        pd.concat(r_rframes, ignore_index=True).to_csv(rdir / "oof_rankings.csv", index=False, encoding="utf-8-sig")
        write_json(rdir / "REPEAT_COMPLETE.json", {
            "status": "complete", "repeat": repeat, "folds_merged": folds,
            "n_oof_queries_per_method": len(df),
        })

    repeat_metrics = pd.DataFrame(repeat_metric_rows)
    repeat_metrics.to_csv(run / "tables" / "repeat_oof_metrics.csv", index=False, encoding="utf-8-sig")
    repeat_queries = pd.concat(repeat_query_frames, ignore_index=True)
    repeat_rankings = pd.concat(repeat_rank_frames, ignore_index=True)
    repeat_queries.to_csv(run / "tables" / "repeat_oof_per_query.csv", index=False, encoding="utf-8-sig")
    repeat_rankings.to_csv(run / "tables" / "repeat_oof_rankings.csv", index=False, encoding="utf-8-sig")

    repeat_summary = _aggregate_metric_frame(repeat_metrics, metadata_cols=("n_oof_queries",))
    repeat_summary.to_csv(run / "tables" / "repeat_oof_metrics_summary.csv", index=False, encoding="utf-8-sig")
    _paper_main_table(repeat_summary).to_csv(run / "tables" / "paper_main_table.csv", index=False, encoding="utf-8-sig")
    # Fold aggregation is explicitly diagnostic, not the final inferential unit.
    _aggregate_metric_frame(fold_metrics).to_csv(run / "tables" / "fold_metrics_summary_DIAGNOSTIC_ONLY.csv", index=False, encoding="utf-8-sig")

    scale_summary = scales_all.groupby(["q", "clique_size"], sort=True).agg(
        n_folds=("outer_id", "size"),
        mean_rms_distance_scale=("rms_distance_scale", "mean"),
        std_rms_distance_scale=("rms_distance_scale", "std"),
        min_rms_distance_scale=("rms_distance_scale", "min"),
        max_rms_distance_scale=("rms_distance_scale", "max"),
        mean_active_train_pair_count=("active_train_pair_count", "mean"),
    ).reset_index()
    scale_summary.to_csv(run / "tables" / "train_only_q_scales_summary.csv", index=False, encoding="utf-8-sig")

    alpha_summary = _alpha_summary(alpha_df)
    alpha_summary.to_csv(run / "tables" / "single_alpha_summary.csv", index=False, encoding="utf-8-sig")

    free_summary = free_weights.groupby(["scaling", "q", "clique_size"], sort=False).agg(
        n_folds=("outer_id", "size"),
        mean_weight=("reference_weight", "mean"),
        std_weight=("reference_weight", "std"),
        median_weight=("reference_weight", "median"),
        mean_train_pair_effective_weight=("mean_train_pair_effective_weight", "mean"),
        mean_test_train_pair_effective_weight=("mean_test_train_pair_effective_weight", "mean"),
    ).reset_index()
    free_summary.to_csv(run / "tables" / "free_q_weights_summary.csv", index=False, encoding="utf-8-sig")
    cosine_pairs, cosine_summary = _free_q_cosine_tables(free_weights)
    cosine_pairs.to_csv(run / "tables" / "free_q_weight_cosine_similarity.csv", index=False, encoding="utf-8-sig")
    cosine_summary.to_csv(run / "tables" / "free_q_weight_cosine_summary.csv", index=False, encoding="utf-8-sig")

    if bool(learning.get("generate_plots", True)):
        from .real_q_weight_scale_cv_plotting import (
            plot_alpha_distribution,
            plot_free_q_heatmap,
            plot_free_q_mean_sd,
            plot_layer_scale_heatmap,
            plot_repeat_metric_distributions,
            plot_repeat_metric_summary,
        )
        plot_repeat_metric_summary(repeat_summary, run / "plots" / "repeat_oof_mean_std")
        plot_repeat_metric_distributions(repeat_metrics, run / "plots" / "repeat_oof_distributions")
        plot_alpha_distribution(alpha_df, run / "plots" / "single_alpha_stability")
        plot_layer_scale_heatmap(scales_all, run / "plots" / "train_only_q_scale_heatmap")
        for scaling in ("raw", "scaled"):
            plot_free_q_heatmap(free_weights[free_weights["scaling"] == scaling], run / "plots" / f"free_q_{scaling}_weight_heatmap", title=f"Free-q weights across 50 folds — {scaling}")
            plot_free_q_mean_sd(free_summary[free_summary["scaling"] == scaling], run / "plots" / f"free_q_{scaling}_weight_mean_sd", title=f"Free-q weight stability — {scaling}")

    pd.DataFrame(runtime_rows).to_csv(run / "tables" / "runtime_rows.csv", index=False, encoding="utf-8-sig")
    write_json(run / "run_manifest.json", {
        "architecture": "train-only q-scale normalization diagnosis over frozen independent-q Hodge VNext core",
        "method_core_changed": False,
        "cache_schema": CACHE_SCHEMA,
        "benchmark_id": benchmark_id,
        "num_networks": len(df),
        "domains": unique_domains,
        "fixed_hodge_settings": {
            "max_clique_size": MAX_CLIQUE_SIZE,
            "computed_maximal_cliques": "K1..K10 only",
            "computed_q": "0..9",
            "max_hodge_order": "all",
            "min_layer_q": 0,
            "normalization": "per_simplex",
        },
        "cv_protocol": {
            "strategy": "repeated_stratified_kfold",
            "n_splits": folds,
            "n_repeats": repeats,
            "seed": outer_seed,
            "n_fold_evaluations": len(outer_splits),
            "final_statistics_unit": "repeat-level merged out-of-fold predictions",
            "network_test_appearances_total": repeats,
        },
        "scale_normalization": {
            "type": "train-only per-q RMS scale-only normalization",
            "cached_tensor_semantics": "C_q is a squared layer distance contribution",
            "distance_scale_formula": "s_q=sqrt(mean_train_active_pairs(C_q))",
            "scaled_squared_contribution_formula": "Ctilde_q=C_q/(s_q+eps)^2",
            "eps": eps,
            "test_leakage": "forbidden; scales use train-train pairs only",
        },
        "weighting_strategies": {
            "Uniform": "beta_q=0, pair-adaptive masked softmax",
            "Single-alpha": {
                "parameterization": "beta_q=alpha*(q/9); pair-adaptive masked softmax",
                "important": "this diagnostic alpha is not numerically comparable to the legacy A_q**alpha aggregation alpha",
                "search": alpha_kwargs,
            },
            "Free-q": {
                "parameterization": "10 logits beta_q; pair-adaptive masked softmax",
                "optimizer": "Adam",
                "objective": "NCA / soft-nearest-neighbour",
                "regularization": "none",
                "settings": opt_kwargs,
            },
        },
        "methods_reported": [name for _, _, name in METHOD_SPECS] + baseline_methods,
        "scientific_boundaries": [
            "Hodge construction, exclusive-q closure, boundary matrices, spectra, heat signatures and staged cache identity are unchanged",
            "train-only scaling is an experiment-protocol transformation applied after cached layer squared distances are constructed",
            "50 folds are correlated partitions; final mean±SD is over 10 repeat-level OOF metrics, not over 50 folds",
            "domain labels are external annotations and not unique structural ground truth",
            "no new regularization hyperparameter is introduced",
            "Single-alpha uses exp(alpha*q/9) as specified by the diagnosis plan and is distinct from the legacy A_q**alpha family",
        ],
        "effects": {
            "method_math_definition": "Hodge core unchanged; adds supervised/normalized aggregation diagnostics",
            "cache_identity": "staged hodge-vnext-stage-cache-v2 unchanged",
            "experiment_statistics": "changed from repeated holdout to repeated stratified 5-fold with repeat-level OOF aggregation",
            "random_graph_seed": "not applicable / unchanged",
            "baseline_semantics": "unchanged",
            "existing_result_comparability": "new CV/scaling protocol must be reported separately from prior 10×80/20 results",
        },
    })
    write_json(complete, {
        "status": "complete",
        "run_dir": str(run),
        "n_fold_evaluations": len(outer_splits),
        "n_repeat_evaluations": repeats,
        "final_statistics": "mean ± SD across 10 repeat-level OOF metrics",
    })
    progress(f"[COMPLETE] {run}")
    return run
