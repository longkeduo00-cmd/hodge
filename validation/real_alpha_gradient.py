from __future__ import annotations

"""Gradient-learned alpha for real-network independent-q Hodge dissimilarity.

Scientific contract
-------------------
* Hodge core remains unchanged.
* qmin is fixed at 0 and normalization at per_simplex for this experiment.
* Formal config uses S=all and Q=all: no label-driven structural truncation.
* alpha is a continuous scalar in R learned by Adam from outer-training labels;
  there is no alpha candidate grid and no alpha min/max bound.
* The differentiable training objective is NCA/soft-nearest-neighbour loss.
  Existing non-differentiable metrics remain evaluation metrics only.
* Repeated outer evaluation never exposes outer-test labels to alpha fitting.
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

from .alpha_gradient import (
    build_pair_layer_energies,
    distance_matrix_from_alpha,
    fit_alpha_adam,
)
from .config import method_from_payload, resolve_path
from .methods import MethodManager, PRIMARY_METHOD
from .real_domains import _prepare_manifest
from .real_learning import (
    _domain_metric_rows,
    _plot_square_method,
    _square_metrics,
    cross_partition_metrics,
    make_cv_splits,
)
from .real_learning_nested import (
    _aggregate_metric_frame,
    _compute_or_load_baseline_matrix,
    _coverage_table,
    _outer_split_membership_table,
    make_repeated_outer_splits,
)


LEARNED_METHOD = "HodgeLearnedAlpha"
ALPHA0_METHOD = "HodgeAlpha0Full"


def _progress(msg: str) -> None:
    print(msg, flush=True)


def _validate_scientific_fixed_config(cfg) -> None:
    if cfg.max_clique_size is not None:
        raise ValueError("gradient-alpha formal experiment requires method.max_clique_size=null (S=all)")
    if cfg.max_hodge_order is not None:
        raise ValueError("gradient-alpha formal experiment requires method.max_hodge_order=null (Q=all)")
    if cfg.min_layer_q != 0:
        raise ValueError("gradient-alpha experiment requires method.min_layer_q=0")
    if cfg.normalization != "per_simplex":
        raise ValueError("gradient-alpha experiment requires method.normalization='per_simplex'")


def _optimizer_kwargs(learning: dict[str, Any]) -> dict[str, Any]:
    opt = dict(learning.get("gradient_optimizer", {}))
    return {
        "initial_alpha": float(opt.get("initial_alpha", 0.0)),
        "learning_rate": float(opt.get("learning_rate", 0.05)),
        "max_epochs": int(opt.get("max_epochs", 2000)),
        "beta1": float(opt.get("beta1", 0.9)),
        "beta2": float(opt.get("beta2", 0.999)),
        "adam_eps": float(opt.get("adam_eps", 1e-8)),
        "grad_tolerance": float(opt.get("grad_tolerance", 1e-8)),
        "loss_tolerance": float(opt.get("loss_tolerance", 1e-10)),
        "patience": int(opt.get("patience", 100)),
        "temperature_multiplier": float(opt.get("temperature_multiplier", 1.0)),
    }


def _run_stem(n: int, benchmark_id: str, payload: dict[str, Any], config_path: Path) -> str:
    cfg = method_from_payload(payload, config_path)
    learning = dict(payload.get("learning", {}))
    outer = dict(learning.get("outer_cv", {}))
    token = stable_hash({
        "benchmark_id": benchmark_id,
        "n": n,
        "method": {
            "S": cfg.max_clique_size,
            "Q": cfg.max_hodge_order,
            "qmin": cfg.min_layer_q,
            "normalization": cfg.normalization,
            "solver": cfg.solver,
            "slq": cfg.slq.__dict__,
            "times": [cfg.time_min, cfg.time_max, cfg.time_points],
        },
        "optimizer": _optimizer_kwargs(learning),
        "outer": outer,
        "inner_diagnostic_folds": int(learning.get("inner_diagnostic_folds", 5)),
        "inner_diagnostic_seed": int(learning.get("inner_diagnostic_seed", 20260930)),
    }, length=12)
    return f"real{n}_{safe_token(benchmark_id)}_gradient-alpha_{token}"


def _compute_profile_worker(index: int, rec: GraphRecord, cfg) -> tuple[int, GraphProfile, float]:
    """Process-safe single-network profile computation for server parallelism."""
    t0 = time.perf_counter()
    profile = compute_graph_profile(rec, cfg, progress=None)
    return index, profile, float(time.perf_counter() - t0)


def _load_or_compute_layer_energies(
    records: Sequence[GraphRecord],
    cfg,
    run: Path,
    runtime_rows: list[dict[str, Any]],
    *,
    recompute: bool,
    profile_workers: int,
    progress: Callable[[str], None],
) -> tuple[list[GraphProfile], np.ndarray, np.ndarray]:
    path = run / "arrays" / "hodge_pair_layer_energies.npz"
    workers = max(1, min(int(profile_workers), len(records))) if records else 1
    profiles_tmp: list[GraphProfile | None] = [None] * len(records)

    if workers == 1:
        for i, rec in enumerate(records):
            idx, profile, elapsed = _compute_profile_worker(i, rec, cfg)
            profiles_tmp[idx] = profile
            runtime_rows.append({
                "stage": "gradient_profile",
                "label": rec.source.get("label", str(i)),
                "elapsed_sec": elapsed,
                "solver": cfg.solver,
                "max_clique_size": cfg.max_clique_size,
                "max_hodge_order": cfg.max_hodge_order,
                "profile_workers": workers,
            })
            progress(f"[PROFILE] {i + 1}/{len(records)} {rec.source.get('label', str(i))} completed")
    else:
        progress(f"[PROFILE PARALLEL] workers={workers}; one network per process; Hodge mathematics unchanged")
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as executor:
            future_map = {executor.submit(_compute_profile_worker, i, rec, cfg): (i, rec) for i, rec in enumerate(records)}
            completed = 0
            for future in as_completed(future_map):
                i, rec = future_map[future]
                idx, profile, elapsed = future.result()
                profiles_tmp[idx] = profile
                completed += 1
                runtime_rows.append({
                    "stage": "gradient_profile",
                    "label": rec.source.get("label", str(i)),
                    "elapsed_sec": elapsed,
                    "solver": cfg.solver,
                    "max_clique_size": cfg.max_clique_size,
                    "max_hodge_order": cfg.max_hodge_order,
                    "profile_workers": workers,
                })
                progress(f"[PROFILE PARALLEL] {completed}/{len(records)} {rec.source.get('label', str(i))} completed")

    if any(profile is None for profile in profiles_tmp):
        raise RuntimeError("profile parallel stage completed with missing network profiles")
    profiles = [profile for profile in profiles_tmp if profile is not None]

    if path.exists() and not recompute:
        obj = np.load(path)
        energies = np.asarray(obj["energies"], dtype=float)
        q_star = np.asarray(obj["q_star"], dtype=np.int32)
        if energies.shape[:2] == (len(records), len(records)) and q_star.shape == (len(records), len(records)):
            progress("[HODGE CACHE] pair-layer energy tensor hit")
            return profiles, energies, q_star

    energies, q_star = build_pair_layer_energies(
        profiles,
        min_layer_q=cfg.min_layer_q,
        max_hodge_order=cfg.max_hodge_order,
    )
    np.savez_compressed(path, energies=energies, q_star=q_star)
    progress(f"[HODGE CACHE] saved pair-layer energy tensor shape={energies.shape}")
    return profiles, energies, q_star


def _audit_energy_reconstruction(profiles, energies, q_star, cfg) -> pd.DataFrame:
    """Verify the differentiable decomposition exactly reconstructs core distance."""
    rows: list[dict[str, Any]] = []
    n = len(profiles)
    test_alphas = (-16.0, 0.0, 2.0)
    pairs = [(i, j) for i in range(min(n, 4)) for j in range(i + 1, min(n, 4))]
    for alpha in test_alphas:
        D = distance_matrix_from_alpha(
            alpha, energies, q_star,
            min_layer_q=cfg.min_layer_q,
            max_hodge_order=cfg.max_hodge_order,
        )
        cfg_a = replace(cfg, alpha=float(alpha))
        for i, j in pairs:
            core = profile_distance(profiles[i], profiles[j], cfg_a).total_distance
            recon = float(D[i, j])
            err = abs(core - recon)
            rows.append({"alpha": alpha, "i": i, "j": j, "core_distance": core, "reconstructed_distance": recon, "abs_error": err})
            if err > 1e-10 * max(1.0, abs(core)):
                raise AssertionError(f"gradient decomposition mismatch alpha={alpha} pair=({i},{j}) err={err}")
    return pd.DataFrame(rows)


def run_real_alpha_gradient(
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
        raise ValueError("gradient alpha learning requires at least two domains")

    strategy = str(outer_cfg.get("strategy", "repeated_stratified_kfold"))
    repeats = int(outer_cfg.get("repeats", 5))
    folds = int(outer_cfg.get("folds", 4))
    outer_seed = int(outer_cfg.get("seed", 20260930))
    test_fraction = float(outer_cfg.get("test_fraction", 0.2))
    min_test = int(outer_cfg.get("require_min_test_per_domain", 2))
    inner_folds = int(learning.get("inner_diagnostic_folds", 5))
    inner_seed_base = int(learning.get("inner_diagnostic_seed", 20260930))

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

    benchmark_id = str(exp.get("benchmark_id", "real48-gradient-alpha"))
    results_root = resolve_path(learning.get("results_root", "results/validation/real_alpha_gradient"), config_path)
    run = results_root / _run_stem(len(df), benchmark_id, payload, config_path)
    complete = run / "RUN_COMPLETE.json"
    if complete.exists() and not force:
        progress(f"[SKIP] completed gradient-alpha run exists: {run}")
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

    effective = json.loads(json.dumps(payload))
    effective["cache_schema"] = CACHE_SCHEMA
    effective["gradient_alpha_contract"] = {
        "alpha_domain": "unbounded real line; no candidate grid / min / max",
        "training_loss": "NCA / soft-nearest-neighbour differentiable surrogate",
        "qmin_fixed": 0,
        "normalization_fixed": "per_simplex",
        "S": "all" if cfg.max_clique_size is None else cfg.max_clique_size,
        "Q": "all" if cfg.max_hodge_order is None else cfg.max_hodge_order,
        "final_metrics_not_optimized_directly": ["1-NN Macro-F1", "AUROC", "mAP", "NDCG", "ARI", "NMI", "MST"],
    }
    write_json(run / "effective_config.json", effective)

    records: list[GraphRecord] = []
    for _, row in df.iterrows():
        rec = create_real_record(row.path, cfg.cache.root, str(row.label), row.nodes_path or None)
        records.append(rec)
        progress(f"[REAL] {row.domain}/{row.label}: N={rec.graph.number_of_nodes()} E={rec.graph.number_of_edges()}")

    runtime_rows: list[dict[str, Any]] = []
    profiles, energies, q_star = _load_or_compute_layer_energies(
        records, cfg, run, runtime_rows,
        recompute=recompute_hodge, profile_workers=profile_workers, progress=progress,
    )
    audit_df = _audit_energy_reconstruction(profiles, energies, q_star, cfg)
    audit_df.to_csv(run / "tables" / "gradient_distance_reconstruction_audit.csv", index=False, encoding="utf-8-sig")

    # Full-information alpha=0 comparator uses the same cached profiles/energies.
    alpha0_D = distance_matrix_from_alpha(
        0.0, energies, q_star,
        min_layer_q=cfg.min_layer_q,
        max_hodge_order=cfg.max_hodge_order,
    )
    np.save(run / "arrays" / f"distance_matrix_{ALPHA0_METHOD}_full.npy", alpha0_D)
    pd.DataFrame(alpha0_D, index=labels, columns=labels).to_csv(
        run / "tables" / f"distance_matrix_{ALPHA0_METHOD}_full.csv", encoding="utf-8-sig"
    )

    # Existing baselines are computed once; they are label-free and shared by all outer splits.
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

    outer_fit_rows: list[dict[str, Any]] = []
    outer_traj_frames: list[pd.DataFrame] = []
    inner_rows: list[dict[str, Any]] = []
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

        # Inner CV is diagnostic only: it measures alpha-learning stability and never
        # selects a candidate or exposes the outer test set.
        inner_splits = make_cv_splits(outer.train_idx, domains, inner_folds, inner_seed_base + oi - 1)
        for inner_fold, (itr, iva) in enumerate(inner_splits, start=1):
            fit = fit_alpha_adam(
                energies, q_star, itr, domains,
                min_layer_q=cfg.min_layer_q,
                max_hodge_order=cfg.max_hodge_order,
                **opt_kwargs,
            )
            D_inner = distance_matrix_from_alpha(
                fit.alpha, energies, q_star,
                min_layer_q=cfg.min_layer_q,
                max_hodge_order=cfg.max_hodge_order,
            )
            met, _, _ = cross_partition_metrics(D_inner, iva, itr, domains, labels)
            inner_rows.append({
                "outer_id": outer.outer_id,
                "repeat": outer.repeat,
                "outer_fold": outer.fold,
                "inner_fold": inner_fold,
                "alpha": fit.alpha,
                "train_loss": fit.best_loss,
                "final_grad": fit.final_grad,
                "epochs": fit.epochs,
                "converged": fit.converged,
                "stop_reason": fit.stop_reason,
                "temperature": fit.temperature,
                **met,
            })

        fit = fit_alpha_adam(
            energies, q_star, outer.train_idx, domains,
            min_layer_q=cfg.min_layer_q,
            max_hodge_order=cfg.max_hodge_order,
            **opt_kwargs,
        )
        fit_row = {
            "outer_id": outer.outer_id,
            "repeat": outer.repeat,
            "fold": outer.fold,
            "outer_seed": outer.seed,
            "train_size": len(outer.train_idx),
            "test_size": len(outer.test_idx),
            "alpha": fit.alpha,
            "train_loss": fit.best_loss,
            "final_grad": fit.final_grad,
            "epochs": fit.epochs,
            "converged": fit.converged,
            "stop_reason": fit.stop_reason,
            "temperature": fit.temperature,
        }
        outer_fit_rows.append(fit_row)
        traj = fit.trajectory.copy()
        traj.insert(0, "outer_id", outer.outer_id)
        outer_traj_frames.append(traj)
        fit.trajectory.to_csv(outer_dir / "alpha_optimization_trajectory.csv", index=False, encoding="utf-8-sig")
        write_json(outer_dir / "learned_alpha.json", fit_row)

        learned_D = distance_matrix_from_alpha(
            fit.alpha, energies, q_star,
            min_layer_q=cfg.min_layer_q,
            max_hodge_order=cfg.max_hodge_order,
        )
        np.save(outer_dir / "distance_matrix_HodgeLearnedAlpha_full.npy", learned_D)

        matrices: dict[str, np.ndarray] = {
            LEARNED_METHOD: learned_D,
            ALPHA0_METHOD: alpha0_D,
            **baseline_matrices,
        }
        pd.DataFrame({
            "index": np.arange(len(df), dtype=int),
            "label": labels,
            "domain": domains,
            "partition": ["test" if i in set(outer.test_idx.tolist()) else "train" for i in range(len(df))],
        }).to_csv(outer_dir / "split.csv", index=False, encoding="utf-8-sig")

        per_outer_inductive: list[dict[str, Any]] = []
        per_outer_square: list[dict[str, Any]] = []
        for method, D in matrices.items():
            met, qdf, rdf = cross_partition_metrics(D, outer.test_idx, outer.train_idx, domains, labels)
            meta = {
                "outer_id": outer.outer_id, "repeat": outer.repeat, "fold": outer.fold,
                "outer_seed": outer.seed, "train_size": len(outer.train_idx), "test_size": len(outer.test_idx),
                "learned_alpha": fit.alpha, "method": method,
            }
            row = {**meta, **met}
            inductive_rows.append(row)
            per_outer_inductive.append(row)
            for col, val in reversed(list(meta.items())):
                qdf.insert(0, col, val)
                rdf.insert(0, col, val)
            inductive_qframes.append(qdf)
            inductive_rframes.append(rdf)

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
            for drow in _domain_metric_rows(method, sq_qdf):
                square_domain_rows.append({
                    "outer_id": outer.outer_id, "repeat": outer.repeat, "fold": outer.fold,
                    "outer_seed": outer.seed, "learned_alpha": fit.alpha, **drow,
                })
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
        pd.DataFrame(per_outer_square).to_csv(outer_dir / "square_test_metrics.csv", index=False, encoding="utf-8-sig")
        write_json(outer_dir / "OUTER_COMPLETE.json", {"status": "complete", **fit_row})

        # Long-run checkpoints.
        pd.DataFrame(outer_fit_rows).to_csv(run / "tables" / "learned_alpha_by_outer_split.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(inductive_rows).to_csv(run / "tables" / "generalization_metrics_by_outer_split.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(square_rows).to_csv(run / "tables" / "square_test_metrics_by_outer_split.csv", index=False, encoding="utf-8-sig")

    outer_fit_df = pd.DataFrame(outer_fit_rows)
    inner_df = pd.DataFrame(inner_rows)
    inductive_df = pd.DataFrame(inductive_rows)
    square_df = pd.DataFrame(square_rows)
    traj_df = pd.concat(outer_traj_frames, ignore_index=True)
    inductive_qdf = pd.concat(inductive_qframes, ignore_index=True)
    inductive_rdf = pd.concat(inductive_rframes, ignore_index=True)
    square_qdf = pd.concat(square_qframes, ignore_index=True)
    square_rdf = pd.concat(square_rframes, ignore_index=True)
    square_domain_df = pd.DataFrame(square_domain_rows)

    outer_fit_df.to_csv(run / "tables" / "learned_alpha_by_outer_split.csv", index=False, encoding="utf-8-sig")
    inner_df.to_csv(run / "tables" / "inner_alpha_stability_diagnostics.csv", index=False, encoding="utf-8-sig")
    traj_df.to_csv(run / "tables" / "outer_alpha_optimization_trajectories.csv", index=False, encoding="utf-8-sig")
    inductive_df.to_csv(run / "tables" / "generalization_metrics_by_outer_split.csv", index=False, encoding="utf-8-sig")
    inductive_qdf.to_csv(run / "tables" / "generalization_per_query.csv", index=False, encoding="utf-8-sig")
    inductive_rdf.to_csv(run / "tables" / "generalization_rankings.csv", index=False, encoding="utf-8-sig")
    square_df.to_csv(run / "tables" / "square_test_metrics_by_outer_split.csv", index=False, encoding="utf-8-sig")
    square_qdf.to_csv(run / "tables" / "square_test_dataset_metrics.csv", index=False, encoding="utf-8-sig")
    square_rdf.to_csv(run / "tables" / "square_test_rankings.csv", index=False, encoding="utf-8-sig")
    square_domain_df.to_csv(run / "tables" / "square_test_domain_metrics.csv", index=False, encoding="utf-8-sig")

    inductive_summary = _aggregate_metric_frame(inductive_df, metadata_cols=("learned_alpha",))
    square_summary = _aggregate_metric_frame(square_df, metadata_cols=("learned_alpha",))
    inductive_summary.to_csv(run / "tables" / "generalization_metrics_summary.csv", index=False, encoding="utf-8-sig")
    square_summary.to_csv(run / "tables" / "square_test_metrics_summary.csv", index=False, encoding="utf-8-sig")
    alpha_summary = {
        "n_outer_evaluations": int(len(outer_fit_df)),
        "mean_alpha": float(outer_fit_df["alpha"].mean()),
        "std_alpha": float(outer_fit_df["alpha"].std(ddof=1)) if len(outer_fit_df) > 1 else 0.0,
        "median_alpha": float(outer_fit_df["alpha"].median()),
        "min_alpha": float(outer_fit_df["alpha"].min()),
        "max_alpha": float(outer_fit_df["alpha"].max()),
        "converged_fraction": float(outer_fit_df["converged"].mean()),
    }
    write_json(run / "learned_alpha_summary.json", alpha_summary)

    if bool(learning.get("generate_plots", True)):
        from .real_learning_nested_plotting import (
            plot_nested_confusion,
            plot_nested_metric_summary,
            plot_outer_metric_distributions,
            plot_outer_test_coverage,
        )
        from .real_alpha_gradient_plotting import (
            plot_alpha_by_outer,
            plot_alpha_histogram,
            plot_outer_loss_trajectories,
            plot_outer_training_trajectories,
        )
        plot_nested_metric_summary(inductive_summary, run / "plots" / "generalization_mean_std", title="Gradient-learned alpha: outer-test → outer-train generalization")
        plot_outer_metric_distributions(inductive_df, run / "plots" / "generalization_distributions")
        plot_outer_test_coverage(coverage, run / "plots" / "outer_test_coverage")
        learned_queries = inductive_qdf[inductive_qdf["method"] == LEARNED_METHOD].copy()
        plot_nested_confusion(learned_queries, unique_domains, run / "plots" / "HodgeLearnedAlpha_aggregate_outer_confusion")
        plot_alpha_by_outer(outer_fit_df, run / "plots" / "learned_alpha_by_outer")
        plot_alpha_histogram(outer_fit_df, run / "plots" / "learned_alpha_distribution")
        plot_outer_training_trajectories(traj_df, run / "plots" / "alpha_optimization_trajectories")
        plot_outer_loss_trajectories(traj_df, run / "plots" / "nca_loss_trajectories")

        # If the outer protocol is a single hold-out, also emit the full existing
        # square-method visual grammar for direct compatibility with the prior runner.
        if len(outer_splits) == 1:
            outer = outer_splits[0]
            fit = outer_fit_df.iloc[0]
            learned_D = np.load(run / "outer" / outer.outer_id / "distance_matrix_HodgeLearnedAlpha_full.npy")
            matrices = {LEARNED_METHOD: learned_D, ALPHA0_METHOD: alpha0_D, **baseline_matrices}
            plot_options = dict(exp.get("plotting", {})); plot_options.update(payload.get("plotting", {}))
            for method, D in matrices.items():
                _, _, _, clustering = _square_metrics(D, outer.test_idx, labels, domains, method, linkage_method, mst_permutations, eval_seed)
                _plot_square_method(run, "test_holdout", method, D, outer.test_idx, labels, domains, clustering, plot_options, eval_seed, skip_existing=not force)

    pd.DataFrame(runtime_rows).to_csv(run / "tables" / "runtime_rows.csv", index=False, encoding="utf-8-sig")
    write_json(run / "run_manifest.json", {
        "architecture": "continuous gradient alpha learning over frozen independent-q Hodge VNext core",
        "method_core_changed": False,
        "cache_schema": CACHE_SCHEMA,
        "benchmark_id": benchmark_id,
        "num_networks": len(df),
        "domains": unique_domains,
        "fixed_scientific_settings": {
            "max_clique_size": "all" if cfg.max_clique_size is None else cfg.max_clique_size,
            "max_hodge_order": "all" if cfg.max_hodge_order is None else cfg.max_hodge_order,
            "min_layer_q": cfg.min_layer_q,
            "normalization": cfg.normalization,
        },
        "learned_parameter": {
            "name": "alpha",
            "domain": "R (unbounded)",
            "optimizer": "Adam",
            "objective": "NCA/soft-nearest-neighbour loss",
            "optimizer_settings": opt_kwargs,
        },
        "outer_evaluation": {
            "strategy": strategy, "repeats": repeats, "folds": folds,
            "seed": outer_seed, "num_outer_evaluations": len(outer_splits),
        },
        "inner_diagnostic_cv": {
            "folds": inner_folds,
            "role": "stability diagnostic only; not candidate-grid selection",
        },
        "methods_reported": [LEARNED_METHOD, ALPHA0_METHOD, *baseline_methods],
        "execution": {
            "profile_workers": profile_workers,
            "gpu_backend": "none; Hodge/SLQ remains NumPy/SciPy CPU",
            "parallelism_scope": "independent networks only",
        },
        "scientific_boundaries": [
            "outer-test labels are never used to fit alpha",
            "1-NN Macro-F1/AUROC/mAP and other discrete/ranking metrics are evaluation metrics, not gradient objectives",
            "alpha has no candidate grid and no hard min/max bound",
            "S and Q are fixed structural settings, not differentiable parameters; formal config uses all/all to avoid manual truncation",
            "qmin=0 and normalization=per_simplex are fixed by experiment contract",
            "domain labels are external annotations, not unique structural ground truth",
            "repeated outer scores are correlated resamples; empirical quantiles are not iid confidence intervals",
        ],
        "effects": {
            "method_math_definition": "unchanged core; existing alpha-parameterized aggregation is optimized continuously",
            "cache_identity": "unchanged staged cache; new label-free pair-layer-energy run artifact only",
            "experiment_statistics": "new supervised metric-learning protocol with repeated outer evaluation",
            "random_graph_seed": "not applicable / unchanged",
            "baseline_semantics": "unchanged",
            "existing_result_comparability": "separate supervised extension; do not merge with original unsupervised Hodge benchmark",
        },
    })
    write_json(complete, {"status": "complete", "run_dir": str(run), "num_outer_evaluations": len(outer_splits), **alpha_summary})
    progress(f"[COMPLETE] {run}")
    return run
