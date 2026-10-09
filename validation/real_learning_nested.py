from __future__ import annotations

"""Repeated nested resampling for supervised Hodge hyper-parameter learning.

This module is an *additional validation protocol* layered on top of the frozen
Hodge VNext mathematical core.  It does not alter clique enumeration, exclusive
q-layers, downward closure, Hodge operators, heat signatures, or the existing
pair-adaptive distance definition.

The expensive graph representations / candidate distance matrices are computed
once for the full fixed real-network benchmark.  Labels are used only to define
outer/inner resampling partitions and to score candidate configurations inside
outer-training data.  Precomputing a label-free distance matrix is therefore
mathematically equivalent to recomputing the same fixed candidate on each fold,
but substantially cheaper.

Supported outer protocols
-------------------------
``repeated_stratified_kfold`` (recommended for formal real48 analysis)
    Repeated StratifiedKFold.  For real48, 4 outer folds are recommended rather
    than 5 because the two nine-network domains then still contribute >=2 test
    networks per fold, so the complete existing square-test metric grammar
    (retrieval, clustering and MST diagnostics) remains defined.

``repeated_stratified_holdout``
    Repeated deterministic stratified train/test splits.  With test_fraction=.2
    on the current real48 benchmark this yields 38 train / 10 test and exactly
    two test networks per domain, matching the single-split learning experiment.

Every outer evaluation performs a fresh inner stratified K-fold CV on its own
outer-training partition and selects a Hodge candidate without using outer-test
labels.  Outer-test scores are then aggregated as a resampling-stability study;
repeated scores are correlated and are not treated as iid samples.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence
import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from hodge_vnext.cache import CACHE_SCHEMA
from hodge_vnext.graph_io import GraphRecord, create_real_record
from hodge_vnext.utils import safe_token, stable_hash, write_json

from .config import method_from_payload, resolve_path
from .methods import MethodManager, PRIMARY_METHOD
from .real_domains import _prepare_manifest
from .real_metrics import validate_distance_matrix
from .real_learning import (
    Candidate,
    LEARNED_METHOD,
    FROZEN_METHOD,
    _candidate_distance_matrix,
    _cv_score_candidate,
    _domain_metric_rows,
    _square_metrics,
    candidate_config,
    cross_partition_metrics,
    enumerate_candidates,
    make_cv_splits,
    make_outer_split,
    select_best_candidate,
)


@dataclass(frozen=True)
class NestedOuterSplit:
    outer_id: str
    repeat: int
    fold: int
    seed: int
    train_idx: np.ndarray
    test_idx: np.ndarray


def _progress(msg: str) -> None:
    print(msg, flush=True)


def _normalise_outer_strategy(value: str) -> str:
    v = str(value).strip().lower().replace("-", "_")
    aliases = {
        "kfold": "repeated_stratified_kfold",
        "repeated_kfold": "repeated_stratified_kfold",
        "stratified_kfold": "repeated_stratified_kfold",
        "repeated_stratified_kfold": "repeated_stratified_kfold",
        "holdout": "repeated_stratified_holdout",
        "repeated_holdout": "repeated_stratified_holdout",
        "stratified_holdout": "repeated_stratified_holdout",
        "repeated_stratified_holdout": "repeated_stratified_holdout",
    }
    if v not in aliases:
        raise ValueError(
            "learning.nested_cv.outer_strategy must be one of "
            "repeated_stratified_kfold or repeated_stratified_holdout"
        )
    return aliases[v]


def make_repeated_outer_splits(
    domains: Sequence[str],
    *,
    strategy: str,
    repeats: int,
    seed: int,
    outer_folds: int = 4,
    test_fraction: float = 0.2,
    require_min_test_per_domain: int = 1,
) -> list[NestedOuterSplit]:
    """Create deterministic repeated outer resampling partitions.

    For repeated K-fold, each network appears in the test set exactly once per
    repeat.  For repeated hold-out, each repeat is an independently-seeded
    stratified split.
    """
    labels = np.asarray(domains, dtype=str)
    strategy = _normalise_outer_strategy(strategy)
    repeats = int(repeats)
    seed = int(seed)
    require_min_test_per_domain = int(require_min_test_per_domain)
    if repeats < 1:
        raise ValueError("learning.nested_cv.outer_repeats must be >=1")
    if require_min_test_per_domain < 1:
        raise ValueError("require_min_test_per_domain must be >=1")

    counts = pd.Series(labels).value_counts()
    out: list[NestedOuterSplit] = []
    if strategy == "repeated_stratified_kfold":
        outer_folds = int(outer_folds)
        if outer_folds < 2:
            raise ValueError("learning.nested_cv.outer_folds must be >=2")
        if outer_folds > int(counts.min()):
            raise ValueError(
                f"outer_folds={outer_folds} exceeds smallest domain count={int(counts.min())}"
            )
        for repeat in range(1, repeats + 1):
            repeat_seed = seed + repeat - 1
            skf = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=repeat_seed)
            for fold, (tr, te) in enumerate(skf.split(np.zeros(len(labels)), labels), start=1):
                tr = np.sort(np.asarray(tr, dtype=int))
                te = np.sort(np.asarray(te, dtype=int))
                test_counts = pd.Series(labels[te]).value_counts()
                if int(test_counts.min()) < require_min_test_per_domain:
                    raise ValueError(
                        f"outer repeat={repeat} fold={fold} has minimum test-domain count "
                        f"{int(test_counts.min())}, below required {require_min_test_per_domain}; "
                        "reduce outer_folds or lower the compatibility requirement"
                    )
                out.append(NestedOuterSplit(
                    outer_id=f"r{repeat:02d}_f{fold:02d}",
                    repeat=repeat,
                    fold=fold,
                    seed=repeat_seed,
                    train_idx=tr,
                    test_idx=te,
                ))
    else:
        for repeat in range(1, repeats + 1):
            repeat_seed = seed + repeat - 1
            split = make_outer_split(labels, float(test_fraction), repeat_seed)
            test_counts = pd.Series(labels[split.test_idx]).value_counts()
            if int(test_counts.min()) < require_min_test_per_domain:
                raise ValueError(
                    f"outer repeat={repeat} has minimum test-domain count {int(test_counts.min())}, "
                    f"below required {require_min_test_per_domain}; increase test_fraction"
                )
            out.append(NestedOuterSplit(
                outer_id=f"r{repeat:02d}_h01",
                repeat=repeat,
                fold=1,
                seed=repeat_seed,
                train_idx=split.train_idx,
                test_idx=split.test_idx,
            ))
    return out


def _outer_split_membership_table(
    df: pd.DataFrame, splits: Sequence[NestedOuterSplit]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split in splits:
        test = set(int(x) for x in split.test_idx)
        for i, row in df.reset_index(drop=True).iterrows():
            rows.append({
                "outer_id": split.outer_id,
                "repeat": split.repeat,
                "fold": split.fold,
                "outer_seed": split.seed,
                "index": int(i),
                "label": str(row.label),
                "domain": str(row.domain),
                "partition": "test" if i in test else "train",
            })
    return pd.DataFrame(rows)


def _aggregate_metric_frame(
    frame: pd.DataFrame,
    *,
    group_col: str = "method",
    metadata_cols: Sequence[str] = (),
) -> pd.DataFrame:
    """Aggregate repeated outer scores without pretending the splits are iid.

    The q025/q975 columns are empirical resampling quantiles, not confidence
    intervals.  This distinction is carried into the run manifest.
    """
    skip = {
        group_col, "outer_id", "repeat", "fold", "outer_seed", "inner_seed",
        "train_size", "test_size", "selected_candidate_id", "candidate_id",
        *metadata_cols,
    }
    metric_cols = [
        c for c in frame.columns
        if c not in skip and pd.api.types.is_numeric_dtype(frame[c])
    ]
    rows: list[dict[str, Any]] = []
    for group, sub in frame.groupby(group_col, sort=False):
        row: dict[str, Any] = {group_col: group, "n_outer_evaluations": int(len(sub))}
        for col in metric_cols:
            vals = sub[col].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                row[f"mean_{col}"] = np.nan
                row[f"std_{col}"] = np.nan
                row[f"median_{col}"] = np.nan
                row[f"q025_{col}"] = np.nan
                row[f"q975_{col}"] = np.nan
                continue
            row[f"mean_{col}"] = float(np.mean(vals))
            row[f"std_{col}"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            row[f"median_{col}"] = float(np.median(vals))
            row[f"q025_{col}"] = float(np.quantile(vals, 0.025))
            row[f"q975_{col}"] = float(np.quantile(vals, 0.975))
        rows.append(row)
    return pd.DataFrame(rows)


def _selection_frequency(selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    key_cols = [
        "candidate_id", "max_clique_size", "max_hodge_order", "min_layer_q",
        "normalization", "alpha",
    ]
    cand = (
        selected.groupby(key_cols, dropna=False, sort=False)
        .size().rename("count").reset_index()
        .sort_values(["count", "candidate_id"], ascending=[False, True])
        .reset_index(drop=True)
    )
    cand["proportion"] = cand["count"] / max(1, len(selected))

    rows: list[dict[str, Any]] = []
    for param in ["max_clique_size", "max_hodge_order", "min_layer_q", "normalization", "alpha"]:
        values = selected[param].copy()
        values = values.where(~values.isna(), "all")
        vc = values.astype(str).value_counts(dropna=False, sort=False)
        for value, count in vc.items():
            rows.append({
                "parameter": param,
                "value": str(value),
                "count": int(count),
                "proportion": float(count / max(1, len(selected))),
            })
    return cand, pd.DataFrame(rows)


def _coverage_table(membership: pd.DataFrame) -> pd.DataFrame:
    g = membership.groupby(["index", "label", "domain"], sort=False)
    out = g["partition"].agg(
        outer_evaluations="size",
        test_appearances=lambda s: int(np.sum(np.asarray(s) == "test")),
        train_appearances=lambda s: int(np.sum(np.asarray(s) == "train")),
    ).reset_index()
    out["test_fraction_observed"] = out["test_appearances"] / out["outer_evaluations"]
    return out


def _nested_run_stem(
    n: int,
    benchmark_id: str,
    learning: dict[str, Any],
    candidates: Sequence[Candidate],
) -> str:
    nested = dict(learning.get("nested_cv", {}))
    payload = {
        "outer_strategy": _normalise_outer_strategy(nested.get("outer_strategy", "repeated_stratified_kfold")),
        "outer_repeats": int(nested.get("outer_repeats", 5)),
        "outer_folds": int(nested.get("outer_folds", 4)),
        "outer_seed": int(nested.get("outer_seed", 20260930)),
        "test_fraction": float(nested.get("test_fraction", learning.get("test_fraction", 0.2))),
        "inner_folds": int(nested.get("inner_folds", learning.get("cv_folds", 5))),
        "inner_seed": int(nested.get("inner_seed", learning.get("cv_seed", 20260930))),
        "primary_metric": str(learning.get("primary_metric", "nn1_macro_f1")),
        "candidates": [c.as_dict() for c in candidates],
    }
    return f"bench-{safe_token(benchmark_id)}_n{n}_nested-{stable_hash(payload, 12)}"


def _clear_nested_report_outputs(run: Path) -> None:
    # Candidate/full matrices are intentionally preserved under --force.
    for name in ("tables", "plots", "outer"):
        p = run / name
        if p.exists():
            import shutil
            shutil.rmtree(p)
    for name in ("RUN_COMPLETE.json", "run_manifest.json", "effective_config.json"):
        p = run / name
        if p.exists():
            p.unlink()


def _matrix_matches_candidate(base_cfg, candidate: Candidate) -> bool:
    return (
        candidate.max_clique_size == base_cfg.max_clique_size
        and candidate.max_hodge_order == base_cfg.max_hodge_order
        and candidate.min_layer_q == base_cfg.min_layer_q
        and candidate.normalization == base_cfg.normalization
        and float(candidate.alpha) == float(base_cfg.alpha)
    )


def _compute_or_load_baseline_matrix(
    method: str,
    records: Sequence[GraphRecord],
    labels: Sequence[str],
    manager: MethodManager,
    run: Path,
    runtime_rows: list[dict[str, Any]],
    *,
    recompute: bool,
    progress: Callable[[str], None],
) -> np.ndarray:
    path = run / "arrays" / f"distance_matrix_{method}_full.npy"
    if path.exists() and not recompute:
        D = validate_distance_matrix(np.load(path))
        if D.shape == (len(records), len(records)):
            pd.DataFrame(D, index=labels, columns=labels).to_csv(
                run / "tables" / f"distance_matrix_{method}_full.csv", encoding="utf-8-sig"
            )
            progress(f"[BASELINE CACHE] {method}: full distance matrix hit")
            return D
    reps = []
    for i, rec in enumerate(records):
        t0 = time.perf_counter()
        rep, elapsed, hit = manager.representation(method, rec)
        reps.append(rep)
        runtime_rows.append({
            "stage": "baseline_representation",
            "method": method,
            "label": labels[i],
            "elapsed_sec": float(elapsed),
            "wall_sec": float(time.perf_counter() - t0),
            "cache_hit": bool(hit),
        })
    D = np.zeros((len(records), len(records)), dtype=float)
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            d = manager.distance(
                method, reps[i], reps[j],
                records[i].graph.number_of_nodes(), records[j].graph.number_of_nodes(),
            )
            D[i, j] = D[j, i] = float(d)
    np.save(path, D)
    pd.DataFrame(D, index=labels, columns=labels).to_csv(
        run / "tables" / f"distance_matrix_{method}_full.csv", encoding="utf-8-sig"
    )
    progress(f"[BASELINE CACHE] {method}: saved full distance matrix")
    return D


def run_real_learning_nested(
    payload: dict[str, Any],
    config_path: Path,
    *,
    force: bool = False,
    recompute_candidates: bool = False,
    recompute_baselines: bool = False,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Run repeated nested supervised hyper-parameter learning on real networks."""
    progress = progress or _progress
    base_cfg = method_from_payload(payload, config_path)
    exp = dict(payload["real_validation"])
    learning = dict(payload.get("learning", {}))
    nested = dict(learning.get("nested_cv", {}))
    df = _prepare_manifest(payload, config_path)
    labels = df["label"].astype(str).tolist()
    domains = df["domain"].astype(str).tolist()
    unique_domains = list(dict.fromkeys(domains))
    if len(unique_domains) < 2:
        raise ValueError("nested real learning requires at least two domains")

    candidates = enumerate_candidates(base_cfg, learning)
    if not candidates:
        raise ValueError("learning search space produced no candidates")

    strategy = _normalise_outer_strategy(nested.get("outer_strategy", "repeated_stratified_kfold"))
    outer_repeats = int(nested.get("outer_repeats", 5))
    outer_folds = int(nested.get("outer_folds", 4))
    outer_seed = int(nested.get("outer_seed", 20260930))
    test_fraction = float(nested.get("test_fraction", learning.get("test_fraction", 0.2)))
    inner_folds = int(nested.get("inner_folds", learning.get("cv_folds", 5)))
    inner_seed_base = int(nested.get("inner_seed", learning.get("cv_seed", 20260930)))
    require_min_test = int(nested.get("require_min_test_per_domain", 2))
    primary_metric = str(learning.get("primary_metric", "nn1_macro_f1"))

    outer_splits = make_repeated_outer_splits(
        domains,
        strategy=strategy,
        repeats=outer_repeats,
        seed=outer_seed,
        outer_folds=outer_folds,
        test_fraction=test_fraction,
        require_min_test_per_domain=require_min_test,
    )

    benchmark_id = str(exp.get("benchmark_id", "real-learning")).strip() or "real-learning"
    results_root = resolve_path(
        learning.get("results_root", "results/validation/real_learning_nested"), config_path
    )
    run = results_root / _nested_run_stem(len(df), benchmark_id, learning, candidates)
    complete = run / "RUN_COMPLETE.json"
    if complete.exists() and not force:
        progress(f"[SKIP] completed nested-learning run exists: {run}")
        return run
    run.mkdir(parents=True, exist_ok=True)
    if force:
        _clear_nested_report_outputs(run)
    for d in ("tables", "plots", "arrays", "candidate_matrices", "outer"):
        (run / d).mkdir(parents=True, exist_ok=True)

    membership = _outer_split_membership_table(df, outer_splits)
    membership.to_csv(run / "tables" / "outer_split_membership.csv", index=False, encoding="utf-8-sig")
    coverage = _coverage_table(membership)
    coverage.to_csv(run / "tables" / "outer_test_coverage.csv", index=False, encoding="utf-8-sig")
    split_counts = (
        membership.groupby(["outer_id", "repeat", "fold", "partition", "domain"], sort=False)
        .size().rename("count").reset_index()
    )
    split_counts.to_csv(run / "tables" / "outer_split_domain_counts.csv", index=False, encoding="utf-8-sig")

    effective = json.loads(json.dumps(payload))
    effective["cache_schema"] = CACHE_SCHEMA
    effective["nested_learning_protocol"] = {
        "outer_strategy": strategy,
        "outer_repeats": outer_repeats,
        "outer_folds": outer_folds if strategy == "repeated_stratified_kfold" else None,
        "test_fraction": test_fraction if strategy == "repeated_stratified_holdout" else None,
        "outer_seed": outer_seed,
        "inner_folds": inner_folds,
        "inner_seed_base": inner_seed_base,
        "candidate_selection": "inner stratified CV on each outer-training partition only",
        "primary_generalization": "outer-test queries -> corresponding outer-training gallery",
        "square_outer_test": "existing real-domain metric grammar; enabled because each test domain has >= required minimum",
        "resampling_intervals": "empirical q025/q975 across correlated outer evaluations; not iid confidence intervals",
        "solver_tuned": False,
        "numerical_accuracy_controls_tuned": False,
    }
    write_json(run / "effective_config.json", effective)

    records: list[GraphRecord] = []
    for _, row in df.iterrows():
        rec = create_real_record(row.path, base_cfg.cache.root, str(row.label), row.nodes_path or None)
        records.append(rec)
        progress(f"[REAL] {row.domain}/{row.label}: N={rec.graph.number_of_nodes()} E={rec.graph.number_of_edges()}")

    runtime_rows: list[dict[str, Any]] = []
    candidate_matrices: dict[str, np.ndarray] = {}
    progress(
        f"[CANDIDATE MATRICES] candidates={len(candidates)} networks={len(records)}; "
        "computed once and reused across all outer/inner splits"
    )
    for ci, candidate in enumerate(candidates, start=1):
        progress(f"[CANDIDATE] {ci}/{len(candidates)} {candidate.as_dict()}")
        candidate_matrices[candidate.candidate_id] = _candidate_distance_matrix(
            records, base_cfg, candidate, run / "candidate_matrices", runtime_rows,
            recompute=recompute_candidates, progress=progress,
        )

    # Frozen Hodge is exactly the input method block.  Reuse a candidate matrix
    # if the search space already contains that exact configuration.
    frozen_match = next((c for c in candidates if _matrix_matches_candidate(base_cfg, c)), None)
    if frozen_match is not None:
        frozen_D = candidate_matrices[frozen_match.candidate_id]
        progress(f"[FROZEN] reused candidate matrix {frozen_match.candidate_id}")
    else:
        frozen_candidate = Candidate(
            "frozen",
            base_cfg.max_clique_size,
            base_cfg.max_hodge_order,
            base_cfg.min_layer_q,
            base_cfg.normalization,
            base_cfg.alpha,
        )
        frozen_D = _candidate_distance_matrix(
            records, base_cfg, frozen_candidate, run / "candidate_matrices", runtime_rows,
            recompute=recompute_candidates, progress=progress,
        )
    np.save(run / "arrays" / "distance_matrix_HodgeFrozen_full.npy", frozen_D)
    pd.DataFrame(frozen_D, index=labels, columns=labels).to_csv(
        run / "tables" / "distance_matrix_HodgeFrozen_full.csv", encoding="utf-8-sig"
    )

    requested_baselines = [m for m in exp.get("methods", []) if m != PRIMARY_METHOD]
    baseline_methods = MethodManager.validate_methods(requested_baselines) if requested_baselines else []
    manager = MethodManager.from_config(base_cfg, payload.get("baselines", {}))
    baseline_matrices: dict[str, np.ndarray] = {}
    for method in baseline_methods:
        progress(f"[BASELINE] {method}")
        baseline_matrices[method] = _compute_or_load_baseline_matrix(
            method, records, labels, manager, run, runtime_rows,
            recompute=recompute_baselines, progress=progress,
        )

    inner_fold_rows: list[dict[str, Any]] = []
    inner_summary_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    inductive_metric_rows: list[dict[str, Any]] = []
    inductive_query_frames: list[pd.DataFrame] = []
    inductive_rank_frames: list[pd.DataFrame] = []
    square_metric_rows: list[dict[str, Any]] = []
    square_query_frames: list[pd.DataFrame] = []
    square_rank_frames: list[pd.DataFrame] = []
    square_domain_rows: list[dict[str, Any]] = []

    linkage_method = str(exp.get("linkage", "average"))
    mst_permutations = int(nested.get("mst_permutations", exp.get("mst_permutations", 5000)))
    eval_seed_base = int(exp.get("seed", 20260826))

    for oi, outer in enumerate(outer_splits, start=1):
        inner_seed = inner_seed_base + (outer.repeat - 1) * 1000 + outer.fold - 1
        inner_splits = make_cv_splits(outer.train_idx, domains, inner_folds, inner_seed)
        progress(
            f"[OUTER {oi}/{len(outer_splits)}] {outer.outer_id} "
            f"train={len(outer.train_idx)} test={len(outer.test_idx)} inner_cv={inner_folds}"
        )

        outer_fold_rows: list[dict[str, Any]] = []
        outer_summary_rows: list[dict[str, Any]] = []
        for candidate in candidates:
            frows, srow = _cv_score_candidate(
                candidate_matrices[candidate.candidate_id], candidate, inner_splits, domains, labels
            )
            for row in frows:
                row.update({
                    "outer_id": outer.outer_id,
                    "repeat": outer.repeat,
                    "outer_fold": outer.fold,
                    "outer_seed": outer.seed,
                    "inner_seed": inner_seed,
                })
            srow.update({
                "outer_id": outer.outer_id,
                "repeat": outer.repeat,
                "outer_fold": outer.fold,
                "outer_seed": outer.seed,
                "inner_seed": inner_seed,
            })
            outer_fold_rows.extend(frows)
            outer_summary_rows.append(srow)

        summary_df = pd.DataFrame(outer_summary_rows)
        best = select_best_candidate(summary_df, primary_metric)
        best_id = str(best["candidate_id"])
        best_candidate = next(c for c in candidates if c.candidate_id == best_id)
        best_cfg = candidate_config(base_cfg, best_candidate)
        selected_row = {
            "outer_id": outer.outer_id,
            "repeat": outer.repeat,
            "fold": outer.fold,
            "outer_seed": outer.seed,
            "inner_seed": inner_seed,
            "train_size": int(len(outer.train_idx)),
            "test_size": int(len(outer.test_idx)),
            **best_candidate.as_dict(),
            f"inner_mean_{primary_metric}": float(best[f"mean_{primary_metric}"]),
            f"inner_std_{primary_metric}": float(best[f"std_{primary_metric}"]),
            "inner_mean_mAP": float(best["mean_mAP"]),
            "inner_mean_auroc_same_domain": float(best["mean_auroc_same_domain"]),
        }
        selected_rows.append(selected_row)
        progress(f"[SELECTED {outer.outer_id}] {best_candidate.as_dict()}")

        inner_fold_rows.extend(outer_fold_rows)
        inner_summary_rows.extend(outer_summary_rows)

        learned_D = candidate_matrices[best_id]
        matrices: dict[str, np.ndarray] = {
            LEARNED_METHOD: learned_D,
            FROZEN_METHOD: frozen_D,
            **baseline_matrices,
        }
        outer_dir = run / "outer" / outer.outer_id
        outer_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "index": np.arange(len(df), dtype=int),
            "label": labels,
            "domain": domains,
            "partition": ["test" if i in set(outer.test_idx.tolist()) else "train" for i in range(len(df))],
        }).to_csv(outer_dir / "split.csv", index=False, encoding="utf-8-sig")
        write_json(outer_dir / "selected_hyperparameters.json", {
            **best_candidate.as_dict(),
            "primary_metric": primary_metric,
            "outer_id": outer.outer_id,
            "frozen_numerical_controls": {
                "solver": best_cfg.solver,
                "exact": best_cfg.exact.__dict__,
                "slq": best_cfg.slq.__dict__,
                "time_min": best_cfg.time_min,
                "time_max": best_cfg.time_max,
                "time_points": best_cfg.time_points,
            },
        })

        per_outer_inductive_rows = []
        per_outer_square_rows = []
        for method, D in matrices.items():
            met, qdf, rdf = cross_partition_metrics(
                D, outer.test_idx, outer.train_idx, domains, labels
            )
            meta = {
                "outer_id": outer.outer_id,
                "repeat": outer.repeat,
                "fold": outer.fold,
                "outer_seed": outer.seed,
                "inner_seed": inner_seed,
                "train_size": int(len(outer.train_idx)),
                "test_size": int(len(outer.test_idx)),
                "selected_candidate_id": best_id,
                "method": method,
            }
            inductive_metric_rows.append({**meta, **met})
            per_outer_inductive_rows.append({**meta, **met})
            for col, val in reversed(list(meta.items())):
                qdf.insert(0, col, val)
                rdf.insert(0, col, val)
            inductive_query_frames.append(qdf)
            inductive_rank_frames.append(rdf)

            square, sq_qdf, sq_rdf, clustering = _square_metrics(
                D, outer.test_idx, labels, domains, method, linkage_method,
                mst_permutations, eval_seed_base + oi - 1,
            )
            sq_meta = dict(meta)
            square_metric_rows.append({**sq_meta, **{k: v for k, v in square.items() if k != "method"}})
            per_outer_square_rows.append({**sq_meta, **{k: v for k, v in square.items() if k != "method"}})
            for col, val in reversed(list(sq_meta.items())):
                sq_qdf.insert(0, col, val)
                sq_rdf.insert(0, col, val)
            square_query_frames.append(sq_qdf)
            square_rank_frames.append(sq_rdf)
            for row in _domain_metric_rows(method, sq_qdf):
                square_domain_rows.append({
                    "outer_id": outer.outer_id,
                    "repeat": outer.repeat,
                    "fold": outer.fold,
                    "outer_seed": outer.seed,
                    "selected_candidate_id": best_id,
                    **row,
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
                index=[labels[i] for i in idx],
                columns=[labels[i] for i in idx],
            ).to_csv(outer_dir / f"test_distance_matrix_{method}.csv", encoding="utf-8-sig")

        pd.DataFrame(outer_fold_rows).to_csv(outer_dir / "inner_cv_fold_metrics.csv", index=False, encoding="utf-8-sig")
        summary_df.to_csv(outer_dir / "inner_cv_candidate_summary.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(per_outer_inductive_rows).to_csv(outer_dir / "inductive_metrics.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(per_outer_square_rows).to_csv(outer_dir / "square_test_metrics.csv", index=False, encoding="utf-8-sig")
        write_json(outer_dir / "OUTER_COMPLETE.json", {
            "status": "complete",
            "outer_id": outer.outer_id,
            "selected_candidate": best_candidate.as_dict(),
        })

        # Checkpoint aggregate tables after every outer evaluation so a long run
        # still leaves inspectable partial results if interrupted during later folds.
        pd.DataFrame(selected_rows).to_csv(run / "tables" / "selected_hyperparameters_by_outer_split.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(inductive_metric_rows).to_csv(run / "tables" / "nested_generalization_metrics_by_outer_split.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(square_metric_rows).to_csv(run / "tables" / "nested_square_test_metrics_by_outer_split.csv", index=False, encoding="utf-8-sig")

    inner_folds_df = pd.DataFrame(inner_fold_rows)
    inner_summary_df = pd.DataFrame(inner_summary_rows)
    selected_df = pd.DataFrame(selected_rows)
    inductive_df = pd.DataFrame(inductive_metric_rows)
    square_df = pd.DataFrame(square_metric_rows)
    square_domain_df = pd.DataFrame(square_domain_rows)
    inductive_qdf = pd.concat(inductive_query_frames, ignore_index=True)
    inductive_rdf = pd.concat(inductive_rank_frames, ignore_index=True)
    square_qdf = pd.concat(square_query_frames, ignore_index=True)
    square_rdf = pd.concat(square_rank_frames, ignore_index=True)

    inner_folds_df.to_csv(run / "tables" / "nested_inner_cv_fold_metrics.csv", index=False, encoding="utf-8-sig")
    inner_summary_df.to_csv(run / "tables" / "nested_inner_cv_candidate_summary.csv", index=False, encoding="utf-8-sig")
    selected_df.to_csv(run / "tables" / "selected_hyperparameters_by_outer_split.csv", index=False, encoding="utf-8-sig")
    inductive_df.to_csv(run / "tables" / "nested_generalization_metrics_by_outer_split.csv", index=False, encoding="utf-8-sig")
    inductive_qdf.to_csv(run / "tables" / "nested_generalization_per_query.csv", index=False, encoding="utf-8-sig")
    inductive_rdf.to_csv(run / "tables" / "nested_generalization_rankings.csv", index=False, encoding="utf-8-sig")
    square_df.to_csv(run / "tables" / "nested_square_test_metrics_by_outer_split.csv", index=False, encoding="utf-8-sig")
    square_qdf.to_csv(run / "tables" / "nested_square_test_dataset_metrics.csv", index=False, encoding="utf-8-sig")
    square_rdf.to_csv(run / "tables" / "nested_square_test_rankings.csv", index=False, encoding="utf-8-sig")
    square_domain_df.to_csv(run / "tables" / "nested_square_test_domain_metrics.csv", index=False, encoding="utf-8-sig")

    inductive_summary = _aggregate_metric_frame(inductive_df)
    square_summary = _aggregate_metric_frame(square_df)
    inductive_summary.to_csv(run / "tables" / "nested_generalization_metrics_summary.csv", index=False, encoding="utf-8-sig")
    square_summary.to_csv(run / "tables" / "nested_square_test_metrics_summary.csv", index=False, encoding="utf-8-sig")

    cand_freq, param_freq = _selection_frequency(selected_df)
    cand_freq.to_csv(run / "tables" / "nested_candidate_selection_frequency.csv", index=False, encoding="utf-8-sig")
    param_freq.to_csv(run / "tables" / "nested_parameter_selection_frequency.csv", index=False, encoding="utf-8-sig")

    if bool(learning.get("generate_plots", True)):
        from .real_learning_nested_plotting import (
            plot_candidate_selection_frequency,
            plot_nested_confusion,
            plot_nested_metric_summary,
            plot_outer_metric_distributions,
            plot_outer_test_coverage,
            plot_parameter_selection_frequency,
        )
        plot_nested_metric_summary(
            inductive_summary,
            run / "plots" / "nested_generalization_mean_std",
            title="Repeated nested CV: outer-test → outer-train generalization",
        )
        plot_outer_metric_distributions(
            inductive_df,
            run / "plots" / "nested_generalization_distributions",
        )
        plot_candidate_selection_frequency(
            cand_freq,
            run / "plots" / "candidate_selection_frequency",
        )
        plot_parameter_selection_frequency(
            param_freq,
            run / "plots" / "parameter_selection_frequency",
        )
        plot_outer_test_coverage(
            coverage,
            run / "plots" / "outer_test_coverage",
        )
        learned_queries = inductive_qdf[inductive_qdf["method"] == LEARNED_METHOD].copy()
        plot_nested_confusion(
            learned_queries,
            unique_domains,
            run / "plots" / "HodgeLearned_aggregate_outer_confusion",
        )

    pd.DataFrame(runtime_rows).to_csv(run / "tables" / "runtime_rows.csv", index=False, encoding="utf-8-sig")
    write_json(run / "run_manifest.json", {
        "architecture": "repeated nested supervised learning layer over frozen independent-q Hodge VNext core",
        "method_core_changed": False,
        "cache_schema": CACHE_SCHEMA,
        "benchmark_id": benchmark_id,
        "num_networks": len(df),
        "domains": unique_domains,
        "outer": {
            "strategy": strategy,
            "repeats": outer_repeats,
            "folds": outer_folds if strategy == "repeated_stratified_kfold" else None,
            "test_fraction": test_fraction if strategy == "repeated_stratified_holdout" else None,
            "seed": outer_seed,
            "num_outer_evaluations": len(outer_splits),
            "require_min_test_per_domain": require_min_test,
        },
        "inner_cv": {
            "folds": inner_folds,
            "seed_base": inner_seed_base,
            "primary_metric": primary_metric,
        },
        "methods_reported": [LEARNED_METHOD, FROZEN_METHOD, *baseline_methods],
        "primary_generalization_table": "tables/nested_generalization_metrics_summary.csv",
        "per_outer_generalization_table": "tables/nested_generalization_metrics_by_outer_split.csv",
        "square_test_compatibility_table": "tables/nested_square_test_metrics_summary.csv",
        "selection_frequency_table": "tables/nested_candidate_selection_frequency.csv",
        "scientific_boundaries": [
            "outer-test labels are never used in inner candidate selection",
            "domain labels are external annotations, not unique structural ground truth",
            "HodgeLearned is a supervised hyper-parameter-selected extension, distinct from the original unsupervised Hodge benchmark",
            "candidate distance matrices are label-free and are precomputed once only as an exact computational reuse optimization",
            "repeated outer evaluations are correlated resamples; reported q025/q975 are empirical resampling intervals, not iid confidence intervals",
            "solver/SLQ numerical controls are frozen and are not label-tuned",
        ],
        "effects": {
            "method_math_definition": "unchanged",
            "cache_identity": "unchanged staged cache schema; adds run-level candidate/full distance matrices only",
            "experiment_statistics": "new repeated nested resampling protocol",
            "random_graph_seed": "not applicable / unchanged",
            "baseline_semantics": "unchanged",
            "existing_result_comparability": "original fixed-split and unsupervised runs are unchanged; nested learned scores are a separate supervised stability experiment",
        },
    })
    write_json(complete, {
        "status": "complete",
        "run_dir": str(run),
        "method_core_changed": False,
        "num_outer_evaluations": len(outer_splits),
        "outer_strategy": strategy,
    })
    progress(f"[COMPLETE] {run}")
    return run
