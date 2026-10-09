from __future__ import annotations

"""Supervised hyper-parameter learning for the real-domain Hodge experiment.

This module is deliberately outside ``hodge_vnext/``.  It does not alter the
mathematical core.  Domain labels are used only to select an existing Hodge
configuration on the training partition; the frozen Hodge distance is still
computed by :func:`hodge_vnext.distance.profile_distance`.

Protocol
--------
1. deterministic stratified 80/20 outer split;
2. stratified K-fold CV *inside the 80% training partition only*;
3. candidate selection from scientific Hodge parameters (S/Q/qmin/
   normalization/alpha); solver and numerical-accuracy controls are frozen;
4. one final hold-out evaluation with test queries against the train gallery;
5. compatibility reports on the square test-only distance matrices using the
   same metrics/plots as ``validation.real_domains``;
6. optional full-48 reports are explicitly descriptive, not hold-out scores.
"""

from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence
import json
import math
import shutil
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from hodge_vnext.cache import CACHE_SCHEMA
from hodge_vnext.config import MethodConfig
from hodge_vnext.distance import profile_distance
from hodge_vnext.graph_io import GraphRecord, create_real_record
from hodge_vnext.profile import GraphProfile, compute_graph_profile
from hodge_vnext.utils import safe_token, stable_hash, write_json

from .config import method_from_payload, resolve_path
from .methods import MethodManager, PRIMARY_METHOD
from .real_domains import _prepare_manifest
from .real_metrics import global_metrics, hierarchical_clustering, retrieval_metrics, validate_distance_matrix


LEARNED_METHOD = "HodgeLearned"
FROZEN_METHOD = "HodgeFrozen"


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    max_clique_size: int | None
    max_hodge_order: int | None
    min_layer_q: int
    normalization: str
    alpha: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "max_clique_size": self.max_clique_size,
            "max_hodge_order": self.max_hodge_order,
            "min_layer_q": self.min_layer_q,
            "normalization": self.normalization,
            "alpha": self.alpha,
        }


@dataclass(frozen=True)
class OuterSplit:
    train_idx: np.ndarray
    test_idx: np.ndarray


def _progress(msg: str) -> None:
    print(msg, flush=True)


def _as_cutoff(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"all", "none", "null"}:
        return None
    return int(value)


def _list_or_inherit(search: dict[str, Any], key: str, inherited: Any) -> list[Any]:
    values = search.get(key)
    if values is None:
        return [inherited]
    if not isinstance(values, list) or not values:
        raise ValueError(f"learning.search_space.{key} must be a non-empty list")
    return values


def enumerate_candidates(base_cfg: MethodConfig, learning: dict[str, Any]) -> list[Candidate]:
    """Enumerate unique scientific Hodge candidates.

    ``solver``, time grid, exact tolerances and SLQ controls are intentionally
    absent from the search space: they are numerical controls, not label-tuned
    scientific parameters.
    """
    search = dict(learning.get("search_space", {}))
    Ss = [_as_cutoff(x) for x in _list_or_inherit(search, "max_clique_size", base_cfg.max_clique_size)]
    Qs = [_as_cutoff(x) for x in _list_or_inherit(search, "max_hodge_order", base_cfg.max_hodge_order)]
    qmins = [int(x) for x in _list_or_inherit(search, "min_layer_q", base_cfg.min_layer_q)]
    norms = [str(x) for x in _list_or_inherit(search, "normalization", base_cfg.normalization)]
    alphas = [float(x) for x in _list_or_inherit(search, "alpha", base_cfg.alpha)]

    unique: dict[str, Candidate] = {}
    for S, Q, qmin, norm, alpha in product(Ss, Qs, qmins, norms, alphas):
        # Canonicalize redundant Q values.  For a finite S, retained q <= S-1,
        # therefore Q >= S-1 is mathematically identical to Q=all.
        q_eff = Q
        if S is not None and Q is not None and Q >= max(S - 1, 0):
            q_eff = None
        payload = {
            "max_clique_size": S,
            "max_hodge_order": q_eff,
            "min_layer_q": qmin,
            "normalization": norm,
            "alpha": alpha,
        }
        cfg = replace(
            base_cfg,
            max_clique_size=S,
            max_hodge_order=q_eff,
            min_layer_q=qmin,
            normalization=norm,
            alpha=alpha,
        )
        cfg.validate()
        cid = stable_hash(payload, length=12)
        unique[cid] = Candidate(cid, S, q_eff, qmin, norm, alpha)
    return sorted(unique.values(), key=lambda c: c.candidate_id)


def candidate_config(base_cfg: MethodConfig, candidate: Candidate) -> MethodConfig:
    cfg = replace(
        base_cfg,
        max_clique_size=candidate.max_clique_size,
        max_hodge_order=candidate.max_hodge_order,
        min_layer_q=candidate.min_layer_q,
        normalization=candidate.normalization,
        alpha=candidate.alpha,
    )
    cfg.validate()
    return cfg


def make_outer_split(domains: Sequence[str], test_fraction: float, seed: int) -> OuterSplit:
    labels = np.asarray(domains, dtype=str)
    if not (0.0 < float(test_fraction) < 1.0):
        raise ValueError("learning.test_fraction must be between 0 and 1")
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=float(test_fraction), random_state=int(seed))
    train_idx, test_idx = next(splitter.split(np.zeros(len(labels)), labels))
    return OuterSplit(np.sort(train_idx.astype(int)), np.sort(test_idx.astype(int)))


def make_cv_splits(train_idx: np.ndarray, domains: Sequence[str], folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    train_idx = np.asarray(train_idx, dtype=int)
    y = np.asarray(domains, dtype=str)[train_idx]
    if int(folds) < 2:
        raise ValueError("learning.cv_folds must be >=2")
    counts = pd.Series(y).value_counts()
    if int(folds) > int(counts.min()):
        raise ValueError(f"cv_folds={folds} exceeds smallest training-domain count={int(counts.min())}")
    skf = StratifiedKFold(n_splits=int(folds), shuffle=True, random_state=int(seed))
    out = []
    for local_tr, local_va in skf.split(np.zeros(len(train_idx)), y):
        out.append((train_idx[local_tr], train_idx[local_va]))
    return out


def _macro_f1(true: Sequence[str], pred: Sequence[str]) -> float:
    y = np.asarray(true, dtype=str)
    p = np.asarray(pred, dtype=str)
    scores = []
    for label in sorted(set(y.tolist())):
        tp = int(np.sum((y == label) & (p == label)))
        fp = int(np.sum((y != label) & (p == label)))
        fn = int(np.sum((y == label) & (p != label)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(scores)) if scores else float("nan")


def _rectangular_auroc(scores: list[tuple[float, int]]) -> float:
    pos = [d for d, y in scores if y == 1]
    neg = [d for d, y in scores if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p < n else (0.5 if p == n else 0.0)
    return float(wins / (len(pos) * len(neg)))


def cross_partition_metrics(
    D: np.ndarray,
    query_idx: Sequence[int],
    gallery_idx: Sequence[int],
    labels: Sequence[str],
    ids: Sequence[str] | None = None,
    ks: Sequence[int] = (1, 3, 5, 7),
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    """Inductive retrieval/classification metrics: query set -> fixed gallery.

    No query-query distance is used.  This is the primary generalization metric
    for the 80/20 experiment; it is stricter than evaluating the square test set
    transductively.
    """
    D = validate_distance_matrix(D)
    qidx = np.asarray(query_idx, dtype=int)
    gidx = np.asarray(gallery_idx, dtype=int)
    y = np.asarray(labels, dtype=str)
    names = np.asarray(ids if ids is not None else [str(i) for i in range(len(D))], dtype=str)
    if set(qidx.tolist()) & set(gidx.tolist()):
        raise ValueError("query and gallery partitions must be disjoint")
    if len(qidx) == 0 or len(gidx) == 0:
        raise ValueError("query and gallery partitions must be non-empty")

    rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    pair_scores: list[tuple[float, int]] = []
    pred: list[str] = []

    for qi in qidx:
        ranked = sorted(gidx.tolist(), key=lambda gj: (float(D[qi, gj]), names[gj]))
        rel = np.asarray([int(y[gj] == y[qi]) for gj in ranked], dtype=int)
        R = int(rel.sum())
        if R == 0:
            raise ValueError(f"query {names[qi]} has no same-domain example in the training gallery")
        cum = np.cumsum(rel)
        precision_all = cum / np.arange(1, len(rel) + 1)
        ap = float(np.dot(precision_all, rel) / R)
        k7 = min(7, len(rel))
        discounts = 1.0 / np.log2(np.arange(1, k7 + 1) + 1)
        idcg = float(discounts[: min(R, k7)].sum())
        ndcg = float(np.dot(rel[:k7], discounts) / idcg) if idcg > 0 else float("nan")
        nn = ranked[0]
        pred.append(y[nn])
        same_d = [float(D[qi, gj]) for gj in gidx if y[gj] == y[qi]]
        other_d = [float(D[qi, gj]) for gj in gidx if y[gj] != y[qi]]
        first_rel = int(np.flatnonzero(rel == 1)[0] + 1)
        row: dict[str, Any] = {
            "query": names[qi],
            "domain": y[qi],
            "nearest_neighbor": names[nn],
            "nearest_neighbor_domain": y[nn],
            "nearest_neighbor_distance": float(D[qi, nn]),
            "nearest_neighbor_correct": int(y[nn] == y[qi]),
            "relevant_total": R,
            "first_relevant_rank": first_rel,
            "average_precision": ap,
            "ndcg_at_7": ndcg,
            "nearest_same_domain_distance": float(np.min(same_d)),
            "mean_same_domain_distance": float(np.mean(same_d)),
            "mean_other_domain_distance": float(np.mean(other_d)),
            "query_between_within_ratio": float(np.mean(other_d) / np.mean(same_d)) if np.mean(same_d) > 0 else float("inf"),
        }
        for k in ks:
            ek = min(int(k), len(rel))
            hits = int(cum[ek - 1])
            row[f"precision_at_{k}"] = hits / ek
            row[f"recall_at_{k}"] = hits / R
        rows.append(row)
        for rank, gj in enumerate(ranked, start=1):
            relevant = int(y[gj] == y[qi])
            d = float(D[qi, gj])
            pair_scores.append((d, relevant))
            rank_rows.append({
                "query": names[qi], "query_domain": y[qi], "rank": rank,
                "candidate": names[gj], "candidate_domain": y[gj],
                "distance": d, "relevant": relevant,
            })

    qdf = pd.DataFrame(rows)
    summary: dict[str, float] = {
        "auroc_same_domain": _rectangular_auroc(pair_scores),
        "nn1_accuracy": float(qdf["nearest_neighbor_correct"].mean()),
        "nn1_macro_f1": _macro_f1(y[qidx], pred),
        "mAP": float(qdf["average_precision"].mean()),
        "ndcg_at_7": float(qdf["ndcg_at_7"].mean()),
        "within_domain_mean_distance": float(qdf["mean_same_domain_distance"].mean()),
        "between_domain_mean_distance": float(qdf["mean_other_domain_distance"].mean()),
        "between_within_ratio": float(qdf["mean_other_domain_distance"].mean() / qdf["mean_same_domain_distance"].mean()) if float(qdf["mean_same_domain_distance"].mean()) > 0 else float("inf"),
    }
    for k in ks:
        summary[f"precision_at_{k}"] = float(qdf[f"precision_at_{k}"].mean())
        summary[f"recall_at_{k}"] = float(qdf[f"recall_at_{k}"].mean())
    return summary, qdf, pd.DataFrame(rank_rows)


def _candidate_distance_matrix(
    records: Sequence[GraphRecord],
    base_cfg: MethodConfig,
    candidate: Candidate,
    cache_dir: Path,
    runtime_rows: list[dict[str, Any]],
    *,
    recompute: bool,
    progress: Callable[[str], None],
) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = cache_dir / f"candidate_{candidate.candidate_id}.npy"
    if matrix_path.exists() and not recompute:
        D = validate_distance_matrix(np.load(matrix_path))
        if D.shape == (len(records), len(records)):
            progress(f"[SEARCH CACHE] {candidate.candidate_id}: distance matrix hit")
            return D
    cfg = candidate_config(base_cfg, candidate)
    profiles: list[GraphProfile] = []
    for i, rec in enumerate(records):
        t0 = time.perf_counter()
        prof = compute_graph_profile(rec, cfg, progress=None)
        elapsed = time.perf_counter() - t0
        profiles.append(prof)
        runtime_rows.append({
            "stage": "candidate_profile", "candidate_id": candidate.candidate_id,
            "label": rec.source.get("label", str(i)), "elapsed_sec": float(elapsed),
            "solver": cfg.solver,
        })
    n = len(records)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = profile_distance(profiles[i], profiles[j], cfg).total_distance
            D[i, j] = D[j, i] = float(d)
    np.save(matrix_path, D)
    progress(f"[SEARCH CACHE] {candidate.candidate_id}: saved")
    return D


def _cv_score_candidate(
    D: np.ndarray,
    candidate: Candidate,
    cv_splits: Sequence[tuple[np.ndarray, np.ndarray]],
    labels: Sequence[str],
    ids: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fold_rows = []
    for fold, (tr, va) in enumerate(cv_splits, start=1):
        metrics, _, _ = cross_partition_metrics(D, va, tr, labels, ids)
        fold_rows.append({"candidate_id": candidate.candidate_id, "fold": fold, **candidate.as_dict(), **metrics})
    frame = pd.DataFrame(fold_rows)
    summary: dict[str, Any] = candidate.as_dict()
    metric_cols = [c for c in frame.columns if c not in {"candidate_id", "fold", "max_clique_size", "max_hodge_order", "min_layer_q", "normalization", "alpha"}]
    for col in metric_cols:
        vals = frame[col].to_numpy(dtype=float)
        summary[f"mean_{col}"] = float(np.nanmean(vals))
        summary[f"std_{col}"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
    return fold_rows, summary


def select_best_candidate(summary: pd.DataFrame, primary_metric: str) -> pd.Series:
    primary = f"mean_{primary_metric}"
    required = [primary, "mean_mAP", "mean_auroc_same_domain", f"std_{primary_metric}", "candidate_id"]
    missing = [c for c in required if c not in summary.columns]
    if missing:
        raise ValueError(f"candidate summary missing selection columns: {missing}")
    ordered = summary.sort_values(
        by=[primary, "mean_mAP", "mean_auroc_same_domain", f"std_{primary_metric}", "candidate_id"],
        ascending=[False, False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return ordered.iloc[0]


def _split_table(df: pd.DataFrame, split: OuterSplit) -> pd.DataFrame:
    part = np.full(len(df), "train", dtype=object)
    part[split.test_idx] = "test"
    out = df[["label", "domain", "path", "nodes_path"]].copy()
    out.insert(0, "index", np.arange(len(df), dtype=int))
    out["partition"] = part
    return out


def _square_metrics(
    D: np.ndarray,
    indices: Sequence[int],
    labels: Sequence[str],
    domains: Sequence[str],
    method: str,
    linkage_method: str,
    mst_permutations: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, Any]:
    idx = np.asarray(indices, dtype=int)
    sub = D[np.ix_(idx, idx)]
    sub_labels = [labels[i] for i in idx]
    sub_domains = [domains[i] for i in idx]
    clustering = hierarchical_clustering(sub, len(set(sub_domains)), method=linkage_method)
    retrieval, per_query, ranks = retrieval_metrics(sub, sub_labels, sub_domains)
    glob = global_metrics(sub, sub_domains, clustering.cluster_labels, mst_permutations, seed)
    return {"method": method, **retrieval, **glob}, per_query, ranks, clustering


def _domain_metric_rows(method: str, per_query: pd.DataFrame) -> list[dict[str, Any]]:
    aggregate_cols = [
        "nearest_neighbor_correct", "average_precision", "ndcg_at_7",
        "precision_at_1", "precision_at_3", "precision_at_5", "precision_at_7",
        "recall_at_1", "recall_at_3", "recall_at_5", "recall_at_7",
        "first_relevant_rank", "nearest_same_domain_distance",
        "mean_same_domain_distance", "mean_other_domain_distance", "query_between_within_ratio",
    ]
    rows = []
    for domain, grp in per_query.groupby("domain", sort=False):
        row: dict[str, Any] = {"method": method, "domain": str(domain), "num_queries": int(len(grp))}
        for col in aggregate_cols:
            if col in grp.columns:
                row[col if col != "nearest_neighbor_correct" else "nn1_accuracy"] = float(grp[col].mean())
        if "average_precision" in row:
            row["mAP"] = row.pop("average_precision")
        rows.append(row)
    return rows


def _plot_square_method(
    run: Path,
    scope: str,
    method: str,
    D: np.ndarray,
    indices: Sequence[int],
    labels: Sequence[str],
    domains: Sequence[str],
    clustering: Any,
    plot_options: dict[str, Any],
    seed: int,
    *,
    skip_existing: bool = False,
) -> None:
    from .real_plotting import plot_cluster_heatmap, plot_domain_block_heatmap, plot_mst, plot_within_domain_panels, plot_mds

    idx = np.asarray(indices, dtype=int)
    sub = D[np.ix_(idx, idx)]
    labs = [labels[i] for i in idx]
    doms = [domains[i] for i in idx]
    pdir = run / "plots" / scope / method
    expected = [
        pdir / "global_cluster_heatmap.pdf",
        pdir / "global_domain_block_heatmap.pdf",
        pdir / "global_mst.pdf",
        pdir / "four_domain_heatmap_mst.pdf",
        pdir / "global_mds_2d.pdf",
    ]
    if skip_existing and all(path.exists() for path in expected):
        return
    title_scope = "hold-out test" if scope == "test_holdout" else "full descriptive"
    plot_cluster_heatmap(sub, labs, clustering, pdir / "global_cluster_heatmap", f"{method}: {title_scope}", plot_options=plot_options)
    plot_domain_block_heatmap(sub, labs, doms, pdir / "global_domain_block_heatmap", f"{method}: {title_scope}", plot_options=plot_options)
    plot_mst(sub, labs, doms, pdir / "global_mst", f"{method}: {title_scope}", seed=seed)
    plot_within_domain_panels(sub, labs, doms, pdir / "four_domain_heatmap_mst", f"{method}: within-domain structure", seed=seed, plot_options=plot_options)
    plot_mds(sub, labs, doms, pdir / "global_mds_2d", f"{method}: {title_scope}")


def _clear_report_outputs(run: Path) -> None:
    for name in ("tables", "plots"):
        p = run / name
        if p.exists():
            shutil.rmtree(p)
    for name in ("RUN_COMPLETE.json", "run_manifest.json"):
        p = run / name
        if p.exists():
            p.unlink()


def _run_stem(n: int, benchmark_id: str, learning: dict[str, Any], candidates: Sequence[Candidate]) -> str:
    payload = {
        "test_fraction": learning.get("test_fraction", 0.2),
        "split_seed": learning.get("split_seed", 20260930),
        "cv_folds": learning.get("cv_folds", 5),
        "cv_seed": learning.get("cv_seed", 20260930),
        "primary_metric": learning.get("primary_metric", "nn1_macro_f1"),
        "candidates": [c.as_dict() for c in candidates],
    }
    return f"bench-{safe_token(benchmark_id)}_n{n}_learning-{stable_hash(payload, 12)}"


def run_real_learning(
    payload: dict[str, Any],
    config_path: Path,
    *,
    force: bool = False,
    recompute_candidates: bool = False,
    progress: Callable[[str], None] | None = None,
) -> Path:
    progress = progress or _progress
    base_cfg = method_from_payload(payload, config_path)
    exp = dict(payload["real_validation"])
    learning = dict(payload.get("learning", {}))
    df = _prepare_manifest(payload, config_path)
    if len(df) < 10:
        raise ValueError("real learning requires enough networks for a five-domain train/test split")
    labels = df["label"].astype(str).tolist()
    domains = df["domain"].astype(str).tolist()
    unique_domains = list(dict.fromkeys(domains))
    if len(unique_domains) < 2:
        raise ValueError("real learning requires at least two domains")

    candidates = enumerate_candidates(base_cfg, learning)
    if not candidates:
        raise ValueError("learning search space produced no candidates")

    benchmark_id = str(exp.get("benchmark_id", "real-learning")).strip() or "real-learning"
    results_root = resolve_path(learning.get("results_root", "results/validation/real_learning"), config_path)
    run = results_root / _run_stem(len(df), benchmark_id, learning, candidates)
    complete = run / "RUN_COMPLETE.json"
    if complete.exists() and not force:
        progress(f"[SKIP] completed learning run exists: {run}")
        return run
    run.mkdir(parents=True, exist_ok=True)
    if force:
        _clear_report_outputs(run)
    for d in ("tables", "plots", "arrays", "candidate_matrices"):
        (run / d).mkdir(parents=True, exist_ok=True)

    split = make_outer_split(domains, float(learning.get("test_fraction", 0.2)), int(learning.get("split_seed", 20260930)))
    cv_splits = make_cv_splits(split.train_idx, domains, int(learning.get("cv_folds", 5)), int(learning.get("cv_seed", 20260930)))
    split_df = _split_table(df, split)
    split_df.to_csv(run / "tables" / "outer_split.csv", index=False, encoding="utf-8-sig")
    split_counts = split_df.groupby(["partition", "domain"], sort=False).size().rename("count").reset_index()
    split_counts.to_csv(run / "tables" / "outer_split_domain_counts.csv", index=False, encoding="utf-8-sig")

    effective = json.loads(json.dumps(payload))
    effective["cache_schema"] = CACHE_SCHEMA
    effective["selected_networks"] = split_df.to_dict("records")
    effective["learning_protocol"] = {
        "outer_split": "deterministic stratified hold-out; labels used for partitioning only",
        "candidate_selection": "stratified K-fold CV inside training partition only",
        "primary_generalization": "test-query to training-gallery metrics",
        "square_test_reports": "compatibility diagnostics using existing real-domain metric/plot grammar",
        "full_dataset_reports": "descriptive only; not hold-out generalization",
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
    fold_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    candidate_matrices: dict[str, np.ndarray] = {}
    progress(f"[SEARCH] candidates={len(candidates)} train={len(split.train_idx)} test={len(split.test_idx)} cv={len(cv_splits)}")
    for ci, candidate in enumerate(candidates, start=1):
        progress(f"[CANDIDATE] {ci}/{len(candidates)} {candidate.as_dict()}")
        D = _candidate_distance_matrix(
            records, base_cfg, candidate, run / "candidate_matrices", runtime_rows,
            recompute=recompute_candidates, progress=progress,
        )
        candidate_matrices[candidate.candidate_id] = D
        frows, srow = _cv_score_candidate(D, candidate, cv_splits, domains, labels)
        fold_rows.extend(frows)
        summary_rows.append(srow)

    folds_df = pd.DataFrame(fold_rows)
    summary_df = pd.DataFrame(summary_rows)
    folds_df.to_csv(run / "tables" / "cv_fold_metrics.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(run / "tables" / "cv_candidate_summary.csv", index=False, encoding="utf-8-sig")
    primary_metric = str(learning.get("primary_metric", "nn1_macro_f1"))
    best = select_best_candidate(summary_df, primary_metric)
    best_id = str(best["candidate_id"])
    best_candidate = next(c for c in candidates if c.candidate_id == best_id)
    best_cfg = candidate_config(base_cfg, best_candidate)
    write_json(run / "selected_hyperparameters.json", {
        **best_candidate.as_dict(),
        "primary_metric": primary_metric,
        "selection_rule": [
            f"maximize mean_{primary_metric}",
            "then maximize mean_mAP",
            "then maximize mean_auroc_same_domain",
            f"then minimize std_{primary_metric}",
            "then candidate_id lexical order for deterministic final tie",
        ],
        "frozen_numerical_controls": {
            "solver": best_cfg.solver,
            "exact": best_cfg.exact.__dict__,
            "slq": best_cfg.slq.__dict__,
            "time_min": best_cfg.time_min,
            "time_max": best_cfg.time_max,
            "time_points": best_cfg.time_points,
        },
    })
    progress(f"[SELECTED] {best_candidate.as_dict()}")

    learned_D = candidate_matrices[best_id]
    np.save(run / "arrays" / "distance_matrix_HodgeLearned_full.npy", learned_D)
    pd.DataFrame(learned_D, index=labels, columns=labels).to_csv(run / "tables" / "distance_matrix_HodgeLearned_full.csv", encoding="utf-8-sig")

    # Frozen Hodge comparator: exactly the method block from the input config.
    frozen_payload = {
        "max_clique_size": base_cfg.max_clique_size,
        "max_hodge_order": base_cfg.max_hodge_order,
        "min_layer_q": base_cfg.min_layer_q,
        "normalization": base_cfg.normalization,
        "alpha": base_cfg.alpha,
    }
    frozen_candidate = Candidate("frozen", **frozen_payload)
    frozen_D = _candidate_distance_matrix(
        records, base_cfg, frozen_candidate, run / "candidate_matrices", runtime_rows,
        recompute=recompute_candidates, progress=progress,
    )
    np.save(run / "arrays" / "distance_matrix_HodgeFrozen_full.npy", frozen_D)
    pd.DataFrame(frozen_D, index=labels, columns=labels).to_csv(run / "tables" / "distance_matrix_HodgeFrozen_full.csv", encoding="utf-8-sig")

    # Classical baselines use the unmodified existing baseline implementations.
    requested_baselines = [m for m in exp.get("methods", []) if m != PRIMARY_METHOD]
    baseline_methods = MethodManager.validate_methods(requested_baselines) if requested_baselines else []
    manager = MethodManager.from_config(base_cfg, payload.get("baselines", {}))
    matrices: dict[str, np.ndarray] = {LEARNED_METHOD: learned_D, FROZEN_METHOD: frozen_D}
    for method in baseline_methods:
        progress(f"[BASELINE] {method}")
        reps = []
        for i, rec in enumerate(records):
            t0 = time.perf_counter()
            rep, elapsed, hit = manager.representation(method, rec)
            reps.append(rep)
            runtime_rows.append({
                "stage": "baseline_representation", "method": method, "label": labels[i],
                "elapsed_sec": float(elapsed), "wall_sec": float(time.perf_counter() - t0), "cache_hit": bool(hit),
            })
        D = np.zeros((len(records), len(records)), dtype=float)
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                d = manager.distance(method, reps[i], reps[j], records[i].graph.number_of_nodes(), records[j].graph.number_of_nodes())
                D[i, j] = D[j, i] = float(d)
        matrices[method] = D
        np.save(run / "arrays" / f"distance_matrix_{method}_full.npy", D)
        pd.DataFrame(D, index=labels, columns=labels).to_csv(run / "tables" / f"distance_matrix_{method}_full.csv", encoding="utf-8-sig")

    # Primary inductive hold-out evaluation: test queries -> train gallery.
    inductive_rows = []
    inductive_query_frames = []
    inductive_rank_frames = []
    for method, D in matrices.items():
        met, qdf, rdf = cross_partition_metrics(D, split.test_idx, split.train_idx, domains, labels)
        inductive_rows.append({"method": method, **met})
        qdf.insert(0, "method", method)
        rdf.insert(0, "method", method)
        inductive_query_frames.append(qdf)
        inductive_rank_frames.append(rdf)
    inductive = pd.DataFrame(inductive_rows)
    inductive.to_csv(run / "tables" / "inductive_test_to_train_metrics_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(inductive_query_frames, ignore_index=True).to_csv(run / "tables" / "inductive_test_to_train_per_query.csv", index=False, encoding="utf-8-sig")
    pd.concat(inductive_rank_frames, ignore_index=True).to_csv(run / "tables" / "inductive_test_to_train_rankings.csv", index=False, encoding="utf-8-sig")

    linkage_method = str(exp.get("linkage", "average"))
    mst_permutations = int(exp.get("mst_permutations", 5000))
    eval_seed = int(exp.get("seed", 20260826))
    plot_options = dict(exp.get("plotting", {}))
    plot_options.update(payload.get("plotting", {}))
    generate_plots = bool(learning.get("generate_plots", True))

    # Compatibility evaluation on square hold-out test matrix.
    test_metric_rows = []
    test_domain_rows = []
    test_per_query_frames = []
    test_clusterings: dict[str, Any] = {}
    for method, D in matrices.items():
        summary, per_query, ranks, clustering = _square_metrics(
            D, split.test_idx, labels, domains, method, linkage_method, mst_permutations, eval_seed,
        )
        test_clusterings[method] = clustering
        test_metric_rows.append(summary)
        per_query.insert(0, "method", method)
        ranks.insert(0, "method", method)
        test_per_query_frames.append(per_query)
        test_domain_rows.extend(_domain_metric_rows(method, per_query))
        idx = np.asarray(split.test_idx, dtype=int)
        sub_labels = [labels[i] for i in idx]
        pd.DataFrame({
            "label": sub_labels,
            "domain": [domains[i] for i in idx],
            "cluster": clustering.cluster_labels,
            "leaf_order_position": np.argsort(clustering.leaf_order),
        }).to_csv(run / "tables" / f"test_cluster_assignments_{method}.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(D[np.ix_(idx, idx)], index=sub_labels, columns=sub_labels).to_csv(run / "tables" / f"test_distance_matrix_{method}.csv", encoding="utf-8-sig")

    test_metrics = pd.DataFrame(test_metric_rows)
    test_domains = pd.DataFrame(test_domain_rows)
    test_metrics.to_csv(run / "tables" / "test_holdout_metrics_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(test_per_query_frames, ignore_index=True).to_csv(run / "tables" / "test_holdout_dataset_metrics_all_methods.csv", index=False, encoding="utf-8-sig")
    test_domains.to_csv(run / "tables" / "test_holdout_domain_metrics_all_methods.csv", index=False, encoding="utf-8-sig")

    if generate_plots:
        from .real_plotting import plot_core_metrics_comparison, plot_domain_metrics_comparison, plot_retrieval_summary
        from .real_learning_plotting import plot_cv_search, plot_inductive_metrics, plot_split_counts, plot_inductive_confusion

        for method, D in matrices.items():
            _plot_square_method(
                run, "test_holdout", method, D, split.test_idx, labels, domains,
                test_clusterings[method], plot_options, eval_seed, skip_existing=not force,
            )
        plot_core_metrics_comparison(test_metrics, run / "plots" / "test_holdout" / "core_metrics_all_methods", plot_options=plot_options)
        if not test_domains.empty:
            plot_domain_metrics_comparison(test_domains, run / "plots" / "test_holdout" / "core_metrics_by_domain", plot_options=plot_options)
        hrow = test_metrics[test_metrics.method == LEARNED_METHOD]
        if not hrow.empty:
            plot_retrieval_summary(hrow, run / "plots" / "test_holdout" / LEARNED_METHOD / "retrieval_summary")
        plot_cv_search(summary_df, primary_metric, best_id, run / "plots" / "learning" / "cv_search")
        plot_inductive_metrics(inductive, run / "plots" / "learning" / "inductive_test_to_train_core_metrics")
        plot_split_counts(split_counts, run / "plots" / "learning" / "outer_split_domain_counts")
        learned_qdf = inductive_query_frames[0]
        plot_inductive_confusion(learned_qdf, unique_domains, run / "plots" / "learning" / "HodgeLearned_test_to_train_confusion")

    # Full-dataset compatibility report is optional and explicitly descriptive.
    full_descriptive = bool(learning.get("full_dataset_descriptive", True))
    if full_descriptive:
        full_metric_rows = []
        full_domain_rows = []
        full_clusterings: dict[str, Any] = {}
        all_idx = np.arange(len(df), dtype=int)
        for method, D in matrices.items():
            summary, per_query, ranks, clustering = _square_metrics(
                D, all_idx, labels, domains, method, linkage_method, mst_permutations, eval_seed,
            )
            full_clusterings[method] = clustering
            summary["interpretation"] = "descriptive_full_dataset_not_holdout_generalization"
            full_metric_rows.append(summary)
            full_domain_rows.extend(_domain_metric_rows(method, per_query))
        full_metrics = pd.DataFrame(full_metric_rows)
        full_domains = pd.DataFrame(full_domain_rows)
        full_metrics.to_csv(run / "tables" / "full_descriptive_metrics_summary.csv", index=False, encoding="utf-8-sig")
        full_domains.to_csv(run / "tables" / "full_descriptive_domain_metrics_all_methods.csv", index=False, encoding="utf-8-sig")
        if generate_plots:
            for method, D in matrices.items():
                _plot_square_method(
                    run, "full_descriptive", method, D, all_idx, labels, domains,
                    full_clusterings[method], plot_options, eval_seed, skip_existing=not force,
                )
            plot_core_metrics_comparison(full_metrics, run / "plots" / "full_descriptive" / "core_metrics_all_methods", plot_options=plot_options)
            if not full_domains.empty:
                plot_domain_metrics_comparison(full_domains, run / "plots" / "full_descriptive" / "core_metrics_by_domain", plot_options=plot_options)

    pd.DataFrame(runtime_rows).to_csv(run / "tables" / "runtime_rows.csv", index=False, encoding="utf-8-sig")
    write_json(run / "run_manifest.json", {
        "architecture": "independent supervised learning layer over frozen Hodge VNext core",
        "method_core_changed": False,
        "cache_schema": CACHE_SCHEMA,
        "benchmark_id": benchmark_id,
        "num_networks": len(df),
        "domains": unique_domains,
        "outer_split": {"train": int(len(split.train_idx)), "test": int(len(split.test_idx)), "test_fraction": float(learning.get("test_fraction", 0.2)), "seed": int(learning.get("split_seed", 20260930))},
        "cv": {"folds": int(learning.get("cv_folds", 5)), "seed": int(learning.get("cv_seed", 20260930)), "primary_metric": primary_metric},
        "selected_candidate": best_candidate.as_dict(),
        "methods_reported": list(matrices.keys()),
        "primary_generalization_table": "tables/inductive_test_to_train_metrics_summary.csv",
        "compatibility_test_table": "tables/test_holdout_metrics_summary.csv",
        "full_dataset_table": "tables/full_descriptive_metrics_summary.csv" if full_descriptive else None,
        "scientific_boundaries": [
            "domain labels are external annotations, not unique structural ground truth",
            "HodgeLearned is a supervised hyper-parameter-selected extension and must not be presented as the original unsupervised Hodge method",
            "candidate selection never reads hold-out test labels",
            "full-dataset metrics are descriptive and not hold-out generalization estimates",
            "solver/SLQ numerical controls are frozen and are not label-tuned",
        ],
        "effects": {
            "method_math_definition": "unchanged core; selected among existing parameterized definitions",
            "cache_identity": "unchanged staged cache schema; candidate matrix cache is new run-level artifact only",
            "experiment_statistics": "new supervised stratified hold-out + train-only CV protocol",
            "random_graph_seed": "not applicable / unchanged",
            "baseline_semantics": "unchanged",
            "existing_result_comparability": "original runs unchanged; learned scores are a distinct supervised experiment and should not be merged with unsupervised benchmark claims",
        },
    })
    write_json(complete, {"status": "complete", "run_dir": str(run), "method_core_changed": False, "selected_candidate": best_candidate.as_dict()})
    progress(f"[COMPLETE] {run}")
    return run
