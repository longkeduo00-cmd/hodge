from __future__ import annotations

"""Matched-density ER/WS/RGG generative-mechanism discrimination.

Scientific contract
-------------------
* Six fine mechanism states: ER, RGG, WS(beta=1e-3,1e-2,1e-1,1).
* Shared target expected mean-degree coordinate kappa.
* ER uses p=kappa/(N-1).
* WS uses degree=kappa (therefore kappa must be an even integer).
* 2-D Euclidean unit-square RGG solves the finite-N expected-degree equation.
* Every (state,kappa,repeat) graph is independently sampled through the existing
  stable graph-cache identity. No trajectory coupling is introduced.
* Five methods use the existing MethodManager unchanged.
"""

from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterable
import hashlib
import json
import math
import shutil
import time

import networkx as nx
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage, optimal_leaf_ordering
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.metrics import (
    adjusted_rand_score,
    balanced_accuracy_score,
    f1_score,
    normalized_mutual_info_score,
    roc_auc_score,
    silhouette_score,
)

from hodge_vnext.cache import ArtifactStore, CACHE_SCHEMA
from hodge_vnext.cliques import enumerate_maximal_clique_layers
from hodge_vnext.graph_io import GraphRecord
from hodge_vnext.models import synthetic_graph_record
from hodge_vnext.utils import stable_hash, write_json

from .config import method_from_payload, resolve_path
from .methods import DEFAULT_METHODS, MethodManager
from .plotting import plot_model_discrimination




def rgg_square_pair_probability(radius: float) -> float:
    r = float(radius)
    if not 0.0 <= r <= 1.0:
        raise ValueError("finite-square formula requires 0 <= radius <= 1")
    return float(math.pi * r * r - (8.0 / 3.0) * r ** 3 + 0.5 * r ** 4)


def rgg_radius_for_expected_degree(n: int, target_mean_degree: float) -> float:
    n = int(n)
    kappa = float(target_mean_degree)
    if n < 2 or not 0.0 <= kappa <= n - 1:
        raise ValueError("require N>=2 and target mean degree in [0,N-1]")
    if kappa == 0.0:
        return 0.0
    target = kappa / float(n - 1)
    if target > rgg_square_pair_probability(1.0) + 1e-15:
        raise ValueError("target requires radius>1; outside this protocol")
    lo, hi = 0.0, 1.0
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        if rgg_square_pair_probability(mid) < target:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))

FINE_STATE_ORDER = (
    "ER",
    "RGG",
    "WS-beta1e-3",
    "WS-beta1e-2",
    "WS-beta1e-1",
    "WS-beta1",
)


def _progress(msg: str) -> None:
    print(msg, flush=True)


def _clear_run_outputs(run_dir: Path) -> None:
    for name in ("tables", "arrays", "plots"):
        p = run_dir / name
        if p.exists():
            shutil.rmtree(p)
    for name in ("RUN_COMPLETE.json", "run_manifest.json", "effective_config.json"):
        p = run_dir / name
        if p.exists():
            p.unlink()


def _coarse_label(state: str) -> str:
    return "WS" if str(state).startswith("WS-") else str(state)


def _parse_state(state: str) -> tuple[str, float | None]:
    if state == "ER":
        return "ER", None
    if state == "RGG":
        return "RGG", None
    prefix = "WS-beta"
    if not state.startswith(prefix):
        raise ValueError(f"unsupported mechanism state: {state}")
    token = state[len(prefix):]
    mapping = {"1e-3": 1e-3, "1e-2": 1e-2, "1e-1": 1e-1, "1": 1.0}
    if token not in mapping:
        raise ValueError(f"unsupported WS state: {state}")
    return "WS", float(mapping[token])


def _state_from_beta(beta: float) -> str:
    choices = [(1e-3, "WS-beta1e-3"), (1e-2, "WS-beta1e-2"), (1e-1, "WS-beta1e-1"), (1.0, "WS-beta1")]
    for value, label in choices:
        if abs(float(beta) - value) <= 1e-14:
            return label
    raise ValueError("formal protocol supports WS beta in {1e-3,1e-2,1e-1,1}")


def _params_for_state(state: str, n: int, kappa: float) -> tuple[str, dict[str, Any], dict[str, float | int | None]]:
    model, beta = _parse_state(state)
    if not 0 < float(kappa) < n - 1:
        raise ValueError("target mean degree must satisfy 0 < kappa < N-1")
    if model == "ER":
        p = float(kappa) / float(n - 1)
        return model, {"p": p}, {"er_p": p, "ws_degree": None, "ws_beta": None, "rgg_radius": None}
    if model == "WS":
        degree = int(round(float(kappa)))
        if abs(float(kappa) - degree) > 1e-12 or degree % 2:
            raise ValueError("WS matched-density protocol requires even integer target degrees")
        return model, {"degree": degree, "p": float(beta)}, {
            "er_p": None, "ws_degree": degree, "ws_beta": float(beta), "rgg_radius": None,
        }
    radius = rgg_radius_for_expected_degree(n, float(kappa))
    nominal_lambda = float(n * math.pi * radius * radius)
    # Use the established RGG generator contract (lambda, dim, p_norm); the
    # finite-square inversion above chooses lambda so the expected mean degree
    # matches kappa rather than equating lambda with kappa.
    return "RGG", {"lambda": nominal_lambda, "dim": 2, "p_norm": 2.0}, {
        "er_p": None, "ws_degree": None, "ws_beta": None, "rgg_radius": radius,
    }


def build_parameter_cells(n: int, target_degrees: Iterable[float], ws_betas: Iterable[float]) -> pd.DataFrame:
    kappas = [float(x) for x in target_degrees]
    if len(kappas) == 0 or len(set(kappas)) != len(kappas):
        raise ValueError("target_degrees must be non-empty and unique")
    if any(b <= a for a, b in zip(kappas[:-1], kappas[1:])):
        raise ValueError("target_degrees must be strictly increasing")
    states = ["ER", "RGG"] + [_state_from_beta(float(b)) for b in ws_betas]
    if tuple(states) != FINE_STATE_ORDER:
        raise ValueError("formal protocol requires WS betas [1e-3,1e-2,1e-1,1] in that order")
    rows = []
    cell = 0
    for state in states:
        for kappa in kappas:
            model, params, derived = _params_for_state(state, int(n), kappa)
            radius = derived["rgg_radius"]
            nominal_lambda = float(n * math.pi * radius * radius) if radius is not None else None
            rows.append({
                "cell_index": cell,
                "cell_label": f"{state}-k{format(kappa, '.6g')}",
                "mechanism_state": state,
                "coarse_mechanism": _coarse_label(state),
                "model": model,
                "target_mean_degree": kappa,
                "er_p": derived["er_p"],
                "ws_degree": derived["ws_degree"],
                "ws_beta": derived["ws_beta"],
                "rgg_dim": 2 if model == "RGG" else None,
                "rgg_p_norm": 2.0 if model == "RGG" else None,
                "rgg_radius": radius,
                "rgg_nominal_lambda": nominal_lambda,
                "generator_params_json": json.dumps(params, sort_keys=True, separators=(",", ":")),
            })
            cell += 1
    return pd.DataFrame(rows)


def _graph_observables(G: nx.Graph) -> dict[str, float | int]:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    components = sorted((len(c) for c in nx.connected_components(G)), reverse=True)
    largest = components[0] if components else 0
    isolates = int(nx.number_of_isolates(G))
    return {
        "num_nodes": int(n),
        "num_edges": int(m),
        "realized_mean_degree": float(2.0 * m / max(n, 1)),
        "num_components": int(len(components)),
        "largest_component_fraction": float(largest / max(n, 1)),
        "num_isolates": isolates,
        "isolated_fraction": float(isolates / max(n, 1)),
        "average_clustering": float(nx.average_clustering(G)),
    }


def _cell_distance_matrix(D: np.ndarray, graph_cells: np.ndarray, num_cells: int) -> tuple[np.ndarray, np.ndarray]:
    means = np.zeros((num_cells, num_cells), dtype=float)
    counts = np.zeros((num_cells, num_cells), dtype=int)
    for a in range(num_cells):
        ia = np.flatnonzero(graph_cells == a)
        if len(ia) < 2:
            raise ValueError(f"cell {a} needs at least two repeats")
        vals = D[np.ix_(ia, ia)][np.triu_indices(len(ia), 1)]
        means[a, a] = float(np.mean(vals))
        counts[a, a] = int(len(vals))
        for b in range(a + 1, num_cells):
            ib = np.flatnonzero(graph_cells == b)
            vals = D[np.ix_(ia, ib)].ravel()
            means[a, b] = means[b, a] = float(np.mean(vals))
            counts[a, b] = counts[b, a] = int(len(vals))
    return means, counts



def classical_mds_2d(D: np.ndarray, *, zero_diagonal: bool = False) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """Deterministic classical MDS embedding of a precomputed dissimilarity matrix.

    Parameters
    ----------
    D:
        Symmetric dissimilarity matrix. Graph-level matrices are expected to
        have a zero diagonal. Cell-level matrices in this experiment store
        same-cell realization noise on the diagonal; set ``zero_diagonal=True``
        so the embedding uses the mathematically required self-distance zero
        while the original noise diagonal remains available in the saved cell
        matrix and tables.

    Returns
    -------
    coords:
        ``(n, 2)`` embedding. Axis signs are canonicalized for reproducible
        plotting; rotations inside an exactly degenerate eigenspace remain
        mathematically arbitrary.
    diagnostics:
        Positive-spectrum 2-D fraction, non-Euclidean negative-spectrum
        fraction, original-vs-2D distance Spearman correlation and normalized
        raw-distance stress. These diagnostics are descriptive only and are
        not classification metrics.
    """
    D = np.asarray(D, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("MDS requires a square dissimilarity matrix")
    n = int(D.shape[0])
    if n < 2:
        raise ValueError("MDS requires at least two objects")
    if not np.all(np.isfinite(D)):
        raise ValueError("MDS dissimilarity matrix contains non-finite values")
    if not np.allclose(D, D.T, rtol=1e-10, atol=1e-12):
        raise ValueError("MDS dissimilarity matrix must be symmetric")
    if np.any(D < -1e-12):
        raise ValueError("MDS dissimilarities must be non-negative")

    work = 0.5 * (D + D.T)
    if zero_diagonal:
        work = work.copy()
        np.fill_diagonal(work, 0.0)
    elif not np.allclose(np.diag(work), 0.0, atol=1e-12, rtol=0):
        raise ValueError("MDS graph-level self-distances must be zero")

    J = np.eye(n) - np.ones((n, n), dtype=float) / float(n)
    B = -0.5 * J @ (work * work) @ J
    B = 0.5 * (B + B.T)
    eigvals, eigvecs = np.linalg.eigh(B)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    tol = max(1.0, float(np.max(np.abs(eigvals)))) * 1e-12
    positive = eigvals > tol
    positive_vals = eigvals[positive]
    positive_vecs = eigvecs[:, positive]

    coords = np.zeros((n, 2), dtype=float)
    take = min(2, len(positive_vals))
    if take:
        coords[:, :take] = positive_vecs[:, :take] * np.sqrt(positive_vals[:take])[None, :]

    # Eigenvector signs are arbitrary; canonicalize them for stable exports.
    for axis in range(2):
        col = coords[:, axis]
        if np.allclose(col, 0.0):
            continue
        pivot = int(np.argmax(np.abs(col)))
        if col[pivot] < 0:
            coords[:, axis] *= -1.0

    original = work[np.triu_indices(n, 1)]
    embedded = pdist(coords, metric="euclidean")
    denom = float(np.sum(original * original))
    stress = float(np.sqrt(np.sum((original - embedded) ** 2) / denom)) if denom > 0 else float("nan")
    if len(original) >= 2 and np.std(original) > 0 and np.std(embedded) > 0:
        rho = float(spearmanr(original, embedded)[0])
        pearson = float(np.corrcoef(original, embedded)[0, 1])
    else:
        rho = float("nan")
        pearson = float("nan")

    positive_sum = float(np.sum(positive_vals))
    explained_2d = float(np.sum(positive_vals[:take]) / positive_sum) if positive_sum > 0 else float("nan")
    negative_abs = float(np.sum(np.abs(eigvals[eigvals < -tol])))
    total_abs = float(np.sum(np.abs(eigvals[np.abs(eigvals) > tol])))
    negative_fraction = float(negative_abs / total_abs) if total_abs > 0 else 0.0

    diagnostics: dict[str, float | int | str] = {
        "n_objects": n,
        "positive_eigenvalue_count": int(np.count_nonzero(positive)),
        "positive_eigenvalue_fraction_2d": explained_2d,
        "negative_eigenvalue_abs_fraction": negative_fraction,
        "distance_spearman_2d": rho,
        "distance_pearson_2d": pearson,
        "normalized_stress_2d": stress,
        "diagonal_policy": "zeroed_for_embedding" if zero_diagonal else "native_zero",
    }
    return coords, diagnostics


def _write_mds_artifacts(run_dir: Path, methods: Iterable[str]) -> pd.DataFrame:
    """Materialize deterministic graph-level and cell-level 2-D MDS tables.

    This is a pure run-level post-processing step. It does not alter method
    representations, pairwise distances, cache identities, random seeds or
    evaluation statistics.
    """
    graph_df = pd.read_csv(run_dir / "tables" / "graph_index.csv").sort_values("graph_index")
    cells = pd.read_csv(run_dir / "tables" / "parameter_cells.csv").sort_values("cell_index")
    diag_rows: list[dict[str, Any]] = []

    graph_cols = [
        "graph_index", "graph_hash", "cell_index", "cell_label", "mechanism_state",
        "coarse_mechanism", "target_mean_degree", "repeat",
    ]
    cell_cols = [
        "cell_index", "cell_label", "mechanism_state", "coarse_mechanism", "target_mean_degree",
    ]
    for method in methods:
        graph_path = run_dir / "arrays" / f"graph_distance_{method}.npy"
        cell_path = run_dir / "arrays" / f"cell_distance_{method}.npy"
        if not graph_path.exists() or not cell_path.exists():
            continue

        graph_coords, graph_diag = classical_mds_2d(np.load(graph_path), zero_diagonal=False)
        graph_out = graph_df[graph_cols].copy()
        graph_out["mds_x"] = graph_coords[:, 0]
        graph_out["mds_y"] = graph_coords[:, 1]
        graph_out.to_csv(run_dir / "tables" / f"mds_graph_{method}.csv", index=False)
        diag_rows.append({"method": method, "level": "graph", **graph_diag})

        # The cell matrix diagonal stores within-cell realization noise by
        # design. Classical MDS requires d(i,i)=0, so only the embedding copy
        # has its diagonal reset to zero; the saved raw cell matrix is untouched.
        cell_coords, cell_diag = classical_mds_2d(np.load(cell_path), zero_diagonal=True)
        cell_out = cells[cell_cols].copy()
        cell_out["mds_x"] = cell_coords[:, 0]
        cell_out["mds_y"] = cell_coords[:, 1]
        cell_out.to_csv(run_dir / "tables" / f"mds_cell_{method}.csv", index=False)
        diag_rows.append({"method": method, "level": "cell", **cell_diag})

    diag_df = pd.DataFrame(diag_rows)
    diag_df.to_csv(run_dir / "tables" / "mds_diagnostics.csv", index=False)
    return diag_df


def _tie_key(query_id: str, candidate_id: str, seed: int) -> str:
    payload = json.dumps([int(seed), str(query_id), str(candidate_id)], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _eligible_mask(kappa: np.ndarray, i: int, protocol: str) -> np.ndarray:
    mask = np.arange(len(kappa)) != i
    if protocol == "matched_density":
        mask &= np.isclose(kappa, kappa[i], rtol=0, atol=1e-12)
    elif protocol == "cross_density":
        mask &= ~np.isclose(kappa, kappa[i], rtol=0, atol=1e-12)
    elif protocol == "global":
        pass
    else:
        raise ValueError(protocol)
    return mask


def _retrieval_metrics(
    D: np.ndarray,
    graph_ids: np.ndarray,
    labels: np.ndarray,
    kappa: np.ndarray,
    protocol: str,
    tie_seed: int,
) -> tuple[dict[str, float | int], pd.DataFrame, pd.DataFrame]:
    labels = np.asarray(labels, dtype=str)
    preds: list[str] = []
    query_rows = []
    pair_y: list[int] = []
    pair_score: list[float] = []
    n = len(labels)
    # Pairwise AUROC uses each unordered eligible pair exactly once.
    for i in range(n):
        for j in range(i + 1, n):
            eligible = (protocol == "global") or (
                protocol == "matched_density" and math.isclose(float(kappa[i]), float(kappa[j]), abs_tol=1e-12)
            ) or (
                protocol == "cross_density" and not math.isclose(float(kappa[i]), float(kappa[j]), abs_tol=1e-12)
            )
            if eligible:
                pair_y.append(int(labels[i] == labels[j]))
                pair_score.append(float(-D[i, j]))

    for i in range(n):
        mask = _eligible_mask(kappa, i, protocol)
        candidates = np.flatnonzero(mask).tolist()
        if not candidates:
            raise ValueError(f"query {i} has no eligible candidates for {protocol}")
        ranked = sorted(
            candidates,
            key=lambda j: (float(D[i, j]), _tie_key(str(graph_ids[i]), str(graph_ids[j]), tie_seed), str(graph_ids[j])),
        )
        pred = str(labels[ranked[0]])
        preds.append(pred)
        rel = np.asarray([labels[j] == labels[i] for j in ranked], dtype=int)
        total_rel = int(rel.sum())
        cum = np.cumsum(rel)
        precision = cum / np.arange(1, len(rel) + 1)
        ap = float(np.dot(precision, rel) / total_rel) if total_rel else float("nan")
        k5 = min(5, len(rel))
        discounts = 1.0 / np.log2(np.arange(1, k5 + 1) + 1.0)
        ideal = min(5, total_rel)
        idcg = float(np.sum(1.0 / np.log2(np.arange(1, ideal + 1) + 1.0))) if ideal else 0.0
        ndcg = float(np.dot(rel[:k5], discounts) / idcg) if idcg else float("nan")
        row: dict[str, Any] = {
            "query_index": i,
            "graph_id": str(graph_ids[i]),
            "true_label": str(labels[i]),
            "predicted_label": pred,
            "nearest_graph_id": str(graph_ids[ranked[0]]),
            "nearest_distance": float(D[i, ranked[0]]),
            "gallery_size": len(ranked),
            "relevant_total": total_rel,
            "average_precision": ap,
            "ndcg_at_5": ndcg,
        }
        for kk in (1, 3, 5):
            ek = min(kk, len(rel))
            hits = int(cum[ek - 1])
            row[f"precision_at_{kk}"] = float(hits / ek)
            row[f"recall_at_{kk}"] = float(hits / total_rel) if total_rel else float("nan")
        query_rows.append(row)

    preds_arr = np.asarray(preds, dtype=str)
    eligible_queries = pd.DataFrame(query_rows)
    result: dict[str, float | int] = {
        "n_queries": n,
        "accuracy": float(np.mean(preds_arr == labels)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds_arr)),
        "macro_f1": float(f1_score(labels, preds_arr, average="macro", zero_division=0)),
        "same_class_auroc": float(roc_auc_score(pair_y, pair_score)) if len(set(pair_y)) == 2 else float("nan"),
        "mean_average_precision": float(np.nanmean(eligible_queries.average_precision)),
        "ndcg_at_5": float(np.nanmean(eligible_queries.ndcg_at_5)),
        "precision_at_1": float(eligible_queries.precision_at_1.mean()),
        "precision_at_3": float(eligible_queries.precision_at_3.mean()),
        "precision_at_5": float(eligible_queries.precision_at_5.mean()),
        "recall_at_1": float(np.nanmean(eligible_queries.recall_at_1)),
        "recall_at_3": float(np.nanmean(eligible_queries.recall_at_3)),
        "recall_at_5": float(np.nanmean(eligible_queries.recall_at_5)),
    }
    classes = sorted(set(labels.tolist()))
    confusion = pd.crosstab(
        pd.Series(labels, name="true_label"),
        pd.Series(preds_arr, name="predicted_label"),
    ).reindex(index=classes, columns=classes, fill_value=0)
    return result, eligible_queries, confusion


def _global_unsupervised(D: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    classes, y = np.unique(labels.astype(str), return_inverse=True)
    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method="average")
    if len(D) > 2:
        Z = optimal_leaf_ordering(Z, condensed)
    pred = fcluster(Z, t=len(classes), criterion="maxclust") - 1
    try:
        sil = float(silhouette_score(D, y, metric="precomputed"))
    except Exception:
        sil = float("nan")
    return {
        "ari": float(adjusted_rand_score(y, pred)),
        "nmi": float(normalized_mutual_info_score(y, pred)),
        "silhouette": sil,
    }


def _mechanism_pair_separation(
    D: np.ndarray,
    states: np.ndarray,
    kappas: np.ndarray,
    state_order: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for kappa in sorted(set(float(x) for x in kappas)):
        for ai, a in enumerate(state_order):
            ia = np.flatnonzero((states == a) & np.isclose(kappas, kappa))
            noise_a = D[np.ix_(ia, ia)][np.triu_indices(len(ia), 1)]
            for b in state_order[ai + 1:]:
                ib = np.flatnonzero((states == b) & np.isclose(kappas, kappa))
                noise_b = D[np.ix_(ib, ib)][np.triu_indices(len(ib), 1)]
                between = D[np.ix_(ia, ib)].ravel()
                baseline = 0.5 * (float(noise_a.mean()) + float(noise_b.mean()))
                rows.append({
                    "target_mean_degree": kappa,
                    "mechanism_a": a,
                    "mechanism_b": b,
                    "between_mean": float(between.mean()),
                    "noise_a_mean": float(noise_a.mean()),
                    "noise_b_mean": float(noise_b.mean()),
                    "mean_noise": baseline,
                    "separation_ratio": float(between.mean() / baseline) if baseline > 0 else float("nan"),
                    "between_pair_count": int(len(between)),
                })
    return rows


def _run_stem(exp: dict[str, Any], cfg_dict: dict[str, Any], baseline_dict: dict[str, Any]) -> str:
    identity = {
        "schema": "er-ws-rgg-mechanism-discrimination-v1",
        "n": int(exp["n"]),
        "repeats": int(exp["repeats"]),
        "base_seed": int(exp.get("base_seed", 22)),
        "target_degrees": [float(x) for x in exp["target_degrees"]],
        "ws_betas": [float(x) for x in exp["ws_betas"]],
        "methods": list(exp["methods"]),
        "tie_seed": int(exp.get("tie_seed", 20260916)),
        "method": cfg_dict,
        "baselines": baseline_dict,
    }
    token = stable_hash(identity, length=12)
    return f"N{identity['n']}_rep{identity['repeats']}_mech6_k{len(identity['target_degrees'])}_id-{token}"


def run_model_discrimination(
    payload: dict[str, Any],
    config_path: Path,
    *,
    force: bool = False,
    progress: Callable[[str], None] | None = None,
) -> Path:
    progress = progress or _progress
    cfg = method_from_payload(payload, config_path)
    exp = dict(payload["validation"])
    n = int(exp["n"])
    repeats = int(exp["repeats"])
    base_seed = int(exp.get("base_seed", 22))
    if repeats < 2:
        raise ValueError("repeats must be >=2")
    target_degrees = [float(x) for x in exp["target_degrees"]]
    ws_betas = [float(x) for x in exp["ws_betas"]]
    methods = MethodManager.validate_methods(list(exp.get("methods", DEFAULT_METHODS)))
    manager = MethodManager.from_config(cfg, payload.get("baselines", {}))
    tie_seed = int(exp.get("tie_seed", 20260916))

    # Explicit hard guards for the frozen experiment definition.
    if cfg.alpha != 0.0 or cfg.normalization != "per_simplex" or cfg.min_layer_q != 0:
        raise ValueError("formal discrimination protocol requires alpha=0, per_simplex, min_layer_q=0")
    if int(payload.get("baselines", {}).get("count_min_layer_q", -1)) != 0:
        raise ValueError("formal discrimination protocol requires Count count_min_layer_q=0")

    cells = build_parameter_cells(n, target_degrees, ws_betas)
    state_order = list(FINE_STATE_ORDER)
    results_root = resolve_path(exp.get("results_root", "results/validation/model_discrimination"), config_path)
    run_dir = results_root / _run_stem(exp, payload["method"], payload.get("baselines", {}))
    complete = run_dir / "RUN_COMPLETE.json"
    if complete.exists() and not force:
        progress(f"[POSTPROCESS] completed run exists; refreshing derived MDS/plots only: {run_dir}")
        _write_mds_artifacts(run_dir, methods)
        plot_model_discrimination(run_dir)
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    if force:
        _clear_run_outputs(run_dir)
    for name in ("tables", "arrays", "plots"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)

    effective = json.loads(json.dumps(payload))
    effective["resolved_protocol"] = {
        "schema": "er-ws-rgg-mechanism-discrimination-v1",
        "fine_states": state_order,
        "target_degrees": target_degrees,
        "num_cells": int(len(cells)),
        "num_graphs": int(len(cells) * repeats),
        "density_matching": {
            "ER": "p=kappa/(N-1)",
            "WS": "degree=kappa",
            "RGG": "(N-1)*(pi*r^2-(8/3)r^3+(1/2)r^4)=kappa",
        },
        "visualization": {
            "mds": "deterministic classical MDS in 2D from each method's raw dissimilarity matrix",
            "graph_level": "all graph realizations",
            "cell_level": "36 parameter cells; same-cell noise diagonal is zeroed only for MDS embedding",
            "interpretation": "visualization aid only; quantitative conclusions use retrieval/clustering metrics",
        },
    }
    effective["cache_schema"] = CACHE_SCHEMA
    write_json(run_dir / "effective_config.json", effective)
    cells.to_csv(run_dir / "tables" / "parameter_cells.csv", index=False)

    # ------------------------------------------------------------------
    # Independent graph generation + structural/clique audit.
    # ------------------------------------------------------------------
    graph_rows: list[dict[str, Any]] = []
    records: list[GraphRecord] = []
    for cell_row in cells.itertuples(index=False):
        state = str(cell_row.mechanism_state)
        kappa = float(cell_row.target_mean_degree)
        model, params, _ = _params_for_state(state, n, kappa)
        progress(f"[CELL {int(cell_row.cell_index)+1:02d}/{len(cells):02d}] {state} kappa={kappa:g}")
        for repeat in range(1, repeats + 1):
            rec = synthetic_graph_record(cfg.cache.root, model, n, params, base_seed, repeat)
            records.append(rec)
            obs = _graph_observables(rec.graph)
            store = ArtifactStore(rec.cache_root, cfg.cache)
            layers = enumerate_maximal_clique_layers(rec.graph, store, progress=None)
            qmax = max(layers) if layers else -1
            clique_count = sum(len(v) for v in layers.values())
            graph_rows.append({
                "graph_index": len(graph_rows),
                "cell_index": int(cell_row.cell_index),
                "cell_label": str(cell_row.cell_label),
                "mechanism_state": state,
                "coarse_mechanism": _coarse_label(state),
                "model": model,
                "target_mean_degree": kappa,
                "repeat": repeat,
                "graph_hash": rec.graph_hash,
                "graph_seed": int(rec.source["graph_seed"]),
                "cache_graph_root": str(rec.cache_root),
                "num_maximal_cliques": int(clique_count),
                "maximal_q": int(qmax),
                **obs,
            })
    graph_df = pd.DataFrame(graph_rows)
    graph_df.to_csv(run_dir / "tables" / "graph_index.csv", index=False)

    audit = graph_df.groupby(["mechanism_state", "target_mean_degree"], sort=False, as_index=False).agg(
        graph_count=("graph_hash", "count"),
        realized_mean_degree_mean=("realized_mean_degree", "mean"),
        realized_mean_degree_std=("realized_mean_degree", "std"),
        average_clustering_mean=("average_clustering", "mean"),
        average_clustering_std=("average_clustering", "std"),
        largest_component_fraction_mean=("largest_component_fraction", "mean"),
        isolated_fraction_mean=("isolated_fraction", "mean"),
        maximal_q_mean=("maximal_q", "mean"),
        maximal_q_max=("maximal_q", "max"),
        num_maximal_cliques_mean=("num_maximal_cliques", "mean"),
    )
    audit.to_csv(run_dir / "tables" / "matched_structural_audit.csv", index=False)

    graph_cells = graph_df.cell_index.to_numpy(dtype=int)
    graph_states = graph_df.mechanism_state.astype(str).to_numpy()
    coarse_labels = graph_df.coarse_mechanism.astype(str).to_numpy()
    graph_kappas = graph_df.target_mean_degree.to_numpy(dtype=float)
    graph_ids = graph_df.graph_hash.astype(str).to_numpy()

    metrics_rows: list[dict[str, Any]] = []
    pair_sep_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Method representations, all graph-pair distances, cell distances.
    # ------------------------------------------------------------------
    for method in methods:
        progress(f"[METHOD] {method}: representations {len(records)} graphs")
        reps: list[Any] = []
        for gi, rec in enumerate(records):
            rep, elapsed, cache_hit = manager.representation(method, rec)
            reps.append(rep)
            runtime_rows.append({
                "method": method,
                "graph_index": gi,
                "graph_hash": rec.graph_hash,
                "representation_seconds": float(elapsed),
                "cache_hit": bool(cache_hit),
            })
            if (gi + 1) % max(1, len(records) // 10) == 0 or gi + 1 == len(records):
                progress(f"  [REP] {gi+1}/{len(records)}")

        m = len(records)
        D = np.zeros((m, m), dtype=float)
        progress(f"[METHOD] {method}: {m*(m-1)//2} pair distances")
        for i in range(m):
            ni = records[i].graph.number_of_nodes()
            for j in range(i + 1, m):
                d = manager.distance(method, reps[i], reps[j], ni, records[j].graph.number_of_nodes())
                D[i, j] = D[j, i] = float(d)
        np.save(run_dir / "arrays" / f"graph_distance_{method}.npy", D)
        cell_D, cell_counts = _cell_distance_matrix(D, graph_cells, len(cells))
        np.save(run_dir / "arrays" / f"cell_distance_{method}.npy", cell_D)
        np.save(run_dir / "arrays" / f"cell_pair_count_{method}.npy", cell_counts)

        for row in _mechanism_pair_separation(D, graph_states, graph_kappas, state_order):
            pair_sep_rows.append({"method": method, **row})

        for level, labels in (("fine", graph_states), ("coarse", coarse_labels)):
            for protocol in ("matched_density", "cross_density", "global"):
                stats, per_query, confusion = _retrieval_metrics(D, graph_ids, labels, graph_kappas, protocol, tie_seed)
                unsup = _global_unsupervised(D, labels) if protocol == "global" else {
                    "ari": float("nan"), "nmi": float("nan"), "silhouette": float("nan")
                }
                metrics_rows.append({
                    "method": method,
                    "label_level": level,
                    "protocol": protocol,
                    **stats,
                    **unsup,
                })
                per_query.insert(0, "method", method)
                per_query.insert(1, "label_level", level)
                per_query.insert(2, "protocol", protocol)
                per_query.to_csv(run_dir / "tables" / f"query_metrics_{protocol}_{level}_{method}.csv", index=False)
                if level == "fine" and protocol in {"matched_density", "cross_density"}:
                    confusion.to_csv(run_dir / "tables" / f"confusion_{protocol}_{method}.csv")
        del reps

    pd.DataFrame(runtime_rows).to_csv(run_dir / "tables" / "runtime_per_graph.csv", index=False)
    pd.DataFrame(metrics_rows).to_csv(run_dir / "tables" / "metrics_all_methods.csv", index=False)
    pd.DataFrame(pair_sep_rows).to_csv(run_dir / "tables" / "mechanism_pair_separation.csv", index=False)

    # Deterministic 2-D MDS is derived strictly from the already-computed raw
    # distance matrices. It is an explanatory visualization, not an evaluation
    # metric and not part of the method/cache identity.
    _write_mds_artifacts(run_dir, methods)

    # Plot only after all tables/arrays are committed.
    plot_model_discrimination(run_dir)

    manifest = {
        "schema": "er-ws-rgg-mechanism-discrimination-v1",
        "cache_schema": CACHE_SCHEMA,
        "method": asdict(cfg),
        "methods": methods,
        "n": n,
        "repeats": repeats,
        "base_seed": base_seed,
        "target_degrees": target_degrees,
        "ws_betas": ws_betas,
        "num_cells": int(len(cells)),
        "num_graphs": int(len(graph_df)),
        "fine_states": state_order,
        "count_min_layer_q": int(payload["baselines"]["count_min_layer_q"]),
        "visualization": {
            "mds": "classical-2d",
            "levels": ["graph", "cell"],
            "cell_diagonal_policy": "zeroed only for embedding; raw cell diagonal remains same-cell noise",
        },
    }
    write_json(run_dir / "run_manifest.json", manifest)
    write_json(run_dir / "RUN_COMPLETE.json", {"completed": True, "schema": manifest["schema"]})
    progress(f"[DONE] {run_dir}")
    return run_dir
