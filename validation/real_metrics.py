from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Sequence

import networkx as nx
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage, optimal_leaf_ordering
from scipy.spatial.distance import squareform


@dataclass(frozen=True)
class ClusterResult:
    linkage_matrix: np.ndarray
    cluster_labels: np.ndarray
    leaf_order: np.ndarray


def validate_distance_matrix(matrix: np.ndarray, atol: float = 1e-9) -> np.ndarray:
    D = np.asarray(matrix, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("distance matrix must be square")
    if not np.all(np.isfinite(D)):
        raise ValueError("distance matrix contains non-finite values")
    if np.max(np.abs(D - D.T)) > atol:
        raise ValueError("distance matrix must be symmetric")
    if np.min(D) < -atol:
        raise ValueError("distance matrix contains negative values")
    D = 0.5 * (D + D.T)
    D[D < 0] = 0.0
    np.fill_diagonal(D, 0.0)
    return D


def hierarchical_clustering(D: np.ndarray, n_clusters: int, method: str = "average") -> ClusterResult:
    D = validate_distance_matrix(D)
    if len(D) < 2:
        return ClusterResult(np.empty((0, 4)), np.ones(len(D), dtype=int), np.arange(len(D)))
    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method=method)
    if len(D) > 2:
        Z = optimal_leaf_ordering(Z, condensed)
    labels = fcluster(Z, t=int(n_clusters), criterion="maxclust").astype(int)
    return ClusterResult(Z, labels, leaves_list(Z).astype(int))


def _comb2(x):
    x = np.asarray(x)
    return x * (x - 1) / 2


def adjusted_rand_index(true_labels: Sequence[str], pred_labels: Sequence[int]) -> float:
    _, ti = np.unique(np.asarray(true_labels, dtype=object), return_inverse=True)
    _, pi = np.unique(np.asarray(pred_labels), return_inverse=True)
    table = np.zeros((ti.max() + 1, pi.max() + 1), dtype=np.int64)
    for a, b in zip(ti, pi):
        table[a, b] += 1
    n = int(table.sum())
    if n < 2:
        return 1.0
    nij = float(_comb2(table).sum())
    ai = float(_comb2(table.sum(axis=1)).sum())
    bj = float(_comb2(table.sum(axis=0)).sum())
    total = float(_comb2(n))
    expected = ai * bj / total if total else 0.0
    maximum = 0.5 * (ai + bj)
    return float((nij - expected) / (maximum - expected)) if maximum != expected else 1.0


def normalized_mutual_information(true_labels: Sequence[str], pred_labels: Sequence[int]) -> float:
    _, ti = np.unique(np.asarray(true_labels, dtype=object), return_inverse=True)
    _, pi = np.unique(np.asarray(pred_labels), return_inverse=True)
    table = np.zeros((ti.max() + 1, pi.max() + 1), dtype=float)
    for a, b in zip(ti, pi):
        table[a, b] += 1
    P = table / table.sum()
    pr, pc = P.sum(axis=1), P.sum(axis=0)
    mi = 0.0
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            if P[i, j] > 0:
                mi += P[i, j] * math.log(P[i, j] / (pr[i] * pc[j]))
    hr = -sum(float(x) * math.log(float(x)) for x in pr if x > 0)
    hc = -sum(float(x) * math.log(float(x)) for x in pc if x > 0)
    denom = math.sqrt(hr * hc)
    return float(mi / denom) if denom > 0 else 1.0


def silhouette_precomputed(D: np.ndarray, labels: Sequence[str]) -> float:
    D = validate_distance_matrix(D)
    y = np.asarray(labels, dtype=object)
    unique = list(dict.fromkeys(y.tolist()))
    scores = []
    for i in range(len(y)):
        same = np.flatnonzero(y == y[i])
        same = same[same != i]
        a = float(np.mean(D[i, same])) if len(same) else 0.0
        bs = [float(np.mean(D[i, np.flatnonzero(y == lab)])) for lab in unique if lab != y[i]]
        b = min(bs) if bs else 0.0
        scores.append((b - a) / max(a, b) if max(a, b) > 0 else 0.0)
    return float(np.mean(scores))


def same_domain_auroc(D: np.ndarray, labels: Sequence[str]) -> float:
    D = validate_distance_matrix(D)
    labels = list(labels)
    pos, neg = [], []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            (pos if labels[i] == labels[j] else neg).append(float(D[i, j]))
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p < n else (0.5 if p == n else 0.0)
    return float(wins / (len(pos) * len(neg)))


def _tie_key(seed: int, query: str, candidate: str) -> str:
    raw = json.dumps([int(seed), str(query), str(candidate)], separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _macro_f1(true: Sequence[str], pred: Sequence[str]) -> float:
    y, p = np.asarray(true, dtype=object), np.asarray(pred, dtype=object)
    vals = []
    for label in sorted(set(y.tolist())):
        tp = int(np.sum((y == label) & (p == label)))
        fp = int(np.sum((y != label) & (p == label)))
        fn = int(np.sum((y == label) & (p != label)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        vals.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(vals))


def retrieval_metrics(
    D: np.ndarray,
    ids: Sequence[str],
    domains: Sequence[str],
    ks: Sequence[int] = (1, 3, 5, 7),
    tie_seed: int = 20260909,
):
    D = validate_distance_matrix(D)
    ids = np.asarray(ids, dtype=str)
    labels = np.asarray(domains, dtype=str)
    rows, ranks = [], []
    for i, query in enumerate(ids):
        candidates = [j for j in range(len(ids)) if j != i]
        ranked = sorted(candidates, key=lambda j: (D[i, j], _tie_key(tie_seed, query, ids[j]), ids[j]))
        rel = (labels[ranked] == labels[i]).astype(int)
        cum = np.cumsum(rel)
        R = int(rel.sum())
        precision_all = cum / np.arange(1, len(rel) + 1)
        ap = float(np.dot(precision_all, rel) / R) if R else np.nan
        k7 = min(7, len(rel))
        discounts = 1.0 / np.log2(np.arange(1, k7 + 1) + 1)
        idcg = float(discounts[:min(R, k7)].sum())
        ndcg = float(np.dot(rel[:k7], discounts) / idcg) if idcg > 0 else np.nan
        nn = ranked[0]
        nearest_ties = int(np.sum(D[i, candidates] == D[i, nn]))
        same_idx = [j for j in candidates if labels[j] == labels[i]]
        other_idx = [j for j in candidates if labels[j] != labels[i]]
        first_rel_positions = np.flatnonzero(rel == 1)
        first_relevant_rank = int(first_rel_positions[0] + 1) if len(first_rel_positions) else np.nan
        nearest_same_domain_distance = float(np.min(D[i, same_idx])) if same_idx else np.nan
        mean_same_domain_distance = float(np.mean(D[i, same_idx])) if same_idx else np.nan
        mean_other_domain_distance = float(np.mean(D[i, other_idx])) if other_idx else np.nan
        query_separation_ratio = (
            mean_other_domain_distance / mean_same_domain_distance
            if np.isfinite(mean_same_domain_distance) and mean_same_domain_distance > 0
            else np.nan
        )
        row = {
            "query": query, "domain": labels[i], "relevant_total": R,
            "nearest_neighbor": ids[nn], "nearest_neighbor_domain": labels[nn],
            "nearest_neighbor_correct": int(labels[nn] == labels[i]),
            "nearest_neighbor_distance": float(D[i, nn]), "nearest_tie_count": nearest_ties,
            "first_relevant_rank": first_relevant_rank,
            "nearest_same_domain_distance": nearest_same_domain_distance,
            "mean_same_domain_distance": mean_same_domain_distance,
            "mean_other_domain_distance": mean_other_domain_distance,
            "query_between_within_ratio": query_separation_ratio,
            "average_precision": ap, "ndcg_at_7": ndcg,
        }
        for k in ks:
            ek = min(int(k), len(rel))
            hits = int(cum[ek - 1])
            row[f"precision_at_{k}"] = hits / ek
            row[f"recall_at_{k}"] = hits / R if R else np.nan
        rows.append(row)
        for rank, j in enumerate(ranked, start=1):
            ranks.append({
                "query": query, "domain": labels[i], "rank": rank,
                "candidate": ids[j], "candidate_domain": labels[j],
                "distance": float(D[i, j]), "relevant": int(labels[j] == labels[i]),
            })
    per_query = pd.DataFrame(rows)
    pred = per_query.nearest_neighbor_domain.to_numpy(dtype=str)
    eligible = per_query[per_query.relevant_total > 0]
    summary = {
        "auroc_same_domain": same_domain_auroc(D, labels),
        "nn1_accuracy": float(per_query.nearest_neighbor_correct.mean()),
        "nn1_macro_f1": _macro_f1(labels, pred),
        "mAP": float(eligible.average_precision.mean()),
        "ndcg_at_7": float(eligible.ndcg_at_7.mean()),
        "queries_with_nearest_tie": int((per_query.nearest_tie_count > 1).sum()),
    }
    for k in ks:
        summary[f"precision_at_{k}"] = float(per_query[f"precision_at_{k}"].mean())
        summary[f"recall_at_{k}"] = float(eligible[f"recall_at_{k}"].mean())
    return summary, per_query, pd.DataFrame(ranks)


def minimum_spanning_tree(D: np.ndarray) -> nx.Graph:
    D = validate_distance_matrix(D)
    G = nx.Graph()
    G.add_nodes_from(range(len(D)))
    for i in range(len(D)):
        for j in range(i + 1, len(D)):
            G.add_edge(i, j, weight=float(D[i, j]))
    return nx.minimum_spanning_tree(G, algorithm="kruskal", weight="weight") if len(D) > 1 else G


def mst_metrics(D: np.ndarray, domains: Sequence[str], permutations: int = 5000, seed: int = 20260826) -> dict:
    T = minimum_spanning_tree(D)
    labels = np.asarray(domains, dtype=object)
    edges = list(T.edges())
    obs = float(sum(labels[u] == labels[v] for u, v in edges) / len(edges)) if edges else 1.0
    if permutations <= 0 or not edges:
        return {"mst_same_domain_ratio": obs, "mst_permutation_pvalue": np.nan}
    rng = np.random.default_rng(int(seed)); ge = 0
    for _ in range(int(permutations)):
        x = labels.copy(); rng.shuffle(x)
        ratio = sum(x[u] == x[v] for u, v in edges) / len(edges)
        ge += ratio >= obs - 1e-15
    return {"mst_same_domain_ratio": obs, "mst_permutation_pvalue": (ge + 1) / (permutations + 1)}


def global_metrics(D: np.ndarray, domains: Sequence[str], clusters: Sequence[int], permutations: int, seed: int) -> dict:
    D = validate_distance_matrix(D)
    within, between = [], []
    for i in range(len(D)):
        for j in range(i + 1, len(D)):
            (within if domains[i] == domains[j] else between).append(float(D[i, j]))
    w = float(np.mean(within)); b = float(np.mean(between))
    out = {
        "ARI": adjusted_rand_index(domains, clusters),
        "NMI": normalized_mutual_information(domains, clusters),
        "silhouette_true_domains": silhouette_precomputed(D, domains),
        "within_domain_mean_distance": w,
        "between_domain_mean_distance": b,
        "between_within_ratio": b / w if w > 0 else np.inf,
    }
    out.update(mst_metrics(D, domains, permutations, seed))
    return out


def classical_mds(D: np.ndarray, ndim: int = 2) -> tuple[np.ndarray, np.ndarray]:
    D = validate_distance_matrix(D)
    n = len(D)
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    vals, vecs = np.linalg.eigh(B)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    positive = np.maximum(vals[:ndim], 0.0)
    coords = vecs[:, :ndim] * np.sqrt(positive)
    return coords, vals
