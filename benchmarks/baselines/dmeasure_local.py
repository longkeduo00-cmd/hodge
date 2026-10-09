from __future__ import annotations

"""Local reproduction of Schieber et al.'s D-measure.

Official algorithm/code reference:
    tischieber/Quantifying-Network-Structural-Dissimilarities
    commit 27e1361884da3f7f28b38bfced4279422231ad38
    file d_distance.r (blob 620ac811c6706531eff91cf66a3349477d4b50df)

Paper:
    Schieber et al., Quantification of network structural dissimilarities,
    Nature Communications 8, 13928 (2017).

The Python implementation preserves the official R logic, including:
  * unreachable shortest-path distances mapped to N;
  * N distance bins (1..N-1 plus N for unreachable nodes);
  * the exact NND normalization used in d_distance.r;
  * alpha centrality x = (I - alpha A^T)^(-1)e with
        alpha=1/N and e=degree/(N-1);
  * sorting alpha centralities, dividing by N^2, appending residual mass;
  * left-padding alpha vectors when graph sizes differ;
  * recommended D weights 0.45, 0.45, 0.10;
  * full D averages alpha-centrality root-JS terms from graph and complement;
  * lowercase d omits the complement term.

Efficiency-only changes:
  1. all-pairs shortest paths are evaluated in exact SciPy chunks instead of
     materializing the full N x N distance matrix;
  2. complement alpha-centrality is solved exactly with a Sherman-Morrison
     identity, so the dense complement adjacency matrix is never built.
These change memory/runtime, not the mathematical D value (up to floating-point
roundoff).
"""

from dataclasses import dataclass
import math
import numpy as np
import networkx as nx
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
from scipy.sparse.csgraph import shortest_path

OFFICIAL_REPOSITORY = "https://github.com/tischieber/Quantifying-Network-Structural-Dissimilarities"
OFFICIAL_COMMIT = "27e1361884da3f7f28b38bfced4279422231ad38"
OFFICIAL_R_BLOB = "620ac811c6706531eff91cf66a3349477d4b50df"


def entropy_nats(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    a = a[a > 0]
    if a.size == 0:
        return 0.0
    return float(-np.sum(a * np.log(a)))


def _csr_adjacency(graph: nx.Graph, nodelist: list) -> sps.csr_matrix:
    A = nx.to_scipy_sparse_array(
        graph,
        nodelist=nodelist,
        dtype=float,
        weight=None,
        format="csr",
    )
    # scipy.csgraph is most robust with csr_matrix and 32-bit structural indices
    # for the graph sizes in this benchmark.  Values are unchanged.
    A = sps.csr_matrix(A, dtype=float)
    if A.indices.dtype != np.int32:
        A.indices = A.indices.astype(np.int32, copy=False)
    if A.indptr.dtype != np.int32:
        A.indptr = A.indptr.astype(np.int32, copy=False)
    return A


@dataclass(frozen=True)
class DMeasureRepresentation:
    n: int
    mu: np.ndarray
    nnd: float
    alpha: np.ndarray | None
    alpha_complement: np.ndarray | None


def distance_profile_and_nnd(
    graph: nx.Graph,
    chunk_size: int = 128,
) -> tuple[np.ndarray, float]:
    """Exact official node_distance()+nnd() result without NxN storage."""
    n = graph.number_of_nodes()
    if n <= 0:
        raise ValueError("D-measure requires a non-empty graph")
    if n == 1:
        # The official R node_distance() special-cases N=1, while nnd() itself
        # is not meaningfully defined for graph comparison.  VNext benchmark
        # graphs all have N>1; keep a deterministic extension for completeness.
        return np.asarray([1.0], dtype=float), 0.0

    nodes = list(graph.nodes())
    A = _csr_adjacency(graph, nodes)
    mu_sum = np.zeros(n, dtype=float)
    node_entropy_sum = 0.0
    denom = float(n - 1)

    chunk_size = max(1, int(chunk_size))
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        idx = np.arange(start, stop, dtype=np.int32)
        dist = shortest_path(
            A,
            method="D",
            directed=False,
            unweighted=True,
            indices=idx,
            return_predecessors=False,
        )
        dist = np.asarray(dist)
        if dist.ndim == 1:
            dist = dist[None, :]

        for local_row, source_index in enumerate(range(start, stop)):
            row = dist[local_row]
            bins = np.empty(n, dtype=np.int64)
            finite = np.isfinite(row)
            # Unweighted shortest paths are integral. rint protects against a
            # floating representation without changing the graph distance.
            bins[finite] = np.rint(row[finite]).astype(np.int64)
            bins[~finite] = n
            bins[source_index] = 0  # official node_distance excludes distance 0
            counts = np.bincount(bins, minlength=n + 1)[1 : n + 1].astype(float)
            p = counts / denom
            mu_sum += p
            node_entropy_sum += entropy_nats(p)

    mu = mu_sum / float(n)
    finite_positive_bins = int(np.count_nonzero(mu[: n - 1] > 0))
    norm = math.log(max(2, finite_positive_bins + 1))
    raw = entropy_nats(mu) - node_entropy_sum / float(n)
    nnd = max(0.0, raw) / norm
    return mu, float(nnd)


def _alpha_centrality_sorted_distribution(
    graph: nx.Graph,
    complement: bool,
) -> np.ndarray:
    """Exact alpha() distribution from the official R implementation."""
    n = graph.number_of_nodes()
    if n <= 0:
        raise ValueError("D-measure requires a non-empty graph")
    if n == 1:
        return np.asarray([0.0, 1.0], dtype=float)

    nodes = list(graph.nodes())
    A = _csr_adjacency(graph, nodes)
    degree = np.asarray(A.sum(axis=1)).ravel()
    alpha = 1.0 / float(n)
    eye = sps.eye(n, format="csc", dtype=float)

    if not complement:
        exo = degree / float(n - 1)
        M = eye - alpha * A.transpose().tocsc()
        x = spsl.spsolve(M, exo)
    else:
        # Complement adjacency: Ac = J - I - A (simple loopless graph).
        # Official alpha centrality solves
        #   (I - Ac/N) x = degree(Ac)/(N-1).
        # Write I - Ac/N = B - (1/N) 11^T with
        #   B=((N+1)/N)I + A/N,
        # then apply Sherman-Morrison. This avoids materializing dense J/Ac.
        comp_degree = (n - 1) - degree
        exo = comp_degree / float(n - 1)
        B = (((n + 1.0) / n) * eye + alpha * A.tocsc()).tocsc()
        lu = spsl.splu(B)
        y = lu.solve(exo)
        u = np.full(n, 1.0 / n, dtype=float)
        z = lu.solve(u)
        ones_dot_z = float(np.sum(z))
        denom = 1.0 - ones_dot_z
        if abs(denom) < 1e-14:
            raise np.linalg.LinAlgError("Unexpected near-singular complement alpha-centrality system")
        x = y + z * (float(np.sum(y)) / denom)

    r = np.sort(np.asarray(x, dtype=float)) / float(n * n)
    residual = max(0.0, 1.0 - float(np.sum(r)))
    return np.concatenate([r, np.asarray([residual])])


def dmeasure_representation(
    graph: nx.Graph,
    *,
    include_alpha: bool = True,
    include_complement: bool = True,
    shortest_path_chunk_size: int = 128,
) -> DMeasureRepresentation:
    mu, nnd = distance_profile_and_nnd(graph, chunk_size=shortest_path_chunk_size)
    alpha = (
        _alpha_centrality_sorted_distribution(graph, complement=False)
        if include_alpha
        else None
    )
    alpha_c = (
        _alpha_centrality_sorted_distribution(graph, complement=True)
        if include_alpha and include_complement
        else None
    )
    return DMeasureRepresentation(
        n=graph.number_of_nodes(),
        mu=mu,
        nnd=nnd,
        alpha=alpha,
        alpha_complement=alpha_c,
    )


def _root_js_nats(P: np.ndarray, Q: np.ndarray) -> float:
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    M = 0.5 * (P + Q)
    js = entropy_nats(M) - 0.5 * (entropy_nats(P) + entropy_nats(Q))
    return math.sqrt(max(js / math.log(2.0), 0.0))


def _aligned_mu_mix(g: DMeasureRepresentation, h: DMeasureRepresentation) -> np.ndarray:
    # Exact PM construction from official R: finite-distance bins align from
    # the left; each graph's unreachable bin is moved to the final common bin.
    m = max(g.n, h.n)
    PM = np.zeros(m, dtype=float)
    PM[: g.n - 1] = g.mu[: g.n - 1]
    PM[-1] = g.mu[g.n - 1]
    PM[: h.n - 1] += h.mu[: h.n - 1]
    PM[-1] += h.mu[h.n - 1]
    return PM / 2.0


def _left_pad(v: np.ndarray, length: int) -> np.ndarray:
    out = np.zeros(length, dtype=float)
    out[length - len(v) :] = v
    return out


def dmeasure_distance(
    g: DMeasureRepresentation,
    h: DMeasureRepresentation,
    *,
    w1: float = 0.45,
    w2: float = 0.45,
    w3: float = 0.10,
    include_complement: bool = True,
) -> float:
    """Full official D, or lowercase d if include_complement=False."""
    if not np.isclose(w1 + w2 + w3, 1.0):
        # Official code accepts arbitrary values; this warning-like validation
        # is intentionally not enforced. Keep values exactly as supplied.
        pass

    first = 0.0
    second = 0.0
    third = 0.0

    if w1 + w2 > 0:
        PM = _aligned_mu_mix(g, h)
        first_js = (
            entropy_nats(PM)
            - 0.5 * (entropy_nats(g.mu) + entropy_nats(h.mu))
        ) / math.log(2.0)
        first = math.sqrt(max(first_js, 0.0))
        second = abs(math.sqrt(g.nnd) - math.sqrt(h.nnd))

    if w3 > 0:
        if g.alpha is None or h.alpha is None:
            raise ValueError("alpha distributions were not computed")
        m = max(len(g.alpha), len(h.alpha))
        pg = _left_pad(g.alpha, m)
        ph = _left_pad(h.alpha, m)
        alpha_term = _root_js_nats(pg, ph)

        if include_complement:
            if g.alpha_complement is None or h.alpha_complement is None:
                raise ValueError("complement alpha distributions were not computed")
            mc = max(len(g.alpha_complement), len(h.alpha_complement))
            pgc = _left_pad(g.alpha_complement, mc)
            phc = _left_pad(h.alpha_complement, mc)
            complement_term = _root_js_nats(pgc, phc)
            # Exact uppercase D: each centrality root-JS contributes half.
            third = 0.5 * alpha_term + 0.5 * complement_term
        else:
            # Exact lowercase d: original-graph alpha term only, no /2.
            third = alpha_term

    return float(w1 * first + w2 * second + w3 * third)
