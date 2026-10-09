from __future__ import annotations

"""Local reproductions of the two netrd baselines used by the benchmark.

Reference implementation:
    netsiphd/netrd
    commit c1c85aa1517de49d335fb3c0e62cc0edbc5c161b

DegreeJS reproduces netrd/distance/degree_divergence.py.
NetSimile reproduces netrd/distance/netsimile.py.  The implementation below
uses algebraically equivalent adjacency-set formulas to avoid materializing an
ego graph for every node; the seven node features and the 35 graph-signature
moments are unchanged.
"""

from collections import Counter
import numpy as np
import networkx as nx
from scipy.spatial.distance import canberra
from scipy.stats import entropy as sp_entropy, skew, kurtosis

OFFICIAL_REPOSITORY = "https://github.com/netsiphd/netrd"
OFFICIAL_COMMIT = "c1c85aa1517de49d335fb3c0e62cc0edbc5c161b"


def js_divergence_net_rd(P: np.ndarray, Q: np.ndarray) -> float:
    """Direct netrd utilities.entropy.js_divergence convention (bits)."""
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    M = 0.5 * (P + Q)
    jsd = 0.5 * (sp_entropy(P, M, base=2) + sp_entropy(Q, M, base=2))
    if np.isclose(jsd, 0.0):
        return 0.0
    return float(jsd)


def degree_histogram_official(graph: nx.Graph) -> np.ndarray:
    """Faithful translation of netrd DegreeDivergence histogram construction."""
    vec = np.array(list(dict(graph.degree()).values()))
    if next(nx.selfloop_edges(graph), False):
        max_deg = len(graph)
    else:
        max_deg = len(graph) - 1
    counter = Counter(vec)
    # Keep the upstream range(max_deg) behavior exactly.
    return np.array([counter[v] for v in range(max_deg)], dtype=float)


def degree_js_distance(hist1: np.ndarray, hist2: np.ndarray) -> float:
    max_len = max(len(hist1), len(hist2))
    p1 = np.pad(hist1, (0, max_len - len(hist1)), "constant", constant_values=0)
    p2 = np.pad(hist2, (0, max_len - len(hist2)), "constant", constant_values=0)
    return js_divergence_net_rd(p1, p2)


def netsimile_feature_extraction(graph: nx.Graph) -> np.ndarray:
    """NetSimile's exact seven features, computed without explicit ego graphs."""
    node_list = sorted(graph.nodes())
    n = len(node_list)
    node_features = np.zeros((n, 7), dtype=float)
    degree = dict(graph.degree())
    clustering = nx.clustering(graph)
    adj = {u: set(graph.neighbors(u)) for u in node_list}

    for row, u in enumerate(node_list):
        nbrs = adj[u]
        du = degree[u]

        node_features[row, 0] = du
        node_features[row, 1] = clustering[u]

        if du > 0:
            node_features[row, 2] = np.mean([degree[v] for v in nbrs])
            node_features[row, 3] = np.mean([clustering[v] for v in nbrs])

            # Edges in the radius-1 ego network: ego-neighbor edges + edges
            # between neighbors. Algebraically identical to nx.ego_graph(...)
            # followed by number_of_edges().
            neighbor_neighbor_twice = sum(len(adj[v] & nbrs) for v in nbrs)
            internal_edges = du + neighbor_neighbor_twice / 2.0
            node_features[row, 4] = internal_edges

            # Upstream feature 5 counts edges leaving the ego node set.
            ego_nodes = nbrs | {u}
            sum_deg_ego = sum(degree[v] for v in ego_nodes)
            node_features[row, 5] = sum_deg_ego - 2.0 * internal_edges

            # Unique distance-two candidates, excluding u and direct neighbors.
            two_hop = set()
            for v in nbrs:
                two_hop.update(adj[v])
            two_hop.difference_update(nbrs)
            two_hop.discard(u)
            node_features[row, 6] = len(two_hop)
        else:
            node_features[row, 2:] = 0.0

    # Upstream function returns np.nan_to_num(node_features); the features above
    # are finite already, but keep the exact final guard.
    return np.nan_to_num(node_features)


def netsimile_graph_signature(node_features: np.ndarray) -> np.ndarray:
    """Official 35-dimensional NetSimile signature."""
    node_features = np.asarray(node_features, dtype=float)
    signature_vec = np.zeros(7 * 5, dtype=float)
    for k in range(7):
        col = node_features[:, k]
        signature_vec[k * 5] = col.mean()
        signature_vec[k * 5 + 1] = np.median(col)
        signature_vec[k * 5 + 2] = col.std()
        signature_vec[k * 5 + 3] = skew(col)
        signature_vec[k * 5 + 4] = kurtosis(col)
    return signature_vec


def netsimile_signature(graph: nx.Graph) -> np.ndarray:
    return netsimile_graph_signature(netsimile_feature_extraction(graph))


def netsimile_distance(sig1: np.ndarray, sig2: np.ndarray) -> float:
    # Keep upstream behavior, including NaN propagation for undefined moments
    # on perfectly constant features (e.g. a fully regular WS p=0 realization).
    return float(abs(canberra(np.asarray(sig1), np.asarray(sig2))))
