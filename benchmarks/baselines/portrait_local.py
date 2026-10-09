from __future__ import annotations

"""Memory-efficient exact reproduction of unweighted Network Portrait Divergence.

Reference implementation:
    bagrow/network-portrait-divergence
    commit 72993c368114c2e834142787579466d673232fa4

The upstream portrait matrix B is sparse in practice, but stored densely.  Here
only occupied (shell, shell-size) cells are stored.  The final probability mass
B[l,k] * k / sum(B[l,k]*k) is identical; zero-shell-size cells have zero weight
and therefore need not be materialized.  This is an efficiency-only change.
"""

from collections import Counter, defaultdict
import numpy as np
import networkx as nx
from scipy.stats import entropy

OFFICIAL_REPOSITORY = "https://github.com/bagrow/network-portrait-divergence"
OFFICIAL_COMMIT = "72993c368114c2e834142787579466d673232fa4"


def portrait_distribution_sparse(graph: nx.Graph) -> dict[tuple[int, int], float]:
    B: dict[tuple[int, int], float] = defaultdict(float)
    for source in graph.nodes():
        lengths = nx.single_source_shortest_path_length(graph, source)
        shell_counts = Counter(int(d) for d in lengths.values())
        for shell, k in shell_counts.items():
            B[(int(shell), int(k))] += 1.0

    weighted: dict[tuple[int, int], float] = {}
    total = 0.0
    for (shell, k), count in B.items():
        w = float(count) * float(k)
        if w != 0.0:
            weighted[(shell, k)] = w
            total += w
    if total == 0.0:
        return {}
    return {key: value / total for key, value in weighted.items()}


def portrait_divergence_sparse(
    rep_g: dict[tuple[int, int], float],
    rep_h: dict[tuple[int, int], float],
) -> float:
    keys = sorted(set(rep_g) | set(rep_h))
    if not keys:
        return 0.0
    P = np.asarray([rep_g.get(k, 0.0) for k in keys], dtype=float)
    Q = np.asarray([rep_h.get(k, 0.0) for k in keys], dtype=float)
    M = 0.5 * (P + Q)
    return float(0.5 * (entropy(P, M, base=2) + entropy(Q, M, base=2)))
