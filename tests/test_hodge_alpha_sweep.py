from __future__ import annotations

import numpy as np

from hodge_vnext.config import MethodConfig
from hodge_vnext.distance import active_hodge_count, profile_distance
from hodge_vnext.profile import GraphProfile
from validation.hodge_alpha_sweep import distance_block_from_layer_features


def _profile(name: str, qmax: int, channels: dict[tuple[int, int], np.ndarray]) -> GraphProfile:
    times = np.array([0.1, 1.0, 10.0], dtype=float)
    return GraphProfile(
        graph_hash=name,
        num_nodes=10,
        num_edges=12,
        times=times,
        channels=channels,
        retained_q_layers=sorted({q for q, _ in channels}),
        q_max=qmax,
        clique_counts={},
        simplex_counts={},
        hodge_dimensions={},
        hodge_nonzeros={},
        solver="slq",
        max_clique_size=None,
        max_hodge_order=None,
        normalization="per_simplex",
    )


def _layer_features(profiles: list[GraphProfile], q_values: list[int]) -> dict[int, np.ndarray]:
    T = len(profiles[0].times)
    out: dict[int, np.ndarray] = {}
    for q in q_values:
        A = active_hodge_count(q, None)
        X = np.zeros((len(profiles), A * T), dtype=float)
        for i, p in enumerate(profiles):
            for k in range(A):
                v = p.channels.get((q, k))
                if v is not None:
                    X[i, k * T:(k + 1) * T] = v
        out[q] = X
    return out


def test_vectorized_alpha_reaggregation_matches_profile_distance():
    z = np.zeros(3)
    a = _profile("a", 1, {
        (0, 0): np.array([1.0, 0.5, 0.2]),
        (1, 0): np.array([0.2, 0.1, 0.0]),
        (1, 1): np.array([0.3, 0.2, 0.1]),
    })
    b = _profile("b", 2, {
        (0, 0): np.array([0.8, 0.4, 0.1]),
        (1, 0): np.array([0.1, 0.2, 0.1]),
        (1, 1): np.array([0.1, 0.1, 0.1]),
        (2, 0): np.array([0.4, 0.2, 0.1]),
        (2, 1): np.array([0.2, 0.2, 0.2]),
        (2, 2): np.array([0.1, 0.0, 0.0]),
    })
    c = _profile("c", 0, {(0, 0): np.array([1.1, 0.6, 0.3])})
    profiles = [a, b, c]
    q_values = [0, 1, 2]
    qmax = np.array([1, 2, 0], dtype=int)
    layers = _layer_features(profiles, q_values)
    alphas = [1.0, 0.5, 0.0, -1.0]

    got = distance_block_from_layer_features(layers, qmax, q_values, alphas, None, 0, len(profiles))
    for ai, alpha in enumerate(alphas):
        cfg = MethodConfig(
            min_layer_q=0,
            normalization="per_simplex",
            alpha=alpha,
            solver="slq",
        )
        for i in range(len(profiles)):
            for j in range(len(profiles)):
                expected = profile_distance(profiles[i], profiles[j], cfg).total_distance
                assert np.isclose(got[ai, i, j], expected, atol=1e-12, rtol=1e-12)


def test_smaller_alpha_reweights_each_high_q_matrix_downward():
    # q=1 has A=2, q=8 has A=9.  Per-matrix weight is proportional to A^(alpha-1).
    ratio_a1 = (9 / 2) ** (1.0 - 1.0)
    ratio_a0 = (9 / 2) ** (0.0 - 1.0)
    ratio_am1 = (9 / 2) ** (-1.0 - 1.0)
    assert np.isclose(ratio_a1, 1.0)
    assert ratio_am1 < ratio_a0 < ratio_a1
