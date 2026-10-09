from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .config import MethodConfig
from .profile import GraphProfile


@dataclass
class DistanceResult:
    total_distance: float
    q_star: int | None
    layer_weights: dict[int, float]
    matrix_weights: dict[tuple[int, int], float]
    layer_distances: dict[int, float]
    matrix_distances: dict[tuple[int, int], float]


def _pair_q_star(a: GraphProfile, b: GraphProfile, min_layer_q: int) -> int | None:
    candidates = [q for q in (a.q_max, b.q_max) if q is not None]
    if not candidates:
        return None
    q_star = max(candidates)
    return q_star if q_star >= min_layer_q else None


def active_hodge_count(q: int, max_hodge_order: int | None) -> int:
    return q + 1 if max_hodge_order is None else min(q, max_hodge_order) + 1


def pair_weights(q_star: int, cfg: MethodConfig) -> tuple[dict[int, float], dict[tuple[int, int], float]]:
    qs = list(range(cfg.min_layer_q, q_star + 1))
    counts = {q: active_hodge_count(q, cfg.max_hodge_order) for q in qs}
    # W_q=A_q^alpha/sum_r A_r^alpha; every L_k in q receives W_q/A_q.
    denom = sum(float(counts[q]) ** cfg.alpha for q in qs)
    if denom <= 0:
        raise RuntimeError("Invalid layer-weight denominator.")
    layer_weights = {q: (float(counts[q]) ** cfg.alpha) / denom for q in qs}
    matrix_weights: dict[tuple[int, int], float] = {}
    for q in qs:
        per_matrix = layer_weights[q] / counts[q]
        for k in range(counts[q]):
            matrix_weights[(q, k)] = per_matrix
    return layer_weights, matrix_weights


def weighted_feature(profile: GraphProfile, q_star: int, cfg: MethodConfig) -> np.ndarray:
    """Flatten one profile into the pair-adaptive weighted Euclidean space.

    A missing layer/matrix contributes an all-zero time signature.  The feature
    uses sqrt(matrix_weight), so ordinary Euclidean distance between two such
    features is exactly the final weighted distance for this q_star.
    """
    _, matrix_weights = pair_weights(q_star, cfg)
    zero = np.zeros_like(profile.times)
    chunks: list[np.ndarray] = []
    for q in range(cfg.min_layer_q, q_star + 1):
        for k in range(active_hodge_count(q, cfg.max_hodge_order)):
            trace = profile.channels.get((q, k), zero)
            chunks.append(np.sqrt(matrix_weights[(q, k)]) * trace)
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=float)


def profile_distance(a: GraphProfile, b: GraphProfile, cfg: MethodConfig) -> DistanceResult:
    if a.solver != b.solver or a.solver != cfg.solver:
        raise ValueError("Both profiles and the run config must use the same global solver.")
    if not np.allclose(a.times, b.times):
        raise ValueError("Profiles must use the same time grid.")
    q_star = _pair_q_star(a, b, cfg.min_layer_q)
    if q_star is None:
        return DistanceResult(0.0, None, {}, {}, {}, {})

    layer_weights, matrix_weights = pair_weights(q_star, cfg)
    zero = np.zeros_like(a.times)
    matrix_distances: dict[tuple[int, int], float] = {}
    layer_sq_unweighted: dict[int, float] = {}
    total_sq = 0.0

    # Pair-adaptive continuous schema: every q from min_layer_q to q_star is present.
    # Missing signatures are zero. Common absence is therefore similarity information.
    for q in range(cfg.min_layer_q, q_star + 1):
        count = active_hodge_count(q, cfg.max_hodge_order)
        local_sq = 0.0
        for k in range(count):
            va = a.channels.get((q, k), zero)
            vb = b.channels.get((q, k), zero)
            d = float(np.linalg.norm(va - vb, ord=2))
            matrix_distances[(q, k)] = d
            local_sq += d * d
            total_sq += matrix_weights[(q, k)] * d * d
        layer_sq_unweighted[q] = local_sq

    layer_distances = {q: math.sqrt(v) for q, v in layer_sq_unweighted.items()}
    return DistanceResult(
        total_distance=math.sqrt(max(total_sq, 0.0)),
        q_star=q_star,
        layer_weights=layer_weights,
        matrix_weights=matrix_weights,
        layer_distances=layer_distances,
        matrix_distances=matrix_distances,
    )
