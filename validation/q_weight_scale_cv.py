from __future__ import annotations

"""Core utilities for train-only q-layer scale normalization diagnostics.

The cached independent-q Hodge tensor used by this project stores a *squared*
layer contribution C_q(i,j):

    C_q(i,j) = (1/A_q) sum_k ||phi_{q,k}(G_i)-phi_{q,k}(G_j)||_2^2.

The experiment design describes an RMS scale for the corresponding layer
distance d_q=sqrt(C_q):

    s_q = sqrt(mean_train_pairs d_q^2) = sqrt(mean_train_pairs C_q).

Therefore the normalized squared contribution entering D^2 is exactly

    Ctilde_q(i,j) = C_q(i,j) / (s_q + eps)^2.

This module implements that mathematically equivalent form without changing the
Hodge core or rebuilding profiles.  All scale estimates use train-train pairs
only and respect the existing pair-specific q* schema.
"""

from dataclasses import dataclass
from typing import Sequence
import math

import numpy as np
import pandas as pd

from .q_weight_learning import (
    distance_matrix_from_beta,
    nca_loss_and_grad_beta,
    training_temperature,
)


@dataclass(frozen=True)
class LayerScaleResult:
    distance_scale: np.ndarray
    squared_denominator: np.ndarray
    active_train_pairs: np.ndarray
    mean_train_squared_distance: np.ndarray
    eps: float


@dataclass(frozen=True)
class SingleAlphaFitResult:
    alpha: float
    beta: np.ndarray
    best_loss: float
    temperature: float
    boundary_hit: bool
    trajectory: pd.DataFrame


def _active_train_pair_values(
    squared_layers: np.ndarray,
    q_star: np.ndarray,
    train_idx: Sequence[int],
    q: int,
) -> np.ndarray:
    idx = np.asarray(train_idx, dtype=int)
    vals: list[float] = []
    for a in range(len(idx)):
        i = int(idx[a])
        for b in range(a + 1, len(idx)):
            j = int(idx[b])
            if int(q_star[i, j]) >= int(q):
                vals.append(float(squared_layers[i, j, q]))
    return np.asarray(vals, dtype=float)


def estimate_train_only_layer_scales(
    squared_layers: np.ndarray,
    q_star: np.ndarray,
    train_idx: Sequence[int],
    *,
    eps: float = 1e-8,
    require_identifiable: bool = True,
) -> LayerScaleResult:
    """Estimate one RMS scale per q using train-train active pairs only.

    ``squared_layers`` contains C_q, already a squared layer distance.  The
    returned ``distance_scale`` is s_q=sqrt(mean(C_q)), matching the design
    document's RMS normalization of d_q=sqrt(C_q).  The denominator applied to
    C_q is (s_q+eps)^2.
    """
    C = np.asarray(squared_layers, dtype=float)
    qs = np.asarray(q_star, dtype=np.int32)
    if C.ndim != 3 or qs.shape != C.shape[:2]:
        raise ValueError("squared_layers must have shape (n,n,q) and q_star shape (n,n)")
    if eps <= 0 or not np.isfinite(eps):
        raise ValueError("eps must be finite and positive")

    num_q = int(C.shape[2])
    scales = np.zeros(num_q, dtype=float)
    denom = np.zeros(num_q, dtype=float)
    counts = np.zeros(num_q, dtype=np.int64)
    means = np.zeros(num_q, dtype=float)

    for q in range(num_q):
        vals = _active_train_pair_values(C, qs, train_idx, q)
        if vals.size == 0:
            if require_identifiable:
                raise ValueError(f"q={q} has no active train-train pairs; train-only scale is not identifiable")
            means[q] = 0.0
            scales[q] = 1.0
            denom[q] = 1.0
            continue
        if not np.all(np.isfinite(vals)) or np.any(vals < -1e-12):
            raise ValueError(f"q={q} contains invalid squared distances")
        vals = np.maximum(vals, 0.0)
        mean_c = float(np.mean(vals))
        if mean_c <= 0.0:
            if require_identifiable:
                raise ValueError(f"q={q} has zero RMS train-only scale")
            means[q] = 0.0
            scales[q] = 1.0
            denom[q] = 1.0
            counts[q] = int(vals.size)
            continue
        s = math.sqrt(mean_c)
        means[q] = mean_c
        scales[q] = s
        denom[q] = (s + float(eps)) ** 2
        counts[q] = int(vals.size)

    return LayerScaleResult(
        distance_scale=scales,
        squared_denominator=denom,
        active_train_pairs=counts,
        mean_train_squared_distance=means,
        eps=float(eps),
    )


def apply_layer_scales_to_squared_distances(
    squared_layers: np.ndarray,
    scale: LayerScaleResult,
) -> np.ndarray:
    C = np.asarray(squared_layers, dtype=float)
    if C.ndim != 3 or C.shape[2] != len(scale.squared_denominator):
        raise ValueError("scale q dimension does not match squared_layers")
    return C / scale.squared_denominator[None, None, :]


def layer_scale_table(scale: LayerScaleResult) -> pd.DataFrame:
    rows = []
    for q in range(len(scale.distance_scale)):
        rows.append({
            "q": q,
            "clique_size": q + 1,
            "active_train_pair_count": int(scale.active_train_pairs[q]),
            "mean_train_squared_distance": float(scale.mean_train_squared_distance[q]),
            "rms_distance_scale": float(scale.distance_scale[q]),
            "squared_distance_denominator": float(scale.squared_denominator[q]),
            "eps": float(scale.eps),
        })
    return pd.DataFrame(rows)


def single_alpha_beta(alpha: float, num_q: int) -> np.ndarray:
    """One-parameter monotone order trend from the experiment design.

    x_q=q/(num_q-1), beta_q=alpha*x_q.  Pair-specific masking/renormalization
    is applied later by ``distance_matrix_from_beta``.  This alpha is a new
    diagnostic parameterization and is intentionally distinct from the legacy
    A_q**alpha Hodge aggregation parameter.
    """
    if num_q < 1:
        return np.zeros(0, dtype=float)
    if num_q == 1:
        return np.zeros(1, dtype=float)
    x = np.arange(num_q, dtype=float) / float(num_q - 1)
    return float(alpha) * x


def _inclusive_grid(lo: float, hi: float, step: float) -> np.ndarray:
    if step <= 0 or hi < lo:
        raise ValueError("invalid grid bounds/step")
    n = int(math.floor((hi - lo) / step + 1e-12))
    vals = lo + step * np.arange(n + 1, dtype=float)
    if vals[-1] < hi - 1e-10:
        vals = np.append(vals, hi)
    vals[0] = lo
    vals[-1] = hi
    return vals


def fit_single_alpha_train_only(
    squared_layers: np.ndarray,
    q_star: np.ndarray,
    train_idx: Sequence[int],
    labels: Sequence[str],
    *,
    coarse_min: float = -10.0,
    coarse_max: float = 10.0,
    coarse_step: float = 1.0,
    fine_half_width: float = 1.0,
    fine_step: float = 0.05,
) -> SingleAlphaFitResult:
    """Deterministic coarse+fine train-only NCA search for the 1-D alpha model."""
    C = np.asarray(squared_layers, dtype=float)
    num_q = int(C.shape[2])
    beta0 = single_alpha_beta(0.0, num_q)
    tau = training_temperature(beta0, C, np.asarray(q_star), train_idx)

    rows: list[dict[str, float | str]] = []

    def evaluate(values: np.ndarray, stage: str) -> tuple[float, float]:
        best_a = float(values[0])
        best_loss = float("inf")
        for a in values:
            beta = single_alpha_beta(float(a), num_q)
            loss, _ = nca_loss_and_grad_beta(
                beta, C, np.asarray(q_star), train_idx, labels, temperature=tau
            )
            rows.append({"stage": stage, "alpha": float(a), "nca_loss": float(loss)})
            # Deterministic tie break prefers the smaller |alpha|, then smaller alpha.
            if (
                loss < best_loss - 1e-14
                or (abs(loss - best_loss) <= 1e-14 and abs(float(a)) < abs(best_a) - 1e-14)
                or (abs(loss - best_loss) <= 1e-14 and abs(abs(float(a)) - abs(best_a)) <= 1e-14 and float(a) < best_a)
            ):
                best_a = float(a)
                best_loss = float(loss)
        return best_a, best_loss

    coarse_vals = _inclusive_grid(float(coarse_min), float(coarse_max), float(coarse_step))
    coarse_best, _ = evaluate(coarse_vals, "coarse")

    fine_lo = max(float(coarse_min), coarse_best - float(fine_half_width))
    fine_hi = min(float(coarse_max), coarse_best + float(fine_half_width))
    fine_vals = _inclusive_grid(fine_lo, fine_hi, float(fine_step))
    best_alpha, best_loss = evaluate(fine_vals, "fine")

    boundary_hit = bool(
        abs(best_alpha - float(coarse_min)) <= max(1e-12, fine_step * 0.51)
        or abs(best_alpha - float(coarse_max)) <= max(1e-12, fine_step * 0.51)
    )
    traj = pd.DataFrame(rows)
    traj["selected"] = np.isclose(traj["alpha"], best_alpha, atol=1e-12) & np.isclose(traj["nca_loss"], best_loss, atol=1e-12)
    return SingleAlphaFitResult(
        alpha=float(best_alpha),
        beta=single_alpha_beta(best_alpha, num_q),
        best_loss=float(best_loss),
        temperature=float(tau),
        boundary_hit=boundary_hit,
        trajectory=traj,
    )


def distance_matrix_single_alpha(
    alpha: float,
    squared_layers: np.ndarray,
    q_star: np.ndarray,
) -> np.ndarray:
    return distance_matrix_from_beta(single_alpha_beta(alpha, squared_layers.shape[2]), squared_layers, q_star)


def cosine_similarity_rows(weight_matrix: np.ndarray) -> np.ndarray:
    W = np.asarray(weight_matrix, dtype=float)
    if W.ndim != 2:
        raise ValueError("weight_matrix must be 2D")
    norms = np.linalg.norm(W, axis=1)
    denom = norms[:, None] * norms[None, :]
    return np.divide(W @ W.T, denom, out=np.zeros((len(W), len(W))), where=denom > 0)
