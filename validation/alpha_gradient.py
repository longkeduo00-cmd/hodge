from __future__ import annotations

"""Continuous alpha learning for the independent-q Hodge distance.

The Hodge mathematical core is not changed.  This module only exploits the
existing alpha-parameterized aggregation

    D^2 = sum_q W_q(alpha) * e_q,
    W_q(alpha) = softmax_q(alpha * log A_q),
    e_q = (1/A_q) * sum_k d_{q,k}^2,

to learn the single continuous scalar alpha from training labels.  The
optimization objective is a differentiable NCA / soft-nearest-neighbour loss;
non-differentiable reporting metrics (1-NN Macro-F1, AUROC, mAP, ...) remain
strictly evaluation metrics.
"""

from dataclasses import dataclass
from typing import Sequence
import math

import numpy as np
import pandas as pd

from hodge_vnext.distance import active_hodge_count
from hodge_vnext.profile import GraphProfile


@dataclass(frozen=True)
class AlphaFitResult:
    alpha: float
    best_loss: float
    final_loss: float
    final_grad: float
    epochs: int
    converged: bool
    stop_reason: str
    temperature: float
    trajectory: pd.DataFrame


def build_pair_layer_energies(
    profiles: Sequence[GraphProfile],
    *,
    min_layer_q: int = 0,
    max_hodge_order: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute label-free per-pair layer energies.

    Returns
    -------
    energies:
        ``(n,n,q_global+1)`` array.  For an active q,
        ``energies[i,j,q] = mean_k ||phi_i(q,k)-phi_j(q,k)||_2^2``.
    q_star:
        ``(n,n)`` integer array holding pair-specific q*.  ``-1`` means that
        the pair has no active layer at/above ``min_layer_q``.

    The final Hodge distance for any alpha is reconstructed exactly from these
    arrays.  No labels are used here.
    """
    if min_layer_q < 0:
        raise ValueError("min_layer_q must be >=0")
    n = len(profiles)
    if n == 0:
        return np.zeros((0, 0, 0), dtype=float), np.zeros((0, 0), dtype=np.int32)
    solver = profiles[0].solver
    times = profiles[0].times
    for p in profiles:
        if p.solver != solver:
            raise ValueError("all profiles must use the same global solver")
        if not np.allclose(p.times, times):
            raise ValueError("all profiles must use the same time grid")

    q_global = max((p.q_max for p in profiles if p.q_max is not None), default=-1)
    energies = np.zeros((n, n, max(q_global + 1, 0)), dtype=float)
    q_star = np.full((n, n), -1, dtype=np.int32)
    zero = np.zeros_like(times)

    for i in range(n):
        for j in range(i + 1, n):
            candidates = [q for q in (profiles[i].q_max, profiles[j].q_max) if q is not None]
            if not candidates:
                continue
            qs = max(candidates)
            if qs < min_layer_q:
                continue
            q_star[i, j] = q_star[j, i] = int(qs)
            for q in range(min_layer_q, qs + 1):
                A = active_hodge_count(q, max_hodge_order)
                local_sq = 0.0
                for k in range(A):
                    a = profiles[i].channels.get((q, k), zero)
                    b = profiles[j].channels.get((q, k), zero)
                    diff = a - b
                    local_sq += float(np.dot(diff, diff))
                energies[i, j, q] = energies[j, i, q] = local_sq / float(A)
    return energies, q_star


def _pair_d2_grad(
    alpha: float,
    e: np.ndarray,
    q_star: int,
    *,
    min_layer_q: int = 0,
    max_hodge_order: int | None = None,
) -> tuple[float, float]:
    if q_star < min_layer_q:
        return 0.0, 0.0
    qs = np.arange(min_layer_q, q_star + 1, dtype=int)
    counts = np.array([active_hodge_count(int(q), max_hodge_order) for q in qs], dtype=float)
    logA = np.log(counts)
    z = float(alpha) * logA
    z -= np.max(z)  # exact softmax stabilization; imposes no alpha bound.
    w = np.exp(z)
    w /= np.sum(w)
    eq = e[qs]
    d2 = float(np.dot(w, eq))
    mean_logA = float(np.dot(w, logA))
    # dD^2/dalpha = sum_q W_q (log A_q - E_W[log A]) e_q.
    grad = float(np.dot(w * (logA - mean_logA), eq))
    return max(d2, 0.0), grad


def distance_squared_and_grad(
    alpha: float,
    energies: np.ndarray,
    q_star: np.ndarray,
    *,
    min_layer_q: int = 0,
    max_hodge_order: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return full D^2(alpha) and d D^2 / d alpha matrices."""
    n = int(q_star.shape[0])
    D2 = np.zeros((n, n), dtype=float)
    G = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d2, g = _pair_d2_grad(
                alpha, energies[i, j], int(q_star[i, j]),
                min_layer_q=min_layer_q,
                max_hodge_order=max_hodge_order,
            )
            D2[i, j] = D2[j, i] = d2
            G[i, j] = G[j, i] = g
    return D2, G


def distance_matrix_from_alpha(
    alpha: float,
    energies: np.ndarray,
    q_star: np.ndarray,
    *,
    min_layer_q: int = 0,
    max_hodge_order: int | None = None,
) -> np.ndarray:
    D2, _ = distance_squared_and_grad(
        alpha, energies, q_star,
        min_layer_q=min_layer_q,
        max_hodge_order=max_hodge_order,
    )
    return np.sqrt(np.maximum(D2, 0.0))


def training_temperature(
    alpha: float,
    energies: np.ndarray,
    q_star: np.ndarray,
    train_idx: Sequence[int],
    *,
    min_layer_q: int = 0,
    max_hodge_order: int | None = None,
    multiplier: float = 1.0,
) -> float:
    """Robust label-free distance scale, estimated from training pairs only."""
    idx = np.asarray(train_idx, dtype=int)
    D2, _ = distance_squared_and_grad(
        alpha, energies, q_star,
        min_layer_q=min_layer_q,
        max_hodge_order=max_hodge_order,
    )
    vals: list[float] = []
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            v = float(D2[idx[a], idx[b]])
            if np.isfinite(v) and v > 0:
                vals.append(v)
    base = float(np.median(vals)) if vals else 1.0
    tau = base * float(multiplier)
    return max(tau, np.finfo(float).tiny)


def _logsumexp(x: np.ndarray) -> tuple[float, np.ndarray]:
    m = float(np.max(x))
    ex = np.exp(x - m)
    s = float(np.sum(ex))
    probs = ex / s
    return m + math.log(s), probs


def nca_loss_and_grad(
    alpha: float,
    energies: np.ndarray,
    q_star: np.ndarray,
    train_idx: Sequence[int],
    labels: Sequence[str],
    *,
    temperature: float,
    min_layer_q: int = 0,
    max_hodge_order: int | None = None,
) -> tuple[float, float]:
    """Differentiable soft-nearest-neighbour/NCA objective and analytic gradient."""
    idx = np.asarray(train_idx, dtype=int)
    y = np.asarray(labels, dtype=str)
    if len(idx) < 2:
        raise ValueError("alpha learning requires at least two training networks")
    D2, G = distance_squared_and_grad(
        alpha, energies, q_star,
        min_layer_q=min_layer_q,
        max_hodge_order=max_hodge_order,
    )
    tau = float(temperature)
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("temperature must be finite and positive")

    losses: list[float] = []
    grads: list[float] = []
    for i in idx:
        js = idx[idx != i]
        same = y[js] == y[i]
        if not np.any(same):
            raise ValueError(f"training network {i} has no same-domain peer")
        logits = -D2[i, js] / tau
        log_all, p_all = _logsumexp(logits)
        log_same, p_same_local = _logsumexp(logits[same])
        losses.append(log_all - log_same)

        gd2 = G[i, js]
        # d(logit)/dalpha = -d(D^2)/dalpha / tau.
        ds = -gd2 / tau
        mean_all = float(np.dot(p_all, ds))
        mean_same = float(np.dot(p_same_local, ds[same]))
        grads.append(mean_all - mean_same)

    return float(np.mean(losses)), float(np.mean(grads))


def fit_alpha_adam(
    energies: np.ndarray,
    q_star: np.ndarray,
    train_idx: Sequence[int],
    labels: Sequence[str],
    *,
    initial_alpha: float = 0.0,
    learning_rate: float = 0.05,
    max_epochs: int = 2000,
    beta1: float = 0.9,
    beta2: float = 0.999,
    adam_eps: float = 1e-8,
    grad_tolerance: float = 1e-8,
    loss_tolerance: float = 1e-10,
    patience: int = 100,
    temperature_multiplier: float = 1.0,
    min_layer_q: int = 0,
    max_hodge_order: int | None = None,
) -> AlphaFitResult:
    """Learn alpha on the full real line with Adam; no alpha search range/bounds."""
    if max_epochs < 1 or learning_rate <= 0:
        raise ValueError("max_epochs and learning_rate must be positive")
    alpha = float(initial_alpha)
    tau = training_temperature(
        alpha, energies, q_star, train_idx,
        min_layer_q=min_layer_q,
        max_hodge_order=max_hodge_order,
        multiplier=temperature_multiplier,
    )
    m = 0.0
    v = 0.0
    best_alpha = alpha
    best_loss = float("inf")
    stale = 0
    last_loss: float | None = None
    rows: list[dict[str, float | int]] = []
    converged = False
    stop_reason = "max_epochs"
    final_grad = float("nan")
    final_loss = float("nan")

    for epoch in range(1, int(max_epochs) + 1):
        loss, grad = nca_loss_and_grad(
            alpha, energies, q_star, train_idx, labels,
            temperature=tau,
            min_layer_q=min_layer_q,
            max_hodge_order=max_hodge_order,
        )
        if not np.isfinite(loss) or not np.isfinite(grad):
            raise FloatingPointError(f"non-finite alpha-learning state at epoch={epoch}: loss={loss}, grad={grad}")
        rows.append({"epoch": epoch, "alpha": alpha, "loss": loss, "grad": grad})
        final_loss, final_grad = loss, grad

        if loss + loss_tolerance < best_loss:
            best_loss = loss
            best_alpha = alpha
            stale = 0
        else:
            stale += 1

        if abs(grad) <= grad_tolerance:
            converged = True
            stop_reason = "gradient_tolerance"
            break
        if last_loss is not None and abs(last_loss - loss) <= loss_tolerance and stale >= patience:
            converged = True
            stop_reason = "loss_plateau"
            break
        if stale >= patience * 5:
            stop_reason = "no_improvement_patience"
            break

        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad * grad)
        m_hat = m / (1.0 - beta1 ** epoch)
        v_hat = v / (1.0 - beta2 ** epoch)
        alpha -= learning_rate * m_hat / (math.sqrt(v_hat) + adam_eps)
        last_loss = loss

    # Evaluate the best observed alpha once so the returned loss/grad correspond
    # exactly to the parameter used downstream.
    best_final_loss, best_final_grad = nca_loss_and_grad(
        best_alpha, energies, q_star, train_idx, labels,
        temperature=tau,
        min_layer_q=min_layer_q,
        max_hodge_order=max_hodge_order,
    )
    traj = pd.DataFrame(rows)
    return AlphaFitResult(
        alpha=float(best_alpha),
        best_loss=float(best_loss),
        final_loss=float(best_final_loss),
        final_grad=float(best_final_grad),
        epochs=int(len(rows)),
        converged=bool(converged),
        stop_reason=stop_reason,
        temperature=float(tau),
        trajectory=traj,
    )
