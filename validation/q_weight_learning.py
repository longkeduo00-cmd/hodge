from __future__ import annotations

"""Supervised learning of independent exclusive-q Hodge layer weights.

This module does not change the Hodge construction.  It learns only the final
cross-q aggregation weights from training labels.  For a graph pair with
pair-specific q* the learned logits are masked to q=0..q* and renormalized:

    w_q^(ij) = softmax(beta_0, ..., beta_q*)_q
    D_ij^2   = sum_{q=0}^{q*} w_q^(ij) e_q(i,j)

where e_q is the existing matrix-wise mean squared Hodge-signature distance for
that exclusive q-layer.  Thus the original pair-adaptive q schema is retained.
No smoothness/entropy/L1/L2 regularization is applied in this first version.
"""

from dataclasses import dataclass
from typing import Sequence
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class QWeightFitResult:
    beta: np.ndarray
    reference_weights: np.ndarray
    best_loss: float
    final_loss: float
    final_grad_norm: float
    epochs: int
    converged: bool
    stop_reason: str
    temperature: float
    trajectory: pd.DataFrame


def _validate_shapes(beta: np.ndarray, energies: np.ndarray, q_star: np.ndarray) -> np.ndarray:
    b = np.asarray(beta, dtype=float).reshape(-1)
    if energies.ndim != 3:
        raise ValueError("energies must have shape (n,n,num_q)")
    if q_star.shape != energies.shape[:2]:
        raise ValueError("q_star shape must match energies[:2]")
    if energies.shape[2] != len(b):
        raise ValueError(f"beta length {len(b)} != energy q dimension {energies.shape[2]}")
    return b


def softmax_weights(beta: Sequence[float]) -> np.ndarray:
    """Full-schema reference weights. Pairwise distances use masked renormalization."""
    b = np.asarray(beta, dtype=float).reshape(-1)
    if b.size == 0:
        return b.copy()
    z = b - float(np.max(b))
    w = np.exp(z)
    w /= float(np.sum(w))
    return w


def pair_effective_weights(beta: Sequence[float], q_star: int) -> np.ndarray:
    """Return q-dimensional effective weights for one pair, zero above q*."""
    b = np.asarray(beta, dtype=float).reshape(-1)
    out = np.zeros_like(b)
    if q_star < 0:
        return out
    top = min(int(q_star), len(b) - 1)
    z = b[: top + 1] - float(np.max(b[: top + 1]))
    w = np.exp(z)
    w /= float(np.sum(w))
    out[: top + 1] = w
    return out


def pair_distance_squared_and_grad_beta(
    beta: Sequence[float],
    e: np.ndarray,
    q_star: int,
) -> tuple[float, np.ndarray]:
    """One pair's D^2 and analytic derivative with respect to all beta_q."""
    b = np.asarray(beta, dtype=float).reshape(-1)
    grad = np.zeros_like(b)
    if q_star < 0:
        return 0.0, grad
    top = min(int(q_star), len(b) - 1)
    z = b[: top + 1] - float(np.max(b[: top + 1]))
    w = np.exp(z)
    w /= float(np.sum(w))
    eq = np.asarray(e[: top + 1], dtype=float)
    d2 = float(np.dot(w, eq))
    # d(sum_q w_q e_q)/d beta_r = w_r (e_r - D^2)
    grad[: top + 1] = w * (eq - d2)
    return max(d2, 0.0), grad


def distance_squared_and_grad_beta(
    beta: Sequence[float],
    energies: np.ndarray,
    q_star: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Full D^2 matrix and dD^2/dbeta tensor.

    Fully vectorized exact implementation of the pair-adaptive masked softmax.
    For every pair (i,j), only q<=q_star[i,j] is active and the active logits
    are renormalized exactly as in ``pair_effective_weights``.  Pair-specific
    prefix maxima are used for numerical stability, so this remains equivalent
    to the original per-pair implementation even when beta magnitudes grow.
    """
    b = _validate_shapes(np.asarray(beta, dtype=float), energies, q_star)
    num_q = len(b)
    tops = np.asarray(q_star, dtype=np.int32)
    safe_top = np.clip(tops, 0, num_q - 1)

    # max(beta_0..beta_q*) for each pair, matching the stable masked softmax.
    prefix_max = np.maximum.accumulate(b)
    pair_max = prefix_max[safe_top]
    q_index = np.arange(num_q, dtype=np.int32)[None, None, :]
    active = q_index <= tops[:, :, None]

    z = b[None, None, :] - pair_max[:, :, None]
    expz = np.where(active, np.exp(z), 0.0)
    denom = np.sum(expz, axis=2, keepdims=True)
    weights = np.divide(expz, denom, out=np.zeros_like(expz), where=denom > 0)

    D2 = np.sum(weights * np.asarray(energies, dtype=float), axis=2)
    D2 = np.maximum(D2, 0.0)
    G = weights * (np.asarray(energies, dtype=float) - D2[:, :, None])

    # q_star<0 denotes self-pairs/inactive schema. The mask already makes both
    # D2 and G exactly zero there.
    return D2, G


def distance_matrix_from_beta(
    beta: Sequence[float],
    energies: np.ndarray,
    q_star: np.ndarray,
) -> np.ndarray:
    D2, _ = distance_squared_and_grad_beta(beta, energies, q_star)
    return np.sqrt(np.maximum(D2, 0.0))


def beta_from_scalar_alpha(alpha: float, num_q: int, *, max_hodge_order: int | None = None) -> np.ndarray:
    """Embed the original scalar-alpha family in the free-q logit family.

    For Q=all, A_q=q+1.  More generally A_q=min(q,Q)+1.  With
    beta_q=alpha*log(A_q), masked softmax reproduces the original pair-adaptive
    W_q(alpha) exactly for every pair-specific q*.
    """
    if num_q < 1:
        return np.zeros(0, dtype=float)
    counts = []
    for q in range(num_q):
        A = q + 1 if max_hodge_order is None else min(q, int(max_hodge_order)) + 1
        counts.append(float(A))
    return float(alpha) * np.log(np.asarray(counts, dtype=float))


def _logsumexp(x: np.ndarray) -> tuple[float, np.ndarray]:
    m = float(np.max(x))
    ex = np.exp(x - m)
    s = float(np.sum(ex))
    return m + math.log(s), ex / s


def training_temperature(
    beta: Sequence[float],
    energies: np.ndarray,
    q_star: np.ndarray,
    train_idx: Sequence[int],
) -> float:
    """Fixed label-free scale from positive training-pair squared distances."""
    idx = np.asarray(train_idx, dtype=int)
    D2, _ = distance_squared_and_grad_beta(beta, energies, q_star)
    vals: list[float] = []
    for a in range(len(idx)):
        for bb in range(a + 1, len(idx)):
            v = float(D2[idx[a], idx[bb]])
            if np.isfinite(v) and v > 0:
                vals.append(v)
    return max(float(np.median(vals)) if vals else 1.0, np.finfo(float).tiny)


def nca_loss_and_grad_beta(
    beta: Sequence[float],
    energies: np.ndarray,
    q_star: np.ndarray,
    train_idx: Sequence[int],
    labels: Sequence[str],
    *,
    temperature: float,
) -> tuple[float, np.ndarray]:
    """NCA / soft-nearest-neighbour loss and analytic beta gradient.

    No regularization term is present by design in this first q-weight model.
    """
    b = _validate_shapes(np.asarray(beta, dtype=float), energies, q_star)
    idx = np.asarray(train_idx, dtype=int)
    y = np.asarray(labels, dtype=str)
    if len(idx) < 2:
        raise ValueError("q-weight learning requires at least two training networks")
    tau = float(temperature)
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("temperature must be finite and positive")

    D2, G = distance_squared_and_grad_beta(b, energies, q_star)
    losses: list[float] = []
    grads: list[np.ndarray] = []
    for i in idx:
        js = idx[idx != i]
        same = y[js] == y[i]
        if not np.any(same):
            raise ValueError(f"training network {i} has no same-domain peer")
        logits = -D2[i, js] / tau
        log_all, p_all = _logsumexp(logits)
        log_same, p_same = _logsumexp(logits[same])
        losses.append(log_all - log_same)

        # d logit_ij / d beta = - d D2_ij / d beta / tau
        ds = -G[i, js, :] / tau
        mean_all = np.sum(p_all[:, None] * ds, axis=0)
        mean_same = np.sum(p_same[:, None] * ds[same, :], axis=0)
        grads.append(mean_all - mean_same)

    return float(np.mean(losses)), np.mean(np.stack(grads, axis=0), axis=0)


def _trajectory_row(epoch: int, beta: np.ndarray, loss: float, grad: np.ndarray) -> dict[str, float | int]:
    w = softmax_weights(beta)
    row: dict[str, float | int] = {
        "epoch": int(epoch),
        "loss": float(loss),
        "grad_norm": float(np.linalg.norm(grad)),
    }
    for q in range(len(beta)):
        row[f"beta_q{q}"] = float(beta[q])
        row[f"weight_q{q}"] = float(w[q])
    return row


def fit_q_weights_adam(
    energies: np.ndarray,
    q_star: np.ndarray,
    train_idx: Sequence[int],
    labels: Sequence[str],
    *,
    learning_rate: float = 0.05,
    max_epochs: int = 2000,
    beta1: float = 0.9,
    beta2: float = 0.999,
    adam_eps: float = 1e-8,
    grad_tolerance: float = 1e-8,
    loss_tolerance: float = 1e-10,
    patience: int = 100,
) -> QWeightFitResult:
    """Learn free q-layer logits by Adam, initialized at uniform q weights.

    The softmax gauge is fixed after every update by subtracting mean(beta).
    This changes neither pairwise weights nor distances and introduces no model
    hyperparameter.  There are deliberately no regularization coefficients.
    """
    if max_epochs < 1 or learning_rate <= 0:
        raise ValueError("max_epochs and learning_rate must be positive")
    num_q = int(energies.shape[2])
    if num_q < 1:
        raise ValueError("q-weight learning requires at least one q layer")
    beta = np.zeros(num_q, dtype=float)
    tau = training_temperature(beta, energies, q_star, train_idx)

    m = np.zeros_like(beta)
    v = np.zeros_like(beta)
    best_beta = beta.copy()
    best_loss = float("inf")
    stale = 0
    last_loss: float | None = None
    rows: list[dict[str, float | int]] = []
    converged = False
    stop_reason = "max_epochs"
    final_grad = np.full_like(beta, np.nan)

    for epoch in range(1, int(max_epochs) + 1):
        loss, grad = nca_loss_and_grad_beta(
            beta, energies, q_star, train_idx, labels, temperature=tau
        )
        if not np.isfinite(loss) or not np.all(np.isfinite(grad)):
            raise FloatingPointError(f"non-finite q-weight state at epoch={epoch}: loss={loss}, grad={grad}")
        rows.append(_trajectory_row(epoch, beta, loss, grad))
        final_grad = grad.copy()

        if loss + loss_tolerance < best_loss:
            best_loss = float(loss)
            best_beta = beta.copy()
            stale = 0
        else:
            stale += 1

        grad_norm = float(np.linalg.norm(grad))
        if grad_norm <= grad_tolerance:
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
        beta -= learning_rate * m_hat / (np.sqrt(v_hat) + adam_eps)
        # Gauge fixing: softmax(beta+c) is identical.  Centering prevents an
        # arbitrary common drift and makes saved beta values interpretable.
        beta -= float(np.mean(beta))
        last_loss = float(loss)

    best_final_loss, best_final_grad = nca_loss_and_grad_beta(
        best_beta, energies, q_star, train_idx, labels, temperature=tau
    )
    return QWeightFitResult(
        beta=best_beta.copy(),
        reference_weights=softmax_weights(best_beta),
        best_loss=float(best_loss),
        final_loss=float(best_final_loss),
        final_grad_norm=float(np.linalg.norm(best_final_grad)),
        epochs=int(len(rows)),
        converged=bool(converged),
        stop_reason=stop_reason,
        temperature=float(tau),
        trajectory=pd.DataFrame(rows),
    )


def weight_diagnostics(weights: Sequence[float]) -> dict[str, float | int]:
    w = np.asarray(weights, dtype=float)
    positive = w[w > 0]
    entropy = float(-np.sum(positive * np.log(positive))) if len(positive) else 0.0
    return {
        "weight_entropy": entropy,
        "effective_num_layers": float(np.exp(entropy)),
        "max_weight_q": int(np.argmax(w)) if len(w) else -1,
        "max_weight": float(np.max(w)) if len(w) else float("nan"),
    }


def average_effective_pair_weights(
    beta: Sequence[float],
    q_star: np.ndarray,
    left_idx: Sequence[int],
    right_idx: Sequence[int] | None = None,
    *,
    exclude_diagonal: bool = True,
) -> np.ndarray:
    """Average actual pair-adaptive weights over a requested pair collection."""
    b = np.asarray(beta, dtype=float).reshape(-1)
    left = np.asarray(left_idx, dtype=int)
    right = left if right_idx is None else np.asarray(right_idx, dtype=int)
    total = np.zeros_like(b)
    count = 0
    same_collection = right_idx is None
    for ai, i in enumerate(left):
        for bj, j in enumerate(right):
            if exclude_diagonal and i == j:
                continue
            if same_collection and bj <= ai:
                continue
            total += pair_effective_weights(b, int(q_star[i, j]))
            count += 1
    return total / float(count) if count else total
