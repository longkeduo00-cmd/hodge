from __future__ import annotations

import numpy as np

from validation.alpha_gradient import (
    _pair_d2_grad,
    distance_matrix_from_alpha,
    fit_alpha_adam,
    nca_loss_and_grad,
)


def test_pair_gradient_matches_finite_difference():
    e = np.array([0.2, 1.3, 2.1, 0.8], dtype=float)
    alpha = -3.7
    d2, grad = _pair_d2_grad(alpha, e, 3, min_layer_q=0, max_hodge_order=None)
    h = 1e-6
    d2p, _ = _pair_d2_grad(alpha + h, e, 3, min_layer_q=0, max_hodge_order=None)
    d2m, _ = _pair_d2_grad(alpha - h, e, 3, min_layer_q=0, max_hodge_order=None)
    numeric = (d2p - d2m) / (2 * h)
    assert np.isclose(grad, numeric, rtol=1e-6, atol=1e-8)
    assert d2 >= 0


def _toy_energies():
    # Two classes, two samples each. q=0 separates classes; q=1 is deliberately
    # misleading. Optimizing alpha should therefore move strongly negative with
    # no explicit alpha lower bound.
    n = 4
    E = np.zeros((n, n, 2), dtype=float)
    qs = np.ones((n, n), dtype=np.int32)
    np.fill_diagonal(qs, -1)
    y = ["A", "A", "B", "B"]
    for i in range(n):
        for j in range(i + 1, n):
            same = y[i] == y[j]
            E[i, j, 0] = E[j, i, 0] = 0.05 if same else 4.0
            E[i, j, 1] = E[j, i, 1] = 4.0 if same else 0.05
    return E, qs, y


def test_nca_gradient_matches_finite_difference():
    E, qs, y = _toy_energies()
    idx = np.arange(4)
    alpha = -1.3
    loss, grad = nca_loss_and_grad(alpha, E, qs, idx, y, temperature=1.0)
    h = 1e-6
    lp, _ = nca_loss_and_grad(alpha + h, E, qs, idx, y, temperature=1.0)
    lm, _ = nca_loss_and_grad(alpha - h, E, qs, idx, y, temperature=1.0)
    numeric = (lp - lm) / (2 * h)
    assert np.isclose(grad, numeric, rtol=1e-5, atol=1e-7)
    assert loss >= 0


def test_adam_learning_is_unbounded_and_can_move_below_old_grid():
    E, qs, y = _toy_energies()
    fit = fit_alpha_adam(
        E, qs, np.arange(4), y,
        initial_alpha=0.0,
        learning_rate=0.1,
        max_epochs=400,
        patience=80,
        grad_tolerance=1e-10,
        loss_tolerance=1e-12,
    )
    assert fit.alpha < -2.0
    assert "alpha" in fit.trajectory.columns


def test_distance_matrix_has_expected_symmetry():
    E, qs, _ = _toy_energies()
    D = distance_matrix_from_alpha(-5.0, E, qs)
    assert np.allclose(D, D.T)
    assert np.allclose(np.diag(D), 0.0)
    assert np.all(D >= 0)
