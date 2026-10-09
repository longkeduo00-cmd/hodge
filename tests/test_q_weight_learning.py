from __future__ import annotations

import numpy as np

from validation.q_weight_learning import (
    beta_from_scalar_alpha,
    distance_matrix_from_beta,
    distance_squared_and_grad_beta,
    pair_distance_squared_and_grad_beta,
    fit_q_weights_adam,
    nca_loss_and_grad_beta,
    pair_effective_weights,
    softmax_weights,
)


def _feature_energies(features_by_q: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    n = len(features_by_q[0])
    qn = len(features_by_q)
    e = np.zeros((n, n, qn), dtype=float)
    for q, x in enumerate(features_by_q):
        x = np.asarray(x, dtype=float)
        e[:, :, q] = (x[:, None] - x[None, :]) ** 2
    q_star = np.full((n, n), qn - 1, dtype=np.int32)
    np.fill_diagonal(q_star, -1)
    return e, q_star


def test_uniform_beta_is_uniform_full_schema_weight():
    w = softmax_weights(np.zeros(10))
    assert np.allclose(w, np.full(10, 0.1))


def test_pair_specific_mask_renormalizes_and_ignores_higher_q():
    beta = np.array([-5.0, 10.0, 20.0])
    w = pair_effective_weights(beta, q_star=0)
    assert np.allclose(w, [1.0, 0.0, 0.0])


def test_scalar_alpha_family_embedding_matches_direct_pair_formula():
    e, q_star = _feature_energies([
        np.array([0.0, 1.0, 4.0]),
        np.array([0.0, 3.0, 2.0]),
        np.array([2.0, 0.0, 5.0]),
    ])
    for alpha in (-16.0, 0.0, 2.0):
        beta = beta_from_scalar_alpha(alpha, 3, max_hodge_order=None)
        D = distance_matrix_from_beta(beta, e, q_star)
        A = np.array([1.0, 2.0, 3.0])
        w = A ** alpha
        w /= w.sum()
        expected = np.sqrt(np.maximum(np.tensordot(e, w, axes=([2], [0])), 0.0))
        np.fill_diagonal(expected, 0.0)
        assert np.allclose(D, expected, atol=1e-12, rtol=1e-12)


def test_nca_beta_gradient_matches_finite_difference():
    e, q_star = _feature_energies([
        np.array([0.0, 0.2, 3.0, 3.2]),
        np.array([0.0, 2.0, 0.1, 2.1]),
        np.array([1.0, 0.0, 1.2, 0.2]),
    ])
    labels = np.array(["A", "A", "B", "B"])
    idx = np.arange(4)
    beta = np.array([0.2, -0.1, 0.05])
    loss, grad = nca_loss_and_grad_beta(beta, e, q_star, idx, labels, temperature=1.3)
    eps = 1e-6
    fd = np.zeros_like(beta)
    for k in range(len(beta)):
        bp = beta.copy(); bm = beta.copy()
        bp[k] += eps; bm[k] -= eps
        lp, _ = nca_loss_and_grad_beta(bp, e, q_star, idx, labels, temperature=1.3)
        lm, _ = nca_loss_and_grad_beta(bm, e, q_star, idx, labels, temperature=1.3)
        fd[k] = (lp - lm) / (2 * eps)
    assert np.isfinite(loss)
    assert np.allclose(grad, fd, atol=2e-6, rtol=2e-5)
    assert abs(float(np.sum(grad))) < 1e-10  # softmax gauge invariance


def test_adam_learns_informative_q_layer_without_regularization():
    # q=0 separates A/B strongly; q=1 deliberately mixes the two domains.
    e, q_star = _feature_energies([
        np.array([0.0, 0.1, -0.1, 5.0, 5.1, 4.9]),
        np.array([0.0, 5.0, 2.5, 0.1, 5.1, 2.6]),
    ])
    labels = np.array(["A", "A", "A", "B", "B", "B"])
    fit = fit_q_weights_adam(
        e, q_star, np.arange(6), labels,
        learning_rate=0.05, max_epochs=800, patience=80,
    )
    assert fit.reference_weights[0] > fit.reference_weights[1]
    assert fit.reference_weights[0] > 0.8
    assert fit.best_loss < np.log(5.0)


def test_pair_mask_with_mixed_qstar_is_finite():
    e, q_star = _feature_energies([
        np.array([0.0, 0.2, 3.0, 3.2]),
        np.array([0.0, 1.0, 0.0, 1.0]),
        np.array([0.0, 2.0, 3.0, 5.0]),
    ])
    q_star[0, 1] = q_star[1, 0] = 0
    q_star[0, 2] = q_star[2, 0] = 1
    D = distance_matrix_from_beta(np.array([0.0, 1.0, -1.0]), e, q_star)
    assert np.all(np.isfinite(D))
    assert np.allclose(D, D.T)
    assert np.allclose(np.diag(D), 0.0)


def test_vectorized_distance_gradient_matches_pairwise_reference():
    rng = np.random.default_rng(12345)
    n, qn = 7, 5
    raw = rng.uniform(0.0, 3.0, size=(n, n, qn))
    energies = 0.5 * (raw + np.transpose(raw, (1, 0, 2)))
    for i in range(n):
        energies[i, i, :] = 0.0
    q_star = rng.integers(0, qn, size=(n, n), dtype=np.int32)
    q_star = np.minimum(q_star, q_star.T)
    np.fill_diagonal(q_star, -1)
    beta = rng.normal(size=qn)

    D2, G = distance_squared_and_grad_beta(beta, energies, q_star)
    ref_D2 = np.zeros((n, n), dtype=float)
    ref_G = np.zeros((n, n, qn), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d2, g = pair_distance_squared_and_grad_beta(beta, energies[i, j], int(q_star[i, j]))
            ref_D2[i, j] = ref_D2[j, i] = d2
            ref_G[i, j] = ref_G[j, i] = g
    assert np.allclose(D2, ref_D2, atol=1e-12, rtol=1e-12)
    assert np.allclose(G, ref_G, atol=1e-12, rtol=1e-12)
