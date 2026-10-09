from __future__ import annotations

import numpy as np
from scipy import sparse

from hodge_vnext.zero_modes import ZeroModeSettings, analytic_single_simplex_zero_mode, zero_mode_diagnostic


def graph_laplacian(n: int, edges: list[tuple[int, int]]) -> sparse.csr_matrix:
    A = np.zeros((n, n), dtype=float)
    for i, j in edges:
        A[i, j] = A[j, i] = 1.0
    D = np.diag(A.sum(axis=1))
    return sparse.csr_matrix(D - A)


def test_k0_zero_count_is_exact_connected_components():
    L = graph_laplacian(5, [(0, 1), (1, 2), (3, 4)])
    res = zero_mode_diagnostic(L, 0, ZeroModeSettings(dense_dimension_guard=10))
    assert res.count_resolved
    assert res.zero_count == 2
    assert res.zero_count_lower_bound == 2
    assert np.isclose(res.zero_fraction, 2 / 5)
    assert res.status == "resolved_exact_k0"


def test_dense_higher_order_zero_count_and_gap():
    L = sparse.diags([0.0, 0.0, 2.0, 5.0], format="csr")
    res = zero_mode_diagnostic(L, 1, ZeroModeSettings(dense_dimension_guard=10))
    assert res.count_resolved
    assert res.zero_count == 2
    assert np.isclose(res.smallest_positive_eigenvalue, 2.0)


def test_sparse_higher_order_reports_resolved_count_when_positive_is_seen():
    vals = np.r_[np.zeros(3), np.arange(1.0, 18.0)]
    L = sparse.diags(vals, format="csr")
    settings = ZeroModeSettings(
        dense_dimension_guard=4,
        sparse_initial_k=4,
        sparse_max_k=8,
        sparse_tolerance=1e-12,
        random_seed=7,
    )
    res = zero_mode_diagnostic(L, 2, settings)
    assert res.count_resolved
    assert res.zero_count == 3
    assert res.zero_count_lower_bound == 3
    assert res.smallest_positive_eigenvalue is not None
    assert np.isclose(res.smallest_positive_eigenvalue, 1.0, atol=1e-8)


def test_sparse_higher_order_never_promotes_censored_lower_bound():
    vals = np.r_[np.zeros(10), np.arange(1.0, 11.0)]
    L = sparse.diags(vals, format="csr")
    settings = ZeroModeSettings(
        dense_dimension_guard=4,
        sparse_initial_k=3,
        sparse_max_k=5,
        sparse_tolerance=1e-12,
        random_seed=9,
    )
    res = zero_mode_diagnostic(L, 3, settings)
    assert not res.count_resolved
    assert res.zero_count is None
    assert res.zero_count_lower_bound == 5
    assert res.status == "censored_zero_count_lower_bound"


def test_single_simplex_analytic_shortcut_is_exact_without_matrix_materialization():
    r0 = analytic_single_simplex_zero_mode(8, 0)
    assert r0.zero_count == 1
    assert r0.dimension == 9
    assert np.isclose(r0.smallest_positive_eigenvalue, 9.0)
    r4 = analytic_single_simplex_zero_mode(8, 4)
    assert r4.zero_count == 0
    assert r4.dimension == 126
    assert np.isclose(r4.smallest_positive_eigenvalue, 9.0)
