from itertools import combinations

import numpy as np
from scipy import sparse

from hodge_vnext.hodge import boundary_matrix_from_lists
from hodge_vnext.spectral import _constant_diagonal_value


def _full_simplex_hodge(q: int, k: int):
    verts = tuple(range(q + 1))
    simplices = {r: list(combinations(verts, r + 1)) for r in range(q + 1)}
    n_k = len(simplices[k])
    L = sparse.csr_matrix((n_k, n_k), dtype=float)
    if k >= 1:
        Bk = boundary_matrix_from_lists(simplices[k - 1], simplices[k], k)
        L = L + Bk.T @ Bk
    if k < q:
        Bn = boundary_matrix_from_lists(simplices[k], simplices[k + 1], k + 1)
        L = L + Bn @ Bn.T
    return L.tocsr()


def test_full_simplex_positive_hodge_is_scalar_identity():
    for q in range(1, 7):
        for k in range(1, q + 1):
            L = _full_simplex_hodge(q, k)
            target = (q + 1) * sparse.identity(L.shape[0], format="csr")
            diff = L - target
            assert diff.nnz == 0 or np.allclose(diff.data, 0.0)


def test_constant_diagonal_detection():
    L = 75.0 * sparse.identity(100, format="csr")
    assert _constant_diagonal_value(L) == 75.0
    bad = L.tolil()
    bad[0, 1] = 1.0
    assert _constant_diagonal_value(bad.tocsr()) is None
