from __future__ import annotations

"""Zero-mode diagnostics for cached Hodge Laplacians.

This module is diagnostic only.  It does not alter the Hodge construction,
heat signatures, pair weighting, or cache identity of the scientific method.

For k=0, the nullity of the combinatorial Laplacian is obtained exactly from
connected components of the Laplacian sparsity graph.  For k>0, small matrices
are diagonalized densely; large matrices use an adaptive smallest-eigenvalue
probe.  A large-matrix result is reported as *censored* whenever the requested
small-eigenvalue budget is exhausted before a positive eigenvalue is observed.
The code never promotes a lower bound to an exact nullity.
"""

from dataclasses import asdict, dataclass
from typing import Any
from math import comb

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import ArpackNoConvergence, eigsh


ZERO_MODE_SCHEMA = "hodge-zero-mode-diagnostic-v1"


@dataclass(frozen=True)
class ZeroModeSettings:
    zero_tolerance: float = 1e-10
    negative_tolerance: float = 1e-8
    dense_dimension_guard: int = 1200
    sparse_initial_k: int = 16
    sparse_max_k: int = 128
    sparse_tolerance: float = 1e-9
    sparse_maxiter: int = 200000
    random_seed: int = 20260924

    def validate(self) -> None:
        if self.zero_tolerance <= 0 or self.negative_tolerance <= 0:
            raise ValueError("zero/negative tolerances must be positive")
        if self.dense_dimension_guard < 1:
            raise ValueError("dense_dimension_guard must be >= 1")
        if self.sparse_initial_k < 1 or self.sparse_max_k < 1:
            raise ValueError("sparse eigenvalue budgets must be positive")
        if self.sparse_initial_k > self.sparse_max_k:
            raise ValueError("sparse_initial_k must be <= sparse_max_k")
        if self.sparse_tolerance <= 0 or self.sparse_maxiter < 1:
            raise ValueError("invalid sparse eigensolver settings")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class ZeroModeResult:
    dimension: int
    zero_count_lower_bound: int
    zero_count: int | None
    count_resolved: bool
    zero_fraction_lower_bound: float
    zero_fraction: float | None
    smallest_positive_eigenvalue: float | None
    method: str
    status: str
    eigenvalues_computed: int
    min_eigenvalue_observed: float | None
    trace: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _trace_csr(L: sparse.csr_matrix) -> float:
    return float(np.asarray(L.diagonal(), dtype=float).sum())


def _check_negative(values: np.ndarray, settings: ZeroModeSettings) -> None:
    if values.size and float(np.min(values)) < -settings.negative_tolerance:
        raise RuntimeError(
            "Hodge matrix produced a materially negative eigenvalue: "
            f"min={float(np.min(values)):.6g}, tolerance={settings.negative_tolerance:.6g}"
        )


def _classify(values: np.ndarray, settings: ZeroModeSettings) -> tuple[int, float | None]:
    vals = np.asarray(values, dtype=float)
    _check_negative(vals, settings)
    zero = int(np.count_nonzero(np.abs(vals) <= settings.zero_tolerance))
    positive = vals[vals > settings.zero_tolerance]
    gap = float(np.min(positive)) if positive.size else None
    return zero, gap


def _k0_components(L: sparse.csr_matrix) -> int:
    n = int(L.shape[0])
    if n == 0:
        return 0
    A = L.copy().tocsr()
    A.setdiag(0.0)
    A.eliminate_zeros()
    if A.nnz:
        A.data[:] = 1.0
    return int(connected_components(A, directed=False, return_labels=False))


def _result(
    *,
    n: int,
    lower: int,
    count: int | None,
    gap: float | None,
    method: str,
    status: str,
    eigenvalues_computed: int,
    min_eig: float | None,
    trace: float,
) -> ZeroModeResult:
    resolved = count is not None
    return ZeroModeResult(
        dimension=n,
        zero_count_lower_bound=int(lower),
        zero_count=None if count is None else int(count),
        count_resolved=bool(resolved),
        zero_fraction_lower_bound=(float(lower) / n if n else 0.0),
        zero_fraction=(float(count) / n if resolved and n else (0.0 if resolved else None)),
        smallest_positive_eigenvalue=gap,
        method=method,
        status=status,
        eigenvalues_computed=int(eigenvalues_computed),
        min_eigenvalue_observed=min_eig,
        trace=float(trace),
    )




def analytic_single_simplex_zero_mode(q: int, k: int) -> ZeroModeResult:
    """Exact zero-mode result for one full q-simplex downward closure.

    For k=0 the 1-skeleton is connected and L_0 has spectrum
    {0, q+1 (multiplicity q)}.  For every k>=1, L_k=(q+1)I.
    This mirrors the analytic fast path in :mod:`hodge_vnext.hodge` without
    materializing a potentially enormous identity matrix.
    """
    q = int(q); k = int(k)
    if q < 0 or k < 0 or k > q:
        raise ValueError("Require 0 <= k <= q")
    n = comb(q + 1, k + 1)
    if k == 0:
        count = 1
        gap = float(q + 1) if q >= 1 else None
        tr = float(q * (q + 1))
    else:
        count = 0
        gap = float(q + 1)
        tr = float(n * (q + 1))
    return _result(
        n=n, lower=count, count=count, gap=gap, method="analytic_single_simplex",
        status="resolved_exact_single_simplex", eigenvalues_computed=0,
        min_eig=0.0 if count else gap, trace=tr,
    )


def zero_mode_diagnostic(
    L: sparse.spmatrix,
    k: int,
    settings: ZeroModeSettings,
) -> ZeroModeResult:
    """Count Hodge zero modes without silently approximating unresolved nullity.

    Parameters
    ----------
    L:
        Symmetric positive-semidefinite Hodge Laplacian.
    k:
        Hodge order.  k=0 has an exact component-count shortcut.
    settings:
        Numerical resolution policy for higher-order matrices.
    """
    settings.validate()
    A = sparse.csr_matrix(L, dtype=float)
    if A.shape[0] != A.shape[1]:
        raise ValueError("Hodge Laplacian must be square")
    n = int(A.shape[0])
    tr = _trace_csr(A)
    if n == 0:
        return _result(
            n=0, lower=0, count=0, gap=None, method="empty", status="resolved_empty",
            eigenvalues_computed=0, min_eig=None, trace=tr,
        )

    # Exact and cheap for all L_0^(q): zero multiplicity equals connected components.
    if int(k) == 0:
        count = _k0_components(A)
        gap: float | None = None
        eig_count = 0
        min_eig: float | None = None
        if n <= settings.dense_dimension_guard:
            vals = np.linalg.eigvalsh(A.toarray())
            _check_negative(vals, settings)
            numeric_count, gap = _classify(vals, settings)
            if numeric_count != count:
                raise RuntimeError(
                    "k=0 combinatorial component count disagrees with numerical nullity: "
                    f"components={count}, numerical={numeric_count}"
                )
            eig_count = n
            min_eig = float(vals[0]) if vals.size else None
        return _result(
            n=n, lower=count, count=count, gap=gap, method="k0_connected_components",
            status="resolved_exact_k0", eigenvalues_computed=eig_count,
            min_eig=min_eig, trace=tr,
        )

    # Trivial all-zero matrix: every basis vector is harmonic.
    if A.nnz == 0 or abs(tr) <= settings.zero_tolerance:
        return _result(
            n=n, lower=n, count=n, gap=None, method="zero_matrix",
            status="resolved_exact_zero_matrix", eigenvalues_computed=0,
            min_eig=0.0, trace=tr,
        )

    if n == 1:
        val = float(A[0, 0])
        _check_negative(np.asarray([val]), settings)
        count = 1 if abs(val) <= settings.zero_tolerance else 0
        return _result(
            n=1, lower=count, count=count,
            gap=None if count else val,
            method="scalar", status="resolved_scalar", eigenvalues_computed=1,
            min_eig=val, trace=tr,
        )

    # Small matrices: full numerical spectrum, hence a resolved count under the
    # declared zero tolerance.
    if n <= settings.dense_dimension_guard:
        vals = np.linalg.eigvalsh(A.toarray())
        count, gap = _classify(vals, settings)
        return _result(
            n=n, lower=count, count=count, gap=gap, method="dense_eigvalsh",
            status="resolved_under_tolerance", eigenvalues_computed=n,
            min_eig=float(vals[0]) if vals.size else None, trace=tr,
        )

    # Large matrices: adaptively request the smallest algebraic eigenvalues.
    # If a positive value appears, every zero mode is necessarily already in the
    # returned prefix.  If the budget is exhausted first, report a lower bound.
    budget_cap = min(settings.sparse_max_k, n - 1)
    request = min(max(1, settings.sparse_initial_k), budget_cap)
    rng = np.random.default_rng(settings.random_seed + 1009 * int(k) + 17 * n)
    v0 = rng.standard_normal(n)
    last_vals = np.empty(0, dtype=float)

    while request >= 1:
        try:
            vals = eigsh(
                A,
                k=request,
                which="SA",
                return_eigenvectors=False,
                tol=settings.sparse_tolerance,
                maxiter=settings.sparse_maxiter,
                v0=v0,
            )
        except ArpackNoConvergence as exc:
            vals = np.asarray(exc.eigenvalues if exc.eigenvalues is not None else [], dtype=float)
            if vals.size == 0:
                return _result(
                    n=n, lower=0, count=None, gap=None, method="sparse_eigsh",
                    status="censored_arpack_no_convergence", eigenvalues_computed=0,
                    min_eig=None, trace=tr,
                )
        vals = np.sort(np.asarray(vals, dtype=float))
        _check_negative(vals, settings)
        last_vals = vals
        count, gap = _classify(vals, settings)
        if gap is not None:
            return _result(
                n=n, lower=count, count=count, gap=gap, method="sparse_eigsh",
                status="resolved_under_tolerance", eigenvalues_computed=len(vals),
                min_eig=float(vals[0]) if vals.size else None, trace=tr,
            )

        if request >= budget_cap:
            break
        request = min(budget_cap, max(request + 1, request * 2))

    lower = int(np.count_nonzero(np.abs(last_vals) <= settings.zero_tolerance))
    # If n-1 smallest eigenvalues are all zero but trace is positive, the only
    # unobserved eigenvalue is positive.  Nullity is then exactly n-1.
    if budget_cap == n - 1 and len(last_vals) == n - 1:
        if abs(tr) <= settings.zero_tolerance:
            return _result(
                n=n, lower=n, count=n, gap=None, method="sparse_eigsh",
                status="resolved_all_zero_from_trace", eigenvalues_computed=len(last_vals),
                min_eig=float(last_vals[0]) if last_vals.size else None, trace=tr,
            )
        return _result(
            n=n, lower=n - 1, count=n - 1, gap=tr, method="sparse_eigsh",
            status="resolved_n_minus_one_zero_modes", eigenvalues_computed=len(last_vals),
            min_eig=float(last_vals[0]) if last_vals.size else None, trace=tr,
        )

    return _result(
        n=n, lower=lower, count=None, gap=None, method="sparse_eigsh",
        status="censored_zero_count_lower_bound", eigenvalues_computed=len(last_vals),
        min_eig=float(last_vals[0]) if last_vals.size else None, trace=tr,
    )
