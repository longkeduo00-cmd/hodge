from __future__ import annotations

"""Faithful local NetLSD heat-signature reproduction.

Primary references
------------------
Paper:
    Tsitsulin et al., "NetLSD: Hearing the Shape of a Graph", KDD 2018.
Official implementation:
    https://github.com/xgfs/NetLSD
    commit 86c73cc8c6533b4547cbc12a2b81e7cc23f9817a
    netlsd/kernels.py, netlsd/util.py

Protocol used by Hodge VNext
----------------------------
* normalized graph Laplacian;
* 250 log-spaced times in [1e-2, 1e2] by default;
* heat trace;
* empty-graph normalization (divide by n);
* Euclidean distance between signatures;
* ``eigenvalues='auto'`` official size-dependent strategy:
    - n <= 1024: full eigendecomposition;
    - n > 1024: 150 eigenvalues from each end of the spectrum (300 total),
      followed by the official linear interpolation of the uncomputed middle.

Important: NetLSD does **not** use stochastic Lanczos quadrature (SLQ).  The
large-graph branch computes extreme eigenvalues with scipy ``eigsh`` (the paper
mentions a block Krylov-Schur eigensolver/SLEPc for its scalable implementation)
and linearly interpolates the middle spectrum.  Hodge's SLQ setting is wholly
independent and is never reused by this baseline.
"""

from typing import Literal
import numpy as np
import networkx as nx
import scipy.linalg as spl
import scipy.sparse as sps
import scipy.sparse.linalg as spsl

OFFICIAL_REPOSITORY = "https://github.com/xgfs/NetLSD"
OFFICIAL_COMMIT = "86c73cc8c6533b4547cbc12a2b81e7cc23f9817a"

# Exact constants in the upstream ``eigenvalues_auto`` implementation.
AUTO_FULL_THRESHOLD = 1024
AUTO_EIGENVALUES_PER_SIDE = 150
AUTO_TOTAL_EXTREME_EIGENVALUES = 2 * AUTO_EIGENVALUES_PER_SIDE
DEFAULT_TIMESCALES = np.logspace(-2, 2, 250)


def graph_to_laplacian(graph: nx.Graph, normalized: bool = True):
    """Faithful NetworkX branch of upstream ``graph_to_laplacian``."""
    if normalized:
        return nx.normalized_laplacian_matrix(graph)
    return nx.laplacian_matrix(graph)


def paper_auto_strategy(num_nodes: int) -> str:
    """Return the exact upstream ``auto`` spectral strategy for graph size."""
    n = int(num_nodes)
    return "full" if n <= AUTO_FULL_THRESHOLD else "two_sided_linear_approx"


def updown_linear_approx(
    eigvals_lower: np.ndarray,
    eigvals_upper: np.ndarray,
    nv: int,
) -> np.ndarray:
    """Direct translation of the official linear spectrum interpolation."""
    eigvals_lower = np.asarray(eigvals_lower, dtype=float)
    eigvals_upper = np.asarray(eigvals_upper, dtype=float)
    nal = len(eigvals_lower)
    nau = len(eigvals_upper)
    if nv < nal + nau:
        raise ValueError(
            f"Number of supplied eigenvalues ({nal} lower and {nau} upper) "
            f"is higher than number of nodes ({nv})!"
        )
    ret = np.zeros(int(nv), dtype=float)
    ret[:nal] = eigvals_lower
    ret[-nau:] = eigvals_upper
    # Preserve the endpoint-inclusive interpolation used upstream.
    ret[nal - 1 : -nau + 1] = np.linspace(
        eigvals_lower[-1], eigvals_upper[0], int(nv) - nal - nau + 2
    )
    return ret


def _full_eigenvalues(mat) -> np.ndarray:
    if sps.issparse(mat):
        mat = mat.todense()
    return np.asarray(spl.eigvalsh(mat), dtype=float)


def _two_sided_sparse_eigenvalues(mat, n_lower: int, n_upper: int) -> np.ndarray:
    """Official sparse extreme-eigenvalue logic, without SLQ."""
    nv = int(mat.shape[0])
    if n_lower == n_upper:
        # This is the exact upstream fast branch used by eigenvalues='auto'.
        tr_eivals = spsl.eigsh(
            mat,
            2 * int(n_lower),
            which="BE",
            return_eigenvectors=False,
        )
        return updown_linear_approx(
            tr_eivals[: int(n_upper)],
            tr_eivals[int(n_upper) :],
            nv,
        )

    # Retained for completeness of the official tuple branch.
    lo_eivals = spsl.eigsh(
        mat, int(n_lower), which="SM", return_eigenvectors=False
    )[::-1]
    up_eivals = spsl.eigsh(
        mat, int(n_upper), which="LM", return_eigenvectors=False
    )
    return updown_linear_approx(lo_eivals, up_eivals, nv)


def _two_sided_dense_eigenvalues(mat: np.ndarray, n_lower: int, n_upper: int) -> np.ndarray:
    """Modern SciPy equivalent of the old upstream dense subset branch."""
    nv = int(mat.shape[0])
    lo = spl.eigvalsh(mat, subset_by_index=(0, int(n_lower) - 1))
    hi = spl.eigvalsh(mat, subset_by_index=(nv - int(n_upper), nv - 1))
    return updown_linear_approx(np.asarray(lo), np.asarray(hi), nv)


def eigenvalues_auto(
    mat,
    n_eivals: Literal["auto", "full"] | int | tuple[int, int] = "auto",
) -> np.ndarray:
    """Faithful upstream eigenvalue policy.

    ``auto`` is the protocol used by all released Hodge VNext JSON configs.
    The 1024 threshold is from the official code.  The paper itself states the
    conceptual policy as full eigendecomposition for small graphs and a
    300-eigenvalue (150 from each side) approximation for the large REDDIT-L
    setting.
    """
    nv = int(mat.shape[0])
    do_full = True
    n_lower = AUTO_EIGENVALUES_PER_SIDE
    n_upper = AUTO_EIGENVALUES_PER_SIDE

    if n_eivals == "auto":
        if nv > AUTO_FULL_THRESHOLD:
            do_full = False
    elif n_eivals == "full":
        do_full = True
    elif isinstance(n_eivals, int):
        if n_eivals < 1:
            raise ValueError("integer eigenvalue count must be >=1")
        n_lower = n_upper = int(n_eivals)
        do_full = False
    elif isinstance(n_eivals, tuple) and len(n_eivals) == 2:
        n_lower, n_upper = map(int, n_eivals)
        if n_lower < 1 or n_upper < 1:
            raise ValueError("tuple eigenvalue counts must both be >=1")
        do_full = False
    else:
        raise ValueError("eigenvalues must be 'auto', 'full', int, or (lower, upper)")

    if do_full:
        return _full_eigenvalues(mat)

    if nv < n_lower + n_upper:
        # Explicit approximation requests on a graph too small to hold the
        # requested extreme spectrum are mathematically unnecessary; computing
        # the full spectrum is exact and avoids eigsh k>=N failures.  This never
        # changes the official auto branch because auto approximates only n>1024.
        return _full_eigenvalues(mat)

    if sps.issparse(mat):
        return _two_sided_sparse_eigenvalues(mat, n_lower, n_upper)
    return _two_sided_dense_eigenvalues(np.asarray(mat), n_lower, n_upper)


def heat_signature(
    graph: nx.Graph,
    timescales: np.ndarray | None = None,
    eigenvalues: Literal["auto", "full"] | int | tuple[int, int] = "auto",
    normalization: Literal["empty", "complete", "none"] | None = "empty",
    normalized_laplacian: bool = True,
) -> np.ndarray:
    """NetLSD heat signature using the official mathematical semantics."""
    if timescales is None:
        timescales = DEFAULT_TIMESCALES
    timescales = np.asarray(timescales, dtype=float)
    if timescales.ndim != 1:
        raise ValueError("timescales must be a one-dimensional array")

    mat = graph_to_laplacian(graph, normalized=normalized_laplacian)
    eivals = eigenvalues_auto(mat, eigenvalues)

    nv = int(eivals.shape[0])
    hkt = np.asarray(
        [np.sum(np.exp(-float(t) * eivals)) for t in timescales],
        dtype=float,
    )

    # Direct upstream normalization rules.
    if normalization == "empty":
        return hkt / nv
    if normalization == "complete":
        if normalized_laplacian:
            return hkt / (1.0 + (nv - 1) * np.exp(-timescales))
        return hkt / (1.0 + nv * np.exp(-nv * timescales))
    return hkt


def compare(descriptor1: np.ndarray, descriptor2: np.ndarray) -> float:
    """Official NetLSD comparison: Euclidean/L2 norm."""
    return float(
        np.linalg.norm(np.asarray(descriptor1, dtype=float) - np.asarray(descriptor2, dtype=float))
    )
