from __future__ import annotations

"""Zero-eigenvalue diagnostics for independent exclusive-q Hodge Laplacians.

This module is diagnostic-only.  It reuses the cached Q2/Q3/Q4 Hodge artifacts
and stores diagnostic spectra under D0_diagnostics.  It does not alter heat
signatures, profile normalization, pairwise distance definitions, or the Q5
spectral cache used by exact/SLQ Hodge runs.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import math

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh, ArpackNoConvergence

from hodge_vnext.cache import ArtifactStore
from hodge_vnext.utils import stable_hash, stable_seed, write_json


ZERO_MODE_SCHEMA = 'hodge-zero-mode-diagnostic-v2'


@dataclass(frozen=True)
class ZeroModeSettings:
    zero_tolerance: float = 1e-10
    negative_tolerance: float = 1e-8
    dense_dimension_guard: int = 512
    sparse_initial_k: int = 16
    sparse_max_k: int = 128
    sparse_tolerance: float = 1e-9
    sparse_maxiter: int = 200000
    random_seed: int = 20261007
    store_eigenvalues: bool = True

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> 'ZeroModeSettings':
        out = cls(**{k: d[k] for k in d if k in cls.__dataclass_fields__})
        if out.zero_tolerance <= 0 or out.negative_tolerance <= 0:
            raise ValueError('eigenvalue tolerances must be positive')
        if out.dense_dimension_guard < 1:
            raise ValueError('dense_dimension_guard must be >=1')
        if out.sparse_initial_k < 1 or out.sparse_max_k < out.sparse_initial_k:
            raise ValueError('invalid sparse prefix settings')
        return out


def _validate(values: np.ndarray, settings: ZeroModeSettings) -> np.ndarray:
    v = np.asarray(values, dtype=float).reshape(-1).copy()
    if v.size == 0:
        return v
    minv = float(np.min(v))
    if minv < -settings.negative_tolerance:
        raise ValueError(f'Hodge Laplacian has significant negative eigenvalue {minv:.6e}')
    v[np.abs(v) <= settings.zero_tolerance] = 0.0
    return np.maximum(v, 0.0)


def _scalar_identity_value(L: sparse.csr_matrix) -> float | None:
    n, m = L.shape
    if n != m:
        return None
    if n == 0:
        return 0.0
    C = L.tocsr()
    if C.nnz == 0:
        return 0.0
    if C.nnz != n or not np.all(np.diff(C.indptr) == 1):
        return None
    if not np.array_equal(C.indices, np.arange(n, dtype=C.indices.dtype)):
        return None
    x = float(C.data[0])
    if np.allclose(C.data, x, rtol=0.0, atol=0.0):
        return x
    return None


def exact_k0_nullity_from_laplacian(L: sparse.csr_matrix) -> int:
    """Exact multiplicity of lambda=0 for a graph Laplacian L_0."""
    n = int(L.shape[0])
    if n == 0:
        return 0
    C = L.tocsr().copy()
    C.setdiag(0.0)
    C.eliminate_zeros()
    if C.nnz:
        C.data = np.ones_like(C.data)
    ncomp, _ = connected_components(C, directed=False, return_labels=True)
    return int(ncomp)


def _cache_root(store: ArtifactStore, q: int, k: int, settings: ZeroModeSettings) -> Path:
    key = {
        'schema': ZERO_MODE_SCHEMA,
        'q': int(q), 'k': int(k),
        'zero_tolerance': settings.zero_tolerance,
        'negative_tolerance': settings.negative_tolerance,
        'dense_dimension_guard': settings.dense_dimension_guard,
        'sparse_initial_k': settings.sparse_initial_k,
        'sparse_max_k': settings.sparse_max_k,
        'sparse_tolerance': settings.sparse_tolerance,
        'sparse_maxiter': settings.sparse_maxiter,
        'random_seed': settings.random_seed,
    }
    p = store.diagnostics_dir() / 'zero_modes_v2' / f'q{q:04d}' / f'k{k:04d}_{stable_hash(key,16)}'
    p.mkdir(parents=True, exist_ok=True)
    return p


def diagnose_hodge_zero_modes(
    L: sparse.csr_matrix,
    graph_hash: str,
    q: int,
    k: int,
    store: ArtifactStore,
    settings: ZeroModeSettings,
    *,
    force: bool = False,
) -> tuple[dict[str, Any], np.ndarray]:
    """Diagnose zero multiplicity and smallest positive eigenvalue.

    Small matrices receive the full dense spectrum.  Large matrices use an
    increasing sparse smallest-eigenvalue prefix.  k=0 nullity is always exact
    from connected components; higher-k sparse nullity is explicitly marked as
    unresolved when the prefix contains only zeros up to the configured cap.
    """
    root = _cache_root(store, q, k, settings)
    summary_path = root / 'summary.json'
    eig_path = root / 'eigenvalues.npy'
    if summary_path.exists() and eig_path.exists() and not force:
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        return summary, np.load(eig_path, allow_pickle=False)

    n = int(L.shape[0])
    trace = float(np.asarray(L.diagonal(), dtype=float).sum())
    identity_value = _scalar_identity_value(L)
    spectrum_scope = 'smallest_prefix'
    method = ''
    status = ''
    resolved = False
    zero_count: int | None = None
    zero_lb = 0
    smallest_positive = math.nan

    if n == 0:
        values = np.empty(0, dtype=float)
        spectrum_scope = 'full'
        method = 'empty'
        status = 'resolved_empty'
        resolved = True
        zero_count = 0
    elif identity_value is not None:
        values = np.asarray([max(float(identity_value), 0.0)], dtype=float)
        method = 'analytic_scalar_identity'
        spectrum_scope = 'analytic_compact'
        if abs(identity_value) <= settings.zero_tolerance:
            zero_count = n
            zero_lb = n
            smallest_positive = math.nan
        else:
            zero_count = 0
            zero_lb = 0
            smallest_positive = float(identity_value)
        status = 'resolved_analytic'
        resolved = True
    elif n <= settings.dense_dimension_guard:
        values = _validate(np.linalg.eigvalsh(L.toarray()), settings)
        spectrum_scope = 'full'
        method = 'dense_eigvalsh'
        zero_count = int(np.sum(values <= settings.zero_tolerance))
        zero_lb = zero_count
        pos = values[values > settings.zero_tolerance]
        smallest_positive = float(pos[0]) if pos.size else math.nan
        status = 'resolved_full_spectrum'
        resolved = True
    else:
        method = 'sparse_eigsh'
        k0_exact = exact_k0_nullity_from_laplacian(L) if k == 0 else None
        if k0_exact is not None:
            zero_count = int(k0_exact)
            zero_lb = int(k0_exact)
            resolved = True
            status = 'resolved_exact_k0'

        max_request = min(int(settings.sparse_max_k), max(n - 1, 1))
        request = min(max(int(settings.sparse_initial_k), 1), max_request)
        values = np.empty(0, dtype=float)
        while request >= 1:
            v0_rng = np.random.default_rng(stable_seed(settings.random_seed, graph_hash, q, k, request))
            v0 = v0_rng.normal(size=n)
            try:
                vals = eigsh(
                    L, k=request, which='SM', return_eigenvectors=False,
                    tol=float(settings.sparse_tolerance), maxiter=int(settings.sparse_maxiter), v0=v0,
                )
            except ArpackNoConvergence as exc:
                vals = np.asarray(exc.eigenvalues if exc.eigenvalues is not None else [], dtype=float)
                if vals.size == 0:
                    status = 'sparse_no_convergence_no_eigenvalues'
                    break
            vals = np.sort(_validate(vals, settings))
            values = vals
            zero_prefix = int(np.sum(vals <= settings.zero_tolerance))
            zero_lb = max(zero_lb, zero_prefix)
            pos = vals[vals > settings.zero_tolerance]
            if pos.size:
                smallest_positive = float(pos[0])
                if k != 0:
                    zero_count = zero_prefix
                    resolved = True
                    status = 'resolved_under_tolerance'
                break
            if request >= max_request:
                if k != 0:
                    status = 'unresolved_zero_lower_bound'
                break
            request = min(max_request, max(request * 2, zero_prefix + 1))

        if k == 0 and resolved and not status:
            status = 'resolved_exact_k0'
        if k != 0 and not resolved and not status:
            status = 'unresolved_sparse_prefix'

    values = np.asarray(values, dtype=float)
    if zero_count is None:
        zero_fraction = math.nan
    else:
        zero_fraction = float(zero_count / n) if n else 0.0
    summary: dict[str, Any] = {
        'schema': ZERO_MODE_SCHEMA,
        'q': int(q), 'k': int(k), 'dimension': n,
        'trace': trace,
        'spectrum_scope': spectrum_scope,
        'spectrum_method': method,
        'eigenvalues_computed': int(values.size),
        'analytic_multiplicity': int(n) if identity_value is not None else None,
        'zero_count_lower_bound': int(zero_lb),
        'zero_count': int(zero_count) if zero_count is not None else None,
        'count_resolved': bool(resolved),
        'zero_fraction_lower_bound': float(zero_lb / n) if n else 0.0,
        'zero_fraction': zero_fraction,
        'smallest_positive_eigenvalue': smallest_positive,
        'min_eigenvalue_observed': float(np.min(values)) if values.size else math.nan,
        'max_eigenvalue_observed': float(np.max(values)) if values.size else math.nan,
        'status': status,
    }
    np.save(eig_path, values)
    write_json(summary_path, summary)
    return summary, values
