from __future__ import annotations

from typing import Callable, Optional
from math import comb
import numpy as np
from scipy import sparse

from .cache import ArtifactStore
from .simplices import layer_simplices_at_order


def boundary_matrix_from_lists(
    lower: list[tuple[int, ...]],
    upper: list[tuple[int, ...]],
    r: int,
) -> sparse.csr_matrix:
    if r < 1:
        raise ValueError("Boundary B_r requires r >= 1.")
    lower_index = {sigma: i for i, sigma in enumerate(lower)}
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for j, sigma in enumerate(upper):
        if len(sigma) != r + 1:
            raise ValueError("Upper simplex has inconsistent dimension.")
        # Sorted vertex order fixes orientation: deleting vertex i contributes (-1)^i.
        for i in range(r + 1):
            face = sigma[:i] + sigma[i + 1 :]
            try:
                row = lower_index[face]
            except KeyError as exc:
                raise RuntimeError("Induced subcomplex is missing a required face.") from exc
            rows.append(row)
            cols.append(j)
            data.append(-1.0 if i % 2 else 1.0)
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(lower), len(upper)), dtype=float)


def get_boundary(
    q: int,
    r: int,
    cliques: list[tuple[int, ...]],
    store: ArtifactStore,
    progress: Optional[Callable[[str], None]] = None,
) -> sparse.csr_matrix:
    path = store.boundary_file(q, r)
    cached = store.load_sparse(path)
    if cached is not None:
        if progress:
            progress(f"[BOUNDARY] q={q}, r={r}: cache hit shape={cached.shape}")
        return cached
    lower = layer_simplices_at_order(q, cliques, r - 1, store, progress)
    upper = layer_simplices_at_order(q, cliques, r, store, progress)
    B = boundary_matrix_from_lists(lower, upper, r)
    store.save_sparse(path, B, enabled=store.config.persist_boundaries)
    if progress:
        progress(f"[BOUNDARY] q={q}, r={r}: built shape={B.shape}, nnz={B.nnz}")
    return B


def get_hodge_laplacian(
    q: int,
    k: int,
    cliques: list[tuple[int, ...]],
    store: ArtifactStore,
    progress: Optional[Callable[[str], None]] = None,
) -> sparse.csr_matrix:
    if not 0 <= k <= q:
        raise ValueError(f"Hodge order k={k} invalid for q={q} layer.")
    path = store.hodge_file(q, k)
    cached = store.load_sparse(path)
    if cached is not None:
        if progress:
            progress(f"[HODGE] q={q}, k={k}: cache hit dim={cached.shape[0]}, nnz={cached.nnz}")
        return cached

    # A single full q-simplex has L_k=(q+1)I for k>=1; avoid boundary fill-in.
    if len(cliques) == 1 and k >= 1:
        n_k = comb(q + 1, k + 1)
        L = sparse.identity(n_k, format="csr", dtype=float)
        L.data *= float(q + 1)
        store.save_sparse(path, L, enabled=store.config.persist_hodge)
        if progress:
            progress(
                f"[HODGE] q={q}, k={k}: analytic full-simplex fast path "
                f"dim={n_k}, nnz={n_k}, value={q + 1}"
            )
        return L

    k_simplices = layer_simplices_at_order(q, cliques, k, store, progress)
    n_k = len(k_simplices)
    L = sparse.csr_matrix((n_k, n_k), dtype=float)
    # L_k = B_k^T B_k + B_{k+1} B_{k+1}^T, within this q layer only.
    if k >= 1:
        Bk = get_boundary(q, k, cliques, store, progress)
        L = L + Bk.T @ Bk
    if k < q:
        Bnext = get_boundary(q, k + 1, cliques, store, progress)
        L = L + Bnext @ Bnext.T
    L = L.tocsr()
    store.save_sparse(path, L, enabled=store.config.persist_hodge)
    if progress:
        progress(f"[HODGE] q={q}, k={k}: built dim={L.shape[0]}, nnz={L.nnz}")
    return L
