from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from scipy import sparse
from scipy.linalg import eigh_tridiagonal

from .cache import ArtifactStore
from .config import MethodConfig
from .utils import stable_seed, write_json


def _validate_and_clip_eigenvalues(values: np.ndarray, cfg: MethodConfig) -> np.ndarray:
    values = np.asarray(values, dtype=float).copy()
    values[np.abs(values) < cfg.exact.eigen_zero_tolerance] = 0.0
    minimum = float(values.min(initial=0.0))
    if minimum < -cfg.exact.eigen_negative_tolerance:
        raise ValueError(f"Hodge Laplacian has significant negative eigenvalue {minimum:.6e}.")
    return np.maximum(values, 0.0)


def exact_eigenvalues(
    L: sparse.csr_matrix,
    q: int,
    k: int,
    cfg: MethodConfig,
    store: ArtifactStore,
    progress: Optional[Callable[[str], None]] = None,
) -> np.ndarray:
    n = int(L.shape[0])
    guard = cfg.exact.dimension_guard
    if guard is not None and n > guard:
        raise RuntimeError(
            f"Exact solver requested globally, but L_{k}^({q}) has dimension {n} > "
            f"exact.dimension_guard={guard}. The run is stopped; it will NOT switch to SLQ."
        )
    key = {
        "zero_tol": cfg.exact.eigen_zero_tolerance,
        "negative_tol": cfg.exact.eigen_negative_tolerance,
    }
    root = store.exact_dir(q, k, key)
    path = root / "eigenvalues.npy"
    cached = store.load_array(path)
    if cached is not None:
        if progress:
            progress(f"[EXACT] q={q}, k={k}: eigenvalue cache hit")
        return cached

    if n == 0:
        eig = np.empty(0, dtype=float)
    else:
        eig = np.linalg.eigvalsh(L.toarray())
        eig = _validate_and_clip_eigenvalues(eig, cfg)
    store.save_array(path, eig, enabled=store.config.persist_spectral)
    store.write_manifest_once(root / "manifest.json", {"solver": "exact", **key, "dimension": n})
    if progress:
        progress(f"[EXACT] q={q}, k={k}: computed {eig.size} eigenvalues")
    return eig


def lanczos_tridiagonal_full_reorth(
    L: sparse.csr_matrix,
    start: np.ndarray,
    steps: int,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Lanczos tridiagonalization with full reorthogonalization.

    Returns diagonal alpha, off-diagonal beta, and ||start||^2. The projected
    matrix is represented only by its tridiagonal bands; no dense T is formed.
    """
    start = np.asarray(start, dtype=float)
    norm_sq = float(start @ start)
    if norm_sq <= 0:
        raise ValueError("Lanczos start vector must be nonzero.")
    n = int(L.shape[0])
    max_steps = min(int(steps), n)
    if max_steps < 1:
        return np.empty(0), np.empty(0), norm_sq

    Q = np.empty((n, max_steps), dtype=float)
    q = start / np.sqrt(norm_sq)
    q_prev = np.zeros_like(q)
    beta_prev = 0.0
    alphas: list[float] = []
    betas: list[float] = []

    for j in range(max_steps):
        Q[:, j] = q
        z = L @ q - beta_prev * q_prev
        alpha = float(q @ z)
        z = z - alpha * q

        # Full reorthogonalization against every Lanczos basis vector built so far.
        basis = Q[:, : j + 1]
        coeff = basis.T @ z
        z = z - basis @ coeff

        beta = float(np.linalg.norm(z))
        alphas.append(alpha)
        if beta < tolerance or j + 1 == max_steps:
            break
        betas.append(beta)
        q_prev, q = q, z / beta
        beta_prev = beta

    return np.asarray(alphas, dtype=float), np.asarray(betas, dtype=float), norm_sq


def _slq_numerical_key(cfg: MethodConfig) -> dict:
    return {
        "lanczos_steps": cfg.slq.lanczos_steps,
        "slq_seed": cfg.slq.seed,
        "breakdown_tolerance": cfg.slq.breakdown_tolerance,
        "probe_type": "rademacher",
        "reorthogonalization": "full",
        "small_eigensolver": "symmetric_tridiagonal",
    }


def _constant_diagonal_value(L: sparse.csr_matrix) -> float | None:
    """Return lambda when L is exactly lambda*I, else None."""
    n, m = L.shape
    if n != m:
        return None
    if n == 0:
        return 0.0
    if L.nnz == 0:
        return 0.0
    if L.nnz != n:
        return None
    C = L.tocsr()
    if not np.all(np.diff(C.indptr) == 1):
        return None
    if not np.array_equal(C.indices, np.arange(n, dtype=C.indices.dtype)):
        return None
    value = float(C.data[0])
    if not np.all(C.data == value):
        return None
    return value


def slq_probe_quadrature(
    L: sparse.csr_matrix,
    graph_hash: str,
    q: int,
    k: int,
    probe_index: int,
    cfg: MethodConfig,
    store: ArtifactStore,
    progress: Optional[Callable[[str], None]] = None,
) -> dict[str, np.ndarray | float | int]:
    """Return one probe's SLQ quadrature nodes and weights, cached independently.

    Probe indices are 1-based. Increasing R reuses probes 1..R_old and computes
    only the missing probes. Changing M creates a distinct spectral-cache branch.
    """
    if probe_index < 1:
        raise ValueError("probe_index is 1-based.")
    key = _slq_numerical_key(cfg)
    root = store.slq_dir(q, k, key)
    path = root / f"probe_{probe_index:05d}.npz"
    if store.config.enabled and path.exists():
        data = np.load(path, allow_pickle=False)
        if progress:
            progress(f"[SLQ] q={q}, k={k}, probe={probe_index}: cache hit")
        return {
            "theta": data["theta"],
            "weights": data["weights"],
            "norm_sq": float(data["norm_sq"]),
            "effective_steps": int(data["effective_steps"]),
            "probe_seed": int(data["probe_seed"]),
        }

    n = int(L.shape[0])
    probe_seed = stable_seed(cfg.slq.seed, graph_hash, q, k, probe_index)
    constant_diag = _constant_diagonal_value(L)
    if n == 0:
        theta = np.empty(0, dtype=float)
        weights = np.empty(0, dtype=float)
        norm_sq = 0.0
        effective_steps = 0
    elif constant_diag is not None:
        # For L=lambda*I, a Rademacher probe has ||z||^2=n and Lanczos
        # terminates exactly after one step. Avoid allocating the n x M
        # full-reorthogonalization basis.
        theta = np.asarray([max(float(constant_diag), 0.0)], dtype=float)
        weights = np.ones(1, dtype=float)
        norm_sq = float(n)
        effective_steps = 1
        if progress:
            progress(
                f"[SLQ] q={q}, k={k}, probe={probe_index}: "
                f"scalar-identity shortcut lambda={constant_diag:g}"
            )
    else:
        rng = np.random.default_rng(probe_seed)
        z = rng.choice(np.asarray([-1.0, 1.0]), size=n)
        alpha, beta, norm_sq = lanczos_tridiagonal_full_reorth(
            L, z, cfg.slq.lanczos_steps, cfg.slq.breakdown_tolerance
        )
        effective_steps = int(alpha.size)
        if effective_steps == 0:
            theta = np.empty(0, dtype=float)
            weights = np.empty(0, dtype=float)
        elif effective_steps == 1:
            theta = alpha.copy()
            weights = np.ones(1, dtype=float)
        else:
            theta, U = eigh_tridiagonal(alpha, beta, eigvals_only=False, check_finite=False)
            weights = np.asarray(U[0, :] ** 2, dtype=float)
        theta = np.maximum(theta, 0.0)

    if store.config.enabled and store.config.persist_spectral:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            np.savez_compressed(
                path,
                theta=theta,
                weights=weights,
                norm_sq=np.float64(norm_sq),
                effective_steps=np.int64(effective_steps),
                probe_seed=np.int64(probe_seed),
            )
    store.write_manifest_once(
        root / "manifest.json",
        {"solver": "slq", **key, "matrix_dimension": n},
    )
    if progress:
        progress(f"[SLQ] q={q}, k={k}, probe={probe_index}: computed M_eff={effective_steps}")
    return {
        "theta": theta,
        "weights": weights,
        "norm_sq": norm_sq,
        "effective_steps": effective_steps,
        "probe_seed": probe_seed,
    }


def time_grid(cfg: MethodConfig) -> np.ndarray:
    return np.logspace(np.log10(cfg.time_min), np.log10(cfg.time_max), cfg.time_points)


def heat_trace(
    L: sparse.csr_matrix,
    graph_hash: str,
    q: int,
    k: int,
    cfg: MethodConfig,
    store: ArtifactStore,
    progress: Optional[Callable[[str], None]] = None,
) -> np.ndarray:
    times = time_grid(cfg)
    time_key = {"mode": "logspace", "min": cfg.time_min, "max": cfg.time_max, "points": cfg.time_points}

    if cfg.solver == "exact":
        solver_key = {
            "solver": "exact",
            "zero_tol": cfg.exact.eigen_zero_tolerance,
            "negative_tol": cfg.exact.eigen_negative_tolerance,
        }
    elif cfg.solver == "slq":
        solver_key = {"solver": "slq", **_slq_numerical_key(cfg), "num_probes": cfg.slq.num_probes}
    else:
        raise ValueError("Mixed/auto solver modes are forbidden.")

    heat_file = store.heat_file(q, k, solver_key, time_key)
    cached = store.load_array(heat_file)
    if cached is not None:
        if progress:
            progress(f"[HEAT] q={q}, k={k}: cache hit")
        return cached

    if cfg.solver == "exact":
        eig = exact_eigenvalues(L, q, k, cfg, store, progress)
        raw = np.exp(-np.outer(times, eig)).sum(axis=1) if eig.size else np.zeros_like(times)
    else:
        estimate = np.zeros_like(times, dtype=float)
        for probe_idx in range(1, cfg.slq.num_probes + 1):
            quad = slq_probe_quadrature(L, graph_hash, q, k, probe_idx, cfg, store, progress)
            theta = np.asarray(quad["theta"], dtype=float)
            weights = np.asarray(quad["weights"], dtype=float)
            norm_sq = float(quad["norm_sq"])
            if theta.size:
                estimate += norm_sq * (np.exp(-np.outer(times, theta)) @ weights)
        raw = np.maximum(estimate / cfg.slq.num_probes, 0.0)

    store.save_array(heat_file, raw, enabled=store.config.persist_heat)
    if progress:
        progress(f"[HEAT] q={q}, k={k}: computed on {times.size} log-spaced times")
    return raw
