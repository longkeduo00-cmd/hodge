from __future__ import annotations

"""Real-network Hodge zero-mode and low-spectrum diagnostics.

This module is diagnostic-only.  It reuses the existing globally-maximal-clique
C1 cache and Q2--Q4 Hodge artifacts, then stores zero-mode / observed-spectrum
results under D0_diagnostics.  It does not change heat signatures or Hodge
pairwise distances.

For large Hodge matrices a complete eigendecomposition is intentionally not
attempted.  The run therefore distinguishes:

* full spectrum: all eigenvalues are stored (small matrices);
* smallest-prefix spectrum: only the smallest algebraic eigenvalues are stored;
* analytic compact spectrum: exact value/multiplicity pairs for a single full
  q-simplex, without materialising a huge repeated eigenvalue vector.

The zero-mode count itself follows the existing diagnostic contract: k=0 is
exact from connected components, while higher-k sparse counts can be censored
if the adaptive smallest-eigenvalue budget is exhausted before a positive
value is seen.
"""

from dataclasses import asdict
from pathlib import Path
from typing import Any
import json
import math
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import ArpackNoConvergence, eigsh

from hodge_vnext.cache import ArtifactStore, CACHE_SCHEMA
from hodge_vnext.cliques import enumerate_maximal_clique_layers, retained_layers
from hodge_vnext.graph_io import create_real_record
from hodge_vnext.hodge import get_hodge_laplacian
from hodge_vnext.simplices import required_max_hodge_order
from hodge_vnext.utils import read_json, safe_token, stable_hash, write_json
from hodge_vnext.zero_modes import ZERO_MODE_SCHEMA, ZeroModeSettings, analytic_single_simplex_zero_mode, zero_mode_diagnostic

from .config import method_from_payload, resolve_path
from .real_domains import _prepare_manifest
from .style import DMEASURE_CMAP, FIG_DPI, LABEL_SIZE, TICK_SIZE, TITLE_SIZE, apply_style
from .real_zero_mode_reporting import generate_real_zero_mode_report


REAL_ZERO_MODE_SCHEMA = "hodge-real-zero-mode-spectrum-v1"


def _settings_from_payload(diag: dict[str, Any], cfg) -> ZeroModeSettings:
    # The archive's newer synthetic diagnostics replaced this legacy adapter.
    values = {key: value for key, value in diag.items() if key in ZeroModeSettings.__dataclass_fields__}
    values.setdefault("zero_tolerance", cfg.exact.eigen_zero_tolerance)
    values.setdefault("negative_tolerance", cfg.exact.eigen_negative_tolerance)
    settings = ZeroModeSettings(**values)
    settings.validate()
    return settings


def _savefig(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _diag_identity(q: int, k: int, settings: ZeroModeSettings) -> dict[str, Any]:
    return {
        "schema": ZERO_MODE_SCHEMA,
        "q": int(q),
        "k": int(k),
        "settings": settings.to_dict(),
    }


def _zero_cache_path(store: ArtifactStore, q: int, k: int, settings: ZeroModeSettings) -> Path:
    root = store.diagnostics_dir() / "zero_modes"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"q{q:04d}_k{k:04d}_{stable_hash(_diag_identity(q, k, settings), 16)}.json"


def _spectrum_cache_path(store: ArtifactStore, q: int, k: int, settings: ZeroModeSettings) -> Path:
    root = store.diagnostics_dir() / "zero_modes" / "spectra"
    root.mkdir(parents=True, exist_ok=True)
    ident = {**_diag_identity(q, k, settings), "capture": "smallest-spectrum-prefix-v1"}
    return root / f"q{q:04d}_k{k:04d}_{stable_hash(ident, 16)}.npz"


def _load_or_compute_zero(
    store: ArtifactStore,
    L: sparse.spmatrix | None,
    q: int,
    k: int,
    settings: ZeroModeSettings,
    *,
    force: bool,
    analytic=None,
) -> tuple[dict[str, Any], bool, float]:
    path = _zero_cache_path(store, q, k, settings)
    if path.exists() and not force:
        return read_json(path), True, 0.0
    t0 = time.perf_counter()
    result = analytic if analytic is not None else zero_mode_diagnostic(L, k, settings)
    payload = {
        "schema": ZERO_MODE_SCHEMA,
        "q": int(q),
        "k": int(k),
        "settings": settings.to_dict(),
        **result.to_dict(),
    }
    write_json(path, payload)
    return payload, False, time.perf_counter() - t0


def _analytic_spectrum_rows(q: int, k: int, dimension: int) -> list[dict[str, Any]]:
    """Exact compact spectrum for the downward closure of one q-simplex."""
    if k == 0:
        rows = [{"eigenvalue": 0.0, "multiplicity": 1}]
        if dimension > 1:
            rows.append({"eigenvalue": float(q + 1), "multiplicity": int(dimension - 1)})
        return rows
    return [{"eigenvalue": float(q + 1), "multiplicity": int(dimension)}]


def _compute_observed_spectrum(
    L: sparse.spmatrix,
    k: int,
    settings: ZeroModeSettings,
    known_zero_count: int | None,
) -> tuple[np.ndarray, str, str]:
    """Return full spectrum for small matrices or a smallest-eigenvalue prefix.

    The prefix is a numerical diagnostic only.  It is never interpreted as a
    complete spectrum when ``n > dense_dimension_guard``.
    """
    A = sparse.csr_matrix(L, dtype=float)
    n = int(A.shape[0])
    if n == 0:
        return np.empty(0, dtype=float), "full", "empty"
    if n == 1:
        return np.asarray([float(A[0, 0])], dtype=float), "full", "scalar"
    if n <= settings.dense_dimension_guard:
        vals = np.linalg.eigvalsh(A.toarray())
        return np.sort(np.asarray(vals, dtype=float)), "full", "dense_eigvalsh"

    cap = min(int(settings.sparse_max_k), n - 1)
    if cap < 1:
        return np.empty(0, dtype=float), "smallest_prefix", "no_sparse_budget"
    request = min(max(1, int(settings.sparse_initial_k)), cap)
    if known_zero_count is not None:
        # If the exact/combinatorial nullity is modest, request at least one
        # positive eigenvalue beyond the zero block when the budget permits.
        request = min(cap, max(request, int(known_zero_count) + 1))
    rng = np.random.default_rng(settings.random_seed + 7919 * int(k) + 31 * n)
    v0 = rng.standard_normal(n)
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
        return np.sort(np.asarray(vals, dtype=float)), "smallest_prefix", "sparse_eigsh"
    except ArpackNoConvergence as exc:
        vals = np.asarray(exc.eigenvalues if exc.eigenvalues is not None else [], dtype=float)
        return np.sort(vals), "smallest_prefix", "sparse_eigsh_partial_no_convergence"


def _load_or_compute_spectrum(
    store: ArtifactStore,
    L: sparse.spmatrix,
    q: int,
    k: int,
    settings: ZeroModeSettings,
    known_zero_count: int | None,
    *,
    force: bool,
) -> tuple[np.ndarray, str, str, bool, float]:
    path = _spectrum_cache_path(store, q, k, settings)
    if path.exists() and not force:
        with np.load(path, allow_pickle=False) as data:
            vals = np.asarray(data["eigenvalues"], dtype=float)
            scope = str(data["scope"].item())
            method = str(data["method"].item())
        return vals, scope, method, True, 0.0
    t0 = time.perf_counter()
    vals, scope, method = _compute_observed_spectrum(L, k, settings, known_zero_count)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, eigenvalues=vals, scope=np.asarray(scope), method=np.asarray(method))
    return vals, scope, method, False, time.perf_counter() - t0


def _domain_order(labels: list[str], domains: list[str]) -> tuple[np.ndarray, list[int]]:
    order = np.arange(len(labels), dtype=int)
    boundaries: list[int] = []
    for i in range(1, len(domains)):
        if domains[i] != domains[i - 1]:
            boundaries.append(i)
    return order, boundaries


def _matrix_from_channel(rows: pd.DataFrame, labels: list[str], q_values: list[int], column: str) -> np.ndarray:
    M = np.full((len(labels), len(q_values)), np.nan, dtype=float)
    li = {x: i for i, x in enumerate(labels)}
    qi = {q: i for i, q in enumerate(q_values)}
    for r in rows.itertuples(index=False):
        label = str(getattr(r, "label"))
        q = int(getattr(r, "q"))
        val = getattr(r, column)
        if label in li and q in qi and val is not None and not pd.isna(val):
            M[li[label], qi[q]] = float(val)
    return M


def _plot_network_q_heatmap(
    rows: pd.DataFrame,
    labels: list[str],
    domains: list[str],
    q_values: list[int],
    column: str,
    title: str,
    cbar_label: str,
    stem: Path,
) -> None:
    if not q_values or rows.empty:
        return
    M = _matrix_from_channel(rows, labels, q_values, column)
    if not np.isfinite(M).any():
        return
    apply_style()
    width = max(13.0, 1.05 * len(q_values) + 7.0)
    height = max(11.0, 0.42 * len(labels) + 4.0)
    fig, ax = plt.subplots(figsize=(width, height))
    masked = np.ma.masked_invalid(M)
    im = ax.imshow(masked, aspect="auto", interpolation="nearest", cmap=DMEASURE_CMAP)
    ax.set_xticks(np.arange(len(q_values)), [str(q) for q in q_values])
    display = [f"{domains[i][:4]} | {labels[i]}" for i in range(len(labels))]
    ax.set_yticks(np.arange(len(labels)), display)
    ax.tick_params(axis="y", labelsize=max(9, min(TICK_SIZE, 13)))
    ax.set_xlabel("Exclusive q-layer")
    ax.set_ylabel("Real network (domain prefix | label)")
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold")
    for i in range(1, len(domains)):
        if domains[i] != domains[i - 1]:
            ax.axhline(i - 0.5, color="white", linewidth=3.0)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
    cb.set_label(cbar_label, fontsize=LABEL_SIZE)
    cb.ax.tick_params(labelsize=TICK_SIZE)
    fig.tight_layout()
    _savefig(fig, stem)


def _plot_qk_heatmap(rows: pd.DataFrame, value: str, title: str, cbar_label: str, stem: Path) -> None:
    if rows.empty:
        return
    pivot = rows.groupby(["q", "k"])[value].mean().unstack("k")
    if pivot.empty:
        return
    qvals = pivot.index.to_numpy(int)
    kvals = pivot.columns.to_numpy(int)
    M = pivot.to_numpy(float)
    apply_style()
    fig, ax = plt.subplots(figsize=(max(11, 1.0 * len(kvals) + 5), max(8, 0.75 * len(qvals) + 4)))
    im = ax.imshow(np.ma.masked_invalid(M), origin="lower", aspect="auto", interpolation="nearest", cmap=DMEASURE_CMAP)
    ax.set_xticks(np.arange(len(kvals)), [str(k) for k in kvals])
    ax.set_yticks(np.arange(len(qvals)), [str(q) for q in qvals])
    ax.set_xlabel("Hodge order k")
    ax.set_ylabel("Exclusive q-layer")
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(cbar_label, fontsize=LABEL_SIZE)
    _savefig(fig, stem)


def _plot_domain_k0_lines(rows: pd.DataFrame, plots: Path) -> None:
    k0 = rows[(rows["k"] == 0) & rows["count_resolved"].astype(bool)].copy()
    if k0.empty:
        return
    apply_style()
    summary = (
        k0.groupby(["domain", "q"], as_index=False)
        .agg(
            median_beta0=("zero_count", "median"),
            median_zero_fraction=("zero_fraction", "median"),
            median_gap=("smallest_positive_eigenvalue", "median"),
        )
    )
    specs = [
        ("median_beta0", r"Median $\beta_0^{(q)}$", "k0_domain_median_beta0"),
        ("median_zero_fraction", r"Median $\beta_0^{(q)}/\dim L_0^{(q)}$", "k0_domain_median_zero_fraction"),
        ("median_gap", r"Median smallest positive eigenvalue", "k0_domain_median_positive_gap"),
    ]
    for col, ylabel, name in specs:
        fig, ax = plt.subplots(figsize=(15, 9))
        for domain in dict.fromkeys(summary["domain"].astype(str).tolist()):
            s = summary[summary["domain"] == domain].sort_values("q")
            if s[col].notna().any():
                ax.plot(s["q"], s[col], marker="o", linewidth=2.6, markersize=7.5, label=domain)
        ax.set_xlabel("Exclusive q-layer")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Real networks: k=0 connectivity by domain — {ylabel}", fontsize=TITLE_SIZE, fontweight="bold")
        ax.legend(frameon=False, ncol=2)
        ax.grid(alpha=0.18)
        _savefig(fig, plots / name)


def run_real_zero_mode_diagnostics(payload: dict[str, Any], config_path: Path, *, force: bool = False) -> Path:
    cfg = method_from_payload(payload, config_path)
    diag = dict(payload.get("real_zero_mode_diagnostics", {}))
    settings = _settings_from_payload(diag, cfg)
    settings.validate()
    max_q = diag.get("max_q", None)
    max_q = None if max_q is None else int(max_q)
    max_k = diag.get("max_k", None)
    max_k = None if max_k is None else int(max_k)
    force_diag = bool(diag.get("force_diagnostic_cache", False))
    store_eigenvalues = bool(diag.get("store_eigenvalues", True))

    df = _prepare_manifest(payload, config_path)
    if df.empty:
        raise ValueError("real zero-mode diagnostics require at least one real network")
    labels = df["label"].astype(str).tolist()
    domains = df["domain"].astype(str).tolist()

    identity = {
        "schema": REAL_ZERO_MODE_SCHEMA,
        "labels": labels,
        "domains": domains,
        "method_structural": {
            "max_clique_size": cfg.max_clique_size,
            "max_hodge_order": cfg.max_hodge_order,
            "min_layer_q": cfg.min_layer_q,
        },
        "max_q": max_q,
        "max_k": max_k,
        "zero_mode_settings": settings.to_dict(),
        "store_eigenvalues": store_eigenvalues,
    }
    root = resolve_path(diag.get("results_root", "results/validation/real_zero_modes"), config_path)
    run = root / f"real_zero_modes_{stable_hash(identity, 12)}"
    complete = run / "RUN_COMPLETE.json"
    if complete.exists() and not force:
        return run
    run.mkdir(parents=True, exist_ok=True)
    if force:
        for name in ("tables", "plots", "arrays"):
            p = run / name
            if p.exists():
                shutil.rmtree(p)
    tables = run / "tables"; plots = run / "plots"; arrays = run / "arrays"
    for p in (tables, plots, arrays):
        p.mkdir(parents=True, exist_ok=True)
    eigdir = arrays / "eigenvalues"
    eigdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(run / "manifest_snapshot.csv", index=False, encoding="utf-8-sig")

    effective = json.loads(json.dumps(payload))
    effective["cache_schema"] = CACHE_SCHEMA
    effective["real_zero_mode_schema"] = REAL_ZERO_MODE_SCHEMA
    effective["resolved_zero_mode_settings"] = settings.to_dict()
    effective["scientific_note"] = (
        "Diagnostic only. k=0 nullity is exact connected-component count. Higher-k unresolved sparse nullities "
        "remain explicitly censored. Small Hodge matrices store the full spectrum; large matrices store only a "
        "smallest-eigenvalue prefix. No heat signature or pairwise Hodge distance is changed."
    )
    write_json(run / "effective_config.json", effective)

    graph_rows: list[dict[str, Any]] = []
    channel_rows: list[dict[str, Any]] = []
    spectrum_index_rows: list[dict[str, Any]] = []
    analytic_rows: list[dict[str, Any]] = []

    for gi, row in enumerate(df.itertuples(index=False)):
        label = str(row.label); domain = str(row.domain)
        rec = create_real_record(row.path, cfg.cache.root, label, row.nodes_path or None)
        store = ArtifactStore(rec.cache_root, cfg.cache)
        print(f"[REAL-ZERO] {gi+1}/{len(df)} {domain}/{label}: N={rec.graph.number_of_nodes()} E={rec.graph.number_of_edges()}", flush=True)
        all_layers = enumerate_maximal_clique_layers(rec.graph, store, progress=None)
        layers = retained_layers(all_layers, cfg.max_clique_size, cfg.min_layer_q)
        if max_q is not None:
            layers = {q: cl for q, cl in layers.items() if q <= max_q}
        graph_rows.append({
            "graph_index": gi,
            "label": label,
            "domain": domain,
            "graph_hash": rec.graph_hash,
            "N": int(rec.graph.number_of_nodes()),
            "E": int(rec.graph.number_of_edges()),
            "retained_q_max": max(layers) if layers else None,
            "retained_q_count": len(layers),
        })

        for q, cliques in sorted(layers.items()):
            kmax = required_max_hodge_order(q, cfg.max_hodge_order)
            if max_k is not None:
                kmax = min(kmax, max_k)
            for k in range(kmax + 1):
                analytic = analytic_single_simplex_zero_mode(q, k) if len(cliques) == 1 else None
                L = None if analytic is not None else get_hodge_laplacian(q, k, cliques, store, progress=None)
                result, zero_hit, zero_seconds = _load_or_compute_zero(
                    store, L, q, k, settings, force=force_diag, analytic=analytic
                )

                spectrum_scope = "not_requested"
                spectrum_method = "not_requested"
                spectrum_hit = False
                spectrum_seconds = 0.0
                spectrum_count = 0
                spectrum_file = ""
                spectrum_min = None
                spectrum_max = None

                if store_eigenvalues:
                    if analytic is not None:
                        spectrum_scope = "analytic_compact"
                        spectrum_method = "analytic_full_simplex"
                        comp = _analytic_spectrum_rows(q, k, int(result["dimension"]))
                        spectrum_count = int(sum(int(x["multiplicity"]) for x in comp))
                        spectrum_min = float(min(x["eigenvalue"] for x in comp)) if comp else None
                        spectrum_max = float(max(x["eigenvalue"] for x in comp)) if comp else None
                        for item in comp:
                            analytic_rows.append({
                                "graph_index": gi, "label": label, "domain": domain, "q": q, "k": k,
                                "eigenvalue": item["eigenvalue"], "multiplicity": item["multiplicity"],
                                "dimension": int(result["dimension"]),
                            })
                    else:
                        known = int(result["zero_count"]) if result.get("count_resolved") and result.get("zero_count") is not None else None
                        vals, spectrum_scope, spectrum_method, spectrum_hit, spectrum_seconds = _load_or_compute_spectrum(
                            store, L, q, k, settings, known, force=force_diag
                        )
                        spectrum_count = int(len(vals))
                        spectrum_min = float(vals[0]) if len(vals) else None
                        spectrum_max = float(vals[-1]) if len(vals) else None
                        out = eigdir / f"{safe_token(label)}_q{q:03d}_k{k:03d}.npy"
                        np.save(out, vals)
                        spectrum_file = str(out.relative_to(run))

                channel_rows.append({
                    "graph_index": gi,
                    "label": label,
                    "domain": domain,
                    "graph_hash": rec.graph_hash,
                    "N": int(rec.graph.number_of_nodes()),
                    "E": int(rec.graph.number_of_edges()),
                    "q": q,
                    "k": k,
                    "num_maximal_cliques_in_q": len(cliques),
                    "zero_diagnostic_cache_hit": zero_hit,
                    "zero_diagnostic_seconds": zero_seconds,
                    "spectrum_cache_hit": spectrum_hit,
                    "spectrum_seconds": spectrum_seconds,
                    "spectrum_scope": spectrum_scope,
                    "spectrum_method": spectrum_method,
                    "eigenvalues_saved": spectrum_count,
                    "spectrum_file": spectrum_file,
                    "observed_eigenvalue_min": spectrum_min,
                    "observed_eigenvalue_max": spectrum_max,
                    **{kk: vv for kk, vv in result.items() if kk not in {"schema", "q", "k", "settings"}},
                })
                spectrum_index_rows.append({
                    "graph_index": gi, "label": label, "domain": domain, "q": q, "k": k,
                    "dimension": int(result["dimension"]), "spectrum_scope": spectrum_scope,
                    "spectrum_method": spectrum_method, "eigenvalues_saved": spectrum_count,
                    "spectrum_file": spectrum_file, "observed_eigenvalue_min": spectrum_min,
                    "observed_eigenvalue_max": spectrum_max,
                    "full_spectrum_available": bool(spectrum_scope in {"full", "analytic_compact"}),
                })

    gdf = pd.DataFrame(graph_rows)
    rdf = pd.DataFrame(channel_rows)
    sdf = pd.DataFrame(spectrum_index_rows)
    adf = pd.DataFrame(analytic_rows)
    gdf.to_csv(tables / "real_graph_index.csv", index=False, encoding="utf-8-sig")
    rdf.to_csv(tables / "real_zero_modes_per_network_channel.csv", index=False, encoding="utf-8-sig")
    sdf.to_csv(tables / "real_eigenvalue_index.csv", index=False, encoding="utf-8-sig")
    if not adf.empty:
        adf.to_csv(tables / "real_analytic_spectra_compact.csv", index=False, encoding="utf-8-sig")

    if not rdf.empty:
        resolved = rdf[rdf["count_resolved"].astype(bool)].copy()
        if not resolved.empty:
            domain_summary = (
                resolved.groupby(["domain", "q", "k"], as_index=False)
                .agg(
                    networks=("label", "count"),
                    median_zero_count=("zero_count", "median"),
                    mean_zero_count=("zero_count", "mean"),
                    median_zero_fraction=("zero_fraction", "median"),
                    mean_zero_fraction=("zero_fraction", "mean"),
                    median_dimension=("dimension", "median"),
                    median_smallest_positive_eigenvalue=("smallest_positive_eigenvalue", "median"),
                )
            )
            domain_summary.to_csv(tables / "real_zero_modes_domain_summary.csv", index=False, encoding="utf-8-sig")
            global_summary = (
                resolved.groupby(["q", "k"], as_index=False)
                .agg(
                    networks=("label", "count"),
                    median_zero_count=("zero_count", "median"),
                    mean_zero_count=("zero_count", "mean"),
                    median_zero_fraction=("zero_fraction", "median"),
                    mean_zero_fraction=("zero_fraction", "mean"),
                    median_smallest_positive_eigenvalue=("smallest_positive_eigenvalue", "median"),
                )
            )
            global_summary.to_csv(tables / "real_zero_modes_global_summary.csv", index=False, encoding="utf-8-sig")
            _plot_qk_heatmap(
                resolved, "zero_fraction", "Real networks: mean resolved Hodge zero-mode fraction",
                "Mean zero-mode fraction", plots / "global_resolved_zero_fraction_q_by_k",
            )
        resolution = (
            rdf.assign(resolved=rdf["count_resolved"].astype(bool).astype(int))
            .groupby(["q", "k"], as_index=False)
            .agg(rows=("label", "count"), resolved=("resolved", "sum"))
        )
        resolution["resolution_rate"] = resolution["resolved"] / resolution["rows"]
        resolution.to_csv(tables / "real_zero_mode_resolution_by_qk.csv", index=False, encoding="utf-8-sig")
        _plot_qk_heatmap(
            resolution, "resolution_rate", "Real networks: zero-mode count resolution rate",
            "Resolution rate", plots / "global_zero_mode_resolution_rate_q_by_k",
        )

        q_values = sorted(int(x) for x in rdf["q"].unique())
        k0 = rdf[(rdf["k"] == 0) & rdf["count_resolved"].astype(bool)].copy()
        if not k0.empty:
            k0[["label", "domain", "q", "zero_count", "zero_fraction", "dimension", "smallest_positive_eigenvalue"]].to_csv(
                tables / "real_k0_connectivity_per_network_q.csv", index=False, encoding="utf-8-sig"
            )
            _plot_network_q_heatmap(
                k0, labels, domains, q_values, "zero_count",
                r"Real networks: exact $\beta_0^{(q)}$ across exclusive q-layers",
                r"$\beta_0^{(q)}$ / connected components", plots / "network_by_q_k0_beta0",
            )
            _plot_network_q_heatmap(
                k0, labels, domains, q_values, "zero_fraction",
                r"Real networks: normalized k=0 harmonic fraction",
                r"$\beta_0^{(q)}/\dim L_0^{(q)}$", plots / "network_by_q_k0_zero_fraction",
            )
            gap_rows = k0[k0["smallest_positive_eigenvalue"].notna()].copy()
            if not gap_rows.empty:
                _plot_network_q_heatmap(
                    gap_rows, labels, domains, q_values, "smallest_positive_eigenvalue",
                    "Real networks: k=0 smallest positive Hodge eigenvalue",
                    "Smallest positive eigenvalue", plots / "network_by_q_k0_positive_gap",
                )
            _plot_domain_k0_lines(rdf, plots)

    write_json(run / "run_manifest.json", {
        "schema": REAL_ZERO_MODE_SCHEMA,
        "cache_schema": CACHE_SCHEMA,
        "num_networks": int(len(df)),
        "labels": labels,
        "domains": list(dict.fromkeys(domains)),
        "method_structural": {
            "max_clique_size": cfg.max_clique_size,
            "max_hodge_order": cfg.max_hodge_order,
            "min_layer_q": cfg.min_layer_q,
        },
        "diagnostic_max_q": max_q,
        "diagnostic_max_k": max_k,
        "zero_mode_settings": asdict(settings),
        "store_eigenvalues": store_eigenvalues,
        "mathematics_changed": False,
        "cache_identity_changed": False,
        "experimental_statistics_changed": False,
        "random_graph_seed_protocol_changed": False,
        "baseline_semantics_changed": False,
        "existing_result_comparability": (
            "This run is diagnostic-only and does not alter previous Hodge distances. Full spectra are only claimed "
            "where real_eigenvalue_index.csv marks full_spectrum_available=true; large-matrix prefixes are not full spectra."
        ),
    })
    # Generate report tables and figures directly from the completed run artifacts.
    # This post-processing does not touch C1--Q7 cache artifacts or any distance.
    report_cfg = dict(diag.get("reporting", {}))
    if bool(report_cfg.get("enabled", True)):
        generate_real_zero_mode_report(
            run,
            labels=None,
            max_q=max_q,
            spectral_dashboards=bool(report_cfg.get("spectral_dashboards", True)),
            network_heatmaps=bool(report_cfg.get("network_heatmaps", True)),
            save_pdf=bool(report_cfg.get("save_pdf", False)),
        )

    write_json(complete, {"completed": True, "schema": REAL_ZERO_MODE_SCHEMA, "run_dir": str(run)})
    print(f"[DONE] {run}", flush=True)
    return run
