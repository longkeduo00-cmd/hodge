from __future__ import annotations

"""Low-order weighting sweep for the existing pair-adaptive Hodge distance.

Only the final scalar aggregation parameter alpha is varied.  Graph generation,
exclusive-q complexes, Hodge matrices, solver outputs, heat signatures and
normalization are unchanged.  The runner reuses Q7 profiles and recomputes only
pairwise aggregation/evaluation.
"""

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any
import json
import math
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap
from scipy.stats import spearmanr

from hodge_vnext.cache import CACHE_SCHEMA
from hodge_vnext.cliques import enumerate_maximal_clique_layers, retained_layers
from hodge_vnext.cache import ArtifactStore
from hodge_vnext.distance import active_hodge_count
from hodge_vnext.models import synthetic_graph_record
from hodge_vnext.profile import compute_graph_profile
from hodge_vnext.utils import stable_hash, write_json

from .config import method_from_payload, resolve_path
from .model_discrimination import _params_for_state, _state_from_beta
from .model_joint_discrimination import (
    FINE_STATE_ORDER,
    _cell_stats,
    _retrieval_metrics_joint,
    build_joint_cells,
)
from .style import FIG_DPI, TICK_SIZE, TITLE_SIZE, apply_style


ALPHA_SWEEP_SCHEMA = "hodge-alpha-low-order-sweep-v1"
PROTOCOLS = ["matched_N_k", "cross_N_only", "cross_k_only", "cross_N_k", "global"]
CORE_METRICS = [
    ("balanced_accuracy", "Balanced accuracy"),
    ("macro_f1", "Macro-F1"),
    ("same_class_auroc", "AUROC"),
    ("mean_average_precision", "mAP"),
    ("ndcg_at_5", "NDCG@5"),
]


def _alpha_slug(alpha: float) -> str:
    s = f"{float(alpha):.6g}".replace("-", "m").replace(".", "p")
    return f"a{s}"


def _validate_alphas(values: list[float]) -> list[float]:
    out = [float(x) for x in values]
    if not out or not all(math.isfinite(x) for x in out):
        raise ValueError("alpha_sweep.alphas must be a non-empty finite list")
    if len(set(out)) != len(out):
        raise ValueError("alpha_sweep.alphas must be unique")
    return out


def alpha_terms(q_values: list[int], alphas: list[float], max_hodge_order: int | None) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Return per-layer numerator coefficients and pair-q* denominators.

    For one layer q, local squared distance S_q = sum_k d_{q,k}^2.
    The pair-adaptive distance is

        D_alpha^2 = [sum_q A_q^(alpha-1) S_q] / [sum_{r<=q*} A_r^alpha].
    """
    if not q_values:
        return np.empty((len(alphas), 0), dtype=float), {}
    counts = np.asarray([active_hodge_count(q, max_hodge_order) for q in q_values], dtype=float)
    coeff = np.asarray([[a ** (alpha - 1.0) for a in counts] for alpha in alphas], dtype=float)
    denominators: dict[int, np.ndarray] = {}
    cumulative = np.zeros(len(alphas), dtype=float)
    for qi, q in enumerate(q_values):
        cumulative = cumulative + np.asarray([counts[qi] ** alpha for alpha in alphas], dtype=float)
        denominators[q] = cumulative.copy()
    return coeff, denominators


def distance_block_from_layer_features(
    layer_features: dict[int, np.ndarray],
    q_max: np.ndarray,
    q_values: list[int],
    alphas: list[float],
    max_hodge_order: int | None,
    row_start: int,
    row_end: int,
) -> np.ndarray:
    """Vectorized exact aggregation for a row block.

    Returns shape (n_alpha, row_end-row_start, n_graphs).  This helper is also
    used by regression tests against profile_distance.
    """
    n = len(q_max)
    b = row_end - row_start
    coeff, denominators = alpha_terms(q_values, alphas, max_hodge_order)
    numer = np.zeros((len(alphas), b, n), dtype=np.float64)
    for qi, q in enumerate(q_values):
        X = np.asarray(layer_features[q], dtype=np.float64)
        xb = X[row_start:row_end]
        na = np.einsum("ij,ij->i", xb, xb)
        nb = np.einsum("ij,ij->i", X, X)
        sq = na[:, None] + nb[None, :] - 2.0 * (xb @ X.T)
        np.maximum(sq, 0.0, out=sq)
        for ai in range(len(alphas)):
            numer[ai] += coeff[ai, qi] * sq

    qstar = np.maximum(q_max[row_start:row_end, None], q_max[None, :])
    out = np.zeros_like(numer)
    qmin = q_values[0] if q_values else 0
    valid = qstar >= qmin
    for ai in range(len(alphas)):
        denom = np.ones((b, n), dtype=np.float64)
        for q in q_values:
            denom[qstar == q] = denominators[q][ai]
        vals = np.zeros((b, n), dtype=np.float64)
        vals[valid] = np.sqrt(np.maximum(numer[ai][valid] / denom[valid], 0.0))
        for local, gi in enumerate(range(row_start, row_end)):
            vals[local, gi] = 0.0
        out[ai] = vals
    return out


def _feature_layout(q_values: list[int], time_points: int, max_hodge_order: int | None) -> tuple[dict[int, slice], int]:
    offsets: dict[int, slice] = {}
    pos = 0
    for q in q_values:
        width = active_hodge_count(q, max_hodge_order) * int(time_points)
        offsets[q] = slice(pos, pos + width)
        pos += width
    return offsets, pos


def _build_records_index(
    cfg,
    cells: pd.DataFrame,
    repeats: int,
    base_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for c in cells.itertuples(index=False):
        state = str(c.mechanism_state)
        n = int(c.network_size)
        kappa = float(c.target_mean_degree)
        model, params, _ = _params_for_state(state, n, kappa)
        print(f"[INDEX] {state} N={n} kappa={kappa:g}", flush=True)
        for repeat in range(1, repeats + 1):
            rec = synthetic_graph_record(cfg.cache.root, model, n, params, base_seed, repeat)
            layers = enumerate_maximal_clique_layers(rec.graph, ArtifactStore(rec.cache_root, cfg.cache), progress=None)
            kept = retained_layers(layers, cfg.max_clique_size, cfg.min_layer_q)
            rows.append({
                "graph_index": len(rows),
                "cell_index": int(c.cell_index),
                "cell_label": str(c.cell_label),
                "mechanism_state": state,
                "model": model,
                "network_size": n,
                "target_mean_degree": kappa,
                "repeat": repeat,
                "graph_hash": rec.graph_hash,
                "graph_seed": int(rec.source["graph_seed"]),
                "retained_q_max": max(kept) if kept else -1,
            })
    return pd.DataFrame(rows)


def _materialize_feature_matrix(
    cfg,
    gdf: pd.DataFrame,
    q_values: list[int],
    offsets: dict[int, slice],
    total_dim: int,
    scratch_file: Path,
    dtype: str,
) -> np.memmap:
    X = open_memmap(scratch_file, mode="w+", dtype=np.dtype(dtype), shape=(len(gdf), total_dim))
    X[:] = 0
    T = int(cfg.time_points)
    for row in gdf.itertuples(index=False):
        if int(row.graph_index) % 100 == 0:
            print(f"[PROFILE] {int(row.graph_index)}/{len(gdf)}", flush=True)
        state = str(row.mechanism_state)
        n = int(row.network_size)
        kappa = float(row.target_mean_degree)
        model, params, _ = _params_for_state(state, n, kappa)
        rec = synthetic_graph_record(cfg.cache.root, model, n, params, int(row.base_seed), int(row.repeat)) if hasattr(row, "base_seed") else None
        if rec is None:
            raise RuntimeError("internal graph index is missing base_seed")
        profile = compute_graph_profile(rec, cfg, progress=None)
        if int(profile.q_max if profile.q_max is not None else -1) != int(row.retained_q_max):
            raise RuntimeError("profile q_max does not match graph-index retained_q_max")
        gi = int(row.graph_index)
        for q in q_values:
            sl = offsets[q]
            count = active_hodge_count(q, cfg.max_hodge_order)
            cursor = sl.start
            for k in range(count):
                channel = profile.channels.get((q, k))
                if channel is not None:
                    if len(channel) != T:
                        raise RuntimeError("inconsistent time-grid length in cached profile")
                    X[gi, cursor:cursor + T] = channel
                cursor += T
    X.flush()
    return X


def _cell_nuisance_correlations(Dcell: np.ndarray, cells: pd.DataFrame) -> dict[str, float]:
    tri = np.triu_indices(len(cells), 1)
    d = np.asarray(Dcell, dtype=float)[tri]
    ns = cells["network_size"].to_numpy(float)
    ks = cells["target_mean_degree"].to_numpy(float)
    dn = np.abs(ns[:, None] - ns[None, :])[tri]
    dk = np.abs(ks[:, None] - ks[None, :])[tri]
    state = cells["mechanism_state"].astype(str).to_numpy()
    same_state = (state[:, None] == state[None, :])[tri]
    def corr(x, mask=None):
        if mask is not None:
            x2, d2 = x[mask], d[mask]
        else:
            x2, d2 = x, d
        if len(x2) < 3 or np.allclose(x2, x2[0]) or np.allclose(d2, d2[0]):
            return float("nan")
        return float(spearmanr(d2, x2).statistic)
    return {
        "spearman_distance_abs_delta_N_all_cells": corr(dn),
        "spearman_distance_abs_delta_k_all_cells": corr(dk),
        "spearman_distance_abs_delta_N_within_mechanism": corr(dn, same_state),
        "spearman_distance_abs_delta_k_within_mechanism": corr(dk, same_state),
    }


def _plot_alpha_metrics(metrics: pd.DataFrame, plots: Path) -> None:
    apply_style()
    protocols = PROTOCOLS
    cmap = plt.get_cmap("tab10")
    for metric, label in CORE_METRICS:
        fig, ax = plt.subplots(figsize=(14.5, 9))
        for i, protocol in enumerate(protocols):
            s = metrics[metrics["protocol"] == protocol].sort_values("alpha", ascending=False)
            ax.plot(s["alpha"], s[metric], marker="o", linewidth=2.4, markersize=7,
                    label=protocol, color=cmap(i % 10))
        ax.set_xlabel(r"$\alpha$ (smaller = stronger low-order emphasis)")
        ax.set_ylabel(label)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Hodge low-order weighting sweep — {label}", fontsize=TITLE_SIZE, fontweight="bold")
        ax.legend(frameon=False, ncol=2)
        ax.grid(alpha=0.18)
        fig.savefig((plots / f"alpha_sweep_{metric}.png"), dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
        fig.savefig((plots / f"alpha_sweep_{metric}.pdf"), bbox_inches="tight", facecolor="white")
        plt.close(fig)


def _plot_nuisance(corrs: pd.DataFrame, plots: Path) -> None:
    apply_style()
    cols = [
        ("spearman_distance_abs_delta_N_within_mechanism", r"$|\Delta N|$ within mechanism"),
        ("spearman_distance_abs_delta_k_within_mechanism", r"$|\Delta k|$ within mechanism"),
    ]
    fig, ax = plt.subplots(figsize=(14.5, 9))
    for col, label in cols:
        s = corrs.sort_values("alpha", ascending=False)
        ax.plot(s["alpha"], s[col], marker="o", linewidth=2.4, markersize=7, label=label)
    ax.axhline(0, color="0.25", linewidth=1.3)
    ax.set_xlabel(r"$\alpha$ (smaller = stronger low-order emphasis)")
    ax.set_ylabel("Spearman correlation with cell-mean Hodge distance")
    ax.set_title("Nuisance sensitivity across the alpha sweep", fontsize=TITLE_SIZE, fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(alpha=0.18)
    fig.savefig(plots / "alpha_sweep_nuisance_correlations.png", dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(plots / "alpha_sweep_nuisance_correlations.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run_hodge_alpha_sweep(payload: dict[str, Any], config_path: Path, *, force: bool = False) -> Path:
    cfg = method_from_payload(payload, config_path)
    sweep = dict(payload.get("alpha_sweep", {}))
    alphas = _validate_alphas(sweep.get("alphas", [1.0, 0.75, 0.5, 0.25, 0.0, -0.5, -1.0]))
    sizes = [int(x) for x in sweep.get("network_sizes", [100, 250, 500, 1000, 2000])]
    kappas = [float(x) for x in sweep.get("target_degrees", [4, 6, 8, 10, 12, 14])]
    repeats = int(sweep.get("repeats", 50))
    base_seed = int(sweep.get("base_seed", 22))
    ws_betas = [float(x) for x in sweep.get("ws_betas", [0.001, 0.01, 0.1, 1.0])]
    tie_seed = int(sweep.get("tie_seed", 20260922))
    block_size = int(sweep.get("block_size", 64))
    feature_dtype = str(sweep.get("feature_dtype", "float64"))
    distance_dtype = str(sweep.get("distance_dtype", "float32"))
    persist_graph = bool(sweep.get("persist_graph_distance_matrices", False))
    keep_scratch = bool(sweep.get("keep_scratch", False))
    if repeats < 2:
        raise ValueError("alpha sweep repeats must be >= 2")
    if block_size < 1:
        raise ValueError("alpha sweep block_size must be >= 1")
    if feature_dtype not in {"float32", "float64"} or distance_dtype not in {"float32", "float64"}:
        raise ValueError("feature_dtype/distance_dtype must be float32 or float64")
    states = ["ER", "RGG"] + [_state_from_beta(b) for b in ws_betas]
    if tuple(states) != FINE_STATE_ORDER:
        raise ValueError("alpha sweep requires WS betas [1e-3,1e-2,1e-1,1]")

    cells = build_joint_cells(sizes, kappas, ws_betas)
    # alpha is excluded from the Q7 profile cache key already; freeze a neutral
    # value here to make that independence explicit in the effective config.
    profile_cfg = replace(cfg, alpha=1.0)
    identity = {
        "schema": ALPHA_SWEEP_SCHEMA,
        "alphas": alphas,
        "network_sizes": sizes,
        "target_degrees": kappas,
        "repeats": repeats,
        "base_seed": base_seed,
        "ws_betas": ws_betas,
        "method_without_alpha": {k: v for k, v in asdict(profile_cfg).items() if k != "alpha"},
        "block_size": block_size,
    }
    root = resolve_path(sweep.get("results_root", "results/validation/hodge_alpha_sweep"), config_path)
    run = root / f"alpha_sweep_{stable_hash(identity, 12)}"
    complete = run / "RUN_COMPLETE.json"
    if complete.exists() and not force:
        return run
    run.mkdir(parents=True, exist_ok=True)
    if force:
        for name in ("tables", "arrays", "plots", "scratch"):
            p = run / name
            if p.exists():
                shutil.rmtree(p)
    tables = run / "tables"; arrays = run / "arrays"; plots = run / "plots"; scratch = run / "scratch"
    for p in (tables, arrays, plots, scratch):
        p.mkdir(parents=True, exist_ok=True)

    eff = json.loads(json.dumps(payload))
    eff["cache_schema"] = CACHE_SCHEMA
    eff["alpha_sweep_schema"] = ALPHA_SWEEP_SCHEMA
    eff["resolved_profile_alpha"] = 1.0
    eff["scientific_note"] = (
        "Only final pairwise alpha aggregation is varied. Q7 profiles are shared across alphas; "
        "no clique/Hodge/spectral/heat definition is changed."
    )
    write_json(run / "effective_config.json", eff)
    cells.to_csv(tables / "parameter_cells.csv", index=False, encoding="utf-8-sig")

    gdf = _build_records_index(profile_cfg, cells, repeats, base_seed)
    gdf["base_seed"] = base_seed
    gdf.to_csv(tables / "graph_index.csv", index=False, encoding="utf-8-sig")
    global_qmax = int(gdf["retained_q_max"].max()) if len(gdf) else -1
    q_values = list(range(profile_cfg.min_layer_q, global_qmax + 1)) if global_qmax >= profile_cfg.min_layer_q else []
    offsets, total_dim = _feature_layout(q_values, profile_cfg.time_points, profile_cfg.max_hodge_order)
    write_json(tables / "feature_layout.json", {
        "q_values": q_values,
        "time_points": profile_cfg.time_points,
        "total_feature_dimension": total_dim,
        "per_q": {str(q): {"start": offsets[q].start, "stop": offsets[q].stop,
                            "active_hodge_count": active_hodge_count(q, profile_cfg.max_hodge_order)} for q in q_values},
    })

    feature_file = scratch / "unweighted_qk_features.npy"
    X = _materialize_feature_matrix(profile_cfg, gdf, q_values, offsets, total_dim, feature_file, feature_dtype)
    qmax = gdf["retained_q_max"].to_numpy(int)
    layer_features = {q: X[:, offsets[q]] for q in q_values}

    # Create one temporary graph-distance memmap per alpha.  The q-resolved
    # squared distances are computed once per row block and combined for every
    # alpha before the block is discarded.
    dmaps: list[np.memmap] = []
    dpaths: list[Path] = []
    for alpha in alphas:
        p = scratch / f"graph_distance_{_alpha_slug(alpha)}.npy"
        dpaths.append(p)
        dmaps.append(open_memmap(p, mode="w+", dtype=np.dtype(distance_dtype), shape=(len(gdf), len(gdf))))

    for start in range(0, len(gdf), block_size):
        end = min(len(gdf), start + block_size)
        print(f"[DISTANCE] rows {start}:{end}/{len(gdf)}", flush=True)
        blocks = distance_block_from_layer_features(
            layer_features, qmax, q_values, alphas, profile_cfg.max_hodge_order, start, end
        )
        for ai, Dm in enumerate(dmaps):
            block = blocks[ai]
            for local, gi in enumerate(range(start, end)):
                block[local, gi] = 0.0
            Dm[start:end, :] = block.astype(distance_dtype, copy=False)
    for Dm in dmaps:
        Dm.flush()

    ids = gdf["graph_hash"].astype(str).to_numpy()
    labels = gdf["mechanism_state"].astype(str).to_numpy()
    ns = gdf["network_size"].to_numpy(int)
    ks = gdf["target_mean_degree"].to_numpy(float)
    cell_idx = gdf["cell_index"].to_numpy(int)
    metrics_rows: list[dict[str, Any]] = []
    nuisance_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []

    _, denominators = alpha_terms(q_values, alphas, profile_cfg.max_hodge_order)
    for ai, alpha in enumerate(alphas):
        D = np.asarray(dmaps[ai])
        mean, std, count = _cell_stats(D, cell_idx, len(cells))
        slug = _alpha_slug(alpha)
        np.save(arrays / f"cell_distance_Hodge_{slug}.npy", mean)
        np.save(arrays / f"cell_distance_std_Hodge_{slug}.npy", std)
        np.save(arrays / f"cell_pair_count_Hodge_{slug}.npy", count)
        if persist_graph:
            np.save(arrays / f"graph_distance_Hodge_{slug}.npy", D)

        for protocol in PROTOCOLS:
            stats, qdf = _retrieval_metrics_joint(D, ids, labels, ns, ks, protocol, tie_seed)
            metrics_rows.append({"alpha": alpha, "protocol": protocol, **stats})
            qdf.to_csv(tables / f"query_metrics_{protocol}_Hodge_{slug}.csv", index=False, encoding="utf-8-sig")

        nuisance_rows.append({"alpha": alpha, **_cell_nuisance_correlations(mean, cells)})
        if q_values:
            qstar = q_values[-1]
            denom = denominators[qstar][ai]
            for q in q_values:
                Aq = active_hodge_count(q, profile_cfg.max_hodge_order)
                layer_weight = (float(Aq) ** alpha) / denom
                matrix_weight = layer_weight / Aq
                weight_rows.append({
                    "alpha": alpha, "reference_q_star": qstar, "q": q,
                    "active_hodge_count": Aq,
                    "layer_weight": layer_weight, "per_matrix_weight": matrix_weight,
                })
        del D

    metrics = pd.DataFrame(metrics_rows)
    nuisance = pd.DataFrame(nuisance_rows)
    weights = pd.DataFrame(weight_rows)
    metrics.to_csv(tables / "alpha_sweep_metrics.csv", index=False, encoding="utf-8-sig")
    nuisance.to_csv(tables / "alpha_sweep_nuisance_correlations.csv", index=False, encoding="utf-8-sig")
    weights.to_csv(tables / "alpha_sweep_reference_weights.csv", index=False, encoding="utf-8-sig")
    _plot_alpha_metrics(metrics, plots)
    _plot_nuisance(nuisance, plots)

    manifest = {
        "schema": ALPHA_SWEEP_SCHEMA,
        "cache_schema": CACHE_SCHEMA,
        "alphas": alphas,
        "num_graphs": len(gdf),
        "num_cells": len(cells),
        "q_values": q_values,
        "profile_method": asdict(profile_cfg),
        "mathematics_changed": False,
        "cache_identity_changed": False,
        "experimental_statistics_changed": False,
        "random_graph_seed_protocol_changed": False,
        "baseline_semantics_changed": False,
        "existing_result_comparability": (
            "Each alpha value is a distinct final aggregation setting. Upstream Q7 signatures are directly shared; "
            "Hodge distances across different alpha values should not be mixed as if they were the same method setting."
        ),
    }
    write_json(run / "run_manifest.json", manifest)
    write_json(run / "RUN_COMPLETE.json", {"completed": True, "schema": ALPHA_SWEEP_SCHEMA})

    # Close memmaps explicitly before optional scratch deletion.  This matters
    # on Windows, where an open memory map prevents file removal.
    del layer_features
    X.flush()
    x_mmap = getattr(X, "_mmap", None)
    if x_mmap is not None:
        x_mmap.close()
    del X
    for Dm in dmaps:
        Dm.flush()
        d_mmap = getattr(Dm, "_mmap", None)
        if d_mmap is not None:
            d_mmap.close()
    dmaps.clear()
    if not keep_scratch:
        shutil.rmtree(scratch, ignore_errors=False)
    print(f"[DONE] {run}", flush=True)
    return run
