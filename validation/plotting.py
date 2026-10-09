from __future__ import annotations

"""MCID-style synthetic plotting for the Hodge VNext validation runner.

Revision 2026-09-09
-------------------
Line figures that compare *methods* use a strict vertical-panel contract:
one method per row and one independent y-axis per method. Distance heatmaps are
written as separate square figures, one file per method. Methods are never
overlaid on the same line axis.  This is intentional because Hodge,
DMeasure, NetLSD, Portrait and Count have different numerical scales; an
overlay can visually compress several methods and hide their transition shape.

The contract applies uniformly to ER, WS, SBM and RGG.  Order-resolved Hodge
figures and clique-abundance figures are not method-comparison figures, so they
continue to overlay q-series on one axis.
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, optimal_leaf_ordering, leaves_list
from scipy.spatial.distance import squareform

from .style import (
    DMEASURE_CMAP, FIG_DPI, LABEL_SIZE, LEGEND_SIZE, LINE_WIDTH, MARKER_SIZE,
    METHOD_COLORS, SUPTITLE_SIZE, TICK_SIZE, TITLE_SIZE, apply_style,
)
from .theory import model_references


def _save(fig, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_mechanism(fig, base: Path, *, overwrite: bool = False) -> None:
    """Publication-quality save with restart-safe figure reuse."""
    base.parent.mkdir(parents=True, exist_ok=True)
    png = base.with_suffix(".png")
    pdf = base.with_suffix(".pdf")
    if not overwrite and png.exists() and pdf.exists():
        plt.close(fig)
        return
    fig.savefig(png, dpi=min(FIG_DPI, 220), bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _reference_lines(ax, refs: dict[str, float], model: str, *, show_labels: bool = True) -> None:
    for i, (name, x) in enumerate(refs.items()):
        ls = "--" if i == 0 else ":"
        ax.axvline(
            float(x), linestyle=ls, linewidth=2.6, color="0.15",
            label=name if show_labels else None,
        )


def _axis_scale(ax, model: str) -> None:
    if model in {"ER", "WS", "RGG"}:
        ax.set_xscale("log")


def _heatmap_tick_positions(coords: np.ndarray, max_ticks: int = 10):
    n = len(coords)
    idx = np.unique(np.linspace(0, n - 1, min(max_ticks, n), dtype=int))
    return idx, [f"{coords[i]:.3g}" for i in idx]


def _method_order(methods: list[str]) -> list[str]:
    preferred = ["Hodge", "Proposed", "DMeasure", "NetLSD", "Portrait", "Count"]
    out = [m for m in preferred if m in methods]
    out.extend(m for m in methods if m not in out)
    # Hodge and Proposed are aliases in different plotting layers; never show
    # both when only one is actually present.
    return list(dict.fromkeys(out))


def _stacked_method_axes(num_methods: int, *, row_height: float = 4.25, width: float = 15.5, sharex: bool = True):
    fig, axes = plt.subplots(
        num_methods,
        1,
        figsize=(width, max(7.5, row_height * num_methods)),
        sharex=sharex,
        squeeze=False,
    )
    return fig, axes[:, 0]


def _annotate_bars(ax, bars, values, *, fmt: str = "{:.3f}", fontsize: int = 12, zero_floor: float = 1e-12) -> None:
    """Write exact numeric values above bars.

    This is purely a plotting-layer change requested for synthetic
    ER/WS/RGG comparison figures; it does not modify any metric value.
    """
    ymax = float(ax.get_ylim()[1]) if ax.get_ylim()[1] > 0 else 1.0
    offset = 0.018 * ymax
    for rect, value in zip(bars, values):
        v = float(value)
        label = "0.000" if abs(v) <= zero_floor else fmt.format(v)
        x = rect.get_x() + rect.get_width() / 2.0
        y = rect.get_height()
        ypos = min(y + offset, ymax - 0.03 * ymax)
        ax.text(x, ypos, label, ha="center", va="bottom", fontsize=fontsize)


def _method_panel_legend(ax, method: str, refs: dict[str, float], *, show_ref_labels: bool) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            labels,
            frameon=True,
            loc="best",
            fontsize=max(12, LEGEND_SIZE - 2),
            ncol=1 if show_ref_labels else 1,
        )


def _single_method_distance_heatmap(D, coords, coordinate_label, refs, *, model, method, figure_size=10.8):
    fig, ax = plt.subplots(figsize=(figure_size, figure_size))
    im = ax.imshow(D, origin="lower", cmap=DMEASURE_CMAP, aspect="equal", vmin=0)
    idx, labels = _heatmap_tick_positions(coords)
    ax.set_xticks(idx, labels, rotation=45, ha="right")
    ax.set_yticks(idx, labels)
    ax.set_xlabel(coordinate_label)
    ax.set_ylabel(coordinate_label)
    ax.set_title(f"{model}: {method} raw dissimilarity", fontweight="bold", pad=10)
    for ri, (_, ref) in enumerate(refs.items()):
        if coords[0] <= ref <= coords[-1]:
            pos = float(np.interp(ref, coords, np.arange(len(coords))))
            ax.axvline(pos, color="white", linestyle="--" if ri == 0 else ":", linewidth=2.2)
            ax.axhline(pos, color="white", linestyle="--" if ri == 0 else ":", linewidth=2.2)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Raw dissimilarity", fontsize=18)
    cbar.ax.tick_params(labelsize=max(13, TICK_SIZE - 2))
    fig.tight_layout()
    return fig, ax


def plot_synthetic_validation(run_dir: str | Path, *, model: str, display_max_q: int | None = 8) -> None:
    apply_style()
    run = Path(run_dir)
    plots = run / "plots"
    plots.mkdir(exist_ok=True)
    cfg = json.loads((run / "effective_config.json").read_text(encoding="utf-8"))
    grid = cfg["resolved_grid"]
    coords = np.asarray(grid["control_coordinates"], dtype=float)
    validation = cfg["validation"]
    plot_options = dict(validation.get("plotting", {}))
    fixed = dict(validation.get("fixed_params", {}))
    n = int(validation["n"])
    methods = _method_order(list(validation.get("methods", ["Hodge"])))
    refs = model_references(model, n, fixed)

    # ------------------------------------------------------------------
    # 1) Raw distance heatmaps: one independent SQUARE figure per method.
    #    Raw scales remain method-specific and are never normalized for display.
    # ------------------------------------------------------------------
    for method in methods:
        p = run / "arrays" / f"distance_mean_{method}.npy"
        if not p.exists():
            continue
        D = np.load(p)
        side = float(plot_options.get("method_heatmap_square_size", 10.8))
        fig, ax = _single_method_distance_heatmap(
            D, coords, grid["coordinate_label"], refs, model=model, method=method, figure_size=side
        )
        _save(fig, plots / f"{model.lower()}_distance_heatmap_{method}")

    # ------------------------------------------------------------------
    # 2) Raw adjacent-distance curves: one method per ROW.
    #    This replaces the old five-line overlay.
    # ------------------------------------------------------------------
    adj = pd.read_csv(run / "tables" / "adjacent_distance_summary.csv")
    adj_methods = _method_order(list(dict.fromkeys(adj["method"].astype(str))))
    if adj_methods:
        fig, axes = _stacked_method_axes(
            len(adj_methods),
            row_height=float(plot_options.get("method_curve_row_height", 4.0)),
            width=float(plot_options.get("method_panel_width", 15.5)),
            sharex=True,
        )
        for idx, (ax, method) in enumerate(zip(axes, adj_methods)):
            sub = adj[adj.method.astype(str) == method].sort_values("left_index")
            c = METHOD_COLORS.get(method, None)
            line, = ax.plot(
                sub.coordinate_mid,
                sub.distance_mean,
                marker="o",
                linewidth=LINE_WIDTH,
                markersize=MARKER_SIZE,
                label=method,
                color=c,
                zorder=3,
            )
            ax.fill_between(
                sub.coordinate_mid,
                sub.ci95_low,
                sub.ci95_high,
                alpha=0.16,
                color=c,
                zorder=2,
            )
            _reference_lines(ax, refs, model, show_labels=(idx == 0))
            _axis_scale(ax, model)
            ax.set_ylabel("Raw\ndissimilarity")
            ax.set_title(method, fontweight="bold", loc="left", pad=7)
            ax.grid(False)
            if idx == 0:
                _method_panel_legend(ax, method, refs, show_ref_labels=True)
            else:
                ax.legend([line], [method], frameon=True, loc="best", fontsize=max(12, LEGEND_SIZE - 2))
        axes[-1].set_xlabel(grid["coordinate_label"])
        fig.suptitle(
            f"{model}: adjacent-parameter network dissimilarity — one method per row",
            fontsize=SUPTITLE_SIZE,
            fontweight="bold",
            y=0.998,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.992), h_pad=1.6)
        _save(fig, plots / f"{model.lower()}_adjacent_raw")

    # ------------------------------------------------------------------
    # 3) Hodge order-resolved adjacent contribution (single method, many q).
    # ------------------------------------------------------------------
    qpath = run / "tables" / "hodge_order_adjacent.csv"
    if qpath.exists() and qpath.stat().st_size:
        qdf = pd.read_csv(qpath)
        if display_max_q is not None:
            qdf = qdf[qdf.q <= display_max_q]
        if not qdf.empty:
            fig, ax = plt.subplots(figsize=(14, 9))
            for q, sub in qdf.groupby("q"):
                sub = sub.sort_values("left_index")
                ax.plot(sub.coordinate_mid, sub.weighted_q_distance_mean, marker="o", label=f"q={int(q)}")
            _reference_lines(ax, refs, model)
            _axis_scale(ax, model)
            ax.set_xlabel(grid["coordinate_label"])
            ax.set_ylabel("Weighted single-q Hodge contribution")
            ax.set_title(f"{model}: order-resolved Hodge response", fontweight="bold")
            ax.legend(frameon=False, ncol=2)
            _save(fig, plots / f"{model.lower()}_hodge_order_resolved")

    # ------------------------------------------------------------------
    # 4) Maximal-clique abundance by q (structural q-series, not methods).
    # ------------------------------------------------------------------
    cdf = pd.read_csv(run / "tables" / "clique_abundance_summary.csv")
    if display_max_q is not None:
        cdf = cdf[cdf.q <= display_max_q]
    if not cdf.empty:
        fig, ax = plt.subplots(figsize=(14, 9))
        for q, sub in cdf.groupby("q"):
            sub = sub.sort_values("parameter_index")
            ax.plot(sub.coordinate, sub.mean_count, marker="o", label=f"q={int(q)}")
        _reference_lines(ax, refs, model)
        _axis_scale(ax, model)
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_xlabel(grid["coordinate_label"])
        ax.set_ylabel("Mean exclusive maximal-simplex count")
        ax.set_title(f"{model}: exclusive q-layer abundance", fontweight="bold")
        ax.legend(frameon=False, ncol=2)
        _save(fig, plots / f"{model.lower()}_clique_abundance")

    # ------------------------------------------------------------------
    # 5) Signal/noise ratio: one method per ROW (no overlay).
    # ------------------------------------------------------------------
    snr = pd.read_csv(run / "tables" / "signal_noise_ratio.csv")
    snr_methods = _method_order(list(dict.fromkeys(snr["method"].astype(str))))
    if snr_methods:
        fig, axes = _stacked_method_axes(
            len(snr_methods),
            row_height=float(plot_options.get("method_curve_row_height", 4.0)),
            width=float(plot_options.get("method_panel_width", 15.5)),
            sharex=True,
        )
        for idx, (ax, method) in enumerate(zip(axes, snr_methods)):
            sub = snr[snr.method.astype(str) == method].sort_values("left_index")
            line, = ax.plot(
                sub.coordinate_mid,
                sub.signal_noise_ratio,
                marker="o",
                label=method,
                color=METHOD_COLORS.get(method, None),
                linewidth=LINE_WIDTH,
                markersize=MARKER_SIZE,
            )
            _reference_lines(ax, refs, model, show_labels=(idx == 0))
            _axis_scale(ax, model)
            ax.set_ylabel("Signal /\nnoise")
            ax.set_title(method, fontweight="bold", loc="left", pad=7)
            ax.grid(False)
            if idx == 0:
                _method_panel_legend(ax, method, refs, show_ref_labels=True)
            else:
                ax.legend([line], [method], frameon=True, loc="best", fontsize=max(12, LEGEND_SIZE - 2))
        axes[-1].set_xlabel(grid["coordinate_label"])
        fig.suptitle(
            f"{model}: adjacent signal-to-noise ratio — one method per row",
            fontsize=SUPTITLE_SIZE,
            fontweight="bold",
            y=0.998,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.992), h_pad=1.6)
        _save(fig, plots / f"{model.lower()}_signal_noise_ratio")

    # ------------------------------------------------------------------
    # 6) Model observables (not method-comparison figures).
    # ------------------------------------------------------------------
    g = pd.read_csv(run / "tables" / "graph_observables.csv")
    agg = g.groupby("parameter_index", as_index=False).agg(
        coordinate=("coordinate", "first"),
        largest_component_fraction=("largest_component_fraction", "mean"),
        second_component_fraction=("second_component_fraction", "mean"),
        finite_cluster_susceptibility=("finite_cluster_susceptibility", "mean"),
        isolated_fraction=("isolated_fraction", "mean"),
        average_clustering=("average_clustering", "mean"),
        lcc_average_path_length=("lcc_average_path_length", "mean"),
    )
    if model == "ER":
        fig, ax = plt.subplots(figsize=(13.5, 8.5))
        ax.plot(agg.coordinate, agg.largest_component_fraction, marker="o", label="Largest component fraction")
        ax.plot(agg.coordinate, agg.second_component_fraction, marker="s", label="Second component fraction")
        ax.plot(agg.coordinate, agg.isolated_fraction, marker="^", label="Isolated fraction")
        _reference_lines(ax, refs, model); _axis_scale(ax, model)
        ax.set_xlabel(grid["coordinate_label"]); ax.set_ylabel("Fraction")
        ax.set_title("ER structural observables", fontweight="bold"); ax.legend(frameon=False)
        _save(fig, plots / "er_structural_observables")
    elif model == "WS":
        fig, ax = plt.subplots(figsize=(13.5, 8.5))
        first_apl = float(agg.lcc_average_path_length.iloc[0])
        first_c = float(agg.average_clustering.iloc[0])
        ax.plot(agg.coordinate, agg.lcc_average_path_length / max(first_apl, 1e-15), marker="o", label="L/L(0)")
        ax.plot(agg.coordinate, agg.average_clustering / max(first_c, 1e-15), marker="s", label="C/C(0)")
        _reference_lines(ax, refs, model); _axis_scale(ax, model)
        ax.set_xlabel(grid["coordinate_label"]); ax.set_ylabel("Normalized observable")
        ax.set_title("WS path length and clustering", fontweight="bold"); ax.legend(frameon=False)
        _save(fig, plots / "ws_structural_observables")
    elif model == "RGG":
        fig, ax = plt.subplots(figsize=(13.5, 8.5))
        ax.plot(agg.coordinate, agg.largest_component_fraction, marker="o", label="Largest component fraction")
        ax.plot(agg.coordinate, agg.isolated_fraction, marker="^", label="Isolated fraction")
        ax.plot(agg.coordinate, agg.average_clustering, marker="s", label="Average clustering")
        _reference_lines(ax, refs, model); _axis_scale(ax, model)
        ax.set_xlabel(grid["coordinate_label"]); ax.set_ylabel("Fraction / clustering")
        ax.set_title("RGG structural observables", fontweight="bold"); ax.legend(frameon=False)
        _save(fig, plots / "rgg_structural_observables")


def _cluster_order_from_distance(D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    D = np.asarray(D, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("distance matrix must be square")
    if not np.all(np.isfinite(D)):
        raise ValueError("distance matrix contains non-finite values")
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method="average")
    if len(D) > 2:
        Z = optimal_leaf_ordering(Z, condensed)
    order = leaves_list(Z).astype(int)
    return Z, order


def _plot_clustered_cell_heatmap(
    D: np.ndarray,
    labels: list[str],
    method: str,
    base: Path,
) -> np.ndarray:
    """Square cell heatmap with average-linkage + optimal-leaf dendrogram."""
    D = np.asarray(D, dtype=float)
    Z, order = _cluster_order_from_distance(D)
    ordered = D[np.ix_(order, order)]
    ordered_labels = [labels[i] for i in order]

    fig = plt.figure(figsize=(15.0, 17.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[3.3, 12.0], width_ratios=[12.0, 0.55], hspace=0.015, wspace=0.12)
    ax_den = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0])
    cax = fig.add_subplot(gs[1, 1])
    d = dendrogram(Z, ax=ax_den, no_labels=True, color_threshold=None, above_threshold_color="0.15", link_color_func=lambda _k: "0.15")
    den_order = np.asarray(d["leaves"], dtype=int)
    if not np.array_equal(den_order, order):
        raise RuntimeError("dendrogram leaf order does not match heatmap order")
    ax_den.set_xticks([])
    ax_den.set_yticks([])
    ax_den.set_xlim(0, 10 * len(order))
    for sp in ax_den.spines.values():
        sp.set_visible(False)

    im = ax.imshow(ordered, origin="upper", cmap=DMEASURE_CMAP, aspect="equal", vmin=0)
    positions = np.arange(len(order))
    ax.set_xticks(positions, ordered_labels, rotation=90, fontsize=max(8, TICK_SIZE - 7))
    ax.set_yticks(positions, ordered_labels, fontsize=max(8, TICK_SIZE - 7))
    ax.set_xlabel("Parameter cell")
    ax.set_ylabel("Parameter cell")
    fig.suptitle(f"{method}: matched-density mechanism dissimilarity", fontweight="bold", fontsize=TITLE_SIZE + 2, y=0.995)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Raw dissimilarity", fontsize=18)
    cax.tick_params(labelsize=max(11, TICK_SIZE - 4))
    _save_mechanism(fig, base)
    return order



_MDS_KAPPA_MARKERS = ("o", "s", "^", "D", "P", "X", "v", "<", ">")


def _mds_color_map(states: list[str]) -> dict[str, object]:
    cmap = plt.get_cmap("tab10")
    return {state: cmap(i % 10) for i, state in enumerate(states)}


def _mds_marker_map(kappas: list[float]) -> dict[float, str]:
    vals = sorted(float(x) for x in kappas)
    if len(vals) > len(_MDS_KAPPA_MARKERS):
        raise ValueError("too many kappa levels for the fixed MDS marker vocabulary")
    return {k: _MDS_KAPPA_MARKERS[i] for i, k in enumerate(vals)}


def _mds_diagnostic_caption(diagnostics: pd.DataFrame, method: str, level: str) -> str:
    if diagnostics.empty or not {"method", "level"}.issubset(diagnostics.columns):
        return ""
    row = diagnostics[(diagnostics.method.astype(str) == str(method)) & (diagnostics.level.astype(str) == str(level))]
    if row.empty:
        return ""
    r = row.iloc[0]
    explained = float(r["positive_eigenvalue_fraction_2d"])
    rho = float(r["distance_spearman_2d"])
    stress = float(r["normalized_stress_2d"])
    return f"2-D positive-spectrum fraction={explained:.2f}   distance Spearman={rho:.2f}   stress={stress:.2f}"


def _mds_legends(ax, states: list[str], kappas: list[float], colors: dict[str, object], markers: dict[float, str]) -> None:
    mechanism_handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=10, markerfacecolor=colors[s],
               markeredgecolor="white", markeredgewidth=0.8, label=s)
        for s in states
    ]
    kappa_handles = [
        Line2D([0], [0], marker=markers[float(k)], linestyle="", markersize=9,
               markerfacecolor="0.72", markeredgecolor="0.18", markeredgewidth=0.8,
               label=f"κ={float(k):g}")
        for k in kappas
    ]
    first = ax.legend(
        handles=mechanism_handles,
        title="Mechanism",
        frameon=True,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.00),
        fontsize=max(11, LEGEND_SIZE - 3),
        title_fontsize=max(12, LEGEND_SIZE - 2),
    )
    ax.add_artist(first)
    ax.legend(
        handles=kappa_handles,
        title="Target mean degree",
        frameon=True,
        loc="lower left",
        bbox_to_anchor=(1.02, 0.00),
        fontsize=max(11, LEGEND_SIZE - 3),
        title_fontsize=max(12, LEGEND_SIZE - 2),
    )


def _plot_graph_mds(df: pd.DataFrame, diagnostics: pd.DataFrame, method: str, base: Path) -> None:
    states = df["mechanism_state"].astype(str).drop_duplicates().tolist()
    kappas = sorted(df["target_mean_degree"].astype(float).unique().tolist())
    colors = _mds_color_map(states)
    markers = _mds_marker_map(kappas)

    fig, ax = plt.subplots(figsize=(15.5, 10.5))
    for state in states:
        state_df = df[df.mechanism_state.astype(str) == state]
        for kappa in kappas:
            sub = state_df[np.isclose(state_df.target_mean_degree.astype(float), float(kappa))]
            if sub.empty:
                continue
            ax.scatter(
                sub.mds_x,
                sub.mds_y,
                s=105,
                marker=markers[float(kappa)],
                c=[colors[state]],
                edgecolors="white",
                linewidths=0.9,
                alpha=0.78,
                zorder=3,
            )

    caption = _mds_diagnostic_caption(diagnostics, method, "graph")
    ax.set_xlabel("Classical MDS dimension 1")
    ax.set_ylabel("Classical MDS dimension 2")
    ax.set_title(f"{method}: graph-level ER–WS–RGG 2-D MDS\n{caption}", fontweight="bold", pad=12)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(False)
    _mds_legends(ax, states, kappas, colors, markers)
    fig.tight_layout(rect=(0, 0, 0.80, 1))
    _save_mechanism(fig, base)


def _plot_cell_mds(df: pd.DataFrame, diagnostics: pd.DataFrame, method: str, base: Path) -> None:
    states = df["mechanism_state"].astype(str).drop_duplicates().tolist()
    kappas = sorted(df["target_mean_degree"].astype(float).unique().tolist())
    colors = _mds_color_map(states)
    markers = _mds_marker_map(kappas)

    fig, ax = plt.subplots(figsize=(15.5, 10.5))
    for state in states:
        state_df = df[df.mechanism_state.astype(str) == state].sort_values("target_mean_degree")
        # Connect parameter cells in increasing kappa only as a visual guide;
        # these are independent ensembles, not a coupled graph trajectory.
        ax.plot(
            state_df.mds_x,
            state_df.mds_y,
            color=colors[state],
            linewidth=1.8,
            alpha=0.45,
            zorder=1,
        )
        for row in state_df.itertuples(index=False):
            kappa = float(row.target_mean_degree)
            ax.scatter(
                [float(row.mds_x)],
                [float(row.mds_y)],
                s=170,
                marker=markers[kappa],
                c=[colors[state]],
                edgecolors="white",
                linewidths=1.1,
                alpha=0.95,
                zorder=3,
            )

    caption = _mds_diagnostic_caption(diagnostics, method, "cell")
    ax.set_xlabel("Classical MDS dimension 1")
    ax.set_ylabel("Classical MDS dimension 2")
    ax.set_title(f"{method}: 36-cell ER–WS–RGG 2-D MDS\n{caption}", fontweight="bold", pad=12)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(False)
    _mds_legends(ax, states, kappas, colors, markers)
    fig.tight_layout(rect=(0, 0, 0.80, 1))
    _save_mechanism(fig, base)


def plot_model_discrimination(run_dir: str | Path) -> None:
    """Plots for the ER/WS/RGG matched-density discrimination experiment.

    This lives in the authoritative synthetic plotting module intentionally;
    no replot-only code path is introduced.
    """
    apply_style()
    run = Path(run_dir)
    plots = run / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(run / "tables" / "parameter_cells.csv").sort_values("cell_index")
    labels = cells["cell_label"].astype(str).tolist()
    methods = _method_order(pd.read_csv(run / "tables" / "metrics_all_methods.csv")["method"].astype(str).unique().tolist())

    # One square clustered heatmap per method.
    for method in methods:
        path = run / "arrays" / f"cell_distance_{method}.npy"
        if not path.exists():
            continue
        D = np.load(path)
        order = _plot_clustered_cell_heatmap(D, labels, method, plots / f"mechanism_heatmap_{method}")
        pd.DataFrame({"order": np.arange(len(order)), "cell_index": order, "cell_label": [labels[i] for i in order]}).to_csv(
            run / "tables" / f"mechanism_heatmap_order_{method}.csv", index=False
        )

    # Deterministic classical-MDS views of the same raw dissimilarities.
    # Graph-level points expose realization overlap/noise; cell-level points
    # show the ensemble-mean mechanism/density geometry. These figures are
    # explanatory only and do not enter any reported classification metric.
    diag_path = run / "tables" / "mds_diagnostics.csv"
    diagnostics = pd.read_csv(diag_path) if diag_path.exists() else pd.DataFrame()
    for method in methods:
        graph_mds = run / "tables" / f"mds_graph_{method}.csv"
        cell_mds = run / "tables" / f"mds_cell_{method}.csv"
        if graph_mds.exists():
            _plot_graph_mds(pd.read_csv(graph_mds), diagnostics, method, plots / f"mds_graph_{method}")
        if cell_mds.exists():
            _plot_cell_mds(pd.read_csv(cell_mds), diagnostics, method, plots / f"mds_cell_{method}")

    # Classification metric panels: methods are rows, metrics are columns.
    metrics = pd.read_csv(run / "tables" / "metrics_all_methods.csv")
    core_names = ["balanced_accuracy", "macro_f1", "same_class_auroc", "mean_average_precision", "ndcg_at_5"]
    for protocol in ["matched_density", "cross_density"]:
        sub = metrics[(metrics["protocol"] == protocol) & (metrics["label_level"] == "fine")].copy()
        if sub.empty:
            continue
        fig, axes = _stacked_method_axes(len(methods), row_height=3.6, width=15.5, sharex=True)
        x = np.arange(len(core_names))
        for ax, method in zip(axes, methods):
            row = sub[sub.method == method]
            if row.empty:
                ax.axis("off")
                continue
            vals = [float(row.iloc[0][m]) for m in core_names]
            bars = ax.bar(x, vals, color=METHOD_COLORS.get(method, None))
            ax.set_ylim(0, 1.05)
            ax.set_ylabel(method)
            ax.grid(False)
            _annotate_bars(ax, bars, vals, fmt="{:.3f}", fontsize=max(10, TICK_SIZE - 5))
        axes[-1].set_xticks(x, ["Bal. Acc.", "Macro-F1", "AUROC", "mAP", "NDCG@5"], rotation=20, ha="right")
        fig.suptitle(f"ER–WS–RGG mechanism discrimination: {protocol.replace('_', ' ')}", fontweight="bold", fontsize=SUPTITLE_SIZE)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        # Core metric bars intentionally overwrite any legacy copy so a
        # completed run can be re-plotted from saved tables and immediately
        # pick up the new numeric annotations without recomputing distances.
        _save_mechanism(fig, plots / f"core_metrics_{protocol}", overwrite=True)

    # Fine-class confusion matrices: one compact comparison figure per protocol.
    for protocol in ["matched_density", "cross_density"]:
        fig, axes = plt.subplots(1, len(methods), figsize=(5.4 * len(methods), 5.8), squeeze=False)
        axes = axes[0]
        max_count = 0.0
        frames = {}
        for method in methods:
            p = run / "tables" / f"confusion_{protocol}_{method}.csv"
            if p.exists():
                frames[method] = pd.read_csv(p, index_col=0)
                max_count = max(max_count, float(frames[method].to_numpy(dtype=float).max()))
        image = None
        for ax, method in zip(axes, methods):
            if method not in frames:
                ax.axis("off")
                continue
            df = frames[method]
            labs = df.index.astype(str).tolist()
            image = ax.imshow(df.to_numpy(dtype=float), cmap=DMEASURE_CMAP, aspect="equal", vmin=0, vmax=max_count or None)
            ax.set_xticks(np.arange(len(labs)), labs, rotation=60, ha="right", fontsize=max(8, TICK_SIZE - 7))
            ax.set_yticks(np.arange(len(labs)), labs, fontsize=max(8, TICK_SIZE - 7))
            ax.set_title(method, fontweight="bold")
            ax.set_xlabel("Predicted")
        if len(axes):
            axes[0].set_ylabel("True mechanism")
        if image is not None:
            fig.colorbar(image, ax=axes.tolist(), fraction=0.018, pad=0.015, label="Query count")
        fig.suptitle(f"ER–WS–RGG: {protocol.replace('_',' ')} 1-NN confusion", fontweight="bold", fontsize=SUPTITLE_SIZE, y=1.02)
        fig.subplots_adjust(left=0.04, right=0.95, bottom=0.20, top=0.86, wspace=0.35)
        _save_mechanism(fig, plots / f"confusion_{protocol}_all_methods")

    # Structural audit: mean degree matching and clustering differences.
    audit = pd.read_csv(run / "tables" / "matched_structural_audit.csv")
    for metric, ylabel in [("realized_mean_degree_mean", "Realized mean degree"), ("average_clustering_mean", "Average clustering")]:
        fig, ax = plt.subplots(figsize=(14.5, 9.0))
        for state, sub in audit.groupby("mechanism_state", sort=False):
            sub = sub.sort_values("target_mean_degree")
            ax.plot(sub.target_mean_degree, sub[metric], marker="o", linewidth=LINE_WIDTH, markersize=MARKER_SIZE, label=state)
        if metric == "realized_mean_degree_mean":
            lo = float(audit.target_mean_degree.min()); hi = float(audit.target_mean_degree.max())
            ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=2.2, color="0.25", label="target")
        ax.set_xlabel("Target mean degree κ")
        ax.set_ylabel(ylabel)
        ax.set_title(f"ER–WS–RGG structural audit: {ylabel}", fontweight="bold")
        ax.legend(frameon=False, ncol=2)
        fig.tight_layout()
        _save_mechanism(fig, plots / f"structural_{metric}")
