from __future__ import annotations

"""Post-processing and plotting for real-network Hodge zero-mode runs.

This module consumes an existing ``real_zero_modes_*`` run directory.  It does
not recompute cliques, Hodge Laplacians, spectra, heat signatures, or distances.
Its purpose is to turn the diagnostic tables/arrays into:

* exact/resolved zero-mode count tables by network, exclusive q-layer and
  Hodge order k;
* zero/positive spectral-mass tables;
* per-network q-by-k heatmaps; and
* one spectral dashboard per (network, q), with positive-spectrum panels and
  an exact zero-vs-positive mass bar for every available k.

Important interpretation rule
-----------------------------
``zero_count`` is used only when ``count_resolved`` is true.  For k=0 this is
an exact combinatorial connected-component count.  For k>0 it is a numerical
nullity resolved under the zero tolerance recorded in ``effective_config``.
A ``smallest_prefix`` spectrum is never presented as a complete positive
spectrum; its panel is explicitly labelled as a prefix.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hodge_vnext.utils import safe_token
from .style import DMEASURE_CMAP, FIG_DPI, LABEL_SIZE, TICK_SIZE, TITLE_SIZE, apply_style


REPORT_SCHEMA = "hodge-real-zero-mode-report-v1"


@dataclass(frozen=True)
class ReportResult:
    run_dir: Path
    tables_dir: Path
    plots_dir: Path
    networks: int
    channels: int
    resolved_channels: int


def _savefig(fig: plt.Figure, stem: Path, *, pdf: bool = True) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    if pdf:
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _require_columns(df: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _load_run(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    run_dir = Path(run_dir).resolve()
    tables = run_dir / "tables"
    channels_path = tables / "real_zero_modes_per_network_channel.csv"
    graph_path = tables / "real_graph_index.csv"
    spectra_path = tables / "real_eigenvalue_index.csv"
    analytic_path = tables / "real_analytic_spectra_compact.csv"
    if not channels_path.exists():
        raise FileNotFoundError(channels_path)
    if not graph_path.exists():
        raise FileNotFoundError(graph_path)
    if not spectra_path.exists():
        raise FileNotFoundError(spectra_path)

    channels = pd.read_csv(channels_path)
    graphs = pd.read_csv(graph_path)
    spectra = pd.read_csv(spectra_path)
    analytic = pd.read_csv(analytic_path) if analytic_path.exists() else pd.DataFrame()
    _require_columns(
        channels,
        ["label", "domain", "q", "k", "dimension", "zero_count", "count_resolved", "zero_fraction", "spectrum_scope"],
        channels_path.name,
    )
    _require_columns(
        spectra,
        ["label", "domain", "q", "k", "dimension", "spectrum_scope", "spectrum_file", "full_spectrum_available"],
        spectra_path.name,
    )

    tol = 1e-10
    effective = run_dir / "effective_config.json"
    if effective.exists():
        payload = json.loads(effective.read_text(encoding="utf-8"))
        tol = float(
            payload.get("resolved_zero_mode_settings", {}).get(
                "zero_tolerance",
                payload.get("method", {}).get("exact", {}).get("eigen_zero_tolerance", tol),
            )
        )
    return channels, graphs, spectra, analytic, tol


def _channel_name(q: int, k: int) -> str:
    return f"q{int(q)}_k{int(k)}"


def _sorted_channel_columns(columns: Iterable[str]) -> list[str]:
    vals = []
    for c in columns:
        if not (c.startswith("q") and "_k" in c):
            continue
        qtxt, ktxt = c.split("_k", 1)
        try:
            vals.append((int(qtxt[1:]), int(ktxt), c))
        except ValueError:
            pass
    return [c for _, _, c in sorted(vals)]


def _write_exact_tables(channels: pd.DataFrame, out: Path) -> None:
    resolved = channels[channels["count_resolved"].astype(bool)].copy()
    unresolved = channels[~channels["count_resolved"].astype(bool)].copy()
    if not unresolved.empty:
        unresolved.to_csv(out / "real_zero_count_unresolved_channels.csv", index=False, encoding="utf-8-sig")

    cols = [
        "graph_index", "label", "domain", "N", "E", "q", "k", "dimension",
        "zero_count", "zero_fraction", "smallest_positive_eigenvalue", "method", "status",
        "spectrum_scope", "spectrum_method",
    ]
    cols = [c for c in cols if c in resolved.columns]
    long = resolved[cols].sort_values(["domain", "label", "q", "k"])
    long.to_csv(out / "real_zero_count_exact_long.csv", index=False, encoding="utf-8-sig")

    # One row per network/q with k0...k9 columns.  This is the most direct
    # representation of "for this network and this exclusive q-layer, how many
    # zero eigenvalues does each L_k^(q) have?".
    qk = resolved.pivot_table(
        index=["label", "domain", "N", "E", "q"], columns="k", values="zero_count", aggfunc="first"
    ).reset_index()
    qk.columns = [f"k{int(c)}" if isinstance(c, (int, np.integer)) else str(c) for c in qk.columns]
    kcols = sorted([c for c in qk.columns if c.startswith("k") and c[1:].isdigit()], key=lambda x: int(x[1:]))
    qk = qk[["label", "domain", "N", "E", "q"] + kcols]
    qk.to_csv(out / "real_zero_count_exact_by_network_q.csv", index=False, encoding="utf-8-sig")

    # Wide network x (q,k) table.
    work = resolved.copy()
    work["channel"] = [_channel_name(q, k) for q, k in zip(work["q"], work["k"])]
    meta = ["label", "domain", "N", "E"]
    wide = work.pivot_table(index=meta, columns="channel", values="zero_count", aggfunc="first").reset_index()
    ccols = _sorted_channel_columns([str(c) for c in wide.columns])
    wide = wide[meta + ccols]
    wide.to_csv(out / "real_zero_count_exact_wide.csv", index=False, encoding="utf-8-sig")

    frac = work.pivot_table(index=meta, columns="channel", values="zero_fraction", aggfunc="first").reset_index()
    ccols = _sorted_channel_columns([str(c) for c in frac.columns])
    frac = frac[meta + ccols]
    frac.to_csv(out / "real_zero_fraction_exact_wide.csv", index=False, encoding="utf-8-sig")

    # Compact network-level index, useful for PPT/appendix navigation.
    summary_rows: list[dict[str, object]] = []
    for (label, domain, N, E), g in resolved.groupby(["label", "domain", "N", "E"], sort=False):
        positive = g[g["zero_count"].astype(float) > 0].copy()
        summary_rows.append({
            "label": label,
            "domain": domain,
            "N": int(N),
            "E": int(E),
            "retained_q_max": int(g["q"].max()),
            "channels_total": int(len(g)),
            "channels_with_zero_modes": int(len(positive)),
            "highest_q_with_zero_modes": (int(positive["q"].max()) if not positive.empty else np.nan),
            "highest_k_with_zero_modes": (int(positive["k"].max()) if not positive.empty else np.nan),
            "max_zero_count": int(g["zero_count"].max()),
            "max_zero_fraction": float(g["zero_fraction"].max()),
        })
    pd.DataFrame(summary_rows).sort_values(["domain", "label"]).to_csv(
        out / "real_zero_mode_network_summary.csv", index=False, encoding="utf-8-sig"
    )


def _load_channel_spectrum(
    run_dir: Path,
    spectrum_row: pd.Series,
    analytic: pd.DataFrame,
) -> tuple[np.ndarray, str]:
    scope = str(spectrum_row["spectrum_scope"])
    label = str(spectrum_row["label"])
    q = int(spectrum_row["q"])
    k = int(spectrum_row["k"])
    if scope == "analytic_compact":
        if analytic.empty:
            return np.empty(0, dtype=float), scope
        a = analytic[(analytic["label"].astype(str) == label) & (analytic["q"] == q) & (analytic["k"] == k)]
        vals: list[np.ndarray] = []
        for r in a.itertuples(index=False):
            vals.append(np.full(int(r.multiplicity), float(r.eigenvalue), dtype=float))
        return (np.concatenate(vals) if vals else np.empty(0, dtype=float)), scope
    rel = str(spectrum_row.get("spectrum_file", ""))
    if not rel:
        return np.empty(0, dtype=float), scope
    # Run tables may have been produced on Windows and therefore contain
    # backslash separators even when post-processing on another platform.
    rel = rel.replace("\\", "/")
    p = run_dir / Path(rel)
    if not p.exists():
        raise FileNotFoundError(p)
    return np.asarray(np.load(p), dtype=float), scope


def _positive_spectrum_plot(ax: plt.Axes, values: np.ndarray, zero_tol: float, scope: str) -> None:
    positive = np.asarray(values, dtype=float)
    positive = positive[np.isfinite(positive) & (positive > zero_tol)]
    if positive.size == 0:
        ax.text(0.5, 0.5, "No observed positive eigenvalues", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    xmin = float(np.min(positive)); xmax = float(np.max(positive))
    if positive.size == 1 or math.isclose(xmin, xmax, rel_tol=1e-12, abs_tol=1e-15):
        ax.axvline(xmin, linewidth=2.4)
        ax.set_xlim(max(zero_tol, xmin * 0.8), xmin * 1.2 if xmin > 0 else 1.0)
    else:
        nbins = int(min(42, max(10, round(math.sqrt(positive.size)))))
        span = xmax / xmin if xmin > 0 else 1.0
        if xmin > 0 and span >= 50:
            bins = np.logspace(np.log10(xmin), np.log10(xmax), nbins + 1)
            ax.hist(positive, bins=bins, weights=np.ones_like(positive) / positive.size, alpha=0.86)
            ax.set_xscale("log")
        else:
            ax.hist(positive, bins=nbins, weights=np.ones_like(positive) / positive.size, alpha=0.86)
    ax.set_ylabel("Positive spectral mass", fontsize=max(9, TICK_SIZE - 3))
    ax.tick_params(labelsize=max(8, TICK_SIZE - 4))
    ax.grid(alpha=0.15)
    if scope == "smallest_prefix":
        ax.text(
            0.02, 0.95, "smallest-prefix only", transform=ax.transAxes,
            va="top", ha="left", fontsize=max(8, TICK_SIZE - 4),
        )


def _zero_positive_mass_bar(ax: plt.Axes, zero_count: int, dimension: int) -> None:
    if dimension <= 0:
        ax.axis("off")
        return
    zf = float(zero_count) / float(dimension)
    pf = 1.0 - zf
    ax.barh([0], [zf], height=0.56, label="zero")
    ax.barh([0], [pf], left=[zf], height=0.56, label=">0")
    ax.set_xlim(0.0, 1.0)
    ax.set_yticks([])
    ax.set_xlabel("Spectral mass fraction", fontsize=max(9, TICK_SIZE - 3))
    ax.tick_params(axis="x", labelsize=max(8, TICK_SIZE - 4))
    ax.text(0.01, 0.92, f"zero={zero_count}/{dimension} = {zf:.4g}", transform=ax.transAxes,
            ha="left", va="top", fontsize=max(8, TICK_SIZE - 4))


def _plot_network_q_dashboard(
    run_dir: Path,
    label: str,
    domain: str,
    q: int,
    channels: pd.DataFrame,
    spectra: pd.DataFrame,
    analytic: pd.DataFrame,
    zero_tol: float,
    out_stem: Path,
    *,
    save_pdf: bool,
) -> None:
    sub = channels[(channels["label"].astype(str) == label) & (channels["q"] == q)].sort_values("k")
    if sub.empty:
        return
    m = len(sub)
    ncols = min(4, m)
    blocks = int(math.ceil(m / ncols))
    apply_style()
    fig = plt.figure(figsize=(5.7 * ncols, 5.8 * blocks + 1.3))
    gs = fig.add_gridspec(2 * blocks, ncols, height_ratios=sum(([3.2, 1.0] for _ in range(blocks)), []), hspace=0.55, wspace=0.32)

    for idx, (_, row) in enumerate(sub.iterrows()):
        block = idx // ncols
        col = idx % ncols
        ax_spec = fig.add_subplot(gs[2 * block, col])
        ax_mass = fig.add_subplot(gs[2 * block + 1, col])
        k = int(row["k"])
        sr = spectra[(spectra["label"].astype(str) == label) & (spectra["q"] == q) & (spectra["k"] == k)]
        if len(sr) != 1:
            raise RuntimeError(f"Expected one spectrum-index row for {label} q={q} k={k}, got {len(sr)}")
        vals, scope = _load_channel_spectrum(run_dir, sr.iloc[0], analytic)
        _positive_spectrum_plot(ax_spec, vals, zero_tol, scope)
        ax_spec.set_title(rf"$L_{{{k}}}^{{({q})}}$  |  {scope}", fontsize=max(11, TITLE_SIZE - 7), fontweight="bold")
        ax_spec.set_xlabel(r"Positive Hodge eigenvalue $\lambda$", fontsize=max(9, TICK_SIZE - 3))
        zc = int(row["zero_count"])
        dim = int(row["dimension"])
        _zero_positive_mass_bar(ax_mass, zc, dim)
        gap = row.get("smallest_positive_eigenvalue", np.nan)
        gap_txt = "NA" if pd.isna(gap) else f"{float(gap):.4g}"
        ax_mass.text(0.99, 0.92, rf"$\lambda_+$={gap_txt}", transform=ax_mass.transAxes,
                     ha="right", va="top", fontsize=max(8, TICK_SIZE - 4))

    # Hide empty cells.
    for idx in range(m, blocks * ncols):
        block = idx // ncols; col = idx % ncols
        a = fig.add_subplot(gs[2 * block, col]); a.axis("off")
        b = fig.add_subplot(gs[2 * block + 1, col]); b.axis("off")

    fig.suptitle(
        f"{label} ({domain}) — exclusive q={q} Hodge spectra and zero-mode mass",
        fontsize=TITLE_SIZE, fontweight="bold", y=0.995,
    )
    _savefig(fig, out_stem, pdf=save_pdf)


def _plot_network_qk_heatmap(
    channels: pd.DataFrame,
    label: str,
    domain: str,
    value: str,
    title: str,
    cbar_label: str,
    out_stem: Path,
    *,
    annotate: bool,
    save_pdf: bool,
) -> None:
    sub = channels[(channels["label"].astype(str) == label) & channels["count_resolved"].astype(bool)].copy()
    if sub.empty:
        return
    qvals = sorted(int(x) for x in sub["q"].unique())
    kmax = int(sub["k"].max())
    M = np.full((len(qvals), kmax + 1), np.nan, dtype=float)
    qi = {q: i for i, q in enumerate(qvals)}
    for r in sub.itertuples(index=False):
        M[qi[int(r.q)], int(r.k)] = float(getattr(r, value))
    apply_style()
    fig, ax = plt.subplots(figsize=(max(12.5, 1.05 * (kmax + 1) + 6.0), max(8.0, 0.88 * len(qvals) + 4.0)))
    im = ax.imshow(np.ma.masked_invalid(M), origin="lower", aspect="auto", interpolation="nearest", cmap=DMEASURE_CMAP)
    ax.set_xticks(np.arange(kmax + 1), [str(k) for k in range(kmax + 1)])
    ax.set_yticks(np.arange(len(qvals)), [str(q) for q in qvals])
    ax.set_xlabel("Hodge order k")
    ax.set_ylabel("Exclusive q-layer")
    ax.set_title(f"{label} ({domain}) — {title}", fontsize=TITLE_SIZE, fontweight="bold")
    if annotate:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if not np.isfinite(M[i, j]):
                    continue
                text = f"{int(round(M[i,j]))}" if value == "zero_count" else f"{M[i,j]:.3g}"
                ax.text(j, i, text, ha="center", va="center", fontsize=max(8, TICK_SIZE - 5))
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.set_label(cbar_label, fontsize=LABEL_SIZE)
    cb.ax.tick_params(labelsize=TICK_SIZE)
    fig.tight_layout()
    _savefig(fig, out_stem, pdf=save_pdf)


def generate_real_zero_mode_report(
    run_dir: str | Path,
    *,
    labels: list[str] | None = None,
    max_q: int | None = None,
    spectral_dashboards: bool = True,
    network_heatmaps: bool = True,
    save_pdf: bool = False,
) -> ReportResult:
    run_dir = Path(run_dir).resolve()
    channels, graphs, spectra, analytic, zero_tol = _load_run(run_dir)
    tables_out = run_dir / "tables" / "zero_mode_report"
    plots_out = run_dir / "plots" / "zero_mode_report"
    tables_out.mkdir(parents=True, exist_ok=True)
    plots_out.mkdir(parents=True, exist_ok=True)

    if labels:
        wanted = set(labels)
        channels = channels[channels["label"].astype(str).isin(wanted)].copy()
        graphs = graphs[graphs["label"].astype(str).isin(wanted)].copy()
        spectra = spectra[spectra["label"].astype(str).isin(wanted)].copy()
        if not analytic.empty:
            analytic = analytic[analytic["label"].astype(str).isin(wanted)].copy()
    if max_q is not None:
        channels = channels[channels["q"] <= int(max_q)].copy()
        spectra = spectra[spectra["q"] <= int(max_q)].copy()
        if not analytic.empty:
            analytic = analytic[analytic["q"] <= int(max_q)].copy()

    _write_exact_tables(channels, tables_out)

    # Audit table: every requested count must be resolved before the report calls
    # it exact/resolved.
    audit = channels.groupby(["q", "k"], as_index=False).agg(
        rows=("label", "count"),
        resolved=("count_resolved", lambda s: int(pd.Series(s).astype(bool).sum())),
    )
    audit["resolution_rate"] = audit["resolved"] / audit["rows"]
    audit.to_csv(tables_out / "real_zero_count_resolution_audit.csv", index=False, encoding="utf-8-sig")

    if network_heatmaps or spectral_dashboards:
        for r in graphs.sort_values(["domain", "label"]).itertuples(index=False):
            label = str(r.label); domain = str(r.domain)
            if label not in set(channels["label"].astype(str)):
                continue
            token = safe_token(label)
            net_root = plots_out / token
            if network_heatmaps:
                _plot_network_qk_heatmap(
                    channels, label, domain, "zero_count", "resolved zero-eigenvalue count by (q,k)",
                    "Zero-eigenvalue count", net_root / f"{token}_zero_count_qk", annotate=True, save_pdf=save_pdf,
                )
                _plot_network_qk_heatmap(
                    channels, label, domain, "zero_fraction", "zero-mode fraction by (q,k)",
                    "Zero-mode fraction", net_root / f"{token}_zero_fraction_qk", annotate=True, save_pdf=save_pdf,
                )
            if spectral_dashboards:
                for q in sorted(int(x) for x in channels[channels["label"].astype(str) == label]["q"].unique()):
                    _plot_network_q_dashboard(
                        run_dir, label, domain, q, channels, spectra, analytic, zero_tol,
                        net_root / f"{token}_q{q:02d}_spectral_dashboard", save_pdf=save_pdf,
                    )

    manifest = {
        "schema": REPORT_SCHEMA,
        "source_run": str(run_dir),
        "zero_tolerance": zero_tol,
        "networks": int(channels["label"].nunique()),
        "channels": int(len(channels)),
        "resolved_channels": int(channels["count_resolved"].astype(bool).sum()),
        "all_requested_counts_resolved": bool(channels["count_resolved"].astype(bool).all()),
        "scientific_note": (
            "k=0 zero_count is exact combinatorial nullity. k>0 zero_count is a numerical nullity resolved under "
            "the recorded zero tolerance. Positive-spectrum panels marked smallest-prefix are not full-spectrum distributions."
        ),
    }
    (run_dir / "zero_mode_report_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return ReportResult(
        run_dir=run_dir,
        tables_dir=tables_out,
        plots_dir=plots_out,
        networks=int(channels["label"].nunique()),
        channels=int(len(channels)),
        resolved_channels=int(channels["count_resolved"].astype(bool).sum()),
    )
