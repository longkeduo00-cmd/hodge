from __future__ import annotations

"""Plotting for synthetic-network zero-mode diagnostics.

The primary report mirrors the real-network zero-mode spectral report used in
this project.  For a fixed Hodge order k (k=0 by default), each column shows
one exclusive q-layer matrix L_k^(q):

  * top: positive-eigenvalue histogram, where bin heights are fractions of the
    *full matrix dimension* ("positive spectral mass"), not a probability
    density over positive eigenvalues;
  * bottom: stacked zero / positive spectral-mass fractions;
  * annotation: zero multiplicity, zero fraction, and the smallest positive
    eigenvalue lambda_+.

For N=200 synthetic graphs, every k=0 matrix has dimension <= N and the formal
configuration's dense guard (1024) therefore provides the full L_0^(q)
spectrum.  Higher-k reports are allowed only when the diagnostic actually
stored a full spectrum; unresolved sparse prefixes are never presented as full
spectral histograms.
"""

from pathlib import Path
from typing import Iterable, Sequence
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


ZERO_COLOR = '#1f77b4'
POSITIVE_COLOR = '#ff7f0e'
GRID_ALPHA = 0.20


def _save(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix('.png'), dpi=220, bbox_inches='tight')
    fig.savefig(stem.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def _triangular_heatmap(summary: pd.DataFrame, model: str, value: str, title: str, cbar_label: str, stem: Path) -> None:
    sub = summary[summary['model'] == model]
    if sub.empty:
        return
    max_q = int(sub['q'].max())
    M = np.full((max_q + 1, max_q + 1), np.nan, dtype=float)
    for r in sub.itertuples(index=False):
        M[int(r.k), int(r.q)] = float(getattr(r, value))
    fig, ax = plt.subplots(figsize=(11.5, 8.5))
    im = ax.imshow(M, origin='lower', aspect='equal', interpolation='nearest')
    ax.set_xlabel('exclusive q-layer', fontsize=17)
    ax.set_ylabel('Hodge order k', fontsize=17)
    ax.set_title(f'{model}: {title}', fontsize=20)
    ax.set_xticks(range(max_q + 1)); ax.set_yticks(range(max_q + 1))
    ax.tick_params(labelsize=13)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=14)
    _save(fig, stem)


def plot_model_heatmaps(summary: pd.DataFrame, plots_dir: Path) -> None:
    for model in summary['model'].drop_duplicates().tolist():
        _triangular_heatmap(summary, model, 'mean_zero_fraction', 'mean zero-eigenvalue fraction', 'mean zero fraction', plots_dir / f'{model}_mean_zero_fraction_q_by_k')
        _triangular_heatmap(summary, model, 'median_smallest_positive_eigenvalue', 'median smallest positive eigenvalue', 'median positive gap', plots_dir / f'{model}_median_positive_gap_q_by_k')
        _triangular_heatmap(summary, model, 'resolution_rate', 'zero-mode resolution rate', 'resolution rate', plots_dir / f'{model}_resolution_rate_q_by_k')


def plot_k0_by_q_degree(channel_summary: pd.DataFrame, plots_dir: Path) -> None:
    sub = channel_summary[channel_summary['k'] == 0]
    for model, mdf in sub.groupby('model', sort=False):
        fig, ax = plt.subplots(figsize=(12.5, 7.8))
        for degree, ddf in mdf.groupby('target_mean_degree', sort=True):
            ddf = ddf.sort_values('q')
            ax.plot(ddf['q'], ddf['mean_zero_fraction'], marker='o', linewidth=2.1, markersize=7, label=f'<k> target={degree:g}')
        ax.set_xlabel('exclusive q-layer', fontsize=17)
        ax.set_ylabel('mean k=0 zero fraction', fontsize=17)
        ax.set_title(f'{model}: k=0 zero modes across q', fontsize=20)
        ax.tick_params(labelsize=13)
        ax.grid(alpha=0.22)
        ax.legend(fontsize=12)
        _save(fig, plots_dir / f'{model}_k0_zero_fraction_vs_q')


def plot_layer_presence(presence: pd.DataFrame, plots_dir: Path) -> None:
    for model, sub in presence.groupby('model', sort=False):
        pivot = sub.pivot(index='target_mean_degree', columns='q', values='presence_fraction').sort_index()
        fig, ax = plt.subplots(figsize=(12.5, 6.8))
        im = ax.imshow(pivot.to_numpy(dtype=float), aspect='auto', origin='lower', vmin=0.0, vmax=1.0)
        ax.set_title(f'{model}: exclusive q-layer presence', fontsize=20)
        ax.set_xlabel('q', fontsize=17); ax.set_ylabel('target mean degree', fontsize=17)
        ax.set_xticks(range(len(pivot.columns)), [str(int(x)) for x in pivot.columns])
        ax.set_yticks(range(len(pivot.index)), [f'{float(x):g}' for x in pivot.index])
        ax.tick_params(labelsize=12)
        cbar = fig.colorbar(im, ax=ax); cbar.set_label('fraction of realizations with layer', fontsize=13)
        _save(fig, plots_dir / f'{model}_layer_presence')


def _safe_token(text: str) -> str:
    out = ''.join(ch if ch.isalnum() or ch in '-_.' else '_' for ch in str(text))
    return out.strip('_') or 'item'


def _full_spectrum_from_row(row: pd.Series, run_dir: Path) -> tuple[np.ndarray | None, int]:
    """Load a full spectrum when the diagnostic representation supports it.

    Returns (eigenvalues, multiplicity).  ``multiplicity`` is used for the
    analytic scalar-identity shortcut, where a single stored eigenvalue
    represents an n-fold repeated eigenvalue.  For ordinary full spectra the
    multiplicity is one.
    """
    rel = str(row.get('spectrum_file', '') or '')
    if not rel:
        return None, 1
    path = run_dir / rel
    if not path.exists():
        return None, 1
    vals = np.asarray(np.load(path, allow_pickle=False), dtype=float).reshape(-1)
    scope = str(row.get('spectrum_scope', ''))
    if scope == 'full':
        return vals, 1
    if scope == 'analytic_compact':
        mult = int(row.get('analytic_multiplicity', row.get('dimension', 1)) or 1)
        return vals, max(mult, 1)
    return None, 1


def _positive_histogram_bins(pos: np.ndarray, *, max_bins: int = 24) -> np.ndarray:
    """Numerically robust positive-eigenvalue bins.

    The report uses a log x-axis, so construct explicit positive log-spaced
    edges even when the spectrum is tightly clustered or exactly repeated.
    This avoids NumPy's equal-range integer-bin failure for scalar-identity or
    nearly degenerate spectra.
    """
    pos = np.asarray(pos, dtype=float)
    lo = float(np.min(pos)); hi = float(np.max(pos))
    if not (np.isfinite(lo) and np.isfinite(hi) and lo > 0 and hi > 0):
        raise ValueError('positive histogram requires finite positive eigenvalues')
    n = min(max_bins, max(3, int(np.sqrt(max(pos.size, 1))) + 1))
    # Expand an exactly/nearly degenerate range multiplicatively.
    if hi <= lo * (1.0 + 1e-10):
        center = max(lo, np.finfo(float).tiny)
        lo = center / 1.05
        hi = center * 1.05
        n = max(2, min(n, 4))
    edges = np.logspace(np.log10(lo), np.log10(hi), n + 1)
    # Defensive fallback for extreme floating-point clustering.
    if not np.all(np.diff(edges) > 0):
        lo = max(lo / 1.1, np.finfo(float).tiny)
        hi = hi * 1.1
        edges = np.logspace(np.log10(lo), np.log10(hi), n + 1)
    return edges


def plot_single_graph_spectrum_report(
    graph_channels: pd.DataFrame,
    run_dir: Path,
    stem: Path,
    *,
    k: int = 0,
    q_values: Sequence[int] | None = None,
    zero_tolerance: float = 1e-10,
    suptitle: str | None = None,
) -> dict[str, object]:
    """Render the requested real-style spectral report for one graph.

    A panel is produced for every requested q.  Missing layers are explicitly
    labelled ``absent``.  If a higher-k channel only has a sparse prefix, the
    panel is labelled ``prefix only`` rather than drawing a misleading full
    spectral histogram.
    """
    sub = graph_channels[graph_channels['k'].astype(int) == int(k)].copy()
    if q_values is None:
        q_values = sorted(int(x) for x in sub['q'].unique().tolist())
    q_values = [int(q) for q in q_values]
    if not q_values:
        return {'plotted': False, 'reason': 'no q values'}

    ncols = len(q_values)
    fig = plt.figure(figsize=(4.0 * ncols, 6.6))
    gs = GridSpec(2, ncols, figure=fig, height_ratios=[4.0, 1.0], hspace=0.38, wspace=0.42)

    full_count = 0
    for col, q in enumerate(q_values):
        ax = fig.add_subplot(gs[0, col])
        ax_mass = fig.add_subplot(gs[1, col])
        rows = sub[sub['q'].astype(int) == q]
        if rows.empty:
            ax.text(0.5, 0.5, 'layer absent', ha='center', va='center', fontsize=13, transform=ax.transAxes)
            ax.set_axis_off()
            ax_mass.text(0.5, 0.5, 'absent', ha='center', va='center', fontsize=11, transform=ax_mass.transAxes)
            ax_mass.set_axis_off()
            continue
        row = rows.iloc[0]
        dim = int(row['dimension'])
        zero_count = int(row['zero_count']) if bool(row['count_resolved']) and pd.notna(row['zero_count']) else None
        zero_fraction = float(row['zero_fraction']) if zero_count is not None and pd.notna(row['zero_fraction']) else math.nan
        gap = float(row['smallest_positive_eigenvalue']) if pd.notna(row['smallest_positive_eigenvalue']) else math.nan
        vals, multiplicity = _full_spectrum_from_row(row, run_dir)
        scope = str(row.get('spectrum_scope', ''))
        title_scope = 'full' if vals is not None else 'prefix only'
        ax.set_title(rf'$L_{{{k}}}^{{({q})}}$  |  {title_scope}', fontsize=14, fontweight='bold')

        if vals is not None:
            full_count += 1
            if multiplicity > 1 and vals.size == 1:
                vals_eff = np.repeat(vals, multiplicity)
            else:
                vals_eff = vals
            pos = vals_eff[vals_eff > float(zero_tolerance)]
            if pos.size:
                bins = _positive_histogram_bins(pos)
                weights = np.full(pos.size, 1.0 / max(dim, 1), dtype=float)
                ax.hist(pos, bins=bins, weights=weights, color=ZERO_COLOR, alpha=0.86)
                ax.set_xscale('log')
                ax.set_xlabel(r'Positive Hodge eigenvalue $\lambda$', fontsize=10)
            else:
                ax.text(0.5, 0.5, 'no positive eigenvalues', ha='center', va='center', fontsize=11, transform=ax.transAxes)
                ax.set_xlabel(r'Positive Hodge eigenvalue $\lambda$', fontsize=10)
            ax.set_ylabel('Positive spectral mass' if col == 0 else '', fontsize=10)
            ax.grid(alpha=GRID_ALPHA)
        else:
            ax.text(0.5, 0.55, 'full spectrum unavailable', ha='center', va='center', fontsize=11, transform=ax.transAxes)
            ax.text(0.5, 0.42, f'scope: {scope}', ha='center', va='center', fontsize=9, transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])

        if zero_count is not None:
            zf = float(zero_fraction)
            pf = max(0.0, 1.0 - zf)
            ax_mass.barh([0], [zf], left=[0], color=ZERO_COLOR, height=0.65)
            ax_mass.barh([0], [pf], left=[zf], color=POSITIVE_COLOR, height=0.65)
            gap_text = f'{gap:.4g}' if np.isfinite(gap) else 'none'
            ax_mass.text(0.02, 0.84, f'zero={zero_count}/{dim} = {zf:.4f}', transform=ax_mass.transAxes, fontsize=9, va='top')
            ax_mass.text(0.98, 0.84, rf'$\lambda_+$={gap_text}', transform=ax_mass.transAxes, fontsize=9, va='top', ha='right')
            ax_mass.set_xlim(0.0, 1.0)
            ax_mass.set_yticks([])
            ax_mass.set_xlabel('Spectral mass fraction', fontsize=9)
            ax_mass.grid(axis='x', alpha=GRID_ALPHA)
        else:
            ax_mass.text(0.5, 0.5, 'nullity unresolved', ha='center', va='center', fontsize=10, transform=ax_mass.transAxes)
            ax_mass.set_axis_off()

    if suptitle:
        fig.suptitle(suptitle, fontsize=16, fontweight='bold', y=0.995)
    fig.subplots_adjust(top=0.91 if suptitle else 0.96)
    _save(fig, stem)
    return {'plotted': True, 'requested_q': len(q_values), 'full_spectrum_panels': full_count}


def render_per_graph_spectrum_reports(
    channels: pd.DataFrame,
    run_dir: Path,
    plots_dir: Path,
    *,
    k_orders: Iterable[int] = (0,),
    q_min: int = 0,
    q_max: int | None = None,
    columns: int = 5,
    zero_tolerance: float = 1e-10,
) -> pd.DataFrame:
    """Create real-style zero-mode spectral reports for every synthetic graph.

    Long q ranges are paginated into ``columns`` q-layers per figure.  This is
    important for q=0..9: each graph gets two compact pages instead of an
    unreadable ten-column strip.
    """
    if columns < 1:
        raise ValueError('columns must be >=1')
    if channels.empty:
        return pd.DataFrame()
    q_hi = int(channels['q'].max()) if q_max is None else int(q_max)
    q_all = list(range(int(q_min), q_hi + 1))
    rows: list[dict[str, object]] = []
    out_root = plots_dir / 'zero_mode_report'
    for gid, gdf in channels.groupby('graph_id', sort=False):
        meta = gdf.iloc[0]
        model = str(meta['model'])
        degree = float(meta['target_mean_degree'])
        rep = int(meta['repeat'])
        for k in [int(x) for x in k_orders]:
            for page_idx, start in enumerate(range(0, len(q_all), columns), start=1):
                qs = q_all[start:start + columns]
                stem = (
                    out_root / _safe_token(model) / f'degree_{degree:g}' /
                    f'{_safe_token(gid)}_k{k}_q{qs[0]:02d}-q{qs[-1]:02d}'
                )
                status = plot_single_graph_spectrum_report(
                    gdf, run_dir, stem, k=k, q_values=qs,
                    zero_tolerance=zero_tolerance,
                    suptitle=f'{model} | target <k>={degree:g} | repeat={rep} | Hodge order k={k}',
                )
                rows.append({
                    'graph_id': gid, 'model': model, 'target_mean_degree': degree,
                    'repeat': rep, 'k': k, 'page': page_idx,
                    'q_min': qs[0], 'q_max': qs[-1],
                    'plot_png': str(stem.with_suffix('.png').relative_to(run_dir)),
                    'plot_pdf': str(stem.with_suffix('.pdf').relative_to(run_dir)),
                    **status,
                })
    return pd.DataFrame(rows)


def zero_count_statistics(channels: pd.DataFrame) -> pd.DataFrame:
    """Explicitly summarize how many zero eigenvalues each L_k^(q) has."""
    rows: list[dict[str, object]] = []
    group_cols = ['model', 'target_mean_degree', 'q', 'k']
    for key, sub_all in channels.groupby(group_cols, sort=True):
        sub = sub_all[sub_all['count_resolved'] & sub_all['zero_count'].notna()].copy()
        zc = sub['zero_count'].to_numpy(dtype=float)
        zf = sub['zero_fraction'].to_numpy(dtype=float)
        gaps = sub['smallest_positive_eigenvalue'].to_numpy(dtype=float)
        gaps = gaps[np.isfinite(gaps)]
        rows.append({
            'model': key[0], 'target_mean_degree': float(key[1]),
            'q': int(key[2]), 'k': int(key[3]),
            'matrix': f'L_{int(key[3])}^({int(key[2])})',
            'graphs_with_layer': int(len(sub_all)),
            'resolved_graphs': int(len(sub)),
            'resolution_rate': float(len(sub) / len(sub_all)) if len(sub_all) else np.nan,
            'mean_zero_count': float(np.mean(zc)) if zc.size else np.nan,
            'sd_zero_count': float(np.std(zc, ddof=1)) if zc.size > 1 else 0.0 if zc.size == 1 else np.nan,
            'median_zero_count': float(np.median(zc)) if zc.size else np.nan,
            'q1_zero_count': float(np.quantile(zc, 0.25)) if zc.size else np.nan,
            'q3_zero_count': float(np.quantile(zc, 0.75)) if zc.size else np.nan,
            'min_zero_count': float(np.min(zc)) if zc.size else np.nan,
            'max_zero_count': float(np.max(zc)) if zc.size else np.nan,
            'mean_zero_fraction': float(np.mean(zf)) if zf.size else np.nan,
            'sd_zero_fraction': float(np.std(zf, ddof=1)) if zf.size > 1 else 0.0 if zf.size == 1 else np.nan,
            'median_zero_fraction': float(np.median(zf)) if zf.size else np.nan,
            'median_smallest_positive_eigenvalue': float(np.median(gaps)) if gaps.size else np.nan,
        })
    return pd.DataFrame(rows)


def plot_k_order_zero_count_boxplots(channels: pd.DataFrame, plots_dir: Path, *, k: int = 0) -> None:
    """Across realizations, show the zero-count distribution for each L_k^(q)."""
    sub0 = channels[(channels['k'].astype(int) == int(k)) & channels['count_resolved'] & channels['zero_count'].notna()].copy()
    for (model, degree), sub in sub0.groupby(['model', 'target_mean_degree'], sort=True):
        qs = sorted(int(q) for q in sub['q'].unique())
        data = [sub.loc[sub['q'].astype(int) == q, 'zero_count'].astype(float).to_numpy() for q in qs]
        if not data:
            continue
        fig, ax = plt.subplots(figsize=(max(10.5, 1.0 * len(qs) + 4.0), 7.2))
        ax.boxplot(data, tick_labels=[rf'$L_{{{k}}}^{{({q})}}$' for q in qs], showmeans=True)
        ax.set_ylabel('Zero-eigenvalue count', fontsize=16)
        ax.set_xlabel('Hodge matrix', fontsize=16)
        ax.set_title(f'{model} | target <k>={float(degree):g} | zero multiplicity across realizations', fontsize=18)
        ax.tick_params(labelsize=12)
        ax.grid(axis='y', alpha=GRID_ALPHA)
        _save(fig, plots_dir / 'zero_count_summary' / f'{_safe_token(model)}_degree_{float(degree):g}_k{k}_zero_count_boxplot')


def plot_k_order_zero_fraction_boxplots(channels: pd.DataFrame, plots_dir: Path, *, k: int = 0) -> None:
    sub0 = channels[(channels['k'].astype(int) == int(k)) & channels['count_resolved'] & channels['zero_fraction'].notna()].copy()
    for (model, degree), sub in sub0.groupby(['model', 'target_mean_degree'], sort=True):
        qs = sorted(int(q) for q in sub['q'].unique())
        data = [sub.loc[sub['q'].astype(int) == q, 'zero_fraction'].astype(float).to_numpy() for q in qs]
        if not data:
            continue
        fig, ax = plt.subplots(figsize=(max(10.5, 1.0 * len(qs) + 4.0), 7.2))
        ax.boxplot(data, tick_labels=[rf'$L_{{{k}}}^{{({q})}}$' for q in qs], showmeans=True)
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel('Zero-eigenvalue fraction', fontsize=16)
        ax.set_xlabel('Hodge matrix', fontsize=16)
        ax.set_title(f'{model} | target <k>={float(degree):g} | zero fraction across realizations', fontsize=18)
        ax.tick_params(labelsize=12)
        ax.grid(axis='y', alpha=GRID_ALPHA)
        _save(fig, plots_dir / 'zero_count_summary' / f'{_safe_token(model)}_degree_{float(degree):g}_k{k}_zero_fraction_boxplot')
