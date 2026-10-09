from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage, optimal_leaf_ordering
from scipy.spatial.distance import pdist, squareform

from .style import DMEASURE_CMAP, FIG_DPI, LINE_WIDTH, MARKER_SIZE, METHOD_COLORS, SUPTITLE_SIZE, TICK_SIZE, TITLE_SIZE, apply_style


def _save(fig, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix('.png'), dpi=FIG_DPI, bbox_inches='tight', facecolor='white')
    fig.savefig(base.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig)


def _annotate(ax, bars, vals) -> None:
    ymax = ax.get_ylim()[1]
    for bar, value in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.018*ymax, f'{float(value):.3f}', ha='center', va='bottom', fontsize=max(9, TICK_SIZE-6))


def _cluster_order_from_distance(D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    D = np.asarray(D, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError('distance matrix must be square')
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method='average')
    if len(D) > 2:
        Z = optimal_leaf_ordering(Z, condensed)
    order = leaves_list(Z).astype(int)
    return Z, order


def _plot_clustered_heatmap(D: np.ndarray, labels: list[str], title: str, base: Path) -> np.ndarray:
    Z, order = _cluster_order_from_distance(D)
    ordered = np.asarray(D, dtype=float)[np.ix_(order, order)]
    ordered_labels = [labels[i] for i in order]

    fig = plt.figure(figsize=(15.0, 17.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[3.2, 12.0], width_ratios=[12.0, 0.55], hspace=0.02, wspace=0.12)
    ax_den = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0])
    cax = fig.add_subplot(gs[1, 1])
    d = dendrogram(Z, ax=ax_den, no_labels=True, color_threshold=None, above_threshold_color='0.15', link_color_func=lambda _k: '0.15')
    den_order = np.asarray(d['leaves'], dtype=int)
    if not np.array_equal(den_order, order):
        raise RuntimeError('dendrogram leaf order does not match heatmap order')
    ax_den.set_xticks([]); ax_den.set_yticks([]); ax_den.set_xlim(0, 10 * len(order))
    for sp in ax_den.spines.values():
        sp.set_visible(False)

    im = ax.imshow(ordered, origin='upper', cmap=DMEASURE_CMAP, aspect='equal', vmin=0)
    positions = np.arange(len(order))
    fs = max(8, TICK_SIZE - 7)
    ax.set_xticks(positions, ordered_labels, rotation=90, fontsize=fs)
    ax.set_yticks(positions, ordered_labels, fontsize=fs)
    ax.set_xlabel('Parameter cell')
    ax.set_ylabel('Parameter cell')
    fig.suptitle(title, fontweight='bold', fontsize=TITLE_SIZE + 1, y=0.995)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label('Raw dissimilarity', fontsize=18)
    cax.tick_params(labelsize=max(11, TICK_SIZE - 4))
    _save(fig, base)
    return order


_MDS_MARKERS = ('o', 's', '^', 'D', 'P', 'X', 'v', '<', '>', '*')


def _classical_mds_2d(D: np.ndarray, *, zero_diagonal: bool = False) -> tuple[np.ndarray, dict[str, float]]:
    D = np.asarray(D, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError('MDS requires a square matrix')
    n = D.shape[0]
    if n < 2:
        raise ValueError('need at least two points')
    D_use = 0.5 * (D + D.T)
    if zero_diagonal:
        np.fill_diagonal(D_use, 0.0)
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D_use ** 2) @ J
    evals, evecs = np.linalg.eigh(B)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    pos = np.clip(evals[:2], a_min=0.0, a_max=None)
    coords = evecs[:, :2] * np.sqrt(pos)[None, :]
    # deterministic axis sign
    for k in range(coords.shape[1]):
        col = coords[:, k]
        pivot = int(np.argmax(np.abs(col)))
        if col[pivot] < 0:
            coords[:, k] *= -1.0
    # diagnostics
    pdist_orig = squareform(D_use, checks=False)
    pdist_2d = pdist(coords)
    if np.std(pdist_orig) > 0 and np.std(pdist_2d) > 0:
        rho = float(np.corrcoef(np.argsort(np.argsort(pdist_orig)), np.argsort(np.argsort(pdist_2d)))[0, 1])
    else:
        rho = float('nan')
    stress = float(np.sqrt(np.sum((pdist_orig - pdist_2d) ** 2) / max(np.sum(pdist_orig ** 2), 1e-15)))
    pos_sum = float(np.clip(evals, a_min=0.0, a_max=None).sum())
    neg_sum = float(np.clip(-evals, a_min=0.0, a_max=None).sum())
    frac2 = float(pos.sum() / max(pos_sum, 1e-15))
    neg_frac = float(neg_sum / max(pos_sum + neg_sum, 1e-15))
    return coords, {
        'positive_eigenvalue_fraction_2d': frac2,
        'negative_eigenvalue_fraction': neg_frac,
        'distance_spearman_2d': rho,
        'normalized_stress_2d': stress,
    }


def _marker_map(unique_sizes: list[int]) -> dict[int, str]:
    vals = [int(x) for x in sorted(unique_sizes)]
    if len(vals) > len(_MDS_MARKERS):
        raise ValueError('too many size levels for fixed marker vocabulary')
    return {n: _MDS_MARKERS[i] for i, n in enumerate(vals)}


def _state_color_map(states: list[str]) -> dict[str, object]:
    cmap = plt.get_cmap('tab10')
    return {state: cmap(i % 10) for i, state in enumerate(states)}


def _plot_graph_mds(df: pd.DataFrame, diag: dict[str, float], method: str, base: Path) -> None:
    states = df['mechanism_state'].astype(str).drop_duplicates().tolist()
    sizes = sorted(df['network_size'].astype(int).unique().tolist())
    colors = _state_color_map(states)
    markers = _marker_map(sizes)
    fig, ax = plt.subplots(figsize=(15.5, 10.5))
    for state in states:
        sdf = df[df.mechanism_state.astype(str) == state]
        for n in sizes:
            sub = sdf[sdf.network_size.astype(int) == int(n)]
            if sub.empty:
                continue
            ax.scatter(sub.mds_x, sub.mds_y, s=105, marker=markers[n], c=[colors[state]], edgecolors='white', linewidths=0.9, alpha=0.80, zorder=3)
    caption = f"2-D positive-spectrum fraction={diag['positive_eigenvalue_fraction_2d']:.2f}   distance Spearman={diag['distance_spearman_2d']:.2f}   stress={diag['normalized_stress_2d']:.2f}"
    ax.set_xlabel('Classical MDS dimension 1')
    ax.set_ylabel('Classical MDS dimension 2')
    ax.set_title(f'{method}: graph-level ER–WS–RGG size 2-D MDS\n{caption}', fontweight='bold', pad=12)
    ax.set_aspect('equal', adjustable='datalim')
    mech_handles = [plt.Line2D([0], [0], marker='o', linestyle='', markersize=10, markerfacecolor=colors[s], markeredgecolor='white', markeredgewidth=0.8, label=s) for s in states]
    size_handles = [plt.Line2D([0], [0], marker=markers[n], linestyle='', markersize=9, markerfacecolor='0.72', markeredgecolor='0.18', markeredgewidth=0.8, label=f'N={n}') for n in sizes]
    first = ax.legend(handles=mech_handles, title='Mechanism', frameon=True, loc='upper left', bbox_to_anchor=(1.02, 1.00), fontsize=max(11, TICK_SIZE - 4), title_fontsize=max(12, TICK_SIZE - 3))
    ax.add_artist(first)
    ax.legend(handles=size_handles, title='Network size', frameon=True, loc='lower left', bbox_to_anchor=(1.02, 0.00), fontsize=max(11, TICK_SIZE - 4), title_fontsize=max(12, TICK_SIZE - 3))
    fig.tight_layout(rect=(0, 0, 0.80, 1))
    _save(fig, base)


def _plot_cell_mds(df: pd.DataFrame, diag: dict[str, float], method: str, base: Path) -> None:
    states = df['mechanism_state'].astype(str).drop_duplicates().tolist()
    sizes = sorted(df['network_size'].astype(int).unique().tolist())
    colors = _state_color_map(states)
    markers = _marker_map(sizes)
    fig, ax = plt.subplots(figsize=(15.5, 10.5))
    for state in states:
        sdf = df[df.mechanism_state.astype(str) == state].sort_values('network_size')
        ax.plot(sdf.mds_x, sdf.mds_y, color=colors[state], linewidth=1.8, alpha=0.45, zorder=1)
        for row in sdf.itertuples(index=False):
            n = int(row.network_size)
            ax.scatter([float(row.mds_x)], [float(row.mds_y)], s=170, marker=markers[n], c=[colors[state]], edgecolors='white', linewidths=1.1, alpha=0.95, zorder=3)
    caption = f"2-D positive-spectrum fraction={diag['positive_eigenvalue_fraction_2d']:.2f}   distance Spearman={diag['distance_spearman_2d']:.2f}   stress={diag['normalized_stress_2d']:.2f}"
    ax.set_xlabel('Classical MDS dimension 1')
    ax.set_ylabel('Classical MDS dimension 2')
    ax.set_title(f'{method}: state × size 2-D MDS\n{caption}', fontweight='bold', pad=12)
    ax.set_aspect('equal', adjustable='datalim')
    mech_handles = [plt.Line2D([0], [0], marker='o', linestyle='', markersize=10, markerfacecolor=colors[s], markeredgecolor='white', markeredgewidth=0.8, label=s) for s in states]
    size_handles = [plt.Line2D([0], [0], marker=markers[n], linestyle='', markersize=9, markerfacecolor='0.72', markeredgecolor='0.18', markeredgewidth=0.8, label=f'N={n}') for n in sizes]
    first = ax.legend(handles=mech_handles, title='Mechanism', frameon=True, loc='upper left', bbox_to_anchor=(1.02, 1.00), fontsize=max(11, TICK_SIZE - 4), title_fontsize=max(12, TICK_SIZE - 3))
    ax.add_artist(first)
    ax.legend(handles=size_handles, title='Network size', frameon=True, loc='lower left', bbox_to_anchor=(1.02, 0.00), fontsize=max(11, TICK_SIZE - 4), title_fontsize=max(12, TICK_SIZE - 3))
    fig.tight_layout(rect=(0, 0, 0.80, 1))
    _save(fig, base)


def _compute_and_save_mds(run: Path, method: str, graph_index: pd.DataFrame, cells: pd.DataFrame) -> None:
    graph_path = run / 'arrays' / f'graph_distance_{method}.npy'
    cell_path = run / 'arrays' / f'cell_distance_{method}.npy'
    if not graph_path.exists() or not cell_path.exists():
        return
    gD = np.load(graph_path)
    cD = np.load(cell_path)
    tables = run / 'tables'
    # graph-level
    gcoords, gdiag = _classical_mds_2d(gD, zero_diagonal=False)
    gdf = graph_index.copy()
    gdf['mds_x'] = gcoords[:, 0]
    gdf['mds_y'] = gcoords[:, 1]
    gdf.to_csv(tables / f'mds_graph_{method}.csv', index=False)
    # cell-level
    ccoords, cdiag = _classical_mds_2d(cD, zero_diagonal=True)
    cdf = cells.copy()
    cdf['mds_x'] = ccoords[:, 0]
    cdf['mds_y'] = ccoords[:, 1]
    cdf.to_csv(tables / f'mds_cell_{method}.csv', index=False)
    # diagnostics append/update
    diag_path = tables / 'mds_diagnostics.csv'
    rows = []
    if diag_path.exists():
        rows = pd.read_csv(diag_path).to_dict('records')
        rows = [r for r in rows if not (str(r.get('method')) == method and str(r.get('level')) in {'graph', 'cell'})]
    rows.extend([
        {'method': method, 'level': 'graph', **gdiag},
        {'method': method, 'level': 'cell', **cdiag},
    ])
    pd.DataFrame(rows).to_csv(diag_path, index=False)
    _plot_graph_mds(gdf, gdiag, method, run / 'plots' / f'mds_graph_{method}')
    _plot_cell_mds(cdf, cdiag, method, run / 'plots' / f'mds_cell_{method}')


def plot_size_discrimination(run_dir: str | Path) -> None:
    apply_style()
    run = Path(run_dir)
    plots = run / 'plots'
    plots.mkdir(exist_ok=True)
    metrics = pd.read_csv(run / 'tables' / 'metrics_all_methods.csv')
    methods = [m for m in ['Hodge','DMeasure','NetLSD','Portrait','Count'] if m in set(metrics.method)]
    core = ['balanced_accuracy','macro_f1','same_class_auroc','mean_average_precision','ndcg_at_5']
    labels = ['Bal. Acc.','Macro-F1','AUROC','mAP','NDCG@5']

    for protocol in ['matched_size','cross_size']:
        sub = metrics[(metrics.protocol == protocol) & (metrics.label_level == 'fine')]
        if sub.empty:
            continue
        fig, axes = plt.subplots(len(methods), 1, figsize=(15.5, max(8, 3.6*len(methods))), sharex=True, squeeze=False)
        x = np.arange(len(core))
        for ax, method in zip(axes[:,0], methods):
            row = sub[sub.method == method]
            if row.empty:
                ax.axis('off'); continue
            vals = [float(row.iloc[0][c]) for c in core]
            bars = ax.bar(x, vals, color=METHOD_COLORS.get(method))
            ax.set_ylim(0, 1.06)
            ax.set_ylabel(method)
            _annotate(ax, bars, vals)
        axes[-1,0].set_xticks(x, labels, rotation=20, ha='right')
        title = 'same-size mechanism discrimination' if protocol == 'matched_size' else 'cross-size mechanism generalization'
        fig.suptitle(f'ER–WS–RGG size experiment: {title}', fontweight='bold', fontsize=SUPTITLE_SIZE)
        fig.tight_layout(rect=(0,0,1,0.98))
        _save(fig, plots / f'core_metrics_{protocol}')

    audit = pd.read_csv(run / 'tables' / 'structural_audit.csv')
    fig, ax = plt.subplots(figsize=(13.5, 8.5))
    for state, sub in audit.groupby('mechanism_state', sort=False):
        sub = sub.sort_values('network_size')
        ax.plot(sub.network_size, sub.realized_mean_degree_mean, marker='o', linewidth=LINE_WIDTH, markersize=MARKER_SIZE, label=state)
    target = float(audit.target_mean_degree.iloc[0])
    ax.axhline(target, linestyle='--', linewidth=2.0, color='0.25', label=f'target κ={target:g}')
    ax.set_xscale('log')
    ax.set_xlabel('Network size N')
    ax.set_ylabel('Realized mean degree')
    ax.set_title('Size experiment structural audit: mean degree held approximately fixed', fontweight='bold')
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    _save(fig, plots / 'structural_mean_degree_vs_size')

    sens_path = run / 'tables' / 'size_sensitivity.csv'
    if sens_path.exists():
        sens = pd.read_csv(sens_path)
        fig, axes = plt.subplots(len(methods), 1, figsize=(15.5, max(9, 4.1*len(methods))), sharex=True, squeeze=False)
        for ax, method in zip(axes[:,0], methods):
            sm = sens[(sens.method == method) & (sens.adjacent_size_pair == True)]
            for state, sub in sm.groupby('mechanism_state', sort=False):
                sub = sub.sort_values('size_a')
                x = np.sqrt(sub.size_a.to_numpy(float) * sub.size_b.to_numpy(float))
                ax.plot(x, sub.between_mean, marker='o', linewidth=LINE_WIDTH, markersize=MARKER_SIZE, label=state)
            ax.set_ylabel(f'{method}\nraw distance')
            ax.set_xscale('log')
            ax.legend(frameon=False, ncol=3, fontsize=max(9,TICK_SIZE-5))
        axes[-1,0].set_xlabel('Adjacent network-size pair (geometric midpoint of N)')
        fig.suptitle('Same-mechanism sensitivity to changing network size', fontweight='bold', fontsize=SUPTITLE_SIZE)
        fig.tight_layout(rect=(0,0,1,0.985))
        _save(fig, plots / 'size_sensitivity_all_methods')

    cells = pd.read_csv(run / 'tables' / 'parameter_cells.csv').sort_values('cell_index')
    cell_labels = cells.cell_label.astype(str).tolist()
    graph_index = pd.read_csv(run / 'tables' / 'graph_index.csv').sort_values('graph_index') if (run / 'tables' / 'graph_index.csv').exists() else pd.DataFrame()
    for method in methods:
        p = run / 'arrays' / f'cell_distance_{method}.npy'
        if not p.exists():
            continue
        D = np.load(p)
        order = _plot_clustered_heatmap(D, cell_labels, f'{method}: state × network-size raw dissimilarity', plots / f'size_cell_heatmap_{method}')
        pd.DataFrame({'order': np.arange(len(order)), 'cell_index': order, 'cell_label': [cell_labels[i] for i in order]}).to_csv(run / 'tables' / f'size_heatmap_order_{method}.csv', index=False)
        if not graph_index.empty:
            _compute_and_save_mds(run, method, graph_index, cells)
