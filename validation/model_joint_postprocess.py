from __future__ import annotations

from pathlib import Path
import json
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .style import FIG_DPI, METHOD_COLORS, SUPTITLE_SIZE, TITLE_SIZE, TICK_SIZE, LEGEND_SIZE, apply_style
from .model_joint_plotting import _cluster_heatmap, _cmds

PROTOCOLS = [
    ("matched_N_k", "Matched N & k"),
    ("cross_N_only", "Cross N only"),
    ("cross_k_only", "Cross k only"),
    ("cross_N_k", "Cross N & k"),
    ("global", "Global"),
]
CORE_METRICS = [
    ("balanced_accuracy", "Balanced Accuracy"),
    ("macro_f1", "Macro-F1"),
    ("same_class_auroc", "AUROC"),
    ("mean_average_precision", "mAP"),
    ("ndcg_at_5", "NDCG@5"),
]
METHOD_ORDER = ["Hodge", "DMeasure", "NetLSD", "Portrait", "Count"]
MARKERS = ['o', 's', '^', 'D', 'P', 'X', 'v', '*']


def _save(fig, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix('.png'), dpi=FIG_DPI, bbox_inches='tight', facecolor='white')
    fig.savefig(base.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig)


def _discover_methods(run: Path, metrics: pd.DataFrame | None = None) -> list[str]:
    methods: set[str] = set()
    if metrics is not None and not metrics.empty and 'method' in metrics.columns:
        methods.update(metrics['method'].astype(str).tolist())
    arrays = run / 'arrays'
    if arrays.exists():
        for p in arrays.glob('cell_distance_*.npy'):
            suffix = p.stem.replace('cell_distance_', '')
            # Exclude uncertainty/count artifacts such as cell_distance_std_Hodge.npy.
            if suffix.startswith('std_') or suffix.startswith('pair_count_'):
                continue
            methods.add(suffix)
        for p in arrays.glob('graph_distance_*.npy'):
            suffix = p.stem.replace('graph_distance_', '')
            if suffix.startswith('std_') or suffix.startswith('pair_count_'):
                continue
            methods.add(suffix)
    tables = run / 'tables'
    if tables.exists():
        for p in tables.glob('mds_cell_*.csv'):
            suffix = p.stem.replace('mds_cell_', '')
            if suffix.startswith('std_'):
                continue
            methods.add(suffix)
    # Never interpret diagnostic artifacts as scientific methods.
    methods = {m for m in methods if not m.startswith('std_') and not m.startswith('pair_count_')}
    ordered = [m for m in METHOD_ORDER if m in methods]
    extras = sorted(m for m in methods if m not in METHOD_ORDER)
    return ordered + extras


def _read_parameters(run: Path) -> pd.DataFrame:
    path = run / 'tables' / 'parameter_cells.csv'
    if not path.exists():
        raise FileNotFoundError(f'Missing parameter cell table: {path}')
    df = pd.read_csv(path).sort_values('cell_index').reset_index(drop=True)
    required = {'cell_index', 'cell_label', 'mechanism_state', 'network_size', 'target_mean_degree'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'parameter_cells.csv missing columns: {sorted(missing)}')
    return df


def _matrix_to_csv(matrix: np.ndarray, labels: Iterable[str], out_csv: Path) -> None:
    labels = list(labels)
    df = pd.DataFrame(matrix, index=labels, columns=labels)
    df.index.name = 'label'
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, encoding='utf-8-sig')


def export_distance_matrices(run_dir: str | Path) -> list[Path]:
    run = Path(run_dir)
    tables = run / 'tables'
    arrays = run / 'arrays'
    if not arrays.exists():
        print(f'[WARN] arrays directory not found: {arrays}')
        return []
    params = _read_parameters(run)
    labels = params['cell_label'].astype(str).tolist()
    exported: list[Path] = []
    methods = _discover_methods(run)
    graph_meta_path = tables / 'graph_index.csv'
    graph_labels: list[str] | None = None
    if graph_meta_path.exists():
        gdf = pd.read_csv(graph_meta_path)
        if 'graph_hash' in gdf.columns:
            graph_labels = gdf['graph_hash'].astype(str).tolist()
        elif 'graph_index' in gdf.columns:
            graph_labels = [f'graph_{int(x)}' for x in gdf['graph_index']]
    for method in methods:
        cell = arrays / f'cell_distance_{method}.npy'
        if cell.exists():
            M = np.load(cell)
            for name in [f'cell_distance_{method}.csv', f'distance_matrix_cell_{method}.csv', f'distance_matrix_{method}.csv']:
                out = tables / name
                _matrix_to_csv(M, labels, out)
                exported.append(out)
        std = arrays / f'cell_distance_std_{method}.npy'
        if std.exists():
            out = tables / f'cell_distance_std_{method}.csv'
            _matrix_to_csv(np.load(std), labels, out)
            exported.append(out)
        cnt = arrays / f'cell_pair_count_{method}.npy'
        if cnt.exists():
            out = tables / f'cell_pair_count_{method}.csv'
            _matrix_to_csv(np.load(cnt), labels, out)
            exported.append(out)
        graph = arrays / f'graph_distance_{method}.npy'
        if graph.exists() and graph_labels is not None:
            out = tables / f'graph_distance_{method}.csv'
            _matrix_to_csv(np.load(graph), graph_labels, out)
            exported.append(out)
    return exported


def plot_protocol_metric_panels(run_dir: str | Path) -> list[Path]:
    apply_style()
    run = Path(run_dir)
    metrics_path = run / 'tables' / 'metrics_all_methods.csv'
    metrics = pd.read_csv(metrics_path)
    methods = _discover_methods(run, metrics)
    plots = run / 'plots'
    outputs: list[Path] = []
    # Remove stale AUROC-by-protocol artifacts: AUROC is now reported by a dedicated GLOBAL-only panel.
    for stale in [
        run / 'plots' / 'metric_same_class_auroc_by_protocol.png',
        run / 'plots' / 'metric_same_class_auroc_by_protocol.pdf',
        run / 'tables' / 'metric_same_class_auroc_by_protocol.csv',
    ]:
        if stale.exists():
            stale.unlink()
    # Overwrite the existing protocol-wise summary with an explicit method legend.
    for protocol, protocol_label in PROTOCOLS:
        sub = metrics[metrics['protocol'].astype(str) == protocol].copy()
        x = np.arange(len(CORE_METRICS), dtype=float)
        width = 0.82 / max(1, len(methods))
        fig, ax = plt.subplots(figsize=(17, 10.5))
        for i, method in enumerate(methods):
            r = sub[sub['method'].astype(str) == method]
            if r.empty:
                continue
            vals = [float(r.iloc[0][col]) for col, _ in CORE_METRICS]
            offset = (i - (len(methods) - 1) / 2.0) * width
            bars = ax.bar(x + offset, vals, width=width * 0.92, color=METHOD_COLORS.get(method), label=method)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2.0, v + 0.015, f'{v:.3f}', ha='center', va='bottom', fontsize=max(9, TICK_SIZE - 5), rotation=90)
        ax.set_xticks(x, [lab for _, lab in CORE_METRICS], rotation=20, ha='right')
        ax.set_ylim(0, 1.08)
        ax.set_ylabel('Score')
        ax.set_title(f'ER / WS / RGG joint N × mean-degree comparison — {protocol_label}', fontweight='bold', fontsize=TITLE_SIZE)
        ax.legend(frameon=False, ncol=min(len(methods), 5), loc='upper center')
        fig.tight_layout()
        out = plots / f'core_metrics_{protocol}'
        _save(fig, out)
        outputs.extend([out.with_suffix('.png'), out.with_suffix('.pdf')])

    # One figure per metric comparing all methods across protocols.
    # Exception: AUROC also gets a dedicated GLOBAL-only figure for cleaner reporting.
    for metric, metric_label in CORE_METRICS:
        if metric == 'same_class_auroc':
            # AUROC is reported separately as a dedicated GLOBAL-only figure.
            continue
        fig, ax = plt.subplots(figsize=(17, 10.5))
        x = np.arange(len(PROTOCOLS), dtype=float)
        width = 0.82 / max(1, len(methods))
        for i, method in enumerate(methods):
            vals = []
            for protocol, _ in PROTOCOLS:
                r = metrics[(metrics['method'].astype(str) == method) & (metrics['protocol'].astype(str) == protocol)]
                vals.append(float(r.iloc[0][metric]) if not r.empty else np.nan)
            offset = (i - (len(methods) - 1) / 2.0) * width
            bars = ax.bar(x + offset, vals, width=width * 0.92, color=METHOD_COLORS.get(method, '#808080'), label=method)
            for b, v in zip(bars, vals):
                if np.isfinite(v):
                    ax.text(b.get_x() + b.get_width() / 2.0, v + 0.015, f'{v:.3f}', ha='center', va='bottom', fontsize=max(9, TICK_SIZE - 5), rotation=90)
        ax.set_xticks(x, [lab for _, lab in PROTOCOLS], rotation=20, ha='right')
        ax.set_ylim(0, 1.08)
        ax.set_ylabel(metric_label)
        ax.set_title(f'{metric_label} — method comparison across evaluation protocols', fontweight='bold', fontsize=TITLE_SIZE)
        ax.legend(frameon=False, ncol=min(len(methods), 5), loc='upper center')
        fig.tight_layout()
        stem = plots / f'metric_{metric}_by_protocol'
        _save(fig, stem)
        outputs.extend([stem.with_suffix('.png'), stem.with_suffix('.pdf')])
        rows = []
        for protocol, protocol_label in PROTOCOLS:
            row = {'protocol': protocol, 'protocol_label': protocol_label}
            for method in methods:
                r = metrics[(metrics['method'].astype(str) == method) & (metrics['protocol'].astype(str) == protocol)]
                row[method] = float(r.iloc[0][metric]) if not r.empty else np.nan
            rows.append(row)
        pd.DataFrame(rows).to_csv(run / 'tables' / f'metric_{metric}_by_protocol.csv', index=False, encoding='utf-8-sig')

    global_only = metrics[metrics['protocol'].astype(str) == 'global'].copy()
    if not global_only.empty:
        metric = 'same_class_auroc'
        metric_label = 'AUROC'
        fig, ax = plt.subplots(figsize=(12.5, 8.5))
        vals = []
        x = np.arange(len(methods), dtype=float)
        for method in methods:
            r = global_only[global_only['method'].astype(str) == method]
            vals.append(float(r.iloc[0][metric]) if not r.empty else np.nan)
        bars = ax.bar(x, vals, color=[METHOD_COLORS.get(m, '#808080') for m in methods], width=0.75)
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2.0, v + 0.015, f'{v:.3f}', ha='center', va='bottom', fontsize=max(10, TICK_SIZE - 4))
        ax.set_xticks(x, methods, rotation=20, ha='right')
        ax.set_ylim(0, 1.08)
        ax.set_ylabel(metric_label)
        ax.set_title('Global AUROC — method comparison', fontweight='bold', fontsize=TITLE_SIZE)
        fig.tight_layout()
        stem = plots / 'global_auroc_method_comparison'
        _save(fig, stem)
        outputs.extend([stem.with_suffix('.png'), stem.with_suffix('.pdf')])
        pd.DataFrame({'method': methods, 'protocol': 'global', 'same_class_auroc': vals}).to_csv(run / 'tables' / 'global_auroc_method_comparison.csv', index=False, encoding='utf-8-sig')
    return outputs


def _size_to_area(vals: np.ndarray, v: float) -> float:
    lo = float(vals.min())
    hi = float(vals.max())
    frac = 0.5 if abs(hi - lo) < 1e-12 else (float(v) - lo) / (hi - lo)
    return 90.0 + 110.0 * frac


def plot_joint_mds_with_legends(run_dir: str | Path) -> list[Path]:
    apply_style()
    run = Path(run_dir)
    params = _read_parameters(run)
    metrics_path = run / 'tables' / 'metrics_all_methods.csv'
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    methods = _discover_methods(run, metrics)
    arrays = run / 'arrays'
    tables = run / 'tables'
    plots = run / 'plots'
    outputs: list[Path] = []

    states = params['mechanism_state'].astype(str).drop_duplicates().tolist()
    cmap = plt.get_cmap('tab10')
    state_colors = {s: cmap(i % 10) for i, s in enumerate(states)}
    sizes = sorted(params['network_size'].astype(int).unique().tolist())
    marker_map = {n: MARKERS[i % len(MARKERS)] for i, n in enumerate(sizes)}
    k_levels = sorted(params['target_mean_degree'].astype(float).unique().tolist())
    k_values = params['target_mean_degree'].astype(float).to_numpy()

    for method in methods:
        cell = arrays / f'cell_distance_{method}.npy'
        if cell.exists():
            D = np.load(cell)
            order = _cluster_heatmap(
                D,
                params['cell_label'].astype(str).tolist(),
                f'{method}: joint mechanism × N × mean-degree raw dissimilarity',
                plots / f'joint_cell_heatmap_{method}',
            )
            pd.DataFrame({
                'order': np.arange(len(order)),
                'cell_index': order,
                'cell_label': [params.iloc[i]['cell_label'] for i in order],
            }).to_csv(tables / f'joint_heatmap_order_{method}.csv', index=False, encoding='utf-8-sig')
            X = _cmds(D)
            out = params[['cell_index', 'cell_label', 'mechanism_state', 'network_size', 'target_mean_degree']].copy()
            out['mds_x'] = X[:, 0]
            out['mds_y'] = X[:, 1]
            out.to_csv(tables / f'mds_cell_{method}.csv', index=False, encoding='utf-8-sig')
        else:
            mds_csv = tables / f'mds_cell_{method}.csv'
            if not mds_csv.exists():
                print(f'[WARN] skip MDS for {method}: no cell distance array or mds_cell csv')
                continue
            out = pd.read_csv(mds_csv)

        # Use a dedicated legend column so the legend can never overflow the canvas.
        fig = plt.figure(figsize=(18.5, 10.5))
        gs = fig.add_gridspec(1, 2, width_ratios=[5.2, 1.35], wspace=0.03)
        ax = fig.add_subplot(gs[0, 0])
        lax = fig.add_subplot(gs[0, 1])
        lax.axis('off')

        for r in out.itertuples(index=False):
            area = _size_to_area(k_values, float(r.target_mean_degree))
            ax.scatter(
                r.mds_x, r.mds_y,
                s=area,
                marker=marker_map[int(r.network_size)],
                c=[state_colors[str(r.mechanism_state)]],
                edgecolors='white', linewidths=0.9, alpha=0.88,
            )
        ax.set_xlabel('Classical MDS dimension 1')
        ax.set_ylabel('Classical MDS dimension 2')
        ax.set_title(f'{method}: cell-level 2-D MDS', fontweight='bold', pad=10)
        ax.axhline(0.0, color='0.88', linewidth=1.2, zorder=0)
        ax.axvline(0.0, color='0.88', linewidth=1.2, zorder=0)

        mech_handles = [Patch(facecolor=state_colors[s], edgecolor='none', label=s) for s in states]
        n_handles = [
            Line2D([0], [0], marker=marker_map[n], linestyle='None',
                   markerfacecolor='0.45', markeredgecolor='0.45', color='0.45',
                   markersize=9, label=f'N={n}')
            for n in sizes
        ]
        # Show every actual mean-degree level; no synthetic median such as k=9.
        size_handles = [
            lax.scatter([], [], s=_size_to_area(k_values, v), color='0.55', alpha=0.75, label=f'k={v:g}')
            for v in k_levels
        ]

        leg1 = lax.legend(
            handles=mech_handles, title='Mechanism', frameon=False,
            loc='upper left', bbox_to_anchor=(0.0, 1.0),
            fontsize=10.5, title_fontsize=12.0, borderaxespad=0.0,
            handletextpad=0.55, labelspacing=0.55,
        )
        lax.add_artist(leg1)
        leg2 = lax.legend(
            handles=n_handles, title='Network size', frameon=False,
            loc='upper left', bbox_to_anchor=(0.0, 0.61),
            fontsize=10.5, title_fontsize=12.0, borderaxespad=0.0,
            handletextpad=0.55, labelspacing=0.55,
        )
        lax.add_artist(leg2)
        lax.legend(
            handles=size_handles, title='Mean degree', frameon=False,
            loc='upper left', bbox_to_anchor=(0.0, 0.27),
            fontsize=10.5, title_fontsize=12.0, borderaxespad=0.0,
            scatterpoints=1, handletextpad=0.55, labelspacing=0.48,
        )
        fig.subplots_adjust(top=0.93, bottom=0.11, left=0.08, right=0.98)
        stem = plots / f'joint_mds_cell_{method}'
        _save(fig, stem)
        outputs.extend([stem.with_suffix('.png'), stem.with_suffix('.pdf')])
    return outputs


def postprocess_joint_run(run_dir: str | Path) -> dict[str, object]:
    run = Path(run_dir)
    metrics = run / 'tables' / 'metrics_all_methods.csv'
    params = run / 'tables' / 'parameter_cells.csv'
    if not metrics.exists():
        raise FileNotFoundError(f'Missing metrics table: {metrics}')
    if not params.exists():
        raise FileNotFoundError(f'Missing parameter cell table: {params}')
    exported = export_distance_matrices(run)
    metric_plots = plot_protocol_metric_panels(run)
    mds_outputs = plot_joint_mds_with_legends(run)
    manifest = {
        'run_dir': str(run),
        'exported_csv_count': len(exported),
        'metric_plot_count': len(metric_plots),
        'mds_plot_count': len(mds_outputs),
        'note': 'Pure post-processing only: no method distances were recomputed. Outputs were generated from existing arrays/*.npy and tables/*.csv artifacts.',
    }
    (run / 'postprocess_joint_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest
