from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
import json
import time

import networkx as nx
import numpy as np
import pandas as pd

from hodge_vnext.cache import ArtifactStore, CACHE_SCHEMA
from hodge_vnext.cliques import enumerate_maximal_clique_layers, retained_layers
from hodge_vnext.config import MethodConfig
from hodge_vnext.hodge import get_hodge_laplacian
from hodge_vnext.models import synthetic_graph_record
from hodge_vnext.utils import params_slug, safe_token, stable_hash, write_json

from .config import method_from_payload, resolve_path
from .zero_mode_diagnostics import ZeroModeSettings, diagnose_hodge_zero_modes


def _progress(msg: str) -> None:
    print(msg, flush=True)


def _params_for_target(model: str, n: int, degree: float, model_cfg: dict[str, Any]) -> dict[str, Any]:
    model = model.upper()
    if model == 'ER':
        return {'p': float(degree) / float(n - 1)}
    if model == 'WS':
        d = int(round(degree))
        if d % 2:
            raise ValueError('WS target mean degree must be even in this matched-degree diagnostic')
        return {'degree': d, 'p': float(model_cfg.get('beta', 0.1))}
    if model == 'SBM':
        return {
            'blocks': int(model_cfg.get('blocks', 4)),
            'mean_degree': float(degree),
            'mu': float(model_cfg.get('mu', 0.2)),
        }
    if model == 'RGG':
        return {
            'lambda': float(degree),
            'dim': int(model_cfg.get('dim', 2)),
            'p_norm': float(model_cfg.get('p_norm', 2.0)),
        }
    raise ValueError(f'unsupported model {model}')


def _run_stem(exp: dict[str, Any]) -> str:
    token = stable_hash({
        'schema': 'synthetic-zero-modes-v2-spectrum-report',
        'n': exp.get('n'), 'models': exp.get('models'),
        'target_mean_degrees': exp.get('target_mean_degrees'),
        'repeats': exp.get('repeats'), 'base_seed': exp.get('base_seed'),
        'max_q': exp.get('max_q'), 'max_k': exp.get('max_k'),
        'diagnostics': exp.get('diagnostics'),
    }, length=12)
    return f"synthetic_zero_modes_{token}"


def _summary_tables(channels: pd.DataFrame, graph_index: pd.DataFrame, max_q: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    resolved = channels[channels['count_resolved']].copy()
    group_cols = ['model', 'target_mean_degree', 'q', 'k']
    rows = []
    for key, sub_all in channels.groupby(group_cols, sort=True):
        sub = sub_all[sub_all['count_resolved']]
        zc = sub['zero_count'].to_numpy(dtype=float) if len(sub) else np.asarray([], dtype=float)
        zf = sub['zero_fraction'].to_numpy(dtype=float) if len(sub) else np.asarray([], dtype=float)
        gaps = sub['smallest_positive_eigenvalue'].to_numpy(dtype=float) if len(sub) else np.asarray([], dtype=float)
        gaps = gaps[np.isfinite(gaps)]
        rows.append({
            'model': key[0], 'target_mean_degree': float(key[1]), 'q': int(key[2]), 'k': int(key[3]),
            'rows': int(len(sub_all)), 'resolved': int(len(sub)),
            'resolution_rate': float(len(sub) / len(sub_all)) if len(sub_all) else np.nan,
            'median_zero_count': float(np.median(zc)) if zc.size else np.nan,
            'mean_zero_count': float(np.mean(zc)) if zc.size else np.nan,
            'median_zero_fraction': float(np.median(zf)) if zf.size else np.nan,
            'mean_zero_fraction': float(np.mean(zf)) if zf.size else np.nan,
            'median_smallest_positive_eigenvalue': float(np.median(gaps)) if gaps.size else np.nan,
        })
    by_degree = pd.DataFrame(rows)

    rows2 = []
    for (model, q, k), sub_all in channels.groupby(['model','q','k'], sort=True):
        sub = sub_all[sub_all['count_resolved']]
        zf = sub['zero_fraction'].to_numpy(dtype=float) if len(sub) else np.asarray([], dtype=float)
        gaps = sub['smallest_positive_eigenvalue'].to_numpy(dtype=float) if len(sub) else np.asarray([], dtype=float)
        gaps = gaps[np.isfinite(gaps)]
        rows2.append({
            'model': model, 'q': int(q), 'k': int(k),
            'rows': int(len(sub_all)), 'resolved': int(len(sub)),
            'resolution_rate': float(len(sub)/len(sub_all)) if len(sub_all) else np.nan,
            'mean_zero_fraction': float(np.mean(zf)) if zf.size else np.nan,
            'median_zero_fraction': float(np.median(zf)) if zf.size else np.nan,
            'median_smallest_positive_eigenvalue': float(np.median(gaps)) if gaps.size else np.nan,
        })
    by_model = pd.DataFrame(rows2)

    presence_rows = []
    for (model, degree), gsub in graph_index.groupby(['model','target_mean_degree'], sort=True):
        total = int(len(gsub))
        ids = set(gsub['graph_id'].astype(str))
        csub = channels[(channels['model'] == model) & (channels['target_mean_degree'] == degree)]
        for q in range(max_q + 1):
            with_layer = int(csub.loc[csub['q'] == q, 'graph_id'].astype(str).nunique())
            presence_rows.append({
                'model': model, 'target_mean_degree': float(degree), 'q': q,
                'graphs_total': total, 'graphs_with_layer': with_layer,
                'presence_fraction': float(with_layer / total) if total else np.nan,
            })
    presence = pd.DataFrame(presence_rows)
    return by_degree, by_model, presence


def run_synthetic_zero_modes(
    payload: dict[str, Any],
    config_path: Path,
    *,
    force: bool = False,
    force_diagnostics: bool = False,
    progress: Callable[[str], None] | None = None,
) -> Path:
    progress = progress or _progress
    cfg = method_from_payload(payload, config_path)
    exp = dict(payload.get('synthetic_zero_modes', {}))
    settings = ZeroModeSettings.from_dict(dict(exp.get('diagnostics', {})))
    n = int(exp.get('n', 200))
    repeats = int(exp.get('repeats', 10))
    base_seed = int(exp.get('base_seed', 20261007))
    degrees = [float(x) for x in exp.get('target_mean_degrees', [6,10,14])]
    models_cfg = dict(exp.get('models', {'ER':{}, 'WS':{}, 'SBM':{}}))
    models = [str(m).upper() for m in models_cfg]
    max_q = int(exp.get('max_q', 4))
    max_k = int(exp.get('max_k', max_q))
    if max_q < 0 or max_k < 0:
        raise ValueError('max_q/max_k must be nonnegative')
    if cfg.max_clique_size is not None and int(cfg.max_clique_size) < max_q + 1:
        raise ValueError('method.max_clique_size must be at least max_q+1 for the zero-mode diagnostic')
    if repeats < 1:
        raise ValueError('repeats must be >=1')

    results_root = resolve_path(exp.get('results_root', 'results/validation/synthetic_zero_modes'), config_path)
    run = results_root / _run_stem(exp)
    complete = run / 'RUN_COMPLETE.json'
    if complete.exists() and not force:
        progress(f'[SKIP] completed synthetic zero-mode run exists: {run}')
        return run
    run.mkdir(parents=True, exist_ok=True)
    for d in ('arrays/eigenvalues', 'tables', 'plots'):
        (run / d).mkdir(parents=True, exist_ok=True)
    if force:
        complete.unlink(missing_ok=True)

    channel_rows: list[dict[str, Any]] = []
    graph_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []

    graph_counter = 0
    for model in models:
        mcfg = dict(models_cfg.get(model, models_cfg.get(model.lower(), {})))
        for degree in degrees:
            params = _params_for_target(model, n, degree, mcfg)
            for rep in range(1, repeats + 1):
                graph_counter += 1
                rec = synthetic_graph_record(cfg.cache.root, model, n, params, base_seed, rep)
                G = rec.graph
                gid = f'{model}_d{degree:g}_r{rep:03d}_{rec.graph_hash}'
                realized_degree = float(2.0 * G.number_of_edges() / G.number_of_nodes())
                graph_rows.append({
                    'graph_index': graph_counter - 1, 'graph_id': gid, 'model': model,
                    'target_mean_degree': degree, 'repeat': rep,
                    'graph_hash': rec.graph_hash, 'N': G.number_of_nodes(), 'E': G.number_of_edges(),
                    'realized_mean_degree': realized_degree,
                    'params': json.dumps(params, sort_keys=True),
                    'graph_seed': rec.source.get('graph_seed'),
                })
                progress(f'[GRAPH {graph_counter}] {model} target_degree={degree:g} repeat={rep}/{repeats} actual={realized_degree:.3f}')
                store = ArtifactStore(rec.cache_root, cfg.cache)
                layers_all = enumerate_maximal_clique_layers(G, store, progress=None)
                layers = retained_layers(layers_all, max_q + 1, 0)
                for q in range(max_q + 1):
                    cliques = layers.get(q, [])
                    layer_rows.append({
                        'graph_id': gid, 'model': model, 'target_mean_degree': degree, 'repeat': rep,
                        'q': q, 'clique_size': q + 1, 'num_maximal_cliques_in_q': len(cliques),
                        'layer_present': bool(cliques),
                    })
                    if not cliques:
                        continue
                    for k in range(0, min(q, max_k) + 1):
                        t0 = time.perf_counter()
                        L = get_hodge_laplacian(q, k, cliques, store, progress=None)
                        summary, eig = diagnose_hodge_zero_modes(
                            L, rec.graph_hash, q, k, store, settings, force=force_diagnostics,
                        )
                        elapsed = time.perf_counter() - t0
                        eig_rel = ''
                        if settings.store_eigenvalues:
                            eig_name = f'{safe_token(gid)}_q{q:03d}_k{k:03d}.npy'
                            eig_path = run / 'arrays' / 'eigenvalues' / eig_name
                            np.save(eig_path, eig)
                            eig_rel = str(eig_path.relative_to(run))
                        row = {
                            'graph_index': graph_counter - 1, 'graph_id': gid, 'model': model,
                            'target_mean_degree': degree, 'repeat': rep,
                            'graph_hash': rec.graph_hash, 'N': G.number_of_nodes(), 'E': G.number_of_edges(),
                            'realized_mean_degree': realized_degree,
                            'q': q, 'k': k, 'num_maximal_cliques_in_q': len(cliques),
                            'diagnostic_seconds': float(elapsed),
                            'spectrum_file': eig_rel,
                            **summary,
                        }
                        channel_rows.append(row)

    channels = pd.DataFrame(channel_rows)
    graphs = pd.DataFrame(graph_rows)
    layers_df = pd.DataFrame(layer_rows)
    by_degree, by_model, presence = _summary_tables(channels, graphs, max_q)

    global_rows = []
    for (q, k), sub_all in channels.groupby(['q','k'], sort=True):
        sub = sub_all[sub_all['count_resolved']]
        zc = sub['zero_count'].to_numpy(dtype=float) if len(sub) else np.asarray([], dtype=float)
        zf = sub['zero_fraction'].to_numpy(dtype=float) if len(sub) else np.asarray([], dtype=float)
        gaps = sub['smallest_positive_eigenvalue'].to_numpy(dtype=float) if len(sub) else np.asarray([], dtype=float)
        gaps = gaps[np.isfinite(gaps)]
        global_rows.append({
            'q': int(q), 'k': int(k), 'graphs': int(len(sub_all)),
            'resolved': int(len(sub)), 'resolution_rate': float(len(sub)/len(sub_all)) if len(sub_all) else np.nan,
            'median_zero_count': float(np.median(zc)) if zc.size else np.nan,
            'mean_zero_count': float(np.mean(zc)) if zc.size else np.nan,
            'median_zero_fraction': float(np.median(zf)) if zf.size else np.nan,
            'mean_zero_fraction': float(np.mean(zf)) if zf.size else np.nan,
            'median_smallest_positive_eigenvalue': float(np.median(gaps)) if gaps.size else np.nan,
        })
    global_summary = pd.DataFrame(global_rows)

    channels.to_csv(run / 'tables' / 'synthetic_zero_modes_per_graph_channel.csv', index=False, encoding='utf-8-sig')
    graphs.to_csv(run / 'tables' / 'synthetic_graph_index.csv', index=False, encoding='utf-8-sig')
    layers_df.to_csv(run / 'tables' / 'synthetic_layer_presence_per_graph.csv', index=False, encoding='utf-8-sig')
    by_degree.to_csv(run / 'tables' / 'synthetic_zero_modes_by_model_degree_qk.csv', index=False, encoding='utf-8-sig')
    by_model.to_csv(run / 'tables' / 'synthetic_zero_modes_by_model_qk.csv', index=False, encoding='utf-8-sig')
    global_summary.to_csv(run / 'tables' / 'synthetic_zero_modes_global_summary.csv', index=False, encoding='utf-8-sig')
    channels[channels['k'] == 0][['graph_id','model','target_mean_degree','repeat','q','zero_count','zero_fraction','dimension','smallest_positive_eigenvalue','count_resolved']].to_csv(run / 'tables' / 'synthetic_k0_connectivity_per_graph_q.csv', index=False, encoding='utf-8-sig')
    presence.to_csv(run / 'tables' / 'synthetic_layer_presence_summary.csv', index=False, encoding='utf-8-sig')

    resolution = channels.groupby(['q','k'], sort=True).agg(rows=('count_resolved','size'), resolved=('count_resolved','sum')).reset_index()
    resolution['resolution_rate'] = resolution['resolved'] / resolution['rows']
    resolution.to_csv(run / 'tables' / 'synthetic_zero_mode_resolution_by_qk.csv', index=False, encoding='utf-8-sig')

    # Explicit matrix-order zero-count statistics.  This is the primary table
    # for answering "how many zero eigenvalues does L_k^(q) have?" across
    # synthetic realizations.
    from .synthetic_zero_modes_plotting import zero_count_statistics
    matrix_stats = zero_count_statistics(channels)
    matrix_stats.to_csv(
        run / 'tables' / 'synthetic_matrix_zero_statistics_by_model_degree_qk.csv',
        index=False, encoding='utf-8-sig'
    )

    # Wide k=0 tables make the real-style q-layer report easy to audit.
    k0 = channels[(channels['k'] == 0) & channels['count_resolved']].copy()
    if not k0.empty:
        for value, fname in [
            ('zero_count', 'synthetic_k0_zero_count_by_graph_q.csv'),
            ('zero_fraction', 'synthetic_k0_zero_fraction_by_graph_q.csv'),
            ('smallest_positive_eigenvalue', 'synthetic_k0_positive_gap_by_graph_q.csv'),
        ]:
            wide = k0.pivot_table(
                index=['graph_id','model','target_mean_degree','repeat'],
                columns='q', values=value, aggfunc='first'
            ).reset_index()
            wide.columns = [
                f'q{int(c)}' if isinstance(c, (int, np.integer)) else str(c)
                for c in wide.columns
            ]
            wide.to_csv(run / 'tables' / fname, index=False, encoding='utf-8-sig')

    report_index = pd.DataFrame()
    if bool(exp.get('generate_plots', True)):
        from .synthetic_zero_modes_plotting import (
            plot_k0_by_q_degree, plot_layer_presence, plot_model_heatmaps,
            render_per_graph_spectrum_reports, plot_k_order_zero_count_boxplots,
            plot_k_order_zero_fraction_boxplots,
        )
        plot_model_heatmaps(by_model, run / 'plots')
        plot_k0_by_q_degree(by_degree, run / 'plots')
        plot_layer_presence(presence, run / 'plots')

        report_cfg = dict(exp.get('spectrum_reports', {}))
        if bool(report_cfg.get('enabled', True)):
            k_orders = [int(x) for x in report_cfg.get('k_orders', [0])]
            q_min_report = int(report_cfg.get('q_min', 0))
            q_max_report = int(report_cfg.get('q_max', max_q))
            report_columns = int(report_cfg.get('columns', 5))
            report_index = render_per_graph_spectrum_reports(
                channels, run, run / 'plots',
                k_orders=k_orders,
                q_min=q_min_report, q_max=q_max_report,
                columns=report_columns,
                zero_tolerance=float(settings.zero_tolerance),
            )
            report_index.to_csv(
                run / 'tables' / 'synthetic_zero_mode_report_index.csv',
                index=False, encoding='utf-8-sig'
            )
            for k_report in k_orders:
                plot_k_order_zero_count_boxplots(channels, run / 'plots', k=k_report)
                plot_k_order_zero_fraction_boxplots(channels, run / 'plots', k=k_report)

    effective = json.loads(json.dumps(payload))
    effective['cache_schema'] = CACHE_SCHEMA
    effective['zero_mode_schema'] = 'hodge-zero-mode-diagnostic-v2'
    effective['scientific_note'] = (
        'Diagnostic only. Globally maximal cliques are separated into independent exclusive q-layers; '
        'Hodge Laplacians are unchanged. Small matrices use full dense spectra; large matrices use a '
        'smallest-eigenvalue prefix. k=0 nullity is exact from connected components; unresolved higher-k '
        'sparse nullities remain explicitly censored.'
    )
    write_json(run / 'effective_config.json', effective)
    write_json(run / 'run_manifest.json', {
        'architecture': 'synthetic independent-q Hodge zero-eigenvalue diagnostic',
        'models': models, 'n': n, 'target_mean_degrees': degrees, 'repeats': repeats,
        'num_graphs': int(len(graphs)), 'max_q': max_q, 'max_k': max_k,
        'independent_sampling': True,
        'method_math_definition_changed': False,
        'cache_identity_changed': False,
        'experiment_statistics_changed': True,
        'random_graph_seed_contract': 'stable model + full params + repeat seed via synthetic_graph_record',
        'baseline_semantics_changed': False,
        'existing_result_comparability': 'diagnostic complements real zero-mode experiment; not a distance benchmark',
        'diagnostics': settings.__dict__,
        'spectrum_report': {
            'style': 'real-style per-graph L_k^(q) positive-spectrum histogram + zero/positive mass bar',
            'primary_hodge_order': 0,
            'report_index_rows': int(len(report_index)),
            'scientific_scope': 'k=0 reports are full spectra under the formal N=200,dense_guard=1024 contract; higher-k prefix spectra are never mislabeled as full',
        },
    })
    write_json(complete, {'status':'complete', 'run_dir':str(run), 'num_graphs':int(len(graphs)), 'num_channels':int(len(channels))})
    progress(f'[DONE] synthetic zero modes: {run}')
    return run
