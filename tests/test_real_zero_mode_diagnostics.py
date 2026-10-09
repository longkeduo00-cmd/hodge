from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from validation.real_zero_mode_diagnostics import run_real_zero_mode_diagnostics


def _write_graph(path: Path, edges: list[tuple[int, int]], nodes: int) -> tuple[Path, Path]:
    e = path.with_suffix('.edgelist')
    n = path.with_suffix('.nodes')
    e.write_text('\n'.join(f'{u} {v}' for u, v in edges) + '\n', encoding='utf-8')
    n.write_text('\n'.join(str(i) for i in range(nodes)) + '\n', encoding='utf-8')
    return e, n


def test_real_zero_mode_runner_saves_real_spectra_and_exact_k0(tmp_path: Path):
    e1, n1 = _write_graph(tmp_path / 'triangle', [(0, 1), (1, 2), (0, 2)], 3)
    e2, n2 = _write_graph(tmp_path / 'path', [(0, 1), (1, 2), (2, 3)], 4)
    manifest = tmp_path / 'manifest.csv'
    pd.DataFrame([
        {'label': 'triangle', 'domain': 'A', 'path': str(e1), 'nodes_path': str(n1)},
        {'label': 'path', 'domain': 'B', 'path': str(e2), 'nodes_path': str(n2)},
    ]).to_csv(manifest, index=False)
    payload = {
        'method': {
            'max_clique_size': 3,
            'max_hodge_order': None,
            'min_layer_q': 0,
            'time_min': 0.01,
            'time_max': 100.0,
            'time_points': 8,
            'normalization': 'per_simplex',
            'alpha': 1.0,
            'solver': 'exact',
            'exact': {'dimension_guard': None, 'eigen_zero_tolerance': 1e-10, 'eigen_negative_tolerance': 1e-8},
            'slq': {'num_probes': 4, 'lanczos_steps': 8, 'seed': 3, 'breakdown_tolerance': 1e-12},
            'cache': {
                'root': str(tmp_path / 'cache'), 'enabled': True,
                'persist_simplices': True, 'persist_boundaries': True, 'persist_hodge': True,
                'persist_spectral': True, 'persist_heat': True, 'persist_profiles': True,
            },
        },
        'real_validation': {'manifest': str(manifest), 'results_root': str(tmp_path / 'unused'), 'methods': ['Hodge']},
        'real_zero_mode_diagnostics': {
            'results_root': str(tmp_path / 'zero'), 'max_q': 2, 'max_k': 2,
            'store_eigenvalues': True, 'dense_dimension_guard': 100, 'sparse_initial_k': 4,
            'sparse_max_k': 16, 'random_seed': 7,
        },
    }
    config = tmp_path / 'config.json'
    config.write_text(json.dumps(payload), encoding='utf-8')
    run = run_real_zero_mode_diagnostics(payload, config, force=True)

    rows = pd.read_csv(run / 'tables' / 'real_zero_modes_per_network_channel.csv')
    tri_k0 = rows[(rows.label == 'triangle') & (rows.q == 2) & (rows.k == 0)].iloc[0]
    assert bool(tri_k0['count_resolved'])
    assert int(tri_k0['zero_count']) == 1
    assert tri_k0['spectrum_scope'] == 'analytic_compact'

    path_k0 = rows[(rows.label == 'path') & (rows.q == 1) & (rows.k == 0)].iloc[0]
    assert int(path_k0['zero_count']) == 1
    assert path_k0['spectrum_scope'] == 'full'
    assert (run / str(path_k0['spectrum_file'])).exists()

    assert (run / 'tables' / 'real_eigenvalue_index.csv').exists()
    assert (run / 'plots' / 'network_by_q_k0_beta0.png').exists()
