from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import validation.real_alpha_sweep as mod


def test_real_alpha_sweep_collects_full_run_outputs_and_audits_baselines(tmp_path: Path, monkeypatch):
    g1 = tmp_path / 'a.edgelist'; g1.write_text('0 1\n1 2\n', encoding='utf-8')
    g2 = tmp_path / 'b.edgelist'; g2.write_text('0 1\n0 2\n', encoding='utf-8')
    manifest = tmp_path / 'manifest.csv'
    pd.DataFrame([
        {'label': 'a', 'domain': 'D1', 'path': str(g1), 'nodes_path': ''},
        {'label': 'b', 'domain': 'D2', 'path': str(g2), 'nodes_path': ''},
    ]).to_csv(manifest, index=False)

    payload = {
        'method': {
            'max_clique_size': 3, 'max_hodge_order': None, 'min_layer_q': 0,
            'time_min': 0.01, 'time_max': 100.0, 'time_points': 8,
            'normalization': 'per_simplex', 'alpha': 1.0, 'solver': 'exact',
            'exact': {'dimension_guard': None, 'eigen_zero_tolerance': 1e-10, 'eigen_negative_tolerance': 1e-8},
            'slq': {'num_probes': 4, 'lanczos_steps': 8, 'seed': 3, 'breakdown_tolerance': 1e-12},
            'cache': {'root': str(tmp_path / 'cache'), 'enabled': True, 'persist_simplices': True,
                      'persist_boundaries': True, 'persist_hodge': True, 'persist_spectral': True,
                      'persist_heat': True, 'persist_profiles': True},
        },
        'real_validation': {'manifest': str(manifest), 'results_root': str(tmp_path / 'inner'), 'methods': ['Hodge', 'DMeasure']},
        'real_alpha_sweep': {'results_root': str(tmp_path / 'sweep'), 'alphas': [1.0, 0.0], 'reference_alpha': 1.0},
        'baselines': {},
    }
    config = tmp_path / 'config.json'; config.write_text(json.dumps(payload), encoding='utf-8')

    def fake_real_runner(local, config_path, force=False):
        alpha = float(local['method']['alpha'])
        expected_slug = mod._alpha_slug(alpha)
        assert local['real_validation']['run_name'] == expected_slug
        out = Path(local['real_validation']['results_root']) / expected_slug
        (out / 'tables').mkdir(parents=True, exist_ok=True)
        (out / 'arrays').mkdir(parents=True, exist_ok=True)
        (out / 'plots' / 'Hodge').mkdir(parents=True, exist_ok=True)
        # Hodge changes; DMeasure is invariant across alpha.
        h = 0.7 + 0.1 * alpha
        metric_rows = [
            {'method': 'Hodge', 'auroc_same_domain': h, 'nn1_accuracy': h, 'nn1_macro_f1': h,
             'precision_at_1': h, 'mAP': h, 'ndcg_at_7': h},
        ]
        domain_rows = [
            {'method': 'Hodge', 'domain': 'D1', 'num_queries': 1, 'nn1_accuracy': h, 'mAP': h, 'ndcg_at_7': h},
            {'method': 'Hodge', 'domain': 'D2', 'num_queries': 1, 'nn1_accuracy': h, 'mAP': h, 'ndcg_at_7': h},
        ]
        if 'DMeasure' in local['real_validation'].get('methods', []):
            metric_rows.append({'method': 'DMeasure', 'auroc_same_domain': 0.6, 'nn1_accuracy': 0.5, 'nn1_macro_f1': 0.5,
                                'precision_at_1': 0.5, 'mAP': 0.55, 'ndcg_at_7': 0.57})
            domain_rows.extend([
                {'method': 'DMeasure', 'domain': 'D1', 'num_queries': 1, 'nn1_accuracy': .5, 'mAP': .55, 'ndcg_at_7': .57},
                {'method': 'DMeasure', 'domain': 'D2', 'num_queries': 1, 'nn1_accuracy': .5, 'mAP': .55, 'ndcg_at_7': .57},
            ])
        pd.DataFrame(metric_rows).to_csv(out / 'tables' / 'real_metrics_summary.csv', index=False)
        pd.DataFrame(domain_rows).to_csv(out / 'tables' / 'domain_metrics_all_methods.csv', index=False)
        D = np.asarray([[0.0, 1.0 + alpha], [1.0 + alpha, 0.0]])
        np.save(out / 'arrays' / 'distance_matrix_Hodge.npy', D)
        return out

    monkeypatch.setattr(mod, 'run_real_domain_validation', fake_real_runner)
    run = mod.run_real_alpha_sweep(payload, config, force=True)

    hodge = pd.read_csv(run / 'tables' / 'hodge_real_alpha_metrics.csv')
    assert set(hodge['alpha']) == {0.0, 1.0}
    audit = pd.read_csv(run / 'tables' / 'baseline_invariance_audit.csv')
    assert audit['passed'].astype(bool).all()
    index = pd.read_csv(run / 'tables' / 'alpha_run_index.csv')
    assert len(index) == 2
    assert set(index['run_name']) == {'a1', 'a0'}
    assert all(Path(x).name in {'a1', 'a0'} for x in index['run_dir'])
    assert (run / 'plots' / 'hodge_alpha_auroc_same_domain.png').exists()
