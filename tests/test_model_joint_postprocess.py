from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from validation.model_joint_postprocess import postprocess_joint_run, _discover_methods


def _make_fake_run(tmp_path: Path) -> Path:
    run = tmp_path / 'joint_run'
    (run / 'arrays').mkdir(parents=True)
    (run / 'tables').mkdir(parents=True)
    (run / 'plots').mkdir(parents=True)
    params = pd.DataFrame([
        {'cell_index': 0, 'cell_label': 'ER-N50-k4', 'mechanism_state': 'ER', 'network_size': 50, 'target_mean_degree': 4.0},
        {'cell_index': 1, 'cell_label': 'RGG-N50-k6', 'mechanism_state': 'RGG', 'network_size': 50, 'target_mean_degree': 6.0},
        {'cell_index': 2, 'cell_label': 'WS-beta1e-3-N100-k8', 'mechanism_state': 'WS-beta1e-3', 'network_size': 100, 'target_mean_degree': 8.0},
    ])
    params.to_csv(run / 'tables' / 'parameter_cells.csv', index=False)
    pd.DataFrame({'graph_index': [0, 1, 2], 'graph_hash': ['g0', 'g1', 'g2']}).to_csv(run / 'tables' / 'graph_index.csv', index=False)
    rows=[]
    protocols=['matched_N_k','cross_N_only','cross_k_only','cross_N_k','global']
    for method,base in [('Hodge',0.8),('Count',0.4)]:
        for i,p in enumerate(protocols):
            rows.append({'method':method,'protocol':p,'balanced_accuracy':base-.01*i,'macro_f1':base-.02*i,
                         'same_class_auroc':base-.03*i,'mean_average_precision':base-.015*i,'ndcg_at_5':base-.025*i})
    pd.DataFrame(rows).to_csv(run/'tables'/'metrics_all_methods.csv',index=False)
    D=np.array([[0.,1.,2.],[1.,0.,1.5],[2.,1.5,0.]])
    np.save(run/'arrays'/'cell_distance_Hodge.npy',D)
    np.save(run/'arrays'/'cell_distance_Count.npy',D*.5)
    np.save(run/'arrays'/'cell_distance_std_Hodge.npy',np.zeros_like(D))
    np.save(run/'arrays'/'cell_distance_std_Count.npy',np.zeros_like(D))
    np.save(run/'arrays'/'cell_pair_count_Hodge.npy',np.ones_like(D))
    np.save(run/'arrays'/'graph_distance_Hodge.npy',D)
    return run


def test_discover_methods_ignores_std_artifacts(tmp_path: Path):
    run=_make_fake_run(tmp_path)
    methods=_discover_methods(run,pd.read_csv(run/'tables'/'metrics_all_methods.csv'))
    assert methods == ['Hodge','Count']


def test_postprocess_joint_run_exports_csv_and_plots(tmp_path: Path):
    run=_make_fake_run(tmp_path)
    manifest=postprocess_joint_run(run)
    assert manifest['exported_csv_count'] >= 4
    assert (run/'tables'/'distance_matrix_Hodge.csv').exists()
    assert (run/'plots'/'global_auroc_method_comparison.png').exists()
    assert not (run/'plots'/'metric_same_class_auroc_by_protocol.png').exists()
    assert (run/'plots'/'joint_mds_cell_Hodge.png').exists()
