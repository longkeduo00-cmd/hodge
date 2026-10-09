from __future__ import annotations

import json
from pathlib import Path
import networkx as nx

from hodge_vnext.cache import ArtifactStore
from hodge_vnext.cliques import enumerate_maximal_clique_layers
from hodge_vnext.config import CacheConfig
from hodge_vnext.graph_io import create_real_record
from hodge_vnext.hodge import get_hodge_laplacian
from hodge_vnext.models import synthetic_graph_record
from validation.zero_mode_diagnostics import ZeroModeSettings, diagnose_hodge_zero_modes


def _diag_for_graph(G: nx.Graph, q: int, k: int, tmp_path: Path):
    rec = create_real_record(G, tmp_path, f'g_q{q}_k{k}')
    store = ArtifactStore(rec.cache_root, CacheConfig(root=str(tmp_path)))
    layers = enumerate_maximal_clique_layers(G, store)
    L = get_hodge_laplacian(q, k, layers[q], store)
    settings = ZeroModeSettings(dense_dimension_guard=1024, store_eigenvalues=True)
    return diagnose_hodge_zero_modes(L, rec.graph_hash, q, k, store, settings)


def test_cycle_c4_q1_has_beta0_one_and_beta1_one(tmp_path):
    G = nx.cycle_graph(4)
    s0, _ = _diag_for_graph(G, 1, 0, tmp_path)
    s1, _ = _diag_for_graph(G, 1, 1, tmp_path)
    assert s0['zero_count'] == 1
    assert s1['zero_count'] == 1
    assert s0['count_resolved'] and s1['count_resolved']


def test_single_triangle_q2_has_only_beta0_zero_mode(tmp_path):
    G = nx.complete_graph(3)
    vals = []
    for k in range(3):
        s, _ = _diag_for_graph(G, 2, k, tmp_path)
        vals.append(s['zero_count'])
    assert vals == [1, 0, 0]


def test_two_disjoint_triangles_q2_beta0_two(tmp_path):
    G = nx.disjoint_union(nx.complete_graph(3), nx.complete_graph(3))
    s, _ = _diag_for_graph(G, 2, 0, tmp_path)
    assert s['zero_count'] == 2


def test_synthetic_seed_contract_is_deterministic(tmp_path):
    params = {'p': 0.1}
    a = synthetic_graph_record(tmp_path, 'ER', 30, params, 22, 1)
    b = synthetic_graph_record(tmp_path, 'ER', 30, params, 22, 1)
    assert a.graph_hash == b.graph_hash
    assert a.source['graph_seed'] == b.source['graph_seed']


def test_formal_synthetic_zero_mode_config_contract():
    path = Path(__file__).resolve().parents[1] / 'experiments' / 'synthetic' / 'ZERO_MODES' / 'synthetic_zero_modes_formal.json'
    p = json.loads(path.read_text(encoding='utf-8'))
    e = p['synthetic_zero_modes']
    assert e['n'] == 200
    assert e['target_mean_degrees'] == [6,10,14]
    assert e['repeats'] == 10
    assert list(e['models']) == ['ER','WS','SBM']
    assert e['max_q'] == 9
    assert e['max_k'] == 9
    assert p['method']['max_clique_size'] == 10


def test_spectrum_report_config_matches_real_style_k0():
    path = Path(__file__).resolve().parents[1] / 'experiments' / 'synthetic' / 'ZERO_MODES' / 'synthetic_zero_modes_formal.json'
    p = json.loads(path.read_text(encoding='utf-8'))
    report = p['synthetic_zero_modes']['spectrum_reports']
    assert report['enabled'] is True
    assert report['k_orders'] == [0]
    assert report['q_min'] == 0
    assert report['q_max'] == 9
    assert report['columns'] == 5


def test_zero_count_statistics_reports_matrix_order():
    import pandas as pd
    from validation.synthetic_zero_modes_plotting import zero_count_statistics
    df = pd.DataFrame([
        {'model':'ER','target_mean_degree':6.0,'q':1,'k':0,'count_resolved':True,'zero_count':2,'zero_fraction':0.2,'smallest_positive_eigenvalue':0.5},
        {'model':'ER','target_mean_degree':6.0,'q':1,'k':0,'count_resolved':True,'zero_count':4,'zero_fraction':0.4,'smallest_positive_eigenvalue':0.4},
        {'model':'ER','target_mean_degree':6.0,'q':2,'k':1,'count_resolved':True,'zero_count':3,'zero_fraction':0.3,'smallest_positive_eigenvalue':0.2},
    ])
    out = zero_count_statistics(df)
    r = out[(out['q']==1) & (out['k']==0)].iloc[0]
    assert r['matrix'] == 'L_0^(1)'
    assert r['mean_zero_count'] == 3.0
    import pytest
    assert r['median_zero_fraction'] == pytest.approx(0.3)


def test_single_graph_spectrum_report_writes_png_pdf(tmp_path):
    import numpy as np
    import pandas as pd
    from validation.synthetic_zero_modes_plotting import plot_single_graph_spectrum_report
    run = tmp_path / 'run'
    eig_dir = run / 'arrays' / 'eigenvalues'
    eig_dir.mkdir(parents=True)
    rows=[]
    for q, vals in [(1, np.array([0.,0.,1.,2.,3.])), (2, np.array([0.,0.5,1.5,4.]))]:
        f=eig_dir / f'g_q{q}.npy'
        np.save(f, vals)
        z=int(np.sum(vals<=1e-10)); pos=vals[vals>1e-10]
        rows.append({
            'graph_id':'g','model':'ER','target_mean_degree':6.0,'repeat':1,'q':q,'k':0,
            'dimension':len(vals),'zero_count':z,'zero_fraction':z/len(vals),'count_resolved':True,
            'smallest_positive_eigenvalue':float(pos.min()) if len(pos) else np.nan,
            'spectrum_scope':'full','spectrum_file':str(f.relative_to(run)),
            'analytic_multiplicity':None,
        })
    df=pd.DataFrame(rows)
    stem=run/'plots'/'report'
    status=plot_single_graph_spectrum_report(df,run,stem,k=0,q_values=[1,2])
    assert status['plotted'] is True
    assert status['full_spectrum_panels'] == 2
    assert stem.with_suffix('.png').exists()
    assert stem.with_suffix('.pdf').exists()
