from dataclasses import replace
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from hodge_vnext.config import CacheConfig, MethodConfig
from hodge_vnext.distance import profile_distance
from hodge_vnext.graph_io import canonicalize_graph, create_real_record
from hodge_vnext.hodge import boundary_matrix_from_lists
from hodge_vnext.profile import _normalize, _profile_cache_key, compute_graph_profile

ROOT = Path(__file__).resolve().parents[1]


def test_all_48_graphs_match_manifest_and_preserve_isolates():
    frame = pd.read_csv(ROOT / "data/real_domains/manifests/real_domains_48.csv")
    assert len(frame) == frame.label.nunique() == 48
    assert sorted(frame.domain.value_counts()) == [9, 9, 10, 10, 10]
    for row in frame.itertuples():
        graph = canonicalize_graph(ROOT / row.path, ROOT / row.nodes_path)
        assert graph.number_of_nodes() == row.nodes
        assert graph.number_of_edges() == row.edges
        assert nx.number_connected_components(graph) == row.connected_components


def test_oriented_boundaries_form_a_chain_complex():
    vertices = [(0,), (1,), (2,)]
    edges = [(0, 1), (0, 2), (1, 2)]
    B1 = boundary_matrix_from_lists(vertices, edges, 1)
    B2 = boundary_matrix_from_lists(edges, [(0, 1, 2)], 2)
    assert np.allclose((B1 @ B2).toarray(), 0)


def test_heat_normalization_is_trace_amplitude_not_eigenvalue_scale():
    raw = np.array([4.0, 2.0])
    cfg = MethodConfig(normalization="per_simplex")
    assert np.allclose(_normalize(raw, cfg, 10, 4), [1.0, 0.5])
    assert np.allclose(_normalize(raw, replace(cfg, normalization="per_node"), 10, 4), [0.4, 0.2])


def test_alpha_change_reuses_profile_and_clique_cutoff_reuses_enumeration(tmp_path):
    graph = nx.complete_graph(4)
    graph.add_edges_from([(3, 4), (4, 5)])
    cfg = MethodConfig(
        max_clique_size=4, min_layer_q=0, solver="exact", time_points=8,
        normalization="per_simplex", cache=CacheConfig(root=str(tmp_path)),
    )
    record = create_real_record(graph, tmp_path, "fixture")
    profile = compute_graph_profile(record, cfg)
    alpha_cfg = replace(cfg, alpha=-1)
    assert _profile_cache_key(cfg) == _profile_cache_key(alpha_cfg)
    stamps = {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*") if p.is_file()}
    reused = compute_graph_profile(record, alpha_cfg)
    assert stamps == {p: p.stat().st_mtime_ns for p in stamps}
    assert profile_distance(profile, reused, cfg).total_distance == 0
    clique_file = next(tmp_path.rglob("maximal_cliques.pkl"))
    stamp = clique_file.stat().st_mtime_ns
    smaller = compute_graph_profile(record, replace(cfg, max_clique_size=3))
    assert smaller.retained_q_layers == [1]
    assert clique_file.stat().st_mtime_ns == stamp
