from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import networkx as nx
import numpy as np

from hodge_vnext.cache import ArtifactStore, CACHE_SCHEMA
from hodge_vnext.config import CacheConfig, MethodConfig
from hodge_vnext.graph_io import create_real_record
from hodge_vnext.profile import compute_graph_profile
from validation.real_metrics import hierarchical_clustering, retrieval_metrics


def _graph():
    G=nx.Graph()
    G.add_edges_from((u,v) for u in [0,1,2,3] for v in [0,1,2,3] if u<v)
    G.add_edges_from([(3,4),(3,5),(4,5),(5,6)])
    return G


def _cfg(tmp_path):
    return MethodConfig(
        max_clique_size=None,max_hodge_order=None,min_layer_q=1,
        time_min=.1,time_max=2,time_points=8,normalization="per_node",alpha=1,
        solver="exact",cache=CacheConfig(root=str(tmp_path),enabled=True),
    )


def test_independent_q_layers_survive_refactor(tmp_path):
    cfg=_cfg(tmp_path); rec=create_real_record(_graph(),tmp_path,"g")
    full=compute_graph_profile(rec,cfg)
    cut=compute_graph_profile(rec,replace(cfg,max_clique_size=3))
    assert full.retained_q_layers==[1,2,3]
    assert cut.retained_q_layers==[1,2]
    for q in (1,2):
        for k in range(q+1):
            assert np.array_equal(full.channels[(q,k)],cut.channels[(q,k)])


def test_stage_cache_layout(tmp_path):
    cfg=_cfg(tmp_path); rec=create_real_record(nx.path_graph(5),tmp_path,"p")
    store=ArtifactStore(rec.cache_root,cfg.cache)
    assert CACHE_SCHEMA in str(store.root)
    assert "C1_cliques" in str(store.cliques_file())
    assert "Q2_simplices" in str(store.simplex_file(1,0))
    assert "Q3_boundary" in str(store.boundary_file(1,1))
    assert "Q4_hodge" in str(store.hodge_file(1,0))
    assert "Q6_heat" in str(store.heat_file(1,0,{"solver":"exact"},{"t":1}))
    assert (rec.cache_root/"C0_graph"/"graph.npz").exists()


def test_real_retrieval_metrics_and_clustering():
    D=np.array([[0,.1,1,1.1],[.1,0,1.2,1],[1,1.2,0,.1],[1.1,1,.1,0]],float)
    labels=["a","b","c","d"]; domains=["X","X","Y","Y"]
    c=hierarchical_clustering(D,2)
    summary, per_query, ranks=retrieval_metrics(D,labels,domains)
    assert summary["auroc_same_domain"]==1.0
    assert summary["nn1_accuracy"]==1.0
    assert summary["precision_at_1"]==1.0
    assert len(c.cluster_labels)==4 and len(per_query)==4 and len(ranks)==12


def test_netlsd_official_auto_policy_and_no_slq(tmp_path):
    from benchmarks.baselines.netlsd_local import (
        AUTO_FULL_THRESHOLD,
        AUTO_EIGENVALUES_PER_SIDE,
        paper_auto_strategy,
        heat_signature,
    )
    from benchmarks.baselines.methods import BaselineContext

    assert AUTO_FULL_THRESHOLD == 1024
    assert AUTO_EIGENVALUES_PER_SIDE == 150
    assert paper_auto_strategy(1024) == "full"
    assert paper_auto_strategy(1025) == "two_sided_linear_approx"

    # Small graph: official auto must be numerically identical to full.
    G = nx.path_graph(12)
    times = np.logspace(-2, 2, 25)
    auto = heat_signature(G, timescales=times, eigenvalues="auto")
    full = heat_signature(G, timescales=times, eigenvalues="full")
    assert np.allclose(auto, full, rtol=0, atol=1e-12)

    ctx = BaselineContext(method_cfg=_cfg(tmp_path))
    settings = ctx.method_settings("NetLSD")
    assert settings["eigenvalues"] == "auto"
    assert settings["auto_policy"]["full_eigendecomposition_if_n_le"] == 1024
    assert settings["auto_policy"]["large_graph_eigenvalues_per_side"] == 150
    assert settings["auto_policy"]["large_graph_middle_spectrum"] == "linear_interpolation"
    assert settings["auto_policy"]["uses_slq"] is False


def test_method_comparison_axes_are_vertical_rows():
    import matplotlib.pyplot as plt
    from validation.plotting import _stacked_method_axes

    fig, axes = _stacked_method_axes(5, row_height=2.0, width=8.0, sharex=True)
    try:
        assert len(axes) == 5
        # Same left/right geometry; strictly descending vertical positions.
        boxes = [ax.get_position() for ax in axes]
        assert all(abs(b.x0 - boxes[0].x0) < 1e-12 for b in boxes)
        assert all(abs(b.width - boxes[0].width) < 1e-12 for b in boxes)
        assert all(boxes[i].y0 > boxes[i + 1].y0 for i in range(4))
    finally:
        plt.close(fig)




def test_synthetic_heatmaps_are_separate_square_figures():
    import matplotlib.pyplot as plt
    from validation.plotting import _single_method_distance_heatmap

    coords=np.asarray([0.1,1.0,10.0],dtype=float)
    D=np.asarray([[0.0,0.2,0.5],[0.2,0.0,0.3],[0.5,0.3,0.0]],dtype=float)
    fig,ax=_single_method_distance_heatmap(
        D,coords,"control",{"guide":1.0},
        model="ER",method="Hodge",figure_size=8.0,
    )
    try:
        assert fig.get_figwidth()==fig.get_figheight()==8.0
        assert float(ax.get_aspect())==1.0
        assert len(ax.images)==1
        assert ax.images[0].get_array().shape==(3,3)
    finally:
        plt.close(fig)








def test_runtime_plot_handles_all_cache_hits(tmp_path):
    import pandas as pd
    from validation.real_plotting import plot_runtime_comparison

    runtime=pd.DataFrame({
        "method":["Count","Count","NetLSD","NetLSD"],
        "label":["a","b","a","b"],
        "elapsed_sec":[0.0,0.0,0.0,0.0],
        "cache_hit":[True,True,True,True],
        "wall_sec":[0.001,0.001,0.002,0.002],
    })
    base=tmp_path/"runtime"
    plot_runtime_comparison(runtime,base)
    assert base.with_suffix(".png").exists()
    assert base.with_suffix(".pdf").exists()


def test_real_plot_export_dpi_caps_large_raster_memory():
    import matplotlib.pyplot as plt
    from validation.real_plotting import _effective_png_dpi

    small=plt.figure(figsize=(12,8))
    large=plt.figure(figsize=(30,31))
    try:
        assert _effective_png_dpi(small)==300
        dpi=_effective_png_dpi(large)
        assert 150 <= dpi < 300
        assert 30*31*dpi*dpi <= 36_000_000
    finally:
        plt.close(small); plt.close(large)


def test_five_domain_within_panels_export_without_tight_bbox_pathology(tmp_path):
    from validation.real_plotting import plot_within_domain_panels

    domains=[]; labels=[]
    for d in range(5):
        for j in range(2):
            domains.append(f"D{d}")
            labels.append(f"d{d}_{j}")
    n=len(labels)
    x=np.arange(n,dtype=float)
    D=np.abs(x[:,None]-x[None,:])
    np.fill_diagonal(D,0.0)
    base=tmp_path/"five_domain_panels"
    plot_within_domain_panels(
        D,labels,domains,base,"test",
        plot_options={"within_domain_label_fontsize":10},
    )
    assert base.with_suffix(".png").exists()
    assert base.with_suffix(".pdf").exists()


def test_real_cluster_heatmap_large_labels_and_leaf_alignment(tmp_path):
    from validation.real_metrics import hierarchical_clustering
    from validation.real_plotting import plot_cluster_heatmap

    # 32-network symmetric distance matrix with a unique deterministic structure.
    n = 32
    x = np.arange(n, dtype=float)
    D = np.abs(x[:, None] - x[None, :]) + 0.001 * np.abs((x[:, None] % 4) - (x[None, :] % 4))
    np.fill_diagonal(D, 0.0)
    labels = [f"dataset_{i:02d}" for i in range(n)]
    clustering = hierarchical_clustering(D, 4, method="average")
    base = tmp_path / "cluster"
    plot_cluster_heatmap(
        D,
        labels,
        clustering,
        base,
        "test",
        plot_options={
            "heatmap_label_fontsize": 15,
            "heatmap_x_label_fontsize": 14,
            "cluster_figure_width": 18.0,
            "cluster_figure_height": 18.0,
        },
    )
    assert base.with_suffix(".png").exists()
    assert base.with_suffix(".pdf").exists()
    order = np.loadtxt(tmp_path / "cluster_order.csv", delimiter=",", dtype=str, skiprows=1)
    assert order.shape[0] == 32
