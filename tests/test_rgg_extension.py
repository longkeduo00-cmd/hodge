from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from hodge_vnext.config import CacheConfig, MethodConfig
from hodge_vnext.grid import ParameterGrid
from hodge_vnext.models import (
    GENERATOR_VERSION,
    RGG_GENERATOR_VERSION,
    generate_independent_graph,
    lp_unit_ball_volume,
    rgg_radius_from_lambda,
    synthetic_graph_record,
)
from validation.plotting import _axis_scale
from validation.runner import _run_stem
from validation.theory import model_references


def _cfg(tmp_path: Path) -> MethodConfig:
    return MethodConfig(
        max_clique_size=None,
        max_hodge_order=None,
        min_layer_q=1,
        time_min=0.1,
        time_max=2.0,
        time_points=8,
        normalization="per_node",
        alpha=1.0,
        solver="exact",
        cache=CacheConfig(root=str(tmp_path), enabled=True),
    )


def test_rgg_lambda_radius_mapping_generalizes_dim_and_p():
    assert math.isclose(lp_unit_ball_volume(2, 2.0), math.pi, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(lp_unit_ball_volume(3, 2.0), 4.0 * math.pi / 3.0, rel_tol=0, abs_tol=1e-12)
    n = 1000
    lam = 4.512
    r = rgg_radius_from_lambda(n, lam, dim=2, p_norm=2.0)
    assert math.isclose(r, math.sqrt(lam / (n * math.pi)), rel_tol=0, abs_tol=1e-15)
    assert rgg_radius_from_lambda(n, lam, dim=3, p_norm=2.0) != r


def test_rgg_generation_is_seed_reproducible_and_parameter_specific(tmp_path):
    params = {"lambda": 4.512, "dim": 2, "p_norm": 2.0}
    G1 = generate_independent_graph("RGG", 80, params, 12345)
    G2 = generate_independent_graph("RGG", 80, params, 12345)
    assert set(G1.edges()) == set(G2.edges())

    rec1 = synthetic_graph_record(tmp_path, "RGG", 80, params, 22, 1)
    rec2 = synthetic_graph_record(tmp_path, "RGG", 80, params, 22, 1)
    rec3 = synthetic_graph_record(tmp_path, "RGG", 80, {**params, "lambda": 6.0}, 22, 1)
    assert rec1.graph_hash == rec2.graph_hash
    assert rec1.source["graph_seed"] == rec2.source["graph_seed"]
    assert rec1.source["graph_seed"] != rec3.source["graph_seed"]
    assert math.isclose(
        rec1.source["resolved_params"]["radius"],
        rgg_radius_from_lambda(80, 4.512, 2, 2.0),
    )

    # Adding RGG must not perturb the generator identity used by legacy models.
    assert GENERATOR_VERSION == "vnext-independent-2"
    assert RGG_GENERATOR_VERSION != GENERATOR_VERSION


def test_rgg_run_identity_changes_with_fixed_geometry_and_internal_grid(tmp_path):
    cfg = _cfg(tmp_path)
    grid_a = ParameterGrid(
        values=np.asarray([1.0, 2.0, 4.0]),
        coordinates=np.asarray([1.0, 2.0, 4.0]),
        parameter_name="lambda",
        coordinate_label="lambda",
    )
    grid_b = ParameterGrid(
        values=np.asarray([1.0, 3.0, 4.0]),
        coordinates=np.asarray([1.0, 3.0, 4.0]),
        parameter_name="lambda",
        coordinate_label="lambda",
    )
    base = {"model": "RGG", "n": 100, "repeats": 3, "base_seed": 22, "fixed_params": {"dim": 2, "p_norm": 2.0}}
    stem_a = _run_stem(base, cfg, grid_a)
    stem_grid = _run_stem(base, cfg, grid_b)
    stem_dim = _run_stem({**base, "fixed_params": {"dim": 3, "p_norm": 2.0}}, cfg, grid_a)
    assert stem_a != stem_grid
    assert stem_a != stem_dim
    assert "dim2" in stem_a and "p2" in stem_a and "rggid-" in stem_a


def test_rgg_references_and_plot_scale_are_dynamic():
    refs = model_references("RGG", 1000, {"dim": 2, "p_norm": 2.0})
    assert math.isclose(refs["2D percolation guide"], 4.512)
    assert math.isclose(refs["connectivity guide ln N"], math.log(1000.0))

    refs_3d = model_references("RGG", 500, {"dim": 3, "p_norm": 2.0})
    assert "2D percolation guide" not in refs_3d
    assert "connectivity guide ln N" not in refs_3d

    fig, ax = plt.subplots()
    try:
        _axis_scale(ax, "RGG")
        assert ax.get_xscale() == "log"
    finally:
        plt.close(fig)
