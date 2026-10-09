from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hodge_vnext.models import GENERATOR_VERSION, RGG_GENERATOR_VERSION, synthetic_graph_record
from validation.model_discrimination import (
    FINE_STATE_ORDER,
    rgg_radius_for_expected_degree,
    rgg_square_pair_probability,
    _cell_distance_matrix,
    classical_mds_2d,
    _retrieval_metrics,
    build_parameter_cells,
)


def test_rgg_finite_square_expected_degree_roundtrip():
    n = 500
    for kappa in (4.0, 8.0, 14.0):
        r = rgg_radius_for_expected_degree(n, kappa)
        recovered = (n - 1) * rgg_square_pair_probability(r)
        assert abs(recovered - kappa) < 1e-10
        assert 0 < r < 1


def test_formal_parameter_cells_are_6_by_6_and_density_matched():
    cells = build_parameter_cells(1000, [4, 6, 8, 10, 12, 14], [1e-3, 1e-2, 1e-1, 1.0])
    assert len(cells) == 36
    assert tuple(cells.mechanism_state.drop_duplicates()) == FINE_STATE_ORDER
    assert set(cells.groupby("mechanism_state").size()) == {6}
    assert np.allclose(cells[cells.model == "ER"].er_p.to_numpy() * 999.0, [4, 6, 8, 10, 12, 14])
    assert np.all(cells[cells.model == "WS"].ws_degree.to_numpy(dtype=float) % 2 == 0)


def test_existing_generators_keep_old_version_and_rgg_is_separate(tmp_path: Path):
    er = synthetic_graph_record(tmp_path, "ER", 30, {"p": 0.1}, 22, 1)
    rgg = synthetic_graph_record(tmp_path, "RGG", 30, {"lambda": 4.0, "dim": 2, "p_norm": 2.0}, 22, 1)
    assert er.source["generator_version"] == GENERATOR_VERSION
    assert rgg.source["generator_version"] == RGG_GENERATOR_VERSION
    assert er.source["graph_seed"] != rgg.source["graph_seed"]


def test_cell_matrix_diagonal_is_same_cell_noise_not_zero():
    D = np.array([
        [0, 2, 5, 6],
        [2, 0, 7, 8],
        [5, 7, 0, 4],
        [6, 8, 4, 0],
    ], dtype=float)
    cells = np.array([0, 0, 1, 1])
    C, counts = _cell_distance_matrix(D, cells, 2)
    assert C[0, 0] == 2
    assert C[1, 1] == 4
    assert C[0, 1] == 6.5
    assert counts[0, 0] == 1
    assert counts[0, 1] == 4


def test_matched_and_cross_density_candidate_protocols():
    # Four mechanism classes, two densities, two repeats each -> 16 graphs.
    states = np.array([s for k in (4.0, 8.0) for s in ("ER", "RGG", "WS-a", "WS-b") for _ in range(2)])
    kappas = np.array([k for k in (4.0, 8.0) for _s in range(4) for _ in range(2)], dtype=float)
    ids = np.array([f"g{i}" for i in range(len(states))])
    # Distance primarily separates mechanism, with tiny density penalty.
    D = np.zeros((len(states), len(states)), dtype=float)
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            d = (0.1 if states[i] == states[j] else 3.0) + (0.01 if kappas[i] != kappas[j] else 0.0)
            D[i, j] = D[j, i] = d
    matched, _, _ = _retrieval_metrics(D, ids, states, kappas, "matched_density", 7)
    cross, _, _ = _retrieval_metrics(D, ids, states, kappas, "cross_density", 7)
    assert matched["balanced_accuracy"] == 1.0
    assert cross["balanced_accuracy"] == 1.0


def test_frozen_config_contract():
    path = Path("experiments/synthetic/ER_WS_RGG/er_ws_rgg_matched.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["method"]["alpha"] == 0.0
    assert payload["method"]["normalization"] == "per_simplex"
    assert payload["method"]["min_layer_q"] == 0
    assert payload["baselines"]["count_min_layer_q"] == 0
    assert payload["validation"]["target_degrees"] == [4, 6, 8, 10, 12, 14]
    assert payload["validation"]["ws_betas"] == [0.001, 0.01, 0.1, 1.0]



def test_classical_mds_2d_recovers_planar_euclidean_geometry():
    points = np.array([
        [0.0, 0.0],
        [2.1, 0.2],
        [0.3, 1.7],
        [2.4, 1.1],
        [1.1, 0.45],
    ])
    D = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    coords, diag = classical_mds_2d(D)
    recovered = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    assert np.allclose(recovered, D, atol=1e-10)
    assert diag["positive_eigenvalue_fraction_2d"] > 1.0 - 1e-12
    assert diag["distance_spearman_2d"] > 1.0 - 1e-12
    assert diag["normalized_stress_2d"] < 1e-10
    assert diag["diagonal_policy"] == "native_zero"


def test_cell_mds_zeroes_noise_diagonal_only_for_embedding():
    # The cell-distance diagonal is within-cell noise, not a self-distance.
    D = np.array([
        [0.4, 2.0, 3.0],
        [2.0, 0.6, 2.5],
        [3.0, 2.5, 0.5],
    ])
    original = D.copy()
    coords, diag = classical_mds_2d(D, zero_diagonal=True)
    assert coords.shape == (3, 2)
    assert np.all(np.isfinite(coords))
    assert np.array_equal(D, original)  # helper must not mutate raw cell matrix
    assert diag["diagonal_policy"] == "zeroed_for_embedding"
