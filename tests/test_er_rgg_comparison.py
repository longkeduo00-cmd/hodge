from __future__ import annotations

import math

import numpy as np

from hodge_vnext.grid import build_parameter_grid
from validation.er_rgg_comparison import (
    matched_rgg_lambda_from_er_c,
    square_euclidean_connection_probability,
)


def test_explicit_pairs_grid_preserves_generator_and_control_coordinates():
    exp = {
        "n": 100,
        "parameter_name": "lambda",
        "coordinate_label": "matched c",
        "parameter_values": [1.1, 2.2, 3.3],
        "control_values": [1.0, 2.0, 3.0],
        "grid": {"mode": "explicit_pairs"},
    }
    grid = build_parameter_grid(exp)
    assert np.allclose(grid.values, [1.1, 2.2, 3.3])
    assert np.allclose(grid.coordinates, [1.0, 2.0, 3.0])
    assert grid.parameter_name == "lambda"


def test_finite_square_er_rgg_expected_degree_matching():
    for n in (100, 500, 1000):
        for c in (0.25, 1.0, 4.512, 6.9, 15.0):
            lam, radius = matched_rgg_lambda_from_er_c(n, c)
            p = square_euclidean_connection_probability(radius)
            assert math.isclose(p, c / n, rel_tol=0.0, abs_tol=2e-13)
            assert math.isclose((n - 1) * p, (n - 1) * c / n, rel_tol=0.0, abs_tol=2e-11)
            assert math.isclose(lam, n * math.pi * radius * radius, rel_tol=0.0, abs_tol=1e-12)


def test_finite_square_matching_rejects_nonstandard_geometry():
    try:
        matched_rgg_lambda_from_er_c(100, 3.0, dim=3, p_norm=2.0)
    except ValueError as exc:
        assert "dim=2" in str(exc)
    else:
        raise AssertionError("non-2D geometry must be rejected for exact finite-square matching")
