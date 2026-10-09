from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from validation.q_weight_learning import distance_matrix_from_beta
from validation.q_weight_scale_cv import (
    apply_layer_scales_to_squared_distances,
    estimate_train_only_layer_scales,
    fit_single_alpha_train_only,
    single_alpha_beta,
)
from validation.real_learning_nested import make_repeated_outer_splits, _outer_split_membership_table
from validation.real_q_weight_scale_cv import _repeat_oof_metrics, _repeat_split_audit


def _domains() -> list[str]:
    counts = [10, 9, 10, 10, 9]
    names = ["A", "B", "C", "D", "E"]
    out: list[str] = []
    for name, n in zip(names, counts):
        out.extend([name] * n)
    return out


def _feature_layers(features: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    n = len(features[0])
    C = np.zeros((n, n, len(features)), dtype=float)
    for q, x in enumerate(features):
        x = np.asarray(x, dtype=float)
        C[:, :, q] = (x[:, None] - x[None, :]) ** 2
    q_star = np.full((n, n), len(features) - 1, dtype=np.int32)
    np.fill_diagonal(q_star, -1)
    return C, q_star


def test_repeated_5fold_gives_exactly_one_test_per_repeat_and_ten_total():
    domains = _domains()
    splits = make_repeated_outer_splits(
        domains,
        strategy="repeated_stratified_kfold",
        repeats=10,
        seed=42,
        outer_folds=5,
        require_min_test_per_domain=1,
    )
    assert len(splits) == 50
    df = pd.DataFrame({"label": [f"g{i}" for i in range(48)], "domain": domains})
    membership = _outer_split_membership_table(df, splits)
    audit = _repeat_split_audit(membership, 48, 10, 5)
    assert audit["all_networks_tested_exactly_once"].all()
    counts = membership.assign(test=(membership.partition == "test").astype(int)).groupby("index")["test"].sum()
    assert np.all(counts.to_numpy() == 10)


def test_train_only_scale_ignores_test_related_values():
    C, q_star = _feature_layers([
        np.array([0.0, 1.0, 2.0, 20.0]),
        np.array([0.0, 2.0, 4.0, 40.0]),
    ])
    train = np.array([0, 1, 2])
    s1 = estimate_train_only_layer_scales(C, q_star, train)
    C2 = C.copy()
    C2[3, :, :] *= 1e6
    C2[:, 3, :] = C2[3, :, :]
    C2[3, 3, :] = 0.0
    s2 = estimate_train_only_layer_scales(C2, q_star, train)
    assert np.allclose(s1.distance_scale, s2.distance_scale)
    assert np.allclose(s1.squared_denominator, s2.squared_denominator)


def test_scaled_squared_contribution_matches_rms_distance_definition():
    C, q_star = _feature_layers([
        np.array([0.0, 1.0, 3.0]),
        np.array([0.0, 2.0, 5.0]),
    ])
    train = np.array([0, 1, 2])
    scale = estimate_train_only_layer_scales(C, q_star, train, eps=1e-8)
    out = apply_layer_scales_to_squared_distances(C, scale)
    for q in range(2):
        vals = np.array([C[0, 1, q], C[0, 2, q], C[1, 2, q]])
        s = np.sqrt(np.mean(vals))
        expected = C[:, :, q] / (s + 1e-8) ** 2
        assert np.allclose(out[:, :, q], expected)


def test_single_alpha_zero_is_uniform_pair_adaptive():
    C, q_star = _feature_layers([
        np.array([0.0, 1.0, 2.0]),
        np.array([0.0, 3.0, 1.0]),
        np.array([0.0, 5.0, 2.0]),
    ])
    q_star[0, 1] = q_star[1, 0] = 1
    D0 = distance_matrix_from_beta(single_alpha_beta(0.0, 3), C, q_star)
    Du = distance_matrix_from_beta(np.zeros(3), C, q_star)
    assert np.allclose(D0, Du, atol=1e-12, rtol=1e-12)


def test_single_alpha_fit_is_train_only():
    C, q_star = _feature_layers([
        np.array([0.0, 0.1, 4.0, 4.1, 20.0]),
        np.array([0.0, 3.0, 0.2, 3.2, 30.0]),
        np.array([0.0, 2.0, 4.0, 6.0, 40.0]),
    ])
    labels = np.array(["A", "A", "B", "B", "C"])
    train = np.array([0, 1, 2, 3])
    fit1 = fit_single_alpha_train_only(
        C, q_star, train, labels,
        coarse_min=-3, coarse_max=3, coarse_step=1,
        fine_half_width=1, fine_step=0.25,
    )
    C2 = C.copy()
    C2[4, :, :] *= 1e5
    C2[:, 4, :] = C2[4, :, :]
    C2[4, 4, :] = 0.0
    fit2 = fit_single_alpha_train_only(
        C2, q_star, train, labels,
        coarse_min=-3, coarse_max=3, coarse_step=1,
        fine_half_width=1, fine_step=0.25,
    )
    assert fit1.alpha == fit2.alpha
    assert np.isclose(fit1.best_loss, fit2.best_loss)


def test_repeat_oof_metrics_requires_one_prediction_per_network():
    qdf = pd.DataFrame({
        "query": ["a", "b", "c", "d"],
        "domain": ["A", "A", "B", "B"],
        "nearest_neighbor_domain": ["A", "B", "B", "B"],
        "nearest_neighbor_correct": [1, 0, 1, 1],
        "average_precision": [1.0, 0.5, 1.0, 0.8],
        "ndcg_at_7": [1.0, 0.6, 1.0, 0.9],
        "mean_same_domain_distance": [1.0, 1.2, 1.0, 1.1],
        "mean_other_domain_distance": [2.0, 1.8, 2.2, 2.0],
        "precision_at_1": [1, 0, 1, 1], "recall_at_1": [1, 0, 1, 1],
        "precision_at_3": [0.5, 0.5, 0.5, 0.5], "recall_at_3": [1, 1, 1, 1],
        "precision_at_5": [0.5, 0.5, 0.5, 0.5], "recall_at_5": [1, 1, 1, 1],
        "precision_at_7": [0.5, 0.5, 0.5, 0.5], "recall_at_7": [1, 1, 1, 1],
    })
    rdf = pd.DataFrame({
        "distance": [0.1, 0.3, 0.2, 0.4, 0.1, 0.5, 0.1, 0.6],
        "relevant": [1, 0, 0, 1, 1, 0, 1, 0],
    })
    metrics = _repeat_oof_metrics(qdf, rdf, 4)
    assert np.isclose(metrics["nn1_accuracy"], 0.75)
    assert np.isfinite(metrics["auroc_same_domain"])


def test_formal_config_is_10x5_and_no_regularization():
    path = Path(__file__).resolve().parents[1] / "experiments" / "real" / "q_scale_cv.json"
    cfg = json.loads(path.read_text(encoding="utf-8"))
    outer = cfg["learning"]["outer_cv"]
    assert outer["strategy"] == "repeated_stratified_kfold"
    assert outer["repeats"] == 10
    assert outer["folds"] == 5
    assert cfg["method"]["max_clique_size"] == 10
    assert cfg["method"]["max_hodge_order"] is None
    assert cfg["method"]["min_layer_q"] == 0
    assert cfg["method"]["normalization"] == "per_simplex"
    assert "regularization" not in cfg["learning"]
