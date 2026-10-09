from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from validation.real_learning import make_cv_splits
from validation.real_learning_nested_plotting import _boxplot_compat
from validation.real_learning_nested import (
    _aggregate_metric_frame,
    _selection_frequency,
    make_repeated_outer_splits,
)


def _real48():
    df = pd.read_csv(Path("data/real_domains/manifests/real_domains_48.csv"))
    return df, df.domain.astype(str).tolist()


def test_repeated_outer_4fold_gives_20_evaluations_and_every_network_tested_5_times():
    df, domains = _real48()
    splits = make_repeated_outer_splits(
        domains,
        strategy="repeated_stratified_kfold",
        repeats=5,
        seed=20260930,
        outer_folds=4,
        require_min_test_per_domain=2,
    )
    assert len(splits) == 20
    test_counts = np.zeros(len(df), dtype=int)
    for split in splits:
        assert set(split.train_idx).isdisjoint(set(split.test_idx))
        assert set(split.train_idx) | set(split.test_idx) == set(range(len(df)))
        vc = pd.Series(np.asarray(domains)[split.test_idx]).value_counts()
        assert int(vc.min()) >= 2
        test_counts[split.test_idx] += 1
    assert np.all(test_counts == 5)


def test_nested_inner_cv_never_touches_outer_test():
    _, domains = _real48()
    outer = make_repeated_outer_splits(
        domains,
        strategy="repeated_stratified_kfold",
        repeats=1,
        seed=20260930,
        outer_folds=4,
        require_min_test_per_domain=2,
    )[0]
    inner = make_cv_splits(outer.train_idx, domains, 5, 20260930)
    outer_test = set(outer.test_idx.tolist())
    outer_train = set(outer.train_idx.tolist())
    for tr, va in inner:
        assert set(tr).isdisjoint(outer_test)
        assert set(va).isdisjoint(outer_test)
        assert set(tr).isdisjoint(set(va))
        assert set(tr) | set(va) == outer_train


def test_repeated_holdout_preserves_two_test_networks_per_domain_for_real48():
    _, domains = _real48()
    splits = make_repeated_outer_splits(
        domains,
        strategy="repeated_stratified_holdout",
        repeats=7,
        seed=20260930,
        test_fraction=0.2,
        require_min_test_per_domain=2,
    )
    assert len(splits) == 7
    for split in splits:
        assert len(split.train_idx) == 38
        assert len(split.test_idx) == 10
        vc = pd.Series(np.asarray(domains)[split.test_idx]).value_counts().to_dict()
        assert set(vc.values()) == {2}


def test_aggregate_metric_frame_reports_resampling_statistics_not_fake_ci():
    frame = pd.DataFrame([
        {"method": "A", "outer_id": "1", "repeat": 1, "fold": 1, "nn1_macro_f1": 0.4},
        {"method": "A", "outer_id": "2", "repeat": 1, "fold": 2, "nn1_macro_f1": 0.8},
        {"method": "B", "outer_id": "1", "repeat": 1, "fold": 1, "nn1_macro_f1": 0.5},
        {"method": "B", "outer_id": "2", "repeat": 1, "fold": 2, "nn1_macro_f1": 0.5},
    ])
    out = _aggregate_metric_frame(frame)
    a = out[out.method == "A"].iloc[0]
    assert a["n_outer_evaluations"] == 2
    assert np.isclose(a["mean_nn1_macro_f1"], 0.6)
    assert a["std_nn1_macro_f1"] > 0
    assert a["q025_nn1_macro_f1"] < a["q975_nn1_macro_f1"]


def test_selection_frequency_counts_candidate_and_marginal_parameter_choices():
    selected = pd.DataFrame([
        {"candidate_id":"a","max_clique_size":3,"max_hodge_order":None,"min_layer_q":0,"normalization":"per_node","alpha":1.0},
        {"candidate_id":"a","max_clique_size":3,"max_hodge_order":None,"min_layer_q":0,"normalization":"per_node","alpha":1.0},
        {"candidate_id":"b","max_clique_size":3,"max_hodge_order":1,"min_layer_q":1,"normalization":"per_simplex","alpha":0.0},
    ])
    cand, param = _selection_frequency(selected)
    assert int(cand.iloc[0]["count"]) == 2
    assert np.isclose(float(cand.iloc[0]["proportion"]), 2/3)
    q = param[(param.parameter == "max_hodge_order") & (param.value == "all")]
    assert int(q.iloc[0]["count"]) == 2


def test_nested_formal_config_keeps_method_and_baseline_contract_and_uses_4x5_outer():
    base = json.loads(Path("experiments/real/comparison.json").read_text(encoding="utf-8"))
    nested = json.loads(Path("experiments/real/nested_cv.json").read_text(encoding="utf-8"))
    assert nested["method"] == base["method"]
    assert nested["baselines"] == base["baselines"]
    ncv = nested["learning"]["nested_cv"]
    assert ncv["outer_strategy"] == "repeated_stratified_kfold"
    assert ncv["outer_repeats"] == 5
    assert ncv["outer_folds"] == 4
    assert ncv["inner_folds"] == 5
    assert ncv["require_min_test_per_domain"] == 2
    assert "solver" not in nested["learning"]["search_space"]
    assert "slq" not in nested["learning"]["search_space"]


def test_boxplot_compat_prefers_tick_labels_for_new_matplotlib_api():
    class NewAxis:
        def boxplot(self, data, *, tick_labels=None, **kwargs):
            assert tick_labels == ["A", "B"]
            assert kwargs["patch_artist"] is True
            return {"boxes": []}

    out = _boxplot_compat(NewAxis(), [[1.0], [2.0]], ["A", "B"], patch_artist=True)
    assert out == {"boxes": []}


def test_boxplot_compat_falls_back_to_labels_for_matplotlib_38_api():
    class OldAxis:
        def __init__(self):
            self.calls = []

        def boxplot(self, data, **kwargs):
            self.calls.append(dict(kwargs))
            if "tick_labels" in kwargs:
                raise TypeError("Axes.boxplot() got an unexpected keyword argument 'tick_labels'")
            assert kwargs["labels"] == ["A", "B"]
            assert kwargs["showmeans"] is True
            return {"boxes": []}

    ax = OldAxis()
    out = _boxplot_compat(ax, [[1.0], [2.0]], ["A", "B"], showmeans=True)
    assert out == {"boxes": []}
    assert len(ax.calls) == 2
