from __future__ import annotations

import numpy as np
import pandas as pd

from validation.real_learning_nested import make_repeated_outer_splits
from validation.real_q_weight_learning import _split_inventory_tables


def _real48_like_domains() -> list[str]:
    counts = {
        "Biological": 10,
        "Collaboration": 9,
        "Infrastructure_Transportation": 10,
        "Interaction_Communication": 10,
        "Social_Cultural": 9,
    }
    out: list[str] = []
    for domain, n in counts.items():
        out.extend([domain] * n)
    return out


def test_ten_repeated_holdouts_are_38_10_and_stratified_two_per_domain():
    domains = _real48_like_domains()
    splits = make_repeated_outer_splits(
        domains,
        strategy="repeated_stratified_holdout",
        repeats=10,
        seed=20260930,
        test_fraction=0.2,
        require_min_test_per_domain=2,
    )
    assert len(splits) == 10
    y = np.asarray(domains)
    for split in splits:
        assert len(split.train_idx) == 38
        assert len(split.test_idx) == 10
        counts = pd.Series(y[split.test_idx]).value_counts()
        assert set(counts.tolist()) == {2}
        assert len(counts) == 5


def test_ten_repeated_holdouts_have_unique_test_sets_and_are_auditable():
    domains = _real48_like_domains()
    labels = [f"g{i:02d}" for i in range(len(domains))]
    splits = make_repeated_outer_splits(
        domains,
        strategy="repeated_stratified_holdout",
        repeats=10,
        seed=20260930,
        test_fraction=0.2,
        require_min_test_per_domain=2,
    )
    signatures = [tuple(split.test_idx.tolist()) for split in splits]
    assert len(set(signatures)) == 10

    inventory, overlap = _split_inventory_tables(splits, labels, domains)
    assert len(inventory) == 10
    assert len(overlap) == 45
    assert not bool(overlap["identical_test_set"].any())
    assert (inventory["train_size"] == 38).all()
    assert (inventory["test_size"] == 10).all()
    assert inventory["test_domain_counts"].str.count(":2").eq(5).all()


def test_repeated_holdout_test_coverage_sums_to_100_assignments():
    domains = _real48_like_domains()
    splits = make_repeated_outer_splits(
        domains,
        strategy="repeated_stratified_holdout",
        repeats=10,
        seed=20260930,
        test_fraction=0.2,
        require_min_test_per_domain=2,
    )
    counts = np.zeros(48, dtype=int)
    for split in splits:
        counts[split.test_idx] += 1
    assert int(counts.sum()) == 100
    # Repeated holdout is deliberately not balanced like repeated K-fold.
    assert counts.min() >= 0
    assert counts.max() <= 10
