from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .style import FIG_DPI, METHOD_COLORS, apply_style

apply_style()


def _save(fig, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _color(method: str):
    if method == "HodgeLearned":
        return METHOD_COLORS.get("Hodge")
    if method == "HodgeFrozen":
        return "0.45"
    return METHOD_COLORS.get(method)


def _boxplot_compat(ax, data, labels, **kwargs):
    """Call Axes.boxplot across Matplotlib 3.8 and >=3.9/3.11 APIs.

    Matplotlib renamed ``labels`` to ``tick_labels`` in 3.9 and removes the
    legacy keyword in newer releases.  The project currently supports
    matplotlib>=3.8, so use the new spelling first and fall back only when an
    older Matplotlib rejects ``tick_labels``.
    """
    try:
        return ax.boxplot(data, tick_labels=labels, **kwargs)
    except TypeError as exc:
        if "tick_labels" not in str(exc):
            raise
        return ax.boxplot(data, labels=labels, **kwargs)


def plot_nested_metric_summary(summary: pd.DataFrame, base: Path, title: str) -> None:
    metrics = [
        ("auroc_same_domain", "AUROC"),
        ("nn1_accuracy", "1-NN Acc."),
        ("nn1_macro_f1", "Macro-F1"),
        ("precision_at_1", "P@1"),
        ("mAP", "mAP"),
        ("ndcg_at_7", "NDCG@7"),
    ]
    methods = summary["method"].astype(str).tolist()
    x = np.arange(len(metrics), dtype=float)
    width = 0.82 / max(1, len(methods))
    fig, ax = plt.subplots(figsize=(20, 11.5))
    for mi, method in enumerate(methods):
        row = summary.iloc[mi]
        means = np.asarray([float(row[f"mean_{m}"]) for m, _ in metrics])
        stds = np.asarray([float(row[f"std_{m}"]) for m, _ in metrics])
        offset = (mi - (len(methods) - 1) / 2) * width
        ax.bar(
            x + offset, means, width=width * 0.92, yerr=stds, capsize=4,
            label=method, color=_color(method), ecolor="black",
        )
    ax.set_xticks(x, [lab for _, lab in metrics])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Outer-test score (mean ± SD across resamples)")
    ax.set_title(title, fontweight="bold")
    ax.legend(frameon=False, ncol=min(3, len(methods)), loc="upper center")
    ax.text(
        0.01, 0.01,
        "Outer evaluations are correlated resamples; error bars describe observed split-to-split variability, not iid uncertainty.",
        transform=ax.transAxes, fontsize=11, va="bottom",
    )
    fig.tight_layout()
    _save(fig, base)


def plot_outer_metric_distributions(frame: pd.DataFrame, base: Path) -> None:
    metrics = [
        ("auroc_same_domain", "AUROC"),
        ("nn1_macro_f1", "Macro-F1"),
        ("mAP", "mAP"),
        ("ndcg_at_7", "NDCG@7"),
    ]
    methods = list(dict.fromkeys(frame["method"].astype(str)))
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    for ax, (metric, label) in zip(axes.flat, metrics):
        data = [frame.loc[frame.method == method, metric].to_numpy(float) for method in methods]
        bp = _boxplot_compat(ax, data, methods, patch_artist=True, showmeans=True)
        for patch, method in zip(bp["boxes"], methods):
            patch.set_facecolor(_color(method) or "0.8")
            patch.set_alpha(0.75)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(label)
        ax.set_title(f"Outer-resample distribution: {label}", fontweight="bold")
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Repeated nested CV generalization variability", fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, base)


def plot_candidate_selection_frequency(freq: pd.DataFrame, base: Path, top_n: int = 20) -> None:
    shown = freq.head(int(top_n)).copy().iloc[::-1]
    labels = []
    for r in shown.itertuples(index=False):
        S = "all" if pd.isna(r.max_clique_size) else str(int(r.max_clique_size))
        Q = "all" if pd.isna(r.max_hodge_order) else str(int(r.max_hodge_order))
        labels.append(f"S={S}, Q={Q}, qmin={int(r.min_layer_q)}, {r.normalization}, a={float(r.alpha):g}")
    y = np.arange(len(shown))
    fig, ax = plt.subplots(figsize=(18, max(9, 0.55 * len(shown) + 3)))
    bars = ax.barh(y, shown["proportion"].to_numpy(float))
    for b, n in zip(bars, shown["count"].to_numpy(int)):
        ax.text(b.get_width() + 0.006, b.get_y() + b.get_height()/2, str(int(n)), va="center", fontsize=11)
    ax.set_yticks(y, labels, fontsize=11)
    ax.set_xlim(0, min(1.0, max(0.10, float(shown["proportion"].max()) * 1.18 if len(shown) else 1.0)))
    ax.set_xlabel("Selection proportion across outer evaluations")
    ax.set_title("Hodge candidate selection stability", fontweight="bold")
    fig.tight_layout()
    _save(fig, base)


def plot_parameter_selection_frequency(freq: pd.DataFrame, base: Path) -> None:
    params = list(dict.fromkeys(freq["parameter"].astype(str)))
    fig, axes = plt.subplots(len(params), 1, figsize=(15, max(12, 4.5 * len(params))))
    axes = np.atleast_1d(axes)
    for ax, param in zip(axes, params):
        sub = freq[freq.parameter == param].copy()
        x = np.arange(len(sub))
        bars = ax.bar(x, sub["proportion"].to_numpy(float))
        for b, n in zip(bars, sub["count"].to_numpy(int)):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.012, str(int(n)), ha="center", va="bottom", fontsize=12)
        ax.set_xticks(x, sub["value"].astype(str).tolist())
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("Proportion")
        ax.set_title(param, fontweight="bold")
    fig.suptitle("Marginal hyper-parameter selection frequency", fontweight="bold", y=1.005)
    fig.tight_layout()
    _save(fig, base)


def plot_outer_test_coverage(coverage: pd.DataFrame, base: Path) -> None:
    shown = coverage.sort_values(["domain", "label"], kind="mergesort")
    x = np.arange(len(shown))
    fig, ax = plt.subplots(figsize=(24, 10))
    ax.bar(x, shown["test_appearances"].to_numpy(float))
    ax.set_xticks(x, shown["label"].astype(str).tolist(), rotation=90, fontsize=9)
    ax.set_ylabel("Number of outer-test appearances")
    ax.set_title("Outer-test coverage of all real networks", fontweight="bold")
    fig.tight_layout()
    _save(fig, base)


def plot_nested_confusion(per_query: pd.DataFrame, domains: Sequence[str], base: Path) -> None:
    doms = list(domains)
    index = {d: i for i, d in enumerate(doms)}
    C = np.zeros((len(doms), len(doms)), dtype=int)
    for row in per_query.itertuples(index=False):
        C[index[str(row.domain)], index[str(row.nearest_neighbor_domain)]] += 1
    fig, ax = plt.subplots(figsize=(12, 11))
    im = ax.imshow(C, cmap="Blues", interpolation="nearest")
    for i in range(len(doms)):
        for j in range(len(doms)):
            ax.text(j, i, str(C[i, j]), ha="center", va="center", fontsize=16)
    ax.set_xticks(range(len(doms)), doms, rotation=30, ha="right")
    ax.set_yticks(range(len(doms)), doms)
    ax.set_xlabel("Predicted domain from nearest outer-training network")
    ax.set_ylabel("True outer-test domain")
    ax.set_title("HodgeLearned aggregate outer-test 1-NN confusion", fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Repeated outer-test predictions")
    fig.tight_layout()
    _save(fig, base)
