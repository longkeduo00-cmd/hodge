from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CORE_METRICS = [
    ("auroc_same_domain", "AUROC"),
    ("nn1_accuracy", "1-NN Acc."),
    ("nn1_macro_f1", "Macro-F1"),
    ("mAP", "mAP"),
    ("ndcg_at_7", "NDCG@7"),
]


def _save(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _boxplot(ax, data, labels):
    try:
        return ax.boxplot(data, tick_labels=labels, patch_artist=True, showmeans=True)
    except TypeError:
        return ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True)


def plot_repeat_metric_summary(summary: pd.DataFrame, stem: Path) -> None:
    methods = summary["method"].astype(str).tolist()
    fig, axes = plt.subplots(len(CORE_METRICS), 1, figsize=(20, 22), constrained_layout=True)
    x = np.arange(len(methods))
    for ax, (metric, title) in zip(axes, CORE_METRICS):
        mean = summary[f"mean_{metric}"].to_numpy(float)
        std = summary[f"std_{metric}"].to_numpy(float)
        ax.bar(x, mean, yerr=std, capsize=5)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel(title, fontsize=15)
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(axis="y", alpha=0.2)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(methods, rotation=25, ha="right", fontsize=11)
    for ax in axes[:-1]:
        ax.set_xticks(x)
        ax.set_xticklabels([])
    fig.suptitle("Repeat-level out-of-fold generalization: mean ± SD across repeats", fontsize=20)
    _save(fig, stem)


def plot_repeat_metric_distributions(frame: pd.DataFrame, stem: Path) -> None:
    methods = frame["method"].drop_duplicates().astype(str).tolist()
    fig, axes = plt.subplots(3, 2, figsize=(22, 18), constrained_layout=True)
    axes = axes.ravel()
    for ax, (metric, title) in zip(axes, CORE_METRICS):
        data = [frame.loc[frame["method"] == m, metric].to_numpy(float) for m in methods]
        _boxplot(ax, data, methods)
        ax.set_ylim(0, 1.0)
        ax.set_title(title, fontsize=17)
        ax.tick_params(axis="x", rotation=28, labelsize=9)
        ax.tick_params(axis="y", labelsize=11)
        ax.grid(axis="y", alpha=0.2)
    axes[-1].axis("off")
    fig.suptitle("Repeat-level OOF metric distributions (10 repeats)", fontsize=20)
    _save(fig, stem)


def plot_alpha_distribution(alpha_df: pd.DataFrame, stem: Path) -> None:
    groups = [g for g in ("raw", "scaled") if g in set(alpha_df["scaling"].astype(str))]
    data = [alpha_df.loc[alpha_df["scaling"] == g, "alpha"].to_numpy(float) for g in groups]
    fig, ax = plt.subplots(figsize=(12, 8))
    _boxplot(ax, data, groups)
    for x, vals in enumerate(data, start=1):
        jitter = np.linspace(-0.08, 0.08, len(vals)) if len(vals) else np.array([])
        ax.scatter(np.full(len(vals), x) + jitter, vals, s=24, alpha=0.65)
    ax.axhline(0.0, linewidth=1.2, alpha=0.5)
    ax.set_ylabel("Selected order-trend α", fontsize=15)
    ax.set_title("Single-α stability across 50 folds", fontsize=18)
    ax.tick_params(labelsize=12)
    ax.grid(axis="y", alpha=0.2)
    _save(fig, stem)


def plot_free_q_heatmap(frame: pd.DataFrame, stem: Path, *, title: str) -> None:
    pivot = frame.pivot(index="outer_id", columns="q", values="reference_weight").sort_index()
    arr = pivot.to_numpy(float)
    fig, ax = plt.subplots(figsize=(16, max(11, 0.26 * len(pivot))))
    im = ax.imshow(arr, aspect="auto", interpolation="nearest", cmap="jet", vmin=0.0)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"K{int(q)+1}" for q in pivot.columns], fontsize=12)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_xlabel("Exclusive maximal-clique layer", fontsize=15)
    ax.set_ylabel("CV fold", fontsize=15)
    ax.set_title(title, fontsize=18)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label("Free-q reference weight", fontsize=13)
    cbar.ax.tick_params(labelsize=10)
    _save(fig, stem)


def plot_free_q_mean_sd(summary: pd.DataFrame, stem: Path, *, title: str) -> None:
    sub = summary.sort_values("q")
    x = np.arange(len(sub))
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.errorbar(x, sub["mean_weight"], yerr=sub["std_weight"], marker="o", linewidth=2.0, capsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"K{int(k)}" for k in sub["clique_size"]], fontsize=12)
    ax.set_ylabel("Weight (mean ± fold SD)", fontsize=15)
    ax.set_xlabel("Exclusive maximal-clique layer", fontsize=15)
    ax.set_ylim(bottom=0)
    ax.set_title(title, fontsize=18)
    ax.grid(alpha=0.2)
    _save(fig, stem)


def plot_layer_scale_heatmap(scales: pd.DataFrame, stem: Path) -> None:
    pivot = scales.pivot(index="outer_id", columns="q", values="rms_distance_scale").sort_index()
    arr = pivot.to_numpy(float)
    fig, ax = plt.subplots(figsize=(16, max(11, 0.26 * len(pivot))))
    im = ax.imshow(arr, aspect="auto", interpolation="nearest", cmap="jet")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"K{int(q)+1}" for q in pivot.columns], fontsize=12)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_xlabel("Exclusive maximal-clique layer", fontsize=15)
    ax.set_ylabel("CV fold", fontsize=15)
    ax.set_title("Train-only RMS distance scales by fold", fontsize=18)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label("RMS scale s_q", fontsize=13)
    cbar.ax.tick_params(labelsize=10)
    _save(fig, stem)
