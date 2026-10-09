from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .style import FIG_DPI, METHOD_COLORS, SUPTITLE_SIZE, apply_style

apply_style()


def _save(fig, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_cv_search(summary: pd.DataFrame, primary_metric: str, selected_id: str, base: Path, top_n: int = 24) -> None:
    col = f"mean_{primary_metric}"
    std = f"std_{primary_metric}"
    shown = summary.sort_values(col, ascending=False).head(int(top_n)).copy().iloc[::-1]
    labels = [
        f"S={('all' if pd.isna(r.max_clique_size) else int(r.max_clique_size))}, Q={('all' if pd.isna(r.max_hodge_order) else int(r.max_hodge_order))}, qmin={int(r.min_layer_q)}, {r.normalization}, a={r.alpha:g}"
        for r in shown.itertuples(index=False)
    ]
    y = np.arange(len(shown))
    fig, ax = plt.subplots(figsize=(18, max(10, 0.55 * len(shown) + 3)))
    bars = ax.barh(y, shown[col].to_numpy(float), xerr=shown[std].to_numpy(float), capsize=4)
    for b, cid in zip(bars, shown["candidate_id"].astype(str)):
        if cid == str(selected_id):
            b.set_hatch("///")
            b.set_linewidth(2.5)
    ax.set_yticks(y, labels, fontsize=11)
    ax.set_xlabel(f"Training-CV {primary_metric} (mean ± SD)")
    ax.set_title("Hodge hyper-parameter search — training partition only", fontweight="bold")
    ax.text(0.01, 1.01, "Hatched bar = selected candidate; hold-out test labels are not used here.", transform=ax.transAxes, fontsize=12, va="bottom")
    fig.tight_layout()
    _save(fig, base)


def plot_inductive_metrics(metrics: pd.DataFrame, base: Path) -> None:
    cols = ["auroc_same_domain", "nn1_accuracy", "nn1_macro_f1", "precision_at_1", "mAP", "ndcg_at_7"]
    labs = ["AUROC", "1-NN Acc.", "Macro-F1", "P@1", "mAP", "NDCG@7"]
    methods = metrics["method"].astype(str).tolist()
    x = np.arange(len(cols), dtype=float)
    width = 0.82 / max(1, len(methods))
    fig, ax = plt.subplots(figsize=(19, 11))
    for mi, method in enumerate(methods):
        row = metrics.iloc[mi]
        vals = [float(row[c]) for c in cols]
        offset = (mi - (len(methods) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width=width * 0.92, label=method, color=METHOD_COLORS.get(method))
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=10, rotation=90)
    ax.set_xticks(x, labs)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score (higher is better)")
    ax.set_title("Hold-out generalization: test queries → training gallery", fontweight="bold")
    ax.legend(frameon=False, ncol=min(4, len(methods)), loc="upper center")
    fig.tight_layout()
    _save(fig, base)


def plot_split_counts(counts: pd.DataFrame, base: Path) -> None:
    domains = list(dict.fromkeys(counts["domain"].astype(str)))
    piv = counts.pivot(index="domain", columns="partition", values="count").reindex(domains).fillna(0)
    x = np.arange(len(domains))
    fig, ax = plt.subplots(figsize=(16, 9))
    bottom = np.zeros(len(domains))
    for part in ("train", "test"):
        vals = piv[part].to_numpy(float) if part in piv.columns else np.zeros(len(domains))
        bars = ax.bar(x, vals, bottom=bottom, label=part)
        for b, v, bot in zip(bars, vals, bottom):
            if v > 0:
                ax.text(b.get_x() + b.get_width() / 2, bot + v / 2, f"{int(v)}", ha="center", va="center", fontsize=14)
        bottom += vals
    ax.set_xticks(x, domains, rotation=18, ha="right")
    ax.set_ylabel("Networks")
    ax.set_title("Outer stratified train/test split by domain", fontweight="bold")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, base)


def plot_inductive_confusion(per_query: pd.DataFrame, domains: Sequence[str], base: Path) -> None:
    doms = list(domains)
    index = {d: i for i, d in enumerate(doms)}
    C = np.zeros((len(doms), len(doms)), dtype=int)
    for row in per_query.itertuples(index=False):
        C[index[str(row.domain)], index[str(row.nearest_neighbor_domain)]] += 1
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(C, cmap="Blues", interpolation="nearest")
    for i in range(len(doms)):
        for j in range(len(doms)):
            ax.text(j, i, str(C[i, j]), ha="center", va="center", fontsize=16)
    ax.set_xticks(range(len(doms)), doms, rotation=30, ha="right")
    ax.set_yticks(range(len(doms)), doms)
    ax.set_xlabel("Predicted domain from nearest training network")
    ax.set_ylabel("True test domain")
    ax.set_title("HodgeLearned inductive 1-NN confusion matrix", fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Test networks")
    fig.tight_layout()
    _save(fig, base)
