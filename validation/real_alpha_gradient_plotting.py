from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _save(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix('.png'), dpi=300, bbox_inches='tight')
    fig.savefig(stem.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def plot_alpha_by_outer(frame: pd.DataFrame, stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 8))
    x = np.arange(len(frame))
    ax.plot(x, frame['alpha'].to_numpy(float), marker='o', linewidth=2.2, markersize=6)
    ax.axhline(0.0, linewidth=1.4)
    ax.set_xlabel('Outer evaluation', fontsize=16)
    ax.set_ylabel('Learned alpha', fontsize=16)
    ax.set_title('Gradient-learned alpha across outer evaluations', fontsize=18)
    ax.tick_params(labelsize=13)
    ax.grid(alpha=0.25)
    _save(fig, stem)


def plot_alpha_histogram(frame: pd.DataFrame, stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 8))
    vals = frame['alpha'].to_numpy(float)
    bins = min(max(8, int(np.sqrt(max(len(vals), 1)) * 2)), 24)
    ax.hist(vals, bins=bins, edgecolor='black', linewidth=1.0)
    ax.axvline(float(np.median(vals)), linestyle='--', linewidth=2.0, label=f"median={np.median(vals):.3g}")
    ax.set_xlabel('Learned alpha', fontsize=16)
    ax.set_ylabel('Outer-evaluation count', fontsize=16)
    ax.set_title('Distribution of gradient-learned alpha', fontsize=18)
    ax.tick_params(labelsize=13)
    ax.legend(fontsize=13)
    _save(fig, stem)


def plot_outer_training_trajectories(frame: pd.DataFrame, stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    for outer_id, sub in frame.groupby('outer_id', sort=False):
        ax.plot(sub['epoch'], sub['alpha'], linewidth=1.5, alpha=0.75, label=str(outer_id))
    ax.set_xlabel('Adam epoch', fontsize=16)
    ax.set_ylabel('Alpha', fontsize=16)
    ax.set_title('Outer-training alpha optimization trajectories', fontsize=18)
    ax.tick_params(labelsize=13)
    if frame['outer_id'].nunique() <= 12:
        ax.legend(fontsize=9, ncol=2)
    ax.grid(alpha=0.2)
    _save(fig, stem)


def plot_outer_loss_trajectories(frame: pd.DataFrame, stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    for outer_id, sub in frame.groupby('outer_id', sort=False):
        ax.plot(sub['epoch'], sub['loss'], linewidth=1.5, alpha=0.75, label=str(outer_id))
    ax.set_xlabel('Adam epoch', fontsize=16)
    ax.set_ylabel('NCA training loss', fontsize=16)
    ax.set_title('Outer-training differentiable loss trajectories', fontsize=18)
    ax.tick_params(labelsize=13)
    if frame['outer_id'].nunique() <= 12:
        ax.legend(fontsize=9, ncol=2)
    ax.grid(alpha=0.2)
    _save(fig, stem)
