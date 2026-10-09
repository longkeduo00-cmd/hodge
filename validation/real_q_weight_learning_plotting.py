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


def plot_learned_q_weights(frame: pd.DataFrame, stem: Path, *, title: str = 'Learned exclusive-q Hodge layer weights') -> None:
    sub = frame.sort_values('q')
    fig, ax = plt.subplots(figsize=(15, 8))
    x = np.arange(len(sub))
    values = sub['train_schema_weight'].to_numpy(float) if 'train_schema_weight' in sub else sub['reference_weight'].to_numpy(float)
    ax.bar(x, values)
    ax.set_xticks(x)
    ax.set_xticklabels([f"K{int(v)}\n(q={int(q)})" for v, q in zip(sub['clique_size'], sub['q'])], fontsize=12)
    ax.set_ylabel('Learned train-schema weight', fontsize=16)
    ax.set_xlabel('Exclusive maximal-clique layer', fontsize=16)
    ax.set_title(title, fontsize=18)
    ax.tick_params(axis='y', labelsize=13)
    ax.set_ylim(bottom=0)
    ax.grid(axis='y', alpha=0.22)
    _save(fig, stem)


def plot_effective_vs_reference_weights(frame: pd.DataFrame, stem: Path) -> None:
    sub = frame.sort_values('q')
    fig, ax = plt.subplots(figsize=(15, 8))
    x = np.arange(len(sub))
    if 'train_schema_weight' in sub:
        ax.plot(x, sub['train_schema_weight'], marker='o', linewidth=2.3, markersize=7, label='Train-schema learned')
    else:
        ax.plot(x, sub['reference_weight'], marker='o', linewidth=2.3, markersize=7, label='Full-schema reference')
    if 'mean_train_pair_effective_weight' in sub:
        ax.plot(x, sub['mean_train_pair_effective_weight'], marker='s', linewidth=2.0, markersize=6, label='Mean train-pair effective')
    if 'mean_test_train_pair_effective_weight' in sub:
        ax.plot(x, sub['mean_test_train_pair_effective_weight'], marker='^', linewidth=2.0, markersize=6, label='Mean test→train effective')
    ax.set_xticks(x)
    ax.set_xticklabels([f"K{int(v)}" for v in sub['clique_size']], fontsize=12)
    ax.set_xlabel('Exclusive maximal-clique layer', fontsize=16)
    ax.set_ylabel('Weight', fontsize=16)
    ax.set_title('Reference vs pair-adaptive effective q weights', fontsize=18)
    ax.tick_params(axis='y', labelsize=13)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.22)
    _save(fig, stem)


def plot_q_weight_trajectories(frame: pd.DataFrame, stem: Path) -> None:
    weight_cols = [c for c in frame.columns if c.startswith('weight_q')]
    fig, ax = plt.subplots(figsize=(16, 9))
    for col in weight_cols:
        q = int(col.replace('weight_q', ''))
        ax.plot(frame['epoch'], frame[col], linewidth=1.8, label=f'q={q} / K{q+1}')
    ax.set_xlabel('Adam epoch', fontsize=16)
    ax.set_ylabel('Full-schema reference weight', fontsize=16)
    ax.set_title('q-layer weight optimization trajectory', fontsize=18)
    ax.tick_params(labelsize=13)
    ax.legend(fontsize=10, ncol=2)
    ax.grid(alpha=0.2)
    _save(fig, stem)


def plot_q_weight_loss(frame: pd.DataFrame, stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(frame['epoch'], frame['loss'], linewidth=2.2)
    ax.set_xlabel('Adam epoch', fontsize=16)
    ax.set_ylabel('NCA training loss', fontsize=16)
    ax.set_title('q-weight training loss', fontsize=18)
    ax.tick_params(labelsize=13)
    ax.grid(alpha=0.22)
    _save(fig, stem)


def plot_outer_weight_heatmap(frame: pd.DataFrame, stem: Path) -> None:
    value_col = 'train_schema_weight' if 'train_schema_weight' in frame else 'reference_weight'
    pivot = frame.pivot(index='outer_id', columns='q', values=value_col)
    arr = pivot.to_numpy(float)
    fig, ax = plt.subplots(figsize=(15, max(8, 0.45 * len(pivot))))
    im = ax.imshow(arr, aspect='auto', interpolation='nearest', cmap='jet', vmin=0.0)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f'K{int(q)+1}' for q in pivot.columns], fontsize=12)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_xlabel('Exclusive maximal-clique layer', fontsize=16)
    ax.set_ylabel('Outer evaluation', fontsize=16)
    ax.set_title('Learned q-weight stability across outer evaluations', fontsize=18)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label('Learned train-schema weight', fontsize=14)
    cbar.ax.tick_params(labelsize=11)
    _save(fig, stem)
