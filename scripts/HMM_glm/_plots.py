"""Diagnostic plots for the GLM-HMM fits."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_state_occupancy_heatmap(occupancy_df: pd.DataFrame, K: int,
                                    out_path: Path, title: str = ""):
    """occupancy_df: index = state 0..K-1, columns = session labels (e.g. 'S4_fed').
    Cells = occupancy proportion."""
    fig, ax = plt.subplots(figsize=(max(6, len(occupancy_df.columns)*0.5), 4))
    im = ax.imshow(occupancy_df.values, aspect="auto", cmap="viridis",
                    vmin=0, vmax=occupancy_df.values.max())
    ax.set_xticks(np.arange(len(occupancy_df.columns)))
    ax.set_xticklabels(occupancy_df.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(np.arange(K))
    ax.set_yticklabels([f"S{k}" for k in range(K)], fontsize=8)
    ax.set_ylabel("HMM state")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="occupancy")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_transition_matrices(trans_by_state: np.ndarray, state_labels: list[str],
                              out_path: Path, title: str = ""):
    """trans_by_state: (M, K, K) array. Plot M panels side by side."""
    M = trans_by_state.shape[0]
    K = trans_by_state.shape[1]
    fig, axes = plt.subplots(1, M, figsize=(4*M, 4))
    if M == 1:
        axes = [axes]
    for m, ax in enumerate(axes):
        im = ax.imshow(trans_by_state[m], cmap="magma", vmin=0, vmax=1, aspect="auto")
        ax.set_title(f"{state_labels[m]}")
        ax.set_xlabel("z_t")
        ax.set_ylabel("z_{t-1}")
        ax.set_xticks(np.arange(K))
        ax.set_yticks(np.arange(K))
        ax.set_xticklabels([str(k) for k in range(K)], fontsize=7)
        ax.set_yticklabels([str(k) for k in range(K)], fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_emission_heatmap(emissions: np.ndarray, pooled_ids: np.ndarray,
                           out_path: Path, title: str = ""):
    """emissions: (K, D) log-rate. pooled_ids: (D, 2) (session_num, cluster_id)."""
    K, D = emissions.shape
    # Sort units by state with highest log-rate
    argmax_states = np.argmax(emissions, axis=0)
    order = np.argsort(argmax_states)
    em_sorted = emissions[:, order]
    fig, ax = plt.subplots(figsize=(max(8, D * 0.05), max(4, K * 0.4)))
    im = ax.imshow(em_sorted, aspect="auto", cmap="RdBu_r",
                    vmin=-emissions.max(), vmax=emissions.max())
    ax.set_xlabel(f"Unit (sorted by argmax state); n={D}")
    ax.set_ylabel("HMM state")
    ax.set_yticks(np.arange(K))
    ax.set_yticklabels([f"S{k}" for k in range(K)])
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="log rate")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_state_timeline(z: np.ndarray, K: int, out_path: Path, title: str = ""):
    """Bar-style timeline of Viterbi state assignments."""
    fig, ax = plt.subplots(figsize=(12, 2))
    cmap = plt.get_cmap("tab10")
    for k in range(K):
        idx = np.flatnonzero(z == k)
        ax.scatter(idx, np.full_like(idx, k), color=cmap(k % 10), s=1)
    ax.set_xlabel("bin")
    ax.set_ylabel("state")
    ax.set_yticks(np.arange(K))
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_cv_curve(k_vals: list[int], mean_held_out: np.ndarray,
                   sem_held_out: np.ndarray, train_ll: np.ndarray,
                   out_path: Path, title: str = ""):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(k_vals, mean_held_out, yerr=sem_held_out,
                  marker="o", lw=1.5, color="C0", label="held-out LL/bin (mean±SEM)")
    ax.plot(k_vals, train_ll, marker="s", lw=1.5, color="C3",
             label="training LL/bin")
    ax.set_xlabel("K (number of states)")
    ax.set_ylabel("log-likelihood per bin")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
