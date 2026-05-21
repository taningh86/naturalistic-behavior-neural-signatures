"""
Dual-Probe Manifold Geometry: Layer 1b — Persistent Homology
=============================================================
Computes Vietoris-Rips persistent homology on neural manifolds.

For each region (ACA, LHA):
  1. PCA-reduce to intrinsic dimensionality (ACA K=10, LHA K=5)
  2. Subsample to manageable size (landmark selection)
  3. Compute persistent homology H0, H1, H2
  4. Generate persistence diagrams + Betti curves
  5. Shuffle null comparison (N=20 circular-shift shuffles)
  6. Quantify: max persistence per Hk, total persistence, # features above null
"""

import yaml
import json
import sys
import time as timer
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from ripser import ripser
from persim import plot_diagrams
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

from dp_avalanche_criticality import (
    get_good_units_p0, get_good_units_p1_lha,
    load_spike_times_for_region, FS,
)
import spikeinterface.extractors as se

with open("paths.yaml") as f:
    cfg = yaml.safe_load(f)
sessions_cfg = cfg["double_probe"]["coordinates_1"]["mouse01"]["sessions"]

BIN_MS = 50.0
SMOOTH_SIGMA = 1.0
SUBSAMPLE_N = 1000       # Landmark points for Rips complex
N_SHUFFLES = 20           # Null shuffles
K_PCS = {'ACA': 10, 'LHA': 5}  # Intrinsic dim from Layer 1a
MAX_DIM = 2               # Compute H0, H1, H2

outdir = Path("data/manifold")
figdir = Path("figures/manifold")
outdir.mkdir(parents=True, exist_ok=True)
figdir.mkdir(parents=True, exist_ok=True)


def load_and_preprocess(session_num, region):
    """Load, bin, smooth, z-score neural data."""
    sval = sessions_cfg[f"session_{session_num}"]
    if region == 'ACA':
        sp = Path(sval['probe_0_aca']['sorted'])
        uids = get_good_units_p0(sp)
    else:
        sp = Path(sval['probe_1_lha_rsp']['sorted'])
        uids = get_good_units_p1_lha(sp)
    sorting = se.read_kilosort(sp)
    avail = set(sorting.get_unit_ids())
    uids = np.array([u for u in uids if u in avail])
    spike_dict = load_spike_times_for_region(sorting, uids)
    all_sp = np.concatenate(list(spike_dict.values()))
    dur = float(all_sp.max()) + 1.0
    dt = BIN_MS / 1000.0
    n_bins = int(dur / dt)
    bin_edges = np.arange(0, n_bins + 1) * dt
    unit_ids = sorted(spike_dict.keys())
    matrix = np.zeros((n_bins, len(unit_ids)))
    for j, uid in enumerate(unit_ids):
        counts, _ = np.histogram(spike_dict[uid], bins=bin_edges)
        matrix[:, j] = gaussian_filter1d(counts.astype(float), sigma=SMOOTH_SIGMA)
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    stds[stds == 0] = 1.0
    matrix = (matrix - means) / stds
    return matrix, len(unit_ids)


def circular_shift_shuffle(X):
    """Circular shift each neuron independently."""
    X_shuf = np.empty_like(X)
    rng = np.random.default_rng()
    for j in range(X.shape[1]):
        shift = rng.integers(1, X.shape[0])
        X_shuf[:, j] = np.roll(X[:, j], shift)
    return X_shuf


def maxmin_subsample(X, n):
    """Greedy farthest-point (maxmin) landmark selection.
    Better than random for covering the manifold geometry.
    """
    rng = np.random.default_rng(42)
    N = X.shape[0]
    if N <= n:
        return X.copy()

    indices = [rng.integers(N)]
    dists = np.full(N, np.inf)
    for _ in range(n - 1):
        last = X[indices[-1]]
        d = np.sum((X - last) ** 2, axis=1)
        dists = np.minimum(dists, d)
        indices.append(np.argmax(dists))
    return X[np.array(indices)]


def compute_persistence(X_sub, max_dim=MAX_DIM):
    """Run Rips persistent homology via ripser."""
    result = ripser(X_sub, maxdim=max_dim, do_cocycles=False)
    return result['dgms']


def persistence_stats(dgms):
    """Compute summary statistics from persistence diagrams."""
    stats = {}
    for k in range(len(dgms)):
        dgm = dgms[k]
        # Remove infinite death features for stats
        finite = dgm[np.isfinite(dgm[:, 1])]
        if len(finite) == 0:
            lifetimes = np.array([0.0])
        else:
            lifetimes = finite[:, 1] - finite[:, 0]

        stats[f'H{k}_n_features'] = len(finite)
        stats[f'H{k}_max_persistence'] = float(np.max(lifetimes)) if len(lifetimes) > 0 else 0.0
        stats[f'H{k}_total_persistence'] = float(np.sum(lifetimes))
        stats[f'H{k}_mean_persistence'] = float(np.mean(lifetimes)) if len(lifetimes) > 0 else 0.0

        # Top-3 lifetimes
        top3 = np.sort(lifetimes)[::-1][:3]
        stats[f'H{k}_top3'] = top3.tolist()

    return stats


def betti_curve(dgms, n_steps=200):
    """Compute Betti number as function of filtration scale."""
    # Find global range
    all_births = np.concatenate([d[:, 0] for d in dgms])
    all_deaths = np.concatenate([d[np.isfinite(d[:, 1]), 1] for d in dgms if len(d[np.isfinite(d[:, 1])]) > 0])
    if len(all_deaths) == 0:
        return np.zeros(n_steps), np.zeros((len(dgms), n_steps))
    eps_range = np.linspace(0, np.max(all_deaths) * 1.1, n_steps)

    curves = np.zeros((len(dgms), n_steps))
    for k, dgm in enumerate(dgms):
        for birth, death in dgm:
            if not np.isfinite(death):
                continue
            alive = (eps_range >= birth) & (eps_range < death)
            curves[k] += alive

    return eps_range, curves


def run_region(session_num, region, matrix, n_units):
    """Full persistent homology pipeline for one region."""
    K = K_PCS[region]
    print(f"\n  PCA to {K} dims...", end='', flush=True)
    pca = PCA(n_components=K)
    X_pca = pca.fit_transform(matrix)
    var_expl = sum(pca.explained_variance_ratio_) * 100
    print(f" {var_expl:.1f}% variance explained")

    # Subsample
    print(f"  Maxmin landmark selection ({SUBSAMPLE_N} points)...", end='', flush=True)
    t0 = timer.time()
    X_sub = maxmin_subsample(X_pca, SUBSAMPLE_N)
    print(f" {timer.time()-t0:.1f}s")

    # Compute persistent homology
    print(f"  Computing Rips persistence (H0-H{MAX_DIM})...", end='', flush=True)
    t0 = timer.time()
    dgms = compute_persistence(X_sub)
    dt_data = timer.time() - t0
    print(f" {dt_data:.1f}s")

    stats_data = persistence_stats(dgms)
    for k in range(MAX_DIM + 1):
        top3 = stats_data[f'H{k}_top3']
        top3_str = ', '.join(f'{v:.3f}' for v in top3)
        print(f"    H{k}: {stats_data[f'H{k}_n_features']} features, "
              f"max_pers={stats_data[f'H{k}_max_persistence']:.3f}, "
              f"total_pers={stats_data[f'H{k}_total_persistence']:.3f}, "
              f"top3=[{top3_str}]")

    eps_data, betti_data = betti_curve(dgms)

    # Null distribution
    print(f"  Null: {N_SHUFFLES} circular-shift shuffles...")
    null_stats = {f'H{k}_max_persistence': [] for k in range(MAX_DIM + 1)}
    null_stats.update({f'H{k}_total_persistence': [] for k in range(MAX_DIM + 1)})
    null_betti_all = []

    for i in range(N_SHUFFLES):
        X_shuf = circular_shift_shuffle(matrix)
        X_shuf_pca = pca.transform(X_shuf)  # Project using same PCA axes
        X_shuf_sub = maxmin_subsample(X_shuf_pca, SUBSAMPLE_N)
        dgms_shuf = compute_persistence(X_shuf_sub)
        ss = persistence_stats(dgms_shuf)
        for k in range(MAX_DIM + 1):
            null_stats[f'H{k}_max_persistence'].append(ss[f'H{k}_max_persistence'])
            null_stats[f'H{k}_total_persistence'].append(ss[f'H{k}_total_persistence'])
        _, bc = betti_curve(dgms_shuf, n_steps=len(eps_data))
        null_betti_all.append(bc)
        if (i + 1) % 5 == 0:
            print(f"    Shuffle {i+1}/{N_SHUFFLES}")

    # Compute null percentiles
    print(f"  Null comparison:")
    sig_results = {}
    for k in range(MAX_DIM + 1):
        null_max = np.array(null_stats[f'H{k}_max_persistence'])
        null_total = np.array(null_stats[f'H{k}_total_persistence'])
        data_max = stats_data[f'H{k}_max_persistence']
        data_total = stats_data[f'H{k}_total_persistence']
        pct_max = np.mean(null_max >= data_max)
        pct_total = np.mean(null_total >= data_total)
        sig_max = '*' if pct_max < 0.05 else 'ns'
        sig_total = '*' if pct_total < 0.05 else 'ns'
        print(f"    H{k}: max_pers data={data_max:.3f} vs null={np.mean(null_max):.3f}+/-{np.std(null_max):.3f} "
              f"(p={pct_max:.3f} {sig_max}); "
              f"total_pers data={data_total:.3f} vs null={np.mean(null_total):.3f}+/-{np.std(null_total):.3f} "
              f"(p={pct_total:.3f} {sig_total})")
        sig_results[f'H{k}'] = {
            'max_pers_data': data_max,
            'max_pers_null_mean': float(np.mean(null_max)),
            'max_pers_null_std': float(np.std(null_max)),
            'max_pers_p': float(pct_max),
            'total_pers_data': data_total,
            'total_pers_null_mean': float(np.mean(null_total)),
            'total_pers_null_std': float(np.std(null_total)),
            'total_pers_p': float(pct_total),
        }

    null_betti = np.array(null_betti_all)  # (N_SHUFFLES, n_dims, n_steps)

    # === Figure ===
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Panel 1: Persistence diagram (all Hk)
    ax = fig.add_subplot(gs[0, 0])
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    labels = ['H0', 'H1', 'H2']
    max_val = 0
    for k, dgm in enumerate(dgms):
        finite = dgm[np.isfinite(dgm[:, 1])]
        if len(finite) > 0:
            ax.scatter(finite[:, 0], finite[:, 1], s=15, alpha=0.6,
                       c=colors[k], label=f'{labels[k]} ({len(finite)})', zorder=3)
            max_val = max(max_val, finite.max())
    ax.plot([0, max_val * 1.1], [0, max_val * 1.1], 'k--', alpha=0.3, lw=1)
    ax.set_xlabel('Birth')
    ax.set_ylabel('Death')
    ax.set_title('Persistence Diagram')
    ax.legend(fontsize=9)

    # Panel 2: Persistence barcode (top features per Hk)
    ax = fig.add_subplot(gs[0, 1])
    y_offset = 0
    for k, dgm in enumerate(dgms):
        finite = dgm[np.isfinite(dgm[:, 1])]
        if len(finite) == 0:
            continue
        lifetimes = finite[:, 1] - finite[:, 0]
        order = np.argsort(lifetimes)[::-1][:15]  # Top 15 per Hk
        for idx in order:
            ax.barh(y_offset, lifetimes[idx], left=finite[idx, 0],
                    height=0.8, color=colors[k], alpha=0.7)
            y_offset += 1
        y_offset += 1  # Gap between Hk groups
    ax.set_xlabel('Filtration scale')
    ax.set_ylabel('Features (sorted by persistence)')
    ax.set_title('Barcode (top 15 per Hk)')
    # Add legend manually
    from matplotlib.patches import Patch
    patches = [Patch(color=colors[k], label=labels[k]) for k in range(len(dgms))]
    ax.legend(handles=patches, fontsize=9)

    # Panel 3: Lifetime distribution per Hk
    ax = fig.add_subplot(gs[0, 2])
    for k, dgm in enumerate(dgms):
        finite = dgm[np.isfinite(dgm[:, 1])]
        if len(finite) > 0:
            lifetimes = finite[:, 1] - finite[:, 0]
            ax.hist(lifetimes, bins=30, alpha=0.5, color=colors[k], label=labels[k])
    ax.set_xlabel('Persistence (lifetime)')
    ax.set_ylabel('Count')
    ax.set_title('Lifetime Distribution')
    ax.legend(fontsize=9)

    # Panel 4-6: Betti curves with null envelope
    for k in range(MAX_DIM + 1):
        ax = fig.add_subplot(gs[1, k])
        # Null envelope
        if null_betti.shape[2] == len(eps_data):
            null_k = null_betti[:, k, :]
            null_mean = null_k.mean(axis=0)
            null_lo = np.percentile(null_k, 2.5, axis=0)
            null_hi = np.percentile(null_k, 97.5, axis=0)
            ax.fill_between(eps_data, null_lo, null_hi, alpha=0.2, color='gray', label='Null 95% CI')
            ax.plot(eps_data, null_mean, color='gray', alpha=0.5, lw=1, label='Null mean')
        # Data
        ax.plot(eps_data, betti_data[k], color=colors[k], lw=2, label=f'{labels[k]} (data)')
        ax.set_xlabel('Filtration scale (epsilon)')
        ax.set_ylabel(f'Betti-{k}')
        ax.set_title(f'Betti-{k} Curve')
        ax.legend(fontsize=8)

    fig.suptitle(f'Persistent Homology -- S{session_num} {region} (N={SUBSAMPLE_N} landmarks, K={K} PCs)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path = figdir / f"S{session_num}_{region}_persistent_homology.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {fig_path}")

    # === Report ===
    lines = [
        f"# Persistent Homology -- S{session_num} {region}",
        f"",
        f"N = {n_units} units, K = {K} PCs ({var_expl:.1f}% var explained), "
        f"{SUBSAMPLE_N} maxmin landmarks, {N_SHUFFLES} null shuffles.",
        f"",
        f"## Persistence Statistics",
        f"",
        f"| Hk | # Features | Max Persistence | Total Persistence | Top-3 Lifetimes |",
        f"|-----|-----------|-----------------|-------------------|-----------------|",
    ]
    for k in range(MAX_DIM + 1):
        top3 = stats_data[f'H{k}_top3']
        top3_str = ', '.join(f'{v:.3f}' for v in top3)
        lines.append(f"| H{k} | {stats_data[f'H{k}_n_features']} | "
                     f"{stats_data[f'H{k}_max_persistence']:.3f} | "
                     f"{stats_data[f'H{k}_total_persistence']:.3f} | "
                     f"{top3_str} |")

    lines.extend([
        f"",
        f"## Null Comparison (circular-shift shuffle)",
        f"",
        f"| Hk | Metric | Data | Null (mean +/- SD) | p-value | Sig |",
        f"|-----|--------|------|--------------------|---------|-----|",
    ])
    for k in range(MAX_DIM + 1):
        sr = sig_results[f'H{k}']
        sig_max = '***' if sr['max_pers_p'] < 0.001 else '**' if sr['max_pers_p'] < 0.01 else '*' if sr['max_pers_p'] < 0.05 else 'ns'
        sig_total = '***' if sr['total_pers_p'] < 0.001 else '**' if sr['total_pers_p'] < 0.01 else '*' if sr['total_pers_p'] < 0.05 else 'ns'
        lines.append(f"| H{k} | max_pers | {sr['max_pers_data']:.3f} | "
                     f"{sr['max_pers_null_mean']:.3f} +/- {sr['max_pers_null_std']:.3f} | "
                     f"{sr['max_pers_p']:.3f} | {sig_max} |")
        lines.append(f"| H{k} | total_pers | {sr['total_pers_data']:.3f} | "
                     f"{sr['total_pers_null_mean']:.3f} +/- {sr['total_pers_null_std']:.3f} | "
                     f"{sr['total_pers_p']:.3f} | {sig_total} |")

    lines.extend([
        f"",
        f"---",
        f"*Vietoris-Rips complex via ripser. {BIN_MS}ms bins, Gaussian smooth sigma={SMOOTH_SIGMA}.*",
    ])

    rp = outdir / f"S{session_num}_{region}_layer1b.md"
    with open(rp, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  Saved {rp}")

    # Save JSON
    jp = outdir / f"S{session_num}_{region}_layer1b.json"
    save_data = {
        'stats': stats_data,
        'null_comparison': sig_results,
        'K': K,
        'N_landmarks': SUBSAMPLE_N,
        'N_shuffles': N_SHUFFLES,
        'var_explained_pct': var_expl,
    }
    with open(jp, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"  Saved {jp}")


def main():
    session_num = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    sval = sessions_cfg[f"session_{session_num}"]
    print(f"{'='*70}")
    print(f"PERSISTENT HOMOLOGY -- S{session_num} ({sval['state']}/{sval['phase']})")
    print(f"{'='*70}")

    for region in ['ACA', 'LHA']:
        print(f"\n{'='*50}")
        print(f"  {region}")
        print(f"{'='*50}")
        t0 = timer.time()
        matrix, n_units = load_and_preprocess(session_num, region)
        print(f"  {n_units} units, {matrix.shape[0]} bins")
        run_region(session_num, region, matrix, n_units)
        print(f"  Region time: {(timer.time()-t0)/60:.1f} min")

    print(f"\n{'='*70}")
    print(f"DONE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
