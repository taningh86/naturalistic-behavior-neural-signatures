"""
Dual-Probe Manifold Geometry: Layer 1c — Cross-Region CCA
==========================================================
Canonical Correlation Analysis between ACA and LHA manifolds.

For the same session:
  1. PCA-reduce each region to intrinsic dimensionality
  2. Run CCA on temporally aligned PC scores
  3. Report canonical correlations + null comparison
  4. Visualize canonical variates colored by behavior
"""

import yaml
import json
import sys
import time as timer
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
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
K_PCS = {'ACA': 10, 'LHA': 5}
N_CCA = 5           # Number of canonical components (min of the two K values)
N_SHUFFLES = 100     # Null shuffles for significance
MIN_EVENTS = 50

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
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    unit_ids = sorted(spike_dict.keys())
    matrix = np.zeros((n_bins, len(unit_ids)))
    for j, uid in enumerate(unit_ids):
        counts, _ = np.histogram(spike_dict[uid], bins=bin_edges)
        matrix[:, j] = gaussian_filter1d(counts.astype(float), sigma=SMOOTH_SIGMA)
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    stds[stds == 0] = 1.0
    matrix = (matrix - means) / stds
    return matrix, bin_centers, len(unit_ids)


def circular_shift_shuffle(X):
    """Circular shift each neuron independently."""
    X_shuf = np.empty_like(X)
    rng = np.random.default_rng()
    for j in range(X.shape[1]):
        shift = rng.integers(1, X.shape[0])
        X_shuf[:, j] = np.roll(X[:, j], shift)
    return X_shuf


def run_cca(X_aca_pcs, X_lha_pcs, n_components):
    """Run CCA and return canonical correlations and variates."""
    cca = CCA(n_components=n_components, max_iter=1000)
    X_c, Y_c = cca.fit_transform(X_aca_pcs, X_lha_pcs)
    # Canonical correlations = correlation of each pair of variates
    canon_corrs = np.array([np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1]
                            for i in range(n_components)])
    return canon_corrs, X_c, Y_c, cca


def load_behavioral(session_num):
    """Load EthoVision xlsx."""
    sval = sessions_cfg[f"session_{session_num}"]
    raw = pd.read_excel(sval['behavior'], header=None)
    col_names = list(raw.iloc[34].values)
    data = raw.iloc[36:].copy()
    data.columns = col_names
    data = data.reset_index(drop=True)
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    return data


def align_to_bins(behav_df, bin_centers):
    behav_times = behav_df['Trial time'].values.astype(float)
    indices = np.searchsorted(behav_times, bin_centers, side='left')
    indices = np.clip(indices, 0, len(behav_times) - 1)
    prev = np.clip(indices - 1, 0, len(behav_times) - 1)
    use_prev = np.abs(behav_times[prev] - bin_centers) < np.abs(behav_times[indices] - bin_centers)
    indices[use_prev] = prev[use_prev]
    return indices


def extract_compartment(behav_df, indices):
    """Extract compartment labels."""
    pot_zone_cols = [c for c in behav_df.columns
                     if 'Zone(Pot-' in str(c) and ' zone' not in c
                     and 'Distance' not in str(c)]
    home_col = [c for c in behav_df.columns if 'Zone(Home' in str(c)
                and 'corner' not in c and 'Distance' not in str(c)]
    ladder_col = [c for c in behav_df.columns if 'Zone(ladder' in str(c)
                  and 'Distance' not in str(c)]
    compartment = np.full(len(indices), 'Arena', dtype=object)
    if home_col:
        compartment[behav_df[home_col[0]].values[indices] == 1] = 'Home'
    if ladder_col:
        compartment[behav_df[ladder_col[0]].values[indices] == 1] = 'Ladder'
    if pot_zone_cols:
        at_pot = np.zeros(len(indices), dtype=bool)
        for c in pot_zone_cols:
            at_pot |= (behav_df[c].values[indices] == 1)
        compartment[at_pot] = 'AtPot'
    return compartment


def extract_scored_behaviors(behav_df, indices):
    """Extract scored behavioral annotations."""
    behaviors = {}
    skip_prefixes = ['Trial time', 'Recording', 'X ', 'Y ', 'Area', 'Elongation',
                     'Direction', 'Distance', 'Velocity', 'Zone(', 'Result']
    for col in behav_df.columns:
        if col is None or str(col) == 'nan':
            continue
        col_str = str(col)
        if any(col_str.startswith(p) for p in skip_prefixes):
            continue
        vals = behav_df[col].values
        n_active = (vals == 1).sum()
        if n_active < MIN_EVENTS:
            continue
        binned = vals[indices].astype(float)
        clean_name = col_str.strip().replace(' ', '_').lower()
        behaviors[clean_name] = binned == 1
    return behaviors


def main():
    session_num = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    sval = sessions_cfg[f"session_{session_num}"]
    print(f"{'='*70}")
    print(f"CROSS-REGION CCA -- S{session_num} ({sval['state']}/{sval['phase']})")
    print(f"{'='*70}")

    # Load both regions
    print("\nLoading ACA...")
    X_aca, bin_centers_aca, n_aca = load_and_preprocess(session_num, 'ACA')
    print(f"  {n_aca} units, {X_aca.shape[0]} bins")

    print("Loading LHA...")
    X_lha, bin_centers_lha, n_lha = load_and_preprocess(session_num, 'LHA')
    print(f"  {n_lha} units, {X_lha.shape[0]} bins")

    # Ensure same number of time bins
    n_bins = min(X_aca.shape[0], X_lha.shape[0])
    X_aca = X_aca[:n_bins]
    X_lha = X_lha[:n_bins]
    bin_centers = bin_centers_aca[:n_bins]
    print(f"  Aligned: {n_bins} bins")

    # PCA reduce
    print(f"\nPCA: ACA -> {K_PCS['ACA']} dims, LHA -> {K_PCS['LHA']} dims")
    pca_aca = PCA(n_components=K_PCS['ACA'])
    pca_lha = PCA(n_components=K_PCS['LHA'])
    X_aca_pcs = pca_aca.fit_transform(X_aca)
    X_lha_pcs = pca_lha.fit_transform(X_lha)
    print(f"  ACA: {100*sum(pca_aca.explained_variance_ratio_):.1f}% var explained")
    print(f"  LHA: {100*sum(pca_lha.explained_variance_ratio_):.1f}% var explained")

    # Run CCA
    print(f"\nCCA ({N_CCA} components)...")
    t0 = timer.time()
    canon_corrs, X_c, Y_c, cca_model = run_cca(X_aca_pcs, X_lha_pcs, N_CCA)
    print(f"  Done ({timer.time()-t0:.1f}s)")
    print(f"  Canonical correlations:")
    for i, r in enumerate(canon_corrs):
        print(f"    CC{i+1}: r = {r:.4f}")

    # Null distribution
    print(f"\nNull: {N_SHUFFLES} circular-shift shuffles of LHA...")
    t0 = timer.time()
    null_corrs = np.zeros((N_SHUFFLES, N_CCA))
    for i in range(N_SHUFFLES):
        X_lha_shuf = circular_shift_shuffle(X_lha)
        X_lha_shuf_pcs = pca_lha.transform(X_lha_shuf)
        cc_null, _, _, _ = run_cca(X_aca_pcs, X_lha_shuf_pcs, N_CCA)
        null_corrs[i] = cc_null
        if (i + 1) % 20 == 0:
            print(f"  Shuffle {i+1}/{N_SHUFFLES}")
    print(f"  Done ({timer.time()-t0:.1f}s)")

    # Significance
    print(f"\nSignificance:")
    p_values = np.zeros(N_CCA)
    for i in range(N_CCA):
        p_values[i] = np.mean(null_corrs[:, i] >= canon_corrs[i])
        sig = '***' if p_values[i] < 0.001 else '**' if p_values[i] < 0.01 else '*' if p_values[i] < 0.05 else 'ns'
        print(f"  CC{i+1}: r={canon_corrs[i]:.4f} vs null={null_corrs[:, i].mean():.4f}+/-{null_corrs[:, i].std():.4f} "
              f"(p={p_values[i]:.3f} {sig})")

    # Load behavioral data for coloring
    print("\nLoading behavioral data for visualization...")
    behav_df = load_behavioral(session_num)
    indices = align_to_bins(behav_df, bin_centers)
    compartment = extract_compartment(behav_df, indices)
    velocity = behav_df['Velocity(Center-point)'].values[indices].astype(float)
    velocity = np.where(np.isnan(velocity), 0, velocity)
    behaviors = extract_scored_behaviors(behav_df, indices)

    # === Figure 1: Canonical correlations + null ===
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Panel 1: Canonical correlation spectrum
    ax = fig.add_subplot(gs[0, 0])
    x_pos = np.arange(N_CCA)
    ax.bar(x_pos, canon_corrs, width=0.4, color='steelblue', label='Data', zorder=3)
    null_mean = null_corrs.mean(axis=0)
    null_std = null_corrs.std(axis=0)
    null_95 = np.percentile(null_corrs, 95, axis=0)
    ax.bar(x_pos + 0.4, null_mean, width=0.4, color='gray', alpha=0.5, label='Null mean', zorder=3)
    ax.errorbar(x_pos + 0.4, null_mean, yerr=null_std, fmt='none', color='black', capsize=3, zorder=4)
    ax.plot(x_pos + 0.4, null_95, 'k^', markersize=6, label='Null 95th pct', zorder=5)
    for i in range(N_CCA):
        if p_values[i] < 0.05:
            ax.text(x_pos[i], canon_corrs[i] + 0.02, '*', ha='center', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos + 0.2)
    ax.set_xticklabels([f'CC{i+1}' for i in range(N_CCA)])
    ax.set_ylabel('Canonical Correlation')
    ax.set_title('Canonical Correlations: ACA-LHA')
    ax.legend(fontsize=8)
    ax.set_ylim(0, min(1.0, max(canon_corrs) * 1.3))

    # Panel 2: Null distribution for CC1
    ax = fig.add_subplot(gs[0, 1])
    ax.hist(null_corrs[:, 0], bins=25, color='gray', alpha=0.6, label='Null')
    ax.axvline(canon_corrs[0], color='steelblue', lw=2, label=f'Data r={canon_corrs[0]:.3f}')
    ax.set_xlabel('Canonical Correlation')
    ax.set_ylabel('Count')
    ax.set_title(f'CC1 Null Distribution (p={p_values[0]:.3f})')
    ax.legend(fontsize=9)

    # Panel 3: Null distribution for CC2
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(null_corrs[:, 1], bins=25, color='gray', alpha=0.6, label='Null')
    ax.axvline(canon_corrs[1], color='darkorange', lw=2, label=f'Data r={canon_corrs[1]:.3f}')
    ax.set_xlabel('Canonical Correlation')
    ax.set_ylabel('Count')
    ax.set_title(f'CC2 Null Distribution (p={p_values[1]:.3f})')
    ax.legend(fontsize=9)

    # Panel 4: CC1 ACA vs CC1 LHA colored by compartment
    ax = fig.add_subplot(gs[1, 0])
    comp_colors = {'Home': 'blue', 'Ladder': 'green', 'Arena': 'gray', 'AtPot': 'red'}
    for label, color in comp_colors.items():
        mask = compartment == label
        if mask.sum() > 0:
            ax.scatter(X_c[mask, 0], Y_c[mask, 0], c=color, s=0.5,
                       alpha=0.3, label=label, rasterized=True)
    ax.set_xlabel('ACA CC1')
    ax.set_ylabel('LHA CC1')
    ax.set_title('CC1: ACA vs LHA (compartment)')
    ax.legend(markerscale=10, fontsize=8)

    # Panel 5: CC1 vs CC2 (ACA variates) colored by compartment
    ax = fig.add_subplot(gs[1, 1])
    for label, color in comp_colors.items():
        mask = compartment == label
        if mask.sum() > 0:
            ax.scatter(X_c[mask, 0], X_c[mask, 1], c=color, s=0.5,
                       alpha=0.3, label=label, rasterized=True)
    ax.set_xlabel('ACA CC1')
    ax.set_ylabel('ACA CC2')
    ax.set_title('ACA Canonical Variates (compartment)')
    ax.legend(markerscale=10, fontsize=8)

    # Panel 6: CC1 ACA vs CC1 LHA colored by feeding/digging
    ax = fig.add_subplot(gs[1, 2])
    feed_mask = behaviors.get('feeding', np.zeros(n_bins, dtype=bool))
    dig_mask = behaviors.get('digging_sand', np.zeros(n_bins, dtype=bool))
    other = ~feed_mask & ~dig_mask
    ax.scatter(X_c[other, 0], Y_c[other, 0], c='lightgray', s=0.3,
               alpha=0.15, rasterized=True, label='Other')
    if feed_mask.sum() > 0:
        ax.scatter(X_c[feed_mask, 0], Y_c[feed_mask, 0], c='darkorange', s=1.5,
                   alpha=0.6, rasterized=True, label=f'Feeding ({feed_mask.sum()})')
    if dig_mask.sum() > 0:
        ax.scatter(X_c[dig_mask, 0], Y_c[dig_mask, 0], c='purple', s=1.5,
                   alpha=0.6, rasterized=True, label=f'Digging ({dig_mask.sum()})')
    ax.set_xlabel('ACA CC1')
    ax.set_ylabel('LHA CC1')
    ax.set_title('CC1: ACA vs LHA (behaviors)')
    ax.legend(markerscale=5, fontsize=8)

    fig.suptitle(f'Cross-Region CCA -- S{session_num} ACA-LHA',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path = figdir / f"S{session_num}_crossregion_cca.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved {fig_path}")

    # === Figure 2: Canonical variates over time ===
    fig, axes = plt.subplots(N_CCA, 1, figsize=(16, 3 * N_CCA), sharex=True)
    if N_CCA == 1:
        axes = [axes]
    for i in range(N_CCA):
        ax = axes[i]
        sig = '*' if p_values[i] < 0.05 else 'ns'
        ax.plot(bin_centers / 60, X_c[:, i], alpha=0.5, lw=0.5, color='steelblue', label='ACA')
        ax.plot(bin_centers / 60, Y_c[:, i], alpha=0.5, lw=0.5, color='darkorange', label='LHA')
        ax.set_ylabel(f'CC{i+1} (r={canon_corrs[i]:.3f} {sig})')
        if i == 0:
            ax.legend(fontsize=9, loc='upper right')
    axes[-1].set_xlabel('Time (min)')
    fig.suptitle(f'Canonical Variates Over Time -- S{session_num}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig_path2 = figdir / f"S{session_num}_crossregion_cca_timeseries.png"
    plt.savefig(fig_path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {fig_path2}")

    # === Report ===
    lines = [
        f"# Cross-Region CCA -- S{session_num}",
        f"",
        f"ACA: {n_aca} units -> {K_PCS['ACA']} PCs ({100*sum(pca_aca.explained_variance_ratio_):.1f}% var)",
        f"LHA: {n_lha} units -> {K_PCS['LHA']} PCs ({100*sum(pca_lha.explained_variance_ratio_):.1f}% var)",
        f"CCA: {N_CCA} components, {N_SHUFFLES} null shuffles (circular shift of LHA).",
        f"",
        f"## Canonical Correlations",
        f"",
        f"| CC | Data r | Null mean +/- SD | Null 95th | p-value | Sig |",
        f"|----|--------|------------------|-----------|---------|-----|",
    ]
    for i in range(N_CCA):
        sig = '***' if p_values[i] < 0.001 else '**' if p_values[i] < 0.01 else '*' if p_values[i] < 0.05 else 'ns'
        lines.append(f"| CC{i+1} | {canon_corrs[i]:.4f} | "
                     f"{null_corrs[:, i].mean():.4f} +/- {null_corrs[:, i].std():.4f} | "
                     f"{np.percentile(null_corrs[:, i], 95):.4f} | "
                     f"{p_values[i]:.3f} | {sig} |")

    n_sig = (p_values < 0.05).sum()
    lines.extend([
        f"",
        f"**{n_sig}/{N_CCA} canonical dimensions significant (p < 0.05).**",
        f"",
        f"---",
        f"*Circular-shift null preserves per-neuron autocorrelation, destroys cross-region coupling.*",
    ])

    rp = outdir / f"S{session_num}_crossregion_cca.md"
    with open(rp, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"Saved {rp}")

    # Save JSON
    jp = outdir / f"S{session_num}_crossregion_cca.json"
    save_data = {
        'canonical_correlations': canon_corrs.tolist(),
        'p_values': p_values.tolist(),
        'null_mean': null_corrs.mean(axis=0).tolist(),
        'null_std': null_corrs.std(axis=0).tolist(),
        'null_95th': np.percentile(null_corrs, 95, axis=0).tolist(),
        'n_aca': n_aca, 'n_lha': n_lha,
        'K_aca': K_PCS['ACA'], 'K_lha': K_PCS['LHA'],
        'N_shuffles': N_SHUFFLES,
    }
    with open(jp, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"Saved {jp}")

    print(f"\n{'='*70}")
    print(f"DONE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
