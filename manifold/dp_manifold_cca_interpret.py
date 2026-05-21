"""
Interpret CCA canonical variates — what does each CC encode?
Correlate CC variates with behavioral variables, plot CC1 time course
with behavioral annotations overlaid.
"""

import yaml
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr, spearmanr, pointbiserialr
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
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
N_CCA = 5
MIN_EVENTS = 50

figdir = Path("figures/manifold")


def load_and_preprocess(session_num, region):
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
    return matrix, bin_centers


def load_behavioral(session_num):
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


def main():
    session_num = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    sval = sessions_cfg[f"session_{session_num}"]
    print(f"CCA Interpretation -- S{session_num} ({sval['state']}/{sval['phase']})")

    # Load and align
    X_aca, bin_centers = load_and_preprocess(session_num, 'ACA')
    X_lha, _ = load_and_preprocess(session_num, 'LHA')
    n_bins = min(X_aca.shape[0], X_lha.shape[0])
    X_aca, X_lha = X_aca[:n_bins], X_lha[:n_bins]
    bin_centers = bin_centers[:n_bins]

    # PCA + CCA
    pca_aca = PCA(n_components=K_PCS['ACA'])
    pca_lha = PCA(n_components=K_PCS['LHA'])
    X_aca_pcs = pca_aca.fit_transform(X_aca)
    X_lha_pcs = pca_lha.fit_transform(X_lha)
    cca = CCA(n_components=N_CCA, max_iter=1000)
    X_c, Y_c = cca.fit_transform(X_aca_pcs, X_lha_pcs)
    canon_corrs = [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] for i in range(N_CCA)]

    # Load behavior
    behav_df = load_behavioral(session_num)
    indices = align_to_bins(behav_df, bin_centers)

    # === Extract ALL behavioral variables ===
    behav_vars = {}

    # Continuous
    behav_vars['velocity'] = behav_df['Velocity(Center-point)'].values[indices].astype(float)
    behav_vars['y_position'] = behav_df['Y center'].values[indices].astype(float)
    behav_vars['x_position'] = behav_df['X center'].values[indices].astype(float)
    direction = behav_df['Direction'].values[indices].astype(float)
    behav_vars['heading_sin'] = np.sin(np.deg2rad(direction))
    behav_vars['heading_cos'] = np.cos(np.deg2rad(direction))

    # Distance to zones
    dist_cols = [c for c in behav_df.columns if 'Distance' in str(c) and 'Zone' in str(c)]
    for c in dist_cols:
        name = str(c).replace('Distance to zone(', '').replace(' / center-point)', '')
        name = name.strip().replace(' ', '_').lower()
        behav_vars[f'dist_{name}'] = behav_df[c].values[indices].astype(float)

    # Compartment as numeric
    pot_zone_cols = [c for c in behav_df.columns
                     if 'Zone(Pot-' in str(c) and ' zone' not in c
                     and 'Distance' not in str(c)]
    home_col = [c for c in behav_df.columns if 'Zone(Home' in str(c)
                and 'corner' not in c and 'Distance' not in str(c)]
    ladder_col = [c for c in behav_df.columns if 'Zone(ladder' in str(c)
                  and 'Distance' not in str(c)]
    compartment = np.full(n_bins, 'Arena', dtype=object)
    if home_col:
        compartment[behav_df[home_col[0]].values[indices] == 1] = 'Home'
    if ladder_col:
        compartment[behav_df[ladder_col[0]].values[indices] == 1] = 'Ladder'
    if pot_zone_cols:
        at_pot = np.zeros(n_bins, dtype=bool)
        for c in pot_zone_cols:
            at_pot |= (behav_df[c].values[indices] == 1)
        compartment[at_pot] = 'AtPot'

    # Binary: at_home, at_pot, at_ladder, at_arena
    behav_vars['at_home'] = (compartment == 'Home').astype(float)
    behav_vars['at_ladder'] = (compartment == 'Ladder').astype(float)
    behav_vars['at_arena'] = (compartment == 'Arena').astype(float)
    behav_vars['at_pot'] = (compartment == 'AtPot').astype(float)

    # Scored behaviors
    skip_prefixes = ['Trial time', 'Recording', 'X ', 'Y ', 'Area', 'Elongation',
                     'Direction', 'Distance', 'Velocity', 'Zone(', 'Result']
    for col in behav_df.columns:
        if col is None or str(col) == 'nan':
            continue
        col_str = str(col)
        if any(col_str.startswith(p) for p in skip_prefixes):
            continue
        vals = behav_df[col].values
        if (vals == 1).sum() < MIN_EVENTS:
            continue
        clean_name = col_str.strip().replace(' ', '_').lower()
        behav_vars[clean_name] = vals[indices].astype(float)

    # === Correlate each CC with each behavioral variable ===
    print(f"\n{'Variable':<40} {'CC1-ACA':>10} {'CC1-LHA':>10} {'CC2-ACA':>10} {'CC2-LHA':>10}")
    print("-" * 82)

    results = {}
    for vname, vals in behav_vars.items():
        valid = ~np.isnan(vals)
        if valid.sum() < 100:
            continue
        row = {}
        for cc_i in range(min(3, N_CCA)):
            for src, cc_vals in [('ACA', X_c), ('LHA', Y_c)]:
                r, p = pearsonr(cc_vals[valid, cc_i], vals[valid])
                row[f'CC{cc_i+1}_{src}_r'] = r
                row[f'CC{cc_i+1}_{src}_p'] = p
        results[vname] = row

        # Print top 2 CCs
        print(f"{vname:<40} {row['CC1_ACA_r']:>+10.3f} {row['CC1_LHA_r']:>+10.3f} "
              f"{row['CC2_ACA_r']:>+10.3f} {row['CC2_LHA_r']:>+10.3f}")

    # Sort by |CC1_ACA_r|
    print(f"\n--- Sorted by |CC1 ACA correlation| ---")
    sorted_vars = sorted(results.items(), key=lambda x: abs(x[1]['CC1_ACA_r']), reverse=True)
    print(f"\n{'Variable':<40} {'CC1-ACA':>10} {'CC1-LHA':>10}")
    print("-" * 62)
    for vname, row in sorted_vars:
        print(f"{vname:<40} {row['CC1_ACA_r']:>+10.3f} {row['CC1_LHA_r']:>+10.3f}")

    # Sort by |CC2_ACA_r|
    print(f"\n--- Sorted by |CC2 ACA correlation| ---")
    sorted_vars2 = sorted(results.items(), key=lambda x: abs(x[1]['CC2_ACA_r']), reverse=True)
    print(f"\n{'Variable':<40} {'CC2-ACA':>10} {'CC2-LHA':>10}")
    print("-" * 62)
    for vname, row in sorted_vars2:
        print(f"{vname:<40} {row['CC2_ACA_r']:>+10.3f} {row['CC2_LHA_r']:>+10.3f}")

    # === Figure: CC1 time course with behavioral overlays ===
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 2, figure=fig, hspace=0.4, wspace=0.3,
                  height_ratios=[2, 2, 1.5, 1.5])

    time_min = bin_centers / 60

    # Panel 1: CC1 time course colored by compartment
    ax = fig.add_subplot(gs[0, :])
    # Plot as background
    comp_colors_map = {'Home': 'blue', 'Ladder': 'green', 'Arena': 'lightgray', 'AtPot': 'red'}
    for label, color in comp_colors_map.items():
        mask = compartment == label
        if mask.sum() > 0:
            ax.scatter(time_min[mask], X_c[mask, 0], c=color, s=0.3, alpha=0.4,
                       rasterized=True, label=label)
    ax.set_ylabel('ACA CC1')
    ax.set_title(f'CC1 (r={canon_corrs[0]:.3f}) colored by compartment', fontweight='bold')
    ax.legend(markerscale=10, fontsize=9, loc='upper right')

    # Panel 2: CC1 with feeding/digging highlighted
    ax = fig.add_subplot(gs[1, :])
    ax.plot(time_min, X_c[:, 0], color='lightgray', lw=0.3, alpha=0.5, zorder=1)
    feed = behav_vars.get('feeding', np.zeros(n_bins))
    dig = behav_vars.get('digging_sand', np.zeros(n_bins))
    feed_mask = feed == 1
    dig_mask = dig == 1
    if feed_mask.sum() > 0:
        ax.scatter(time_min[feed_mask], X_c[feed_mask, 0], c='darkorange', s=2,
                   alpha=0.7, label=f'Feeding ({feed_mask.sum()})', zorder=3, rasterized=True)
    if dig_mask.sum() > 0:
        ax.scatter(time_min[dig_mask], X_c[dig_mask, 0], c='purple', s=2,
                   alpha=0.7, label=f'Digging ({dig_mask.sum()})', zorder=3, rasterized=True)
    ax.set_ylabel('ACA CC1')
    ax.set_title('CC1 with feeding & digging', fontweight='bold')
    ax.legend(markerscale=5, fontsize=9, loc='upper right')

    # Panel 3: Top behavioral correlates of CC1 (bar chart)
    ax = fig.add_subplot(gs[2, 0])
    top_cc1 = sorted_vars[:12]
    names = [v[0] for v in top_cc1]
    r_aca = [v[1]['CC1_ACA_r'] for v in top_cc1]
    r_lha = [v[1]['CC1_LHA_r'] for v in top_cc1]
    y = np.arange(len(names))
    ax.barh(y - 0.2, r_aca, 0.35, color='steelblue', label='ACA CC1')
    ax.barh(y + 0.2, r_lha, 0.35, color='darkorange', label='LHA CC1')
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Pearson r')
    ax.set_title('CC1 behavioral correlates', fontweight='bold')
    ax.legend(fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color='black', lw=0.5)

    # Panel 4: Top behavioral correlates of CC2 (bar chart)
    ax = fig.add_subplot(gs[2, 1])
    top_cc2 = sorted_vars2[:12]
    names2 = [v[0] for v in top_cc2]
    r_aca2 = [v[1]['CC2_ACA_r'] for v in top_cc2]
    r_lha2 = [v[1]['CC2_LHA_r'] for v in top_cc2]
    y2 = np.arange(len(names2))
    ax.barh(y2 - 0.2, r_aca2, 0.35, color='steelblue', label='ACA CC2')
    ax.barh(y2 + 0.2, r_lha2, 0.35, color='darkorange', label='LHA CC2')
    ax.set_yticks(y2)
    ax.set_yticklabels(names2, fontsize=8)
    ax.set_xlabel('Pearson r')
    ax.set_title('CC2 behavioral correlates', fontweight='bold')
    ax.legend(fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color='black', lw=0.5)

    # Panel 5: CC1 vs y_position
    ax = fig.add_subplot(gs[3, 0])
    ypos = behav_vars['y_position']
    valid = ~np.isnan(ypos)
    sc = ax.scatter(X_c[valid, 0], ypos[valid], c=bin_centers[valid], s=0.3,
                    alpha=0.2, cmap='viridis', rasterized=True)
    ax.set_xlabel('ACA CC1')
    ax.set_ylabel('Y position (cm)')
    r_ypos = results['y_position']['CC1_ACA_r']
    ax.set_title(f'CC1 vs Y position (r={r_ypos:.3f})', fontweight='bold')
    plt.colorbar(sc, ax=ax, label='Time (s)')

    # Panel 6: CC1 vs velocity
    ax = fig.add_subplot(gs[3, 1])
    vel = behav_vars['velocity']
    valid_v = ~np.isnan(vel)
    sc = ax.scatter(X_c[valid_v, 0], vel[valid_v], c=bin_centers[valid_v], s=0.3,
                    alpha=0.2, cmap='viridis', rasterized=True)
    ax.set_xlabel('ACA CC1')
    ax.set_ylabel('Velocity (cm/s)')
    r_vel = results['velocity']['CC1_ACA_r']
    ax.set_title(f'CC1 vs Velocity (r={r_vel:.3f})', fontweight='bold')
    plt.colorbar(sc, ax=ax, label='Time (s)')

    fig.suptitle(f'CCA Interpretation -- S{session_num}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fp = figdir / f"S{session_num}_cca_interpretation.png"
    plt.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved {fp}")
    print("Done.")


if __name__ == '__main__':
    main()
