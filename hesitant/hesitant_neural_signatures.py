"""
Neural signatures of hesitant exploration bouts — Session 1.

Approach 1: Population firing rate comparison
  - Mean firing rate (per unit, population) during hesitant vs committed vs task-engaged bouts
  - Separate for LHA and RSP

Approach 3: GRU-ODE latent dynamics
  - Trajectory projections (PCA) colored by bout type
  - Trajectory speed in latent space
  - Distance from session-mean hidden state

Excursion groups:
  A) Hesitant: no feed/dig, farthest != Pot, reversals >= 1, duration >= 2s
  B) Committed exploratory: reaches arena/pots, no feed/dig, similar duration to hesitant
  C) Task-engaged: feeding or digging excursions (reference)
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spikeinterface.extractors as se
from scipy import stats as sp_stats
from sklearn.decomposition import PCA
import warnings
import sys

warnings.filterwarnings('ignore')

FS = 30000
BIN_SIZE_MS = 100  # 100ms bins for firing rate analysis
BIN_SAMPLES = int(BIN_SIZE_MS * FS / 1000)  # 3000 samples
LHA_DEPTH_MAX = 1300
RSP_DEPTH_MIN = 1300

with open("paths.yaml") as f:
    paths_config = yaml.safe_load(f)


# =============================================================================
# DATA LOADING
# =============================================================================

def get_good_units_by_region(sorted_path_obj):
    ci = sorted_path_obj / "cluster_info.tsv"
    if not ci.exists():
        return np.array([]), np.array([])
    df = pd.read_csv(ci, sep='\t')
    if 'depth' not in df.columns:
        return np.array([]), np.array([])
    label_col = None
    if 'group' in df.columns and df['group'].eq('good').any():
        label_col = 'group'
    elif 'KSLabel' in df.columns:
        label_col = 'KSLabel'
    if label_col is None:
        return np.array([]), np.array([])
    good = df[df[label_col] == 'good']
    lha_ids = good[good['depth'] < LHA_DEPTH_MAX]['cluster_id'].values
    rsp_ids = good[good['depth'] >= RSP_DEPTH_MIN]['cluster_id'].values
    return lha_ids, rsp_ids


def bin_spike_trains_100ms(sorting, unit_ids):
    """Bin spikes at 100ms, return raw counts (not z-scored) and time array."""
    spike_trains = {}
    all_min, all_max = np.inf, 0
    for uid in unit_ids:
        st = sorting.get_unit_spike_train(uid)
        spike_trains[uid] = st
        if len(st) > 0:
            all_min = min(all_min, np.min(st))
            all_max = max(all_max, np.max(st))

    n_bins = int((all_max - all_min) / BIN_SAMPLES) + 1
    data = np.zeros((n_bins, len(unit_ids)), dtype=np.float32)

    for i, uid in enumerate(unit_ids):
        st = spike_trains[uid]
        if len(st) > 0:
            b = ((st - all_min) // BIN_SAMPLES).astype(int)
            b = b[(b >= 0) & (b < n_bins)]
            np.add.at(data[:, i], b, 1)

    # Convert counts to firing rate (Hz): counts / bin_size_in_seconds
    firing_rates = data / (BIN_SIZE_MS / 1000.0)

    # Time array in seconds from recording start
    time_sec = (np.arange(n_bins) * BIN_SIZE_MS / 1000) + (all_min / FS)

    return firing_rates, time_sec, n_bins, all_min


def time_to_spike_bin(t_sec, all_min, bin_size_ms=100):
    """Convert behavior time (seconds) to spike data bin index."""
    offset_sec = all_min / FS
    return int((t_sec - offset_sec) / (bin_size_ms / 1000.0))


def time_to_gru_bin(t_sec, all_min, bin_size_ms=10):
    """Convert behavior time (seconds) to GRU-ODE hidden state bin index."""
    offset_sec = all_min / FS
    return int((t_sec - offset_sec) / (bin_size_ms / 1000.0))


# =============================================================================
# DEFINE EXCURSION GROUPS
# =============================================================================

def define_groups(exc_df):
    """Define hesitant, committed-exploratory, and task-engaged groups."""
    s1 = exc_df[exc_df['session'] == 1].copy()

    # Group A: Hesitant
    hesitant = s1[
        (s1['feeding_bins'] == 0) &
        (s1['digging_bins'] == 0) &
        (s1['farthest_zone'] != 'Pot') &
        (s1['reversals'] >= 1) &
        (s1['duration'] >= 2.0)
    ].copy()
    hesitant['group'] = 'Hesitant'

    # Duration stats for hesitant bouts (for matching)
    hes_dur_min = hesitant['duration'].min()
    hes_dur_max = hesitant['duration'].max()
    hes_dur_median = hesitant['duration'].median()

    # Group C: Task-engaged (feeding or digging)
    task = s1[
        (s1['feeding_bins'] > 0) | (s1['digging_bins'] > 0)
    ].copy()
    task['group'] = 'Task-engaged'

    # Group B: Committed exploratory — reaches arena/pots, no feed/dig,
    # duration >= hesitant min but < task median (if available)
    non_hes_non_task = s1[
        ~s1.index.isin(hesitant.index) &
        ~s1.index.isin(task.index)
    ].copy()

    committed = non_hes_non_task[
        (non_hes_non_task['reached_arena'] == True) &
        (non_hes_non_task['duration'] >= hes_dur_min)
    ].copy()

    # If task group exists, cap duration at task median
    if len(task) > 0:
        task_dur_median = task['duration'].median()
        committed = committed[committed['duration'] <= task_dur_median]

    committed['group'] = 'Committed'

    return hesitant, committed, task, s1


# =============================================================================
# APPROACH 1: POPULATION FIRING RATES
# =============================================================================

def extract_bout_firing_rates(fr_data, time_sec, all_min, bouts_df):
    """Extract mean firing rate for each bout. Returns (n_bouts, n_units)."""
    bout_rates = []
    valid_bouts = []

    for _, row in bouts_df.iterrows():
        b_start = time_to_spike_bin(row['start_time'], all_min, BIN_SIZE_MS)
        b_end = time_to_spike_bin(row['end_time'], all_min, BIN_SIZE_MS)

        # Clamp to valid range
        b_start = max(0, b_start)
        b_end = min(fr_data.shape[0], b_end)

        if b_end - b_start < 2:
            continue

        mean_fr = fr_data[b_start:b_end, :].mean(axis=0)
        bout_rates.append(mean_fr)
        valid_bouts.append(row)

    if len(bout_rates) == 0:
        return np.array([]), pd.DataFrame()

    return np.array(bout_rates), pd.DataFrame(valid_bouts)


def plot_firing_rates(lha_rates, rsp_rates, group_labels, group_names, colors):
    """Plot population firing rate comparisons."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Approach 1: Population Firing Rates During Excursion Types\n'
                 'Session 1 (Fed, Exploration)', fontsize=13, fontweight='bold')

    for row_idx, (region_name, rates_by_group) in enumerate(
            [('LHA', lha_rates), ('RSP', rsp_rates)]):

        # Panel 1: Mean population FR per bout (boxplot)
        ax = axes[row_idx, 0]
        pop_means = []
        labels = []
        for gname in group_names:
            if gname in rates_by_group and len(rates_by_group[gname]) > 0:
                pop_means.append(rates_by_group[gname].mean(axis=1))
                labels.append(gname)
        if len(pop_means) > 0:
            bp = ax.boxplot(pop_means, labels=labels, patch_artist=True,
                           showfliers=True, flierprops={'markersize': 3})
            for patch, lbl in zip(bp['boxes'], labels):
                patch.set_facecolor(colors[lbl])
                patch.set_alpha(0.6)
            # Overlay individual points
            for i, (data, lbl) in enumerate(zip(pop_means, labels)):
                x = np.random.normal(i + 1, 0.04, len(data))
                ax.scatter(x, data, c=colors[lbl], alpha=0.4, s=10, zorder=3)
        ax.set_ylabel('Mean Firing Rate (Hz)')
        ax.set_title(f'{region_name}: Population Mean FR')

        # Panel 2: Per-unit mean FR across groups (heatmap-style)
        ax = axes[row_idx, 1]
        group_unit_means = []
        group_labels_list = []
        for gname in group_names:
            if gname in rates_by_group and len(rates_by_group[gname]) > 0:
                group_unit_means.append(rates_by_group[gname].mean(axis=0))
                group_labels_list.append(gname)
        if len(group_unit_means) > 0:
            n_units = len(group_unit_means[0])
            x = np.arange(n_units)
            for i, (means, lbl) in enumerate(zip(group_unit_means, group_labels_list)):
                ax.bar(x + i * 0.25, means, width=0.24, color=colors[lbl],
                       alpha=0.7, label=lbl)
            ax.set_xlabel('Unit index')
            ax.set_ylabel('Mean FR (Hz)')
            ax.set_title(f'{region_name}: Per-Unit Mean FR')
            ax.legend(fontsize=8)

        # Panel 3: Statistical comparison (Mann-Whitney U)
        ax = axes[row_idx, 2]
        text_lines = [f'{region_name} — Statistical Tests\n']
        for i, g1 in enumerate(group_names):
            for g2 in group_names[i+1:]:
                if g1 in rates_by_group and g2 in rates_by_group:
                    r1 = rates_by_group[g1]
                    r2 = rates_by_group[g2]
                    if len(r1) > 0 and len(r2) > 0:
                        pop1 = r1.mean(axis=1)
                        pop2 = r2.mean(axis=1)
                        stat, pval = sp_stats.mannwhitneyu(pop1, pop2,
                                                           alternative='two-sided')
                        sig = '*' if pval < 0.05 else 'ns'
                        text_lines.append(
                            f'{g1} vs {g2}:\n'
                            f'  n={len(pop1)} vs {len(pop2)}\n'
                            f'  median={np.median(pop1):.2f} vs {np.median(pop2):.2f}\n'
                            f'  U={stat:.0f}, p={pval:.4f} {sig}\n'
                        )
        ax.text(0.05, 0.95, '\n'.join(text_lines), transform=ax.transAxes,
                verticalalignment='top', fontsize=9, fontfamily='monospace')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('figures/hesitant_neural_firing_rates.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print("  Saved: figures/hesitant_neural_firing_rates.png")


# =============================================================================
# APPROACH 3: GRU-ODE LATENT DYNAMICS
# =============================================================================

def extract_bout_trajectories(hidden_states, all_min, bouts_df):
    """Extract GRU-ODE hidden state trajectories for each bout."""
    trajectories = []
    valid_bouts = []
    n_total = hidden_states.shape[0]

    for _, row in bouts_df.iterrows():
        b_start = time_to_gru_bin(row['start_time'], all_min, 10)
        b_end = time_to_gru_bin(row['end_time'], all_min, 10)

        b_start = max(0, b_start)
        b_end = min(n_total, b_end)

        if b_end - b_start < 10:  # at least 100ms
            continue

        traj = hidden_states[b_start:b_end, :]
        trajectories.append(traj)
        valid_bouts.append(row)

    return trajectories, pd.DataFrame(valid_bouts)


def compute_trajectory_stats(trajectories):
    """Compute speed, curvature, mean position for each trajectory."""
    stats = []
    for traj in trajectories:
        # Speed: norm of difference between consecutive hidden states
        if len(traj) < 2:
            stats.append({'speed_mean': np.nan, 'speed_std': np.nan,
                         'total_distance': np.nan, 'displacement': np.nan,
                         'tortuosity': np.nan, 'mean_norm': np.nan})
            continue

        diffs = np.diff(traj, axis=0)
        speeds = np.linalg.norm(diffs, axis=1)
        total_dist = np.sum(speeds)
        displacement = np.linalg.norm(traj[-1] - traj[0])
        tortuosity = total_dist / displacement if displacement > 1e-8 else np.nan
        mean_norm = np.mean(np.linalg.norm(traj, axis=1))

        stats.append({
            'speed_mean': np.mean(speeds),
            'speed_std': np.std(speeds),
            'total_distance': total_dist,
            'displacement': displacement,
            'tortuosity': tortuosity,
            'mean_norm': mean_norm,
        })
    return pd.DataFrame(stats)


def plot_latent_dynamics(trajectories_by_group, stats_by_group,
                         hidden_states, all_min, group_names, colors,
                         region_name, exc_df_s1):
    """Plot GRU-ODE trajectory analysis for one region."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Approach 3: GRU-ODE Latent Dynamics — {region_name}\n'
                 f'Session 1 (Fed, Exploration)', fontsize=13, fontweight='bold')

    # Fit PCA on full session hidden states
    pca = PCA(n_components=3)
    pca.fit(hidden_states)

    # Panel 1: PCA trajectories colored by group
    ax = axes[0, 0]
    for gname in group_names:
        if gname not in trajectories_by_group:
            continue
        trajs = trajectories_by_group[gname]
        for traj in trajs:
            proj = pca.transform(traj)
            ax.plot(proj[:, 0], proj[:, 1], color=colors[gname],
                    alpha=0.3, linewidth=0.5)
            ax.scatter(proj[0, 0], proj[0, 1], color=colors[gname],
                      s=8, zorder=4, marker='o')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('Trajectories in PC space')
    # Legend
    for gname in group_names:
        ax.plot([], [], color=colors[gname], linewidth=2, label=gname)
    ax.legend(fontsize=8)

    # Panel 2: Mean trajectory per group in PC space
    ax = axes[0, 1]
    for gname in group_names:
        if gname not in trajectories_by_group:
            continue
        trajs = trajectories_by_group[gname]
        # Resample all trajectories to same length, then average
        n_resample = 50
        resampled = []
        for traj in trajs:
            proj = pca.transform(traj)
            indices = np.linspace(0, len(proj) - 1, n_resample).astype(int)
            resampled.append(proj[indices, :2])
        if len(resampled) > 0:
            mean_traj = np.mean(resampled, axis=0)
            sem_traj = np.std(resampled, axis=0) / np.sqrt(len(resampled))
            ax.plot(mean_traj[:, 0], mean_traj[:, 1], color=colors[gname],
                    linewidth=2, label=f'{gname} (n={len(trajs)})')
            ax.scatter(mean_traj[0, 0], mean_traj[0, 1], color=colors[gname],
                      s=40, zorder=4, marker='o', edgecolors='black')
            ax.scatter(mean_traj[-1, 0], mean_traj[-1, 1], color=colors[gname],
                      s=40, zorder=4, marker='s', edgecolors='black')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('Mean trajectory (o=start, □=end)')
    ax.legend(fontsize=8)

    # Panel 3: Trajectory speed boxplot
    ax = axes[0, 2]
    speed_data = []
    speed_labels = []
    for gname in group_names:
        if gname in stats_by_group and len(stats_by_group[gname]) > 0:
            speeds = stats_by_group[gname]['speed_mean'].dropna().values
            if len(speeds) > 0:
                speed_data.append(speeds)
                speed_labels.append(gname)
    if len(speed_data) > 0:
        bp = ax.boxplot(speed_data, labels=speed_labels, patch_artist=True,
                       showfliers=True, flierprops={'markersize': 3})
        for patch, lbl in zip(bp['boxes'], speed_labels):
            patch.set_facecolor(colors[lbl])
            patch.set_alpha(0.6)
        for i, (data, lbl) in enumerate(zip(speed_data, speed_labels)):
            x = np.random.normal(i + 1, 0.04, len(data))
            ax.scatter(x, data, c=colors[lbl], alpha=0.4, s=10, zorder=3)
    ax.set_ylabel('Mean latent speed')
    ax.set_title('Trajectory speed (||dh/dt||)')

    # Panel 4: Tortuosity boxplot
    ax = axes[1, 0]
    tort_data = []
    tort_labels = []
    for gname in group_names:
        if gname in stats_by_group and len(stats_by_group[gname]) > 0:
            tort = stats_by_group[gname]['tortuosity'].dropna().values
            if len(tort) > 0:
                tort_data.append(tort)
                tort_labels.append(gname)
    if len(tort_data) > 0:
        bp = ax.boxplot(tort_data, labels=tort_labels, patch_artist=True,
                       showfliers=True, flierprops={'markersize': 3})
        for patch, lbl in zip(bp['boxes'], tort_labels):
            patch.set_facecolor(colors[lbl])
            patch.set_alpha(0.6)
        for i, (data, lbl) in enumerate(zip(tort_data, tort_labels)):
            x = np.random.normal(i + 1, 0.04, len(data))
            ax.scatter(x, data, c=colors[lbl], alpha=0.4, s=10, zorder=3)
    ax.set_ylabel('Tortuosity (path/displacement)')
    ax.set_title('Trajectory tortuosity')

    # Panel 5: Mean hidden state norm
    ax = axes[1, 1]
    norm_data = []
    norm_labels = []
    for gname in group_names:
        if gname in stats_by_group and len(stats_by_group[gname]) > 0:
            norms = stats_by_group[gname]['mean_norm'].dropna().values
            if len(norms) > 0:
                norm_data.append(norms)
                norm_labels.append(gname)
    if len(norm_data) > 0:
        bp = ax.boxplot(norm_data, labels=norm_labels, patch_artist=True,
                       showfliers=True, flierprops={'markersize': 3})
        for patch, lbl in zip(bp['boxes'], norm_labels):
            patch.set_facecolor(colors[lbl])
            patch.set_alpha(0.6)
        for i, (data, lbl) in enumerate(zip(norm_data, norm_labels)):
            x = np.random.normal(i + 1, 0.04, len(data))
            ax.scatter(x, data, c=colors[lbl], alpha=0.4, s=10, zorder=3)
    ax.set_ylabel('Mean ||h||')
    ax.set_title('Hidden state magnitude')

    # Panel 6: Stats text
    ax = axes[1, 2]
    text_lines = [f'{region_name} — Latent Dynamics Stats\n']
    for metric in ['speed_mean', 'tortuosity', 'mean_norm']:
        text_lines.append(f'--- {metric} ---')
        for i, g1 in enumerate(group_names):
            for g2 in group_names[i+1:]:
                if g1 in stats_by_group and g2 in stats_by_group:
                    v1 = stats_by_group[g1][metric].dropna().values
                    v2 = stats_by_group[g2][metric].dropna().values
                    if len(v1) > 1 and len(v2) > 1:
                        stat, pval = sp_stats.mannwhitneyu(
                            v1, v2, alternative='two-sided')
                        sig = '*' if pval < 0.05 else 'ns'
                        text_lines.append(
                            f'  {g1} vs {g2}: '
                            f'med={np.median(v1):.4f} vs {np.median(v2):.4f}, '
                            f'p={pval:.4f} {sig}'
                        )
        text_lines.append('')
    ax.text(0.02, 0.95, '\n'.join(text_lines), transform=ax.transAxes,
            verticalalignment='top', fontsize=8, fontfamily='monospace')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'figures/hesitant_neural_latent_{region_name.lower()}.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: figures/hesitant_neural_latent_{region_name.lower()}.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("  Neural Signatures of Hesitant Exploration — Session 1")
    print("  Approach 1: Population firing rates")
    print("  Approach 3: GRU-ODE latent dynamics")
    print("=" * 80)
    sys.stdout.flush()

    # --- Load excursion data and define groups ---
    exc_df = pd.read_csv('data/excursion_features_all_sessions.csv')
    hesitant, committed, task, s1_all = define_groups(exc_df)

    print(f"\n  Excursion groups (Session 1):")
    print(f"    Hesitant:    {len(hesitant)} bouts, "
          f"dur median={hesitant['duration'].median():.1f}s "
          f"(range {hesitant['duration'].min():.1f}-{hesitant['duration'].max():.1f})")
    print(f"    Committed:   {len(committed)} bouts, "
          f"dur median={committed['duration'].median():.1f}s "
          f"(range {committed['duration'].min():.1f}-{committed['duration'].max():.1f})")
    print(f"    Task-engaged:{len(task)} bouts, "
          f"dur median={task['duration'].median():.1f}s "
          f"(range {task['duration'].min():.1f}-{task['duration'].max():.1f})")
    sys.stdout.flush()

    # --- Load spike data ---
    print("\n  Loading spike data for Session 1...")
    sys.stdout.flush()
    sp = paths_config['single_probe']['coordinates_1']['mouse01']['sessions']
    sc = sp['session_1']
    sorted_path = Path(sc['sorted'])

    sorting = se.read_kilosort(sorted_path)
    lha_ids, rsp_ids = get_good_units_by_region(sorted_path)
    print(f"    LHA: {len(lha_ids)} good units")
    print(f"    RSP: {len(rsp_ids)} good units")

    # Bin at 100ms for firing rate analysis
    print("  Binning spikes at 100ms...")
    sys.stdout.flush()
    lha_fr, lha_time, lha_nbins, lha_allmin = bin_spike_trains_100ms(sorting, lha_ids)
    rsp_fr, rsp_time, rsp_nbins, rsp_allmin = bin_spike_trains_100ms(sorting, rsp_ids)
    print(f"    LHA: {lha_nbins} bins, {lha_fr.shape[1]} units, "
          f"offset={lha_allmin/FS:.3f}s")
    print(f"    RSP: {rsp_nbins} bins, {rsp_fr.shape[1]} units, "
          f"offset={rsp_allmin/FS:.3f}s")
    sys.stdout.flush()

    # --- Load GRU-ODE hidden states ---
    print("  Loading GRU-ODE hidden states (10ms)...")
    h_lha = np.load('data/gru_ode_10ms_hidden_lha_s1.npy')
    h_rsp = np.load('data/gru_ode_10ms_hidden_rsp_s1.npy')
    print(f"    LHA: {h_lha.shape}")
    print(f"    RSP: {h_rsp.shape}")
    sys.stdout.flush()

    # =================================================================
    # APPROACH 1: Population firing rates
    # =================================================================
    print("\n  === APPROACH 1: Population Firing Rates ===")
    sys.stdout.flush()

    group_names = ['Hesitant', 'Committed', 'Task-engaged']
    colors = {'Hesitant': '#E53935', 'Committed': '#1E88E5', 'Task-engaged': '#43A047'}
    groups = {'Hesitant': hesitant, 'Committed': committed, 'Task-engaged': task}

    lha_rates = {}
    rsp_rates = {}
    for gname, gdf in groups.items():
        lr, lv = extract_bout_firing_rates(lha_fr, lha_time, lha_allmin, gdf)
        rr, rv = extract_bout_firing_rates(rsp_fr, rsp_time, rsp_allmin, gdf)
        if len(lr) > 0:
            lha_rates[gname] = lr
        if len(rr) > 0:
            rsp_rates[gname] = rr
        print(f"    {gname}: LHA {lr.shape if len(lr)>0 else '(0)'}, "
              f"RSP {rr.shape if len(rr)>0 else '(0)'}")

    plot_firing_rates(lha_rates, rsp_rates, groups, group_names, colors)

    # Print summary numbers
    print("\n  Population mean FR (Hz) — median [IQR]:")
    for region_name, rates in [('LHA', lha_rates), ('RSP', rsp_rates)]:
        print(f"    {region_name}:")
        for gname in group_names:
            if gname in rates and len(rates[gname]) > 0:
                pop = rates[gname].mean(axis=1)
                q25, q50, q75 = np.percentile(pop, [25, 50, 75])
                print(f"      {gname}: {q50:.2f} [{q25:.2f}-{q75:.2f}] "
                      f"(n={len(pop)})")

    sys.stdout.flush()

    # =================================================================
    # APPROACH 3: GRU-ODE latent dynamics
    # =================================================================
    print("\n  === APPROACH 3: GRU-ODE Latent Dynamics ===")
    sys.stdout.flush()

    # Use LHA all_min for GRU-ODE alignment (both LHA and RSP models
    # were trained on data starting from their respective all_min)
    for region_name, h_states, allmin in [('LHA', h_lha, lha_allmin),
                                           ('RSP', h_rsp, rsp_allmin)]:
        print(f"\n  {region_name}:")
        trajs_by_group = {}
        stats_by_group = {}

        for gname, gdf in groups.items():
            trajs, valid = extract_bout_trajectories(h_states, allmin, gdf)
            if len(trajs) > 0:
                trajs_by_group[gname] = trajs
                tstats = compute_trajectory_stats(trajs)
                stats_by_group[gname] = tstats
                print(f"    {gname}: {len(trajs)} trajectories")
                print(f"      speed: {tstats['speed_mean'].median():.4f} median")
                print(f"      tortuosity: {tstats['tortuosity'].median():.2f} median")
                print(f"      ||h||: {tstats['mean_norm'].median():.3f} median")

        plot_latent_dynamics(trajs_by_group, stats_by_group,
                            h_states, allmin, group_names, colors,
                            region_name, s1_all)

    sys.stdout.flush()

    # =================================================================
    # Save summary
    # =================================================================
    summary_rows = []
    for region_name, rates, h_states, allmin in [
            ('LHA', lha_rates, h_lha, lha_allmin),
            ('RSP', rsp_rates, h_rsp, rsp_allmin)]:
        for gname, gdf in groups.items():
            row = {'region': region_name, 'group': gname, 'n_bouts': len(gdf)}
            if gname in rates and len(rates[gname]) > 0:
                pop = rates[gname].mean(axis=1)
                row['fr_median'] = np.median(pop)
                row['fr_q25'] = np.percentile(pop, 25)
                row['fr_q75'] = np.percentile(pop, 75)

            trajs, _ = extract_bout_trajectories(h_states, allmin, gdf)
            if len(trajs) > 0:
                tstats = compute_trajectory_stats(trajs)
                row['latent_speed_median'] = tstats['speed_mean'].median()
                row['latent_tortuosity_median'] = tstats['tortuosity'].median()
                row['latent_norm_median'] = tstats['mean_norm'].median()

            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv('data/hesitant_neural_signatures_summary.csv', index=False)
    print(f"\n  Saved: data/hesitant_neural_signatures_summary.csv")
    print("\nDone!")


if __name__ == "__main__":
    main()
