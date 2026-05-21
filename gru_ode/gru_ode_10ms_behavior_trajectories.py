"""
Behavior-Triggered Latent Trajectory Visualization
====================================================
Visualize how GRU-ODE latent dynamics evolve around specific behavioral events.

Two visualization types:
1. Peri-event trajectories: aligned to behavior onset, -5s to +10s
2. Full-session state-space: trajectory colored by active behavior
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from scipy.stats import mannwhitneyu
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

SUBSAMPLE_RATIO = 10  # 10ms -> 100ms
PRE_BINS = 50         # 5s before onset (at 100ms)
POST_BINS = 100       # 10s after onset (at 100ms)
MIN_BOUT_BINS = 3     # Minimum 300ms bout duration

with open("paths.yaml") as f:
    paths_config = yaml.safe_load(f)

SESSION_INFO = {
    1: {'state': 'Fed', 'phase': 'Exploration'},
    2: {'state': 'Fed', 'phase': 'Foraging'},
    3: {'state': 'Fed', 'phase': 'Exploration'},
    4: {'state': 'Fed', 'phase': 'Foraging'},
    5: {'state': 'Fasted', 'phase': 'Exploration'},
    6: {'state': 'Fasted', 'phase': 'Foraging'},
    7: {'state': 'Fasted', 'phase': 'Exploration'},
    8: {'state': 'Fasted', 'phase': 'Foraging'},
}

BEHAVIORS = [
    'Feeding',
    'Quick and hasty exploration at home',
    'Quick one loop at home',
    'Longer exploration at home',
    'Transition wall exploration',
    'Hiding in corners',
    'Incomplete home return',
    'Contemplation at T-zone',
]

SHORT_NAMES = {
    'Feeding': 'Feeding',
    'Quick and hasty exploration at home': 'Quick Hasty Exp',
    'Quick one loop at home': 'Quick Loop',
    'Longer exploration at home': 'Long Exp Home',
    'Transition wall exploration': 'Trans Wall Exp',
    'Hiding in corners': 'Hiding Corners',
    'Incomplete home return': 'Incomplete Return',
    'Contemplation at T-zone': 'Contemp T-zone',
}

BEHAVIOR_COLORS = {
    'Feeding': '#E53935',
    'Quick and hasty exploration at home': '#1E88E5',
    'Quick one loop at home': '#43A047',
    'Longer exploration at home': '#8E24AA',
    'Transition wall exploration': '#FB8C00',
    'Hiding in corners': '#00ACC1',
    'Incomplete home return': '#D81B60',
    'Contemplation at T-zone': '#6D4C41',
}

FIGURES_DIR = Path("figures")
DATA_DIR = Path("data")


# =============================================================================
# DATA LOADING
# =============================================================================

def load_hidden_states(region, session_num):
    """Load cached 10ms hidden states, subsample to 100ms."""
    path = DATA_DIR / f"gru_ode_10ms_hidden_{region}_s{session_num}.npy"
    h_10ms = np.load(path)
    return h_10ms[::SUBSAMPLE_RATIO]  # (T_100ms, 32)


def load_behavior(session_num):
    sp = paths_config['single_probe']['coordinates_1']['mouse01']['sessions']
    key = f"session_{session_num}"
    behav_path = Path(sp[key]['behavior'])
    behav_raw = pd.read_csv(behav_path, index_col=0)
    behav_df = behav_raw.T.reset_index(drop=True)
    behav_df.columns = behav_df.columns.str.strip()
    for col in behav_df.columns:
        behav_df[col] = pd.to_numeric(behav_df[col], errors='coerce')
    return behav_df


def detect_onsets(binary_signal, min_bout=MIN_BOUT_BINS, pre_margin=PRE_BINS,
                  post_margin=POST_BINS):
    """Find onset indices of behavior bouts (0->1 transitions with min duration)."""
    signal = (binary_signal > 0).astype(int)
    diff = np.diff(signal, prepend=0)
    onsets = np.where(diff == 1)[0]

    valid_onsets = []
    for onset in onsets:
        # Check minimum bout duration
        bout_end = onset
        while bout_end < len(signal) and signal[bout_end] == 1:
            bout_end += 1
        bout_len = bout_end - onset
        if bout_len < min_bout:
            continue
        # Check window margins
        if onset < pre_margin or onset + post_margin > len(signal):
            continue
        valid_onsets.append((onset, bout_len))

    return valid_onsets


# =============================================================================
# PERI-EVENT TRAJECTORIES
# =============================================================================

def plot_perievent(region, all_pca, all_behav, pca_model):
    """Peri-event scatter plot: mean positions around behavior onset."""
    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    axes = axes.ravel()

    total_window = PRE_BINS + POST_BINS

    for b_idx, behavior in enumerate(BEHAVIORS):
        ax = axes[b_idx]

        fed_trajectories = []
        fasted_trajectories = []
        fed_n_events = 0
        fasted_n_events = 0

        for sn in range(1, 9):
            if sn not in all_pca or sn not in all_behav:
                continue
            pca_traj = all_pca[sn]
            behav_df = all_behav[sn]
            n_use = min(len(pca_traj), len(behav_df))

            if behavior not in behav_df.columns:
                continue

            signal = behav_df[behavior].values[:n_use].astype(float)
            signal = np.nan_to_num(signal, nan=0.0)
            onsets = detect_onsets(signal)

            for onset_idx, bout_len in onsets:
                window = pca_traj[onset_idx - PRE_BINS:onset_idx + POST_BINS]
                if len(window) == total_window:
                    if SESSION_INFO[sn]['state'] == 'Fed':
                        fed_trajectories.append(window)
                        fed_n_events += 1
                    else:
                        fasted_trajectories.append(window)
                        fasted_n_events += 1

        has_data = False

        for label, trajs, n_events, cmap_name in [
            ('Fed', fed_trajectories, fed_n_events, 'Blues'),
            ('Fasted', fasted_trajectories, fasted_n_events, 'Reds'),
        ]:
            if len(trajs) < 1:
                continue
            has_data = True
            trajs = np.array(trajs)  # (n_events, total_window, 2)
            mean_traj = trajs.mean(axis=0)

            # Time color: light (pre) -> dark (post)
            n_pts = len(mean_traj)
            cmap = plt.cm.get_cmap(cmap_name)
            time_colors = cmap(np.linspace(0.25, 0.85, n_pts))

            # Scatter all mean points with time-coded color
            ax.scatter(mean_traj[:, 0], mean_traj[:, 1],
                       c=np.arange(n_pts), cmap=cmap_name, vmin=-20, vmax=n_pts + 20,
                       s=15, alpha=0.7, edgecolors='none', zorder=3)

            # Mark onset with star
            base_color = '#2196F3' if label == 'Fed' else '#F44336'
            ax.scatter(mean_traj[PRE_BINS, 0], mean_traj[PRE_BINS, 1],
                       marker='*', s=200, c=base_color,
                       edgecolors='k', linewidths=0.8, zorder=10)

            # Start marker (triangle)
            ax.scatter(mean_traj[0, 0], mean_traj[0, 1],
                       marker='^', s=80, c=base_color, alpha=0.6,
                       edgecolors='k', linewidths=0.5, zorder=8)

            ax.annotate(f'{label}: n={n_events}',
                        xy=(0.02, 0.98 if label == 'Fed' else 0.90),
                        xycoords='axes fraction', fontsize=8, va='top',
                        color=base_color, fontweight='bold')

        if not has_data:
            ax.text(0.5, 0.5, 'No events', transform=ax.transAxes,
                    ha='center', va='center', fontsize=12, color='gray')

        ax.set_title(SHORT_NAMES[behavior], fontsize=11, fontweight='bold')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.tick_params(labelsize=8)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2196F3',
               markersize=8, label='Fed (light=pre, dark=post)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#F44336',
               markersize=8, label='Fasted (light=pre, dark=post)'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='gray',
               markersize=12, label='Onset', markeredgecolor='k'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray',
               markersize=8, label='Start (-5s)', alpha=0.6),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               fontsize=10, bbox_to_anchor=(0.5, -0.02))

    plt.suptitle(f'{region.upper()} — Peri-Event Latent States (Scatter)\n'
                 f'(-5s to +10s around behavior onset, mean across events)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    path = FIGURES_DIR / f'gru_ode_10ms_perievent_{region}.png'
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# FULL-SESSION STATE-SPACE
# =============================================================================

def plot_statespace(region, all_pca, all_behav):
    """Full-session state-space scatter colored by active behavior."""
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))

    for sn in range(1, 9):
        row = 0 if sn <= 4 else 1
        col = (sn - 1) % 4
        ax = axes[row, col]

        if sn not in all_pca or sn not in all_behav:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
            continue

        pca_traj = all_pca[sn]
        behav_df = all_behav[sn]
        n_use = min(len(pca_traj), len(behav_df))
        pca_traj = pca_traj[:n_use]

        # Background: all points in light gray
        ax.scatter(pca_traj[:, 0], pca_traj[:, 1], s=1, c='#E0E0E0',
                   alpha=0.3, zorder=1, rasterized=True)

        # Overlay behavior-active points
        for behavior in BEHAVIORS:
            if behavior not in behav_df.columns:
                continue
            signal = behav_df[behavior].values[:n_use].astype(float)
            signal = np.nan_to_num(signal, nan=0.0)
            active = (signal > 0)

            if active.sum() < 2:
                continue

            color = BEHAVIOR_COLORS[behavior]
            ax.scatter(pca_traj[active, 0], pca_traj[active, 1],
                       s=4, c=color, alpha=0.6, zorder=2, rasterized=True)

        info = SESSION_INFO[sn]
        ax.set_title(f'S{sn}: {info["state"]} {info["phase"]}',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('PC1', fontsize=8)
        ax.set_ylabel('PC2', fontsize=8)
        ax.tick_params(labelsize=7)

    # Row labels
    axes[0, 0].annotate('FED', xy=(-0.35, 0.5), xycoords='axes fraction',
                         fontsize=14, fontweight='bold', rotation=90,
                         va='center', color='#2196F3')
    axes[1, 0].annotate('FASTED', xy=(-0.35, 0.5), xycoords='axes fraction',
                         fontsize=14, fontweight='bold', rotation=90,
                         va='center', color='#F44336')

    # Legend
    legend_elements = [Line2D([0], [0], color=BEHAVIOR_COLORS[b], lw=2,
                              label=SHORT_NAMES[b]) for b in BEHAVIORS]
    legend_elements.append(Line2D([0], [0], color='#E0E0E0', lw=1,
                                  label='No behavior', alpha=0.5))
    fig.legend(handles=legend_elements, loc='lower center', ncol=5,
               fontsize=8, bbox_to_anchor=(0.5, -0.02))

    plt.suptitle(f'{region.upper()} — Full-Session Latent State-Space\n'
                 f'(colored by active behavior)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    path = FIGURES_DIR / f'gru_ode_10ms_statespace_{region}.png'
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# PERI-EVENT STATISTICS
# =============================================================================

def compute_perievent_stats(region, all_pca, all_behav):
    """Compute per-session peri-event trajectory metrics."""
    rows = []

    for behavior in BEHAVIORS:
        for sn in range(1, 9):
            if sn not in all_pca or sn not in all_behav:
                continue
            pca_traj = all_pca[sn]
            behav_df = all_behav[sn]
            n_use = min(len(pca_traj), len(behav_df))

            if behavior not in behav_df.columns:
                continue

            signal = behav_df[behavior].values[:n_use].astype(float)
            signal = np.nan_to_num(signal, nan=0.0)
            onsets = detect_onsets(signal)

            if len(onsets) == 0:
                continue

            # Metrics across events in this session
            pre_speeds = []
            post_speeds = []
            onset_displacements = []

            for onset_idx, bout_len in onsets:
                window = pca_traj[onset_idx - PRE_BINS:onset_idx + POST_BINS]
                if len(window) < PRE_BINS + POST_BINS:
                    continue

                # Speed: mean step size in PCA space
                pre_steps = np.linalg.norm(np.diff(window[:PRE_BINS], axis=0), axis=1)
                post_steps = np.linalg.norm(np.diff(window[PRE_BINS:PRE_BINS + min(bout_len, POST_BINS)], axis=0), axis=1)

                pre_speeds.append(pre_steps.mean())
                if len(post_steps) > 0:
                    post_speeds.append(post_steps.mean())

                # Displacement from onset to 2s after
                t_2s = min(20, POST_BINS, bout_len)  # 20 bins = 2s
                onset_displacements.append(
                    np.linalg.norm(window[PRE_BINS + t_2s] - window[PRE_BINS])
                )

            info = SESSION_INFO[sn]
            rows.append({
                'region': region, 'behavior': behavior,
                'session': sn, 'state': info['state'], 'phase': info['phase'],
                'n_events': len(onsets),
                'mean_pre_speed': np.mean(pre_speeds) if pre_speeds else np.nan,
                'mean_post_speed': np.mean(post_speeds) if post_speeds else np.nan,
                'speed_change': (np.mean(post_speeds) - np.mean(pre_speeds)) if pre_speeds and post_speeds else np.nan,
                'mean_onset_displacement': np.mean(onset_displacements) if onset_displacements else np.nan,
            })

    df = pd.DataFrame(rows)

    # Print summary with fed vs fasted stats
    print(f"\n  {region.upper()} Peri-Event Statistics:")
    for behavior in BEHAVIORS:
        bdf = df[df['behavior'] == behavior]
        fed = bdf[bdf['state'] == 'Fed']
        fas = bdf[bdf['state'] == 'Fasted']

        n_fed = fed['n_events'].sum() if len(fed) > 0 else 0
        n_fas = fas['n_events'].sum() if len(fas) > 0 else 0

        if n_fed == 0 and n_fas == 0:
            print(f"    {SHORT_NAMES[behavior]:20s}: no events")
            continue

        # Speed change comparison
        fed_sc = fed['speed_change'].dropna().values
        fas_sc = fas['speed_change'].dropna().values

        p_str = ''
        if len(fed_sc) >= 2 and len(fas_sc) >= 2:
            _, p = mannwhitneyu(fed_sc, fas_sc, alternative='two-sided')
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            p_str = f'p={p:.4f} {sig}'

        print(f"    {SHORT_NAMES[behavior]:20s}: Fed={n_fed} events, Fasted={n_fas} events, "
              f"speed_change: Fed={np.nanmean(fed_sc):.4f}, Fas={np.nanmean(fas_sc):.4f} {p_str}")

    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Loading data...")

    all_behav = {}
    for sn in range(1, 9):
        all_behav[sn] = load_behavior(sn)
        print(f"  S{sn}: {len(all_behav[sn])} behavior bins")

    for region in ['lha', 'rsp']:
        print(f"\n{'='*60}")
        print(f"  {region.upper()}")
        print(f"{'='*60}")

        # Load and subsample hidden states
        print("  Loading cached hidden states...")
        all_h100 = {}
        for sn in range(1, 9):
            all_h100[sn] = load_hidden_states(region, sn)
            print(f"    S{sn}: {all_h100[sn].shape}")

        # Joint PCA across all sessions
        print("  Fitting joint PCA...")
        pooled = np.vstack([all_h100[sn] for sn in range(1, 9)])
        pca = PCA(n_components=2)
        pca.fit(pooled)
        print(f"    Variance explained: PC1={pca.explained_variance_ratio_[0]:.1%}, "
              f"PC2={pca.explained_variance_ratio_[1]:.1%}")

        all_pca = {}
        for sn in range(1, 9):
            all_pca[sn] = pca.transform(all_h100[sn])

        # Behavior onset counts
        print("\n  Behavior onset counts:")
        for behavior in BEHAVIORS:
            counts = {}
            for sn in range(1, 9):
                behav_df = all_behav[sn]
                n_use = min(len(all_pca[sn]), len(behav_df))
                if behavior not in behav_df.columns:
                    counts[sn] = 0
                    continue
                signal = behav_df[behavior].values[:n_use].astype(float)
                signal = np.nan_to_num(signal, nan=0.0)
                onsets = detect_onsets(signal)
                counts[sn] = len(onsets)
            total = sum(counts.values())
            fed_total = sum(counts[s] for s in [1, 2, 3, 4])
            fas_total = sum(counts[s] for s in [5, 6, 7, 8])
            detail = ', '.join(f'S{s}:{counts[s]}' for s in range(1, 9))
            print(f"    {SHORT_NAMES[behavior]:20s}: {total} total "
                  f"(Fed={fed_total}, Fasted={fas_total}) [{detail}]")

        # Generate plots
        print("\n  Generating peri-event trajectories...")
        plot_perievent(region, all_pca, all_behav, pca)

        print("  Generating state-space plots...")
        plot_statespace(region, all_pca, all_behav)

        # Statistics
        stats_df = compute_perievent_stats(region, all_pca, all_behav)

        # Save stats
        if region == 'lha':
            all_stats = stats_df
        else:
            all_stats = pd.concat([all_stats, stats_df], ignore_index=True)

    # Save combined stats
    stats_path = DATA_DIR / 'gru_ode_10ms_behavior_trajectories_stats.csv'
    all_stats.to_csv(stats_path, index=False, float_format='%.4f')
    print(f"\nSaved: {stats_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
