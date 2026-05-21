"""
SPIKE-synchronization aligned with ALL behavioral variables.
Loads pre-computed sync traces from coor1_spike_sync_traces.npz.
Compares sync during each behavioral state (binary) and by continuous variable quartiles.
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, kruskal
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG
# =============================================================================
with open("paths.yaml") as f:
    cfg = yaml.safe_load(f)

FS = 30000
BIN_SEC = 0.1
sessions_cfg = cfg["single_probe"]["coordinates_1"]["mouse01"]["sessions"]
session_meta = {
    1: ('fed', 'exploration'), 2: ('fed', 'foraging'),
    3: ('fed', 'exploration'), 4: ('fed', 'foraging'),
    5: ('fasted', 'exploration'), 6: ('fasted', 'foraging'),
    7: ('fasted', 'exploration'), 8: ('fasted', 'foraging'),
}

NETWORK_COLORS = {'LHA-LHA': '#e74c3c', 'RSP-RSP': '#27ae60', 'LHA-RSP': '#2980b9'}
STATE_COLORS = {'fed': '#3498db', 'fasted': '#e74c3c'}

# Binary behavior variables to test (active vs inactive)
BINARY_BEHAVIORS = [
    'Feeding', 'Digging', 'Grooming',
    'Longer exploration at home', 'Quick and hasty exploration at home',
    'Quick one loop at home', 'Incomplete home return',
    'Contemplation at T-zone', 'Transition wall exploration',
    'Arena wall exploration', 'Hesitant exploration',
    'Random switching between pots', 'Intentional switching between pots',
    'Quick arena exploration', 'Hiding in corners',
    'Low acceleration',
]

# Continuous variables to test (quartile split)
CONTINUOUS_BEHAVIORS = [
    'Velocity', 'Distance to Pot-2', 'Distance to Pot-4',
    'Distance to Home', 'Meander',
]

MIN_ACTIVE_BINS = 20  # need at least 20 active bins (2 seconds) to include


# =============================================================================
# LOAD BEHAVIOR
# =============================================================================
def load_behavior(behav_path):
    df_raw = pd.read_csv(behav_path, header=None)
    var_names = df_raw.iloc[:, 0].values
    time_vals = df_raw.iloc[1, 1:].astype(float).values
    data = df_raw.iloc[:, 1:].values

    behav = {'time': time_vals}
    for i, name in enumerate(var_names):
        if isinstance(name, str):
            behav[name.strip()] = data[i].astype(float)
    return behav


# =============================================================================
# LOAD SYNC TRACES
# =============================================================================
traces_npz = np.load("data/coor1_spike_sync_traces.npz")

print("=" * 90)
print("SPIKE-SYNCHRONIZATION vs ALL BEHAVIORAL VARIABLES")
print("=" * 90)

all_results = []

for snum in range(1, 9):
    sc = sessions_cfg[f"session_{snum}"]
    state, phase = session_meta[snum]

    # Load sync traces for this session
    time_key = f's{snum}_time'
    if time_key not in traces_npz:
        continue
    time_bins = traces_npz[time_key]
    n_bins = len(time_bins)

    traces = {}
    for nt in ['LHA-LHA', 'RSP-RSP', 'LHA-RSP']:
        key = f's{snum}_{nt}'
        if key in traces_npz:
            traces[nt] = traces_npz[key]

    if not traces:
        continue

    # Load behavior
    behav_path = sc.get('behavior')
    if not behav_path or not Path(behav_path).exists():
        print(f"\n  S{snum}: no behavior data")
        continue

    behav = load_behavior(behav_path)
    behav_time = behav['time']

    # Map sync time bins -> behavior indices
    sync_to_behav = np.searchsorted(behav_time, time_bins)
    sync_to_behav = np.clip(sync_to_behav, 0, len(behav_time) - 1)

    print(f"\n  S{snum} ({state}/{phase}):")

    # --- Binary behaviors ---
    for var_name in BINARY_BEHAVIORS:
        if var_name not in behav:
            continue

        raw_vals = behav[var_name]
        aligned = raw_vals[sync_to_behav]
        active_mask = aligned > 0.5
        inactive_mask = ~active_mask
        n_active = active_mask.sum()
        n_inactive = inactive_mask.sum()

        if n_active < MIN_ACTIVE_BINS or n_inactive < MIN_ACTIVE_BINS:
            continue

        for nt, trace in traces.items():
            sync_active = trace[active_mask]
            sync_inactive = trace[inactive_mask]

            mean_active = np.nanmean(sync_active)
            mean_inactive = np.nanmean(sync_inactive)

            # Mann-Whitney U
            stat, p = mannwhitneyu(sync_active, sync_inactive, alternative='two-sided')
            pct_change = (mean_active - mean_inactive) / mean_inactive * 100 if mean_inactive > 0 else 0

            all_results.append({
                'session': snum, 'state': state, 'phase': phase,
                'network': nt, 'var_type': 'binary', 'variable': var_name,
                'mean_active': mean_active, 'mean_inactive': mean_inactive,
                'pct_change': pct_change,
                'n_active': int(n_active), 'n_inactive': int(n_inactive),
                'U_stat': stat, 'p_value': p,
            })

    # --- Continuous behaviors (quartile split) ---
    for var_name in CONTINUOUS_BEHAVIORS:
        if var_name not in behav:
            continue

        raw_vals = behav[var_name]
        aligned = raw_vals[sync_to_behav]

        # Remove zeros/NaN for distance variables
        valid = np.isfinite(aligned) & (aligned != 0) if 'Distance' in var_name else np.isfinite(aligned)
        if valid.sum() < 100:
            continue

        q25, q50, q75 = np.percentile(aligned[valid], [25, 50, 75])

        quartile_masks = {
            'Q1 (low)': aligned <= q25,
            'Q4 (high)': aligned >= q75,
        }

        for nt, trace in traces.items():
            q1_sync = trace[quartile_masks['Q1 (low)'] & valid]
            q4_sync = trace[quartile_masks['Q4 (high)'] & valid]

            if len(q1_sync) < MIN_ACTIVE_BINS or len(q4_sync) < MIN_ACTIVE_BINS:
                continue

            mean_q1 = np.nanmean(q1_sync)
            mean_q4 = np.nanmean(q4_sync)
            stat, p = mannwhitneyu(q1_sync, q4_sync, alternative='two-sided')
            pct_change = (mean_q4 - mean_q1) / mean_q1 * 100 if mean_q1 > 0 else 0

            all_results.append({
                'session': snum, 'state': state, 'phase': phase,
                'network': nt, 'var_type': 'continuous', 'variable': var_name,
                'mean_active': mean_q4, 'mean_inactive': mean_q1,
                'pct_change': pct_change,
                'n_active': int(quartile_masks['Q4 (high)'].sum()),
                'n_inactive': int(quartile_masks['Q1 (low)'].sum()),
                'U_stat': stat, 'p_value': p,
            })

    # Count how many binary variables had enough data
    s_results = [r for r in all_results if r['session'] == snum]
    binary_count = sum(1 for r in s_results if r['var_type'] == 'binary')
    cont_count = sum(1 for r in s_results if r['var_type'] == 'continuous')
    print(f"    {binary_count} binary tests + {cont_count} continuous tests")


# =============================================================================
# SAVE & SUMMARIZE
# =============================================================================
results_df = pd.DataFrame(all_results)
results_df.to_csv("data/coor1_spike_sync_all_behaviors.csv", index=False)
print(f"\nSaved {len(results_df)} tests to data/coor1_spike_sync_all_behaviors.csv")

# FDR correction across all tests
from statsmodels.stats.multitest import multipletests
if len(results_df) > 0:
    reject, p_adj, _, _ = multipletests(results_df['p_value'].values, method='fdr_bh', alpha=0.05)
    results_df['p_fdr'] = p_adj
    results_df['significant'] = reject
    results_df.to_csv("data/coor1_spike_sync_all_behaviors.csv", index=False)

# --- Print significant results ---
print("\n" + "=" * 90)
print("SIGNIFICANT RESULTS (FDR < 0.05)")
print("=" * 90)
sig_df = results_df[results_df['significant'] == True].sort_values('p_fdr')
if len(sig_df) == 0:
    print("  No results survive FDR correction.")
else:
    for _, r in sig_df.iterrows():
        direction = "higher" if r['pct_change'] > 0 else "lower"
        print(f"  S{r['session']} ({r['state']}/{r['phase']}) {r['network']} | "
              f"{r['variable']}: sync {direction} during active ({r['pct_change']:+.1f}%), "
              f"p_fdr={r['p_fdr']:.4f}")

# --- Print uncorrected p < 0.05 for exploration ---
print("\n" + "=" * 90)
print("UNCORRECTED p < 0.05 (for exploration only)")
print("=" * 90)
unc_df = results_df[(results_df['p_value'] < 0.05) & (results_df['significant'] == False)]
unc_df = unc_df.sort_values('p_value')
for _, r in unc_df.head(30).iterrows():
    direction = "higher" if r['pct_change'] > 0 else "lower"
    print(f"  S{r['session']} ({r['state']}/{r['phase']}) {r['network']} | "
          f"{r['variable']}: sync {direction} ({r['pct_change']:+.1f}%), "
          f"p={r['p_value']:.4f}, p_fdr={r['p_fdr']:.3f}")

# --- Aggregate: which variables are most often significant across sessions? ---
print("\n" + "=" * 90)
print("VARIABLE CONSISTENCY: how often p < 0.05 uncorrected across sessions")
print("=" * 90)
sig_unc = results_df[results_df['p_value'] < 0.05]
if len(sig_unc) > 0:
    consistency = (sig_unc.groupby(['variable', 'network'])
                   .agg(n_sessions=('session', 'nunique'),
                        sessions=('session', lambda x: sorted(x.unique())),
                        mean_pct=('pct_change', 'mean'),
                        directions=('pct_change', lambda x: f"{(x>0).sum()}+/{(x<0).sum()}-"))
                   .sort_values('n_sessions', ascending=False))
    for (var, nt), row in consistency.iterrows():
        print(f"  {nt:<10} {var:<45} {row['n_sessions']}/8 sessions, "
              f"mean {row['mean_pct']:+.1f}%, {row['directions']}, S={row['sessions']}")


# =============================================================================
# FIGURE: Top behavioral modulators of synchrony
# =============================================================================

# For each network, find variables with consistent effects (>= 3 sessions p < 0.05)
print("\n" + "=" * 90)
print("GENERATING FIGURES")
print("=" * 90)

# Figure 1: Heatmap of effect sizes (pct_change) per variable × session
for nt in ['LHA-LHA', 'RSP-RSP', 'LHA-RSP']:
    nt_df = results_df[results_df['network'] == nt]
    if len(nt_df) == 0:
        continue

    # Get variables that appear in at least 4 sessions
    var_counts = nt_df.groupby('variable')['session'].nunique()
    common_vars = var_counts[var_counts >= 4].index.tolist()
    if not common_vars:
        continue

    # Build matrix: variables × sessions
    pivot = nt_df[nt_df['variable'].isin(common_vars)].pivot_table(
        values='pct_change', index='variable', columns='session', aggfunc='mean')

    # Also get significance markers
    sig_pivot = nt_df[nt_df['variable'].isin(common_vars)].pivot_table(
        values='p_value', index='variable', columns='session', aggfunc='mean')

    fig, ax = plt.subplots(figsize=(12, max(4, len(common_vars) * 0.45)))

    # Sort by mean absolute effect
    row_order = pivot.abs().mean(axis=1).sort_values(ascending=True).index
    pivot = pivot.loc[row_order]
    sig_pivot = sig_pivot.loc[row_order]

    im = ax.imshow(pivot.values, aspect='auto', cmap='RdBu_r',
                   vmin=-max(3, pivot.values[np.isfinite(pivot.values)].max()),
                   vmax=max(3, pivot.values[np.isfinite(pivot.values)].max()))

    # Mark significant cells
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            if not np.isnan(sig_pivot.values[i, j]):
                if sig_pivot.values[i, j] < 0.001:
                    ax.text(j, i, '***', ha='center', va='center', fontsize=8, fontweight='bold')
                elif sig_pivot.values[i, j] < 0.01:
                    ax.text(j, i, '**', ha='center', va='center', fontsize=8, fontweight='bold')
                elif sig_pivot.values[i, j] < 0.05:
                    ax.text(j, i, '*', ha='center', va='center', fontsize=8)

    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([f"S{c}\n{session_meta[c][0][:3]}/{session_meta[c][1][:3]}"
                        for c in pivot.columns], fontsize=9)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_xlabel('Session', fontsize=10)
    plt.colorbar(im, ax=ax, label='% change in sync (active vs inactive)', shrink=0.8)
    ax.set_title(f'{nt} — Behavioral Modulation of SPIKE-Synchronization\n'
                 f'(* p<0.05, ** p<0.01, *** p<0.001 uncorrected)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'figures/spike_sync_behav_heatmap_{nt.lower().replace("-","_")}.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved figures/spike_sync_behav_heatmap_{nt.lower().replace('-','_')}.png")


# Figure 2: Feeding — sync during feeding vs not-feeding (all sessions, 3 networks)
feeding_df = results_df[results_df['variable'] == 'Feeding']
if len(feeding_df) > 0:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ai, nt in enumerate(['LHA-LHA', 'RSP-RSP', 'LHA-RSP']):
        ax = axes[ai]
        nt_df = feeding_df[feeding_df['network'] == nt]

        for si, s in enumerate(['fed', 'fasted']):
            s_df = nt_df[nt_df['state'] == s]
            if len(s_df) == 0:
                continue

            x_feed = si * 2.5
            x_nofeed = si * 2.5 + 1

            feed_vals = s_df['mean_active'].values
            nofeed_vals = s_df['mean_inactive'].values

            ax.bar(x_nofeed, nofeed_vals.mean(), 0.8, color=STATE_COLORS[s], alpha=0.4,
                   yerr=nofeed_vals.std()/np.sqrt(len(nofeed_vals)) if len(nofeed_vals) > 1 else 0,
                   capsize=3, label=f'{s} no-feed' if ai == 0 else '')
            ax.bar(x_feed, feed_vals.mean(), 0.8, color=STATE_COLORS[s], alpha=0.9,
                   yerr=feed_vals.std()/np.sqrt(len(feed_vals)) if len(feed_vals) > 1 else 0,
                   capsize=3, label=f'{s} feeding' if ai == 0 else '')

            ax.scatter([x_feed]*len(feed_vals), feed_vals, color='black', s=20, zorder=5, alpha=0.6)
            ax.scatter([x_nofeed]*len(nofeed_vals), nofeed_vals, color='black', s=20, zorder=5, alpha=0.6)

            # Mark significant sessions
            for _, r in s_df.iterrows():
                if r['p_value'] < 0.05:
                    y_pos = max(r['mean_active'], r['mean_inactive']) + 0.003
                    ax.text(x_feed + 0.5, y_pos, f"S{int(r['session'])}*", fontsize=7,
                            ha='center', color='red')

        ax.set_xticks([0, 1, 2.5, 3.5])
        ax.set_xticklabels(['Fed\nFeed', 'Fed\nNo-feed', 'Fast\nFeed', 'Fast\nNo-feed'], fontsize=9)
        ax.set_ylabel('Mean SPIKE-Sync', fontsize=10)
        ax.set_title(nt, fontsize=12, fontweight='bold')

    axes[0].legend(fontsize=8, loc='upper right')
    fig.suptitle('SPIKE-Synchronization During Feeding vs Not-Feeding',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('figures/spike_sync_feeding.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved figures/spike_sync_feeding.png")


# Figure 3: Digging
digging_df = results_df[results_df['variable'] == 'Digging']
if len(digging_df) > 0:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ai, nt in enumerate(['LHA-LHA', 'RSP-RSP', 'LHA-RSP']):
        ax = axes[ai]
        nt_df = digging_df[digging_df['network'] == nt]

        for si, s in enumerate(['fed', 'fasted']):
            s_df = nt_df[nt_df['state'] == s]
            if len(s_df) == 0:
                continue

            x_dig = si * 2.5
            x_nodig = si * 2.5 + 1

            dig_vals = s_df['mean_active'].values
            nodig_vals = s_df['mean_inactive'].values

            ax.bar(x_nodig, nodig_vals.mean(), 0.8, color=STATE_COLORS[s], alpha=0.4,
                   yerr=nodig_vals.std()/np.sqrt(len(nodig_vals)) if len(nodig_vals) > 1 else 0,
                   capsize=3)
            ax.bar(x_dig, dig_vals.mean(), 0.8, color=STATE_COLORS[s], alpha=0.9,
                   yerr=dig_vals.std()/np.sqrt(len(dig_vals)) if len(dig_vals) > 1 else 0,
                   capsize=3)

            ax.scatter([x_dig]*len(dig_vals), dig_vals, color='black', s=20, zorder=5, alpha=0.6)
            ax.scatter([x_nodig]*len(nodig_vals), nodig_vals, color='black', s=20, zorder=5, alpha=0.6)

            for _, r in s_df.iterrows():
                if r['p_value'] < 0.05:
                    y_pos = max(r['mean_active'], r['mean_inactive']) + 0.003
                    ax.text(x_dig + 0.5, y_pos, f"S{int(r['session'])}*", fontsize=7,
                            ha='center', color='red')

        ax.set_xticks([0, 1, 2.5, 3.5])
        ax.set_xticklabels(['Fed\nDig', 'Fed\nNo-dig', 'Fast\nDig', 'Fast\nNo-dig'], fontsize=9)
        ax.set_ylabel('Mean SPIKE-Sync', fontsize=10)
        ax.set_title(nt, fontsize=12, fontweight='bold')

    fig.suptitle('SPIKE-Synchronization During Digging vs Not-Digging',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('figures/spike_sync_digging.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved figures/spike_sync_digging.png")


# Figure 4: Velocity effect (Q1 vs Q4)
vel_df = results_df[results_df['variable'] == 'Velocity']
if len(vel_df) > 0:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ai, nt in enumerate(['LHA-LHA', 'RSP-RSP', 'LHA-RSP']):
        ax = axes[ai]
        nt_df = vel_df[vel_df['network'] == nt]

        for si, s in enumerate(['fed', 'fasted']):
            s_df = nt_df[nt_df['state'] == s]
            if len(s_df) == 0:
                continue

            x_slow = si * 2.5
            x_fast = si * 2.5 + 1

            slow_vals = s_df['mean_inactive'].values  # Q1 = low velocity
            fast_vals = s_df['mean_active'].values     # Q4 = high velocity

            ax.bar(x_slow, slow_vals.mean(), 0.8, color=STATE_COLORS[s], alpha=0.9,
                   yerr=slow_vals.std()/np.sqrt(len(slow_vals)) if len(slow_vals) > 1 else 0,
                   capsize=3)
            ax.bar(x_fast, fast_vals.mean(), 0.8, color=STATE_COLORS[s], alpha=0.4,
                   yerr=fast_vals.std()/np.sqrt(len(fast_vals)) if len(fast_vals) > 1 else 0,
                   capsize=3)

            ax.scatter([x_slow]*len(slow_vals), slow_vals, color='black', s=20, zorder=5, alpha=0.6)
            ax.scatter([x_fast]*len(fast_vals), fast_vals, color='black', s=20, zorder=5, alpha=0.6)

            for _, r in s_df.iterrows():
                if r['p_value'] < 0.05:
                    y_pos = max(r['mean_active'], r['mean_inactive']) + 0.003
                    ax.text(x_slow + 0.5, y_pos, f"S{int(r['session'])}*", fontsize=7,
                            ha='center', color='red')

        ax.set_xticks([0, 1, 2.5, 3.5])
        ax.set_xticklabels(['Fed\nSlow', 'Fed\nFast', 'Fast\nSlow', 'Fast\nFast'], fontsize=9)
        ax.set_ylabel('Mean SPIKE-Sync', fontsize=10)
        ax.set_title(nt, fontsize=12, fontweight='bold')

    fig.suptitle('SPIKE-Synchronization: Low vs High Velocity (Q1 vs Q4)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('figures/spike_sync_velocity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved figures/spike_sync_velocity.png")

print("\n[DONE]")
