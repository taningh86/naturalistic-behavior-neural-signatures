"""
Entropy-Neural Signatures — All 8 Sessions (M1, Coordinates 1)

Investigates neural correlates of behavioral entropy:
1. Sliding-window correlation between entropy and neural metrics (FR, PC1-3) for LHA and RSP
2. Cross-correlation (temporal lead/lag) — does neural state predict entropy or vice versa?
3. High vs Low entropy epoch comparison (top/bottom quartile)
4. Peri-event analysis around entropy dips (threshold crossings)
5. Cross-session summary

Uses same 60s window / 10s step entropy as behavioral_sequence_mining.py.
Neural metrics smoothed to match entropy timescale.
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import spikeinterface.extractors as se
from sklearn.decomposition import PCA
from scipy.stats import spearmanr, mannwhitneyu, pearsonr
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate
from collections import Counter
from scipy.stats import entropy as sp_entropy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import warnings
import time as timer

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG
# =============================================================================
with open("paths.yaml") as f:
    cfg = yaml.safe_load(f)

FS = 30000
LHA_DEPTH_MAX = 1300
MIN_FR = 0.3
MIN_AMP = 48

# Neural binning — coarse to match entropy timescale
NEURAL_BIN_SEC = 1.0       # 1s bins for neural data
SMOOTH_SEC = 10.0          # smooth neural to ~10s scale (matching entropy step)

# Entropy params (must match behavioral_sequence_mining.py)
ENTROPY_WINDOW_SEC = 60
ENTROPY_STEP_SEC = 10

sessions_cfg = cfg["single_probe"]["coordinates_1"]["mouse01"]["sessions"]
session_meta = {
    1: ('fed', 'exploration'), 2: ('fed', 'foraging'),
    3: ('fed', 'exploration'), 4: ('fed', 'foraging'),
    5: ('fasted', 'exploration'), 6: ('fasted', 'foraging'),
    7: ('fasted', 'exploration'), 8: ('fasted', 'foraging'),
}

priority_order = [
    'Right corner', 'Left corner', 'Arna center', 'Foraging arena',
    'Home', 'Ladder', 'Transition zone',
    'Pot-1 zone', 'Pot-2 zone', 'Pot-3 zone', 'Pot-4 zone',
    'Pot-1', 'Pot-2', 'Pot-3', 'Pot-4',
]

zone_short = {
    'Home': 'H', 'Ladder': 'L', 'Transition zone': 'T',
    'Foraging arena': 'FA', 'Arna center': 'AC',
    'Pot-1': 'P1', 'Pot-2': 'P2', 'Pot-3': 'P3', 'Pot-4': 'P4',
    'Pot-1 zone': 'P1z', 'Pot-2 zone': 'P2z', 'Pot-3 zone': 'P3z', 'Pot-4 zone': 'P4z',
    'Right corner': 'RC', 'Left corner': 'LC', 'other': 'O',
}


# =============================================================================
# HELPERS
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


def get_zones(behav):
    n = len(behav['time'])
    zones = np.full(n, 'O', dtype=object)
    for var_name in priority_order:
        if var_name in behav:
            mask = behav[var_name] > 0.5
            zones[mask] = zone_short.get(var_name, var_name)
    return zones


def compute_entropy_trace(zones, time_vals):
    """Sliding-window transition entropy matching behavioral_sequence_mining.py."""
    dt = np.median(np.diff(time_vals))
    window_bins = int(ENTROPY_WINDOW_SEC / dt)
    step_bins = int(ENTROPY_STEP_SEC / dt)

    ent_times, ent_vals = [], []
    for start_idx in range(0, len(zones) - window_bins, step_bins):
        end_idx = start_idx + window_bins
        wz = zones[start_idx:end_idx]
        transitions = []
        for j in range(1, len(wz)):
            if wz[j] != wz[j-1]:
                transitions.append(f"{wz[j-1]}->{wz[j]}")
        if len(transitions) < 3:
            continue
        counts = Counter(transitions)
        probs = np.array(list(counts.values()), dtype=float)
        probs /= probs.sum()
        h = sp_entropy(probs, base=2)
        ent_times.append(time_vals[start_idx + window_bins // 2])
        ent_vals.append(h)

    return np.array(ent_times), np.array(ent_vals)


def get_good_units(sorted_path):
    ci = Path(sorted_path) / "cluster_info.tsv"
    df = pd.read_csv(ci, sep='\t')
    label_col = 'group' if ('group' in df.columns and df['group'].eq('good').any()) else 'KSLabel'
    good = df[(df[label_col] == 'good') & (df['fr'] > MIN_FR) & (df['amp'] > MIN_AMP)]
    lha = good[good['depth'] < LHA_DEPTH_MAX]['cluster_id'].values
    rsp = good[good['depth'] >= LHA_DEPTH_MAX]['cluster_id'].values
    return lha, rsp


def zscore_1d(x):
    mu, sd = np.mean(x), np.std(x)
    if sd < 1e-6:
        return x - mu
    return (x - mu) / sd


# =============================================================================
# MAIN LOOP
# =============================================================================
all_stats = []
all_session_data = {}

for snum in range(1, 9):
    t0 = timer.time()
    state, phase = session_meta[snum]
    sc = sessions_cfg[f"session_{snum}"]
    sorted_path = Path(sc['sorted'])
    behav_path = sc.get('behavior')

    print(f"\n{'='*70}")
    print(f"SESSION {snum} ({state}/{phase})")
    print(f"{'='*70}")

    if not behav_path or not Path(behav_path).exists():
        print("  SKIP -- no behavior data")
        continue

    # --- Behavior: entropy trace ---
    behav = load_behavior(behav_path)
    time_vals = behav['time']
    zones = get_zones(behav)
    ent_times, ent_vals = compute_entropy_trace(zones, time_vals)
    print(f"  Entropy: {len(ent_vals)} windows, mean={np.mean(ent_vals):.2f}, range={np.min(ent_vals):.2f}-{np.max(ent_vals):.2f}")

    # --- Neural data ---
    lha_ids, rsp_ids = get_good_units(sorted_path)
    sorting = se.read_kilosort(sorted_path)
    avail = set(sorting.get_unit_ids())
    lha_ids = np.array([u for u in lha_ids if u in avail])
    rsp_ids = np.array([u for u in rsp_ids if u in avail])
    print(f"  LHA: {len(lha_ids)} units, RSP: {len(rsp_ids)} units")

    if len(lha_ids) < 2 or len(rsp_ids) < 2:
        print("  SKIP -- insufficient units")
        continue

    # Bin spikes at 1s resolution
    rec_duration = time_vals[-1] + NEURAL_BIN_SEC
    bin_edges = np.arange(0, rec_duration + NEURAL_BIN_SEC, NEURAL_BIN_SEC)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    n_bins = len(bin_centers)

    lha_fr = np.array([np.histogram(sorting.get_unit_spike_train(u) / FS, bins=bin_edges)[0] / NEURAL_BIN_SEC
                        for u in lha_ids])
    rsp_fr = np.array([np.histogram(sorting.get_unit_spike_train(u) / FS, bins=bin_edges)[0] / NEURAL_BIN_SEC
                        for u in rsp_ids])

    # Population mean FR (z-scored per unit, then averaged)
    lha_z = np.array([(r - r.mean()) / max(r.std(), 1e-6) for r in lha_fr])
    rsp_z = np.array([(r - r.mean()) / max(r.std(), 1e-6) for r in rsp_fr])
    lha_pop_fr = np.mean(lha_z, axis=0)
    rsp_pop_fr = np.mean(rsp_z, axis=0)

    # PCA on z-scored data
    lha_pca_model = PCA(n_components=min(3, len(lha_ids))).fit(lha_z.T)
    rsp_pca_model = PCA(n_components=min(3, len(rsp_ids))).fit(rsp_z.T)
    lha_pcs = lha_pca_model.transform(lha_z.T)  # (n_bins, 3)
    rsp_pcs = rsp_pca_model.transform(rsp_z.T)

    print(f"  LHA PCA var explained: {lha_pca_model.explained_variance_ratio_[:3]}")
    print(f"  RSP PCA var explained: {rsp_pca_model.explained_variance_ratio_[:3]}")

    # Smooth neural signals (Gaussian, sigma = SMOOTH_SEC / NEURAL_BIN_SEC)
    sigma = SMOOTH_SEC / NEURAL_BIN_SEC
    lha_pop_fr_s = gaussian_filter1d(lha_pop_fr, sigma)
    rsp_pop_fr_s = gaussian_filter1d(rsp_pop_fr, sigma)
    lha_pc1_s = gaussian_filter1d(lha_pcs[:, 0], sigma)
    rsp_pc1_s = gaussian_filter1d(rsp_pcs[:, 0], sigma)
    lha_pc2_s = gaussian_filter1d(lha_pcs[:, 1], sigma)
    rsp_pc2_s = gaussian_filter1d(rsp_pcs[:, 1], sigma)

    # Population state velocity (rate of change in PCA space)
    lha_pcs_s = np.column_stack([gaussian_filter1d(lha_pcs[:, i], sigma) for i in range(lha_pcs.shape[1])])
    rsp_pcs_s = np.column_stack([gaussian_filter1d(rsp_pcs[:, i], sigma) for i in range(rsp_pcs.shape[1])])
    lha_vel = np.sqrt(np.sum(np.diff(lha_pcs_s, axis=0)**2, axis=1))
    rsp_vel = np.sqrt(np.sum(np.diff(rsp_pcs_s, axis=0)**2, axis=1))
    lha_vel = np.concatenate([[lha_vel[0]], lha_vel])
    rsp_vel = np.concatenate([[rsp_vel[0]], rsp_vel])
    lha_vel_s = gaussian_filter1d(lha_vel, sigma)
    rsp_vel_s = gaussian_filter1d(rsp_vel, sigma)

    # --- Interpolate neural metrics at entropy time points ---
    neural_metrics = {
        'LHA FR': np.interp(ent_times, bin_centers, lha_pop_fr_s),
        'RSP FR': np.interp(ent_times, bin_centers, rsp_pop_fr_s),
        'LHA PC1': np.interp(ent_times, bin_centers, lha_pc1_s),
        'RSP PC1': np.interp(ent_times, bin_centers, rsp_pc1_s),
        'LHA PC2': np.interp(ent_times, bin_centers, lha_pc2_s),
        'RSP PC2': np.interp(ent_times, bin_centers, rsp_pc2_s),
        'LHA Vel': np.interp(ent_times, bin_centers, lha_vel_s),
        'RSP Vel': np.interp(ent_times, bin_centers, rsp_vel_s),
    }

    # =========================================================================
    # 1. CORRELATION: entropy vs each neural metric
    # =========================================================================
    print(f"\n  --- Spearman correlations (entropy vs neural) ---")
    session_corrs = {}
    for metric_name, metric_vals in neural_metrics.items():
        rho, p = spearmanr(ent_vals, metric_vals)
        session_corrs[metric_name] = (rho, p)
        sig = '*' if p < 0.05 else ''
        print(f"    {metric_name:>10}: rho={rho:+.3f}, p={p:.4f} {sig}")

        all_stats.append({
            'session': snum, 'state': state, 'phase': phase,
            'metric': metric_name, 'analysis': 'spearman',
            'rho': rho, 'p': p, 'significant': p < 0.05
        })

    # =========================================================================
    # 2. CROSS-CORRELATION (temporal lead/lag)
    # =========================================================================
    print(f"\n  --- Cross-correlation peak lags ---")
    max_lag_steps = 12  # each step = 10s, so +/-120s
    lags_sec = np.arange(-max_lag_steps, max_lag_steps + 1) * ENTROPY_STEP_SEC

    ent_z = zscore_1d(ent_vals)
    for metric_name in ['LHA FR', 'RSP FR', 'LHA PC1', 'RSP PC1']:
        metric_z = zscore_1d(neural_metrics[metric_name])
        # Manual cross-correlation at discrete lags
        cc_vals = []
        for lag in range(-max_lag_steps, max_lag_steps + 1):
            if lag >= 0:
                cc = np.corrcoef(ent_z[:len(ent_z)-lag], metric_z[lag:])[0, 1]
            else:
                cc = np.corrcoef(ent_z[-lag:], metric_z[:len(metric_z)+lag])[0, 1]
            cc_vals.append(cc)
        cc_vals = np.array(cc_vals)
        peak_idx = np.argmax(np.abs(cc_vals))
        peak_lag = lags_sec[peak_idx]
        peak_cc = cc_vals[peak_idx]
        # Positive lag = neural LEADS entropy
        lead_str = "neural leads" if peak_lag > 0 else ("entropy leads" if peak_lag < 0 else "zero lag")
        print(f"    {metric_name:>10}: peak r={peak_cc:+.3f} at lag={peak_lag:+.0f}s ({lead_str})")

        all_stats.append({
            'session': snum, 'state': state, 'phase': phase,
            'metric': metric_name, 'analysis': 'xcorr_peak',
            'rho': peak_cc, 'lag_sec': peak_lag, 'p': np.nan
        })

    # =========================================================================
    # 3. HIGH vs LOW ENTROPY EPOCHS
    # =========================================================================
    q25 = np.percentile(ent_vals, 25)
    q75 = np.percentile(ent_vals, 75)
    low_mask = ent_vals <= q25
    high_mask = ent_vals >= q75

    print(f"\n  --- High vs Low entropy (Q75 vs Q25) ---")
    print(f"    Thresholds: low<={q25:.2f}, high>={q75:.2f}")
    print(f"    N windows: low={np.sum(low_mask)}, high={np.sum(high_mask)}")

    for metric_name, metric_vals in neural_metrics.items():
        low_vals = metric_vals[low_mask]
        high_vals = metric_vals[high_mask]
        if len(low_vals) < 3 or len(high_vals) < 3:
            continue
        u_stat, p = mannwhitneyu(low_vals, high_vals, alternative='two-sided')
        sig = '*' if p < 0.05 else ''
        pct_diff = 100 * (np.mean(high_vals) - np.mean(low_vals)) / (abs(np.mean(low_vals)) + 1e-10)
        print(f"    {metric_name:>10}: low={np.mean(low_vals):+.3f}, high={np.mean(high_vals):+.3f}, "
              f"diff={pct_diff:+.1f}%, p={p:.4f} {sig}")

        all_stats.append({
            'session': snum, 'state': state, 'phase': phase,
            'metric': metric_name, 'analysis': 'high_vs_low',
            'low_mean': np.mean(low_vals), 'high_mean': np.mean(high_vals),
            'pct_diff': pct_diff, 'p': p, 'significant': p < 0.05
        })

    # =========================================================================
    # 4. PERI-DIP ANALYSIS — entropy drops below session median
    # =========================================================================
    ent_median = np.median(ent_vals)
    ent_below = ent_vals < ent_median
    # Find onset of dip episodes (transition from above to below median)
    dip_onsets = []
    for i in range(1, len(ent_below)):
        if ent_below[i] and not ent_below[i-1]:
            dip_onsets.append(i)

    pre_steps = 6   # 60s before dip
    post_steps = 12  # 120s after dip
    peri_time_ax = np.arange(-pre_steps, post_steps) * ENTROPY_STEP_SEC

    if len(dip_onsets) >= 3:
        print(f"\n  --- Peri-dip analysis ({len(dip_onsets)} dip onsets, median={ent_median:.2f}) ---")

        peri_data = {mn: [] for mn in neural_metrics}
        peri_data['entropy'] = []

        for onset in dip_onsets:
            if onset - pre_steps < 0 or onset + post_steps > len(ent_vals):
                continue
            peri_data['entropy'].append(ent_vals[onset - pre_steps:onset + post_steps])
            for mn, mv in neural_metrics.items():
                peri_data[mn].append(mv[onset - pre_steps:onset + post_steps])

        n_valid = len(peri_data['entropy'])
        if n_valid >= 3:
            print(f"    Valid dips: {n_valid}")
            for mn in ['LHA FR', 'RSP FR', 'LHA PC1', 'RSP PC1']:
                arr = np.array(peri_data[mn])
                pre_mean = np.mean(arr[:, :pre_steps], axis=1)
                post_mean = np.mean(arr[:, pre_steps:], axis=1)
                try:
                    stat, p = mannwhitneyu(pre_mean, post_mean, alternative='two-sided')
                except ValueError:
                    p = 1.0
                sig = '*' if p < 0.05 else ''
                print(f"    {mn:>10}: pre={np.mean(pre_mean):+.3f}, post={np.mean(post_mean):+.3f}, p={p:.4f} {sig}")
    else:
        print(f"\n  --- Too few dip onsets ({len(dip_onsets)}) for peri-dip analysis ---")
        n_valid = 0
        peri_data = {}

    # Store for plotting
    all_session_data[snum] = {
        'state': state, 'phase': phase,
        'ent_times': ent_times, 'ent_vals': ent_vals,
        'neural_metrics': neural_metrics,
        'corrs': session_corrs,
        'peri_data': peri_data if n_valid >= 3 else {},
        'peri_time': peri_time_ax,
        'n_dips': n_valid if len(dip_onsets) >= 3 else 0,
    }

    elapsed = timer.time() - t0
    print(f"\n  Done in {elapsed:.1f}s")


# =============================================================================
# SAVE STATS
# =============================================================================
stats_df = pd.DataFrame(all_stats)
stats_df.to_csv("data/entropy_neural_stats.csv", index=False)
print(f"\nSaved data/entropy_neural_stats.csv ({len(stats_df)} rows)")


# =============================================================================
# FIGURE 1: Per-session time series overlay (entropy + neural)
# =============================================================================
fig, axes = plt.subplots(4, 4, figsize=(24, 16))
fig.suptitle("Behavioral Entropy vs Neural Metrics Over Time", fontsize=14, fontweight='bold')

for idx, snum in enumerate(range(1, 9)):
    if snum not in all_session_data:
        continue
    d = all_session_data[snum]
    col = idx % 4
    row_top = (idx // 4) * 2
    row_bot = row_top + 1

    ax1 = axes[row_top, col]
    ax2 = axes[row_bot, col]

    color_ent = '#333333'
    color_lha = '#e74c3c'
    color_rsp = '#3498db'

    t = d['ent_times']

    # Top: entropy + FR
    ax1.plot(t, d['ent_vals'], color=color_ent, linewidth=1.5, label='Entropy')
    ax1.set_ylabel("Entropy (bits)", color=color_ent)
    ax1b = ax1.twinx()
    ax1b.plot(t, d['neural_metrics']['LHA FR'], color=color_lha, linewidth=1, alpha=0.7, label='LHA FR')
    ax1b.plot(t, d['neural_metrics']['RSP FR'], color=color_rsp, linewidth=1, alpha=0.7, label='RSP FR')
    ax1b.set_ylabel("z-scored FR", fontsize=8)

    rho_l = d['corrs']['LHA FR'][0]
    rho_r = d['corrs']['RSP FR'][0]
    p_l = d['corrs']['LHA FR'][1]
    p_r = d['corrs']['RSP FR'][1]
    ax1.set_title(f"S{snum} ({d['state']}/{d['phase'][:3]})\n"
                  f"LHA rho={rho_l:+.2f} p={p_l:.3f}, RSP rho={rho_r:+.2f} p={p_r:.3f}",
                  fontsize=9)

    # Bottom: entropy + PC1
    ax2.plot(t, d['ent_vals'], color=color_ent, linewidth=1.5)
    ax2.set_ylabel("Entropy (bits)", color=color_ent)
    ax2b = ax2.twinx()
    ax2b.plot(t, d['neural_metrics']['LHA PC1'], color=color_lha, linewidth=1, alpha=0.7, label='LHA PC1')
    ax2b.plot(t, d['neural_metrics']['RSP PC1'], color=color_rsp, linewidth=1, alpha=0.7, label='RSP PC1')
    ax2b.set_ylabel("PC1 score", fontsize=8)
    ax2.set_xlabel("Time (s)")

    rho_l = d['corrs']['LHA PC1'][0]
    rho_r = d['corrs']['RSP PC1'][0]
    p_l = d['corrs']['LHA PC1'][1]
    p_r = d['corrs']['RSP PC1'][1]
    ax2.set_title(f"LHA PC1 rho={rho_l:+.2f} p={p_l:.3f}, RSP PC1 rho={rho_r:+.2f} p={p_r:.3f}",
                  fontsize=9)

    if col == 0:
        ax1b.legend(fontsize=7, loc='upper right')
        ax2b.legend(fontsize=7, loc='upper right')

plt.tight_layout()
plt.savefig("figures/entropy_neural_timeseries.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/entropy_neural_timeseries.png")


# =============================================================================
# FIGURE 2: Cross-session correlation summary
# =============================================================================
spearman_df = stats_df[stats_df['analysis'] == 'spearman'].copy()
metrics_order = ['LHA FR', 'RSP FR', 'LHA PC1', 'RSP PC1', 'LHA PC2', 'RSP PC2', 'LHA Vel', 'RSP Vel']

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: heatmap of rho values
rho_matrix = np.zeros((8, len(metrics_order)))
sig_matrix = np.zeros((8, len(metrics_order)), dtype=bool)
for _, row in spearman_df.iterrows():
    si = int(row['session']) - 1
    mi = metrics_order.index(row['metric']) if row['metric'] in metrics_order else -1
    if mi >= 0:
        rho_matrix[si, mi] = row['rho']
        sig_matrix[si, mi] = row['significant']

ax = axes[0]
im = ax.imshow(rho_matrix, cmap='RdBu_r', vmin=-0.6, vmax=0.6, aspect='auto')
ax.set_xticks(range(len(metrics_order)))
ax.set_xticklabels(metrics_order, rotation=45, ha='right', fontsize=9)
ax.set_yticks(range(8))
ylabels = [f"S{s+1} ({session_meta[s+1][0][:3]}/{session_meta[s+1][1][:3]})" for s in range(8)]
ax.set_yticklabels(ylabels, fontsize=9)
# Mark significant with asterisk
for i in range(8):
    for j in range(len(metrics_order)):
        if sig_matrix[i, j]:
            ax.text(j, i, '*', ha='center', va='center', fontsize=14, fontweight='bold', color='black')
        ax.text(j, i+0.3, f"{rho_matrix[i,j]:+.2f}", ha='center', va='center', fontsize=7, color='gray')
plt.colorbar(im, ax=ax, shrink=0.8, label='Spearman rho')
ax.set_title("Entropy-Neural Spearman Correlations\n(* p<0.05)", fontsize=11)

# Right: bar chart of mean |rho| by condition
ax = axes[1]
conditions = [('fed', 'Fed'), ('fasted', 'Fasted')]
x = np.arange(len(metrics_order))
width = 0.35
colors = {'fed': '#3498db', 'fasted': '#e74c3c'}

for ci, (state_key, label) in enumerate(conditions):
    means = []
    sems = []
    for mn in metrics_order:
        vals = spearman_df[(spearman_df['metric'] == mn) & (spearman_df['state'] == state_key)]['rho'].values
        means.append(np.mean(vals))
        sems.append(np.std(vals) / max(np.sqrt(len(vals)), 1))
    ax.bar(x + ci * width - width/2, means, width, yerr=sems, label=label,
           color=colors[state_key], alpha=0.8, capsize=3)

ax.set_xticks(x)
ax.set_xticklabels(metrics_order, rotation=45, ha='right', fontsize=9)
ax.set_ylabel("Mean Spearman rho")
ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
ax.legend()
ax.set_title("Mean Entropy-Neural Correlation\nby Metabolic State", fontsize=11)

plt.tight_layout()
plt.savefig("figures/entropy_neural_correlation_summary.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/entropy_neural_correlation_summary.png")


# =============================================================================
# FIGURE 3: High vs Low entropy neural comparison
# =============================================================================
hl_df = stats_df[stats_df['analysis'] == 'high_vs_low'].copy()

fig, axes = plt.subplots(2, 4, figsize=(20, 8))
fig.suptitle("Neural Metrics: High vs Low Entropy Epochs (Q75 vs Q25)", fontsize=13, fontweight='bold')

for idx, snum in enumerate(range(1, 9)):
    if snum not in all_session_data:
        continue
    ax = axes[idx // 4, idx % 4]
    d = all_session_data[snum]
    sess_hl = hl_df[hl_df['session'] == snum]

    metrics_plot = ['LHA FR', 'RSP FR', 'LHA PC1', 'RSP PC1']
    x_pos = np.arange(len(metrics_plot))

    low_means = []
    high_means = []
    sig_flags = []
    for mn in metrics_plot:
        row = sess_hl[sess_hl['metric'] == mn]
        if len(row) > 0:
            low_means.append(row.iloc[0]['low_mean'])
            high_means.append(row.iloc[0]['high_mean'])
            sig_flags.append(row.iloc[0]['significant'])
        else:
            low_means.append(0)
            high_means.append(0)
            sig_flags.append(False)

    w = 0.35
    ax.bar(x_pos - w/2, low_means, w, label='Low entropy', color='#e74c3c', alpha=0.7)
    ax.bar(x_pos + w/2, high_means, w, label='High entropy', color='#3498db', alpha=0.7)
    for i, sf in enumerate(sig_flags):
        if sf:
            ymax = max(abs(low_means[i]), abs(high_means[i]))
            ax.text(i, ymax * 1.1, '*', ha='center', fontsize=14, fontweight='bold')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics_plot, rotation=30, ha='right', fontsize=8)
    ax.set_title(f"S{snum} ({d['state']}/{d['phase'][:3]})", fontsize=10)
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    if idx == 0:
        ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("figures/entropy_neural_high_vs_low.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/entropy_neural_high_vs_low.png")


# =============================================================================
# FIGURE 4: Peri-dip analysis (sessions with enough dips)
# =============================================================================
dip_sessions = [s for s in all_session_data if all_session_data[s]['n_dips'] >= 3]
if dip_sessions:
    n_panels = len(dip_sessions)
    fig, axes = plt.subplots(2, n_panels, figsize=(5*n_panels, 8))
    if n_panels == 1:
        axes = axes[:, np.newaxis]
    fig.suptitle("Peri-Entropy-Dip Neural Activity\n(aligned to entropy drop below median)", fontsize=13, fontweight='bold')

    for ci, snum in enumerate(dip_sessions):
        d = all_session_data[snum]
        pt = d['peri_time']

        # Top: entropy + FR
        ax = axes[0, ci]
        ent_peri = np.array(d['peri_data']['entropy'])
        ax.plot(pt, np.mean(ent_peri, axis=0), color='black', linewidth=2, label='Entropy')
        ax.fill_between(pt, np.mean(ent_peri, axis=0) - np.std(ent_peri, axis=0)/np.sqrt(len(ent_peri)),
                        np.mean(ent_peri, axis=0) + np.std(ent_peri, axis=0)/np.sqrt(len(ent_peri)),
                        alpha=0.2, color='gray')
        ax.set_ylabel("Entropy (bits)")
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(f"S{snum} ({d['state']}/{d['phase'][:3]}), n={d['n_dips']} dips", fontsize=10)

        ax2 = ax.twinx()
        for mn, color in [('LHA FR', '#e74c3c'), ('RSP FR', '#3498db')]:
            arr = np.array(d['peri_data'][mn])
            mean_trace = np.mean(arr, axis=0)
            sem = np.std(arr, axis=0) / np.sqrt(len(arr))
            ax2.plot(pt, mean_trace, color=color, linewidth=1.5, label=mn)
            ax2.fill_between(pt, mean_trace - sem, mean_trace + sem, alpha=0.15, color=color)
        ax2.set_ylabel("z-scored FR", fontsize=8)
        if ci == 0:
            ax2.legend(fontsize=7, loc='upper right')

        # Bottom: entropy + PC1
        ax = axes[1, ci]
        ax.plot(pt, np.mean(ent_peri, axis=0), color='black', linewidth=2)
        ax.fill_between(pt, np.mean(ent_peri, axis=0) - np.std(ent_peri, axis=0)/np.sqrt(len(ent_peri)),
                        np.mean(ent_peri, axis=0) + np.std(ent_peri, axis=0)/np.sqrt(len(ent_peri)),
                        alpha=0.2, color='gray')
        ax.set_ylabel("Entropy (bits)")
        ax.set_xlabel("Time from dip onset (s)")
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)

        ax2 = ax.twinx()
        for mn, color in [('LHA PC1', '#e74c3c'), ('RSP PC1', '#3498db')]:
            arr = np.array(d['peri_data'][mn])
            mean_trace = np.mean(arr, axis=0)
            sem = np.std(arr, axis=0) / np.sqrt(len(arr))
            ax2.plot(pt, mean_trace, color=color, linewidth=1.5, label=mn)
            ax2.fill_between(pt, mean_trace - sem, mean_trace + sem, alpha=0.15, color=color)
        ax2.set_ylabel("PC1 score", fontsize=8)
        if ci == 0:
            ax2.legend(fontsize=7, loc='upper right')

    plt.tight_layout()
    plt.savefig("figures/entropy_neural_peri_dip.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved figures/entropy_neural_peri_dip.png")


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("CROSS-SESSION SUMMARY — Significant Spearman Correlations (p<0.05)")
print("=" * 70)
sig_corrs = spearman_df[spearman_df['significant'] == True]
for _, row in sig_corrs.iterrows():
    print(f"  S{int(row['session'])} ({row['state']}/{row['phase'][:3]}): "
          f"{row['metric']} rho={row['rho']:+.3f}, p={row['p']:.4f}")

print(f"\nTotal significant: {len(sig_corrs)} / {len(spearman_df)} tests")
print(f"By region: LHA={len(sig_corrs[sig_corrs['metric'].str.startswith('LHA')])}, "
      f"RSP={len(sig_corrs[sig_corrs['metric'].str.startswith('RSP')])}")
print(f"By state: fed={len(sig_corrs[sig_corrs['state']=='fed'])}, "
      f"fasted={len(sig_corrs[sig_corrs['state']=='fasted'])}")
