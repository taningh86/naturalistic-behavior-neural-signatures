"""
Entropy Mode-Specific Neural Signatures — All 8 Sessions (M1, Coor1)

For each entropy time point, classify the dominant behavioral mode:
  - P4_exploit: >40% P4-involving transitions (food pot exploitation)
  - P2_exploit: >40% P2-involving transitions (visible food exploitation)
  - HL_shuttle: >40% Home/Ladder transitions (home base shuttling)
  - Mixed: none of the above dominates (diverse exploration)

Then check: does the LHA-RSP entropy-neural opposition hold across ALL modes,
or is it specific to food-directed stereotypy?
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import spikeinterface.extractors as se
from sklearn.decomposition import PCA
from scipy.stats import spearmanr, mannwhitneyu, kruskal
from scipy.ndimage import gaussian_filter1d
from collections import Counter
from scipy.stats import entropy as sp_entropy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import warnings
import time as timer

warnings.filterwarnings('ignore')

with open("paths.yaml") as f:
    cfg = yaml.safe_load(f)

FS = 30000
LHA_DEPTH_MAX = 1300
MIN_FR = 0.3
MIN_AMP = 48
NEURAL_BIN_SEC = 1.0
SMOOTH_SEC = 10.0
ENTROPY_WINDOW_SEC = 60
ENTROPY_STEP_SEC = 10

# Mode classification threshold: a mode dominates if its transition % exceeds this
MODE_THRESH = 0.35

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


def get_good_units(sorted_path):
    ci = Path(sorted_path) / "cluster_info.tsv"
    df = pd.read_csv(ci, sep='\t')
    label_col = 'group' if ('group' in df.columns and df['group'].eq('good').any()) else 'KSLabel'
    good = df[(df[label_col] == 'good') & (df['fr'] > MIN_FR) & (df['amp'] > MIN_AMP)]
    lha = good[good['depth'] < LHA_DEPTH_MAX]['cluster_id'].values
    rsp = good[good['depth'] >= LHA_DEPTH_MAX]['cluster_id'].values
    return lha, rsp


def classify_mode(transitions_counter):
    """Classify a window's behavioral mode from its transition counts."""
    total = sum(transitions_counter.values())
    if total == 0:
        return 'empty'

    p4_frac = sum(v for k, v in transitions_counter.items()
                  if 'P4' in k) / total
    p2_frac = sum(v for k, v in transitions_counter.items()
                  if 'P2' in k) / total
    # H/L: transitions involving Home, Ladder, or between them via Transition zone
    hl_frac = sum(v for k, v in transitions_counter.items()
                  if any(x in k for x in ['H->', '->H', 'L->', '->L'])) / total

    # Priority: most dominant wins if above threshold
    fracs = {'P4_exploit': p4_frac, 'P2_exploit': p2_frac, 'HL_shuttle': hl_frac}
    best = max(fracs, key=fracs.get)
    if fracs[best] >= MODE_THRESH:
        return best
    return 'Mixed'


# =============================================================================
# MAIN LOOP
# =============================================================================
print("=" * 80)
print("ENTROPY MODE-SPECIFIC NEURAL SIGNATURES")
print("=" * 80)

all_rows = []
all_mode_neural = []  # For figure: (snum, mode, entropy, lha_fr, rsp_fr, lha_pc1, rsp_pc1)

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

    behav = load_behavior(behav_path)
    time_vals = behav['time']
    zones = get_zones(behav)

    dt = np.median(np.diff(time_vals))
    window_bins = int(ENTROPY_WINDOW_SEC / dt)
    step_bins = int(ENTROPY_STEP_SEC / dt)

    # Compute entropy + classify each window's behavioral mode
    ent_times, ent_vals, modes = [], [], []
    for start_idx in range(0, len(zones) - window_bins, step_bins):
        wz = zones[start_idx:start_idx + window_bins]
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
        modes.append(classify_mode(counts))

    ent_times = np.array(ent_times)
    ent_vals = np.array(ent_vals)
    modes = np.array(modes)

    # Mode summary
    mode_counts = Counter(modes)
    print(f"  Mode counts: {dict(mode_counts)}")
    for mode in ['P4_exploit', 'P2_exploit', 'HL_shuttle', 'Mixed']:
        mask = modes == mode
        if np.sum(mask) > 0:
            print(f"    {mode:12s}: n={np.sum(mask):3d}, entropy mean={np.mean(ent_vals[mask]):.2f}, "
                  f"std={np.std(ent_vals[mask]):.2f}")

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

    # Bin spikes
    rec_duration = time_vals[-1] + NEURAL_BIN_SEC
    bin_edges = np.arange(0, rec_duration + NEURAL_BIN_SEC, NEURAL_BIN_SEC)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    lha_fr = np.array([np.histogram(sorting.get_unit_spike_train(u) / FS, bins=bin_edges)[0] / NEURAL_BIN_SEC
                        for u in lha_ids])
    rsp_fr = np.array([np.histogram(sorting.get_unit_spike_train(u) / FS, bins=bin_edges)[0] / NEURAL_BIN_SEC
                        for u in rsp_ids])

    lha_z = np.array([(r - r.mean()) / max(r.std(), 1e-6) for r in lha_fr])
    rsp_z = np.array([(r - r.mean()) / max(r.std(), 1e-6) for r in rsp_fr])
    lha_pop_fr = np.mean(lha_z, axis=0)
    rsp_pop_fr = np.mean(rsp_z, axis=0)

    # PCA
    lha_pca_model = PCA(n_components=min(3, len(lha_ids))).fit(lha_z.T)
    rsp_pca_model = PCA(n_components=min(3, len(rsp_ids))).fit(rsp_z.T)
    lha_pcs = lha_pca_model.transform(lha_z.T)
    rsp_pcs = rsp_pca_model.transform(rsp_z.T)

    # Smooth
    sigma = SMOOTH_SEC / NEURAL_BIN_SEC
    lha_pop_fr_s = gaussian_filter1d(lha_pop_fr, sigma)
    rsp_pop_fr_s = gaussian_filter1d(rsp_pop_fr, sigma)
    lha_pc1_s = gaussian_filter1d(lha_pcs[:, 0], sigma)
    rsp_pc1_s = gaussian_filter1d(rsp_pcs[:, 0], sigma)

    # Interpolate neural at entropy time points
    lha_fr_at_ent = np.interp(ent_times, bin_centers, lha_pop_fr_s)
    rsp_fr_at_ent = np.interp(ent_times, bin_centers, rsp_pop_fr_s)
    lha_pc1_at_ent = np.interp(ent_times, bin_centers, lha_pc1_s)
    rsp_pc1_at_ent = np.interp(ent_times, bin_centers, rsp_pc1_s)

    # Store per-window data for figures
    for i in range(len(ent_times)):
        all_mode_neural.append({
            'session': snum, 'state': state, 'phase': phase,
            'time': ent_times[i], 'entropy': ent_vals[i], 'mode': modes[i],
            'lha_fr': lha_fr_at_ent[i], 'rsp_fr': rsp_fr_at_ent[i],
            'lha_pc1': lha_pc1_at_ent[i], 'rsp_pc1': rsp_pc1_at_ent[i],
        })

    # =========================================================================
    # Per-mode neural analysis
    # =========================================================================
    print(f"\n  --- Neural metrics by behavioral mode ---")

    for metric_name, metric_vals in [('LHA FR', lha_fr_at_ent), ('RSP FR', rsp_fr_at_ent),
                                      ('LHA PC1', lha_pc1_at_ent), ('RSP PC1', rsp_pc1_at_ent)]:

        # 1) Spearman within each mode
        print(f"\n  {metric_name}:")
        mode_means = {}
        for mode in ['P4_exploit', 'P2_exploit', 'HL_shuttle', 'Mixed']:
            mask = modes == mode
            n = np.sum(mask)
            if n < 5:
                continue
            rho, p = spearmanr(ent_vals[mask], metric_vals[mask])
            m_mean = np.mean(metric_vals[mask])
            m_ent = np.mean(ent_vals[mask])
            mode_means[mode] = m_mean
            sig = '*' if p < 0.05 else 'ns'
            print(f"    {mode:12s} (n={n:3d}): rho={rho:+.3f} p={p:.4f} {sig}  "
                  f"mean_neural={m_mean:+.3f}  mean_entropy={m_ent:.2f}")

            all_rows.append({
                'session': snum, 'state': state, 'phase': phase,
                'metric': metric_name, 'mode': mode, 'n': n,
                'rho': rho, 'p': p, 'significant': p < 0.05,
                'mean_neural': m_mean, 'mean_entropy': m_ent,
            })

        # 2) Compare neural level across modes (is LHA higher during HL vs P4?)
        mode_list = [m for m in ['P4_exploit', 'P2_exploit', 'HL_shuttle', 'Mixed']
                     if np.sum(modes == m) >= 5]
        if len(mode_list) >= 2:
            groups = [metric_vals[modes == m] for m in mode_list]
            if len(mode_list) >= 3:
                stat, p_kw = kruskal(*groups)
                print(f"    Kruskal-Wallis across modes: H={stat:.2f}, p={p_kw:.4f} "
                      f"{'*' if p_kw < 0.05 else 'ns'}")
            # Pairwise: key comparison is HL vs P4 and HL vs Mixed
            for i, m1 in enumerate(mode_list):
                for m2 in mode_list[i+1:]:
                    g1 = metric_vals[modes == m1]
                    g2 = metric_vals[modes == m2]
                    stat, p_pw = mannwhitneyu(g1, g2, alternative='two-sided')
                    d1, d2 = np.mean(g1), np.mean(g2)
                    print(f"      {m1} vs {m2}: {d1:+.3f} vs {d2:+.3f}, "
                          f"MWU p={p_pw:.4f} {'*' if p_pw < 0.05 else 'ns'}")

    elapsed = timer.time() - t0
    print(f"\n  Session {snum} done in {elapsed:.1f}s")

# Save stats
stats_df = pd.DataFrame(all_rows)
stats_df.to_csv("data/entropy_mode_neural_stats.csv", index=False)
print(f"\nSaved data/entropy_mode_neural_stats.csv ({len(stats_df)} rows)")

mode_df = pd.DataFrame(all_mode_neural)
mode_df.to_csv("data/entropy_mode_neural_raw.csv", index=False)
print(f"Saved data/entropy_mode_neural_raw.csv ({len(mode_df)} rows)")


# =============================================================================
# FIGURE 1: Per-session mode-colored entropy + neural traces
# =============================================================================
mode_colors = {
    'P4_exploit': '#cc0000',  # red
    'P2_exploit': '#3366cc',  # blue
    'HL_shuttle': '#999999',  # gray
    'Mixed': '#66aa66',       # green
}

fig, axes = plt.subplots(8, 3, figsize=(22, 28))
fig.suptitle("Entropy & Neural Metrics by Behavioral Mode — All 8 Sessions",
             fontsize=14, fontweight='bold')

for idx, snum in enumerate(range(1, 9)):
    sdf = mode_df[mode_df['session'] == snum]
    if len(sdf) == 0:
        for j in range(3):
            axes[idx, j].text(0.5, 0.5, f'S{snum} — no data', ha='center', va='center')
        continue

    state = sdf['state'].iloc[0]
    phase = sdf['phase'].iloc[0]

    # Col 0: Entropy colored by mode
    ax = axes[idx, 0]
    for mode, color in mode_colors.items():
        mask = sdf['mode'] == mode
        if mask.sum() > 0:
            ax.scatter(sdf.loc[mask, 'time'], sdf.loc[mask, 'entropy'],
                      c=color, s=8, alpha=0.7, label=mode if idx == 0 else '')
    ax.plot(sdf['time'], sdf['entropy'], color='black', linewidth=0.5, alpha=0.3)
    ax.set_ylabel("Entropy (bits)")
    ax.set_title(f"S{snum} ({state}/{phase})", fontsize=10)
    ax.set_ylim(0, 5.5)

    # Col 1: LHA FR colored by mode
    ax = axes[idx, 1]
    for mode, color in mode_colors.items():
        mask = sdf['mode'] == mode
        if mask.sum() > 0:
            ax.scatter(sdf.loc[mask, 'time'], sdf.loc[mask, 'lha_fr'],
                      c=color, s=8, alpha=0.7)
    ax.plot(sdf['time'], sdf['lha_fr'], color='black', linewidth=0.5, alpha=0.3)
    ax.set_ylabel("LHA FR (z)")
    ax.set_title(f"LHA FR", fontsize=10)

    # Col 2: RSP FR colored by mode
    ax = axes[idx, 2]
    for mode, color in mode_colors.items():
        mask = sdf['mode'] == mode
        if mask.sum() > 0:
            ax.scatter(sdf.loc[mask, 'time'], sdf.loc[mask, 'rsp_fr'],
                      c=color, s=8, alpha=0.7)
    ax.plot(sdf['time'], sdf['rsp_fr'], color='black', linewidth=0.5, alpha=0.3)
    ax.set_ylabel("RSP FR (z)")
    ax.set_title(f"RSP FR", fontsize=10)

axes[-1, 0].set_xlabel("Time (s)")
axes[-1, 1].set_xlabel("Time (s)")
axes[-1, 2].set_xlabel("Time (s)")

# Legend
legend_elements = [Patch(facecolor=c, label=m) for m, c in mode_colors.items()]
fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=11,
           bbox_to_anchor=(0.5, -0.01))

plt.tight_layout(rect=[0, 0.02, 1, 0.98])
plt.savefig("figures/entropy_mode_neural_traces.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/entropy_mode_neural_traces.png")


# =============================================================================
# FIGURE 2: Box/strip plots — neural level by mode, per session
# =============================================================================
fig, axes = plt.subplots(8, 4, figsize=(20, 28))
fig.suptitle("Neural Metrics by Behavioral Mode (box + strip)\n"
             "Does LHA/RSP respond the same to all types of stereotypy?",
             fontsize=14, fontweight='bold')

metric_cols = [('lha_fr', 'LHA FR'), ('rsp_fr', 'RSP FR'),
               ('lha_pc1', 'LHA PC1'), ('rsp_pc1', 'RSP PC1')]

for idx, snum in enumerate(range(1, 9)):
    sdf = mode_df[mode_df['session'] == snum]
    if len(sdf) == 0:
        for j in range(4):
            axes[idx, j].text(0.5, 0.5, 'no data', ha='center', va='center')
        continue

    state = sdf['state'].iloc[0]
    phase = sdf['phase'].iloc[0]

    for j, (col, label) in enumerate(metric_cols):
        ax = axes[idx, j]
        modes_present = [m for m in ['P4_exploit', 'P2_exploit', 'HL_shuttle', 'Mixed']
                         if (sdf['mode'] == m).sum() >= 3]
        if len(modes_present) == 0:
            ax.text(0.5, 0.5, 'no modes', ha='center', va='center')
            continue

        positions = []
        bp_data = []
        colors_bp = []
        for k, mode in enumerate(modes_present):
            vals = sdf.loc[sdf['mode'] == mode, col].values
            bp_data.append(vals)
            positions.append(k)
            colors_bp.append(mode_colors[mode])

        bp = ax.boxplot(bp_data, positions=positions, widths=0.6, patch_artist=True,
                       showfliers=False)
        for patch, color in zip(bp['boxes'], colors_bp):
            patch.set_facecolor(color)
            patch.set_alpha(0.4)

        # Strip plot
        for k, (mode, vals) in enumerate(zip(modes_present, bp_data)):
            jitter = np.random.normal(0, 0.08, len(vals))
            ax.scatter(np.full(len(vals), k) + jitter, vals,
                      c=mode_colors[mode], s=5, alpha=0.4, zorder=3)

        ax.set_xticks(positions)
        ax.set_xticklabels([m.replace('_', '\n') for m in modes_present], fontsize=7)
        if idx == 0:
            ax.set_title(label, fontsize=10, fontweight='bold')
        if j == 0:
            ax.set_ylabel(f"S{snum} ({state[:3]}/{phase[:3]})", fontsize=9)

plt.tight_layout()
plt.savefig("figures/entropy_mode_neural_boxplots.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/entropy_mode_neural_boxplots.png")


# =============================================================================
# FIGURE 3: Summary — mode-specific Spearman rho heatmap
# =============================================================================
fig, axes = plt.subplots(1, 4, figsize=(20, 8))
fig.suptitle("Entropy-Neural Spearman rho by Behavioral Mode\n"
             "(+) = neural UP with high entropy, (-) = neural UP with low entropy",
             fontsize=13, fontweight='bold')

for j, metric in enumerate(['LHA FR', 'RSP FR', 'LHA PC1', 'RSP PC1']):
    ax = axes[j]
    mdf = stats_df[stats_df['metric'] == metric]

    sessions = sorted(mdf['session'].unique())
    modes_all = ['P4_exploit', 'P2_exploit', 'HL_shuttle', 'Mixed']

    matrix = np.full((len(sessions), len(modes_all)), np.nan)
    sig_matrix = np.full((len(sessions), len(modes_all)), False)

    for i, snum in enumerate(sessions):
        for k, mode in enumerate(modes_all):
            row = mdf[(mdf['session'] == snum) & (mdf['mode'] == mode)]
            if len(row) == 1:
                matrix[i, k] = row['rho'].values[0]
                sig_matrix[i, k] = row['significant'].values[0]

    im = ax.imshow(matrix, cmap='RdBu_r', vmin=-0.8, vmax=0.8, aspect='auto')

    # Annotate
    for i in range(len(sessions)):
        for k in range(len(modes_all)):
            if not np.isnan(matrix[i, k]):
                txt = f"{matrix[i, k]:.2f}"
                if sig_matrix[i, k]:
                    txt += "*"
                ax.text(k, i, txt, ha='center', va='center', fontsize=8,
                       fontweight='bold' if sig_matrix[i, k] else 'normal')

    ax.set_xticks(range(len(modes_all)))
    ax.set_xticklabels([m.replace('_', '\n') for m in modes_all], fontsize=8)
    ax.set_yticks(range(len(sessions)))
    ylabels = [f"S{s} ({session_meta[s][0][:3]}/{session_meta[s][1][:3]})" for s in sessions]
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.set_title(metric, fontsize=11, fontweight='bold')

plt.colorbar(im, ax=axes, shrink=0.6, label='Spearman rho')
plt.tight_layout()
plt.savefig("figures/entropy_mode_neural_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/entropy_mode_neural_heatmap.png")


# =============================================================================
# SUMMARY TABLE
# =============================================================================
print("\n\n" + "=" * 80)
print("SUMMARY: IS THE NEURAL SIGNATURE CONSISTENT ACROSS MODES?")
print("=" * 80)

for metric in ['LHA FR', 'RSP FR', 'LHA PC1', 'RSP PC1']:
    mdf = stats_df[stats_df['metric'] == metric]
    print(f"\n{metric}:")
    print(f"  {'Mode':<14} {'N_sig/N_total':>14} {'Mean rho':>10} {'Sign consistency':>18}")
    for mode in ['P4_exploit', 'P2_exploit', 'HL_shuttle', 'Mixed']:
        sub = mdf[mdf['mode'] == mode]
        if len(sub) == 0:
            continue
        n_sig = sub['significant'].sum()
        n_tot = len(sub)
        mean_rho = sub['rho'].mean()
        # Sign consistency: how many have same sign as mean
        if mean_rho != 0:
            same_sign = np.sum(np.sign(sub['rho'].values) == np.sign(mean_rho))
        else:
            same_sign = 0
        print(f"  {mode:<14} {n_sig:>5}/{n_tot:<5}       {mean_rho:+.3f}     "
              f"{same_sign}/{n_tot} same sign")

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
