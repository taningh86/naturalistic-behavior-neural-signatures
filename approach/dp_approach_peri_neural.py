"""
Dual-Probe: Peri-Approach Neural Analysis
==========================================
Align ACA and LHA neural data to approach target-arrival (t = 0)
for each approach type extracted in dp_approach_events.py.

Window: -5 s to +3 s, 250 ms bins.
Baseline: -5 s to -3 s  (pre-approach, before locomotion ramp).
Approach phase: -3 s to 0 s (transit).
Post-arrival:   0 to +3 s (contact / dig / feed / ladder mount).

Metrics per event (baseline-subtracted mean per window):
  ACA FR (z-scored population)
  LHA FR (z-scored population)
  ACA PC1 (session PCA, first component)
  LHA PC1
  Velocity

Stats:
  Wilcoxon signed-rank: approach vs baseline, post vs baseline
  Mann-Whitney U / Kruskal-Wallis across states (fed / fasted / fed-HFD)
  Per approach type (pot_approach, ladder_from_arena, pre_dig, pre_feed)
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import wilcoxon, mannwhitneyu, kruskal
from sklearn.decomposition import PCA
import spikeinterface.extractors as se
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
import time as timer

warnings.filterwarnings('ignore')

with open("paths.yaml") as f:
    cfg = yaml.safe_load(f)

# ---- Constants ----
FS = 30000
LHA_DEPTH_MIN = 0
LHA_DEPTH_MAX = 345
P0_MIN_FR = 0.2
P1_MIN_FR = 0.2
P1_MIN_AMP = 43

BIN_SEC = 0.25
PRE_SEC = 5.0
POST_SEC = 3.0
SMOOTH_SIGMA = 2  # in bins → 500 ms

BASELINE_WINDOW = (-5.0, -3.0)
APPROACH_WINDOW = (-3.0, -0.5)
AT_ARRIVAL_WINDOW = (-0.5, 0.5)
POST_WINDOW = (0.5, 3.0)

SKIP_SESSIONS = {23, 24}

EVENT_TYPES = ['pot_approach', 'ladder_from_arena', 'pre_dig', 'pre_feed']
METRIC_NAMES = ['Velocity', 'ACA FR', 'LHA FR', 'ACA PC1', 'LHA PC1']
STATE_ORDER = ['fed', 'fasted', 'fed-HFD']
STATE_COLORS = {'fed': '#2471A3', 'fasted': '#C0392B', 'fed-HFD': '#8E44AD'}
TYPE_COLORS = {
    'pot_approach': '#16A085',
    'ladder_from_arena': '#E67E22',
    'pre_dig': '#8B4513',
    'pre_feed': '#D4AC0D',
}

OUT_DIR = Path("data/approach_peri")
FIG_DIR = Path("figures/approach_peri")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

sessions_cfg = cfg["double_probe"]["coordinates_1"]["mouse01"]["sessions"]


def load_behavior_xlsx_vel(path):
    df = pd.read_excel(path, header=None)
    col_names = df.iloc[34].tolist()
    data = df.iloc[36:].reset_index(drop=True)
    data.columns = col_names
    time_vals = pd.to_numeric(data['Recording time'], errors='coerce').values
    vel = pd.to_numeric(data['Velocity(Center-point)'], errors='coerce').values
    vel = np.nan_to_num(vel, nan=0.0)
    vel = np.clip(vel, 0, None)
    return time_vals, vel


def get_good_units_p0(sorted_path):
    ci = Path(sorted_path) / "cluster_info.tsv"
    if not ci.exists():
        return np.array([])
    df = pd.read_csv(ci, sep='\t')
    label_col = 'group' if ('group' in df.columns and df['group'].eq('good').any()) else 'KSLabel'
    if label_col not in df.columns:
        return np.array([])
    good = df[(df[label_col] == 'good') & (df['fr'] > P0_MIN_FR)]
    return good['cluster_id'].values


def get_good_units_p1_lha(sorted_path):
    ci = Path(sorted_path) / "cluster_info.tsv"
    if not ci.exists():
        return np.array([])
    df = pd.read_csv(ci, sep='\t')
    label_col = 'group' if ('group' in df.columns and df['group'].eq('good').any()) else 'KSLabel'
    if label_col not in df.columns or 'depth' not in df.columns:
        return np.array([])
    good = df[(df[label_col] == 'good') &
              (df['fr'] > P1_MIN_FR) &
              (df['amp'] > P1_MIN_AMP) &
              (df['depth'] >= LHA_DEPTH_MIN) &
              (df['depth'] <= LHA_DEPTH_MAX)]
    return good['cluster_id'].values


# ========================================================================
# Load events
# ========================================================================
events_df = pd.read_csv("data/dp_approach_events.csv")
events_df = events_df[events_df['approach_type'].isin(EVENT_TYPES)].reset_index(drop=True)
print(f"Loaded {len(events_df)} approach events across {len(EVENT_TYPES)} types.")

# ========================================================================
# Discover sessions
# ========================================================================
session_meta = {}
for skey, sval in sessions_cfg.items():
    snum = int(skey.split('_')[1])
    if snum in SKIP_SESSIONS:
        continue
    p0_sorted = sval.get('probe_0_aca', {}).get('sorted')
    p1_sorted = sval.get('probe_1_lha_rsp', {}).get('sorted')
    behav = sval.get('behavior')
    if not p0_sorted or not p1_sorted or not behav:
        continue
    if not Path(p0_sorted).exists() or not Path(behav).exists():
        continue
    session_meta[snum] = {
        'state': sval['state'], 'phase': sval['phase'],
        'p0_sorted': p0_sorted, 'p1_sorted': p1_sorted,
        'behavior': behav,
    }

print(f"Found {len(session_meta)} sessions with sorted data + behavior.")

# ========================================================================
# Time axis
# ========================================================================
n_pre_bins = int(PRE_SEC / BIN_SEC)
n_post_bins = int(POST_SEC / BIN_SEC)
n_total_bins = n_pre_bins + n_post_bins
time_axis = np.arange(-n_pre_bins, n_post_bins) * BIN_SEC + BIN_SEC / 2

# Window indices
def win_mask(t0, t1):
    return (time_axis >= t0) & (time_axis < t1)

BASE_MASK = win_mask(*BASELINE_WINDOW)
APPR_MASK = win_mask(*APPROACH_WINDOW)
AT_MASK = win_mask(*AT_ARRIVAL_WINDOW)
POST_MASK = win_mask(*POST_WINDOW)

# Storage: for each metric, array of (n_events, n_bins)
traces = {m: [] for m in METRIC_NAMES}
# Parallel meta list (session, state, phase, event_idx, approach_type, target_zone, origin_zone, duration)
meta_rows = []

# ========================================================================
# Process each session
# ========================================================================
print("\n" + "=" * 100)
print("PERI-APPROACH EXTRACTION")
print("=" * 100)

for snum in sorted(session_meta.keys()):
    t0 = timer.time()
    meta = session_meta[snum]
    state, phase = meta['state'], meta['phase']
    sess_events = events_df[events_df['session'] == snum]
    if len(sess_events) == 0:
        continue
    print(f"\n  S{snum} ({state}/{phase}): {len(sess_events)} events,", end='', flush=True)

    time_vals, vel = load_behavior_xlsx_vel(meta['behavior'])

    # Neural
    p0_path = Path(meta['p0_sorted'])
    p1_path = Path(meta['p1_sorted'])
    aca_ids = get_good_units_p0(p0_path)
    lha_ids = get_good_units_p1_lha(p1_path)

    try:
        sorting_p0 = se.read_kilosort(p0_path)
        avail_p0 = set(sorting_p0.get_unit_ids())
        aca_ids = np.array([u for u in aca_ids if u in avail_p0])
    except Exception:
        aca_ids = np.array([])
    try:
        sorting_p1 = se.read_kilosort(p1_path)
        avail_p1 = set(sorting_p1.get_unit_ids())
        lha_ids = np.array([u for u in lha_ids if u in avail_p1])
    except Exception:
        lha_ids = np.array([])

    print(f" ACA={len(aca_ids)}, LHA={len(lha_ids)},", end='', flush=True)

    has_aca = len(aca_ids) >= 2
    has_lha = len(lha_ids) >= 2

    # Session-level binning
    session_end = time_vals[-1] + 1
    bin_edges = np.arange(0, session_end + BIN_SEC, BIN_SEC)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Velocity
    vel_binned = np.interp(bin_centers, time_vals, vel)

    # ACA
    if has_aca:
        aca_counts = np.array([np.histogram(sorting_p0.get_unit_spike_train(u) / FS,
                                            bins=bin_edges)[0] for u in aca_ids])
        aca_z = np.array([(r - r.mean()) / max(r.std(), 1e-6) for r in aca_counts])
        aca_pop = np.mean(aca_z, axis=0)
        aca_pc1 = PCA(n_components=min(3, len(aca_ids))).fit_transform(aca_z.T)[:, 0]
        # smooth session-level traces lightly
        aca_pop = gaussian_filter1d(aca_pop, SMOOTH_SIGMA)
        aca_pc1 = gaussian_filter1d(aca_pc1, SMOOTH_SIGMA)

    # LHA
    if has_lha:
        lha_counts = np.array([np.histogram(sorting_p1.get_unit_spike_train(u) / FS,
                                            bins=bin_edges)[0] for u in lha_ids])
        lha_z = np.array([(r - r.mean()) / max(r.std(), 1e-6) for r in lha_counts])
        lha_pop = np.mean(lha_z, axis=0)
        lha_pc1 = PCA(n_components=min(3, len(lha_ids))).fit_transform(lha_z.T)[:, 0]
        lha_pop = gaussian_filter1d(lha_pop, SMOOTH_SIGMA)
        lha_pc1 = gaussian_filter1d(lha_pc1, SMOOTH_SIGMA)

    n_valid = 0
    for _, ev in sess_events.iterrows():
        t_arrival = ev['t_end']
        center_bin = int(t_arrival / BIN_SEC)
        start_bin = center_bin - n_pre_bins
        end_bin = center_bin + n_post_bins
        if start_bin < 0 or end_bin > len(bin_centers):
            continue

        # Extract traces
        w_vel = vel_binned[start_bin:end_bin]
        w_aca_fr = aca_pop[start_bin:end_bin] if has_aca else np.full(n_total_bins, np.nan)
        w_lha_fr = lha_pop[start_bin:end_bin] if has_lha else np.full(n_total_bins, np.nan)
        w_aca_pc1 = aca_pc1[start_bin:end_bin] if has_aca else np.full(n_total_bins, np.nan)
        w_lha_pc1 = lha_pc1[start_bin:end_bin] if has_lha else np.full(n_total_bins, np.nan)

        # Baseline-subtract (mean over BASE_MASK)
        def bsub(w):
            if np.all(np.isnan(w)):
                return w
            return w - np.nanmean(w[BASE_MASK])

        traces['Velocity'].append(bsub(w_vel))
        traces['ACA FR'].append(bsub(w_aca_fr))
        traces['LHA FR'].append(bsub(w_lha_fr))
        traces['ACA PC1'].append(bsub(w_aca_pc1))
        traces['LHA PC1'].append(bsub(w_lha_pc1))

        meta_rows.append({
            'session': snum, 'state': state, 'phase': phase,
            'event_idx': int(ev['event_idx']),
            'approach_type': ev['approach_type'],
            'target_zone': ev['target_zone'],
            'origin_zone': ev['origin_zone'],
            'duration': float(ev['duration']),
            'has_aca': has_aca,
            'has_lha': has_lha,
        })
        n_valid += 1

    dt = timer.time() - t0
    print(f" {n_valid} events extracted ({dt:.1f}s)")

# ========================================================================
# Build pooled arrays
# ========================================================================
for m in METRIC_NAMES:
    traces[m] = np.array(traces[m])
meta_df = pd.DataFrame(meta_rows)
print(f"\nTotal pooled events: {len(meta_df)}")
print(meta_df.groupby(['approach_type', 'state']).size().unstack(fill_value=0))

# ========================================================================
# Per-event window means (CSV)
# ========================================================================
rows = []
for i, m in meta_df.iterrows():
    row = dict(m)
    for mname in METRIC_NAMES:
        arr = traces[mname][i]
        if np.all(np.isnan(arr)):
            row[f'{mname}_approach_mean'] = np.nan
            row[f'{mname}_at_mean'] = np.nan
            row[f'{mname}_post_mean'] = np.nan
            continue
        row[f'{mname}_approach_mean'] = float(np.nanmean(arr[APPR_MASK]))
        row[f'{mname}_at_mean'] = float(np.nanmean(arr[AT_MASK]))
        row[f'{mname}_post_mean'] = float(np.nanmean(arr[POST_MASK]))
    rows.append(row)

per_event = pd.DataFrame(rows)
per_event.to_csv(OUT_DIR / 'peri_event_window_means.csv', index=False)
print(f"\nSaved {OUT_DIR / 'peri_event_window_means.csv'}")

# Save raw traces for downstream window analyses
np.savez_compressed(
    OUT_DIR / 'peri_event_traces.npz',
    time_axis=time_axis,
    **{m.replace(' ', '_'): traces[m] for m in METRIC_NAMES},
)
meta_df.to_csv(OUT_DIR / 'peri_event_meta.csv', index=False)
print(f"Saved {OUT_DIR / 'peri_event_traces.npz'} + peri_event_meta.csv")

# ========================================================================
# Statistics
# ========================================================================
print("\n" + "=" * 100)
print("STATS: approach vs baseline (Wilcoxon signed-rank on baseline-subtracted means)")
print("=" * 100)

stats_rows = []
for atype in EVENT_TYPES:
    sub = per_event[per_event['approach_type'] == atype]
    n = len(sub)
    if n < 5:
        continue
    print(f"\n  {atype} (n={n})")
    for mname in METRIC_NAMES:
        a_col = f'{mname}_approach_mean'
        at_col = f'{mname}_at_mean'
        p_col = f'{mname}_post_mean'
        a_vals = sub[a_col].dropna().values
        at_vals = sub[at_col].dropna().values
        p_vals = sub[p_col].dropna().values
        if len(a_vals) < 5:
            continue
        try:
            w_a = wilcoxon(a_vals).pvalue
        except Exception:
            w_a = np.nan
        try:
            w_at = wilcoxon(at_vals).pvalue
        except Exception:
            w_at = np.nan
        try:
            w_p = wilcoxon(p_vals).pvalue
        except Exception:
            w_p = np.nan
        print(f"    {mname:10s} approach={a_vals.mean():+.3f} (p={w_a:.3f})  "
              f"at_arrival={at_vals.mean():+.3f} (p={w_at:.3f})  "
              f"post={p_vals.mean():+.3f} (p={w_p:.3f})")
        stats_rows.append({
            'approach_type': atype, 'metric': mname,
            'n': len(a_vals),
            'approach_mean': a_vals.mean(), 'approach_p': w_a,
            'at_arrival_mean': at_vals.mean(), 'at_arrival_p': w_at,
            'post_mean': p_vals.mean(), 'post_p': w_p,
        })

# State comparison (Kruskal-Wallis)
print("\n" + "=" * 100)
print("STATS: across states (Kruskal-Wallis + pairwise Mann-Whitney)")
print("=" * 100)

state_stats = []
for atype in EVENT_TYPES:
    sub = per_event[per_event['approach_type'] == atype]
    if len(sub) < 15:
        continue
    print(f"\n  {atype}")
    for mname in METRIC_NAMES:
        col = f'{mname}_approach_mean'
        groups = [sub[sub['state'] == st][col].dropna().values for st in STATE_ORDER]
        sizes = [len(g) for g in groups]
        if any(s < 3 for s in sizes):
            continue
        try:
            kw_p = kruskal(*groups).pvalue
        except Exception:
            kw_p = np.nan
        print(f"    {mname:10s} KW p={kw_p:.3f}  "
              f"fed={groups[0].mean():+.3f} (n={sizes[0]}) "
              f"fas={groups[1].mean():+.3f} (n={sizes[1]}) "
              f"hfd={groups[2].mean():+.3f} (n={sizes[2]})")
        state_stats.append({
            'approach_type': atype, 'metric': mname, 'window': 'approach',
            'kw_p': kw_p,
            'fed_mean': groups[0].mean(), 'fasted_mean': groups[1].mean(),
            'hfd_mean': groups[2].mean(),
            'fed_n': sizes[0], 'fasted_n': sizes[1], 'hfd_n': sizes[2],
        })

pd.DataFrame(stats_rows).to_csv(OUT_DIR / 'peri_event_wilcoxon.csv', index=False)
pd.DataFrame(state_stats).to_csv(OUT_DIR / 'peri_event_state_kw.csv', index=False)

# ========================================================================
# FIGURES
# ========================================================================
print("\nBuilding figures...")

# Figure 1: Pooled mean ± SEM per event type × metric (all states combined)
fig, axes = plt.subplots(len(METRIC_NAMES), len(EVENT_TYPES),
                         figsize=(4 * len(EVENT_TYPES), 2.4 * len(METRIC_NAMES)),
                         sharex=True, constrained_layout=True)

for mi, mname in enumerate(METRIC_NAMES):
    for ei, atype in enumerate(EVENT_TYPES):
        ax = axes[mi, ei]
        mask = (meta_df['approach_type'] == atype).values
        arr = traces[mname][mask]
        # drop all-NaN events
        valid = ~np.all(np.isnan(arr), axis=1)
        arr = arr[valid]
        if len(arr) == 0:
            ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes)
            continue
        mean = np.nanmean(arr, axis=0)
        sem = np.nanstd(arr, axis=0) / np.sqrt(np.sum(~np.isnan(arr), axis=0))
        color = TYPE_COLORS[atype]
        ax.plot(time_axis, mean, color=color, lw=2)
        ax.fill_between(time_axis, mean - sem, mean + sem, color=color, alpha=0.25)
        ax.axvline(0, color='k', lw=0.8, ls='--')
        ax.axhline(0, color='gray', lw=0.5, ls=':')
        ax.axvspan(*BASELINE_WINDOW, color='gray', alpha=0.08)
        if mi == 0:
            ax.set_title(f"{atype} (n={len(arr)})", fontsize=11)
        if ei == 0:
            ax.set_ylabel(mname, fontsize=11)
        if mi == len(METRIC_NAMES) - 1:
            ax.set_xlabel("time from arrival (s)")
        ax.grid(alpha=0.2)

fig.suptitle('Peri-Approach Neural Dynamics (baseline-subtracted)', fontsize=14, fontweight='bold')
fig.savefig(FIG_DIR / 'dp_peri_approach_pooled.png', dpi=150)
plt.close(fig)
print(f"Saved {FIG_DIR / 'dp_peri_approach_pooled.png'}")

# Figure 2: Same by state
fig, axes = plt.subplots(len(METRIC_NAMES), len(EVENT_TYPES),
                         figsize=(4 * len(EVENT_TYPES), 2.4 * len(METRIC_NAMES)),
                         sharex=True, constrained_layout=True)

for mi, mname in enumerate(METRIC_NAMES):
    for ei, atype in enumerate(EVENT_TYPES):
        ax = axes[mi, ei]
        for state in STATE_ORDER:
            mask = ((meta_df['approach_type'] == atype) &
                    (meta_df['state'] == state)).values
            arr = traces[mname][mask]
            valid = ~np.all(np.isnan(arr), axis=1)
            arr = arr[valid]
            if len(arr) < 3:
                continue
            mean = np.nanmean(arr, axis=0)
            sem = np.nanstd(arr, axis=0) / np.sqrt(np.sum(~np.isnan(arr), axis=0))
            color = STATE_COLORS[state]
            ax.plot(time_axis, mean, color=color, lw=1.6, label=f"{state} (n={len(arr)})")
            ax.fill_between(time_axis, mean - sem, mean + sem, color=color, alpha=0.15)
        ax.axvline(0, color='k', lw=0.8, ls='--')
        ax.axhline(0, color='gray', lw=0.5, ls=':')
        if mi == 0:
            ax.set_title(atype, fontsize=11)
            ax.legend(fontsize=8, loc='upper left')
        if ei == 0:
            ax.set_ylabel(mname, fontsize=11)
        if mi == len(METRIC_NAMES) - 1:
            ax.set_xlabel("time from arrival (s)")
        ax.grid(alpha=0.2)

fig.suptitle('Peri-Approach Neural Dynamics by State', fontsize=14, fontweight='bold')
fig.savefig(FIG_DIR / 'dp_peri_approach_by_state.png', dpi=150)
plt.close(fig)
print(f"Saved {FIG_DIR / 'dp_peri_approach_by_state.png'}")

print("\nDone.")
