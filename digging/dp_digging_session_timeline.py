"""
Dual-Probe: Full-Session Digging Timeline
==========================================
Plots the ENTIRE session as continuous traces with dig bouts shaded and
labeled by pot location. Each session gets one tall figure showing:
  - Velocity
  - ACA FR (population mean, smoothed)
  - LHA FR (population mean, smoothed)
  - ACA PC1
  - LHA PC1
  - ACA Fano factor
  - LHA Fano factor
  - ACA-LHA cross-correlation (50ms lag, 1s sliding window)

Dig bouts are shaded in light color and labeled with pot ID.
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
import spikeinterface.extractors as se
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings

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

BIN_SEC = 1.0
SMOOTH_SIGMA = 3
MIN_DIG_DURATION = 2.0
MIN_INTER_DIG = 10.0

SKIP_SESSIONS = {23, 24}
TARGET_SESSIONS = None  # None = all sessions with digging

sessions_cfg = cfg["double_probe"]["coordinates_1"]["mouse01"]["sessions"]

zone_priority = [
    'Home corner left', 'Home corner right', 'Central Arena Zone',
    'Foraging arena', 'Home', 'ladder to Arena', 'Transition Zone',
    'Pot-1 zone', 'Pot-2 Zone', 'Pot-3 zone', 'Pot-4 zone',
    'Pot-1', 'Pot-2', 'Pot-3', 'Pot-4',
]
zone_short = {
    'Home': 'H', 'ladder to Arena': 'L', 'Transition Zone': 'T',
    'Foraging arena': 'FA', 'Central Arena Zone': 'CA',
    'Pot-1': 'P1', 'Pot-2': 'P2', 'Pot-3': 'P3', 'Pot-4': 'P4',
    'Pot-1 zone': 'P1z', 'Pot-2 Zone': 'P2z', 'Pot-3 zone': 'P3z', 'Pot-4 zone': 'P4z',
    'Home corner left': 'HCL', 'Home corner right': 'HCR',
}

DIG_COLORS = {
    'P1': '#e6194b', 'P2': '#3cb44b', 'P3': '#4363d8', 'P4': '#f58231',
    'unknown': '#999999',
}


def load_behavior_xlsx(path):
    df = pd.read_excel(path, header=None)
    col_names = df.iloc[34].tolist()
    data = df.iloc[36:].reset_index(drop=True)
    data.columns = col_names

    time_vals = pd.to_numeric(data['Recording time'], errors='coerce').values
    vel = pd.to_numeric(data['Velocity(Center-point)'], errors='coerce').values
    vel = np.nan_to_num(vel, nan=0.0)
    vel = np.clip(vel, 0, None)

    zones = np.full(len(time_vals), 'O', dtype=object)
    for zname in zone_priority:
        col_match = [c for c in col_names if isinstance(c, str) and
                     c.startswith('Zone(') and zname in c]
        if col_match:
            vals = pd.to_numeric(data[col_match[0]], errors='coerce').values
            mask = vals > 0.5
            short = zone_short.get(zname, zname[:3])
            zones[mask] = short

    dig_col = 'Digging sand'
    if dig_col in col_names:
        dig_vals = pd.to_numeric(data[dig_col], errors='coerce').values
        dig_vals = np.nan_to_num(dig_vals, nan=0.0)
    else:
        dig_vals = np.zeros(len(time_vals))

    return time_vals, vel, zones, dig_vals


def extract_dig_bouts(dig_vals, time_vals, min_duration, min_inter_dig):
    mask = dig_vals > 0.5
    diff = np.diff(mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if mask[0]:
        starts = np.concatenate([[0], starts])
    if mask[-1]:
        ends = np.concatenate([ends, [len(mask)]])
    if len(starts) == 0:
        return []
    bout_times = [(time_vals[s], time_vals[min(e - 1, len(time_vals) - 1)])
                  for s, e in zip(starts, ends)]
    merged = [bout_times[0]]
    for s, e in bout_times[1:]:
        if s - merged[-1][1] < min_inter_dig:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    bouts = []
    for s, e in merged:
        dur = e - s
        if dur >= min_duration:
            bouts.append({'start_time': s, 'end_time': e, 'duration': dur})
    return bouts


def get_pot_at_dig(zones, time_vals, dig_start):
    idx = np.searchsorted(time_vals, dig_start)
    dt = np.median(np.diff(time_vals))
    window = int(2.0 / dt)
    start = max(0, idx - window)
    end = min(len(zones), idx + window)
    segment = zones[start:end]
    pot_zones = [z for z in segment if z.startswith('P') and not z.endswith('z')]
    if pot_zones:
        from collections import Counter
        return Counter(pot_zones).most_common(1)[0][0]
    return 'unknown'


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


def compute_fano_factor(spike_counts_per_unit):
    mean_fr = np.mean(spike_counts_per_unit, axis=0)
    var_fr = np.var(spike_counts_per_unit, axis=0)
    mean_fr_safe = np.where(mean_fr > 0.01, mean_fr, 0.01)
    return var_fr / mean_fr_safe


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

print(f"Found {len(session_meta)} sessions")

out_dir = Path('figures/dp_digging_per_session')
out_dir.mkdir(exist_ok=True)

targets = TARGET_SESSIONS if TARGET_SESSIONS else sorted(session_meta.keys())

# Storage for pooled traces (normalized per session for cross-session comparison)
pooled_data = []

for snum in targets:
    if snum not in session_meta:
        print(f"S{snum} not available, skipping")
        continue

    meta = session_meta[snum]
    state, phase = meta['state'], meta['phase']
    print(f"\nProcessing S{snum} ({state}/{phase})...")

    # ---- Load behavior ----
    time_vals, vel, zones, dig_vals = load_behavior_xlsx(meta['behavior'])
    session_dur = time_vals[-1]
    dt_behav = np.median(np.diff(time_vals))

    bouts = extract_dig_bouts(dig_vals, time_vals, MIN_DIG_DURATION, MIN_INTER_DIG)
    print(f"  {len(bouts)} dig bouts")

    # Determine pot for each bout
    for b in bouts:
        b['pot'] = get_pot_at_dig(zones, time_vals, b['start_time'])
        print(f"    Dig {b['start_time']:.1f}-{b['end_time']:.1f}s "
              f"({b['duration']:.1f}s) at {b['pot']}")

    if len(bouts) == 0:
        print("  No valid dig bouts, skipping")
        continue

    # ---- Load neural data ----
    p0_path = Path(meta['p0_sorted'])
    p1_path = Path(meta['p1_sorted'])
    aca_ids = get_good_units_p0(p0_path)
    lha_ids = get_good_units_p1_lha(p1_path)

    sorting_p0 = se.read_kilosort(p0_path)
    avail_p0 = set(sorting_p0.get_unit_ids())
    aca_ids = np.array([u for u in aca_ids if u in avail_p0])

    sorting_p1 = se.read_kilosort(p1_path)
    avail_p1 = set(sorting_p1.get_unit_ids())
    lha_ids = np.array([u for u in lha_ids if u in avail_p1])

    has_aca = len(aca_ids) >= 2
    has_lha = len(lha_ids) >= 2
    print(f"  ACA={len(aca_ids)} units, LHA={len(lha_ids)} units")

    # ---- Compute 1s-binned spike counts for full session ----
    n_bins = int(session_dur / BIN_SEC)
    bin_edges = np.arange(n_bins + 1) * BIN_SEC
    time_centers = bin_edges[:-1] + BIN_SEC / 2

    if has_aca:
        aca_counts = np.zeros((len(aca_ids), n_bins))
        for i, uid in enumerate(aca_ids):
            st = sorting_p0.get_unit_spike_train(uid) / FS
            aca_counts[i] = np.histogram(st, bins=bin_edges)[0]

        aca_mean_fr = np.mean(aca_counts, axis=0)
        aca_mean_fr_smooth = gaussian_filter1d(aca_mean_fr, SMOOTH_SIGMA)

        # Fano factor
        aca_fano = compute_fano_factor(aca_counts)
        aca_fano_smooth = gaussian_filter1d(aca_fano, SMOOTH_SIGMA)

        # PC1
        aca_z = (aca_counts - aca_counts.mean(axis=1, keepdims=True)) / \
                (aca_counts.std(axis=1, keepdims=True) + 1e-6)
        pca_aca = PCA(n_components=1)
        aca_pc1 = pca_aca.fit_transform(aca_z.T).ravel()
        aca_pc1_smooth = gaussian_filter1d(aca_pc1, SMOOTH_SIGMA)

    if has_lha:
        lha_counts = np.zeros((len(lha_ids), n_bins))
        for i, uid in enumerate(lha_ids):
            st = sorting_p1.get_unit_spike_train(uid) / FS
            lha_counts[i] = np.histogram(st, bins=bin_edges)[0]

        lha_mean_fr = np.mean(lha_counts, axis=0)
        lha_mean_fr_smooth = gaussian_filter1d(lha_mean_fr, SMOOTH_SIGMA)

        # Fano factor
        lha_fano = compute_fano_factor(lha_counts)
        lha_fano_smooth = gaussian_filter1d(lha_fano, SMOOTH_SIGMA)

        # PC1
        lha_z = (lha_counts - lha_counts.mean(axis=1, keepdims=True)) / \
                (lha_counts.std(axis=1, keepdims=True) + 1e-6)
        pca_lha = PCA(n_components=1)
        lha_pc1 = pca_lha.fit_transform(lha_z.T).ravel()
        lha_pc1_smooth = gaussian_filter1d(lha_pc1, SMOOTH_SIGMA)

    # ---- Velocity trace (binned to 1s) ----
    vel_binned = np.zeros(n_bins)
    for i in range(n_bins):
        t0, t1 = bin_edges[i], bin_edges[i + 1]
        mask = (time_vals >= t0) & (time_vals < t1)
        if np.any(mask):
            vel_binned[i] = np.mean(vel[mask])
    vel_smooth = gaussian_filter1d(vel_binned, SMOOTH_SIGMA)

    # ---- Build panel list ----
    panels = []
    panels.append(('Velocity (cm/s)', time_centers, vel_smooth, '#555555'))
    if has_aca:
        panels.append(('ACA FR (spk/s, pop mean)', time_centers, aca_mean_fr_smooth, '#1f77b4'))
    if has_lha:
        panels.append(('LHA FR (spk/s, pop mean)', time_centers, lha_mean_fr_smooth, '#d62728'))
    if has_aca:
        panels.append(('ACA PC1', time_centers, aca_pc1_smooth, '#1f77b4'))
    if has_lha:
        panels.append(('LHA PC1', time_centers, lha_pc1_smooth, '#d62728'))
    if has_aca:
        panels.append(('ACA Fano Factor', time_centers, aca_fano_smooth, '#1f77b4'))
    if has_lha:
        panels.append(('LHA Fano Factor', time_centers, lha_fano_smooth, '#d62728'))

    n_panels = len(panels)

    # ---- PLOT ----
    fig, axes = plt.subplots(n_panels, 1, figsize=(28, 3.5 * n_panels),
                             sharex=True)
    if n_panels == 1:
        axes = [axes]

    for ax_idx, (label, t, trace, color) in enumerate(panels):
        ax = axes[ax_idx]
        ax.plot(t, trace, color=color, linewidth=1.2)

        # Shade dig bouts
        for bi, bout in enumerate(bouts):
            pot = bout['pot']
            dig_color = DIG_COLORS.get(pot, '#999999')
            ax.axvspan(bout['start_time'], bout['end_time'],
                       color=dig_color, alpha=0.15)

            # Label pot on top panel only
            if ax_idx == 0:
                mid = (bout['start_time'] + bout['end_time']) / 2
                y_top = ax.get_ylim()[1]
                ax.text(mid, y_top * 0.95, f"{pot}\n{bout['duration']:.0f}s",
                        ha='center', va='top', fontsize=10, fontweight='bold',
                        color=dig_color,
                        bbox=dict(boxstyle='round,pad=0.2', fc='white',
                                  ec=dig_color, alpha=0.8))

        ax.set_ylabel(label, fontsize=15)
        ax.tick_params(labelsize=13)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[-1].set_xlabel('Time (s)', fontsize=15)
    axes[-1].set_xlim(0, session_dur)

    # Re-do pot labels after ylim is set (first pass may have wrong ylim)
    ax0 = axes[0]
    # Clear old text and redo
    for txt in ax0.texts:
        txt.remove()
    y_min, y_max = ax0.get_ylim()
    for bi, bout in enumerate(bouts):
        pot = bout['pot']
        dig_color = DIG_COLORS.get(pot, '#999999')
        mid = (bout['start_time'] + bout['end_time']) / 2
        ax0.text(mid, y_max - (y_max - y_min) * 0.03,
                 f"{pot}\n{bout['duration']:.0f}s",
                 ha='center', va='top', fontsize=10, fontweight='bold',
                 color=dig_color,
                 bbox=dict(boxstyle='round,pad=0.2', fc='white',
                           ec=dig_color, alpha=0.8))

    # Legend for pot colors
    from matplotlib.patches import Patch
    pots_seen = sorted(set(b['pot'] for b in bouts))
    legend_handles = [Patch(facecolor=DIG_COLORS.get(p, '#999'), alpha=0.3,
                            edgecolor=DIG_COLORS.get(p, '#999'), label=p)
                      for p in pots_seen]
    axes[0].legend(handles=legend_handles, loc='upper right', fontsize=14,
                   title='Dig Location', title_fontsize=15)

    fig.suptitle(f'S{snum} — {state.capitalize()} / {phase.capitalize()} — '
                 f'Full Session Timeline ({len(bouts)} dig bouts, '
                 f'ACA={len(aca_ids)}, LHA={len(lha_ids)})',
                 fontsize=20, fontweight='bold', y=1.01)
    plt.tight_layout()

    fname = out_dir / f'S{snum}_{state}_{phase}_session_timeline.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {fname}")

    # ---- Store peri-dig windows for pooled figure ----
    PRE_WIN = 30  # seconds before onset (for baseline z-scoring)
    POST_WIN = 15
    n_pre = int(PRE_WIN / BIN_SEC)
    n_post = int(POST_WIN / BIN_SEC)
    n_win = n_pre + n_post

    for bout in bouts:
        onset = bout['start_time']
        onset_bin = int(onset / BIN_SEC)
        start_bin = onset_bin - n_pre
        end_bin = onset_bin + n_post

        if start_bin < 0 or end_bin > n_bins:
            continue

        entry = {
            'session': snum, 'state': state, 'phase': phase,
            'pot': bout['pot'], 'duration': bout['duration'],
            'vel': vel_smooth[start_bin:end_bin],
        }
        if has_aca:
            entry['aca_fr'] = aca_mean_fr_smooth[start_bin:end_bin]
            entry['aca_pc1'] = aca_pc1_smooth[start_bin:end_bin]
            entry['aca_fano'] = aca_fano_smooth[start_bin:end_bin]
        if has_lha:
            entry['lha_fr'] = lha_mean_fr_smooth[start_bin:end_bin]
            entry['lha_pc1'] = lha_pc1_smooth[start_bin:end_bin]
            entry['lha_fano'] = lha_fano_smooth[start_bin:end_bin]
        pooled_data.append(entry)

# ========================================================================
# POOLED FIGURE: peri-dig onset, all sessions, z-scored to baseline
# ========================================================================
print(f"\nPooled: {len(pooled_data)} dig events across all sessions")

if len(pooled_data) > 0:
    from matplotlib.patches import Patch

    PRE_WIN = 30
    POST_WIN = 15
    n_pre = int(PRE_WIN / BIN_SEC)
    n_post = int(POST_WIN / BIN_SEC)
    n_win = n_pre + n_post
    peri_time = np.arange(-n_pre, n_post) * BIN_SEC + BIN_SEC / 2

    # Baseline window for z-scoring: [-30s, -15s]
    bl_start, bl_end = 0, n_pre - 15

    metric_keys = [
        ('vel', 'Velocity (Z)', '#555555'),
        ('aca_fr', 'ACA FR (Z)', '#1f77b4'),
        ('lha_fr', 'LHA FR (Z)', '#d62728'),
        ('aca_pc1', 'ACA PC1 (Z)', '#1f77b4'),
        ('lha_pc1', 'LHA PC1 (Z)', '#d62728'),
        ('aca_fano', 'ACA Fano (Z)', '#1f77b4'),
        ('lha_fano', 'LHA Fano (Z)', '#d62728'),
    ]

    state_colors = {'fed': '#4e79a7', 'fasted': '#e15759', 'fed-HFD': '#f28e2b'}
    state_labels = {'fed': 'Fed', 'fasted': 'Fasted', 'fed-HFD': 'HFD'}

    def zscore_to_bl(trace, bl_s, bl_e):
        bl = trace[bl_s:bl_e]
        m, s = np.mean(bl), np.std(bl)
        if s < 1e-6:
            s = 1e-6
        return (trace - m) / s

    # ---- Pooled: all events ----
    fig, axes = plt.subplots(len(metric_keys), 1, figsize=(24, 3.5 * len(metric_keys)),
                             sharex=True)

    for ax_idx, (key, label, color) in enumerate(metric_keys):
        ax = axes[ax_idx]
        traces = []
        for d in pooled_data:
            if key in d:
                z = zscore_to_bl(d[key], bl_start, bl_end)
                traces.append(z)
        if len(traces) == 0:
            ax.set_title(f'{label} — no data')
            continue
        traces = np.array(traces)
        mean_tr = np.nanmean(traces, axis=0)
        sem_tr = np.nanstd(traces, axis=0) / np.sqrt(len(traces))

        ax.plot(peri_time, mean_tr, color=color, linewidth=2.5,
                label=f'All (n={len(traces)})')
        ax.fill_between(peri_time, mean_tr - sem_tr, mean_tr + sem_tr,
                        color=color, alpha=0.15)

        ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axvspan(-3, 0, color='#2196F3', alpha=0.08)
        ax.axvspan(1, 4, color='#4CAF50', alpha=0.08)
        ax.set_xlim(-5, 10)
        ax.set_ylabel(label, fontsize=15)
        ax.tick_params(labelsize=13)
        ax.legend(fontsize=13)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[-1].set_xlabel('Time from dig onset (s)', fontsize=15)
    fig.suptitle(f'All Sessions Pooled — Peri-Dig Onset (n={len(pooled_data)} events)',
                 fontsize=20, fontweight='bold', y=1.01)
    plt.tight_layout()
    fname = out_dir / 'pooled_all_sessions_peri_dig.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {fname}")

    # ---- Pooled: by state ----
    fig, axes = plt.subplots(len(metric_keys), 1, figsize=(24, 3.5 * len(metric_keys)),
                             sharex=True)

    for ax_idx, (key, label, color) in enumerate(metric_keys):
        ax = axes[ax_idx]

        for sf, sc in state_colors.items():
            traces = []
            for d in pooled_data:
                if d['state'] == sf and key in d:
                    z = zscore_to_bl(d[key], bl_start, bl_end)
                    traces.append(z)
            if len(traces) < 2:
                continue
            traces = np.array(traces)
            mean_tr = np.nanmean(traces, axis=0)
            sem_tr = np.nanstd(traces, axis=0) / np.sqrt(len(traces))

            ax.plot(peri_time, mean_tr, color=sc, linewidth=2,
                    label=f'{state_labels[sf]} (n={len(traces)})')
            ax.fill_between(peri_time, mean_tr - sem_tr, mean_tr + sem_tr,
                            color=sc, alpha=0.12)

        ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axvspan(-3, 0, color='#2196F3', alpha=0.08)
        ax.axvspan(1, 4, color='#4CAF50', alpha=0.08)
        ax.set_xlim(-5, 10)
        ax.set_ylabel(label, fontsize=15)
        ax.tick_params(labelsize=13)
        ax.legend(fontsize=13)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[-1].set_xlabel('Time from dig onset (s)', fontsize=15)
    fig.suptitle(f'By State — Peri-Dig Onset (n={len(pooled_data)} events)',
                 fontsize=20, fontweight='bold', y=1.01)
    plt.tight_layout()
    fname = out_dir / 'pooled_by_state_peri_dig.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {fname}")

print("\nDone.")
