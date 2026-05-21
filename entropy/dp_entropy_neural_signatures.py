"""
Dual-Probe: Behavioral Entropy vs Neural Signatures
=====================================================
Adapts single-probe entropy analysis for dual-probe data:
- Probe 0: ACA
- Probe 1: LHA (depth 0-345 um)
- Behavior: xlsx format (36-row header, 25Hz / 0.04s steps)

Analysis:
1. Per-session Spearman correlations (raw + partial|velocity)
2. Peri-inflection analysis (peaks/troughs of entropy)
3. Pooled peri-inflection across sessions

CAUSAL ENTROPY: assigned to END of 60s window.
Unit selection: Probe 0 FR>0.2, no AMP; Probe 1 FR>0.2, AMP>43
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import entropy as sp_entropy, spearmanr, mannwhitneyu, wilcoxon, rankdata
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema
from sklearn.decomposition import PCA
from collections import Counter
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
P0_MIN_FR = 0.2   # ACA: FR > 0.2, no AMP filter
P1_MIN_FR = 0.2   # LHA: FR > 0.2
P1_MIN_AMP = 43   # LHA: AMP > 43

ENTROPY_WINDOW_SEC = 60
ENTROPY_STEP_SEC = 10
PERI_WINDOW = 12       # +/-120s in entropy steps
SMOOTH_SIGMA = 3       # for inflection detection
MIN_AMPLITUDE = 0.3    # min entropy swing for inflection

sessions_cfg = cfg["double_probe"]["coordinates_1"]["mouse01"]["sessions"]

# Skip sessions with different behavior paradigm (NEW_PARADIGM from 9-11-25)
SKIP_SESSIONS = {23, 24}

# Session metadata — only sessions with sorted data AND behavior
session_meta = {}
for skey, sval in sessions_cfg.items():
    snum = int(skey.split('_')[1])
    if snum in SKIP_SESSIONS:
        continue
    # Need both probes sorted and behavior
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

print(f"Found {len(session_meta)} sessions with sorted data + behavior:")
for snum, meta in sorted(session_meta.items()):
    print(f"  S{snum}: {meta['state']}/{meta['phase']}")

# Zone mapping for dual-probe behavior xlsx
# Column names: "Zone(XXX / any of Center-point, Nose-point)"
# Priority order (higher priority zones override lower)
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
    'Lever choice zone': 'LCZ', 'Lever Zone': 'LZ',
    'Pot choice zone': 'PCZ', 'Sand Pots Zone': 'SPZ',
    'Lever1 food zone': 'L1F', 'Lever2 food zone': 'L2F',
}


def load_behavior_xlsx(path):
    """Load dual-probe behavior xlsx (36-row header, 25Hz)."""
    df = pd.read_excel(path, header=None)
    col_names = df.iloc[34].tolist()
    data = df.iloc[36:].reset_index(drop=True)
    data.columns = col_names

    time_vals = pd.to_numeric(data['Recording time'], errors='coerce').values
    vel = pd.to_numeric(data['Velocity(Center-point)'], errors='coerce').values
    vel = np.nan_to_num(vel, nan=0.0)
    vel = np.clip(vel, 0, None)

    # Build zone array using priority
    zones = np.full(len(time_vals), 'O', dtype=object)
    for zname in zone_priority:
        col_match = [c for c in col_names if isinstance(c, str) and
                     c.startswith('Zone(') and zname in c]
        if col_match:
            vals = pd.to_numeric(data[col_match[0]], errors='coerce').values
            mask = vals > 0.5
            short = zone_short.get(zname, zname[:3])
            zones[mask] = short

    return time_vals, vel, zones


def compute_entropy_causal(zones, time_vals, vel, window_sec, step_sec):
    """Compute behavioral entropy (causal, end-assigned)."""
    dt = np.median(np.diff(time_vals))
    window_bins = int(window_sec / dt)
    step_bins = int(step_sec / dt)

    ent_times, ent_vals, vel_means = [], [], []
    for start_idx in range(0, len(zones) - window_bins, step_bins):
        wz = zones[start_idx:start_idx + window_bins]
        transitions = []
        for j in range(1, len(wz)):
            if wz[j] != wz[j - 1]:
                transitions.append(f"{wz[j-1]}->{wz[j]}")
        if len(transitions) < 3:
            continue
        counts = Counter(transitions)
        probs = np.array(list(counts.values()), dtype=float)
        probs /= probs.sum()
        h = sp_entropy(probs, base=2)
        ent_times.append(time_vals[start_idx + window_bins - 1])  # CAUSAL
        ent_vals.append(h)
        vel_means.append(np.nanmean(vel[start_idx:start_idx + window_bins]))

    return np.array(ent_times), np.array(ent_vals), np.array(vel_means)


def get_good_units_p0(sorted_path):
    """Probe 0 (ACA): KSLabel='good' + FR > 0.2, no AMP filter."""
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
    """Probe 1 (LHA): KSLabel='good' + FR > 0.2 + AMP > 43, depth 0-345um."""
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


def partial_spearman(x, y, z):
    """Partial Spearman correlation controlling for z."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    def resid(a, b):
        slope = np.polyfit(b, a, 1)
        return a - np.polyval(slope, b)
    return spearmanr(resid(rx, rz), resid(ry, rz))


def find_inflections(values, smooth_sigma, min_amp, order=3):
    smoothed = gaussian_filter1d(values, smooth_sigma)
    peaks = list(argrelextrema(smoothed, np.greater, order=order)[0])
    troughs = list(argrelextrema(smoothed, np.less, order=order)[0])
    all_ext = sorted([(p, 'peak', smoothed[p]) for p in peaks] +
                     [(t, 'trough', smoothed[t]) for t in troughs],
                     key=lambda x: x[0])
    if len(all_ext) < 2:
        return [], [], smoothed
    filtered = [all_ext[0]]
    for i in range(1, len(all_ext)):
        if all_ext[i][1] == filtered[-1][1]:
            if all_ext[i][1] == 'peak':
                if all_ext[i][2] > filtered[-1][2]:
                    filtered[-1] = all_ext[i]
            else:
                if all_ext[i][2] < filtered[-1][2]:
                    filtered[-1] = all_ext[i]
        else:
            amp = abs(all_ext[i][2] - filtered[-1][2])
            if amp >= min_amp:
                filtered.append(all_ext[i])
    final_peaks = [f[0] for f in filtered if f[1] == 'peak']
    final_troughs = [f[0] for f in filtered if f[1] == 'trough']
    return final_peaks, final_troughs, smoothed


# ========================================================================
# MAIN ANALYSIS
# ========================================================================
print("\n" + "=" * 110)
print("DUAL-PROBE: BEHAVIORAL ENTROPY vs NEURAL SIGNATURES")
print("ACA (probe 0) and LHA (probe 1, 0-345um)")
print("=" * 110)

all_corr_results = []
all_infl_results = []
metric_names = ['Entropy', 'Velocity', 'ACA FR', 'LHA FR', 'ACA PC1', 'LHA PC1']
neural_metrics = ['ACA FR', 'LHA FR', 'ACA PC1', 'LHA PC1']

# Pooled peri-inflection storage — track state per metric per event
pooled_peaks = {m: [] for m in metric_names}
pooled_troughs = {m: [] for m in metric_names}
peak_states_per_metric = {m: [] for m in metric_names}
trough_states_per_metric = {m: [] for m in metric_names}
peak_sessions = []
trough_sessions = []
peak_states = []
trough_states = []
STATE_ORDER = ['fed', 'fasted', 'fed-HFD', 'fasted-HFD']
STATE_LABELS = {'fed': 'Fed', 'fasted': 'Fasted', 'fed-HFD': 'HFD', 'fasted-HFD': 'HFD-Fasted'}

for snum in sorted(session_meta.keys()):
    t0 = timer.time()
    meta = session_meta[snum]
    state, phase = meta['state'], meta['phase']

    # ---- Load behavior ----
    print(f"\n  S{snum} ({state}/{phase}): loading behavior...", end='', flush=True)
    time_vals, vel, zones = load_behavior_xlsx(meta['behavior'])
    ent_times, ent_vals, vel_means = compute_entropy_causal(
        zones, time_vals, vel, ENTROPY_WINDOW_SEC, ENTROPY_STEP_SEC)

    if len(ent_vals) < 20:
        print(f" too few entropy points ({len(ent_vals)}), skipping")
        continue

    print(f" entropy={len(ent_vals)} pts, mean={np.mean(ent_vals):.2f} bits", end='')

    # ---- Load neural data ----
    p0_path = Path(meta['p0_sorted'])
    p1_path = Path(meta['p1_sorted'])

    aca_ids = get_good_units_p0(p0_path)
    lha_ids = get_good_units_p1_lha(p1_path)

    # Filter by available units in sorting
    try:
        sorting_p0 = se.read_kilosort(p0_path)
        avail_p0 = set(sorting_p0.get_unit_ids())
        aca_ids = np.array([u for u in aca_ids if u in avail_p0])
    except Exception as e:
        print(f" P0 load error: {e}")
        aca_ids = np.array([])

    try:
        sorting_p1 = se.read_kilosort(p1_path)
        avail_p1 = set(sorting_p1.get_unit_ids())
        lha_ids = np.array([u for u in lha_ids if u in avail_p1])
    except Exception as e:
        print(f" P1 load error: {e}")
        lha_ids = np.array([])

    print(f", ACA={len(aca_ids)}, LHA={len(lha_ids)}", end='')

    if len(aca_ids) < 2 and len(lha_ids) < 2:
        print(" — skipping (too few units)")
        continue

    # ---- Compute FR and PCA ----
    bin_edges = np.arange(0, time_vals[-1] + 2, 1.0)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    metrics_at_ent = {
        'Entropy': ent_vals,
        'Velocity': vel_means,
    }

    for region, unit_ids, sorting, prefix in [
        ('ACA', aca_ids, sorting_p0 if len(aca_ids) >= 2 else None, 'ACA'),
        ('LHA', lha_ids, sorting_p1 if len(lha_ids) >= 2 else None, 'LHA'),
    ]:
        if sorting is None or len(unit_ids) < 2:
            metrics_at_ent[f'{prefix} FR'] = np.full(len(ent_vals), np.nan)
            metrics_at_ent[f'{prefix} PC1'] = np.full(len(ent_vals), np.nan)
            continue

        fr = np.array([np.histogram(sorting.get_unit_spike_train(u) / FS,
                                     bins=bin_edges)[0] for u in unit_ids])
        z = np.array([(r - r.mean()) / max(r.std(), 1e-6) for r in fr])
        pop_fr = gaussian_filter1d(np.mean(z, axis=0), 10)
        pc1 = gaussian_filter1d(
            PCA(n_components=min(3, len(unit_ids))).fit_transform(z.T)[:, 0], 10)

        metrics_at_ent[f'{prefix} FR'] = np.interp(ent_times, bin_centers, pop_fr)
        metrics_at_ent[f'{prefix} PC1'] = np.interp(ent_times, bin_centers, pc1)

    # ---- Steady-state Spearman correlations ----
    print(f" [{timer.time()-t0:.1f}s]")
    print(f"    Steady-state Spearman:")
    print(f"    {'Metric':<12s}  {'rho(raw)':>10s}  {'p(raw)':>10s}  {'rho(part)':>10s}  {'p(part)':>10s}")

    for mname in neural_metrics:
        mdata = metrics_at_ent[mname]
        if np.all(np.isnan(mdata)):
            print(f"    {mname:<12s}  {'N/A':>10s}")
            continue

        rho, p = spearmanr(ent_vals, mdata)
        rho_part, p_part = partial_spearman(ent_vals, mdata, vel_means)
        sig = '*' if p < 0.05 else ''
        sig_p = '*' if p_part < 0.05 else ''

        print(f"    {mname:<12s}  {rho:+10.3f} {sig:>1s}  {p:10.4f}  "
              f"{rho_part:+10.3f} {sig_p:>1s}  {p_part:10.4f}")

        all_corr_results.append({
            'session': snum, 'state': state, 'phase': phase,
            'metric': mname, 'rho_raw': rho, 'p_raw': p,
            'rho_partial': rho_part, 'p_partial': p_part,
        })

    # ---- Peri-inflection analysis ----
    peaks, troughs, _ = find_inflections(ent_vals, SMOOTH_SIGMA, MIN_AMPLITUDE)
    peaks = [p for p in peaks if PERI_WINDOW <= p < len(ent_vals) - PERI_WINDOW]
    troughs = [t for t in troughs if PERI_WINDOW <= t < len(ent_vals) - PERI_WINDOW]

    print(f"    Inflections: {len(peaks)} peaks, {len(troughs)} troughs")

    for infl_type, infl_indices, label in [
        ('peak', peaks, 'Peak'), ('trough', troughs, 'Trough')
    ]:
        if len(infl_indices) < 2:
            continue

        print(f"    {label} (N={len(infl_indices)}):")
        print(f"    {'Metric':<12s}  {'Pre':>8s}  {'At':>8s}  {'Post':>8s}  "
              f"{'Delta':>8s}  {'MWU p':>10s}")

        for mname in metric_names:
            mdata = metrics_at_ent[mname]
            if np.all(np.isnan(mdata)):
                continue

            windows = np.array([mdata[idx - PERI_WINDOW: idx + PERI_WINDOW + 1]
                                for idx in infl_indices])
            pre_sl = slice(PERI_WINDOW - 6, PERI_WINDOW)
            post_sl = slice(PERI_WINDOW + 1, PERI_WINDOW + 7)

            pre_vals = np.nanmean(windows[:, pre_sl], axis=1)
            at_vals = windows[:, PERI_WINDOW]
            post_vals = np.nanmean(windows[:, post_sl], axis=1)
            delta = np.nanmean(post_vals) - np.nanmean(pre_vals)

            if len(pre_vals) >= 2 and len(post_vals) >= 2:
                _, p_mwu = mannwhitneyu(pre_vals, post_vals, alternative='two-sided')
            else:
                p_mwu = 1.0
            sig = '*' if p_mwu < 0.05 else ''

            print(f"    {mname:<12s}  {np.nanmean(pre_vals):+8.3f}  {np.nanmean(at_vals):+8.3f}  "
                  f"{np.nanmean(post_vals):+8.3f}  {delta:+8.3f}  {p_mwu:10.4f} {sig:>1s}")

            all_infl_results.append({
                'session': snum, 'state': state, 'phase': phase,
                'inflection': infl_type, 'n_events': len(infl_indices),
                'metric': mname, 'pre_mean': np.nanmean(pre_vals),
                'at_inflection': np.nanmean(at_vals),
                'post_mean': np.nanmean(post_vals), 'delta': delta,
                'p_mwu': p_mwu,
            })

        # Collect for pooled analysis (z-scored to baseline)
        baseline_sl = slice(0, 6)
        for idx in infl_indices:
            for mname in metric_names:
                mdata = metrics_at_ent[mname]
                if np.all(np.isnan(mdata)):
                    continue
                window = mdata[idx - PERI_WINDOW: idx + PERI_WINDOW + 1]
                bl = window[baseline_sl]
                bl_mean = np.nanmean(bl)
                bl_std = np.nanstd(bl)
                if bl_std > 1e-10:
                    window_z = (window - bl_mean) / bl_std
                else:
                    window_z = window - bl_mean
                if infl_type == 'peak':
                    pooled_peaks[mname].append(window_z)
                    peak_states_per_metric[mname].append(state)
                else:
                    pooled_troughs[mname].append(window_z)
                    trough_states_per_metric[mname].append(state)
            if infl_type == 'peak':
                peak_sessions.append(snum)
                peak_states.append(state)
            else:
                trough_sessions.append(snum)
                trough_states.append(state)

# Convert pooled to arrays
for mname in metric_names:
    pooled_peaks[mname] = np.array(pooled_peaks[mname]) if len(pooled_peaks[mname]) > 0 \
        else np.array([]).reshape(0, 2*PERI_WINDOW+1)
    pooled_troughs[mname] = np.array(pooled_troughs[mname]) if len(pooled_troughs[mname]) > 0 \
        else np.array([]).reshape(0, 2*PERI_WINDOW+1)
    peak_states_per_metric[mname] = np.array(peak_states_per_metric[mname])
    trough_states_per_metric[mname] = np.array(trough_states_per_metric[mname])

peak_states_arr = np.array(peak_states)
trough_states_arr = np.array(trough_states)

n_peaks = len(pooled_peaks['Entropy'])
n_troughs = len(pooled_troughs['Entropy'])

# ---- Save per-session results ----
df_corr = pd.DataFrame(all_corr_results)
df_corr.to_csv("data/dp_entropy_neural_corr_stats.csv", index=False)
print(f"\nSaved data/dp_entropy_neural_corr_stats.csv ({len(df_corr)} rows)")

df_infl = pd.DataFrame(all_infl_results)
df_infl.to_csv("data/dp_entropy_inflection_stats.csv", index=False)
print(f"Saved data/dp_entropy_inflection_stats.csv ({len(df_infl)} rows)")

# ========================================================================
# GRAND SUMMARY: Steady-state (overall + per-state)
# ========================================================================
print("\n" + "=" * 110)
print("STEADY-STATE SUMMARY: Neural vs Entropy (Spearman + partial|velocity)")
print("=" * 110)

def print_corr_summary(label, subset):
    print(f"\n  --- {label} ---")
    for mname in neural_metrics:
        mdf = subset[subset['metric'] == mname]
        if len(mdf) == 0:
            continue
        n_raw = (mdf['p_raw'] < 0.05).sum()
        n_part = (mdf['p_partial'] < 0.05).sum()
        n_tot = len(mdf)
        mean_rho = mdf['rho_raw'].mean()
        mean_rho_p = mdf['rho_partial'].mean()
        pos_raw = (mdf['rho_raw'] > 0).sum()
        neg_raw = (mdf['rho_raw'] < 0).sum()
        print(f"    {mname:<12s}: raw {n_raw}/{n_tot} sig, mean rho={mean_rho:+.3f} ({pos_raw}+/{neg_raw}-) "
              f"| partial {n_part}/{n_tot} sig, mean rho={mean_rho_p:+.3f}")

print_corr_summary("ALL SESSIONS", df_corr)
for st in STATE_ORDER:
    sub = df_corr[df_corr['state'] == st]
    if len(sub) > 0:
        print_corr_summary(STATE_LABELS[st].upper(), sub)

# ========================================================================
# GRAND SUMMARY: Per-session inflections (overall + per-state)
# ========================================================================
print("\n" + "=" * 110)
print("PER-SESSION INFLECTION SUMMARY")
print("=" * 110)

def print_infl_summary(label, subset):
    print(f"\n  --- {label} ---")
    for infl_type, il in [('peak', 'PEAKS'), ('trough', 'TROUGHS')]:
        idf = subset[subset['inflection'] == infl_type]
        if len(idf) == 0:
            continue
        print(f"  {il}:")
        for mname in metric_names:
            mdf = idf[idf['metric'] == mname]
            if len(mdf) == 0:
                continue
            n_sig = (mdf['p_mwu'] < 0.05).sum()
            n_tot = len(mdf)
            mean_d = mdf['delta'].mean()
            pos = (mdf['delta'] > 0).sum()
            neg = (mdf['delta'] < 0).sum()
            print(f"    {mname:<12s}: {n_sig}/{n_tot} sig, mean delta={mean_d:+.4f} ({pos}+/{neg}-)")

print_infl_summary("ALL SESSIONS", df_infl)
for st in STATE_ORDER:
    sub = df_infl[df_infl['state'] == st]
    if len(sub) > 0:
        print_infl_summary(STATE_LABELS[st].upper(), sub)

# ========================================================================
# POOLED PERI-INFLECTION — OVERALL + PER-STATE
# ========================================================================

time_axis = np.arange(-PERI_WINDOW, PERI_WINDOW + 1) * ENTROPY_STEP_SEC
pre_sl = slice(PERI_WINDOW - 6, PERI_WINDOW)
post_sl = slice(PERI_WINDOW + 1, PERI_WINDOW + 7)

def run_pooled_analysis(storage, group_label, state_filter=None):
    """Run pooled peri-inflection analysis, optionally filtered by state."""
    results = []
    pk_states_pm = peak_states_per_metric
    tr_states_pm = trough_states_per_metric
    for infl_type, infl_storage, infl_spm, label in [
        ('peak', storage[0], pk_states_pm, 'PEAKS'),
        ('trough', storage[1], tr_states_pm, 'TROUGHS'),
    ]:
        # Count using Entropy metric as reference for N
        ref_states = infl_spm.get('Entropy', np.array([]))
        if state_filter is not None:
            n_ev = int((ref_states == state_filter).sum())
        else:
            n_ev = len(ref_states)
        if n_ev < 4:
            continue
        print(f"\n  {label} (N={n_ev}):")
        print(f"    {'Metric':<12s}  {'Pre(z)':>8s}  {'At(z)':>8s}  {'Post(z)':>8s}  "
              f"{'Delta':>8s}  {'MWU p':>10s}  {'Wilcox p':>10s}")
        for mname in metric_names:
            windows = infl_storage[mname]
            m_states = infl_spm.get(mname, np.array([]))
            if len(windows) < 4:
                continue
            if state_filter is not None:
                infl_mask = m_states == state_filter
            else:
                infl_mask = np.ones(len(windows), dtype=bool)
            windows_f = windows[infl_mask]
            if len(windows_f) < 4:
                continue
            pre_vals = np.nanmean(windows_f[:, pre_sl], axis=1)
            at_vals = windows_f[:, PERI_WINDOW]
            post_vals = np.nanmean(windows_f[:, post_sl], axis=1)
            delta = np.nanmean(post_vals) - np.nanmean(pre_vals)
            _, p_mwu = mannwhitneyu(pre_vals, post_vals, alternative='two-sided')
            try:
                _, p_wil = wilcoxon(pre_vals, post_vals)
            except ValueError:
                p_wil = 1.0
            sig_m = '*' if p_mwu < 0.05 else ''
            sig_w = '*' if p_wil < 0.05 else ''
            print(f"    {mname:<12s}  {np.nanmean(pre_vals):+8.3f}  {np.nanmean(at_vals):+8.3f}  "
                  f"{np.nanmean(post_vals):+8.3f}  {delta:+8.3f}  {p_mwu:10.4f} {sig_m:>1s}  "
                  f"{p_wil:10.4f} {sig_w:>1s}")
            results.append({
                'group': group_label, 'inflection': infl_type, 'metric': mname,
                'n_events': n_ev,
                'pre_mean_z': np.nanmean(pre_vals), 'at_mean_z': np.nanmean(at_vals),
                'post_mean_z': np.nanmean(post_vals), 'delta_z': delta,
                'p_mwu': p_mwu, 'p_wilcoxon': p_wil,
            })
    return results

storage_pair = (pooled_peaks, pooled_troughs)
pooled_stats = []

# Overall
print("\n" + "=" * 110)
print(f"POOLED PERI-INFLECTION — ALL SESSIONS (N={n_peaks} peaks, {n_troughs} troughs)")
print("=" * 110)
pooled_stats.extend(run_pooled_analysis(storage_pair, 'all'))

# Per-state
for st in STATE_ORDER:
    n_pk = int((peak_states_arr == st).sum()) if len(peak_states_arr) > 0 else 0
    n_tr = int((trough_states_arr == st).sum()) if len(trough_states_arr) > 0 else 0
    if n_pk + n_tr < 4:
        continue
    print(f"\n{'=' * 110}")
    print(f"POOLED PERI-INFLECTION — {STATE_LABELS[st].upper()} (N={n_pk} peaks, {n_tr} troughs)")
    print("=" * 110)
    pooled_stats.extend(run_pooled_analysis(storage_pair, STATE_LABELS[st],
                                            state_filter=st))

df_pooled = pd.DataFrame(pooled_stats)
df_pooled.to_csv("data/dp_entropy_inflection_pooled_stats.csv", index=False)
print(f"\nSaved data/dp_entropy_inflection_pooled_stats.csv ({len(df_pooled)} rows)")

# ========================================================================
# FIGURES
# ========================================================================

# Figure 1: Per-session correlation bar chart (partial rho), color-coded by state
state_bar_colors = {'fed': 'tab:blue', 'fasted': 'tab:red',
                    'fed-HFD': 'tab:purple', 'fasted-HFD': 'tab:orange'}
fig, axes = plt.subplots(1, len(neural_metrics), figsize=(4 * len(neural_metrics), 6))
if len(neural_metrics) == 1:
    axes = [axes]

all_sessions = sorted(session_meta.keys())
for mi, mname in enumerate(neural_metrics):
    ax = axes[mi]
    mat = np.full(len(all_sessions), np.nan)
    pmat = np.full(len(all_sessions), np.nan)
    labels = []
    bar_colors = []
    for si, snum in enumerate(all_sessions):
        row = df_corr[(df_corr['metric'] == mname) & (df_corr['session'] == snum)]
        meta = session_meta[snum]
        labels.append(f"S{snum}\n{meta['state'][:3]}/{meta['phase'][:3]}")
        bar_colors.append(state_bar_colors.get(meta['state'], 'gray'))
        if len(row) > 0:
            mat[si] = row['rho_partial'].values[0]
            pmat[si] = row['p_partial'].values[0]

    bars = ax.bar(range(len(all_sessions)), mat, color=bar_colors, alpha=0.7,
                  edgecolor='black', linewidth=0.5)
    for si in range(len(all_sessions)):
        if not np.isnan(pmat[si]) and pmat[si] < 0.05:
            ax.text(si, mat[si], '*', ha='center', va='bottom' if mat[si] > 0 else 'top',
                    fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(all_sessions)))
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_title(mname, fontsize=10, fontweight='bold')
    ax.set_ylabel('Partial rho (entropy|velocity)' if mi == 0 else '')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_ylim(-0.6, 0.6)

# Legend for states
from matplotlib.patches import Patch
legend_handles = [Patch(facecolor=state_bar_colors[st], label=STATE_LABELS[st])
                  for st in STATE_ORDER if any(session_meta[s]['state'] == st for s in all_sessions)]
axes[-1].legend(handles=legend_handles, fontsize=7, loc='lower right')

plt.suptitle('Dual-Probe: Neural-Entropy Partial Correlations', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("figures/dp_entropy_neural_correlations.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/dp_entropy_neural_correlations.png")

# Figure 2: Pooled peri-inflection traces — ALL sessions
fig, axes = plt.subplots(len(metric_names), 2, figsize=(12, len(metric_names) * 2), sharex=True)
colors_metric = {
    'Entropy': 'black', 'Velocity': 'gray',
    'ACA FR': 'tab:green', 'LHA FR': 'tab:red',
    'ACA PC1': 'darkgreen', 'LHA PC1': 'tab:orange',
}

all_stats = [s for s in pooled_stats if s['group'] == 'all']
for col, (infl_type, storage, label) in enumerate([
    ('peak', pooled_peaks, f'PEAKS (N={n_peaks})'),
    ('trough', pooled_troughs, f'TROUGHS (N={n_troughs})'),
]):
    for row, mname in enumerate(metric_names):
        ax = axes[row, col]
        windows = storage[mname]
        color = colors_metric.get(mname, 'gray')

        if len(windows) < 4:
            ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes, ha='center')
            continue

        mean_trace = np.nanmean(windows, axis=0)
        sem_trace = np.nanstd(windows, axis=0) / np.sqrt(len(windows))

        ax.fill_between(time_axis, mean_trace - sem_trace, mean_trace + sem_trace,
                        alpha=0.3, color=color)
        ax.plot(time_axis, mean_trace, color=color, linewidth=2)
        ax.axvline(0, color='black', linestyle='--', alpha=0.5, linewidth=0.8)
        ax.axhline(0, color='gray', linestyle=':', alpha=0.3)
        ax.axvspan(-60, -10, alpha=0.05, color='green')
        ax.axvspan(10, 60, alpha=0.05, color='red')
        ax.set_ylabel(mname, fontsize=8)

        stat_row = [s for s in all_stats
                    if s['metric'] == mname and s['inflection'] == infl_type]
        if stat_row:
            sr = stat_row[0]
            sig = '***' if sr['p_wilcoxon'] < 0.001 else ('**' if sr['p_wilcoxon'] < 0.01 else
                  ('*' if sr['p_wilcoxon'] < 0.05 else 'ns'))
            ax.text(0.98, 0.95, f"d={sr['delta_z']:+.2f} {sig}",
                    transform=ax.transAxes, fontsize=7, ha='right', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        if row == 0:
            ax.set_title(label, fontsize=11, fontweight='bold')

    axes[-1, col].set_xlabel('Time from inflection (s)', fontsize=10)

plt.suptitle('Dual-Probe: Pooled Peri-Inflection (z-scored, ACA + LHA)',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig("figures/dp_entropy_inflection_pooled.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/dp_entropy_inflection_pooled.png")

# Figure 3: Per-state pooled peri-inflection comparison
state_colors = {'fed': 'tab:blue', 'fasted': 'tab:red', 'fed-HFD': 'tab:purple', 'fasted-HFD': 'tab:orange'}
# Only plot states with enough events
ref_pk_states = peak_states_per_metric.get('Entropy', np.array([]))
ref_tr_states = trough_states_per_metric.get('Entropy', np.array([]))
active_states = [st for st in STATE_ORDER
                 if int((ref_pk_states == st).sum()) >= 4 or int((ref_tr_states == st).sum()) >= 4]
if len(active_states) >= 2:
    # Plot key neural metrics per state
    key_metrics = ['ACA FR', 'LHA FR', 'ACA PC1', 'LHA PC1']
    fig, axes = plt.subplots(len(key_metrics), 2, figsize=(14, len(key_metrics) * 2.5), sharex=True)

    for col, (infl_type, storage, spm, label_base) in enumerate([
        ('peak', pooled_peaks, peak_states_per_metric, 'PEAKS'),
        ('trough', pooled_troughs, trough_states_per_metric, 'TROUGHS'),
    ]):
        for row, mname in enumerate(key_metrics):
            ax = axes[row, col]
            windows_all = storage[mname]
            m_states = spm.get(mname, np.array([]))
            if len(windows_all) < 4:
                ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes, ha='center')
                continue

            for st in active_states:
                mask = m_states == st
                if mask.sum() < 4:
                    continue
                windows_st = windows_all[mask]
                mean_trace = np.nanmean(windows_st, axis=0)
                sem_trace = np.nanstd(windows_st, axis=0) / np.sqrt(len(windows_st))
                sc = state_colors[st]
                ax.fill_between(time_axis, mean_trace - sem_trace, mean_trace + sem_trace,
                                alpha=0.15, color=sc)
                ax.plot(time_axis, mean_trace, color=sc, linewidth=2,
                        label=f"{STATE_LABELS[st]} (N={mask.sum()})")

            ax.axvline(0, color='black', linestyle='--', alpha=0.5, linewidth=0.8)
            ax.axhline(0, color='gray', linestyle=':', alpha=0.3)
            ax.set_ylabel(mname, fontsize=9)

            # Overlay entropy trace on secondary y-axis
            ent_windows = storage['Entropy']
            if len(ent_windows) >= 4:
                ax2 = ax.twinx()
                ent_mean = np.nanmean(ent_windows, axis=0)
                ax2.plot(time_axis, ent_mean, color='black', linewidth=1.2,
                         linestyle='--', alpha=0.4, label='Entropy')
                ax2.set_ylabel('Entropy (z)', fontsize=7, color='gray', alpha=0.5)
                ax2.tick_params(axis='y', labelsize=6, colors='gray')

            if row == 0:
                ref_st = spm.get('Entropy', np.array([]))
                n_counts = ', '.join([f"{STATE_LABELS[st]}={int((ref_st == st).sum())}"
                                      for st in active_states if (ref_st == st).sum() >= 4])
                ax.set_title(f'{label_base} ({n_counts})', fontsize=10, fontweight='bold')
                ax.legend(fontsize=7, loc='upper left')

        axes[-1, col].set_xlabel('Time from inflection (s)', fontsize=10)

    plt.suptitle('Dual-Probe: Per-State Pooled Peri-Inflection (z-scored)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig("figures/dp_entropy_inflection_by_state.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved figures/dp_entropy_inflection_by_state.png")
