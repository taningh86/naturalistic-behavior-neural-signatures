"""
Dual-Probe: Scale-Free Analysis — Part A (Subsampled Avalanches) + Part B (DFA & 1/f)
=======================================================================================
Pilot: S3 ACA (225 units, 30 min, fed/exploration).

Part B runs first (fast): DFA Hurst exponent + 1/f spectral slope on full population.
Part A then runs subsampled avalanche analysis (N=30,60,100,150 x 20 draws).

Reuses avalanche pipeline from dp_avalanche_criticality.py as callable functions.
"""

import yaml
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import powerlaw
import spikeinterface.extractors as se
from scipy.signal import welch
from scipy.stats import linregress
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
import time as timer

warnings.filterwarnings('ignore')

# Import avalanche functions from the pilot script
from dp_avalanche_criticality import (
    get_good_units_p0, get_good_units_p1_lha,
    load_spike_times_for_region, build_population_rate,
    compute_mean_iei, extract_avalanches, fit_power_law,
    compute_mean_size_vs_duration, fit_scaling_exponent,
    compute_shape_collapse, optimize_collapse_exponent,
    shuffle_spike_times,
    FS, P0_MIN_FR, PRIMARY_THRESHOLD_SD, MIN_AVALANCHES_HARD,
    MIN_AVALANCHES_WARN, COLLAPSE_T_MIN, COLLAPSE_T_MAX,
)

with open("paths.yaml") as f:
    cfg = yaml.safe_load(f)

sessions_cfg = cfg["double_probe"]["coordinates_1"]["mouse01"]["sessions"]

# Default pilot session (overridable via CLI arg)
PILOT_SESSION = 3

outdir = Path("data/avalanche")
outdir.mkdir(parents=True, exist_ok=True)
figdir = Path("figures/avalanche")
figdir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# PART B: DFA + 1/f SPECTRAL SLOPE (runs first — fast)
# ============================================================================

def dfa(signal, window_sizes=None):
    """Detrended Fluctuation Analysis.
    Manual implementation — straightforward algorithm:
      1. Integrate signal (cumulative sum of deviations from mean)
      2. For each window size n, divide integrated signal into windows
      3. In each window, fit linear trend, compute RMS of residuals
      4. F(n) = RMS fluctuation averaged across windows
      5. H = slope of log(F(n)) vs log(n)

    Returns window_sizes, fluctuations arrays."""
    N = len(signal)
    # Integrate: cumulative sum of (x - mean)
    y = np.cumsum(signal - np.mean(signal))

    if window_sizes is None:
        # Log-spaced from 4 to N/10
        min_win = 4
        max_win = N // 10
        if max_win <= min_win:
            max_win = N // 4
        n_sizes = 30  # number of window sizes to try
        window_sizes = np.unique(np.logspace(
            np.log10(min_win), np.log10(max_win), n_sizes).astype(int))
        window_sizes = window_sizes[window_sizes >= 4]

    fluctuations = []
    valid_sizes = []

    for n in window_sizes:
        # Number of non-overlapping windows
        n_windows = N // n
        if n_windows < 2:
            continue

        # Compute RMS of detrended residuals in each window
        rms_list = []
        for i in range(n_windows):
            segment = y[i * n:(i + 1) * n]
            # Linear detrending
            x_axis = np.arange(n)
            coeffs = np.polyfit(x_axis, segment, 1)
            trend = np.polyval(coeffs, x_axis)
            residual = segment - trend
            rms = np.sqrt(np.mean(residual ** 2))
            rms_list.append(rms)

        fluctuations.append(np.mean(rms_list))
        valid_sizes.append(n)

    return np.array(valid_sizes), np.array(fluctuations)


def compute_hurst_exponent(sizes, fluctuations, fit_range=None):
    """Fit H from log-log slope of DFA fluctuations.
    Returns H, (ci_lo, ci_hi), r_squared."""
    log_n = np.log10(sizes)
    log_f = np.log10(fluctuations)

    if fit_range is not None:
        mask = (sizes >= fit_range[0]) & (sizes <= fit_range[1])
        log_n = log_n[mask]
        log_f = log_f[mask]

    if len(log_n) < 5:
        return np.nan, (np.nan, np.nan), np.nan

    slope, intercept, r, p, se = linregress(log_n, log_f)
    # Bootstrap CI
    n = len(log_n)
    slopes = []
    for _ in range(500):
        idx = np.random.randint(0, n, n)
        s, _, _, _, _ = linregress(log_n[idx], log_f[idx])
        slopes.append(s)
    ci = (float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5)))

    return float(slope), ci, float(r ** 2)


def compute_psd_slope(signal, dt, f_range=(0.1, 50.0)):
    """Compute 1/f^beta spectral slope via Welch's method.
    dt in seconds.
    Returns beta, (ci_lo, ci_hi), freqs, psd arrays."""
    fs = 1.0 / dt
    # Segment length: ~10 seconds for good low-freq resolution
    nperseg = min(int(10.0 / dt), len(signal) // 4)
    nperseg = max(nperseg, 256)  # at least 256 samples

    freqs, psd = welch(signal, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)

    # Fit range: clip to valid frequencies
    f_lo = max(f_range[0], freqs[1])  # exclude DC
    f_hi = min(f_range[1], fs / 2)
    mask = (freqs >= f_lo) & (freqs <= f_hi) & (psd > 0)

    if mask.sum() < 5:
        return np.nan, (np.nan, np.nan), freqs, psd

    log_f = np.log10(freqs[mask])
    log_p = np.log10(psd[mask])

    slope, intercept, r, p, se = linregress(log_f, log_p)
    beta = -slope  # P(f) ~ 1/f^beta, so log(P) = -beta*log(f) + c

    # Bootstrap CI
    n = len(log_f)
    betas = []
    for _ in range(500):
        idx = np.random.randint(0, n, n)
        s, _, _, _, _ = linregress(log_f[idx], log_p[idx])
        betas.append(-s)
    ci = (float(np.percentile(betas, 2.5)), float(np.percentile(betas, 97.5)))

    return float(beta), ci, freqs, psd


# ============================================================================
# PART A: Subsampled avalanche pipeline (one draw)
# ============================================================================

def run_subsample_avalanche(spike_dict_full, session_duration, unit_ids_subset):
    """Run avalanche pipeline on a subset of units.
    Uses mean-IEI dt for the subsample and PRIMARY_THRESHOLD_SD.
    Returns results dict with exponents, criteria, etc.
    Lightweight: no bootstrap, no full sweep, no figures."""

    # Subset the spike dict
    spike_dict = {uid: spike_dict_full[uid] for uid in unit_ids_subset}
    n_units = len(unit_ids_subset)

    # Compute IEI for this subsample
    mean_iei = compute_mean_iei(spike_dict)
    dt = np.clip(mean_iei, 0.0001, 0.1)  # clamp to 0.1-100ms

    # Build population rate and threshold
    bin_edges, pop_counts = build_population_rate(spike_dict, dt, session_duration)
    threshold = pop_counts.mean() + PRIMARY_THRESHOLD_SD * pop_counts.std()

    # Extract avalanches
    avalanches = extract_avalanches(pop_counts, threshold)
    n_av = len(avalanches)

    result = {
        'n_units': n_units,
        'mean_iei_ms': round(mean_iei * 1000, 3),
        'dt_ms': round(dt * 1000, 3),
        'threshold': round(float(threshold), 2),
        'n_avalanches': n_av,
    }

    if n_av < MIN_AVALANCHES_HARD:
        result['tau'] = np.nan
        result['alpha'] = np.nan
        result['gamma'] = np.nan
        result['scaling_ratio'] = np.nan
        result['scaling_residual'] = np.nan
        result['tau_decades'] = 0
        result['alpha_decades'] = 0
        result['max_duration'] = 0
        result['collapse_error'] = np.nan
        return result

    sizes = np.array([a['size'] for a in avalanches])
    durations = np.array([a['duration'] for a in avalanches])
    result['max_duration'] = int(durations.max())

    # Power law fits (no bootstrap for speed)
    s_fit = fit_power_law(sizes, "S", bootstrap=False)
    t_fit = fit_power_law(durations, "T", bootstrap=False)

    result['tau'] = s_fit.get('exponent', np.nan)
    result['alpha'] = t_fit.get('exponent', np.nan)
    result['tau_decades'] = s_fit.get('decades', 0)
    result['alpha_decades'] = t_fit.get('decades', 0)

    # Scaling relation
    unique_T, mean_S = compute_mean_size_vs_duration(avalanches)
    gamma, _ = fit_scaling_exponent(unique_T, mean_S)
    result['gamma'] = gamma if isinstance(gamma, float) else np.nan

    tau_val = result['tau']
    alpha_val = result['alpha']
    if (not np.isnan(tau_val) and not np.isnan(alpha_val)
            and (tau_val - 1) > 0.01):
        sr = (alpha_val - 1) / (tau_val - 1)
        result['scaling_ratio'] = sr
        if not np.isnan(result['gamma']):
            result['scaling_residual'] = abs(sr - result['gamma'])
        else:
            result['scaling_residual'] = np.nan
    else:
        result['scaling_ratio'] = np.nan
        result['scaling_residual'] = np.nan

    # Shape collapse (quick)
    _, ce, _ = compute_shape_collapse(avalanches,
                                       result['gamma'] if not np.isnan(result['gamma']) else 2.0)
    result['collapse_error'] = ce if isinstance(ce, float) else np.nan

    # Shuffle control (quick — 1 shuffle, no bootstrap)
    shuf_dict = shuffle_spike_times(spike_dict, session_duration)
    shuf_edges, shuf_pop = build_population_rate(shuf_dict, dt, session_duration)
    shuf_thresh = shuf_pop.mean() + PRIMARY_THRESHOLD_SD * shuf_pop.std()
    shuf_avs = extract_avalanches(shuf_pop, shuf_thresh)
    result['shuffle_n_avalanches'] = len(shuf_avs)
    if len(shuf_avs) >= MIN_AVALANCHES_HARD:
        shuf_sizes = np.array([a['size'] for a in shuf_avs])
        shuf_s = fit_power_law(shuf_sizes, "S", bootstrap=False)
        result['shuffle_tau'] = shuf_s.get('exponent', np.nan)
    else:
        result['shuffle_tau'] = np.nan

    return result


# ============================================================================
# MAIN
# ============================================================================

def main(region='ACA', session_num=None):
    region = region.upper()
    sess_num = session_num if session_num is not None else PILOT_SESSION
    sess_val = sessions_cfg[f"session_{sess_num}"]
    sess_state = sess_val['state']
    sess_phase = sess_val['phase']

    print("=" * 100)
    print("SCALE-FREE ANALYSIS -- Part A (Subsampled Avalanches) + Part B (DFA & 1/f)")
    print(f"S{sess_num} {region} ({sess_state}/{sess_phase})")
    print("=" * 100)

    # ---- Load data ----
    if region == 'ACA':
        sorted_path = Path(sess_val['probe_0_aca']['sorted'])
        print(f"\nLoading ACA units from {sorted_path}...")
        unit_ids = get_good_units_p0(sorted_path)
        sorting = se.read_kilosort(sorted_path)
    elif region == 'LHA':
        sorted_path = Path(sess_val['probe_1_lha_rsp']['sorted'])
        print(f"\nLoading LHA units from {sorted_path}...")
        unit_ids = get_good_units_p1_lha(sorted_path)
        sorting = se.read_kilosort(sorted_path)
    else:
        raise ValueError(f"Unknown region: {region}")

    avail = set(sorting.get_unit_ids())
    unit_ids = np.array([u for u in unit_ids if u in avail])
    n_units = len(unit_ids)
    print(f"  {n_units} good {region} units")

    print("Loading spike times...")
    spike_dict = load_spike_times_for_region(sorting, unit_ids)
    all_spikes = np.concatenate(list(spike_dict.values()))
    session_duration = float(all_spikes.max()) + 1.0
    total_spikes = len(all_spikes)
    print(f"  Session: {session_duration:.1f}s, {total_spikes:,} spikes")

    # ==================================================================
    # PART B: DFA + 1/f (fast — run first)
    # ==================================================================
    print("\n" + "=" * 100)
    print("PART B: DFA & 1/f SPECTRAL SLOPE")
    print("=" * 100)

    partB = {}
    BIN_SIZES_MS = [1.0, 10.0, 100.0]  # 1ms, 10ms (primary), 100ms
    PRIMARY_BIN_MS = 10.0

    for bin_ms in BIN_SIZES_MS:
        dt = bin_ms / 1000.0
        label = f"{bin_ms}ms"
        print(f"\n  Bin size: {label}")

        # Build population rate (all 225 units)
        bin_edges, pop_rate = build_population_rate(spike_dict, dt, session_duration)
        n_bins = len(pop_rate)
        print(f"    {n_bins} bins, mean={pop_rate.mean():.2f}, std={pop_rate.std():.2f}")

        # ---- DFA ----
        t0 = timer.time()
        sizes, fluct = dfa(pop_rate)
        H, H_ci, H_r2 = compute_hurst_exponent(sizes, fluct)
        dfa_time = timer.time() - t0
        print(f"    DFA: H={H:.3f} [{H_ci[0]:.3f}, {H_ci[1]:.3f}], R2={H_r2:.4f} ({dfa_time:.1f}s)")

        # ---- 1/f slope ----
        # Fit range: 0.1-50 Hz for 1ms/10ms bins; 0.1-5 Hz for 100ms bins
        f_hi = min(50.0, 1.0 / (2 * dt))  # Nyquist / 2
        f_range = (0.1, f_hi)

        t0 = timer.time()
        beta, beta_ci, freqs, psd = compute_psd_slope(pop_rate, dt, f_range=f_range)
        psd_time = timer.time() - t0
        print(f"    PSD: beta={beta:.3f} [{beta_ci[0]:.3f}, {beta_ci[1]:.3f}] "
              f"(fit {f_range[0]}-{f_range[1]:.0f} Hz, {psd_time:.1f}s)")

        # ---- Shuffle control ----
        print(f"    Shuffle control...", end='', flush=True)
        shuf_dict = shuffle_spike_times(spike_dict, session_duration)
        _, shuf_pop = build_population_rate(shuf_dict, dt, session_duration)

        shuf_sizes, shuf_fluct = dfa(shuf_pop)
        shuf_H, shuf_H_ci, shuf_H_r2 = compute_hurst_exponent(shuf_sizes, shuf_fluct)

        shuf_beta, shuf_beta_ci, shuf_freqs, shuf_psd = compute_psd_slope(
            shuf_pop, dt, f_range=f_range)
        print(f" H_shuf={shuf_H:.3f}, beta_shuf={shuf_beta:.3f}")

        # Flag if shuffle produces high values
        if shuf_H > 0.7:
            print(f"    *** WARNING: Shuffled H={shuf_H:.3f} is elevated -- check pipeline ***")
        if shuf_beta > 0.5:
            print(f"    *** WARNING: Shuffled beta={shuf_beta:.3f} is elevated -- check pipeline ***")

        partB[label] = {
            'bin_ms': bin_ms, 'n_bins': n_bins,
            'H': H, 'H_ci': H_ci, 'H_r2': H_r2,
            'beta': beta, 'beta_ci': beta_ci, 'f_range': f_range,
            'shuffle_H': shuf_H, 'shuffle_H_ci': shuf_H_ci,
            'shuffle_beta': shuf_beta, 'shuffle_beta_ci': shuf_beta_ci,
            # Store arrays for plotting (primary bin only)
            'dfa_sizes': sizes.tolist() if bin_ms == PRIMARY_BIN_MS else None,
            'dfa_fluct': fluct.tolist() if bin_ms == PRIMARY_BIN_MS else None,
            'shuf_dfa_sizes': shuf_sizes.tolist() if bin_ms == PRIMARY_BIN_MS else None,
            'shuf_dfa_fluct': shuf_fluct.tolist() if bin_ms == PRIMARY_BIN_MS else None,
            'psd_freqs': freqs.tolist() if bin_ms == PRIMARY_BIN_MS else None,
            'psd_power': psd.tolist() if bin_ms == PRIMARY_BIN_MS else None,
            'shuf_psd_freqs': shuf_freqs.tolist() if bin_ms == PRIMARY_BIN_MS else None,
            'shuf_psd_power': shuf_psd.tolist() if bin_ms == PRIMARY_BIN_MS else None,
        }

    # ---- Part B Figure ----
    primary = partB[f"{PRIMARY_BIN_MS}ms"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: DFA
    ax = axes[0]
    dfa_s = np.array(primary['dfa_sizes'])
    dfa_f = np.array(primary['dfa_fluct'])
    ax.loglog(dfa_s, dfa_f, 'ko-', markersize=4, linewidth=1, label='Data')
    # Fit line
    log_s = np.log10(dfa_s)
    log_f = np.log10(dfa_f)
    slope, intercept, _, _, _ = linregress(log_s, log_f)
    fit_line = 10 ** (slope * log_s + intercept)
    ax.loglog(dfa_s, fit_line, 'r--', linewidth=2,
              label=f'H={primary["H"]:.3f} [{primary["H_ci"][0]:.3f},{primary["H_ci"][1]:.3f}]')
    # Shuffle
    shuf_s = np.array(primary['shuf_dfa_sizes'])
    shuf_f = np.array(primary['shuf_dfa_fluct'])
    ax.loglog(shuf_s, shuf_f, 'gx-', markersize=3, linewidth=0.7, alpha=0.5,
              label=f'Shuffle H={primary["shuffle_H"]:.3f}')
    ax.set_xlabel('Window size n (bins)')
    ax.set_ylabel('F(n)')
    ax.set_title(f'DFA (bin={PRIMARY_BIN_MS}ms)', fontweight='bold')
    ax.legend(fontsize=9)

    # Panel 2: PSD
    ax = axes[1]
    psd_f = np.array(primary['psd_freqs'])
    psd_p = np.array(primary['psd_power'])
    mask = psd_f > 0
    ax.loglog(psd_f[mask], psd_p[mask], 'k-', linewidth=0.5, alpha=0.7, label='Data')
    # Fit range
    f_lo, f_hi = primary['f_range']
    fit_mask = (psd_f >= f_lo) & (psd_f <= f_hi) & (psd_p > 0)
    if fit_mask.sum() > 0:
        ax.loglog(psd_f[fit_mask], psd_p[fit_mask], 'b-', linewidth=1.5)
        # Fit line
        log_ff = np.log10(psd_f[fit_mask])
        log_pp = np.log10(psd_p[fit_mask])
        s, i, _, _, _ = linregress(log_ff, log_pp)
        fl = 10 ** (s * log_ff + i)
        ax.loglog(psd_f[fit_mask], fl, 'r--', linewidth=2,
                  label=f'1/f^{primary["beta"]:.2f} [{primary["beta_ci"][0]:.2f},{primary["beta_ci"][1]:.2f}]')
    # Shuffle
    shuf_ff = np.array(primary['shuf_psd_freqs'])
    shuf_pp = np.array(primary['shuf_psd_power'])
    mask_s = shuf_ff > 0
    ax.loglog(shuf_ff[mask_s], shuf_pp[mask_s], 'g-', linewidth=0.5, alpha=0.4,
              label=f'Shuffle 1/f^{primary["shuffle_beta"]:.2f}')
    ax.axvline(f_lo, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(f_hi, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power')
    ax.set_title(f'Power Spectrum (bin={PRIMARY_BIN_MS}ms)', fontweight='bold')
    ax.legend(fontsize=9)

    # Panel 3: Summary across bin sizes
    ax = axes[2]
    bins = sorted(partB.keys(), key=lambda x: float(x.replace('ms', '')))
    x = range(len(bins))
    H_vals = [partB[b]['H'] for b in bins]
    H_shuf = [partB[b]['shuffle_H'] for b in bins]
    beta_vals = [partB[b]['beta'] for b in bins]
    beta_shuf = [partB[b]['shuffle_beta'] for b in bins]

    ax.bar([i - 0.2 for i in x], H_vals, 0.18, label='H (data)', color='steelblue')
    ax.bar([i + 0.0 for i in x], H_shuf, 0.18, label='H (shuffle)', color='steelblue', alpha=0.3)
    ax.bar([i + 0.2 for i in x], beta_vals, 0.18, label='beta (data)', color='darkorange')
    ax.bar([i + 0.4 for i in x], beta_shuf, 0.18, label='beta (shuffle)', color='darkorange', alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(bins)
    ax.set_xlabel('Bin size')
    ax.set_ylabel('Exponent')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='H=0.5 (uncorrelated)')
    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.5, label='beta=1 (pink noise)')
    ax.set_title('Scale-Free Metrics Across Bin Sizes', fontweight='bold')
    ax.legend(fontsize=7, ncol=2)

    fig.suptitle(f'Part B: DFA & 1/f -- S{sess_num} {region} ({sess_state}/{sess_phase}, N={n_units})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    partB_fig = figdir / f"scale_free_partB_S{sess_num}_{region}.png"
    plt.savefig(partB_fig, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved {partB_fig}")

    # Save Part B JSON
    partB_json = outdir / f"scale_free_partB_S{sess_num}_{region}.json"
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, tuple):
            return list(obj)
        return obj
    with open(partB_json, 'w') as f:
        json.dump(partB, f, indent=2, default=convert)
    print(f"  Saved {partB_json}")

    # ==================================================================
    # PART A: Subsampled Avalanche Analysis
    # ==================================================================
    print("\n" + "=" * 100)
    print("PART A: SUBSAMPLED AVALANCHE ANALYSIS")
    print("=" * 100)

    SUBSAMPLE_SIZES = [30, 60, 100, 150]
    N_DRAWS = 20
    all_unit_ids = list(spike_dict.keys())

    partA = {}

    for N in SUBSAMPLE_SIZES:
        if N > len(all_unit_ids):
            print(f"\n  N={N}: SKIP (only {len(all_unit_ids)} units)")
            continue

        print(f"\n  N={N}: {N_DRAWS} draws...", flush=True)
        t0 = timer.time()
        draw_results = []

        for draw_i in range(N_DRAWS):
            # Random subsample without replacement
            subset = np.random.choice(all_unit_ids, N, replace=False)
            res = run_subsample_avalanche(spike_dict, session_duration, subset)
            draw_results.append(res)

            # Progress
            if (draw_i + 1) % 5 == 0:
                elapsed = timer.time() - t0
                eta = elapsed / (draw_i + 1) * (N_DRAWS - draw_i - 1)
                print(f"    draw {draw_i+1}/{N_DRAWS} "
                      f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)", flush=True)

        total_time = timer.time() - t0
        print(f"    Done: {total_time:.1f}s")

        # Aggregate statistics across draws
        df_draws = pd.DataFrame(draw_results)

        summary = {
            'N': N,
            'n_draws': N_DRAWS,
            'total_time_s': round(total_time, 1),
        }

        for metric in ['n_avalanches', 'max_duration', 'mean_iei_ms', 'dt_ms',
                        'tau', 'alpha', 'gamma', 'scaling_ratio',
                        'scaling_residual', 'tau_decades', 'alpha_decades',
                        'collapse_error', 'shuffle_tau']:
            vals = df_draws[metric].dropna()
            if len(vals) > 0:
                summary[f'{metric}_median'] = float(vals.median())
                summary[f'{metric}_mean'] = float(vals.mean())
                summary[f'{metric}_lo'] = float(vals.quantile(0.025))
                summary[f'{metric}_hi'] = float(vals.quantile(0.975))
                summary[f'{metric}_n_valid'] = int(len(vals))
            else:
                summary[f'{metric}_median'] = None
                summary[f'{metric}_n_valid'] = 0

        # Criticality criteria: fraction of draws meeting each
        n_pl_s = sum(1 for r in draw_results
                     if not np.isnan(r.get('tau_decades', 0))
                     and r.get('tau_decades', 0) >= 1.5)
        n_pl_t = sum(1 for r in draw_results
                     if not np.isnan(r.get('alpha_decades', 0))
                     and r.get('alpha_decades', 0) >= 1.0)
        n_scaling = sum(1 for r in draw_results
                        if isinstance(r.get('scaling_residual'), float)
                        and not np.isnan(r['scaling_residual'])
                        and r['scaling_residual'] < 0.3)
        n_collapse = sum(1 for r in draw_results
                         if isinstance(r.get('collapse_error'), float)
                         and not np.isnan(r['collapse_error'])
                         and r['collapse_error'] < 0.1)
        n_shuffle = sum(1 for r in draw_results
                        if isinstance(r.get('shuffle_tau'), float)
                        and not np.isnan(r['shuffle_tau'])
                        and isinstance(r.get('tau'), float)
                        and not np.isnan(r['tau'])
                        and abs(r['shuffle_tau'] - r['tau']) > 0.2)
        summary['frac_pl_s'] = n_pl_s / N_DRAWS
        summary['frac_pl_t'] = n_pl_t / N_DRAWS
        summary['frac_scaling'] = n_scaling / N_DRAWS
        summary['frac_collapse'] = n_collapse / N_DRAWS
        summary['frac_shuffle_pass'] = n_shuffle / N_DRAWS

        partA[N] = summary

        print(f"    tau: {summary.get('tau_median', 'N/A')}"
              f" (decades: {summary.get('tau_decades_median', 'N/A')})")
        print(f"    alpha: {summary.get('alpha_median', 'N/A')}"
              f" (decades: {summary.get('alpha_decades_median', 'N/A')})")
        print(f"    n_avalanches: {summary.get('n_avalanches_median', 'N/A')}"
              f", max_dur: {summary.get('max_duration_median', 'N/A')}")
        print(f"    Criteria pass rates: P(S)={summary['frac_pl_s']:.0%}, "
              f"P(T)={summary['frac_pl_t']:.0%}, "
              f"scaling={summary['frac_scaling']:.0%}, "
              f"collapse={summary['frac_collapse']:.0%}, "
              f"shuffle={summary['frac_shuffle_pass']:.0%}")

    # ---- Part A Figure ----
    Ns = sorted(partA.keys())
    if len(Ns) > 0:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # Panel 1: tau vs N
        ax = axes[0, 0]
        tau_meds = [partA[n].get('tau_median') for n in Ns]
        tau_los = [partA[n].get('tau_lo') for n in Ns]
        tau_his = [partA[n].get('tau_hi') for n in Ns]
        ax.errorbar(Ns, tau_meds,
                     yerr=[[m - lo if m and lo else 0 for m, lo in zip(tau_meds, tau_los)],
                           [hi - m if m and hi else 0 for m, hi in zip(tau_meds, tau_his)]],
                     fmt='ko-', capsize=5, linewidth=2, markersize=8)
        ax.axhline(1.5, color='red', linestyle='--', alpha=0.5, label='Theory (1.5)')
        ax.set_xlabel('Subsample size N')
        ax.set_ylabel('tau (P(S) exponent)')
        ax.set_title('tau vs subsample size', fontweight='bold')
        ax.legend()

        # Panel 2: alpha vs N
        ax = axes[0, 1]
        alpha_meds = [partA[n].get('alpha_median') for n in Ns]
        alpha_los = [partA[n].get('alpha_lo') for n in Ns]
        alpha_his = [partA[n].get('alpha_hi') for n in Ns]
        valid_alpha = [m is not None and not np.isnan(m) for m in alpha_meds]
        if any(valid_alpha):
            ax.errorbar(Ns, [m if m and not np.isnan(m) else 0 for m in alpha_meds],
                         yerr=[[m - lo if m and lo and not np.isnan(m) else 0
                                for m, lo in zip(alpha_meds, alpha_los)],
                               [hi - m if m and hi and not np.isnan(m) else 0
                                for m, hi in zip(alpha_meds, alpha_his)]],
                         fmt='ko-', capsize=5, linewidth=2, markersize=8)
            ax.axhline(2.0, color='red', linestyle='--', alpha=0.5, label='Theory (2.0)')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No valid alpha fits', transform=ax.transAxes, ha='center')
        ax.set_xlabel('Subsample size N')
        ax.set_ylabel('alpha (P(T) exponent)')
        ax.set_title('alpha vs subsample size', fontweight='bold')

        # Panel 3: gamma vs N
        ax = axes[0, 2]
        gamma_meds = [partA[n].get('gamma_median') for n in Ns]
        valid_gamma = [m is not None and not np.isnan(m) for m in gamma_meds]
        if any(valid_gamma):
            gamma_los = [partA[n].get('gamma_lo') for n in Ns]
            gamma_his = [partA[n].get('gamma_hi') for n in Ns]
            ax.errorbar(Ns, [m if m and not np.isnan(m) else 0 for m in gamma_meds],
                         yerr=[[m - lo if m and lo and not np.isnan(m) else 0
                                for m, lo in zip(gamma_meds, gamma_los)],
                               [hi - m if m and hi and not np.isnan(m) else 0
                                for m, hi in zip(gamma_meds, gamma_his)]],
                         fmt='ko-', capsize=5, linewidth=2, markersize=8)
            ax.axhline(2.0, color='red', linestyle='--', alpha=0.5, label='Theory (2.0)')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No valid gamma fits', transform=ax.transAxes, ha='center')
        ax.set_xlabel('Subsample size N')
        ax.set_ylabel('gamma (<S>~T^gamma)')
        ax.set_title('gamma vs subsample size', fontweight='bold')

        # Panel 4: n_avalanches vs N
        ax = axes[1, 0]
        nav_meds = [partA[n].get('n_avalanches_median', 0) for n in Ns]
        ax.bar(range(len(Ns)), nav_meds, color='steelblue', alpha=0.7)
        ax.set_xticks(range(len(Ns)))
        ax.set_xticklabels(Ns)
        ax.axhline(1000, color='red', linestyle='--', alpha=0.5, label='Min recommended')
        ax.set_xlabel('Subsample size N')
        ax.set_ylabel('N avalanches (median)')
        ax.set_title('Avalanche count', fontweight='bold')
        ax.legend()

        # Panel 5: decades vs N
        ax = axes[1, 1]
        dec_s = [partA[n].get('tau_decades_median', 0) for n in Ns]
        dec_t = [partA[n].get('alpha_decades_median', 0) for n in Ns]
        x_pos = np.arange(len(Ns))
        ax.bar(x_pos - 0.15, dec_s, 0.3, label='P(S) decades', color='steelblue', alpha=0.7)
        ax.bar(x_pos + 0.15, [d if d and not np.isnan(d) else 0 for d in dec_t],
               0.3, label='P(T) decades', color='darkorange', alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(Ns)
        ax.axhline(1.5, color='red', linestyle='--', alpha=0.5, label='Min for P(S)')
        ax.set_xlabel('Subsample size N')
        ax.set_ylabel('Decades of scaling')
        ax.set_title('Power-law fit range', fontweight='bold')
        ax.legend(fontsize=8)

        # Panel 6: criteria pass rates
        ax = axes[1, 2]
        criteria_names = ['P(S)', 'P(T)', 'Scaling', 'Collapse', 'Shuffle']
        x_pos = np.arange(len(criteria_names))
        width = 0.8 / len(Ns)
        for i, n in enumerate(Ns):
            rates = [partA[n]['frac_pl_s'], partA[n]['frac_pl_t'],
                     partA[n]['frac_scaling'], partA[n]['frac_collapse'],
                     partA[n]['frac_shuffle_pass']]
            ax.bar(x_pos + i * width, rates, width, label=f'N={n}', alpha=0.7)
        ax.set_xticks(x_pos + width * len(Ns) / 2)
        ax.set_xticklabels(criteria_names)
        ax.set_ylabel('Fraction of draws passing')
        ax.set_title('Criticality criteria pass rates', fontweight='bold')
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)

        fig.suptitle(f'Part A: Subsampled Avalanches -- S{sess_num} {region} ({sess_state}/{sess_phase})',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        partA_fig = figdir / f"scale_free_partA_S{sess_num}_{region}.png"
        plt.savefig(partA_fig, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  Saved {partA_fig}")

    # Save Part A JSON
    partA_json = outdir / f"scale_free_partA_S{sess_num}_{region}.json"
    with open(partA_json, 'w') as f:
        json.dump({str(k): v for k, v in partA.items()}, f, indent=2, default=convert)
    print(f"  Saved {partA_json}")

    # ==================================================================
    # COMBINED REPORT
    # ==================================================================
    print("\n" + "=" * 100)
    print("WRITING COMBINED REPORT")
    print("=" * 100)

    # Build interpretive summary
    primary_B = partB[f"{PRIMARY_BIN_MS}ms"]
    H_val = primary_B['H']
    beta_val = primary_B['beta']
    H_shuf_val = primary_B['shuffle_H']
    beta_shuf_val = primary_B['shuffle_beta']

    # Check Part B signals
    H_elevated = H_val > 0.6 and (H_val - H_shuf_val) > 0.1
    beta_elevated = beta_val > 0.5 and (beta_val - beta_shuf_val) > 0.3

    # Check Part A: any subsample size produce interpretable results?
    best_N = None
    best_decades = 0
    for n in Ns:
        d = partA[n].get('tau_decades_median', 0)
        if d and not np.isnan(d) and d > best_decades:
            best_decades = d
            best_N = n

    report_lines = [
        f"# Scale-Free Analysis Report -- S{sess_num} {region} ({sess_state}/{sess_phase})",
        f"",
        f"## Part B: DFA & 1/f Spectral Slope",
        f"",
        f"| Bin size | H (data) | H (shuffle) | beta (data) | beta (shuffle) |",
        f"|---------|----------|-------------|-------------|----------------|",
    ]
    for b in sorted(partB.keys(), key=lambda x: float(x.replace('ms', ''))):
        pb = partB[b]
        report_lines.append(
            f"| {b} | {pb['H']:.3f} [{pb['H_ci'][0]:.3f},{pb['H_ci'][1]:.3f}] | "
            f"{pb['shuffle_H']:.3f} | "
            f"{pb['beta']:.3f} [{pb['beta_ci'][0]:.3f},{pb['beta_ci'][1]:.3f}] | "
            f"{pb['shuffle_beta']:.3f} |")

    report_lines.extend([
        f"",
        f"**DFA interpretation:** H={H_val:.3f} at 10ms bins "
        f"({'ELEVATED (long-range correlations)' if H_elevated else 'near 0.5 or not above shuffle'}). "
        f"Shuffle control H={H_shuf_val:.3f} "
        f"({'as expected ~0.5' if H_shuf_val < 0.6 else 'WARNING: elevated shuffle'}).",
        f"",
        f"**1/f interpretation:** beta={beta_val:.3f} at 10ms bins "
        f"({'scale-free (pink-noise-like)' if beta_elevated else 'near white noise or not distinct from shuffle'}). "
        f"Shuffle beta={beta_shuf_val:.3f}.",
        f"",
        f"**Consistency check:** For related processes, beta ~ 2H - 1 = {2*H_val - 1:.3f}. "
        f"Measured beta = {beta_val:.3f}. "
        f"{'Consistent.' if abs(beta_val - (2*H_val - 1)) < 0.3 else 'Discrepant -- may indicate different scaling regimes.'}",
        f"",
    ])

    report_lines.extend([
        f"## Part A: Subsampled Avalanche Analysis",
        f"",
        f"| N | n_av (median) | max_dur | tau | tau decades | alpha | P(S) pass | P(T) pass | Scaling | Collapse | Shuffle |",
        f"|---|--------------|---------|-----|------------|-------|-----------|-----------|---------|----------|---------|",
    ])
    for n in Ns:
        s = partA[n]
        def fmt(val):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return "N/A"
            return f"{val:.3f}" if isinstance(val, float) else str(val)

        report_lines.append(
            f"| {n} | {fmt(s.get('n_avalanches_median'))} | "
            f"{fmt(s.get('max_duration_median'))} | "
            f"{fmt(s.get('tau_median'))} | {fmt(s.get('tau_decades_median'))} | "
            f"{fmt(s.get('alpha_median'))} | "
            f"{s['frac_pl_s']:.0%} | {s['frac_pl_t']:.0%} | "
            f"{s['frac_scaling']:.0%} | {s['frac_collapse']:.0%} | "
            f"{s['frac_shuffle_pass']:.0%} |")

    report_lines.extend([
        f"",
        f"**Subsample interpretation:** "
        f"{'No subsample size produced avalanches with substantial power-law range (best: ' + str(round(best_decades, 2)) + ' decades at N=' + str(best_N) + ').' if best_decades < 1.5 else 'N=' + str(best_N) + ' produced ' + str(round(best_decades, 2)) + ' decades -- possible power-law scaling.'}",
        f"",
    ])

    # Integrative interpretation
    report_lines.extend([
        f"## Integrative Interpretation",
        f"",
    ])

    if H_elevated and beta_elevated and best_decades >= 1.5:
        interp = (
            f"{region} shows converging evidence for scale-free dynamics during foraging: "
            f"DFA reveals long-range temporal correlations (H={H_val:.3f}), "
            f"power spectrum shows 1/f-like scaling (beta={beta_val:.3f}), "
            f"and subsampled avalanche analysis at N={best_N} produces {best_decades:.2f} decades "
            f"of power-law scaling. The consistency across methods supports genuine "
            f"scale-free dynamics rather than an artifact of any single method."
        )
    elif (H_elevated or beta_elevated) and best_decades < 1.5:
        interp = (
            f"{region} shows scale-free temporal structure by continuous metrics "
            f"(H={H_val:.3f}, beta={beta_val:.3f}) but NOT in the specific form "
            f"the avalanche framework assumes (best: {best_decades:.2f} decades at N={best_N}). "
            f"This is informative: the population rate has long-range temporal correlations "
            f"and/or 1/f spectral structure, but individual activity cascades do not "
            f"propagate as power-law-distributed avalanches. This pattern is consistent with "
            f"scale-free temporal dynamics that arise from modulation of overall excitability "
            f"(e.g., metabolic/arousal state fluctuations) rather than from local critical "
            f"branching processes. The dense, tonically active population rate does not "
            f"organize into discrete propagating events."
        )
    elif not H_elevated and not beta_elevated:
        interp = (
            f"Strong evidence that {region} during foraging does not show scale-free dynamics "
            f"by any framework tested. DFA (H={H_val:.3f}) shows no long-range correlations "
            f"beyond shuffle, spectral slope (beta={beta_val:.3f}) is near white noise, "
            f"and avalanche analysis fails at all subsample sizes. "
            f"Consider manifold-level or other approaches."
        )
    else:
        interp = (
            f"Results are mixed. H={H_val:.3f} "
            f"({'elevated' if H_elevated else 'not elevated'}), "
            f"beta={beta_val:.3f} ({'elevated' if beta_elevated else 'not elevated'}), "
            f"avalanche best={best_decades:.2f} decades. "
            f"No clear conclusion -- further investigation needed."
        )

    report_lines.append(interp)
    report_lines.extend([
        f"",
        f"*Note: {n_units} {region} units represent a tiny subsample of the full {region} population. "
        f"Subsampling is known to distort avalanche exponents and can affect DFA/PSD "
        f"estimates. These results characterize the recorded population, not necessarily "
        f"the full region.*",
    ])

    report_path = outdir / f"scale_free_report_S{sess_num}_{region}.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f"  Saved {report_path}")

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == '__main__':
    region_arg = sys.argv[1] if len(sys.argv) > 1 else 'ACA'
    session_arg = int(sys.argv[2]) if len(sys.argv) > 2 else None
    main(region=region_arg, session_num=session_arg)
