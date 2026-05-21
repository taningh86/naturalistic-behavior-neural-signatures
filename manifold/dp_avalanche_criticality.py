"""
Dual-Probe: Neuronal Avalanche Criticality Analysis
====================================================
Whole-session avalanche analysis on ACA and LHA separately.
Diagnostic first pass — NO behavioral stratification.

Pipeline per (session, region):
  1. Load quality-curated spike times
  2. Build population event time series, threshold at X SD
  3. Extract avalanches (contiguous active bins)
  4. Fit power laws via MLE (powerlaw package)
  5. Test scaling relation: (alpha-1)/(tau-1) vs measured 1/sigma*nu*z
  6. Shape collapse of avalanche profiles
  7. Shuffle control (per-unit ISI shuffle)
  8. Cross-region coincidence diagnostic

References:
  Beggs & Plenz 2003 (original avalanches)
  Alstott, Bullmore & Plenz 2014 (powerlaw package)
  Marshall et al. 2016 (shape collapse)

Run order per the spec:
  Phase 1: ONE pilot session, ACA only → stop & show
  Phase 2: LHA on same session (after review)
  Phase 3: Batch remaining sessions
"""

import yaml
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import powerlaw
import spikeinterface.extractors as se
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
import time as timer

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
FS = 30000  # Neuropixels sampling rate

# Unit selection thresholds (from memory / existing scripts)
P0_MIN_FR = 0.2       # ACA: KSLabel='good' + FR > 0.2, no AMP filter
P1_MIN_FR = 0.2       # LHA: KSLabel='good' + FR > 0.2 + AMP > 43
P1_MIN_AMP = 43
LHA_DEPTH_MIN = 0     # LHA depth range on probe 1
LHA_DEPTH_MAX = 345

# Avalanche parameters
# Primary threshold: 3.0 SD above mean population rate
# Sweep: [2.0, 2.5, 3.0, 3.5] SD
THRESHOLD_SDS = [2.0, 2.5, 3.0, 3.5]
PRIMARY_THRESHOLD_SD = 3.0
# Fixed dt values in ms — appropriate for in vivo Neuropixels cortical data.
# Population mean IEI (~0.5ms for 200+ units) is too fine and produces only
# single-bin coincidences, not cascades. Use 1-8ms range (Fontenele et al. 2019,
# Dahmen et al. 2019) and look for the bin width giving best power-law scaling.
# Sweep from fine (1ms) to coarse (50ms). Coarser bins may reveal
# cascading on slower timescales relevant for freely behaving animals.
DT_VALUES_MS = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
PRIMARY_DT_MS = 4.0  # 4ms — standard for cortical avalanches

# Minimum avalanche count for credible fits
MIN_AVALANCHES_WARN = 1000
MIN_AVALANCHES_HARD = 100

# Bootstrap CIs
N_BOOTSTRAP = 500

# Shape collapse
COLLAPSE_T_MIN = 3   # minimum avalanche duration (in bins) for profile analysis
COLLAPSE_T_MAX = 50   # maximum — avoid very rare long ones

# Shuffle control
N_SHUFFLES = 1        # 1 shuffle is enough for diagnostic (spec says rerun Steps 2-5)

# Sessions to skip
SKIP_SESSIONS = {1, 2, 23, 24, 25, 26}

# Pilot session
PILOT_SESSION = 3

with open("paths.yaml") as f:
    cfg = yaml.safe_load(f)

sessions_cfg = cfg["double_probe"]["coordinates_1"]["mouse01"]["sessions"]

outdir = Path("data/avalanche")
outdir.mkdir(parents=True, exist_ok=True)
figdir = Path("figures/avalanche")
figdir.mkdir(parents=True, exist_ok=True)


# ============================================================================
# UNIT LOADING
# ============================================================================

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


def load_spike_times_for_region(sorting, unit_ids):
    """Extract spike times (in seconds) for a set of unit IDs.
    Returns dict: unit_id -> np.array of spike times in seconds."""
    spike_dict = {}
    for uid in unit_ids:
        st = sorting.get_unit_spike_train(uid)  # in samples
        spike_dict[uid] = st / FS
    return spike_dict


# ============================================================================
# AVALANCHE EXTRACTION
# ============================================================================

def build_population_rate(spike_dict, dt, session_duration):
    """Sum all spikes across units into bins of width dt.
    Returns bin_edges, population_counts."""
    bin_edges = np.arange(0, session_duration + dt, dt)
    pop_counts = np.zeros(len(bin_edges) - 1, dtype=np.float64)
    for uid, times in spike_dict.items():
        counts, _ = np.histogram(times, bins=bin_edges)
        pop_counts += counts
    return bin_edges, pop_counts


def compute_mean_iei(spike_dict):
    """Compute mean inter-event interval across the population.
    Pool all spikes, sort, compute median ISI. This sets the natural
    timescale for avalanche binning (Beggs & Plenz 2003)."""
    all_spikes = np.concatenate(list(spike_dict.values()))
    all_spikes.sort()
    if len(all_spikes) < 2:
        return 0.001  # fallback
    isis = np.diff(all_spikes)
    # Use median rather than mean to be robust to outliers
    return float(np.median(isis))


def extract_avalanches(pop_counts, threshold):
    """Extract avalanches from thresholded population activity.
    An avalanche = contiguous run of bins above threshold,
    bounded by at least 1 bin below threshold.

    Returns list of dicts with 'size' (total spikes in avalanche),
    'duration' (number of bins), 'profile' (per-bin spike counts)."""
    above = pop_counts > threshold
    avalanches = []
    i = 0
    n = len(above)
    while i < n:
        if above[i]:
            # Start of an avalanche
            j = i
            while j < n and above[j]:
                j += 1
            # Avalanche spans bins [i, j)
            profile = pop_counts[i:j].copy()
            avalanches.append({
                'size': float(np.sum(profile)),
                'duration': j - i,
                'profile': profile,
                'start_bin': i,
            })
            i = j
        else:
            i += 1
    return avalanches


# ============================================================================
# POWER LAW FITTING
# ============================================================================

def fit_power_law(data, data_label="S", bootstrap=True):
    """Fit power law to data using the powerlaw package (MLE + KS).
    bootstrap=False skips CI computation (used in sensitivity sweep for speed).
    Returns dict with exponent, xmin, xmax, CI, LR tests, decades."""
    if len(data) < MIN_AVALANCHES_HARD:
        return {'exponent': np.nan, 'xmin': np.nan, 'error': 'too few data points'}

    data = np.array(data, dtype=float)
    # powerlaw package handles discrete/continuous automatically
    # For sizes (continuous) and durations (discrete integers)
    discrete = data_label == "T"

    # For durations, exclude T=1 (single-bin events carry no shape info)
    if discrete:
        data = data[data > 1]
        if len(data) < MIN_AVALANCHES_HARD:
            return {'exponent': np.nan, 'xmin': np.nan,
                    'error': 'too few multi-bin avalanches'}

    try:
        fit = powerlaw.Fit(data, discrete=discrete, verbose=False)
    except Exception as e:
        return {'exponent': np.nan, 'xmin': np.nan, 'error': str(e)}

    try:
        alpha = fit.power_law.alpha
        xmin = fit.power_law.xmin
    except (ValueError, Exception) as e:
        return {'exponent': np.nan, 'xmin': np.nan, 'error': str(e)}

    xmax = data.max()

    # Decades of scaling
    if xmin > 0:
        decades = np.log10(xmax / xmin)
    else:
        decades = 0

    result = {
        'exponent': float(alpha),
        'xmin': float(xmin),
        'xmax': float(xmax),
        'decades': float(decades),
        'n_above_xmin': int(np.sum(data >= xmin)),
    }

    if not bootstrap:
        # Quick mode for sweep — exponent only, no CI or LR tests
        return result

    # Likelihood ratio tests vs alternatives
    lr_tests = {}
    for alt in ['lognormal', 'exponential', 'truncated_power_law']:
        try:
            R, p = fit.distribution_compare('power_law', alt, normalized_ratio=True)
            lr_tests[alt] = {'R': float(R), 'p': float(p)}
        except Exception:
            lr_tests[alt] = {'R': np.nan, 'p': np.nan}
    result['lr_tests'] = lr_tests

    # Bootstrap CI for the exponent (fix xmin to avoid re-estimating each time)
    bootstrap_alphas = []
    n = len(data)
    for _ in range(N_BOOTSTRAP):
        resample = data[np.random.randint(0, n, n)]
        try:
            bfit = powerlaw.Fit(resample, discrete=discrete, xmin=xmin, verbose=False)
            ba = bfit.power_law.alpha
            if np.isfinite(ba):
                bootstrap_alphas.append(ba)
        except (ValueError, Exception):
            pass

    ci_lo, ci_hi = np.nan, np.nan
    if len(bootstrap_alphas) > 10:
        ci_lo = float(np.percentile(bootstrap_alphas, 2.5))
        ci_hi = float(np.percentile(bootstrap_alphas, 97.5))
    result['ci_lo'] = ci_lo
    result['ci_hi'] = ci_hi

    return result


# ============================================================================
# SCALING RELATION
# ============================================================================

def compute_mean_size_vs_duration(avalanches):
    """Compute <S>(T): mean avalanche size as a function of duration.
    Returns arrays of unique durations and corresponding mean sizes."""
    durations = np.array([a['duration'] for a in avalanches])
    sizes = np.array([a['size'] for a in avalanches])

    unique_T = np.unique(durations)
    mean_S = np.array([sizes[durations == t].mean() for t in unique_T])

    return unique_T, mean_S


def fit_scaling_exponent(unique_T, mean_S, min_T=2):
    """Fit <S>(T) ~ T^gamma via log-log linear regression.
    gamma should equal 1/(sigma*nu*z) ≈ 2.0 at criticality.
    Only use T >= min_T to avoid edge effects at T=1."""
    mask = unique_T >= min_T
    if mask.sum() < 3:
        return np.nan, np.nan
    logT = np.log10(unique_T[mask])
    logS = np.log10(mean_S[mask])
    # Linear regression in log-log space
    coeffs = np.polyfit(logT, logS, 1)
    gamma = coeffs[0]
    # Bootstrap CI
    n = len(logT)
    gammas = []
    for _ in range(N_BOOTSTRAP):
        idx = np.random.randint(0, n, n)
        c = np.polyfit(logT[idx], logS[idx], 1)
        gammas.append(c[0])
    ci = (float(np.percentile(gammas, 2.5)), float(np.percentile(gammas, 97.5)))
    return float(gamma), ci


# ============================================================================
# SHAPE COLLAPSE
# ============================================================================

def compute_shape_collapse(avalanches, gamma, t_min=COLLAPSE_T_MIN, t_max=COLLAPSE_T_MAX):
    """Compute avalanche shape collapse.
    For each duration T, compute mean profile s(t, T).
    Rescale: t -> t/T, s -> s * T^(gamma - 1).
    Returns collapsed profiles and collapse error."""
    durations = np.array([a['duration'] for a in avalanches])

    # Group avalanches by duration
    unique_T = np.unique(durations)
    unique_T = unique_T[(unique_T >= t_min) & (unique_T <= t_max)]

    if len(unique_T) < 3:
        return None, np.nan, None

    # Compute mean profile for each T
    profiles = {}
    for T in unique_T:
        T_avs = [a['profile'] for a in avalanches if a['duration'] == T]
        if len(T_avs) < 5:  # need enough avalanches per duration
            continue
        # Average profile
        mean_prof = np.mean(T_avs, axis=0)
        profiles[T] = mean_prof

    if len(profiles) < 3:
        return None, np.nan, None

    # Rescale and compute collapse
    collapsed = {}
    for T, prof in profiles.items():
        t_norm = np.linspace(0, 1, len(prof))
        # Rescale amplitude: s * T^(gamma - 1)
        # This is the Marshall et al. 2016 convention
        s_rescaled = prof * T**(gamma - 1)
        collapsed[T] = (t_norm, s_rescaled)

    # Quantify collapse error:
    # Interpolate all to common grid, compute variance relative to mean
    t_grid = np.linspace(0, 1, 100)
    interp_profiles = []
    for T, (t_norm, s_rescaled) in collapsed.items():
        interp = np.interp(t_grid, t_norm, s_rescaled)
        interp_profiles.append(interp)
    interp_profiles = np.array(interp_profiles)

    mean_profile = np.mean(interp_profiles, axis=0)
    variance = np.mean(np.var(interp_profiles, axis=0))
    mean_sq = np.mean(mean_profile**2)
    collapse_error = float(variance / mean_sq) if mean_sq > 0 else np.nan

    return collapsed, collapse_error, (t_grid, mean_profile)


def optimize_collapse_exponent(avalanches, gamma_range=(1.0, 3.5), n_steps=50,
                                t_min=COLLAPSE_T_MIN, t_max=COLLAPSE_T_MAX):
    """Find the gamma that minimizes shape collapse error.
    This is independent of the power-law fits — a self-consistency check."""
    gammas = np.linspace(gamma_range[0], gamma_range[1], n_steps)
    errors = []
    for g in gammas:
        _, err, _ = compute_shape_collapse(avalanches, g, t_min, t_max)
        errors.append(err if not np.isnan(err) else 1e10)
    errors = np.array(errors)
    best_idx = np.argmin(errors)
    return float(gammas[best_idx]), float(errors[best_idx])


# ============================================================================
# SHUFFLE CONTROL
# ============================================================================

def shuffle_spike_times(spike_dict, session_duration):
    """Shuffle spike times within each unit by circular shift.
    Preserves per-unit rate and ISI structure, destroys cross-unit timing."""
    shuffled = {}
    for uid, times in spike_dict.items():
        if len(times) == 0:
            shuffled[uid] = times.copy()
            continue
        # Random circular shift within session
        shift = np.random.uniform(0, session_duration)
        shifted = (times + shift) % session_duration
        shifted.sort()
        shuffled[uid] = shifted
    return shuffled


# ============================================================================
# CROSS-REGION COINCIDENCE DIAGNOSTIC
# ============================================================================

def cross_region_coincidence(spike_dict_aca, spike_dict_lha, dt=0.001, max_lag=0.05):
    """Compute cross-region population rate coincidence at zero lag vs jittered.
    Pool all spikes per region, bin at dt, compute cross-correlation.
    Returns (lags, cross_corr, zero_lag_value, jitter_mean, jitter_std)."""
    all_aca = np.concatenate(list(spike_dict_aca.values()))
    all_lha = np.concatenate(list(spike_dict_lha.values()))

    session_dur = max(all_aca.max(), all_lha.max()) + 1.0
    bins = np.arange(0, session_dur, dt)
    aca_rate, _ = np.histogram(all_aca, bins=bins)
    lha_rate, _ = np.histogram(all_lha, bins=bins)

    # Normalize
    aca_rate = (aca_rate - aca_rate.mean()) / max(aca_rate.std(), 1e-10)
    lha_rate = (lha_rate - lha_rate.mean()) / max(lha_rate.std(), 1e-10)

    # Cross-correlation at lags
    max_lag_bins = int(max_lag / dt)
    lags = np.arange(-max_lag_bins, max_lag_bins + 1) * dt
    cc = np.zeros(len(lags))
    n = len(aca_rate)
    for i, lag_bins in enumerate(range(-max_lag_bins, max_lag_bins + 1)):
        if lag_bins >= 0:
            cc[i] = np.dot(aca_rate[:n-lag_bins], lha_rate[lag_bins:]) / n
        else:
            cc[i] = np.dot(aca_rate[-lag_bins:], lha_rate[:n+lag_bins]) / n

    zero_lag = cc[max_lag_bins]

    # Jitter control: circular shift one region
    jitter_zeros = []
    for _ in range(20):
        shift = np.random.randint(int(1.0 / dt), n)  # shift by at least 1s
        lha_shifted = np.roll(lha_rate, shift)
        jitter_zeros.append(np.dot(aca_rate, lha_shifted) / n)

    return lags, cc, zero_lag, float(np.mean(jitter_zeros)), float(np.std(jitter_zeros))


# ============================================================================
# FULL ANALYSIS FOR ONE REGION, ONE SESSION
# ============================================================================

def run_avalanche_analysis(spike_dict, session_duration, region_name, session_num,
                            state, phase, n_units):
    """Full avalanche pipeline for one region.
    Returns results dict and generates diagnostic figure."""

    t0 = timer.time()
    results = {
        'session': session_num, 'region': region_name,
        'state': state, 'phase': phase,
        'n_units': n_units, 'session_duration_s': session_duration,
    }

    # Step 2: Mean IEI for dt
    mean_iei = compute_mean_iei(spike_dict)
    results['mean_iei_ms'] = round(mean_iei * 1000, 3)
    print(f"    Mean IEI = {mean_iei*1000:.2f} ms", flush=True)

    # Total spikes
    total_spikes = sum(len(v) for v in spike_dict.values())
    results['total_spikes'] = total_spikes
    pop_rate_hz = total_spikes / session_duration
    results['pop_rate_hz'] = round(pop_rate_hz, 1)

    # ---- Sensitivity sweep: all (threshold, dt) combos ----
    sweep_results = []
    primary_avalanches = None
    primary_dt = None
    primary_threshold = None

    for sd_thresh in THRESHOLD_SDS:
        for dt_ms in DT_VALUES_MS:
            dt = dt_ms / 1000.0  # convert to seconds

            bin_edges, pop_counts = build_population_rate(spike_dict, dt, session_duration)
            threshold = pop_counts.mean() + sd_thresh * pop_counts.std()
            avalanches = extract_avalanches(pop_counts, threshold)
            n_av = len(avalanches)

            sizes = np.array([a['size'] for a in avalanches]) if n_av > 0 else np.array([])
            durations = np.array([a['duration'] for a in avalanches]) if n_av > 0 else np.array([])

            # Fit power laws if enough avalanches (no bootstrap in sweep — speed)
            tau = np.nan
            alpha_exp = np.nan
            if n_av >= MIN_AVALANCHES_HARD:
                s_fit = fit_power_law(sizes, "S", bootstrap=False)
                t_fit = fit_power_law(durations, "T", bootstrap=False)
                tau = s_fit['exponent']
                alpha_exp = t_fit['exponent']

            row = {
                'threshold_sd': sd_thresh, 'dt_ms': dt_ms,
                'threshold_value': round(float(threshold), 2),
                'n_avalanches': n_av,
                'max_duration': int(durations.max()) if len(durations) > 0 else 0,
                'tau': round(tau, 3) if not np.isnan(tau) else None,
                'alpha': round(alpha_exp, 3) if not np.isnan(alpha_exp) else None,
            }
            sweep_results.append(row)
            print(f"      sweep: {sd_thresh}SD / {dt_ms}ms -> {n_av} av, "
                  f"max_T={row['max_duration']}, tau={row['tau']}, alpha={row['alpha']}", flush=True)

            # Store primary combo for detailed analysis
            if sd_thresh == PRIMARY_THRESHOLD_SD and dt_ms == PRIMARY_DT_MS:
                primary_avalanches = avalanches
                primary_dt = dt
                primary_threshold = threshold
                primary_pop_counts = pop_counts
                primary_bin_edges = bin_edges

    results['sweep'] = sweep_results

    # ---- Detailed analysis on primary (threshold=3SD, dt=1x IEI) ----
    if primary_avalanches is None or len(primary_avalanches) < MIN_AVALANCHES_HARD:
        results['error'] = f'Too few avalanches at primary settings: {len(primary_avalanches) if primary_avalanches else 0}'
        print(f"    *** TOO FEW AVALANCHES — STOPPING ***")
        return results

    n_av = len(primary_avalanches)
    results['n_avalanches'] = n_av
    results['dt_ms'] = round(primary_dt * 1000, 3)
    results['threshold_sd'] = PRIMARY_THRESHOLD_SD
    results['threshold_value'] = round(float(primary_threshold), 2)

    if n_av < MIN_AVALANCHES_WARN:
        results['warning'] = f'Only {n_av} avalanches — below recommended 1000 for credible fits'
        print(f"    WARNING: only {n_av} avalanches (want >=1000)")

    sizes = np.array([a['size'] for a in primary_avalanches])
    durations = np.array([a['duration'] for a in primary_avalanches])

    print(f"    {n_av} avalanches (dt={primary_dt*1000:.2f}ms, thresh={primary_threshold:.1f})")
    print(f"    Size: median={np.median(sizes):.0f}, max={sizes.max():.0f}")
    print(f"    Duration: median={np.median(durations)}, max={durations.max()}")

    # Step 4: Power law fits
    print(f"    Fitting P(S)...", end='', flush=True)
    s_fit = fit_power_law(sizes, "S")
    tau_ci_lo = s_fit.get('ci_lo', np.nan)
    tau_ci_hi = s_fit.get('ci_hi', np.nan)
    print(f" tau={s_fit['exponent']:.3f}"
          f" [{tau_ci_lo:.3f}, {tau_ci_hi:.3f}]"
          f" ({s_fit.get('decades',0):.2f} decades)")
    results['tau'] = s_fit

    print(f"    Fitting P(T)...", end='', flush=True)
    t_fit = fit_power_law(durations, "T")
    alpha_ci_lo = t_fit.get('ci_lo', np.nan)
    alpha_ci_hi = t_fit.get('ci_hi', np.nan)
    print(f" alpha={t_fit['exponent']:.3f}"
          f" [{alpha_ci_lo:.3f}, {alpha_ci_hi:.3f}]"
          f" ({t_fit.get('decades',0):.2f} decades)")
    results['alpha'] = t_fit

    # Step 5: Scaling relation
    unique_T, mean_S = compute_mean_size_vs_duration(primary_avalanches)
    gamma_measured, gamma_ci = fit_scaling_exponent(unique_T, mean_S)
    results['gamma_measured'] = gamma_measured
    results['gamma_ci'] = gamma_ci

    # Scaling relation: (alpha-1)/(tau-1) should equal gamma
    tau_val = s_fit['exponent']
    alpha_val = t_fit['exponent']
    if not np.isnan(tau_val) and not np.isnan(alpha_val) and (tau_val - 1) > 0.01:
        scaling_ratio = (alpha_val - 1) / (tau_val - 1)
    else:
        scaling_ratio = np.nan
    results['scaling_ratio'] = round(scaling_ratio, 3) if not np.isnan(scaling_ratio) else None
    results['scaling_relation_residual'] = round(abs(scaling_ratio - gamma_measured), 3) \
        if not np.isnan(scaling_ratio) and not np.isnan(gamma_measured) else None

    if isinstance(scaling_ratio, (int, float)) and not np.isnan(scaling_ratio):
        print(f"    <S>(T) scaling: gamma={gamma_measured:.3f}, "
              f"(alpha-1)/(tau-1)={scaling_ratio:.3f}")
    else:
        print(f"    <S>(T) scaling: gamma={gamma_measured}, ratio=N/A")

    # Step 6: Shape collapse
    print(f"    Computing shape collapse...", flush=True)
    gamma_opt, collapse_err_opt = optimize_collapse_exponent(primary_avalanches)
    collapsed, collapse_err, collapse_mean = compute_shape_collapse(
        primary_avalanches, gamma_measured if not np.isnan(gamma_measured) else 2.0)
    results['collapse_error'] = collapse_err
    results['collapse_error_optimized'] = collapse_err_opt
    results['gamma_collapse_optimized'] = gamma_opt
    print(f"    Collapse error: {collapse_err:.4f} (optimized: {collapse_err_opt:.4f} at gamma={gamma_opt:.2f})")

    # Step 7: Shuffle control
    print(f"    Running shuffle control...", flush=True)
    shuf_spike_dict = shuffle_spike_times(spike_dict, session_duration)
    shuf_bin_edges, shuf_pop_counts = build_population_rate(shuf_spike_dict, primary_dt, session_duration)
    shuf_threshold = shuf_pop_counts.mean() + PRIMARY_THRESHOLD_SD * shuf_pop_counts.std()
    shuf_avalanches = extract_avalanches(shuf_pop_counts, shuf_threshold)
    n_shuf_av = len(shuf_avalanches)
    results['shuffle_n_avalanches'] = n_shuf_av

    if n_shuf_av >= MIN_AVALANCHES_HARD:
        shuf_sizes = np.array([a['size'] for a in shuf_avalanches])
        shuf_durations = np.array([a['duration'] for a in shuf_avalanches])
        shuf_s_fit = fit_power_law(shuf_sizes, "S", bootstrap=False)
        shuf_t_fit = fit_power_law(shuf_durations, "T", bootstrap=False)
        results['shuffle_tau'] = shuf_s_fit['exponent']
        results['shuffle_alpha'] = shuf_t_fit['exponent']
        shuf_unique_T, shuf_mean_S = compute_mean_size_vs_duration(shuf_avalanches)
        shuf_gamma, _ = fit_scaling_exponent(shuf_unique_T, shuf_mean_S)
        results['shuffle_gamma'] = shuf_gamma

        # Flag if shuffle passes power law — pipeline problem
        shuf_s_lr = shuf_s_fit.get('lr_tests', {}).get('exponential', {}).get('p', 1.0)
        shuf_t_lr = shuf_t_fit.get('lr_tests', {}).get('exponential', {}).get('p', 1.0)
        if (not np.isnan(shuf_s_fit['exponent']) and
            shuf_s_lr < 0.05 and shuf_t_lr < 0.05):
            results['shuffle_warning'] = ('SHUFFLE SHOWS POWER LAWS — '
                                          'pipeline may have issues')
            print(f"    *** WARNING: SHUFFLED DATA SHOWS POWER LAWS ***")
        else:
            print(f"    Shuffle: tau={shuf_s_fit['exponent']:.3f}, "
                  f"alpha={shuf_t_fit['exponent']:.3f}, "
                  f"n_av={n_shuf_av}")
    else:
        results['shuffle_tau'] = None
        results['shuffle_alpha'] = None
        print(f"    Shuffle: only {n_shuf_av} avalanches (expected fewer than real)")

    elapsed = timer.time() - t0
    results['elapsed_s'] = round(elapsed, 1)
    print(f"    Done in {elapsed:.1f}s")

    # ---- DIAGNOSTIC FIGURE ----
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.35)

    # Panel A: Population rate trace with threshold
    ax = fig.add_subplot(gs[0, :2])
    t_centers = (primary_bin_edges[:-1] + primary_bin_edges[1:]) / 2
    # Show first 60s only for readability
    show_sec = 60
    show_bins = int(show_sec / primary_dt)
    show_bins = min(show_bins, len(primary_pop_counts))
    ax.plot(t_centers[:show_bins], primary_pop_counts[:show_bins],
            'k-', linewidth=0.3, alpha=0.7)
    ax.axhline(primary_threshold, color='red', linestyle='--', linewidth=1,
               label=f'threshold ({PRIMARY_THRESHOLD_SD}σ)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Population spike count')
    ax.set_title(f'Population Rate (first {show_sec}s)')
    ax.legend(fontsize=9)

    # Panel B: P(S) log-log with fit
    ax = fig.add_subplot(gs[0, 2])
    if not np.isnan(s_fit['exponent']):
        fit_obj = powerlaw.Fit(sizes, verbose=False)
        fit_obj.plot_pdf(ax=ax, color='blue', linewidth=0, marker='o',
                         markersize=3, label='Data')
        fit_obj.power_law.plot_pdf(ax=ax, color='red', linestyle='--',
                                    linewidth=2, label=f'τ={s_fit["exponent"]:.2f}')
        # Overlay shuffle
        if n_shuf_av >= MIN_AVALANCHES_HARD:
            shuf_fit_obj = powerlaw.Fit(shuf_sizes, verbose=False)
            shuf_fit_obj.plot_pdf(ax=ax, color='gray', linewidth=0,
                                   marker='x', markersize=3, alpha=0.5,
                                   label='Shuffle')
        ax.set_title(f'P(S): τ={s_fit["exponent"]:.2f} [{s_fit.get("ci_lo",0):.2f},{s_fit.get("ci_hi",0):.2f}]')
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, 'Fit failed', transform=ax.transAxes, ha='center')
    ax.set_xlabel('Avalanche size S')
    ax.set_ylabel('P(S)')

    # Panel C: P(T) log-log with fit
    ax = fig.add_subplot(gs[0, 3])
    if not np.isnan(t_fit['exponent']):
        fit_obj = powerlaw.Fit(durations, discrete=True, verbose=False)
        fit_obj.plot_pdf(ax=ax, color='blue', linewidth=0, marker='o',
                         markersize=3, label='Data')
        fit_obj.power_law.plot_pdf(ax=ax, color='red', linestyle='--',
                                    linewidth=2, label=f'α={t_fit["exponent"]:.2f}')
        if n_shuf_av >= MIN_AVALANCHES_HARD:
            shuf_fit_obj = powerlaw.Fit(shuf_durations, discrete=True, verbose=False)
            shuf_fit_obj.plot_pdf(ax=ax, color='gray', linewidth=0,
                                   marker='x', markersize=3, alpha=0.5,
                                   label='Shuffle')
        ax.set_title(f'P(T): α={t_fit["exponent"]:.2f} [{t_fit.get("ci_lo",0):.2f},{t_fit.get("ci_hi",0):.2f}]')
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, 'Fit failed', transform=ax.transAxes, ha='center')
    ax.set_xlabel('Avalanche duration T (bins)')
    ax.set_ylabel('P(T)')

    # Panel D: <S>(T) scaling
    ax = fig.add_subplot(gs[1, 0])
    if not np.isnan(gamma_measured):
        ax.loglog(unique_T, mean_S, 'ko', markersize=4, label='Data')
        # Fit line
        T_fit = np.logspace(np.log10(2), np.log10(unique_T.max()), 50)
        S_fit = T_fit**gamma_measured * (mean_S[unique_T >= 2][0] / (2**gamma_measured))
        ax.loglog(T_fit, S_fit, 'r--', linewidth=2,
                  label=f'γ={gamma_measured:.2f}')
        # Theory line (gamma=2)
        S_theory = T_fit**2.0 * (mean_S[unique_T >= 2][0] / (2**2.0))
        ax.loglog(T_fit, S_theory, 'b:', linewidth=1, alpha=0.5,
                  label='γ=2.0 (theory)')
        _sr_title = (f'\n(a-1)/(t-1)={scaling_ratio:.2f}'
                     if isinstance(scaling_ratio, (int, float)) and not np.isnan(scaling_ratio) else '')
        ax.set_title(f'<S>(T): γ={gamma_measured:.2f}{_sr_title}')
        ax.legend(fontsize=8)
    ax.set_xlabel('Duration T (bins)')
    ax.set_ylabel('<S>(T)')

    # Panel E: Shape collapse
    ax = fig.add_subplot(gs[1, 1])
    if collapsed is not None:
        cmap = plt.cm.viridis
        T_vals = sorted(collapsed.keys())
        norm = plt.Normalize(min(T_vals), max(T_vals))
        for T_val in T_vals:
            t_norm, s_rescaled = collapsed[T_val]
            ax.plot(t_norm, s_rescaled, color=cmap(norm(T_val)), alpha=0.7, linewidth=1)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        plt.colorbar(sm, ax=ax, label='T (bins)')
        ax.set_xlabel('t / T')
        ax.set_ylabel(f's · T^(γ-1)')
        ax.set_title(f'Shape Collapse (err={collapse_err:.4f})\n'
                     f'Opt: γ={gamma_opt:.2f}, err={collapse_err_opt:.4f}')
    else:
        ax.text(0.5, 0.5, 'Insufficient data\nfor collapse', transform=ax.transAxes, ha='center')
        ax.set_title('Shape Collapse')

    # Panel F: Sensitivity sweep heatmap (tau)
    ax = fig.add_subplot(gs[1, 2])
    sweep_df = pd.DataFrame(sweep_results)
    tau_pivot = sweep_df.pivot(index='threshold_sd', columns='dt_ms', values='tau')
    if not tau_pivot.empty:
        im = ax.imshow(tau_pivot.values, aspect='auto', cmap='RdYlBu_r',
                       vmin=1.0, vmax=2.5)
        ax.set_xticks(range(len(DT_VALUES_MS)))
        ax.set_xticklabels([f'{m}ms' for m in DT_VALUES_MS])
        ax.set_yticks(range(len(THRESHOLD_SDS)))
        ax.set_yticklabels([f'{s}σ' for s in THRESHOLD_SDS])
        ax.set_xlabel('Δt')
        ax.set_ylabel('Threshold (SD)')
        ax.set_title('τ sensitivity')
        for i in range(len(THRESHOLD_SDS)):
            for j in range(len(DT_VALUES_MS)):
                v = tau_pivot.values[i, j]
                if not np.isnan(v) if isinstance(v, float) else v is not None:
                    ax.text(j, i, f'{float(v):.2f}', ha='center', va='center', fontsize=8)
        plt.colorbar(im, ax=ax)

    # Panel G: Sensitivity sweep heatmap (alpha)
    ax = fig.add_subplot(gs[1, 3])
    alpha_pivot = sweep_df.pivot(index='threshold_sd', columns='dt_ms', values='alpha')
    if not alpha_pivot.empty:
        im = ax.imshow(alpha_pivot.values, aspect='auto', cmap='RdYlBu_r',
                       vmin=1.0, vmax=3.5)
        ax.set_xticks(range(len(DT_VALUES_MS)))
        ax.set_xticklabels([f'{m}ms' for m in DT_VALUES_MS])
        ax.set_yticks(range(len(THRESHOLD_SDS)))
        ax.set_yticklabels([f'{s}σ' for s in THRESHOLD_SDS])
        ax.set_xlabel('Δt')
        ax.set_ylabel('Threshold (SD)')
        ax.set_title('α sensitivity')
        for i in range(len(THRESHOLD_SDS)):
            for j in range(len(DT_VALUES_MS)):
                v = alpha_pivot.values[i, j]
                if not np.isnan(v) if isinstance(v, float) else v is not None:
                    ax.text(j, i, f'{float(v):.2f}', ha='center', va='center', fontsize=8)
        plt.colorbar(im, ax=ax)

    # Panel H: Sensitivity sweep (n_avalanches)
    ax = fig.add_subplot(gs[2, 0])
    nav_pivot = sweep_df.pivot(index='threshold_sd', columns='dt_ms', values='n_avalanches')
    if not nav_pivot.empty:
        im = ax.imshow(nav_pivot.values, aspect='auto', cmap='YlOrRd')
        ax.set_xticks(range(len(DT_VALUES_MS)))
        ax.set_xticklabels([f'{m}ms' for m in DT_VALUES_MS])
        ax.set_yticks(range(len(THRESHOLD_SDS)))
        ax.set_yticklabels([f'{s}σ' for s in THRESHOLD_SDS])
        ax.set_xlabel('Δt')
        ax.set_ylabel('Threshold (SD)')
        ax.set_title('N avalanches')
        for i in range(len(THRESHOLD_SDS)):
            for j in range(len(DT_VALUES_MS)):
                v = nav_pivot.values[i, j]
                if not np.isnan(v) if isinstance(v, float) else v is not None:
                    ax.text(j, i, f'{int(v)}', ha='center', va='center', fontsize=8)
        plt.colorbar(im, ax=ax)

    # Panel I: Size and duration distributions (histograms)
    ax = fig.add_subplot(gs[2, 1])
    ax.hist(sizes, bins=50, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.3)
    ax.set_xlabel('Avalanche size S')
    ax.set_ylabel('Count')
    ax.set_title(f'Size distribution (N={n_av})')
    ax.set_yscale('log')

    ax2 = fig.add_subplot(gs[2, 2])
    ax2.hist(durations, bins=np.arange(0.5, durations.max()+1.5, 1),
             color='darkorange', alpha=0.7, edgecolor='black', linewidth=0.3)
    ax2.set_xlabel('Avalanche duration T (bins)')
    ax2.set_ylabel('Count')
    ax2.set_title(f'Duration distribution (N={n_av})')
    ax2.set_yscale('log')

    # Panel J: Scaling relation summary text
    ax = fig.add_subplot(gs[2, 3])
    ax.axis('off')
    summary_lines = [
        f"Session S{session_num} — {region_name} ({state}/{phase})",
        f"Units: {n_units}   Spikes: {total_spikes:,}",
        f"Δt = {primary_dt*1000:.2f} ms",
        f"Threshold: {PRIMARY_THRESHOLD_SD}σ = {primary_threshold:.1f}",
        f"Avalanches: {n_av}",
        "",
        f"τ (P(S) exponent): {s_fit['exponent']:.3f} [{s_fit.get('ci_lo',0):.3f}, {s_fit.get('ci_hi',0):.3f}]",
        f"  Decades: {s_fit.get('decades',0):.2f}",
        f"  vs lognormal: R={s_fit.get('lr_tests',{}).get('lognormal',{}).get('R','?')}, "
        f"p={s_fit.get('lr_tests',{}).get('lognormal',{}).get('p','?')}",
        "",
        f"α (P(T) exponent): {t_fit['exponent']:.3f} [{t_fit.get('ci_lo',0):.3f}, {t_fit.get('ci_hi',0):.3f}]",
        f"  Decades: {t_fit.get('decades',0):.2f}",
        "",
        f"γ measured (<S>~T^γ): {gamma_measured:.3f}",
        f"(a-1)/(t-1) = {scaling_ratio:.3f}" if isinstance(scaling_ratio, (int, float)) and not np.isnan(scaling_ratio) else "(a-1)/(t-1) = N/A",
        f"Scaling residual: {abs(scaling_ratio - gamma_measured):.3f}" if results.get('scaling_relation_residual') else "",
        "",
        f"Collapse error: {collapse_err:.4f}" if not np.isnan(collapse_err) else "Collapse: N/A",
        f"Optimal collapse: γ={gamma_opt:.2f}, err={collapse_err_opt:.4f}",
        "",
        f"Shuffle: τ={results.get('shuffle_tau','N/A')}, "
        f"α={results.get('shuffle_alpha','N/A')}, "
        f"n={results.get('shuffle_n_avalanches','N/A')}",
    ]
    ax.text(0.02, 0.98, '\n'.join(summary_lines), transform=ax.transAxes,
            va='top', ha='left', fontsize=8, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    fig.suptitle(f'Neuronal Avalanche Analysis — S{session_num} {region_name} ({state}/{phase})',
                 fontsize=16, fontweight='bold')

    figpath = figdir / f"avalanche_S{session_num}_{region_name}.png"
    plt.savefig(figpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved {figpath}")

    # Save results JSON
    jsonpath = outdir / f"avalanche_S{session_num}_{region_name}.json"
    # Convert numpy types for JSON serialization
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

    with open(jsonpath, 'w') as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"    Saved {jsonpath}")

    return results


# ============================================================================
# MAIN — PILOT: Session 3, ACA only
# ============================================================================

def main():
    print("=" * 100)
    print("NEURONAL AVALANCHE CRITICALITY ANALYSIS — PILOT")
    print("=" * 100)

    skey = f"session_{PILOT_SESSION}"
    sval = sessions_cfg[skey]
    state = sval['state']
    phase = sval['phase']

    print(f"\nPilot: S{PILOT_SESSION} ({state}/{phase})")

    # Load ACA (probe 0)
    p0_sorted = sval['probe_0_aca']['sorted']
    p0_path = Path(p0_sorted)
    print(f"\n  Loading ACA units from {p0_path}...")

    aca_ids = get_good_units_p0(p0_path)
    try:
        sorting_p0 = se.read_kilosort(p0_path)
        avail_p0 = set(sorting_p0.get_unit_ids())
        aca_ids = np.array([u for u in aca_ids if u in avail_p0])
    except Exception as e:
        print(f"  ERROR loading probe 0: {e}")
        sys.exit(1)

    print(f"  ACA: {len(aca_ids)} good units")
    if len(aca_ids) < 5:
        print(f"  ERROR: only {len(aca_ids)} ACA units — need at least 5")
        sys.exit(1)

    # Get spike times
    print(f"  Loading spike times...")
    aca_spikes = load_spike_times_for_region(sorting_p0, aca_ids)

    # Session duration: max spike time + buffer
    all_spike_times = np.concatenate(list(aca_spikes.values()))
    session_duration = float(all_spike_times.max()) + 1.0
    print(f"  Session duration: {session_duration:.1f}s ({session_duration/60:.1f} min)")

    # Run ACA analysis
    print(f"\n  === ACA AVALANCHE ANALYSIS ===")
    aca_results = run_avalanche_analysis(
        aca_spikes, session_duration, 'ACA', PILOT_SESSION,
        state, phase, len(aca_ids))

    # ---- Write markdown summary ----
    md_path = outdir / f"avalanche_S{PILOT_SESSION}_ACA_summary.md"
    tau = aca_results.get('tau', {})
    alpha = aca_results.get('alpha', {})
    gamma = aca_results.get('gamma_measured', np.nan)
    sr = aca_results.get('scaling_ratio', np.nan)
    ce = aca_results.get('collapse_error', np.nan)

    # Assess criticality criteria
    criteria = []
    # 1. Power law P(S)
    if isinstance(tau, dict) and not np.isnan(tau.get('exponent', np.nan)):
        tau_val = tau['exponent']
        tau_decades = tau.get('decades', 0)
        tau_lr_ln = tau.get('lr_tests', {}).get('lognormal', {}).get('p', 1)
        pl_s_pass = tau_decades >= 1.5 and (isinstance(tau_lr_ln, float) and tau_lr_ln < 0.05)
        criteria.append(('P(S) power law (>=1.5 decades + LR vs lognormal)',
                         'PASS' if pl_s_pass else 'FAIL',
                         f'tau={tau_val:.3f}, {tau_decades:.2f} decades, LR p={tau_lr_ln}'))
    else:
        criteria.append(('P(S) power law', 'FAIL', 'Fit failed'))

    # 2. Power law P(T)
    if isinstance(alpha, dict) and not np.isnan(alpha.get('exponent', np.nan)):
        alpha_val = alpha['exponent']
        alpha_decades = alpha.get('decades', 0)
        alpha_lr_ln = alpha.get('lr_tests', {}).get('lognormal', {}).get('p', 1)
        pl_t_pass = alpha_decades >= 1.0  # T has fewer decades typically
        criteria.append(('P(T) power law',
                         'PASS' if pl_t_pass else 'MARGINAL',
                         f'alpha={alpha_val:.3f}, {alpha_decades:.2f} decades'))
    else:
        criteria.append(('P(T) power law', 'FAIL', 'Fit failed'))

    # 3. Scaling relation
    _sr = sr if isinstance(sr, (int, float)) else np.nan
    _gamma = gamma if isinstance(gamma, (int, float)) else np.nan
    if not np.isnan(_sr) and not np.isnan(_gamma):
        sr_residual = abs(_sr - _gamma)
        sr_pass = sr_residual < 0.3  # generous tolerance for in vivo
        criteria.append(('Scaling relation (alpha-1)/(tau-1) = gamma',
                         'PASS' if sr_pass else 'FAIL',
                         f'ratio={_sr:.3f}, gamma={_gamma:.3f}, residual={sr_residual:.3f}'))
    else:
        criteria.append(('Scaling relation', 'FAIL', 'Could not compute'))

    # 4. Shape collapse
    _ce = ce if isinstance(ce, (int, float)) else np.nan
    if not np.isnan(_ce):
        ce_pass = _ce < 0.1  # arbitrary but reasonable
        criteria.append(('Shape collapse',
                         'PASS' if ce_pass else 'MARGINAL',
                         f'error={_ce:.4f}'))
    else:
        criteria.append(('Shape collapse', 'FAIL', 'N/A'))

    # 5. Shuffle fails
    shuf_tau = aca_results.get('shuffle_tau')
    shuf_n = aca_results.get('shuffle_n_avalanches', 0)
    if shuf_tau is not None and isinstance(shuf_tau, (int, float)) and not np.isnan(shuf_tau):
        # Shuffle should NOT reproduce the real exponents
        real_tau = tau.get('exponent', 0) if isinstance(tau, dict) else 0
        shuf_differs = abs(shuf_tau - real_tau) > 0.2
        criteria.append(('Shuffle control (fails to reproduce)',
                         'PASS' if shuf_differs or shuf_n < MIN_AVALANCHES_HARD else 'FAIL',
                         f'shuffle tau={shuf_tau:.3f} vs real tau={real_tau:.3f}'))
    elif shuf_n < MIN_AVALANCHES_HARD:
        criteria.append(('Shuffle control', 'PASS', f'Only {shuf_n} shuffle avalanches'))
    else:
        criteria.append(('Shuffle control', 'N/A', ''))

    # Overall verdict
    passes = sum(1 for _, v, _ in criteria if v == 'PASS')
    total = len(criteria)

    md_lines = [
        f"# Avalanche Criticality -- S{PILOT_SESSION} ACA ({state}/{phase})\n",
        f"## Summary",
        f"- **Units**: {len(aca_ids)}",
        f"- **Session duration**: {session_duration:.1f}s",
        f"- **Avalanches**: {aca_results.get('n_avalanches', 'N/A')} "
        f"(dt={PRIMARY_DT_MS}ms, threshold={PRIMARY_THRESHOLD_SD} SD)",
        f"- **Total spikes**: {aca_results.get('total_spikes', '?'):,}",
        f"- **Mean IEI**: {aca_results.get('mean_iei_ms', '?')}ms\n",
        f"## Exponents",
        f"| Parameter | Value | 95% CI | Decades | Theory |",
        f"|-----------|-------|--------|---------|--------|",
    ]
    if isinstance(tau, dict) and not np.isnan(tau.get('exponent', np.nan)):
        md_lines.append(f"| tau (P(S)) | {tau['exponent']:.3f} | "
                        f"[{tau.get('ci_lo',0):.3f}, {tau.get('ci_hi',0):.3f}] | "
                        f"{tau.get('decades',0):.2f} | ~1.5 |")
    if isinstance(alpha, dict) and not np.isnan(alpha.get('exponent', np.nan)):
        md_lines.append(f"| alpha (P(T)) | {alpha['exponent']:.3f} | "
                        f"[{alpha.get('ci_lo',0):.3f}, {alpha.get('ci_hi',0):.3f}] | "
                        f"{alpha.get('decades',0):.2f} | ~2.0 |")
    gamma_str = f"{gamma:.3f}" if isinstance(gamma, (int, float)) and not np.isnan(gamma) else "N/A"
    sr_str = f"{sr:.3f}" if isinstance(sr, (int, float)) and not np.isnan(sr) else "N/A"
    md_lines.append(f"| gamma (<S>~T^gamma) | {gamma_str} | {aca_results.get('gamma_ci','?')} | - | ~2.0 |")
    md_lines.append(f"| (alpha-1)/(tau-1) | {sr_str} | - | - | = gamma |")

    md_lines.extend([
        f"\n## Criticality Criteria",
        f"| Criterion | Result | Details |",
        f"|-----------|--------|---------|",
    ])
    for name, result, details in criteria:
        md_lines.append(f"| {name} | **{result}** | {details} |")

    md_lines.extend([
        f"\n## Verdict",
        f"**{passes}/{total} criteria pass.**",
        f"",
        f"*Note: Subsampling (recording ~{len(aca_ids)} of millions of ACA neurons) "
        f"is known to distort exponents. Freely behaving in vivo data is harder than "
        f"slice preparations -- exact theoretical values (tau=1.5, alpha=2.0) are not expected.*",
    ])

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    print(f"\n  Saved {md_path}")

    print("\n" + "=" * 100)
    print("PILOT COMPLETE — Review diagnostic figure and results before proceeding")
    print("=" * 100)


if __name__ == '__main__':
    main()
