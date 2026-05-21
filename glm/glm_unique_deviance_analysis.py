"""
GLM Unique Deviance Analysis: Identifying Preferred Encoding Per Neuron

Poisson GLMs with a cleaned predictor set, using leave-one-out unique deviance
to identify each neuron's dominant predictor and functional category.

Reports three complementary dominance metrics:
  1. GLM unique deviance (excitatory & inhibitory separately)
  2. Rate elevation (fold-change in firing rate when predictor is active)
  3. Spike-weighted dominance (excess spikes above baseline)

Outputs:
  - Per-session detailed CSV: one row per (unit x predictor)
  - Per-neuron summary CSV: dominant predictors by all three metrics
  - Combined all-sessions CSVs
  - Summary figures

Usage:
    conda activate si_env
    python glm_unique_deviance_analysis.py
"""

import yaml
import time
from pathlib import Path
import numpy as np
import pandas as pd
import spikeinterface.extractors as se
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy as scipy_entropy, pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================
PATHS_YAML = Path("paths.yaml")
METRICS_CSV = Path("data/all_sessions_unit_metrics_by_region.csv")
OUTPUT_DIR = Path("figures")
DATA_DIR = Path("data")
FS = 30000  # Neuropixels sampling rate (Hz)
BIN_WIDTH = 0.1  # 100 ms bins

# --- Predictor set ---
# Pot-1 through Pot-4: tight interaction zones at each pot
# Pot-2 zone, Pot-4 zone: broader zones around pot 2 and pot 4
# All distance variables removed (redundant with zone occupancy)
# Distance moved removed (collinear with Velocity)
CONTINUOUS_VARS = ['Velocity', 'Meander']
ZONE_VARS = [
    'Home', 'Foraging arena', 'Transition zone',
    'Pot-1', 'Pot-2', 'Pot-3', 'Pot-4',
    'Pot-2 zone', 'Pot-4 zone',
]
BEHAVIORAL_STATE_VARS = ['Feeding', 'Grooming']
EXPLORATION_COMBINE = [
    'Longer exploration at home',
    'Quick and hasty exploration at home',
    'Hesitant exploration',
]

# Category definitions for leave-one-category-out analysis
PREDICTOR_CATEGORIES = {
    'Velocity': 'Locomotion',
    'Meander': 'Locomotion',
    'Home': 'Spatial zones',
    'Foraging arena': 'Spatial zones',
    'Transition zone': 'Spatial zones',
    'Pot-1': 'Spatial zones',
    'Pot-2': 'Spatial zones',
    'Pot-3': 'Spatial zones',
    'Pot-4': 'Spatial zones',
    'Pot-2 zone': 'Spatial zones',
    'Pot-4 zone': 'Spatial zones',
    'Feeding': 'Behavioral state',
    'Grooming': 'Behavioral state',
    'Active_exploration': 'Behavioral state',
}

CATEGORY_NAMES = ['Locomotion', 'Spatial zones', 'Behavioral state']

# All binary predictor names (for rate elevation / spike-weighted computation)
ALL_BINARY_PREDICTORS = ZONE_VARS + BEHAVIORAL_STATE_VARS  # Active_exploration added dynamically

# Cross-validation settings
ALPHA_GRID = np.logspace(-4, 1, 20)
N_CV_FOLDS = 5


# =============================================================================
# Data loading
# =============================================================================

def load_behavior(behavior_path: Path) -> pd.DataFrame:
    """Load transposed behavior CSV and return DataFrame with time bins as rows."""
    behav_raw = pd.read_csv(behavior_path, index_col=0)
    behav_df = behav_raw.T.reset_index(drop=True)
    behav_df.columns = behav_df.columns.str.strip()
    for col in behav_df.columns:
        behav_df[col] = pd.to_numeric(behav_df[col], errors='coerce')
    return behav_df


def bin_spikes(
    spike_times_samples: np.ndarray,
    rec_time: np.ndarray,
    fs: float,
) -> np.ndarray:
    """Bin spike times into 100ms bins aligned to behavior timestamps."""
    spike_times_sec = spike_times_samples / fs
    dt = np.median(np.diff(rec_time))
    bin_edges = np.concatenate([rec_time - dt / 2, [rec_time[-1] + dt / 2]])
    counts, _ = np.histogram(spike_times_sec, bins=bin_edges)
    return counts


# =============================================================================
# Predictor preparation
# =============================================================================

def prepare_predictors(
    behav_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Extract and preprocess the predictor set.

    Returns
    -------
    predictors : pd.DataFrame
        Shape (n_time_bins, n_predictors).
    predictor_names : list[str]
        Ordered list of predictor names.
    """
    predictors = pd.DataFrame(index=behav_df.index)
    predictor_names = []

    # --- Continuous variables (z-scored) ---
    for var in CONTINUOUS_VARS:
        if var not in behav_df.columns:
            print(f"    [WARN] Missing continuous variable: {var}")
            continue
        vals = behav_df[var].values.astype(float).copy()
        vals[~np.isfinite(vals)] = 0.0
        std = np.std(vals)
        if std == 0:
            print(f"    [WARN] Zero-variance continuous variable: {var}")
            continue
        scaler = StandardScaler()
        predictors[var] = scaler.fit_transform(vals.reshape(-1, 1)).ravel()
        predictor_names.append(var)

    # --- Zone occupancy (binary) ---
    for var in ZONE_VARS:
        if var not in behav_df.columns:
            print(f"    [WARN] Missing zone variable: {var}")
            continue
        binary_vals = (behav_df[var].values > 0).astype(float)
        occupancy_frac = binary_vals.mean()
        if occupancy_frac < 0.005:
            print(f"    [WARN] Very sparse zone variable: {var} "
                  f"({100*occupancy_frac:.2f}% occupancy) -- skipping")
            continue
        predictors[var] = binary_vals
        predictor_names.append(var)

    # --- Behavioral states (binary) ---
    for var in BEHAVIORAL_STATE_VARS:
        if var not in behav_df.columns:
            print(f"    [WARN] Missing behavioral state: {var}")
            continue
        binary_vals = (behav_df[var].values > 0).astype(float)
        occupancy_frac = binary_vals.mean()
        if occupancy_frac < 0.005:
            print(f"    [WARN] Very sparse behavioral state: {var} "
                  f"({100*occupancy_frac:.2f}% occupancy) -- skipping")
            continue
        predictors[var] = binary_vals
        predictor_names.append(var)

    # --- Combined active exploration ---
    active_expl = np.zeros(len(behav_df))
    for var in EXPLORATION_COMBINE:
        if var in behav_df.columns:
            vals = behav_df[var].values.astype(float)
            vals[~np.isfinite(vals)] = 0.0
            active_expl = np.maximum(active_expl, vals)
    active_expl_binary = (active_expl > 0).astype(float)
    occupancy_frac = active_expl_binary.mean()
    if occupancy_frac >= 0.005:
        predictors['Active_exploration'] = active_expl_binary
        predictor_names.append('Active_exploration')
    else:
        print(f"    [WARN] Active_exploration very sparse "
              f"({100*occupancy_frac:.2f}%) -- skipping")

    predictors = predictors.fillna(0)
    return predictors, predictor_names


# =============================================================================
# GLM fitting and deviance computations
# =============================================================================

def fit_glm_get_deviance(y: np.ndarray, X: np.ndarray) -> tuple[float, bool]:
    """Fit Poisson GLM and return deviance."""
    X_const = sm.add_constant(X, has_constant='add')
    try:
        model = sm.GLM(
            y, X_const,
            family=sm.families.Poisson(link=sm.families.links.Log()),
        )
        result = model.fit(maxiter=100, method='IRLS')
        return result.deviance, True
    except Exception:
        return np.nan, False


def fit_full_glm(y: np.ndarray, X: np.ndarray, predictor_names: list[str]) -> dict:
    """Fit full Poisson GLM and return all inference results plus deviance."""
    X_const = sm.add_constant(X, has_constant='add')
    n_pred = len(predictor_names)

    try:
        model = sm.GLM(
            y, X_const,
            family=sm.families.Poisson(link=sm.families.links.Log()),
        )
        result = model.fit(maxiter=100, method='IRLS')

        coefs = result.params[1:]
        std_errors = result.bse[1:]
        z_scores = coefs / std_errors
        p_values = result.pvalues[1:]

        null_model = sm.GLM(
            y, np.ones((len(y), 1)),
            family=sm.families.Poisson(),
        )
        null_result = null_model.fit()
        pseudo_r2 = 1 - (result.llf / null_result.llf)

        return {
            'coefficients': coefs,
            'std_errors': std_errors,
            'z_scores': z_scores,
            'p_values': p_values,
            'pseudo_r2': pseudo_r2,
            'aic': result.aic,
            'deviance': result.deviance,
            'converged': True,
        }
    except Exception as e:
        print(f" [GLM error: {e}]", end='')
        return {
            'coefficients': np.full(n_pred, np.nan),
            'std_errors': np.full(n_pred, np.nan),
            'z_scores': np.full(n_pred, np.nan),
            'p_values': np.full(n_pred, np.nan),
            'pseudo_r2': np.nan,
            'aic': np.nan,
            'deviance': np.nan,
            'converged': False,
        }


def compute_unique_deviance(
    y: np.ndarray, X: np.ndarray, predictor_names: list[str], deviance_full: float,
) -> np.ndarray:
    """Compute unique deviance for each predictor via leave-one-out."""
    n_pred = len(predictor_names)
    unique_dev = np.full(n_pred, np.nan)

    for i in range(n_pred):
        X_reduced = np.delete(X, i, axis=1)
        dev_reduced, converged = fit_glm_get_deviance(y, X_reduced)
        if converged and np.isfinite(deviance_full):
            unique_dev[i] = dev_reduced - deviance_full
        else:
            unique_dev[i] = np.nan

    return unique_dev


def compute_category_deviance(
    y: np.ndarray, X: np.ndarray, predictor_names: list[str], deviance_full: float,
) -> dict[str, float]:
    """Compute unique deviance for each category via leave-one-category-out."""
    cat_deviance = {}

    for cat in CATEGORY_NAMES:
        drop_indices = [
            i for i, p in enumerate(predictor_names)
            if PREDICTOR_CATEGORIES.get(p) == cat
        ]

        if not drop_indices:
            cat_deviance[cat] = np.nan
            continue

        X_reduced = np.delete(X, drop_indices, axis=1)

        if X_reduced.shape[1] == 0:
            X_const = np.ones((len(y), 1))
            try:
                model = sm.GLM(y, X_const, family=sm.families.Poisson())
                result = model.fit(maxiter=100, method='IRLS')
                dev_reduced = result.deviance
            except Exception:
                cat_deviance[cat] = np.nan
                continue
        else:
            dev_reduced, converged = fit_glm_get_deviance(y, X_reduced)
            if not converged:
                cat_deviance[cat] = np.nan
                continue

        if np.isfinite(deviance_full):
            cat_deviance[cat] = dev_reduced - deviance_full
        else:
            cat_deviance[cat] = np.nan

    return cat_deviance


def compute_selectivity_index(unique_deviances: np.ndarray) -> float:
    """Compute selectivity index from unique deviance values.

    selectivity = 1 - (entropy / log(N))
    Range: 0 (equally driven) to 1 (driven by one only).
    """
    valid = unique_deviances[np.isfinite(unique_deviances)]
    valid = np.maximum(valid, 0.0)

    if len(valid) == 0 or valid.sum() == 0:
        return np.nan

    proportions = valid / valid.sum()
    n = len(proportions)
    if n <= 1:
        return 1.0

    h = scipy_entropy(proportions)
    max_h = np.log(n)
    return 1.0 - (h / max_h)


def cv_poisson_regression(y: np.ndarray, X: np.ndarray) -> tuple[float, float]:
    """Cross-validated D2 using sklearn PoissonRegressor with TimeSeriesSplit."""
    tscv = TimeSeriesSplit(n_splits=N_CV_FOLDS)
    best_alpha = ALPHA_GRID[0]
    best_score = -np.inf

    for alpha in ALPHA_GRID:
        fold_scores = []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            try:
                model = PoissonRegressor(alpha=alpha, max_iter=1000)
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
                fold_scores.append(score)
            except Exception:
                fold_scores.append(-np.inf)
        mean_score = np.mean(fold_scores)
        if mean_score > best_score:
            best_score = mean_score
            best_alpha = alpha

    return best_score, best_alpha


# =============================================================================
# Rate elevation & spike-weighted metrics
# =============================================================================

def compute_rate_metrics(
    spike_counts: np.ndarray,
    behav_df: pd.DataFrame,
    predictor_names: list[str],
) -> dict[str, dict]:
    """Compute rate elevation and excess spikes for each predictor.

    Returns dict: predictor_name -> {rate_elevation, excess_spikes,
                                     rate_when_active_hz, rate_when_inactive_hz,
                                     spikes_when_active, n_active_bins, occupancy_pct}
    """
    n_bins = len(spike_counts)
    total_spikes = spike_counts.sum()
    overall_rate = total_spikes / n_bins / BIN_WIDTH
    mean_count_per_bin = total_spikes / n_bins

    results = {}

    for pred in predictor_names:
        cat = PREDICTOR_CATEGORIES.get(pred, 'Unknown')

        # Continuous predictors: use correlation
        if pred in CONTINUOUS_VARS:
            if pred in behav_df.columns:
                vals = behav_df[pred].values.astype(float)
                valid = np.isfinite(vals)
                if valid.sum() >= 10:
                    r, _ = pearsonr(vals[valid], spike_counts[valid])
                    results[pred] = {
                        'rate_elevation': r,
                        'excess_spikes': np.nan,
                        'rate_when_active_hz': overall_rate,
                        'rate_when_inactive_hz': 0.0,
                        'spikes_when_active': int(total_spikes),
                        'n_active_bins': int(valid.sum()),
                        'occupancy_pct': 100.0,
                    }
            continue

        # Binary predictors: compute directly
        if pred == 'Active_exploration':
            active_expl = np.zeros(n_bins)
            for var in EXPLORATION_COMBINE:
                if var in behav_df.columns:
                    vals = behav_df[var].values.astype(float)
                    vals[~np.isfinite(vals)] = 0.0
                    active_expl = np.maximum(active_expl, vals)
            mask = active_expl > 0
        elif pred in behav_df.columns:
            mask = behav_df[pred].values > 0
        else:
            continue

        n_active = mask.sum()
        n_inactive = (~mask).sum()
        if n_active < 5 or n_inactive < 5:
            continue

        spikes_in = spike_counts[mask].sum()
        spikes_out = spike_counts[~mask].sum()
        rate_in = spikes_in / n_active / BIN_WIDTH
        rate_out = spikes_out / n_inactive / BIN_WIDTH
        rate_elevation = rate_in / rate_out if rate_out > 0 else np.inf
        expected_spikes = mean_count_per_bin * n_active
        excess_spikes = spikes_in - expected_spikes

        results[pred] = {
            'rate_elevation': rate_elevation,
            'excess_spikes': excess_spikes,
            'rate_when_active_hz': rate_in,
            'rate_when_inactive_hz': rate_out,
            'spikes_when_active': int(spikes_in),
            'n_active_bins': int(n_active),
            'occupancy_pct': 100 * n_active / n_bins,
        }

    return results


# =============================================================================
# Session processing
# =============================================================================

def process_session(
    session_num: int,
    session_config: dict,
    metrics_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process a single session: GLM + unique deviance + rate metrics for each unit."""
    state = session_config['state']
    phase = session_config['phase']
    sorted_path = Path(session_config['sorted'])
    behavior_path = Path(session_config['behavior'])
    session_label = f"mouse01_coordinates_1_session_{session_num}"

    print(f"\n{'='*60}")
    print(f"Session {session_num}: {session_label}")
    print(f"  State: {state} | Phase: {phase}")
    print(f"{'='*60}")

    if not sorted_path.exists():
        print(f"  [SKIP] Sorted path not found: {sorted_path}")
        return pd.DataFrame(), pd.DataFrame()
    if not behavior_path.exists():
        print(f"  [SKIP] Behavior path not found: {behavior_path}")
        return pd.DataFrame(), pd.DataFrame()

    # --- Load behavior ---
    print("  Loading behavior data...")
    behav_df = load_behavior(behavior_path)
    rec_time = behav_df['Recording time'].values
    predictors, predictor_names = prepare_predictors(behav_df)
    X = predictors.values
    n_bins = len(rec_time)
    print(f"  Behavior: {n_bins} time bins, {len(predictor_names)} predictors")
    print(f"  Predictors: {predictor_names}")

    if len(predictor_names) == 0:
        print("  [SKIP] No valid predictors after filtering")
        return pd.DataFrame(), pd.DataFrame()

    pred_categories = [PREDICTOR_CATEGORIES.get(p, 'Unknown') for p in predictor_names]

    # --- Load sorted spikes ---
    print("  Loading spike sorting data...")
    sorting = se.read_kilosort(sorted_path)
    fs = sorting.get_sampling_frequency()

    # --- Filter to good units ---
    session_metrics = metrics_df[
        (metrics_df['session'] == session_label)
        & (metrics_df['passes_qc'] == True)
    ]
    good_unit_ids = session_metrics['unit_id'].values
    good_unit_regions = dict(
        zip(session_metrics['unit_id'], session_metrics['region'])
    )

    n_lha = sum(1 for r in good_unit_regions.values() if r == 'LHA')
    n_rsp = sum(1 for r in good_unit_regions.values() if r == 'RSP')
    print(f"  Good units: {len(good_unit_ids)} (LHA={n_lha}, RSP={n_rsp})")

    if len(good_unit_ids) == 0:
        print("  [SKIP] No good units for this session")
        return pd.DataFrame(), pd.DataFrame()

    # --- Process each unit ---
    detailed_rows = []
    summary_rows = []

    for i, unit_id in enumerate(good_unit_ids):
        region = good_unit_regions[unit_id]
        print(f"  [{i+1}/{len(good_unit_ids)}] Unit {unit_id} ({region})",
              end='', flush=True)

        spike_times = sorting.get_unit_spike_train(unit_id)
        spike_counts = bin_spikes(spike_times, rec_time, fs)

        total_spikes = int(spike_counts.sum())
        if total_spikes < 10:
            print(f" -- skipped ({total_spikes} spikes)")
            continue

        overall_rate = total_spikes / n_bins / BIN_WIDTH

        # --- GLM ---
        full_results = fit_full_glm(spike_counts, X, predictor_names)
        if not full_results['converged']:
            print(" -- FAILED (full model)")
            continue

        deviance_full = full_results['deviance']
        unique_dev = compute_unique_deviance(spike_counts, X, predictor_names, deviance_full)
        cat_dev = compute_category_deviance(spike_counts, X, predictor_names, deviance_full)
        selectivity = compute_selectivity_index(unique_dev)
        cv_d2, best_alpha = cv_poisson_regression(spike_counts, X)

        # --- Rate metrics ---
        rate_metrics = compute_rate_metrics(spike_counts, behav_df, predictor_names)

        # --- GLM dominant predictors (sign-aware) ---
        coefs = full_results['coefficients']
        valid_mask = np.isfinite(unique_dev) & np.isfinite(coefs)
        excit_mask = valid_mask & (coefs > 0)
        inhib_mask = valid_mask & (coefs < 0)

        if excit_mask.any():
            excit_devs = np.where(excit_mask, unique_dev, -np.inf)
            excit_idx = np.argmax(excit_devs)
            dom_excit_pred = predictor_names[excit_idx]
            dom_excit_dev = unique_dev[excit_idx]
        else:
            dom_excit_pred = 'None'
            dom_excit_dev = np.nan

        if inhib_mask.any():
            inhib_devs = np.where(inhib_mask, unique_dev, -np.inf)
            inhib_idx = np.argmax(inhib_devs)
            dom_inhib_pred = predictor_names[inhib_idx]
            dom_inhib_dev = unique_dev[inhib_idx]
        else:
            dom_inhib_pred = 'None'
            dom_inhib_dev = np.nan

        # Excitatory category (sum of excitatory unique deviances per category)
        cat_excit_dev = {}
        for cat in CATEGORY_NAMES:
            cat_indices = [
                j for j, p in enumerate(predictor_names)
                if PREDICTOR_CATEGORIES.get(p) == cat
            ]
            excit_in_cat = [
                unique_dev[j] for j in cat_indices
                if excit_mask[j] and np.isfinite(unique_dev[j])
            ]
            cat_excit_dev[cat] = sum(excit_in_cat) if excit_in_cat else 0.0

        if any(v > 0 for v in cat_excit_dev.values()):
            dom_excit_cat = max(cat_excit_dev, key=cat_excit_dev.get)
            dom_excit_cat_dev = cat_excit_dev[dom_excit_cat]
        else:
            dom_excit_cat = 'None'
            dom_excit_cat_dev = np.nan

        # Inhibitory category
        cat_inhib_dev = {}
        for cat in CATEGORY_NAMES:
            cat_indices = [
                j for j, p in enumerate(predictor_names)
                if PREDICTOR_CATEGORIES.get(p) == cat
            ]
            inhib_in_cat = [
                unique_dev[j] for j in cat_indices
                if inhib_mask[j] and np.isfinite(unique_dev[j])
            ]
            cat_inhib_dev[cat] = sum(inhib_in_cat) if inhib_in_cat else 0.0

        if any(v > 0 for v in cat_inhib_dev.values()):
            dom_inhib_cat = max(cat_inhib_dev, key=cat_inhib_dev.get)
            dom_inhib_cat_dev = cat_inhib_dev[dom_inhib_cat]
        else:
            dom_inhib_cat = 'None'
            dom_inhib_cat_dev = np.nan

        # --- Rate elevation dominant (binary predictors, excitatory: >1, inhibitory: <1) ---
        binary_rate = {p: m for p, m in rate_metrics.items() if p not in CONTINUOUS_VARS}

        excit_rate = {p: m for p, m in binary_rate.items() if m['rate_elevation'] > 1.0}
        if excit_rate:
            dom_rate_excit_pred = max(excit_rate, key=lambda p: excit_rate[p]['rate_elevation'])
            dom_rate_excit_val = excit_rate[dom_rate_excit_pred]['rate_elevation']
        else:
            dom_rate_excit_pred = 'None'
            dom_rate_excit_val = np.nan

        inhib_rate = {p: m for p, m in binary_rate.items() if m['rate_elevation'] < 1.0}
        if inhib_rate:
            dom_rate_inhib_pred = min(inhib_rate, key=lambda p: inhib_rate[p]['rate_elevation'])
            dom_rate_inhib_val = inhib_rate[dom_rate_inhib_pred]['rate_elevation']
        else:
            dom_rate_inhib_pred = 'None'
            dom_rate_inhib_val = np.nan

        # --- Spike-weighted dominant (excitatory: excess>0, inhibitory: excess<0) ---
        excit_spk = {p: m for p, m in binary_rate.items()
                     if np.isfinite(m['excess_spikes']) and m['excess_spikes'] > 0}
        if excit_spk:
            dom_spk_excit_pred = max(excit_spk, key=lambda p: excit_spk[p]['excess_spikes'])
            dom_spk_excit_val = excit_spk[dom_spk_excit_pred]['excess_spikes']
            dom_spk_excit_rate = excit_spk[dom_spk_excit_pred]['rate_when_active_hz']
        else:
            dom_spk_excit_pred = 'None'
            dom_spk_excit_val = np.nan
            dom_spk_excit_rate = np.nan

        inhib_spk = {p: m for p, m in binary_rate.items()
                     if np.isfinite(m['excess_spikes']) and m['excess_spikes'] < 0}
        if inhib_spk:
            dom_spk_inhib_pred = min(inhib_spk, key=lambda p: inhib_spk[p]['excess_spikes'])
            dom_spk_inhib_val = inhib_spk[dom_spk_inhib_pred]['excess_spikes']
            dom_spk_inhib_rate = inhib_spk[dom_spk_inhib_pred]['rate_when_active_hz']
        else:
            dom_spk_inhib_pred = 'None'
            dom_spk_inhib_val = np.nan
            dom_spk_inhib_rate = np.nan

        # Spike-weighted category (excitatory)
        cat_spk_excit = {}
        for cat in CATEGORY_NAMES:
            cat_preds = [p for p in excit_spk if PREDICTOR_CATEGORIES.get(p) == cat]
            cat_spk_excit[cat] = sum(excit_spk[p]['excess_spikes'] for p in cat_preds)

        if any(v > 0 for v in cat_spk_excit.values()):
            dom_spk_excit_cat = max(cat_spk_excit, key=cat_spk_excit.get)
        else:
            dom_spk_excit_cat = 'None'

        status_str = (
            f" -- R2={full_results['pseudo_r2']:.4f}"
            f" | excit_glm={dom_excit_pred}"
            f" | excit_spk={dom_spk_excit_pred}"
            f" | inhib_glm={dom_inhib_pred}"
        )
        print(status_str)

        # --- Build detailed rows (one per predictor) ---
        for j, pred_name in enumerate(predictor_names):
            rm = rate_metrics.get(pred_name, {})
            detailed_rows.append({
                'session': session_label,
                'session_num': session_num,
                'state': state,
                'phase': phase,
                'unit_id': int(unit_id),
                'region': region,
                'overall_rate_hz': overall_rate,
                'total_spikes': total_spikes,
                'predictor': pred_name,
                'category': pred_categories[j],
                # GLM results
                'coefficient': full_results['coefficients'][j],
                'std_error': full_results['std_errors'][j],
                'z_score': full_results['z_scores'][j],
                'p_value': full_results['p_values'][j],
                'unique_deviance': unique_dev[j],
                'coef_sign': 'excitatory' if coefs[j] > 0 else ('inhibitory' if coefs[j] < 0 else 'zero'),
                'is_dominant_excitatory': (pred_name == dom_excit_pred),
                'is_dominant_inhibitory': (pred_name == dom_inhib_pred),
                # Rate metrics
                'rate_when_active_hz': rm.get('rate_when_active_hz', np.nan),
                'rate_when_inactive_hz': rm.get('rate_when_inactive_hz', np.nan),
                'rate_elevation': rm.get('rate_elevation', np.nan),
                'excess_spikes': rm.get('excess_spikes', np.nan),
                'spikes_when_active': rm.get('spikes_when_active', np.nan),
                'n_active_bins': rm.get('n_active_bins', np.nan),
                'occupancy_pct': rm.get('occupancy_pct', np.nan),
            })

        # --- Build summary row (one per unit) ---
        summary_rows.append({
            'session': session_label,
            'session_num': session_num,
            'state': state,
            'phase': phase,
            'unit_id': int(unit_id),
            'region': region,
            'overall_rate_hz': overall_rate,
            'total_spikes': total_spikes,
            'pseudo_r2_mcfadden': full_results['pseudo_r2'],
            'cv_d2': cv_d2,
            'aic': full_results['aic'],
            'selectivity_index': selectivity,
            # GLM unique deviance dominance
            'glm_excitatory_predictor': dom_excit_pred,
            'glm_excitatory_unique_deviance': dom_excit_dev,
            'glm_excitatory_category': dom_excit_cat,
            'glm_excitatory_category_deviance': dom_excit_cat_dev,
            'glm_inhibitory_predictor': dom_inhib_pred,
            'glm_inhibitory_unique_deviance': dom_inhib_dev,
            'glm_inhibitory_category': dom_inhib_cat,
            'glm_inhibitory_category_deviance': dom_inhib_cat_dev,
            # Rate elevation dominance
            'rate_elevation_excitatory_predictor': dom_rate_excit_pred,
            'rate_elevation_excitatory_value': dom_rate_excit_val,
            'rate_elevation_inhibitory_predictor': dom_rate_inhib_pred,
            'rate_elevation_inhibitory_value': dom_rate_inhib_val,
            # Spike-weighted dominance
            'spike_weighted_excitatory_predictor': dom_spk_excit_pred,
            'spike_weighted_excitatory_excess': dom_spk_excit_val,
            'spike_weighted_excitatory_rate_hz': dom_spk_excit_rate,
            'spike_weighted_excitatory_category': dom_spk_excit_cat,
            'spike_weighted_inhibitory_predictor': dom_spk_inhib_pred,
            'spike_weighted_inhibitory_excess': dom_spk_inhib_val,
            'spike_weighted_inhibitory_rate_hz': dom_spk_inhib_rate,
            # Category deviances
            'category_locomotion_deviance': cat_dev.get('Locomotion', np.nan),
            'category_spatial_zones_deviance': cat_dev.get('Spatial zones', np.nan),
            'category_behavioral_state_deviance': cat_dev.get('Behavioral state', np.nan),
            'n_significant_predictors': 0,  # Updated after FDR
        })

    if not detailed_rows:
        print("  No units produced results.")
        return pd.DataFrame(), pd.DataFrame()

    detailed_df = pd.DataFrame(detailed_rows)
    summary_df = pd.DataFrame(summary_rows)

    # --- FDR correction ---
    valid_p = detailed_df['p_value'].notna()
    detailed_df['p_value_fdr'] = np.nan
    detailed_df['significant'] = False

    if valid_p.sum() > 0:
        _, p_fdr, _, _ = multipletests(
            detailed_df.loc[valid_p, 'p_value'].values,
            alpha=0.05, method='fdr_bh',
        )
        detailed_df.loc[valid_p, 'p_value_fdr'] = p_fdr
        detailed_df.loc[valid_p, 'significant'] = p_fdr < 0.05

    sig_counts = (
        detailed_df[detailed_df['significant']]
        .groupby('unit_id').size().to_dict()
    )
    summary_df['n_significant_predictors'] = summary_df['unit_id'].map(
        lambda uid: sig_counts.get(uid, 0)
    )

    n_sig = detailed_df['significant'].sum()
    n_total = valid_p.sum()
    if n_total > 0:
        print(f"  FDR results: {n_sig}/{n_total} significant "
              f"({100*n_sig/n_total:.1f}%)")

    return detailed_df, summary_df


# =============================================================================
# Plotting
# =============================================================================

def plot_glm_summary(
    combined_detail: pd.DataFrame,
    combined_summary: pd.DataFrame,
    output_dir: Path,
):
    """Generate summary plots comparing all three dominance metrics."""
    sns.set_style('whitegrid')

    palette_region = {'LHA': '#E74C3C', 'RSP': '#3498DB'}
    category_palette = {
        'Locomotion': '#2ECC71',
        'Spatial zones': '#9B59B6',
        'Behavioral state': '#E67E22',
        'None': '#95A5A6',
    }

    # ==========================================
    # Figure 1: Main summary (6 panels)
    # ==========================================
    fig = plt.figure(figsize=(22, 16))
    gs = gridspec.GridSpec(3, 2, hspace=0.40, wspace=0.30,
                           left=0.07, right=0.96, top=0.94, bottom=0.06)

    # A) Three dominance metrics comparison by region — excitatory category
    ax_a = fig.add_subplot(gs[0, 0])
    metrics_list = [
        ('glm_excitatory_category', 'GLM Unique Dev.'),
        ('spike_weighted_excitatory_category', 'Spike-Weighted'),
    ]
    x_positions = np.arange(len(metrics_list))
    bar_width = 0.35

    for r_idx, region in enumerate(['LHA', 'RSP']):
        region_data = combined_summary[combined_summary['region'] == region]
        if region_data.empty:
            continue
        for m_idx, (col, label) in enumerate(metrics_list):
            cat_counts = region_data[col].value_counts()
            n_region = len(region_data)
            bottom = 0
            x = m_idx + r_idx * (len(metrics_list) + 0.5)
            for cat in CATEGORY_NAMES:
                val = 100 * cat_counts.get(cat, 0) / n_region
                ax_a.bar(x, val, bottom=bottom, width=bar_width,
                         color=category_palette.get(cat, '#95A5A6'),
                         edgecolor='white', linewidth=0.5,
                         label=cat if (r_idx == 0 and m_idx == 0) else None)
                if val > 8:
                    ax_a.text(x, bottom + val / 2, f'{val:.0f}%',
                              ha='center', va='center', fontsize=7, fontweight='bold')
                bottom += val

    xtick_positions = [0, 1, 2.5, 3.5]
    xtick_labels = ['GLM\n(LHA)', 'Spike-Wt\n(LHA)', 'GLM\n(RSP)', 'Spike-Wt\n(RSP)']
    ax_a.set_xticks(xtick_positions)
    ax_a.set_xticklabels(xtick_labels, fontsize=8)
    ax_a.set_ylabel('Percent of Neurons')
    ax_a.set_title('A) Excitatory Category: GLM vs Spike-Weighted', fontweight='bold')
    ax_a.legend(loc='upper right', fontsize=7)
    ax_a.set_ylim(0, 105)

    # B) Top excitatory predictors by spike-weighted
    ax_b = fig.add_subplot(gs[0, 1])
    pred_counts = combined_summary['spike_weighted_excitatory_predictor'].value_counts()
    top_preds = pred_counts.head(14)
    colors = [category_palette.get(PREDICTOR_CATEGORIES.get(p, 'None'), '#95A5A6')
              for p in top_preds.index]
    ax_b.barh(range(len(top_preds)), top_preds.values, color=colors,
              edgecolor='white', linewidth=0.5)
    ax_b.set_yticks(range(len(top_preds)))
    ax_b.set_yticklabels(top_preds.index, fontsize=9)
    ax_b.invert_yaxis()
    ax_b.set_xlabel('Number of Neurons')
    ax_b.set_title('B) Spike-Weighted Excitatory Dominant Predictor', fontweight='bold')

    # C) Unique deviance by predictor (box plots)
    ax_c = fig.add_subplot(gs[1, :])
    plot_data = combined_detail.copy()
    plot_data['unique_deviance_clamped'] = plot_data['unique_deviance'].clip(lower=0)
    pred_order = (
        plot_data.groupby('predictor')['unique_deviance_clamped']
        .median().sort_values(ascending=False).index.tolist()
    )
    sns.boxplot(
        data=plot_data, x='predictor', y='unique_deviance_clamped',
        hue='region', order=pred_order, palette=palette_region,
        showfliers=False, ax=ax_c, linewidth=0.8,
    )
    ax_c.set_xlabel('')
    ax_c.set_ylabel('Unique Deviance')
    ax_c.set_title('C) Per-Predictor Unique Deviance by Region', fontweight='bold')
    ax_c.tick_params(axis='x', rotation=45)
    ax_c.legend(title='Region', fontsize=8)

    # D) Selectivity index by region
    ax_d = fig.add_subplot(gs[2, 0])
    for region in ['LHA', 'RSP']:
        region_data = combined_summary[combined_summary['region'] == region]
        if region_data.empty:
            continue
        vals = region_data['selectivity_index'].dropna()
        ax_d.hist(vals, bins=30, alpha=0.6, label=f'{region} (n={len(vals)})',
                  color=palette_region[region], edgecolor='white', linewidth=0.5)
    ax_d.set_xlabel('Selectivity Index')
    ax_d.set_ylabel('Count')
    ax_d.set_title('D) Selectivity Index Distribution', fontweight='bold')
    ax_d.legend(fontsize=9)
    ax_d.set_xlim(0, 1)

    # E) Model quality
    ax_e = fig.add_subplot(gs[2, 1])
    plot_metrics = []
    for _, row in combined_summary.iterrows():
        plot_metrics.append({'region': row['region'], 'metric': 'Pseudo-R2', 'value': row['pseudo_r2_mcfadden']})
        plot_metrics.append({'region': row['region'], 'metric': 'CV D2', 'value': row['cv_d2']})
    metrics_df_plot = pd.DataFrame(plot_metrics)
    sns.boxplot(
        data=metrics_df_plot, x='metric', y='value', hue='region',
        palette=palette_region, showfliers=False, ax=ax_e, linewidth=0.8,
    )
    ax_e.set_xlabel('')
    ax_e.set_ylabel('Score')
    ax_e.set_title('E) Model Quality by Region', fontweight='bold')
    ax_e.legend(title='Region', fontsize=8)
    ax_e.axhline(0, color='grey', linestyle='--', linewidth=0.5, alpha=0.7)

    fig_path = output_dir / 'glm_unique_deviance_summary.png'
    fig.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  -> Saved figure: {fig_path}")

    # ==========================================
    # Figure 2: Fed vs. Fasted comparison
    # ==========================================
    states = combined_summary['state'].unique()
    if len(states) >= 2:
        fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
        fig2.suptitle('GLM Encoding: Fed vs. Fasted Comparison', fontweight='bold', fontsize=14)
        state_palette = {'fed': '#3498DB', 'fasted': '#E74C3C'}

        ax2a = axes2[0]
        for state in ['fed', 'fasted']:
            state_data = combined_summary[combined_summary['state'] == state]
            if state_data.empty:
                continue
            cat_counts = state_data['spike_weighted_excitatory_category'].value_counts()
            n_state = len(state_data)
            bottom = 0
            for cat in CATEGORY_NAMES:
                val = 100 * cat_counts.get(cat, 0) / n_state
                ax2a.bar(state.capitalize(), val, bottom=bottom,
                         color=category_palette.get(cat, '#95A5A6'),
                         label=cat if state == 'fed' else None,
                         edgecolor='white', linewidth=0.5)
                if val > 5:
                    ax2a.text(state.capitalize(), bottom + val / 2, f'{val:.0f}%',
                              ha='center', va='center', fontsize=9, fontweight='bold')
                bottom += val
        ax2a.set_ylabel('Percent of Neurons')
        ax2a.set_title('Excitatory Category by State (Spike-Wt)')
        ax2a.legend(loc='upper right', fontsize=7)
        ax2a.set_ylim(0, 105)

        ax2b = axes2[1]
        for state in ['fed', 'fasted']:
            state_data = combined_summary[combined_summary['state'] == state]
            if state_data.empty:
                continue
            vals = state_data['selectivity_index'].dropna()
            ax2b.hist(vals, bins=25, alpha=0.6,
                      label=f'{state.capitalize()} (n={len(vals)})',
                      color=state_palette[state], edgecolor='white', linewidth=0.5)
        ax2b.set_xlabel('Selectivity Index')
        ax2b.set_ylabel('Count')
        ax2b.set_title('Selectivity by State')
        ax2b.legend(fontsize=9)
        ax2b.set_xlim(0, 1)

        ax2c = axes2[2]
        sns.boxplot(
            data=combined_summary, x='state', y='cv_d2', hue='region',
            palette=palette_region, showfliers=False, ax=ax2c, linewidth=0.8,
            order=['fed', 'fasted'],
        )
        ax2c.set_xlabel('State')
        ax2c.set_ylabel('Cross-validated D2')
        ax2c.set_title('Model Quality by State & Region')
        ax2c.axhline(0, color='grey', linestyle='--', linewidth=0.5, alpha=0.7)
        ax2c.legend(title='Region', fontsize=8)

        fig2.tight_layout()
        fig2_path = output_dir / 'glm_fed_vs_fasted_comparison.png'
        fig2.savefig(fig2_path, dpi=200, bbox_inches='tight')
        plt.close(fig2)
        print(f"  -> Saved figure: {fig2_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    """Run GLM unique deviance analysis for all 8 sessions of Mouse01 Coordinates-1."""
    t_start = time.time()

    print("=" * 60)
    print("GLM Unique Deviance Analysis")
    print("With Rate Elevation & Spike-Weighted Dominance")
    print("Mouse01, Coordinates-1, Sessions 1-8")
    print("=" * 60)

    with open(PATHS_YAML) as f:
        paths_config = yaml.safe_load(f)
    sessions = paths_config['single_probe']['coordinates_1']['mouse01']['sessions']

    metrics_df = pd.read_csv(METRICS_CSV)
    total_qc = metrics_df['passes_qc'].sum()
    print(f"Loaded {len(metrics_df)} unit records, {total_qc} pass QC")

    all_detailed = []
    all_summary = []

    for session_num in range(1, 9):
        session_key = f"session_{session_num}"
        session_config = sessions[session_key]

        if session_config.get('behavior') is None:
            print(f"\n[SKIP] Session {session_num}: no behavior data")
            continue

        detailed_df, summary_df = process_session(
            session_num, session_config, metrics_df
        )

        if not detailed_df.empty:
            state = session_config['state']
            phase = session_config['phase']

            detail_path = DATA_DIR / f"glm_detailed_session{session_num}_{state}_{phase}.csv"
            detailed_df.to_csv(detail_path, index=False)
            print(f"  -> Saved: {detail_path}")

            summary_path = DATA_DIR / f"glm_summary_session{session_num}_{state}_{phase}.csv"
            summary_df.to_csv(summary_path, index=False)
            print(f"  -> Saved: {summary_path}")

            all_detailed.append(detailed_df)
            all_summary.append(summary_df)

    # =================================================================
    # Population summary
    # =================================================================
    print(f"\n{'='*60}")
    print("POPULATION SUMMARY")
    print(f"{'='*60}")

    if not all_summary:
        print("No sessions were processed successfully.")
        return

    combined_detail = pd.concat(all_detailed, ignore_index=True)
    combined_summary = pd.concat(all_summary, ignore_index=True)

    combined_detail.to_csv(DATA_DIR / "glm_detailed_all_sessions.csv", index=False)
    combined_summary.to_csv(DATA_DIR / "glm_summary_all_sessions.csv", index=False)
    print(f"  -> Saved combined CSVs")

    n_units = len(combined_summary)
    n_sessions = combined_summary['session_num'].nunique()
    print(f"\nSessions processed: {n_sessions}")
    print(f"Total units: {n_units}")

    # --- Three dominance metrics comparison ---
    for metric_name, excit_col, inhib_col in [
        ('GLM UNIQUE DEVIANCE', 'glm_excitatory_predictor', 'glm_inhibitory_predictor'),
        ('RATE ELEVATION', 'rate_elevation_excitatory_predictor', 'rate_elevation_inhibitory_predictor'),
        ('SPIKE-WEIGHTED', 'spike_weighted_excitatory_predictor', 'spike_weighted_inhibitory_predictor'),
    ]:
        print(f"\n--- {metric_name} ---")
        print(f"  Top EXCITATORY:")
        ec = combined_summary[excit_col].value_counts().head(8)
        for pred, count in ec.items():
            print(f"    {pred:25s}  {count:4d}  ({100*count/n_units:.1f}%)")
        print(f"  Top INHIBITORY:")
        ic = combined_summary[inhib_col].value_counts().head(8)
        for pred, count in ic.items():
            print(f"    {pred:25s}  {count:4d}  ({100*count/n_units:.1f}%)")

    # --- By region ---
    for region in ['LHA', 'RSP']:
        r_df = combined_summary[combined_summary['region'] == region]
        if r_df.empty:
            continue
        print(f"\n  {region} ({len(r_df)} units):")
        for label, col in [
            ('GLM excit. category', 'glm_excitatory_category'),
            ('Spike-wt excit. category', 'spike_weighted_excitatory_category'),
        ]:
            print(f"    {label}:")
            cc = r_df[col].value_counts()
            for cat, count in cc.items():
                print(f"      {cat:25s}  {count:3d}  ({100*count/len(r_df):.1f}%)")
        print(f"    Median selectivity: {r_df['selectivity_index'].median():.3f}")
        print(f"    Median pseudo-R2:   {r_df['pseudo_r2_mcfadden'].median():.4f}")

    # --- Selectivity ---
    print(f"\nSelectivity index:")
    print(f"  Median: {combined_summary['selectivity_index'].median():.3f}")
    print(f"  Mean:   {combined_summary['selectivity_index'].mean():.3f}")

    # --- FDR significance ---
    n_total_tests = combined_detail['p_value'].notna().sum()
    n_sig = combined_detail['significant'].sum()
    if n_total_tests > 0:
        print(f"\nSignificance (FDR < 0.05): {n_sig}/{n_total_tests} "
              f"({100*n_sig/n_total_tests:.1f}%)")

    # --- Verification ---
    neg_dev = (combined_detail['unique_deviance'] < -0.01).sum()
    if neg_dev > 0:
        print(f"\n  [WARNING] {neg_dev} entries with negative unique deviance")
    else:
        print(f"\n  [OK] No materially negative unique deviances")

    # --- Unit 395 check ---
    u395 = combined_summary[(combined_summary['unit_id'] == 395) & (combined_summary['session_num'] == 1)]
    if not u395.empty:
        r = u395.iloc[0]
        print(f"\n--- Unit 395 (session 1) verification ---")
        print(f"  GLM excitatory:     {r['glm_excitatory_predictor']} (udev={r['glm_excitatory_unique_deviance']:.1f})")
        print(f"  GLM inhibitory:     {r['glm_inhibitory_predictor']} (udev={r['glm_inhibitory_unique_deviance']:.1f})")
        print(f"  Rate elev. excit.:  {r['rate_elevation_excitatory_predictor']} ({r['rate_elevation_excitatory_value']:.2f}x)")
        print(f"  Rate elev. inhib.:  {r['rate_elevation_inhibitory_predictor']} ({r['rate_elevation_inhibitory_value']:.2f}x)")
        print(f"  Spike-wt excit.:    {r['spike_weighted_excitatory_predictor']} (excess={r['spike_weighted_excitatory_excess']:.0f}, rate={r['spike_weighted_excitatory_rate_hz']:.2f} Hz)")
        print(f"  Spike-wt inhib.:    {r['spike_weighted_inhibitory_predictor']} (excess={r['spike_weighted_inhibitory_excess']:.0f})")

    # --- Plots ---
    print(f"\n{'='*60}")
    print("GENERATING PLOTS")
    print(f"{'='*60}")
    plot_glm_summary(combined_detail, combined_summary, OUTPUT_DIR)

    elapsed = time.time() - t_start
    print(f"\nTotal runtime: {elapsed/60:.1f} minutes")
    print(f"Done. Output saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
