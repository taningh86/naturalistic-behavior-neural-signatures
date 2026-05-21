"""
GLM Encoding Model: Single Neuron-Behavior Relationships

Fits Poisson GLM encoding models to identify which behavioral variables
predict each neuron's firing rate. Processes all 8 sessions of Mouse01
Coordinates-1 (fed and fasted, LHA + RSP).

Outputs per-session CSVs with coefficients, p-values (FDR-corrected),
and model quality metrics.

Usage:
    conda activate si_env
    python glm_neuron_behavior_analysis.py
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import spikeinterface.extractors as se
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
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

# Predictor variable names (must match behavior CSV column names exactly)
CONTINUOUS_VARS = ['Velocity', 'Distance moved', 'Meander']
ZONE_VARS = [
    'Home', 'Foraging arena', 'Transition zone',
    'Pot-1 zone', 'Pot-2 zone', 'Pot-3 zone', 'Pot-4 zone',
]
DISTANCE_VARS = [
    'Distance to Pot-2', 'Distance to Pot-4',
    'Distance to Home', 'Distance to Foraging arena',
]
BEHAVIORAL_STATE_VARS = ['Feeding', 'Grooming']
EXPLORATION_COMBINE = [
    'Longer exploration at home',
    'Quick and hasty exploration at home',
    'Hesitant exploration',
]

# Cross-validation settings
ALPHA_GRID = np.logspace(-4, 1, 20)
N_CV_FOLDS = 5


# =============================================================================
# Data loading
# =============================================================================

def load_behavior(behavior_path: Path) -> pd.DataFrame:
    """Load transposed behavior CSV (variables as rows) and return
    a DataFrame with time bins as rows and variables as columns.

    Parameters
    ----------
    behavior_path : Path
        Path to the 100ms-binned behavior CSV.

    Returns
    -------
    pd.DataFrame
        Shape (n_time_bins, n_variables). Column names are the
        behavioral variable names.
    """
    behav_raw = pd.read_csv(behavior_path, index_col=0)
    behav_df = behav_raw.T.reset_index(drop=True)

    # Strip whitespace from column names
    behav_df.columns = behav_df.columns.str.strip()

    # Ensure numeric
    for col in behav_df.columns:
        behav_df[col] = pd.to_numeric(behav_df[col], errors='coerce')

    return behav_df


def prepare_predictors(
    behav_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Extract and preprocess predictor variables from behavior data.

    - Continuous variables are z-scored.
    - Zone occupancy variables are binarized (>0 → 1).
    - Distance variables are z-scored.
    - Behavioral states are binarized.
    - Exploration subtypes are combined into a single 'Active_exploration'.
    - Predictors with zero variance are dropped.

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
        # Replace inf/nan with 0 before scaling
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
        # Only include if at least 0.5% occupancy
        occupancy_frac = binary_vals.mean()
        if occupancy_frac < 0.005:
            print(f"    [WARN] Very sparse zone variable: {var} "
                  f"({100*occupancy_frac:.2f}% occupancy) — skipping")
            continue
        predictors[var] = binary_vals
        predictor_names.append(var)

    # --- Distance variables (z-scored) ---
    for var in DISTANCE_VARS:
        if var not in behav_df.columns:
            print(f"    [WARN] Missing distance variable: {var}")
            continue
        vals = behav_df[var].values.astype(float).copy()
        vals[~np.isfinite(vals)] = 0.0
        std = np.std(vals)
        if std == 0:
            print(f"    [WARN] Zero-variance distance variable: {var}")
            continue
        scaler = StandardScaler()
        predictors[var] = scaler.fit_transform(vals.reshape(-1, 1)).ravel()
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
                  f"({100*occupancy_frac:.2f}% occupancy) — skipping")
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
              f"({100*occupancy_frac:.2f}%) — skipping")

    # Fill any remaining NaN
    predictors = predictors.fillna(0)

    return predictors, predictor_names


def bin_spikes(
    spike_times_samples: np.ndarray,
    rec_time: np.ndarray,
    fs: float,
) -> np.ndarray:
    """Bin spike times into 100ms bins aligned to behavior timestamps.

    Parameters
    ----------
    spike_times_samples : np.ndarray
        Spike times in samples (at `fs` Hz).
    rec_time : np.ndarray
        Recording time values from behavior (seconds), one per bin.
    fs : float
        Sampling rate.

    Returns
    -------
    np.ndarray
        Spike count per behavior time bin.
    """
    spike_times_sec = spike_times_samples / fs
    dt = np.median(np.diff(rec_time))
    bin_edges = np.concatenate([rec_time - dt / 2, [rec_time[-1] + dt / 2]])
    counts, _ = np.histogram(spike_times_sec, bins=bin_edges)
    return counts


# =============================================================================
# GLM fitting
# =============================================================================

def fit_statsmodels_glm(
    y: np.ndarray,
    X: np.ndarray,
    predictor_names: list[str],
) -> dict:
    """Fit unregularized Poisson GLM with statsmodels for inference.

    Returns coefficients, standard errors, z-scores, p-values (Wald test),
    McFadden pseudo-R², AIC, BIC, and deviance.
    """
    X_const = sm.add_constant(X, has_constant='add')
    n_pred = len(predictor_names)

    try:
        model = sm.GLM(
            y, X_const,
            family=sm.families.Poisson(link=sm.families.links.Log()),
        )
        result = model.fit(maxiter=100, method='IRLS')

        # Skip intercept (index 0)
        coefs = result.params[1:]
        std_errors = result.bse[1:]
        z_scores = coefs / std_errors
        p_values = result.pvalues[1:]

        # McFadden pseudo-R²
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
            'bic': result.bic_llf,
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
            'bic': np.nan,
            'deviance': np.nan,
            'converged': False,
        }


def cv_poisson_regression(y: np.ndarray, X: np.ndarray) -> tuple[float, float]:
    """Cross-validated D² (deviance explained) using sklearn PoissonRegressor.

    Selects the best L2 regularization alpha via 5-fold TimeSeriesSplit,
    then returns the mean CV D² and the best alpha.

    Returns
    -------
    best_cv_d2 : float
        Mean D² across folds at best alpha.
    best_alpha : float
        The alpha that achieved the best CV score.
    """
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
                score = model.score(X_test, y_test)  # D² deviance explained
                fold_scores.append(score)
            except Exception:
                fold_scores.append(-np.inf)

        mean_score = np.mean(fold_scores)
        if mean_score > best_score:
            best_score = mean_score
            best_alpha = alpha

    return best_score, best_alpha


# =============================================================================
# Session processing
# =============================================================================

def process_session(
    session_num: int,
    session_config: dict,
    metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    """Process a single session: load data, fit GLM for each good unit.

    Returns a DataFrame with one row per (unit × predictor).
    """
    state = session_config['state']
    phase = session_config['phase']
    sorted_path = Path(session_config['sorted'])
    behavior_path = Path(session_config['behavior'])
    session_label = f"mouse01_coordinates_1_session_{session_num}"

    print(f"\n{'='*60}")
    print(f"Session {session_num}: {session_label}")
    print(f"  State: {state} | Phase: {phase}")
    print(f"{'='*60}")

    # --- Verify paths ---
    if not sorted_path.exists():
        print(f"  [SKIP] Sorted path not found: {sorted_path}")
        return pd.DataFrame()
    if not behavior_path.exists():
        print(f"  [SKIP] Behavior path not found: {behavior_path}")
        return pd.DataFrame()

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
        return pd.DataFrame()

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
        return pd.DataFrame()

    # --- Fit GLM for each unit ---
    results_rows = []

    for i, unit_id in enumerate(good_unit_ids):
        region = good_unit_regions[unit_id]
        print(f"  [{i+1}/{len(good_unit_ids)}] Unit {unit_id} ({region})",
              end='')

        # Bin spikes
        spike_times = sorting.get_unit_spike_train(unit_id)
        spike_counts = bin_spikes(spike_times, rec_time, fs)

        total_spikes = int(spike_counts.sum())
        if total_spikes < 10:
            print(f" — skipped ({total_spikes} spikes)")
            continue

        # 1) Statsmodels GLM for p-values
        glm_results = fit_statsmodels_glm(spike_counts, X, predictor_names)

        # 2) Cross-validated D² with sklearn
        cv_d2, best_alpha = cv_poisson_regression(spike_counts, X)

        status = "OK" if glm_results['converged'] else "FAIL"
        print(f" — {status} | pseudo-R²={glm_results['pseudo_r2']:.4f}"
              f" | CV-D²={cv_d2:.4f}")

        # Collect per-predictor rows
        for j, pred_name in enumerate(predictor_names):
            results_rows.append({
                'session': session_label,
                'session_num': session_num,
                'state': state,
                'phase': phase,
                'unit_id': int(unit_id),
                'region': region,
                'predictor': pred_name,
                'coefficient': glm_results['coefficients'][j],
                'std_error': glm_results['std_errors'][j],
                'z_score': glm_results['z_scores'][j],
                'p_value': glm_results['p_values'][j],
                'pseudo_r2_mcfadden': glm_results['pseudo_r2'],
                'cv_d2': cv_d2,
                'cv_best_alpha': best_alpha,
                'aic': glm_results['aic'],
                'bic': glm_results['bic'],
                'deviance': glm_results['deviance'],
                'converged': glm_results['converged'],
                'total_spikes_in_bins': total_spikes,
            })

    if not results_rows:
        print("  No units produced results.")
        return pd.DataFrame()

    results_df = pd.DataFrame(results_rows)

    # --- FDR correction (Benjamini-Hochberg) across all p-values ---
    valid_p = results_df['p_value'].notna()
    results_df['p_value_fdr'] = np.nan
    results_df['significant'] = False

    if valid_p.sum() > 0:
        _, p_fdr, _, _ = multipletests(
            results_df.loc[valid_p, 'p_value'].values,
            alpha=0.05,
            method='fdr_bh',
        )
        results_df.loc[valid_p, 'p_value_fdr'] = p_fdr
        results_df.loc[valid_p, 'significant'] = p_fdr < 0.05

    n_sig = results_df['significant'].sum()
    n_total = valid_p.sum()
    print(f"  FDR results: {n_sig}/{n_total} significant "
          f"({100*n_sig/n_total:.1f}%)" if n_total > 0 else "")

    return results_df


# =============================================================================
# Main
# =============================================================================

def main():
    """Run GLM encoding model for all 8 sessions of Mouse01 Coordinates-1."""
    print("=" * 60)
    print("GLM Encoding Model: Single Neuron–Behavior Relationships")
    print("Mouse01, Coordinates-1, Sessions 1–8")
    print("=" * 60)

    # Load configuration
    with open(PATHS_YAML) as f:
        paths_config = yaml.safe_load(f)

    sessions = (
        paths_config['single_probe']['coordinates_1']['mouse01']['sessions']
    )

    # Load unit quality metrics
    metrics_df = pd.read_csv(METRICS_CSV)
    total_qc = metrics_df['passes_qc'].sum()
    print(f"Loaded {len(metrics_df)} unit records, {total_qc} pass QC")

    # Process each session
    all_results = []
    for session_num in range(1, 9):
        session_key = f"session_{session_num}"
        session_config = sessions[session_key]

        if session_config.get('behavior') is None:
            print(f"\n[SKIP] Session {session_num}: no behavior data")
            continue

        results_df = process_session(session_num, session_config, metrics_df)

        if not results_df.empty:
            state = session_config['state']
            phase = session_config['phase']
            out_path = (
                DATA_DIR
                / f"glm_neuron_behavior_session{session_num}_{state}_{phase}.csv"
            )
            results_df.to_csv(out_path, index=False)
            print(f"  -> Saved: {out_path}")
            all_results.append(results_df)

    # --- Summary ---
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    if not all_results:
        print("No sessions were processed successfully.")
        return

    combined = pd.concat(all_results, ignore_index=True)
    n_units = combined.groupby(['session', 'unit_id']).ngroups
    n_total_tests = combined['p_value'].notna().sum()
    n_sig = combined['significant'].sum()

    unit_pr2 = (
        combined
        .groupby(['session', 'unit_id'])['pseudo_r2_mcfadden']
        .first()
    )

    print(f"Sessions processed:        {len(all_results)}")
    print(f"Total units fitted:        {n_units}")
    print(f"Total predictor tests:     {n_total_tests}")
    print(f"Significant (FDR < 0.05):  {n_sig} "
          f"({100*n_sig/n_total_tests:.1f}%)" if n_total_tests > 0 else "")
    print(f"Pseudo-R² (McFadden):")
    print(f"  Median: {unit_pr2.median():.4f}")
    print(f"  Mean:   {unit_pr2.mean():.4f}")
    print(f"  Range:  [{unit_pr2.min():.4f}, {unit_pr2.max():.4f}]")

    # Per-region summary
    for region in ['LHA', 'RSP']:
        region_df = combined[combined['region'] == region]
        if region_df.empty:
            continue
        r_units = region_df.groupby(['session', 'unit_id']).ngroups
        r_sig = region_df['significant'].sum()
        r_total = region_df['p_value'].notna().sum()
        r_pr2 = (
            region_df
            .groupby(['session', 'unit_id'])['pseudo_r2_mcfadden']
            .first()
            .median()
        )
        print(f"\n  {region}: {r_units} units, "
              f"{r_sig}/{r_total} significant, "
              f"median pseudo-R²={r_pr2:.4f}")

    # Per-predictor summary (most frequently significant)
    print("\nTop predictors by significance rate:")
    pred_stats = (
        combined[combined['p_value'].notna()]
        .groupby('predictor')
        .agg(
            n_tests=('significant', 'count'),
            n_sig=('significant', 'sum'),
        )
    )
    pred_stats['sig_rate'] = pred_stats['n_sig'] / pred_stats['n_tests']
    pred_stats = pred_stats.sort_values('sig_rate', ascending=False)
    for pred, row in pred_stats.iterrows():
        print(f"  {pred:30s}  {row['n_sig']:3.0f}/{row['n_tests']:3.0f} "
              f"({100*row['sig_rate']:5.1f}%)")

    print(f"\nDone. Output CSVs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
