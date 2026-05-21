"""
Dual-Probe: Manifold Geometry Analysis -- ACA & LHA
=====================================================
Layer 1 (structure-first) + Layer 2 (behavioral mapping)
Pilot: S3, ACA and LHA separately.

Run order:
  1. Preprocessing (50ms bins, Gaussian smooth, z-score)
  2. Layer 1a: Dimensionality suite (PR, Two-NN, CorrelDim, Isomap)
  3. Layer 1d: Visualization (PCA 2D/3D, UMAP 2D/3D)
  4. Layer 2: Behavioral mapping (decodability, MI, silhouette)
  CHECKPOINT -- present results before continuing to Layer 1b/1c.
"""

import yaml
import json
import sys
import time as timer
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import shortest_path
from scipy.stats import linregress
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, balanced_accuracy_score, silhouette_score
from sklearn.metrics import mutual_info_score
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# Reuse data loaders from avalanche pipeline
from dp_avalanche_criticality import (
    get_good_units_p0, get_good_units_p1_lha,
    load_spike_times_for_region, FS,
)
import spikeinterface.extractors as se

# ============================================================================
# CONSTANTS
# ============================================================================

with open("paths.yaml") as f:
    cfg = yaml.safe_load(f)

sessions_cfg = cfg["double_probe"]["coordinates_1"]["mouse01"]["sessions"]

PILOT_SESSION = 3
BIN_MS = 50.0          # 50ms bins
SMOOTH_SIGMA = 1.0     # Gaussian sigma in bins
BLOCK_SIZE = 200       # Block bootstrap block size in bins (~10s at 50ms)
N_BOOT = 100           # Bootstrap iterations for CIs
N_BOOT_FAST = 200      # For fast metrics (PR)
CORRDIM_SUBSAMPLE = 3000  # Points for correlation dimension
ISOMAP_SUBSAMPLE = 3000   # Points for Isomap
ISOMAP_K_VALUES = [10, 15, 20, 30]  # k-NN for Isomap sensitivity
ISOMAP_K_PRIMARY = 15
N_KMEANS = 20          # Clusters for MI computation
N_CV_FOLDS = 5         # CV folds for decodability
N_SUBSAMPLE_DRAWS = 10 # Draws for ACA->111 unit subsampling

outdir = Path("data/manifold")
outdir.mkdir(parents=True, exist_ok=True)
figdir = Path("figures/manifold")
figdir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

def load_neural_data(session_num, region):
    """Load spike times for a region, return spike_dict and session_duration."""
    sval = sessions_cfg[f"session_{session_num}"]
    if region == 'ACA':
        sorted_path = Path(sval['probe_0_aca']['sorted'])
        unit_ids = get_good_units_p0(sorted_path)
    elif region == 'LHA':
        sorted_path = Path(sval['probe_1_lha_rsp']['sorted'])
        unit_ids = get_good_units_p1_lha(sorted_path)
    else:
        raise ValueError(f"Unknown region: {region}")

    sorting = se.read_kilosort(sorted_path)
    avail = set(sorting.get_unit_ids())
    unit_ids = np.array([u for u in unit_ids if u in avail])
    spike_dict = load_spike_times_for_region(sorting, unit_ids)
    all_spikes = np.concatenate(list(spike_dict.values()))
    session_duration = float(all_spikes.max()) + 1.0
    return spike_dict, session_duration, unit_ids


def preprocess_neural(spike_dict, session_duration, bin_ms=BIN_MS):
    """Bin at bin_ms, Gaussian smooth (sigma=1 bin), z-score per unit.
    Returns: matrix (T x N), bin_centers (T,), unit_ids list."""
    dt = bin_ms / 1000.0
    n_bins = int(session_duration / dt)
    bin_edges = np.arange(0, n_bins + 1) * dt
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    unit_ids = sorted(spike_dict.keys())
    n_units = len(unit_ids)

    matrix = np.zeros((n_bins, n_units))
    for j, uid in enumerate(unit_ids):
        counts, _ = np.histogram(spike_dict[uid], bins=bin_edges)
        matrix[:, j] = counts

    # Gaussian smooth (sigma = 1 bin)
    for j in range(n_units):
        matrix[:, j] = gaussian_filter1d(matrix[:, j], sigma=SMOOTH_SIGMA)

    # Z-score per unit
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    stds[stds == 0] = 1.0
    matrix = (matrix - means) / stds

    return matrix, bin_centers, unit_ids


def circular_shift_shuffle(spike_dict, session_duration):
    """Circular shift each neuron independently by random offset.
    Preserves per-neuron rate + autocorrelation; destroys population coordination."""
    shifted = {}
    for uid, times in spike_dict.items():
        offset = np.random.uniform(0, session_duration)
        shifted[uid] = np.sort((times + offset) % session_duration)
    return shifted


def load_behavioral_data(session_num):
    """Load EthoVision behavioral xlsx for a session."""
    sval = sessions_cfg[f"session_{session_num}"]
    behav_path = sval.get('behavior', None)
    if behav_path is None:
        raise ValueError(f"No behavior path for session {session_num}")

    raw = pd.read_excel(behav_path, header=None)
    col_names = list(raw.iloc[34].values)
    data = raw.iloc[36:].copy()
    data.columns = col_names
    data = data.reset_index(drop=True)

    # Convert all columns to numeric where possible
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    return data


def align_behavior_to_bins(behav_df, bin_centers):
    """Align behavioral variables to neural bin centers via nearest-sample lookup.
    behav_df has 'Trial time' column at 40ms sampling.
    bin_centers at 50ms. For each bin, find nearest behavioral sample."""
    behav_times = behav_df['Trial time'].values.astype(float)
    aligned = {}

    # For each bin center, find nearest behavioral time index
    indices = np.searchsorted(behav_times, bin_centers, side='left')
    indices = np.clip(indices, 0, len(behav_times) - 1)

    # Also check the previous index for closer match
    prev_indices = np.clip(indices - 1, 0, len(behav_times) - 1)
    d_curr = np.abs(behav_times[indices] - bin_centers)
    d_prev = np.abs(behav_times[prev_indices] - bin_centers)
    use_prev = d_prev < d_curr
    indices[use_prev] = prev_indices[use_prev]

    aligned['_indices'] = indices
    aligned['_behav_times'] = behav_times
    return indices


def extract_behavioral_variables(behav_df, indices):
    """Extract and organize behavioral variables at aligned time points."""
    variables = {}

    # --- Continuous variables ---
    # Position
    variables['x_position'] = {
        'values': behav_df['X center'].values[indices].astype(float),
        'type': 'continuous', 'unit': 'cm'
    }
    variables['y_position'] = {
        'values': behav_df['Y center'].values[indices].astype(float),
        'type': 'continuous', 'unit': 'cm'
    }

    # Velocity
    variables['velocity'] = {
        'values': behav_df['Velocity(Center-point)'].values[indices].astype(float),
        'type': 'continuous', 'unit': 'cm/s'
    }

    # Heading (convert to sin/cos to handle wrap-around)
    direction = behav_df['Direction'].values[indices].astype(float)
    direction_rad = np.deg2rad(direction)
    variables['heading_sin'] = {
        'values': np.sin(direction_rad),
        'type': 'continuous', 'unit': 'sin(deg)'
    }
    variables['heading_cos'] = {
        'values': np.cos(direction_rad),
        'type': 'continuous', 'unit': 'cos(deg)'
    }

    # Distance to nearest pot
    pot_dist_cols = [c for c in behav_df.columns if 'Distance to zone(Pot-' in str(c)]
    if pot_dist_cols:
        pot_dists = np.stack([
            behav_df[c].values[indices].astype(float) for c in pot_dist_cols
        ], axis=1)
        variables['dist_nearest_pot'] = {
            'values': np.nanmin(pot_dists, axis=1),
            'type': 'continuous', 'unit': 'cm'
        }

    # Distance to home
    home_dist_col = [c for c in behav_df.columns if 'Distance to zone(Home' in str(c)]
    if home_dist_col:
        variables['dist_home'] = {
            'values': behav_df[home_dist_col[0]].values[indices].astype(float),
            'type': 'continuous', 'unit': 'cm'
        }

    # Time in session
    variables['time_in_session'] = {
        'values': behav_df['Trial time'].values[indices].astype(float),
        'type': 'continuous', 'unit': 's'
    }

    # Time since last pot visit
    # Pot zone columns: "Zone(Pot-1 / ...)" but NOT "Zone(Pot-1 zone / ...)" or "Distance to zone(Pot-..."
    pot_zone_cols = [c for c in behav_df.columns
                     if 'Zone(Pot-' in str(c) and ' zone' not in c
                     and 'Distance' not in str(c)]
    if pot_zone_cols:
        # At any pot at each time step (full behavioral resolution)
        all_times = behav_df['Trial time'].values.astype(float)
        at_pot = np.zeros(len(behav_df), dtype=bool)
        for c in pot_zone_cols:
            at_pot |= (behav_df[c].values == 1)
        # Compute time since last pot visit
        last_pot_time = np.full(len(behav_df), np.nan)
        last_t = np.nan
        for i in range(len(behav_df)):
            if at_pot[i]:
                last_t = all_times[i]
            last_pot_time[i] = last_t
        time_since = all_times - last_pot_time
        time_since[np.isnan(last_pot_time)] = np.nan
        variables['time_since_pot'] = {
            'values': time_since[indices],
            'type': 'continuous', 'unit': 's'
        }

    # --- Categorical variables ---
    # Compartment: Home / Ladder / AtPot / Arena
    home_col = [c for c in behav_df.columns if 'Zone(Home' in str(c) and 'corner' not in c
                and 'Distance' not in str(c)]
    ladder_col = [c for c in behav_df.columns if 'Zone(ladder' in str(c)
                  and 'Distance' not in str(c)]
    pot_zone_cols_cat = pot_zone_cols  # reuse from above

    compartment = np.full(len(indices), 'Arena', dtype=object)
    if home_col:
        home_vals = behav_df[home_col[0]].values[indices]
        compartment[home_vals == 1] = 'Home'
    if ladder_col:
        ladder_vals = behav_df[ladder_col[0]].values[indices]
        compartment[ladder_vals == 1] = 'Ladder'
    if pot_zone_cols_cat:
        at_pot_bins = np.zeros(len(indices), dtype=bool)
        for c in pot_zone_cols_cat:
            at_pot_bins |= (behav_df[c].values[indices] == 1)
        compartment[at_pot_bins] = 'AtPot'

    variables['compartment'] = {
        'values': compartment,
        'type': 'categorical',
        'classes': ['Home', 'Ladder', 'Arena', 'AtPot']
    }

    return variables


# ============================================================================
# LAYER 1a: DIMENSIONALITY ESTIMATES
# ============================================================================

def participation_ratio(X):
    """PR = (sum lambda_i)^2 / sum(lambda_i^2) from PCA eigenvalues.
    X: (T, N) matrix."""
    pca = PCA(n_components=min(X.shape))
    pca.fit(X)
    evals = pca.explained_variance_
    pr = (evals.sum()) ** 2 / (evals ** 2).sum()
    return pr, evals


def two_nn_dimension(X):
    """Two-NN intrinsic dimensionality estimator (Facco et al. 2017).
    For each point, compute mu = r2/r1 (ratio of 2nd to 1st NN distance).
    Fit: log(1 - i/N) = -d * log(mu_(i))."""
    nn = NearestNeighbors(n_neighbors=3, algorithm='auto')
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    r1 = distances[:, 1]  # nearest neighbor
    r2 = distances[:, 2]  # second nearest

    # Filter zero distances
    valid = r1 > 0
    mu = r2[valid] / r1[valid]
    mu = np.sort(mu)
    N = len(mu)

    # Empirical survival function
    F = np.arange(1, N + 1) / N

    # Exclude extremes (top 10%, bottom 1%)
    mask = (F > 0.01) & (F < 0.90)
    if mask.sum() < 10:
        return np.nan

    log_mu = np.log(mu[mask])
    log_surv = np.log(1.0 - F[mask])

    slope, _, r_value, _, _ = linregress(log_mu, log_surv)
    d = -slope
    return d


def correlation_dimension(X, n_sub=CORRDIM_SUBSAMPLE):
    """Grassberger-Procaccia correlation dimension.
    Subsample to n_sub points, compute pairwise distances,
    fit slope of log(C(r)) vs log(r) in scaling region."""
    if len(X) > n_sub:
        idx = np.random.choice(len(X), n_sub, replace=False)
        X_sub = X[idx]
    else:
        X_sub = X

    dists = pdist(X_sub)
    n_pairs = len(dists)

    # Log-spaced r values spanning 5th to 95th percentile of distances
    r_lo = np.percentile(dists, 5)
    r_hi = np.percentile(dists, 95)
    if r_lo <= 0:
        r_lo = np.min(dists[dists > 0])
    r_vals = np.logspace(np.log10(r_lo), np.log10(r_hi), 40)

    C_r = np.array([np.sum(dists < r) / n_pairs for r in r_vals])

    # Filter valid range: C(r) in [0.02, 0.80]
    valid = (C_r > 0.02) & (C_r < 0.80)
    if valid.sum() < 5:
        return np.nan, r_vals, C_r

    log_r = np.log10(r_vals[valid])
    log_C = np.log10(C_r[valid])

    slope, _, r_value, _, _ = linregress(log_r, log_C)
    return slope, r_vals, C_r


def isomap_dimension(X, k=ISOMAP_K_PRIMARY, n_sub=ISOMAP_SUBSAMPLE):
    """Isomap-based intrinsic dimensionality.
    Build k-NN graph, compute geodesic distances, do classical MDS,
    count eigenvalues above noise floor (>5% of max eigenvalue)."""
    if len(X) > n_sub:
        idx = np.random.choice(len(X), n_sub, replace=False)
        X_sub = X[idx]
    else:
        X_sub = X

    # Build k-NN graph
    nn = NearestNeighbors(n_neighbors=k, algorithm='auto')
    nn.fit(X_sub)
    knn_graph = nn.kneighbors_graph(mode='distance')

    # Make symmetric
    knn_graph = knn_graph.maximum(knn_graph.T)

    # Shortest paths (geodesic distances)
    geo_dist = shortest_path(knn_graph, method='D', directed=False)

    # Check for disconnected components
    if np.any(np.isinf(geo_dist)):
        # Use largest connected component
        connected = np.all(np.isfinite(geo_dist), axis=1)
        if connected.sum() < 100:
            return np.nan
        geo_dist = geo_dist[np.ix_(connected, connected)]

    n = len(geo_dist)

    # Classical MDS: double-center squared distance matrix
    D2 = geo_dist ** 2
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ D2 @ H

    # Eigendecomposition (only need top eigenvalues)
    eigenvalues = np.linalg.eigvalsh(B)
    eigenvalues = np.sort(eigenvalues)[::-1]  # descending

    # Count positive eigenvalues above 5% of max
    pos_evals = eigenvalues[eigenvalues > 0]
    if len(pos_evals) == 0:
        return np.nan
    threshold = 0.05 * pos_evals[0]
    dim = int(np.sum(pos_evals > threshold))

    return dim


def block_bootstrap_resample(X, block_size=BLOCK_SIZE):
    """Resample time series by blocks with replacement.
    Returns a new matrix with same shape as X."""
    T, N = X.shape
    n_blocks = int(np.ceil(T / block_size))
    block_starts = np.arange(0, T, block_size)

    # Sample blocks with replacement
    chosen = np.random.choice(len(block_starts), n_blocks, replace=True)
    rows = []
    for b in chosen:
        start = block_starts[b]
        end = min(start + block_size, T)
        rows.append(X[start:end])
    resampled = np.concatenate(rows, axis=0)[:T]  # trim to original length
    return resampled


def run_dimensionality_suite(X, n_boot=N_BOOT, label='data'):
    """Run all 4 dimensionality estimates with block bootstrap CIs."""
    results = {}
    T, N = X.shape
    print(f"    Matrix: {T} time bins x {N} units")

    # 1. Participation ratio
    t0 = timer.time()
    pr, evals = participation_ratio(X)
    boot_pr = []
    for _ in range(n_boot):
        X_b = block_bootstrap_resample(X)
        pr_b, _ = participation_ratio(X_b)
        boot_pr.append(pr_b)
    pr_ci = [float(np.percentile(boot_pr, 2.5)), float(np.percentile(boot_pr, 97.5))]
    results['PR'] = {'value': float(pr), 'ci': pr_ci}
    print(f"    PR = {pr:.1f} [{pr_ci[0]:.1f}, {pr_ci[1]:.1f}] ({timer.time()-t0:.1f}s)")

    # 2. Two-NN
    # Note: block bootstrap creates duplicate rows; add tiny jitter to break ties
    t0 = timer.time()
    tnn = two_nn_dimension(X)
    boot_tnn = []
    for _ in range(n_boot):
        X_b = block_bootstrap_resample(X)
        # Jitter to break ties from resampled duplicates (1e-6 of std per feature)
        X_b = X_b + np.random.randn(*X_b.shape) * 1e-6
        boot_tnn.append(two_nn_dimension(X_b))
    boot_tnn = [v for v in boot_tnn if not np.isnan(v)]
    tnn_ci = [float(np.percentile(boot_tnn, 2.5)),
              float(np.percentile(boot_tnn, 97.5))] if boot_tnn else [np.nan, np.nan]
    results['TwoNN'] = {'value': float(tnn), 'ci': tnn_ci}
    print(f"    Two-NN = {tnn:.1f} [{tnn_ci[0]:.1f}, {tnn_ci[1]:.1f}] ({timer.time()-t0:.1f}s)")

    # 3. Correlation dimension
    t0 = timer.time()
    cd, r_vals, C_r = correlation_dimension(X)
    boot_cd = []
    n_boot_cd = min(n_boot, 50)  # Fewer iterations for expensive metric
    for _ in range(n_boot_cd):
        X_b = block_bootstrap_resample(X)
        cd_b, _, _ = correlation_dimension(X_b)
        boot_cd.append(cd_b)
    boot_cd = [v for v in boot_cd if not np.isnan(v)]
    cd_ci = [float(np.percentile(boot_cd, 2.5)),
             float(np.percentile(boot_cd, 97.5))] if boot_cd else [np.nan, np.nan]
    results['CorrDim'] = {'value': float(cd) if not np.isnan(cd) else None, 'ci': cd_ci}
    print(f"    CorrDim = {cd:.1f} [{cd_ci[0]:.1f}, {cd_ci[1]:.1f}] ({timer.time()-t0:.1f}s)")

    # 4. Isomap dimension (primary k + sensitivity)
    t0 = timer.time()
    iso_results = {}
    for k in ISOMAP_K_VALUES:
        iso_d = isomap_dimension(X, k=k)
        iso_results[k] = float(iso_d) if not np.isnan(iso_d) else None
    # Bootstrap only for primary k
    boot_iso = []
    n_boot_iso = min(n_boot, 20)  # Expensive
    for _ in range(n_boot_iso):
        X_b = block_bootstrap_resample(X)
        d_b = isomap_dimension(X_b, k=ISOMAP_K_PRIMARY)
        if not np.isnan(d_b):
            boot_iso.append(d_b)
    iso_ci = [float(np.percentile(boot_iso, 2.5)),
              float(np.percentile(boot_iso, 97.5))] if boot_iso else [np.nan, np.nan]
    results['Isomap'] = {
        'value': iso_results[ISOMAP_K_PRIMARY],
        'k_sensitivity': iso_results,
        'ci': iso_ci
    }
    iso_val = iso_results[ISOMAP_K_PRIMARY]
    iso_str = f"{iso_val:.0f}" if iso_val else "N/A"
    print(f"    Isomap(k={ISOMAP_K_PRIMARY}) = {iso_str} [{iso_ci[0]:.0f}, {iso_ci[1]:.0f}] ({timer.time()-t0:.1f}s)")
    k_str = ", ".join([f"k={k}:{v}" for k, v in iso_results.items()])
    print(f"      Sensitivity: {k_str}")

    # Store eigenvalue spectrum for figures
    results['_eigenvalues'] = evals.tolist()[:50]  # Top 50

    return results


# ============================================================================
# LAYER 1d: VISUALIZATION
# ============================================================================

def run_visualization(X, session_num, region, behav_vars, bin_centers):
    """PCA 2D/3D and UMAP 2D/3D projections."""
    results = {}

    # PCA
    t0 = timer.time()
    pca = PCA(n_components=min(50, X.shape[1]))
    pca_coords = pca.fit_transform(X)
    results['pca'] = pca_coords
    results['pca_var_explained'] = pca.explained_variance_ratio_.tolist()
    print(f"    PCA: top 3 explain {100*sum(pca.explained_variance_ratio_[:3]):.1f}% ({timer.time()-t0:.1f}s)")

    # UMAP
    t0 = timer.time()
    import umap
    reducer_2d = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    umap_2d = reducer_2d.fit_transform(X)
    reducer_3d = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42)
    umap_3d = reducer_3d.fit_transform(X)
    results['umap_2d'] = umap_2d
    results['umap_3d'] = umap_3d
    print(f"    UMAP 2D+3D ({timer.time()-t0:.1f}s)")

    # --- Figure: manifold visualization ---
    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3)

    # PCA 2D colored by time
    ax = fig.add_subplot(gs[0, 0])
    sc = ax.scatter(pca_coords[:, 0], pca_coords[:, 1], c=bin_centers, s=0.5,
                    alpha=0.3, cmap='viridis', rasterized=True)
    ax.set_xlabel(f'PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)')
    ax.set_ylabel(f'PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)')
    ax.set_title('PCA 2D (time)')
    plt.colorbar(sc, ax=ax, label='Time (s)')

    # PCA 3D colored by time
    ax = fig.add_subplot(gs[0, 1], projection='3d')
    sc = ax.scatter(pca_coords[:, 0], pca_coords[:, 1], pca_coords[:, 2],
                    c=bin_centers, s=0.3, alpha=0.2, cmap='viridis', rasterized=True)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('PCA 3D (time)')

    # UMAP 2D colored by time
    ax = fig.add_subplot(gs[0, 2])
    sc = ax.scatter(umap_2d[:, 0], umap_2d[:, 1], c=bin_centers, s=0.5,
                    alpha=0.3, cmap='viridis', rasterized=True)
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.set_title('UMAP 2D (time)')
    plt.colorbar(sc, ax=ax, label='Time (s)')

    # UMAP 3D colored by time
    ax = fig.add_subplot(gs[0, 3], projection='3d')
    sc = ax.scatter(umap_3d[:, 0], umap_3d[:, 1], umap_3d[:, 2],
                    c=bin_centers, s=0.3, alpha=0.2, cmap='viridis', rasterized=True)
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.set_zlabel('UMAP3')
    ax.set_title('UMAP 3D (time)')

    # Bottom row: colored by velocity and compartment
    vel = behav_vars.get('velocity', {}).get('values', np.zeros(len(bin_centers)))
    vel_clean = np.where(np.isnan(vel), 0, vel)

    ax = fig.add_subplot(gs[1, 0])
    sc = ax.scatter(pca_coords[:, 0], pca_coords[:, 1], c=vel_clean, s=0.5,
                    alpha=0.3, cmap='hot', rasterized=True,
                    vmin=0, vmax=np.percentile(vel_clean[vel_clean > 0], 95) if np.any(vel_clean > 0) else 1)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('PCA 2D (velocity)')
    plt.colorbar(sc, ax=ax, label='cm/s')

    ax = fig.add_subplot(gs[1, 1])
    sc = ax.scatter(umap_2d[:, 0], umap_2d[:, 1], c=vel_clean, s=0.5,
                    alpha=0.3, cmap='hot', rasterized=True,
                    vmin=0, vmax=np.percentile(vel_clean[vel_clean > 0], 95) if np.any(vel_clean > 0) else 1)
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.set_title('UMAP 2D (velocity)')
    plt.colorbar(sc, ax=ax, label='cm/s')

    # Compartment coloring
    comp = behav_vars.get('compartment', {}).get('values', np.full(len(bin_centers), 'Arena'))
    comp_colors = {'Home': 'blue', 'Ladder': 'green', 'Arena': 'gray', 'AtPot': 'red'}

    ax = fig.add_subplot(gs[1, 2])
    for label, color in comp_colors.items():
        mask = comp == label
        if mask.sum() > 0:
            ax.scatter(pca_coords[mask, 0], pca_coords[mask, 1], c=color, s=0.5,
                       alpha=0.3, label=label, rasterized=True)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('PCA 2D (compartment)')
    ax.legend(markerscale=10, fontsize=8)

    ax = fig.add_subplot(gs[1, 3])
    for label, color in comp_colors.items():
        mask = comp == label
        if mask.sum() > 0:
            ax.scatter(umap_2d[mask, 0], umap_2d[mask, 1], c=color, s=0.5,
                       alpha=0.3, label=label, rasterized=True)
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.set_title('UMAP 2D (compartment)')
    ax.legend(markerscale=10, fontsize=8)

    fig.suptitle(f'Manifold Visualization -- S{session_num} {region}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path = figdir / f"S{session_num}_{region}_manifold_viz.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved {fig_path}")

    return results


# ============================================================================
# LAYER 2: BEHAVIORAL MAPPING
# ============================================================================

def temporal_cv_folds(n_samples, n_folds=N_CV_FOLDS):
    """Create contiguous temporal CV folds."""
    fold_size = n_samples // n_folds
    folds = []
    for i in range(n_folds):
        start = i * fold_size
        end = (i + 1) * fold_size if i < n_folds - 1 else n_samples
        test_idx = np.arange(start, end)
        train_idx = np.concatenate([np.arange(0, start), np.arange(end, n_samples)])
        folds.append((train_idx, test_idx))
    return folds


def compute_decodability_continuous(pcs, target, folds):
    """Ridge regression from PCs to continuous target. Returns R2 mean and per-fold."""
    valid = ~np.isnan(target)
    if valid.sum() < 100:
        return np.nan, []

    r2s = []
    for train_idx, test_idx in folds:
        tr_valid = valid[train_idx]
        te_valid = valid[test_idx]
        if tr_valid.sum() < 50 or te_valid.sum() < 20:
            continue
        model = Ridge(alpha=1.0)
        model.fit(pcs[train_idx][tr_valid], target[train_idx][tr_valid])
        pred = model.predict(pcs[test_idx][te_valid])
        r2 = r2_score(target[test_idx][te_valid], pred)
        r2s.append(r2)
    return float(np.mean(r2s)) if r2s else np.nan, r2s


def compute_decodability_categorical(pcs, labels, folds):
    """Linear SVM from PCs to categorical label. Returns balanced accuracy."""
    le = LabelEncoder()
    encoded = le.fit_transform(labels)

    # Skip if fewer than 2 classes
    if len(le.classes_) < 2:
        return np.nan, []

    accs = []
    for train_idx, test_idx in folds:
        # Check that both train and test have all classes
        tr_classes = set(encoded[train_idx])
        te_classes = set(encoded[test_idx])
        if len(tr_classes) < 2 or len(te_classes) < 2:
            continue
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(pcs[train_idx])
        X_te = scaler.transform(pcs[test_idx])
        model = LinearSVC(max_iter=5000, random_state=42)
        model.fit(X_tr, encoded[train_idx])
        pred = model.predict(X_te)
        acc = balanced_accuracy_score(encoded[test_idx], pred)
        accs.append(acc)
    return float(np.mean(accs)) if accs else np.nan, accs


def compute_mi(cluster_labels, variable, is_categorical=False):
    """Mutual information between k-means cluster labels and behavioral variable.
    For continuous variables, discretize into 10 quantile bins."""
    valid = ~np.isnan(variable) if not is_categorical else np.ones(len(variable), dtype=bool)
    if valid.sum() < 100:
        return np.nan

    cl = cluster_labels[valid]
    var = variable[valid] if not is_categorical else variable[valid]

    if not is_categorical:
        # Discretize continuous variable into 10 quantile bins
        try:
            var_binned = pd.qcut(var, q=10, labels=False, duplicates='drop')
        except ValueError:
            var_binned = pd.cut(var, bins=10, labels=False)
        var_binned = np.nan_to_num(var_binned, nan=0).astype(int)
    else:
        le = LabelEncoder()
        var_binned = le.fit_transform(var)

    return float(mutual_info_score(cl, var_binned))


def run_behavioral_mapping(neural_matrix, behav_vars, dim_results, session_num, region):
    """Run decodability, MI, silhouette for all behavioral variables."""
    results = {}

    # Determine intrinsic dimensionality for PC count
    dim_estimates = []
    for key in ['TwoNN', 'CorrDim', 'Isomap']:
        val = dim_results.get(key, {}).get('value')
        if val is not None and not np.isnan(val):
            dim_estimates.append(val)
    if dim_estimates:
        K = max(3, int(np.ceil(np.mean(dim_estimates))))
    else:
        K = 10  # fallback
    K = min(K, neural_matrix.shape[1])  # can't exceed n_units
    print(f"    Using K={K} PCs (mean intrinsic dim estimate)")

    # PCA projection to K dimensions
    pca = PCA(n_components=K)
    pcs = pca.fit_transform(neural_matrix)

    # K-means clustering on PCA projection
    kmeans = KMeans(n_clusters=N_KMEANS, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(pcs)

    # CV folds
    folds = temporal_cv_folds(len(neural_matrix))

    # --- Process each variable ---
    print(f"    Behavioral variables ({len(behav_vars)}):")
    for var_name, var_info in behav_vars.items():
        values = var_info['values']
        vtype = var_info['type']

        if vtype == 'continuous':
            # Decodability
            r2, fold_r2s = compute_decodability_continuous(pcs, values, folds)
            # MI
            mi = compute_mi(cluster_labels, values, is_categorical=False)
            # Silhouette: N/A for continuous
            sil = np.nan

            results[var_name] = {
                'type': 'continuous',
                'decodability': r2,
                'fold_scores': fold_r2s,
                'MI': mi,
                'silhouette': sil,
            }
            r2_str = f"{r2:.3f}" if not np.isnan(r2) else "N/A"
            mi_str = f"{mi:.4f}" if not np.isnan(mi) else "N/A"
            print(f"      {var_name}: R2={r2_str}, MI={mi_str}")

        elif vtype == 'categorical':
            # Decodability
            acc, fold_accs = compute_decodability_categorical(pcs, values, folds)
            # MI (encode as integers)
            le = LabelEncoder()
            encoded = le.fit_transform(values)
            mi = compute_mi(cluster_labels, encoded.astype(float), is_categorical=False)
            # Silhouette
            if len(np.unique(values)) >= 2:
                le2 = LabelEncoder()
                sil = float(silhouette_score(pcs, le2.fit_transform(values),
                                             sample_size=min(5000, len(pcs))))
            else:
                sil = np.nan

            results[var_name] = {
                'type': 'categorical',
                'decodability': acc,
                'fold_scores': fold_accs,
                'MI': mi,
                'silhouette': sil,
                'classes': list(np.unique(values)),
            }
            acc_str = f"{acc:.3f}" if not np.isnan(acc) else "N/A"
            print(f"      {var_name}: Acc={acc_str}, MI={mi:.4f}, Sil={sil:.3f}")

    # --- Null baseline: circular-shifted neural data ---
    print("    Computing null baseline (circular shift)...", flush=True)
    # Use the shuffle that was already computed and stored, or create new
    # For simplicity, create a permuted version of cluster labels
    np.random.seed(42)
    null_perm = np.random.permutation(len(cluster_labels))
    null_clusters = cluster_labels[null_perm]

    # Decodability null: shuffle target labels relative to PCs
    null_results = {}
    for var_name, var_info in behav_vars.items():
        values = var_info['values']
        vtype = var_info['type']
        shuffled_values = values[null_perm]

        if vtype == 'continuous':
            r2_null, _ = compute_decodability_continuous(pcs, shuffled_values, folds)
            mi_null = compute_mi(null_clusters, values, is_categorical=False)
            null_results[var_name] = {'decodability_null': r2_null, 'MI_null': mi_null}
        elif vtype == 'categorical':
            acc_null, _ = compute_decodability_categorical(pcs, shuffled_values, folds)
            le = LabelEncoder()
            encoded = le.fit_transform(values)
            mi_null = compute_mi(null_clusters, encoded.astype(float), is_categorical=False)
            null_results[var_name] = {'decodability_null': acc_null, 'MI_null': mi_null}

    # Merge null results
    for var_name in results:
        if var_name in null_results:
            results[var_name].update(null_results[var_name])

    results['_K'] = K
    results['_n_clusters'] = N_KMEANS
    print("    Done.")

    return results


# ============================================================================
# FIGURE GENERATION
# ============================================================================

def plot_dimensionality(data_results, shuffle_results, session_num, region, n_units):
    """Plot dimensionality comparison: data vs shuffle, all 4 metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Bar chart of all 4 estimates
    ax = axes[0]
    metrics = ['PR', 'TwoNN', 'CorrDim', 'Isomap']
    labels = ['PR', 'Two-NN', 'Corr. Dim', 'Isomap']
    x = np.arange(len(metrics))

    data_vals = []
    data_errs_lo = []
    data_errs_hi = []
    shuf_vals = []
    for m in metrics:
        dv = data_results.get(m, {}).get('value')
        sv = shuffle_results.get(m, {}).get('value')
        ci = data_results.get(m, {}).get('ci', [np.nan, np.nan])
        data_vals.append(dv if dv is not None else 0)
        shuf_vals.append(sv if sv is not None else 0)
        data_errs_lo.append(max(0, (dv or 0) - ci[0]) if dv else 0)
        data_errs_hi.append(max(0, ci[1] - (dv or 0)) if dv else 0)

    ax.bar(x - 0.15, data_vals, 0.3, label='Data', color='steelblue',
           yerr=[data_errs_lo, data_errs_hi], capsize=5)
    ax.bar(x + 0.15, shuf_vals, 0.3, label='Shuffle', color='gray', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Dimensionality')
    ax.set_title('Dimensionality Estimates', fontweight='bold')
    ax.legend()

    # Panel 2: Eigenvalue spectrum
    ax = axes[1]
    evals = data_results.get('_eigenvalues', [])
    shuf_evals = shuffle_results.get('_eigenvalues', [])
    if evals:
        ax.plot(range(1, len(evals)+1), evals, 'ko-', markersize=3, label='Data')
    if shuf_evals:
        ax.plot(range(1, len(shuf_evals)+1), shuf_evals, 'g--', alpha=0.5, label='Shuffle')
    ax.set_xlabel('PC index')
    ax.set_ylabel('Eigenvalue')
    ax.set_title('PCA Eigenvalue Spectrum', fontweight='bold')
    ax.set_yscale('log')
    ax.legend()

    # Panel 3: Isomap k sensitivity
    ax = axes[2]
    k_sens = data_results.get('Isomap', {}).get('k_sensitivity', {})
    if k_sens:
        ks = sorted(k_sens.keys())
        vals = [k_sens[k] if k_sens[k] is not None else 0 for k in ks]
        ax.plot(ks, vals, 'ko-', markersize=6, linewidth=2)
        ax.set_xlabel('k (neighbors)')
        ax.set_ylabel('Isomap Dimensionality')
        ax.set_title('Isomap k Sensitivity', fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No Isomap results', transform=ax.transAxes, ha='center')

    fig.suptitle(f'Layer 1a: Dimensionality -- S{session_num} {region} (N={n_units})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path = figdir / f"S{session_num}_{region}_dimensionality.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved {fig_path}")


def plot_behavioral_results(behav_results, viz_results, behav_vars,
                            session_num, region):
    """Plot behavioral mapping: ranking + UMAP colored by top variables."""
    # --- Ranking figure ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Get variable names and scores, sorted by decodability
    var_names = [k for k in behav_results if not k.startswith('_')]
    decode_scores = []
    mi_scores = []
    null_scores = []
    for vn in var_names:
        d = behav_results[vn].get('decodability', np.nan)
        m = behav_results[vn].get('MI', np.nan)
        dn = behav_results[vn].get('decodability_null', np.nan)
        decode_scores.append(d if not np.isnan(d) else 0)
        mi_scores.append(m if not np.isnan(m) else 0)
        null_scores.append(dn if not np.isnan(dn) else 0)

    # Sort by decodability
    sort_idx = np.argsort(decode_scores)[::-1]
    sorted_names = [var_names[i] for i in sort_idx]
    sorted_decode = [decode_scores[i] for i in sort_idx]
    sorted_null = [null_scores[i] for i in sort_idx]
    sorted_mi = [mi_scores[i] for i in sort_idx]

    # Panel 1: Decodability ranking
    ax = axes[0]
    y = range(len(sorted_names))
    ax.barh(y, sorted_decode, 0.4, label='Data', color='steelblue')
    ax.barh([i + 0.4 for i in y], sorted_null, 0.4, label='Null', color='gray', alpha=0.5)
    ax.set_yticks([i + 0.2 for i in y])
    ax.set_yticklabels(sorted_names, fontsize=9)
    ax.set_xlabel('R2 / Balanced Accuracy')
    ax.set_title('Decodability Ranking', fontweight='bold')
    ax.legend(fontsize=8)
    ax.invert_yaxis()

    # Panel 2: MI ranking
    ax = axes[1]
    mi_sort = np.argsort(sorted_mi)[::-1]
    mi_names = [sorted_names[i] for i in mi_sort]
    mi_vals = [sorted_mi[i] for i in mi_sort]
    ax.barh(range(len(mi_names)), mi_vals, color='darkorange')
    ax.set_yticks(range(len(mi_names)))
    ax.set_yticklabels(mi_names, fontsize=9)
    ax.set_xlabel('Mutual Information (bits)')
    ax.set_title('MI Ranking', fontweight='bold')
    ax.invert_yaxis()

    # Panel 3: Silhouette (categorical only)
    ax = axes[2]
    cat_vars = [(vn, behav_results[vn].get('silhouette', np.nan))
                for vn in var_names if behav_results[vn].get('type') == 'categorical']
    if cat_vars:
        names, sils = zip(*cat_vars)
        ax.barh(range(len(names)), sils, color='forestgreen')
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel('Silhouette Score')
        ax.set_title('Silhouette (categorical)', fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No categorical variables', transform=ax.transAxes, ha='center')

    fig.suptitle(f'Layer 2: Behavioral Mapping -- S{session_num} {region}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path = figdir / f"S{session_num}_{region}_behav_ranking.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved {fig_path}")

    # --- UMAP colored by continuous variables ---
    umap_2d = viz_results.get('umap_2d')
    if umap_2d is None:
        return

    cont_vars = [(vn, vi) for vn, vi in behav_vars.items() if vi['type'] == 'continuous']
    n_vars = len(cont_vars)
    ncols = 4
    nrows = int(np.ceil(n_vars / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)

    for i, (vn, vi) in enumerate(cont_vars):
        row, col = divmod(i, ncols)
        ax = axes[row, col]
        vals = vi['values'].copy()
        valid = ~np.isnan(vals)
        if valid.sum() < 10:
            ax.set_title(vn)
            continue
        vmin = np.percentile(vals[valid], 2)
        vmax = np.percentile(vals[valid], 98)
        sc = ax.scatter(umap_2d[valid, 0], umap_2d[valid, 1],
                        c=vals[valid], s=0.3, alpha=0.3, cmap='viridis',
                        vmin=vmin, vmax=vmax, rasterized=True)
        ax.set_title(vn, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(sc, ax=ax, shrink=0.8)

    # Hide unused axes
    for i in range(n_vars, nrows * ncols):
        row, col = divmod(i, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle(f'UMAP colored by behavior -- S{session_num} {region}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path = figdir / f"S{session_num}_{region}_behav_viz.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved {fig_path}")


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_region_report(dim_data, dim_shuf, behav_results, session_num, region, n_units):
    """Generate markdown summary for one region."""
    lines = [
        f"# Manifold Geometry -- S{session_num} {region}",
        f"",
        f"## Layer 1a: Dimensionality",
        f"",
        f"| Metric | Data | 95% CI | Shuffle | Data/Shuffle |",
        f"|--------|------|--------|---------|-------------|",
    ]
    for key, label in [('PR', 'Participation Ratio'), ('TwoNN', 'Two-NN'),
                        ('CorrDim', 'Correlation Dim'), ('Isomap', f'Isomap (k={ISOMAP_K_PRIMARY})')]:
        dv = dim_data.get(key, {}).get('value')
        sv = dim_shuf.get(key, {}).get('value')
        ci = dim_data.get(key, {}).get('ci', [None, None])

        dv_str = f"{dv:.1f}" if dv is not None else "N/A"
        sv_str = f"{sv:.1f}" if sv is not None else "N/A"
        ci_str = f"[{ci[0]:.1f}, {ci[1]:.1f}]" if ci[0] is not None and not np.isnan(ci[0]) else "N/A"
        ratio = f"{dv/sv:.2f}" if dv and sv and sv > 0 else "N/A"
        lines.append(f"| {label} | {dv_str} | {ci_str} | {sv_str} | {ratio} |")

    # PR vs intrinsic gap
    pr = dim_data.get('PR', {}).get('value', 0)
    intrinsic = []
    for k in ['TwoNN', 'CorrDim', 'Isomap']:
        v = dim_data.get(k, {}).get('value')
        if v is not None and not np.isnan(v):
            intrinsic.append(v)
    mean_intrinsic = np.mean(intrinsic) if intrinsic else None

    lines.extend([
        f"",
        f"**PR vs intrinsic gap:** PR={pr:.1f}, mean intrinsic={f'{mean_intrinsic:.1f}' if mean_intrinsic else 'N/A'}. ",
    ])
    if mean_intrinsic and pr > 0:
        ratio = pr / mean_intrinsic
        if ratio > 2.0:
            lines.append(f"PR >> intrinsic ({ratio:.1f}x) -- curved manifold (high curvature inflates linear PR).")
        elif ratio > 1.3:
            lines.append(f"PR > intrinsic ({ratio:.1f}x) -- mild curvature or nonlinearity.")
        else:
            lines.append(f"PR ~ intrinsic ({ratio:.1f}x) -- approximately linear manifold.")

    # Isomap sensitivity
    k_sens = dim_data.get('Isomap', {}).get('k_sensitivity', {})
    if k_sens:
        lines.extend([
            f"",
            f"**Isomap k sensitivity:** " + ", ".join([f"k={k}: {v}" for k, v in sorted(k_sens.items())]),
        ])

    # Layer 2: Behavioral mapping
    lines.extend([
        f"",
        f"## Layer 2: Behavioral Mapping",
        f"",
        f"K = {behav_results.get('_K', '?')} PCs, {behav_results.get('_n_clusters', '?')} k-means clusters.",
        f"",
        f"| Variable | Type | Decodability | Null | MI | Silhouette |",
        f"|----------|------|-------------|------|-----|-----------|",
    ])

    var_items = [(k, v) for k, v in behav_results.items() if not k.startswith('_')]
    # Sort by decodability descending
    var_items.sort(key=lambda x: x[1].get('decodability', 0) or 0, reverse=True)

    for vn, vi in var_items:
        d = vi.get('decodability')
        dn = vi.get('decodability_null')
        mi = vi.get('MI')
        sil = vi.get('silhouette')
        vt = vi.get('type', '?')

        d_str = f"{d:.3f}" if d is not None and not np.isnan(d) else "N/A"
        dn_str = f"{dn:.3f}" if dn is not None and not np.isnan(dn) else "N/A"
        mi_str = f"{mi:.4f}" if mi is not None and not np.isnan(mi) else "N/A"
        sil_str = f"{sil:.3f}" if sil is not None and not np.isnan(sil) else "-"
        metric_label = "R2" if vt == 'continuous' else "Acc"
        lines.append(f"| {vn} | {vt} ({metric_label}) | {d_str} | {dn_str} | {mi_str} | {sil_str} |")

    # Top 5 ranking
    top5 = var_items[:5]
    lines.extend([
        f"",
        f"**Top 5 manifold-organizing variables (by decodability):**",
    ])
    for i, (vn, vi) in enumerate(top5):
        d = vi.get('decodability', 0) or 0
        lines.append(f"  {i+1}. {vn} ({d:.3f})")

    lines.extend([
        f"",
        f"---",
        f"*N = {n_units} units, {BIN_MS}ms bins, Gaussian smooth sigma={SMOOTH_SIGMA} bins.*",
    ])

    report_path = outdir / f"S{session_num}_{region}_manifold_summary.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"    Saved {report_path}")
    return lines


def generate_cross_region_report(aca_dim, aca_dim_shuf, aca_behav,
                                  lha_dim, lha_dim_shuf, lha_behav,
                                  aca_sub_dim, session_num, n_aca, n_lha):
    """Generate cross-region comparison report."""
    lines = [
        f"# Cross-Region Manifold Comparison -- S{session_num}",
        f"",
        f"## Dimensionality Comparison",
        f"",
        f"| Metric | ACA (N={n_aca}) | ACA sub (N={n_lha}) | LHA (N={n_lha}) | ACA/shuf | LHA/shuf |",
        f"|--------|----------------|---------------------|-----------------|----------|----------|",
    ]

    for key, label in [('PR', 'PR'), ('TwoNN', 'Two-NN'), ('CorrDim', 'CorrDim'), ('Isomap', 'Isomap')]:
        aca_v = aca_dim.get(key, {}).get('value')
        lha_v = lha_dim.get(key, {}).get('value')
        aca_sv = aca_dim_shuf.get(key, {}).get('value')
        lha_sv = lha_dim_shuf.get(key, {}).get('value')
        sub_v = aca_sub_dim.get(key, {}).get('value') if aca_sub_dim else None

        aca_str = f"{aca_v:.1f}" if aca_v is not None else "N/A"
        lha_str = f"{lha_v:.1f}" if lha_v is not None else "N/A"
        sub_str = f"{sub_v:.1f}" if sub_v is not None else "N/A"
        aca_ratio = f"{aca_v/aca_sv:.2f}" if aca_v and aca_sv and aca_sv > 0 else "N/A"
        lha_ratio = f"{lha_v/lha_sv:.2f}" if lha_v and lha_sv and lha_sv > 0 else "N/A"
        lines.append(f"| {label} | {aca_str} | {sub_str} | {lha_str} | {aca_ratio} | {lha_ratio} |")

    # Behavioral ranking comparison
    lines.extend([
        f"",
        f"## Top Manifold-Organizing Variables",
        f"",
        f"| Rank | ACA | ACA score | LHA | LHA score |",
        f"|------|-----|-----------|-----|-----------|",
    ])

    aca_vars = [(k, v) for k, v in aca_behav.items() if not k.startswith('_')]
    lha_vars = [(k, v) for k, v in lha_behav.items() if not k.startswith('_')]
    aca_vars.sort(key=lambda x: x[1].get('decodability', 0) or 0, reverse=True)
    lha_vars.sort(key=lambda x: x[1].get('decodability', 0) or 0, reverse=True)

    for i in range(min(5, len(aca_vars), len(lha_vars))):
        aca_n, aca_s = aca_vars[i][0], aca_vars[i][1].get('decodability', 0) or 0
        lha_n, lha_s = lha_vars[i][0], lha_vars[i][1].get('decodability', 0) or 0
        lines.append(f"| {i+1} | {aca_n} | {aca_s:.3f} | {lha_n} | {lha_s:.3f} |")

    # Shared organizing variables
    aca_top5 = set([v[0] for v in aca_vars[:5]])
    lha_top5 = set([v[0] for v in lha_vars[:5]])
    shared = aca_top5 & lha_top5
    aca_unique = aca_top5 - lha_top5
    lha_unique = lha_top5 - aca_top5

    lines.extend([
        f"",
        f"**Shared top-5:** {', '.join(shared) if shared else 'None'}",
        f"**ACA-specific:** {', '.join(aca_unique) if aca_unique else 'None'}",
        f"**LHA-specific:** {', '.join(lha_unique) if lha_unique else 'None'}",
        f"",
        f"---",
        f"*ACA subsampled to N={n_lha} units (mean of {N_SUBSAMPLE_DRAWS} random draws) for direct comparison.*",
        f"*Dimensionality normalized to shuffle control (data/shuffle ratio) for sample-size-robust cross-region metric.*",
    ])

    report_path = outdir / f"S{session_num}_crossregion_manifold.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"    Saved {report_path}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_region_pipeline(session_num, region, behav_df, bin_centers_ref=None):
    """Full Layer 1a + 1d + Layer 2 for one region."""
    print(f"\n{'='*80}")
    print(f"  REGION: {region}")
    print(f"{'='*80}")

    # Load and preprocess neural data
    print(f"\n  Loading neural data...")
    spike_dict, session_duration, unit_ids = load_neural_data(session_num, region)
    n_units = len(unit_ids)
    print(f"    {n_units} good {region} units, {sum(len(v) for v in spike_dict.values()):,} spikes")

    print(f"  Preprocessing ({BIN_MS}ms bins, smooth sigma={SMOOTH_SIGMA})...")
    matrix, bin_centers, _ = preprocess_neural(spike_dict, session_duration)
    print(f"    Neural matrix: {matrix.shape[0]} bins x {matrix.shape[1]} units")

    # Align behavior
    print(f"  Aligning behavioral data...")
    indices = align_behavior_to_bins(behav_df, bin_centers)
    behav_vars = extract_behavioral_variables(behav_df, indices)

    # Spot-check alignment: verify "at pot" matches low pot distance
    pot_mask = behav_vars['compartment']['values'] == 'AtPot'
    if pot_mask.sum() > 0 and 'dist_nearest_pot' in behav_vars:
        mean_dist_at_pot = np.nanmean(behav_vars['dist_nearest_pot']['values'][pot_mask])
        mean_dist_not_pot = np.nanmean(behav_vars['dist_nearest_pot']['values'][~pot_mask])
        print(f"    Alignment check: dist_nearest_pot at pot={mean_dist_at_pot:.1f}cm, "
              f"not at pot={mean_dist_not_pot:.1f}cm")
        if mean_dist_at_pot > mean_dist_not_pot:
            print(f"    *** WARNING: 'At pot' has HIGHER distance to pot -- check alignment! ***")

    # Print behavioral variable summary
    comp = behav_vars['compartment']['values']
    for label in ['Home', 'Ladder', 'Arena', 'AtPot']:
        n = (comp == label).sum()
        pct = 100 * n / len(comp)
        print(f"    Compartment '{label}': {n} bins ({pct:.1f}%)")

    # ---- Layer 1a: Dimensionality ----
    print(f"\n  Layer 1a: Dimensionality Suite")
    print(f"  {'='*40}")

    print(f"  Data:")
    dim_data = run_dimensionality_suite(matrix, label='data')

    print(f"\n  Shuffle control (circular shift):")
    shuf_dict = circular_shift_shuffle(spike_dict, session_duration)
    shuf_matrix, _, _ = preprocess_neural(shuf_dict, session_duration)
    dim_shuf = run_dimensionality_suite(shuf_matrix, n_boot=50, label='shuffle')

    # Dimensionality figure
    plot_dimensionality(dim_data, dim_shuf, session_num, region, n_units)

    # ---- Layer 1d: Visualization ----
    print(f"\n  Layer 1d: Visualization")
    print(f"  {'='*40}")
    viz_results = run_visualization(matrix, session_num, region, behav_vars, bin_centers)

    # ---- Layer 2: Behavioral Mapping ----
    print(f"\n  Layer 2: Behavioral Mapping")
    print(f"  {'='*40}")
    behav_results = run_behavioral_mapping(matrix, behav_vars, dim_data,
                                            session_num, region)

    # Behavioral figures
    plot_behavioral_results(behav_results, viz_results, behav_vars,
                            session_num, region)

    # Region report
    generate_region_report(dim_data, dim_shuf, behav_results,
                           session_num, region, n_units)

    # Save JSON results
    def convert_json(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, tuple):
            return list(obj)
        return obj

    # Save layer 1a
    json_path = outdir / f"S{session_num}_{region}_layer1a.json"
    save_data = {'data': dim_data, 'shuffle': dim_shuf}
    # Remove large arrays
    for d in [save_data['data'], save_data['shuffle']]:
        d.pop('_eigenvalues', None)
    with open(json_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=convert_json)
    print(f"    Saved {json_path}")

    # Save layer 2
    json_path = outdir / f"S{session_num}_{region}_layer2.json"
    save_behav = {k: v for k, v in behav_results.items() if not k.startswith('_')}
    # Remove fold scores (verbose)
    for v in save_behav.values():
        v.pop('fold_scores', None)
    save_behav['_K'] = behav_results.get('_K')
    save_behav['_n_clusters'] = behav_results.get('_n_clusters')
    with open(json_path, 'w') as f:
        json.dump(save_behav, f, indent=2, default=convert_json)
    print(f"    Saved {json_path}")

    return {
        'dim_data': dim_data,
        'dim_shuf': dim_shuf,
        'behav_results': behav_results,
        'viz_results': viz_results,
        'matrix': matrix,
        'spike_dict': spike_dict,
        'session_duration': session_duration,
        'n_units': n_units,
        'bin_centers': bin_centers,
    }


def run_aca_subsampled(aca_results, n_target, session_num):
    """Subsample ACA to n_target units, compute dimensionality.
    Average over N_SUBSAMPLE_DRAWS random draws."""
    print(f"\n  ACA Subsampled to N={n_target} (x{N_SUBSAMPLE_DRAWS} draws)...")
    matrix_full = aca_results['matrix']
    n_full = matrix_full.shape[1]

    if n_target >= n_full:
        print(f"    SKIP: n_target={n_target} >= n_units={n_full}")
        return None

    all_results = {k: [] for k in ['PR', 'TwoNN', 'CorrDim', 'Isomap']}

    for draw in range(N_SUBSAMPLE_DRAWS):
        cols = np.random.choice(n_full, n_target, replace=False)
        X_sub = matrix_full[:, cols]

        pr, _ = participation_ratio(X_sub)
        tnn = two_nn_dimension(X_sub)
        cd, _, _ = correlation_dimension(X_sub)
        iso = isomap_dimension(X_sub, k=ISOMAP_K_PRIMARY)

        all_results['PR'].append(pr)
        all_results['TwoNN'].append(tnn)
        all_results['CorrDim'].append(cd)
        all_results['Isomap'].append(iso)

        if (draw + 1) % 5 == 0:
            print(f"    Draw {draw+1}/{N_SUBSAMPLE_DRAWS}")

    # Average across draws
    summary = {}
    for key in all_results:
        vals = [v for v in all_results[key] if v is not None and not np.isnan(v)]
        if vals:
            summary[key] = {
                'value': float(np.mean(vals)),
                'ci': [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))],
            }
        else:
            summary[key] = {'value': None, 'ci': [np.nan, np.nan]}
        print(f"    {key}: {summary[key]['value']:.1f}" if summary[key]['value'] else f"    {key}: N/A")

    return summary


def main():
    session_num = int(sys.argv[1]) if len(sys.argv) > 1 else PILOT_SESSION
    sval = sessions_cfg[f"session_{session_num}"]
    state = sval['state']
    phase = sval['phase']

    print("=" * 80)
    print(f"MANIFOLD GEOMETRY ANALYSIS -- S{session_num} ({state}/{phase})")
    print(f"ACA + LHA, Layer 1a + 1d + Layer 2")
    print("=" * 80)

    # Load behavioral data (shared across regions)
    print("\nLoading behavioral data...")
    behav_df = load_behavioral_data(session_num)
    print(f"  {len(behav_df)} samples at ~40ms")

    # Run both regions
    t_start = timer.time()

    aca_results = run_region_pipeline(session_num, 'ACA', behav_df)
    lha_results = run_region_pipeline(session_num, 'LHA', behav_df)

    # Cross-region: subsample ACA to LHA unit count
    n_lha = lha_results['n_units']
    aca_sub_dim = run_aca_subsampled(aca_results, n_lha, session_num)

    # Cross-region comparison figure
    print("\n  Cross-Region Comparison")
    print("  " + "=" * 40)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Raw dimensionality
    ax = axes[0]
    metrics = ['PR', 'TwoNN', 'CorrDim', 'Isomap']
    labels = ['PR', 'Two-NN', 'CorrDim', 'Isomap']
    x = np.arange(len(metrics))
    w = 0.25

    aca_vals = [aca_results['dim_data'].get(m, {}).get('value', 0) or 0 for m in metrics]
    lha_vals = [lha_results['dim_data'].get(m, {}).get('value', 0) or 0 for m in metrics]
    sub_vals = [aca_sub_dim.get(m, {}).get('value', 0) or 0 for m in metrics] if aca_sub_dim else [0]*4

    ax.bar(x - w, aca_vals, w, label=f'ACA (N={aca_results["n_units"]})', color='steelblue')
    ax.bar(x, sub_vals, w, label=f'ACA sub (N={n_lha})', color='steelblue', alpha=0.5)
    ax.bar(x + w, lha_vals, w, label=f'LHA (N={n_lha})', color='darkorange')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Dimensionality')
    ax.set_title('Raw Dimensionality', fontweight='bold')
    ax.legend(fontsize=9)

    # Panel 2: Normalized to shuffle
    ax = axes[1]
    aca_ratios = []
    lha_ratios = []
    for m in metrics:
        dv = aca_results['dim_data'].get(m, {}).get('value')
        sv = aca_results['dim_shuf'].get(m, {}).get('value')
        aca_ratios.append(dv / sv if dv and sv and sv > 0 else 0)
        dv = lha_results['dim_data'].get(m, {}).get('value')
        sv = lha_results['dim_shuf'].get(m, {}).get('value')
        lha_ratios.append(dv / sv if dv and sv and sv > 0 else 0)

    ax.bar(x - 0.15, aca_ratios, 0.3, label='ACA', color='steelblue')
    ax.bar(x + 0.15, lha_ratios, 0.3, label='LHA', color='darkorange')
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='Shuffle baseline')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Data / Shuffle ratio')
    ax.set_title('Shuffle-Normalized Dimensionality', fontweight='bold')
    ax.legend(fontsize=9)

    fig.suptitle(f'Cross-Region Dimensionality -- S{session_num}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path = figdir / f"S{session_num}_crossregion_dimensionality.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved {fig_path}")

    # Cross-region report
    generate_cross_region_report(
        aca_results['dim_data'], aca_results['dim_shuf'], aca_results['behav_results'],
        lha_results['dim_data'], lha_results['dim_shuf'], lha_results['behav_results'],
        aca_sub_dim, session_num, aca_results['n_units'], n_lha
    )

    total_time = timer.time() - t_start
    print(f"\n{'='*80}")
    print(f"DONE -- Total time: {total_time/60:.1f} minutes")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
