"""
Manifold Geometry Detection
============================
Detect geometric structure (rings, tori, loops) in neural population
activity during individual excursions.

Approaches:
  1. UMAP with min_dist=0.0 (tight structure preservation) — scatter plots
  2. Isomap (geodesic-preserving embedding)
  3. Recurrence analysis — autocorrelation of neural state to detect periodicity
  4. Spectral analysis of distance matrix — eigenvalue gap for circular topology
  5. Multiple time resolutions

Focus: Session 1 (Fed Exploration), longest complete excursions, LHA & RSP.
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import umap
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import uniform_filter1d
from scipy.spatial.distance import pdist, squareform
from scipy.signal import find_peaks
import spikeinterface.extractors as se
import warnings
import time as timer

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG
# =============================================================================

FS = 30000
LHA_DEPTH_MAX = 1300
RSP_DEPTH_MIN = 1300

with open("paths.yaml") as f:
    paths_config = yaml.safe_load(f)

ZONE_COLORS = {0: '#BDBDBD', 1: '#2196F3', 2: '#FF9800', 3: '#9C27B0', 4: '#4CAF50'}
ZONE_NAMES = {0: 'None', 1: 'Home', 2: 'Ladder', 3: 'Transition', 4: 'Arena'}


# =============================================================================
# DATA LOADING
# =============================================================================

def get_good_units_by_region(sorted_path_obj):
    ci = sorted_path_obj / "cluster_info.tsv"
    if not ci.exists():
        return np.array([]), np.array([])
    df = pd.read_csv(ci, sep='\t')
    if 'depth' not in df.columns:
        return np.array([]), np.array([])
    label_col = None
    if 'group' in df.columns and df['group'].eq('good').any():
        label_col = 'group'
    elif 'KSLabel' in df.columns:
        label_col = 'KSLabel'
    if label_col is None:
        return np.array([]), np.array([])
    good = df[df[label_col] == 'good']
    lha_ids = good[good['depth'] < LHA_DEPTH_MAX]['cluster_id'].values
    rsp_ids = good[good['depth'] >= RSP_DEPTH_MIN]['cluster_id'].values
    return lha_ids, rsp_ids


def bin_and_smooth_flexible(sorting, unit_ids, bin_ms=10, smooth_ms=100):
    """Bin spikes with flexible resolution, smooth, z-score."""
    bin_samples = int(bin_ms * FS / 1000)
    smooth_bins = max(1, int(smooth_ms / bin_ms))

    spike_trains = {}
    all_min, all_max = np.inf, 0
    for uid in unit_ids:
        st = sorting.get_unit_spike_train(uid)
        spike_trains[uid] = st
        if len(st) > 0:
            all_min = min(all_min, np.min(st))
            all_max = max(all_max, np.max(st))

    n_bins = int((all_max - all_min) / bin_samples) + 1
    data = np.zeros((n_bins, len(unit_ids)), dtype=np.float32)
    for i, uid in enumerate(unit_ids):
        st = spike_trains[uid]
        if len(st) > 0:
            b = ((st - all_min) // bin_samples).astype(int)
            b = b[(b >= 0) & (b < n_bins)]
            np.add.at(data[:, i], b, 1)

    if smooth_bins > 1:
        for i in range(data.shape[1]):
            data[:, i] = uniform_filter1d(data[:, i], size=smooth_bins, mode='constant')

    means = data.mean(axis=0, keepdims=True)
    stds = data.std(axis=0, keepdims=True)
    stds[stds < 1e-8] = 1.0
    zscore_data = (data - means) / stds

    time_sec = (np.arange(n_bins) * bin_ms / 1000) + (all_min / FS)
    return zscore_data, time_sec


def load_behavior_zones(beh_path):
    df = pd.read_csv(beh_path, header=None)
    labels = df.iloc[:, 0].values

    def get_row(name):
        idx = np.where(labels == name)[0]
        return df.iloc[idx[0], 1:].astype(float).values if len(idx) > 0 else None

    beh_times = get_row('Recording time')
    home = get_row('Home')
    ladder = get_row('Ladder')
    transition = get_row('Transition zone')
    arena = get_row('Foraging arena')

    zone_ids = np.zeros(len(beh_times), dtype=int)
    zone_ids[home == 1] = 1
    zone_ids[ladder == 1] = 2
    zone_ids[transition == 1] = 3
    zone_ids[arena == 1] = 4
    return beh_times, zone_ids


def extract_excursion_data(zscore, time_sec, exc_start, exc_end):
    mask = (time_sec >= exc_start) & (time_sec <= exc_end)
    return zscore[mask], time_sec[mask]


def map_zones(neural_times, beh_times, beh_zones):
    zones = np.zeros(len(neural_times), dtype=int)
    for i, t in enumerate(neural_times):
        idx = min(np.searchsorted(beh_times, t), len(beh_zones) - 1)
        zones[i] = beh_zones[idx]
    return zones


# =============================================================================
# TOPOLOGY DETECTION (without ripser)
# =============================================================================

def recurrence_analysis(data, max_lag=None):
    """Compute autocorrelation of neural state vector to detect periodicity.

    If the manifold has a ring/loop, the neural state will recur
    and the autocorrelation will show periodic peaks.
    """
    n = len(data)
    if max_lag is None:
        max_lag = n // 2

    # Use PCA-reduced data for efficiency
    n_comp = min(10, data.shape[1], n - 1)
    pca = PCA(n_components=n_comp)
    data_pca = pca.fit_transform(data)

    # Compute self-similarity (cosine similarity at each lag)
    norms = np.linalg.norm(data_pca, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    data_normed = data_pca / norms

    autocorr = np.zeros(max_lag)
    for lag in range(max_lag):
        if lag == 0:
            autocorr[lag] = 1.0
        else:
            dots = np.sum(data_normed[:-lag] * data_normed[lag:], axis=1)
            autocorr[lag] = np.mean(dots)

    return autocorr


def compute_recurrence_matrix(data, threshold_pct=10):
    """Compute recurrence plot — binary matrix of state recurrences.

    Points where distance < threshold_pct percentile of all distances.
    """
    n_comp = min(10, data.shape[1], len(data) - 1)
    pca = PCA(n_components=n_comp)
    data_pca = pca.fit_transform(data)

    # Subsample if too large
    if len(data_pca) > 1000:
        idx = np.linspace(0, len(data_pca) - 1, 1000, dtype=int)
        data_pca = data_pca[idx]

    D = squareform(pdist(data_pca, 'euclidean'))
    threshold = np.percentile(D[D > 0], threshold_pct)
    R = (D < threshold).astype(float)
    return R, D


def spectral_circularity(data, k=10):
    """Detect circular topology via spectral analysis of kNN graph Laplacian.

    A ring manifold has a characteristic eigenvalue gap: the first two
    non-zero eigenvalues of the graph Laplacian are nearly equal (both
    correspond to sin/cos of the circular coordinate).

    Returns: eigenvalue ratio (close to 1.0 = circular), eigenvalues
    """
    n_comp = min(15, data.shape[1], len(data) - 1)
    pca = PCA(n_components=n_comp)
    data_pca = pca.fit_transform(data)

    n = len(data_pca)
    k = min(k, n - 1)

    # Build kNN graph
    nn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    nn.fit(data_pca)
    distances, indices = nn.kneighbors(data_pca)

    # Build adjacency matrix (symmetric)
    W = np.zeros((n, n))
    for i in range(n):
        for j_idx in range(k):
            j = indices[i, j_idx]
            d = distances[i, j_idx]
            weight = np.exp(-d ** 2 / (2 * np.median(distances[:, 1]) ** 2))
            W[i, j] = max(W[i, j], weight)
            W[j, i] = max(W[j, i], weight)

    # Graph Laplacian
    D = np.diag(W.sum(axis=1))
    L = D - W

    # Eigenvalues (smallest)
    n_eigs = min(8, n - 1)
    eigenvalues = np.sort(np.linalg.eigvalsh(L))[:n_eigs]

    # Circularity: ratio of 2nd to 3rd smallest non-zero eigenvalue
    # For a circle, lambda_1 ≈ lambda_2 (both ≈ (2*pi/n)^2)
    # and lambda_3 ≈ 4*lambda_1
    nonzero = eigenvalues[eigenvalues > 1e-8]
    if len(nonzero) >= 3:
        ratio_12 = nonzero[1] / nonzero[0] if nonzero[0] > 0 else 0
        ratio_23 = nonzero[2] / nonzero[1] if nonzero[1] > 0 else 0
        circularity = nonzero[0] / nonzero[1] if nonzero[1] > 0 else 0
    else:
        ratio_12 = 0
        ratio_23 = 0
        circularity = 0

    return circularity, eigenvalues, nonzero


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Manifold Geometry Detection")
    print("=" * 50)

    sp = paths_config['single_probe']['coordinates_1']['mouse01']['sessions']
    sc = sp['session_1']
    sorted_path = Path(sc['sorted'])
    beh_path = sc['behavior']

    sorting = se.read_kilosort(sorted_path)
    lha_ids, rsp_ids = get_good_units_by_region(sorted_path)

    exc_df = pd.read_csv("data/excursions_session_1.csv")
    complete = exc_df[exc_df['label'] == 'complete'].sort_values('duration', ascending=False)

    top_exc = complete.head(6)
    print(f"  Top 6 longest complete excursions:")
    for _, row in top_exc.iterrows():
        print(f"    Exc {int(row['excursion_id'])}: {row['duration']:.0f}s")

    beh_times, beh_zones = load_behavior_zones(beh_path)

    resolutions = [
        {'bin_ms': 50, 'smooth_ms': 200, 'label': '50ms/200ms'},
        {'bin_ms': 200, 'smooth_ms': 500, 'label': '200ms/500ms'},
        {'bin_ms': 500, 'smooth_ms': 1000, 'label': '500ms/1s'},
    ]

    for region, region_label, unit_ids in [('lha', 'LHA', lha_ids),
                                            ('rsp', 'RSP', rsp_ids)]:
        if len(unit_ids) < 5:
            continue

        print(f"\n{'='*60}")
        print(f"  {region_label}: {len(unit_ids)} neurons")
        print(f"{'='*60}")

        for res in resolutions:
            print(f"\n  Resolution: {res['label']}")

            zscore, time_sec = bin_and_smooth_flexible(
                sorting, unit_ids,
                bin_ms=res['bin_ms'], smooth_ms=res['smooth_ms'])

            n_exc = len(top_exc)

            # Figure: n_exc rows x 6 cols
            # Col 0: UMAP 2D scatter (zone color)
            # Col 1: UMAP 2D scatter (time color)
            # Col 2: UMAP 3D scatter (zone color)
            # Col 3: Isomap 2D scatter
            # Col 4: Recurrence plot
            # Col 5: Autocorrelation + spectral circularity
            fig = plt.figure(figsize=(30, 5 * n_exc + 1))
            fig.suptitle(f"{region_label} Session 1 — Manifold Geometry\n"
                         f"Res: {res['label']}, {len(unit_ids)} neurons, "
                         f"UMAP min_dist=0.0",
                         fontsize=14, fontweight='bold')

            for ei, (_, erow) in enumerate(top_exc.iterrows()):
                eid = int(erow['excursion_id'])
                exc_data, exc_times = extract_excursion_data(
                    zscore, time_sec, erow['start_time'], erow['end_time'])

                if len(exc_data) < 10:
                    print(f"    Exc {eid}: {len(exc_data)} pts, skip")
                    continue

                exc_zones = map_zones(exc_times, beh_times, beh_zones)
                zone_colors = [ZONE_COLORS[z] for z in exc_zones]
                time_frac = np.linspace(0, 1, len(exc_data))

                print(f"    Exc {eid}: {len(exc_data)} pts, {erow['duration']:.0f}s")

                # --- UMAP 2D (zone color) ---
                nn = min(30, len(exc_data) - 1)
                reducer = umap.UMAP(n_components=2, n_neighbors=nn,
                                     min_dist=0.0, spread=0.5,
                                     metric='euclidean', random_state=42)
                emb_umap = reducer.fit_transform(exc_data)

                ax = fig.add_subplot(n_exc, 6, ei * 6 + 1)
                ax.scatter(emb_umap[:, 0], emb_umap[:, 1],
                           c=zone_colors, s=20, alpha=0.8, rasterized=True)
                ax.set_xlabel('UMAP 1', fontsize=8)
                ax.set_ylabel('UMAP 2', fontsize=8)
                ax.set_title(f'Exc {eid} ({erow["duration"]:.0f}s)\n'
                             f'UMAP — zone', fontsize=9)
                ax.tick_params(labelsize=7)

                # --- UMAP 2D (time color) ---
                ax = fig.add_subplot(n_exc, 6, ei * 6 + 2)
                sc = ax.scatter(emb_umap[:, 0], emb_umap[:, 1],
                                c=time_frac, cmap='viridis', s=20,
                                alpha=0.8, rasterized=True)
                plt.colorbar(sc, ax=ax, label='Time', shrink=0.8)
                ax.set_xlabel('UMAP 1', fontsize=8)
                ax.set_ylabel('UMAP 2', fontsize=8)
                ax.set_title(f'UMAP — time', fontsize=9)
                ax.tick_params(labelsize=7)

                # --- UMAP 3D (zone color) ---
                reducer3d = umap.UMAP(n_components=3, n_neighbors=nn,
                                       min_dist=0.0, spread=0.5,
                                       metric='euclidean', random_state=42)
                emb_3d = reducer3d.fit_transform(exc_data)

                ax = fig.add_subplot(n_exc, 6, ei * 6 + 3, projection='3d')
                ax.scatter(emb_3d[:, 0], emb_3d[:, 1], emb_3d[:, 2],
                           c=zone_colors, s=12, alpha=0.7, rasterized=True)
                ax.view_init(elev=25, azim=45)
                ax.set_xlabel('U1', fontsize=7)
                ax.set_ylabel('U2', fontsize=7)
                ax.set_zlabel('U3', fontsize=7)
                ax.set_title('UMAP 3D — zone', fontsize=9)
                ax.tick_params(labelsize=6)

                # --- Isomap 2D ---
                n_iso = min(10, len(exc_data) - 1)
                ax = fig.add_subplot(n_exc, 6, ei * 6 + 4)
                try:
                    iso = Isomap(n_components=2, n_neighbors=n_iso)
                    emb_iso = iso.fit_transform(exc_data)
                    ax.scatter(emb_iso[:, 0], emb_iso[:, 1],
                               c=zone_colors, s=20, alpha=0.8, rasterized=True)
                    ax.set_title('Isomap — zone', fontsize=9)
                except Exception:
                    ax.text(0.5, 0.5, 'Isomap failed', ha='center',
                            va='center', transform=ax.transAxes)
                    ax.set_title('Isomap — failed', fontsize=9)
                ax.set_xlabel('Iso 1', fontsize=8)
                ax.set_ylabel('Iso 2', fontsize=8)
                ax.tick_params(labelsize=7)

                # --- Recurrence plot ---
                ax = fig.add_subplot(n_exc, 6, ei * 6 + 5)
                R, D = compute_recurrence_matrix(exc_data, threshold_pct=10)
                ax.imshow(R, cmap='binary', aspect='auto', origin='lower',
                          interpolation='none')
                ax.set_xlabel('Time bin', fontsize=8)
                ax.set_ylabel('Time bin', fontsize=8)
                ax.set_title('Recurrence Plot\n(10th pctile threshold)', fontsize=9)
                ax.tick_params(labelsize=7)

                # --- Autocorrelation + spectral ---
                ax = fig.add_subplot(n_exc, 6, ei * 6 + 6)
                max_lag = min(len(exc_data) // 2, 200)
                if max_lag > 5:
                    autocorr = recurrence_analysis(exc_data, max_lag=max_lag)
                    lag_times = np.arange(max_lag) * res['bin_ms'] / 1000
                    ax.plot(lag_times, autocorr, 'b-', linewidth=1)

                    # Find peaks in autocorrelation
                    if len(autocorr) > 10:
                        peaks, props = find_peaks(autocorr[5:],
                                                   height=0.1, distance=5)
                        peaks += 5
                        if len(peaks) > 0:
                            ax.scatter(lag_times[peaks], autocorr[peaks],
                                       c='red', s=30, zorder=3,
                                       label=f'{len(peaks)} peaks')
                            ax.legend(fontsize=7)

                    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
                    ax.set_xlabel('Lag (s)', fontsize=8)
                    ax.set_ylabel('Autocorrelation', fontsize=8)

                    # Spectral circularity
                    if len(exc_data) > 20:
                        circ, evals, nonzero_evals = spectral_circularity(
                            exc_data, k=min(10, len(exc_data) - 1))
                        circ_str = f'λ1/λ2={circ:.2f}'
                        if circ > 0.7:
                            circ_str += ' (CIRCULAR?)'
                        ax.set_title(f'Autocorrelation\n{circ_str}', fontsize=9)
                    else:
                        ax.set_title('Autocorrelation', fontsize=9)
                else:
                    ax.text(0.5, 0.5, 'Too few pts', ha='center',
                            va='center', transform=ax.transAxes)
                    ax.set_title('Autocorrelation', fontsize=9)
                ax.tick_params(labelsize=7)

            # Zone legend
            legend_elements = [
                Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=ZONE_COLORS[z], markersize=8,
                       label=ZONE_NAMES[z])
                for z in [1, 2, 3, 4, 0]
            ]
            fig.legend(handles=legend_elements, loc='lower center', ncol=5,
                       fontsize=10, bbox_to_anchor=(0.5, -0.01))

            plt.tight_layout(rect=[0, 0.02, 1, 0.94])
            res_tag = f"{res['bin_ms']}ms"
            outpath = (Path("figures") /
                       f"manifold_geometry_{region}_s1_{res_tag}.png")
            fig.savefig(outpath, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"    Saved: {outpath}")

    print("\nDone!")


if __name__ == "__main__":
    main()
