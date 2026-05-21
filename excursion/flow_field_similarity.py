"""
Flow Field Similarity Analysis
===============================
Compute flow field features for ALL 52 excursions in LHA (and RSP),
then rank by similarity to Excursion 81 (feeding).

Features:
  - Elongation: PC1 var / PC2 var (high = one-dimensional)
  - Convergence: fraction of flow vectors pointing toward densest region
  - Speed stats: mean, std, skewness of embedding speed
  - Dwell concentration: max_dwell / mean_dwell (peaky vs uniform)
  - Dwell entropy: Shannon entropy of dwell histogram (low = concentrated)
  - Effective dimensionality: participation ratio from PCA eigenvalues
  - Tortuosity: path length / displacement
  - Flow alignment: mean cosine similarity of consecutive velocity vectors
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
from scipy.ndimage import uniform_filter1d, gaussian_filter
from scipy.spatial.distance import cdist
from scipy.stats import skew, zscore as scipy_zscore
import spikeinterface.extractors as se
import warnings

warnings.filterwarnings('ignore')

FS = 30000
LHA_DEPTH_MAX = 1300
RSP_DEPTH_MIN = 1300

with open("paths.yaml") as f:
    paths_config = yaml.safe_load(f)

BEHAVIOR_COLORS = {
    'Feeding': '#D32F2F',
    'Digging': '#FF9800',
    'Grooming': '#4CAF50',
    'Quick arena exploration': '#00BCD4',
    'Arena wall exploration': '#9C27B0',
    'Transition wall exploration': '#2196F3',
    'Hesitant exploration': '#795548',
    'Quick one loop at home': '#E91E63',
}


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


def bin_and_smooth(sorting, unit_ids, bin_ms, smooth_ms):
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


def load_behavior_timeseries(session_num, time_sec):
    sp = paths_config['single_probe']['coordinates_1']['mouse01']['sessions']
    sc = sp[f'session_{session_num}']
    behav_path = sc.get('behavior')
    if not behav_path or not Path(behav_path).exists():
        return {}
    behav_df = pd.read_csv(behav_path, header=None)
    result = {}
    for row_idx in range(behav_df.shape[0]):
        name = str(behav_df.iloc[row_idx, 0]).strip()
        if not name or name == 'nan':
            continue
        row_data = pd.to_numeric(behav_df.iloc[row_idx, 1:], errors='coerce').values
        aligned = np.zeros(len(time_sec))
        for ti, t in enumerate(time_sec):
            bi = int(t / 0.1)
            if 0 <= bi < len(row_data) and not np.isnan(row_data[bi]):
                aligned[ti] = row_data[bi]
        result[name] = aligned
    return result


def get_dominant_behavior(behav_dict, exc_mask):
    n = exc_mask.sum()
    labels = np.full(n, 'Other', dtype=object)
    priority = ['Feeding', 'Digging', 'Grooming', 'Quick arena exploration',
                'Arena wall exploration', 'Transition wall exploration',
                'Hesitant exploration', 'Quick one loop at home']
    for bname in reversed(priority):
        if bname in behav_dict:
            bdata = behav_dict[bname][exc_mask]
            labels[bdata > 0] = bname
    return labels


def compute_flow_features(exc_data, coords_2d):
    """Compute a feature vector characterizing the flow field."""
    n_pts = len(exc_data)
    features = {}

    # --- Elongation: ratio of PC1 to PC2 variance ---
    pca_full = PCA().fit(exc_data)
    evals = pca_full.explained_variance_
    features['elongation'] = evals[0] / (evals[1] + 1e-10)

    # --- Participation ratio (effective dimensionality) ---
    evals_norm = evals / evals.sum()
    features['participation_ratio'] = 1.0 / np.sum(evals_norm**2)

    # --- Variance explained by PC1 ---
    features['pc1_var_explained'] = evals[0] / evals.sum()

    # --- Flow vectors ---
    dx = np.diff(coords_2d[:, 0])
    dy = np.diff(coords_2d[:, 1])
    speed = np.sqrt(dx**2 + dy**2)

    features['speed_mean'] = np.mean(speed)
    features['speed_std'] = np.std(speed)
    features['speed_skewness'] = skew(speed)
    features['speed_cv'] = np.std(speed) / (np.mean(speed) + 1e-10)

    # --- Tortuosity: total path length / displacement ---
    path_length = np.sum(speed)
    displacement = np.sqrt((coords_2d[-1, 0] - coords_2d[0, 0])**2 +
                           (coords_2d[-1, 1] - coords_2d[0, 1])**2)
    features['tortuosity'] = path_length / (displacement + 1e-10)

    # --- Convergence: fraction of vectors pointing toward center of mass ---
    com = coords_2d.mean(axis=0)
    n_converging = 0
    for i in range(len(dx)):
        # Vector from point to center of mass
        to_com_x = com[0] - coords_2d[i, 0]
        to_com_y = com[1] - coords_2d[i, 1]
        # Dot product with velocity
        dot = dx[i] * to_com_x + dy[i] * to_com_y
        if dot > 0:
            n_converging += 1
    features['convergence'] = n_converging / len(dx)

    # --- Flow alignment: cosine similarity of consecutive velocity vectors ---
    cos_sims = []
    for i in range(len(dx) - 1):
        mag1 = np.sqrt(dx[i]**2 + dy[i]**2)
        mag2 = np.sqrt(dx[i+1]**2 + dy[i+1]**2)
        if mag1 > 1e-10 and mag2 > 1e-10:
            cos = (dx[i]*dx[i+1] + dy[i]*dy[i+1]) / (mag1 * mag2)
            cos_sims.append(cos)
    if len(cos_sims) > 0:
        features['flow_alignment'] = np.mean(cos_sims)
        features['flow_alignment_std'] = np.std(cos_sims)
    else:
        features['flow_alignment'] = 0.0
        features['flow_alignment_std'] = 0.0

    # --- Dwell-time concentration ---
    n_grid = 30
    heatmap, _, _ = np.histogram2d(coords_2d[:, 0], coords_2d[:, 1], bins=n_grid)
    heatmap_flat = heatmap.ravel()
    occupied = heatmap_flat[heatmap_flat > 0]
    features['dwell_max_over_mean'] = occupied.max() / (occupied.mean() + 1e-10)
    # Shannon entropy of dwell distribution
    p = occupied / occupied.sum()
    features['dwell_entropy'] = -np.sum(p * np.log(p + 1e-10))
    # Fraction of bins occupied
    features['occupancy_fraction'] = len(occupied) / len(heatmap_flat)

    # --- Convex hull area (approximate via 2D range) ---
    x_range = coords_2d[:, 0].max() - coords_2d[:, 0].min()
    y_range = coords_2d[:, 1].max() - coords_2d[:, 1].min()
    features['spread_area'] = x_range * y_range

    # --- Recurrence: fraction of pairwise distances below threshold ---
    if n_pts <= 500:
        dists = cdist(coords_2d, coords_2d)
    else:
        # Subsample for speed
        idx = np.random.choice(n_pts, 500, replace=False)
        dists = cdist(coords_2d[idx], coords_2d[idx])
    thresh = np.percentile(dists, 10)
    features['recurrence_rate'] = (dists < thresh).sum() / dists.size

    # --- HD speed (in full neural space) ---
    hd_diffs = np.diff(exc_data, axis=0)
    hd_speed = np.sqrt(np.sum(hd_diffs**2, axis=1))
    features['hd_speed_mean'] = np.mean(hd_speed)
    features['hd_speed_std'] = np.std(hd_speed)
    features['hd_speed_skewness'] = skew(hd_speed)

    return features


def main():
    print("Flow Field Similarity Analysis")
    print("=" * 50)

    sp = paths_config['single_probe']['coordinates_1']['mouse01']['sessions']
    sc = sp['session_1']
    sorted_path = Path(sc['sorted'])
    sorting = se.read_kilosort(sorted_path)
    lha_ids, rsp_ids = get_good_units_by_region(sorted_path)

    exc_df = pd.read_csv("data/excursions_session_1.csv")
    complete = exc_df[exc_df['label'] == 'complete']

    # Load behavior profiles for labels
    behav_profiles = pd.read_csv("data/excursion_behavior_profiles.csv")

    for region, unit_ids in [('lha', lha_ids), ('rsp', rsp_ids)]:
        region_label = region.upper()

        for bin_ms, smooth_ms, res_label in [
            (50, 200, '50ms/200ms'),
            (200, 500, '200ms/500ms'),
        ]:
            print(f"\n  {region_label} @ {res_label}")
            zscore, time_sec = bin_and_smooth(sorting, unit_ids, bin_ms, smooth_ms)
            behav_dict = load_behavior_timeseries(1, time_sec)

            all_features = []

            for _, erow in complete.iterrows():
                eid = int(erow['excursion_id'])
                mask = (time_sec >= erow['start_time']) & (time_sec <= erow['end_time'])
                exc_data = zscore[mask]
                n_pts = len(exc_data)

                if n_pts < 15:
                    continue

                coords_2d = PCA(n_components=2).fit_transform(exc_data)
                feats = compute_flow_features(exc_data, coords_2d)
                feats['excursion_id'] = eid
                feats['duration'] = erow['duration']
                feats['n_pts'] = n_pts

                # Get dominant behavior for this excursion
                behav_labels = get_dominant_behavior(behav_dict, mask)
                unique, counts = np.unique(behav_labels, return_counts=True)
                dominant = unique[np.argmax(counts)]
                feats['dominant_behavior'] = dominant

                # Get feeding/digging fractions from profiles
                bp_row = behav_profiles[behav_profiles['excursion_id'] == eid]
                if len(bp_row) > 0:
                    feats['feeding_frac'] = bp_row.iloc[0].get('Feeding_frac', 0)
                    feats['digging_frac'] = bp_row.iloc[0].get('Digging_frac', 0)
                else:
                    feats['feeding_frac'] = 0
                    feats['digging_frac'] = 0

                all_features.append(feats)

            feat_df = pd.DataFrame(all_features)
            print(f"    Computed features for {len(feat_df)} excursions")

            # Save features
            out_csv = Path("data") / f"flow_field_features_{region}_{bin_ms}ms.csv"
            feat_df.to_csv(out_csv, index=False)
            print(f"    Saved: {out_csv}")

            # ============================================================
            # Similarity to Excursion 81
            # ============================================================
            feature_cols = [c for c in feat_df.columns if c not in
                           ['excursion_id', 'duration', 'n_pts',
                            'dominant_behavior', 'feeding_frac', 'digging_frac']]

            # Z-score features
            feat_matrix = feat_df[feature_cols].values.astype(float)
            feat_z = scipy_zscore(feat_matrix, axis=0)
            feat_z = np.nan_to_num(feat_z, 0)

            # Find Exc 81 index
            exc81_idx = feat_df[feat_df['excursion_id'] == 81].index
            if len(exc81_idx) == 0:
                print("    Exc 81 not found, skipping similarity")
                continue
            exc81_idx = exc81_idx[0]
            exc81_z = feat_z[exc81_idx]

            # Euclidean distance to Exc 81
            distances = np.sqrt(np.sum((feat_z - exc81_z)**2, axis=1))
            feat_df['dist_to_exc81'] = distances
            feat_df['rank'] = feat_df['dist_to_exc81'].rank().astype(int)

            # Sort by similarity
            ranked = feat_df.sort_values('dist_to_exc81')

            print(f"\n    Top 15 most similar to Exc 81 ({region_label}):")
            print(f"    {'Rank':<5} {'ExcID':<7} {'Dist':<7} {'Dur(s)':<8} "
                  f"{'Elong':<7} {'PR':<6} {'Conv':<6} {'Tort':<7} "
                  f"{'DwellEnt':<9} {'FlowAln':<8} {'Behav':<20} {'Feed%':<7}")
            print("    " + "-" * 105)
            for i, (_, row) in enumerate(ranked.head(15).iterrows()):
                print(f"    {i+1:<5} {int(row['excursion_id']):<7} "
                      f"{row['dist_to_exc81']:<7.2f} {row['duration']:<8.1f} "
                      f"{row['elongation']:<7.1f} {row['participation_ratio']:<6.1f} "
                      f"{row['convergence']:<6.2f} {row['tortuosity']:<7.1f} "
                      f"{row['dwell_entropy']:<9.2f} {row['flow_alignment']:<8.3f} "
                      f"{row['dominant_behavior']:<20} {row['feeding_frac']:<7.2f}")

            print(f"\n    Bottom 5 (most different from Exc 81):")
            for i, (_, row) in enumerate(ranked.tail(5).iterrows()):
                r = len(ranked) - 4 + i
                print(f"    {r:<5} {int(row['excursion_id']):<7} "
                      f"{row['dist_to_exc81']:<7.2f} {row['duration']:<8.1f} "
                      f"{row['elongation']:<7.1f} {row['participation_ratio']:<6.1f} "
                      f"{row['convergence']:<6.2f} {row['tortuosity']:<7.1f} "
                      f"{row['dwell_entropy']:<9.2f} {row['flow_alignment']:<8.3f} "
                      f"{row['dominant_behavior']:<20} {row['feeding_frac']:<7.2f}")

            # ============================================================
            # FIGURE 1: Feature comparison — Exc 81 vs all others
            # ============================================================
            n_feats = len(feature_cols)
            n_cols = 5
            n_rows = int(np.ceil(n_feats / n_cols))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(28, 4 * n_rows))
            fig.suptitle(
                f"{region_label} — Flow Field Features Across All Excursions\n"
                f"Excursion 81 (Feeding) highlighted in red | {res_label}",
                fontsize=14, fontweight='bold', y=0.98)

            for ax_i, feat_name in enumerate(feature_cols):
                ax = axes.ravel()[ax_i]
                vals = feat_df[feat_name].values
                exc_ids = feat_df['excursion_id'].values
                colors = []
                for eid in exc_ids:
                    if eid == 81:
                        colors.append('#D32F2F')
                    elif eid == 57:
                        colors.append('#FF9800')
                    else:
                        colors.append('#90CAF9')

                ax.bar(range(len(vals)), vals, color=colors, edgecolor='none', alpha=0.8)
                # Highlight Exc 81 value
                idx81 = np.where(exc_ids == 81)[0]
                if len(idx81) > 0:
                    ax.axhline(vals[idx81[0]], color='#D32F2F', linestyle='--',
                               linewidth=1.5, alpha=0.7)
                ax.set_title(feat_name, fontsize=9, fontweight='bold')
                ax.set_xticks([])
                ax.tick_params(labelsize=7)

            # Hide unused axes
            for ax_i in range(len(feature_cols), len(axes.ravel())):
                axes.ravel()[ax_i].set_visible(False)

            plt.tight_layout(rect=[0, 0, 1, 0.94])
            out_fig1 = Path("figures") / f"flow_field_features_{region}_{bin_ms}ms.png"
            fig.savefig(out_fig1, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"\n    Saved: {out_fig1}")

            # ============================================================
            # FIGURE 2: Similarity ranking scatter
            # ============================================================
            fig, axes = plt.subplots(2, 3, figsize=(22, 13))
            fig.suptitle(
                f"{region_label} — Similarity to Excursion 81 (Feeding) | {res_label}",
                fontsize=14, fontweight='bold', y=0.98)

            # Panel (0,0): Distance ranking bar chart
            ax = axes[0, 0]
            ranked_reset = ranked.reset_index(drop=True)
            bar_colors = []
            for _, row in ranked_reset.iterrows():
                eid = int(row['excursion_id'])
                if eid == 81:
                    bar_colors.append('#D32F2F')
                elif eid == 57:
                    bar_colors.append('#FF9800')
                elif row['dist_to_exc81'] < ranked_reset['dist_to_exc81'].quantile(0.2):
                    bar_colors.append('#4CAF50')
                else:
                    bar_colors.append('#90CAF9')
            ax.barh(range(len(ranked_reset)), ranked_reset['dist_to_exc81'].values,
                    color=bar_colors, edgecolor='none', alpha=0.8)
            # Label top 10
            for i in range(min(10, len(ranked_reset))):
                eid = int(ranked_reset.iloc[i]['excursion_id'])
                ax.text(ranked_reset.iloc[i]['dist_to_exc81'] + 0.1, i,
                        f"Exc {eid}", fontsize=7, va='center')
            ax.set_xlabel('Distance to Exc 81', fontsize=10)
            ax.set_ylabel('Rank', fontsize=10)
            ax.set_title('Similarity Ranking\n(green = top 20%)', fontsize=11)
            ax.invert_yaxis()

            # Panel (0,1): Elongation vs Participation Ratio
            ax = axes[0, 1]
            for _, row in feat_df.iterrows():
                eid = int(row['excursion_id'])
                c = '#D32F2F' if eid == 81 else '#FF9800' if eid == 57 else '#90CAF9'
                s = 200 if eid == 81 else 120 if eid == 57 else 40
                z = 10 if eid in [81, 57] else 1
                ax.scatter(row['elongation'], row['participation_ratio'],
                           c=c, s=s, zorder=z, edgecolors='black', linewidths=0.5)
                if eid in [81, 57] or row['dist_to_exc81'] < ranked['dist_to_exc81'].quantile(0.15):
                    ax.annotate(f"{eid}", (row['elongation'], row['participation_ratio']),
                                fontsize=7, ha='left', va='bottom')
            ax.set_xlabel('Elongation (PC1/PC2 var ratio)', fontsize=10)
            ax.set_ylabel('Participation Ratio', fontsize=10)
            ax.set_title('Shape: Elongation vs Dimensionality', fontsize=11)

            # Panel (0,2): Convergence vs Flow alignment
            ax = axes[0, 2]
            for _, row in feat_df.iterrows():
                eid = int(row['excursion_id'])
                c = '#D32F2F' if eid == 81 else '#FF9800' if eid == 57 else '#90CAF9'
                s = 200 if eid == 81 else 120 if eid == 57 else 40
                z = 10 if eid in [81, 57] else 1
                ax.scatter(row['convergence'], row['flow_alignment'],
                           c=c, s=s, zorder=z, edgecolors='black', linewidths=0.5)
                if eid in [81, 57] or row['dist_to_exc81'] < ranked['dist_to_exc81'].quantile(0.15):
                    ax.annotate(f"{eid}", (row['convergence'], row['flow_alignment']),
                                fontsize=7, ha='left', va='bottom')
            ax.set_xlabel('Convergence (frac toward COM)', fontsize=10)
            ax.set_ylabel('Flow Alignment (cos sim)', fontsize=10)
            ax.set_title('Dynamics: Convergence vs Persistence', fontsize=11)

            # Panel (1,0): Tortuosity vs Dwell entropy
            ax = axes[1, 0]
            for _, row in feat_df.iterrows():
                eid = int(row['excursion_id'])
                c = '#D32F2F' if eid == 81 else '#FF9800' if eid == 57 else '#90CAF9'
                s = 200 if eid == 81 else 120 if eid == 57 else 40
                z = 10 if eid in [81, 57] else 1
                ax.scatter(row['tortuosity'], row['dwell_entropy'],
                           c=c, s=s, zorder=z, edgecolors='black', linewidths=0.5)
                if eid in [81, 57] or row['dist_to_exc81'] < ranked['dist_to_exc81'].quantile(0.15):
                    ax.annotate(f"{eid}", (row['tortuosity'], row['dwell_entropy']),
                                fontsize=7, ha='left', va='bottom')
            ax.set_xlabel('Tortuosity (path/displacement)', fontsize=10)
            ax.set_ylabel('Dwell Entropy', fontsize=10)
            ax.set_title('Trajectory: Complexity vs Spread', fontsize=11)

            # Panel (1,1): Duration vs Distance to Exc 81
            ax = axes[1, 1]
            for _, row in feat_df.iterrows():
                eid = int(row['excursion_id'])
                c = '#D32F2F' if eid == 81 else '#FF9800' if eid == 57 else '#90CAF9'
                s = 200 if eid == 81 else 120 if eid == 57 else 40
                z = 10 if eid in [81, 57] else 1
                ax.scatter(row['duration'], row['dist_to_exc81'],
                           c=c, s=s, zorder=z, edgecolors='black', linewidths=0.5)
                if eid in [81, 57] or row['dist_to_exc81'] < ranked['dist_to_exc81'].quantile(0.15):
                    ax.annotate(f"{eid}", (row['duration'], row['dist_to_exc81']),
                                fontsize=7, ha='left', va='bottom')
            ax.set_xlabel('Duration (s)', fontsize=10)
            ax.set_ylabel('Distance to Exc 81', fontsize=10)
            ax.set_title('Duration vs Similarity', fontsize=11)

            # Panel (1,2): PCA of feature space — 2D map of all excursions
            ax = axes[1, 2]
            feat_pca = PCA(n_components=2).fit_transform(feat_z)
            for i, (_, row) in enumerate(feat_df.iterrows()):
                eid = int(row['excursion_id'])
                c = '#D32F2F' if eid == 81 else '#FF9800' if eid == 57 else '#90CAF9'
                s = 200 if eid == 81 else 120 if eid == 57 else 40
                z = 10 if eid in [81, 57] else 1
                ax.scatter(feat_pca[i, 0], feat_pca[i, 1],
                           c=c, s=s, zorder=z, edgecolors='black', linewidths=0.5)
                if eid in [81, 57] or row['dist_to_exc81'] < ranked['dist_to_exc81'].quantile(0.15):
                    ax.annotate(f"{eid}", (feat_pca[i, 0], feat_pca[i, 1]),
                                fontsize=7, ha='left', va='bottom')
            ax.set_xlabel('Feature PC1', fontsize=10)
            ax.set_ylabel('Feature PC2', fontsize=10)
            ax.set_title('Feature Space (PCA)\n(excursion similarity map)', fontsize=11)

            # Legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#D32F2F',
                       markersize=12, label='Exc 81 (Feeding)'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF9800',
                       markersize=10, label='Exc 57 (Digging)'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#4CAF50',
                       markersize=8, label='Top 20% similar'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#90CAF9',
                       markersize=8, label='Other'),
            ]
            ax.legend(handles=legend_elements, fontsize=8, loc='best')

            plt.tight_layout(rect=[0, 0, 1, 0.94])
            out_fig2 = Path("figures") / f"flow_field_similarity_{region}_{bin_ms}ms.png"
            fig.savefig(out_fig2, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"    Saved: {out_fig2}")

    print("\nDone!")


if __name__ == "__main__":
    main()
