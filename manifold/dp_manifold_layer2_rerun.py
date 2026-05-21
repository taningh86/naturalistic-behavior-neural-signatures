"""
Re-run Layer 2 behavioral mapping with cleaned variable set.
- Drop: x/y position, dist_nearest_pot, dist_home, time_in_session, time_since_pot
- Keep: velocity, heading_sin, heading_cos, compartment
- Add: scored behavioral annotations that have events in this session
Reuses saved PCA/UMAP from dp_manifold_geometry.py (no Layer 1a recomputation).
"""

import yaml
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, balanced_accuracy_score, silhouette_score
from sklearn.metrics import mutual_info_score
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from dp_avalanche_criticality import (
    get_good_units_p0, get_good_units_p1_lha,
    load_spike_times_for_region, FS,
)
import spikeinterface.extractors as se

with open("paths.yaml") as f:
    cfg = yaml.safe_load(f)
sessions_cfg = cfg["double_probe"]["coordinates_1"]["mouse01"]["sessions"]

BIN_MS = 50.0
SMOOTH_SIGMA = 1.0
N_KMEANS = 20
N_CV_FOLDS = 5
MIN_EVENTS = 50  # Minimum frames for a behavioral annotation to be included

outdir = Path("data/manifold")
figdir = Path("figures/manifold")


def load_and_preprocess(session_num, region):
    """Load spikes, bin, smooth, z-score."""
    sval = sessions_cfg[f"session_{session_num}"]
    if region == 'ACA':
        sp = Path(sval['probe_0_aca']['sorted'])
        uids = get_good_units_p0(sp)
    else:
        sp = Path(sval['probe_1_lha_rsp']['sorted'])
        uids = get_good_units_p1_lha(sp)
    sorting = se.read_kilosort(sp)
    avail = set(sorting.get_unit_ids())
    uids = np.array([u for u in uids if u in avail])
    spike_dict = load_spike_times_for_region(sorting, uids)
    all_sp = np.concatenate(list(spike_dict.values()))
    dur = float(all_sp.max()) + 1.0

    dt = BIN_MS / 1000.0
    n_bins = int(dur / dt)
    bin_edges = np.arange(0, n_bins + 1) * dt
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    unit_ids = sorted(spike_dict.keys())
    matrix = np.zeros((n_bins, len(unit_ids)))
    for j, uid in enumerate(unit_ids):
        counts, _ = np.histogram(spike_dict[uid], bins=bin_edges)
        matrix[:, j] = gaussian_filter1d(counts.astype(float), sigma=SMOOTH_SIGMA)
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    stds[stds == 0] = 1.0
    matrix = (matrix - means) / stds
    return matrix, bin_centers, len(unit_ids)


def load_behavioral(session_num):
    """Load EthoVision xlsx."""
    sval = sessions_cfg[f"session_{session_num}"]
    raw = pd.read_excel(sval['behavior'], header=None)
    col_names = list(raw.iloc[34].values)
    data = raw.iloc[36:].copy()
    data.columns = col_names
    data = data.reset_index(drop=True)
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    return data


def align_to_bins(behav_df, bin_centers):
    """Nearest-sample alignment."""
    behav_times = behav_df['Trial time'].values.astype(float)
    indices = np.searchsorted(behav_times, bin_centers, side='left')
    indices = np.clip(indices, 0, len(behav_times) - 1)
    prev = np.clip(indices - 1, 0, len(behav_times) - 1)
    use_prev = np.abs(behav_times[prev] - bin_centers) < np.abs(behav_times[indices] - bin_centers)
    indices[use_prev] = prev[use_prev]
    return indices


def extract_variables(behav_df, indices):
    """Extract cleaned behavioral variables:
    - Kinematic: velocity, heading_sin, heading_cos
    - Compartment (categorical)
    - Scored behavioral annotations with sufficient events
    """
    variables = {}

    # --- Kinematics ---
    variables['velocity'] = {
        'values': behav_df['Velocity(Center-point)'].values[indices].astype(float),
        'type': 'continuous', 'unit': 'cm/s'
    }
    direction = behav_df['Direction'].values[indices].astype(float)
    direction_rad = np.deg2rad(direction)
    variables['heading_sin'] = {
        'values': np.sin(direction_rad), 'type': 'continuous', 'unit': 'sin(deg)'
    }
    variables['heading_cos'] = {
        'values': np.cos(direction_rad), 'type': 'continuous', 'unit': 'cos(deg)'
    }

    # --- Compartment (categorical) ---
    pot_zone_cols = [c for c in behav_df.columns
                     if 'Zone(Pot-' in str(c) and ' zone' not in c
                     and 'Distance' not in str(c)]
    home_col = [c for c in behav_df.columns if 'Zone(Home' in str(c)
                and 'corner' not in c and 'Distance' not in str(c)]
    ladder_col = [c for c in behav_df.columns if 'Zone(ladder' in str(c)
                  and 'Distance' not in str(c)]

    compartment = np.full(len(indices), 'Arena', dtype=object)
    if home_col:
        compartment[behav_df[home_col[0]].values[indices] == 1] = 'Home'
    if ladder_col:
        compartment[behav_df[ladder_col[0]].values[indices] == 1] = 'Ladder'
    if pot_zone_cols:
        at_pot = np.zeros(len(indices), dtype=bool)
        for c in pot_zone_cols:
            at_pot |= (behav_df[c].values[indices] == 1)
        compartment[at_pot] = 'AtPot'

    variables['compartment'] = {
        'values': compartment, 'type': 'categorical',
        'classes': ['Home', 'Ladder', 'Arena', 'AtPot']
    }

    # --- Scored behavioral annotations ---
    # These are columns after the distance-to-zone columns
    # Identify by checking: not Zone, not Distance, not position/velocity/area columns
    skip_prefixes = ['Trial time', 'Recording', 'X ', 'Y ', 'Area', 'Elongation',
                     'Direction', 'Distance', 'Velocity', 'Zone(', 'Result']
    for col in behav_df.columns:
        if col is None or str(col) == 'nan':
            continue
        col_str = str(col)
        # Skip non-behavioral columns
        if any(col_str.startswith(p) for p in skip_prefixes):
            continue
        # Check for sufficient events
        vals_full = behav_df[col].values
        n_active_full = (vals_full == 1).sum()
        if n_active_full < MIN_EVENTS:
            continue

        # This is a scored behavior with enough events
        vals_binned = vals_full[indices].astype(float)
        n_active = (vals_binned == 1).sum()
        pct = 100.0 * n_active / len(vals_binned)

        # Clean the name
        clean_name = col_str.strip().replace(' ', '_').lower()

        variables[clean_name] = {
            'values': vals_binned.astype(str),  # Categorical: "0.0" / "1.0"
            'type': 'categorical',
            'classes': ['0.0', '1.0'],
            'n_active': int(n_active),
            'pct_active': round(pct, 1),
        }

    return variables


def temporal_cv_folds(n, k=N_CV_FOLDS):
    folds = []
    sz = n // k
    for i in range(k):
        start = i * sz
        end = (i + 1) * sz if i < k - 1 else n
        test = np.arange(start, end)
        train = np.concatenate([np.arange(0, start), np.arange(end, n)])
        folds.append((train, test))
    return folds


def decode_continuous(pcs, target, folds):
    valid = ~np.isnan(target)
    if valid.sum() < 100:
        return np.nan, []
    r2s = []
    for tr, te in folds:
        trv, tev = valid[tr], valid[te]
        if trv.sum() < 50 or tev.sum() < 20:
            continue
        m = Ridge(alpha=1.0)
        m.fit(pcs[tr][trv], target[tr][trv])
        r2s.append(r2_score(target[te][tev], m.predict(pcs[te][tev])))
    return float(np.mean(r2s)) if r2s else np.nan, r2s


def decode_categorical(pcs, labels, folds):
    le = LabelEncoder()
    enc = le.fit_transform(labels)
    if len(le.classes_) < 2:
        return np.nan, []
    accs = []
    for tr, te in folds:
        if len(set(enc[tr])) < 2 or len(set(enc[te])) < 2:
            continue
        sc = StandardScaler()
        m = LinearSVC(max_iter=5000, random_state=42)
        m.fit(sc.fit_transform(pcs[tr]), enc[tr])
        accs.append(balanced_accuracy_score(enc[te], m.predict(sc.transform(pcs[te]))))
    return float(np.mean(accs)) if accs else np.nan, accs


def compute_mi(cluster_labels, variable, is_cat=False):
    valid = ~np.isnan(variable) if not is_cat else np.ones(len(variable), dtype=bool)
    if valid.sum() < 100:
        return np.nan
    cl = cluster_labels[valid]
    var = variable[valid]
    if not is_cat:
        try:
            vb = pd.qcut(var, q=10, labels=False, duplicates='drop')
        except ValueError:
            vb = pd.cut(var, bins=10, labels=False)
        vb = np.nan_to_num(vb, nan=0).astype(int)
    else:
        vb = LabelEncoder().fit_transform(var)
    return float(mutual_info_score(cl, vb))


def run_layer2(session_num, region, matrix, behav_vars, K):
    """Run behavioral mapping with cleaned variables."""
    pca = PCA(n_components=K)
    pcs = pca.fit_transform(matrix)
    kmeans = KMeans(n_clusters=N_KMEANS, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(pcs)
    folds = temporal_cv_folds(len(matrix))

    results = {}
    print(f"    Variables ({len(behav_vars)}):")
    for vn, vi in behav_vars.items():
        vals = vi['values']
        vtype = vi['type']
        extra = ""
        if 'pct_active' in vi:
            extra = f" [{vi['n_active']} bins, {vi['pct_active']}%]"

        if vtype == 'continuous':
            r2, _ = decode_continuous(pcs, vals, folds)
            mi = compute_mi(clusters, vals)
            results[vn] = {'type': 'continuous', 'decodability': r2, 'MI': mi}
            print(f"      {vn}: R2={r2:.3f}, MI={mi:.4f}{extra}")
        elif vtype == 'categorical':
            acc, _ = decode_categorical(pcs, vals, folds)
            mi = compute_mi(clusters, vals, is_cat=True)
            if len(np.unique(vals)) >= 2:
                le2 = LabelEncoder()
                sil = float(silhouette_score(pcs, le2.fit_transform(vals),
                                             sample_size=min(5000, len(pcs))))
            else:
                sil = np.nan
            results[vn] = {'type': 'categorical', 'decodability': acc, 'MI': mi,
                           'silhouette': sil}
            print(f"      {vn}: Acc={acc:.3f}, MI={mi:.4f}, Sil={sil:.3f}{extra}")

    # Null baseline
    print("    Null baseline...", end='', flush=True)
    perm = np.random.RandomState(42).permutation(len(clusters))
    null_cl = clusters[perm]
    for vn, vi in behav_vars.items():
        vals = vi['values']
        shuffled = vals[perm]
        if vi['type'] == 'continuous':
            r2_null, _ = decode_continuous(pcs, shuffled, folds)
            mi_null = compute_mi(null_cl, vals)
        else:
            r2_null, _ = decode_categorical(pcs, shuffled, folds)
            mi_null = compute_mi(null_cl, vals, is_cat=True)
        results[vn]['decodability_null'] = r2_null
        results[vn]['MI_null'] = mi_null
    print(" done.")

    results['_K'] = K
    return results


def plot_ranking(results, session_num, region):
    """Ranking figure."""
    var_items = [(k, v) for k, v in results.items() if not k.startswith('_')]
    var_items.sort(key=lambda x: x[1].get('decodability', 0) or 0, reverse=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, max(5, 0.4 * len(var_items))))

    # Decodability
    ax = axes[0]
    names = [v[0] for v in var_items]
    scores = [v[1].get('decodability', 0) or 0 for v in var_items]
    nulls = [v[1].get('decodability_null', 0) or 0 for v in var_items]
    y = range(len(names))
    ax.barh(y, scores, 0.4, label='Data', color='steelblue')
    ax.barh([i + 0.4 for i in y], nulls, 0.4, label='Null', color='gray', alpha=0.5)
    ax.set_yticks([i + 0.2 for i in y])
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('R2 / Balanced Accuracy')
    ax.set_title('Decodability Ranking', fontweight='bold')
    ax.legend(fontsize=8)
    ax.invert_yaxis()

    # MI
    ax = axes[1]
    mi_items = sorted(var_items, key=lambda x: x[1].get('MI', 0) or 0, reverse=True)
    mi_names = [v[0] for v in mi_items]
    mi_vals = [v[1].get('MI', 0) or 0 for v in mi_items]
    ax.barh(range(len(mi_names)), mi_vals, color='darkorange')
    ax.set_yticks(range(len(mi_names)))
    ax.set_yticklabels(mi_names, fontsize=9)
    ax.set_xlabel('Mutual Information (bits)')
    ax.set_title('MI Ranking', fontweight='bold')
    ax.invert_yaxis()

    fig.suptitle(f'Layer 2 (revised): Behavioral Mapping -- S{session_num} {region}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fp = figdir / f"S{session_num}_{region}_behav_ranking_v2.png"
    plt.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved {fp}")


def write_report(results, session_num, region, n_units, K):
    """Updated markdown report for Layer 2."""
    var_items = [(k, v) for k, v in results.items() if not k.startswith('_')]
    var_items.sort(key=lambda x: x[1].get('decodability', 0) or 0, reverse=True)

    lines = [
        f"# Layer 2 (revised) -- S{session_num} {region}",
        f"",
        f"K = {K} PCs, {N_KMEANS} k-means clusters. N = {n_units} units.",
        f"",
        f"Variables: compartment + kinematics (velocity, heading) + scored behavioral annotations.",
        f"Dropped: x/y position, dist_nearest_pot, dist_home, time_in_session, time_since_pot (collinear/non-behavioral).",
        f"",
        f"| Variable | Type | Decodability | Null | MI |",
        f"|----------|------|-------------|------|-----|",
    ]
    for vn, vi in var_items:
        d = vi.get('decodability')
        dn = vi.get('decodability_null')
        mi = vi.get('MI')
        vt = vi.get('type', '?')
        metric = "R2" if vt == 'continuous' else "Acc"
        d_str = f"{d:.3f}" if d is not None and not np.isnan(d) else "N/A"
        dn_str = f"{dn:.3f}" if dn is not None and not np.isnan(dn) else "N/A"
        mi_str = f"{mi:.4f}" if mi is not None and not np.isnan(mi) else "N/A"
        lines.append(f"| {vn} | {vt} ({metric}) | {d_str} | {dn_str} | {mi_str} |")

    lines.extend([
        f"",
        f"**Top 5 manifold-organizing variables:**",
    ])
    for i, (vn, vi) in enumerate(var_items[:5]):
        d = vi.get('decodability', 0) or 0
        lines.append(f"  {i+1}. {vn} ({d:.3f})")

    rp = outdir / f"S{session_num}_{region}_layer2_v2.md"
    with open(rp, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"    Saved {rp}")

    jp = outdir / f"S{session_num}_{region}_layer2_v2.json"
    save = {k: v for k, v in results.items()}
    with open(jp, 'w') as f:
        json.dump(save, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else int(o) if isinstance(o, (np.integer,)) else o)
    print(f"    Saved {jp}")


def main():
    session_num = int(sys.argv[1]) if len(sys.argv) > 1 else 3

    sval = sessions_cfg[f"session_{session_num}"]
    print(f"{'='*80}")
    print(f"LAYER 2 RERUN -- S{session_num} ({sval['state']}/{sval['phase']})")
    print(f"Cleaned variables: compartment + kinematics + scored behaviors")
    print(f"{'='*80}")

    # Load behavioral data
    print("\nLoading behavioral data...")
    behav_df = load_behavioral(session_num)

    # Use intrinsic dimensionality from Layer 1a for K
    # Read from saved results
    K_values = {'ACA': 10, 'LHA': 5}  # Based on Two-NN/CorrDim from previous run
    # ACA: Two-NN=10, CorrDim=7.9 -> mean ~9 -> K=10
    # LHA: Two-NN=6.9, CorrDim=3.8 -> mean ~5 -> K=6
    # Excluding Isomap (unreliable for LHA)

    for region in ['ACA', 'LHA']:
        print(f"\n{'='*60}")
        print(f"  {region}")
        print(f"{'='*60}")

        # Load and preprocess
        print("  Loading neural data...")
        matrix, bin_centers, n_units = load_and_preprocess(session_num, region)
        print(f"    {n_units} units, {matrix.shape[0]} bins")

        # Align and extract variables
        indices = align_to_bins(behav_df, bin_centers)
        behav_vars = extract_variables(behav_df, indices)

        # Print compartment summary
        comp = behav_vars['compartment']['values']
        for label in ['Home', 'Ladder', 'Arena', 'AtPot']:
            n = (comp == label).sum()
            print(f"    {label}: {n} bins ({100*n/len(comp):.1f}%)")

        # Run Layer 2
        K = K_values[region]
        print(f"\n  Layer 2 (K={K} PCs):")
        results = run_layer2(session_num, region, matrix, behav_vars, K)

        # Figures and report
        plot_ranking(results, session_num, region)
        write_report(results, session_num, region, n_units, K)

    print(f"\n{'='*80}")
    print("DONE")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
