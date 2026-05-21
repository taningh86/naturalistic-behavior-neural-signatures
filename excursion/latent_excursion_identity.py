"""
Investigate WHY GRU-ODE hidden states separate excursions (7.6x lift)
but fail to decode behavior/zone/speed (all at chance).

Hypotheses tested:
1. Slow drift: excursion ID decoding is just temporal position
2. Between > within variance: hidden states vary more across excursions
   than within them, leaving no room for behavior encoding
3. Hidden state geometry: excursions form tight clusters with no
   internal structure corresponding to behavior

Diagnostics:
A. Confusion matrix of excursion ID — do temporally adjacent excursions
   get confused more?
B. Pairwise hidden-state distance vs temporal distance between excursions
C. Within-excursion vs between-excursion variance decomposition
D. Remove slow drift (detrend) then re-decode behavior
E. PCA colored by excursion ID vs by behavior — visual comparison
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import spikeinterface.extractors as se
import torch
import torch.nn as nn
from torchdiffeq import odeint
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
import warnings

warnings.filterwarnings('ignore')

# Config — must match the 10ms Poisson GRU-ODE training
BIN_SIZE_MS = 10
FS = 30000
BIN_SAMPLES = int(BIN_SIZE_MS * FS / 1000)
SEQ_LEN = 50
PRED_BINS = 10
D_SHARED = 32
HIDDEN_SIZE = 32
ODE_GATE_HIDDEN = 64
ODE_SOLVER = 'rk4'
ODE_STEP_SIZE = 1.0
ODE_DT = 1.0
LHA_DEPTH_MAX = 1300
RSP_DEPTH_MIN = 1300
STRIDE = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("paths.yaml") as f:
    paths_config = yaml.safe_load(f)


# --- Model classes (same as latent_decoding.py) ---
class GRUODEFunc(nn.Module):
    def __init__(self, hidden_size, gate_hidden=64):
        super().__init__()
        self.update_gate = nn.Sequential(
            nn.Linear(hidden_size, gate_hidden), nn.Tanh(),
            nn.Linear(gate_hidden, hidden_size), nn.Sigmoid(),
        )
        self.candidate = nn.Sequential(
            nn.Linear(hidden_size, gate_hidden), nn.Tanh(),
            nn.Linear(gate_hidden, hidden_size), nn.Tanh(),
        )

    def forward(self, t, h):
        z = self.update_gate(h)
        n = self.candidate(h)
        return (1 - z) * (n - h)


class PooledGRUODE(nn.Module):
    def __init__(self, session_neuron_counts, d_shared, hidden_size,
                 gate_hidden=64, pred_steps=10):
        super().__init__()
        self.d_shared = d_shared
        self.hidden_size = hidden_size
        self.pred_steps = pred_steps
        self.input_projections = nn.ModuleDict()
        for sn, n_neurons in session_neuron_counts.items():
            self.input_projections[str(sn)] = nn.Linear(n_neurons, d_shared)
        self.ode_func = GRUODEFunc(hidden_size, gate_hidden)
        self.obs_cell = nn.GRUCell(input_size=d_shared, hidden_size=hidden_size)
        self.fc_shared = nn.Linear(hidden_size, d_shared)
        self.output_projections = nn.ModuleDict()
        for sn, n_neurons in session_neuron_counts.items():
            self.output_projections[str(sn)] = nn.Linear(d_shared, n_neurons)
        self.register_buffer('t_span', torch.tensor([0.0, ODE_DT]))

    def _ode_evolve(self, h):
        h_evolved = odeint(
            self.ode_func, h, self.t_span,
            method=ODE_SOLVER, options={'step_size': ODE_STEP_SIZE},
        )
        return h_evolved[-1]

    def extract_hidden_states(self, x, session_num):
        sn_key = str(session_num)
        with torch.no_grad():
            h = torch.zeros(x.shape[0], self.hidden_size, device=x.device)
            for k in range(x.shape[1]):
                h = self._ode_evolve(h)
                x_proj = self.input_projections[sn_key](x[:, k, :])
                h = self.obs_cell(x_proj, h)
            return h


# --- Data loading ---
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


def bin_spike_trains(sorting, unit_ids):
    spike_trains = {}
    all_min, all_max = np.inf, 0
    for uid in unit_ids:
        st = sorting.get_unit_spike_train(uid)
        spike_trains[uid] = st
        if len(st) > 0:
            all_min = min(all_min, np.min(st))
            all_max = max(all_max, np.max(st))
    n_bins = int((all_max - all_min) / BIN_SAMPLES) + 1
    data = np.zeros((n_bins, len(unit_ids)), dtype=np.float32)
    for i, uid in enumerate(unit_ids):
        st = spike_trains[uid]
        if len(st) > 0:
            b = ((st - all_min) // BIN_SAMPLES).astype(int)
            b = b[(b >= 0) & (b < n_bins)]
            np.add.at(data[:, i], b, 1)
    means = data.mean(axis=0, keepdims=True)
    stds = data.std(axis=0, keepdims=True)
    stds[stds < 1e-8] = 1.0
    zscore_data = (data - means) / stds
    time_sec = (np.arange(n_bins) * BIN_SIZE_MS / 1000) + (all_min / FS)
    return zscore_data, time_sec, n_bins


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


def get_behavior_labels(behav_dict, n_pts):
    labels = np.full(n_pts, 'Other', dtype=object)
    priority = ['Feeding', 'Digging', 'Grooming', 'Quick arena exploration',
                'Arena wall exploration', 'Transition wall exploration',
                'Hesitant exploration', 'Quick one loop at home']
    for bname in reversed(priority):
        if bname in behav_dict:
            labels[behav_dict[bname] > 0] = bname
    return labels


def get_zone_labels(behav_dict, n_pts):
    labels = np.full(n_pts, 'Other', dtype=object)
    for zone in ['Home', 'Ladder', 'Transition zone', 'Foraging arena']:
        if zone in behav_dict:
            labels[behav_dict[zone] > 0] = zone
    return labels


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("  Why Do Hidden States Separate Excursions But Not Behaviors?")
    print("=" * 70)

    sp = paths_config['single_probe']['coordinates_1']['mouse01']['sessions']
    sc = sp['session_1']
    sorted_path = Path(sc['sorted'])
    sorting = se.read_kilosort(sorted_path)
    lha_ids, rsp_ids = get_good_units_by_region(sorted_path)

    exc_df = pd.read_csv("data/excursions_session_1.csv")
    complete = exc_df[exc_df['label'] == 'complete']

    for region, unit_ids in [('lha', lha_ids), ('rsp', rsp_ids)]:
        region_label = region.upper()
        print(f"\n{'='*70}")
        print(f"  {region_label} — {len(unit_ids)} neurons")
        print(f"{'='*70}")

        # Load model
        model_path = Path("data") / f"gru_ode_10ms_poisson_{region}_fed_model.pt"
        if not model_path.exists():
            print(f"  Model not found: {model_path}")
            continue

        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        model = PooledGRUODE(
            checkpoint['neuron_counts'],
            checkpoint['config']['d_shared'],
            checkpoint['config']['hidden_size'],
            checkpoint['config'].get('gate_hidden', ODE_GATE_HIDDEN),
            checkpoint['config'].get('pred_bins', PRED_BINS),
        ).to(DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Extract hidden states
        s_data, s_time, s_bins = bin_spike_trains(sorting, unit_ids)
        print(f"  Extracting hidden states ({s_bins} bins, stride={STRIDE})...")

        seqs = []
        for i in range(0, s_bins - SEQ_LEN, STRIDE):
            seqs.append(s_data[i:i + SEQ_LEN])
        seqs_np = np.array(seqs)

        chunk_size = 128
        all_hidden = []
        for start in range(0, len(seqs_np), chunk_size):
            chunk = torch.tensor(seqs_np[start:start + chunk_size],
                                 dtype=torch.float32).to(DEVICE)
            h = model.extract_hidden_states(chunk, 1)
            all_hidden.append(h.cpu().numpy())
        hidden_all = np.concatenate(all_hidden, axis=0)
        print(f"  {hidden_all.shape[0]} hidden states x {hidden_all.shape[1]}D")

        # Time alignment
        hs_indices = [i * STRIDE + SEQ_LEN - 1 for i in range(len(hidden_all))]
        hs_time_sec = s_time[np.array(hs_indices)]

        # Load behavior
        behav_dict = load_behavior_timeseries(1, hs_time_sec)
        behavior_labels = get_behavior_labels(behav_dict, len(hs_time_sec))
        zone_labels = get_zone_labels(behav_dict, len(hs_time_sec))

        # Assign excursion IDs
        exc_ids = np.full(len(hs_time_sec), -1, dtype=int)
        for _, erow in complete.iterrows():
            mask = (hs_time_sec >= erow['start_time']) & (hs_time_sec <= erow['end_time'])
            exc_ids[mask] = int(erow['excursion_id'])

        # Filter to in-excursion points
        in_exc = exc_ids >= 0
        H = hidden_all[in_exc]
        B = behavior_labels[in_exc]
        Z = zone_labels[in_exc]
        E = exc_ids[in_exc]
        T = hs_time_sec[in_exc]

        unique_exc = np.unique(E)
        n_exc = len(unique_exc)
        print(f"  {len(H)} points in {n_exc} excursions")

        # =================================================================
        # A. VARIANCE DECOMPOSITION: between-excursion vs within-excursion
        # =================================================================
        print(f"\n  --- A. Variance Decomposition ---")

        global_mean = H.mean(axis=0)
        total_var = np.sum(np.var(H, axis=0))

        # Between-excursion variance: variance of excursion centroids
        centroids = np.array([H[E == eid].mean(axis=0) for eid in unique_exc])
        exc_sizes = np.array([(E == eid).sum() for eid in unique_exc])
        weighted_between = 0
        for i, eid in enumerate(unique_exc):
            diff = centroids[i] - global_mean
            weighted_between += exc_sizes[i] * np.sum(diff ** 2)
        between_var = weighted_between / len(H)

        # Within-excursion variance: average variance within each excursion
        weighted_within = 0
        for i, eid in enumerate(unique_exc):
            exc_data = H[E == eid]
            within = np.sum(np.var(exc_data, axis=0)) * len(exc_data)
            weighted_within += within
        within_var = weighted_within / len(H)

        print(f"  Total variance:          {total_var:.4f}")
        print(f"  Between-excursion var:   {between_var:.4f} ({100*between_var/total_var:.1f}%)")
        print(f"  Within-excursion var:    {within_var:.4f} ({100*within_var/total_var:.1f}%)")
        print(f"  Ratio (between/within):  {between_var/within_var:.2f}")

        # Now do the same decomposition for behavior
        beh_unique = np.unique(B)
        beh_centroids = np.array([H[B == b].mean(axis=0) for b in beh_unique])
        beh_sizes = np.array([(B == b).sum() for b in beh_unique])
        beh_between = 0
        for i, b in enumerate(beh_unique):
            diff = beh_centroids[i] - global_mean
            beh_between += beh_sizes[i] * np.sum(diff ** 2)
        beh_between_var = beh_between / len(H)

        beh_within = 0
        for i, b in enumerate(beh_unique):
            beh_data = H[B == b]
            beh_within += np.sum(np.var(beh_data, axis=0)) * len(beh_data)
        beh_within_var = beh_within / len(H)

        print(f"\n  Behavior grouping:")
        print(f"  Between-behavior var:    {beh_between_var:.4f} ({100*beh_between_var/total_var:.1f}%)")
        print(f"  Within-behavior var:     {beh_within_var:.4f} ({100*beh_within_var/total_var:.1f}%)")
        print(f"  Ratio (between/within):  {beh_between_var/beh_within_var:.4f}")

        # =================================================================
        # B. CENTROID DISTANCE vs TEMPORAL DISTANCE
        # =================================================================
        print(f"\n  --- B. Centroid Distance vs Temporal Distance ---")

        exc_mid_times = np.array([T[E == eid].mean() for eid in unique_exc])
        centroid_dists = squareform(pdist(centroids, 'euclidean'))
        time_dists = squareform(pdist(exc_mid_times.reshape(-1, 1), 'euclidean'))

        # Upper triangle only
        triu_idx = np.triu_indices(n_exc, k=1)
        cd_flat = centroid_dists[triu_idx]
        td_flat = time_dists[triu_idx]

        r_pearson, p_pearson = pearsonr(td_flat, cd_flat)
        r_spearman, p_spearman = spearmanr(td_flat, cd_flat)
        print(f"  Pearson r(time_dist, centroid_dist):  {r_pearson:.3f}  p={p_pearson:.2e}")
        print(f"  Spearman r(time_dist, centroid_dist): {r_spearman:.3f}  p={p_spearman:.2e}")

        # =================================================================
        # C. EXCURSION ID CONFUSION vs TEMPORAL PROXIMITY
        # =================================================================
        print(f"\n  --- C. Excursion ID Confusion vs Temporal Proximity ---")

        # Filter to excursions with >= 30 points (same as decoding script)
        exc_counts = pd.Series(E).value_counts()
        valid_exc_id = exc_counts[exc_counts >= 30].index.tolist()
        eid_mask = np.isin(E, valid_exc_id)
        H_eid = H[eid_mask]
        E_eid = E[eid_mask]
        T_eid = T[eid_mask]

        le_exc = LabelEncoder()
        E_enc = le_exc.fit_transform(E_eid)
        n_exc_classes = len(le_exc.classes_)

        scaler = StandardScaler()
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        all_preds_eid = np.full(len(E_enc), -1, dtype=int)

        for train_idx, test_idx in skf.split(H_eid, E_enc):
            X_train = scaler.fit_transform(H_eid[train_idx])
            X_test = scaler.transform(H_eid[test_idx])
            clf = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs',
                                     multi_class='multinomial')
            clf.fit(X_train, E_enc[train_idx])
            all_preds_eid[test_idx] = clf.predict(X_test)

        # Compute confusion matrix
        cm = confusion_matrix(E_enc, all_preds_eid)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        # For misclassified points, how far in time is the predicted excursion?
        exc_order = le_exc.classes_  # original excursion IDs in encoded order
        exc_mid_map = {eid: T[E == eid].mean() for eid in exc_order}

        misclassified = all_preds_eid != E_enc
        if misclassified.sum() > 0:
            true_eids = le_exc.inverse_transform(E_enc[misclassified])
            pred_eids = le_exc.inverse_transform(all_preds_eid[misclassified])
            time_gaps = np.array([abs(exc_mid_map[t] - exc_mid_map[p])
                                  for t, p in zip(true_eids, pred_eids)])
            print(f"  Misclassified points: {misclassified.sum()} / {len(E_enc)}")
            print(f"  Time gap for misclassifications:")
            print(f"    Mean:   {time_gaps.mean():.1f}s")
            print(f"    Median: {np.median(time_gaps):.1f}s")
            print(f"    <60s:   {(time_gaps < 60).sum()} ({100*(time_gaps < 60).mean():.1f}%)")
            print(f"    <120s:  {(time_gaps < 120).sum()} ({100*(time_gaps < 120).mean():.1f}%)")
            print(f"    <300s:  {(time_gaps < 300).sum()} ({100*(time_gaps < 300).mean():.1f}%)")

            # Compare to random baseline: mean pairwise time gap
            all_time_gaps = []
            for eid1 in exc_order:
                for eid2 in exc_order:
                    if eid1 != eid2:
                        all_time_gaps.append(abs(exc_mid_map[eid1] - exc_mid_map[eid2]))
            print(f"    Random baseline mean gap: {np.mean(all_time_gaps):.1f}s")

        # =================================================================
        # D. DETREND (remove slow drift) then re-decode behavior
        # =================================================================
        print(f"\n  --- D. Detrend Hidden States, Then Re-decode ---")

        # Method 1: Subtract per-excursion centroid (removes between-exc variance)
        H_centered = np.zeros_like(H)
        for eid in unique_exc:
            emask = E == eid
            H_centered[emask] = H[emask] - H[emask].mean(axis=0)

        # Method 2: Regress out time (linear detrend in 32D)
        from sklearn.linear_model import LinearRegression
        time_reg = LinearRegression()
        time_reg.fit(T.reshape(-1, 1), H)
        H_detrended = H - time_reg.predict(T.reshape(-1, 1))

        for method_name, H_mod in [("Centroid-subtracted", H_centered),
                                     ("Time-detrended", H_detrended)]:
            # Re-decode behavior
            beh_counts = pd.Series(B).value_counts()
            valid_behaviors = beh_counts[beh_counts >= 30].index.tolist()
            beh_mask = np.isin(B, valid_behaviors)
            H_beh = H_mod[beh_mask]
            B_beh = B[beh_mask]
            E_beh = E[beh_mask]

            le_beh = LabelEncoder()
            B_enc = le_beh.fit_transform(B_beh)
            n_classes = len(le_beh.classes_)

            valid_exc = [eid for eid in np.unique(E_beh)
                         if (E_beh == eid).sum() >= 10]

            all_preds_beh = np.full(len(B_enc), -1, dtype=int)
            for eid in valid_exc:
                test_mask = E_beh == eid
                train_mask = ~test_mask
                if len(np.unique(B_enc[train_mask])) < 2:
                    continue
                X_train = scaler.fit_transform(H_beh[train_mask])
                X_test = scaler.transform(H_beh[test_mask])
                clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs',
                                         multi_class='multinomial')
                clf.fit(X_train, B_enc[train_mask])
                all_preds_beh[test_mask] = clf.predict(X_test)

            pred_mask = all_preds_beh >= 0
            if pred_mask.sum() > 0:
                acc = accuracy_score(B_enc[pred_mask], all_preds_beh[pred_mask])
                bal_acc = balanced_accuracy_score(B_enc[pred_mask], all_preds_beh[pred_mask])
                chance = np.max(np.bincount(B_enc[pred_mask]) / pred_mask.sum())
                print(f"  {method_name}: acc={acc:.3f} (chance={chance:.3f}), bal_acc={bal_acc:.3f}")

            # Re-decode zone
            zone_counts = pd.Series(Z).value_counts()
            valid_zones = zone_counts[zone_counts >= 30].index.tolist()
            zone_mask = np.isin(Z, valid_zones)
            H_zone = H_mod[zone_mask]
            Z_zone = Z[zone_mask]
            E_zone = E[zone_mask]

            le_zone = LabelEncoder()
            Z_enc = le_zone.fit_transform(Z_zone)

            valid_exc_z = [eid for eid in np.unique(E_zone)
                           if (E_zone == eid).sum() >= 10]

            all_preds_zone = np.full(len(Z_enc), -1, dtype=int)
            for eid in valid_exc_z:
                test_mask = E_zone == eid
                train_mask = ~test_mask
                if len(np.unique(Z_enc[train_mask])) < 2:
                    continue
                X_train = scaler.fit_transform(H_zone[train_mask])
                X_test = scaler.transform(H_zone[test_mask])
                clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs',
                                         multi_class='multinomial')
                clf.fit(X_train, Z_enc[train_mask])
                all_preds_zone[test_mask] = clf.predict(X_test)

            pred_mask_z = all_preds_zone >= 0
            if pred_mask_z.sum() > 0:
                z_acc = accuracy_score(Z_enc[pred_mask_z], all_preds_zone[pred_mask_z])
                z_bal = balanced_accuracy_score(Z_enc[pred_mask_z], all_preds_zone[pred_mask_z])
                z_chance = np.max(np.bincount(Z_enc[pred_mask_z]) / pred_mask_z.sum())
                print(f"  {method_name}: zone acc={z_acc:.3f} (chance={z_chance:.3f}), bal_acc={z_bal:.3f}")

            # Re-decode excursion ID (should drop if drift was the signal)
            eid_mask2 = np.isin(E, valid_exc_id)
            H_eid2 = H_mod[eid_mask2]
            E_eid2 = E[eid_mask2]
            le_exc2 = LabelEncoder()
            E_enc2 = le_exc2.fit_transform(E_eid2)

            fold_accs = []
            for train_idx, test_idx in skf.split(H_eid2, E_enc2):
                X_train = scaler.fit_transform(H_eid2[train_idx])
                X_test = scaler.transform(H_eid2[test_idx])
                clf = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs',
                                         multi_class='multinomial')
                clf.fit(X_train, E_enc2[train_idx])
                fold_accs.append(accuracy_score(E_enc2[test_idx], clf.predict(X_test)))
            eid_acc2 = np.mean(fold_accs)
            eid_chance2 = 1.0 / len(le_exc2.classes_)
            print(f"  {method_name}: exc ID acc={eid_acc2:.3f} (chance={eid_chance2:.3f}, lift={eid_acc2/eid_chance2:.1f}x)")

        # =================================================================
        # E. Within-excursion behavior variance
        # =================================================================
        print(f"\n  --- E. Within-Excursion Behavior Separation ---")

        # For each excursion: can we separate behaviors WITHIN that excursion?
        exc_beh_results = []
        for eid in unique_exc:
            emask = E == eid
            h_exc = H_centered[emask]  # use centroid-subtracted
            b_exc = B[emask]

            # Need at least 2 behaviors with >= 5 points each
            bc = pd.Series(b_exc).value_counts()
            valid_b = bc[bc >= 5].index.tolist()
            if len(valid_b) < 2:
                continue

            b_mask = np.isin(b_exc, valid_b)
            if b_mask.sum() < 15:
                continue

            h_sub = h_exc[b_mask]
            b_sub = b_exc[b_mask]

            # Compute within-excursion between-behavior variance ratio
            exc_total = np.sum(np.var(h_sub, axis=0))
            beh_centroids_exc = np.array([h_sub[b_sub == b].mean(axis=0) for b in valid_b])
            exc_global = h_sub.mean(axis=0)
            beh_between_exc = 0
            for i, b in enumerate(valid_b):
                diff = beh_centroids_exc[i] - exc_global
                beh_between_exc += (b_sub == b).sum() * np.sum(diff ** 2)
            beh_between_exc /= len(h_sub)

            exc_beh_results.append({
                'excursion_id': eid,
                'n_points': emask.sum(),
                'n_behaviors': len(valid_b),
                'total_var': exc_total,
                'behavior_between_var': beh_between_exc,
                'pct_behavior': 100 * beh_between_exc / exc_total if exc_total > 0 else 0,
            })

        if exc_beh_results:
            ebr_df = pd.DataFrame(exc_beh_results)
            print(f"  {len(ebr_df)} excursions with >=2 behaviors")
            print(f"  % variance explained by behavior WITHIN excursions:")
            print(f"    Mean:   {ebr_df['pct_behavior'].mean():.1f}%")
            print(f"    Median: {ebr_df['pct_behavior'].median():.1f}%")
            print(f"    Max:    {ebr_df['pct_behavior'].max():.1f}%")
            print(f"    Min:    {ebr_df['pct_behavior'].min():.1f}%")

        # =================================================================
        # F. AUTOCORRELATION of hidden states
        # =================================================================
        print(f"\n  --- F. Hidden State Autocorrelation ---")

        # Sample from within excursions
        lags_sec = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
        lag_bins = [int(l / (STRIDE * BIN_SIZE_MS / 1000)) for l in lags_sec]

        for lag_s, lag_b in zip(lags_sec, lag_bins):
            if lag_b < 1:
                continue
            cosines = []
            for eid in unique_exc:
                emask = E == eid
                h_exc = H[emask]
                if len(h_exc) <= lag_b:
                    continue
                for t in range(len(h_exc) - lag_b):
                    h1 = h_exc[t]
                    h2 = h_exc[t + lag_b]
                    norm1 = np.linalg.norm(h1)
                    norm2 = np.linalg.norm(h2)
                    if norm1 > 0 and norm2 > 0:
                        cosines.append(np.dot(h1, h2) / (norm1 * norm2))
            if cosines:
                print(f"    Lag {lag_s:5.1f}s: cosine similarity = {np.mean(cosines):.4f} "
                      f"(std={np.std(cosines):.4f}, n={len(cosines)})")

        # =================================================================
        # FIGURE
        # =================================================================
        print(f"\n  Creating figure...")
        fig, axes = plt.subplots(2, 3, figsize=(22, 14))
        fig.suptitle(f"{region_label} — Why Hidden States Separate Excursions But Not Behaviors",
                     fontsize=14, fontweight='bold')

        # Panel (0,0): PCA colored by excursion ID
        ax = axes[0, 0]
        pca = PCA(n_components=3).fit(H)
        H_pca = pca.transform(H)
        cmap = plt.cm.turbo
        exc_norm = (E - E.min()) / (E.max() - E.min() + 1e-8)
        ax.scatter(H_pca[:, 0], H_pca[:, 1], c=exc_norm, cmap=cmap,
                   s=3, alpha=0.4, rasterized=True)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax.set_title('Hidden states colored by Excursion ID\n(temporal order = color order)')
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(E.min(), E.max()))
        plt.colorbar(sm, ax=ax, label='Excursion ID', shrink=0.8)

        # Panel (0,1): PCA colored by behavior
        ax = axes[0, 1]
        beh_colors = {
            'Other': '#CCCCCC', 'Feeding': '#D32F2F', 'Digging': '#FF9800',
            'Grooming': '#9C27B0', 'Quick arena exploration': '#2196F3',
            'Arena wall exploration': '#4CAF50', 'Transition wall exploration': '#795548',
            'Hesitant exploration': '#607D8B', 'Quick one loop at home': '#FFC107'
        }
        for bname in np.unique(B):
            mask = B == bname
            color = beh_colors.get(bname, '#999999')
            ax.scatter(H_pca[mask, 0], H_pca[mask, 1], c=color, s=3,
                       alpha=0.4, label=bname, rasterized=True)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax.set_title('Hidden states colored by Behavior')
        ax.legend(fontsize=6, markerscale=3, loc='upper right')

        # Panel (0,2): PCA colored by zone
        ax = axes[0, 2]
        zone_colors = {
            'Home': '#4CAF50', 'Ladder': '#FFC107',
            'Transition zone': '#FF9800', 'Foraging arena': '#2196F3',
            'Other': '#CCCCCC'
        }
        for zname in np.unique(Z):
            mask = Z == zname
            color = zone_colors.get(zname, '#999999')
            ax.scatter(H_pca[mask, 0], H_pca[mask, 1], c=color, s=3,
                       alpha=0.4, label=zname, rasterized=True)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax.set_title('Hidden states colored by Zone')
        ax.legend(fontsize=8, markerscale=3)

        # Panel (1,0): Centroid distance vs temporal distance
        ax = axes[1, 0]
        ax.scatter(td_flat, cd_flat, s=8, alpha=0.3, c='#2196F3', rasterized=True)
        # Fit line
        z_poly = np.polyfit(td_flat, cd_flat, 1)
        x_line = np.linspace(td_flat.min(), td_flat.max(), 100)
        ax.plot(x_line, np.polyval(z_poly, x_line), 'r-', linewidth=2)
        ax.set_xlabel('Temporal distance between excursions (s)')
        ax.set_ylabel('Centroid distance in 32D hidden space')
        ax.set_title(f'Centroid dist vs time dist\nr={r_pearson:.3f}, p={p_pearson:.1e}')

        # Panel (1,1): Variance decomposition bar chart
        ax = axes[1, 1]
        labels_bar = ['Excursion\n(between)', 'Excursion\n(within)',
                      'Behavior\n(between)', 'Behavior\n(within)']
        vals = [100*between_var/total_var, 100*within_var/total_var,
                100*beh_between_var/total_var, 100*beh_within_var/total_var]
        colors_bar = ['#D32F2F', '#FFCDD2', '#2196F3', '#BBDEFB']
        bars = ax.bar(range(4), vals, color=colors_bar, edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(4))
        ax.set_xticklabels(labels_bar, fontsize=9)
        ax.set_ylabel('% of total variance')
        ax.set_title('Variance Decomposition')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, val + 1,
                    f'{val:.1f}%', ha='center', fontweight='bold', fontsize=10)

        # Panel (1,2): Time series of PC1 with excursion boundaries
        ax = axes[1, 2]
        # Plot PC1 of ALL hidden states (not just in-excursion) as time series
        H_all_pca = pca.transform(hidden_all)
        ax.plot(hs_time_sec, H_all_pca[:, 0], linewidth=0.3, color='#666666', alpha=0.5)

        # Overlay excursion segments colored by excursion ID
        for eid in unique_exc:
            emask = E == eid
            t_exc = T[emask]
            pc1_exc = H_pca[emask, 0]
            color = cmap(exc_norm[emask][0])
            ax.plot(t_exc, pc1_exc, linewidth=1.0, color=color, alpha=0.8)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('PC1 of hidden state')
        ax.set_title('PC1 over session time\n(colored segments = excursions)')

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        outpath = Path("figures") / f"latent_identity_investigation_{region}.png"
        fig.savefig(outpath, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {outpath}")

    print("\nDone!")


if __name__ == "__main__":
    main()
