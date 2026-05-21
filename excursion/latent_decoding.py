"""
Decode behavior, zone, and speed from GRU-ODE hidden states.
Tests whether the 32D latent states contain moment-to-moment information
that is linearly accessible, even though PC1-PC2 projections look similar
across excursions.

Uses leave-one-excursion-out cross-validation to ensure generalization.
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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, r2_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
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
STRIDE = 10  # one hidden state per 100ms
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("paths.yaml") as f:
    paths_config = yaml.safe_load(f)


# --- Model classes ---
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
    """Assign dominant behavior label to each time point."""
    labels = np.full(n_pts, 'Other', dtype=object)
    priority = ['Feeding', 'Digging', 'Grooming', 'Quick arena exploration',
                'Arena wall exploration', 'Transition wall exploration',
                'Hesitant exploration', 'Quick one loop at home']
    for bname in reversed(priority):
        if bname in behav_dict:
            labels[behav_dict[bname] > 0] = bname
    return labels


def get_zone_labels(behav_dict, n_pts):
    """Assign zone label to each time point."""
    labels = np.full(n_pts, 'Other', dtype=object)
    for zone in ['Home', 'Ladder', 'Transition zone', 'Foraging arena']:
        if zone in behav_dict:
            labels[behav_dict[zone] > 0] = zone
    return labels


def get_speed_values(behav_dict, n_pts):
    """Get animal speed at each time point if available."""
    for key in behav_dict:
        if 'velocity' in key.lower() or 'speed' in key.lower():
            return behav_dict[key]
    # Try 'Distance moved' as proxy
    if 'Distance moved' in behav_dict:
        return behav_dict['Distance moved']
    return None


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  Decoding Behavior from GRU-ODE Hidden States")
    print("  Session 1 (Fed) — Linear Readout from 32D Latent Space")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    sp = paths_config['single_probe']['coordinates_1']['mouse01']['sessions']
    sc = sp['session_1']
    sorted_path = Path(sc['sorted'])
    sorting = se.read_kilosort(sorted_path)
    lha_ids, rsp_ids = get_good_units_by_region(sorted_path)

    exc_df = pd.read_csv("data/excursions_session_1.csv")
    complete = exc_df[exc_df['label'] == 'complete']

    results_all = []

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
        print(f"  Loaded model: {model_path.name}")

        # Extract hidden states for Session 1
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

        # Load behavior labels
        behav_dict = load_behavior_timeseries(1, hs_time_sec)
        behavior_labels = get_behavior_labels(behav_dict, len(hs_time_sec))
        zone_labels = get_zone_labels(behav_dict, len(hs_time_sec))
        speed_values = get_speed_values(behav_dict, len(hs_time_sec))

        # Assign excursion IDs to each hidden state
        exc_ids = np.full(len(hs_time_sec), -1, dtype=int)
        for _, erow in complete.iterrows():
            mask = (hs_time_sec >= erow['start_time']) & (hs_time_sec <= erow['end_time'])
            exc_ids[mask] = int(erow['excursion_id'])

        # Only use time points within excursions
        in_exc = exc_ids >= 0
        H = hidden_all[in_exc]
        B = behavior_labels[in_exc]
        Z = zone_labels[in_exc]
        E = exc_ids[in_exc]
        T = hs_time_sec[in_exc]

        print(f"  {len(H)} time points within excursions")
        print(f"  Behavior distribution:")
        for bname, count in pd.Series(B).value_counts().items():
            print(f"    {bname}: {count} ({100*count/len(B):.1f}%)")
        print(f"  Zone distribution:")
        for zname, count in pd.Series(Z).value_counts().items():
            print(f"    {zname}: {count} ({100*count/len(Z):.1f}%)")

        # =====================================================================
        # 1. BEHAVIOR DECODING — Leave-one-excursion-out CV
        # =====================================================================
        print(f"\n  --- Behavior Decoding (Leave-One-Excursion-Out) ---")

        # Filter to behaviors with enough samples
        beh_counts = pd.Series(B).value_counts()
        valid_behaviors = beh_counts[beh_counts >= 30].index.tolist()
        beh_mask = np.isin(B, valid_behaviors)
        H_beh = H[beh_mask]
        B_beh = B[beh_mask]
        E_beh = E[beh_mask]

        le_beh = LabelEncoder()
        B_enc = le_beh.fit_transform(B_beh)
        n_classes = len(le_beh.classes_)
        print(f"  {n_classes} behavior classes with >=30 samples: {list(le_beh.classes_)}")

        unique_exc = np.unique(E_beh)
        # Filter excursions that have at least 2 classes
        valid_exc = []
        for eid in unique_exc:
            emask = E_beh == eid
            if len(np.unique(B_enc[emask])) >= 1 and emask.sum() >= 10:
                valid_exc.append(eid)

        all_preds_beh = np.full(len(B_enc), -1, dtype=int)
        all_true_beh = B_enc.copy()
        exc_accuracies = []

        scaler = StandardScaler()

        for eid in valid_exc:
            test_mask = E_beh == eid
            train_mask = ~test_mask

            # Need at least 2 classes in training
            if len(np.unique(B_enc[train_mask])) < 2:
                continue

            X_train = scaler.fit_transform(H_beh[train_mask])
            X_test = scaler.transform(H_beh[test_mask])
            y_train = B_enc[train_mask]
            y_test = B_enc[test_mask]

            clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs',
                                     multi_class='multinomial')
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            all_preds_beh[test_mask] = preds
            acc = accuracy_score(y_test, preds)
            exc_accuracies.append({'excursion_id': eid, 'accuracy': acc,
                                   'n_points': test_mask.sum()})

        predicted_mask = all_preds_beh >= 0
        overall_acc = accuracy_score(all_true_beh[predicted_mask],
                                     all_preds_beh[predicted_mask])
        overall_bal_acc = balanced_accuracy_score(all_true_beh[predicted_mask],
                                                  all_preds_beh[predicted_mask])

        # Chance level
        class_freqs = np.bincount(all_true_beh[predicted_mask]) / predicted_mask.sum()
        chance_acc = np.max(class_freqs)  # majority class baseline

        print(f"  Overall accuracy: {overall_acc:.3f} (chance = {chance_acc:.3f})")
        print(f"  Balanced accuracy: {overall_bal_acc:.3f} (chance = {1/n_classes:.3f})")
        print(f"  Lift over chance: {overall_acc / chance_acc:.2f}x")

        exc_acc_df = pd.DataFrame(exc_accuracies)
        print(f"  Per-excursion accuracy: mean={exc_acc_df['accuracy'].mean():.3f}, "
              f"std={exc_acc_df['accuracy'].std():.3f}")

        results_all.append({
            'region': region_label,
            'task': 'Behavior',
            'accuracy': overall_acc,
            'balanced_accuracy': overall_bal_acc,
            'chance': chance_acc,
            'lift': overall_acc / chance_acc,
            'n_classes': n_classes,
        })

        # =====================================================================
        # 2. ZONE DECODING
        # =====================================================================
        print(f"\n  --- Zone Decoding (Leave-One-Excursion-Out) ---")

        zone_counts = pd.Series(Z).value_counts()
        valid_zones = zone_counts[zone_counts >= 30].index.tolist()
        zone_mask = np.isin(Z, valid_zones)
        H_zone = H[zone_mask]
        Z_zone = Z[zone_mask]
        E_zone = E[zone_mask]

        le_zone = LabelEncoder()
        Z_enc = le_zone.fit_transform(Z_zone)
        n_zones = len(le_zone.classes_)
        print(f"  {n_zones} zone classes with >=30 samples: {list(le_zone.classes_)}")

        unique_exc_z = np.unique(E_zone)
        valid_exc_z = [eid for eid in unique_exc_z
                       if (E_zone == eid).sum() >= 10]

        all_preds_zone = np.full(len(Z_enc), -1, dtype=int)
        all_true_zone = Z_enc.copy()

        for eid in valid_exc_z:
            test_mask = E_zone == eid
            train_mask = ~test_mask
            if len(np.unique(Z_enc[train_mask])) < 2:
                continue

            X_train = scaler.fit_transform(H_zone[train_mask])
            X_test = scaler.transform(H_zone[test_mask])
            y_train = Z_enc[train_mask]
            y_test = Z_enc[test_mask]

            clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs',
                                     multi_class='multinomial')
            clf.fit(X_train, y_train)
            all_preds_zone[test_mask] = clf.predict(X_test)

        pred_mask_z = all_preds_zone >= 0
        zone_acc = accuracy_score(all_true_zone[pred_mask_z], all_preds_zone[pred_mask_z])
        zone_bal_acc = balanced_accuracy_score(all_true_zone[pred_mask_z],
                                               all_preds_zone[pred_mask_z])
        zone_chance = np.max(np.bincount(all_true_zone[pred_mask_z]) / pred_mask_z.sum())

        print(f"  Overall accuracy: {zone_acc:.3f} (chance = {zone_chance:.3f})")
        print(f"  Balanced accuracy: {zone_bal_acc:.3f} (chance = {1/n_zones:.3f})")
        print(f"  Lift over chance: {zone_acc / zone_chance:.2f}x")

        results_all.append({
            'region': region_label,
            'task': 'Zone',
            'accuracy': zone_acc,
            'balanced_accuracy': zone_bal_acc,
            'chance': zone_chance,
            'lift': zone_acc / zone_chance,
            'n_classes': n_zones,
        })

        # =====================================================================
        # 3. SPEED DECODING (regression)
        # =====================================================================
        if speed_values is not None:
            print(f"\n  --- Speed Decoding (Leave-One-Excursion-Out) ---")
            S_exc = speed_values[in_exc]
            valid_speed = ~np.isnan(S_exc) & (S_exc >= 0)
            H_spd = H[valid_speed]
            S_spd = S_exc[valid_speed]
            E_spd = E[valid_speed]

            unique_exc_s = np.unique(E_spd)
            valid_exc_s = [eid for eid in unique_exc_s
                           if (E_spd == eid).sum() >= 10]

            all_preds_spd = np.full(len(S_spd), np.nan)
            all_true_spd = S_spd.copy()

            for eid in valid_exc_s:
                test_mask = E_spd == eid
                train_mask = ~test_mask

                X_train = scaler.fit_transform(H_spd[train_mask])
                X_test = scaler.transform(H_spd[test_mask])
                y_train = S_spd[train_mask]
                y_test = S_spd[test_mask]

                reg = Ridge(alpha=1.0)
                reg.fit(X_train, y_train)
                all_preds_spd[test_mask] = reg.predict(X_test)

            pred_mask_s = ~np.isnan(all_preds_spd)
            speed_r2 = r2_score(all_true_spd[pred_mask_s], all_preds_spd[pred_mask_s])
            speed_corr = np.corrcoef(all_true_spd[pred_mask_s],
                                     all_preds_spd[pred_mask_s])[0, 1]
            print(f"  R² = {speed_r2:.3f}")
            print(f"  Pearson r = {speed_corr:.3f}")

            results_all.append({
                'region': region_label,
                'task': 'Speed',
                'accuracy': speed_r2,
                'balanced_accuracy': speed_corr,
                'chance': 0.0,
                'lift': speed_r2,
                'n_classes': 0,
            })
        else:
            print(f"\n  No speed/velocity variable found in behavior data")
            S_exc = None

        # =====================================================================
        # 4. EXCURSION IDENTITY DECODING
        # =====================================================================
        print(f"\n  --- Excursion Identity Decoding (5-fold CV) ---")

        # Can we tell which excursion a time point belongs to?
        # Use only excursions with >= 30 points
        exc_counts = pd.Series(E).value_counts()
        valid_exc_id = exc_counts[exc_counts >= 30].index.tolist()
        exc_id_mask = np.isin(E, valid_exc_id)
        H_eid = H[exc_id_mask]
        E_eid = E[exc_id_mask]

        le_exc = LabelEncoder()
        E_enc = le_exc.fit_transform(E_eid)
        n_exc_classes = len(le_exc.classes_)
        print(f"  {n_exc_classes} excursions with >=30 points")

        # Stratified 5-fold (not leave-one-out since excursion IS the label)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_accs = []
        all_preds_eid = np.full(len(E_enc), -1, dtype=int)

        for train_idx, test_idx in skf.split(H_eid, E_enc):
            X_train = scaler.fit_transform(H_eid[train_idx])
            X_test = scaler.transform(H_eid[test_idx])
            y_train = E_enc[train_idx]
            y_test = E_enc[test_idx]

            clf = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs',
                                     multi_class='multinomial')
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            all_preds_eid[test_idx] = preds
            fold_accs.append(accuracy_score(y_test, preds))

        eid_acc = np.mean(fold_accs)
        eid_chance = 1.0 / n_exc_classes
        print(f"  Accuracy: {eid_acc:.3f} (chance = {eid_chance:.3f})")
        print(f"  Lift over chance: {eid_acc / eid_chance:.1f}x")

        results_all.append({
            'region': region_label,
            'task': 'Excursion ID',
            'accuracy': eid_acc,
            'balanced_accuracy': eid_acc,
            'chance': eid_chance,
            'lift': eid_acc / eid_chance,
            'n_classes': n_exc_classes,
        })

        # =====================================================================
        # 5. TEMPORAL POSITION within excursion
        # =====================================================================
        print(f"\n  --- Temporal Position Decoding (early/mid/late) ---")

        # For each excursion, assign early (0-33%), mid (33-66%), late (66-100%)
        phase_labels = np.full(len(E), '', dtype=object)
        for eid in np.unique(E):
            emask = E == eid
            n = emask.sum()
            if n < 15:
                phase_labels[emask] = 'skip'
                continue
            indices = np.where(emask)[0]
            third = n // 3
            phase_labels[indices[:third]] = 'early'
            phase_labels[indices[third:2*third]] = 'mid'
            phase_labels[indices[2*third:]] = 'late'

        phase_valid = phase_labels != 'skip'
        H_phase = H[phase_valid]
        P_phase = phase_labels[phase_valid]
        E_phase = E[phase_valid]

        le_phase = LabelEncoder()
        P_enc = le_phase.fit_transform(P_phase)

        # Leave-one-excursion-out
        unique_exc_p = np.unique(E_phase)
        valid_exc_p = [eid for eid in unique_exc_p if (E_phase == eid).sum() >= 15]

        all_preds_phase = np.full(len(P_enc), -1, dtype=int)

        for eid in valid_exc_p:
            test_mask = E_phase == eid
            train_mask = ~test_mask
            if len(np.unique(P_enc[train_mask])) < 3:
                continue

            X_train = scaler.fit_transform(H_phase[train_mask])
            X_test = scaler.transform(H_phase[test_mask])
            y_train = P_enc[train_mask]
            y_test = P_enc[test_mask]

            clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs',
                                     multi_class='multinomial')
            clf.fit(X_train, y_train)
            all_preds_phase[test_mask] = clf.predict(X_test)

        pred_mask_p = all_preds_phase >= 0
        phase_acc = accuracy_score(P_enc[pred_mask_p], all_preds_phase[pred_mask_p])
        phase_bal_acc = balanced_accuracy_score(P_enc[pred_mask_p],
                                                all_preds_phase[pred_mask_p])
        print(f"  Accuracy: {phase_acc:.3f} (chance = 0.333)")
        print(f"  Balanced accuracy: {phase_bal_acc:.3f}")

        results_all.append({
            'region': region_label,
            'task': 'Temporal Phase',
            'accuracy': phase_acc,
            'balanced_accuracy': phase_bal_acc,
            'chance': 0.333,
            'lift': phase_acc / 0.333,
            'n_classes': 3,
        })

        # =====================================================================
        # FIGURE: Summary of decoding results + confusion matrices
        # =====================================================================
        print(f"\n  Creating figures...")

        fig, axes = plt.subplots(2, 3, figsize=(22, 14))
        fig.suptitle(f"{region_label} — Linear Decoding from 32D GRU-ODE Hidden States\n"
                     f"Session 1 (Fed) | Leave-One-Excursion-Out Cross-Validation",
                     fontsize=14, fontweight='bold')

        # Panel (0,0): Behavior confusion matrix
        ax = axes[0, 0]
        pred_mask_b = all_preds_beh >= 0
        cm_beh = confusion_matrix(all_true_beh[pred_mask_b], all_preds_beh[pred_mask_b])
        cm_beh_norm = cm_beh.astype(float) / cm_beh.sum(axis=1, keepdims=True)
        im = ax.imshow(cm_beh_norm, cmap='Blues', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, shrink=0.8)
        class_names_short = [c[:12] for c in le_beh.classes_]
        ax.set_xticks(range(n_classes))
        ax.set_xticklabels(class_names_short, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(n_classes))
        ax.set_yticklabels(class_names_short, fontsize=8)
        for i in range(n_classes):
            for j in range(n_classes):
                ax.text(j, i, f'{cm_beh_norm[i,j]:.2f}',
                       ha='center', va='center', fontsize=7,
                       color='white' if cm_beh_norm[i,j] > 0.5 else 'black')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'Behavior (acc={overall_acc:.3f}, chance={chance_acc:.3f})',
                    fontsize=11)

        # Panel (0,1): Zone confusion matrix
        ax = axes[0, 1]
        cm_zone = confusion_matrix(all_true_zone[pred_mask_z], all_preds_zone[pred_mask_z])
        cm_zone_norm = cm_zone.astype(float) / cm_zone.sum(axis=1, keepdims=True)
        im = ax.imshow(cm_zone_norm, cmap='Greens', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, shrink=0.8)
        zone_names = list(le_zone.classes_)
        ax.set_xticks(range(n_zones))
        ax.set_xticklabels(zone_names, rotation=45, ha='right', fontsize=9)
        ax.set_yticks(range(n_zones))
        ax.set_yticklabels(zone_names, fontsize=9)
        for i in range(n_zones):
            for j in range(n_zones):
                ax.text(j, i, f'{cm_zone_norm[i,j]:.2f}',
                       ha='center', va='center', fontsize=9,
                       color='white' if cm_zone_norm[i,j] > 0.5 else 'black')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'Zone (acc={zone_acc:.3f}, chance={zone_chance:.3f})',
                    fontsize=11)

        # Panel (0,2): Temporal phase confusion matrix
        ax = axes[0, 2]
        cm_phase = confusion_matrix(P_enc[pred_mask_p], all_preds_phase[pred_mask_p])
        cm_phase_norm = cm_phase.astype(float) / cm_phase.sum(axis=1, keepdims=True)
        im = ax.imshow(cm_phase_norm, cmap='Oranges', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, shrink=0.8)
        phase_names = list(le_phase.classes_)
        ax.set_xticks(range(3))
        ax.set_xticklabels(phase_names, fontsize=10)
        ax.set_yticks(range(3))
        ax.set_yticklabels(phase_names, fontsize=10)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f'{cm_phase_norm[i,j]:.2f}',
                       ha='center', va='center', fontsize=11,
                       color='white' if cm_phase_norm[i,j] > 0.5 else 'black')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'Temporal Phase (acc={phase_acc:.3f}, chance=0.333)',
                    fontsize=11)

        # Panel (1,0): Bar chart of all decoding accuracies
        ax = axes[1, 0]
        region_results = [r for r in results_all if r['region'] == region_label]
        tasks = [r['task'] for r in region_results]
        accs = [r['accuracy'] for r in region_results]
        chances = [r['chance'] for r in region_results]
        x_pos = np.arange(len(tasks))
        bars = ax.bar(x_pos, accs, 0.5, color=['#2196F3', '#4CAF50', '#FF9800',
                                                 '#9C27B0', '#D32F2F'][:len(tasks)],
                      edgecolor='black', linewidth=0.5)
        ax.bar(x_pos, chances, 0.5, fill=False, edgecolor='red',
               linewidth=2, linestyle='--', label='Chance')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(tasks, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('Accuracy / R²')
        ax.set_title('Decoding Performance Summary', fontsize=11)
        ax.legend()
        ax.set_ylim(0, 1)
        for bar, acc, chance in zip(bars, accs, chances):
            lift = acc / chance if chance > 0 else acc
            ax.text(bar.get_x() + bar.get_width()/2, acc + 0.02,
                   f'{acc:.3f}\n({lift:.1f}x)',
                   ha='center', va='bottom', fontsize=8, fontweight='bold')

        # Panel (1,1): Per-excursion behavior accuracy
        ax = axes[1, 1]
        exc_acc_sorted = exc_acc_df.sort_values('accuracy')
        colors = ['#D32F2F' if eid in [81] else '#FF9800' if eid in [57]
                  else '#2196F3' for eid in exc_acc_sorted['excursion_id']]
        ax.barh(range(len(exc_acc_sorted)), exc_acc_sorted['accuracy'],
                color=colors, edgecolor='black', linewidth=0.3)
        ax.set_yticks(range(len(exc_acc_sorted)))
        ax.set_yticklabels([f"Exc {int(eid)}" for eid in exc_acc_sorted['excursion_id']],
                          fontsize=5)
        ax.axvline(chance_acc, color='red', linestyle='--', linewidth=1.5,
                  label=f'Chance ({chance_acc:.2f})')
        ax.set_xlabel('Behavior decoding accuracy')
        ax.set_title('Per-Excursion Accuracy', fontsize=11)
        ax.legend(fontsize=8)

        # Panel (1,2): Speed decoding scatter (if available)
        ax = axes[1, 2]
        if S_exc is not None and 'all_preds_spd' in dir():
            pred_mask_s_local = ~np.isnan(all_preds_spd)
            # Subsample for visualization
            n_show = min(3000, pred_mask_s_local.sum())
            idx_show = np.random.choice(np.where(pred_mask_s_local)[0],
                                        n_show, replace=False)
            ax.scatter(all_true_spd[idx_show], all_preds_spd[idx_show],
                      s=3, alpha=0.3, c='#2196F3', rasterized=True)
            lims = [min(all_true_spd[pred_mask_s_local].min(),
                       all_preds_spd[pred_mask_s_local].min()),
                    max(all_true_spd[pred_mask_s_local].max(),
                       all_preds_spd[pred_mask_s_local].max())]
            ax.plot(lims, lims, 'r--', linewidth=1.5)
            ax.set_xlabel('True speed')
            ax.set_ylabel('Predicted speed')
            ax.set_title(f'Speed Decoding (R²={speed_r2:.3f}, r={speed_corr:.3f})',
                        fontsize=11)
        else:
            ax.text(0.5, 0.5, 'No speed data available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Speed Decoding', fontsize=11)

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        outpath = Path("figures") / f"latent_decoding_{region}.png"
        fig.savefig(outpath, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {outpath}")

    # Final summary
    print(f"\n{'='*70}")
    print(f"  DECODING SUMMARY")
    print(f"{'='*70}")
    res_df = pd.DataFrame(results_all)
    print(res_df.to_string(index=False))
    res_df.to_csv("data/latent_decoding_results.csv", index=False, float_format='%.4f')
    print(f"\n  Saved: data/latent_decoding_results.csv")
    print("\nDone!")


if __name__ == "__main__":
    main()
