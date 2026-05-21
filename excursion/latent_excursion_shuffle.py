"""
Shuffle test: Is excursion ID decoding from GRU-ODE hidden states
meaningful, or is it just the hidden state acting as a noisy clock?

Tests:
1. True excursion ID decoding (baseline)
2. Temporally-shuffled excursion labels: randomly reassign excursion IDs
   to time windows of the same size, preserving the temporal structure
   of the hidden states but breaking the neural-excursion link
3. Circular shift: shift all excursion labels forward in time by a random
   offset, preserving excursion duration structure but misaligning with
   neural data
4. Raw spike PCA control: can raw neural data (no GRU-ODE) separate
   excursions just as well? If yes, the GRU-ODE adds nothing.
"""

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spikeinterface.extractors as se
import torch
import torch.nn as nn
from torchdiffeq import odeint
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from scipy.ndimage import uniform_filter1d
import warnings

warnings.filterwarnings('ignore')

BIN_SIZE_MS = 10
FS = 30000
BIN_SAMPLES = int(BIN_SIZE_MS * FS / 1000)
SEQ_LEN = 50
PRED_BINS = 10
HIDDEN_SIZE = 32
ODE_GATE_HIDDEN = 64
ODE_SOLVER = 'rk4'
ODE_STEP_SIZE = 1.0
ODE_DT = 1.0
LHA_DEPTH_MAX = 1300
RSP_DEPTH_MIN = 1300
STRIDE = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_SHUFFLES = 100
import sys

with open("paths.yaml") as f:
    paths_config = yaml.safe_load(f)


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


def cv_exc_accuracy(X, E, valid_exc, n_splits=5):
    """Excursion ID classification accuracy with stratified CV."""
    mask = np.isin(E, valid_exc)
    X_sub = X[mask]
    E_sub = E[mask]
    le = LabelEncoder()
    E_enc = le.fit_transform(E_sub)
    n_classes = len(le.classes_)
    if n_classes < 2:
        return 0.0, 1.0
    scaler = StandardScaler()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs = []
    for train_idx, test_idx in skf.split(X_sub, E_enc):
        X_tr = scaler.fit_transform(X_sub[train_idx])
        X_te = scaler.transform(X_sub[test_idx])
        clf = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs',
                                 multi_class='multinomial')
        clf.fit(X_tr, E_enc[train_idx])
        accs.append(accuracy_score(E_enc[test_idx], clf.predict(X_te)))
    return np.mean(accs), 1.0 / n_classes


def main():
    print("=" * 70)
    print("  Excursion ID Shuffle Test")
    print(f"  {N_SHUFFLES} shuffle iterations")
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

        # Load model and extract hidden states
        model_path = Path("data") / f"gru_ode_10ms_poisson_{region}_fed_model.pt"
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

        s_data, s_time, s_bins = bin_spike_trains(sorting, unit_ids)
        print(f"  Extracting hidden states...")

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

        hs_indices = [i * STRIDE + SEQ_LEN - 1 for i in range(len(hidden_all))]
        hs_time_sec = s_time[np.array(hs_indices)]

        # Assign true excursion IDs
        exc_ids = np.full(len(hs_time_sec), -1, dtype=int)
        for _, erow in complete.iterrows():
            mask = (hs_time_sec >= erow['start_time']) & (hs_time_sec <= erow['end_time'])
            exc_ids[mask] = int(erow['excursion_id'])

        in_exc = exc_ids >= 0
        H = hidden_all[in_exc]
        E = exc_ids[in_exc]
        T = hs_time_sec[in_exc]

        exc_counts = pd.Series(E).value_counts()
        valid_exc = exc_counts[exc_counts >= 30].index.values
        valid_mask = np.isin(E, valid_exc)
        H_v = H[valid_mask]
        E_v = E[valid_mask]
        T_v = T[valid_mask]
        unique_exc = np.sort(np.unique(E_v))
        n_exc = len(unique_exc)

        print(f"  {len(H_v)} points in {n_exc} excursions")
        sys.stdout.flush()

        # =================================================================
        # 1. TRUE accuracy
        # =================================================================
        true_acc, chance = cv_exc_accuracy(H_v, E_v, unique_exc)
        print(f"\n  TRUE excursion ID accuracy: {true_acc:.4f} (chance={chance:.4f}, lift={true_acc/chance:.1f}x)")
        sys.stdout.flush()

        # =================================================================
        # 2. SHUFFLE: randomly permute excursion labels across time points
        #    This breaks excursion-neural link but preserves the marginal
        #    distribution of labels and the hidden state distribution
        # =================================================================
        print(f"\n  Running {N_SHUFFLES} point-wise shuffles...")
        shuffle_accs = []
        rng = np.random.RandomState(42)
        for i in range(N_SHUFFLES):
            E_shuf = rng.permutation(E_v)
            acc_shuf, _ = cv_exc_accuracy(H_v, E_shuf, unique_exc)
            shuffle_accs.append(acc_shuf)
            if (i + 1) % 25 == 0:
                print(f"    {i+1}/{N_SHUFFLES} done, mean so far: {np.mean(shuffle_accs):.4f}")
                sys.stdout.flush()
        shuffle_accs = np.array(shuffle_accs)
        p_value = (np.sum(shuffle_accs >= true_acc) + 1) / (N_SHUFFLES + 1)
        print(f"  Point-shuffle: mean={shuffle_accs.mean():.4f}, std={shuffle_accs.std():.4f}")
        print(f"  p-value (true >= shuffle): {p_value:.4f}")

        # =================================================================
        # 3. BLOCK SHUFFLE: permute which excursion ID is assigned to which
        #    time window. This preserves temporal contiguity within each
        #    "excursion" but scrambles which neural activity maps to which ID.
        # =================================================================
        print(f"\n  Running {N_SHUFFLES} block shuffles (permute excursion labels)...")
        block_shuffle_accs = []
        for i in range(N_SHUFFLES):
            # Create a random mapping: real exc ID -> shuffled exc ID
            perm = rng.permutation(unique_exc)
            label_map = dict(zip(unique_exc, perm))
            E_block_shuf = np.array([label_map[e] for e in E_v])
            acc_bs, _ = cv_exc_accuracy(H_v, E_block_shuf, unique_exc)
            block_shuffle_accs.append(acc_bs)
            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{N_SHUFFLES} done, mean so far: {np.mean(block_shuffle_accs):.4f}")
        block_shuffle_accs = np.array(block_shuffle_accs)
        p_block = (np.sum(block_shuffle_accs >= true_acc) + 1) / (N_SHUFFLES + 1)
        print(f"  Block-shuffle: mean={block_shuffle_accs.mean():.4f}, std={block_shuffle_accs.std():.4f}")
        print(f"  p-value (true >= block-shuffle): {p_block:.4f}")

        # =================================================================
        # 4. RAW SPIKE PCA CONTROL: does raw neural data (no GRU-ODE)
        #    separate excursions equally well?
        # =================================================================
        print(f"\n  Raw spike PCA control...")
        # Use same binning as GRU-ODE input but at stride=10 to match
        # the hidden state time points
        raw_indices = np.array(hs_indices)
        raw_at_hs = s_data[raw_indices]  # raw z-scored spike counts at same times
        raw_in_exc = raw_at_hs[in_exc]
        raw_v = raw_in_exc[valid_mask]

        # PCA to 32D to match hidden state dimensionality
        pca_raw = PCA(n_components=32).fit(raw_v)
        raw_pca = pca_raw.transform(raw_v)
        acc_raw_pca, _ = cv_exc_accuracy(raw_pca, E_v, unique_exc)
        print(f"  Raw spike PCA (32D): {acc_raw_pca:.4f} (lift={acc_raw_pca/chance:.1f}x)")

        # Also try raw spikes without PCA
        acc_raw_full, _ = cv_exc_accuracy(raw_v, E_v, unique_exc)
        print(f"  Raw spikes (full {raw_v.shape[1]}D): {acc_raw_full:.4f} (lift={acc_raw_full/chance:.1f}x)")

        # Smoothed raw spikes (500ms window = 50 bins of 10ms)
        smooth_bins = 50
        raw_smooth = np.zeros_like(s_data)
        for col in range(s_data.shape[1]):
            raw_smooth[:, col] = uniform_filter1d(s_data[:, col], size=smooth_bins, mode='constant')
        raw_smooth_at_hs = raw_smooth[raw_indices]
        raw_smooth_exc = raw_smooth_at_hs[in_exc][valid_mask]
        pca_smooth = PCA(n_components=32).fit(raw_smooth_exc)
        raw_smooth_pca = pca_smooth.transform(raw_smooth_exc)
        acc_smooth_pca, _ = cv_exc_accuracy(raw_smooth_pca, E_v, unique_exc)
        print(f"  Smoothed spike PCA (32D, 500ms): {acc_smooth_pca:.4f} (lift={acc_smooth_pca/chance:.1f}x)")

        # =================================================================
        # FIGURE
        # =================================================================
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"{region_label} — Excursion ID Shuffle Test ({N_SHUFFLES} iterations)",
                     fontsize=14, fontweight='bold')

        # Panel (0,0): Point-shuffle null distribution
        ax = axes[0, 0]
        ax.hist(shuffle_accs, bins=30, color='#BBDEFB', edgecolor='#1565C0',
                linewidth=0.5, alpha=0.8, label='Point-shuffle null')
        ax.axvline(true_acc, color='#D32F2F', linewidth=2.5,
                   label=f'True acc = {true_acc:.4f}')
        ax.axvline(chance, color='grey', linewidth=1.5, linestyle='--',
                   label=f'Chance = {chance:.4f}')
        ax.set_xlabel('Accuracy')
        ax.set_ylabel('Count')
        ax.set_title(f'Point-Shuffle Null Distribution\np = {p_value:.4f}')
        ax.legend(fontsize=9)

        # Panel (0,1): Block-shuffle null distribution
        ax = axes[0, 1]
        ax.hist(block_shuffle_accs, bins=30, color='#C8E6C9', edgecolor='#2E7D32',
                linewidth=0.5, alpha=0.8, label='Block-shuffle null')
        ax.axvline(true_acc, color='#D32F2F', linewidth=2.5,
                   label=f'True acc = {true_acc:.4f}')
        ax.axvline(chance, color='grey', linewidth=1.5, linestyle='--',
                   label=f'Chance = {chance:.4f}')
        ax.set_xlabel('Accuracy')
        ax.set_ylabel('Count')
        ax.set_title(f'Block-Shuffle Null Distribution\np = {p_block:.4f}')
        ax.legend(fontsize=9)

        # Panel (1,0): Comparison bar chart
        ax = axes[1, 0]
        methods = ['True\n(GRU-ODE 32D)', 'Point-shuffle\nmean', 'Block-shuffle\nmean',
                   'Raw spike\nPCA 32D', f'Raw spike\nfull {raw_v.shape[1]}D',
                   'Smoothed spike\nPCA 32D', 'Chance']
        values = [true_acc, shuffle_accs.mean(), block_shuffle_accs.mean(),
                  acc_raw_pca, acc_raw_full, acc_smooth_pca, chance]
        colors = ['#D32F2F', '#90CAF9', '#A5D6A7',
                  '#FFB74D', '#FFCC80', '#FFE082', '#E0E0E0']
        bars = ax.bar(range(len(methods)), values, color=colors,
                      edgecolor='black', linewidth=0.5)
        # Error bars for shuffles
        ax.errorbar(1, shuffle_accs.mean(), yerr=shuffle_accs.std()*2,
                    fmt='none', color='black', capsize=5, linewidth=2)
        ax.errorbar(2, block_shuffle_accs.mean(), yerr=block_shuffle_accs.std()*2,
                    fmt='none', color='black', capsize=5, linewidth=2)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, fontsize=8)
        ax.set_ylabel('Accuracy')
        ax.set_title('All Methods Compared')
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.003,
                    f'{val:.3f}\n({val/chance:.1f}x)',
                    ha='center', fontsize=8, fontweight='bold')

        # Panel (1,1): Interpretation text
        ax = axes[1, 1]
        ax.axis('off')

        # Build interpretation
        lines = []
        lines.append(f"TRUE accuracy: {true_acc:.4f} ({true_acc/chance:.1f}x chance)")
        lines.append(f"")
        lines.append(f"POINT-SHUFFLE null: {shuffle_accs.mean():.4f} +/- {shuffle_accs.std():.4f}")
        lines.append(f"  p = {p_value:.4f}")
        if p_value < 0.05:
            lines.append(f"  => TRUE > point-shuffle (significant)")
            lines.append(f"  => Hidden states carry excursion-specific info")
            lines.append(f"     beyond random label assignment")
        else:
            lines.append(f"  => TRUE not better than point-shuffle")
            lines.append(f"  => Hidden states DON'T carry excursion-specific info")
        lines.append(f"")
        lines.append(f"BLOCK-SHUFFLE null: {block_shuffle_accs.mean():.4f} +/- {block_shuffle_accs.std():.4f}")
        lines.append(f"  p = {p_block:.4f}")
        if p_block < 0.05:
            lines.append(f"  => TRUE > block-shuffle (significant)")
            lines.append(f"  => The classifier uses temporal contiguity")
            lines.append(f"     (which points belong together in time)")
        else:
            lines.append(f"  => TRUE matches block-shuffle")
            lines.append(f"  => Block-shuffled labels are just as classifiable")
            lines.append(f"  => Classifier just uses temporal contiguity,")
            lines.append(f"     not neural content specific to each excursion")
        lines.append(f"")
        lines.append(f"RAW SPIKE CONTROLS:")
        lines.append(f"  PCA 32D: {acc_raw_pca:.4f} ({acc_raw_pca/chance:.1f}x)")
        lines.append(f"  Full {raw_v.shape[1]}D: {acc_raw_full:.4f} ({acc_raw_full/chance:.1f}x)")
        lines.append(f"  Smoothed PCA 32D: {acc_smooth_pca:.4f} ({acc_smooth_pca/chance:.1f}x)")
        if acc_raw_pca > true_acc * 0.8:
            lines.append(f"  => Raw spikes separate excursions comparably")
            lines.append(f"  => GRU-ODE adds little beyond raw neural signal")

        text = '\n'.join(lines)
        ax.text(0.05, 0.95, text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        outpath = Path("figures") / f"latent_excursion_shuffle_{region}.png"
        fig.savefig(outpath, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"\n  Saved: {outpath}")

    print("\nDone!")


if __name__ == "__main__":
    main()
