"""
How exactly does the classifier assign excursion IDs from hidden states?

Tests:
1. First-point-only: can just the initial hidden state classify the excursion?
   (tests if it's purely initial conditions / where in the slow drift you are)
2. Centroid-only: classify using just the mean hidden state per excursion
3. Centroid-subtracted: within-excursion shape only (already shows 5x lift)
4. Feature ablation: which dimensions of hidden state carry excursion info?
5. Temporal position control: can session time alone predict excursion ID
   as well as hidden states can?
6. Within-excursion spread analysis: do excursions differ in cluster shape?
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
import warnings

warnings.filterwarnings('ignore')

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


def cv_accuracy(X, y, n_splits=5):
    """Stratified k-fold CV accuracy."""
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(le.classes_)
    if n_classes < 2:
        return 0.0, 1.0 / max(n_classes, 1)
    scaler = StandardScaler()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs = []
    for train_idx, test_idx in skf.split(X, y_enc):
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        clf = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs',
                                 multi_class='multinomial')
        clf.fit(X_tr, y_enc[train_idx])
        accs.append(accuracy_score(y_enc[test_idx], clf.predict(X_te)))
    chance = 1.0 / n_classes
    return np.mean(accs), chance


def main():
    print("=" * 70)
    print("  How Does the Classifier Assign Excursion IDs?")
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

        exc_ids = np.full(len(hs_time_sec), -1, dtype=int)
        for _, erow in complete.iterrows():
            mask = (hs_time_sec >= erow['start_time']) & (hs_time_sec <= erow['end_time'])
            exc_ids[mask] = int(erow['excursion_id'])

        in_exc = exc_ids >= 0
        H = hidden_all[in_exc]
        E = exc_ids[in_exc]
        T = hs_time_sec[in_exc]

        # Filter to excursions with >= 30 points
        exc_counts = pd.Series(E).value_counts()
        valid_exc = exc_counts[exc_counts >= 30].index.values
        valid_mask = np.isin(E, valid_exc)
        H = H[valid_mask]
        E = E[valid_mask]
        T = T[valid_mask]
        unique_exc = np.unique(E)
        n_exc = len(unique_exc)

        print(f"  {len(H)} points in {n_exc} excursions (>=30 pts each)")

        # =================================================================
        # 1. BASELINE: full hidden state -> excursion ID
        # =================================================================
        acc_full, chance = cv_accuracy(H, E)
        print(f"\n  1. Full 32D hidden state -> Exc ID:")
        print(f"     Accuracy: {acc_full:.3f}  (chance={chance:.3f}, lift={acc_full/chance:.1f}x)")

        # =================================================================
        # 2. TIME-ONLY CONTROL: can raw timestamp predict excursion ID?
        # =================================================================
        acc_time, _ = cv_accuracy(T.reshape(-1, 1), E)
        print(f"\n  2. Time-only (1D) -> Exc ID:")
        print(f"     Accuracy: {acc_time:.3f}  (lift={acc_time/chance:.1f}x)")

        # Polynomial time features (quadratic)
        T_poly = np.column_stack([T, T**2, T**3])
        acc_time_poly, _ = cv_accuracy(T_poly, E)
        print(f"     Time cubic (3D): {acc_time_poly:.3f}  (lift={acc_time_poly/chance:.1f}x)")

        # =================================================================
        # 3. FIRST POINT ONLY: initial hidden state per excursion
        # =================================================================
        first_h = []
        first_e = []
        for eid in unique_exc:
            emask = E == eid
            idx = np.where(emask)[0]
            first_h.append(H[idx[0]])
            first_e.append(eid)
        first_h = np.array(first_h)
        first_e = np.array(first_e)
        # Can't do CV with 1 sample per class, so use nearest-centroid
        # Leave-one-out: for each excursion, find nearest centroid among others
        correct = 0
        for i in range(n_exc):
            dists = np.linalg.norm(first_h - first_h[i], axis=1)
            dists[i] = np.inf  # exclude self
            nearest = np.argmin(dists)
            # Is the nearest first-point from a temporally adjacent excursion?
        # Instead, just report how well first-point separates in PCA
        print(f"\n  3. First-point analysis:")
        pca_first = PCA(n_components=min(5, n_exc)).fit(first_h)
        print(f"     First-point PCA var: {pca_first.explained_variance_ratio_[:3]*100}")

        # Nearest-neighbor leave-one-out on first points
        nn_correct = 0
        for i in range(n_exc):
            dists = np.linalg.norm(first_h - first_h[i], axis=1)
            dists[i] = np.inf
            nn_correct += (first_e[np.argmin(dists)] == first_e[i])
        # This is trivially 0 since each excursion appears once...
        # Instead: assign ALL points using nearest first-point
        first_h_map = {eid: first_h[i] for i, eid in enumerate(first_e)}
        preds_nn = []
        for h_pt in H:
            dists = np.array([np.linalg.norm(h_pt - first_h_map[eid])
                              for eid in unique_exc])
            preds_nn.append(unique_exc[np.argmin(dists)])
        acc_nn = accuracy_score(E, preds_nn)
        print(f"     Nearest-first-point classifier: {acc_nn:.3f} (lift={acc_nn/chance:.1f}x)")

        # =================================================================
        # 4. CENTROID-ONLY: nearest centroid classifier
        # =================================================================
        centroids = {eid: H[E == eid].mean(axis=0) for eid in unique_exc}
        preds_cent = []
        for h_pt in H:
            dists = np.array([np.linalg.norm(h_pt - centroids[eid])
                              for eid in unique_exc])
            preds_cent.append(unique_exc[np.argmin(dists)])
        acc_cent = accuracy_score(E, preds_cent)
        print(f"\n  4. Nearest-centroid classifier: {acc_cent:.3f} (lift={acc_cent/chance:.1f}x)")

        # =================================================================
        # 5. RELATIVE TIME within excursion -> excursion ID
        #    (tests if trajectory shape/speed differs)
        # =================================================================
        T_rel = np.zeros_like(T)
        for eid in unique_exc:
            emask = E == eid
            t_exc = T[emask]
            T_rel[emask] = (t_exc - t_exc.min()) / (t_exc.max() - t_exc.min() + 1e-8)

        # Hidden state + relative time vs hidden state alone
        H_with_trel = np.column_stack([H, T_rel])
        acc_with_trel, _ = cv_accuracy(H_with_trel, E)
        print(f"\n  5. Hidden state + relative time: {acc_with_trel:.3f} (vs {acc_full:.3f} without)")

        # =================================================================
        # 6. DIMENSION-BY-DIMENSION: which hidden dims carry exc info?
        # =================================================================
        print(f"\n  6. Per-dimension excursion separability (F-ratio):")
        f_ratios = []
        for d in range(HIDDEN_SIZE):
            # One-way ANOVA F-ratio for this dimension
            grand_mean = H[:, d].mean()
            ss_between = sum((E == eid).sum() * (H[E == eid, d].mean() - grand_mean)**2
                             for eid in unique_exc)
            ss_within = sum(np.sum((H[E == eid, d] - H[E == eid, d].mean())**2)
                            for eid in unique_exc)
            df_between = n_exc - 1
            df_within = len(H) - n_exc
            f_ratio = (ss_between / df_between) / (ss_within / df_within + 1e-10)
            f_ratios.append(f_ratio)
        f_ratios = np.array(f_ratios)
        top_dims = np.argsort(f_ratios)[::-1]
        print(f"     Top 5 dims: {top_dims[:5]} (F={f_ratios[top_dims[:5]]})")
        print(f"     Bottom 5 dims: {top_dims[-5:]} (F={f_ratios[top_dims[-5:]]})")
        print(f"     Mean F: {f_ratios.mean():.1f}, Max F: {f_ratios.max():.1f}, Min F: {f_ratios.min():.1f}")

        # Classify using only top-5 dims vs bottom-5 dims
        acc_top5, _ = cv_accuracy(H[:, top_dims[:5]], E)
        acc_bot5, _ = cv_accuracy(H[:, top_dims[-5:]], E)
        acc_top10, _ = cv_accuracy(H[:, top_dims[:10]], E)
        print(f"     Top 5 dims only: {acc_top5:.3f} (lift={acc_top5/chance:.1f}x)")
        print(f"     Top 10 dims only: {acc_top10:.3f} (lift={acc_top10/chance:.1f}x)")
        print(f"     Bottom 5 dims only: {acc_bot5:.3f} (lift={acc_bot5/chance:.1f}x)")

        # =================================================================
        # 7. WITHIN-EXCURSION TRAJECTORY STATS
        # =================================================================
        print(f"\n  7. Within-excursion trajectory properties:")
        exc_stats = []
        for eid in unique_exc:
            emask = E == eid
            h_exc = H[emask]
            t_exc = T[emask]
            n_pts = len(h_exc)

            centroid = h_exc.mean(axis=0)
            spread = np.mean(np.linalg.norm(h_exc - centroid, axis=1))
            max_spread = np.max(np.linalg.norm(h_exc - centroid, axis=1))

            # Trajectory length (total path in 32D)
            diffs = np.diff(h_exc, axis=0)
            path_length = np.sum(np.linalg.norm(diffs, axis=1))
            mean_step = np.mean(np.linalg.norm(diffs, axis=1))

            # Net displacement (start to end)
            displacement = np.linalg.norm(h_exc[-1] - h_exc[0])

            # Tortuosity
            tortuosity = path_length / (displacement + 1e-8)

            # Duration
            duration = t_exc[-1] - t_exc[0]

            exc_stats.append({
                'excursion_id': eid,
                'n_points': n_pts,
                'duration': duration,
                'centroid_norm': np.linalg.norm(centroid),
                'spread': spread,
                'max_spread': max_spread,
                'path_length': path_length,
                'mean_step': mean_step,
                'displacement': displacement,
                'tortuosity': tortuosity,
            })

        stats_df = pd.DataFrame(exc_stats)
        print(f"     {'Metric':<20} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
        for col in ['centroid_norm', 'spread', 'path_length', 'mean_step',
                     'displacement', 'tortuosity']:
            print(f"     {col:<20} {stats_df[col].mean():>8.3f} {stats_df[col].std():>8.3f} "
                  f"{stats_df[col].min():>8.3f} {stats_df[col].max():>8.3f}")

        # Correlation between centroid_norm and session time
        from scipy.stats import spearmanr
        exc_mid_times = np.array([T[E == eid].mean() for eid in unique_exc])
        r_cent_time, p_cent_time = spearmanr(exc_mid_times, stats_df['centroid_norm'])
        print(f"\n     Centroid norm vs session time: rho={r_cent_time:.3f}, p={p_cent_time:.3e}")
        r_spread_dur, p_spread_dur = spearmanr(stats_df['duration'], stats_df['spread'])
        print(f"     Spread vs duration: rho={r_spread_dur:.3f}, p={p_spread_dur:.3e}")

        # =================================================================
        # 8. KEY TEST: Does time alone match hidden states?
        # =================================================================
        print(f"\n  8. Summary comparison:")
        print(f"     {'Method':<35} {'Accuracy':>8} {'Lift':>6}")
        print(f"     {'-'*50}")
        print(f"     {'Full 32D hidden state':<35} {acc_full:>8.3f} {acc_full/chance:>6.1f}x")
        print(f"     {'Time only (1D)':<35} {acc_time:>8.3f} {acc_time/chance:>6.1f}x")
        print(f"     {'Time cubic (3D)':<35} {acc_time_poly:>8.3f} {acc_time_poly/chance:>6.1f}x")
        print(f"     {'Nearest first-point':<35} {acc_nn:>8.3f} {acc_nn/chance:>6.1f}x")
        print(f"     {'Nearest centroid':<35} {acc_cent:>8.3f} {acc_cent/chance:>6.1f}x")
        print(f"     {'Top 5 dims only':<35} {acc_top5:>8.3f} {acc_top5/chance:>6.1f}x")
        print(f"     {'Top 10 dims only':<35} {acc_top10:>8.3f} {acc_top10/chance:>6.1f}x")
        print(f"     {'Bottom 5 dims only':<35} {acc_bot5:>8.3f} {acc_bot5/chance:>6.1f}x")
        print(f"     {'Hidden + relative time':<35} {acc_with_trel:>8.3f} {acc_with_trel/chance:>6.1f}x")
        print(f"     {'Chance':<35} {chance:>8.3f} {1.0:>6.1f}x")

        # =================================================================
        # FIGURE
        # =================================================================
        fig, axes = plt.subplots(2, 3, figsize=(22, 14))
        fig.suptitle(f"{region_label} — How the Classifier Assigns Excursion IDs",
                     fontsize=14, fontweight='bold')

        # Panel (0,0): PCA of hidden states colored by time
        ax = axes[0, 0]
        pca = PCA(n_components=3).fit(H)
        H_pca = pca.transform(H)
        sc = ax.scatter(H_pca[:, 0], H_pca[:, 1], c=T, cmap='viridis',
                        s=3, alpha=0.4, rasterized=True)
        plt.colorbar(sc, ax=ax, label='Session time (s)', shrink=0.8)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax.set_title('Hidden states colored by session TIME')

        # Panel (0,1): Centroid positions in PCA, colored by time
        ax = axes[0, 1]
        cent_array = np.array([centroids[eid] for eid in unique_exc])
        cent_pca = pca.transform(cent_array)
        mid_times = np.array([T[E == eid].mean() for eid in unique_exc])
        sc2 = ax.scatter(cent_pca[:, 0], cent_pca[:, 1], c=mid_times,
                         cmap='viridis', s=80, edgecolors='black', linewidths=0.5,
                         zorder=5)
        for i, eid in enumerate(unique_exc):
            ax.annotate(str(eid), (cent_pca[i, 0], cent_pca[i, 1]),
                        fontsize=5, ha='center', va='center')
        plt.colorbar(sc2, ax=ax, label='Mid-time (s)', shrink=0.8)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax.set_title('Excursion centroids colored by time')

        # Panel (0,2): F-ratio per hidden dimension
        ax = axes[0, 2]
        sorted_f = np.sort(f_ratios)[::-1]
        ax.bar(range(HIDDEN_SIZE), sorted_f, color='#2196F3', edgecolor='black',
               linewidth=0.3)
        ax.set_xlabel('Hidden dimension (sorted by F-ratio)')
        ax.set_ylabel('ANOVA F-ratio')
        ax.set_title('Per-dimension excursion separability')
        ax.axhline(1.0, color='red', linestyle='--', label='F=1 (no separation)')
        ax.legend()

        # Panel (1,0): Method comparison bar chart
        ax = axes[1, 0]
        methods = ['Full 32D', 'Time (1D)', 'Time³ (3D)', 'Near. 1st pt',
                    'Near. centroid', 'Top 5D', 'Top 10D', 'Bot 5D']
        accs_list = [acc_full, acc_time, acc_time_poly, acc_nn,
                     acc_cent, acc_top5, acc_top10, acc_bot5]
        colors = ['#2196F3', '#FF5722', '#FF8A65', '#9C27B0',
                  '#4CAF50', '#FFC107', '#FFD54F', '#BDBDBD']
        bars = ax.bar(range(len(methods)), accs_list, color=colors,
                      edgecolor='black', linewidth=0.5)
        ax.axhline(chance, color='red', linestyle='--', linewidth=2, label=f'Chance ({chance:.3f})')
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('Accuracy')
        ax.set_title('Excursion ID Classification by Method')
        ax.legend()
        for bar, a in zip(bars, accs_list):
            ax.text(bar.get_x() + bar.get_width()/2, a + 0.005,
                    f'{a:.3f}\n({a/chance:.1f}x)', ha='center', fontsize=7,
                    fontweight='bold')

        # Panel (1,1): Excursion trajectory spread vs duration
        ax = axes[1, 1]
        sc3 = ax.scatter(stats_df['duration'], stats_df['spread'],
                         c=mid_times, cmap='viridis', s=50,
                         edgecolors='black', linewidths=0.5)
        plt.colorbar(sc3, ax=ax, label='Session time (s)', shrink=0.8)
        for _, row in stats_df.iterrows():
            ax.annotate(str(int(row['excursion_id'])),
                        (row['duration'], row['spread']),
                        fontsize=5, ha='left')
        ax.set_xlabel('Excursion duration (s)')
        ax.set_ylabel('Mean spread in 32D (from centroid)')
        ax.set_title(f'Trajectory spread vs duration\n(rho={r_spread_dur:.2f}, p={p_spread_dur:.2e})')

        # Panel (1,2): Example excursion trajectories in PCA
        ax = axes[1, 2]
        # Pick 6 excursions spread across session
        sample_exc = unique_exc[np.linspace(0, n_exc-1, 6, dtype=int)]
        cmap_sample = plt.cm.Set1
        for i, eid in enumerate(sample_exc):
            emask = E == eid
            h_exc_pca = H_pca[emask]
            color = cmap_sample(i / 6)
            ax.plot(h_exc_pca[:, 0], h_exc_pca[:, 1], '-', color=color,
                    linewidth=0.8, alpha=0.7)
            ax.scatter(h_exc_pca[0, 0], h_exc_pca[0, 1], c=[color], s=50,
                       marker='o', edgecolors='black', linewidths=0.5, zorder=5)
            ax.scatter(h_exc_pca[-1, 0], h_exc_pca[-1, 1], c=[color], s=50,
                       marker='s', edgecolors='black', linewidths=0.5, zorder=5)
            ax.annotate(f'Exc {eid}', (h_exc_pca[0, 0], h_exc_pca[0, 1]),
                        fontsize=7, fontweight='bold', color=color)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax.set_title('Example excursion trajectories\n(circle=start, square=end)')

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        outpath = Path("figures") / f"latent_excursion_mechanism_{region}.png"
        fig.savefig(outpath, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"\n  Saved: {outpath}")

    print("\nDone!")


if __name__ == "__main__":
    main()
