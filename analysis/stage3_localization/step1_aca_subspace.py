"""
Stage 3 — Step 1: Localize the ACA mean-curvature diet-state effect to a subspace.

Procedure
---------
For each session (excl. 13/23/24):
  1. Load ACA spike-rate matrix (50 ms bins, smoothed, z-scored — same as Stage 1).
  2. PCA. Project onto top-K components for K in {2, 3, 5, 10, 20, "full"}.
  3. Compute curvature (1 - cos(theta)) on the K-dim projection, smoothed sigma=3 bins
     (same as Stage 1 compute_curvature).
  4. For each rising/falling phase (from Stage 1 phase_data), take mean curvature.

Then per K:
  - session-level mean across rising+falling phases
  - bootstrap fed_vs_fasted, fed_vs_HFD, fasted_vs_HFD diff (5000 resamples)
  - report mean, CI, real-effect retention

Output:
  data/stage3_localization/per_session_subspace_curv.csv
  data/stage3_localization/state_diff_vs_K.csv
  figures/stage3_localization/state_diff_vs_K.png
  figures/stage3_localization/explained_variance_per_session.png

Notes:
  - PCA fit per session on full timecourse, then project full timecourse.
  - Curvature in low-dim is on a different scale than full-space — focus on
    *between-state* differences within K, not absolute magnitude.
  - Cache ACA matrices to data/stage3_localization/_cache/ to avoid reloading.
"""
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / 'analysis' / 'cycles_partD'))
sys.path.insert(0, str(REPO / 'analysis' / 'dynamics_stage1'))

from dp_cycles_lib import load_neural

S1D = REPO / 'data' / 'dynamics_stage1'
OUTDIR = REPO / 'data' / 'stage3_localization'
FIGDIR = REPO / 'figures' / 'stage3_localization'
CACHE = OUTDIR / '_cache'
OUTDIR.mkdir(parents=True, exist_ok=True)
FIGDIR.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

CURV_SIGMA = 3
EXCLUDE = {13, 23, 24}
K_LIST = [2, 3, 5, 10, 20]   # plus 'full' added dynamically
PHASE_TYPES = ('rising', 'falling')
N_BOOT = 5000
RNG = np.random.default_rng(20260427)


def load_aca_cached(session):
    f = CACHE / f'session_{session}_aca.npy'
    if f.exists():
        return np.load(f)
    print(f'  S{session}: loading ACA...')
    matrix, _, _ = load_neural(session, 'ACA')
    np.save(f, matrix.astype(np.float32))
    return matrix


def compute_curv(matrix, sigma=CURV_SIGMA, eps=1e-9):
    diff = np.diff(matrix, axis=0)
    norms = np.linalg.norm(diff, axis=1)
    v_a = diff[:-1]
    v_b = diff[1:]
    n_a = norms[:-1]
    n_b = norms[1:]
    cos_t = np.einsum('ij,ij->i', v_a, v_b) / np.maximum(n_a * n_b, eps)
    cos_t = np.clip(cos_t, -1.0, 1.0)
    curv = 1.0 - cos_t
    return gaussian_filter1d(curv, sigma=sigma)


def session_state_lookup():
    df = pd.read_csv(S1D / 'all_sessions_summary.csv')
    return df.groupby('session').state.first().to_dict()


def session_phase_table():
    df = pd.read_csv(S1D / 'all_sessions_summary.csv')
    return df


def per_session_curv_means(session, phases_df, K_list_with_full):
    """Return rows: one per (session, K, phase_id) with mean_curv."""
    matrix = load_aca_cached(session)
    n_units = matrix.shape[1]
    pca = PCA(n_components=min(matrix.shape[0], n_units))
    proj_full = pca.fit_transform(matrix)
    explained = pca.explained_variance_ratio_

    phases_sess = phases_df[(phases_df.session == session) &
                             (phases_df.phase_type.isin(PHASE_TYPES))]
    rows = []
    for K_label in K_list_with_full:
        if K_label == 'full':
            proj = proj_full
            K_eff = proj.shape[1]
        else:
            K_eff = min(K_label, n_units)
            proj = proj_full[:, :K_eff]
        curv = compute_curv(proj)  # length T-2
        for _, ph in phases_sess.iterrows():
            sb = int(ph.start_bin)
            eb = int(min(ph.end_bin, len(curv)))
            if eb - sb < 2:
                continue
            seg = curv[sb:eb]
            rows.append(dict(
                session=session,
                K_label=K_label,
                K_eff=K_eff,
                n_units=n_units,
                phase_id=int(ph.phase_id),
                phase_type=ph.phase_type,
                mean_curv=float(np.mean(seg)),
                duration_bins=eb - sb,
            ))
    return rows, explained


def bootstrap_diff(arr_a, arr_b, n_boot=N_BOOT):
    a = np.asarray(arr_a)
    b = np.asarray(arr_b)
    obs = np.mean(a) - np.mean(b)
    boots = np.empty(n_boot)
    na, nb = len(a), len(b)
    for i in range(n_boot):
        boots[i] = np.mean(RNG.choice(a, na, replace=True)) - \
                   np.mean(RNG.choice(b, nb, replace=True))
    lo = np.percentile(boots, 2.5)
    hi = np.percentile(boots, 97.5)
    return obs, lo, hi


def main():
    state_lookup = session_state_lookup()
    phases_df = session_phase_table()
    sessions = sorted(s for s in state_lookup if s not in EXCLUDE)
    print(f'Sessions: {sessions}')

    K_list_with_full = K_LIST + ['full']

    all_rows = []
    explained_records = []
    for session in sessions:
        try:
            rows, exp = per_session_curv_means(session, phases_df, K_list_with_full)
        except Exception as e:
            print(f'  S{session}: FAILED {e}')
            continue
        all_rows.extend(rows)
        for k in range(min(len(exp), 25)):
            explained_records.append(dict(session=session, comp=k + 1, ev_ratio=float(exp[k])))
        print(f'  S{session}: {len(rows)} phase rows, top-5 var-explained = '
              f'{exp[:5].round(3).tolist()}')

    long = pd.DataFrame(all_rows)
    long['state'] = long['session'].map(state_lookup)
    long.to_csv(OUTDIR / 'per_session_subspace_curv.csv', index=False)
    pd.DataFrame(explained_records).to_csv(OUTDIR / 'explained_variance_long.csv', index=False)

    # Aggregate session-level means (rising+falling pooled per K per session)
    sess_means = long.groupby(['session', 'state', 'K_label'])['mean_curv'].mean().reset_index()
    sess_means.to_csv(OUTDIR / 'session_means_per_K.csv', index=False)

    # State diffs per K
    diff_rows = []
    for K_label in K_list_with_full:
        sub = sess_means[sess_means.K_label == K_label]
        fed = sub[sub.state == 'fed']['mean_curv'].values
        fas = sub[sub.state == 'fasted']['mean_curv'].values
        hfd = sub[sub.state == 'fed-HFD']['mean_curv'].values
        for label, a, b in [('fed_vs_fasted', fed, fas),
                            ('fed_vs_fed-HFD', fed, hfd),
                            ('fasted_vs_fed-HFD', fas, hfd)]:
            obs, lo, hi = bootstrap_diff(a, b)
            diff_rows.append(dict(
                K_label=K_label,
                contrast=label,
                n_a=len(a), n_b=len(b),
                mean_a=float(np.mean(a)),
                mean_b=float(np.mean(b)),
                mean_diff=obs,
                ci_lo=lo, ci_hi=hi,
                excludes_zero=bool((lo > 0) or (hi < 0)),
            ))
    diff_df = pd.DataFrame(diff_rows)
    diff_df.to_csv(OUTDIR / 'state_diff_vs_K.csv', index=False)

    print('\n=== State diff vs K ===')
    print(diff_df.to_string(index=False))

    # ---- Figures ----
    # 1. State diff vs K
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    contrasts = ['fed_vs_fasted', 'fed_vs_fed-HFD', 'fasted_vs_fed-HFD']
    K_xticks = [str(k) for k in K_LIST] + ['full']
    for ax, c in zip(axes, contrasts):
        sub = diff_df[diff_df.contrast == c]
        x = np.arange(len(sub))
        ax.errorbar(x, sub['mean_diff'].values,
                    yerr=[sub['mean_diff'] - sub['ci_lo'],
                          sub['ci_hi'] - sub['mean_diff']],
                    fmt='o-', color='C0', capsize=4)
        for i, row in enumerate(sub.itertuples()):
            if row.excludes_zero:
                ax.scatter(i, row.mean_diff, marker='*', color='red', s=120, zorder=5)
        ax.axhline(0, ls='--', color='k', alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(K_xticks)
        ax.set_xlabel('Subspace dimensionality K')
        ax.set_title(c.replace('_', ' '))
    axes[0].set_ylabel('mean curv diff (a - b)')
    fig.suptitle('ACA curvature state-diff vs PCA subspace dim. (red star = CI excludes 0)')
    fig.tight_layout()
    fig.savefig(FIGDIR / 'state_diff_vs_K.png', dpi=140)
    plt.close(fig)

    # 2. Explained variance per session
    ev_long = pd.DataFrame(explained_records)
    fig, ax = plt.subplots(figsize=(8, 5))
    for sess in sorted(ev_long.session.unique()):
        sub = ev_long[ev_long.session == sess]
        ax.plot(sub['comp'], np.cumsum(sub['ev_ratio']), alpha=0.5)
    for K in K_LIST:
        ax.axvline(K, ls='--', color='k', alpha=0.3)
    ax.set_xlabel('PC index')
    ax.set_ylabel('cumulative variance explained')
    ax.set_title('ACA cumulative variance per session')
    fig.tight_layout()
    fig.savefig(FIGDIR / 'explained_variance_per_session.png', dpi=140)
    plt.close(fig)

    print(f'\nWrote {OUTDIR}')
    print(f'Wrote {FIGDIR}')


if __name__ == '__main__':
    main()
