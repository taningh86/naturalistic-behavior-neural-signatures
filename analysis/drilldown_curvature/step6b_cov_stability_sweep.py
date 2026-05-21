"""
Step 6b: Covariance stability sweep across timescales, in the K=2 ACA subspace.

Motivation: Step 6 used 5 s windows on full-unit covariance and found a directional
but non-significant effect. Two issues:
  - Stage 3 showed the curvature diet-state effect concentrates in the top 2 PCs
    (3x larger Cohen's d at K=2 vs K=full).
  - Curvature itself operates at the 50-150 ms bin-to-bin scale; 5 s windows
    average over many curvature reversals.

This step:
  - Projects ACA to top K=2 PCs per session (matches Stage 3 subspace).
  - Sweeps window sizes: 200 ms, 500 ms, 1 s, 2 s, 5 s (4, 10, 20, 40, 100 bins).
  - Computes covariance stability (Spearman r between consecutive 2x2 covariance
    upper triangles) within rising/falling entropy phases.
  - Bootstraps state contrasts at each window size.

Floor note: 50 ms = 1 bin cannot give a covariance (need >= 2 samples). 200 ms
(4 bins) is the minimum feasible for a 2x2 covariance; even there, estimates
are noisy. The sweep is the test — if a fast-timescale effect exists, it should
appear at one of these scales.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent.parent

CACHE = REPO / 'data' / 'stage3_localization' / '_cache'
S1D = REPO / 'data' / 'dynamics_stage1'
OUTDIR = REPO / 'data' / 'drilldown_curvature'
FIGDIR = REPO / 'figures' / 'drilldown_curvature'
OUTDIR.mkdir(parents=True, exist_ok=True)
FIGDIR.mkdir(parents=True, exist_ok=True)

K_PCS = 2
WINDOW_BINS_LIST = [4, 10, 20, 40, 100]   # 200 ms, 500 ms, 1 s, 2 s, 5 s
PHASE_TYPES = ('rising', 'falling')
EXCLUDE = {13, 23, 24}
N_BOOT = 5000
RNG = np.random.default_rng(20260430)


def load_aca_cached(session):
    f = CACHE / f'session_{session}_aca.npy'
    if not f.exists():
        raise FileNotFoundError(f"ACA cache missing for session {session}: {f}")
    return np.load(f)


def project_k2(matrix, k=K_PCS):
    pca = PCA(n_components=k)
    return pca.fit_transform(matrix), pca.explained_variance_ratio_


def cov_upper_tri_lw(X, k):
    """Ledoit-Wolf covariance upper-triangle (off-diag).

    For k=2, the upper triangle has only one entry (cov[0,1]).
    """
    if X.shape[0] < 2:
        return None
    lw = LedoitWolf()
    lw.fit(X)
    C = lw.covariance_
    iu = np.triu_indices(C.shape[0], k=1)
    return C[iu]


def per_session_at_window(session, phases_df, window_bins, mean_curv_lookup):
    matrix = load_aca_cached(session)
    proj, evr = project_k2(matrix, k=K_PCS)
    T = proj.shape[0]

    phases_sess = phases_df[(phases_df.session == session) &
                            (phases_df.phase_type.isin(PHASE_TYPES))]
    if len(phases_sess) == 0:
        return None

    # K=2 -> upper triangle has 1 entry. Spearman r needs at least 2 elements.
    # Fallback: collect (val_t, val_{t+1}) pairs over all consecutive windows
    # in the session and compute Spearman on the pooled pairs (per session).
    cov_t = []
    cov_tplus = []

    for _, ph in phases_sess.iterrows():
        sb = int(ph.start_bin)
        eb = int(min(ph.end_bin, T))
        if eb - sb < 2 * window_bins:
            continue
        starts = list(range(sb, eb - window_bins + 1, window_bins))
        if len(starts) < 2:
            continue

        triangles = []
        for s in starts:
            seg = proj[s:s + window_bins]
            tri = cov_upper_tri_lw(seg, k=K_PCS)
            if tri is None:
                triangles.append(np.nan)
            else:
                # K=2: tri has 1 entry -> scalar
                triangles.append(float(tri[0]))

        for k in range(len(triangles) - 1):
            a, b = triangles[k], triangles[k + 1]
            if np.isfinite(a) and np.isfinite(b):
                cov_t.append(a)
                cov_tplus.append(b)

    if len(cov_t) < 5:
        return None

    cov_t = np.array(cov_t)
    cov_tplus = np.array(cov_tplus)

    rho, pval = spearmanr(cov_t, cov_tplus)

    return dict(
        session=int(session),
        n_units=int(matrix.shape[1]),
        evr_pc12=float(evr.sum()),
        n_pairs=int(len(cov_t)),
        cov_stability_rho=float(rho) if np.isfinite(rho) else np.nan,
        cov_stability_p=float(pval) if np.isfinite(pval) else np.nan,
        mean_pc12_cov=float(np.mean(cov_t)),
        mean_curvature=mean_curv_lookup.get(int(session), np.nan),
    )


def bootstrap_diff(arr_a, arr_b, n_boot=N_BOOT):
    a = np.asarray(arr_a, dtype=float)
    b = np.asarray(arr_b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return None
    obs = float(np.mean(a) - np.mean(b))
    boots = np.empty(n_boot)
    na, nb = len(a), len(b)
    for i in range(n_boot):
        boots[i] = (np.mean(RNG.choice(a, na, replace=True)) -
                    np.mean(RNG.choice(b, nb, replace=True)))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return obs, float(lo), float(hi)


def main():
    print('Loading phase + state metadata...')
    df = pd.read_csv(S1D / 'all_sessions_summary.csv')
    state_lookup = df.groupby('session').state.first().to_dict()
    df_rf = df[df.phase_type.isin(PHASE_TYPES)]
    mean_curv_lookup = df_rf.groupby('session').mean_curv_ACA.mean().to_dict()

    sessions = sorted(set(df.session) - EXCLUDE)
    print(f'Sweeping {len(WINDOW_BINS_LIST)} window sizes x {len(sessions)} sessions...\n')

    all_rows = []
    state_diffs = []

    for w in WINDOW_BINS_LIST:
        ms = w * 50
        print(f'-- window = {w} bins ({ms} ms) --')
        rows = []
        for s in sessions:
            try:
                row = per_session_at_window(s, df, w, mean_curv_lookup)
            except Exception as e:
                print(f'  S{s}: ERROR {e}')
                continue
            if row is None:
                continue
            row['window_bins'] = w
            row['window_ms'] = ms
            row['state'] = state_lookup.get(s)
            rows.append(row)
            all_rows.append(row)

        out_w = pd.DataFrame(rows)
        if len(out_w) == 0:
            print('  no usable sessions')
            continue

        # Per-state means
        per_state = out_w.groupby('state')['cov_stability_rho'].agg(['mean', 'std', 'count'])
        print(f'  Per-state cov_stability_rho:')
        for state, row in per_state.iterrows():
            print(f"    {state:8s}: mean={row['mean']:+.3f}  std={row['std']:.3f}  n={int(row['count'])}")

        # Bootstrap state contrasts
        for s_a, s_b in [('fed', 'fasted'), ('fed', 'fed-HFD'), ('fasted', 'fed-HFD')]:
            a = out_w[out_w.state == s_a].cov_stability_rho.values
            b = out_w[out_w.state == s_b].cov_stability_rho.values
            res = bootstrap_diff(a, b)
            if res is None:
                continue
            obs, lo, hi = res
            excl = (lo > 0) or (hi < 0)
            flag = '***' if excl else 'ns'
            print(f"    {s_a} vs {s_b}: delta={obs:+.4f}  CI=[{lo:+.4f}, {hi:+.4f}]  {flag}")
            state_diffs.append(dict(
                window_bins=w, window_ms=ms,
                state_a=s_a, state_b=s_b,
                n_a=int(len(a)), n_b=int(len(b)),
                mean_a=float(np.mean(a)), mean_b=float(np.mean(b)),
                obs_diff=obs, ci_lo=lo, ci_hi=hi, ci_excl_zero=bool(excl),
            ))

        # Cross-metric correlation per window
        rho_x, p_x = spearmanr(out_w.mean_curvature, out_w.cov_stability_rho)
        print(f'  Cross-metric (curv vs cov_stability): rho={rho_x:+.3f}, p={p_x:.4f}, n={len(out_w)}')
        print()

    pd.DataFrame(all_rows).to_csv(OUTDIR / 'step6b_per_session_window_sweep.csv', index=False)
    pd.DataFrame(state_diffs).to_csv(OUTDIR / 'step6b_state_diff_window_sweep.csv', index=False)
    print(f"Wrote {OUTDIR / 'step6b_per_session_window_sweep.csv'}")
    print(f"Wrote {OUTDIR / 'step6b_state_diff_window_sweep.csv'}")

    # ---- Figure: per-window state means with CIs ----
    diffs_df = pd.DataFrame(state_diffs)
    if len(diffs_df) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        # Left: per-state mean cov stability vs window size
        ax = axes[0]
        all_df = pd.DataFrame(all_rows)
        state_colors = {'fed': '#1f77b4', 'fasted': '#d62728', 'fed-HFD': '#2ca02c'}
        for state in ['fed', 'fasted', 'fed-HFD']:
            sub = all_df[all_df.state == state]
            if len(sub) == 0:
                continue
            grp = sub.groupby('window_ms').cov_stability_rho.agg(['mean', 'std', 'count'])
            sem = grp['std'] / np.sqrt(grp['count'])
            ax.errorbar(grp.index, grp['mean'], yerr=sem,
                        marker='o', capsize=4, color=state_colors[state], label=state)
        ax.set_xscale('log')
        ax.set_xlabel('Window size (ms)')
        ax.set_ylabel('Cov stability (Spearman r consecutive 2x2 cov)')
        ax.set_title('K=2 ACA cov stability vs window size')
        ax.legend()
        ax.grid(alpha=0.3, which='both')

        # Right: bootstrap state diff vs window
        ax = axes[1]
        contrast_colors = {'fed-fasted': '#9467bd', 'fed-fed-HFD': '#ff7f0e',
                            'fasted-fed-HFD': '#8c564b'}
        for contrast, sub in diffs_df.groupby(['state_a', 'state_b']):
            label = f"{contrast[0]} - {contrast[1]}"
            ax.errorbar(sub.window_ms, sub.obs_diff,
                         yerr=[sub.obs_diff - sub.ci_lo, sub.ci_hi - sub.obs_diff],
                         marker='o', capsize=4, label=label,
                         color=contrast_colors.get(label, 'gray'))
        ax.axhline(0, color='k', lw=0.7, ls='--')
        ax.set_xscale('log')
        ax.set_xlabel('Window size (ms)')
        ax.set_ylabel('State diff in cov stability (95% bootstrap CI)')
        ax.set_title('State contrast vs window size')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which='both')

        plt.tight_layout()
        fig.savefig(FIGDIR / 'step6b_cov_stability_window_sweep.png', dpi=140)
        print(f"Wrote {FIGDIR / 'step6b_cov_stability_window_sweep.png'}")


if __name__ == '__main__':
    main()
