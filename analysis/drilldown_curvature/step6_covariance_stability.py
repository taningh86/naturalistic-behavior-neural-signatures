"""
Step 6: Validate curvature interpretation via population covariance stability.

Hypothesis: if higher ACA curvature in fasted/HFD reflects unstable population
covariance structure across consecutive time windows (rather than a firing-rate
variability effect), then fasted/HFD sessions should show LOWER covariance
stability — the off-diagonal co-firing pattern should be less similar between
adjacent time windows.

Procedure (per session, ACA only):
  1. Load cached z-scored 50 ms-binned ACA matrix.
  2. Load entropy phase definitions (rising / falling only, matching Stage 1).
  3. Within each rising/falling phase, slide non-overlapping windows of
     WINDOW_BINS (5 s default).
  4. In each window, compute Ledoit-Wolf shrinkage covariance matrix.
  5. For each consecutive window pair, compute Spearman correlation between
     the vectorized upper triangles (off-diagonal only).
  6. Aggregate to per-session mean covariance similarity.

Cross-validation:
  - Per-session correlate (mean curvature, mean cov-similarity) — Spearman.
    Expectation: negative correlation (high curv ↔ low cov stability).
  - Bootstrap diff between states (fed vs fasted, fed vs HFD).

Outputs:
  data/drilldown_curvature/step6_per_session_cov_stability.csv
  data/drilldown_curvature/step6_state_diff.csv
  figures/drilldown_curvature/step6_cov_stability_vs_curvature.png
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.covariance import LedoitWolf
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent.parent

CACHE = REPO / 'data' / 'stage3_localization' / '_cache'
S1D = REPO / 'data' / 'dynamics_stage1'
OUTDIR = REPO / 'data' / 'drilldown_curvature'
FIGDIR = REPO / 'figures' / 'drilldown_curvature'
OUTDIR.mkdir(parents=True, exist_ok=True)
FIGDIR.mkdir(parents=True, exist_ok=True)

WINDOW_BINS = 100        # 5 s windows at 50 ms bins
PHASE_TYPES = ('rising', 'falling')
EXCLUDE = {13, 23, 24}
N_BOOT = 5000
RNG = np.random.default_rng(20260430)


def load_aca_cached(session):
    f = CACHE / f'session_{session}_aca.npy'
    if not f.exists():
        raise FileNotFoundError(
            f"ACA cache missing for session {session}: {f}. "
            "Re-run a Stage 3 step or pre-warm the cache.")
    return np.load(f)


def cov_upper_tri(X):
    """Ledoit-Wolf covariance, return vectorized upper triangle (off-diag)."""
    if X.shape[0] < X.shape[1] + 1:
        # Underdetermined - LW still works but flag
        pass
    lw = LedoitWolf()
    lw.fit(X)
    C = lw.covariance_
    iu = np.triu_indices(C.shape[0], k=1)
    return C[iu]


def per_session_cov_stability(session, phases_df, mean_curv_lookup):
    matrix = load_aca_cached(session)
    n_units = matrix.shape[1]
    T = matrix.shape[0]

    phases_sess = phases_df[(phases_df.session == session) &
                            (phases_df.phase_type.isin(PHASE_TYPES))]
    if len(phases_sess) == 0:
        return None

    pair_corrs = []
    n_windows_total = 0

    for _, ph in phases_sess.iterrows():
        sb = int(ph.start_bin)
        eb = int(min(ph.end_bin, T))
        if eb - sb < 2 * WINDOW_BINS:
            continue
        # Build sequential non-overlapping windows
        starts = list(range(sb, eb - WINDOW_BINS + 1, WINDOW_BINS))
        if len(starts) < 2:
            continue

        triangles = []
        for s in starts:
            seg = matrix[s:s + WINDOW_BINS]
            tri = cov_upper_tri(seg)
            triangles.append(tri)
        n_windows_total += len(triangles)

        for k in range(len(triangles) - 1):
            r, _ = spearmanr(triangles[k], triangles[k + 1])
            if np.isfinite(r):
                pair_corrs.append(r)

    if len(pair_corrs) < 3:
        return None

    return dict(
        session=int(session),
        state=str(phases_sess.state.iloc[0]) if 'state' in phases_sess.columns else None,
        n_units=int(n_units),
        n_windows=int(n_windows_total),
        n_pairs=int(len(pair_corrs)),
        mean_cov_stability=float(np.mean(pair_corrs)),
        median_cov_stability=float(np.median(pair_corrs)),
        std_cov_stability=float(np.std(pair_corrs)),
        mean_curvature=mean_curv_lookup.get(int(session), np.nan),
    )


def bootstrap_diff(arr_a, arr_b, n_boot=N_BOOT):
    a = np.asarray(arr_a)
    b = np.asarray(arr_b)
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

    # Pre-compute mean curvature per session (rising+falling phases) to cross-check
    df_rf = df[df.phase_type.isin(PHASE_TYPES)]
    mean_curv_lookup = df_rf.groupby('session').mean_curv_ACA.mean().to_dict()

    sessions = sorted(set(df.session) - EXCLUDE)
    print(f'Processing {len(sessions)} sessions: {sessions}')

    rows = []
    for s in sessions:
        print(f'  S{s} (state={state_lookup.get(s, "?")})...', end=' ', flush=True)
        try:
            row = per_session_cov_stability(s, df, mean_curv_lookup)
        except Exception as e:
            print(f'ERROR: {e}')
            continue
        if row is None:
            print('skipped (insufficient phases)')
            continue
        row['state'] = state_lookup.get(s, row.get('state'))
        rows.append(row)
        print(f"n_pairs={row['n_pairs']}, "
              f"mean_cov_stab={row['mean_cov_stability']:.3f}, "
              f"mean_curv={row['mean_curvature']:.3f}")

    out = pd.DataFrame(rows).sort_values('session')
    out.to_csv(OUTDIR / 'step6_per_session_cov_stability.csv', index=False)
    print(f"\nWrote {OUTDIR / 'step6_per_session_cov_stability.csv'}")

    # ---- Per-state summary ----
    print('\nPer-state means (mean cov stability across sessions):')
    print(out.groupby('state')[['mean_cov_stability', 'mean_curvature', 'n_units']]
          .agg(['mean', 'std', 'count']).round(3))

    # ---- State contrast bootstraps ----
    diffs = []
    for s_a, s_b in [('fed', 'fasted'), ('fed', 'fed-HFD'), ('fasted', 'fed-HFD')]:
        a = out[out.state == s_a].mean_cov_stability.values
        b = out[out.state == s_b].mean_cov_stability.values
        if len(a) < 2 or len(b) < 2:
            continue
        obs, lo, hi = bootstrap_diff(a, b)
        excl = (lo > 0) or (hi < 0)
        diffs.append(dict(
            metric='mean_cov_stability', state_a=s_a, state_b=s_b,
            n_a=len(a), n_b=len(b), mean_a=float(np.mean(a)), mean_b=float(np.mean(b)),
            obs_diff=obs, ci_lo=lo, ci_hi=hi, ci_excl_zero=bool(excl),
        ))
        flag = '***' if excl else 'ns'
        print(f"  {s_a} vs {s_b}: delta={obs:+.4f}  "
              f"CI=[{lo:+.4f}, {hi:+.4f}]  {flag}")
    pd.DataFrame(diffs).to_csv(OUTDIR / 'step6_state_diff.csv', index=False)

    # ---- Cross-check: per-session correlation between curvature and cov stability ----
    rho, pval = spearmanr(out.mean_curvature, out.mean_cov_stability)
    print(f"\nCross-metric Spearman: curvature vs cov_stability: "
          f"rho = {rho:+.3f}, p = {pval:.4f}, n = {len(out)}")
    print('  (expect rho < 0 if curvature reflects covariance instability)')

    # ---- Figure ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: scatter, colored by state
    state_colors = {'fed': '#1f77b4', 'fasted': '#d62728', 'fed-HFD': '#2ca02c'}
    ax = axes[0]
    for s, sub in out.groupby('state'):
        ax.scatter(sub.mean_curvature, sub.mean_cov_stability,
                   color=state_colors.get(s, 'gray'), s=70, edgecolor='k',
                   linewidth=0.5, label=f"{s} (n={len(sub)})")
    ax.set_xlabel('Session mean ACA curvature\n(rising+falling phases)')
    ax.set_ylabel('Session mean covariance stability\n(Spearman r between consecutive 5 s windows)')
    ax.set_title(f'Curvature vs cov stability\nrho = {rho:+.3f}, p = {pval:.4f}')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Right: per-state boxplot of cov stability
    ax = axes[1]
    states_in = ['fed', 'fasted', 'fed-HFD']
    data = [out[out.state == s].mean_cov_stability.values for s in states_in]
    bp = ax.boxplot(data, labels=states_in, showmeans=True, patch_artist=True)
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(state_colors[states_in[i]])
        patch.set_alpha(0.5)
    for i, vals in enumerate(data):
        jitter = np.random.normal(0, 0.05, len(vals))
        ax.scatter(np.full(len(vals), i + 1) + jitter, vals,
                   alpha=0.7, s=40, color='k', zorder=3)
    ax.set_ylabel('mean cov stability (per session)')
    ax.set_title('Cov stability by state')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGDIR / 'step6_cov_stability_vs_curvature.png', dpi=140)
    print(f"Wrote {FIGDIR / 'step6_cov_stability_vs_curvature.png'}")


if __name__ == '__main__':
    main()
