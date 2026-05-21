"""
Step 7: Probe ACA-LHA opposition across multiple metrics, contrasting HFD
with fed/fasted.

Background: dp_entropy_neural_signatures.py established that at entropy
peaks, ACA FR drops and LHA FR rises (and vice versa at troughs); HFD shows
no significant FR / PC1 opposition. We test whether the same state-dependent
opposition pattern shows up in trajectory speed and curvature, supporting
the FR/PC1 finding from a different angle.

Procedure (uses pre-computed Stage 1 phase summary, no recomputation):
  - Load all_sessions_summary.csv (per-session, per-phase metrics).
  - For each metric in {speed, curv, fr, pc1}:
    - Per session, compute Pearson r between mean_X_ACA and mean_X_LHA
      across phases. Negative r = opposition.
    - Two phase pools:
        (a) peak + trough only (peri-inflection focal)
        (b) all phases (rising, falling, peak, trough)
    - Per state, summarize mean r and bootstrap state contrasts.

Hypothesis support: if HFD lacks opposition (consistent with FR/PC1 finding),
then HFD per-session r should be LESS NEGATIVE (closer to zero or positive)
than fed/fasted, in additional metrics beyond FR.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent.parent
S1D = REPO / 'data' / 'dynamics_stage1'
OUTDIR = REPO / 'data' / 'drilldown_curvature'
FIGDIR = REPO / 'figures' / 'drilldown_curvature'
OUTDIR.mkdir(parents=True, exist_ok=True)
FIGDIR.mkdir(parents=True, exist_ok=True)

EXCLUDE_SESSIONS = {13, 23, 24}
N_BOOT = 5000
RNG = np.random.default_rng(20260430)

METRICS = [
    ('speed', 'mean_speed_ACA', 'mean_speed_LHA'),
    ('curv',  'mean_curv_ACA',  'mean_curv_LHA'),
    ('fr',    'mean_fr_ACA',    'mean_fr_LHA'),
    ('pc1',   'mean_pc1_ACA',   'mean_pc1_LHA'),
]


def per_session_r(df_sess, col_a, col_b):
    a = df_sess[col_a].values
    b = df_sess[col_b].values
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 4:
        return np.nan, mask.sum()
    r, _ = pearsonr(a[mask], b[mask])
    return r, mask.sum()


def bootstrap_diff(arr_a, arr_b, n_boot=N_BOOT):
    a = np.asarray(arr_a, dtype=float)
    b = np.asarray(arr_b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return None
    obs = float(np.mean(a) - np.mean(b))
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = (np.mean(RNG.choice(a, len(a), replace=True)) -
                    np.mean(RNG.choice(b, len(b), replace=True)))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return obs, float(lo), float(hi)


def bootstrap_mean(arr, n_boot=N_BOOT):
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) < 2:
        return None
    obs = float(np.mean(a))
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = float(np.mean(RNG.choice(a, len(a), replace=True)))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return obs, float(lo), float(hi)


def analyze_pool(df, pool_label, phase_mask):
    print(f'\n========== Phase pool: {pool_label} ==========')
    rows = []
    summary_rows = []
    contrast_rows = []

    df_use = df[phase_mask].copy()

    for metric_name, col_a, col_b in METRICS:
        print(f'\n  -- metric: {metric_name} (ACA vs LHA cross-region r per session) --')
        per_sess = []
        for snum, sub in df_use.groupby('session'):
            if snum in EXCLUDE_SESSIONS:
                continue
            r, n = per_session_r(sub, col_a, col_b)
            if not np.isfinite(r):
                continue
            state = sub.state.iloc[0]
            rows.append(dict(pool=pool_label, metric=metric_name, session=int(snum),
                             state=state, n_phases=int(n), r=float(r)))
            per_sess.append((state, r))

        per_state = {}
        for state, r in per_sess:
            per_state.setdefault(state, []).append(r)

        for state in ['fed', 'fasted', 'fed-HFD']:
            arr = per_state.get(state, [])
            res = bootstrap_mean(arr)
            if res is None:
                continue
            obs, lo, hi = res
            excl = (lo > 0) or (hi < 0)
            tag = '***' if excl else 'ns'
            print(f"    {state:8s} mean r = {obs:+.3f}  CI=[{lo:+.3f}, {hi:+.3f}]  n={len(arr)}  "
                  f"{'excl 0' if excl else 'CI brackets 0'}  {tag}")
            summary_rows.append(dict(
                pool=pool_label, metric=metric_name, state=state,
                n_sessions=len(arr), mean_r=obs, ci_lo=lo, ci_hi=hi,
                ci_excl_zero=bool(excl),
            ))

        # State contrasts: HFD - fed, HFD - fasted, fed - fasted
        for s_a, s_b in [('fed-HFD', 'fed'), ('fed-HFD', 'fasted'), ('fed', 'fasted')]:
            a = per_state.get(s_a, [])
            b = per_state.get(s_b, [])
            res = bootstrap_diff(a, b)
            if res is None:
                continue
            obs, lo, hi = res
            excl = (lo > 0) or (hi < 0)
            tag = '***' if excl else 'ns'
            print(f"    {s_a} vs {s_b}: delta={obs:+.3f}  CI=[{lo:+.3f}, {hi:+.3f}]  {tag}")
            contrast_rows.append(dict(
                pool=pool_label, metric=metric_name, state_a=s_a, state_b=s_b,
                obs_diff=obs, ci_lo=lo, ci_hi=hi, ci_excl_zero=bool(excl),
            ))

    return rows, summary_rows, contrast_rows


def main():
    df = pd.read_csv(S1D / 'all_sessions_summary.csv')
    print(f'Loaded {len(df)} phase rows across {df.session.nunique()} sessions')

    pools = {
        'peak_trough': df.phase_type.isin(['peak', 'trough']),
        'all_phases':  df.phase_type.isin(['peak', 'trough', 'rising', 'falling']),
    }

    all_rows, all_summary, all_contrast = [], [], []
    for pool_label, mask in pools.items():
        r, s, c = analyze_pool(df, pool_label, mask)
        all_rows += r
        all_summary += s
        all_contrast += c

    pd.DataFrame(all_rows).to_csv(OUTDIR / 'step7_per_session_opposition.csv', index=False)
    pd.DataFrame(all_summary).to_csv(OUTDIR / 'step7_state_summary.csv', index=False)
    pd.DataFrame(all_contrast).to_csv(OUTDIR / 'step7_state_contrasts.csv', index=False)
    print(f"\nWrote {OUTDIR / 'step7_per_session_opposition.csv'}")
    print(f"Wrote {OUTDIR / 'step7_state_summary.csv'}")
    print(f"Wrote {OUTDIR / 'step7_state_contrasts.csv'}")

    # ---- Figure: mean r per state x metric, with CI bars; rows = pools ----
    summary_df = pd.DataFrame(all_summary)
    metrics_order = [m[0] for m in METRICS]
    states = ['fed', 'fasted', 'fed-HFD']
    state_colors = {'fed': '#1f77b4', 'fasted': '#d62728', 'fed-HFD': '#2ca02c'}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, pool_label in zip(axes, pools.keys()):
        sub = summary_df[summary_df.pool == pool_label]
        x = np.arange(len(metrics_order))
        width = 0.27
        for i, state in enumerate(states):
            ssub = sub[sub.state == state].set_index('metric').reindex(metrics_order)
            xs = x + (i - 1) * width
            means = ssub.mean_r.values
            err_lo = ssub.mean_r.values - ssub.ci_lo.values
            err_hi = ssub.ci_hi.values - ssub.mean_r.values
            ax.bar(xs, means, width=width, color=state_colors[state],
                   yerr=[err_lo, err_hi], capsize=3, label=state, alpha=0.85,
                   edgecolor='black', linewidth=0.5)
        ax.axhline(0, color='k', lw=0.7, ls='--')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_order)
        ax.set_xlabel('metric')
        ax.set_title(f'Phase pool: {pool_label}')
        ax.grid(alpha=0.3, axis='y')
        if ax is axes[0]:
            ax.set_ylabel('Mean session-level Pearson r\n(ACA vs LHA across phases)')
            ax.legend(loc='best')
    fig.suptitle('ACA-LHA cross-region correlation by metric and state\n'
                 '(more negative = stronger opposition)')
    plt.tight_layout()
    fig.savefig(FIGDIR / 'step7_aca_lha_opposition.png', dpi=140)
    print(f"Wrote {FIGDIR / 'step7_aca_lha_opposition.png'}")


if __name__ == '__main__':
    main()
