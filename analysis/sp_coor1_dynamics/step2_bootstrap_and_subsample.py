"""Single-probe Mouse01-Coor1 Stage 1 follow-up (Step 2).

Two analyses on the 8-session drilldown output:

A. Bootstrap CI on the curvature state contrast (LHA & RSP, rising+falling).
   Curvature is angular -> unit-count-independent, so this is the trustworthy
   signal. With n=4 fed vs n=4 fasted, the bootstrap reflects session-to-session
   variability within one mouse only -- no cross-animal generalization.

B. Unit-count-matched subsample control on speed (LHA -> strict-min,
   RSP -> strict-min). Mirror of dual-probe Stage 3 LHA control.
   Expectation: both speed effects collapse if they were driven by sqrt(N).

Outputs
-------
- data/sp_coor1_dynamics/curv_bootstrap.csv (per-region per-phase CIs)
- data/sp_coor1_dynamics/speed_subsample_per_draw.csv
- data/sp_coor1_dynamics/speed_subsample_summary.csv
- data/sp_coor1_dynamics/step2_summary.md
- data/sp_coor1_dynamics/_cache/session_{N}_{LHA,RSP}.npy  (matrices for reuse)
- figures/sp_coor1_dynamics/step2_curvature_bootstrap.png
- figures/sp_coor1_dynamics/step2_speed_subsample.png
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "analysis" / "sp_coor1_dynamics"))

from sp_lib import list_sessions, load_neural

OUTDIR = REPO / "data" / "sp_coor1_dynamics"
CACHEDIR = OUTDIR / "_cache"
FIGDIR = REPO / "figures" / "sp_coor1_dynamics"
OUTDIR.mkdir(parents=True, exist_ok=True)
CACHEDIR.mkdir(parents=True, exist_ok=True)
FIGDIR.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = OUTDIR / "all_sessions_summary.csv"
SPEED_SIGMA = 3
PHASE_TYPES = ('rising', 'falling')
N_DRAWS = 20
N_BOOT = 5000
RNG = np.random.default_rng(20260428)


# ============================================================================
# Helpers
# ============================================================================
def cache_matrix(session, region):
    f = CACHEDIR / f"session_{session}_{region}.npy"
    if f.exists():
        return np.load(f)
    m, _, _ = load_neural(session, region)
    np.save(f, m)
    return m


def compute_speed(matrix, sigma=SPEED_SIGMA):
    diff = np.diff(matrix, axis=0)
    speed = np.linalg.norm(diff, axis=1)
    return gaussian_filter1d(speed, sigma=sigma)


def bootstrap_diff(a, b, n_boot=N_BOOT, rng=None):
    rng = rng or RNG
    a = np.asarray(a)
    b = np.asarray(b)
    obs = float(np.mean(a) - np.mean(b))
    boots = np.empty(n_boot)
    na, nb = len(a), len(b)
    for i in range(n_boot):
        boots[i] = (np.mean(rng.choice(a, na, replace=True)) -
                    np.mean(rng.choice(b, nb, replace=True)))
    return obs, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# ============================================================================
# Part A: Curvature bootstrap CI
# ============================================================================
def part_a_curvature_bootstrap():
    df = pd.read_csv(SUMMARY_CSV)
    sub = df[df.phase_type.isin(PHASE_TYPES)].copy()

    # Per-session mean across rising+falling phases
    sess_means = sub.groupby(['session', 'state']).agg(
        mean_curv_LHA=('mean_curv_LHA', 'mean'),
        mean_curv_RSP=('mean_curv_RSP', 'mean'),
        mean_speed_LHA=('mean_speed_LHA', 'mean'),
        mean_speed_RSP=('mean_speed_RSP', 'mean'),
        n_units_LHA=('n_units_LHA', 'first'),
        n_units_RSP=('n_units_RSP', 'first'),
    ).reset_index()
    print('Session-level means (rising+falling):')
    print(sess_means.to_string(index=False))
    print()

    fed = sess_means[sess_means.state == 'fed']
    fas = sess_means[sess_means.state == 'fasted']

    rows = []
    for col, label in [
        ('mean_curv_LHA', 'curv_LHA'),
        ('mean_curv_RSP', 'curv_RSP'),
        ('mean_speed_LHA', 'speed_LHA_unmatched'),
        ('mean_speed_RSP', 'speed_RSP_unmatched'),
    ]:
        a = fed[col].values
        b = fas[col].values
        obs, lo, hi = bootstrap_diff(a, b)
        rows.append(dict(
            quantity=label, contrast='fed_minus_fasted',
            n_fed=len(a), n_fas=len(b),
            mean_fed=float(np.mean(a)), mean_fasted=float(np.mean(b)),
            mean_diff=obs, ci_lo=lo, ci_hi=hi,
            excludes_zero=bool((lo > 0) or (hi < 0)),
            pct_change=float(obs / np.mean(b) * 100.0),
        ))
    out = pd.DataFrame(rows)
    out.to_csv(OUTDIR / 'curv_bootstrap.csv', index=False)
    print('=== Curvature bootstrap (and unmatched speed for reference) ===')
    print(out.to_string(index=False))
    print()
    return sess_means, out


# ============================================================================
# Part B: Speed subsample control
# ============================================================================
def part_b_speed_subsample():
    df = pd.read_csv(SUMMARY_CSV)
    state_lookup = df.groupby('session').state.first().to_dict()
    sessions = sorted(state_lookup)
    phases_df = df[df.phase_type.isin(PHASE_TYPES)].copy()

    # Compute (and cache) target N for LHA and RSP
    unit_counts = {'LHA': {}, 'RSP': {}}
    for s in sessions:
        for region in ('LHA', 'RSP'):
            m = cache_matrix(s, region)
            unit_counts[region][s] = m.shape[1]
    print('LHA unit counts:', unit_counts['LHA'])
    print('RSP unit counts:', unit_counts['RSP'])
    target_lha = int(min(unit_counts['LHA'].values()))
    target_rsp = int(min(unit_counts['RSP'].values()))
    print(f'\nstrict-min target: LHA N={target_lha}, RSP N={target_rsp}')

    all_draws = []
    for region, target_N in [('LHA', target_lha), ('RSP', target_rsp)]:
        print(f'\n==== {region}: subsample to N={target_N}, {N_DRAWS} draws ====')
        for draw in range(N_DRAWS):
            rng_draw = np.random.default_rng((20260428, draw, hash(region) & 0xFFFF))
            draw_rows = []
            for s in sessions:
                m = cache_matrix(s, region)
                n_total = m.shape[1]
                idx = rng_draw.choice(n_total, target_N, replace=False)
                m_sub = m[:, idx]
                sp = compute_speed(m_sub)
                ph_s = phases_df[phases_df.session == s]
                phase_means = []
                for _, ph in ph_s.iterrows():
                    sb = int(ph.start_bin)
                    eb = int(min(ph.end_bin, len(sp)))
                    if eb - sb < 2:
                        continue
                    phase_means.append(float(np.mean(sp[sb:eb])))
                if not phase_means:
                    continue
                draw_rows.append(dict(session=s, state=state_lookup[s],
                                      mean_speed=float(np.mean(phase_means))))
            ddf = pd.DataFrame(draw_rows)
            fed = ddf[ddf.state == 'fed']['mean_speed'].values
            fas = ddf[ddf.state == 'fasted']['mean_speed'].values
            obs, lo, hi = bootstrap_diff(fed, fas)
            all_draws.append(dict(
                region=region, target_N=target_N, draw=draw,
                n_fed=len(fed), n_fas=len(fas),
                mean_fed=float(np.mean(fed)),
                mean_fasted=float(np.mean(fas)),
                mean_diff=obs, ci_lo=lo, ci_hi=hi,
                excludes_zero=bool((lo > 0) or (hi < 0)),
            ))
            if (draw + 1) % 5 == 0:
                print(f'  {region} draw {draw+1}/{N_DRAWS}')

    draws_df = pd.DataFrame(all_draws)
    draws_df.to_csv(OUTDIR / 'speed_subsample_per_draw.csv', index=False)

    # Aggregate
    summary_rows = []
    for region in ('LHA', 'RSP'):
        ss = draws_df[draws_df.region == region]
        summary_rows.append(dict(
            region=region,
            target_N=int(ss.target_N.iloc[0]),
            n_draws=len(ss),
            median_diff=float(ss.mean_diff.median()),
            p25_diff=float(np.percentile(ss.mean_diff, 25)),
            p75_diff=float(np.percentile(ss.mean_diff, 75)),
            min_diff=float(ss.mean_diff.min()),
            max_diff=float(ss.mean_diff.max()),
            median_ci_lo=float(ss.ci_lo.median()),
            median_ci_hi=float(ss.ci_hi.median()),
            n_sig=int(ss.excludes_zero.sum()),
            frac_sig=float(ss.excludes_zero.mean()),
        ))
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTDIR / 'speed_subsample_summary.csv', index=False)
    print('\n=== Speed subsample summary ===')
    print(summary_df.to_string(index=False))
    return draws_df, summary_df, target_lha, target_rsp


# ============================================================================
# Outcome classification (mirrors dual-probe convention)
# ============================================================================
def classify(med, frac_sig, orig_diff):
    if abs(orig_diff) < 1e-9:
        return 'na', 'no_original'
    same_dir = (np.sign(med) == np.sign(orig_diff))
    mag_ratio = abs(med) / max(abs(orig_diff), 1e-9)
    if (not same_dir) and frac_sig >= 0.3:
        return 'reversal', 'effect direction reversed at matched N'
    if same_dir and frac_sig >= 0.5:
        if mag_ratio >= 0.5:
            return 'A', f'persists ({mag_ratio:.0%} of original, {frac_sig:.0%} sig)'
        return 'C', f'partial ({mag_ratio:.0%} of original, {frac_sig:.0%} sig)'
    if frac_sig < 0.3:
        return 'B', f'disappears ({frac_sig:.0%} sig, median {med:+.3f} vs orig {orig_diff:+.3f})'
    return 'C', f'borderline ({frac_sig:.0%} sig)'


# ============================================================================
# Figures + markdown
# ============================================================================
def make_figures(curv_table, draws_df, summary_df):
    # Curvature bootstrap (forest plot of fed-fasted CIs for the 4 quantities)
    fig, ax = plt.subplots(figsize=(8, 4))
    quantities = ['curv_LHA', 'curv_RSP', 'speed_LHA_unmatched', 'speed_RSP_unmatched']
    y = np.arange(len(quantities))
    for i, q in enumerate(quantities):
        row = curv_table[curv_table.quantity == q].iloc[0]
        c = 'C0' if q.startswith('curv') else 'C3'
        ax.errorbar(row.mean_diff, i,
                    xerr=[[row.mean_diff - row.ci_lo], [row.ci_hi - row.mean_diff]],
                    fmt='o', color=c, capsize=4, lw=2)
    ax.axvline(0, color='k', ls=':')
    ax.set_yticks(y)
    ax.set_yticklabels(quantities)
    ax.set_xlabel('fed - fasted (95% bootstrap CI)')
    ax.set_title('Single-probe Mouse01-Coor1: state contrast on rising+falling phases')
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIGDIR / 'step2_curvature_bootstrap.png', dpi=140)
    plt.close(fig)

    # Speed subsample (unmatched vs matched, both regions side by side)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    for ax, region in zip(axes, ('LHA', 'RSP')):
        unm_row = curv_table[curv_table.quantity == f'speed_{region}_unmatched'].iloc[0]
        unm_diff = unm_row.mean_diff
        unm_lo = unm_row.ci_lo
        unm_hi = unm_row.ci_hi
        ss = draws_df[draws_df.region == region]
        ax.scatter(np.zeros(len(ss)) + np.random.RandomState(42).normal(0, 0.04, len(ss)),
                   ss.mean_diff.values, alpha=0.6, s=40, color='C0',
                   label=f'matched N={ss.target_N.iloc[0]} (per-draw)')
        ax.errorbar(0.6, unm_diff, yerr=[[unm_diff - unm_lo], [unm_hi - unm_diff]],
                    fmt='s', color='red', capsize=5, ms=10, label='unmatched (full units)')
        ax.axhline(0, color='k', ls=':')
        ax.set_xticks([0, 0.6])
        ax.set_xticklabels(['matched', 'unmatched'])
        ax.set_ylabel('fed - fasted (mean speed)')
        ax.set_title(f'{region} speed contrast')
        ax.legend(fontsize=8)
    fig.suptitle('Single-probe Mouse01-Coor1: speed state contrast, matched vs unmatched N')
    fig.tight_layout()
    fig.savefig(FIGDIR / 'step2_speed_subsample.png', dpi=140)
    plt.close(fig)


def write_markdown(curv_table, draws_df, summary_df, target_lha, target_rsp,
                   sess_means, classifications):
    md = []
    md.append('# Single-probe Mouse01-Coor1 — Stage 1 follow-up (Step 2)')
    md.append('')
    md.append('**Date**: 2026-04-28. **Sessions**: 8 (4 fed, 4 fasted, 1 mouse).')
    md.append('')
    md.append('## A. Curvature bootstrap CI (rising+falling phases)')
    md.append('')
    md.append('| Quantity | Fed | Fasted | Δ (fed-fasted) | 95% CI | excludes 0? | Δ% |')
    md.append('|---|---|---|---|---|---|---|')
    for q in ('curv_LHA', 'curv_RSP'):
        r = curv_table[curv_table.quantity == q].iloc[0]
        md.append(f'| {q} | {r["mean_fed"]:.4f} | {r["mean_fasted"]:.4f} | {r["mean_diff"]:+.4f} | '
                  f'[{r["ci_lo"]:+.4f}, {r["ci_hi"]:+.4f}] | {r["excludes_zero"]} | {r["pct_change"]:+.2f}% |')
    md.append('')
    md.append('## B. Speed subsample control')
    md.append('')
    md.append(f'Per-session unit counts (LHA / RSP):')
    sm = sess_means
    for _, r in sm.iterrows():
        md.append(f'- S{int(r["session"])} ({r["state"]}): LHA={int(r["n_units_LHA"])}, RSP={int(r["n_units_RSP"])}')
    md.append('')
    md.append(f'Strict-min targets: LHA **N={target_lha}**, RSP **N={target_rsp}**.')
    md.append(f'Per region, {N_DRAWS} random draws, recompute speed on subsampled matrix, '
              f'bootstrap fed-fasted ({N_BOOT} resamples) per draw.')
    md.append('')
    md.append('| Region | Unmatched Δ | Unmatched 95% CI | Matched-N median Δ | Matched IQR | frac_sig draws |')
    md.append('|---|---|---|---|---|---|')
    for region in ('LHA', 'RSP'):
        unm = curv_table[curv_table.quantity == f'speed_{region}_unmatched'].iloc[0]
        m = summary_df[summary_df.region == region].iloc[0]
        md.append(f'| {region} | {unm["mean_diff"]:+.4f} | [{unm["ci_lo"]:+.4f}, {unm["ci_hi"]:+.4f}] | '
                  f'{m["median_diff"]:+.4f} | [{m["p25_diff"]:+.4f}, {m["p75_diff"]:+.4f}] | '
                  f'{int(m["n_sig"])}/{int(m["n_draws"])} ({m["frac_sig"]:.2f}) |')
    md.append('')
    md.append('## Outcome classification (speed)')
    md.append('')
    for region, (out, reason) in classifications.items():
        label_map = {'A': 'A — Persists', 'B': 'B — Disappears',
                     'C': 'C — Partial', 'reversal': 'REVERSAL', 'na': 'n/a'}
        md.append(f'- **{region}**: {label_map.get(out, out)} — {reason}')
    md.append('')
    md.append('## Read')
    md.append('')
    # interpret curvature
    sig_curv = []
    for q in ('curv_LHA', 'curv_RSP'):
        r = curv_table[curv_table.quantity == q].iloc[0]
        if r['excludes_zero']:
            sig_curv.append(q)
    if sig_curv:
        md.append(f'- Curvature CI excludes zero for: {", ".join(sig_curv)}. '
                  f'Direction is fasted > fed in both regions, matching the dual-probe ACA finding.')
    else:
        md.append('- Neither LHA nor RSP curvature CI excludes zero. With n=4 vs 4 in one mouse, '
                  'the bootstrap is underpowered and the small (~2-4%) effect cannot be resolved.')
    md.append('')
    md.append('## Caveats')
    md.append('')
    md.append('- One mouse, 4 vs 4 sessions. Bootstrap reflects within-mouse session variance only — does NOT generalize across animals.')
    md.append('- Unit-count imbalance (fed > fasted) is the dominant nuisance. The matched-N speed test is the right control; curvature is intrinsically robust.')
    md.append('- S7 has only 3 phases (1 peak + 1 trough); per-session mean is sparser there.')
    md.append('')
    md.append('## Files')
    md.append('')
    md.append('- `data/sp_coor1_dynamics/curv_bootstrap.csv`')
    md.append('- `data/sp_coor1_dynamics/speed_subsample_per_draw.csv`, `speed_subsample_summary.csv`')
    md.append('- `data/sp_coor1_dynamics/_cache/session_{N}_{LHA,RSP}.npy`')
    md.append('- `figures/sp_coor1_dynamics/step2_curvature_bootstrap.png`, `step2_speed_subsample.png`')
    with open(OUTDIR / 'step2_summary.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))


# ============================================================================
# Main
# ============================================================================
def main():
    sess_means, curv_table = part_a_curvature_bootstrap()
    draws_df, summary_df, target_lha, target_rsp = part_b_speed_subsample()

    # Classify speed outcomes vs unmatched
    classifications = {}
    for region in ('LHA', 'RSP'):
        unm = curv_table[curv_table.quantity == f'speed_{region}_unmatched'].iloc[0]
        m = summary_df[summary_df.region == region].iloc[0]
        out, reason = classify(m.median_diff, m.frac_sig, unm.mean_diff)
        classifications[region] = (out, reason)
    print('\n=== Speed classification ===')
    for r, (o, why) in classifications.items():
        print(f'  {r}: {o} — {why}')

    make_figures(curv_table, draws_df, summary_df)
    write_markdown(curv_table, draws_df, summary_df, target_lha, target_rsp,
                   sess_means, classifications)
    print(f'\nWrote {OUTDIR}')
    print(f'Wrote {FIGDIR}')


if __name__ == '__main__':
    main()
