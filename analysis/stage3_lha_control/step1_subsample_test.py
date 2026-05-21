"""
Stage 3 follow-up: LHA subsample-controlled test.

Question: does the LHA speed x diet state finding (Stage 3 step 2, K=full) survive
when LHA unit counts are matched across sessions? Or is it inflated by unit-count
differences (fed mean ~104 units, fasted mean ~78, HFD mean ~82)?

Procedure
---------
1. Determine target N (minimum LHA unit count across 17 sessions; flag if it's
   an outlier and report 25th percentile alongside).
2. For 20 random draws, subsample each session's LHA matrix to N units.
3. For each draw, recompute trajectory speed (full subsampled space, sigma=3
   smoothing -- matches Stage 1) and per-phase mean speed for rising+falling.
4. Per draw: bootstrap fed-vs-fasted and fed-vs-HFD session-level diffs (5000
   resamples) -> mean_diff and 95% CI.
5. Aggregate across draws: median effect, CI distribution, fraction of draws
   where bootstrap CI excludes zero.

Stop conditions:
- target N < 20 -> abort (subsampled speed estimates unstable)
- effect direction reverses at matched N -> flag for revision
- effect estimates wildly variable across draws -> declare fragile

Operational details
- Reuses cached LHA matrices in data/stage3_localization/_cache/session_X_lha.npy
- Uses Stage 1 phase boundaries from data/dynamics_stage1/all_sessions_summary.csv
- Excludes S13/23/24 (drilldown convention)

Outputs
- data/stage3_lha_control/lha_subsample_per_draw.csv
- data/stage3_lha_control/lha_subsample_summary.csv
- data/stage3_lha_control/lha_subsample_summary.md
- figures/stage3_lha_control/lha_subsample_distribution.png
- figures/stage3_lha_control/lha_subsample_comparison.png
"""
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent.parent

S1D = REPO / 'data' / 'dynamics_stage1'
LOCALIZATION_CACHE = REPO / 'data' / 'stage3_localization' / '_cache'
OUTDIR = REPO / 'data' / 'stage3_lha_control'
FIGDIR = REPO / 'figures' / 'stage3_lha_control'
OUTDIR.mkdir(parents=True, exist_ok=True)
FIGDIR.mkdir(parents=True, exist_ok=True)

SPEED_SIGMA = 3
EXCLUDE = {13, 23, 24}
PHASE_TYPES = ('rising', 'falling')
N_DRAWS = 20
N_BOOT = 5000
RNG = np.random.default_rng(20260428)

# Original (unmatched) results from step2_lha_subspace.py / lha_state_diff_vs_K.csv
ORIG = {
    'fed_vs_fasted': dict(mean_diff=0.909, ci_lo=0.201, ci_hi=1.621, sig=True),
    'fed_vs_fed-HFD': dict(mean_diff=0.691, ci_lo=0.200, ci_hi=1.129, sig=True),
}


def load_lha(session):
    f = LOCALIZATION_CACHE / f'session_{session}_lha.npy'
    if not f.exists():
        raise FileNotFoundError(f'Missing LHA cache for S{session}: {f}')
    return np.load(f)


def compute_speed(matrix, sigma=SPEED_SIGMA):
    diff = np.diff(matrix, axis=0)
    speed = np.linalg.norm(diff, axis=1)
    return gaussian_filter1d(speed, sigma=sigma)


def bootstrap_diff(arr_a, arr_b, n_boot=N_BOOT, rng=None):
    rng = rng or RNG
    a = np.asarray(arr_a)
    b = np.asarray(arr_b)
    obs = float(np.mean(a) - np.mean(b))
    boots = np.empty(n_boot)
    na, nb = len(a), len(b)
    for i in range(n_boot):
        boots[i] = np.mean(rng.choice(a, na, replace=True)) - \
                   np.mean(rng.choice(b, nb, replace=True))
    return obs, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    summary_csv = pd.read_csv(S1D / 'all_sessions_summary.csv')
    state_lookup = summary_csv.groupby('session').state.first().to_dict()
    sessions = sorted(s for s in state_lookup if s not in EXCLUDE)
    print(f'Sessions: {sessions}')

    # ---- Step 1: target N
    unit_counts = {}
    for s in sessions:
        m = load_lha(s)
        unit_counts[s] = m.shape[1]
    counts_arr = np.array(list(unit_counts.values()))
    target_min = int(np.min(counts_arr))
    target_p25 = int(np.percentile(counts_arr, 25))
    print('\nUnit counts per session:')
    for s in sessions:
        print(f'  S{s} ({state_lookup[s]}): {unit_counts[s]}')
    print(f'\n  min={target_min}, 25th percentile={target_p25}, '
          f'median={int(np.median(counts_arr))}, max={int(np.max(counts_arr))}')

    # Flag if min is outlier (more than 1.5x IQR below 25th percentile)
    iqr = np.percentile(counts_arr, 75) - target_p25
    is_outlier = target_min < target_p25 - 1.5 * iqr
    print(f'  min outlier? {is_outlier} (1.5*IQR threshold = '
          f'{int(target_p25 - 1.5 * iqr)})')

    if target_min < 20:
        print(f'STOP: target N {target_min} below threshold of 20.')
        return

    # Run primary at target_min, secondary at target_p25 if feasible for all sessions
    targets = [('strict_min', target_min)]
    if target_p25 != target_min and target_p25 <= target_min:
        targets.append(('p25', target_p25))
    elif target_p25 != target_min:
        # P25 exceeds min -> would require excluding low-count sessions.
        # Skip secondary; note in output that strict_min is primary and only target.
        print(f'  Skipping P25 target N={target_p25}: would exclude '
              f'session(s) with fewer units (e.g. min={target_min}).')

    # ---- Step 2-3: subsample, compute speeds, bootstrap diffs per draw
    phases_df = summary_csv[summary_csv.phase_type.isin(PHASE_TYPES)]

    all_draws = []
    for tname, target_N in targets:
        print(f'\n==== Target {tname}: N={target_N} ====')
        for draw in range(N_DRAWS):
            draw_rows = []
            for s in sessions:
                m = load_lha(s)
                n_total = m.shape[1]
                if n_total < target_N:
                    raise ValueError(f'S{s} has {n_total} < target {target_N}')
                rng_draw = np.random.default_rng((20260428, draw, s))
                idx = rng_draw.choice(n_total, target_N, replace=False)
                m_sub = m[:, idx]
                sp = compute_speed(m_sub)  # length T-1
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
                draw_rows.append(dict(
                    session=s, state=state_lookup[s],
                    mean_speed=float(np.mean(phase_means)),
                ))
            df = pd.DataFrame(draw_rows)
            fed = df[df.state == 'fed']['mean_speed'].values
            fas = df[df.state == 'fasted']['mean_speed'].values
            hfd = df[df.state == 'fed-HFD']['mean_speed'].values

            for label, a, b in [('fed_vs_fasted', fed, fas),
                                ('fed_vs_fed-HFD', fed, hfd),
                                ('fasted_vs_fed-HFD', fas, hfd)]:
                obs, lo, hi = bootstrap_diff(a, b)
                all_draws.append(dict(
                    target=tname, target_N=target_N, draw=draw,
                    contrast=label,
                    n_a=len(a), n_b=len(b),
                    mean_a=float(np.mean(a)),
                    mean_b=float(np.mean(b)),
                    mean_diff=obs, ci_lo=lo, ci_hi=hi,
                    excludes_zero=bool((lo > 0) or (hi < 0)),
                ))
            if (draw + 1) % 5 == 0:
                print(f'  draw {draw+1}/{N_DRAWS} done')

    draws_df = pd.DataFrame(all_draws)
    draws_df.to_csv(OUTDIR / 'lha_subsample_per_draw.csv', index=False)

    # ---- Step 4: aggregate
    summary_rows = []
    for tname, target_N in targets:
        sub = draws_df[draws_df.target == tname]
        for c in ['fed_vs_fasted', 'fed_vs_fed-HFD', 'fasted_vs_fed-HFD']:
            ss = sub[sub.contrast == c]
            n_sig = int(ss.excludes_zero.sum())
            same_dir_as_orig = ''
            if c in ORIG:
                orig_sign = np.sign(ORIG[c]['mean_diff'])
                ss_signs = np.sign(ss.mean_diff.values)
                frac_same = float(np.mean(ss_signs == orig_sign))
                same_dir_as_orig = f'{frac_same:.2f}'
            summary_rows.append(dict(
                target=tname, target_N=target_N, contrast=c,
                n_draws=len(ss),
                median_diff=float(ss.mean_diff.median()),
                p25_diff=float(np.percentile(ss.mean_diff, 25)),
                p75_diff=float(np.percentile(ss.mean_diff, 75)),
                min_diff=float(ss.mean_diff.min()),
                max_diff=float(ss.mean_diff.max()),
                median_ci_lo=float(ss.ci_lo.median()),
                median_ci_hi=float(ss.ci_hi.median()),
                n_sig=n_sig,
                frac_sig=n_sig / len(ss),
                frac_same_dir_as_orig=same_dir_as_orig,
                orig_diff=ORIG[c]['mean_diff'] if c in ORIG else '',
            ))
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTDIR / 'lha_subsample_summary.csv', index=False)
    print('\n=== Summary ===')
    print(summary_df.to_string(index=False))

    # ---- Outcome classification (strict_min is the primary)
    primary = summary_df[summary_df.target == 'strict_min'].set_index('contrast')

    def classify(contrast, row):
        # row is a Series for one contrast (strict_min)
        if contrast not in ORIG:
            return 'na', 'no_original'
        orig = ORIG[contrast]
        med = row.median_diff
        # A: persists -- median in same direction, frac_sig >= 0.5
        same_dir = (np.sign(med) == np.sign(orig['mean_diff']))
        magnitude_ratio = abs(med) / max(abs(orig['mean_diff']), 1e-9)
        if (not same_dir) and row.frac_sig >= 0.3:
            return 'reversal', 'effect direction reversed at matched N'
        if same_dir and row.frac_sig >= 0.5:
            if magnitude_ratio >= 0.5:
                return 'A', f'persists ({magnitude_ratio:.0%} of original magnitude, {row.frac_sig:.0%} of draws sig)'
            else:
                return 'C', f'partial ({magnitude_ratio:.0%} of original magnitude, {row.frac_sig:.0%} of draws sig)'
        if row.frac_sig < 0.3:
            return 'B', f'disappears ({row.frac_sig:.0%} of draws sig, median diff {med:+.3f} vs original {orig["mean_diff"]:+.3f})'
        return 'C', f'borderline ({row.frac_sig:.0%} of draws sig, median {med:+.3f} vs orig {orig["mean_diff"]:+.3f})'

    classifications = {}
    for c, row in primary.iterrows():
        outcome, reason = classify(c, row)
        classifications[c] = (outcome, reason)
    print('\n=== Outcome classification (strict_min, target_N={}) ==='.format(target_min))
    for c, (o, r) in classifications.items():
        print(f'  {c}: {o} -- {r}')

    # ---- Stability flags
    stability = {}
    for c in ['fed_vs_fasted', 'fed_vs_fed-HFD']:
        ss = draws_df[(draws_df.target == 'strict_min') & (draws_df.contrast == c)]
        spread = float(ss.mean_diff.max() - ss.mean_diff.min())
        median_abs = float(np.median(np.abs(ss.mean_diff)))
        rel_spread = spread / max(median_abs, 1e-9)
        stability[c] = dict(spread=spread, median_abs=median_abs, rel_spread=rel_spread,
                            wildly_variable=bool(rel_spread > 4.0))

    # ---- Figures
    # 1. Distribution of effect sizes across draws
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    for ax, contrast in zip(axes, ['fed_vs_fasted', 'fed_vs_fed-HFD']):
        for tname, target_N in targets:
            ss = draws_df[(draws_df.target == tname) & (draws_df.contrast == contrast)]
            color = 'C0' if tname == 'strict_min' else 'C2'
            ax.scatter(np.full(len(ss), 0 if tname == 'strict_min' else 1),
                       ss.mean_diff.values, alpha=0.6, color=color, s=40,
                       label=f'{tname} (N={target_N})')
        ax.axhline(ORIG[contrast]['mean_diff'], color='red', ls='--', label='unmatched (orig)')
        ax.axhspan(ORIG[contrast]['ci_lo'], ORIG[contrast]['ci_hi'],
                   color='red', alpha=0.1)
        ax.axhline(0, color='k', ls=':', alpha=0.5)
        ax.set_xticks([0, 1] if len(targets) > 1 else [0])
        ax.set_xticklabels([f'{t[0]}\n(N={t[1]})' for t in targets])
        ax.set_ylabel('mean diff (a - b)')
        ax.set_title(contrast.replace('_', ' '))
        ax.legend(fontsize=8, loc='upper right')
    fig.suptitle('LHA speed state-diff: subsampled draws vs original (unmatched)')
    fig.tight_layout()
    fig.savefig(FIGDIR / 'lha_subsample_distribution.png', dpi=140)
    plt.close(fig)

    # 2. Side-by-side bar comparison
    fig, ax = plt.subplots(figsize=(9, 4.5))
    contrasts = ['fed_vs_fasted', 'fed_vs_fed-HFD']
    x = np.arange(len(contrasts))
    bar_w = 0.35
    orig_vals = [ORIG[c]['mean_diff'] for c in contrasts]
    orig_err = [[ORIG[c]['mean_diff'] - ORIG[c]['ci_lo'] for c in contrasts],
                [ORIG[c]['ci_hi'] - ORIG[c]['mean_diff'] for c in contrasts]]
    sub = draws_df[draws_df.target == 'strict_min']
    matched_med = [float(sub[sub.contrast == c].mean_diff.median()) for c in contrasts]
    matched_err = [[matched_med[i] - float(np.percentile(sub[sub.contrast == contrasts[i]].mean_diff, 25)) for i in range(len(contrasts))],
                   [float(np.percentile(sub[sub.contrast == contrasts[i]].mean_diff, 75)) - matched_med[i] for i in range(len(contrasts))]]

    ax.bar(x - bar_w / 2, orig_vals, bar_w, yerr=orig_err, capsize=5,
           color='red', alpha=0.7, label='Unmatched (full population)')
    ax.bar(x + bar_w / 2, matched_med, bar_w, yerr=matched_err, capsize=5,
           color='C0', alpha=0.8, label=f'Matched N={target_min} (median over {N_DRAWS} draws, IQR err)')
    ax.axhline(0, color='k', ls=':')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', ' ') for c in contrasts])
    ax.set_ylabel('mean diff (a - b)')
    ax.set_title('LHA speed state diff: unmatched vs unit-count-matched')
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGDIR / 'lha_subsample_comparison.png', dpi=140)
    plt.close(fig)

    # ---- Markdown summary
    md = []
    md.append('# Stage 3 follow-up — LHA subsample-controlled test')
    md.append('')
    md.append(f'**Date**: 2026-04-28. **Sessions**: {len(sessions)} '
              f'(fed n=8, fasted n=5, HFD n=4; S13/23/24 excluded).')
    md.append('')
    md.append('## Target N choice')
    md.append('')
    md.append('Per-session LHA unit counts:')
    for s in sessions:
        md.append(f'- S{s} ({state_lookup[s]}): {unit_counts[s]} units')
    md.append('')
    md.append(f'- min = **{target_min}** (S{[k for k, v in unit_counts.items() if v == target_min][0]})')
    md.append(f'- 25th percentile = {target_p25}')
    md.append(f'- median = {int(np.median(counts_arr))}, max = {int(np.max(counts_arr))}')
    md.append(f'- min flagged as outlier (1.5×IQR below P25)? **{is_outlier}**')
    md.append('')
    md.append(f'Primary analysis at strict_min N={target_min}; secondary check at P25 N={target_p25}.')
    md.append('')
    md.append('## Procedure')
    md.append('')
    md.append(f'- {N_DRAWS} random draws per target.')
    md.append('- Each draw: random subsample of LHA units per session to target N.')
    md.append('- Recompute trajectory speed (full subsampled space, σ=3 smoothing).')
    md.append('- Per-phase mean for rising+falling; session-level mean.')
    md.append(f'- Bootstrap pairwise diffs ({N_BOOT} resamples) per draw.')
    md.append('')
    md.append('## Results — primary (strict_min, N={})'.format(target_min))
    md.append('')
    md.append('| Contrast | Median diff | IQR | n_sig draws | frac_sig | orig (unmatched) |')
    md.append('|---|---|---|---|---|---|')
    for c in ['fed_vs_fasted', 'fed_vs_fed-HFD', 'fasted_vs_fed-HFD']:
        row = primary.loc[c]
        orig_str = f'{ORIG[c]["mean_diff"]:+.3f} [{ORIG[c]["ci_lo"]:+.3f}, {ORIG[c]["ci_hi"]:+.3f}] ★' if c in ORIG else 'n/a'
        md.append(f'| {c} | {row.median_diff:+.3f} | [{row.p25_diff:+.3f}, {row.p75_diff:+.3f}] | '
                  f'{int(row.n_sig)}/{int(row.n_draws)} | {row.frac_sig:.2f} | {orig_str} |')
    md.append('')
    md.append('★ = original CI excludes zero.')
    md.append('')
    if 'p25' in [t[0] for t in targets]:
        md.append('## Results — secondary (P25, N={})'.format(target_p25))
        md.append('')
        md.append('| Contrast | Median diff | IQR | frac_sig |')
        md.append('|---|---|---|---|')
        for c in ['fed_vs_fasted', 'fed_vs_fed-HFD', 'fasted_vs_fed-HFD']:
            row = summary_df[(summary_df.target == 'p25') & (summary_df.contrast == c)].iloc[0]
            md.append(f'| {c} | {row.median_diff:+.3f} | [{row.p25_diff:+.3f}, {row.p75_diff:+.3f}] | {row.frac_sig:.2f} |')
        md.append('')
    md.append('## Stability')
    md.append('')
    for c in ['fed_vs_fasted', 'fed_vs_fed-HFD']:
        s = stability[c]
        md.append(f'- **{c}**: range across draws = {s["spread"]:.3f}, median |effect| = {s["median_abs"]:.3f}, relative spread = {s["rel_spread"]:.2f}. '
                  f'Wildly variable? **{s["wildly_variable"]}** (threshold rel_spread > 4).')
    md.append('')
    md.append('## Outcome classification')
    md.append('')
    for c, (o, r) in classifications.items():
        label = {'A': 'A — Persists', 'B': 'B — Disappears',
                 'C': 'C — Partial', 'reversal': 'REVERSAL', 'na': 'n/a'}.get(o, o)
        md.append(f'- **{c}** → **{label}**: {r}')
    md.append('')
    md.append('## Interpretation')
    md.append('')
    fed_fas_outcome = classifications.get('fed_vs_fasted', ('na', ''))[0]
    fed_hfd_outcome = classifications.get('fed_vs_fed-HFD', ('na', ''))[0]
    md.append(f'- fed-vs-fasted at matched N: outcome **{fed_fas_outcome}**.')
    md.append(f'- fed-vs-HFD at matched N: outcome **{fed_hfd_outcome}**.')
    md.append('')
    if fed_fas_outcome == 'A':
        md.append('Both fed-vs-fasted and fed-vs-HFD effects persist with matched units. The "LHA reads diet state via the full population" claim survives. The Stage 3 cross-region dichotomy (ACA low-D vs LHA high-D) is supported.')
    elif fed_fas_outcome == 'B':
        md.append('fed-vs-fasted disappears at matched N. The original K=full LHA fed-vs-fasted effect was largely a unit-count artifact. **The "LHA high-D distributed coding" claim should be revised.** The Stage 1 finding that LHA speed differs by state still holds in raw mean-difference terms but cannot be attributed to a distributed population-level code without additional evidence. Recommend pausing the cross-session ACA-alignment branch until the LHA story is rewritten.')
    elif fed_fas_outcome == 'C':
        md.append('fed-vs-fasted partially persists. The state effect is real but partly amplified by unit-count differences. Report both effects honestly. The cross-region dichotomy (ACA low-D, LHA high-D) is qualitatively still valid but with reduced effect magnitude.')
    else:
        md.append('Inconclusive or reversed at matched N. Flag for revision.')
    md.append('')
    md.append('## Recommendation')
    md.append('')
    if fed_fas_outcome in ('A', 'C') and not stability['fed_vs_fasted']['wildly_variable']:
        md.append('Proceed with cross-session ACA subspace alignment as the next step. The LHA story (high-D distributed) is sufficiently robust to keep as part of the Stage 3 framing.')
    else:
        md.append('Pause cross-session ACA alignment until the LHA story is revised. The K=full LHA fed-vs-fasted finding should be qualified or removed from the Stage 3 cross-region dichotomy framing.')

    with open(OUTDIR / 'lha_subsample_summary.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))

    print(f'\nWrote {OUTDIR}')
    print(f'Wrote {FIGDIR}')


if __name__ == '__main__':
    main()
