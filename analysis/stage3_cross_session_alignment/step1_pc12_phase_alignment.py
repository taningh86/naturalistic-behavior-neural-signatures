"""
Stage 3 — Cross-session ACA subspace alignment, Step 1.

Question: are per-session PC1-2 phase-conditioned trajectory SHAPES more similar
across sessions of the SAME diet state than across sessions of DIFFERENT states?

If yes -> Stage 3 Step 1's "top-2 PCs carry diet-state info" upgrades from
"variance ranking is dominant per-session" to "a shared low-D coding axis exists
across mice."

If no -> the K=2 subspace is locally re-derived in each session; same property
("dominant covariance directions encode state") but no shared axis.

Procedure
---------
For each session (excl. 13/23/24):
  1. Load cached ACA matrix; per-session PCA; project to PC1-2.
  2. For each rising/falling phase interval, extract (PC1, PC2) trajectory.
  3. Linear-interpolation resample to fixed length L=50.
  4. Average within session × phase type -> mean shape (L x 2).

Pairwise:
  - scipy.spatial.procrustes(s1, s2) -> standardizes both and finds best
    orthogonal alignment (rotation + reflection). Returns disparity =
    sum-of-squares distance after alignment.
  - Aggregate: per phase type, then averaged across phase types.

Test:
  - Mann-Whitney U one-sided: same-state pair disparities < different-state.
  - Permutation null: shuffle session-state labels 1000x; recompute median gap.

Outputs
-------
  data/stage3_cross_session_alignment/per_session_phase_shapes.npz
  data/stage3_cross_session_alignment/pairwise_disparity.csv
  data/stage3_cross_session_alignment/same_vs_diff_state_test.csv
  data/stage3_cross_session_alignment/cross_session_summary.md
  figures/stage3_cross_session_alignment/per_session_pc12_phase_shapes.png
  figures/stage3_cross_session_alignment/disparity_by_pair_type.png
"""
import sys
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.spatial import procrustes
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent.parent
S1D = REPO / 'data' / 'dynamics_stage1'
LOC_CACHE = REPO / 'data' / 'stage3_localization' / '_cache'
OUTDIR = REPO / 'data' / 'stage3_cross_session_alignment'
FIGDIR = REPO / 'figures' / 'stage3_cross_session_alignment'
OUTDIR.mkdir(parents=True, exist_ok=True)
FIGDIR.mkdir(parents=True, exist_ok=True)

EXCLUDE = {13, 23, 24}
PHASE_TYPES = ('rising', 'falling')
K = 2
RESAMPLE_L = 50
N_PERMS = 1000
RNG = np.random.default_rng(20260428)


def load_aca(session):
    f = LOC_CACHE / f'session_{session}_aca.npy'
    if not f.exists():
        raise FileNotFoundError(f'Missing ACA cache for S{session}: {f}')
    return np.load(f)


def project_pcK(matrix, k):
    pca = PCA(n_components=k)
    return pca.fit_transform(matrix)


def resample_trajectory(traj, L):
    """traj: (n, 2). Resample to (L, 2) along time axis via linear interpolation."""
    n = len(traj)
    if n < 2:
        return None
    t_src = np.linspace(0.0, 1.0, n)
    t_tgt = np.linspace(0.0, 1.0, L)
    out = np.zeros((L, 2))
    for d in range(2):
        out[:, d] = np.interp(t_tgt, t_src, traj[:, d])
    return out


def per_session_phase_shapes(session, phases_df):
    matrix = load_aca(session)
    pcs = project_pcK(matrix, K)  # (n_bins, K)
    sub = phases_df[(phases_df.session == session) &
                    (phases_df.phase_type.isin(PHASE_TYPES))]
    out = {}
    n_events = {}
    for ptype in PHASE_TYPES:
        sub_p = sub[sub.phase_type == ptype]
        shapes = []
        for _, ph in sub_p.iterrows():
            sb = int(ph.start_bin)
            eb = int(min(ph.end_bin, len(pcs)))
            if eb - sb < 5:
                continue
            traj = pcs[sb:eb, :2]
            r = resample_trajectory(traj, RESAMPLE_L)
            if r is not None:
                shapes.append(r)
        n_events[ptype] = len(shapes)
        if shapes:
            out[ptype] = np.mean(np.stack(shapes), axis=0)
        else:
            out[ptype] = None
    return out, n_events


def main():
    summary = pd.read_csv(S1D / 'all_sessions_summary.csv')
    state_lookup = summary.groupby('session').state.first().to_dict()
    sessions = sorted(s for s in state_lookup if s not in EXCLUDE)
    print(f'Sessions: {sessions}')

    # ---- Step 1: per-session phase-conditioned shapes
    shapes_per_session = {}
    n_events_per_session = {}
    for s in sessions:
        sh, n_ev = per_session_phase_shapes(s, summary)
        shapes_per_session[s] = sh
        n_events_per_session[s] = n_ev
        ok = {p: 'OK' if sh[p] is not None else 'MISS' for p in PHASE_TYPES}
        print(f'  S{s} ({state_lookup[s]}): rising n={n_ev["rising"]} {ok["rising"]}, '
              f'falling n={n_ev["falling"]} {ok["falling"]}')

    # save shapes
    save_dict = {}
    for s in sessions:
        for p in PHASE_TYPES:
            sh = shapes_per_session[s][p]
            if sh is not None:
                save_dict[f's{s}_{p}'] = sh
    np.savez(OUTDIR / 'per_session_phase_shapes.npz', **save_dict)

    # ---- Step 2: pairwise Procrustes disparity
    pairs = list(itertools.combinations(sessions, 2))
    disparity_rows = []
    for s1, s2 in pairs:
        st1, st2 = state_lookup[s1], state_lookup[s2]
        same = (st1 == st2)
        for ptype in PHASE_TYPES:
            sh1 = shapes_per_session[s1].get(ptype)
            sh2 = shapes_per_session[s2].get(ptype)
            if sh1 is None or sh2 is None:
                continue
            # scipy.spatial.procrustes standardizes internally; returns (mtx1, mtx2, disparity)
            _, _, disp = procrustes(sh1, sh2)
            disparity_rows.append(dict(
                s_a=s1, s_b=s2, state_a=st1, state_b=st2, same_state=same,
                phase_type=ptype, disparity=float(disp),
            ))
    disp_df = pd.DataFrame(disparity_rows)
    disp_df.to_csv(OUTDIR / 'pairwise_disparity.csv', index=False)

    # avg across phase types per pair
    pair_avg = disp_df.groupby(['s_a', 's_b', 'state_a', 'state_b', 'same_state'],
                               as_index=False).disparity.mean()
    pair_avg.rename(columns={'disparity': 'mean_disparity'}, inplace=True)
    pair_avg.to_csv(OUTDIR / 'pairwise_disparity_avg.csv', index=False)

    # ---- Step 3: same-state vs different-state test
    print('\n=== Disparity per phase type ===')
    test_rows = []
    for ptype in PHASE_TYPES:
        sub = disp_df[disp_df.phase_type == ptype]
        same = sub[sub.same_state].disparity.values
        diff = sub[~sub.same_state].disparity.values
        if len(same) == 0 or len(diff) == 0:
            continue
        u, p = mannwhitneyu(same, diff, alternative='less')
        test_rows.append(dict(
            phase_type=ptype, n_same=len(same), n_diff=len(diff),
            median_same=float(np.median(same)), median_diff=float(np.median(diff)),
            U=float(u), p_one_sided=float(p),
        ))
        print(f'  {ptype}: same n={len(same)} med={np.median(same):.4f}, '
              f'diff n={len(diff)} med={np.median(diff):.4f}, '
              f'MW U={u:.0f} p_1s={p:.4f}')

    same = pair_avg[pair_avg.same_state].mean_disparity.values
    diff = pair_avg[~pair_avg.same_state].mean_disparity.values
    u_p, p_p = mannwhitneyu(same, diff, alternative='less')
    test_rows.append(dict(
        phase_type='avg', n_same=len(same), n_diff=len(diff),
        median_same=float(np.median(same)), median_diff=float(np.median(diff)),
        U=float(u_p), p_one_sided=float(p_p),
    ))
    print(f'  pooled: same n={len(same)} med={np.median(same):.4f}, '
          f'diff n={len(diff)} med={np.median(diff):.4f}, '
          f'MW U={u_p:.0f} p_1s={p_p:.4f}')

    test_df = pd.DataFrame(test_rows)
    test_df.to_csv(OUTDIR / 'same_vs_diff_state_test.csv', index=False)

    # ---- Step 4: permutation null on session-state labels
    obs_gap = float(np.median(diff)) - float(np.median(same))
    sess_arr = np.array(sessions)
    state_arr = np.array([state_lookup[s] for s in sessions])
    pair_avg_lookup = {(int(r.s_a), int(r.s_b)): float(r.mean_disparity)
                       for _, r in pair_avg.iterrows()}

    null_gaps = np.zeros(N_PERMS)
    for i in range(N_PERMS):
        perm_states = RNG.permutation(state_arr)
        perm_lookup = dict(zip(sess_arr, perm_states))
        same_p, diff_p = [], []
        for s1, s2 in pairs:
            d = pair_avg_lookup[(s1, s2)]
            if perm_lookup[s1] == perm_lookup[s2]:
                same_p.append(d)
            else:
                diff_p.append(d)
        if same_p and diff_p:
            null_gaps[i] = np.median(diff_p) - np.median(same_p)
        else:
            null_gaps[i] = np.nan
    null_gaps = null_gaps[~np.isnan(null_gaps)]
    perm_p = float(np.mean(null_gaps >= obs_gap))
    print(f'\nObserved median(diff)-median(same) = {obs_gap:.5f}')
    print(f'Permutation null mean = {np.mean(null_gaps):.5f}, '
          f'95th pct = {np.percentile(null_gaps, 95):.5f}')
    print(f'Permutation p (one-sided, more separation than null) = {perm_p:.4f}')

    # ---- Per-mouse-state-aware test (HFD has only 4, sparse pairs)
    state_pair_breakdown = []
    for sa, sb in itertools.combinations(['fed', 'fasted', 'fed-HFD'], 2):
        sub = pair_avg[((pair_avg.state_a == sa) & (pair_avg.state_b == sb)) |
                       ((pair_avg.state_a == sb) & (pair_avg.state_b == sa))]
        if len(sub) > 0:
            state_pair_breakdown.append(dict(
                contrast=f'{sa}_vs_{sb}', n_pairs=len(sub),
                median_disparity=float(np.median(sub.mean_disparity)),
            ))
    for st in ['fed', 'fasted', 'fed-HFD']:
        sub = pair_avg[(pair_avg.state_a == st) & (pair_avg.state_b == st)]
        if len(sub) > 0:
            state_pair_breakdown.append(dict(
                contrast=f'{st}_vs_{st}', n_pairs=len(sub),
                median_disparity=float(np.median(sub.mean_disparity)),
            ))
    breakdown_df = pd.DataFrame(state_pair_breakdown)
    breakdown_df.to_csv(OUTDIR / 'state_pair_breakdown.csv', index=False)
    print('\n=== State-pair breakdown (avg over phase types) ===')
    print(breakdown_df.to_string(index=False))

    # ---- Figures
    state_color = {'fed': 'C0', 'fasted': 'C1', 'fed-HFD': 'C3'}

    # 1. Per-session phase shapes
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, ptype in zip(axes, PHASE_TYPES):
        # standardize each shape (center + Frob norm) for plotting
        for s in sessions:
            sh = shapes_per_session[s].get(ptype)
            if sh is None:
                continue
            c = sh - sh.mean(axis=0)
            n = np.linalg.norm(c)
            if n > 1e-9:
                c = c / n
            color = state_color.get(state_lookup[s], 'gray')
            ax.plot(c[:, 0], c[:, 1], color=color, alpha=0.6, lw=1.2)
            ax.scatter(c[0, 0], c[0, 1], color=color, s=25, marker='o', zorder=3)
            ax.scatter(c[-1, 0], c[-1, 1], color=color, s=35, marker='x', zorder=3)
        ax.set_title(f'{ptype} (start: o, end: x)')
        ax.set_xlabel('PC1 (centered, unit-Frob)')
        ax.set_ylabel('PC2 (centered, unit-Frob)')
        ax.set_aspect('equal', 'box')
    handles = [plt.Line2D([0], [0], color=c, label=s, lw=2)
               for s, c in state_color.items()]
    axes[0].legend(handles=handles, loc='best', fontsize=9)
    fig.suptitle('Per-session phase-conditioned PC1-2 trajectory shapes (standardized)')
    fig.tight_layout()
    fig.savefig(FIGDIR / 'per_session_pc12_phase_shapes.png', dpi=140)
    plt.close(fig)

    # 2. Disparity boxplot by pair type
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    panels = [
        (disp_df[disp_df.phase_type == 'rising'], 'rising'),
        (disp_df[disp_df.phase_type == 'falling'], 'falling'),
        (pair_avg.assign(disparity=pair_avg.mean_disparity), 'avg over phase types'),
    ]
    for ax, (sub, title) in zip(axes, panels):
        same = sub[sub.same_state].disparity.values
        diff = sub[~sub.same_state].disparity.values
        bp = ax.boxplot([same, diff],
                        tick_labels=[f'same\n(n={len(same)})', f'different\n(n={len(diff)})'],
                        showmeans=True)
        # overlay scatter
        ax.scatter(np.full(len(same), 1) + RNG.uniform(-0.08, 0.08, len(same)),
                   same, alpha=0.5, color='C0', s=12)
        ax.scatter(np.full(len(diff), 2) + RNG.uniform(-0.08, 0.08, len(diff)),
                   diff, alpha=0.5, color='C3', s=12)
        ax.set_ylabel('Procrustes disparity')
        ax.set_title(title)
    fig.suptitle('Pairwise Procrustes disparity: same-state vs different-state pairs')
    fig.tight_layout()
    fig.savefig(FIGDIR / 'disparity_by_pair_type.png', dpi=140)
    plt.close(fig)

    # 3. Permutation null
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(null_gaps, bins=40, color='gray', alpha=0.7, label='null (state-shuffled)')
    ax.axvline(obs_gap, color='red', lw=2, label=f'observed = {obs_gap:.4f}')
    ax.set_xlabel('median(diff) − median(same) disparity')
    ax.set_ylabel('count')
    ax.set_title(f'Permutation null (1000 shuffles), p_1s = {perm_p:.4f}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGDIR / 'permutation_null.png', dpi=140)
    plt.close(fig)

    # ---- Markdown summary
    md = []
    md.append('# Stage 3 — Cross-session ACA subspace alignment (Step 1)')
    md.append('')
    md.append(f'**Date**: 2026-04-28. **Sessions**: {len(sessions)} '
              '(fed n=8, fasted n=5, HFD n=4; S13/23/24 excluded).')
    md.append('')
    md.append('## Question')
    md.append('')
    md.append('Stage 3 Step 1 found that ACA top-2 PCs (per-session PCA) carry the diet-state '
              'curvature signature in each session. Are those 2D subspaces *aligned* across '
              'sessions, or 17 idiosyncratic subspaces that each happen to contain a diet effect?')
    md.append('')
    md.append('## Procedure')
    md.append('')
    md.append('- Per-session PCA on z-scored 50 ms ACA matrix; project to top-2 PCs.')
    md.append('- For each rising/falling phase interval, extract (PC1, PC2) trajectory.')
    md.append(f'- Linear-interpolation resample to fixed length L={RESAMPLE_L}.')
    md.append('- Average resampled trajectories within session × phase type → mean shape (L×2).')
    md.append('- Pairwise `scipy.spatial.procrustes` (auto-standardizes; rotation+reflection alignment).')
    md.append('- Disparity = sum-of-squared-distances after best alignment.')
    md.append('- One-sided Mann-Whitney U: same-state pair disparity < different-state pair.')
    md.append(f'- Permutation null: shuffle session-state labels {N_PERMS}×; recompute median gap.')
    md.append('')
    md.append('## Per-session phase counts')
    md.append('')
    for s in sessions:
        n_ev = n_events_per_session[s]
        md.append(f'- S{s} ({state_lookup[s]}): rising n={n_ev["rising"]}, falling n={n_ev["falling"]}')
    md.append('')
    md.append('## Results')
    md.append('')
    md.append('| Phase | n same-pair | n diff-pair | med disp (same) | med disp (diff) | MW U | p (1-sided) |')
    md.append('|---|---|---|---|---|---|---|')
    for r in test_rows:
        md.append(f'| {r["phase_type"]} | {r["n_same"]} | {r["n_diff"]} | '
                  f'{r["median_same"]:.4f} | {r["median_diff"]:.4f} | '
                  f'{r["U"]:.0f} | {r["p_one_sided"]:.4f} |')
    md.append('')
    md.append(f'**Permutation null** ({N_PERMS} shuffles of session-state labels): '
              f'observed median(diff) − median(same) = **{obs_gap:.5f}**; '
              f'permutation null mean = {np.mean(null_gaps):.5f}; '
              f'95th pct = {np.percentile(null_gaps, 95):.5f}; '
              f'permutation p (one-sided) = **{perm_p:.4f}**.')
    md.append('')
    md.append('## State-pair breakdown')
    md.append('')
    md.append('| Contrast | n pairs | median disparity (avg over phase types) |')
    md.append('|---|---|---|')
    for r in state_pair_breakdown:
        contrast_disp = r["contrast"].replace('_vs_', ' vs ')
        md.append(f'| {contrast_disp} | {r["n_pairs"]} | {r["median_disparity"]:.4f} |')
    md.append('')
    md.append('## Interpretation')
    md.append('')
    pooled_p = test_rows[-1]['p_one_sided']
    if perm_p < 0.05 and pooled_p < 0.05:
        md.append('**Same-state session pairs have measurably smaller PC1-2 phase-conditioned '
                  'trajectory disparity than different-state pairs.** This is evidence for a '
                  'CONSERVED low-D ACA coding axis across mice. The Stage 3 Step 1 finding '
                  '(top-2 PCs carry diet-state info per session) upgrades from a per-session '
                  'property to a shared geometric structure.')
    elif perm_p < 0.10 or pooled_p < 0.10:
        md.append('**Trend toward greater similarity for same-state pairs, but not robustly '
                  'significant.** Cannot definitively claim a conserved cross-session axis. '
                  'Stage 3 Step 1 result holds within session but cross-session generalization '
                  'is weak. Possible reasons: per-session PCA picks up idiosyncratic noise '
                  'directions; HFD n=4 dilutes power; alignment via 2D Procrustes is permissive.')
    else:
        md.append('**No significant cross-session alignment of PC1-2 phase-conditioned '
                  'trajectory shapes by diet state.** Stage 3 Step 1 holds within session but '
                  'does not generalize to a shared geometric axis. The K=2 subspace appears '
                  'to be locally re-derived in each session — same property '
                  '("dominant covariance directions encode state") but different axes.')
    md.append('')
    md.append('## Caveats')
    md.append('')
    md.append('- Per-session PCA: PC sign and axis-swap ambiguity are handled by Procrustes '
              '(orthogonal alignment includes reflection).')
    md.append('- 2D Procrustes is permissive; the test relies on differential disparity, not absolute alignment.')
    md.append('- Phase intervals vary in duration; resampling to fixed L collapses time-rate differences.')
    md.append('- HFD n=4 → only 6 same-state HFD pairs.')
    md.append('- Cross-session alignment is in latent score space, not unit space '
              '(no UnitMatch for ACA).')
    md.append('')
    md.append('## Files')
    md.append('')
    md.append('- `analysis/stage3_cross_session_alignment/step1_pc12_phase_alignment.py`')
    md.append('- `data/stage3_cross_session_alignment/per_session_phase_shapes.npz`')
    md.append('- `data/stage3_cross_session_alignment/pairwise_disparity.csv`, `pairwise_disparity_avg.csv`')
    md.append('- `data/stage3_cross_session_alignment/same_vs_diff_state_test.csv`, `state_pair_breakdown.csv`')
    md.append('- `figures/stage3_cross_session_alignment/per_session_pc12_phase_shapes.png`')
    md.append('- `figures/stage3_cross_session_alignment/disparity_by_pair_type.png`')
    md.append('- `figures/stage3_cross_session_alignment/permutation_null.png`')

    with open(OUTDIR / 'cross_session_summary.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))

    print(f'\nWrote {OUTDIR}')
    print(f'Wrote {FIGDIR}')


if __name__ == '__main__':
    main()
