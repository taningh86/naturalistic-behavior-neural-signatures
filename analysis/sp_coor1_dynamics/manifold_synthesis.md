# Single-probe Mouse01-Coor1 — Manifold Geometry (Layers 1a + 1b)

**Date**: 2026-04-28. **Sessions**: 8 (4 fed, 4 fasted). **Bins**: 50 ms,
σ=1 Gaussian smooth, per-unit z-score. **Cache**: `data/sp_coor1_dynamics/_cache/`.

## Layer 1a — Intrinsic Dimensionality

Per-region, per-session: PR (participation ratio), Two-NN (Facco), CorrDim
(Grassberger-Procaccia), Isomap (k=15 primary, sensitivity over k∈{10,15,20,30}).
Block bootstrap (block=200 bins, ~10 s) for CIs: N_BOOT=100 (PR/TwoNN), 50 (CorrDim), 20 (Isomap).

### State means (across 4 sessions per state)

| Region | State | n_units | PR    | TwoNN | CorrDim | Isomap (k=15) |
|--------|-------|---------|-------|-------|---------|----------------|
| LHA    | fed   | 28.25   | 24.25 | 7.97  | 3.64    | 77.75          |
| LHA    | fasted| 16.50   | 14.89 | 6.23  | 2.89    | 47.75          |
| RSP    | fed   | 77.25   | 47.60 | 6.27  | 4.73    | 68.50          |
| RSP    | fasted| 54.25   | 36.37 | 6.04  | 4.09    | 57.25          |

### State contrast bootstrap (5000 resamples, fed n=4 vs fasted n=4)

| Region | Metric  | Δ fed-fasted | 95% CI            | excl 0 | %    |
|--------|---------|--------------|-------------------|--------|------|
| LHA    | PR      | +9.36        | [+5.54, +13.42]   | yes    | +63% |
| LHA    | TwoNN   | +1.74        | [+0.80, +2.82]    | yes    | +28% |
| LHA    | CorrDim | +0.74        | [+0.33, +1.13]    | yes    | +26% |
| LHA    | Isomap  | +30.0        | [-9.0, +65.5]     | no     | +63% |
| LHA    | n_units | +11.75       | [+8.0, +16.0]     | yes    | +71% |
| RSP    | PR      | +11.22       | [+5.34, +16.89]   | yes    | +31% |
| RSP    | TwoNN   | +0.24        | [-0.03, +0.50]    | no     | +4%  |
| RSP    | CorrDim | +0.64        | [+0.36, +0.87]    | yes    | +16% |
| RSP    | Isomap  | +11.25       | [-3.01, +23.25]   | no     | +20% |
| RSP    | n_units | +23.0        | [+11.5, +35.25]   | yes    | +42% |

### Read

- **PR is N-bound**, not "intrinsic". LHA fed PR is +63% while unit count is +71%; RSP fed PR is +31% while unit count is +42%. PR scales linearly with N when units are uncorrelated, so the bootstrap on PR is essentially restating the unit-count contrast — **not interpretable as a state effect on dimensionality**.
- **TwoNN / CorrDim / Isomap** are *less* N-dependent but still scale somewhat with sample size. Both regions show fed>fasted in CorrDim (LHA +26%, RSP +16%, both CIs exclude 0); LHA additionally shows fed>fasted TwoNN (+28%). RSP TwoNN is essentially flat (+4%, CI crosses 0).
- **Cross-region**: ranked by TwoNN (most N-robust), LHA (~7) ≈ RSP (~6) — both regions live on a low-D manifold (~5–8 dimensions intrinsically) embedded in 12-93 unit recordings. CorrDim agrees: LHA ~3, RSP ~4–5.
- **Critical caveat**: every dim difference is confounded with unit count. Until we run an N-matched subsample control on TwoNN/CorrDim, "fasted has lower intrinsic dim than fed" is not a defensible state claim.

## Layer 1b — Persistent Homology

Per region: PCA → K_PCS components (LHA K=5, RSP K=10), maxmin landmark
subsample (N=600), Vietoris-Rips up to H2 via ripser. Null: 15 circular-shift
shuffles per neuron (PCA axes from data are reused for null projection).

### Per-session significance summary

LHA (8 sessions, K=5):
- **H1 max persistence**: 2/8 sessions p<0.05 (S2, S8); 4/8 marginal (p≈0.07)
- **H1 total persistence**: 7/8 sessions p<0.001 (only S6 fails)
- **H2 max persistence**: 7/8 sessions p<0.001 (only S6 fails)

RSP (8 sessions, K=10):
- **H1 max persistence**: 5/8 sessions p<0.05 (all 4 fed + S5)
- **H1 total persistence**: 7/8 sessions p<0.001 (only S7 fails)
- **H2 max persistence**: 6/8 sessions p<0.05

### State contrast (fed vs fasted, 5000 bootstraps)

| Region | Metric          | Δ      | 95% CI            | excl 0 |
|--------|-----------------|--------|-------------------|--------|
| LHA    | H0_total_pers   | +91    | [+27, +149]       | yes    |
| LHA    | H1_max_pers     | -0.006 | [-0.27, +0.27]    | no     |
| LHA    | H1_total_pers   | +16.8  | [-0.6, +31.4]     | no     |
| LHA    | H2_max_pers     | +0.04  | [-0.02, +0.11]    | no     |
| LHA    | H2_total_pers   | +9.4   | [-2.6, +20.5]     | no     |
| RSP    | H1_max_pers     | **+0.49**  | **[+0.05, +0.95]**    | **yes**    |
| RSP    | H1_total_pers   | **+32.8**  | **[+13.9, +52.2]**    | **yes**    |
| RSP    | H2_total_pers   | +18.2  | [+0.03, +36.4]    | barely |

### Read

- **Both regions carry non-trivial topology in most sessions**: H1 (loops) and H2 (voids) total persistence sit above the circular-shift null in 7–8/8 sessions. The neural manifold is not a simple convex blob even after PCA reduction.
- **H2 is the strongest signal in LHA** (max persistence 7/8 sig, total persistence trending). For LHA the few-loop count vs many small loops aligns with low TwoNN ~7.
- **State effect in RSP H1**: fed mice have 33% larger max H1 persistence and 11% larger total H1 persistence than fasted, both CIs excluding 0. **Caveat**: fed has ~42% more RSP units, and PCA→K=10 with more units may simply build a fuller manifold. This may be unit-count, not state.
- **No state effect on LHA topology**, with the exception of H0 total persistence (modest +8%, mostly reflecting unit-count differences in pairwise distances).
- **S6 LHA is a topology outlier**: 12 units, H1 below null. Insufficient population for stable Rips.

## Surviving claims

1. **Single-probe LHA & RSP carry non-trivial Vietoris-Rips topology** above shuffle null (H1 and especially H2) in 7–8/8 sessions, replicating the dual-probe ACA & LHA finding.
2. **Both regions are intrinsically low-D** (TwoNN ~6–8, CorrDim ~3–5) despite living in 12–93 unit ambient spaces. Direction matches dual-probe (ACA ~8–10, LHA ~4–7).
3. **RSP H1 persistence is larger in fed than fasted** (max +33%, CI excludes 0) — *but* fed has 42% more RSP units, so this is a candidate state effect, not a confirmed one.

## Withdrawn / inconclusive

- All PR-based state contrasts (PR scales ~linearly with N).
- LHA Isomap and RSP TwoNN/Isomap state contrasts (CIs include 0).
- RSP H1 state effect requires N-matched control before we can call it a state signal vs unit-count artifact.

## Caveats

- Single mouse, 4 vs 4 — within-mouse session variance only. No cross-animal generalization.
- All metrics carry varying degrees of N confound; fed sessions have ~1.4–1.7× more units than fasted in both regions.
- Layer 1b uses K_PCS={LHA:5, RSP:10}; chosen to mirror dual-probe defaults (LHA=5, ACA=10). Alternative K choices for RSP (e.g. 6–8) would be defensible given TwoNN ~6.
- Bootstrap CIs at n=4 vs n=4 are tight because session variance is small, not because sample size is large.
- TwoNN block-bootstrap CIs collapse to 0–0.1 because resampled blocks contain duplicate rows; the point estimates are still informative but the CIs are not.

## Files

- `analysis/sp_coor1_dynamics/sp_manifold_layer1a.py` (PR/TwoNN/CorrDim/Isomap)
- `analysis/sp_coor1_dynamics/sp_manifold_layer1b.py` (Rips persistent homology)
- `analysis/sp_coor1_dynamics/sp_manifold_state_contrast.py` (Layer 1a state contrast)
- `analysis/sp_coor1_dynamics/sp_manifold_layer1b_contrast.py` (Layer 1b state contrast)
- `data/sp_coor1_dynamics/manifold_layer1a.csv`, `manifold_layer1a.json`, `manifold_layer1a_state_contrast.csv`
- `data/sp_coor1_dynamics/manifold_layer1b.csv`, `manifold_layer1b_state_contrast.csv`
- `data/sp_coor1_dynamics/S{1..8}_{LHA,RSP}_layer1b.json`
- `figures/sp_coor1_dynamics/manifold_layer1a_dimensionality.png`
- `figures/sp_coor1_dynamics/S{1..8}_{LHA,RSP}_persistent_homology.png`

## Next options

(a) **N-matched subsample on TwoNN/CorrDim/H1** — does the apparent fed>fasted dim/topology survive when fed sessions are subsampled to fasted unit counts? Mirrors Step 2 speed control.
(b) **Layer 1c: CCA** — is there a coupled subspace between LHA and RSP, and how does it shift with state?
(c) **Layer 2: behavioral mapping** — decode entropy phase / compartment from neural manifold; does fasting reorganize spatial→feeding axis as in dual-probe?
