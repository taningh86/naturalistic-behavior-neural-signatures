# Dual-probe Layer 1b — Fed vs Fasted vs HFD Persistent Homology

**Date**: 2026-04-29. **Sessions**: S3, S4 (fed) + S11–S16 (fasted) + S19–S24 (HFD) = 14 sessions × 2 regions (ACA, LHA). **Parameters**: SUBSAMPLE_N=600 maxmin landmarks, N_SHUFFLES=15, MAX_DIM=2, K_PCS={ACA:10, LHA:5} — matched to single-probe pipeline.

## Per-state means

### ACA
| State  | n  | n_units | H0_max  | H0_total | H1_max | H1_total | H2_max | H2_total |
|--------|----|---------|---------|----------|--------|----------|--------|----------|
| fed    | 2  | 217.5   | 7.34    | 3413.1   | 1.78   | 479.4    | 1.21   | 264.1    |
| fasted | 6  | 187.2   | 7.41    | 3140.2   | 1.69   | 417.1    | 1.06   | 237.3    |
| HFD    | 6  | 207.7   | 7.32    | 3009.2   | 1.79   | 373.4    | 1.00   | 172.5    |

### LHA
| State  | n  | n_units | H0_max  | H0_total | H1_max | H1_total | H2_max | H2_total |
|--------|----|---------|---------|----------|--------|----------|--------|----------|
| fed    | 2  | 111.0   | 5.34    | 1771.4   | 1.65   | 243.9    | 0.75   | 81.8     |
| fasted | 6  | 84.3    | 5.10    | 1491.5   | 1.38   | 201.1    | 0.69   | 68.5     |
| HFD    | 6  | 87.5    | 5.48    | 1454.7   | 1.63   | 185.0    | 0.60   | 53.1     |

## Pairwise state contrast (5000-resample bootstrap of group-mean diff)

### ACA (excludes-zero only)
| Contrast            | Metric         | mean_a  | mean_b  | diff   | 95% CI            | %    |
|---------------------|----------------|---------|---------|--------|-------------------|------|
| fed − fasted        | n_units        | 217.5   | 187.2   | +30.3  | [+0.5, +52.5]     | +16% |
| fed − fasted        | H0_total_pers  | 3413    | 3140    | +273   | [+153, +408]      | +9%  |
| fed − fasted        | H1_total_pers  | 479     | 417     | +62    | [+29, +102]       | +15% |
| fed − fasted        | H2_max_pers    | 1.208   | 1.062   | +0.15  | [+0.01, +0.30]    | +14% |
| fed − HFD           | H0_total_pers  | 3413    | 3009    | +404   | [+298, +511]      | +13% |
| fed − HFD           | H1_total_pers  | 479     | 373     | +106   | [+77, +136]       | +28% |
| fed − HFD           | H2_max_pers    | 1.208   | 1.004   | +0.20  | [+0.06, +0.33]    | +20% |
| fed − HFD           | H2_total_pers  | 264     | 173     | +92    | [+62, +122]       | +53% |
| **fasted − HFD**    | **H2_total_pers** | **237** | **173** | **+65** | **[+18, +108]** | **+38%** |

### LHA (excludes-zero only)
| Contrast            | Metric         | mean_a  | mean_b  | diff   | 95% CI            | %     |
|---------------------|----------------|---------|---------|--------|-------------------|-------|
| fed − fasted        | n_units        | 111.0   | 84.3    | +26.7  | [+9.2, +45.7]     | +32%  |
| fed − fasted        | var_expl_pct   | 13.9    | 15.7    | −1.78  | [−3.6, −0.04]     | −11%  |
| fed − fasted        | H0_total_pers  | 1771    | 1491    | +280   | [+213, +343]      | +19%  |
| fed − fasted        | H1_total_pers  | 244     | 201     | +43    | [+25, +60]        | +21%  |
| fed − HFD           | n_units        | 111.0   | 87.5    | +23.5  | [+10.7, +34.7]    | +27%  |
| fed − HFD           | var_expl_pct   | 13.9    | 15.6    | −1.73  | [−3.0, −0.7]      | −11%  |
| fed − HFD           | H0_total_pers  | 1771    | 1455    | +317   | [+223, +434]      | +22%  |
| fed − HFD           | H1_total_pers  | 244     | 185     | +59    | [+39, +79]        | +32%  |
| fed − HFD           | H2_max_pers    | 0.75    | 0.60    | +0.15  | [+0.05, +0.25]    | +25%  |
| fed − HFD           | H2_total_pers  | 81.8    | 53.1    | +28.6  | [+15.0, +42.5]    | +54%  |
| **fasted − HFD**    | **H2_max_pers**    | **0.69** | **0.60** | **+0.09** | **[+0.01, +0.18]** | **+16%** |
| **fasted − HFD**    | **H2_total_pers**  | **69**   | **53**   | **+15**   | **[+0.2, +32]**    | **+29%** |

(All other contrasts include 0.)

## Read

### Headline: H1 (loops) does NOT differ across states
**No state effect on H1 max persistence** in either ACA or LHA. ACA H1 max is essentially identical across fed (1.78), fasted (1.69), HFD (1.79). LHA H1 max: fed 1.65, fasted 1.38, HFD 1.63 — fed≈HFD, fasted nudges down but CIs cross 0. The single-probe **RSP H1 max +33% fed > fasted** finding (CI [+0.05, +0.95]) does not replicate in dual-probe ACA. Possible reasons: (i) different cortex (RSP ≠ ACA); (ii) single-probe RSP has +42% more units in fed which dominates here that the +16% ACA n_units gap does not; (iii) different mice; (iv) fed n=2 in dual-probe is underpowered.

### HFD selectively reduces H2 (voids) in both regions
This is the cleanest unconfounded finding because HFD (n=6) and fasted (n=6) have nearly identical unit counts (ACA 208 vs 187; LHA 87.5 vs 84.3):

- **ACA H2 total**: fasted 237 vs HFD 173, **+38% (CI [+18, +108])**
- **LHA H2 max**:   fasted 0.69 vs HFD 0.60, **+16% (CI [+0.01, +0.18])**
- **LHA H2 total**: fasted 69 vs HFD 53, **+29% (CI [+0.2, +32])**

These contrasts cannot be explained by unit count — HFD has slightly **more** ACA units than fasted, yet substantially fewer voids. The fed > HFD H2 effects (much larger: +53% ACA, +54% LHA H2 total) align in direction but are partly N-confounded since fed has more units.

**Interpretation**: HFD compresses the manifold along its higher-order topological dimensions. The ambient point cloud still organizes into loops (H1 unchanged) but stops enclosing higher-dimensional voids (H2 collapses). Consistent with HFD-induced flattening of neural state space; mirrors no other manifold result so far.

### Fed vs fasted is largely N-confounded
H0 differences are entirely unit-count: more units → more H0 features. H1 total fed > fasted (+15% ACA, +21% LHA) tracks the +16/+32% n_units gap. With fed only n=2, these contrasts have weak power and unreliable CIs.

### Fed-only n=2 caveat
S3 and S4 were re-run at matched 600/15 parameters as a fed reference. Any contrast involving fed is on n=2 vs n=6, so CI tightness is misleading — bootstrap resamples of n=2 cluster narrowly around the two observed values. The fed=2 contrasts here should be treated as exploratory; the **fasted vs HFD** contrasts (n=6 vs n=6) are the most defensible.

## Surviving claims

1. **Both ACA and LHA carry significant Vietoris-Rips topology** above circular-shift null in 14/14 sessions for H1 total persistence (and H0 always). H1 max significant in 14/14 ACA sessions; LHA H1 max marginal in several but H1 total robust.
2. **HFD reduces H2 (voids) in both ACA and LHA** vs fasted at matched unit count. CI excludes 0 for ACA H2 total, LHA H2 max, LHA H2 total.
3. **No fed/fasted/HFD effect on H1 max persistence** in either region.

## Withdrawn / inconclusive

- **Single-probe RSP H1 max fed > fasted does NOT generalize to dual-probe ACA** under matched parameters. Single-probe finding likely reflects either RSP-specific effect, RSP unit-count confound, or cross-mouse difference; not an ACA effect.
- All fed-involving contrasts are weak (n=2 fed) — needs more dual-probe fed sessions or single-probe fed addition.

## Caveats

- Fed n=2 vs fasted n=6 vs HFD n=6 — unbalanced.
- Single mouse (Mouse01) for dual-probe; within-mouse variance only.
- HFD recordings are fed mice on high-fat diet; metabolic/physiological state differs from chronic-fast vs ad-lib chow comparison.
- N_SHUFFLES=15, SUBSAMPLE_N=600 reduced from defaults (1000/20) to keep batch tractable; null CIs are noisier than `dp_manifold_layer1b.py` defaults but matched to single-probe pipeline.
- S13, S23, S24 had only 12k bins (vs 36k typical) — shorter recordings; ran without issue but null might be slightly less stable.

## Files

- `dp_manifold_layer1b_batch.py` — batch driver
- `dp_manifold_layer1b_state_contrast.py` — pairwise bootstrap
- `data/manifold/S{N}_{ACA,LHA}_layer1b_batch.json` (28 files)
- `data/manifold/manifold_layer1b_batch.csv`
- `data/manifold/manifold_layer1b_batch_state_summary.csv`
- `data/manifold/manifold_layer1b_batch_state_contrast.csv`
- `_dp_layer1b_batch_log.txt` (run log)

## Next options

(a) **N-matched subsample on H1/H2** — same fix applied elsewhere; subsample fed/fasted/HFD to common unit count and re-run, especially for H2 fed-vs-HFD claim.
(b) **Add dual-probe fed sessions S5–S10** at matched 600/15 to expand fed n.
(c) **HFD H2 mechanism** — what about HFD organization compresses voids while preserving loops? Probe with phase-locked decomposition or Layer 1c CCA between regions.
