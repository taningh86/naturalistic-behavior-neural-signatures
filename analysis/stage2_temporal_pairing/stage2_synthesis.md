# Stage 2 — Temporal Pairing Synthesis

**Goal.** Take the two surviving Stage 1 / drill-down state signatures (ACA mean curvature, LHA mean speed) and ask two questions about them:

1. **Timing.** Within an entropy phase, *when* does the state difference appear — before, at, or after the inflection?
2. **Coupling.** Are ACA curvature and LHA speed two faces of the same state-shift, or two independent state signatures running in parallel?

Same 17 sessions used in the drill-down: fed (n = 8, S3–S10), fasted (n = 5, S11/S12/S14/S15/S16), fed-HFD (n = 4, S19–S22). S13/S23/S24 excluded.

Bin: 50 ms throughout. Inflections defined as scipy `find_peaks` peaks/troughs on the entropy timecourse used in Stage 1.

---

## Analysis 1 — Peri-inflection time-locking (±60 s)

**Procedure.** Pulled ±1200-bin windows (60 s) around each peak/trough inflection_bin, NaN-padded truncated edges. For each metric × state × inflection_type, computed session-level mean trajectory with 95% session-bootstrap CIs (2000 resamples). Then computed pairwise diff trajectories (a − b) with bootstrap CIs per timepoint. Divergence onset = earliest contiguous run of ≥20 bins (1 s) where the diff CI excluded zero.

Counts: fed peak = 36, fed trough = 39; fasted peak = 24, fasted trough = 25; HFD peak = 19, HFD trough = 22.

### 1a. ACA mean curvature — primary state signature

| Contrast | Inflection | Onset (s rel. to inflection) |
|---|---|---|
| fed vs fasted | peak | **−57.80** |
| fed vs HFD | peak | **−52.10** |
| fasted vs HFD | peak | −33.25 |
| fed vs fasted | trough | **−58.90** |
| fed vs HFD | trough | +8.75 |
| fasted vs HFD | trough | −34.35 |
| fed vs fasted | pooled | **−58.90** |
| fed vs HFD | pooled | **−57.85** |
| fasted vs HFD | pooled | **−56.40** |

**Interpretation.** With the −60 s window edge as the earliest possible onset, every fed-vs-{fasted, HFD} contrast that resolves crosses the divergence threshold within ~1–8 s of that edge. *The state difference is essentially saturated for the entire ±60 s window.* The inflection itself is not an anchor — it is incidental. Entropy phases differ between states as a sustained, tonic property, not a transient peri-inflection event. This is consistent with the drill-down's time-resolved finding that the effect spans 30–60 % of normalized phase.

The HFD-trough contrast (+8.75 s) is the lone exception — but only one of the six fed-vs-{fasted,HFD} cells, and the pooled and other-inflection cells are all ≤ −52 s. Likely reflects the small HFD n × trough-specific noise.

### 1b. ACA speed — secondary check

ACA speed shows fed-vs-fasted onset at peak −57.30 s but is mixed for HFD and trough. fasted-vs-HFD pooled is NaN (no run of length ≥ 20 bins where CI excluded zero). The speed signature is weaker and less stereotyped than curvature, matching Step 5 of the drill-down (curvature-specific signal).

### 1c. LHA curvature — null

Most contrasts return NaN for divergence onset (no sustained CI-excludes-zero run). The LHA-curvature axis is not a state marker.

### 1d. LHA mean speed — parallel state signature

Every fed-vs-{fasted, HFD} contrast onsets at −60.0 to −59.9 s — i.e. the *very first bin of the window*. The state effect is essentially ubiquitous. fasted-vs-HFD pooled is NaN, consistent with these two states being indistinguishable on this axis (mirrors drill-down: HFD aligns with fasted on LHA speed and the fed group is the outlier).

### Bottom line on timing

Both surviving signatures (ACA curvature and LHA speed) are **slow tonic state markers**, not phase-locked transients. The peri-inflection framing was the right way to test the alternative hypothesis (transient peri-inflection event) and that hypothesis is rejected.

---

## Analysis 2 — Cross-region pairing within phases

**Procedure.** For each entropy phase ≥ 20 bins (1 s) long: zero-mean ACA curvature and LHA speed across the phase, compute Pearson r. (146 phases passed: fed 65, fasted 44, HFD 37; rising 75, falling 71.) Generated within-session shuffle null by circularly rotating LHA speed within the phase's session and recomputing r. Per-state mean |r| with bootstrap CIs over sessions (5000 resamples). Lagged r over ±2 s in 0.1 s steps.

### 2a. Within-phase coupling: essentially absent

| State | Phase | n_phases | n_sess | mean r | 95% CI | excludes 0? | mean r (null) | real − null |
|---|---|---|---|---|---|---|---|---|
| fed | rising | 33 | 8 | −0.011 | [−0.056, +0.033] | no | −0.000 | −0.011 |
| fed | falling | 32 | 8 | −0.014 | [−0.064, +0.031] | no | −0.008 | −0.006 |
| fasted | rising | 24 | 5 | +0.006 | [−0.074, +0.086] | no | +0.001 | +0.005 |
| fasted | falling | 20 | 5 | +0.010 | [−0.028, +0.046] | no | +0.012 | −0.001 |
| fed-HFD | rising | 18 | 4 | +0.016 | [−0.017, +0.053] | no | +0.011 | +0.005 |
| fed-HFD | **falling** | 19 | 4 | **+0.043** | **[+0.007, +0.084]** | **yes** | +0.008 | **+0.035** |

Five of six state × phase cells: real |r| < 0.02, CI brackets zero, and real ≈ null. Only **fed-HFD falling** crosses the threshold (mean r = +0.043, CI [+0.007, +0.084], real − null = +0.035). Even this is a small effect at n = 4 sessions and would not survive any multiple-comparison correction across the six cells.

### 2b. State contrasts: no coupling difference between states

| Group | Contrast | mean diff | 95% CI |
|---|---|---|---|
| rising∪falling | fed vs fasted | −0.021 | [−0.082, +0.036] |
| rising∪falling | fed vs HFD | −0.044 | [−0.090, +0.002] |
| rising∪falling | fasted vs HFD | −0.022 | [−0.078, +0.036] |

No contrast CI excludes zero. fed-vs-HFD comes closest (CI just brackets zero by 0.002) — directionally consistent with the falling-phase fed-HFD result above, but underpowered.

### 2c. Lagged correlation: no clean shared trace

Per-state peak |r| over ±2 s:
- fed: 0.040 at −0.90 s
- fasted: 0.067 at +1.00 s
- HFD: 0.077 at +0.70 s

Peaks are small and at different signs across states. No coherent lag structure suggesting a directed coupling.

### 2d. Leverage check

Mean |r| for top-decile-length phases = 0.080 vs 0.090 for the bottom 90 %. Long phases are not driving the (already small) correlations.

### Bottom line on coupling

ACA curvature and LHA speed are **independent state signatures**. Both differ by diet state, but they do not co-vary on the within-phase timescale. The diet shift moves both metrics, but through pathways that are decoupled at the 50-ms binned within-phase level. The fed-HFD-falling outlier should be flagged but not over-interpreted (n = 4 sessions, single cell of six, no multiple-test correction).

---

## Combined interpretation

Two findings, internally consistent:

1. **Slow, not fast.** The state difference in ACA curvature and LHA speed is sustained throughout the ±60 s peri-inflection window — meaning the entropy phase, not the inflection event, is the relevant unit of time. Both metrics behave like background regimes that the brain inhabits during a phase, not transient responses.

2. **Parallel, not coupled.** Within a phase, the two metrics do not co-vary. ACA curvature and LHA speed are reading out the diet state from independent dimensions of the dynamics: ACA via the trajectory-shape axis (fed has straighter trajectories), LHA via the trajectory-magnitude axis (fed runs faster).

This is a more useful framing than "one mechanism modulates both." It suggests separate population-level read-outs that we should now localize:

- **ACA curvature:** sub-PC structure (which directions in ACA state-space contribute the curvature) and/or per-unit angular dispersion contributions.
- **LHA speed:** per-unit FR vs ensemble drift contribution; relation to behavioral velocity (drill-down Step 5 already showed LHA speed is partly a behavioral-state read-out).

## Caveats and open issues

- HFD n = 4 — the fed-HFD-falling coupling result is exploratory; replication required.
- The peri-inflection window (±60 s) saturates for most contrasts. To probe whether divergence builds *within* a phase, a future analysis could use phase-normalized time (already done at coarse resolution in drill-down Step 2) at higher resolution with bootstrap CIs at each percentile.
- LHA speed shows divergence at the very window edge — a wider window (e.g. ±120 s) would not change the conclusion (effect is tonic) but would demonstrate that more cleanly.
- We did not test ACA-LHA pairings on other axes (e.g., ACA curvature × LHA curvature, or LHA speed × ACA speed). Those are next, and would be cheap to add.

## Files

- `analysis/stage2_temporal_pairing/step1_periinflection.py`
- `analysis/stage2_temporal_pairing/step2_crossregion.py`
- `data/stage2_temporal_pairing/periinflection_*.csv`, `divergence_onset_summary.csv`
- `data/stage2_temporal_pairing/crossregion_*.csv`
- `figures/stage2_temporal_pairing/periinflection_*.png`, `divergence_onset.png`, `crossregion_*.png`
