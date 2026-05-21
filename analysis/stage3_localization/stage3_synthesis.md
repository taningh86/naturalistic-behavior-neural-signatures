# Stage 3 — Localization: which subspace carries the diet-state signatures?

**Goal.** The drill-down established two state signatures: ACA mean curvature and LHA mean speed. Both were computed in *full* unit-space. Now: do these signals live in a low-dimensional subspace, or are they distributed across many components? Asked separately for each region.

## Procedure
For each session (n=17, fed=8, fasted=5, fed-HFD=4; S13/23/24 excluded):
1. Load ACA spike-rate matrix (50 ms bins, σ=1 smoothed, z-scored).
2. PCA → project onto top-K components for K ∈ {2, 3, 5, 10, 20, full}.
3. Compute curvature (1 − cos θ between successive bin-to-bin velocity vectors), σ=3 smoothed (matches Stage 1).
4. Phase-mean curvature for each rising/falling phase (using Stage 1 phase definitions).
5. Session-level mean (rising+falling pooled).
6. Bootstrap pairwise state diffs (5000 resamples).

## Results

### Variance is high-dimensional in ACA
Top-5 variance ratios per session: PC1 ≈ 4.4–6.5 %, PC2 ≈ 3.3–4.1 %, …, PC5 ≈ 2.0–2.6 %. Cumulative variance at K=20 is ~50–60 %. ACA is a *high-dimensional* signal — no dominant axis captures the bulk of variance.

### State diff vs K (5000-bootstrap session-level CIs)

| K | fed–fasted | fed–HFD | fasted–HFD |
|---|---|---|---|
| 2 | **−0.037 [−0.050, −0.023]** ★ | **−0.041 [−0.056, −0.026]** ★ | −0.004 [−0.013, +0.004] |
| 3 | **−0.046 [−0.059, −0.033]** ★ | **−0.039 [−0.055, −0.024]** ★ | +0.007 [−0.008, +0.020] |
| 5 | **−0.040 [−0.053, −0.028]** ★ | **−0.029 [−0.041, −0.018]** ★ | **+0.011 [+0.003, +0.020]** ★ |
| 10 | **−0.040 [−0.052, −0.029]** ★ | **−0.025 [−0.035, −0.015]** ★ | **+0.015 [+0.008, +0.023]** ★ |
| 20 | **−0.037 [−0.048, −0.029]** ★ | **−0.024 [−0.033, −0.017]** ★ | **+0.013 [+0.007, +0.021]** ★ |
| full | **−0.018 [−0.023, −0.013]** ★ | **−0.012 [−0.016, −0.008]** ★ | **+0.006 [+0.002, +0.011]** ★ |

★ = bootstrap CI excludes zero. Diff = mean(curv state A) − mean(curv state B).

### Three findings

1. **The fed-vs-{fasted, HFD} signature is captured by the top 2 PCs.** At K=2 the CIs already exclude zero with effect sizes (−0.037, −0.041) comparable to or *larger* than at full dimensionality (−0.018, −0.012).

2. **Full space dilutes, not amplifies.** Effect sizes peak around K=2–10 and shrink ~3-fold by full dimensionality. The high-dimensional residual (~600+ ACA PCs) adds noise to curvature without adding state-specific signal.

3. **HFD-vs-fasted is structurally different from fed-vs-others.** At K=2/3 fasted vs HFD is null; the contrast emerges only at K≥5 (fasted > HFD by ~0.011–0.015). The HFD-fasted distinction lives in higher-order PCs than the fed-vs-fasted distinction does.

## Interpretation

- **The diet-state curvature shift is structurally low-dimensional.** It does not require the full ACA population — the top 2 covariance axes are enough to read it out. This matters because (a) it makes the signal interpretable and (b) it suggests a meaningful population coding axis aligned with metabolic state, not a high-dim noise property.

- **PC1–2 of ACA explain ~8 % of variance but carry the diet-state signature.** This is reminiscent of "neural manifold" results where task-relevant directions are not the highest-variance directions — but here, paradoxically, they *are*: the dominant covariance directions are the ones that distinguish states. Worth noting: this is per-session PCA. A future analysis should test whether a *common* top-2 axis aligns across sessions/states (CCA or generalized PCA).

- **HFD is a partial outlier.** Fed vs HFD looks like fed vs fasted at low K; but at higher K, fasted and HFD pull apart. This is consistent with the drill-down finding that HFD aligns with fasted on the leading mean-curvature axis but has its own structure.

## Caveats

- Per-session PCA with no cross-session alignment. Different K=2 subspaces in different sessions are not the same subspace. The result is "low intrinsic dim is sufficient" not "the same low-dim axis carries the state effect across mice."
- HFD n=4 — the K-dependent fasted-vs-HFD pattern (null at K=2, sig at K≥5) needs replication.
- We did not run the same analysis on LHA speed; LHA is fewer units, lower dimensionality already, and its state signal probably localizes immediately to ~PC1.
- The fed-vs-fasted CI at K=3 (−0.046) is the largest effect — modest hint that the third component matters slightly more than PC1 alone, but step is within the bootstrap noise.

---

# Step 2 — LHA mean speed

Same pipeline, LHA matrix, speed instead of curvature. K ∈ {2, 3, 5, 10, full}.

LHA unit counts (relevant for full-space comparison): fed mean ~104, fasted mean ~78 (S15/S16 only 52/54), HFD mean ~82. So fed > fasted in unit count by ~33 % — flagged as a potential confound in the full-K row.

| K | fed–fasted | fed–HFD | fasted–HFD |
|---|---|---|---|
| 2 | +0.014 [−0.049, +0.066] | **+0.074 [+0.049, +0.097]** ★ | **+0.060 [+0.014, +0.118]** ★ |
| 3 | +0.013 [−0.093, +0.124] | +0.073 [−0.049, +0.185] | +0.060 [−0.062, +0.172] |
| 5 | −0.008 [−0.138, +0.114] | **+0.113 [+0.024, +0.204]** ★ | **+0.121 [+0.014, +0.243]** ★ |
| 10 | +0.046 [−0.061, +0.137] | **+0.113 [+0.006, +0.200]** ★ | +0.067 [−0.059, +0.195] |
| full | **+0.909 [+0.201, +1.621]** ★ | **+0.691 [+0.200, +1.129]** ★ | −0.218 [−1.029, +0.566] |

★ = CI excludes zero.

### LHA pattern is the **opposite** of ACA

1. **fed-vs-fasted is captured only at full dimensionality.** At K=2/5/10 the CI brackets zero. At K=full the diff is large (Δ = +0.91, ~17 % relative) and significant.

2. **fed-vs-HFD is robust at K=2 and K=full.** Δ = +0.07 at K=2 already excludes zero (relative ~12 %), and grows monotonically to +0.69 at K=full. HFD has a *low-dimensional* speed signature distinct from fed; fasted does not.

3. **fasted-vs-HFD signs and significance flip with K.** Positive and CI-excluding-zero at K=2 and K=5; negative and ns at K=full. Likely sensitive to unit-count differences and noise; do not over-interpret.

### LHA full-space caveat

LHA-speed unit counts vary substantially by state (fed 104, fasted 78, HFD 82). Speed scales roughly with √N (it is the L2 norm of the bin-to-bin diff vector summed over units). Some of the K=full fed-vs-fasted effect is therefore mechanically attributable to unit count. But:
- The same scaling would inflate fed-vs-HFD too (HFD ~82 ≈ fasted), and we *do* see fed > HFD at K=full — consistent with a real gain on top of the unit-count effect.
- At K=10 (fixed), fed-vs-fasted is ns (CI brackets zero). So the K=full fed-vs-fasted result is suspect: it could be the full-residual that carries the true state effect, or it could be the unit-count confound. Cannot distinguish from this analysis.

### Interpretation contrast vs. ACA curvature

| Signature | Where the signal lives | Effect at K=2 vs K=full |
|---|---|---|
| ACA curvature | dominant 2 covariance axes | K=2 effect 3× larger than K=full |
| LHA speed | distributed across full population | K=2 effect ≈ 0 (fed-vs-fasted), K=full strongest |

The two surviving state signatures localize to **opposite ends of the dimensionality spectrum**. ACA reads metabolic state via a low-dim coding axis aligned with the dominant covariance direction. LHA does not — its diet-state effect is spread across units.

This sharpens the Stage 2 finding that ACA curvature and LHA speed are *independent* state read-outs: not only do they not co-vary within phases, they live at different dimensionalities of population activity.

---

# Step 3 — LHA unit-count-controlled subsample test (REVISES the Step 2 K=full claim)

To address the unit-count caveat in Step 2 (LHA fed mean ~104 units, fasted ~78, HFD ~82), each session's LHA matrix was randomly subsampled to a matched N=52 (strict minimum, set by S15) for 20 independent draws. Trajectory speed was recomputed in the subsampled space (σ=3 smoothing, same as Stage 1 / Step 2). For each draw, session-level diet-state diffs were bootstrapped (5000 resamples), and outcomes aggregated across draws.

P25 N=86 was infeasible because S15 (52 units) and S16 (54 units) lie below it; secondary check skipped.

| Contrast | Median diff (matched N=52) | IQR across 20 draws | n_sig draws | frac_sig | Original (unmatched) |
|---|---|---|---|---|---|
| fed_vs_fasted | **+0.120** | [+0.105, +0.148] | 4/20 | 0.20 | +0.909 [+0.201, +1.621] ★ |
| fed_vs_fed-HFD | **+0.074** | [+0.057, +0.098] | 2/20 | 0.10 | +0.691 [+0.200, +1.129] ★ |
| fasted_vs_fed-HFD | −0.042 | [−0.073, −0.022] | 0/20 | 0.00 | n/a (was ns at K=full) |

**Outcome classification: both fed-vs-fasted and fed-vs-HFD = B (Disappears).**

- Magnitude collapses to ~13 % of the unmatched K=full effect for fed-vs-fasted, ~11 % for fed-vs-HFD.
- Only 20 % / 10 % of draws reach bootstrap CI excluding zero, despite the direction being preserved in 100 % / 95 % of draws.
- Stability across draws is moderate (rel-spread 1.25 and 2.44 — not "wildly variable" but no draw individually replicates the original effect size).
- The matched-N median diffs (+0.12 fed-vs-fasted, +0.07 fed-vs-HFD) are quantitatively similar to the **K=2** effects from Step 2 (+0.014 and +0.074 respectively), not the K=full effects (+0.91 and +0.69).

### Revision of the Step 2 conclusion

The Step 2 claim that LHA fed-vs-fasted speed signature is "high-dimensional and only emerges at K=full" does **not** survive the unit-count control. The K=full effect was largely produced by the √N scaling of L2-norm speed under a 33 % unit-count gap (fed 104 vs fasted 78). At matched N, the residual fed-vs-fasted speed effect is small and inconsistent.

The fed-vs-HFD signature retains the same magnitude at matched N (+0.074) as it had at K=2 (+0.074) — i.e. the **fed-vs-HFD low-D component from Step 2 is genuine**, but the higher-K growth (to +0.69 at K=full) is also unit-count-driven.

### Revised Stage 3 framing

| Signature | Where the signal lives | What survives controls |
|---|---|---|
| ACA curvature | Top 2 PCs (~8 % var) | Bootstrap-significant at every K from 2 to full; K=2 effect 3× larger than K=full → **structural low-D coding axis** |
| LHA speed (fed-vs-fasted) | None robust | Disappears at matched unit count; K=full effect was a √N artifact |
| LHA speed (fed-vs-HFD) | Top 2 PCs (small) | Persists at K=2 (Δ≈+0.07) and at matched-N (Δ≈+0.07); modest **low-D HFD-specific signature** |

The original "opposite ends of dimensionality spectrum" claim is **withdrawn**. Replace with: ACA carries a robust low-D state-coding axis; LHA does **not** carry a robust diet-state speed signature once unit count is controlled, except for a modest HFD-specific low-D component.

---

## Next steps

- **Cross-session subspace alignment for ACA** — does PC1–2 of one session align with PC1–2 of another (CCA, Procrustes)? If yes, "the same axis" carries state info across mice. *Greenlit — ACA finding is the surviving Stage 3 result.*
- **Per-unit angular contribution for ACA** — which units drive the curvature in PC1–2? *Greenlit.*
- **Revise Step 2 claims in any K01 / paper draft.** The "LHA reads diet state via the full population" framing was wrong. The honest story is: ACA reads diet state via a low-D coding axis; LHA does not have a robust population-level speed signature for fed-vs-fasted, but does have a small low-D HFD signature.
- **Optional**: re-run Step 2 unit-count control for LHA *curvature* (not speed) — curvature is dimensionless and cosine-based, so √N scaling does not apply. Curvature was non-significant in the original Stage 1 LHA analysis, but worth a sanity check at matched N to confirm.

---

# Step 4 — Cross-session ACA PC1–2 alignment (FURTHER REVISES the Step 1 claim)

To test whether the per-session top-2 PC subspaces are *aligned* across sessions (i.e., whether there is a shared low-D coding axis for diet state) or whether each session has its own idiosyncratic 2D subspace that happens to contain a state effect.

**Procedure**: Per session, project ACA matrix to PC1–2; for each rising/falling phase interval, extract the (PC1, PC2) trajectory and resample to fixed length L=50 via linear interpolation; average within session × phase type → mean shape (50×2). Pairwise `scipy.spatial.procrustes` (auto-standardizes; rotation+reflection alignment) yields disparity = sum-of-squared distances after best alignment. One-sided Mann-Whitney U: same-state pair disparity < different-state pair. 1000-shuffle permutation null on session-state labels.

**Sessions**: 17 (fed n=8, fasted n=5, HFD n=4). 44 same-state pairs, 92 different-state pairs.

| Phase | n same | n diff | med disp same | med disp diff | MW U | p (1-sided) |
|---|---|---|---|---|---|---|
| rising | 44 | 92 | 0.9470 | 0.9540 | 1930 | 0.332 |
| falling | 44 | 92 | 0.9628 | 0.9553 | 2202 | 0.797 (reversed) |
| avg | 44 | 92 | 0.9516 | 0.9518 | 2120 | 0.673 |

Permutation null: observed median(diff) − median(same) = +0.00021; null mean = +0.00009; 95th pct = +0.00763; **permutation p = 0.473**.

**State-pair breakdown** (avg over phase types):

| Contrast | n pairs | median disparity |
|---|---|---|
| fed vs fasted | 40 | 0.9499 |
| fed vs fed-HFD | 32 | 0.9619 |
| fasted vs fed-HFD | 20 | 0.9358 |
| fed vs fed | 28 | 0.9581 |
| fasted vs fasted | 10 | **0.9078** |
| fed-HFD vs fed-HFD | 6 | 0.9511 |

### Result: NEGATIVE

Same-state and different-state pair disparities are essentially identical (~0.95). No statistical separation. The K=2 subspace appears to be **locally re-derived in each session** — same property ("dominant covariance directions encode state") but **different axes per mouse**.

The only weak hint is that *fasted vs fasted* same-state pairs have lower disparity (0.91) than other groupings, but n=10 pairs from 5 mice is too small to push the aggregate test, and same-state fed-vs-fed pairs (n=28) are actually slightly *higher* (0.96) than fed-vs-fasted different-state pairs.

### What this means for the K01 framing

The Stage 3 Step 1 result is preserved at the within-session level: ACA top-2 PCs carry diet-state info per session, and the K=2 effect is 3× larger than at full space. But the strong claim "ACA reads diet state via a *shared* low-D coding axis aligned across mice" is **not supported**.

The honest framing for the K01 narrative becomes:
- ACA's diet-state signal is concentrated in each session's dominant covariance directions, but those directions are not the same across mice (in latent score space).
- "Variance ranking is dominant" is a conserved property; "the same axis" is not.
- This is consistent with mechanistic heterogeneity: different mice may use different unit ensembles to encode metabolic state.

### Caveats

- 2D Procrustes is permissive — every shape can be partly aligned with any other, so the test relies on differential disparity, not absolute alignment. The differential is essentially zero here.
- HFD n=4 → only 6 same-state HFD pairs. Could be missing power for that subgroup.
- Cross-session alignment is in latent score space (post-projection), not unit space — UnitMatch is not available for ACA. A *unit-space* alignment via UnitMatch could in principle be more sensitive but requires registration not yet performed.
- Phase intervals vary in duration; resampling to fixed L=50 collapses time-rate differences.
- Test was on K=2. A higher-K version (e.g., 5 PCs) might pick up shared structure that's not visible at K=2 — but Step 1 already showed K=2 is where the per-session diet effect is concentrated, so it's the right K to test for cross-session alignment.

---

## Final revised Stage 3 framing (after Steps 1–4)

| Claim | Status |
|---|---|
| ACA mean curvature shifts with diet state at session level | **Robust** (Stage 1 / Step 1) |
| In each session, the diet effect is concentrated in top-2 PCs (K=2 effect 3× full) | **Robust** (Step 1) |
| LHA mean speed shifts with diet state at K=full | **Withdrawn** — √N unit-count artifact (Step 3) |
| Stage 3 "opposite ends of dimensionality spectrum" framing | **Withdrawn** (Step 3) |
| ACA top-2 PCs are aligned across mice (shared coding axis) | **Not supported** (Step 4) |
| ACA top-2 PCs each carry the state effect in their own session-specific 2D subspace | Supported but weaker than initially claimed |

The surviving Stage 3 result: **ACA's diet-state signature is structurally low-dimensional within each session, even though the specific axis differs across mice.** Useful but more modest than the original framing.

## Files

- `analysis/stage3_localization/step1_aca_subspace.py`, `step2_lha_subspace.py`
- `analysis/stage3_lha_control/step1_subsample_test.py` (Step 3 unit-count control)
- `analysis/stage3_cross_session_alignment/step1_pc12_phase_alignment.py` (Step 4 cross-session)
- ACA: `data/stage3_localization/per_session_subspace_curv.csv`, `session_means_per_K.csv`, `state_diff_vs_K.csv`, `explained_variance_long.csv`
- LHA: `lha_per_session_subspace_speed.csv`, `lha_session_means_per_K.csv`, `lha_state_diff_vs_K.csv`, `lha_explained_variance_long.csv`
- LHA control: `data/stage3_lha_control/lha_subsample_per_draw.csv`, `lha_subsample_summary.csv`, `lha_subsample_summary.md`
- Cross-session: `data/stage3_cross_session_alignment/per_session_phase_shapes.npz`, `pairwise_disparity.csv`, `pairwise_disparity_avg.csv`, `same_vs_diff_state_test.csv`, `state_pair_breakdown.csv`, `cross_session_summary.md`
- `figures/stage3_localization/state_diff_vs_K.png`, `lha_state_diff_vs_K.png`, `explained_variance_per_session.png`, `lha_explained_variance_per_session.png`
- `figures/stage3_lha_control/lha_subsample_distribution.png`, `lha_subsample_comparison.png`
- `figures/stage3_cross_session_alignment/per_session_pc12_phase_shapes.png`, `disparity_by_pair_type.png`, `permutation_null.png`
- Matrix cache: `data/stage3_localization/_cache/session_X_{aca,lha}.npy`
