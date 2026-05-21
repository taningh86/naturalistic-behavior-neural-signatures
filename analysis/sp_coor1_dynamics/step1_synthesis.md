# Single-probe Mouse01-Coor1 — Stage 1 dynamics drill-down (Step 1)

**Date**: 2026-04-28. **Sessions**: 8 (1-4 fed, 5-8 fasted; alternating exploration/foraging).

## Procedure

Mirror of dual-probe Stage 1 (`analysis/dynamics_stage1/stage1_batch.py`):
- 50 ms bins, σ=1 Gaussian smoothing, per-unit z-score.
- LHA = depth < 1300 µm, RSP = depth ≥ 1300 µm (single Neuropixels 2.0).
- Filter: KSLabel='good' ∧ fr > 0.3 Hz ∧ Amplitude > 48 µV.
- Speed = ‖ΔN‖, σ=3 bins; curvature = 1−cos θ between successive Δ vectors, σ=3 bins.
- Behavioral entropy: 60 s window, 10 s step, σ=30 s smoothing on transition-string distribution.
- Inflections: prominence 0.3, min distance 60 s.
- Phases: trough/rising/peak/falling spans between consecutive inflections.

## Per-session summary

| Sess | State / Phase | LHA units | RSP units | Entropy pts | Peaks | Troughs | Phases | LHA spd | RSP spd |
|------|---------------|-----------|-----------|-------------|-------|---------|--------|---------|---------|
| S1   | fed / exp     | 30        | 93        | 174         | 4     | 4       | 15     | 3.02    | 4.09    |
| S2   | fed / for     | 32        | 78        | 173         | 4     | 5       | 17     | 3.24    | 3.72    |
| S3   | fed / exp     | 26        | 65        | 173         | 3     | 3       | 11     | 2.62    | 3.20    |
| S4   | fed / for     | 25        | 73        | 174         | 5     | 6       | 21     | 2.76    | 3.49    |
| S5   | fasted / exp  | 16        | 59        | 159         | 5     | 5       | 19     | 2.18    | 3.27    |
| S6   | fasted / for  | 12        | 63        | 136         | 5     | 6       | 21     | 1.84    | 3.59    |
| S7   | fasted / exp  | 19        | 50        | 115         | 1     | 1       | 3      | 2.34    | 3.05    |
| S8   | fasted / for  | 19        | 45        | 166         | 7     | 6       | 25     | 2.38    | 2.81    |
| **all** | —          | —         | —         | —           | 34    | 36      | 132    | —       | —       |

S7 has only 1 peak + 1 trough (smooth entropy in this session); will be a sparse contributor to phase-conditioned stats.

## State contrast on rising+falling phases (session-mean → group mean, n=4 vs n=4)

| Quantity        | Fed (mean ± SD) | Fasted (mean ± SD) | Δ (fasted−fed) | √N expected from unit count |
|-----------------|-----------------|--------------------|----------------|---------------------------|
| LHA speed       | 2.914 ± 0.215   | 2.178 ± 0.190      | **−0.736 (−25%)** | fasted has 16 units vs fed 28 → √(16/28) = 0.756× → expects fasted ≈ 2.20 |
| RSP speed       | 3.611 ± 0.219   | 3.171 ± 0.241      | **−0.440 (−12%)** | fasted 55 vs fed 77 → √(55/77) = 0.846× → expects fasted ≈ 3.05 |
| LHA curvature   | 0.576 ± 0.003   | 0.590 ± 0.007      | **+0.014 (+2.4%)** | curvature is angular, no √N scaling |
| RSP curvature   | 0.521 ± 0.006   | 0.544 ± 0.010      | **+0.023 (+4.4%)** | — |

## Take-aways (provisional, before any hypothesis test)

- **LHA & RSP speed are confounded with unit count.** Fed sessions have ~1.7× more LHA units and ~1.4× more RSP units than fasted. The observed speed differences are ≈ what √N predicts. **Speed contrasts cannot be interpreted at face value** — same lesson as dual-probe Stage 3 Step 3 (LHA K=full speed effect was a √N artifact).
- **Curvature is the unit-count-robust signal.** Both LHA and RSP show fasted > fed by 2-4%. **Direction matches the dual-probe ACA finding** (fed has lower curvature than fasted/HFD). One-mouse, n=4 vs n=4 — no cross-animal generalization possible.
- **State and exp/for phase are 100% confounded across the 4-vs-4 split**, but each state contains 2 EXP + 2 FOR sessions, so phase can be partialled out within-state.

## Caveats

- Single mouse (Mouse01); the n=4 vs n=4 contrast is across sessions of the same animal. Any "state effect" here is really "session effect across 4 fed sessions and 4 fasted sessions of one mouse" — much weaker than the dual-probe data.
- Strong unit-count imbalance (fed > fasted in both regions). Speed-based claims need the same N-matched subsample control as dual-probe Step 3.
- S7 is sparse (3 phases). Phase-conditioned stats will be dominated by S5/S6/S8 in fasted.
- Behavioral repertoire labels differ from dual-probe (single-probe template uses "Digging" not "digging_sand", "Incomplete home return" not "incomplete_home_returns", etc.); the canonical 5-behavior set is mapped where the row exists, otherwise reported as 0.

## Files

- `analysis/sp_coor1_dynamics/sp_lib.py`, `step1_drilldown.py`, `_test_loaders.py`, `_quick_peek.py`
- `data/sp_coor1_dynamics/session_{1..8}_{speed,curvature,phases,phase_data,summary}.{npy,json,npz,csv}`
- `data/sp_coor1_dynamics/all_sessions_summary.csv`, `batch_log.txt`
- `figures/sp_coor1_dynamics/session_{1..8}_diagnostic.png`

## Suggested next steps

1. **Bootstrap CI on curvature state contrast** (LHA & RSP separately, rising/falling phases). The signal is small (2-4%) but unit-count-independent. Use phase-mean → session-mean → 5000 resamples.
2. **Unit-count subsample control on speed**, matched to the strict-min N (LHA=12, RSP=45). 20 random draws per session, recompute speed at matched N. Expect both speed effects to collapse, mirroring dual-probe LHA Step 3.
3. **Per-session PCA + K∈{2,3,5,10,full} subspace dynamics** (port of Stage 3 Step 1) — can the curvature signal be localized to top 2-3 PCs in LHA & RSP, like ACA?
4. **Cross-session PC1-2 alignment** — given that dual-probe ACA showed no shared axis, will likely also be negative here, but worth confirming as a parallel check.
