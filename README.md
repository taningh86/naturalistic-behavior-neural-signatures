# Naturalistic Behavior Neural Signatures

Code for Neuropixels analyses of LHA / RSP / ACA during foraging in mice across fed, fasted, and high-fat diet states.

K01 work. Aim 1: model LHA gating of cortical dynamics with graph neural ODEs. Aim 2: Neuropixels-Opto perturbation.

## Recordings

Single probe (Neuropixels 2.0, targeting LHA + RSP):
- Coordinates-1: Mouse01 (8 sessions), Mouse02 (4 sessions)
- Coordinates-2: Mouse01 (6 sessions)

Dual probe (two Neuropixels 2.0):
- imec0: ACA
- imec1: LHA + RSP
- Coordinates-1: Mouse01 (24 sessions, fed / fasted / HFD)

Each session: raw `.ap.bin`, Kilosort 3 or 4 sorted output, EthoVision tracking (120 or 100 ms bins).

## Setup

```bash
conda env create -f environment.yml
conda activate spikeinterface
```

Session paths live in `paths.yaml`. Load with:

```python
import yaml
from pathlib import Path

with open("paths.yaml") as f:
    data = yaml.safe_load(f)
```

Don't hardcode paths — they differ between Windows G:/H: drives.

## Folders

Heavy artifacts (`data/`, `figures/`, raw behavior, sorted output) are gitignored.

- `ccg/` — cross-correlograms, kernel-smoothed spike sync, significant-pair FDR for single probe (Coor1 / Coor2)
- `network/` — per-region pair pipelines (ACA-ACA, LHA-LHA, RSP-RSP, LHA-RSP, dual-probe variants). Includes both the original pooled-pairs scripts and the corrected session-level versions
- `single_probe/` — good-unit session-level co-occurrence for Mouse01 Coor1 / Coor2
- `dual_probe/` — good-unit session-level co-occurrence for ACA-ACA, LHA-LHA, LHA-ACA
- `hfd/` — same for HFD, plus 3-way fed / fasted / HFD comparisons
- `gru/` — GRU prediction of population activity, baselines, by-region and pooled variants
- `gru_ode/` — gated neural ODE: pooled-by-region, manifold / topology read-outs, fixed points, autonomous and behaviorally-conditioned runs, GRU comparison
- `entropy/` — behavioral entropy vs FR / PC1 / GRU-ODE metrics; inflection-locked analyses; home-visit categorization; attractor / contraction
- `excursion/` — excursion detection, manifold evolution, latent flow fields, shuffle / identity / mechanism controls
- `foraging/` — bout neural signatures, transit dynamics, transition quantification, shuffle controls
- `retreat/` — retreat detection by source zone, fast-dart filtering, advanced metrics (mostly negative)
- `hesitant/` — hesitant vs committed bouts, pre-bout state, time-resolved latents, transition change-points
- `digging/` — peri-dig ACA / LHA, rate-of-change and duration splits, dig co-occurrence
- `approach/` — pot / ladder / pre-dig / pre-feed approach events, peri-event neural
- `manifold/` — Layer 1a (TwoNN / CorrDim / PR), 1b (Vietoris-Rips H1 / H2), 1c (CCA), Layer 2 (behavioral mapping), avalanche / scale-free controls
- `behavior/` — sequence mining, fed / fasted / HFD metric comparison
- `glm/` — neuron-behavior GLMs, unique-deviance partitioning
- `utils/` — unit QC, population summaries
- `analysis/` — stage drilldowns and curvature / cycles / mapper sub-projects (preserved with original structure)
- `scripts/HMM/` — 19-step HMM / state-modeling pipeline (PyHMM + dynamax), GLM-HMM strategy switches, state-transition neural signatures, Granger ACA-LHA, LFP spectral analysis, SWR detection
- `scripts/HMM_glm/` — GLM-HMM mixed-model fitting (K=6 / K=8)

Top-level: `paths.yaml`, `environment.yml`.

## Approach notes

**Stats.** Session-level throughout. Per-session mean correlation, then Mann-Whitney U on session means. Earlier pooled-pairs versions are kept under `network/` for reference but inflate N.

**Unit selection.** Custom QC thresholds were unreliable. All current analyses require KSLabel = 'good' from Kilosort plus:
- Single probe: FR > 0.3 Hz, AMP > 48 µV
- Dual probe imec0 (ACA): FR > 0.2 Hz, no AMP filter
- Dual probe imec1 (LHA + RSP): FR > 0.2 Hz, AMP > 43 µV

**Depth boundaries.**
- Single probe Coor1: LHA < 1300 µm, RSP ≥ 1300 µm
- Single probe Coor2: LHA ≤ 1410 µm, RSP ≥ 4725 µm
- Dual probe imec1: LHA 0-345 µm, RSP 4680-5025 µm

## Headline results

Single probe (session-level, 10 ms lag):

| pair | state Δ | state p | phase Δ | phase p |
|------|--------|--------|--------|--------|
| LHA-LHA | +92% | 0.029 | +16% | 0.30 |
| RSP-RSP | +65% | 0.029 | +13% | 0.69 |
| LHA-RSP | +460% | 0.11 | +1037% | 0.20 |

Dual probe (session-level, 10 ms lag):

| pair | state Δ | state p | phase Δ | phase p |
|------|--------|--------|--------|--------|
| ACA-ACA | +139% | 0.036 | +87% | 0.23 |
| LHA-LHA | -3.5% | 0.52 | -19% | 0.020 |
| LHA-ACA | +70% | 0.80 | -87% | 0.35 |

HFD 3-way:
- LHA-LHA: HFD >> fed, fasted at 50 / 100 ms (KW p = 0.003)
- ACA-ACA: trending (KW p = 0.074); fed vs fasted at 10 ms p = 0.036
- Extreme correlations (r > 0.1) exist only in HFD; absent from fed / fasted

Modeling:
- GRU-ODE matches GRU R² within 0.003 while reproducing the biological signals
- RSP participation ratio drops with fasting (12.9 → 9.5, p = 0.029)
- RSP latent speed drops with fasting (2.13 → 1.83, p = 0.029)

Entropy-neural opposition:
- LHA-RSP (single probe) and ACA-LHA (dual probe): LHA up during stereotyped low-entropy foraging, cortex up during varied high-entropy exploration
- HFD breaks the ACA-LHA opposition (cortex goes flat)

## Gotchas

- EthoVision tracking has artifacts. QC before use.
- SpikeInterface auto-updates can break old code.
- Big GPU correlation matrices OOM. Chunk by neuron pair.
- Windows paths: always `pathlib.Path`, never raw backslash strings.
- HFD is foraging-arena only (no lever paradigm). Lever-zone scored data must not be included in HFD analyses.
- Dual-probe fed session 2 has null sorted/raw paths. Skip.
- Session 16 fasted uses KS4 for imec0 (not KS3).
