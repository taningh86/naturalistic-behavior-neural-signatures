"""16 — Granger causality between ACA and LHA at S3 (home) transitions.

Tests directionality of pre-exit coupling at S3 (the state with multi-metric
pre-exit signatures from script 15). Both ACA and LHA show pre-exit changes;
this script asks: which region's activity predicts the other's first?

Pipeline:
  1. Per session, 50 ms binning. PC1 of full-session z-scored firing rates
     per region. Population-summed firing rate per region.
  2. Identify S3 (home) runs from merged Viterbi (480 ms grid). Map to 50 ms.
     Build segments: each S3 run + 5 s post-exit (~100 bins).
  3. Skip runs where S3 portion < 5 s (~100 bins).
  4. For each session × signal (pop_sum, PC1) × direction (ACA→LHA, LHA→ACA):
     a. Bivariate VAR-style F-test for Granger causality, fitting per segment
        and pooling design matrices (no lags spanning segment boundaries).
     b. Lag p selected by BIC across p ∈ [1, 20] on the bivariate
        unrestricted model (same p used for both directions to keep them
        comparable).
     c. 100 circular-shift shuffles of the predictor region's full-session
        time series (offset ∈ [200, T-200] bins, 50 ms). Re-extract segments,
        recompute F. Build null.
  5. Cross-session aggregation: per signal/direction, count sessions passing
     shuffle p95; asymmetry index per session.

Out of scope: multivariate Granger, non-S3 transitions, spike-level Granger,
LFP, decoder.
"""
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import f as f_dist
from scipy.stats import binomtest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "HMM"))

import spikeinterface.extractors as se
from dp_avalanche_criticality import (
    get_good_units_p0,
    get_good_units_p1_lha,
    load_spike_times_for_region,
)
from _utils import load_config


# ---- Constants ----
NEURAL_BIN_S = 0.050             # 50 ms
HMM_BIN_S = 0.480
SESSIONS = [4, 6, 8, 12, 14, 16]
S3_STATE = 3
POST_EXIT_S = 5.0
POST_EXIT_BINS = int(POST_EXIT_S / NEURAL_BIN_S)   # 100 bins
MIN_S3_BINS = int(POST_EXIT_S / NEURAL_BIN_S)      # 100 bins (5 s)
LAG_RANGE = list(range(1, 21))
N_SHUFFLES = 100
SHUFFLE_MIN_OFFSET = 200
SHUFFLE_MARGIN = 200
SHUFFLE_SEED = 20260508


def out_dirs():
    base_out = REPO_ROOT / "data" / "HMM" / "neural_alignment" / "granger"
    base_fig = REPO_ROOT / "figures" / "HMM" / "neural_alignment" / "granger"
    base_out.mkdir(parents=True, exist_ok=True)
    base_fig.mkdir(parents=True, exist_ok=True)
    return base_out, base_fig


# ---- Granger helpers ----
def build_lagged(segments_x, segments_y, p):
    """Build pooled lagged design matrix. y[t] ~ y_lags + x_lags + intercept.

    segments_x, segments_y: lists of 1D arrays of equal lengths per segment.
    p: lag order.

    Returns:
      X_full: (N, 2p+1)  — y_lags (p) + x_lags (p) + intercept
      X_y_only: (N, p+1) — y_lags + intercept (restricted model design)
      y: (N,)
    """
    X_full_list = []
    X_y_only_list = []
    y_list = []
    for sx, sy in zip(segments_x, segments_y):
        L = len(sy)
        if L <= p:
            continue
        n = L - p
        y_lags = np.column_stack([sy[p - k:L - k] for k in range(1, p + 1)])
        x_lags = np.column_stack([sx[p - k:L - k] for k in range(1, p + 1)])
        intercept = np.ones((n, 1))
        X_y_only_list.append(np.column_stack([y_lags, intercept]))
        X_full_list.append(np.column_stack([y_lags, x_lags, intercept]))
        y_list.append(sy[p:L])
    if not y_list:
        return None, None, None
    return (np.vstack(X_full_list),
            np.vstack(X_y_only_list),
            np.concatenate(y_list))


def granger_F(segments_x, segments_y, p):
    """Test if x Granger-causes y at lag p. Returns (F, p_val, n_samples)."""
    Xu, Xr, y = build_lagged(segments_x, segments_y, p)
    if y is None:
        return np.nan, np.nan, 0
    n = len(y)
    if n <= 2 * p + 2:
        return np.nan, np.nan, n
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            beta_r, _, _, _ = np.linalg.lstsq(Xr, y, rcond=None)
            beta_u, _, _, _ = np.linalg.lstsq(Xu, y, rcond=None)
        except np.linalg.LinAlgError:
            return np.nan, np.nan, n
    RSS_r = float(np.sum((y - Xr @ beta_r) ** 2))
    RSS_u = float(np.sum((y - Xu @ beta_u) ** 2))
    df_diff = p
    df_u = n - (2 * p + 1)
    if df_u <= 0 or RSS_u <= 1e-12:
        return np.nan, np.nan, n
    F = ((RSS_r - RSS_u) / df_diff) / (RSS_u / df_u)
    p_val = 1.0 - f_dist.cdf(F, df_diff, df_u)
    return float(F), float(p_val), n


def select_lag_BIC(segments_x, segments_y, lag_range=LAG_RANGE):
    """Select lag by BIC on the unrestricted bivariate model.
    BIC = n*log(RSS/n) + k*log(n), where k = 2p+1."""
    best_p = lag_range[0]; best_bic = np.inf
    for p in lag_range:
        Xu, _, y = build_lagged(segments_x, segments_y, p)
        if y is None or len(y) <= 2 * p + 2:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                beta_u, _, _, _ = np.linalg.lstsq(Xu, y, rcond=None)
            except np.linalg.LinAlgError:
                continue
        RSS_u = float(np.sum((y - Xu @ beta_u) ** 2))
        n = len(y)
        if RSS_u <= 1e-12:
            continue
        k = 2 * p + 1
        bic = n * np.log(RSS_u / n) + k * np.log(n)
        if bic < best_bic:
            best_bic = bic; best_p = p
    return best_p


# ---- Session loader ----
def load_session(sn, cfg, paths_data):
    s_paths = (paths_data["double_probe"]["coordinates_1"]["mouse01"]
                ["sessions"][f"session_{sn}"])
    aca_sorted = Path(s_paths["probe_0_aca"]["sorted"])
    lha_sorted = Path(s_paths["probe_1_lha_rsp"]["sorted"])
    aca_uids = [int(u) for u in get_good_units_p0(aca_sorted)]
    lha_uids = [int(u) for u in get_good_units_p1_lha(lha_sorted)]
    aca_sorting = se.read_kilosort(aca_sorted)
    lha_sorting = se.read_kilosort(lha_sorted)
    aca_uids = [u for u in aca_uids if u in set(aca_sorting.get_unit_ids())]
    lha_uids = [u for u in lha_uids if u in set(lha_sorting.get_unit_ids())]
    aca_spikes = load_spike_times_for_region(aca_sorting, aca_uids)
    lha_spikes = load_spike_times_for_region(lha_sorting, lha_uids)

    binned = np.load(REPO_ROOT / cfg["out_dirs"]["binned"]
                      / f"session_{sn}.npz", allow_pickle=True)
    trial_time = np.asarray(binned["trial_time"], dtype=np.float64)
    n_hmm = len(trial_time)
    duration_s = float(trial_time[-1] + HMM_BIN_S)
    n_50 = int(np.ceil(duration_s / NEURAL_BIN_S))
    edges_50 = np.arange(n_50 + 1) * NEURAL_BIN_S

    aca_uid_list = sorted(aca_spikes.keys())
    lha_uid_list = sorted(lha_spikes.keys())
    n_aca = len(aca_uid_list); n_lha = len(lha_uid_list)
    aca_50 = np.zeros((n_aca, n_50)); lha_50 = np.zeros((n_lha, n_50))
    for i, uid in enumerate(aca_uid_list):
        aca_50[i] = np.histogram(aca_spikes[uid], edges_50)[0]
    for i, uid in enumerate(lha_uid_list):
        lha_50[i] = np.histogram(lha_spikes[uid], edges_50)[0]

    post = pd.read_csv(REPO_ROOT / cfg["merge_dirs"]["posteriors"]
                        / f"session_{sn}.csv")
    viterbi_480 = post["viterbi"].values.astype(np.int64)

    history = pd.read_csv(REPO_ROOT / cfg["commitment_dirs"]["out"]
                           / "sampling_history.csv")
    s_hist = history[history.session == sn].iloc[0]
    return dict(sn=sn, n_50=n_50, n_hmm=n_hmm,
                 aca_counts=aca_50, lha_counts=lha_50,
                 viterbi_480=viterbi_480,
                 metabolic_state=s_hist["state"],
                 n_aca=n_aca, n_lha=n_lha)


# ---- Build per-session signals + segments ----
def build_signals(session):
    """Returns dict with pop_sum_aca, pop_sum_lha, pc1_aca, pc1_lha, etc."""
    aca_50 = session["aca_counts"]; lha_50 = session["lha_counts"]

    # Population-summed firing rate (z-scored within session)
    aca_pop = aca_50.sum(axis=0)
    lha_pop = lha_50.sum(axis=0)
    aca_pop_z = (aca_pop - aca_pop.mean()) / (aca_pop.std() + 1e-9)
    lha_pop_z = (lha_pop - lha_pop.mean()) / (lha_pop.std() + 1e-9)

    # PCA on z-scored firing rates per region
    def pcs_top3(rates):
        mu = rates.mean(axis=1, keepdims=True)
        sd = rates.std(axis=1, keepdims=True) + 1e-9
        z = (rates - mu) / sd                          # (n_units, n_bins)
        X = z.T                                        # (n_bins, n_units)
        Xc = X - X.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        pcs = U[:, :3] * S[:3]                          # (n_bins, 3)
        return pcs

    aca_pcs = pcs_top3(aca_50)
    lha_pcs = pcs_top3(lha_50)

    return dict(
        aca_pop=aca_pop_z, lha_pop=lha_pop_z,
        aca_pc1=aca_pcs[:, 0], lha_pc1=lha_pcs[:, 0],
        aca_pc2=aca_pcs[:, 1], lha_pc2=lha_pcs[:, 1],
        aca_pc3=aca_pcs[:, 2], lha_pc3=lha_pcs[:, 2],
    )


def build_s3_segments(viterbi_480, n_50):
    """Return list of (start_50, end_50, exit_50) per S3 run with ≥5s S3 length.

    The 480 ms HMM bin t covers neural-bin range [t*0.48/0.05, (t+1)*0.48/0.05).
    """
    n = len(viterbi_480)
    diff = np.diff(viterbi_480, prepend=-1, append=-1)
    boundaries = np.flatnonzero(diff != 0)
    starts = boundaries[:-1]; ends = boundaries[1:]
    rate_per_hmm = HMM_BIN_S / NEURAL_BIN_S   # 9.6
    segments = []
    for s, e in zip(starts, ends):
        if int(viterbi_480[s]) != S3_STATE:
            continue
        # Map to 50 ms bins
        s_50 = int(round(s * rate_per_hmm))
        e_50 = int(round(e * rate_per_hmm))
        s3_len_50 = e_50 - s_50
        if s3_len_50 < MIN_S3_BINS:
            continue
        # Add post-exit window
        seg_end_50 = min(n_50, e_50 + POST_EXIT_BINS)
        segments.append(dict(s3_start=s_50, s3_end=e_50,
                              segment_end=seg_end_50,
                              s3_len_bins=s3_len_50,
                              segment_len=seg_end_50 - s_50))
    return segments


def extract_segment_arrays(signals, segments, key):
    """Return list of 1D arrays, one per segment, of the named signal."""
    arr = signals[key]
    return [arr[s["s3_start"]:s["segment_end"]] for s in segments]


# ---- Per-session Granger run ----
def run_session(session, base_out, base_fig, rng):
    sn = session["sn"]
    out_dir = base_out / f"session_{sn}"
    fig_dir = base_fig / f"session_{sn}"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n--- S{sn} ({session['metabolic_state']}) "
          f"ACA={session['n_aca']} LHA={session['n_lha']} ---", flush=True)
    t0 = time.time()

    signals = build_signals(session)
    segments = build_s3_segments(session["viterbi_480"], session["n_50"])
    if not segments:
        print(f"  No qualifying S3 segments. Skipping.")
        return None
    seg_lens = [s["segment_len"] for s in segments]
    s3_lens = [s["s3_len_bins"] for s in segments]
    print(f"  S3 segments: n={len(segments)}, "
          f"mean S3 length={np.mean(s3_lens):.1f} bins ({np.mean(s3_lens)*NEURAL_BIN_S:.1f}s), "
          f"mean total segment length={np.mean(seg_lens):.1f} bins")

    # Save segments
    seg_rows = []
    for i, s in enumerate(segments):
        seg_rows.append(dict(
            segment_idx=i, s3_start_50=s["s3_start"], s3_end_50=s["s3_end"],
            segment_end_50=s["segment_end"], s3_len_bins=s["s3_len_bins"],
            segment_len=s["segment_len"],
            s3_start_s=s["s3_start"] * NEURAL_BIN_S,
            s3_end_s=s["s3_end"] * NEURAL_BIN_S,
        ))
    pd.DataFrame(seg_rows).to_csv(out_dir / f"session_{sn}_segments.csv",
                                     index=False)

    # Save signals
    np.savez(out_dir / f"session_{sn}_signals.npz",
              aca_pop=signals["aca_pop"], lha_pop=signals["lha_pop"],
              aca_pc1=signals["aca_pc1"], lha_pc1=signals["lha_pc1"],
              aca_pc2=signals["aca_pc2"], lha_pc2=signals["lha_pc2"],
              aca_pc3=signals["aca_pc3"], lha_pc3=signals["lha_pc3"])

    obs_rows = []
    shuf_rows = []

    SIGNAL_NAMES = [("pop_sum", "aca_pop", "lha_pop"),
                     ("pc1", "aca_pc1", "lha_pc1")]

    for sig_name, aca_key, lha_key in SIGNAL_NAMES:
        aca_segs = extract_segment_arrays(signals, segments, aca_key)
        lha_segs = extract_segment_arrays(signals, segments, lha_key)

        # Lag selection on bivariate unrestricted (use ACA→LHA direction,
        # but the chosen lag also applies to LHA→ACA for symmetry)
        p = select_lag_BIC(aca_segs, lha_segs, LAG_RANGE)
        print(f"  {sig_name}: BIC-selected p={p}", flush=True)

        # Observed F per direction
        F_aca_lha, p_aca_lha, n_aca_lha = granger_F(aca_segs, lha_segs, p)
        F_lha_aca, p_lha_aca, n_lha_aca = granger_F(lha_segs, aca_segs, p)
        print(f"    ACA→LHA: F={F_aca_lha:.2f} p={p_aca_lha:.3g} (n={n_aca_lha})", flush=True)
        print(f"    LHA→ACA: F={F_lha_aca:.2f} p={p_lha_aca:.3g} (n={n_lha_aca})", flush=True)

        # Shuffle: circularly shift the FULL predictor signal, re-extract,
        # recompute F. 100 iterations per direction.
        T = session["n_50"]
        shuf_F_aca_lha = np.full(N_SHUFFLES, np.nan)
        shuf_F_lha_aca = np.full(N_SHUFFLES, np.nan)
        for it in range(N_SHUFFLES):
            offset = int(rng.integers(SHUFFLE_MIN_OFFSET, T - SHUFFLE_MARGIN))
            aca_shifted = np.roll(signals[aca_key], offset)
            lha_shifted = np.roll(signals[lha_key], offset)
            aca_segs_shuf = [aca_shifted[s["s3_start"]:s["segment_end"]]
                              for s in segments]
            lha_segs_shuf = [lha_shifted[s["s3_start"]:s["segment_end"]]
                              for s in segments]
            # ACA→LHA: shift ACA (predictor), keep LHA real
            F_a, _, _ = granger_F(aca_segs_shuf, lha_segs, p)
            shuf_F_aca_lha[it] = F_a
            # LHA→ACA: shift LHA (predictor), keep ACA real
            F_l, _, _ = granger_F(lha_segs_shuf, aca_segs, p)
            shuf_F_lha_aca[it] = F_l

        for direction, obs_F, obs_p, n_obs, shuf_arr in [
            ("ACA->LHA", F_aca_lha, p_aca_lha, n_aca_lha, shuf_F_aca_lha),
            ("LHA->ACA", F_lha_aca, p_lha_aca, n_lha_aca, shuf_F_lha_aca),
        ]:
            valid_shuf = shuf_arr[np.isfinite(shuf_arr)]
            if len(valid_shuf) == 0:
                p95 = np.nan; pct = np.nan; passes = False
            else:
                p95 = float(np.percentile(valid_shuf, 95))
                pct = float((valid_shuf <= obs_F).mean() * 100)
                passes = bool(np.isfinite(obs_F) and obs_F > p95)
            obs_rows.append(dict(
                session=sn, signal=sig_name, direction=direction,
                lag_p=p, observed_F=obs_F,
                F_p_value=obs_p, n_samples=n_obs,
                shuffle_mean=float(np.nanmean(valid_shuf)) if len(valid_shuf) else np.nan,
                shuffle_p95=p95, obs_pctile=pct, exceeds_p95=passes,
            ))
            for ii, F_s in enumerate(shuf_arr):
                shuf_rows.append(dict(session=sn, signal=sig_name,
                                        direction=direction, iter_idx=ii,
                                        shuffled_F=float(F_s) if np.isfinite(F_s) else np.nan))

    df_obs = pd.DataFrame(obs_rows)
    df_obs.to_csv(out_dir / f"session_{sn}_observed.csv", index=False)
    df_shuf = pd.DataFrame(shuf_rows)
    df_shuf.to_csv(out_dir / f"session_{sn}_shuffle.csv", index=False)

    # Per-direction observed-F + shuffle-null figure
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=False)
    axes_flat = axes.flatten()
    panel_specs = [
        ("pop_sum", "ACA->LHA"), ("pop_sum", "LHA->ACA"),
        ("pc1", "ACA->LHA"), ("pc1", "LHA->ACA"),
    ]
    for ax, (sig, direction) in zip(axes_flat, panel_specs):
        obs_row = df_obs[(df_obs.signal == sig) & (df_obs.direction == direction)]
        if len(obs_row) == 0:
            continue
        obs_F = float(obs_row.iloc[0]["observed_F"])
        p95 = float(obs_row.iloc[0]["shuffle_p95"])
        pct = float(obs_row.iloc[0]["obs_pctile"])
        shuf = df_shuf[(df_shuf.signal == sig)
                          & (df_shuf.direction == direction)]["shuffled_F"].dropna().values
        ax.hist(shuf, bins=20, color="#9999cc", edgecolor="white")
        ax.axvline(obs_F, color="red", lw=2,
                    label=f"observed F={obs_F:.2f}")
        ax.axvline(p95, color="black", lw=1, ls="--",
                    label=f"shuf p95={p95:.2f}")
        ax.set_title(f"{sig} — {direction} (obs at {pct:.0f}th pctile)",
                      fontsize=10)
        ax.set_xlabel("F"); ax.set_ylabel("# shuffles")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(f"S{sn} ({session['metabolic_state']}) — Granger F, "
                 f"observed vs 100 circular-shift shuffles", y=1.0)
    fig.tight_layout()
    fig.savefig(fig_dir / "observed_F_per_direction.png", dpi=130)
    plt.close(fig)

    # Segments overlay figure
    fig, ax = plt.subplots(figsize=(13, 3.5))
    v = session["viterbi_480"]
    n_v = len(v)
    time_v = np.arange(n_v) * HMM_BIN_S
    cmap = plt.cm.tab20
    K = max(int(v.max()) + 1, 14)
    state_colors = np.array([cmap(int(s) % cmap.N) for s in v])
    ax.scatter(time_v, np.full(n_v, 0.5), c=state_colors, s=2, marker="s")
    for i, s in enumerate(segments):
        t_s3_start = s["s3_start"] * NEURAL_BIN_S
        t_s3_end = s["s3_end"] * NEURAL_BIN_S
        t_seg_end = s["segment_end"] * NEURAL_BIN_S
        ax.axvspan(t_s3_start, t_s3_end, color="green", alpha=0.18,
                    label="S3 stay" if i == 0 else None)
        ax.axvspan(t_s3_end, t_seg_end, color="orange", alpha=0.25,
                    label="post-exit (5s)" if i == 0 else None)
    ax.set_xlim(0, time_v[-1])
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Time (s)")
    ax.set_title(f"S{sn}: Viterbi (color) with Granger analysis segments overlaid "
                 f"(n={len(segments)})")
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(fig_dir / "segments_overlay.png", dpi=130)
    plt.close(fig)

    print(f"  Done S{sn} [{time.time()-t0:.0f}s]", flush=True)
    return df_obs


# ---- Cross-session aggregation ----
def cross_session(per_sess, base_out, base_fig):
    rows = []
    for df in per_sess:
        if df is not None:
            rows.append(df)
    if not rows:
        return None
    cross = pd.concat(rows, ignore_index=True)
    cross.to_csv(base_out / "cross_session_summary.csv", index=False)

    # Asymmetry index per session × signal
    asym_rows = []
    sessions = sorted(cross["session"].unique())
    sess_state_map = {4: "fed", 6: "fed", 8: "fed",
                       12: "fasted", 14: "fasted", 16: "fasted"}
    for sn in sessions:
        for sig in ("pop_sum", "pc1"):
            f_aca = cross[(cross.session == sn) & (cross.signal == sig)
                            & (cross.direction == "ACA->LHA")]["observed_F"]
            f_lha = cross[(cross.session == sn) & (cross.signal == sig)
                            & (cross.direction == "LHA->ACA")]["observed_F"]
            if not len(f_aca) or not len(f_lha):
                continue
            fa = float(f_aca.iloc[0]); fl = float(f_lha.iloc[0])
            denom = fa + fl
            asym = (fa - fl) / denom if abs(denom) > 1e-9 else np.nan
            asym_rows.append(dict(session=sn, state=sess_state_map.get(sn, "?"),
                                    signal=sig, F_ACA_to_LHA=fa,
                                    F_LHA_to_ACA=fl,
                                    asymmetry_index=asym))
    asym = pd.DataFrame(asym_rows)
    asym.to_csv(base_out / "cross_session_asymmetry.csv", index=False)

    # Per-direction replication count
    rep_rows = []
    for sig in ("pop_sum", "pc1"):
        for direction in ("ACA->LHA", "LHA->ACA"):
            sub = cross[(cross.signal == sig) & (cross.direction == direction)]
            n_total = len(sub)
            n_pass_shuf = int(sub["exceeds_p95"].sum())
            n_pass_F = int((sub["F_p_value"] < 0.05).sum())
            rep_rows.append(dict(signal=sig, direction=direction,
                                   n_sessions=n_total,
                                   n_pass_shuffle_p95=n_pass_shuf,
                                   n_pass_F_p_005=n_pass_F))
    pd.DataFrame(rep_rows).to_csv(base_out / "replication_summary.csv", index=False)

    # Asymmetry plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, sig in zip(axes, ("pop_sum", "pc1")):
        sub = asym[asym.signal == sig]
        for _, row in sub.iterrows():
            color = "#4477aa" if row["state"] == "fed" else "#cc6677"
            ax.scatter(row["session"], row["asymmetry_index"], color=color,
                        s=80, edgecolors="black", linewidths=0.5)
            ax.text(row["session"] + 0.1, row["asymmetry_index"],
                     f"S{int(row['session'])}", fontsize=8, va="center")
        ax.axhline(0, color="black", lw=0.7, ls="--")
        ax.set_xticks(sessions)
        ax.set_xticklabels([f"S{s}" for s in sessions])
        ax.set_ylabel("Asymmetry index (F_ACA→LHA − F_LHA→ACA) / sum")
        ax.set_title(f"{sig}: positive = ACA leads, negative = LHA leads")
        ax.grid(alpha=0.3)
    fig.suptitle("Granger asymmetry per session — fed (blue) vs fasted (red)",
                 y=1.0)
    fig.tight_layout()
    fig.savefig(base_fig / "asymmetry_per_session.png", dpi=130)
    plt.close(fig)

    # Replication summary plot
    fig, ax = plt.subplots(figsize=(8, 4.5))
    rep = pd.read_csv(base_out / "replication_summary.csv")
    x = np.arange(len(rep))
    ax.bar(x - 0.2, rep["n_pass_shuffle_p95"], width=0.4,
            color="#cc4444", label="exceeds shuffle p95")
    ax.bar(x + 0.2, rep["n_pass_F_p_005"], width=0.4,
            color="#4477aa", label="F p-value < 0.05")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['signal']}\n{r['direction']}"
                          for _, r in rep.iterrows()], fontsize=9)
    ax.set_ylabel("# sessions passing")
    ax.set_title("Granger replication — n sessions out of 6")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(base_fig / "replication_summary.png", dpi=130)
    plt.close(fig)

    # F comparison pop vs PC1
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, direction in zip(axes, ("ACA->LHA", "LHA->ACA")):
        for sn in sessions:
            f_pop = cross[(cross.session == sn) & (cross.signal == "pop_sum")
                            & (cross.direction == direction)]["observed_F"]
            f_pc = cross[(cross.session == sn) & (cross.signal == "pc1")
                           & (cross.direction == direction)]["observed_F"]
            if not len(f_pop) or not len(f_pc):
                continue
            color = "#4477aa" if sess_state_map.get(sn) == "fed" else "#cc6677"
            ax.scatter(float(f_pop.iloc[0]), float(f_pc.iloc[0]),
                        color=color, s=80, edgecolors="black", linewidths=0.5)
            ax.text(float(f_pop.iloc[0]) + 0.05, float(f_pc.iloc[0]),
                     f"S{sn}", fontsize=8)
        lim = max(cross["observed_F"].max() * 1.05, 1.0)
        ax.plot([0, lim], [0, lim], "k--", lw=0.5)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("F (pop_sum)"); ax.set_ylabel("F (PC1)")
        ax.set_title(direction)
        ax.grid(alpha=0.3)
    fig.suptitle("Granger F: pop_sum vs PC1 (do they agree?)", y=1.0)
    fig.tight_layout()
    fig.savefig(base_fig / "F_comparison_pop_vs_pc1.png", dpi=130)
    plt.close(fig)

    return cross, asym


def sign_test_asymmetry(asym):
    """Per signal, sign test on F_ACA→LHA > F_LHA→ACA across sessions."""
    rows = []
    for sig in ("pop_sum", "pc1"):
        sub = asym[asym.signal == sig].dropna()
        if not len(sub):
            continue
        n_pos = int((sub["asymmetry_index"] > 0).sum())
        n = len(sub)
        # Two-sided binomial test (exact)
        try:
            p = binomtest(n_pos, n, 0.5, alternative="two-sided").pvalue
        except Exception:
            p = np.nan
        rows.append(dict(signal=sig,
                          n_sessions=n,
                          n_ACA_leads=n_pos,
                          n_LHA_leads=n - n_pos,
                          binom_p=p))
    return pd.DataFrame(rows)


# ---- Main ----
def main():
    cfg = load_config()
    base_out, base_fig = out_dirs()
    with open(REPO_ROOT / cfg["paths_yaml"]) as f:
        paths_data = yaml.safe_load(f)
    rng = np.random.default_rng(SHUFFLE_SEED)

    per_sess = []
    for sn in SESSIONS:
        print(f"\n========== Loading S{sn} ==========", flush=True)
        try:
            session = load_session(sn, cfg, paths_data)
        except Exception as e:
            print(f"  ERROR loading S{sn}: {e}", flush=True)
            continue
        df = run_session(session, base_out, base_fig, rng)
        per_sess.append(df)

    print("\n========== Cross-session aggregation ==========", flush=True)
    out = cross_session(per_sess, base_out, base_fig)
    if out is None:
        print("No results."); return
    cross, asym = out
    print("\nReplication:")
    print(pd.read_csv(base_out / "replication_summary.csv").to_string(index=False))
    print("\nAsymmetry per session:")
    print(asym.to_string(index=False))
    sign_df = sign_test_asymmetry(asym)
    print("\nSign test (ACA→LHA > LHA→ACA across sessions):")
    print(sign_df.to_string(index=False))
    sign_df.to_csv(base_out / "sign_test.csv", index=False)
    print(f"\nDone.", flush=True)


if __name__ == "__main__":
    main()
