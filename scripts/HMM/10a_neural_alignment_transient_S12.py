"""10a — Track A: event-locked transient neural analyses for Session 12.

Session 12 (fasted, food at P4, discovery at t=597.12 s) is the first pass at
neural alignment for the dual-probe foraging HMM project. ACA (probe 0) and
LHA (probe 1, depth 0-345 um) are processed separately and reported side by
side.

Three analyses, each with a ±1 s window around the event onset, 100 ms bins:
  A1: discovery dig (n=1) vs failed digs at food pot (n=2)
  A2: pre-discovery S4 entries (n=15) vs post-discovery S4 entries (Viterbi)
  A3: pre-discovery pot-zone entries vs post-discovery (Viterbi)

Per-unit PETHs and population (sum-across-units) PETHs are saved as figures.
A summary CSV per analysis lists per-unit mean FR pre/post + peak-diff.

Sample sizes are small (1 vs 2 for A1) so no formal stats — descriptive plots
only. Track B (state-conditioned analyses) lives in a separate script.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

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
SESSION_NUM = 12
NEURAL_BIN_S = 0.1            # 100 ms
WINDOW_PRE_S = 1.0
WINDOW_POST_S = 1.0
SMOOTH_SIGMA_BINS = 0.5       # 50 ms gaussian, for visualization only
LOW_FR_THRESHOLD_HZ = 0.5     # flag warning below this in pre-event window

CONTEMPLATION_STATE = 4       # merged S4
POT_ZONE_STATES = {8, 9, 10, 13}

# Discovery / merged-state class membership (matches scripts/HMM/09).
FEEDING_STATES = {2, 11}
DIGGING_STATES = {6}


# ---- Helpers ----
def bin_window_indices(t_event_s, n_bins):
    """Return start, stop indices in the (n_bins,) time series for the ±W window
    centered at t_event_s. Returns None if window falls outside the recording.
    """
    center = int(round(t_event_s / NEURAL_BIN_S))
    pre_b = int(round(WINDOW_PRE_S / NEURAL_BIN_S))
    post_b = int(round(WINDOW_POST_S / NEURAL_BIN_S))
    start = center - pre_b
    stop = center + post_b + 1   # inclusive of +1.0s
    if start < 0 or stop > n_bins:
        return None
    return start, stop


def peth_per_unit(rates, events_t, n_bins):
    """Per-unit average firing rate across events.

    rates: (n_units, n_bins) spike counts per 100 ms bin.
    Returns:
      peth: (n_units, win_len) MEAN COUNTS per bin (divide by NEURAL_BIN_S → Hz)
      n_valid: number of events that fit fully in the recording
    """
    pre_b = int(round(WINDOW_PRE_S / NEURAL_BIN_S))
    post_b = int(round(WINDOW_POST_S / NEURAL_BIN_S))
    win_len = pre_b + post_b + 1
    n_units = rates.shape[0]
    accum = np.zeros((n_units, win_len), dtype=np.float64)
    n_valid = 0
    for t in events_t:
        idx = bin_window_indices(t, n_bins)
        if idx is None:
            continue
        s, e = idx
        accum += rates[:, s:e]
        n_valid += 1
    if n_valid == 0:
        return np.full((n_units, win_len), np.nan), 0
    return accum / n_valid, n_valid


def peth_population(rates, events_t, n_bins):
    """Population (sum-across-units) average across events. Returns (win_len,)."""
    pre_b = int(round(WINDOW_PRE_S / NEURAL_BIN_S))
    post_b = int(round(WINDOW_POST_S / NEURAL_BIN_S))
    win_len = pre_b + post_b + 1
    pop = rates.sum(axis=0)
    accum = np.zeros(win_len)
    n_valid = 0
    for t in events_t:
        idx = bin_window_indices(t, n_bins)
        if idx is None:
            continue
        s, e = idx
        accum += pop[s:e]
        n_valid += 1
    if n_valid == 0:
        return np.full(win_len, np.nan), 0
    return accum / n_valid, n_valid


def make_time_vec():
    pre_b = int(round(WINDOW_PRE_S / NEURAL_BIN_S))
    post_b = int(round(WINDOW_POST_S / NEURAL_BIN_S))
    return np.arange(-pre_b, post_b + 1) * NEURAL_BIN_S


def plot_per_unit_peths(peth_a, peth_b, label_a, label_b, n_a, n_b,
                         region, time_vec, out_path):
    """Per-unit PETH grid; condition A red, condition B grey. Y axis = Hz."""
    n_units = peth_a.shape[0]
    if n_units == 0:
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.text(0.5, 0.5, f"No units in {region}", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        return
    n_cols = 6
    n_rows = (n_units + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(2.4 * n_cols, 1.7 * n_rows + 0.4),
                              sharex=True)
    axes_flat = np.atleast_1d(axes).flatten()
    # Convert mean counts → Hz
    peth_a_hz = peth_a / NEURAL_BIN_S
    peth_b_hz = peth_b / NEURAL_BIN_S
    for u in range(n_units):
        ax = axes_flat[u]
        a = gaussian_filter1d(peth_a_hz[u], sigma=SMOOTH_SIGMA_BINS)
        b = gaussian_filter1d(peth_b_hz[u], sigma=SMOOTH_SIGMA_BINS)
        ax.plot(time_vec, a, color="#cc0000", lw=1.4,
                 label=f"{label_a} (n={n_a})")
        ax.plot(time_vec, b, color="#666666", lw=1.4,
                 label=f"{label_b} (n={n_b})")
        ax.axvline(0, color="black", lw=0.7, ls="--")
        ax.set_title(f"u{u}", fontsize=8)
        ax.tick_params(labelsize=7)
        if u == 0:
            ax.legend(fontsize=7, frameon=False, loc="upper right")
    for u in range(n_units, len(axes_flat)):
        axes_flat[u].axis("off")
    fig.suptitle(f"{region}: per-unit PETH ({label_a} vs {label_b}) — S12",
                 y=0.995, fontsize=11)
    fig.supxlabel("time from event (s)", fontsize=10)
    fig.supylabel("firing rate (Hz)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_population_peth(pop_a_aca, pop_b_aca, pop_a_lha, pop_b_lha,
                          n_a, n_b, label_a, label_b,
                          n_units_aca, n_units_lha,
                          time_vec, out_path):
    """One panel per region (ACA, LHA), both conditions overlaid."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for ax, pop_a, pop_b, region, n_units in [
        (axes[0], pop_a_aca, pop_b_aca, "ACA", n_units_aca),
        (axes[1], pop_a_lha, pop_b_lha, "LHA", n_units_lha),
    ]:
        # Convert summed counts → population firing rate (Hz total) per bin
        a_hz = pop_a / NEURAL_BIN_S
        b_hz = pop_b / NEURAL_BIN_S
        a_smooth = gaussian_filter1d(a_hz, sigma=SMOOTH_SIGMA_BINS)
        b_smooth = gaussian_filter1d(b_hz, sigma=SMOOTH_SIGMA_BINS)
        ax.plot(time_vec, a_smooth, color="#cc0000", lw=2,
                 label=f"{label_a} (n={n_a})")
        ax.plot(time_vec, b_smooth, color="#666666", lw=2,
                 label=f"{label_b} (n={n_b})")
        ax.axvline(0, color="black", lw=0.8, ls="--")
        ax.set_title(f"{region} population (n_units={n_units})")
        ax.set_xlabel("time from event (s)")
        ax.set_ylabel("population FR (Hz, summed across units)")
        ax.legend(fontsize=9, frameon=False)
        ax.grid(alpha=0.3)
    fig.suptitle(f"S12 population PETH — {label_a} vs {label_b}", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def per_unit_summary(peth_a, peth_b, region):
    """Return DataFrame with one row per unit:
      mean_FR_a_pre, mean_FR_a_post, mean_FR_b_pre, mean_FR_b_post,
      peak_diff_bin, peak_diff_value
    All firing rates in Hz.
    """
    pre_b = int(round(WINDOW_PRE_S / NEURAL_BIN_S))
    post_b = int(round(WINDOW_POST_S / NEURAL_BIN_S))
    # Convert mean counts → Hz
    a_hz = peth_a / NEURAL_BIN_S
    b_hz = peth_b / NEURAL_BIN_S
    rows = []
    n_units = peth_a.shape[0]
    time_vec = make_time_vec()
    for u in range(n_units):
        a = a_hz[u]; b = b_hz[u]
        diff = a - b
        if np.all(np.isnan(diff)):
            peak_idx = int(pre_b)
            peak_val = np.nan
        else:
            peak_idx = int(np.argmax(np.abs(diff)))
            peak_val = float(diff[peak_idx])
        rows.append(dict(
            unit_id=u,
            region=region,
            mean_FR_a_pre=float(np.mean(a[:pre_b])),         # bins 0..pre_b-1 = -1.0..-0.1s
            mean_FR_a_post=float(np.mean(a[pre_b + 1:])),    # bins pre_b+1..end = +0.1..+1.0s
            mean_FR_b_pre=float(np.mean(b[:pre_b])),
            mean_FR_b_post=float(np.mean(b[pre_b + 1:])),
            peak_diff_bin=peak_idx,
            peak_diff_time_s=float(time_vec[peak_idx]),
            peak_diff_value=peak_val,
        ))
    return pd.DataFrame(rows)


# ---- Main ----
def main():
    cfg = load_config()
    bin_size_behav_s = float(cfg["target_bin_ms"]) / 1000.0

    # Resolve session-12 sorted dirs
    with open(REPO_ROOT / cfg["paths_yaml"]) as f:
        paths_data = yaml.safe_load(f)
    s12_paths = (paths_data["double_probe"]["coordinates_1"]["mouse01"]
                  ["sessions"][f"session_{SESSION_NUM}"])
    aca_sorted = Path(s12_paths["probe_0_aca"]["sorted"])
    lha_sorted = Path(s12_paths["probe_1_lha_rsp"]["sorted"])

    # Output dirs
    out_dir = REPO_ROOT / "data" / "HMM" / "neural_alignment" / f"transient_S{SESSION_NUM}"
    fig_dir = REPO_ROOT / "figures" / "HMM" / "neural_alignment" / f"transient_S{SESSION_NUM}"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ----- Load good units + spikes -----
    print(f"=== S{SESSION_NUM} neural-data load ===")
    aca_uids_all = get_good_units_p0(aca_sorted)
    lha_uids_all = get_good_units_p1_lha(lha_sorted)
    print(f"  ACA QC-good units: {len(aca_uids_all)}")
    print(f"  LHA QC-good units (depth 0-345 um): {len(lha_uids_all)}")

    aca_sorting = se.read_kilosort(aca_sorted)
    lha_sorting = se.read_kilosort(lha_sorted)
    aca_avail = set(aca_sorting.get_unit_ids())
    lha_avail = set(lha_sorting.get_unit_ids())
    aca_uids = [u for u in aca_uids_all if u in aca_avail]
    lha_uids = [u for u in lha_uids_all if u in lha_avail]
    print(f"  Units present in sorting → ACA: {len(aca_uids)}, LHA: {len(lha_uids)}")

    aca_spikes = load_spike_times_for_region(aca_sorting, aca_uids)
    lha_spikes = load_spike_times_for_region(lha_sorting, lha_uids)

    # ----- Determine session duration -----
    binned = np.load(REPO_ROOT / cfg["out_dirs"]["binned"]
                      / f"session_{SESSION_NUM}.npz", allow_pickle=True)
    trial_time = np.asarray(binned["trial_time"], dtype=np.float64)
    duration_s = float(trial_time[-1] + bin_size_behav_s)
    # Guard against spike times exceeding behavior end
    max_spike = max(
        max((t.max() for t in aca_spikes.values() if len(t)), default=0),
        max((t.max() for t in lha_spikes.values() if len(t)), default=0),
    )
    duration_s = max(duration_s, max_spike + NEURAL_BIN_S)
    n_bins = int(np.ceil(duration_s / NEURAL_BIN_S))
    bin_edges = np.arange(n_bins + 1) * NEURAL_BIN_S
    print(f"  Session duration: {duration_s:.1f} s → {n_bins} bins of {NEURAL_BIN_S*1000:.0f} ms")

    # ----- Bin spikes at 100ms (counts per bin) -----
    aca_uid_list = sorted(aca_spikes.keys())
    lha_uid_list = sorted(lha_spikes.keys())
    aca_rates = np.zeros((len(aca_uid_list), n_bins), dtype=np.float64)
    lha_rates = np.zeros((len(lha_uid_list), n_bins), dtype=np.float64)
    for i, uid in enumerate(aca_uid_list):
        aca_rates[i] = np.histogram(aca_spikes[uid], bin_edges)[0]
    for i, uid in enumerate(lha_uid_list):
        lha_rates[i] = np.histogram(lha_spikes[uid], bin_edges)[0]

    # Mean firing rate per unit (Hz) for low-FR flagging
    aca_mean_fr = aca_rates.mean(axis=1) / NEURAL_BIN_S
    lha_mean_fr = lha_rates.mean(axis=1) / NEURAL_BIN_S
    aca_low = np.where(aca_mean_fr < LOW_FR_THRESHOLD_HZ)[0]
    lha_low = np.where(lha_mean_fr < LOW_FR_THRESHOLD_HZ)[0]
    print(f"  ACA mean FR range: [{aca_mean_fr.min():.2f}, {aca_mean_fr.max():.2f}] Hz; "
          f"low-FR (<{LOW_FR_THRESHOLD_HZ} Hz): {len(aca_low)}/{len(aca_uid_list)}")
    print(f"  LHA mean FR range: [{lha_mean_fr.min():.2f}, {lha_mean_fr.max():.2f}] Hz; "
          f"low-FR (<{LOW_FR_THRESHOLD_HZ} Hz): {len(lha_low)}/{len(lha_uid_list)}")

    # ----- Load events + history + Viterbi -----
    cm_dir = REPO_ROOT / cfg["commitment_dirs"]["out"]
    events_df = pd.read_csv(cm_dir / f"session_{SESSION_NUM}_events.csv")
    history_df = pd.read_csv(cm_dir / "sampling_history.csv")
    history = history_df[history_df.session == SESSION_NUM].iloc[0]
    discovery_bin = int(history["discovery_bin"])
    discovery_time_s = float(history["discovery_time_s"])
    print(f"\n  S{SESSION_NUM} discovery: bin={discovery_bin}, t={discovery_time_s:.2f}s")

    # Merged Viterbi (480 ms behavior bins) for re-deriving post-discovery events
    post_df = pd.read_csv(REPO_ROOT / cfg["merge_dirs"]["posteriors"]
                           / f"session_{SESSION_NUM}.csv")
    viterbi = post_df["viterbi"].values.astype(np.int64)
    behav_t = post_df["time_s"].values.astype(np.float64)

    # Helper: transitions INTO state-set after a given bin
    def post_entries_into(states_set, after_bin):
        in_set = np.isin(viterbi, list(states_set)).astype(np.int64)
        diff = np.diff(in_set, prepend=0)
        bins = np.flatnonzero(diff == 1)
        bins = bins[bins > after_bin]
        return behav_t[bins]

    # ----- Event sets -----
    discovery_t = events_df.loc[events_df.event_type == "discovery_dig",
                                 "time_s"].values
    failed_food_t = events_df.loc[events_df.event_type == "prior_dig_food_pot",
                                   "time_s"].values
    pre_s4_t = events_df.loc[events_df.event_type == "S4_entry", "time_s"].values
    pre_pz_t = events_df.loc[events_df.event_type == "pot_zone_entry",
                              "time_s"].values
    post_s4_t = post_entries_into({CONTEMPLATION_STATE}, discovery_bin)
    post_pz_t = post_entries_into(POT_ZONE_STATES, discovery_bin)

    print(f"\n  Event counts:")
    print(f"    A1 discovery_dig: {len(discovery_t)}, "
          f"failed_food_pot: {len(failed_food_t)}")
    print(f"    A2 pre_S4: {len(pre_s4_t)}, post_S4: {len(post_s4_t)}")
    print(f"    A3 pre_pot_zone: {len(pre_pz_t)}, post_pot_zone: {len(post_pz_t)}")

    time_vec = make_time_vec()

    # ----- A1: discovery vs failed digs at food pot -----
    print("\n=== A1: discovery dig vs failed food-pot digs ===")
    aca_disc, n_aca_disc = peth_per_unit(aca_rates, discovery_t, n_bins)
    aca_fail, n_aca_fail = peth_per_unit(aca_rates, failed_food_t, n_bins)
    lha_disc, n_lha_disc = peth_per_unit(lha_rates, discovery_t, n_bins)
    lha_fail, n_lha_fail = peth_per_unit(lha_rates, failed_food_t, n_bins)
    print(f"  ACA events used: discovery={n_aca_disc}, failed={n_aca_fail}")
    print(f"  LHA events used: discovery={n_lha_disc}, failed={n_lha_fail}")
    plot_per_unit_peths(aca_disc, aca_fail, "discovery", "failed",
                         n_aca_disc, n_aca_fail, "ACA",
                         time_vec, fig_dir / "A1_per_unit_PETH_ACA.png")
    plot_per_unit_peths(lha_disc, lha_fail, "discovery", "failed",
                         n_lha_disc, n_lha_fail, "LHA",
                         time_vec, fig_dir / "A1_per_unit_PETH_LHA.png")
    pop_aca_disc, _ = peth_population(aca_rates, discovery_t, n_bins)
    pop_aca_fail, _ = peth_population(aca_rates, failed_food_t, n_bins)
    pop_lha_disc, _ = peth_population(lha_rates, discovery_t, n_bins)
    pop_lha_fail, _ = peth_population(lha_rates, failed_food_t, n_bins)
    plot_population_peth(pop_aca_disc, pop_aca_fail, pop_lha_disc, pop_lha_fail,
                          n_aca_disc, n_aca_fail, "discovery", "failed",
                          len(aca_uid_list), len(lha_uid_list),
                          time_vec, fig_dir / "A1_population_PETH.png")
    summary = pd.concat([
        per_unit_summary(aca_disc, aca_fail, "ACA")
            .rename(columns={"mean_FR_a_pre": "mean_FR_discovery_pre",
                              "mean_FR_a_post": "mean_FR_discovery_post",
                              "mean_FR_b_pre": "mean_FR_failed_pre",
                              "mean_FR_b_post": "mean_FR_failed_post"}),
        per_unit_summary(lha_disc, lha_fail, "LHA")
            .rename(columns={"mean_FR_a_pre": "mean_FR_discovery_pre",
                              "mean_FR_a_post": "mean_FR_discovery_post",
                              "mean_FR_b_pre": "mean_FR_failed_pre",
                              "mean_FR_b_post": "mean_FR_failed_post"}),
    ], ignore_index=True)
    summary.to_csv(out_dir / "A1_summary.csv", index=False)
    print(f"  → {fig_dir / 'A1_per_unit_PETH_ACA.png'}, _LHA, "
          f"_population, {out_dir / 'A1_summary.csv'}")

    # ----- A2: pre-discovery S4 vs post-discovery S4 -----
    print("\n=== A2: pre-discovery S4 entries vs post-discovery S4 entries ===")
    aca_pre_s4, n_aca_pre = peth_per_unit(aca_rates, pre_s4_t, n_bins)
    aca_post_s4, n_aca_post = peth_per_unit(aca_rates, post_s4_t, n_bins)
    lha_pre_s4, n_lha_pre = peth_per_unit(lha_rates, pre_s4_t, n_bins)
    lha_post_s4, n_lha_post = peth_per_unit(lha_rates, post_s4_t, n_bins)
    print(f"  ACA events used: pre={n_aca_pre}, post={n_aca_post}")
    print(f"  LHA events used: pre={n_lha_pre}, post={n_lha_post}")
    plot_per_unit_peths(aca_pre_s4, aca_post_s4, "pre", "post",
                         n_aca_pre, n_aca_post, "ACA",
                         time_vec, fig_dir / "A2_per_unit_PETH_ACA.png")
    plot_per_unit_peths(lha_pre_s4, lha_post_s4, "pre", "post",
                         n_lha_pre, n_lha_post, "LHA",
                         time_vec, fig_dir / "A2_per_unit_PETH_LHA.png")
    pop_aca_pre, _ = peth_population(aca_rates, pre_s4_t, n_bins)
    pop_aca_post, _ = peth_population(aca_rates, post_s4_t, n_bins)
    pop_lha_pre, _ = peth_population(lha_rates, pre_s4_t, n_bins)
    pop_lha_post, _ = peth_population(lha_rates, post_s4_t, n_bins)
    plot_population_peth(pop_aca_pre, pop_aca_post, pop_lha_pre, pop_lha_post,
                          n_aca_pre, n_aca_post, "pre-discovery", "post-discovery",
                          len(aca_uid_list), len(lha_uid_list),
                          time_vec, fig_dir / "A2_population_PETH.png")
    summary = pd.concat([
        per_unit_summary(aca_pre_s4, aca_post_s4, "ACA")
            .rename(columns={"mean_FR_a_pre": "mean_FR_pre_pre",
                              "mean_FR_a_post": "mean_FR_pre_post",
                              "mean_FR_b_pre": "mean_FR_post_pre",
                              "mean_FR_b_post": "mean_FR_post_post"}),
        per_unit_summary(lha_pre_s4, lha_post_s4, "LHA")
            .rename(columns={"mean_FR_a_pre": "mean_FR_pre_pre",
                              "mean_FR_a_post": "mean_FR_pre_post",
                              "mean_FR_b_pre": "mean_FR_post_pre",
                              "mean_FR_b_post": "mean_FR_post_post"}),
    ], ignore_index=True)
    summary.to_csv(out_dir / "A2_summary.csv", index=False)
    print(f"  → {out_dir / 'A2_summary.csv'}")

    # ----- A3: pre-discovery pot-zone vs post-discovery pot-zone -----
    print("\n=== A3: pre-discovery pot-zone entries vs post-discovery pot-zone entries ===")
    aca_pre_pz, n_aca_pre = peth_per_unit(aca_rates, pre_pz_t, n_bins)
    aca_post_pz, n_aca_post = peth_per_unit(aca_rates, post_pz_t, n_bins)
    lha_pre_pz, n_lha_pre = peth_per_unit(lha_rates, pre_pz_t, n_bins)
    lha_post_pz, n_lha_post = peth_per_unit(lha_rates, post_pz_t, n_bins)
    print(f"  ACA events used: pre={n_aca_pre}, post={n_aca_post}")
    print(f"  LHA events used: pre={n_lha_pre}, post={n_lha_post}")
    plot_per_unit_peths(aca_pre_pz, aca_post_pz, "pre", "post",
                         n_aca_pre, n_aca_post, "ACA",
                         time_vec, fig_dir / "A3_per_unit_PETH_ACA.png")
    plot_per_unit_peths(lha_pre_pz, lha_post_pz, "pre", "post",
                         n_lha_pre, n_lha_post, "LHA",
                         time_vec, fig_dir / "A3_per_unit_PETH_LHA.png")
    pop_aca_pre, _ = peth_population(aca_rates, pre_pz_t, n_bins)
    pop_aca_post, _ = peth_population(aca_rates, post_pz_t, n_bins)
    pop_lha_pre, _ = peth_population(lha_rates, pre_pz_t, n_bins)
    pop_lha_post, _ = peth_population(lha_rates, post_pz_t, n_bins)
    plot_population_peth(pop_aca_pre, pop_aca_post, pop_lha_pre, pop_lha_post,
                          n_aca_pre, n_aca_post, "pre-discovery", "post-discovery",
                          len(aca_uid_list), len(lha_uid_list),
                          time_vec, fig_dir / "A3_population_PETH.png")
    summary = pd.concat([
        per_unit_summary(aca_pre_pz, aca_post_pz, "ACA")
            .rename(columns={"mean_FR_a_pre": "mean_FR_pre_pre",
                              "mean_FR_a_post": "mean_FR_pre_post",
                              "mean_FR_b_pre": "mean_FR_post_pre",
                              "mean_FR_b_post": "mean_FR_post_post"}),
        per_unit_summary(lha_pre_pz, lha_post_pz, "LHA")
            .rename(columns={"mean_FR_a_pre": "mean_FR_pre_pre",
                              "mean_FR_a_post": "mean_FR_pre_post",
                              "mean_FR_b_pre": "mean_FR_post_pre",
                              "mean_FR_b_post": "mean_FR_post_post"}),
    ], ignore_index=True)
    summary.to_csv(out_dir / "A3_summary.csv", index=False)
    print(f"  → {out_dir / 'A3_summary.csv'}")

    # ---- final flags ----
    print("\n=== Low-FR unit flags ===")
    if len(aca_low):
        print(f"  ACA low-FR units (mean FR < {LOW_FR_THRESHOLD_HZ} Hz): "
              f"{aca_low.tolist()} (FR: "
              f"{[f'{aca_mean_fr[i]:.2f}' for i in aca_low]})")
    if len(lha_low):
        print(f"  LHA low-FR units (mean FR < {LOW_FR_THRESHOLD_HZ} Hz): "
              f"{lha_low.tolist()} (FR: "
              f"{[f'{lha_mean_fr[i]:.2f}' for i in lha_low]})")
    if not (len(aca_low) or len(lha_low)):
        print(f"  (none below {LOW_FR_THRESHOLD_HZ} Hz)")

    print(f"\nDone. Outputs in {out_dir} and {fig_dir}")


if __name__ == "__main__":
    main()
