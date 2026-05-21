"""09 — Behavioral commitment-marker extraction (post-merge dynamax).

For each foraging session, identify and timestamp:
  - discovery_dig: first dig at the food pot followed by the pure-feeding state
                   within `discovery_window_s` seconds.
  - prior_dig_food_pot, prior_dig_non_food_pot: dig events before discovery,
                   localised to the food pot or to other pots respectively.
  - pot_zone_entry: pre-discovery transitions INTO any pot-zone state
                   (excluding the pure-digging state itself).
  - S4_entry: pre-discovery transitions INTO any contemplation/transition state.
  - failed_dig: any dig run not followed by feeding within the window
                (regardless of pot).

Compute a per-session sampling-history score at the moment of discovery, and
save per-session event tables, an all-events table, and a sampling-history
summary table.

Inputs:
  - data/HMM/merged_state_profiles_dynamax.csv
  - data/HMM/merged_posteriors_dynamax/session_{N}.csv
  - data/HMM/binned/session_{N}.npz   (provides pot_id, trial_time)
  - cfg["food_pot_per_session"]

Outputs:
  - data/HMM/commitment_markers/session_{N}_events.csv (per session)
  - data/HMM/commitment_markers/sampling_history.csv (one row per session)
  - data/HMM/commitment_markers/all_events_combined.csv
  - data/HMM/commitment_markers/state_classification.csv (which merged states
    were tagged feeding / digging / contemplation / pot-zone)
  - figures/HMM/commitment_markers/session_{N}_events.png
  - figures/HMM/commitment_markers/sampling_history_summary.png

S10 has no food: discovery is null and pot-vs-non-pot dig classification falls
back to all-pots-are-non-food. Sampling-history fields are computed across the
whole session (no truncation point).
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import load_config, ensure_dir, REPO_ROOT


# ---- helpers --------------------------------------------------------------
def find_runs(seq: np.ndarray, target_set):
    """Return a list of (start, end_exclusive) for contiguous runs where
    seq[t] is in target_set. end is exclusive."""
    target_set = set(int(x) for x in target_set)
    in_run = np.array([int(s) in target_set for s in seq])
    if not in_run.any():
        return []
    diff = np.diff(in_run.astype(np.int64), prepend=0, append=0)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def find_entries(seq: np.ndarray, target_set):
    """Return start bins where seq enters a state in target_set
    (i.e., seq[t] in set AND seq[t-1] not in set, with t>=0)."""
    target_set = set(int(x) for x in target_set)
    in_run = np.array([int(s) in target_set for s in seq])
    diff = np.diff(in_run.astype(np.int64), prepend=0)
    return np.flatnonzero(diff == 1).tolist()


def majority_pot_id(pot_id_run: np.ndarray) -> int:
    """Mode of the pot_id over a run, ignoring zeros if any nonzero value exists."""
    if len(pot_id_run) == 0:
        return 0
    nonzero = pot_id_run[pot_id_run > 0]
    if len(nonzero) == 0:
        return 0
    vals, counts = np.unique(nonzero, return_counts=True)
    return int(vals[np.argmax(counts)])


# ---- state classification -------------------------------------------------
def classify_states(profile_df: pd.DataFrame, cfg) -> dict:
    """Return dict of state-class → list of state ids, plus a per-state classification table."""
    feeding_thr = float(cfg["feeding_state_min_prob"])
    digging_thr = float(cfg["digging_state_min_prob"])
    cont_thr = float(cfg["contemplation_event_min_prob"])
    pot_zone_thr = float(cfg["pot_zone_state_min_prob"])

    feeding_states = profile_df.loc[
        profile_df["event_feeding_prob"] >= feeding_thr, "state"
    ].astype(int).tolist()

    digging_states = profile_df.loc[
        profile_df["event_digging_sand_prob"] >= digging_thr, "state"
    ].astype(int).tolist()

    contemplation_states = profile_df.loc[
        (profile_df["event_contemplation_at_transition_prob"] >= cont_thr) |
        (profile_df["event_exploration_at_transition_prob"] >= cont_thr),
        "state"
    ].astype(int).tolist()

    pot_zone_score = (profile_df["zone_pot_prob"]
                       + profile_df["zone_pot_zone_prob"])
    raw_pot_zone_states = profile_df.loc[
        pot_zone_score >= pot_zone_thr, "state"
    ].astype(int).tolist()
    # Exclude pure-digging AND pure-feeding states. Pot-zone entries are meant
    # to capture exploratory pot visits, not the discovery dig or feeding bouts.
    excluded = set(digging_states) | set(feeding_states)
    pot_zone_states = [s for s in raw_pot_zone_states if s not in excluded]

    return dict(
        feeding=feeding_states,
        digging=digging_states,
        contemplation=contemplation_states,
        pot_zone=pot_zone_states,
    )


# ---- per-session extraction ----------------------------------------------
def extract_session(
    sn, state_label, viterbi, time_s, pot_id, food_pot,
    state_classes, discovery_window_s, bin_size_s,
):
    """Run extraction for one session. Returns (events_df, history_dict)."""
    feeding = state_classes["feeding"]
    digging = state_classes["digging"]
    contemplation = state_classes["contemplation"]
    pot_zone = state_classes["pot_zone"]
    win_bins = max(1, int(round(discovery_window_s / bin_size_s)))
    T = len(viterbi)

    # 1) all dig runs
    dig_runs = find_runs(viterbi, digging)

    # 2) per-run: pot identity + feeding-within-window
    run_records = []
    for (s, e) in dig_runs:
        pot = majority_pot_id(pot_id[s:e])
        # Look for any pure-feeding state in [e, e+win_bins)
        end_check = min(T, e + win_bins)
        feeding_after = bool(np.isin(viterbi[e:end_check], feeding).any())
        run_records.append(dict(
            start_bin=s, end_bin=e,  # end exclusive
            duration_bins=e - s,
            duration_s=(e - s) * bin_size_s,
            pot=pot,
            led_to_feeding=feeding_after,
        ))

    # Helper: gap (in bins) from end_bin to next pure-feeding bin (or inf)
    feeding_idx_arr = np.flatnonzero(np.isin(viterbi, list(feeding)))

    def gap_bins_to_next_feeding(end_bin):
        nxt = feeding_idx_arr[feeding_idx_arr >= end_bin]
        if len(nxt) == 0:
            return np.inf
        return int(nxt[0] - end_bin)

    # 3) discovery: first run with pot == food_pot AND led_to_feeding within window.
    # Fallback (used when no clean discovery): the food-pot dig with the smallest
    # gap to the next pure-feeding bin (regardless of window). Marked as such.
    discovery_run = None
    discovery_method = None
    discovery_lag_s = None
    if food_pot is not None:
        # Clean discovery
        for r in run_records:
            if r["pot"] == int(food_pot) and r["led_to_feeding"]:
                discovery_run = r
                discovery_method = "within_window"
                discovery_lag_s = (gap_bins_to_next_feeding(r["end_bin"])
                                    * bin_size_s)
                break
        # Fallback: closest food-pot dig to a pure-feeding state
        if discovery_run is None:
            food_pot_runs = [r for r in run_records if r["pot"] == int(food_pot)]
            best = None; best_gap = np.inf
            for r in food_pot_runs:
                gap = gap_bins_to_next_feeding(r["end_bin"])
                if gap < best_gap:
                    best_gap = gap
                    best = r
            if best is not None and np.isfinite(best_gap):
                discovery_run = best
                discovery_method = "closest_dig_fallback"
                discovery_lag_s = best_gap * bin_size_s
            else:
                discovery_method = "no_dig_at_food_pot"

    # Resolve discovery_bin / time
    if discovery_run is not None:
        discovery_bin = discovery_run["start_bin"]
        discovery_time_s = float(time_s[discovery_bin])
    else:
        discovery_bin = None
        discovery_time_s = None
        if food_pot is None:
            discovery_method = "no_food_session"

    # Flags
    discovery_within_window = (discovery_method == "within_window")
    discovery_failed = (food_pot is not None) and (discovery_run is None)
    if (discovery_method == "closest_dig_fallback") or discovery_failed:
        food_pot_runs = [r for r in run_records if r["pot"] == int(food_pot)]
        n_clean = sum(r["led_to_feeding"] for r in food_pot_runs)
        print(f"    [S{sn} discovery diagnostic] food_pot=P{food_pot}; "
              f"{len(food_pot_runs)} food-pot digs, {n_clean} within "
              f"{discovery_window_s}s window. Method={discovery_method}; "
              f"chosen dig start_bin={discovery_bin}, lag_to_feeding="
              f"{(discovery_lag_s if discovery_lag_s is not None else float('nan')):.2f}s")

    # Truncation point for "prior" events: discovery_bin if defined, else T (whole session)
    cutoff_bin = discovery_bin if discovery_bin is not None else T

    # 4) build per-event rows.  The discovery_dig itself can have start_bin
    # equal to cutoff_bin; allow it through explicitly so it appears in events.
    events = []
    for r in run_records:
        is_discovery = (discovery_bin is not None
                        and r["start_bin"] == discovery_bin)
        if (not is_discovery) and r["start_bin"] >= cutoff_bin:
            continue
        is_food_pot = (food_pot is not None and r["pot"] == int(food_pot))
        if is_discovery:
            ev_type = "discovery_dig"
        elif is_food_pot:
            ev_type = "prior_dig_food_pot"
        else:
            ev_type = "prior_dig_non_food_pot"
        events.append(dict(
            bin=r["start_bin"],
            time_s=float(time_s[r["start_bin"]]),
            event_type=ev_type,
            state_id=int(viterbi[r["start_bin"]]),
            pot_identity=(f"P{r['pot']}" if r["pot"] > 0 else "none"),
            duration_s=r["duration_s"],
        ))
        # failed_dig if it didn't lead to feeding (and not the discovery)
        if (not r["led_to_feeding"]) and r["start_bin"] != discovery_bin:
            events.append(dict(
                bin=r["start_bin"],
                time_s=float(time_s[r["start_bin"]]),
                event_type="failed_dig",
                state_id=int(viterbi[r["start_bin"]]),
                pot_identity=(f"P{r['pot']}" if r["pot"] > 0 else "none"),
                duration_s=r["duration_s"],
            ))

    # 5) pot-zone entries (transitions INTO pot_zone states), pre-discovery
    pot_zone_entry_bins = find_entries(viterbi, pot_zone)
    for b in pot_zone_entry_bins:
        if b >= cutoff_bin:
            continue
        # dwell duration = run length starting at b
        # advance to end of run
        end = b + 1
        while end < T and int(viterbi[end]) in set(pot_zone):
            end += 1
        events.append(dict(
            bin=int(b),
            time_s=float(time_s[b]),
            event_type="pot_zone_entry",
            state_id=int(viterbi[b]),
            pot_identity=(f"P{pot_id[b]}" if pot_id[b] > 0 else "ambiguous"),
            duration_s=(end - b) * bin_size_s,
        ))

    # 6) S4 / contemplation entries pre-discovery
    s4_entry_bins = find_entries(viterbi, contemplation)
    for b in s4_entry_bins:
        if b >= cutoff_bin:
            continue
        end = b + 1
        while end < T and int(viterbi[end]) in set(contemplation):
            end += 1
        events.append(dict(
            bin=int(b),
            time_s=float(time_s[b]),
            event_type="S4_entry",
            state_id=int(viterbi[b]),
            pot_identity="n/a",
            duration_s=(end - b) * bin_size_s,
        ))

    events_df = pd.DataFrame(events).sort_values("bin").reset_index(drop=True)

    # 7) sampling history at moment of discovery (or end-of-session for S10)
    prior_digs_all = [r for r in run_records if r["start_bin"] < cutoff_bin
                      and r["start_bin"] != discovery_bin]
    failed_digs = [r for r in prior_digs_all if not r["led_to_feeding"]]
    pot_zone_entries_pre = [b for b in pot_zone_entry_bins if b < cutoff_bin]
    distinct_pots = set()
    for b in pot_zone_entries_pre:
        if pot_id[b] > 0:
            distinct_pots.add(int(pot_id[b]))
    s4_pre = [b for b in s4_entry_bins if b < cutoff_bin]

    # discovery_dig_was_first_dig: any dig run before discovery (other than itself)?
    if discovery_run is not None:
        any_prior_run = any(r["start_bin"] < discovery_bin for r in run_records)
        discovery_was_first_dig = not any_prior_run
    else:
        discovery_was_first_dig = False  # n/a; no discovery

    history = dict(
        session=sn,
        state=state_label,
        food_pot=food_pot,
        discovery_bin=discovery_bin,
        discovery_time_s=discovery_time_s,
        discovery_method=discovery_method,
        discovery_lag_s=discovery_lag_s,
        discovery_within_window=discovery_within_window,
        discovery_failed=discovery_failed,
        n_prior_pot_digs=len(prior_digs_all),
        n_prior_failed_digs=len(failed_digs),
        n_prior_distinct_pots_visited=len(distinct_pots),
        n_prior_pot_zone_entries=len(pot_zone_entries_pre),
        n_prior_S4_entries=len(s4_pre),
        time_to_discovery_s=(discovery_time_s if discovery_time_s is not None
                              else float(time_s[-1])),
        discovery_dig_was_first_dig=discovery_was_first_dig,
        n_total_dig_runs=len(run_records),
        session_duration_s=float(time_s[-1]),
    )
    return events_df, history


# ---- plots ---------------------------------------------------------------
EVENT_ORDER = [
    "S4_entry",
    "pot_zone_entry",
    "failed_dig",
    "prior_dig_non_food_pot",
    "prior_dig_food_pot",
    "discovery_dig",
]
EVENT_COLOR = {
    "S4_entry": "#7570b3",
    "pot_zone_entry": "#1b9e77",
    "failed_dig": "#bbbbbb",
    "prior_dig_non_food_pot": "#d95f02",
    "prior_dig_food_pot": "#e7298a",
    "discovery_dig": "#e6194b",
}


def plot_session_events(sn, state_label, events_df, history, time_max_s, out_path):
    fig, ax = plt.subplots(figsize=(13, 3.5))
    for i, etype in enumerate(EVENT_ORDER):
        sub = events_df[events_df.event_type == etype]
        if len(sub) == 0:
            continue
        ax.scatter(sub.time_s, np.full(len(sub), i),
                   color=EVENT_COLOR[etype], s=60,
                   label=f"{etype} (n={len(sub)})", edgecolors="black", linewidths=0.4)
    if history["discovery_time_s"] is not None:
        ls = "--" if history.get("discovery_within_window", True) else ":"
        method = history.get("discovery_method", "")
        ax.axvline(history["discovery_time_s"], color="red", lw=1.4, ls=ls,
                   label=f"discovery @ {history['discovery_time_s']:.1f}s ({method})")
    ax.set_yticks(np.arange(len(EVENT_ORDER)))
    ax.set_yticklabels(EVENT_ORDER)
    ax.set_xlabel("Session time (s)")
    food_pot = history["food_pot"]
    method_str = ""
    if history.get("discovery_method") == "closest_dig_fallback":
        method_str = f" [FALLBACK lag={history['discovery_lag_s']:.0f}s]"
    ax.set_title(f"S{sn} ({state_label})  food_pot="
                 f"{('P' + str(food_pot)) if food_pot is not None else 'NONE (extinction)'}{method_str} | "
                 f"prior digs: {history['n_prior_pot_digs']}, "
                 f"distinct pots: {history['n_prior_distinct_pots_visited']}, "
                 f"S4 entries: {history['n_prior_S4_entries']}")
    ax.set_xlim(0, time_max_s)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_sampling_summary(history_df: pd.DataFrame, out_path: Path):
    metrics = [
        ("time_to_discovery_s", "Time to discovery (s)"),
        ("n_prior_pot_digs", "# prior pot digs"),
        ("n_prior_failed_digs", "# prior failed digs"),
        ("n_prior_distinct_pots_visited", "# distinct pots visited"),
        ("n_prior_pot_zone_entries", "# pot-zone entries"),
        ("n_prior_S4_entries", "# S4 (contemplation) entries"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, (m, lab) in zip(axes.flat, metrics):
        for i, row in history_df.iterrows():
            x = i
            y = row[m]
            color = "#4477aa" if row["state"] == "fed" else "#cc6677"
            method = row.get("discovery_method", "")
            if method == "within_window":
                marker = "o"           # clean
            elif method == "closest_dig_fallback":
                marker = "s"           # square = fallback
            elif method == "manual_override_raw_feeding":
                marker = "^"           # triangle = manual override
            elif method == "no_food_session":
                marker = "D"           # diamond = no food
            else:
                marker = "X"           # X = no dig at all
            edge = "black"
            if row.get("discovery_dig_was_first_dig", False):
                edge = "gold"
            ax.scatter(x, y, color=color, s=80, marker=marker,
                       edgecolors=edge, linewidths=1.5,
                       label=row["state"] if i in (0, len(history_df)-1) else None)
            ax.text(x, y, f" S{row['session']}", fontsize=8, va="center")
        ax.set_xticks(range(len(history_df)))
        ax.set_xticklabels([f"S{s}" for s in history_df.session], rotation=0)
        ax.set_ylabel(lab)
        ax.grid(axis="y", alpha=0.3)
    # legend
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#4477aa",
                    markersize=10, label="fed"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#cc6677",
                    markersize=10, label="fasted"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="grey",
                    markersize=10, label="clean discovery (within window)"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="grey",
                    markersize=10, label="fallback discovery (atypical lag)"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="grey",
                    markersize=10, label="manual override (raw feeding)"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="grey",
                    markersize=10, label="no-food session (S10)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Per-session sampling history at discovery", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---- main ----------------------------------------------------------------
def main():
    cfg = load_config()
    out_dir = ensure_dir(REPO_ROOT / cfg["commitment_dirs"]["out"])
    fig_dir = ensure_dir(REPO_ROOT / cfg["commitment_dirs"]["fig"])

    bin_size_s = float(cfg["target_bin_ms"]) / 1000.0
    discovery_window_s = float(cfg["discovery_window_s"])
    food_pot_map = cfg["food_pot_per_session"]

    fed = cfg["sessions"]["fed"]
    fasted = cfg["sessions"]["fasted"]
    all_sessions = fed + fasted

    # Load merged state classification
    profile_df = pd.read_csv(REPO_ROOT / cfg["merge_dirs"]["state_profiles_csv"])
    state_classes = classify_states(profile_df, cfg)

    print("=== State classification ===")
    for cat, ids in state_classes.items():
        print(f"  {cat:>15}: {ids}")
    # Save classification table
    rows = []
    for cat, ids in state_classes.items():
        for s in ids:
            rows.append(dict(category=cat, state_id=int(s)))
    pd.DataFrame(rows).to_csv(out_dir / "state_classification.csv", index=False)

    # Validate that key categories are non-empty
    for cat in ("feeding", "digging", "contemplation"):
        if not state_classes[cat]:
            print(f"WARNING: no state classified as '{cat}' "
                  f"(thresholds in config may need tuning)")

    # Posteriors dir + binned dir
    posteriors_dir = REPO_ROOT / cfg["merge_dirs"]["posteriors"]
    binned_dir = REPO_ROOT / cfg["out_dirs"]["binned"]

    # Per-session extraction
    history_rows = []
    all_events = []
    for sn in all_sessions:
        state_label = "fed" if sn in fed else "fasted"
        food_pot = food_pot_map.get(sn, food_pot_map.get(int(sn)))

        # Load merged Viterbi + time
        post_path = posteriors_dir / f"session_{sn}.csv"
        if not post_path.exists():
            print(f"  SKIP S{sn}: no merged posteriors at {post_path}")
            continue
        df = pd.read_csv(post_path)
        viterbi = df["viterbi"].values.astype(np.int64)
        time_s_post = df["time_s"].values.astype(np.float64)

        # Load pot_id from binned npz
        binned = np.load(binned_dir / f"session_{sn}.npz", allow_pickle=True)
        pot_id = np.asarray(binned["pot_id"], dtype=np.int64)
        time_s_bin = np.asarray(binned["trial_time"], dtype=np.float64)
        if len(pot_id) != len(viterbi):
            n = min(len(pot_id), len(viterbi))
            print(f"  S{sn}: aligning lengths "
                  f"(viterbi={len(viterbi)}, pot_id={len(pot_id)} → {n})")
            viterbi = viterbi[:n]
            pot_id = pot_id[:n]
            time_s_post = time_s_post[:n]

        events_df, history = extract_session(
            sn, state_label, viterbi, time_s_post, pot_id,
            food_pot, state_classes, discovery_window_s, bin_size_s,
        )
        events_df.to_csv(out_dir / f"session_{sn}_events.csv", index=False)
        # Stamp session id then append
        events_df_stamped = events_df.copy()
        events_df_stamped.insert(0, "session", sn)
        events_df_stamped.insert(1, "state", state_label)
        all_events.append(events_df_stamped)

        plot_session_events(
            sn, state_label, events_df, history, float(time_s_post[-1]),
            fig_dir / f"session_{sn}_events.png",
        )

        history_rows.append(history)

        food_str = f"P{food_pot}" if food_pot is not None else "NONE"
        if history["discovery_time_s"] is None:
            disc_str = "no discovery"
        else:
            method = history["discovery_method"]
            tag = "" if method == "within_window" else f" [{method} lag={history['discovery_lag_s']:.0f}s]"
            disc_str = f"@ {history['discovery_time_s']:.1f}s{tag}"
        print(f"  S{sn} ({state_label}, food={food_str}): {disc_str}, "
              f"prior_digs={history['n_prior_pot_digs']}, "
              f"distinct_pots={history['n_prior_distinct_pots_visited']}, "
              f"S4_entries={history['n_prior_S4_entries']}, "
              f"first_dig={history['discovery_dig_was_first_dig']}")

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(out_dir / "sampling_history.csv", index=False)
    print(f"\nSampling history → {out_dir / 'sampling_history.csv'}")

    if all_events:
        all_events_df = pd.concat(all_events, ignore_index=True)
        all_events_df.to_csv(out_dir / "all_events_combined.csv", index=False)
        print(f"All events combined → {out_dir / 'all_events_combined.csv'}")

    plot_sampling_summary(history_df, fig_dir / "sampling_history_summary.png")
    print(f"Summary figure → {fig_dir / 'sampling_history_summary.png'}")

    # ---- final summary ----
    print("\n========== SUMMARY ==========")
    for _, h in history_df.iterrows():
        flags = []
        method = h.get("discovery_method", "")
        if method == "closest_dig_fallback":
            flags.append(f"FALLBACK (lag={h['discovery_lag_s']:.0f}s)")
        elif h["discovery_failed"]:
            flags.append("DISCOVERY-FAILED")
        if h.get("discovery_dig_was_first_dig", False):
            flags.append("STUMBLED (first dig)")
        if h["food_pot"] is None or pd.isna(h["food_pot"]):
            flags.append("no-food-session")
        flag_str = f"  [{', '.join(flags)}]" if flags else ""
        if h["discovery_time_s"] is None or pd.isna(h["discovery_time_s"]):
            t = "n/a"
        else:
            t = f"{h['discovery_time_s']:.1f}s"
        print(f"  S{h['session']:>2} ({h['state']:>6}, food="
              f"{('P' + str(int(h['food_pot']))) if not pd.isna(h['food_pot']) else 'none':>4}): "
              f"discovery={t:>9}, prior_digs={h['n_prior_pot_digs']:>2}, "
              f"distinct_pots={h['n_prior_distinct_pots_visited']:>1}, "
              f"S4={h['n_prior_S4_entries']:>2}{flag_str}")


if __name__ == "__main__":
    main()
