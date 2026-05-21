"""09b — Manual override of S4 discovery time.

WHY THIS OVERRIDE EXISTS
========================
The merged dynamax HMM (final K=14) misclassifies S4's first feeding bout.
At bins 916-924 (t=439.8-443.6 s, dur 4.3 s) the animal is at the food pot
(P4) and raw EthoVision feeding=1 for all 9 bins. However the Viterbi
decoder assigns these bins to merged S1 — a transition/approach state with
P(feeding) = 0.124 — because the bout is short and zone occupancy is split
(pot_zone 31 %, arena 46 %, other 19 %). As a result, neither the strict
within-window discovery rule nor the closest-food-pot-dig fallback in
script 09 finds this bout as a "feeding state" entry.

Manual inspection of the raw EthoVision feeding label confirms the actual
feeding bout starts at bin 916 (t=439.8 s), preceded by a 5.3-s pure-digging
run at the food pot at bins 885-895 (t=424.9-429.7 s). The dig→feed lag is
~14.9 s (from dig start to feeding onset) — atypical but biologically
reasonable.

This override:
  - Sets S4's discovery_time_s = 439.8 (raw feeding onset, bin 916).
  - Sets discovery_method = "manual_override_raw_feeding".
  - Sets discovery_lag_s = exact lag from dig-run start (bin 885) to feeding
    onset (bin 916), recomputed from the trial_time array.
  - Relabels the dig run at bins 885-895 as "discovery_dig" in the events
    table.
  - Recomputes prior_digs / failed_digs / pot_zone_entries / S4_entries /
    distinct_pots using the new cutoff bin (916) — events between t=439.8 s
    (new discovery) and t=496.5 s (old fallback discovery) are REMOVED from
    the pre-discovery lists.
  - Regenerates sampling_history_summary.png and session_4_events.png.
  - Updates the S4 rows in all_events_combined.csv.

Run AFTER scripts/HMM/09_extract_commitment_markers.py. If 09 is re-run for
any reason this override must be reapplied.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import load_config, REPO_ROOT
# Re-use plotting + helpers from script 09. Importing the module loads
# helpers without running its main().
import importlib.util
spec = importlib.util.spec_from_file_location(
    "step09",
    Path(__file__).resolve().parent / "09_extract_commitment_markers.py",
)
step09 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step09)


# Configuration of the override
S4_OVERRIDE_FEEDING_ONSET_BIN = 916
S4_OVERRIDE_DISCOVERY_DIG_START_BIN = 885   # bins 885-895 = pure-digging at P4
S4_OVERRIDE_METHOD = "manual_override_raw_feeding"


def main():
    cfg = load_config()
    cm_dir = REPO_ROOT / cfg["commitment_dirs"]["out"]
    fig_dir = REPO_ROOT / cfg["commitment_dirs"]["fig"]
    bin_size_s = float(cfg["target_bin_ms"]) / 1000.0
    discovery_window_s = float(cfg["discovery_window_s"])

    # ---- Load current S4 state ----
    history_df = pd.read_csv(cm_dir / "sampling_history.csv")
    s4_old = history_df[history_df.session == 4].iloc[0].to_dict()
    print("=== S4 BEFORE override ===")
    for k, v in s4_old.items():
        print(f"  {k}: {v}")

    # ---- Load merged Viterbi + raw arrays for S4 ----
    posteriors_dir = REPO_ROOT / cfg["merge_dirs"]["posteriors"]
    binned_dir = REPO_ROOT / cfg["out_dirs"]["binned"]
    prepared_dir = REPO_ROOT / cfg["dynamax_dirs"]["prepared"]

    post_df = pd.read_csv(posteriors_dir / "session_4.csv")
    viterbi = post_df["viterbi"].values.astype(np.int64)
    time_s = post_df["time_s"].values.astype(np.float64)
    binned = np.load(binned_dir / "session_4.npz", allow_pickle=True)
    pot_id = np.asarray(binned["pot_id"], dtype=np.int64)
    n = min(len(viterbi), len(pot_id))
    viterbi = viterbi[:n]
    pot_id = pot_id[:n]
    time_s = time_s[:n]

    # ---- Resolve state classes from merged profiles + config ----
    profile_df = pd.read_csv(REPO_ROOT / cfg["merge_dirs"]["state_profiles_csv"])
    state_classes = step09.classify_states(profile_df, cfg)
    feeding = state_classes["feeding"]
    digging = state_classes["digging"]
    contemplation = state_classes["contemplation"]
    pot_zone = state_classes["pot_zone"]

    # ---- Compute the override's discovery anchor + dig→feed lag ----
    onset_bin = S4_OVERRIDE_FEEDING_ONSET_BIN
    dig_start_bin = S4_OVERRIDE_DISCOVERY_DIG_START_BIN
    discovery_time_s = float(time_s[onset_bin])
    dig_start_time_s = float(time_s[dig_start_bin])
    dig_to_feed_lag_s = discovery_time_s - dig_start_time_s
    print(f"\nOverride anchors (recomputed from raw bins):")
    print(f"  feeding onset bin = {onset_bin}, time = {discovery_time_s:.4f} s")
    print(f"  discovery dig start bin = {dig_start_bin}, "
          f"time = {dig_start_time_s:.4f} s")
    print(f"  dig→feed lag (start-to-start) = {dig_to_feed_lag_s:.4f} s")

    # ---- Re-derive S4 events with override cutoff ----
    food_pot = int(cfg["food_pot_per_session"][4])
    cutoff_bin = onset_bin   # all events at bin < cutoff_bin are pre-discovery
    win_bins = max(1, int(round(discovery_window_s / bin_size_s)))
    T = len(viterbi)

    # Dig runs across the whole session
    dig_runs = step09.find_runs(viterbi, digging)
    run_records = []
    for (s, e) in dig_runs:
        pot = step09.majority_pot_id(pot_id[s:e])
        end_check = min(T, e + win_bins)
        led = bool(np.isin(viterbi[s:end_check], feeding).any())  # within-window flag
        run_records.append(dict(
            start_bin=s, end_bin=e,
            duration_bins=e - s,
            duration_s=(e - s) * bin_size_s,
            pot=pot,
            led_to_feeding=led,
        ))

    events = []
    for r in run_records:
        if r["start_bin"] >= cutoff_bin:
            continue
        is_food_pot = (r["pot"] == food_pot)
        if r["start_bin"] == dig_start_bin:
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
        # failed_dig (kept consistent with 09 logic)
        if (not r["led_to_feeding"]) and r["start_bin"] != dig_start_bin:
            events.append(dict(
                bin=r["start_bin"],
                time_s=float(time_s[r["start_bin"]]),
                event_type="failed_dig",
                state_id=int(viterbi[r["start_bin"]]),
                pot_identity=(f"P{r['pot']}" if r["pot"] > 0 else "none"),
                duration_s=r["duration_s"],
            ))

    # Pot-zone entries (pre-discovery)
    pot_zone_entry_bins = step09.find_entries(viterbi, pot_zone)
    for b in pot_zone_entry_bins:
        if b >= cutoff_bin:
            continue
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

    # S4 / contemplation entries (pre-discovery)
    s4_entry_bins = step09.find_entries(viterbi, contemplation)
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
    events_csv = cm_dir / "session_4_events.csv"
    events_df.to_csv(events_csv, index=False)
    print(f"\nUpdated {events_csv} ({len(events_df)} rows)")

    # ---- Recompute sampling-history fields for S4 ----
    prior_digs_all = [r for r in run_records
                      if r["start_bin"] < cutoff_bin
                      and r["start_bin"] != dig_start_bin]
    failed_digs = [r for r in prior_digs_all if not r["led_to_feeding"]]
    pot_zone_entries_pre = [b for b in pot_zone_entry_bins if b < cutoff_bin]
    distinct_pots = set()
    for b in pot_zone_entries_pre:
        if pot_id[b] > 0:
            distinct_pots.add(int(pot_id[b]))
    s4_pre = [b for b in s4_entry_bins if b < cutoff_bin]
    any_prior_run = any(r["start_bin"] < dig_start_bin for r in run_records)
    discovery_was_first_dig = not any_prior_run

    s4_new = dict(
        session=4,
        state="fed",
        food_pot=float(food_pot),
        discovery_bin=int(onset_bin),
        discovery_time_s=float(discovery_time_s),
        discovery_method=S4_OVERRIDE_METHOD,
        discovery_lag_s=float(dig_to_feed_lag_s),
        discovery_within_window=False,
        discovery_failed=False,
        n_prior_pot_digs=len(prior_digs_all),
        n_prior_failed_digs=len(failed_digs),
        n_prior_distinct_pots_visited=len(distinct_pots),
        n_prior_pot_zone_entries=len(pot_zone_entries_pre),
        n_prior_S4_entries=len(s4_pre),
        time_to_discovery_s=float(discovery_time_s),
        discovery_dig_was_first_dig=bool(discovery_was_first_dig),
        n_total_dig_runs=len(run_records),
        session_duration_s=float(time_s[-1]),
    )

    print("\n=== S4 AFTER override ===")
    for k, v in s4_new.items():
        print(f"  {k}: {v}")

    # ---- Persist updated sampling_history.csv ----
    history_df = history_df.set_index("session")
    for k, v in s4_new.items():
        if k == "session":
            continue
        if k not in history_df.columns:
            history_df[k] = pd.Series(dtype=object)
        history_df.at[4, k] = v
    history_df = history_df.reset_index()
    history_csv = cm_dir / "sampling_history.csv"
    history_df.to_csv(history_csv, index=False)
    print(f"\nUpdated {history_csv}")

    # ---- Update all_events_combined.csv ----
    all_csv = cm_dir / "all_events_combined.csv"
    all_df = pd.read_csv(all_csv)
    all_df_other = all_df[all_df.session != 4].copy()
    s4_stamped = events_df.copy()
    s4_stamped.insert(0, "session", 4)
    s4_stamped.insert(1, "state", "fed")
    all_df_new = pd.concat([all_df_other, s4_stamped], ignore_index=True)
    all_df_new = all_df_new.sort_values(["session", "bin"]).reset_index(drop=True)
    all_df_new.to_csv(all_csv, index=False)
    print(f"Updated {all_csv}")

    # ---- Regenerate session_4_events.png ----
    step09.plot_session_events(
        4, "fed", events_df, s4_new,
        float(time_s[-1]),
        fig_dir / "session_4_events.png",
    )
    print(f"Regenerated {fig_dir / 'session_4_events.png'}")

    # ---- Regenerate sampling_history_summary.png ----
    # Use the updated history_df. step09 plot expects DataFrame; reorder rows
    # so the X-axis ordering matches the original session order (fed first).
    fed = cfg["sessions"]["fed"]
    fasted = cfg["sessions"]["fasted"]
    order = fed + fasted
    history_df_plot = history_df.set_index("session").loc[order].reset_index()
    step09.plot_sampling_summary(
        history_df_plot,
        fig_dir / "sampling_history_summary.png",
    )
    print(f"Regenerated {fig_dir / 'sampling_history_summary.png'}")

    # ---- Print BEFORE/AFTER side by side for S4 ----
    print("\n========== BEFORE / AFTER (S4) ==========")
    fields = [
        "discovery_bin", "discovery_time_s", "discovery_method",
        "discovery_lag_s", "discovery_within_window", "discovery_failed",
        "n_prior_pot_digs", "n_prior_failed_digs",
        "n_prior_distinct_pots_visited",
        "n_prior_pot_zone_entries", "n_prior_S4_entries",
        "n_total_dig_runs",
    ]
    print(f"  {'field':<35} {'before':>30}    {'after':>30}")
    for f in fields:
        before = s4_old.get(f, "—")
        after = s4_new.get(f, "—")
        print(f"  {f:<35} {str(before):>30} -> {str(after):>30}")


if __name__ == "__main__":
    main()
