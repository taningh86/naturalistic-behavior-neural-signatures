"""01 — Load each foraging-session xlsx, rebin from 40ms → 480ms.

Output: data/HMM/binned/session_{N}.npz with arrays
  speed                (T,)        — mean across 12 sub-bins
  distance_to_pot      (T,)        — row-wise min over 4 pot distances, then mean
  zone                 (T,)  int   — mode of priority-mapped zone label
  pot_id               (T,)  int   — mode of per-bin pot identity (0=none, 1..4)
  events               (T, n_ev)   — mode of each Bernoulli event column
  zone_labels          list[str]   — index → name
  event_names          list[str]   — column index → canonical event name
  trial_time           (T,)        — start time of each rebinned bin (s)
  meta                 dict        — session_num, state, n_raw_bins, n_bins, etc.

pot_id semantics (used downstream by script 09 for commitment-marker extraction):
  - 0 = not at any pot or pot zone
  - 1..4 = at Pot-{i}, derived from the Pot-i and Pot-i-zone columns.
  - Priority: pot[i] (close-to-pot) takes precedence over pot_zone[i] (broader zone).
  - If multiple pots are simultaneously active in the raw 40ms bin, the smallest
    pot index wins (deterministic). Mode-binned to 480ms.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import mode

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import (
    load_config, load_paths_yaml, session_xlsx, session_list,
    ensure_dir, REPO_ROOT,
)


def block_mean(x, factor):
    """Mean over non-overlapping blocks of `factor` samples.

    Truncates trailing partial block.
    """
    n = (len(x) // factor) * factor
    return x[:n].reshape(-1, factor).mean(axis=1)


def block_mode(x, factor):
    """Per-block mode (most common value)."""
    n = (len(x) // factor) * factor
    blocks = x[:n].reshape(-1, factor)
    out = np.empty(blocks.shape[0], dtype=np.int64)
    for i, blk in enumerate(blocks):
        m = mode(blk, keepdims=False)
        out[i] = int(m.mode)
    return out


def assign_zone(df, zone_priority):
    """Map binary zone columns into a single int label per raw bin.

    Returns (zone_int, labels) where zone_int is shape (T_raw,) and labels is
    a list mapping label index -> name.
    """
    labels = list(zone_priority.keys())  # ordered by priority
    label_to_idx = {name: i for i, name in enumerate(labels)}
    T = len(df)
    out = np.full(T, label_to_idx["other"], dtype=np.int64)
    assigned = np.zeros(T, dtype=bool)
    for name in labels:
        cols = zone_priority[name]
        if not cols:
            continue
        # missing columns → silently treat as all-zero
        present = [c for c in cols if c in df.columns]
        if not present:
            continue
        # Coerce to numeric — EthoVision uses '-' for missing in some columns.
        sub = df[present].apply(coerce_numeric).fillna(0).values
        any_active = (sub > 0).any(axis=1)
        new_assign = any_active & ~assigned
        out[new_assign] = label_to_idx[name]
        assigned |= new_assign
    return out, labels


def coerce_numeric(s):
    return pd.to_numeric(s, errors="coerce")


def assign_pot_id(df, zone_priority):
    """Per-bin pot identity from the Pot-i and Pot-i-zone columns.

    Returns int array (T_raw,) where 0 = not at any pot, 1-4 = at pot N.
    Pot ('close to pot') has priority over pot_zone (broader area). Within a
    group, the smallest pot index wins on ties.
    """
    T = len(df)
    pot_id = np.zeros(T, dtype=np.int64)

    # First: 'pot' columns (close-to-pot)
    for i, col in enumerate(zone_priority.get("pot", [])):
        if col in df.columns:
            active = (coerce_numeric(df[col]).fillna(0).values > 0) & (pot_id == 0)
            pot_id[active] = i + 1

    # Then: 'pot_zone' columns (broader zone) for bins not yet assigned
    for i, col in enumerate(zone_priority.get("pot_zone", [])):
        if col in df.columns:
            active = (coerce_numeric(df[col]).fillna(0).values > 0) & (pot_id == 0)
            pot_id[active] = i + 1

    return pot_id


def load_one_xlsx(xlsx_path, cfg):
    sheet = cfg["xlsx_sheet"]
    header = cfg["xlsx_header_row"]
    skip = cfg["xlsx_skip_rows"]
    df = pd.read_excel(xlsx_path, sheet_name=sheet, header=header, skiprows=skip)
    # Coerce numerics; missing values become NaN. EthoVision uses '-' for missing.
    return df


def process_session(session_num, state, xlsx_path, cfg, out_dir):
    print(f"\n=== Session {session_num} ({state}) ===")
    print(f"  xlsx: {Path(xlsx_path).name}")
    df = load_one_xlsx(xlsx_path, cfg)
    T_raw = len(df)
    print(f"  raw bins: {T_raw}, raw bin size: {cfg['raw_bin_ms']} ms")

    # Speed
    speed_col = cfg["continuous_columns"]["speed"]
    speed_raw = coerce_numeric(df[speed_col]).fillna(0.0).values

    # Distance to nearest pot
    pot_cols = cfg["distance_pot_columns"]
    dist = np.full((T_raw, len(pot_cols)), np.inf)
    for i, c in enumerate(pot_cols):
        if c in df.columns:
            dist[:, i] = coerce_numeric(df[c]).fillna(np.inf).values
        else:
            print(f"  WARNING: missing column {c}")
    nearest_pot_dist = dist.min(axis=1)
    # Replace any infs (all-pot-cols missing) with NaN; downstream will fill
    nearest_pot_dist = np.where(np.isfinite(nearest_pot_dist), nearest_pot_dist, np.nan)
    nearest_pot_dist = pd.Series(nearest_pot_dist).ffill().bfill().fillna(0.0).values

    # Events
    event_map = cfg["event_columns"]
    event_names = list(event_map.keys())
    events_raw = np.zeros((T_raw, len(event_names)), dtype=np.int64)
    for i, (canonical, xlsx_col) in enumerate(event_map.items()):
        if xlsx_col in df.columns:
            v = coerce_numeric(df[xlsx_col]).fillna(0.0).values
            events_raw[:, i] = (v > 0).astype(np.int64)
        else:
            print(f"  WARNING: missing event column {xlsx_col!r} → all zeros")

    # Zone
    zone_raw, zone_labels = assign_zone(df, cfg["zone_priority"])

    # Per-bin pot identity (0 = none, 1..4 = at pot N)
    pot_id_raw = assign_pot_id(df, cfg["zone_priority"])

    # Trial time at start of each raw bin
    tt = coerce_numeric(df["Trial time"]).fillna(0.0).values

    # Rebin
    factor = cfg["bin_factor"]
    T_new = T_raw // factor
    print(f"  rebinning: factor={factor}, output bins: {T_new} "
          f"(={T_new * cfg['target_bin_ms'] / 1000:.1f} s)")

    speed = block_mean(speed_raw, factor)
    distance_to_pot = block_mean(nearest_pot_dist, factor)
    zone = block_mode(zone_raw, factor)
    pot_id = block_mode(pot_id_raw, factor)
    events = np.zeros((T_new, len(event_names)), dtype=np.int64)
    for i in range(len(event_names)):
        events[:, i] = block_mode(events_raw[:, i], factor)
    trial_time = tt[:T_new * factor:factor]

    print(f"  speed range: [{speed.min():.2f}, {speed.max():.2f}] cm/s")
    print(f"  distance_to_pot range: [{distance_to_pot.min():.2f}, "
          f"{distance_to_pot.max():.2f}] cm")
    zone_counts = {zone_labels[k]: int((zone == k).sum()) for k in range(len(zone_labels))}
    print(f"  zone occupancy (bins): {zone_counts}")
    pot_counts = {f"P{i}" if i > 0 else "none": int((pot_id == i).sum())
                  for i in range(5)}
    print(f"  pot_id occupancy (bins): {pot_counts}")
    event_counts = {n: int(events[:, i].sum())
                    for i, n in enumerate(event_names)}
    print(f"  event totals (bins active): {event_counts}")

    out_path = out_dir / f"session_{session_num}.npz"
    np.savez(
        out_path,
        speed=speed,
        distance_to_pot=distance_to_pot,
        zone=zone,
        pot_id=pot_id,
        events=events,
        zone_labels=np.array(zone_labels),
        event_names=np.array(event_names),
        trial_time=trial_time,
        meta=np.array(
            {
                "session_num": session_num,
                "state": state,
                "xlsx_name": Path(xlsx_path).name,
                "n_raw_bins": int(T_raw),
                "n_bins": int(T_new),
                "raw_bin_ms": cfg["raw_bin_ms"],
                "target_bin_ms": cfg["target_bin_ms"],
            },
            dtype=object,
        ),
    )
    print(f"  saved {out_path}")


def main():
    cfg = load_config()
    paths_data = load_paths_yaml(cfg)
    out_dir = ensure_dir(REPO_ROOT / cfg["out_dirs"]["binned"])
    print(f"Output dir: {out_dir}")

    np.random.seed(cfg["random_seed"])

    sess = session_list(cfg)
    print(f"Sessions: {sess}")

    for session_num, state in sess:
        xlsx, paths_state = session_xlsx(paths_data, session_num)
        if paths_state != state:
            print(f"  WARNING: state mismatch for S{session_num}: "
                  f"config={state}, paths.yaml={paths_state}")
        if xlsx is None:
            print(f"  SKIP S{session_num}: behavior xlsx is null in paths.yaml")
            continue
        process_session(session_num, state, xlsx, cfg, out_dir)

    print(f"\nDone. Binned data in {out_dir}")


if __name__ == "__main__":
    main()
