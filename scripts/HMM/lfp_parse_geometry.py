"""
Parse static probe geometry from reference SpikeGLX .ap.meta files.

Runs ONCE. Outputs are static; all downstream LFP analyses reference them.

Inputs
------
Two reference .ap.meta files from 6_17_25 EXP (imec0 and imec1) that contain
snsGeomMap entries with (shank:x:y:used) per channel.

Outputs (in data/HMM/neural_alignment/lfp/)
-------------------------------------------
- probe_geometry.csv         (per-channel, both probes stacked)
- bipolar_pairs_imec0.csv    (within-shank, <=30 um, ACA)
- bipolar_pairs_imec1.csv    (within-shank, <=30 um, same-region only)
- geometry_summary.json      (counts, distances, exclusions)
- figures/.../probe_layout_imec0.png
- figures/.../probe_layout_imec1.png
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_IMEC0_META = Path(
    "H:/Neuropixels Data/DOUBLE_Probe/DOUBLE_FED/6_17_25_DOUBLE/"
    "DOUBLE_PROBE_6_17_25_EXP_g0/DOUBLE_PROBE_6_17_25_EXP_g0_t0.imec0.ap.meta"
)
DEFAULT_IMEC1_META = Path(
    "H:/Neuropixels Data/DOUBLE_Probe/DOUBLE_FED/6_17_25_DOUBLE/"
    "DOUBLE_PROBE_6_17_25_EXP_g0/DOUBLE_PROBE_6_17_25_EXP_g0_t0.imec1.ap.meta"
)

REPO = Path("H:/NPX ANALYSIS REPO")
OUT_DIR = REPO / "data" / "HMM" / "neural_alignment" / "lfp"
FIG_DIR = REPO / "figures" / "HMM" / "neural_alignment" / "lfp"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# y boundary on probe 1: LHA below, RSP above
LHA_RSP_Y_SPLIT_UM = 2500.0
# bipolar pairing threshold
BIPOLAR_MAX_DIST_UM = 30.0


def parse_sns_geom_map(meta_path: Path) -> tuple[dict, pd.DataFrame]:
    """Parse the ~snsGeomMap line. Returns (header_dict, channels_df)."""
    text = meta_path.read_text(encoding="utf-8", errors="ignore")
    line = None
    for ln in text.splitlines():
        if ln.startswith("~snsGeomMap="):
            line = ln[len("~snsGeomMap="):].strip()
            break
    if line is None:
        raise RuntimeError(f"No snsGeomMap line in {meta_path}")

    tuples = re.findall(r"\(([^)]+)\)", line)
    if not tuples:
        raise RuntimeError(f"snsGeomMap had no tuples: {meta_path}")

    head = tuples[0].split(",")
    header = dict(
        probe_type=head[0],
        n_shanks=int(head[1]),
        shank_pitch_um=float(head[2]),
        probe_width_um=float(head[3]),
    )
    if header["probe_type"] != "NP2013":
        raise RuntimeError(
            f"Expected probe_type=NP2013, got {header['probe_type']} in {meta_path}"
        )

    rows = []
    for ch_idx, entry in enumerate(tuples[1:]):
        parts = entry.split(":")
        if len(parts) != 4:
            raise RuntimeError(f"Bad channel entry {entry!r} in {meta_path}")
        shank, x_um, y_um, used = parts
        rows.append(
            dict(
                channel_index=ch_idx,
                shank=int(shank),
                x_um=float(x_um),
                y_um=float(y_um),
                used=int(used),
            )
        )
    df = pd.DataFrame(rows)
    return header, df


def assign_region(probe: str, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if probe == "imec0":
        df["region"] = "ACA"
    elif probe == "imec1":
        df["region"] = np.where(df["y_um"].values < LHA_RSP_Y_SPLIT_UM, "LHA", "RSP")
    else:
        raise ValueError(probe)
    return df


def build_bipolar_pairs(
    df: pd.DataFrame,
    max_dist_um: float = BIPOLAR_MAX_DIST_UM,
    same_region_only: bool = False,
) -> tuple[pd.DataFrame, list[int]]:
    """
    For each channel C, find the nearest same-shank neighbor within max_dist_um.
    Ties broken by smaller channel index. Each undirected pair (A<B) appears once.
    Returns (pairs_df, excluded_channel_indices).
    """
    xs = df["x_um"].values
    ys = df["y_um"].values
    shanks = df["shank"].values
    regions = df["region"].values
    chans = df["channel_index"].values
    n = len(df)

    nearest = np.full(n, -1, dtype=np.int64)
    nearest_dist = np.full(n, np.nan)

    for i in range(n):
        # candidates: same shank, different channel, optionally same region
        cand = np.where(shanks == shanks[i])[0]
        cand = cand[cand != i]
        if same_region_only:
            cand = cand[regions[cand] == regions[i]]
        if cand.size == 0:
            continue
        dx = xs[cand] - xs[i]
        dy = ys[cand] - ys[i]
        d = np.sqrt(dx * dx + dy * dy)
        mask = d <= max_dist_um
        if not mask.any():
            continue
        cand = cand[mask]
        d = d[mask]
        # tie break: smallest distance, then smallest channel index
        order = np.lexsort((chans[cand], d))
        nearest[i] = cand[order[0]]
        nearest_dist[i] = d[order[0]]

    excluded = chans[nearest < 0].tolist()

    pair_set = set()
    pair_rows = []
    for i in range(n):
        j = nearest[i]
        if j < 0:
            continue
        a, b = (i, j) if chans[i] < chans[j] else (j, i)
        key = (int(chans[a]), int(chans[b]))
        if key in pair_set:
            continue
        pair_set.add(key)
        ax, ay = xs[a], ys[a]
        bx, by = xs[b], ys[b]
        d = float(np.hypot(ax - bx, ay - by))
        pair_rows.append(
            dict(
                channel_a=int(chans[a]),
                channel_b=int(chans[b]),
                shank=int(shanks[a]),
                mean_x_um=float((ax + bx) / 2),
                mean_y_um=float((ay + by) / 2),
                distance_um=d,
                region=str(regions[a]),
            )
        )

    pairs_df = pd.DataFrame(pair_rows).sort_values(
        ["shank", "mean_y_um", "channel_a"]
    ).reset_index(drop=True)
    pairs_df.insert(0, "pair_index", np.arange(len(pairs_df)))
    return pairs_df, excluded


def plot_layout(
    df: pd.DataFrame,
    pairs_df: pd.DataFrame,
    out_path: Path,
    title: str,
    color_by: str = "shank",
):
    fig, ax = plt.subplots(figsize=(10, 12))

    if color_by == "shank":
        cmap = plt.get_cmap("tab10")
        for sh, sub in df.groupby("shank"):
            ax.scatter(
                sub["x_um"], sub["y_um"], s=14,
                color=cmap(sh % 10), label=f"shank {sh}", zorder=3,
            )
    elif color_by == "region":
        colors = {"ACA": "#1f77b4", "LHA": "#d62728", "RSP": "#2ca02c"}
        for reg, sub in df.groupby("region"):
            ax.scatter(
                sub["x_um"], sub["y_um"], s=14,
                color=colors.get(reg, "k"), label=reg, zorder=3,
            )

    for _, row in pairs_df.iterrows():
        a = df[df["channel_index"] == row["channel_a"]].iloc[0]
        b = df[df["channel_index"] == row["channel_b"]].iloc[0]
        ax.plot(
            [a["x_um"], b["x_um"]],
            [a["y_um"], b["y_um"]],
            color="0.4", lw=0.5, alpha=0.6, zorder=2,
        )

    if "imec1" in title.lower():
        ax.axhline(LHA_RSP_Y_SPLIT_UM, color="k", lw=0.8, ls="--", alpha=0.5)
        ax.text(
            ax.get_xlim()[1], LHA_RSP_Y_SPLIT_UM, "  y=2500 (LHA/RSP split)",
            ha="left", va="center", fontsize=8,
        )

    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def summarize(df: pd.DataFrame, label: str, pairs: pd.DataFrame, excluded: list[int]) -> dict:
    out = dict(
        probe=label,
        n_channels=int(len(df)),
        shank_counts={int(k): int(v) for k, v in df["shank"].value_counts().sort_index().items()},
    )
    by_region = {}
    for reg, sub in df.groupby("region"):
        pair_sub = pairs[pairs["region"] == reg]
        by_region[reg] = dict(
            n_channels=int(len(sub)),
            n_bipolar_pairs=int(len(pair_sub)),
            mean_pair_distance_um=float(pair_sub["distance_um"].mean()) if len(pair_sub) else None,
            min_pair_distance_um=float(pair_sub["distance_um"].min()) if len(pair_sub) else None,
            max_pair_distance_um=float(pair_sub["distance_um"].max()) if len(pair_sub) else None,
        )
    out["by_region"] = by_region
    out["n_excluded_channels"] = int(len(excluded))
    out["excluded_channels"] = excluded
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-imec0", type=Path, default=DEFAULT_IMEC0_META)
    ap.add_argument("--meta-imec1", type=Path, default=DEFAULT_IMEC1_META)
    args = ap.parse_args()

    print(f"Parsing imec0 meta: {args.meta_imec0}")
    head0, df0 = parse_sns_geom_map(args.meta_imec0)
    df0 = assign_region("imec0", df0)
    df0.insert(0, "probe", "imec0")
    print(f"  header={head0}, n_channels={len(df0)}")

    print(f"Parsing imec1 meta: {args.meta_imec1}")
    head1, df1 = parse_sns_geom_map(args.meta_imec1)
    df1 = assign_region("imec1", df1)
    df1.insert(0, "probe", "imec1")
    print(f"  header={head1}, n_channels={len(df1)}")

    geom_all = pd.concat([df0, df1], ignore_index=True)
    geom_path = OUT_DIR / "probe_geometry.csv"
    geom_all.to_csv(geom_path, index=False)
    print(f"Saved {geom_path}")

    pairs0, excl0 = build_bipolar_pairs(df0, same_region_only=False)
    pairs1, excl1 = build_bipolar_pairs(df1, same_region_only=True)
    pairs0_path = OUT_DIR / "bipolar_pairs_imec0.csv"
    pairs1_path = OUT_DIR / "bipolar_pairs_imec1.csv"
    pairs0.to_csv(pairs0_path, index=False)
    pairs1.to_csv(pairs1_path, index=False)
    print(f"Saved {pairs0_path}")
    print(f"Saved {pairs1_path}")

    # Confirm no LHA-RSP cross-region pairs on probe 1
    pair1_regions_a = df1.set_index("channel_index").loc[pairs1["channel_a"], "region"].values
    pair1_regions_b = df1.set_index("channel_index").loc[pairs1["channel_b"], "region"].values
    cross = (pair1_regions_a != pair1_regions_b).sum()
    if cross != 0:
        raise RuntimeError(f"Found {cross} cross-region pairs on imec1 (should be 0)")

    summary = dict(
        imec0=summarize(df0, "imec0", pairs0, excl0),
        imec1=summarize(df1, "imec1", pairs1, excl1),
        meta_files=dict(imec0=str(args.meta_imec0), imec1=str(args.meta_imec1)),
        probe_headers=dict(imec0=head0, imec1=head1),
        bipolar_max_dist_um=BIPOLAR_MAX_DIST_UM,
        lha_rsp_y_split_um=LHA_RSP_Y_SPLIT_UM,
        cross_region_pairs_on_imec1=int(cross),
    )
    summary_path = OUT_DIR / "geometry_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {summary_path}")

    plot_layout(
        df0, pairs0,
        FIG_DIR / "probe_layout_imec0.png",
        title="Probe 0 (imec0, ACA) layout + bipolar pairs",
        color_by="shank",
    )
    plot_layout(
        df1, pairs1,
        FIG_DIR / "probe_layout_imec1.png",
        title="Probe 1 (imec1, LHA+RSP) layout + bipolar pairs",
        color_by="region",
    )
    print(f"Saved figures to {FIG_DIR}")

    print("\n=== SUMMARY ===")
    for probe_label, sub in summary.items():
        if probe_label not in ("imec0", "imec1"):
            continue
        print(f"\n{probe_label}: n_channels={sub['n_channels']}, shanks={sub['shank_counts']}")
        for reg, info in sub["by_region"].items():
            print(
                f"  {reg}: {info['n_channels']} channels, "
                f"{info['n_bipolar_pairs']} bipolar pairs, "
                f"dist um min/mean/max = "
                f"{info['min_pair_distance_um']:.2f}/"
                f"{info['mean_pair_distance_um']:.2f}/"
                f"{info['max_pair_distance_um']:.2f}"
                if info["n_bipolar_pairs"] else
                f"  {reg}: {info['n_channels']} channels, 0 bipolar pairs"
            )
        if sub["excluded_channels"]:
            print(f"  excluded channels: {sub['excluded_channels']}")
        else:
            print(f"  excluded channels: none")
    print(f"\nCross-region (LHA-RSP) pairs on imec1: {summary['cross_region_pairs_on_imec1']} (expected 0)")


if __name__ == "__main__":
    main()
