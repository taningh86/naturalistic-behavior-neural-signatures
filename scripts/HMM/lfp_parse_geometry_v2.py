"""Refined static probe geometry — v2.

Probe 1 region boundaries tightened from the IMRO midpoint (y=2500 µm) to the
spike-defined tissue bands:
  LHA proper: y < 345 µm
  RSP proper: y > 4680 µm
  intermediate (345 ≤ y ≤ 4680): excluded from bipolar pair construction

Probe 0 (ACA) is unchanged — every channel is in ACA tissue.

Reuses `data/HMM/neural_alignment/lfp/probe_geometry.csv` (no re-parsing of
the .ap.meta files). Writes v2 outputs alongside v1.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO = Path("H:/NPX ANALYSIS REPO")
OUT_DIR = REPO / "data" / "HMM" / "neural_alignment" / "lfp"
FIG_DIR = REPO / "figures" / "HMM" / "neural_alignment" / "lfp"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

LHA_Y_MAX = 345.0
RSP_Y_MIN = 4680.0
BIPOLAR_MAX_DIST_UM = 30.0
SPARSE_FLAG_THRESHOLD = 30


def refine_region_v2(probe: str, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if probe == "imec0":
        df["region"] = "ACA"
        return df
    if probe == "imec1":
        y = df["y_um"].values
        region = np.where(y < LHA_Y_MAX, "LHA",
                          np.where(y > RSP_Y_MIN, "RSP", "intermediate"))
        df["region"] = region
        return df
    raise ValueError(probe)


def build_bipolar_pairs_v2(df: pd.DataFrame, max_dist_um: float = BIPOLAR_MAX_DIST_UM):
    """Within-shank AND within-region nearest neighbor ≤ max_dist_um.
    Intermediate channels (on imec1) are excluded entirely."""
    df = df.copy()
    df = df.reset_index(drop=True)
    eligible = df["region"].isin(("ACA", "LHA", "RSP")).values
    xs = df["x_um"].values
    ys = df["y_um"].values
    shanks = df["shank"].values
    regions = df["region"].values
    chans = df["channel_index"].values
    n = len(df)

    nearest = np.full(n, -1, dtype=np.int64)
    nearest_dist = np.full(n, np.nan)

    for i in range(n):
        if not eligible[i]:
            continue
        cand = np.where((shanks == shanks[i])
                          & (regions == regions[i])
                          & eligible)[0]
        cand = cand[cand != i]
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
        order = np.lexsort((chans[cand], d))
        nearest[i] = cand[order[0]]
        nearest_dist[i] = d[order[0]]

    excluded = []
    for i in range(n):
        if not eligible[i]:
            continue
        if nearest[i] < 0:
            excluded.append(int(chans[i]))

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
        pair_rows.append(dict(
            channel_a=int(chans[a]),
            channel_b=int(chans[b]),
            shank=int(shanks[a]),
            mean_x_um=float((ax + bx) / 2),
            mean_y_um=float((ay + by) / 2),
            distance_um=d,
            region=str(regions[a]),
        ))

    pairs_df = pd.DataFrame(pair_rows)
    if len(pairs_df):
        pairs_df = pairs_df.sort_values(["shank", "mean_y_um", "channel_a"]).reset_index(drop=True)
        pairs_df.insert(0, "pair_index", np.arange(len(pairs_df)))
    return pairs_df, excluded


def plot_layout_v2(df, pairs_df, out_path, title, color_by="shank",
                    show_region_boundaries=False):
    fig, ax = plt.subplots(figsize=(10, 12))

    if color_by == "shank":
        cmap = plt.get_cmap("tab10")
        for sh, sub in df.groupby("shank"):
            ax.scatter(sub["x_um"], sub["y_um"], s=14,
                        color=cmap(sh % 10), label=f"shank {sh}", zorder=3)
    elif color_by == "region_v2":
        colors = {"ACA": "#1f77b4", "LHA": "#d62728",
                   "RSP": "#2ca02c", "intermediate": "#999999"}
        for reg, sub in df.groupby("region"):
            ax.scatter(sub["x_um"], sub["y_um"], s=14,
                        color=colors.get(reg, "k"),
                        label=f"{reg} (n={len(sub)})", zorder=3)

    for _, row in pairs_df.iterrows():
        a = df[df["channel_index"] == row["channel_a"]].iloc[0]
        b = df[df["channel_index"] == row["channel_b"]].iloc[0]
        ax.plot([a["x_um"], b["x_um"]],
                 [a["y_um"], b["y_um"]],
                 color="0.4", lw=0.5, alpha=0.6, zorder=2)

    if show_region_boundaries:
        ax.axhline(LHA_Y_MAX, color="k", lw=0.8, ls="--", alpha=0.7)
        ax.axhline(RSP_Y_MIN, color="k", lw=0.8, ls="--", alpha=0.7)
        ax.text(ax.get_xlim()[1], LHA_Y_MAX,
                 f"  y={LHA_Y_MAX:.0f} (LHA upper bound)",
                 ha="left", va="center", fontsize=8)
        ax.text(ax.get_xlim()[1], RSP_Y_MIN,
                 f"  y={RSP_Y_MIN:.0f} (RSP lower bound)",
                 ha="left", va="center", fontsize=8)

    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def summarize(df, label, pairs, excluded):
    out = dict(
        probe=label,
        n_channels=int(len(df)),
        shank_counts={int(k): int(v) for k, v in df["shank"].value_counts().sort_index().items()},
    )
    by_region = {}
    for reg, sub in df.groupby("region"):
        pair_sub = pairs[pairs["region"] == reg] if len(pairs) else pd.DataFrame()
        by_region[reg] = dict(
            n_channels=int(len(sub)),
            n_bipolar_pairs=int(len(pair_sub)),
            mean_pair_distance_um=float(pair_sub["distance_um"].mean()) if len(pair_sub) else None,
            min_pair_distance_um=float(pair_sub["distance_um"].min()) if len(pair_sub) else None,
            max_pair_distance_um=float(pair_sub["distance_um"].max()) if len(pair_sub) else None,
        )
    out["by_region"] = by_region
    out["n_excluded_channels_no_neighbor"] = int(len(excluded))
    out["excluded_channels_no_neighbor"] = excluded
    return out


def main():
    geom_path = OUT_DIR / "probe_geometry.csv"
    if not geom_path.exists():
        raise FileNotFoundError(
            f"Run lfp_parse_geometry.py first; missing {geom_path}"
        )
    geom = pd.read_csv(geom_path)
    print(f"Loaded {len(geom)} channels from {geom_path}")

    # Refine region assignments
    p0 = refine_region_v2("imec0", geom[geom["probe"] == "imec0"])
    p1 = refine_region_v2("imec1", geom[geom["probe"] == "imec1"])
    geom_v2 = pd.concat([p0, p1], ignore_index=True)
    geom_v2_path = OUT_DIR / "probe_geometry_v2.csv"
    geom_v2.to_csv(geom_v2_path, index=False)
    print(f"Saved {geom_v2_path}")

    print("\nProbe 1 region channel counts (refined):")
    print(p1["region"].value_counts())

    pairs0, excl0 = build_bipolar_pairs_v2(p0)
    pairs1, excl1 = build_bipolar_pairs_v2(p1)

    pairs0_path = OUT_DIR / "bipolar_pairs_imec0_v2.csv"
    pairs1_path = OUT_DIR / "bipolar_pairs_imec1_v2.csv"
    pairs0.to_csv(pairs0_path, index=False)
    pairs1.to_csv(pairs1_path, index=False)
    print(f"Saved {pairs0_path}")
    print(f"Saved {pairs1_path}")

    # Sanity check: no cross-region pairs in imec1
    if len(pairs1):
        regions_a = p1.set_index("channel_index").loc[pairs1["channel_a"], "region"].values
        regions_b = p1.set_index("channel_index").loc[pairs1["channel_b"], "region"].values
        cross = int((regions_a != regions_b).sum())
        if cross != 0:
            raise RuntimeError(f"Cross-region pairs found in imec1 v2: {cross}")
        n_intermediate = int(((regions_a == "intermediate") | (regions_b == "intermediate")).sum())
        if n_intermediate != 0:
            raise RuntimeError(f"Intermediate channel(s) in pair list: {n_intermediate}")

    summary = dict(
        imec0=summarize(p0, "imec0", pairs0, excl0),
        imec1=summarize(p1, "imec1", pairs1, excl1),
        region_boundaries=dict(
            lha_y_max_um=LHA_Y_MAX,
            rsp_y_min_um=RSP_Y_MIN,
        ),
        bipolar_max_dist_um=BIPOLAR_MAX_DIST_UM,
    )
    summary_path = OUT_DIR / "geometry_summary_v2.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {summary_path}")

    # Validation figures
    plot_layout_v2(p0, pairs0,
                    FIG_DIR / "probe_layout_imec0_v2.png",
                    title="Probe 0 (imec0, ACA) v2 layout + bipolar pairs",
                    color_by="shank")
    plot_layout_v2(p1, pairs1,
                    FIG_DIR / "probe_layout_imec1_v2.png",
                    title="Probe 1 (imec1) v2 layout — refined LHA / intermediate / RSP",
                    color_by="region_v2",
                    show_region_boundaries=True)
    print(f"Saved figures to {FIG_DIR}")

    # Compare with v1
    v1_summary_path = OUT_DIR / "geometry_summary.json"
    if v1_summary_path.exists():
        v1 = json.loads(v1_summary_path.read_text())
    else:
        v1 = None

    print("\n=== v2 SUMMARY ===")
    for probe_label, sub in summary.items():
        if probe_label not in ("imec0", "imec1"):
            continue
        print(f"\n{probe_label}: n_channels={sub['n_channels']}, "
              f"shanks={sub['shank_counts']}")
        for reg, info in sub["by_region"].items():
            line = (f"  {reg}: {info['n_channels']} channels, "
                    f"{info['n_bipolar_pairs']} bipolar pairs")
            if info["n_bipolar_pairs"]:
                line += (f", dist um min/mean/max = "
                         f"{info['min_pair_distance_um']:.2f}/"
                         f"{info['mean_pair_distance_um']:.2f}/"
                         f"{info['max_pair_distance_um']:.2f}")
            # compare to v1
            if v1 and probe_label in v1:
                v1_info = v1[probe_label]["by_region"].get(reg)
                if v1_info:
                    pct = (100 * info["n_bipolar_pairs"]
                              / max(1, v1_info["n_bipolar_pairs"]))
                    line += (f"  (v1: {v1_info['n_bipolar_pairs']} → "
                             f"v2: {info['n_bipolar_pairs']}, {pct:.0f}% retained)")
            print(line)
        if sub["excluded_channels_no_neighbor"]:
            print(f"  channels excluded for no neighbor within {BIPOLAR_MAX_DIST_UM} um: "
                  f"{sub['excluded_channels_no_neighbor']}")

    # Sparse flag
    flagged = False
    for probe_label, sub in summary.items():
        if probe_label not in ("imec0", "imec1"):
            continue
        for reg, info in sub["by_region"].items():
            if reg in ("intermediate", "unused"):
                continue
            if info["n_bipolar_pairs"] < SPARSE_FLAG_THRESHOLD:
                if not flagged:
                    print("\n*** WARNING: sparse pair counts ***")
                    flagged = True
                print(f"  {probe_label} {reg}: only {info['n_bipolar_pairs']} pairs "
                      f"(< {SPARSE_FLAG_THRESHOLD}). Consider relaxing the within-shank "
                      f"distance threshold from {BIPOLAR_MAX_DIST_UM} µm or accepting "
                      f"reduced statistical power.")
    if not flagged:
        print(f"\nAll regional pair counts ≥ {SPARSE_FLAG_THRESHOLD}. Statistical "
              "power should be acceptable for regional means.")


if __name__ == "__main__":
    main()
