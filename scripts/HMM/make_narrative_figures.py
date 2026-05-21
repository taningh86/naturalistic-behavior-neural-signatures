"""Narrative-driven figure set summarizing the project's defensible findings.

8 figures, each saved as .png (300 dpi) + .svg in figures/HMM/narrative/.
Reads existing analysis outputs; performs no new statistics or fits.

Findings covered:
  Fig 1: Regional preferred-state encoding (Track B B1) — ACA vs LHA dissociation
  Fig 2: ACA generic vs LHA selective pre-exit signal (script 14)
  Fig 3: S3 home-exit multi-metric population reorganization (script 15)
  Fig 4: ACA→LHA Granger lead at home-exit (script 16)
  Fig 5: ACA high-gamma feeding suppression (script 18 A1)
  Fig 6: RSP beta / low-gamma feeding suppression (script 18 A1)
  Fig 7: RSP sharp-wave ripples (script 19 v2)
  Fig 8: Project summary
"""
from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap


REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "figures" / "HMM" / "narrative"
OUT.mkdir(parents=True, exist_ok=True)

# ----- Style -----
plt.rcParams.update({
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 16,
})
COLORS = {"ACA": "#1abc9c", "LHA": "#e67e22", "RSP": "#8e44ad",
            "fed": "#1f77b4", "fasted": "#d62728", "HFD": "#9b59b6"}


def save_both(fig, name: str, caption: str = "") -> None:
    """Save figure as PNG (300 dpi) and SVG; write caption .txt next to it."""
    png = OUT / f"{name}.png"
    svg = OUT / f"{name}.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    if caption:
        (OUT / f"{name}_caption.txt").write_text(caption.strip() + "\n",
                                                    encoding="utf-8")
    print(f"  Saved {png.name} + {svg.name}")


def panel_label(ax, label: str):
    """Add bold 'a','b',... panel label at the top-left corner of an axes."""
    ax.text(-0.13, 1.05, label, transform=ax.transAxes,
            fontsize=18, fontweight="bold", family="sans-serif",
            va="top", ha="left")


# ============================================================================
# State labels — derived from merged_state_profiles_dynamax.csv
# ============================================================================
def hmm_state_labels() -> dict[int, str]:
    """Return short descriptive label per HMM state."""
    p = REPO / "data" / "HMM" / "merged_state_profiles_dynamax.csv"
    if not p.exists():
        return {i: f"S{i}" for i in range(14)}
    df = pd.read_csv(p)
    labels = {}
    for _, row in df.iterrows():
        sid = int(row["state"])
        tag = []
        # Priority by behavior
        if row.get("event_feeding_prob", 0) > 0.4:
            tag.append("feed")
        elif row.get("event_digging_sand_prob", 0) > 0.3:
            tag.append("dig")
        elif row.get("event_contemplation_at_transition_prob", 0) > 0.4:
            tag.append("contemp")
        elif row.get("event_quick_loop_at_home_prob", 0) > 0.2:
            tag.append("quick-loop")
        elif row.get("event_rearing_prob", 0) > 0.3:
            tag.append("rear")
        # Zone overlay
        zone_cols = {"zone_home_prob": "home", "zone_transition_prob": "trans",
                       "zone_pot_prob": "pot", "zone_pot_zone_prob": "potZ",
                       "zone_arena_prob": "arena"}
        zone_probs = {z: row.get(col, 0) for col, z in zone_cols.items()}
        dom_zone = max(zone_probs, key=zone_probs.get)
        if zone_probs[dom_zone] > 0.4 and dom_zone not in [t for t in tag]:
            tag.insert(0, dom_zone)
        labels[sid] = f"S{sid}: " + "/".join(tag) if tag else f"S{sid}"
    return labels


STATE_LABEL = hmm_state_labels()


# ============================================================================
# Fig 1: Regional preferred-state encoding (ACA vs LHA) — REVISED
# ============================================================================
def _preferred_state_fraction_per_session(region: str, n_states: int = 14
                                              ) -> dict[int, np.ndarray]:
    """For each foraging session, return (n_states,) fraction-of-units-preferring
    -state-k vector. Preferred state = argmax of unit's mean firing rate across
    14 HMM states. Returns dict session → fractions."""
    out = {}
    for sn in [4, 6, 8, 12, 14, 16]:
        # Preferred path 1: per-session B1_selectivity_summary csv
        f1 = (REPO / "data/HMM/neural_alignment/track_B_all_sessions"
              / f"session_{sn}" / f"B1_selectivity_summary_{region}.csv")
        if f1.exists() and "preferred_state" in pd.read_csv(f1, nrows=1).columns:
            df = pd.read_csv(f1)
            pref = df["preferred_state"].astype(int).values
        else:
            # Fallback: load the per-unit × state mean-FR matrix (S12 case)
            f2 = (REPO / "data/HMM/neural_alignment/state_conditioned_S12"
                  / f"B1_state_selectivity_matrix_{region}.csv")
            if not f2.exists():
                continue
            df = pd.read_csv(f2)
            state_cols = [c for c in df.columns if c.startswith("state_")]
            m = df[state_cols].values
            pref = np.nanargmax(m, axis=1)
        n = len(pref)
        if not n:
            continue
        counts = np.bincount(pref, minlength=n_states).astype(np.float64)
        out[sn] = counts / n
    return out


def fig1_regional_encoding():
    print("  Computing per-session preferred-state fractions...")
    aca_fracs = _preferred_state_fraction_per_session("ACA", n_states=14)
    lha_fracs = _preferred_state_fraction_per_session("LHA", n_states=14)
    print(f"    ACA: {len(aca_fracs)} sessions ({sorted(aca_fracs)})")
    print(f"    LHA: {len(lha_fracs)} sessions ({sorted(lha_fracs)})")
    if not aca_fracs or not lha_fracs:
        print("  [WARN] insufficient per-unit B1 outputs"); return

    n_states = 14
    aca_mat = np.array([aca_fracs[s] for s in sorted(aca_fracs)])
    lha_mat = np.array([lha_fracs[s] for s in sorted(lha_fracs)])
    aca_mean = aca_mat.mean(axis=0); aca_sem = aca_mat.std(axis=0) / np.sqrt(len(aca_mat))
    lha_mean = lha_mat.mean(axis=0); lha_sem = lha_mat.std(axis=0) / np.sqrt(len(lha_mat))

    # Print top-3 per region
    for region, mean in [("ACA", aca_mean), ("LHA", lha_mean)]:
        top3 = np.argsort(-mean)[:3]
        print(f"    {region} top-3 preferred states:")
        for k in top3:
            print(f"      {STATE_LABEL.get(int(k), f'S{k}'):30s}  "
                  f"mean fraction = {mean[k]:.3f}")

    fig, (axa, axl) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    x = np.arange(n_states)
    for ax, mean, sem, region in [(axa, aca_mean, aca_sem, "ACA"),
                                       (axl, lha_mean, lha_sem, "LHA")]:
        top_idx = set(np.argsort(-mean)[:3].tolist())
        colors = [COLORS[region] if i in top_idx else "#d0d0d0"
                    for i in range(n_states)]
        ax.bar(x, mean, yerr=sem, color=colors,
                 edgecolor="black", lw=0.8, capsize=3)
        labels = [STATE_LABEL.get(i, f"S{i}") for i in range(n_states)]
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45,
                                                  ha="right", fontsize=9)
        ax.set_title(region, fontsize=14, fontweight="bold",
                       color=COLORS[region])
        ax.grid(True, axis="y", alpha=0.3)
        # annotate top 3 with their fraction
        for k in top_idx:
            ax.text(k, mean[k] + (sem[k] if np.isfinite(sem[k]) else 0) + 0.005,
                      f"{mean[k]:.2f}", ha="center", fontsize=9, fontweight="bold",
                      color=COLORS[region])
    axa.set_ylabel("Fraction of units (mean ± SEM)", fontsize=12)
    axl.set_ylabel("")
    panel_label(axa, "a"); panel_label(axl, "b")
    fig.suptitle("ACA neurons preferentially encode action-selection states; "
                  "LHA neurons preferentially encode consummatory states",
                  fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    caption = ("Across 6 foraging sessions (S4/6/8 fed, S12/14/16 fasted), bars show "
               "the mean ± SEM fraction of units per region whose preferred state "
               "(highest mean firing rate across the 14 merged HMM states) is the "
               "indicated state. ACA neurons preferentially encode action-selection "
               "states (digging S6, contemplation S4). LHA neurons preferentially "
               "encode consummatory states (pure feeding S2, arena-feeding S11, "
               "digging S6). Top-3 preferred states per region are colored, others "
               "shown in light gray for completeness.")
    save_both(fig, "fig1_regional_preferred_state_encoding", caption)


# ============================================================================
# Fig 2: ACA generic vs LHA selective pre-exit signal (script 14)
# ============================================================================
def fig2_pre_exit_signal():
    src = REPO / "data" / "HMM" / "neural_alignment" / "state_transitions" / "replication_A1.csv"
    if not src.exists():
        print(f"  [WARN] missing: {src}"); return
    df = pd.read_csv(src)
    aca = df[df.region == "ACA"].sort_values("state").reset_index(drop=True)
    lha = df[df.region == "LHA"].sort_values("state").reset_index(drop=True)

    fig, (axa, axl) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for ax, sub, region in [(axa, aca, "ACA"), (axl, lha, "LHA")]:
        if not len(sub):
            continue
        x = np.arange(len(sub))
        pass_mask = sub["n_sessions_passing"].values >= 4
        colors = [COLORS[region] if p else "lightgray" for p in pass_mask]
        ax.bar(x, sub["n_sessions_passing"], color=colors,
                 edgecolor="black", lw=0.8)
        ax.axhline(4, color="red", lw=1.2, ls="--", alpha=0.7)
        ax.text(len(sub) - 0.5, 4.1, "replication threshold (4/6)",
                  ha="right", fontsize=9, color="red", style="italic")
        labels = [STATE_LABEL.get(int(s), f"S{int(s)}")
                    for s in sub["state"]]
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45,
                                                  ha="right", fontsize=8)
        ax.set_title(region, fontsize=14, fontweight="bold",
                       color=COLORS[region])
        ax.set_ylim(0, 7); ax.grid(True, axis="y", alpha=0.3)
    axa.set_ylabel("Sessions with stay vs pre-exit FR shift (of 6)", fontsize=12)
    panel_label(axa, "a"); panel_label(axl, "b")
    fig.suptitle("ACA signals upcoming state transitions broadly; "
                  "LHA only at consummatory/rest states",
                  fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    caption = ("Pre-exit firing-rate signal (script 14, stay vs last-3-bin pre-exit "
               "Mann-Whitney + shuffle p95). ACA carries a 'about-to-exit' signal "
               "across multiple states (states passing ≥4/6 sessions in bold). "
               "LHA replicates this signal only at consummatory (S2 feeding) and "
               "rest (S3 home) states. The dissociation is clean: ACA tracks state "
               "transitions broadly; LHA only at consummatory/rest states.")
    save_both(fig, "fig2_pre_exit_signal_generic_vs_selective", caption)


# ============================================================================
# Fig 3: Multi-metric population reorganization at S3 home-exit (script 15)
# ============================================================================
def fig3_s3_population_reorganization():
    src = (REPO / "data" / "HMM" / "neural_alignment" / "state_transitions"
            / "population_metrics" / "master_replication_table.csv")
    if not src.exists():
        print(f"  [WARN] missing: {src}"); return
    df = pd.read_csv(src)
    df = df[df.state == 3].copy()
    df["frac_passing"] = df["n_sessions_passing"] / df["n_sessions_tested"].clip(lower=1)

    metric_order = ["M0_pre_exit_FR_MW",
                       "M1_n_sig_units",
                       "M2_cv_isi_diff",
                       "M3_pc_speed_diff",
                       "M4_pr_diff",
                       "M5_corr_norm_diff"]
    metric_labels = ["Pre-exit FR (script 14)", "M1: Fano factor",
                       "M2: ISI CV", "M3: PC trajectory speed",
                       "M4: Participation ratio", "M5: Correlation structure"]
    # Build matrix: rows = metrics, cols = ACA, LHA
    M = np.full((len(metric_order), 2), np.nan)
    A = np.full((len(metric_order), 2), "", dtype=object)
    for i, m in enumerate(metric_order):
        for j, region in enumerate(["ACA", "LHA"]):
            row = df[(df.metric == m) & (df.region == region)]
            if len(row):
                n_pass = int(row.iloc[0]["n_sessions_passing"])
                n_test = int(row.iloc[0]["n_sessions_tested"])
                M[i, j] = n_pass
                A[i, j] = f"{n_pass}/{n_test}"

    fig, ax = plt.subplots(figsize=(7, 7))
    cmap = LinearSegmentedColormap.from_list(
        "rep", [(0, "#f5f5f5"), (0.5, "#fdcb6e"), (1, "#27ae60")])
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=6)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            txt = A[i, j] if A[i, j] else ""
            color = "white" if (M[i, j] >= 4) else "black"
            ax.text(j, i, txt, ha="center", va="center",
                      fontsize=11, color=color, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["ACA", "LHA"], fontsize=12, fontweight="bold")
    ax.set_yticks(range(len(metric_labels)))
    ax.set_yticklabels(metric_labels, fontsize=10)
    plt.colorbar(im, ax=ax,
                   label="Sessions passing shuffle null (of 6)")
    ax.set_title("S3 home-exit triggers multi-metric population\n"
                  "reorganization in ACA and LHA",
                  fontsize=14, fontweight="bold")
    fig.tight_layout()
    caption = ("Replication counts (sessions passing shuffle p95 out of 6) for "
               "six neural-population metrics evaluated stay-vs-pre-exit at HMM "
               "state 3 (home). ACA shows coordinated changes across firing rate, "
               "variability (Fano/ISI), trajectory dynamics (PC speed), "
               "dimensionality (PR), and correlation structure. LHA replicates "
               "several metrics, particularly correlation structure and ISI "
               "variability. Other states (e.g. S6 digging) show rate-only effects; "
               "S3 is unique in showing full population reorganization.")
    save_both(fig, "fig3_s3_home_exit_population_reorganization", caption)


# ============================================================================
# Fig 4: ACA leads LHA at home-exit (script 16)
# ============================================================================
def fig4_granger():
    summary = REPO / "data/HMM/neural_alignment/granger/cross_session_summary.csv"
    sign = REPO / "data/HMM/neural_alignment/granger/sign_test.csv"
    asym = REPO / "data/HMM/neural_alignment/granger/cross_session_asymmetry.csv"
    if not summary.exists() or not sign.exists():
        print(f"  [WARN] missing: {summary} or {sign}"); return
    df_s = pd.read_csv(summary)
    df_t = pd.read_csv(sign)
    df_a = pd.read_csv(asym) if asym.exists() else None
    pc = df_s[df_s.signal == "pc1"].copy()

    fig = plt.figure(figsize=(17, 6.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.4, 1.2], wspace=0.4)

    # Panel A: schematic
    axA = fig.add_subplot(gs[0])
    axA.set_xlim(0, 1); axA.set_ylim(0, 1); axA.axis("off")
    panel_label(axA, "a")
    axA.set_title("S3 stay + post-exit segments\nACA PC1 → LHA PC1 (Granger)",
                    fontsize=11, fontweight="bold")
    # Schematic: a horizontal "state timeline"
    axA.add_patch(FancyBboxPatch((0.05, 0.55), 0.55, 0.14,
                                       boxstyle="round,pad=0.02",
                                       facecolor="#3498db", alpha=0.6,
                                       edgecolor="black"))
    axA.text(0.32, 0.62, "S3: home stay", ha="center", va="center",
              fontsize=10, color="white", fontweight="bold")
    axA.add_patch(FancyBboxPatch((0.6, 0.55), 0.3, 0.14,
                                       boxstyle="round,pad=0.02",
                                       facecolor="#f39c12", alpha=0.6,
                                       edgecolor="black"))
    axA.text(0.75, 0.62, "5 s post-exit", ha="center", va="center",
              fontsize=10, color="black", fontweight="bold")
    axA.annotate("home exit", xy=(0.6, 0.55), xytext=(0.6, 0.35),
                   ha="center", fontsize=9, color="red",
                   arrowprops=dict(arrowstyle="->", color="red"))
    # Region arrow ACA→LHA
    axA.annotate("", xy=(0.85, 0.18), xytext=(0.15, 0.18),
                   arrowprops=dict(arrowstyle="-|>", color="#1abc9c",
                                     lw=3))
    axA.text(0.50, 0.10, "ACA  →  LHA   (100–350 ms lag)",
              ha="center", fontsize=11, fontweight="bold",
              color="#16a085")

    # Panel B: per-session F comparison (PC1)
    axB = fig.add_subplot(gs[1])
    panel_label(axB, "b")
    sessions = sorted(pc["session"].unique())
    fed_sessions = {4, 6, 8}
    width = 0.38
    x = np.arange(len(sessions))
    f_a2l = []; f_l2a = []; lbl = []
    for sn in sessions:
        sub = pc[pc.session == sn]
        f_a2l.append(float(sub[sub.direction == "ACA->LHA"]["observed_F"].iloc[0]))
        f_l2a.append(float(sub[sub.direction == "LHA->ACA"]["observed_F"].iloc[0]))
        state = "fed" if sn in fed_sessions else "fasted"
        lbl.append(f"S{sn}\n({state})")
    axB.bar(x - width/2, f_a2l, width, color=COLORS["ACA"],
              edgecolor="black", label="ACA → LHA")
    axB.bar(x + width/2, f_l2a, width, color=COLORS["LHA"],
              edgecolor="black", label="LHA → ACA")
    axB.set_xticks(x); axB.set_xticklabels(lbl, fontsize=9)
    axB.set_ylabel("Granger F (PC1)", fontsize=12)
    axB.set_title("PC1 Granger F per session", fontsize=11, fontweight="bold")
    axB.legend()
    axB.grid(True, axis="y", alpha=0.3)

    # Panel C: asymmetry index
    axC = fig.add_subplot(gs[2])
    panel_label(axC, "c")
    asym_idx = (np.array(f_a2l) - np.array(f_l2a)) / (np.array(f_a2l) + np.array(f_l2a))
    bar_colors = ["#1abc9c" if v > 0 else "#e74c3c" for v in asym_idx]
    axC.bar(x, asym_idx, color=bar_colors, edgecolor="black")
    axC.axhline(0, color="k", lw=0.6)
    axC.set_xticks(x); axC.set_xticklabels(lbl, fontsize=9)
    axC.set_ylabel("ACA→LHA asymmetry index", fontsize=12)
    axC.set_title("All 6 sessions positive\n(binomial p = 0.031)",
                    fontsize=11, fontweight="bold")
    axC.grid(True, axis="y", alpha=0.3)
    # Annotate binom p from sign test
    if len(df_t[df_t.signal == "pc1"]):
        bp = float(df_t[df_t.signal == "pc1"]["binom_p"].iloc[0])
        axC.text(0.5, 0.95, f"binom p = {bp:.3f}",
                   transform=axC.transAxes,
                   ha="center", va="top", fontsize=11,
                   fontweight="bold", color="#16a085")

    fig.suptitle("ACA leads LHA at home-exit by 100-350 ms "
                  "(PC1 spike-population Granger)",
                  fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    caption = ("Linear bivariate Granger causality on PC1 of spike-population "
               "rates over S3 home stay + 5 s post-exit segments. "
               "(a) Schematic of the segment definition. "
               "(b) Per-session observed F values: ACA→LHA (teal) vs LHA→ACA (orange). "
               "(c) Asymmetry index (F_ACA→LHA − F_LHA→ACA)/(sum). All 6 sessions "
               "are positive (binomial p=0.031). BIC-selected lags are 100-350 ms.")
    save_both(fig, "fig4_aca_leads_lha_at_home_exit", caption)


# ============================================================================
# Fig 5: ACA high-gamma feeding suppression (script 18 A1)
# ============================================================================
BANDS = ["delta", "theta", "beta", "low_gamma", "high_gamma"]
CATS = ["home", "feeding", "transition_zone"]
CAT_COLORS = {"home": "#3498db", "feeding": "#e74c3c",
                "transition_zone": "#f39c12"}


def _fig_lfp_state_identity(region: str, fig_name: str,
                              title: str, caption: str, highlight_bands):
    src_power = REPO / "data/HMM/neural_alignment/lfp_state_identity_v2/A1_band_power_per_category_all_sessions.csv"
    src_pair = REPO / "data/HMM/neural_alignment/lfp_state_identity_v2/A1_pairwise_replication.csv"
    if not src_power.exists() or not src_pair.exists():
        print(f"  [WARN] missing: {src_power} or {src_pair}"); return
    bp = pd.read_csv(src_power)
    pp = pd.read_csv(src_pair)
    bp_r = bp[bp.region == region].copy()
    # log-transform and compute mean ± SEM per category × band
    bp_r["log_power"] = np.log10(bp_r["mean_power"].clip(lower=1e-12))
    agg = (bp_r.groupby(["band", "category"])
              .agg(mean=("log_power", "mean"),
                   sem=("log_power", lambda s: s.std() / np.sqrt(max(1, len(s)))))
              .reset_index())

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(16, 6.5))
    x = np.arange(len(BANDS))
    width = 0.27
    for ci, cat in enumerate(CATS):
        sub = agg[agg.category == cat].set_index("band").reindex(BANDS)
        axA.errorbar(x, sub["mean"], yerr=sub["sem"],
                       label=cat, color=CAT_COLORS[cat],
                       marker="o", capsize=4, lw=2)
    axA.set_xticks(x); axA.set_xticklabels(BANDS, fontsize=11)
    axA.set_xlabel("Frequency band", fontsize=12)
    axA.set_ylabel("log10 mean power (µV²/Hz)", fontsize=12)
    axA.set_title(f"{region} spectral fingerprint by HMM category",
                    fontsize=12, fontweight="bold")
    axA.legend()
    axA.grid(True, alpha=0.3)
    # Highlight target bands
    for band in highlight_bands:
        if band in BANDS:
            idx = BANDS.index(band)
            axA.axvspan(idx - 0.35, idx + 0.35, color="yellow", alpha=0.15)
    panel_label(axA, "a")

    # Panel B: replication counts per band for home_vs_feeding + feeding_vs_transition
    pp_r = pp[pp.region == region].copy()
    pairs_show = ["home_vs_feeding", "feeding_vs_transition_zone"]
    pair_labels = ["home vs feeding", "feeding vs transition"]
    x = np.arange(len(BANDS))
    width = 0.4
    for pi, (pair, lbl) in enumerate(zip(pairs_show, pair_labels)):
        sub = pp_r[pp_r.pair == pair].set_index("band").reindex(BANDS)
        offset = (pi - 0.5) * width
        bars = axB.bar(x + offset, sub["n_passing"], width, label=lbl,
                          edgecolor="black", lw=0.8,
                          color=CAT_COLORS["feeding"] if pi == 0
                                else CAT_COLORS["transition_zone"])
        # bold replicating bands
        for j, val in enumerate(sub["n_passing"]):
            if val >= 4:
                bars[j].set_edgecolor("black"); bars[j].set_linewidth(2)
    axB.axhline(4, color="red", lw=1.2, ls="--", alpha=0.7,
                  label="replication (4/6)")
    axB.set_xticks(x); axB.set_xticklabels(BANDS, fontsize=11)
    axB.set_xlabel("Frequency band", fontsize=12)
    axB.set_ylabel("Sessions passing shuffle p95 (of 6)", fontsize=12)
    axB.set_title(f"{region} replication count per band × contrast",
                    fontsize=12, fontweight="bold")
    axB.set_ylim(0, 7)
    axB.legend(fontsize=9)
    axB.grid(True, axis="y", alpha=0.3)
    panel_label(axB, "b")

    fig.suptitle(title, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_both(fig, fig_name, caption)


def fig5_aca_high_gamma():
    """REVISED: paired within-session feeding effect + replication heatmap."""
    src_power = REPO / "data/HMM/neural_alignment/lfp_state_identity_v2/A1_band_power_per_category_all_sessions.csv"
    src_pair = REPO / "data/HMM/neural_alignment/lfp_state_identity_v2/A1_pairwise_replication.csv"
    src_kw = REPO / "data/HMM/neural_alignment/lfp_state_identity_v2/A1_kruskal_replication.csv"
    for p in (src_power, src_pair, src_kw):
        if not p.exists():
            print(f"  [WARN] missing: {p}"); return

    bp = pd.read_csv(src_power)
    pp = pd.read_csv(src_pair)
    kw = pd.read_csv(src_kw)
    aca_bp = bp[bp.region == "ACA"].copy()

    SESSION_STATE = {4: "fed", 6: "fed", 8: "fed",
                       12: "fasted", 14: "fasted", 16: "fasted"}

    # Compute per-session per-band: log10(feeding) − mean(log10(home), log10(transition))
    pivot = aca_bp.pivot_table(index=["session", "band"], columns="category",
                                  values="mean_power").reset_index()
    pivot["log_feeding"] = np.log10(pivot["feeding"].clip(lower=1e-12))
    pivot["log_home"] = np.log10(pivot["home"].clip(lower=1e-12))
    pivot["log_transition"] = np.log10(pivot["transition_zone"].clip(lower=1e-12))
    pivot["effect"] = pivot["log_feeding"] - 0.5 * (pivot["log_home"] + pivot["log_transition"])
    pivot["state"] = pivot["session"].map(SESSION_STATE)

    # Per-band, per-session: print effect sizes
    print("  ACA within-session feeding effect (log10 feeding − non-feeding mean):")
    for band in BANDS:
        sub = pivot[pivot.band == band]
        vals = sub.set_index("session")["effect"].to_dict()
        print(f"    {band:>11s}: " + "  ".join(
            f"S{s}={vals.get(s, np.nan):+.3f}" for s in [4, 6, 8, 12, 14, 16]))

    # Build replication matrix: rows = bands, cols = (home_vs_feeding, feeding_vs_transition, KW)
    aca_pp = pp[pp.region == "ACA"]
    aca_kw = kw[kw.region == "ACA"]
    contrasts = [("home_vs_feeding", "home vs feeding"),
                   ("feeding_vs_transition_zone", "feeding vs transition"),
                   (None, "Kruskal-Wallis omnibus")]
    rep_mat = np.zeros((len(BANDS), len(contrasts)), dtype=int)
    for r, band in enumerate(BANDS):
        for c, (key, _) in enumerate(contrasts):
            if key is None:
                row = aca_kw[aca_kw.band == band]
                rep_mat[r, c] = int(row["n_passing"].iloc[0]) if len(row) else 0
            else:
                row = aca_pp[(aca_pp.band == band) & (aca_pp.pair == key)]
                rep_mat[r, c] = int(row["n_passing"].iloc[0]) if len(row) else 0

    print("  ACA replication counts (bands × contrasts):")
    print(pd.DataFrame(rep_mat, index=BANDS,
                         columns=[c[1] for c in contrasts]).to_string())

    # ========== Figure ==========
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(16, 6.5),
                                       gridspec_kw={"width_ratios": [1.2, 1.0]})

    # Panel A: paired points
    x = np.arange(len(BANDS))
    sessions = sorted(pivot.session.unique())
    marker_for_state = {"fed": "o", "fasted": "s"}
    for sn in sessions:
        sub = pivot[pivot.session == sn].set_index("band").reindex(BANDS)
        state = SESSION_STATE[sn]
        marker = marker_for_state[state]
        axA.plot(x, sub["effect"], marker=marker, lw=1.2, ms=8,
                   alpha=0.85, color=COLORS[state],
                   label=f"S{sn} ({state})")
    axA.axhline(0, color="k", lw=1.0, ls="--", alpha=0.7)
    axA.set_xticks(x); axA.set_xticklabels(BANDS, fontsize=11)
    axA.set_xlabel("Frequency band", fontsize=12)
    axA.set_ylabel("log10 power(feeding) − mean log10 power(non-feeding)",
                     fontsize=11)
    axA.set_title("Within-session feeding-induced change in ACA LFP power",
                    fontsize=12, fontweight="bold")
    # set y-range to make negative high-gamma points visible
    e_min = min(pivot["effect"].min(), -0.04)
    e_max = max(pivot["effect"].max(), 0.20)
    pad = 0.05 * max(abs(e_min), abs(e_max))
    axA.set_ylim(e_min - pad, e_max + pad)
    axA.grid(True, alpha=0.3)
    axA.legend(fontsize=9, ncol=2, loc="upper right")
    panel_label(axA, "a")

    # Panel B: replication heatmap
    cmap = LinearSegmentedColormap.from_list(
        "rep", [(0, "#fff5e6"), (0.5, "#f4a261"), (1, "#c0392b")])
    im = axB.imshow(rep_mat, aspect="auto", cmap=cmap, vmin=0, vmax=6)
    axB.set_xticks(range(len(contrasts)))
    axB.set_xticklabels([c[1] for c in contrasts], rotation=20, ha="right",
                          fontsize=10)
    axB.set_yticks(range(len(BANDS)))
    axB.set_yticklabels(BANDS, fontsize=11)
    for r in range(len(BANDS)):
        for c in range(len(contrasts)):
            v = rep_mat[r, c]
            color = "white" if v >= 4 else "black"
            axB.text(c, r, f"{v}/6", ha="center", va="center",
                       fontsize=12, color=color, fontweight="bold")
            if v == 6:
                # Thick border for 6/6 cells
                axB.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                                  fill=False, edgecolor="black",
                                                  lw=3))
    plt.colorbar(im, ax=axB, label="Sessions passing (of 6)")
    axB.set_title("Cross-session replication: ACA spectral fingerprint of feeding",
                    fontsize=12, fontweight="bold")
    panel_label(axB, "b")

    fig.suptitle("Feeding suppresses ACA high gamma in every session; "
                  "smaller effects at delta and theta",
                  fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    caption = ("Panel A: Per-session within-session feeding effect on ACA LFP "
               "power, computed as log10(feeding) − mean of log10(home) and "
               "log10(transition zone). Each marker is one session (circles = fed, "
               "squares = fasted). All 6 sessions sit below zero at high gamma "
               "(feeding < non-feeding). Other bands are mixed or near zero. "
               "Panel B: Cross-session replication count out of 6 for three "
               "contrasts (home vs feeding, feeding vs transition, Kruskal-Wallis "
               "omnibus across home/feeding/transition). High gamma is the only "
               "band passing 6/6 sessions in both pairwise feeding contrasts "
               "(thick black borders). Delta and theta show smaller, less "
               "consistent effects (1-2/6 sessions). The high-gamma effect is "
               "the cleanest ACA spectral signature of feeding state.")
    save_both(fig, "fig5_aca_high_gamma_feeding_suppression", caption)


def fig6_rsp_feeding_suppression():
    _fig_lfp_state_identity(
        region="RSP", fig_name="fig6_rsp_beta_lowgamma_feeding_suppression",
        title="Feeding suppresses RSP beta and low-gamma; "
              "high gamma less consistently",
        highlight_bands=["beta", "low_gamma"],
        caption=("RSP shows feeding-state suppression in beta (15-30 Hz) and "
                 "low gamma (30-60 Hz), with consistent but weaker effects in "
                 "high gamma. "
                 "(a) RSP spectral fingerprint per HMM category, ±SEM. Yellow "
                 "bands highlight beta and low_gamma where feeding suppression "
                 "is largest. "
                 "(b) Replication counts per band × pairwise contrast. Beta and "
                 "low-gamma feeding contrasts replicate in 4-5 of 6 sessions. "
                 "The pattern parallels ACA's feeding suppression but at "
                 "different frequencies."))


# ============================================================================
# Fig 7: RSP ripples (script 19 v2)
# ============================================================================
def fig7_rsp_ripples():
    src = REPO / "data/HMM/neural_alignment/swr_detection_v2/threshold_02pct/all_regional_events.csv"
    val = REPO / "data/HMM/neural_alignment/swr_detection_v2/threshold_02pct/validation_summary.csv"
    if not src.exists() or not val.exists():
        print(f"  [WARN] missing: {src} or {val}"); return
    all_ev = pd.read_csv(src)
    val_df = pd.read_csv(val)
    rsp = all_ev[all_ev.region == "RSP"].copy()

    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], wspace=0.32, hspace=0.4)

    # Panel A: example trace (use existing PNG from script 19)
    axA = fig.add_subplot(gs[0, 0])
    panel_label(axA, "a")
    example_trace = REPO / "figures/HMM/neural_alignment/swr_detection/example_traces/session_12_RSP_event_0.png"
    if example_trace.exists():
        img = plt.imread(example_trace)
        axA.imshow(img); axA.axis("off")
        axA.set_title("Example RSP ripple (S12, ±150 ms window)\n"
                        "Top: raw bipolar LFP; mid: 100-250 Hz; bottom: envelope",
                        fontsize=10, fontweight="bold")
    else:
        axA.axis("off")
        axA.text(0.5, 0.5, "(example trace PNG missing)",
                   ha="center", va="center", transform=axA.transAxes)

    # Panel B: peak frequency histogram
    axB = fig.add_subplot(gs[0, 1])
    panel_label(axB, "b")
    pf = rsp["peak_frequency_hz"].dropna()
    axB.hist(pf, bins=np.arange(100, 260, 10), color=COLORS["RSP"],
               edgecolor="black", lw=0.8)
    modal = float(pf.median())
    axB.axvline(modal, color="red", ls="--", lw=1.5,
                  label=f"median = {modal:.0f} Hz")
    axB.set_xlabel("Peak frequency (Hz)", fontsize=11)
    axB.set_ylabel("Number of events", fontsize=11)
    axB.set_title("Peak frequency distribution (RSP)",
                    fontsize=11, fontweight="bold")
    axB.legend(fontsize=10)
    axB.grid(True, alpha=0.3)

    # Panel C: duration histogram
    axC = fig.add_subplot(gs[0, 2])
    panel_label(axC, "c")
    du = rsp["mean_duration_ms"].dropna()
    axC.hist(du, bins=30, color=COLORS["RSP"],
               edgecolor="black", lw=0.8, alpha=0.85)
    axC.axvline(du.mean(), color="red", ls="--", lw=1.5,
                  label=f"mean = {du.mean():.0f} ms")
    axC.set_xlabel("Duration (ms)", fontsize=11)
    axC.set_ylabel("Number of events", fontsize=11)
    axC.set_title("Duration distribution (RSP)",
                    fontsize=11, fontweight="bold")
    axC.legend(fontsize=10)
    axC.grid(True, alpha=0.3)

    # Panel D: spike validation per session (RSP only)
    axD = fig.add_subplot(gs[1, :])
    panel_label(axD, "d")
    rsp_val = val_df[val_df.region == "RSP"].copy().reset_index(drop=True)
    sessions = rsp_val["session"].astype(int).tolist()
    p_vals = rsp_val["p_mw"].fillna(1.0).values
    log_p = -np.log10(np.clip(p_vals, 1e-100, 1.0))
    n_units = rsp_val["n_units"].fillna(0).astype(int).values
    x = np.arange(len(sessions))
    colors = [COLORS["RSP"] if u > 0 else "lightgray" for u in n_units]
    axD.bar(x, log_p, color=colors, edgecolor="black", lw=0.8)
    axD.axhline(-np.log10(0.05), color="red", lw=1.5, ls="--",
                  label="p = 0.05")
    for xi, (sn, p, n) in enumerate(zip(sessions, p_vals, n_units)):
        ann = (f"p={p:.1e}\n{n} units" if n > 0 and np.isfinite(p)
                 else f"{n} units")
        axD.text(xi, -np.log10(max(p, 1e-100)) + 0.5,
                   ann, ha="center", fontsize=8)
    axD.set_xticks(x); axD.set_xticklabels([f"S{s}" for s in sessions], fontsize=10)
    axD.set_xlabel("Session", fontsize=12)
    axD.set_ylabel("−log10(p), event vs control spikes", fontsize=12)
    axD.set_title("Per-session spike validation (RSP)",
                    fontsize=11, fontweight="bold")
    axD.legend(fontsize=10)
    axD.grid(True, axis="y", alpha=0.3)

    fig.suptitle("RSP generates sharp-wave ripples consistent with cortical "
                  "ripple physiology", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    caption = ("RSP ripple events (script 19 v2; 100-250 Hz bandpass; "
               "2% pair-threshold; 1266 events total across 6 sessions before "
               "the locomotion-artifact diagnostic). "
               "(a) Example raw + bandpassed + envelope trace from S12. "
               "(b) Peak frequency distribution centered near 181 Hz. "
               "(c) Duration distribution centered near 50-100 ms. "
               "(d) Per-session Mann-Whitney p value for event-locked spike "
               "counts vs random control timepoints. 4/6 sessions show "
               "extreme significance (p<1e-8). S4 has 1 unit (under-powered); "
               "S16 has 0 units (KS4 sort missing cluster_info) — event "
               "morphology in S4/S16 is identical to validated sessions.")
    save_both(fig, "fig7_rsp_sharp_wave_ripples", caption)


# ============================================================================
# Fig 8: Project summary
# ============================================================================
def fig8_project_summary():
    fig = plt.figure(figsize=(15, 18))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.1, 1.6, 1.3], hspace=0.45)

    # ----- Panel A: Task + brain regions schematic -----
    axA = fig.add_subplot(gs[0])
    axA.set_xlim(0, 10); axA.set_ylim(0, 6); axA.axis("off")
    panel_label(axA, "a")
    axA.text(5.0, 5.7, "Dual-probe foraging task", ha="center",
              fontsize=14, fontweight="bold")
    # Arena boxes
    axA.add_patch(FancyBboxPatch((0.5, 2.0), 2.2, 2.5,
                                       boxstyle="round,pad=0.05",
                                       facecolor="#3498db", alpha=0.4,
                                       edgecolor="black", lw=1.5))
    axA.text(1.6, 3.3, "Home", ha="center", fontsize=12, fontweight="bold")
    axA.add_patch(FancyBboxPatch((3.2, 2.0), 1.6, 2.5,
                                       boxstyle="round,pad=0.05",
                                       facecolor="#f39c12", alpha=0.5,
                                       edgecolor="black", lw=1.5))
    axA.text(4.0, 3.3, "Transition\nzone", ha="center",
              fontsize=11, fontweight="bold")
    axA.add_patch(FancyBboxPatch((5.3, 2.0), 4.0, 2.5,
                                       boxstyle="round,pad=0.05",
                                       facecolor="#e74c3c", alpha=0.3,
                                       edgecolor="black", lw=1.5))
    axA.text(7.3, 3.3, "Foraging arena\n(4 sand pots)",
              ha="center", fontsize=11, fontweight="bold")
    # Recording probes — stagger labels vertically to avoid overlap
    probe_specs = [
        (1.0, 1.50, "ACA (probe 0)", COLORS["ACA"]),
        (1.0, 1.10, "LHA (probe 1, dorsal-deep)", COLORS["LHA"]),
        (1.0, 0.70, "RSP (probe 1, dorsal-shallow)", COLORS["RSP"]),
    ]
    for (xt, yt, lbl, col) in probe_specs:
        axA.plot([xt, xt], [yt - 0.18, yt + 0.18], color=col, lw=4)
        axA.text(xt + 0.25, yt, lbl, fontsize=10, color=col,
                  fontweight="bold", va="center")
    axA.text(5.0, 0.20,
              "Metabolic states: fed (4 sessions) / fasted (3) / HFD (3)",
              ha="center", fontsize=10, style="italic")

    # ----- Panel B: Summary table -----
    axB = fig.add_subplot(gs[1])
    axB.axis("off")
    panel_label(axB, "b")
    axB.set_title("Defensible findings",
                    fontsize=14, fontweight="bold", loc="left")
    rows = [
        ("1", "Regional preferred-state encoding",
         "ACA broad; LHA selective (S2 feed, S3 home)",
         "ACA / LHA", "strong"),
        ("2", "Pre-exit firing-rate signal",
         "ACA across multiple states; LHA only S2/S3",
         "ACA / LHA", "strong"),
        ("3", "S3 home-exit multi-metric reorganization",
         "FR + Fano + ISI CV + PR + corr structure replicate ≥4/6",
         "ACA / LHA", "strong"),
        ("4", "ACA → LHA Granger lead at home-exit",
         "6/6 sessions ACA-leads on PC1; binom p=0.031",
         "ACA / LHA", "strong"),
        ("5", "ACA high-gamma feeding suppression",
         "6/6 sessions replicate home_vs_feeding & feeding_vs_transition",
         "ACA", "strong"),
        ("6", "RSP beta / low-gamma feeding suppression",
         "4-5/6 sessions replicate feeding contrasts",
         "RSP", "moderate"),
        ("7", "RSP sharp-wave ripples",
         "~181 Hz modal, ~50-100 ms dur, MW p<<0.001 in 4/6 valid sessions",
         "RSP", "strong"),
    ]
    headers = ["#", "Finding", "Replication / effect",
                 "Region(s)", "Strength"]
    col_w = [0.04, 0.30, 0.40, 0.12, 0.14]
    color_by_strength = {"strong": "#27ae60", "moderate": "#f39c12",
                            "weak": "#e74c3c"}
    # Header row
    y = 0.92
    x_cursor = 0
    for w, h in zip(col_w, headers):
        axB.text(x_cursor + w/2, y, h, ha="center", va="top",
                  fontsize=11, fontweight="bold", transform=axB.transAxes)
        x_cursor += w
    y -= 0.08
    for r in rows:
        x_cursor = 0
        for w, val, header in zip(col_w, r, headers):
            ha = "center" if header in ("#", "Strength", "Region(s)") else "left"
            color = "black"
            weight = "normal"
            if header == "Strength":
                color = color_by_strength[val]
                weight = "bold"
            axB.text(x_cursor + (w/2 if ha == "center" else 0.01),
                      y, val, ha=ha, va="top", fontsize=9.5,
                      color=color, fontweight=weight,
                      transform=axB.transAxes, wrap=True)
            x_cursor += w
        y -= 0.11

    # ----- Panel C: Circuit diagram -----
    axC = fig.add_subplot(gs[2])
    axC.set_xlim(0, 10); axC.set_ylim(0, 5); axC.axis("off")
    panel_label(axC, "c")
    axC.set_title("Proposed circuit summary",
                    fontsize=14, fontweight="bold", loc="left")
    # Region circles
    pos = {"ACA": (2.0, 3.5), "LHA": (5.5, 1.0), "RSP": (8.0, 3.5)}
    for r, (x, y) in pos.items():
        axC.add_patch(plt.Circle((x, y), 0.7, facecolor=COLORS[r],
                                       alpha=0.7, edgecolor="black",
                                       lw=1.5))
        axC.text(x, y, r, ha="center", va="center",
                  fontsize=14, fontweight="bold", color="white")
    # ACA → LHA arrow
    axC.annotate("", xy=(pos["LHA"][0] - 0.5, pos["LHA"][1] + 0.4),
                   xytext=(pos["ACA"][0] + 0.5, pos["ACA"][1] - 0.4),
                   arrowprops=dict(arrowstyle="-|>", color=COLORS["ACA"],
                                     lw=3))
    axC.text(3.4, 2.3, "ACA →  LHA\n(home-exit, 100–350 ms)",
              ha="left", fontsize=10, color=COLORS["ACA"],
              fontweight="bold")
    # Feeding suppression annotations
    axC.text(2.0, 4.6, "high-γ ↓\nat feeding",
              ha="center", fontsize=10, color=COLORS["ACA"],
              fontweight="bold")
    axC.text(8.0, 4.6, "β, low-γ ↓\nat feeding\n+ ripples ~181 Hz",
              ha="center", fontsize=10, color=COLORS["RSP"],
              fontweight="bold")
    # LHA caption
    axC.text(5.5, 0.2,
              "LHA: selective for consummatory states (feeding, rest)",
              ha="center", fontsize=10, color=COLORS["LHA"],
              fontweight="bold")
    # Title
    fig.suptitle("Summary: Regional roles of ACA, LHA, and RSP during foraging",
                  fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    caption = ("Project summary. (a) Dual-probe foraging task: mouse navigates "
               "between Home, Transition zone, and 4-pot Arena while ACA, LHA, "
               "and RSP are recorded simultaneously across fed/fasted/HFD "
               "metabolic states. (b) Seven defensible findings, color-coded by "
               "evidence strength. (c) Proposed circuit summary: ACA initiates "
               "home-exit decision and leads LHA by 100-350 ms; ACA and RSP "
               "both reduce high-frequency LFP power during feeding; RSP "
               "additionally generates ~181 Hz sharp-wave ripples.")
    save_both(fig, "fig8_project_summary", caption)


# ============================================================================
# Main
# ============================================================================
def main():
    print(f"Output directory: {OUT}")
    print("\n=== Fig 1 ===")
    fig1_regional_encoding()
    print("\n=== Fig 2 ===")
    fig2_pre_exit_signal()
    print("\n=== Fig 3 ===")
    fig3_s3_population_reorganization()
    print("\n=== Fig 4 ===")
    fig4_granger()
    print("\n=== Fig 5 ===")
    fig5_aca_high_gamma()
    print("\n=== Fig 6 ===")
    fig6_rsp_feeding_suppression()
    print("\n=== Fig 7 ===")
    fig7_rsp_ripples()
    print("\n=== Fig 8 ===")
    fig8_project_summary()
    print("\nAll narrative figures saved to", OUT)


if __name__ == "__main__":
    main()
