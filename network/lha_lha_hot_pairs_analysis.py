"""
LHA-LHA Hot Pairs Analysis: Identify unit pairs with biggest correlation changes (fed vs fasted)
Session-level tracking to show consistency across recordings
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("="*80)
print("LHA-LHA HOT PAIRS ANALYSIS (Session-Level Tracking)")
print("="*80)

# Load connectivity data
print("\nLoading connectivity data...")
fed_exp = pd.read_csv("data/lha_lha_connectivity_fed_exploration.csv")
fasted_exp = pd.read_csv("data/lha_lha_connectivity_fasted_exploration.csv")
fed_for = pd.read_csv("data/lha_lha_connectivity_fed_foraging.csv")
fasted_for = pd.read_csv("data/lha_lha_connectivity_fasted_foraging.csv")

# ============================================================================
# EXPLORATION PHASE ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("EXPLORATION PHASE: Sessions 1,3 (Fed) vs Sessions 5,7 (Fasted)")
print("="*80)

def analyze_phase_pairs(fed_df, fasted_df, phase_name, lags=[10, 50]):
    """Analyze which pairs show consistent changes across sessions."""

    results_by_lag = {}

    for lag in lags:
        corr_col = f'correlation_{lag}ms'
        print(f"\n{'-'*80}")
        print(f"{phase_name} - LAG: {lag}ms")
        print(f"{'-'*80}")

        # Create unit pair identifiers
        fed_df_copy = fed_df.copy()
        fasted_df_copy = fasted_df.copy()

        fed_df_copy['pair_id'] = fed_df_copy.apply(
            lambda row: tuple(sorted([row['unit_1'], row['unit_2']])), axis=1
        )
        fasted_df_copy['pair_id'] = fasted_df_copy.apply(
            lambda row: tuple(sorted([row['unit_1'], row['unit_2']])), axis=1
        )

        # Find pairs present in both fed and fasted
        fed_pairs = set(fed_df_copy['pair_id'])
        fasted_pairs = set(fasted_df_copy['pair_id'])
        common_pairs = fed_pairs & fasted_pairs

        print(f"Fed pairs: {len(fed_pairs)}")
        print(f"Fasted pairs: {len(fasted_pairs)}")
        print(f"Common pairs: {len(common_pairs)}")

        # Calculate changes for each pair
        pair_changes = []

        for pair_id in common_pairs:
            fed_rows = fed_df_copy[fed_df_copy['pair_id'] == pair_id]
            fasted_rows = fasted_df_copy[fasted_df_copy['pair_id'] == pair_id]

            if len(fed_rows) > 0 and len(fasted_rows) > 0:
                fed_corr = fed_rows[corr_col].values[0]
                fasted_corr = fasted_rows[corr_col].values[0]
                delta_corr = fasted_corr - fed_corr

                pair_changes.append({
                    'unit_1': pair_id[0],
                    'unit_2': pair_id[1],
                    'fed_corr': fed_corr,
                    'fasted_corr': fasted_corr,
                    'delta_corr': delta_corr,
                    'abs_delta': abs(delta_corr),
                    'direction': 'increased' if delta_corr > 0 else 'decreased'
                })

        # Sort by absolute change magnitude
        pair_changes_df = pd.DataFrame(pair_changes)
        pair_changes_df = pair_changes_df.sort_values('abs_delta', ascending=False)

        print(f"\nTop 30 pairs with BIGGEST CHANGES (10 increased, 10 decreased):")
        print(f"\n--- TOP INCREASES (Fasted >> Fed) ---")
        top_increases = pair_changes_df[pair_changes_df['direction'] == 'increased'].head(10)
        print(top_increases[['unit_1', 'unit_2', 'fed_corr', 'fasted_corr', 'delta_corr']].to_string(index=False))

        print(f"\n--- TOP DECREASES (Fasted << Fed) ---")
        top_decreases = pair_changes_df[pair_changes_df['direction'] == 'decreased'].head(10)
        print(top_decreases[['unit_1', 'unit_2', 'fed_corr', 'fasted_corr', 'delta_corr']].to_string(index=False))

        # Statistics
        print(f"\n--- STATISTICS ---")
        print(f"Mean Delta_corr: {pair_changes_df['delta_corr'].mean():.6f}")
        print(f"Median Delta_corr: {pair_changes_df['delta_corr'].median():.6f}")
        print(f"Std Delta_corr: {pair_changes_df['delta_corr'].std():.6f}")
        print(f"Min Delta_corr: {pair_changes_df['delta_corr'].min():.6f} (pair {pair_changes_df.loc[pair_changes_df['delta_corr'].idxmin(), ['unit_1', 'unit_2']].values})")
        print(f"Max Delta_corr: {pair_changes_df['delta_corr'].max():.6f} (pair {pair_changes_df.loc[pair_changes_df['delta_corr'].idxmax(), ['unit_1', 'unit_2']].values})")

        # Distribution
        n_increased = len(pair_changes_df[pair_changes_df['delta_corr'] > 0])
        n_decreased = len(pair_changes_df[pair_changes_df['delta_corr'] < 0])
        print(f"Pairs with increased correlation (fasted): {n_increased} ({n_increased/len(pair_changes_df)*100:.1f}%)")
        print(f"Pairs with decreased correlation (fasted): {n_decreased} ({n_decreased/len(pair_changes_df)*100:.1f}%)")

        # Save results
        results_by_lag[lag] = pair_changes_df

        # Save to CSV
        output_file = Path(f"data/lha_lha_hot_pairs_{phase_name.lower().replace(' ', '_')}_{lag}ms.csv")
        pair_changes_df.to_csv(output_file, index=False)
        print(f"\n[OK] Saved to: {output_file}")

    return results_by_lag

# Analyze exploration phase
exp_results = analyze_phase_pairs(fed_exp, fasted_exp, "EXPLORATION")

# ============================================================================
# FORAGING PHASE ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("FORAGING PHASE: Sessions 2,4 (Fed) vs Sessions 6,8 (Fasted)")
print("="*80)

for_results = analyze_phase_pairs(fed_for, fasted_for, "FORAGING")

# ============================================================================
# COMPARISON: Exploration vs Foraging
# ============================================================================

print("\n" + "="*80)
print("PHASE COMPARISON")
print("="*80)

for lag in [10, 50]:
    print(f"\n--- LAG {lag}ms ---")
    exp_df = exp_results[lag]
    for_df = for_results[lag]

    print(f"Exploration - Mean Delta_corr: {exp_df['delta_corr'].mean():.6f}, N pairs: {len(exp_df)}")
    print(f"Foraging    - Mean Delta_corr: {for_df['delta_corr'].mean():.6f}, N pairs: {len(for_df)}")

print("\n" + "="*80)
print("[DONE] LHA-LHA hot pairs analysis complete!")
print("="*80)
