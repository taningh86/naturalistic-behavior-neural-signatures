"""Quick aggregated stats from the all-sessions summary."""
import pandas as pd
import numpy as np

df = pd.read_csv("data/sp_coor1_dynamics/all_sessions_summary.csv")
print("rows:", len(df), " cols:", len(df.columns))
print("phase_type counts:")
print(df['phase_type'].value_counts())
print()

sub = df[df['phase_type'].isin(['rising', 'falling'])]
print("Rising+falling means by state and phase_type:")
print(sub.groupby(['state', 'phase_type'])[
    ['mean_speed_LHA', 'mean_speed_RSP', 'mean_curv_LHA', 'mean_curv_RSP',
     'n_units_LHA', 'n_units_RSP']
].mean().round(3))
print()

print("Per-session unit counts:")
print(df.groupby(['session', 'state'])[['n_units_LHA', 'n_units_RSP']].first())
print()

# State-level contrast (session means then group means) on rising+falling
print("Session-level means on rising+falling phases:")
sess_means = sub.groupby(['session', 'state'])[
    ['mean_speed_LHA', 'mean_speed_RSP', 'mean_curv_LHA', 'mean_curv_RSP']
].mean().reset_index()
print(sess_means)
print()
print("State-level means (across sessions):")
print(sess_means.groupby('state').agg(['mean', 'std']).round(4))
