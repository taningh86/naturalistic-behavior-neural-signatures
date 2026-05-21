"""Regenerate the Layer 1a summary figure from the saved CSV."""
import pandas as pd
from sp_manifold_layer1a import make_summary_figure

df = pd.read_csv("data/sp_coor1_dynamics/manifold_layer1a.csv")
make_summary_figure(df)
