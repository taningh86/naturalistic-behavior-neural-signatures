import pandas as pd

df = pd.read_csv("data/sp_coor1_dynamics/manifold_layer1a.csv")
print(df.to_string(index=False))
print("\n--- State means ---")
print(df.groupby(["region", "state"])[["n_units", "PR", "TwoNN", "CorrDim", "Isomap"]].mean().round(2))
