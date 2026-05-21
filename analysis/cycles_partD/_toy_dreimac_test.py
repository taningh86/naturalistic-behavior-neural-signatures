"""
Toy sanity test: dreimac on a noisy 2D circle.
Verifies that CircularCoords produces a phase that correlates with the true angle.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import pycircstat as pc
from dreimac import CircularCoords

figdir = Path(__file__).parent / "figures_toy"
figdir.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)
N = 500
true_theta = rng.uniform(0, 2 * np.pi, N)
R = 1.0
noise = 0.05
X = np.column_stack([R * np.cos(true_theta) + rng.normal(0, noise, N),
                     R * np.sin(true_theta) + rng.normal(0, noise, N)])

print(f"Toy data: {X.shape}, noise={noise}")

# Fit circular coordinates. n_landmarks: how many landmarks maxmin selects.
cc = CircularCoords(X, n_landmarks=200, maxdim=1)
# get_coordinates takes a cocycle index (0 = most persistent H1 class)
phi = cc.get_coordinates(cocycle_idx=0)  # length N
print(f"Phase output shape: {phi.shape}, range [{phi.min():.3f}, {phi.max():.3f}]")
print(f"Phase std: {phi.std():.3f} (expect > 0.5 for a real cycle)")

# Compare to true angle via circular correlation
phi_wrapped = np.mod(phi, 2 * np.pi)
true_wrapped = np.mod(true_theta, 2 * np.pi)
r_cc = pc.corrcc(phi_wrapped, true_wrapped)
print(f"Circular correlation (phi, true theta) = {r_cc:.3f}")

# Also check reverse direction (phi may be reflected)
r_cc_rev = pc.corrcc(phi_wrapped, (2 * np.pi - true_wrapped))
print(f"  reflected: {r_cc_rev:.3f}")

r_best = max(abs(r_cc), abs(r_cc_rev))
print(f"Best |r| = {r_best:.3f} (expect > 0.8 for clean circle)")

# Figure
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].scatter(X[:, 0], X[:, 1], c=true_theta, cmap='hsv', s=8)
axes[0].set_title("Toy data colored by true angle")
axes[0].set_aspect('equal')

axes[1].scatter(X[:, 0], X[:, 1], c=phi_wrapped, cmap='hsv', s=8)
axes[1].set_title(f"Colored by dreimac phi (|r|={r_best:.2f})")
axes[1].set_aspect('equal')

axes[2].scatter(true_wrapped, phi_wrapped, s=8, alpha=0.6)
axes[2].set_xlabel("True angle")
axes[2].set_ylabel("dreimac phi")
axes[2].set_title("True vs recovered")

plt.tight_layout()
out = figdir / "toy_dreimac_circle.png"
plt.savefig(out, dpi=120)
print(f"Saved {out}")

if r_best > 0.8:
    print("\n>>> PASS: dreimac recovers circular structure on toy data.")
else:
    print(f"\n>>> FAIL: best |r|={r_best:.3f} < 0.8. Something is wrong.")
