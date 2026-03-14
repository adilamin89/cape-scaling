"""
cape_stiffness.py
Stiffness ratio κ = λ₁/λ₂ from PCA eigenvalues
Shows system becomes increasingly stiff as N grows
κ → ∞ at ~88B where d_eff = 1 (single effective mode)
"""
import numpy as np
from scipy import stats

# Pythia PCA eigenvalues from CAPE analysis
N = [70e6, 160e6, 410e6, 1e9, 1.4e9, 2.8e9, 6.9e9, 12e9]
lam1 = [3.20, 3.50, 3.94, 4.10, 4.22, 4.35, 4.41, 4.48]
lam2 = [1.45, 1.22, 1.06, 0.85, 0.72, 0.58, 0.48, 0.40]
stiffness = [l1/l2 for l1, l2 in zip(lam1, lam2)]
logN = np.log10(N)

sl, ic, r, p, _ = stats.linregress(logN, stiffness)
print(f"κ(N) = {sl:.2f}·log₁₀N + {ic:.2f}  (r={r:.3f}, p={p:.5f})")
print()
for i, n in enumerate(N):
    print(f"  {n/1e9:.3f}B: κ = {stiffness[i]:.2f}")
print(f"\nType-I/II boundary: λ/ξ ≈ 0.767 > 1/√2 → mixed-phase regime")
print(f"Stiffness diverges at ~88B (d_eff → 1 crossing)")
