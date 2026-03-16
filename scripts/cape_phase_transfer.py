"""
cape_phase_transfer.py
Phase Transfer Matrix T = V_bonus · V_tax^{-1}

Computes the 1-to-1 transformation between benchmark coordinates
in different coupling phases (tax → bonus → frontier).

Key result:
  A_bonus = T · A_tax · T^{-1} + ΔA_coupling
  
  T = pure eigenvector rotation (geometric, computable from PCA data)
  ΔA_coupling = residual coupling change (feed to PySR for symbolic form)

Usage:
  python cape_phase_transfer.py
  
Output:
  - Transfer matrix T
  - Reconstruction quality ||T·A_tax·T^{-1} - A_bonus||
  - Residual matrix for PySR symbolic regression
  - Extension template for Nc2 (HS/TQA → SWE/GPQA)
"""
import numpy as np

# ── ODE coefficient matrices per phase (from PySINDy) ──────────────
# 5-benchmark order: [HS, TQA, ARC, WG, MMLU]
A_tax = np.array([
    [-0.00, -0.72, -0.00, -0.69, -0.00],
    [-0.31,  0.12, -0.00, -0.00, -0.00],
    [-0.00, -0.23,  0.00, -0.00, -0.00],
    [-0.00, -0.19, -0.00,  0.00, -0.00],
    [-0.00, -0.00, -0.00, -0.00,  0.00],
])

A_bonus = np.array([
    [-0.00, -0.11, -0.00, -0.52, -0.00],
    [-0.18,  0.08, -0.00, -0.00, -0.00],
    [-0.00, -0.12,  0.00, -0.00, -0.00],
    [-0.00, -0.09, -0.00,  0.00, -0.00],
    [-0.00, -0.00, -0.00, -0.00,  0.00],
])

# ── PCA eigenvectors per phase (from §7 measurements) ──────────────
# Replace with exact values from your local data files for production
V_tax = np.array([
    [ 0.48,  -0.20,  0.45,  0.52,  0.50],
    [ 0.40,   0.62,  0.35,  0.40,  0.45],
    [ 0.49,  -0.15,  0.48,  0.52,  0.49],
    [ 0.47,  -0.25,  0.46,  0.51,  0.50],
    [ 0.38,   0.70,  0.40,  0.30,  0.38],
])

V_bonus = np.array([
    [ 0.49,   0.22,  0.45,  0.51,  0.50],
    [ 0.38,  -0.71,  0.35,  0.40,  0.45],
    [ 0.50,   0.18,  0.48,  0.52,  0.49],
    [ 0.48,   0.21,  0.46,  0.51,  0.50],
    [ 0.37,  -0.61,  0.40,  0.30,  0.38],
])

# ── Orthonormalize ──────────────────────────────────────────────────
def orthonorm(V):
    Q, _ = np.linalg.qr(V)
    return Q

V_tax_n   = orthonorm(V_tax)
V_bonus_n = orthonorm(V_bonus)

# ── Phase transfer matrix ───────────────────────────────────────────
T = V_bonus_n @ np.linalg.inv(V_tax_n)

print("Phase Transfer Matrix T = V_bonus · V_tax^{-1}")
labels = ["HS","TQA","ARC","WG","MMLU"]
header = f"{'':6}" + "".join(f"{l:8}" for l in labels)
print(header)
for i, row in enumerate(T):
    print(f"  {labels[i]:4}  " + "".join(f"{v:+7.3f} " for v in row))

# ── Reconstruction test ─────────────────────────────────────────────
A_reconstructed = T @ A_tax @ np.linalg.inv(T)
dA_coupling = A_bonus - A_reconstructed

print(f"\nReconstruction residual ||T·A_tax·T^{{-1}} - A_bonus|| = {np.linalg.norm(dA_coupling):.4f}")
print("(0 = pure rotation; nonzero = coupling correction needed)")

print("\nΔA_coupling (feed to PySR for symbolic form):")
for i, row in enumerate(dA_coupling):
    dominant = [(labels[j], row[j]) for j in range(5) if abs(row[j]) > 0.05]
    if dominant:
        terms = ", ".join(f"{v:+.3f}·{l}" for l,v in dominant)
        print(f"  d{labels[i]}/dlogN correction: {terms}")

# ── Extension template for Nc2 ─────────────────────────────────────
print("""
\nNc2 EXTENSION (HS/TQA → SWE/GPQA):
  At Nc2, construct V_frontier from frontier model PCA.
  T2 = V_frontier · V_bonus^{-1}
  A_frontier = T2 · A_bonus · T2^{-1} + ΔA_coupling2
  
  benchmark_frontier = T2 · T1 · benchmark_base
  ODE_frontier = T2 · T1 · ODE_base · (T2·T1)^{-1} + corrections
  
  This gives a 1-to-1 map from base model benchmarks to frontier
  benchmarks, computable from the measured eigenvector rotations.
""")

import json
results = {
    "T": T.tolist(),
    "A_tax": A_tax.tolist(),
    "A_bonus": A_bonus.tolist(),
    "dA_coupling": dA_coupling.tolist(),
    "reconstruction_error": float(np.linalg.norm(dA_coupling)),
    "description": "Phase transfer matrix T=V_bonus@V_tax^{-1}; A_bonus = T@A_tax@T^{-1} + dA_coupling"
}
with open("cape_phase_transfer_results.json","w") as f:
    json.dump(results, f, indent=2)
print("Results saved to cape_phase_transfer_results.json")
