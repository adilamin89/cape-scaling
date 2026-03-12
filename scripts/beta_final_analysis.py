#!/usr/bin/env python3
"""
Final analysis of 6-model gradient scaling results.
Generates plots and runs PySR on the complete dataset.
"""

import numpy as np, json, os
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.optimize import curve_fit

DIR = os.path.dirname(os.path.abspath(__file__))

# Load all 6 model results
with open(os.path.join(os.path.dirname(DIR), "data", "beta_final_6model.json")) as f:
    data = json.load(f)

all_r = data['all_results']
names = [r['name'] for r in all_r]
N = np.array([r['N'] for r in all_r])
G = np.array([r['grad_norm_mean'] for r in all_r])
G_std = np.array([r['grad_norm_std'] for r in all_r])
L = np.array([r['loss_mean'] for r in all_r])

logN = np.log10(N)
logG = np.log10(G)
logL = np.log10(L)

# Sort by N for plotting
idx = np.argsort(N)
N_s, G_s, G_std_s, L_s, logN_s, logG_s, logL_s = N[idx], G[idx], G_std[idx], L[idx], logN[idx], logG[idx], logL[idx]
names_s = [names[i] for i in idx]

print("="*60)
print("FINAL 6-MODEL GRADIENT SCALING ANALYSIS")
print("="*60)

# 1. Power-law fits
s_G, i_G, r_G, p_G, se_G = linregress(logN_s, logG_s)
beta = -s_G
s_L, i_L, r_L, p_L, se_L = linregress(logN_s, logL_s)
alpha = -s_L

print(f"\nPower-law fits:")
print(f"  ||grad|| ~ N^{{{s_G:.4f}}} => beta = {beta:.4f} +/- {se_G:.4f} (r={r_G:.4f})")
print(f"  Loss ~ N^{{{s_L:.4f}}} => alpha = {alpha:.4f} +/- {se_L:.4f} (r={r_L:.4f})")
print(f"  beta/alpha = {beta/alpha:.2f}")
print(f"  alpha+1 = {alpha+1:.4f}")

# 2. Chinchilla-style fit: L = E + A*N^{-alpha}
def chinchilla(logN, E, A, a):
    return E + A * 10**(-a*logN)

try:
    popt, pcov = curve_fit(chinchilla, logN_s, L_s, p0=[2.0, 50.0, 0.3], maxfev=5000)
    E_irr, A_fit, alpha_fit = popt
    L_fit = chinchilla(logN_s, *popt)
    print(f"\nChinchilla fit: L = {E_irr:.3f} + {A_fit:.1f}*N^(-{alpha_fit:.3f})")
    print(f"  R^2 = {1 - np.sum((L_s-L_fit)**2)/np.sum((L_s-np.mean(L_s))**2):.4f}")
    print(f"  Irreducible loss E = {E_irr:.3f}")
except:
    E_irr, alpha_fit = None, None
    print("  Chinchilla fit failed")

# 3. Non-monotonicity analysis
print(f"\nNon-monotonicity check:")
for i in range(len(N_s)):
    monotonic = "ok" if i == 0 or G_s[i] < G_s[i-1] else "NON-MONOTONIC"
    print(f"  {names_s[i]:15s}: N={N_s[i]:.0e}, ||grad||={G_s[i]:.2f}, {monotonic}")

# Check if 1b is an outlier
G_pred_1b = 10**(s_G * np.log10(1e9) + i_G)
G_actual_1b = G_s[np.argmin(np.abs(N_s - 1e9))]
residual_1b = (G_actual_1b - G_pred_1b) / G_pred_1b * 100
print(f"\n  1b deviation from power law: {residual_1b:.1f}%")

# 4. Fit excluding 1b outlier
mask = N_s != 1e9
if mask.sum() >= 3:
    s_G2, i_G2, r_G2, p_G2, se_G2 = linregress(logN_s[mask], logG_s[mask])
    print(f"\nFit EXCLUDING 1b:")
    print(f"  beta = {-s_G2:.4f} +/- {se_G2:.4f} (r={r_G2:.4f})")
    print(f"  Compare: with 1b beta={beta:.4f}, without beta={-s_G2:.4f}")

# 5. Per-parameter gradient
G_pp = G_s / np.sqrt(N_s)
logGpp = np.log10(G_pp)
s_pp, _, r_pp, _, _ = linregress(logN_s, logGpp)
print(f"\nPer-parameter gradient: ||grad||/sqrt(N) ~ N^{{{s_pp:.4f}}}")

# ═══════════════════════════════════════════
# FIGURE: 4-panel analysis
# ═══════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel 1: ||grad|| vs N
ax = axes[0, 0]
ax.errorbar(logN_s, G_s, yerr=G_std_s, fmt='o', markersize=8, capsize=4, color='royalblue', label='Measured')
logN_fit = np.linspace(logN_s.min()-0.1, logN_s.max()+0.1, 100)
ax.plot(logN_fit, 10**(s_G*logN_fit + i_G), 'r--', label=f'Power law: beta={beta:.3f}')
ax.set_xlabel('log10(N)')
ax.set_ylabel('||grad L||')
ax.set_title(f'Gradient Norm vs Model Size (beta={beta:.3f})')
ax.legend()
for i, n in enumerate(names_s):
    ax.annotate(n.replace('pythia-',''), (logN_s[i], G_s[i]), fontsize=7, ha='left')

# Panel 2: Loss vs N
ax = axes[0, 1]
ax.plot(logN_s, L_s, 'o-', markersize=8, color='forestgreen')
logN_fine = np.linspace(logN_s.min()-0.1, logN_s.max()+0.1, 100)
ax.plot(logN_fine, 10**(s_L*logN_fine + i_L), 'r--', label=f'alpha={alpha:.3f}')
ax.set_xlabel('log10(N)')
ax.set_ylabel('Loss')
ax.set_title(f'Loss vs Model Size (alpha={alpha:.3f})')
ax.legend()

# Panel 3: ||grad|| vs Loss
ax = axes[1, 0]
ax.plot(L_s, G_s, 'o', markersize=8, color='darkorange')
s_gl, i_gl, r_gl, _, _ = linregress(logL_s, logG_s)
L_fine = np.linspace(L_s.min()-0.1, L_s.max()+0.1, 100)
ax.plot(L_fine, 10**(s_gl*np.log10(L_fine) + i_gl), 'r--', label=f'||grad|| ~ L^{{{s_gl:.2f}}} (r={r_gl:.3f})')
for i, n in enumerate(names_s):
    ax.annotate(n.replace('pythia-',''), (L_s[i], G_s[i]), fontsize=7)
ax.set_xlabel('Loss')
ax.set_ylabel('||grad L||')
ax.set_title('Gradient vs Loss (direct relationship)')
ax.legend()

# Panel 4: Log-log gradient vs N with N_c region
ax = axes[1, 1]
ax.plot(logN_s, logG_s, 'o', markersize=8, color='crimson')
ax.plot(logN_fit, s_G*logN_fit + i_G, 'b--', alpha=0.5, label=f'Power law')
# Highlight N_c region
ax.axvspan(np.log10(1.1e9), np.log10(5.4e9), alpha=0.15, color='gold', label='N_c region (90% CI)')
ax.axvline(np.log10(2.3e9), color='gold', linestyle=':', alpha=0.5)
for i, n in enumerate(names_s):
    ax.annotate(n.replace('pythia-',''), (logN_s[i], logG_s[i]+0.02), fontsize=7, ha='center')
ax.set_xlabel('log10(N)')
ax.set_ylabel('log10(||grad L||)')
ax.set_title('Log-log: Gradient dip near Lifshitz transition?')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(DIR), "figures", "beta_final_6model.png"), dpi=150)
print(f"\nFigure saved: beta_final_6model.png")

# ═══════════════════════════════════════════
# PySR with 6 data points
# ═══════════════════════════════════════════
print(f"\n{'='*60}")
print("PySR SYMBOLIC REGRESSION (6 models)")
print("="*60)

try:
    from pysr import PySRRegressor

    # Task 1: ||grad|| = f(logN)
    X = logN_s.reshape(-1, 1)
    print("  Running PySR: ||grad|| = f(logN) ...")
    m1 = PySRRegressor(
        niterations=60, binary_operators=["+", "-", "*", "/"],
        unary_operators=["exp", "log", "sqrt", "abs"],
        maxsize=15, populations=15, progress=False, verbosity=0,
    )
    m1.fit(X, G_s, variable_names=["logN"])
    print("  Top equations:")
    for i, eq in enumerate(m1.equations_.itertuples()):
        if i >= 5: break
        print(f"    [{i}] {eq.equation:45s} loss={eq.loss:.3f}")

    # Task 2: ||grad|| = f(logN, L)
    X2 = np.column_stack([logN_s, L_s])
    print(f"\n  Running PySR: ||grad|| = f(logN, L) ...")
    m2 = PySRRegressor(
        niterations=60, binary_operators=["+", "-", "*", "/"],
        unary_operators=["exp", "log", "sqrt"],
        maxsize=15, populations=15, progress=False, verbosity=0,
    )
    m2.fit(X2, G_s, variable_names=["logN", "L"])
    print("  Top equations:")
    for i, eq in enumerate(m2.equations_.itertuples()):
        if i >= 5: break
        print(f"    [{i}] {eq.equation:45s} loss={eq.loss:.3f}")

except ImportError:
    print("  PySR not available")
except Exception as e:
    print(f"  PySR error: {e}")

# ═══════════════════════════════════════════
# THEORY IMPLICATIONS
# ═══════════════════════════════════════════
print(f"\n{'='*60}")
print("THEORY IMPLICATIONS")
print("="*60)
print(f"""
1. beta = {beta:.3f} +/- {se_G:.3f} (6 models, N = 7e7 to 2.8e9)
2. alpha = {alpha:.3f} +/- {se_L:.3f}
3. beta/alpha = {beta/alpha:.2f} (not ~1 as 3-point suggested, not alpha+1=1.12)
4. Mean-field GL prediction beta=alpha+1 FAILS (delta={abs(beta-alpha-1):.3f})
5. The ORIGINAL 3-point beta=0.30 was MISLEADING due to limited range
6. Non-monotonic ||grad|| at 1b — possible Lifshitz transition signature?

GRADIENT NON-MONOTONICITY:
  The gradient norm dips at pythia-1b (N=1e9), which is within the
  Lifshitz transition region (N_c = 2.3B, 90% CI [1.1B, 5.4B]).
  This could indicate the phase transition affects not just benchmark
  correlations but also the loss landscape geometry.

REVISED UNDERSTANDING:
  - beta ~ 0.4 is intermediate between alpha (0.12) and alpha+1 (1.12)
  - ||grad|| ~ L^3.4 — steeper than linear, but not as steep as mean-field
  - The simple power law is a poor fit (r=-0.92) due to the 1b anomaly
  - Need: piecewise or non-power-law model for gradient scaling
""")
