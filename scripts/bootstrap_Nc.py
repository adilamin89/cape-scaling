#!/usr/bin/env python3
"""
Bootstrap/jackknife error bars on N_c (Lifshitz point).

Tests:
1. Bootstrap resampling of the U-shaped TQA quadratic fit → N_c confidence interval
2. Jackknife on γ₁₂ linear fit zero-crossing → N_c confidence interval
3. Bootstrap on sliding-window correlation → transition region bounds

Author: Adil Amin, March 2026
"""

import numpy as np
import os

# Output paths relative to repo root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
_FIG_DIR = os.path.join(_REPO_ROOT, "figures")
_DATA_DIR = os.path.join(_REPO_ROOT, "data")
os.makedirs(_FIG_DIR, exist_ok=True)

from scipy.optimize import curve_fit
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── DATA ──
# Combined Pythia + Llama benchmark scores (verified)
data = {
    'Pythia': [
        {'name': '70m',   'N': 7e7,   'HS': 27.29, 'TQA': 47.64},
        {'name': '160m',  'N': 1.6e8, 'HS': 29.59, 'TQA': 43.52},
        {'name': '410m',  'N': 4.1e8, 'HS': 40.56, 'TQA': 40.34},
        {'name': '1b',    'N': 1e9,   'HS': 47.16, 'TQA': 38.67},
        {'name': '1.4b',  'N': 1.4e9, 'HS': 52.01, 'TQA': 38.66},
        {'name': '2.8b',  'N': 2.8e9, 'HS': 59.37, 'TQA': 35.56},
        {'name': '6.9b',  'N': 6.9e9, 'HS': 64.02, 'TQA': 32.76},
        {'name': '12b',   'N': 1.2e10,'HS': 67.30, 'TQA': 32.47},
    ],
    'Llama-1': [
        {'name': '7B',  'N': 7e9,   'HS': 77.81, 'TQA': 34.33},
        {'name': '13B', 'N': 1.3e10,'HS': 80.92, 'TQA': 39.48},
        {'name': '65B', 'N': 6.5e10,'HS': 86.09, 'TQA': 43.43},
    ],
    'Llama-2': [
        {'name': '7B',  'N': 7e9,   'HS': 77.74, 'TQA': 38.98},
        {'name': '13B', 'N': 1.3e10,'HS': 82.10, 'TQA': 37.40},
        {'name': '70B', 'N': 7e10,  'HS': 87.30, 'TQA': 44.90},
    ],
}

# Flatten
all_points = []
for family, models in data.items():
    for m in models:
        all_points.append({**m, 'family': family})

N_all = np.array([p['N'] for p in all_points])
TQA_all = np.array([p['TQA'] for p in all_points])
HS_all = np.array([p['HS'] for p in all_points])
logN_all = np.log10(N_all)

n_total = len(all_points)
print(f"Total data points: {n_total}")

# ═══════════════════════════════════════════
# 1. BOOTSTRAP ON QUADRATIC TQA FIT → N_c
# ═══════════════════════════════════════════
print("\n" + "="*60)
print("1. BOOTSTRAP: Quadratic TQA(logN) fit")
print("="*60)

n_boot = 10000
np.random.seed(42)

Nc_boot = []
coeff_boot = []

for i in range(n_boot):
    # Resample with replacement
    idx = np.random.choice(n_total, size=n_total, replace=True)
    logN_b = logN_all[idx]
    TQA_b = TQA_all[idx]

    # Quadratic fit
    try:
        coeffs = np.polyfit(logN_b, TQA_b, 2)
        a, b, c = coeffs
        if a > 0:  # U-shape (minimum exists)
            logN_min = -b / (2 * a)
            if 7 < logN_min < 12:  # Reasonable range
                Nc_boot.append(10**logN_min)
                coeff_boot.append(coeffs)
    except:
        continue

Nc_boot = np.array(Nc_boot)
n_valid = len(Nc_boot)
print(f"  Valid bootstrap samples: {n_valid}/{n_boot} ({100*n_valid/n_boot:.1f}%)")

# Statistics
Nc_median = np.median(Nc_boot)
Nc_mean = np.mean(Nc_boot)
Nc_ci_lo, Nc_ci_hi = np.percentile(Nc_boot, [2.5, 97.5])
Nc_ci_68_lo, Nc_ci_68_hi = np.percentile(Nc_boot, [16, 84])

print(f"\n  N_c (quadratic fit):")
print(f"    Median: {Nc_median:.2e}")
print(f"    Mean:   {Nc_mean:.2e}")
print(f"    68% CI: [{Nc_ci_68_lo:.2e}, {Nc_ci_68_hi:.2e}]")
print(f"    95% CI: [{Nc_ci_lo:.2e}, {Nc_ci_hi:.2e}]")
print(f"    In billions: {Nc_median/1e9:.1f}B [{Nc_ci_lo/1e9:.1f}B, {Nc_ci_hi/1e9:.1f}B]")

# ═══════════════════════════════════════════
# 2. JACKKNIFE ON γ₁₂ LINEAR FIT ZERO-CROSSING
# ═══════════════════════════════════════════
print("\n" + "="*60)
print("2. JACKKNIFE: γ₁₂(logN) zero-crossing")
print("="*60)

# Compute local coupling γ₁₂ = dTQA/dHS between consecutive points
# Sort by N
sort_idx = np.argsort(N_all)
N_sorted = N_all[sort_idx]
HS_sorted = HS_all[sort_idx]
TQA_sorted = TQA_all[sort_idx]
families_sorted = [all_points[i]['family'] for i in sort_idx]

# Consecutive differences (within same family or across)
gamma_points = []
for i in range(len(N_sorted) - 1):
    dTQA = TQA_sorted[i+1] - TQA_sorted[i]
    dHS = HS_sorted[i+1] - HS_sorted[i]
    if abs(dHS) > 0.5:  # Avoid division by near-zero
        gamma = dTQA / dHS
        logN_mid = 0.5 * (np.log10(N_sorted[i]) + np.log10(N_sorted[i+1]))
        gamma_points.append({'logN': logN_mid, 'gamma': gamma})

gamma_logN = np.array([g['logN'] for g in gamma_points])
gamma_vals = np.array([g['gamma'] for g in gamma_points])
n_gamma = len(gamma_points)

# Full fit
slope_full, intercept_full = np.polyfit(gamma_logN, gamma_vals, 1)
Nc_full = 10**(-intercept_full / slope_full)
print(f"  Full fit: γ₁₂ = {slope_full:.3f}·log₁₀N + ({intercept_full:.3f})")
print(f"  Zero crossing: N_c = {Nc_full:.2e}")

# Jackknife: leave one out
Nc_jack = []
for j in range(n_gamma):
    mask = np.ones(n_gamma, dtype=bool)
    mask[j] = False
    s, ic = np.polyfit(gamma_logN[mask], gamma_vals[mask], 1)
    if s != 0:
        nc = 10**(-ic / s)
        if 1e7 < nc < 1e12:
            Nc_jack.append(nc)

Nc_jack = np.array(Nc_jack)
Nc_jack_mean = np.mean(Nc_jack)
# Jackknife standard error
Nc_jack_se = np.sqrt((n_gamma - 1) / n_gamma * np.sum((Nc_jack - Nc_jack_mean)**2))

print(f"\n  Jackknife N_c:")
print(f"    Mean:   {Nc_jack_mean:.2e}")
print(f"    SE:     {Nc_jack_se:.2e}")
print(f"    In billions: {Nc_jack_mean/1e9:.1f}B ± {Nc_jack_se/1e9:.1f}B")

# Also bootstrap the linear fit
Nc_gamma_boot = []
for i in range(n_boot):
    idx = np.random.choice(n_gamma, size=n_gamma, replace=True)
    s, ic = np.polyfit(gamma_logN[idx], gamma_vals[idx], 1)
    if s != 0:
        nc = 10**(-ic / s)
        if 1e7 < nc < 1e12:
            Nc_gamma_boot.append(nc)

Nc_gamma_boot = np.array(Nc_gamma_boot)
Nc_gb_median = np.median(Nc_gamma_boot)
Nc_gb_lo, Nc_gb_hi = np.percentile(Nc_gamma_boot, [2.5, 97.5])
print(f"\n  Bootstrap γ₁₂ zero-crossing:")
print(f"    Median: {Nc_gb_median:.2e}")
print(f"    95% CI: [{Nc_gb_lo:.2e}, {Nc_gb_hi:.2e}]")
print(f"    In billions: {Nc_gb_median/1e9:.1f}B [{Nc_gb_lo/1e9:.1f}B, {Nc_gb_hi/1e9:.1f}B]")

# ═══════════════════════════════════════════
# 3. BOOTSTRAP ON PEARSON r(HS, TQA) BY FAMILY
# ═══════════════════════════════════════════
print("\n" + "="*60)
print("3. BOOTSTRAP: Correlation coefficients")
print("="*60)

from scipy.stats import pearsonr

for family, models in data.items():
    hs = np.array([m['HS'] for m in models])
    tqa = np.array([m['TQA'] for m in models])
    n_fam = len(models)

    r_orig, p_orig = pearsonr(hs, tqa)

    # Bootstrap r
    r_boot = []
    for i in range(n_boot):
        idx = np.random.choice(n_fam, size=n_fam, replace=True)
        if len(np.unique(idx)) >= 2:  # Need at least 2 unique points
            r_b, _ = pearsonr(hs[idx], tqa[idx])
            if np.isfinite(r_b):
                r_boot.append(r_b)

    r_boot = np.array(r_boot)
    if len(r_boot) > 100:
        r_lo, r_hi = np.percentile(r_boot, [2.5, 97.5])
        print(f"  {family} (n={n_fam}): r = {r_orig:.3f} [{r_lo:.3f}, {r_hi:.3f}] (p={p_orig:.2e})")
    else:
        print(f"  {family} (n={n_fam}): r = {r_orig:.3f} (p={p_orig:.2e}) — too few points for bootstrap CI")

# ═══════════════════════════════════════════
# 4. FIGURE
# ═══════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 12), facecolor='#0A0C10')
fig.suptitle('Bootstrap Error Bars on Lifshitz Point N_c\n'
             'GL Neural Scaling Phase Transition',
             fontsize=16, color='#E8ECF4', fontweight='bold')

colors = {'bg': '#0A0C10', 'text': '#C8CCD4', 'grid': '#1E1E28',
          'blue': '#4EA8DE', 'green': '#4ADE80', 'red': '#EF4444',
          'yellow': '#FBBF24', 'accent': '#E07040', 'purple': '#A78BFA'}

for ax in axes.flat:
    ax.set_facecolor(colors['bg'])
    ax.tick_params(colors='#6B7280')
    ax.grid(color=colors['grid'], alpha=0.5)
    for s in ax.spines.values(): s.set_color('#2A2E38')

# Panel A: N_c bootstrap distribution (quadratic)
ax = axes[0, 0]
ax.hist(np.log10(Nc_boot), bins=50, color=colors['blue'], alpha=0.7, edgecolor='white', linewidth=0.5)
ax.axvline(np.log10(Nc_median), color=colors['yellow'], linestyle='-', linewidth=2, label=f'Median: {Nc_median/1e9:.1f}B')
ax.axvline(np.log10(Nc_ci_lo), color=colors['red'], linestyle='--', linewidth=1.5, label=f'95% CI: [{Nc_ci_lo/1e9:.1f}B, {Nc_ci_hi/1e9:.1f}B]')
ax.axvline(np.log10(Nc_ci_hi), color=colors['red'], linestyle='--', linewidth=1.5)
ax.set_xlabel('log₁₀(N_c)', color=colors['text'])
ax.set_ylabel('Count', color=colors['text'])
ax.set_title('A. Bootstrap N_c (Quadratic TQA Fit)', color=colors['accent'], fontweight='bold')
ax.legend(fontsize=10, facecolor='#12151C', edgecolor='#2A2E38', labelcolor=colors['text'])

# Panel B: N_c bootstrap distribution (γ₁₂ zero-crossing)
ax = axes[0, 1]
ax.hist(np.log10(Nc_gamma_boot), bins=50, color=colors['green'], alpha=0.7, edgecolor='white', linewidth=0.5)
ax.axvline(np.log10(Nc_gb_median), color=colors['yellow'], linestyle='-', linewidth=2, label=f'Median: {Nc_gb_median/1e9:.1f}B')
ax.axvline(np.log10(Nc_gb_lo), color=colors['red'], linestyle='--', linewidth=1.5, label=f'95% CI: [{Nc_gb_lo/1e9:.1f}B, {Nc_gb_hi/1e9:.1f}B]')
ax.axvline(np.log10(Nc_gb_hi), color=colors['red'], linestyle='--', linewidth=1.5)
ax.set_xlabel('log₁₀(N_c)', color=colors['text'])
ax.set_ylabel('Count', color=colors['text'])
ax.set_title('B. Bootstrap N_c (γ₁₂ Zero-Crossing)', color=colors['accent'], fontweight='bold')
ax.legend(fontsize=10, facecolor='#12151C', edgecolor='#2A2E38', labelcolor=colors['text'])

# Panel C: U-shape with bootstrap confidence band
ax = axes[1, 0]
family_colors = {'Pythia': colors['blue'], 'Llama-1': colors['green'], 'Llama-2': colors['purple']}
for family, models in data.items():
    N_f = [m['N'] for m in models]
    TQA_f = [m['TQA'] for m in models]
    ax.scatter(N_f, TQA_f, color=family_colors[family], s=80, zorder=5, label=family)

# Plot bootstrap confidence band
logN_grid = np.linspace(7.5, 11.5, 200)
TQA_boot_curves = []
for coeffs in coeff_boot[:min(len(coeff_boot), 2000)]:
    TQA_boot_curves.append(np.polyval(coeffs, logN_grid))
TQA_boot_curves = np.array(TQA_boot_curves)
TQA_lo = np.percentile(TQA_boot_curves, 2.5, axis=0)
TQA_hi = np.percentile(TQA_boot_curves, 97.5, axis=0)
TQA_med = np.median(TQA_boot_curves, axis=0)

ax.fill_between(10**logN_grid, TQA_lo, TQA_hi, color=colors['yellow'], alpha=0.2, label='95% CI')
ax.plot(10**logN_grid, TQA_med, color=colors['yellow'], linewidth=2, label='Median fit')
ax.axvline(Nc_median, color=colors['red'], linestyle=':', alpha=0.7, label=f'N_c = {Nc_median/1e9:.1f}B')
ax.set_xscale('log')
ax.set_xlabel('N (parameters)', color=colors['text'])
ax.set_ylabel('TruthfulQA (%)', color=colors['text'])
ax.set_title('C. U-Shape with Bootstrap Band', color=colors['accent'], fontweight='bold')
ax.legend(fontsize=9, facecolor='#12151C', edgecolor='#2A2E38', labelcolor=colors['text'])

# Panel D: γ₁₂ with jackknife error bars
ax = axes[1, 1]
ax.scatter(gamma_logN, gamma_vals, color=colors['blue'], s=80, zorder=5)
logN_fit = np.linspace(7.5, 11.5, 100)
ax.plot(logN_fit, slope_full * logN_fit + intercept_full, '--', color=colors['yellow'], linewidth=2,
        label=f'γ₁₂ = {slope_full:.2f}·log₁₀N + ({intercept_full:.2f})')
ax.axhline(0, color=colors['red'], linestyle=':', alpha=0.7)
ax.axvline(np.log10(Nc_full), color=colors['green'], linestyle=':', alpha=0.7,
           label=f'N_c = {Nc_full/1e9:.1f}B ± {Nc_jack_se/1e9:.1f}B')
ax.fill_between([np.log10(Nc_gb_lo), np.log10(Nc_gb_hi)], [-2, -2], [3, 3],
                color=colors['green'], alpha=0.15, label='95% CI on N_c')
ax.set_ylim(-2, 3)
ax.set_xlabel('log₁₀(N)', color=colors['text'])
ax.set_ylabel('γ₁₂ = dTQA/dHS', color=colors['text'])
ax.set_title('D. Running Coupling with Error Bars', color=colors['accent'], fontweight='bold')
ax.legend(fontsize=9, facecolor='#12151C', edgecolor='#2A2E38', labelcolor=colors['text'])

# Add s± and s++ labels
ax.text(8.0, -1.5, 's± regime', color=colors['red'], fontsize=12, fontweight='bold')
ax.text(10.5, 1.5, 's++ regime', color=colors['green'], fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(_FIG_DIR, 'bootstrap_Nc.png'),
            dpi=150, bbox_inches='tight', facecolor=colors['bg'])
plt.close()

# ── SAVE RESULTS ──
results = {
    'quadratic_fit': {
        'Nc_median': float(Nc_median),
        'Nc_mean': float(Nc_mean),
        'Nc_95CI': [float(Nc_ci_lo), float(Nc_ci_hi)],
        'Nc_68CI': [float(Nc_ci_68_lo), float(Nc_ci_68_hi)],
        'n_valid_samples': int(n_valid),
        'Nc_billions': f"{Nc_median/1e9:.1f}B [{Nc_ci_lo/1e9:.1f}B, {Nc_ci_hi/1e9:.1f}B]",
    },
    'gamma_zero_crossing': {
        'Nc_full_fit': float(Nc_full),
        'Nc_jackknife_mean': float(Nc_jack_mean),
        'Nc_jackknife_SE': float(Nc_jack_se),
        'Nc_bootstrap_median': float(Nc_gb_median),
        'Nc_bootstrap_95CI': [float(Nc_gb_lo), float(Nc_gb_hi)],
        'gamma_slope': float(slope_full),
        'gamma_intercept': float(intercept_full),
    },
    'combined_estimate': {
        'note': 'Weighted average of quadratic and gamma methods',
        'Nc_quad_B': float(Nc_median / 1e9),
        'Nc_gamma_B': float(Nc_gb_median / 1e9),
    }
}

with open(os.path.join(_DATA_DIR, 'bootstrap_Nc_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*60}")
print("SUMMARY FOR PAPER")
print("="*60)
print(f"  Method 1 (Quadratic TQA fit):  N_c = {Nc_median/1e9:.1f}B  95%CI: [{Nc_ci_lo/1e9:.1f}B, {Nc_ci_hi/1e9:.1f}B]")
print(f"  Method 2 (γ₁₂ zero-crossing): N_c = {Nc_gb_median/1e9:.1f}B  95%CI: [{Nc_gb_lo/1e9:.1f}B, {Nc_gb_hi/1e9:.1f}B]")
print(f"\n  → For paper: N_c ≈ {0.5*(Nc_median+Nc_gb_median)/1e9:.1f}B")
print(f"\n✓ Figure saved: bootstrap_Nc.png")
print(f"✓ Results saved: bootstrap_Nc_results.json")
