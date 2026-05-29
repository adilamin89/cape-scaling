#!/usr/bin/env python3
"""
Regenerate ODE figure for Paper 3A (NeurIPS version).
Removes hardcoded "Figure 4" title. LaTeX handles figure numbering.
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os

DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(DIR, '..', 'fig_ode_neurips.png')

# Pythia benchmark data (ground truth)
logN = np.log10([7e7, 1.6e8, 4.1e8, 1e9, 1.4e9, 2.8e9, 6.9e9, 1.2e10])
HS  = np.array([27.3, 31.4, 40.9, 49.7, 52.9, 60.7, 64.0, 67.3])
TQA = np.array([47.1, 44.3, 41.2, 38.9, 38.9, 35.6, 33.0, 32.0])
ARC = np.array([21.6, 24.1, 26.2, 29.1, 31.5, 36.3, 37.0, 38.0])
WG  = np.array([51.5, 51.4, 53.1, 53.6, 58.0, 60.2, 62.0, 64.0])
MMLU= np.array([25.9, 24.9, 27.3, 24.3, 25.8, 26.8, 26.0, 27.0])

benchmarks = {'HellaSwag': (HS, '#4285F4'),
              'TruthfulQA': (TQA, '#DB4437'),
              'ARC': (ARC, '#0F9D58'),
              'WinoGrande': (WG, '#7B1FA2'),
              'MMLU': (MMLU, '#F4B400')}

def refit_and_integrate():
    dt = np.diff(logN)
    all_data = np.column_stack([HS, TQA, ARC, WG, MMLU])
    derivs = np.diff(all_data, axis=0) / dt[:, None]
    midpoints = (all_data[:-1] + all_data[1:]) / 2

    # Simple linear ODE: dB/dlogN = C @ B + c0
    from numpy.linalg import lstsq
    A_mat = np.column_stack([midpoints, np.ones(len(midpoints))])
    coeffs, _, _, _ = lstsq(A_mat, derivs, rcond=None)
    C = coeffs[:5].T
    c0 = coeffs[5]

    def ode_rhs(t, y):
        return C @ y + c0

    y0 = all_data[0]
    t_span = (logN[0], logN[-1])
    t_eval = np.linspace(logN[0], logN[-1], 200)
    sol = solve_ivp(ode_rhs, t_span, y0, t_eval=t_eval, method='RK45')
    return sol, all_data

sol, actual = refit_and_integrate()

# Compute per-benchmark MAE
names_list = list(benchmarks.keys())
fig, axes = plt.subplots(1, 5, figsize=(16, 3.2))
fig.subplots_adjust(wspace=0.35)

for i, (name, (data, color)) in enumerate(benchmarks.items()):
    ax = axes[i]
    # Interpolate ODE prediction at actual logN points
    pred = np.interp(logN, sol.t, sol.y[i])
    mae = np.mean(np.abs(pred - data))
    mae_pct = mae / np.mean(data) * 100

    ax.plot(sol.t, sol.y[i], color=color, linewidth=2, alpha=0.8)
    ax.scatter(logN, data, color=color, s=40, zorder=5, edgecolors='white', linewidth=0.5)

    # Shade Nc region
    ax.axvspan(9.0, 9.6, alpha=0.08, color='orange')

    ax.set_xlabel(r'$\log_{10}(N)$', fontsize=9)
    ax.set_title(name, fontsize=11, fontweight='bold', color=color)

    # MAE badge
    bbox_color = '#FFF3E0' if mae_pct < 5 else '#FFEBEE'
    ax.text(0.95, 0.08, f'{mae_pct:.1f}%',
            transform=ax.transAxes, fontsize=9, fontweight='bold',
            ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=bbox_color, edgecolor=color, alpha=0.9))

    ax.tick_params(labelsize=8)

plt.savefig(OUT, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved: {OUT}')

# Also compute overall MAE for verification
all_pred = np.column_stack([np.interp(logN, sol.t, sol.y[i]) for i in range(5)])
overall_mae = np.mean(np.abs(all_pred - actual))
overall_pct = overall_mae / np.mean(actual) * 100
print(f'Overall MAE: {overall_mae:.2f} ({overall_pct:.1f}%)')
print(f'Paper claims 3.6% mean error — verify this matches.')
