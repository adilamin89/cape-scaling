#!/usr/bin/env python3
"""Per-phase PySINDy: fit ODE separately on s+/- and s++ regimes."""
import numpy as np
import json, os

DIR = os.path.dirname(os.path.abspath(__file__))

names = ['70M', '160M', '410M', '1B', '1.4B', '2.8B', '6.9B', '12B']
logN = np.log10([7e7, 1.6e8, 4.1e8, 1e9, 1.4e9, 2.8e9, 6.9e9, 1.2e10])
HS  = np.array([27.3, 31.4, 40.9, 49.7, 52.9, 60.7, 64.0, 67.3])
TQA = np.array([47.1, 44.3, 41.2, 38.9, 38.9, 35.6, 33.0, 32.0])
ARC = np.array([21.6, 24.1, 26.2, 29.1, 31.5, 36.3, 37.0, 38.0])
WG  = np.array([51.5, 51.4, 53.1, 53.6, 58.0, 60.2, 62.0, 64.0])
MMLU= np.array([25.9, 24.9, 27.3, 24.3, 25.8, 26.8, 26.0, 27.0])

def fit_ode_phase(idx, phase_name):
    sep = '\u2500' * 50
    print(f"\n{sep}")
    print(f"  Phase: {phase_name} (models: {[names[i] for i in idx]})")
    print(sep)

    lN = logN[idx]
    hs, tqa, arc, wg = HS[idx], TQA[idx], ARC[idx], WG[idx]

    dt = np.diff(lN)
    dHS = np.diff(hs) / dt
    dTQA = np.diff(tqa) / dt

    hs_mid = 0.5*(hs[:-1] + hs[1:])
    tqa_mid = 0.5*(tqa[:-1] + tqa[1:])
    wg_mid = 0.5*(wg[:-1] + wg[1:])

    results = {}

    # Fit dHS/dlogN
    n_intervals = len(dt)
    if n_intervals >= 3:
        X = np.column_stack([np.ones(n_intervals), tqa_mid, wg_mid])
        coefs, _, _, _ = np.linalg.lstsq(X, dHS, rcond=None)
        print(f"  HS' = {coefs[0]:.2f} + {coefs[1]:.3f}*TQA + {coefs[2]:.3f}*WG")
        results['HS_eq'] = {'const': float(coefs[0]), 'TQA_coef': float(coefs[1]), 'WG_coef': float(coefs[2])}
    else:
        X2 = np.column_stack([np.ones(n_intervals), tqa_mid])
        coefs2, _, _, _ = np.linalg.lstsq(X2, dHS, rcond=None)
        print(f"  HS' = {coefs2[0]:.2f} + {coefs2[1]:.3f}*TQA")
        results['HS_eq'] = {'const': float(coefs2[0]), 'TQA_coef': float(coefs2[1])}

    # Fit dTQA/dlogN = a + b*HS
    X_tqa = np.column_stack([np.ones(n_intervals), hs_mid])
    coefs_tqa, _, _, _ = np.linalg.lstsq(X_tqa, dTQA, rcond=None)
    print(f"  TQA' = {coefs_tqa[0]:.2f} + {coefs_tqa[1]:.3f}*HS")
    results['TQA_eq'] = {'const': float(coefs_tqa[0]), 'HS_coef': float(coefs_tqa[1])}

    gamma_12 = np.diff(tqa) / np.diff(hs)
    print(f"  gamma_12 values: {[f'{g:.3f}' for g in gamma_12]}")
    print(f"  mean gamma_12 = {np.mean(gamma_12):.3f}")
    results['gamma_12_values'] = gamma_12.tolist()
    results['gamma_12_mean'] = float(np.mean(gamma_12))

    return results

print("=" * 70)
print("PER-PHASE PySINDy: s+/- vs s++ DYNAMICS")
print("=" * 70)

spm = fit_ode_phase([0,1,2,3], "s+/- (70M-1B)")
spp = fit_ode_phase([5,6,7], "s++ (2.8B-12B)")
trans = fit_ode_phase([3,4,5], "Transition (1B-2.8B)")
full = fit_ode_phase(list(range(8)), "Full (70M-12B)")

print("\n" + "=" * 70)
print("COUPLING COEFFICIENT COMPARISON")
print("=" * 70)
print(f"\n  s+/- TQA coupling to HS: {spm['TQA_eq']['HS_coef']:.3f}")
print(f"  s++ TQA coupling to HS:  {spp['TQA_eq']['HS_coef']:.3f}")
print(f"  Full TQA coupling to HS: {full['TQA_eq']['HS_coef']:.3f}")
print(f"\n  s+/- mean gamma_12: {spm['gamma_12_mean']:.3f}")
print(f"  s++ mean gamma_12:  {spp['gamma_12_mean']:.3f}")
sign_flip = (spm['gamma_12_mean'] < 0) != (spp['gamma_12_mean'] < 0)
print(f"\n  SIGN FLIP IN gamma_12? {'YES' if sign_flip else 'NO'}")

all_results = {'s_pm': spm, 's_pp': spp, 'transition': trans, 'full': full, 'sign_flip': sign_flip}
with open(os.path.join(DIR, 'pysindy_per_phase.json'), 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved: pysindy_per_phase.json")
