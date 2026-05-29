#!/usr/bin/env python3
"""Verify Paper 3B headline numbers against supplement data."""
import json, numpy as np, os, sys

DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(DIR, '..', 'data')

passed = failed = 0
def check(name, cond, detail=""):
    global passed, failed
    if cond: print(f"  PASS: {name}"); passed += 1
    else: print(f"  FAIL: {name} — {detail}"); failed += 1

print("="*50)
print("Paper 3B Verification")
print("="*50)

fr = json.load(open(os.path.join(DATA, 'frontier_34models.json')))
check("Total models = 34", fr['total_models'] == 34)
check("Labs = 10", fr['n_labs'] == 10)
check("r_all ≈ 0.72", abs(fr['r_all'] - 0.72) < 0.01)
check("Slope ≈ 0.51", abs(fr['slope'] - 0.51) < 0.01)
check("Intercept ≈ 46.4", abs(fr['intercept'] - 46.4) < 0.5)
check("Holdout MAE ≈ 9.2", abs(fr.get('holdout_mae_mean', 0) - 9.2) < 0.5)
check("DeepSeek swing ≈ 15.9", abs(fr.get('deepseek_swing_pp', 0) - 15.9) < 0.5)

for lab, exp_h in [('Google', 5.5), ('Anthropic', -6.9)]:
    if lab in fr.get('per_lab', {}):
        actual = fr['per_lab'][lab].get('mean_h_core')
        if actual is not None:
            check(f"{lab} core h ≈ {exp_h:+.1f}", abs(actual - exp_h) < 0.5)

print(f"\n{'='*50}")
print(f"RESULTS: {passed} passed, {failed} failed")
if failed == 0: print("All checks PASS.")
else: sys.exit(1)
