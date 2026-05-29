#!/usr/bin/env python3
"""Verify Paper 3A headline numbers against supplement data."""
import json, numpy as np, os, sys

DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(DIR, '..', 'data')

passed = failed = 0
def check(name, cond, detail=""):
    global passed, failed
    if cond: print(f"  PASS: {name}"); passed += 1
    else: print(f"  FAIL: {name} — {detail}"); failed += 1

print("="*50)
print("Paper 3A Verification")
print("="*50)

g4 = json.load(open(os.path.join(DATA, 'gemma4_results.json')))
check("Gemma-4-E4B-it coupling ≈ 0.952", abs(g4['coupling']['HS_TQA']['net'] - 0.952) < 0.001)
check("Gemma-4-E4B-it d_rep ≈ 26.3", abs(g4['d_eff'] - 26.3) < 0.1)

g3 = json.load(open(os.path.join(DATA, 'gemma3_results.json')))
check("Gemma-3-4B coupling ≈ 0.965", abs(g3['coupling']['HS_TQA']['net'] - 0.965) < 0.001)
check("Gemma-3-4B d_rep ≈ 22.4", abs(g3['d_eff'] - 22.4) < 0.1)
check("PLE: Gemma-3 coupling > Gemma-4", g3['coupling']['HS_TQA']['net'] > g4['coupling']['HS_TQA']['net'])
check("PLE: Gemma-4 d_rep > Gemma-3", g4['d_eff'] > g3['d_eff'])

q3 = json.load(open(os.path.join(DATA, 'qwen3_8b_results.json')))
check("Qwen3-8B coupling ≈ 0.741", abs(q3['coupling']['HS_TQA']['net'] - 0.741) < 0.001)

qg = json.load(open(os.path.join(DATA, 'qwen_generation.json')))
check("Qwen generation data loaded", 'finding' in qg)

# Intervention results
iv = json.load(open(os.path.join(DATA, 'intervention_results_v2.json')))
r1b = iv['pythia_1b']
check("1B hidden coupling ≈ 0.725", abs(r1b['hidden_coupling']['cosine'] - 0.725) < 0.01)
check("1B projection COMPRESSES (orig < hidden)", r1b['original_proj_coupling']['cosine'] < r1b['hidden_coupling']['cosine'])
check("1B wider RECOVERS (wider > orig)", r1b['wider_proj_coupling']['cosine'] > r1b['original_proj_coupling']['cosine'])

print(f"\n{'='*50}")
print(f"RESULTS: {passed} passed, {failed} failed")
if failed == 0: print("All checks PASS.")
else: sys.exit(1)
