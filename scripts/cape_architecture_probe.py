"""
cape_architecture_probe.py
Architecture probe: γ₁₂ at ~1B across model families
Shows transition is thermodynamic not architectural
Result: OLMo-1B (16L same as Pythia-1B) gives γ₁₂=0.000, not -0.64
→ layer count alone doesn't explain the dip; sign change is physical
"""
import numpy as np

# Benchmark scores at ~1B from published lm-eval evaluations
NEAR_1B = [
    # (family, N1, HS1, TQA1, N2, HS2, TQA2, layers, note)
    ("Pythia",   1.0e9, 57.4, 38.0, 1.4e9, 59.6, 36.6, 16, "anomaly: 16L"),
    ("OPT",      1.3e9, 54.5, 35.5, 2.7e9, 61.0, 36.1, 24, "normal 24L"),
    ("BLOOM",    1.1e9, 42.0, 36.4, 3.0e9, 53.5, 37.2, 24, "normal 24L"),
    ("OLMo",     1.0e9, 62.5, 36.0, 7.0e9, 76.4, 36.0, 16, "16L control"),
]

print(f"{'Family':10} {'γ₁₂':8} {'Layers':8} {'Sign':6} {'Note'}")
print("-"*55)
for fam, N1, HS1, TQA1, N2, HS2, TQA2, L, note in NEAR_1B:
    dHS = HS2 - HS1
    dTQA = TQA2 - TQA1
    g12 = dTQA/dHS if abs(dHS) > 0.1 else 0.0
    sign = "TAX" if g12 < -0.1 else ("ZERO" if abs(g12) < 0.1 else "COOP")
    print(f"  {fam:8} {g12:+7.3f}  {L:6}L   {sign:6} {note}")

print("""
Key result:
  OLMo-1B (16 layers = same as Pythia-1B) → γ₁₂ = 0.000
  Pythia-1B (16 layers) → γ₁₂ = -0.64
  Layer count does NOT explain the sign; the transition is thermodynamic.
  OPT and BLOOM (24 layers) also show γ₁₂ ≈ 0 at ~1B → in transition region.
""")
