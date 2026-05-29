"""
cape_algebraic_nc.py
Predicts phase boundary (Nc) from ODE isocline condition.

The phase boundary in capability space is where dTQA/d(logN) = 0:
  TQA_c = sqrt(a/b · HS_c)

This is a zero-parameter prediction: calibrate a/b from any model
known to be at the transition (e.g. OLMo with γ₁₂=0.000),
then classify any other model's phase from its (HS, TQA) scores alone.

No parameter fitting, no bootstrap, no power-law fit.
Pure ODE algebra.
"""
import numpy as np

# Calibrate a/b from OLMo (γ₁₂=0.000 — exactly at phase boundary)
HS_olmo_mean = (0.625 + 0.764) / 2   # mean of OLMo 1B and 7B
TQA_olmo = 0.360                       # unchanged across transition
a_over_b = TQA_olmo**2 / HS_olmo_mean
print(f"Phase boundary parameter a/b = {a_over_b:.4f}")
print(f"Calibrated from OLMo (γ₁₂=0.000 at ~4B)")
print()

def phase_from_benchmarks(HS, TQA, a_over_b):
    """
    Classify model phase from benchmark scores.
    Returns: "BONUS" if TQA > TQA_c, "TAX" if TQA < TQA_c, "BOUNDARY" if close
    Also returns the predicted TQA at the phase boundary for this HS value.
    """
    TQA_c = np.sqrt(a_over_b * HS)
    diff = TQA - TQA_c
    if abs(diff) < 0.01:
        phase = "BOUNDARY"
    elif diff > 0:
        phase = "BONUS"
    else:
        phase = "TAX"
    return phase, TQA_c

# Apply to all families
print("PHASE CLASSIFICATION FROM CAPABILITY SPACE")
print(f"{'Family':28} {'HS':6} {'TQA':6} {'TQA_c':7} {'Phase'}")
print("-"*60)
models = [
    ("Pythia-70M",    0.27, 0.48),
    ("Pythia-1B",     0.57, 0.38),
    ("Pythia-2.8B",   0.64, 0.36),
    ("Pythia-6.9B",   0.65, 0.34),
    ("Pythia-12B",    0.67, 0.32),
    ("OLMo-1B",       0.63, 0.36),
    ("OLMo-7B",       0.76, 0.36),
    ("OPT-1.3B",      0.54, 0.35),
    ("OPT-6.7B",      0.67, 0.37),
    ("OPT-30B",       0.73, 0.37),
    ("Llama-1-7B",    0.78, 0.34),
    ("Llama-2-7B",    0.78, 0.39),
    ("Llama-2-70B",   0.87, 0.45),
    ("Llama-3-8B",    0.82, 0.44),
    ("Phi-2.7B",      0.75, 0.44),
    ("Phi-3-mini",    0.79, 0.65),
]
for name, HS, TQA in models:
    phase, TQA_c = phase_from_benchmarks(HS, TQA, a_over_b)
    print(f"  {name:26} {HS:.2f}  {TQA:.2f}  {TQA_c:.3f}  {phase}")

print()
print("PREDICTION FOR Nc2 (SWE/GPQA frontier):")
print("Analogous condition: GPQA_c = sqrt(a₂/b₂ · SWE_c)")
print("a₂/b₂ estimable from slope of SWE-GPQA cooperative fit (0.79)")
print("Nc2 = scale where a model's GPQA first satisfies this condition")
