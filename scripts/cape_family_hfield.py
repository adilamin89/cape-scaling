"""
cape_family_hfield.py
Family h-field signatures — macro-interpretability of training strategy

For each lab, computes mean h-field = deviation of GPQA from the universal
cooperative SWE-GPQA curve. Reveals training philosophy (reasoning-rich vs
coding-rich) from public benchmark scores alone.

Key result (March 2026, n=19 frontier models):
  Google    mean_h = +6.2  (reasoning-rich)
  OpenAI    mean_h = +3.1  (balanced, reasoning lean)
  Moonshot  mean_h = +1.9
  Alibaba   mean_h = +0.7
  DeepSeek  mean_h = -3.9  (coding-rich)
  Anthropic mean_h = -6.0  (coding-rich, with oscillation pattern)

Usage: python cape_family_hfield.py
"""
import numpy as np
from scipy import stats

MODELS = [
    ("Claude 3.5 Sonnet",  49.0, 59.4, "Anthropic"),
    ("Claude 3.7 Sonnet",  62.3, 68.0, "Anthropic"),
    ("Claude Haiku 4.5",   73.3, 71.0, "Anthropic"),
    ("Claude Sonnet 4.5",  77.2, 83.4, "Anthropic"),
    ("Claude Sonnet 4.6",  79.6, 74.1, "Anthropic"),
    ("Claude Opus 4.6",    80.8, 91.3, "Anthropic"),
    ("Gemini 2.5 Pro",     63.8, 84.0, "Google"),
    ("Gemini 3 Flash",     78.0, 90.4, "Google"),
    ("Gemini 3 Pro",       76.2, 91.9, "Google"),
    ("Gemini 3.1 Pro",     80.6, 94.3, "Google"),
    ("GPT-4o",             33.2, 53.6, "OpenAI"),
    ("GPT-5",              74.9, 85.7, "OpenAI"),
    ("GPT-5.1",            76.3, 88.1, "OpenAI"),
    ("GPT-5.2 Pro",        80.0, 93.2, "OpenAI"),
    ("DeepSeek V3.2",      74.4, 79.9, "DeepSeek"),
    ("Kimi K2.5",          76.8, 87.6, "Moonshot"),
    ("Qwen3.5-72B",        73.4, 83.7, "Alibaba"),
]

swe  = np.array([m[1] for m in MODELS])
gpqa = np.array([m[2] for m in MODELS])
sl, ic, r, p, _ = stats.linregress(swe, gpqa)
h_vals = gpqa - (sl * swe + ic)

print(f"Universal cooperative fit: GPQA = {sl:.2f}·SWE + {ic:.1f}")
print(f"r = {r:.3f}, p = {p:.6f}, n = {len(MODELS)}")
print()

# Per-model h
print(f"{'Model':24} {'SWE':5} {'GPQA':5} {'h':7} {'Family'}")
print("-" * 60)
for (name, sv, gv, fam), h in zip(MODELS, h_vals):
    print(f"  {name:22} {sv:4.1f}  {gv:4.1f}  {h:+5.1f}  {fam}")

# Per-family mean h
print()
print("FAMILY h-FIELD SIGNATURES (macro-interpretability):")
print(f"{'Lab':12} {'mean_h':8} {'n':4} {'interpretation'}")
print("-" * 55)
for fam in ["Google", "OpenAI", "Moonshot", "Alibaba", "DeepSeek", "Anthropic"]:
    fam_h = [h for m, h in zip(MODELS, h_vals) if m[3] == fam]
    n = len(fam_h)
    mean_h = np.mean(fam_h)
    interp = "reasoning-rich" if mean_h > 2 else ("coding-rich" if mean_h < -2 else "balanced")
    print(f"  {fam:12} {mean_h:+6.1f}   {n}   {interp}")

# Within-family gamma_12 trajectory
print()
print("WITHIN-FAMILY γ₁₂ TRAJECTORIES:")
for fam in ["Anthropic", "OpenAI", "Google"]:
    mods = sorted([(m[0], m[1], m[2]) for m in MODELS if m[3] == fam], key=lambda x: x[1])
    print(f"\n  {fam}:")
    for i, (name, sv, gv) in enumerate(mods):
        h = gv - (sl * sv + ic)
        if i == 0:
            print(f"    {name:22} SWE={sv:.1f} GPQA={gv:.1f} h={h:+.1f}  [baseline]")
        else:
            dswe = sv - mods[i-1][1]
            dgpqa = gv - mods[i-1][2]
            gamma = dgpqa / dswe if dswe != 0 else float('nan')
            flag = " ← TAX" if gamma < 0 else (" ← STRONG BONUS" if gamma > 5 else "")
            print(f"    {name:22} SWE={sv:.1f} GPQA={gv:.1f} h={h:+.1f}  γ₁₂={gamma:+.2f}{flag}")

# Nc2 algebraic classifier
print()
print("Nc2 ALGEBRAIC CLASSIFIER: GPQA_c = sqrt(0.79 · SWE)")
print("(analogous to Nc1 classifier TQA_c = sqrt(0.187 · HS))")
print(f"{'Model':24} {'GPQA_c':7} {'diff':7} {'phase'}")
print("-" * 55)
for (name, sv, gv, fam), h in zip(MODELS, h_vals):
    gpqa_c = np.sqrt(0.79 * sv)
    diff = gv - gpqa_c
    phase = "BONUS" if diff > 0 else "TAX"
    flag = " ◄" if phase == "TAX" else ""
    print(f"  {name:22} {gpqa_c:5.1f}   {diff:+5.1f}  {phase}{flag}")
