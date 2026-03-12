import numpy as np
import os
from scipy.stats import pearsonr
import json

FRONTIER_2026 = {
  "Claude Sonnet 4.5":   (70,   77.2, 83.4),
  "Claude Sonnet 4.6":   (70,   79.6, 74.1),
  "Claude Opus 4.6":     (500,  80.8, 91.3),
  "GPT-5.2 Pro":         (1000, 80.0, 93.2),
  "Gemini 3 Flash":      (200,  78.0, 90.4),
  "Gemini 3 Pro":        (1000, 76.2, 91.9),
  "Gemini 3.1 Pro":      (1000, 80.6, 94.3),
  "DeepSeek V3.2":       (671,  74.4, 79.9),
  "Kimi K2.5":           (1000, 80.2, 85.2),
  "Qwen3.5-72B":         (72,   73.4, 83.7),
}
# columns: (N_B, SWE_bench_verified, GPQA_Diamond)

models_both = {k: v for k, v in FRONTIER_2026.items() if v[1] is not None and v[2] is not None}
swe = np.array([v[1] for v in models_both.values()])
gpqa = np.array([v[2] for v in models_both.values()])

r_all, p_all = pearsonr(swe, gpqa)
print(f"Cross-model r(SWE, GPQA) = {r_all:.3f}, p={p_all:.4f}, n={len(swe)}")

# Within Anthropic family
anthropic_keys = [k for k in FRONTIER_2026 if 'Claude' in k]
a_swe = np.array([FRONTIER_2026[k][1] for k in anthropic_keys])
a_gpqa = np.array([FRONTIER_2026[k][2] for k in anthropic_keys])
print(f"\nAnthropic family:")
for i, k in enumerate(anthropic_keys):
    print(f"  {k}: SWE={a_swe[i]}, GPQA={a_gpqa[i]}")
for i in range(len(anthropic_keys)-1):
    dswe = a_swe[i+1] - a_swe[i]
    dgpqa = a_gpqa[i+1] - a_gpqa[i]
    g = dgpqa/dswe if abs(dswe) > 0.1 else float('inf')
    print(f"  gamma12 {anthropic_keys[i]}->{anthropic_keys[i+1]}: {g:.2f}  (dSWE={dswe:.1f}, dGPQA={dgpqa:.1f})")

# Within Google family
google_keys = [k for k in FRONTIER_2026 if 'Gemini' in k]
g_swe = np.array([FRONTIER_2026[k][1] for k in google_keys if FRONTIER_2026[k][1] is not None])
g_gpqa = np.array([FRONTIER_2026[k][2] for k in google_keys if FRONTIER_2026[k][2] is not None])
n_g = min(len(g_swe), len(g_gpqa))
print(f"\nGoogle family:")
if n_g >= 2:
    r_g, p_g = pearsonr(g_swe[:n_g], g_gpqa[:n_g])
    print(f"  r(SWE,GPQA)={r_g:.3f}, p={p_g:.4f}, n={n_g}")

print(f"\nSonnet 4.6 anomaly:")
print(f"  Sonnet4.5->4.6: dSWE=+2.4, dGPQA=-9.3, gamma12={(-9.3/2.4):.2f} (NEGATIVE - anomalous)")
print(f"  Sonnet4.6->Opus4.6: dSWE=+1.2, dGPQA=+17.2, gamma12={(17.2/1.2):.2f} (large positive - cooperative)")

results = {
    'r_cross_model': float(r_all),
    'p_cross_model': float(p_all),
    'n_models': int(len(swe)),
    'cross_family_cooperative': bool(r_all > 0),
    'paper_claim_correct': bool(r_all > 0),
    'anthropic_gamma_sonnet45_to_46': float(-9.3/2.4),
    'anthropic_gamma_sonnet46_to_opus46': float(17.2/1.2),
    'sonnet46_anomaly': 'gamma<0 for Sonnet4.5->4.6 (SWE up, GPQA sharply down) — within-family s+/- remnant',
    'paper_update_needed': 'within-family coupling varies by tier; cross-family remains cooperative (r>0)',
}
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'swe_gpqa_coupling_v2.json'), 'w') as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
