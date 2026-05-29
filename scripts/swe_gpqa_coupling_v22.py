"""
swe_gpqa_coupling_v22.py
CAPE frontier coupling analysis — 20 models, 7 labs (Mar 2026)
r=+0.848, p=0.000002
"""
import numpy as np
from scipy import stats

FRONTIER_MODELS = [
    # (name, SWE_verified, GPQA_diamond, family)
    ("Claude 3.5 Sonnet",  49.0, 59.4, "Anthropic"),
    ("Claude 3.7 Sonnet",  62.3, 68.0, "Anthropic"),
    ("Claude Haiku 4.5",   73.3, 71.0, "Anthropic"),  # GPQA estimated
    ("Claude Sonnet 4.5",  77.2, 83.4, "Anthropic"),
    ("Claude Opus 4.5",    80.9, 87.0, "Anthropic"),
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
    ("GPT-5.4",            77.2, 84.2, "OpenAI"),
    ("DeepSeek V3.2",      74.4, 79.9, "DeepSeek"),
    ("Kimi K2.5",          76.8, 87.6, "Moonshot"),
    ("Qwen3.5-397B",       73.4, 88.4, "Alibaba"),
    ("MiniMax M2.5",       80.2, 85.0, "MiniMax"),
]

def run_analysis():
    swe = np.array([m[1] for m in FRONTIER_MODELS])
    gpqa = np.array([m[2] for m in FRONTIER_MODELS])
    sl, ic, r, p, _ = stats.linregress(swe, gpqa)
    h_vals = gpqa - (sl*swe + ic)
    
    print(f"n={len(FRONTIER_MODELS)}: r={r:.3f}, p={p:.6f}")
    print(f"Fit: GPQA = {sl:.2f}*SWE + {ic:.1f}")
    print()
    for (name,sv,gv,fam),h in zip(FRONTIER_MODELS, h_vals):
        hs = ("+" if h>=0 else "") + f"{h:.1f}"
        print(f"  {name:22} SWE={sv:4.1f} GPQA={gv:4.1f} h={hs:6} [{fam}]")
    
    print("\nWithin-family γ₁₂:")
    for fam in ["Anthropic","OpenAI","Google"]:
        mods = sorted([(m[0],m[1],m[2]) for m in FRONTIER_MODELS if m[3]==fam], key=lambda x:x[1])
        print(f"\n  {fam}:")
        for i in range(len(mods)-1):
            ds = mods[i+1][1]-mods[i][1]
            dg = mods[i+1][2]-mods[i][2]
            if abs(ds) > 0.5:
                g12 = dg/ds
                tag = "COOP" if g12>0 else "TAX"
                print(f"    {mods[i][0][:20]:20}→{mods[i+1][0][:20]:20} γ₁₂={g12:+.2f} [{tag}]")

if __name__ == "__main__":
    run_analysis()
