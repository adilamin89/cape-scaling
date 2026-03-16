#!/usr/bin/env python3
"""
CAPE Quickstart — classify any model in 10 lines.

Usage:
    python scripts/quickstart.py
    python scripts/quickstart.py --N 7 --hs 78 --tqa 43
"""
import argparse
import numpy as np

# ── CAPE phase classifier (from paper §2–§5) ──
NC = 3.5e9          # critical scale (parameters)
SLOPE = 0.629       # γ₁₂ running coupling slope
INTERCEPT = -5.886  # γ₁₂ intercept
THETA_STAR = 38.8   # phase-locking angle (degrees)

# Algebraic phase classifier: TQA > √(0.187 · HS) → cooperative
ALPHA_RATIO = 0.187  # a₂/b₂ from isocline analysis

def classify_model(N_params, hs_score, tqa_score):
    """Classify a model into CAPE alignment phase."""
    logN = np.log10(N_params)
    gamma = SLOPE * logN + INTERCEPT

    # Phase from coupling sign
    if gamma < -0.3:
        phase = "TAX"
        desc = "Alignment tax: scaling hurts truthfulness"
    elif gamma < 0.3:
        phase = "TRANSITION"
        desc = f"Near Nc — maximum leverage for interventions"
    else:
        phase = "BONUS"
        desc = "Alignment bonus: scaling helps truthfulness"

    # Algebraic classifier (zero free parameters)
    tqa_boundary = np.sqrt(ALPHA_RATIO * hs_score) * 100
    algebraic = "ABOVE" if tqa_score > tqa_boundary else "BELOW"

    # h-field (data quality offset)
    # Using Phi-class reference: h = 0 for web-trained at this scale
    h_field = tqa_score - (0.187 * hs_score + 25.3)

    return {
        "phase": phase,
        "description": desc,
        "gamma_12": round(gamma, 3),
        "algebraic_classifier": algebraic,
        "tqa_boundary": round(tqa_boundary, 1),
        "h_field": round(h_field, 1),
        "recommendation": (
            "Invest in data curation to shift Nc→0"
            if phase == "TAX" else
            "Maximum RLHF leverage — small changes have outsized effect"
            if phase == "TRANSITION" else
            "Scale uniformly — cooperative regime confirmed"
        )
    }


def classify_frontier(swe_score, gpqa_score):
    """Classify a frontier model on the SWE×GPQA axis."""
    # Cooperative regression: GPQA = 0.788 * SWE + 24.9
    gpqa_pred = 0.788 * swe_score + 24.9
    h = gpqa_score - gpqa_pred

    return {
        "h_field": round(h, 1),
        "interpretation": (
            "Reasoning-rich (above cooperative curve)"
            if h > 3 else
            "Coding-rich (below cooperative curve)"
            if h < -3 else
            "On the cooperative manifold"
        ),
        "recommendation": (
            f"Invest in agentic coding (SWE-type capability)"
            if h > 3 else
            f"Invest in scientific reasoning (GPQA-type capability)"
            if h < -3 else
            "Scale uniformly — already efficient"
        )
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CAPE phase classifier")
    parser.add_argument("--N", type=float, default=7, help="Parameters in billions")
    parser.add_argument("--hs", type=float, default=78, help="HellaSwag score (%)")
    parser.add_argument("--tqa", type=float, default=43, help="TruthfulQA score (%)")
    parser.add_argument("--swe", type=float, default=None, help="SWE-bench score (%) for frontier")
    parser.add_argument("--gpqa", type=float, default=None, help="GPQA Diamond score (%) for frontier")
    args = parser.parse_args()

    print("=" * 55)
    print("CAPE — Capability Coupling Phase Classifier")
    print("=" * 55)

    result = classify_model(args.N * 1e9, args.hs, args.tqa)
    print(f"\nModel: {args.N}B params, HS={args.hs}%, TQA={args.tqa}%")
    print(f"Phase:      {result['phase']}")
    print(f"γ₁₂:       {result['gamma_12']}")
    print(f"h-field:    {result['h_field']}")
    print(f"Algebraic:  TQA {'>' if result['algebraic_classifier']=='ABOVE' else '<'} {result['tqa_boundary']}% boundary")
    print(f"→ {result['description']}")
    print(f"→ {result['recommendation']}")

    if args.swe is not None and args.gpqa is not None:
        fr = classify_frontier(args.swe, args.gpqa)
        print(f"\nFrontier axis: SWE={args.swe}%, GPQA={args.gpqa}%")
        print(f"h-field:    {fr['h_field']}")
        print(f"→ {fr['interpretation']}")
        print(f"→ {fr['recommendation']}")

    # Demo: classify some well-known models
    print("\n" + "=" * 55)
    print("Example classifications:")
    print("=" * 55)
    examples = [
        ("Pythia-1B", 1, 49.7, 38.9),
        ("OPT-6.7B", 6.7, 67.2, 34.9),
        ("Phi-4 (14B)", 14, 84.4, 65),
        ("Llama-3-70B", 70, 85.3, 46.2),
    ]
    for name, n, hs, tqa in examples:
        r = classify_model(n * 1e9, hs, tqa)
        print(f"  {name:20} γ₁₂={r['gamma_12']:+.2f}  h={r['h_field']:+.1f}  [{r['phase']}]")
