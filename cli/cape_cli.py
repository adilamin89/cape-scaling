#!/usr/bin/env python3
"""
CAPE CLI — Capability Coupling Analysis of Phase Emergence
Usage:
  cape h-field --swe 80.8 --gpqa 91.3
  cape coupling --family OPT
  cape steer --model pythia-410m --prompt "Vaccines cause autism because"
  cape predict --check
"""
import argparse
import json
import sys

SLOPE, INTERCEPT = 0.513, 46.4

LAB_H = {
    "Google": 5.5, "xAI": 5.1, "OpenAI": 3.1, "Alibaba": 2.8, "Meta": 2.4,
    "DeepSeek": 1.9, "Moonshot": 1.8, "Zhipu": -0.3, "MiniMax": -2.3, "Anthropic": -6.9
}

OPT_LADDER = {
    "125M": 0.514, "1.3B": 0.645, "6.7B": 0.741,
    "13B": 0.876, "30B": 0.356, "66B": 0.396
}


def cmd_hfield(args):
    """Compute h-field from SWE + GPQA scores."""
    h = args.gpqa - (SLOPE * args.swe + INTERCEPT)
    predicted = SLOPE * args.swe + INTERCEPT

    if h > 10:
        phase = "Reasoning-specialist"
    elif h < -10:
        phase = "Coding-specialist excursion"
    else:
        phase = "Cooperative (on trend)"

    direction = "Reasoning-rich" if h > 0 else "Coding-rich"

    # Find nearest lab
    nearest = min(LAB_H.items(), key=lambda x: abs(x[1] - h))

    print(f"\n  CAPE h-field Diagnostic")
    print(f"  {'='*40}")
    print(f"  SWE-bench:      {args.swe:.1f}%")
    print(f"  GPQA Diamond:   {args.gpqa:.1f}%")
    print(f"  ────────────────────────────────")
    print(f"  h-field:        {h:+.1f} pp")
    print(f"  Phase:          {phase}")
    print(f"  Direction:      {direction}")
    print(f"  Predicted GPQA: {predicted:.1f}%")
    print(f"  Nearest lab:    {nearest[0]} (h={nearest[1]:+.1f})")
    print()

    if h < -5:
        print(f"  Suggestion: Coding-heavy. Reasoning investment has")
        print(f"  highest marginal coupling return.")
    elif h > 5:
        print(f"  Suggestion: Reasoning-saturated. Coding investment")
        print(f"  may have higher marginal value.")
    else:
        print(f"  Balanced on the cooperation trend.")
        print(f"  Either axis investment is efficient.")
    print()

    if args.json:
        result = {
            "swe": args.swe, "gpqa": args.gpqa, "h": round(h, 2),
            "phase": phase, "direction": direction,
            "predicted_gpqa": round(predicted, 2),
            "nearest_lab": nearest[0]
        }
        print(json.dumps(result, indent=2))


def cmd_coupling(args):
    """Show coupling trajectory for a model family."""
    if args.family.upper() == "OPT":
        print(f"\n  OPT Internal Coupling Trajectory")
        print(f"  {'='*45}")
        print(f"  {'Size':<8} {'Coupling':<10} {'Status'}")
        print(f"  {'─'*45}")
        for size, coupling in OPT_LADDER.items():
            if coupling > 0.8:
                status = "PEAK"
            elif coupling < 0.4:
                status = "Nc2 DROP" if size == "30B" else "Recovery"
            else:
                status = "Rising" if coupling < 0.7 else "Cooperative"
            print(f"  {size:<8} {coupling:<10.3f} {status}")
        print()
        print(f"  Pattern: rise → peak (13B) → drop (30B) → recovery (66B)")
        print(f"  This is the Nc2 cascade — same pattern as Nc1 in Pythia")
    else:
        print(f"  Family '{args.family}' — use OPT for full internal ladder")
        print(f"  Available: OPT (125M-66B with internal coupling data)")


def cmd_predict(args):
    """Check prediction status."""
    predictions = [
        ("SWE saturation", "Dec 2026", "Pending"),
        ("IFEval activation", "Dec 2026", "Pending"),
        ("DeepSeek coding-first", "Next 2 releases", "Pending"),
        ("Google reasoning advantage", "Next 2 releases", "Pending"),
        ("Cooperative coupling persists", "May 2027", "Pending"),
        ("IFEval→HLE handoff (Nc4)", "Dec 2027", "Pending"),
        ("SWE-HLE decoupling", "Dec 2026", "Pending"),
    ]
    confirmed = [
        ("OLMo γ₁₂ = 0.000", "Confirmed", "Independent lab, independent training"),
        ("Llama-2 holdout 5.6% MAE", "Confirmed", "Cross-family, 2× polynomial"),
        ("Qwen3 cooperative all scales", "Confirmed", "Tax eliminated by curation"),
    ]

    print(f"\n  CAPE Predictions")
    print(f"  {'='*55}")
    print(f"\n  Already Confirmed:")
    for name, status, note in confirmed:
        print(f"  ✅ {name}: {note}")

    print(f"\n  Falsifiable Predictions:")
    for i, (name, deadline, status) in enumerate(predictions, 1):
        print(f"  ⏳ #{i} {name} (by {deadline}) — {status}")
    print()


def cmd_steer(args):
    """Run live activation-level steering on any HuggingFace model."""
    try:
        from cape_steer import CAPESteerer
    except ImportError:
        print("  Install dependencies: pip install torch transformers")
        print("  Then run from the cli/ directory.")
        return

    steerer = CAPESteerer(args.model, device=getattr(args, 'device', 'auto'))
    result = steerer.steer(args.prompt, max_new_tokens=getattr(args, 'max_tokens', 50))

    print(f"\n  CAPE Self-Steering — Live Activation Intervention")
    print(f"  {'='*55}")
    print(f"  Model: {result.model_name}")
    print(f"  Probe layer: {result.probe_layer}/{result.num_layers} (quarter-depth)")
    print(f"  Phase: {result.phase} (cos_truth={result.cos_truth:.4f})")
    print(f"  Correction strength: {result.correction_strength}")
    print()
    print(f"  WITHOUT steering:")
    prompt_len = len(result.prompt)
    print(f"    {result.normal_text[prompt_len:][:200]}")
    print()
    print(f"  WITH steering (layer {result.probe_layer}):")
    print(f"    {result.steered_text[prompt_len:][:200]}")
    print()
    print(f"  Changed: {'YES' if result.changed else 'NO'}")
    print()


def main():
    parser = argparse.ArgumentParser(
        prog='cape',
        description='CAPE: Capability Coupling Analysis of Phase Emergence'
    )
    sub = parser.add_subparsers(dest='command')

    # h-field
    p_h = sub.add_parser('h-field', help='Compute h-field from SWE + GPQA')
    p_h.add_argument('--swe', type=float, required=True, help='SWE-bench Verified score')
    p_h.add_argument('--gpqa', type=float, required=True, help='GPQA Diamond score')
    p_h.add_argument('--json', action='store_true', help='Output as JSON')

    # coupling
    p_c = sub.add_parser('coupling', help='Show coupling trajectory')
    p_c.add_argument('--family', type=str, default='OPT', help='Model family')

    # predict
    p_p = sub.add_parser('predict', help='Check prediction status')
    p_p.add_argument('--check', action='store_true')

    # steer
    p_s = sub.add_parser('steer', help='Self-aligning steering demo')
    p_s.add_argument('--model', type=str, default='pythia-410m')
    p_s.add_argument('--prompt', type=str, default='The earth is flat because')

    args = parser.parse_args()

    if args.command == 'h-field':
        cmd_hfield(args)
    elif args.command == 'coupling':
        cmd_coupling(args)
    elif args.command == 'predict':
        cmd_predict(args)
    elif args.command == 'steer':
        cmd_steer(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
