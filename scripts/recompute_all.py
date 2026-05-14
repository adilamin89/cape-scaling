#!/usr/bin/env python3
"""
CAPE Data Pipeline — recompute all dashboard data from frozen + post-cutoff sources.

Usage:
    python scripts/recompute_all.py              # recompute from existing data
    python scripts/recompute_all.py --add-model   # interactive: add a new model

Reads:
    data/frontier_final_consolidated.json    (frozen March 2026, NEVER modified)
    data/post_cutoff.json                    (new models added here)

Writes:
    data/computed/frontier_all.json          (merged panel for dashboard)
    data/computed/hfield_per_lab.json        (per-lab h-field diagnostics)
    data/computed/predictions_status.json    (7 predictions with current status)
    data/computed/summary.json              (headline numbers for dashboard header)

The dashboard (index.html or zehenlabs.com/cape/) can fetch these JSON files
instead of hardcoding values.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / "data"
COMPUTED_DIR = DATA_DIR / "computed"
COMPUTED_DIR.mkdir(parents=True, exist_ok=True)

# Paper's frozen regression (NEVER changes)
SLOPE = 0.513
INTERCEPT = 46.4
R_FULL = 0.7235
N_FULL = 34


def load_frozen():
    """Load the March 2026 frozen frontier panel."""
    path = DATA_DIR / "frontier_final_consolidated.json"
    if not path.exists():
        # Try supplement location
        alt = REPO_ROOT.parent / "supplement_3b" / "data" / "frontier_34models.json"
        if alt.exists():
            path = alt
        else:
            print(f"WARNING: No frozen data found at {path}")
            return {"models": []}
    with open(path) as f:
        return json.load(f)


def load_post_cutoff():
    """Load post-cutoff models (added manually)."""
    path = DATA_DIR / "post_cutoff.json"
    if not path.exists():
        # Try supplement location
        alt = REPO_ROOT.parent / "supplement_3b" / "data" / "post_cutoff_april2026.json"
        if alt.exists():
            with open(alt) as f:
                return json.load(f)
        return []
    with open(path) as f:
        return json.load(f)


def compute_hfield(swe, gpqa):
    """Compute h-field using frozen regression."""
    return round(gpqa - (SLOPE * swe + INTERCEPT), 1)


def merge_panels(frozen, post_cutoff):
    """Merge frozen and post-cutoff into unified panel."""
    all_models = []

    # Frozen models
    if "models" in frozen:
        for m in frozen["models"]:
            m["subset"] = m.get("subset", "frozen")
            m["h"] = compute_hfield(m.get("swe", 0), m.get("gpqa", 0))
            all_models.append(m)
    elif isinstance(frozen, list):
        for m in frozen:
            m["subset"] = "frozen"
            m["h"] = compute_hfield(m.get("swe", 0), m.get("gpqa", 0))
            all_models.append(m)

    # Post-cutoff
    if isinstance(post_cutoff, list):
        for m in post_cutoff:
            m["subset"] = "post_cutoff"
            if "h" not in m:
                m["h"] = compute_hfield(m.get("swe", 0), m.get("gpqa", 0))
            all_models.append(m)
    elif isinstance(post_cutoff, dict) and "models" in post_cutoff:
        for m in post_cutoff["models"]:
            m["subset"] = "post_cutoff"
            m["h"] = compute_hfield(m.get("swe", 0), m.get("gpqa", 0))
            all_models.append(m)

    return all_models


def compute_per_lab(models):
    """Compute per-lab h-field statistics."""
    labs = {}
    for m in models:
        lab = m.get("lab", "Unknown")
        if lab not in labs:
            labs[lab] = {"models": [], "h_values": [], "n": 0}
        labs[lab]["models"].append(m.get("model", m.get("name", "unknown")))
        labs[lab]["h_values"].append(m.get("h", 0))
        labs[lab]["n"] += 1

    result = {}
    for lab, data in sorted(labs.items(), key=lambda x: -sum(x[1]["h_values"]) / max(len(x[1]["h_values"]), 1)):
        h_vals = data["h_values"]
        mean_h = round(sum(h_vals) / len(h_vals), 1) if h_vals else 0
        result[lab] = {
            "n": data["n"],
            "mean_h": mean_h,
            "direction": "Reasoning-rich" if mean_h > 3 else ("Coding-rich" if mean_h < -3 else "Balanced"),
            "models": data["models"],
            "h_values": [round(h, 1) for h in h_vals],
        }

    return result


def check_predictions(models):
    """Check 7 predictions against current data."""
    predictions = [
        {
            "id": 1,
            "name": "SWE saturation",
            "deadline": "Dec 2026",
            "pass": "Top-5 SWE spread < 2pp",
            "fail": "Spread > 5pp",
            "status": "pending",
        },
        {
            "id": 2,
            "name": "IFEval activation",
            "deadline": "Dec 2026",
            "pass": "r(GPQA, IFEval) > +0.6, n >= 8",
            "fail": "r < 0.3",
            "status": "pending",
        },
        {
            "id": 3,
            "name": "DeepSeek coding-first",
            "deadline": "Next 2 releases",
            "pass": "h < 0 both",
            "fail": "h > +5 either",
            "status": "pending",
        },
        {
            "id": 4,
            "name": "Google reasoning advantage",
            "deadline": "Next 2 releases",
            "pass": "h > +3 both",
            "fail": "h < 0 either",
            "status": "pending",
        },
        {
            "id": 5,
            "name": "Cooperative coupling persists",
            "deadline": "May 2027",
            "pass": "r(SWE, GPQA) > +0.5, n >= 30",
            "fail": "r < 0.3",
            "status": "pending",
        },
        {
            "id": 6,
            "name": "IFEval -> HLE handoff (Nc4)",
            "deadline": "Dec 2027",
            "pass": "IFEval spread < 3pp, HLE > 15pp",
            "fail": "IFEval > 8pp",
            "status": "pending",
        },
        {
            "id": 7,
            "name": "SWE-HLE decoupling",
            "deadline": "Dec 2026",
            "pass": "r(SWE, HLE) < 0, n >= 10",
            "fail": "r > +0.3",
            "status": "pending",
        },
    ]

    # Check SWE saturation with current data
    swe_scores = sorted([m.get("swe", 0) for m in models if m.get("swe", 0) > 70], reverse=True)
    if len(swe_scores) >= 5:
        top5_spread = swe_scores[0] - swe_scores[4]
        predictions[0]["current_spread"] = round(top5_spread, 1)
        if top5_spread < 2:
            predictions[0]["status"] = "likely_pass"
        elif top5_spread > 5:
            predictions[0]["status"] = "likely_fail"

    return predictions


def compute_summary(models, per_lab):
    """Compute headline summary numbers."""
    frozen = [m for m in models if m.get("subset") != "post_cutoff"]
    post = [m for m in models if m.get("subset") == "post_cutoff"]

    return {
        "n_base": 63,
        "n_families": 16,
        "n_frozen": len(frozen),
        "n_post_cutoff": len(post),
        "n_total": len(models),
        "n_labs": len(per_lab),
        "slope": SLOPE,
        "intercept": INTERCEPT,
        "r_full": R_FULL,
        "nc": 3.5,
        "nc_ci": [2.9, 13.4],
        "r_pythia_tax": -0.989,
        "ode_holdout_mae": 5.6,
        "cross_lab_holdout_mae": 9.2,
        "zero_competing_heads": "38/40",
        "updated": datetime.now().isoformat(),
    }


def add_model_interactive():
    """Interactive prompt to add a new model."""
    print("\n  Add New Frontier Model")
    print("  " + "=" * 40)
    name = input("  Model name: ").strip()
    lab = input("  Lab: ").strip()
    swe = float(input("  SWE-bench Verified (%): "))
    gpqa = float(input("  GPQA Diamond (%): "))
    date = input("  Release date (YYYY-MM-DD): ").strip()

    h = compute_hfield(swe, gpqa)

    model = {
        "model": name,
        "lab": lab,
        "swe": swe,
        "gpqa": gpqa,
        "h": h,
        "date": date,
        "subset": "post_cutoff",
    }

    print(f"\n  h-field: {h:+.1f}")
    print(f"  Phase: {'Reasoning-rich' if h > 5 else ('Coding-rich' if h < -5 else 'Cooperative')}")

    # Load existing post-cutoff and append
    path = DATA_DIR / "post_cutoff.json"
    existing = []
    if path.exists():
        with open(path) as f:
            existing = json.load(f)
    existing.append(model)

    with open(path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\n  Saved to {path}")
    print(f"  Total post-cutoff models: {len(existing)}")
    return model


def main():
    if "--add-model" in sys.argv:
        add_model_interactive()
        print("\n  Now run without --add-model to recompute all data.")
        return

    print("CAPE Data Pipeline")
    print("=" * 50)

    # Load data
    frozen = load_frozen()
    post_cutoff = load_post_cutoff()
    print(f"  Frozen models: {len(frozen.get('models', frozen)) if isinstance(frozen, dict) else len(frozen)}")
    print(f"  Post-cutoff models: {len(post_cutoff) if isinstance(post_cutoff, list) else len(post_cutoff.get('models', []))}")

    # Merge
    all_models = merge_panels(frozen, post_cutoff)
    print(f"  Total models: {len(all_models)}")

    # Compute
    per_lab = compute_per_lab(all_models)
    predictions = check_predictions(all_models)
    summary = compute_summary(all_models, per_lab)

    # Write computed files
    outputs = {
        "frontier_all.json": all_models,
        "hfield_per_lab.json": per_lab,
        "predictions_status.json": predictions,
        "summary.json": summary,
    }

    for filename, data in outputs.items():
        path = COMPUTED_DIR / filename
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Wrote {path}")

    # Print summary
    print(f"\n  Summary:")
    print(f"    Frozen: {summary['n_frozen']} models")
    print(f"    Post-cutoff: {summary['n_post_cutoff']} models")
    print(f"    Labs: {summary['n_labs']}")
    print(f"    Regression: GPQA = {SLOPE}*SWE + {INTERCEPT}")
    print(f"    Updated: {summary['updated']}")

    print(f"\n  Per-lab h-field:")
    for lab, data in per_lab.items():
        print(f"    {lab:12s}  h={data['mean_h']:+5.1f}  n={data['n']}  {data['direction']}")

    print(f"\n  Dashboard can now fetch from data/computed/")
    print(f"  To add a new model: python scripts/recompute_all.py --add-model")


if __name__ == "__main__":
    main()
