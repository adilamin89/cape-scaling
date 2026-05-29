"""
Regenerate all frontier statistics for 34-model / 10-lab panel.
Produces frozen numbers for both papers + updates JSON metadata.
Run once, verify, then use these numbers everywhere.
"""
import json
import numpy as np
from collections import defaultdict
from pathlib import Path

# Load raw data
DATA_PATH = Path(__file__).parent.parent / "data" / "frontier_34models.json"
with open(DATA_PATH) as f:
    raw = json.load(f)

models = raw["models"]
print(f"Total models in file: {len(models)}")

# Extract arrays
names = [m["name"] for m in models]
labs = [m["lab"] for m in models]
swe = np.array([m["swe"] for m in models], dtype=float)
gpqa = np.array([m["gpqa"] for m in models], dtype=float)

unique_labs = sorted(set(labs))
print(f"Unique labs: {len(unique_labs)} — {unique_labs}")

# ── Define core subset (same 20 models as current paper) ──
# Core = one model per release per lab, matched benchmark variants, no compute-tier duplicates
CORE_NAMES = {
    "Claude 3.5 Sonnet", "Claude 3.7 Sonnet", "Claude Haiku 4.5",
    "Claude Sonnet 4.5", "Claude Opus 4.5", "Claude Sonnet 4.6", "Claude Opus 4.6",
    "DeepSeek-V3", "DeepSeek-R1", "DeepSeek V3.2",
    "Gemini 2.5 Pro", "Gemini 3 Flash", "Gemini 3 Pro", "Gemini 3.1 Pro",
    "Llama 4 Maverick", "Kimi K2.5", "Qwen3.5-397B", "MiniMax M2.5",
    "o3-mini", "o3", "GPT-5", "GPT-5.1", "GPT-5.4 std",
}
core_mask = np.array([n in CORE_NAMES for n in names])
core_swe = swe[core_mask]
core_gpqa = gpqa[core_mask]
core_names = [n for n, c in zip(names, core_mask) if c]
core_labs = [l for l, c in zip(labs, core_mask) if c]
print(f"\nCore subset: {len(core_swe)} models from {len(set(core_labs))} labs")

# ── Full-panel regression (34 models) ──
full_slope, full_intercept = np.polyfit(swe, gpqa, 1)
full_r = np.corrcoef(swe, gpqa)[0, 1]
full_n = len(swe)

# ── Core regression (20+ models) ──
core_slope, core_intercept = np.polyfit(core_swe, core_gpqa, 1)
core_r = np.corrcoef(core_swe, core_gpqa)[0, 1]
core_n = len(core_swe)

# ── h-field for ALL models using FULL-PANEL regression ──
h_all = gpqa - (full_slope * swe + full_intercept)

# ── Per-lab stats ──
lab_stats = {}
for lab in unique_labs:
    mask = np.array([l == lab for l in labs])
    core_lab_mask = np.array([l == lab for l in core_labs])

    lab_h = h_all[mask]
    lab_n = int(mask.sum())
    core_h_vals = (core_gpqa - (full_slope * core_swe + full_intercept))[core_lab_mask]

    lab_stats[lab] = {
        "n_total": lab_n,
        "n_core": int(core_lab_mask.sum()),
        "mean_h_all": float(np.mean(lab_h)),
        "mean_h_core": float(np.mean(core_h_vals)) if core_lab_mask.sum() > 0 else None,
        "std_h": float(np.std(lab_h)) if lab_n > 1 else None,
    }

# ── Leave-one-lab-out cross-validation ──
holdout_errors = {}
for held_lab in unique_labs:
    train_mask = np.array([l != held_lab for l in labs])
    test_mask = np.array([l == held_lab for l in labs])

    if test_mask.sum() < 1:
        continue

    train_swe, train_gpqa = swe[train_mask], gpqa[train_mask]
    test_swe, test_gpqa = swe[test_mask], gpqa[test_mask]

    s, i = np.polyfit(train_swe, train_gpqa, 1)
    pred_gpqa = s * test_swe + i
    mae = float(np.mean(np.abs(pred_gpqa - test_gpqa)))
    mae_pct = mae  # already in percentage points since scores are in %
    holdout_errors[held_lab] = {
        "n_test": int(test_mask.sum()),
        "mae_pp": round(mae_pct, 2),
    }

# Labs with ≥3 models for holdout reporting
holdout_labs_3plus = {k: v for k, v in holdout_errors.items() if v["n_test"] >= 3}
holdout_mae_mean = np.mean([v["mae_pp"] for v in holdout_labs_3plus.values()])
holdout_mae_std = np.std([v["mae_pp"] for v in holdout_labs_3plus.values()])

# ── Subset correlations ──
swe_ge40_mask = swe >= 40
no_compute_tier = np.array(["xhigh" not in n and "std" not in n for n in names])

subsets = {
    "full_panel": (np.ones(len(swe), dtype=bool), len(swe)),
    "core": (core_mask, core_mask.sum()),
    "swe_ge40": (swe_ge40_mask, swe_ge40_mask.sum()),
    "no_compute_tier": (no_compute_tier, no_compute_tier.sum()),
}
subset_r = {}
for name_s, (mask_s, n_s) in subsets.items():
    if n_s >= 3:
        subset_r[name_s] = {"r": float(np.corrcoef(swe[mask_s], gpqa[mask_s])[0, 1]), "n": int(n_s)}

# ── Per-model h-field table ──
model_table = []
for i, m in enumerate(models):
    is_core = names[i] in CORE_NAMES
    model_table.append({
        "name": names[i],
        "lab": labs[i],
        "swe": float(swe[i]),
        "gpqa": float(gpqa[i]),
        "h": round(float(h_all[i]), 1),
        "subset": "Core" if is_core else "Extended",
    })

# ── DeepSeek trajectory ──
ds_models = [(names[i], float(h_all[i])) for i in range(len(names)) if labs[i] == "DeepSeek"]
ds_h_vals = [h for _, h in ds_models]
ds_swing = max(ds_h_vals) - min(ds_h_vals)

# ── Print comparison with old values ──
print("\n" + "="*70)
print("COMPARISON: OLD (31/8) vs NEW (34/10)")
print("="*70)

old = {"slope": 0.52, "intercept": 45.7, "r_full": 0.729, "r_core": 0.854, "n": 31, "labs": 8}

print(f"\n{'Metric':30s} {'Old (31)':>12s} {'New (34)':>12s} {'Delta':>10s}")
print("-"*66)
print(f"{'Full panel n':30s} {old['n']:>12d} {full_n:>12d} {full_n-old['n']:>+10d}")
print(f"{'Labs':30s} {old['labs']:>12d} {len(unique_labs):>12d} {len(unique_labs)-old['labs']:>+10d}")
print(f"{'Full r':30s} {old['r_full']:>12.4f} {full_r:>12.4f} {full_r-old['r_full']:>+10.4f}")
print(f"{'Core r':30s} {old['r_core']:>12.4f} {core_r:>12.4f} {core_r-old['r_core']:>+10.4f}")
print(f"{'Slope':30s} {old['slope']:>12.4f} {full_slope:>12.4f} {full_slope-old['slope']:>+10.4f}")
print(f"{'Intercept':30s} {old['intercept']:>12.1f} {full_intercept:>12.1f} {full_intercept-old['intercept']:>+10.1f}")

print(f"\n{'Holdout MAE (labs≥3)':30s} {'7.1±2.4':>12s} {holdout_mae_mean:.1f}±{holdout_mae_std:.1f}")

# Per-lab h comparison
print(f"\n{'Lab':15s} {'n':>3s} {'Old h':>8s} {'New h(core)':>12s} {'Delta':>8s}")
print("-"*50)
old_h = {"Google": 5.7, "Anthropic": -6.7, "DeepSeek": 2.2, "OpenAI": 3.4,
         "Meta": 2.6, "Moonshot": 2.1, "Alibaba": 3.1, "MiniMax": -2.1}
for lab in sorted(lab_stats.keys(), key=lambda x: -(lab_stats[x]["mean_h_core"] or -999)):
    s = lab_stats[lab]
    old_val = old_h.get(lab, None)
    core_h = s["mean_h_core"]
    if core_h is not None and old_val is not None:
        print(f"{lab:15s} {s['n_core']:3d} {old_val:+8.1f} {core_h:+12.1f} {core_h-old_val:+8.1f}")
    elif core_h is not None:
        print(f"{lab:15s} {s['n_core']:3d} {'---':>8s} {core_h:+12.1f} {'NEW':>8s}")
    else:
        print(f"{lab:15s} {s['n_total']:3d} {'---':>8s} {'(ext only)':>12s} {'---':>8s}")

# DeepSeek trajectory
print(f"\nDeepSeek trajectory: ", end="")
for name_m, h_m in sorted(ds_models, key=lambda x: -x[1]):
    print(f"{name_m}={h_m:+.1f}  ", end="")
print(f"\nDeepSeek swing: {ds_swing:.1f} pp")

# Subset correlations
print(f"\nSubset correlations:")
for name_s, vals in subset_r.items():
    print(f"  {name_s:20s}: r={vals['r']:+.3f} (n={vals['n']})")

# Holdout per lab
print(f"\nHoldout MAE per lab (all):")
for lab, v in sorted(holdout_errors.items(), key=lambda x: x[1]["mae_pp"]):
    marker = " ***" if v["n_test"] >= 3 else ""
    print(f"  {lab:15s}: {v['mae_pp']:5.1f} pp (n={v['n_test']}){marker}")

# ── Update and save JSON ──
output = {
    "total_models": full_n,
    "n_labs": len(unique_labs),
    "labs": unique_labs,
    "r_all": round(float(full_r), 4),
    "r_core": round(float(core_r), 4),
    "n_core": int(core_n),
    "slope": round(float(full_slope), 4),
    "intercept": round(float(full_intercept), 1),
    "holdout_mae_mean": round(float(holdout_mae_mean), 1),
    "holdout_mae_std": round(float(holdout_mae_std), 1),
    "holdout_labs_3plus": holdout_labs_3plus,
    "holdout_all": holdout_errors,
    "per_lab": lab_stats,
    "subset_correlations": subset_r,
    "deepseek_swing_pp": round(float(ds_swing), 1),
    "models": model_table,
    "regression_equation": f"GPQA = {full_slope:.3f} * SWE + {full_intercept:.1f}",
    "generated": "2026-05-04 regenerate_frontier_stats.py",
}

OUT_PATH = Path(__file__).parent.parent / "data" / "frontier_34models_regen.json"
with open(OUT_PATH, "w") as f:
    json.dump(output, f, indent=2)
print(f"\n✓ Frozen stats written to {OUT_PATH}")

# Also update the original file's metadata (preserve model rows, update stats)
raw["total_models"] = full_n
raw["n_labs"] = len(unique_labs)
raw["r_all"] = round(float(full_r), 4)
raw["slope"] = round(float(full_slope), 4)
raw["intercept"] = round(float(full_intercept), 1)
with open(DATA_PATH, "w") as f:
    json.dump(raw, f, indent=2)
print(f"✓ Updated metadata in {DATA_PATH}")

print("\n" + "="*70)
print("NUMBERS TO USE IN PAPERS")
print("="*70)
print(f"Full panel: {full_n} models, {len(unique_labs)} labs, r = +{full_r:.3f}")
print(f"Core: {core_n} models, r = +{core_r:.3f}")
print(f"Regression: GPQA = {full_slope:.2f} · SWE + {full_intercept:.1f}")
print(f"Holdout (labs≥3): {holdout_mae_mean:.1f} ± {holdout_mae_std:.1f}%")
print(f"DeepSeek swing: {ds_swing:.1f} pp")
