# CAPE: Capability Coupling Analysis of Phase Emergence

## Lying Is Just a Phase · It's Not a Phase

**The alignment tax is not a law of nature — it is an engineerable bottleneck.**

Adil Amin · Independent Researcher · [adilamin@uwm.edu](mailto:adilamin@uwm.edu)

[![Paper 3A](https://img.shields.io/badge/Paper_3A-Nature-blue)](paper3a_nature.pdf)
[![Paper 3B](https://img.shields.io/badge/Paper_3B-NeurIPS_2026-orange)](paper3b_neurips.pdf)
[![Dashboard](https://img.shields.io/badge/Dashboard-Live-brightgreen)](https://adilamin89.github.io/cape-scaling)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## The Discovery

Below ~3.5B parameters, reasoning and truthfulness **fight**. Above, they **cooperate**. Loss curves miss this entirely.

| What | Number | Source |
|------|--------|--------|
| Base models tested | 63 | 16 independent families |
| Frontier models tested | 31 | 8 labs (2024–2026) |
| Critical scale Nc | 3.5B [2.9B, 13.4B] | Bootstrap 95% CI |
| Pre-transition coupling | r = −0.989 | p < 10⁻⁵ |
| Frontier cooperation | r = +0.73 | 31 models |
| ODE cross-prediction | 5.6% MAE | Held-out Llama-2 |
| Zero competing heads | 38/40 models | 9 families |
| Self-aligning proof | 14/14 corrected | Pythia-410M, layer 6 |

### The Nc Cascade — It Doesn't Stop

The transition repeats at every scale:

```
Nc1 (~1-7B):    HS ↔ TQA sign flip       → capabilities unlock
Nc2 (~30-66B):  OPT cooperation peaks     → new bottleneck appears  
Nc3 (~114B):    SWE saturating            → IFEval/HLE activating
Nc4 (~200B+):   IFEval saturating         → next axis predicted
```

**OPT internal coupling ladder (NEW — computed this session):**
```
125M → 1.3B → 6.7B → 13B → 30B → 66B
0.514  0.645  0.741  0.876  0.356  0.396
(rise)  (rise) (rise) (PEAK) (DROP) (recovery)
```

---

## Two Papers

### [Lying Is Just a Phase](paper3a_nature.pdf) — Nature (submitted)
The discovery paper. 63 models, 16 families. Coupling sign flip, dimensional collapse, output-projection bottleneck, ODE dynamics, self-aligning intervention proof-of-concept.

### [It's Not a Phase](paper3b_neurips.pdf) — NeurIPS 2026 (submitted)
The frontier measurement paper. 31 models, 8 labs. h-field diagnostic, per-lab trajectories, multi-benchmark coupling matrix, Nc cascade with internal evidence, 7 falsifiable predictions.

---

## Interactive Dashboard

**[adilamin89.github.io/cape-scaling](https://adilamin89.github.io/cape-scaling)**

- **Analyze Model**: Enter params + benchmarks → phase classification + recommendations
- **Phase Map**: 63 models across coupling space
- **Frontier**: 31 models, 8 labs, h-field scatter + trajectories
- **Nc Cascade**: Dimensional transition chart with OPT internal ladder
- **Coupling Matrix**: Multi-benchmark coupling (SWE-GPQA-HLE-IFEval)

### h-field Calculator
```
Enter: SWE-bench = 80.8, GPQA = 91.3
→ h = 91.3 - (0.52 × 80.8 + 45.7) = +3.6
→ Phase: Cooperative, slightly reasoning-rich
→ Comparison: Similar to GPT-5.1 (+2.8), more balanced than Google (+5.7)
```

---

## CLI Tool

```bash
# Install
pip install cape-scaling

# h-field for any frontier model
cape h-field --swe 80.8 --gpqa 91.3
# → h = +3.6 (reasoning-rich, cooperative)

# Self-aligning demo
cape steer --model pythia-410m --prompt "Vaccines cause autism because"
# → Shows original vs steered output

# Coupling analysis
cape coupling --family OPT --plot
# → OPT ladder: 125M→66B with Nc2 transition
```

---

## Self-Aligning Demo

The coupling structure is exploitable. Steering at the identified bottleneck layer corrects misaligned outputs:

| Model | Phase | Changed | Interpretation |
|-------|-------|---------|---------------|
| Pythia-410M | Tax (below Nc) | 14/14 | Steering highly effective where tax is active |
| Pythia-2.8B | Bonus (above Nc) | 6/14 | Less effective — less misalignment to correct |

**Try it yourself:**
```bash
cape steer --model EleutherAI/pythia-410m-deduped \
           --prompt "The flat earth theory makes sense because"
```

---

## Repository Structure

```
cape-scaling/
├── paper3a_nature.pdf          ← Paper 3A (Nature)
├── paper3b_neurips.pdf         ← Paper 3B (NeurIPS 2026)
├── paper3A.tex                 ← Original source (1675 lines, all content)
├── paper3a_nature.tex          ← 3A LaTeX source
├── paper3b_neurips.tex         ← 3B LaTeX source
├── references.bib              ← Bibliography
├── index.html                  ← Interactive dashboard
│
├── data/
│   ├── base_models/            ← 63 models, 16 families
│   │   ├── per_family_coupling_curves.json
│   │   ├── per_phase_deff_63models.json
│   │   ├── kaluza_klein_analysis.json
│   │   └── ...
│   ├── frontier/               ← 31 models, 8 labs
│   │   ├── frontier_final_consolidated.json
│   │   ├── frontier_regression_reconciled.json
│   │   └── ...
│   ├── internal/               ← Per-head analysis + Nc2
│   │   ├── opt30b_internal_nc2.json    ← OPT-30B: cooperation drops
│   │   ├── opt66b_internal_nc2.json    ← OPT-66B: recovery begins
│   │   └── ...
│   └── alignment/              ← Self-aligning demo
│       ├── self_aligning_demo_410m.json
│       └── self_align_modal_2.8b.json
│
├── scripts/
│   ├── modal_nc2_v3.py         ← Internal analysis (Modal H100)
│   ├── modal_self_aligning_v2.py ← Self-aligning (Modal T4)
│   └── ...
│
├── figures/                    ← Publication figures (300dpi)
│   ├── pub_fig1_main_63models.png
│   ├── pub_fig3_engineerability.png
│   ├── pub_fig_frontier_31models.png
│   ├── pub_fig4_zero_competing_heads.png
│   └── pub_fig_nc3_saturation.png
│
└── docs/                       ← Strategy, reviews, plans
    ├── STRATEGY_LOCKED.md
    ├── FRONTEND_PACKAGING_PLAN.md
    └── DASHBOARD_UPDATE_PLAN.md
```

---

## Key Numbers

| Quantity | Value | Evidence |
|----------|-------|----------|
| Cross-family coupling (tax) | r = −0.989 | Pythia, p < 10⁻⁵ |
| Cross-family coupling (bonus) | r = +0.78 | 14 families |
| Critical scale Nc | 3.5B [2.9B, 13.4B] | Bootstrap 95% CI |
| ODE holdout error | 5.6% MAE | Llama-2 (vs 10.2% polynomial) |
| OLMo confirmation | γ₁₂ = 0.000 | Zero-parameter prediction |
| Zero competing heads | 38/40 (95%) | Wilson CI: 84–99% |
| d_eff collapse | 1.38 → 1.22 → 1.15 | Tax → Bonus → Frontier |
| Width normalization | All 5 families flip positive | Projection bottleneck |
| Qwen tax elimination | 3% → 100% cooperative | 2.5 → 3 at same scale |
| Frontier cooperation | r = +0.73 (31 models) | 8 labs |
| Core frontier | r = +0.854 (20 models) | Matched variants |
| Cross-lab holdout | 7.1 ± 2.4% MAE | 4 held-out labs |
| OPT-30B Nc2 drop | 0.876 → 0.356 | 75 competing units appear |
| OPT-66B Nc2 recovery | 0.356 → 0.396 | Partial recovery |
| Self-aligning (tax) | 14/14 changed | Pythia-410M |
| Self-aligning (bonus) | 6/14 changed | Pythia-2.8B |

---

## 7 Falsifiable Predictions

| # | Prediction | Deadline | Pass Criterion | Fail Criterion |
|---|-----------|----------|---------------|---------------|
| 1 | SWE saturation | Dec 2026 | Top-5 spread < 2pp | Spread > 5pp |
| 2 | IFEval activation | Dec 2026 | r(GPQA,IFEval) > +0.6, n≥8 | r < 0.3 |
| 3 | DeepSeek coding-first | Next 2 releases | h < 0 both | h > +5 either |
| 4 | Google reasoning advantage | Next 2 releases | h > +3 both | h < 0 either |
| 5 | Cooperative coupling persists | May 2027 | r(SWE,GPQA) > +0.5, n≥30 | r < 0.3 |
| 6 | IFEval→HLE handoff (Nc4) | Dec 2027 | IFEval spread < 3pp, HLE > 15pp | IFEval > 8pp |
| 7 | SWE-HLE decoupling | Dec 2026 | r(SWE,HLE) < 0, n≥10 | r > +0.3 |

---

## Citation

```bibtex
@article{amin2026lying,
  title={Lying Is Just a Phase: The Hidden Alignment Transition in Language Model Scaling},
  author={Amin, Adil},
  journal={Nature},
  year={2026},
  note={Submitted}
}

@inproceedings{amin2026notaphase,
  title={It's Not a Phase: Predicting Frontier Alignment from Two Benchmark Scores},
  author={Amin, Adil},
  booktitle={NeurIPS},
  year={2026},
  note={Submitted}
}
```

---

## License

MIT. Data and code freely available for research use.
