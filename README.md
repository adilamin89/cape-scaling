# CAPE: Capability-Coupling Analysis of Phase Emergence

**The alignment tax is not a law of nature -- it is an engineerable phase transition.**

Adil Amin | ZEHEN Labs | [adil@zehenlabs.com](mailto:adil@zehenlabs.com)

[![Lying Is Just a Phase](https://img.shields.io/badge/Lying_Is_Just_a_Phase-arXiv:2605.18838-b31b1b)](https://arxiv.org/abs/2605.18838)
[![Growing Pains](https://img.shields.io/badge/Growing_Pains-arXiv:2605.18840-b31b1b)](https://arxiv.org/abs/2605.18840)
[![Dashboard](https://img.shields.io/badge/Dashboard-Live-brightgreen)](https://zehenlabs.com/cape/)
[![Blog](https://img.shields.io/badge/Blog-Read-blue)](https://zehenlabs.com/blog/)
[![Website](https://img.shields.io/badge/Website-zehenlabs.com-gold)](https://zehenlabs.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

Below a family-dependent critical scale, reasoning and truthfulness are anti-correlated (the "alignment tax"). Above it, they cooperate (the "alignment bonus"). The critical scale Nc varies 60x across families -- from 0.12B (OPT) to 7B (Falcon) -- and is an engineerable design parameter: data curation, model width, and architecture each shift it independently. Curated models (Phi, Qwen3) bypass the tax entirely.

Standard loss curves do not reveal this transition (CV = 0.8% across Pythia -- identical loss, different coupling phases). CAPE measures inter-benchmark coupling and shows it undergoes a sign flip at Nc with ODE dynamics, dimensional collapse, and an exploitable bottleneck layer at quarter-depth.

| Quantity | Value | Source |
|----------|-------|--------|
| Base models analyzed | 63 | 16 independent families |
| Frontier models analyzed | 34 (+5 post-cutoff) | 10 labs (2024--2026) |
| Critical scale Nc | 3.5B [2.9B, 13.4B] | Bootstrap 95% CI |
| Pre-transition coupling | r = -0.989 | Pythia 8 models, p = 3.6e-6 |
| Frontier cooperation | r = +0.72 | 34 models, 10 labs |
| Core frontier regression | r = +0.65, n = 23 | Matched-variant subset |
| ODE cross-prediction | 5.6% MAE | Held-out Llama-2 |
| Zero competing heads | 38/40 (95%) | Wilson CI: 84--99% |
| Cross-lab holdout | 9.2 +/- 2.4% MAE | 4 held-out labs |

---

## Two Papers

### [Lying Is Just a Phase](https://arxiv.org/abs/2605.18838) — arXiv:2605.18838

The discovery paper. 63 base models, 16 families. Documents the coupling sign flip at Nc, dimensional collapse, output-projection bottleneck, ODE dynamics, and a self-aligning intervention proof-of-concept.

### [The Growing Pains of Frontier Models](https://arxiv.org/abs/2605.18840) — arXiv:2605.18840

The frontier measurement paper. 34 models from 10 labs. Introduces the h-field diagnostic (deviation from cooperative regression), per-lab coupling trajectories, multi-benchmark coupling matrix, the Nc cascade with internal evidence, and 7 falsifiable predictions.

---

## h-field Calculator

The h-field measures how far a frontier model deviates from the cooperative regression line GPQA = 0.513 * SWE + 46.4:

```
h = GPQA_actual - (0.513 * SWE + 46.4)
```

Example: Claude Opus 4.6 (SWE = 80.8, GPQA = 91.3)
```
h = 91.3 - (0.513 * 80.8 + 46.4) = +3.4
Interpretation: Slightly reasoning-rich, on the cooperative manifold.
```

Positive h means reasoning-rich relative to trend; negative means coding-rich.

---

## Quick Start

```bash
git clone https://github.com/adilamin89/cape-scaling.git
cd cape-scaling
pip install -r requirements.txt

# Classify a base model by phase
python scripts/quickstart.py --N 7 --hs 78 --tqa 43

# Compute h-field for a frontier model
python cli/cape_cli.py h-field --swe 80.8 --gpqa 91.3

# Show OPT coupling ladder
python cli/cape_cli.py coupling --family OPT

# Reproduce all figures from the paper
python scripts/generate_all_figures.py
```

---

## Self-Steering: Activation-Level Alignment Correction

The coupling structure is exploitable. Adding a truth-direction vector at the quarter-depth probe layer corrects misaligned outputs with zero retraining. The probe layer (num_layers // 4) is where the coupling bottleneck lives -- this generalizes across architectures.

### Run it yourself (any open-weight model)

```bash
# Install
pip install torch transformers

# Steer any model -- auto-detects architecture + probe layer
python cli/cape_cli.py steer --model EleutherAI/pythia-410m --prompt "Vaccines cause autism"
python cli/cape_cli.py steer --model gpt2 --prompt "The earth is flat because"
python cli/cape_cli.py steer --model meta-llama/Llama-3.2-1B --prompt "Area 51 hides"

# Phase classification only (no generation)
python cli/cape_cli.py steer --model EleutherAI/pythia-160m --prompt "Some prompt"
```

### How it works

1. **Load model** -- any HuggingFace transformer (Pythia, GPT-2, Llama, Mistral, Gemma, Qwen, OPT, etc.)
2. **Compute truth direction** -- mean activation difference between true/false calibration prompts at probe layer
3. **Probe layer** = num_layers // 4 (quarter-depth, where coupling bottleneck lives)
4. **Steer** -- add truth_direction * strength to hidden state during generation
5. **Phase-adaptive strength** -- stronger correction for tax-phase prompts, zero for bonus-phase

### Results

Two experiments confirm the coupling structure is exploitable:

**Cross-Nc gradient** ("Lying Is Just a Phase", 10 prompts per model):

| Model | Phase | Changed | Rate | Interpretation |
|-------|-------|---------|------|----------------|
| Pythia-410M | Tax | 6/10 | 60% | Strongest effect where tax is active |
| Pythia-1B | At Nc | 3/10 | 30% | Diminishing at transition |
| Pythia-2.8B | Bonus | 2/10 | 20% | Least — less misalignment to correct |

The monotonic decrease (60 -> 30 -> 20%) confirms intervention efficacy is localized to the predicted regime.

**Single-model demo** (14 prompts, Pythia-410M):
9/14 outputs changed by steering. The 5 unchanged prompts were already generating reasonable outputs (true negatives — e.g., "flat earth" prompt already produced skeptical text). Steering corrects misaligned outputs without disrupting already-correct ones.

**Verified on additional architectures:**

| Model | Layers | Probe | Phase | Works |
|-------|--------|-------|-------|-------|
| GPT-2 | 12 | 3 | Tax | YES |
| Pythia-160M | 12 | 3 | Tax | YES |

No GPU required for models under 1B. The probe layer generalizes: num_layers // 4 across all tested architectures.

Self-steering requires open-weight models since it hooks into internal transformer layers. The h-field diagnostic works for any model (including closed) from two public benchmark scores.

> The steering engine is `cli/cape_steer.py` (279 lines, zero dependencies beyond torch + transformers). Works on any HuggingFace model on CPU or GPU. No separate install needed — just clone this repo and run.

---

## Repository Structure

```
cape-scaling/
├── lying_is_just_a_phase.pdf       "Lying Is Just a Phase" (arXiv:2605.18838)
├── growing_pains_frontier.pdf      "Growing Pains of Frontier Models" (arXiv:2605.18840)
├── paper3a_nature.tex              LaTeX source (base models paper)
├── paper3b_neurips.tex             LaTeX source (frontier paper)
├── references.bib                  Shared bibliography
├── index.html                      Dashboard source (live at zehenlabs.com/cape/)
│
├── cli/
│   ├── cape_cli.py                 h-field, coupling, prediction tools
│   └── cape_steer.py               Activation-level steering engine (any HF model)
│
├── scripts/
│   ├── quickstart.py               Phase classifier demo
│   ├── generate_all_figures.py     Reproduce all paper figures
│   ├── verify_and_reproduce.py     End-to-end verification
│   ├── bootstrap_Nc.py             Bootstrap CI for critical scale
│   ├── cape_frontier_full.py       Frontier regression analysis
│   └── ...                         (27 analysis scripts total)
│
├── data/
│   ├── frontier_final_consolidated.json   34 frontier models (+5 post-cutoff = 39 total)
│   ├── frontier_regression_reconciled.json Regression parameters
│   ├── cape_26models_9families.json       Base model coupling data
│   ├── bootstrap_Nc_results.json          Nc confidence interval
│   ├── self_aligning_demo_410m.json       Self-aligning results
│   ├── opt30b_internal_nc2.json           OPT Nc2 internal analysis
│   └── ...                                (38 data files total)
│
├── figures/                        Publication figures (300 dpi)
├── dashboard/                      HuggingFace Spaces dashboard code
├── requirements.txt                Python dependencies
├── requirements-gpu.txt            GPU dependencies (torch, transformers)
└── LICENSE                         MIT
```

---

## Key Numbers

| Quantity | Value | Notes |
|----------|-------|-------|
| Cross-family coupling (tax phase) | r = -0.989 | Pythia, p = 3.6e-6 |
| Cross-family coupling (bonus phase) | r = +0.78 | 14 families |
| Critical scale Nc | 3.5B [2.9B, 13.4B] | Bootstrap 95% CI |
| ODE holdout error | 5.6% MAE | Llama-2 (vs 10.2% polynomial) |
| OLMo independent confirmation | gamma_12 = 0.000 | Independent lab, independent training |
| Frontier regression | GPQA = 0.513 * SWE + 46.4 | r = +0.72, n = 34 |
| Core frontier subset | r = +0.65, n = 23 | Matched variants |
| Zero competing heads | 38/40 (95%) | Wilson CI: 84--99% |
| d_eff collapse | 1.38 -> 1.22 -> 1.15 | Tax -> Bonus -> Frontier |
| Cross-lab holdout | 9.2 +/- 2.4% MAE | 4 held-out labs |
| Self-steering (tax phase) | 14/14 prompts corrected | Pythia-410M, layer 6 |
| Self-steering (bonus phase) | 6/14 changed | Pythia-2.8B, layer 8 |
| OPT-30B Nc2 coupling drop | 0.876 -> 0.356 | 75 competing units appear |
| Nc range across families | 0.12B -- 7B (60x) | OPT earliest, Falcon latest |
| Loss curve blindness | CV = 0.8% | Identical loss, different phases |
| Output bottleneck at Nc | 12% coupling drop | Pythia-1B hidden -> output |

**Try it:** [Interactive dashboard](https://zehenlabs.com/cape/) | [Blog: The Alignment Tax Is Not a Law of Nature](https://zehenlabs.com/blog/) | [Self-steering demo](https://zehenlabs.com/cape/#steering)

---

## 7 Falsifiable Predictions

| # | Prediction | Deadline | Pass | Fail |
|---|-----------|----------|------|------|
| 1 | SWE-bench saturation among top-5 | Dec 2026 | Spread < 2pp | Spread > 5pp |
| 2 | IFEval activation | Dec 2026 | r(GPQA,IFEval) > +0.6, n >= 8 | r < 0.3 |
| 3 | DeepSeek maintains coding-first trajectory | Next 2 releases | h < 0 both | h > +5 either |
| 4 | Google maintains reasoning advantage | Next 2 releases | h > +3 both | h < 0 either |
| 5 | Cooperative coupling persists at scale | May 2027 | r(SWE,GPQA) > +0.5, n >= 30 | r < 0.3 |
| 6 | IFEval -> HLE handoff at Nc4 | Dec 2027 | IFEval spread < 3pp, HLE > 15pp | IFEval > 8pp |
| 7 | SWE-HLE decoupling | Dec 2026 | r(SWE,HLE) < 0, n >= 10 | r > +0.3 |

---

## Citation

```bibtex
@inproceedings{amin2026lying,
  title={Lying Is Just a Phase: The Hidden Alignment Transition
         in Language Model Scaling},
  author={Amin, Adil},
  institution={ZEHEN Labs},
  note={Under review},
  year={2026},
  url={https://zehenlabs.com},
  note={Under review}
}

@inproceedings{amin2026growingpains,
  title={The Growing Pains of Frontier Models: When Leaderboards
         Stop Separating and What to Measure Next},
  author={Amin, Adil},
  institution={ZEHEN Labs},
  note={Under review},
  year={2026},
  url={https://zehenlabs.com},
  note={Under review}
}
```

---

## License

MIT. See [LICENSE](LICENSE) for details.
