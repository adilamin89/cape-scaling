# CAPE: Capability Coupling Analysis of Phase Emergence

**When Models Stop Lying — and When They Start Again: Capability Coupling and the Alignment Tax in AI Scaling Laws**

Adil Amin · Independent Researcher · [adilamin@uwm.edu](mailto:adilamin@uwm.edu)

[![arXiv](https://img.shields.io/badge/arXiv-2503.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2503.XXXXX)
[![Dashboard](https://img.shields.io/badge/Dashboard-Live-brightgreen)](https://adilamin89.github.io/cape-scaling)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What This Paper Shows

Standard scaling laws predict how *loss* falls with compute. They say nothing about how *capabilities interact*. This paper shows:

1. **Below ~3.5B parameters**: truthfulness (TruthfulQA) *anticorrelates* with every TQA-paired capability — reasoning (r = −0.989), commonsense (r = −0.941), and knowledge (r = −0.792) — an **alignment tax** built into pretraining.
2. **Above 3.5B**: all couplings reverse. Both improve together — an **alignment bonus**.
3. **Two models with identical loss can be in opposite alignment regimes** — the transition is invisible in loss (CV = 0.8%).
4. **At frontier scale** (20 models across 7 labs, 2024–2026): SWE-bench × GPQA coupling is cooperative (r = +0.85, p < 10⁻⁵), with detectable training-recipe excursions in Anthropic (Sonnet 4.6: h = −13.6), OpenAI (GPT-5.4: h = −1.6), and Google (Gemini 3 Pro: γ₁₂ = −0.83) — all of which recover at the next release. Three labs, same physics.
5. **Data curation is worth ~10× model size at 1B** parameters: curated training eliminates the alignment tax entirely (Phi demonstrates this empirically).
6. **Phase-separated PCA** confirms the restructuring is specific to truthfulness: d_eff peaks at 1.81 at the transition (maximum fluctuations at the critical point), then collapses to 1.20 in the bonus phase.
7. **CAPE's ODE beats polynomial baselines by ~2×** on held-out Llama-2 prediction (5.6% vs 10.2% MAE).

---

## Quick Start

```bash
git clone https://github.com/adilamin89/cape-scaling
cd cape-scaling

# One-command setup (creates venv, installs deps, runs smoke test)
bash setup.sh

# Or manual install
pip install -r requirements.txt

# Classify any model in one command
python scripts/quickstart.py --N 7 --hs 78 --tqa 43

# Frontier model (SWE × GPQA axis)
python scripts/quickstart.py --swe 80.0 --gpqa 93.2

# Reproduce all figures
python scripts/generate_all_figures.py

# Verify all 12 diagnostics
python scripts/verify_and_reproduce.py

# Full 20-model frontier coupling analysis
python scripts/cape_frontier_full.py
```

---

## Interactive Dashboard

**[adilamin89.github.io/cape-scaling](https://adilamin89.github.io/cape-scaling)**

7 tabs covering the full analysis:

| Tab | What it shows |
|-----|---------------|
| **Analyze** | Enter any model's scores → phase classification + recommendations |
| **Phase Map** | Base-model coupling map (44 models, 14 families) |
| **Frontier** | 20 frontier models, within-family trajectories (Anthropic/OpenAI/Google) |
| **Nc Cascade** | How phase transitions stack: Nc₁ → Nc₂ → Nc₃ |
| **Matrix** | Phase-separated correlation matrices, d_eff progression, CV results, RG flow |
| **Physics** | All equations, polynomial baseline, topology (W=0.5), kink soliton, dictionary |
| **Paper** | Key results summary, boosting chain, figure guide |

---

## Key Results (March 2026)

| Finding | Value | Method |
|---------|-------|--------|
| Nc (critical scale) | 3.5B (95% CI: 2.9B–13.4B) | Bootstrap on 14 web-trained models |
| Pre-transition correlation | r = −0.989, p < 10⁻⁵ | Within-Pythia |
| Post-transition correlation | r = +0.77, p = 0.026 | Cross-family at 7B |
| ODE fit error | 3.6% | PySINDy on Pythia |
| Hold-out prediction (CAPE) | **5.6%** MAE | Llama-2 held out |
| Hold-out prediction (poly-2) | 10.2% MAE | Best polynomial baseline |
| OLMo confirmation | γ₁₂ = 0.000 | Zero free parameters |
| Frontier cooperative coupling | r = +0.85, p < 10⁻⁵ | **20 models, 7 labs** |
| d_eff at transition | **1.81** (peak) | Phase-separated PCA |
| d_eff in bonus phase | 1.20 | Phase-separated PCA |
| TQA sign-flip CV | 4/4 pairs flip, 0/6 non-TQA | Leave-one-family-out |
| RG fixed point | γ* = 0.64 (stable) | Beta function analysis |
| Winding number | W = 0.5 (Z₂) | Berry phase calculation |
| Ginzburg number at Nc | Gi = 1.35 | Fluctuation analysis |
| Predicted Nc₃ | ~114–130B | Two-method convergence |

---

## Repository Structure

```
cape-scaling/
├── index.html                          ← Interactive dashboard (GitHub Pages, 7 tabs)
├── paper3A.tex                         ← Main paper (LaTeX source, v2 draft)
├── paper3A.pdf                         ← Compiled PDF
├── cape_supplementary.tex              ← Supplementary appendix (GL derivations)
├── .gitignore
├── requirements.txt
│
├── data/
│   ├── ai_free_energy_data.json        ← Pythia benchmark scores + correlation matrix
│   ├── cape_26models_9families.json    ← 26-model PCA dataset with h(D), eigenvectors
│   ├── frontier_models.json            ← 20 frontier models: SWE-bench + GPQA Diamond
│   ├── beta_final_6model.json          ← β order-parameter exponent data
│   └── bootstrap_Nc_results.json       ← Bootstrap CI results
│
├── figures/
│   ├── fig1_main.png ... fig10_nc3.png ← All paper figures
│
└── scripts/
    ├── quickstart.py                   ← START HERE: classify any model in 10 lines
    ├── verify_and_reproduce.py         ← Verify all 12 diagnostics
    ├── generate_all_figures.py         ← Reproduce all figures
    ├── cape_frontier_full.py           ← 20-model frontier coupling (r=+0.85)
    ├── swe_gpqa_coupling_v22.py        ← Frontier analysis with within-family γ₁₂
    ├── bootstrap_Nc.py                 ← Bootstrap CI on Nc → [2.9B, 13.4B]
    ├── beta_final_analysis.py          ← β order-parameter exponent
    ├── cape_architecture_probe.py      ← Architecture probe (OPT/BLOOM/OLMo)
    ├── cape_stiffness.py               ← Phase stiffness computation
    ├── cape_algebraic_nc.py            ← √(0.187·HS) algebraic classifier
    ├── cape_null_model.py              ← Null model comparison
    ├── cape_nc3_deep.py                ← Nc₃ three-field analysis
    ├── pysindy_per_phase.py            ← Per-phase ODE fitting
    ├── diagnostics.py                  ← Scaling diagnostics (α_eff, κ, D_L)
    ├── pythia_gradient_extraction.py   ← GPU: ‖∇L‖ extraction
    └── test_dashboard.py               ← Playwright automated dashboard tests
```

---

## Phase Classification Summary

| Phase | Scale | γ₁₂ | d_eff | What's happening |
|-------|-------|------|-------|------------------|
| **Tax** | <3.5B | <0 | ~1.53 | TQA anticorrelates with everything — alignment costs compute |
| **Transition** | ~3.5B | ≈0 | **1.81** | Maximum fluctuations — highest leverage for interventions |
| **Bonus** | 3.5–70B | >0 | ~1.20 | All capabilities cooperate — scaling helps alignment |
| **Frontier** | >70B | >0 | ~1.15 | Deep cooperative. New axes (IFEval) activating at Nc₃ |

---

## Citation

```bibtex
@article{amin2026cape,
  title={When Models Stop Lying --- and When They Start Again:
         Capability Coupling and the Alignment Tax in AI Scaling Laws},
  author={Amin, Adil},
  journal={arXiv preprint arXiv:2503.XXXXX},
  year={2026}
}
```

---

## Contact

Adil Amin · [adilamin@uwm.edu](mailto:adilamin@uwm.edu) · [adil89aminx@gmail.com](mailto:adil89aminx@gmail.com)
Independent Researcher · March 2026
