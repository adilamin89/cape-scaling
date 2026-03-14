# CAPE: Capability Coupling Analysis of Phase Emergence

**When Models Stop Lying — and When They Start Again: Capability Coupling and the Alignment Tax in AI Scaling Laws**

Adil Amin · Independent Researcher · [adilamin@uwm.edu](mailto:adilamin@uwm.edu)

[![arXiv](https://img.shields.io/badge/arXiv-2503.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2503.XXXXX)
[![Dashboard](https://img.shields.io/badge/Dashboard-Live-brightgreen)](https://adilamin89.github.io/cape-scaling)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What This Paper Shows

Standard scaling laws predict how *loss* falls with compute. They say nothing about how *capabilities interact*. This paper shows:

1. **Below ~3.5B parameters**: reasoning and truthfulness *anticorrelate* (r = −0.989, p < 10⁻⁵). Scaling one *hurts* the other — an **alignment tax** built into pretraining.
2. **Above 3.5B**: the coupling reverses. Both improve together — an **alignment bonus**.  
3. **Two models with identical loss can be in opposite alignment regimes** — the transition is invisible in loss (CV = 0.8%).
4. **At frontier scale** (19 models across 6 labs, 2024–2026): SWE-bench × GPQA coupling is cooperative (r = +0.85, p < 10⁻⁵), with detectable training-recipe excursions (Sonnet 4.6: γ₁₂ = −3.88) that recover at the next tier (Opus 4.6: γ₁₂ = +14.3).
5. **Data curation is worth ~10× model size at 1B** parameters: curated training eliminates the alignment tax entirely (Phi demonstrates this empirically).

---

## Interactive Dashboard

**[adilamin89.github.io/cape-scaling](https://adilamin89.github.io/cape-scaling)**

Enter any model's parameter count and benchmark scores → get:
- Alignment phase classification (Tax / Crossover / Bonus / Phase 4)
- Coupling γ₁₂, effective dimensionality d_eff, data-quality offset h(𝒟)
- Phase-specific recommendations
- Frontier phase tracking (SWE-bench × GPQA axis)
- Custom benchmark pair input for Nc₂/Nc₃ detection

Works for any model from 70M to frontier scale. Automatically extends to new benchmark pairs as capability axes saturate.

---

## Repository Structure

```
cape-scaling/
├── index.html                          ← Interactive dashboard (GitHub Pages)
├── paper3A.tex                         ← Main paper (LaTeX source)
├── cape_supplementary.tex              ← Supplementary appendix (GL derivations)
│
├── data/
│   ├── ai_free_energy_data.json        ← Pythia benchmark scores + correlation matrix
│   ├── cape_26models_9families.json    ← 26-model PCA dataset with h(D), eigenvectors
│   ├── frontier_models.json            ← 19 frontier models: SWE-bench + GPQA Diamond
│   └── beta_final_6model.json          ← β order-parameter exponent data
│
├── figures/
│   ├── fig1_main.png                   ← Phase transition + U-shape
│   ├── fig2_loss_boost.png             ← Boosting chain L₀→L₄
│   ├── fig3_thermo.png                 ← Thermodynamic observables
│   ├── fig4_gradient.png               ← Gradient scaling anomaly
│   ├── fig5_ode_actual.png             ← Discovered ODE vs data
│   ├── fig6_coupling_jump.png          ← γ₁₂ sign flip at Nc
│   ├── fig7_perphase_coupling.png      ← Per-phase coupling matrix
│   ├── fig8_topology.png               ← Dimensional collapse + topology
│   └── fig9_frontier.png               ← Frontier SWE×GPQA (n=19)
│
└── scripts/
    ├── verify_and_reproduce.py         ← START HERE: all key diagnostics
    ├── generate_all_figures.py         ← Reproduce all 9 figures
    ├── bootstrap_Nc.py                 ← Bootstrap CI on Nc → [2.9B, 13.4B]
    ├── beta_final_analysis.py          ← β order-parameter exponent
    ├── swe_gpqa_coupling.py            ← Frontier coupling (SWE vs GPQA)
    ├── cape_frontier_full.py           ← Full frontier analysis (19 models)
    ├── cape_architecture_probe.py      ← Architecture probe (OPT/BLOOM/OLMo)
    ├── cape_stiffness.py               ← Phase stiffness computation
    ├── pythia_gradient_extraction.py   ← GPU: ‖∇L‖ extraction
    ├── pysindy_per_phase.py            ← Per-phase ODE fitting (s± vs s++)
    ├── diagnostics.py                  ← Scaling diagnostics (α_eff, κ, D_L)
    └── test_dashboard.py               ← Playwright automated dashboard tests
```


---

## Key Results

| Finding | Value | Method |
|---------|-------|--------|
| Nc (critical scale) | 3.5B (95% CI: 2.9B–13.4B) | Bootstrap on 14 web-trained models |
| Pre-transition correlation | r = −0.989, p < 10⁻⁵ | Within-Pythia |
| Post-transition correlation | r = +0.77, p = 0.026 | Cross-family at 7B |
| ODE fit error | 3.6% | PySINDy on Pythia |
| Hold-out prediction error | 5.6% | Llama-2 held out |
| OLMo confirmation | γ₁₂ = 0.000 | Zero free parameters |
| Frontier cooperative coupling | r = +0.85, p < 10⁻⁵ | 19 models, 6 labs |
| Anthropic excursion (Sonnet 4.6) | γ₁₂ = −3.88, h = −13 | Within-family |
| Recovery (Opus 4.6) | γ₁₂ = +14.3, h = +2.8 | Within-family |
| Ginzburg number at Nc | Gi = 1.35 | Fluctuation analysis |
| Predicted Nc₃ | ~114–130B | Two-method convergence |

---

## Phase Classification Summary

| Phase | Scale | γ₁₂ | d_eff | Active benchmarks |
|-------|-------|------|-------|-------------------|
| Tax | <3.5B | <0 | ~1.05 | HellaSwag, TruthfulQA (competing) |
| Crossover | ~3.5B | ≈0 | ~1.1 | HellaSwag, TruthfulQA (decoupled) |
| Bonus | 3.5–70B | >0 | 1.1–1.6 | HS, TQA, MMLU (cooperative) |
| Phase 4 | >70B | >0 | 1.75–2.0 | SWE-bench, GPQA, IFEval |

---

## Reproducibility

```bash
git clone https://github.com/adilamin89/cape-scaling
cd cape-scaling
pip install numpy scipy pysindy matplotlib

# Reproduce all figures
python scripts/generate_all_figures.py

# Verify all 12 diagnostics
python scripts/verify_and_reproduce.py

# Frontier coupling analysis
python scripts/swe_gpqa_coupling.py
```

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
