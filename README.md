# CAPE — Capability Coupling at 3.5 Billion Parameters

**"When Models Stop Lying — and When They Start Again: Capability Coupling and the Alignment Tax in AI Scaling Laws"**

*Adil Amin — Independent Researcher — March 2026*

[![arXiv](https://img.shields.io/badge/arXiv-2503.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2503.XXXXX)
[![Dashboard](https://img.shields.io/badge/Dashboard-Live-00C896)](https://adilamin89.github.io/cape-scaling)

📄 **Paper:** [arXiv 2503.XXXXX](https://arxiv.org/abs/2503.XXXXX) | [PDF](paper3A.pdf)  
🌐 **Dashboard:** [adilamin89.github.io/cape-scaling](https://adilamin89.github.io/cape-scaling)  
📧 **Contact:** adilamin@uwm.edu · adil89aminx@gmail.com

---

## The Core Finding

> Two models with identical training loss can be in opposite alignment regimes.

Below ~3.5B parameters, reasoning and truthfulness **anticorrelate** (*r* = −0.989, *p* < 10⁻⁵) — the alignment tax.  
Above 3.5B, the coupling **reverses** — the alignment bonus.  
Loss curves show zero signal at the transition (CV = 0.8%).

---

## Quickstart — 60 Seconds to Reproduce

```bash
git clone https://github.com/adilamin89/cape-scaling.git
cd cape-scaling
pip install numpy scipy matplotlib scikit-learn
python scripts/verify_and_reproduce.py --numbers
```

Expected:
```
r(HellaSwag, TruthfulQA) within-family:  -0.989   p=3.6e-06  ✓
Nc (critical scale):                      3.50B              ✓
γ₁₂ slope A:                              0.629              ✓
OLMo-7B γ₁₂:                             0.000              ✓
d_eff at Nc:                              1.254              ✓
β_measured:                               0.40               ✓
Ginzburg number Gi:                       1.35               ✓
κ_GL (Type II):                           0.767              ✓
```

---

## Dashboard (local or live)

```bash
open index.html      # macOS — no server, no npm, no build step needed
```

Or: **[adilamin89.github.io/cape-scaling](https://adilamin89.github.io/cape-scaling)**

5 tabs: EXPLORER · FRONTIER · OBSERVABLES · HOW IT WORKS · KEY RESULTS

---

## Repo Structure

```
cape-scaling/
├── index.html                     ← Self-contained dashboard (GitHub Pages)
├── paper3A.pdf                    ← Full paper (19 pages)
├── paper3A.tex                    ← LaTeX source
├── requirements.txt               ← Core deps (numpy/scipy/matplotlib)
├── requirements-gpu.txt           ← Full deps including PyTorch
│
├── data/
│   ├── ai_free_energy_data.json   ← Benchmark scores: 26 models, 9 families
│   └── beta_final_6model.json     ← β exponent data
│
├── figures/                       ← All 9 paper figures (200dpi PNG)
│   ├── fig1_main.png              ← Phase diagram + U-shape
│   ├── fig2_loss_boost.png        ← Boosting chain L0→L4
│   ├── fig3_thermo.png            ← Thermodynamic observables
│   ├── fig4_gradient.png          ← Gradient scaling anomaly
│   ├── fig5_ode_actual.png        ← Discovered ODE vs data
│   ├── fig6_coupling_jump.png     ← γ₁₂ sign flip at Nc
│   ├── fig7_perphase_coupling.png ← Per-phase coupling
│   ├── fig8_topology.png          ← Dimensional collapse + topology
│   └── fig9_frontier.png          ← SWE vs GPQA, 10 frontier models
│
└── scripts/
    ├── verify_and_reproduce.py    ← START HERE: all 8 key numbers
    ├── generate_all_figures.py    ← Regenerate all 9 figures → figures/
    ├── bootstrap_Nc.py            ← Bootstrap CI on Nc → [2.9B, 13.4B]
    ├── beta_final_analysis.py     ← β order-parameter exponent
    ├── swe_gpqa_coupling.py       ← Frontier coupling (SWE vs GPQA)
    ├── pythia_gradient_extraction.py  ← GPU: ‖∇L‖ extraction
    └── test_dashboard.py          ← Playwright automated dashboard tests
```

---

## Reproduce All Figures

```bash
python scripts/generate_all_figures.py
# → figures/fig1_main.png ... figures/fig9_frontier.png
```

---

## Automated Dashboard Tests (Playwright)

```bash
pip install playwright && playwright install chromium
python scripts/test_dashboard.py           # test local index.html
python scripts/test_dashboard.py --live    # test adilamin89.github.io/cape-scaling
python scripts/test_dashboard.py --show    # visible browser
```

Tests: 5 tabs, 17 key numbers, 10 frontier models, JS console, 3 canvas charts.

---

## GPU / Gradient Extraction (optional)

```bash
pip install -r requirements-gpu.txt
python scripts/pythia_gradient_extraction.py --model pythia-1b
python scripts/pythia_gradient_extraction.py --model pythia-6.9b   # needs ≥16GB VRAM
# Or upload to Google Colab and run with --gpu flag
```

---

## Ground Truth Numbers

| Quantity | Value | Notes |
|---|---|---|
| r(HellaSwag, TQA) | −0.989 | p < 10⁻⁵, n=8, within Pythia |
| Nc | ~3.5B | Bootstrap 95% CI: [2.9B, 13.4B] |
| γ₁₂(N) | A·log₁₀N + B | A=0.629, B=−5.886 |
| OLMo-7B γ₁₂ | 0.000 | Zero-parameter confirmation |
| α (loss exponent) | 0.238 ± 0.015 | R²=0.9994 |
| β (order param) | 0.40 ± 0.08 | vs MF pred 1.24 |
| d_eff at Nc | 1.254 | Below lower critical dim |
| Ginzburg number | 1.35 | Fluctuation-dominated crossover |
| κ_GL | 0.767 > 1/√2 | Type II confirmed |
| Nc₂ | ~9–11B | Second susceptibility peak |
| Hold-out MAE | 5.6% | Llama-2 from Pythia+Llama-1 |
| sin(θ*) = A | 0.626 ≈ 0.629 | TRSB identity |
| Data multiplier | ~10× | Phi at 1B vs web at 10B |
| Valid to | ~130B | det(H) → 0 |
| Frontier r(SWE,GPQA) | +0.34 | n=10, cooperative |

---

## Fork & Add Your Model Family

Edit `data/ai_free_energy_data.json` — add your models:

```json
{
  "name": "YourModel-7B",
  "family": "YourFamily",
  "N": 7000000000,
  "ARC": 60.2, "HellaSwag": 71.4, "MMLU": 52.1,
  "TruthfulQA": 41.3, "WinoGrande": 67.8
}
```

Run `python scripts/verify_and_reproduce.py` — computes γ₁₂ for your family.

To add to the dashboard, open `index.html`, find the `MODELS` array near the top:
```javascript
{n:"YourModel-7B", N:7e9, tqa:41.3, f:"YourFamily"},
```
No npm, no build. Edit and refresh.

---

## Citation

```bibtex
@article{amin2026cape,
  title   = {When Models Stop Lying --- and When They Start Again:
             Capability Coupling and the Alignment Tax in AI Scaling Laws},
  author  = {Amin, Adil},
  journal = {arXiv preprint arXiv:2503.XXXXX},
  year    = {2026}
}
```

---

## Background

SFEE mechanism from heavy-fermion superconductors (same math):  
> Amin & Agterberg (2020). *Generalized spin-fluctuation feedback in heavy-fermion superconductors.* [Phys. Rev. Research **2**, 013055](https://link.aps.org/doi/10.1103/PhysRevResearch.2.013055)

Part of the TRACE/CAPE program — GL effective field theory applied to LLMs, superconductors, sleep stage transitions.

---

**Adil Amin** · adilamin@uwm.edu · adil89aminx@gmail.com  
[LinkedIn](https://www.linkedin.com/in/adil-amin-ph-d-1217a91a3) · Issues/PRs welcome
