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

## Dashboard

```bash
open index.html      # macOS — no server, no npm, no build step needed
```

Or: **[adilamin89.github.io/cape-scaling](https://adilamin89.github.io/cape-scaling)**

**6 tabs:**

| Tab | What it does |
|---|---|
| **Overview** | Phase diagram, model browser, key stats. Enter any model size + TruthfulQA to see where it lands. |
| **Explorer** | Full computation engine. Enter parameters → get γ₁₂, regime, d_eff, h(D), susceptibility χ(N), eigenvectors, ODE slope, and scaling predictions. Regime-aware PCA computed separately for tax/bonus/frontier. Click any known model to auto-fill. |
| **Frontier** | SWE-bench vs GPQA Diamond across 16 frontier models. The emerging capability pair where HellaSwag saturates. |
| **h(D) Lab** | Data curation quality ranking. Measures how much training data shifts the phase boundary beyond what scale alone predicts. |
| **Physics** | Dual-language reference: every equation in both physics and ML terms. Boosting chain (L₀→L₄), three-regime theory, physics↔ML dictionary. |
| **Paper** | Abstract, citation, 13 diagnostics, usage guide. |

### Adding new benchmarks

The Explorer tab accepts up to 7 benchmarks (HellaSwag, TruthfulQA, MMLU, ARC-C, WinoGrande, GPQA, SWE-bench). The PCA engine automatically includes any benchmark with ≥3 non-null values and recomputes eigenvectors, participation ratio (d_eff), and the coupling matrix. If a new benchmark becomes standard beyond frontier, add it to the `benchCols` array in `index.html` and the engine handles the rest — no hardcoded benchmark assumptions.

### CLI tool (coming soon)

`cape-cli.js` — standalone Node.js tool that runs the same PCA engine, coupling computation, and regime classification from the command line. Feed it a JSON of model benchmarks and get back the full CAPE analysis.

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

Tests: 6 tabs, 17 key numbers, 16 frontier models, JS console, 4 canvas charts.

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

| Quantity | Value | ML interpretation |
|---|---|---|
| r(HellaSwag, TQA) | −0.989 | Strongest anticorrelation: scaling reasoning hurts truthfulness |
| Nc | ~3.5B | Critical model size where alignment regime flips |
| γ₁₂(N) | A·log₁₀N + B | Running coupling: A=0.629, B=−5.886 |
| χ_γ(N) = 1/\|γ₁₂\| | Diverges at Nc | Sensitivity of regime classification to scale choice |
| OLMo-7B γ₁₂ | 0.000 | Zero-parameter confirmation of Nc |
| α (loss exponent) | 0.238 ± 0.015 | Power-law loss scaling R²=0.9994 |
| β (order param) | 0.40 ± 0.08 | Critical exponent, vs mean-field prediction 1.24 |
| d_eff at Nc | 1.254 | Effective number of independent benchmark axes |
| Ginzburg number | 1.35 | Fluctuation-dominated crossover |
| κ_GL | 0.767 > 1/√2 | Type II confirmed |
| Nc₂ | ~9–11B | Second susceptibility peak |
| Hold-out MAE | 5.6% | Cross-family prediction accuracy |
| sin(θ*) = A | 0.626 ≈ 0.629 | Eigenvector-coupling identity |
| Data multiplier | ~10× | 1 unit data quality ≈ 10× model size at 1B |
| Valid to | ~130B | Where two-field theory predicts its own breakdown |
| Frontier r(SWE,GPQA) | +0.34 | Cooperative coupling at frontier scale |

---

## Physics↔ML Glossary

| Physics term | ML translation | CAPE quantity |
|---|---|---|
| Coupling constant γ₁₂ | How much one benchmark changes per unit gain in another | γ₁₂(N) = 0.629·log₁₀N − 5.886 |
| Phase transition | Regime flip: alignment tax → bonus | Sign change at Nc = 3.5B |
| Susceptibility χ | How sensitively regime responds to scale changes | χ_γ = 1/\|γ₁₂\|, diverges at Nc |
| External field h | Training data quality offset from web baseline | h(D) = TQA_obs − TQA_pred |
| Effective dimension d_eff | Number of independent benchmark axes needed | Participation ratio of eigenvalues |
| Order parameter | Primary quantity that changes sign at the transition | Pearson r(HS, TQA) |
| Soft mode (λ₂) | Benchmark variance along least-constrained direction | λ₂ → 0 means dimensional collapse |
| Eigenvector rotation | Which benchmarks load together and which compete | e₁ = [+HS, −TQA] in tax; [+HS, +TQA] in bonus |
| Free energy F | Loss landscape viewed as function of capability fields | F = a₁φ₁² + a₂φ₂² + γ₁₂φ₁φ₂ |
| Renormalization group | Boosting chain: each model's failure is next-order signal | L₀ → L₁ → L₂ → L₃ → L₄ |
| Hessian det(H) → 0 | Current two-benchmark theory runs out of dimensions | Predicts breakdown at ~130B |
| Specific heat peak | Maximum alignment sensitivity to scale | Susceptibility peak at 1B |

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

To add to the dashboard, open `index.html`, find the `M` array near the top:
```javascript
{n:"YourModel-7B", N:7e9, hs:71.4, tqa:41.3, f:"YourFamily"},
```
No npm, no build. Edit and refresh. The Explorer tab computes everything live.

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

The coupling mathematics underlying CAPE was originally developed for multi-band systems:
> Amin & Agterberg (2020). *Generalized spin-fluctuation feedback.* [Phys. Rev. Research **2**, 013055](https://link.aps.org/doi/10.1103/PhysRevResearch.2.013055)

Part of the CAPE program — applying effective field theory to AI scaling laws.

---

**Adil Amin** · adilamin@uwm.edu · adil89aminx@gmail.com
[LinkedIn](https://www.linkedin.com/in/adil-amin-ph-d-1217a91a3) · Issues/PRs welcome
