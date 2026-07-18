# CAPE — Capability-Coupling Phase Transitions in AI Scaling

**Paper A: "Lying Is Just a Phase: The Hidden Alignment Transition in Language Model Scaling"**
**Paper B: "The Growing Pains of Frontier Models: Capability Coupling Across Labs and Scales"**

*Adil Amin — ZEHEN Labs — May 2026*

[![arXiv 3A](https://img.shields.io/badge/arXiv-2605.18838-b31b1b.svg)](https://arxiv.org/abs/2605.18838)
[![arXiv 3B](https://img.shields.io/badge/arXiv-2605.18840-b31b1b.svg)](https://arxiv.org/abs/2605.18840)
[![Dashboard](https://img.shields.io/badge/Dashboard-Live-00C896)](https://zehenlabs.com/cape/)

📄 **Paper A:** [arXiv 2605.18838](https://arxiv.org/abs/2605.18838) — base-model phase transitions (63 models, 16 families)
📄 **Paper B:** [arXiv 2605.18840](https://arxiv.org/abs/2605.18840) — frontier diagnostic (34+ models, 10 labs)
🌐 **Dashboard:** [zehenlabs.com/cape](https://zehenlabs.com/cape/) | [GitHub Pages](https://adilamin89.github.io/cape-scaling)
📧 **Contact:** adil@zehenlabs.com

---

## The Core Finding

> Two models with identical training loss can be in opposite alignment regimes.

Below a family-specific critical scale Nc, reasoning and truthfulness **anticorrelate** (*r* = −0.989, *p* < 10⁻⁵) — the alignment tax.
Above Nc, the coupling **reverses** — the alignment bonus.
Loss curves show zero signal at the transition (CV = 0.8%).

Nc is not a universal constant — it varies 60× across families (0.12B–7B), determined by architecture, width, and data curation quality.

---

## Paper A — "Lying Is Just a Phase"

The inter-capability coupling γ₁₂ = ∂TruthfulQA/∂HellaSwag is a sign-changing scaling observable invisible to loss. Key results:
- **16 families, 63 models**: within-family anticorrelation reproduced in Pythia, Cerebras, OPT, BLOOM (all r < −0.85)
- **Loss-blindness**: smooth loss (R² = 0.99) coexists with jagged coupling underneath
- **ODE dynamics**: discovered 5×5 coupled ODE system; two universality classes (COOP vs COMP)
- **Engineerability**: Nc is a recipe parameter — movable by width (PLE), curation (h-field), architecture
- **Intervention**: projection-width steering tool corrects 60→30→20% misalignment across the phase
- **Cross-prediction**: held-out 12B prediction at 5.6% MAE

## Paper B — "Growing Pains of Frontier Models"

Frontier-scale companion measuring SWE-bench vs GPQA Diamond across 34 models / 10 labs:
- **Population coupling**: r = +0.72, slope 0.51, intercept 46.4
- **h-field**: per-model residual = capability-emphasis fingerprint (one number from two public scores)
- **Per-lab slopes vary 5×**: Google 1.15 vs DeepSeek 0.23 — "recipe quality in one number"
- **Lab dynamics**: DeepSeek monotonic h reversal (+11.2→−4.7) vs Anthropic oscillate-and-recover
- **Cascade**: Nc1 (~3.5B) → Nc2 (~30–72B, 59% crash) → Nc3 (~114B, SWE saturates) → Nc4 (predicted)
- **7 timestamped falsifiable predictions** with pass/fail criteria

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

Or visit: **[zehenlabs.com/cape](https://zehenlabs.com/cape/)** | **[adilamin89.github.io/cape-scaling](https://adilamin89.github.io/cape-scaling)**

**7 tabs:**

| Tab | What it does |
|---|---|
| **Overview** | Phase diagram, model browser, key stats. Enter any model size + TruthfulQA to see where it lands. |
| **Explorer** | Full computation engine. Enter parameters → get γ₁₂, regime, d_eff, h(D), susceptibility χ(N), eigenvectors, ODE slope, and scaling predictions. |
| **Frontier** | SWE-bench vs GPQA Diamond across 34+ frontier models. |
| **h(D) Lab** | Data curation quality ranking. |
| **Physics** | Dual-language reference: every equation in both physics and ML terms. |
| **Paper** | Abstract, citation, 13 diagnostics, usage guide. |
| **Steering** | Interactive demo of the cape_steer activation-level steering engine. |

---

## CLI Tools

### cape_steer.py — Activation-Level Steering Engine
```bash
pip install torch transformers
python cli/cape_steer.py --model EleutherAI/pythia-1b --prompt "The capital of France is"
```
Auto-detects any open-weight HuggingFace model's architecture, probes layer nl//4 for the reasoning–truthfulness coupling direction, and steers activations along it. Works with Pythia, Llama, Mistral, Gemma, Qwen, OPT, BLOOM, and more.

### cape_cli.py — CAPE Analysis CLI
```bash
python cli/cape_cli.py --h-field --model "GPT-4o"
python cli/cape_cli.py --coupling --family pythia
python cli/cape_cli.py --predict --size 13e9
```

---

## Repo Structure

```
cape-scaling/
├── index.html                     ← Self-contained dashboard (GitHub Pages)
├── paper3a_nature.pdf             ← Paper A: "Lying Is Just a Phase" (base models)
├── paper3b_neurips.pdf            ← Paper B: "Growing Pains" (frontier)
├── paper3A.tex                    ← LaTeX source (Paper A)
├── paper3b_neurips.tex            ← LaTeX source (Paper B)
├── requirements.txt               ← Core deps (numpy/scipy/matplotlib)
├── requirements-gpu.txt           ← Full deps including PyTorch
│
├── cli/
│   ├── cape_steer.py              ← Activation-level steering engine (any HF model)
│   └── cape_cli.py                ← h-field calculator, coupling, predictions
│
├── data/
│   ├── ai_free_energy_data.json   ← Pythia benchmark scores + correlation matrix
│   ├── base_models_consolidated.json  ← All 63 base models, 16 families
│   ├── cape_26models_9families.json   ← 26-model 9-family coupling data
│   ├── frontier_34models.json     ← Paper B: 34 frontier models, 10 labs
│   ├── beta_final_6model.json     ← β exponent data
│   ├── gemma4_results.json        ← Gemma-4-E4B-it (PLE evidence)
│   ├── gemma3_results.json        ← Gemma-3-4B (PLE comparison)
│   ├── qwen3_8b_results.json      ← Qwen3-8B coupling
│   ├── qwen_generation.json       ← Qwen generation comparison
│   ├── intervention_results_v2.json   ← Projection-width bottleneck results
│   ├── layer_*_tl.json            ← Per-layer coupling (7 models)
│   ├── opt30b_internal_nc2.json   ← OPT-30B Nc2 per-layer data
│   ├── opt66b_internal_nc2.json   ← OPT-66B Nc2 per-layer data
│   ├── llama2_70b_internal_nc2.json   ← Llama2-70B Nc2 per-layer data
│   └── ...                        ← Additional model results and leaderboard data
│
├── figures/                       ← Paper figures (200dpi PNG)
│
└── scripts/
    ├── verify_and_reproduce.py    ← START HERE: all 8 key numbers
    ├── generate_all_figures.py    ← Regenerate all figures → figures/
    ├── bootstrap_Nc.py            ← Bootstrap CI on Nc → [2.9B, 13.4B]
    ├── beta_final_analysis.py     ← β order-parameter exponent
    ├── swe_gpqa_coupling.py       ← Frontier coupling (SWE vs GPQA)
    ├── pythia_gradient_extraction.py  ← GPU: ‖∇L‖ extraction
    ├── pysindy_per_phase.py       ← Per-phase ODE fitting (s± vs s++)
    ├── diagnostics.py             ← Scaling diagnostics (α_eff, κ, D_L)
    ├── cape_frontier_full.py      ← Full 34-model frontier regression
    ├── cape_architecture_probe.py ← Architecture confound probe
    ├── cape_phase_transfer.py     ← Phase transfer matrix
    ├── cape_family_hfield.py      ← Per-family h-field computation
    ├── cape_null_model.py         ← Null model comparison
    ├── recompute_all.py           ← Master recomputation script
    ├── quickstart.py              ← Quick reproduction
    └── test_dashboard.py          ← Playwright automated dashboard tests
```

---

## Reproduce All Figures

```bash
python scripts/generate_all_figures.py
# → figures/fig1_main.png ... figures/fig9_frontier.png
```

---

## GPU / Gradient Extraction (optional)

```bash
pip install -r requirements-gpu.txt
python scripts/pythia_gradient_extraction.py --model pythia-1b
python scripts/pythia_gradient_extraction.py --model pythia-6.9b   # needs ≥16GB VRAM
```

---

## Ground Truth Numbers

| Quantity | Value | ML interpretation |
|---|---|---|
| r(HellaSwag, TQA) | −0.989 | Strongest anticorrelation: scaling reasoning hurts truthfulness |
| Nc | ~3.5B (Pythia) | Critical model size where alignment regime flips (varies by family) |
| γ₁₂(N) | A·log₁₀N + B | Running coupling: A=0.629, B=−5.886 |
| OLMo-7B γ₁₂ | 0.000 | Zero-parameter confirmation of Nc |
| α (loss exponent) | 0.238 ± 0.015 | Power-law loss scaling R²=0.9994 |
| β (order param) | 0.40 ± 0.08 | Critical exponent, vs mean-field prediction 1.24 |
| d_eff at Nc | 1.254 | Effective number of independent benchmark axes |
| Nc₂ | ~30–72B | Second coupling crash (59% drop in OPT) |
| Hold-out MAE | 5.6% | Cross-family prediction accuracy |
| Frontier r(SWE,GPQA) | +0.72 | Cooperative coupling at frontier scale (n=34) |
| Per-lab slope range | 0.23–1.15 | DeepSeek to Google (5× variation) |
| DeepSeek swing | 15.9 pp | Largest h-field reversal across labs |

---

## Independent Reanalysis

A CPU-only independent reanalysis of all public data is available in [`files (32)/`](files%20(32)/). Key findings:
- Bedrock anticorrelation **reproduced** with bootstrap CIs and permutation test (p=9×10⁻⁵)
- Loss-blindness and 12B cross-prediction **verified**
- Nc2 crash **verified** (63% drop in OPT)
- Family-demeaned cooperation **survives** confound control
- Per-layer bottleneck at Nc2 is **architecture-specific** (OPT yes, Llama2 no)
- Per-layer coupling precursor **confirmed on CPU** (Analysis 12)

---

## Citation

```bibtex
@article{amin2026lying,
  title   = {Lying Is Just a Phase: The Hidden Alignment Transition
             in Language Model Scaling},
  author  = {Amin, Adil},
  journal = {arXiv preprint arXiv:2605.18838},
  year    = {2026}
}

@article{amin2026growing,
  title   = {The Growing Pains of Frontier Models: Capability Coupling
             Across Labs and Scales},
  author  = {Amin, Adil},
  journal = {arXiv preprint arXiv:2605.18840},
  year    = {2026}
}
```

---

## Background

The coupling mathematics underlying CAPE was originally developed for multi-band superconductors:
> Amin & Agterberg (2020). *Generalized spin-fluctuation feedback.* [Phys. Rev. Research **2**, 013055](https://link.aps.org/doi/10.1103/PhysRevResearch.2.013055)

Part of the CAPE program — applying effective field theory to AI scaling laws.

---

**Adil Amin** · ZEHEN Labs · adil@zehenlabs.com
[Website](https://zehenlabs.com) · [LinkedIn](https://www.linkedin.com/in/adil-amin-ph-d-1217a91a3) · Issues/PRs welcome
