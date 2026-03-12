# Cowork / Claude Code — GitHub Setup & Maintenance Guide

## What this doc covers
1. One-time GitHub setup (create repo, push all files)
2. GitHub Pages (dashboard live at adilamin89.github.io/cape-scaling)
3. Playwright dashboard tests (verify all numbers render correctly)
4. Ongoing workflow (update arXiv ID, re-push figures, run scripts)
5. Archiving old private repos

---

## STEP 1 — Create GitHub Repo (browser, 3 min)

1. Go to **github.com/new**
2. Repository name: `cape-scaling`
3. Visibility: **Public** ← important for GitHub Pages
4. Do NOT initialise with README (we have one)
5. Click **Create repository**

Copy the SSH URL: `git@github.com:adilamin89/cape-scaling.git`

---

## STEP 2 — Push from this folder (Cowork or terminal)

```bash
# Navigate to the repo folder you downloaded/unzipped
cd /path/to/cape-scaling

# Initialise git (if not already done)
git init
git branch -M main

# Add remote
git remote add origin git@github.com:adilamin89/cape-scaling.git

# Stage everything
git add .

# First commit
git commit -m "Initial commit: Paper 3A, dashboard, all scripts and figures"

# Push
git push -u origin main
```

**If you get auth errors:**
```bash
# Use HTTPS instead with a Personal Access Token (PAT)
git remote set-url origin https://github.com/adilamin89/cape-scaling.git
# Then push — it will ask for username and PAT as password
```
Generate a PAT at: github.com/settings/tokens → New classic token → check `repo` scope

---

## STEP 3 — Enable GitHub Pages (browser, 1 min)

1. Go to your repo → **Settings** → **Pages** (left sidebar)
2. Source: **Deploy from a branch**
3. Branch: `main`, folder: `/ (root)`
4. Click **Save**
5. Wait ~2 min → dashboard live at: **https://adilamin89.github.io/cape-scaling**

The `index.html` at repo root IS the dashboard — no build step needed.

---

## STEP 4 — Playwright Dashboard Tests (Claude Code / terminal)

Install Playwright once:
```bash
pip install playwright
playwright install chromium
```

Run the test suite:
```bash
python scripts/test_dashboard.py
```

This opens the dashboard, clicks all 5 tabs, and verifies:
- All key numbers are visible (r=−0.989, Nc=3.5B, A=0.629, etc.)
- All 10 frontier models appear on Frontier tab
- All 26 Pythia/OLMo/Llama models on Explorer tab
- h-field values for Sonnet 4.6 (−13.9), Opus 4.6 (+2.2) are correct
- No JavaScript console errors
- GitHub link resolves

---

## STEP 5 — After arXiv ID lands

Update two lines:

**In `index.html`** — find `arXiv: pending` and replace:
```
arXiv: 2503.XXXXX   →   arXiv: 2503.YOURID
```

**In `README.md`** — find the arXiv badge line and replace:
```
[arXiv: pending](https://arxiv.org)  →  [arXiv 2503.YOURID](https://arxiv.org/abs/2503.YOURID)
```

Then:
```bash
git add index.html README.md
git commit -m "Add arXiv ID 2503.YOURID"
git push
```

---

## STEP 6 — Archive old private repos

For any earlier paper draft repos you want to keep but hide:
1. Go to repo → **Settings** → scroll to **Danger Zone**
2. **Change visibility** → Private
3. Or: **Archive this repository** (read-only, permanent)

Repos to consider archiving:
- Any draft versions before `cape-scaling` (paper3A_v1, paper3A_draft, etc.)

---

## Repo Structure Reference

```
cape-scaling/
├── index.html              ← GitHub Pages dashboard (self-contained, no build)
├── paper3A.pdf             ← Full paper
├── paper3A.tex             ← LaTeX source
├── README.md               ← Landing page for the repo
├── COWORK_SETUP.md         ← This file
├── .gitignore              ← Python/LaTeX/macOS ignores
│
├── data/
│   └── ai_free_energy_data.json   ← All benchmark data, 26 models
│
├── figures/                ← All 9 paper figures (PNG, 200dpi)
│   ├── fig1_main.png
│   ├── fig2_loss_boost.png
│   ├── fig3_thermo.png
│   ├── fig4_gradient.png
│   ├── fig5_ode_actual.png
│   ├── fig6_coupling_jump.png
│   ├── fig7_perphase_coupling.png
│   ├── fig8_topology.png
│   └── fig9_frontier.png
│
└── scripts/
    ├── verify_and_reproduce.py    ← Run this first — reproduces all key numbers
    ├── generate_all_figures.py    ← Regenerates all 9 figures → figures/
    ├── bootstrap_Nc.py            ← Bootstrap CI for Nc transition point
    ├── beta_final_analysis.py     ← β exponent (order parameter) analysis
    ├── swe_gpqa_coupling.py       ← Frontier model SWE vs GPQA coupling
    ├── pythia_gradient_extraction.py  ← Needs GPU: gradient norm extraction
    └── test_dashboard.py          ← Playwright tests for the dashboard
```

---

## Running the Analysis (M2 Mac)

```bash
# Install dependencies
pip install numpy scipy matplotlib scikit-learn

# Quick verify — prints all 8 key numbers in ~5 seconds
python scripts/verify_and_reproduce.py --numbers

# Full verify — prints numbers + saves verification figure
python scripts/verify_and_reproduce.py

# Regenerate all 9 paper figures into figures/
python scripts/generate_all_figures.py

# Bootstrap CI on Nc (takes ~30 seconds)
python scripts/bootstrap_Nc.py

# Frontier coupling
python scripts/swe_gpqa_coupling.py

# GPU gradient extraction (Pythia 6.9B, 12B — needs >16GB RAM or Colab)
python scripts/pythia_gradient_extraction.py --model pythia-6.9b
```

Expected output from `verify_and_reproduce.py --numbers`:
```
r(HellaSwag, TruthfulQA) within-family:  -0.989   p=3.6e-06  ✓
Nc (zero crossing, bootstrap median):     3.50B              ✓
γ₁₂ slope A:                              0.629              ✓
OLMo-7B γ₁₂:                             0.000              ✓
d_eff at Nc:                              1.254              ✓
β_measured:                               0.40               ✓
Ginzburg number Gi:                       1.35               ✓
κ_GL (Type II confirmed):                 0.767              ✓
```

---

## Dashboard — what each tab shows

| Tab | Content | Key chart |
|-----|---------|-----------|
| EXPLORER | TruthfulQA vs N for 26 models + live γ₁₂ calculator | Canvas scatter — U-shape |
| FRONTIER | SWE-bench vs GPQA for 10 frontier models | Canvas scatter + h-field table |
| OBSERVABLES | Thermodynamic table: derivative order, value, measurement method | Canvas: γ₁₂, d_eff, eigenvalue, specific heat |
| HOW IT WORKS | Plain-language 4-panel explanation | Text panels |
| KEY RESULTS | 12 numbered stats + paper card + links | Stats grid |

---

## Claude Code Tasks (run on M2 Mac — GPU/memory tasks)

These are too heavy for Claude Chat — route to Claude Code CLI:

```bash
# Extend β analysis to Pythia 1B, 1.4B, 2.8B
python scripts/beta_final_analysis.py --extend

# 2nd derivative free energy anomaly test
python scripts/priority0_free_energy_2nd_deriv.py

# Leave-one-family-out robustness
python scripts/priority1_leave_one_family_out.py

# GPU gradient extraction for 6.9B and 12B
python scripts/pythia_gradient_extraction.py --model pythia-6.9b
python scripts/pythia_gradient_extraction.py --model pythia-12b
```

---

## Google Colab Tasks (need GPU, >16GB)

Upload `scripts/pythia_gradient_extraction.py` and `data/ai_free_energy_data.json` to a Colab notebook, then:
```python
!python pythia_gradient_extraction.py --model pythia-6.9b --gpu
!python pythia_gradient_extraction.py --model pythia-12b --gpu
```

---

## Contact / Links

- Paper: [arXiv: pending → update after submission]
- Dashboard: https://adilamin89.github.io/cape-scaling
- Email: adilamin@uwm.edu · adil89aminx@gmail.com
