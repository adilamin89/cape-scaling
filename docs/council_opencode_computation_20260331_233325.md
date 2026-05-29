# Council OpenCode Computation Memo

Date: 2026-03-31
Repo: `cape-push-ready`

## Source availability note

I could not read the following requested sources because they are not present in this checkout:

- `docs/cape/CAPE_COMPUTATION_INDEX.md`
- `docs/cape/CC_SESSION_v13.md`
- `docs/sessions/CC_SESSION_v11.md`
- `docs/sessions/CC_SESSION_v12.md`
- `data/internal/cape_micro_macro_results.json`
- `scripts/session13_internal/cape_micro_to_macro.py`
- `basin_memory/physics_engine.py`
- `basin_memory/cape_physics_v4.py`

This memo therefore uses the accessible repo state: `README.md`, `PAPER2_EXTENSIONS.md`, `paper3A.tex`, `data/*.json`, and the scripts currently under `scripts/`.

## 1) Full script inventory with run status

Run-status convention:

- `confirmed artifact`: script output exists in repo
- `documented/no artifact`: script is clearly intended for use but leaves no checked-in output
- `stale-or-unclear`: some output exists but freshness or exact provenance is uncertain
- `blocked by missing path`: script writes to a path that does not exist in this checkout

| Script | Purpose | Run status |
|---|---|---|
| `scripts/verify_and_reproduce.py` | Recompute core CAPE/AI-free-energy numbers and verification figure | confirmed artifact (`figures/ai_free_energy_verification.png`) |
| `scripts/generate_all_figures.py` | Regenerate paper figures 1-9 | confirmed artifact (`figures/fig1_main.png` to `figures/fig9_frontier.png`) |
| `scripts/bootstrap_Nc.py` | Bootstrap and jackknife critical scale `N_c` | confirmed artifact (`data/bootstrap_Nc_results.json`) |
| `scripts/beta_final_analysis.py` | 6-model gradient scaling analysis plus PySR attempt | confirmed artifact (`data/beta_final_6model.json`, `figures/beta_final_6model.png`) |
| `scripts/pythia_gradient_extraction.py` | Heavy gradient extraction across Pythia sizes | documented/no artifact |
| `scripts/pysindy_per_phase.py` | Per-phase ODE fits for tax, transition, and bonus regimes | stale-or-unclear (`scripts/pysindy_per_phase.json` present) |
| `scripts/olmo_gradient_validation.py` | Test OLMo gradient-dip prediction vs Pythia trend | documented/no artifact |
| `scripts/test_dashboard.py` | Playwright checks for dashboard behavior | documented/no artifact |
| `scripts/swe_gpqa_coupling.py` | Frontier SWE x GPQA coupling analysis | documented/no artifact |
| `scripts/swe_gpqa_coupling_v22.py` | Updated frontier coupling variant | documented/no artifact |
| `scripts/cape_frontier_full.py` | Full frontier analysis with family trajectories and h-field | documented/no artifact |
| `scripts/cape_family_hfield.py` | Family-level h-field analysis | documented/no artifact |
| `scripts/cape_architecture_probe.py` | Near-1B architecture control, including OLMo | documented/no artifact |
| `scripts/cape_stiffness.py` | Stiffness ratio `kappa = lambda1/lambda2` vs scale | documented/no artifact |
| `scripts/cape_algebraic_nc.py` | Algebraic phase-boundary classifier calibrated from OLMo | documented/no artifact |
| `scripts/cape_phase_transfer.py` | Phase-transfer matrix and residual coupling matrix | documented/no artifact |
| `scripts/cape_null_model.py` | Null-model and overconstraint argument | documented/no artifact |
| `scripts/cape_nc3_saturation.py` | Nc3 saturation/activation probe | documented/no artifact |
| `scripts/cape_nc3_deep.py` | Deeper Nc3 interpretation pass | documented/no artifact |
| `scripts/gen_fig9_frontier.py` | Standalone frontier figure generator | blocked by missing path |
| `scripts/gen_fig10_nc3.py` | Standalone Nc3 figure generator | blocked by missing path |
| `scripts/diagnostics.py` | Utility module for scaling diagnostics | library/helper, not a standalone run target |

Non-script artifacts already present under `scripts/`:

- `scripts/pysindy_per_phase.json`: likely output of `scripts/pysindy_per_phase.py`
- `scripts/ode_transition_residuals.json`: important for PySR follow-up, but generator is not obvious in this checkout

Immediate repo gaps relative to the requested work:

- no `docs/` subtree existed before this memo
- no `basin_memory/` subtree
- no `scripts/session13_internal/` subtree
- no `data/internal/` subtree
- no checked-in session notes for Sessions 11-13

## 2) Basin memory techniques applicable to CAPE

Because the requested `basin_memory/*.py` files are missing, this section is an adaptation plan rather than a direct extraction.

Most useful basin-memory ideas to port into CAPE:

1. **Metastable basin labeling in capability space**
   - Treat each model as a state in `(phi1, phi2, h_D, d_eff, gamma12)` rather than only `(HS, TQA)`.
   - Define basins by sign and magnitude of `gamma12`, soft-mode size `lambda2`, and local curvature around `N_c`.
   - This gives stable labels such as `deep_tax`, `near_Nc`, `bonus`, `frontier`, and later `Nc3-active`.

2. **Barrier-height tracking across scale**
   - Translate free-energy barrier ideas into the observed quantities already in repo: gradient dip, susceptibility peak, eigenvector rotation, and ODE coefficient jump.
   - Use barrier proxies to rank which transitions are real phase changes versus family-specific recipe excursions.

3. **Path-dependent memory / hysteresis diagnostics**
   - CAPE already hints at memory in `paper3A.tex` via "order-parameter memory".
   - On expanded datasets, test whether distilled, curated, or RLHF-heavy families keep bonus-phase geometry below the nominal `N_c`.
   - This is the cleanest place to formalize Phi-like or post-training-induced hysteresis.

4. **Basin transition operator**
   - `scripts/cape_phase_transfer.py` already approximates a phase-to-phase transfer matrix.
   - A basin-memory engine should generalize that into a learned transition operator between `tax -> crossover -> bonus -> frontier` basins, with residuals passed to PySR.

5. **Attractor stability from local Jacobians**
   - Reuse per-phase ODE coefficients from `scripts/pysindy_per_phase.py` as local linearizations.
   - Compare eigenvalues/eigenvectors of these Jacobians across families to test which regimes are stable attractors and which are narrow transition corridors.

6. **Memory-aware intervention planning**
   - Use the basin label plus local susceptibility to decide where alignment interventions should be cheapest.
   - The repo's own theory suggests the optimal intervention zone is near `N_c`, where restoring forces are weakest.

Recommended CAPE-specific state vector:

```text
z(N) = [phi1, phi2, gamma12, lambda2, theta_evec, h_D, d_eff, grad_ratio]
```

where `grad_ratio = measured_grad / family_baseline_grad` becomes the bridge between weight-space and benchmark-space memory.

## 3) Local vs Colab computation sequence

Best split from current scripts:

### Local first

Run locally for all lightweight, deterministic, and artifact-producing analyses:

1. `python scripts/verify_and_reproduce.py`
2. `python scripts/generate_all_figures.py`
3. `python scripts/bootstrap_Nc.py`
4. `python scripts/pysindy_per_phase.py`
5. `python scripts/cape_architecture_probe.py`
6. `python scripts/cape_phase_transfer.py`
7. `python scripts/cape_stiffness.py`
8. `python scripts/cape_null_model.py`
9. frontier/Nc3 scripts (`swe_gpqa_coupling*.py`, `cape_frontier_full.py`, `cape_nc3_*.py`)

### Colab / remote GPU second

Use Colab or another GPU box for memory-heavy model-weight computations:

1. `python scripts/pythia_gradient_extraction.py --colab --n-batches 10 --batch-size 4 --seq-len 512`
2. shard by model size if needed and merge `*_results.json`
3. `python scripts/olmo_gradient_validation.py`
4. any future Qwen/OLMo/TransformerLens activation or gradient runs
5. PySR Julia jobs if local Julia setup is unstable

### Back on local after GPU jobs finish

1. copy JSON outputs into `data/` or a new `data/internal/`
2. rerun `scripts/beta_final_analysis.py`
3. run PySR on residual matrices and transition operators
4. regenerate summary figures and memo tables

Rule of thumb:

- local = analysis, plotting, bootstrap, ODE fitting, bookkeeping
- Colab = checkpoint loading, backward passes, activation extraction, TransformerLens debugging, Julia-heavy symbolic regression if local install is brittle

## 4) Qwen TransformerLens bug test plan

No Qwen or TransformerLens code is present in this checkout, so this is a proposed test plan.

Goal: determine whether the bug is in model loading, tokenizer/template handling, architecture conversion, or hook semantics.

### Stage A: minimal parity test

For one small Qwen checkpoint and one known-good non-Qwen control:

1. load with raw Hugging Face
2. load with TransformerLens
3. use identical tokenized inputs
4. compare:
   - token ids
   - logits on final position
   - top-10 next-token probabilities
   - selected residual stream norms per layer

Pass condition: max-logit difference is near floating-point tolerance after disabling generation helpers.

### Stage B: tokenizer and chat-template isolation

Run the same prompt four ways:

1. plain text, no chat template
2. HF chat template
3. manual BOS/EOS handling
4. TransformerLens tokenization path

If plain-text parity passes but chat-template parity fails, the bug is tokenizer/template-side, not attention-side.

### Stage C: architecture compatibility checks

Explicitly verify Qwen-specific details that often break conversions:

- rotary embedding scaling/base
- grouped-query or multi-query attention layout
- RMSNorm placement
- tied vs untied embeddings
- BOS/EOS/pad token configuration
- sliding-window or cache assumptions
- attention mask semantics

### Stage D: hook-level consistency

For one prompt, record per-layer tensors from both implementations:

- residual pre
- attention output
- MLP output
- residual post

Binary-search the first layer where divergence exceeds tolerance.

### Stage E: gradient/backward sanity

If the end use is CAPE weight-space analysis, also compare:

1. scalar loss on same input
2. total gradient norm
3. selected layer gradient norms

If forward parity holds but gradient parity fails, the bug sits in hooks, loss handling, or mixed-precision/backward configuration.

### Deliverables

- one notebook or script that writes `qwen_tl_parity_report.json`
- one CSV of layerwise diffs
- one verdict field: `tokenizer`, `conversion`, `attention`, `hooking`, or `backward`

Recommended first target: smallest Qwen model available, then escalate only after exact forward parity is achieved.

## 5) OLMo gradient normalization plan

Current repo logic mixes two notions of normalization:

- total norm `||grad||`
- per-parameter norm `||grad|| / sqrt(N)`

For an OLMo-vs-Pythia comparison on expanded data, use a stricter normalization stack.

### Canonical measurement protocol

1. same loss definition across families
2. same dataset slice (prefer fixed C4 validation text, not random tokens if avoidable)
3. same sequence length, batch size, and number of batches
4. float32 accumulation for norms even if model forward uses lower precision
5. identical treatment of embeddings, LM head, and tied weights

### Save all four gradient metrics

For every model, save:

1. `grad_total = ||g||_2`
2. `grad_per_param = ||g||_2 / sqrt(N_trainable)`
3. `grad_per_token = ||g||_2 / sqrt(tokens_in_batch)`
4. `grad_loss_normalized = ||g||_2 / loss`

### Add layerwise normalization

Also store per-layer:

- raw layer norm
- layer norm divided by `sqrt(params_in_layer)`
- fraction of total gradient mass in early, middle, late layers

This matters because the repo theory explicitly treats the Pythia-1B anomaly as a redistribution of gradient mass, not only a scalar dip.

### Comparison outputs

For OLMo specifically compute:

```text
ratio_to_pythia_fit = measured_grad_per_param / predicted_pythia_grad_per_param(N)
```

Interpretation:

- `< 1`: screened / dip-like / transition-like
- `~ 1`: on-family trend
- `> 1`: stronger-than-expected gradient, likely outside the crossover story

Priority OLMo run order:

1. OLMo-1B
2. OLMo-7B
3. if available, OLMo intermediate sizes or OLMo-2 family for normalization robustness

## 6) Priority computation list for all papers

This prioritization uses the present repo structure: main CAPE paper (`paper3A.tex`), supplementary derivations, and Paper 2 extensions (`PAPER2_EXTENSIONS.md`).

### Tier 0: unblock the main CAPE paper now

1. Full Pythia gradient extraction across all sizes
2. OLMo gradient validation at `gamma12 = 0`
3. Refresh per-phase ODE fits and transition residuals
4. Recompute expanded cross-family dataset tables if 63-model set exists off-repo
5. Regenerate figures/tables from one clean data snapshot

### Tier 1: strengthen the main paper's "independent probes" argument

1. Architecture-control refresh including OLMo and any Qwen near-1B points
2. Recompute stiffness, eigenvector rotation, and susceptibility on expanded dataset
3. Rebuild holdout tests with more than one held-out family
4. Re-run null-model / overconstraint summary using the larger dataset

### Tier 2: Paper 2 / `d_eff > 2` extensions

1. Expanded PCA on all available models with consistent benchmark set
2. `d_eff(N)` fit on 26 -> 63 models
3. Nc3 activation test with third-axis benchmark coverage
4. Leggett-mode search across family trajectories
5. full susceptibility matrix `chi_ij`
6. discontinuity detector across `gamma12`, `lambda2`, `theta`, `d_eff`

### Tier 3: micro-to-macro and symbolic-discovery program

1. PySR on gradient residuals
2. PySR on phase-transfer residual matrix `Delta A_coupling`
3. symbolic fit for Nc2/Nc3 transition operators
4. symbolic bridge between weight-space observables and benchmark-space coupling

### Tier 4: infrastructure

1. Qwen TransformerLens parity harness
2. standardized internal results schema under `data/internal/`
3. reproducible Colab notebooks for heavy runs
4. one consolidated computation index document once missing docs return

## 7) What from Sessions 10-12 techniques needs rerunning on expanded 63-model dataset

The actual Session 11-12 notes requested are missing, so this is the best reconstruction from current code and paper text.

Anything that depends on sample size, family coverage, or cross-family geometry should be rerun.

### Must rerun

1. **PCA / eigen-spectrum / `d_eff`**
   - the 26-model PCA in `data/cape_26models_9families.json` is too small for final claims on universality and Nc3

2. **`h(D)` ranking and residual family offsets**
   - current ranking is one of the clearest cross-family tools and will shift materially with 63 models

3. **Phase labels and algebraic classifier accuracy**
   - any `41/44`-style claim should be recomputed on the full set

4. **Per-phase ODE fits and sign-flip boundaries**
   - phase-specific dynamics should be re-estimated with denser near-`N_c` coverage

5. **Bootstrap `N_c` and confidence intervals**
   - more models should tighten or move the CI; do not keep the old interval by default

6. **Frontier coupling and Nc3 scripts**
   - all SWE/GPQA/IFEval-style computations should be rerun if new frontier families are included

7. **Eigenvector rotation and phase-transfer matrix**
   - these are exactly the kinds of geometric quantities that benefit from denser family sampling

8. **Null-model / overconstraint summary counts**
   - counts like `12/12`, `41/44`, and family-holdout success rates need a clean refresh

### Probably rerun

1. architecture-control tables
2. stiffness trend fits
3. holdout prediction experiments with more withheld families
4. GP baselines for `phi1` and `phi2`

### Need new, not just rerun

1. family-stratified fits so large families do not dominate
2. missingness-robust PCA if the 63-model benchmark matrix is sparse
3. three-axis analyses for models with SWE/GPQA/IFEval or equivalent frontier benchmarks
4. explicit uncertainty propagation for h-field and classifier outputs

## 8) PySR Julia run plan

PySR is already referenced in `scripts/beta_final_analysis.py`, `scripts/cape_phase_transfer.py`, and the residual notes, so the right next step is a clean Julia-backed symbolic-regression lane.

### Environment setup

1. install Julia 1.10+
2. create a dedicated Python env for `pysr`
3. first-run package build outside any long experiment

Suggested bootstrap:

```bash
python -m venv .venv-pysr
source .venv-pysr/bin/activate
pip install pysr numpy pandas scipy
python -c "from pysr import PySRRegressor; print('PySR import ok')"
```

### Smoke test

Before any CAPE data, run a 2-minute toy regression to verify Julia package install, worker launch, and equation export.

### CAPE run order

1. **Gradient scaling residuals**
   - input: `logN`, `loss`, measured gradient metrics
   - objective: explain the 1B dip and departures from simple power law

2. **Per-phase ODE residuals**
   - input: midpoint benchmark states and derivative residuals
   - objective: discover correction terms beyond the current linear phase ODE

3. **Phase-transfer residual matrix**
   - input: `Delta A_coupling` from `scripts/cape_phase_transfer.py`
   - objective: symbolic form for coupling corrections between basins

4. **Nc2 / Nc3 operator search**
   - input: expanded multi-benchmark frontier data
   - objective: compact symbolic trigger for third-axis activation

### Operator budget

Start simple:

- binary: `+`, `-`, `*`, `/`
- unary: `log`, `exp`, `sqrt`, `abs`

Only add trigonometric or piecewise operators if the simple library fails repeatedly; otherwise the search space will explode.

### Runtime strategy

- local laptop: smoke tests and tiny residual sets
- Colab/remote CPU-GPU box: longer `niterations`, larger populations, repeated seeds
- always save hall-of-fame CSV/JSON after each run

### Output schema

For each PySR experiment save:

```text
task_name
input_features
normalization_used
operator_set
random_seed
best_equations
pareto_front
held_out_error
physical_interpretation
```

### Success criteria

Prioritize equations that are not only low-loss but interpretable in CAPE terms:

- sign-flip trigger near `N_c`
- barrier/susceptibility interpretation
- transportable across families
- stable under resampling

## Bottom line

The current checkout is enough to organize the next computation wave, but not enough to directly answer the missing Session 11-13 and basin-memory requests. The highest-value path is:

1. finish full Pythia gradients
2. run OLMo with stricter normalization
3. stand up a Qwen TransformerLens parity harness
4. rerun all geometry/ODE/classifier analyses on the expanded 63-model dataset
5. reserve PySR for residuals and transition operators, not raw benchmark tables first
