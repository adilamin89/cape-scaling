# CAPE Paper 2 — Beyond Two-Field Theory: Extensions at d_eff > 2

**Status:** Dashboard-ready (modular), not yet in paper.
**Depends on:** Frontier data with 3+ independent benchmarks showing d_eff > 2.

---

## 1. d_eff → 3 Transition (The Central Prediction)

**Physics:** When det(H) → 0 at ~130B, the two-field GL free energy F = a₁φ₁² + a₂φ₂² + γ₁₂φ₁φ₂ can no longer capture the system. A third field φ₃ (e.g., SWE-bench or agentic capability) acquires independent variance, and d_eff rises above 2.

**ML translation:** The participation ratio computed from the eigenvalue spectrum exceeds 2 when a third benchmark cannot be predicted from the first two principal components. At that point, the regime classification requires three axes, not two.

**Falsifiable prediction:**
- d_eff(100B) ≈ 2.0 (current data: 1.99, consistent)
- d_eff(500B) should exceed 2.0 if third axis activates
- d_eff(1T+) should approach 3.0

**Detection method:** Monitor eigenvalue spectrum. When λ₃/λ₁ > 0.1 (currently ~0.02), the third mode is active.

**Current numbers from dashboard:**
- d_eff(Pythia-only, <12B): 1.047
- d_eff(extended, <40B): 1.625
- d_eff(all 26 models): 1.987
- Fitted formula: d_eff ≈ −0.27·log₁₀(N) + 3.9

---

## 2. Leggett Modes (Relative Phase Oscillations)

**Physics:** In a multi-band superconductor, when two order parameters couple, their *relative phase* θ = arg(ψ₁) − arg(ψ₂) can oscillate around the equilibrium value. These are Leggett modes — collective excitations of the inter-band phase.

**ML translation:** When a third benchmark axis activates (d_eff > 2), the *relative alignment* between benchmark pairs can fluctuate. For example, the HS-TQA coupling might be positive while the SWE-GPQA coupling oscillates sign — a "Leggett mode" in benchmark space.

**Observable signature:**
- Anticorrelation between γ₁₂(HS,TQA) and γ₁₃(HS,SWE) across model families
- The relative coupling angle θ₁₂ − θ₁₃ oscillates rather than being locked
- Frequency of oscillation (in log₁₀N space) relates to the inter-band coupling strength

**Testable with:** Family-by-family coupling measurements at frontier scale. Need ≥5 models per family with all three benchmarks.

**Current evidence:** Sonnet 4.6 shows γ₁₂(SWE,GPQA) = −3.88 while other Anthropic models are positive. This isolated sign excursion *could* be a Leggett mode — or just noise. Need more data points.

---

## 3. BKT Transition (Vortex-Antivortex Unbinding)

**Physics:** In 2D systems (d_eff ≈ 2), the Berezinskii-Kosterlitz-Thouless transition involves topological defects: vortex-antivortex pairs that are bound below T_BKT and unbind above it. No order parameter changes sign — it's a *topological* transition.

**ML translation:** In 2D benchmark space (d_eff ≈ 2), a "vortex" is a model where the local coupling structure rotates 2π as you traverse neighboring models in parameter space. Below a critical complexity, capability couplings are locally coherent (vortices are bound). Above it, they "unbind" — different model families show qualitatively different coupling structures even at the same scale.

**Observable signature:**
- Compute the "winding number" of eigenvector angle θ(N) as N traverses a closed loop in (family × size) space
- If winding number ≠ 0, a vortex is present
- BKT transition: universal jump in the superfluid stiffness ↔ sudden loss of cross-family universality

**Detection method:**
- Plot θ(N) for each family separately
- If families diverge above some N_BKT while converging below, that's the unbinding
- Currently: families converge (r = +0.770 across 26 models). No vortex unbinding yet detected.

**Falsifiable prediction:** If the cross-family correlation drops sharply above ~500B, and different families show different eigenvector orientations, this is the BKT signature.

---

## 4. PDW (Pair-Density Wave) Analogue

**Physics:** In certain superconductors, the order parameter modulates *spatially* — the pairing amplitude oscillates in real space. This is a pair-density wave.

**ML translation:** The coupling γ₁₂ modulates with a *third parameter* beyond log₁₀(N). Candidates: training data size D, learning rate schedule, or RLHF intensity. If γ₁₂(N, D) shows periodic behavior in D at fixed N, that's a PDW.

**Observable signature:**
- At fixed N (e.g., 7B), compare γ₁₂ across different training runs with different D
- If coupling oscillates (positive → negative → positive) as D increases, this is PDW
- Currently: Phi's h(D) = +23 vs Falcon's h(D) = −2.7 at similar N could be the first hint

**Status:** Speculative. Needs multiple training runs at same N with varying D.

---

## 5. Spin Susceptibility and Multi-Component Order

**Physics:** In a system with multiple order parameter components, the spin susceptibility tensor χ_ij measures how component i responds to a field conjugate to component j. Off-diagonal χ_ij ≠ 0 means components are coupled.

**ML translation:** The full capability susceptibility matrix is:
```
χ_ij(N) = ∂²F/∂h_i∂h_j
```
where h_i is the external field conjugate to benchmark i. Off-diagonal entries measure how perturbing one benchmark (via targeted training) affects another.

**Computable now:**
- χ_γ(N) = 1/|γ₁₂| (scalar coupling susceptibility) — DONE, in paper
- χ₂(N) = 1/λ₂ (soft-mode susceptibility) — DONE, in paper
- Full χ_ij matrix from the Hessian of the covariance — COMPUTABLE from dashboard PCA

**Dashboard implementation:** The Jacobi eigenvalue decomposition already gives the full eigenvalue spectrum. The susceptibility matrix is the inverse of the covariance matrix. Can be added as a display panel.

---

## 6. Automatic Discontinuity Detection

**Physics concept:** A first-order transition shows discontinuity in the first derivative of the free energy. A second-order shows discontinuity in the second derivative.

**ML concept:** Compute dγ₁₂/dlog₁₀N, d²γ₁₂/d(log₁₀N)², and look for jumps.

**Currently known:**
- dγ₁₂/dlog₁₀N = 0.629 (constant — the linear running)
- d²γ₁₂/d(log₁₀N)² = 0 (linear means second derivative vanishes)
- The ODE coupling jump (6.3× across Nc) IS the discontinuity — in the ODE coefficient, not in γ₁₂ itself

**For dashboard:** Sweep any computed quantity (γ₁₂, d_eff, r, χ, eigenvector angle) vs log₁₀N. Compute numerical derivative. Flag where |d/dlogN| exceeds a threshold. This is the "discontinuity detector."

---

## 7. Free Energy and Phase Diagram Construction

**The free energy is:**
```
F(φ_HS, φ_TQA; N) = a₁(N)φ_HS² + a₂(N)φ_TQA² + γ₁₂(N)φ_HS·φ_TQA + (1/4)(φ_HS⁴ + φ_TQA⁴)
```

**Phase boundaries** are where ∂F/∂φ = 0 has degenerate solutions.

**For Paper 2:** Explicitly construct F from the measured couplings and show that the phase boundary matches Nc = 3.5B. Then show what happens to F when d_eff → 3 (the quartic terms mix, and new minima appear).

---

## Summary: What Goes Where

| Extension | Dashboard (now) | Paper 1 | Paper 2 |
|---|---|---|---|
| χ_γ = 1/\|γ₁₂\| divergence | ✅ Explorer tab | ✅ Added as 13th diagnostic | — |
| Regime-aware PCA | ✅ Explorer tab | ✅ Added as footnote | Methodology section |
| d_eff → 3 prediction | ✅ Dashboard formula | ✅ Sharpened conclusion line | Central result |
| Leggett modes | 🔲 Ready to add | — | Prediction + search |
| BKT transition | 🔲 Ready to add | — | Prediction + search |
| PDW analogue | 🔲 Speculative | — | If data supports |
| Full χ_ij matrix | 🔲 Computable from PCA | — | Observable table |
| Discontinuity detector | 🔲 Ready to implement | — | Diagnostic method |
| Free energy construction | 🔲 Equations ready | — | Theory section |

---

*All extensions are designed to be modular — each can be added to the dashboard as a new panel without modifying existing code. The dashboard's PCA engine, Jacobi eigensolver, and regime-aware partitioning already provide the infrastructure.*
