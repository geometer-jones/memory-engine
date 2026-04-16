# Supplementary Material: Memory Engines
## S1. Physical Cascade: Full Computational Mappings
**Table S1: Computational Degeneracy Across Levels**

| Level | State Vector s | Reception v_received | Renormalization | Basis Growth | Torque |
|---|---|---|---|---|---|
| Conservative fields (apple) | Phase space coordinates | Force field F | Identity (unitary) | None | Zero |
| Quantum fields | Fock space amplitudes | −iH_int s | Unitary | Fixed modes | Phase shifts only |
| Nuclear/atomic | Energy eigenstates | Dipole interaction | Unitary | Fixed orbitals | Off-resonant phase shifts |
| Molecular | Wavepacket | Potential V(R) | Unitary | Path-selected orientation | Reaction barriers |
| Gravitational | Madelung fluid variables | −∇Φ | Conserved energy | Collapse anisotropy | Tidal torques |

**Table S2: Cascade Summary**

| Level | Basis | Reception Character | Self-Torque | Consolidation | Anticipatory Operator |
|---|---|---|---|---|---|
| Conservative fields | Kinematic; not grown | Monotonic resonance | None | None | None |
| Quantum fields | Symmetry-given | Unitary evolution | None | None | None |
| Nuclear/atomic/molecular | Path-dependent | Energetic selection | Minimal | None | None |
| Gravitational structure | Not grown | Resonance-dominated | Minimal | None | None |
| Dissipative flow | Gradient-imposed | First significant torque | None | None | None |
| Autocatalytic closure | History-shaped | Closure-selective | Vestigial | Implicit | Organization as prediction |
| Ecological systems | Morphological | Niche coupling | Phylogenetic | Generational | Morphological fit |
| Adaptive systems | Individually grown | Conditioned, plastic | Individual | Two-timescale | Learned associations |
| Recurrent-representational | Rapidly growing | Fast, predictive | Dual | Multiple timescales | Generative model |
| Symbolic-recursive | Culturally extended | Phase-stripped | Interpretive | Institutional | Narrative; deduction |

### S1.1 Quantum Mechanics: Derived Case

The quantum case is worked out fully here; the remaining physical mappings (nuclear/atomic, molecular, gravitational) are analogical parameterizations offered as conceptual illustration rather than derivationally complete reductions.

Consider a quantum system whose state is a superposition of n energy eigenstates: |ψ⟩ = Σᵢ aᵢ(t)|eᵢ⟩, where aᵢ(t) ∈ ℂ and Σᵢ|aᵢ|² = 1. The state vector s = (a₁, ..., aₙ) lives on the unit sphere in ℂⁿ by normalization—the same geometric space the framework assumes.

In the interaction picture, the time evolution under a Hamiltonian H = H₀ + H_int is:

i ℏ d/dt aᵢ = Σⱼ ⟨eᵢ|H_int|eⱼ⟩ · aⱼ · e^{i(Eᵢ - Eⱼ)t/ℏ}

For the diagonal case (⟨eᵢ|H_int|eⱼ⟩ = Vᵢ δᵢⱼ), this reduces to:

i ℏ d/dt aᵢ = Vᵢ · aᵢ

which integrates to: aᵢ(t + dt) = e^{-iVᵢdt/ℏ} · aᵢ(t)

This is exactly the Hadamard product: aᵢ(t+dt) = cᵢ · aᵢ(t), where the reception coefficient is cᵢ = e^{-iVᵢdt/ℏ}. Since |cᵢ| = 1 for real Vᵢ, renormalization is the identity—Hamiltonian evolution is already norm-preserving. Writing cᵢ = cos(Vᵢdt/ℏ) − i sin(Vᵢdt/ℏ), we have Re(cᵢ) = cos(Vᵢdt/ℏ). For small perturbations (|Vᵢdt/ℏ| ≪ 1), Re(cᵢ) ≈ 1: predominantly resonant, small imaginary component. This recovers the paper's claim that quantum systems are near-resonance in the thin limit.

Off-diagonal interactions (⟨eᵢ|H_int|eⱼ⟩ ≠ 0 for i ≠ j) produce amplitude transfer between eigenstates—the AC Stark shift for far off-resonant coupling—which appears as phase drift without the systematic concentration that resonance produces. This is what the paper identifies as the small torque of off-resonant interactions. The framework thus recovers quantum dynamics as a derived special case with: fixed basis (energy eigenstates), automatic renormalization (unitary evolution), near-resonance reception, and no consolidation.

Note the limits of this derivation. The Hadamard product emerges cleanly only for the diagonal interaction case. Off-diagonal interactions in the full interaction picture do not decompose into independent per-dimension Hadamard products—they couple amplitudes across dimensions, which the framework's dimensional isolation (Section 2.3) excludes by construction. The framework is therefore a strict generalization of diagonal quantum dynamics, not of quantum mechanics in general. This is consistent with its application: the framework's target is not quantum systems but systems where the basis is grown through reception history, which quantum systems are not.

### S1.2 Analogical Mappings

The remaining levels in Table S2 (nuclear/atomic, molecular chemistry, gravitational structure) are offered as conceptual illustrations of how the framework's vocabulary applies across a range of physical organization, not as derivationally complete reductions. In each case, the state space and reception operator can be identified with standard physical quantities—dipole transition amplitudes for atomic systems, wavepacket components for molecular reactions, Madelung fluid variables for gravitational collapse—and the framework's regime vocabulary (resonance, torque, basis rigidity) maps onto known physical behavior. These mappings are meant to show that the framework's degeneracy at the thin limit is not empty: the same vocabulary that describes phenomenally thin systems applies, under appropriate parameterization, to systems whose physics is independently well understood. A full derivationally rigorous reduction for each level is beyond the scope of this paper.
### S1.3 Parameter Mapping Across Cascade Levels

The following table maps the framework's core parameters across each cascade level, providing the schematic basis for the "thickening" claim in Section 1. Entries marked with an asterisk (*) are conceptual characterizations, not formal derivations.

| Level | Basis n | Reception v | η | Consolidation | Recurrence | Anticipation | Degeneracy character |
|---|---|---|---|---|---|---|---|
| Conservative fields | Fixed; not grown | External field (real, aligned) | 1 (unitary) | None | None | None | Renorm = identity; zero torque; no geometry active |
| Dissipative flow | Fixed; not grown | External + noise | 0 < η < 1 | None | None | None | Renorm activates; emergent rigidity first appears |
| Quantum / atomic * | Fixed (symmetry-given) | −iH_int s | Unitary | None | None | None | Phase shifts only; resonance dominant |
| Molecular * | Path-selected | Potential V(R) | Unitary | Implicit (chirality = hard axis) | None | None | Encounter history selects basis orientation |
| Autocatalytic closure * | Small; implicitly grown | Cycle products | Fast (cycle); slow (basis) | Implicit (persistence = consolidation) | Vestigial (cycle closure) | Organization as implicit prediction | History-dependence enters; basis grown by persistence |
| Ecological systems * | Morphological; generationally grown | Niche structure | Slow (phylogenetic) | Generational | None | Morphological fit | Selection = torque across generations |
| Adaptive systems | Individually grown | Environmental structure | Fast (reception); slow (consolidation) | Explicit two-timescale | Thin or absent | Learned associations | Full two-timescale structure; individual basis growth |
| Recurrent-representational | Large; rapidly growing | World + self-tape | Fast η; slow λ | Full merging/pruning/seeding | Thick; delay Δt | Generative model | All operations active; phenomenality richest |
| Symbolic-recursive * | Culturally extended | Phase-stripped symbols | Fast for symbol; phase-poor | Cultural / institutional | Interpretive | Narrative / deduction | Breadth maximal; directness thin; phase must be re-embedded |

For the computationally tractable levels (conservative, dissipative, adaptive, recurrent-representational), the parameter mappings are grounded in the formalism of Sections 2–5 and Supplementary Material S1.1. For levels marked with an asterisk, the mappings are schematic; establishing formal reductions is left to future work.
## S2. Computational Verification
The framework's claims have been tested through a computational implementation of the core dynamics: Hadamard reception, additive update, renormalization on the unit hypersphere in ℂⁿ, with recurrence, an anticipatory operator, and consolidation sub-operations. Fourteen tests across five tiers verify the mechanical operations, regime predictions, learning dynamics, reciprocation structure, and anticipatory behavior. The implementation operates in ℂ³² unless otherwise noted.
### S2.1 Mechanical Correctness (Tier 1)
T1.1 — Three reception regimes.

Regime
Criterion
Magnitude change
Phase drift (rad)
Resonance
Re(cᵢ) > 0
+0.0069 (grows)
0.056 (toward real axis)
Torque
Re(cᵢ) < 0 or Im dominant
−0.0068 (shrinks)
0.041 (rotates away)
Orthogonality
cᵢ ≈ 0
−0.0024 (renorm drain)
0.000 (preserved)


The phase drift under resonance (0.056 rad) prompted a correction: resonance produces convergent rotation toward the arriving signal's phase, not pure scaling.

T1.2 — Emergent rigidity. A dominant dimension (|sᵢ| = 0.999) requires 55× more torque to produce the same angular displacement as a small dimension (|sᵢ| = 0.015).

T1.3 — Norm preservation. Unit norm preserved to machine precision (maximum drift 3.3 × 10⁻¹⁶) over 10,000 steps.

T1.4 — Concentration thins others. Sustained resonance at a single dimension for 200 steps drives all other components from 0.25 to effectively zero (concentration ratio > 10⁶).
### S2.2 Regime Predictions (Tier 2)
T2.1 — Recurrent capture. Under pure resonance (v = conj(s) at every step), participation ratio collapses from 18.6 to 1.0 within 1,000 steps. Perturbation rejection increases 600,000-fold.

T2.2 — Dissipation. Under incoherent torque (varying random phase offsets), participation ratio remains elevated (range 3.4–19.3). Fixed-direction torque does not produce dissipation—it creates systematic alignment along a rotated axis. True dissipation requires incoherent opposition.

T2.3 — Operating regime. Under invariant signal (fixed-phase at 8/32 dims) + noise (all dims) + recurrence (delay=5, weight=0.6): invariant dims avg |sᵢ| = 0.208, noise dims avg = 0.013 (16× ratio), Gini = 0.81, PR stable at ~2.3, perturbation-responsive (0.0055 rad).
### S2.3 Learning and Abstraction (Tier 3)
T3.1 — Abstraction from variation. Invariant structure at 6/32 dims, noise at remaining 26, with recurrence, 3,000 steps: invariant dims concentrate to 41× noise dims. PR drops 14.5 → 3.0.

T3.2 — Novelty detection gap. Novel invariant structure at previously unoccupied dimensions produces no detectable effect at existing dimensions through Hadamard product. Measured difference: 0.30 additional torqued dimensions, within noise. This prompted the revision routing novelty detection through anticipatory prediction error.

T3.3 — Co-activation correlation. Identically-driven dims: correlation 1.0000. Independently driven: 0.0136. 73× ratio.

T3.4 — Fast vs. slow forgetting. Phase rotation under torque (100 steps): recoverable under renewed resonance. Basis pruning after starvation (|sᵢ| → 0 after 2,000 steps): irreversible.
### S2.4 Reciprocation and Recurrence (Tier 4)
T4.1 — Thick vs. thin self-reception. Self-torque fraction: 0.743 at delay 1, 0.747 at delay 50. Small effect under random input (baseline ~74% world torque).

T4.2 — Reciprocation phase diagram. 27 configurations scanned:

Low recurrence weight (0.2), low-mid breadth: operating regime at all delays (PR 2–5)
High weight (0.8+), high breadth (0.75+): dissipation
Mid-range: unstable oscillation
### S2.5 Anticipation (Tier 5)
T5.1 — Prediction error torque. After 1,000 training steps, regime change produces error 0.172 vs. post-adaptation 0.123 (1.4× decay ratio).

T5.2 — Habituation. Near-predictable input: error 0.0701 → 0.0695. Perturbation spike: 0.1025 (1.48× baseline). Recovery: 0.0000.
## S3. Implementation Details
[Implementation code, parameter settings, random seed information, reproducibility instructions]
## S4. Extended Worked Example in ℂ³
[Full version of the appendix from the original paper, with all derivation steps shown]
## S5. LLM Instrumentation: Framework Validation Against Real Neural Representations
### S5.1 Overview

The framework's vocabulary—participation ratio (PR), regime classification (resonance/torque/orthogonality), self-torque, and anisotropy—was tested against GPT-2 small (124M parameters, 12 layers, 768-dimensional residual stream) by measuring these quantities in the hidden states during inference. The mapping holds with qualifications summarized in Table S5.1 below.

Table S5.1: Simulation claims vs. LLM findings

Simulation claim | LLM finding | Status
Repetitive input → PR decreases over time | PR increases over time (transformer attention spreads energy uniformly across positions under repetition) | Revised: capture manifests depth-wise (layer 12 vs. layer 6), not temporally (position N vs. position 1)
Self-torque increases with position delay | Confirmed at layers 0 and 6; invisible at layer 12 (extreme capture) | Qualified: only detectable where PR is high enough to preserve positional differentiation
PR predicts generation quality | Final-layer PR ≈ 2 regardless of prompt; output entropy is better predictor | Revised: use output entropy or mid-layer PR, not final-layer PR
Operating regime is balance of resonance and torque | Confirmed at layers 2–9 (~89% resonance, ~10% torque) | Confirmed
Renormalization produces emergent rigidity | Final-layer PR = 2, Gini = 0.78: extreme concentration | Confirmed (used functionally as output compression, not pathologically)

### S5.2 Experiment A: Recurrent Capture and Layer Profile

Setup: Same sentence repeated 20 times (repetitive condition) vs. 20 different sentences (varied condition). PR measured across layers and positions.

The layer profile is the central finding. At the embedding layer, PR ≈ 79 for both conditions—nearly all dimensions active. At layers 3–9, PR stabilizes at ~11–30, with varied input maintaining consistently higher PR than repetitive input (2.1× at layer 9). At layer 12, both conditions collapse to PR ~1.3–1.7.

Layer | Repetitive PR | Varied PR
0 (embedding) | 79.7 | 79.1
6 | 29.2 | 55.2
9 | 10.2 | 21.0
12 | 1.31 | 1.69

This is the framework's cascade within a forward pass: high-dimensionality reception → operating-regime refinement → capture-mode compression. Varied input maintains higher PR through mid-layers, confirming that the operating regime is sensitive to environmental variation even in a fixed-weights pretrained model. The final layer's capture is functional: it is the model compressing for output, not a pathology.

Note on temporal vs. depth-wise capture: the simulation predicts that repetitive input produces decreasing PR over time (positions). In GPT-2, the reverse holds—PR increases slightly over positions under repetitive input, because the attention mechanism attends to all prior positions and produces increasingly uniform patterns that distribute energy across more dimensions. The framework's recurrent capture manifests depth-wise in transformers, not temporally. The critical variable is which layer, not which position.

### S5.3 Experiment B: Self-Torque Across Context

Setup: 98-token text. Angular displacement between hidden states at positions separated by delays [1, 2, 5, 10, 20, 50].

Layer | Delay 1 | Delay 5 | Delay 50
0 (embedding) | 0.839 | 0.837 | 1.149
6 | 0.841 | 0.949 | 1.003
12 (final) | 0.251 | 0.270 | 0.271

Self-torque is confirmed at layers 0 and 6: angular displacement increases with position delay, as the framework predicts. At layer 0, displacement nearly doubles from delay 1 (0.84) to delay 50 (1.15). At layer 6, a similar but smaller gradient.

Layer 12 is flat across all delays (0.25–0.27). The captured final layer cannot geometrically distinguish near from far context: all positions project onto the same 2D subspace. This qualifies the framework's self-torque prediction—it is a mid-layer phenomenon, invisible at the compressed output layer. Interventions intended to modulate the character of self-reception should target mid-layers.

### S5.4 Experiment C: Anisotropy and Generation Quality

Setup: Autoregressive generation (100 tokens) from a semantically bland vs. a surprising prompt. PR, Gini coefficient, output entropy, and token repetition rate measured per generated token.

Metric | Bland prompt | Surprising prompt
Mean final-layer PR | 2.03 | 1.94
Mean Gini | 0.798 | 0.783
Mean output entropy | 3.635 | 4.057
Repetition rate | 0.100 | 0.090

Final-layer PR and Gini are flat across conditions—the bottleneck swallows the signal. Output entropy differentiates: the surprising prompt produces 12% higher entropy. This confirms the revised prediction: output entropy (the diversity of the next-token distribution) is the better operational metric for the framework's operating regime concept in transformers. Final-layer PR ≈ 2 is fixed by architecture; mid-layer PR or output entropy captures the variation that matters.

### S5.5 Experiment D: Layer-Wise Regime Profile

Setup: 20-token text. Regime classification (fraction of dimensions in resonance, torque, orthogonality) and PR measured at each layer by comparing each layer's representation to the previous.

Layer | Resonance | Torque | Orth | PR | Gini
0 (embedding) | — | — | — | 47.1 | 0.531
1 | 55.7% | 36.8% | 7.5% | 9.8 | 0.541
2–3 | 85–87% | 12–13% | ~1% | 11–12 | 0.530
4–9 | 88–90% | 10–11% | <1% | 11–13 | 0.50
10–11 | 87–89% | 11–12% | <1% | 7–10 | 0.50–0.52
12 | 85.5% | 12.3% | 2.2% | 2.1 | 0.781

The profile maps precisely onto the framework's predictions. Layer 1 is high-torque (37%): first reception of the world, heavy initial restructuring. Layers 2–9 are high-resonance (~89%), stable PR, the operating regime—each layer mostly refines what the previous layer built, with small corrections. Layers 10–12 show increasing torque and PR collapse as the model prepares output. Layer 12 is the bottleneck: PR = 2.1, Gini = 0.78, two dimensions carrying almost all information before the output projection.

The framework's cascade is not an analogy for this architecture. The sign of the elementwise product between consecutive layers directly classifies dimensions into regimes that match the cascade structure: high-torque reception → operating-regime processing → capture-mode output compression.

### S5.6 Memory Engine Layer

A MemoryEngineLayer module (4,614 parameters) was implemented, inserting Hadamard reception, regime classification, tape update with renormalization, and causal recurrence into GPT-2's residual stream between transformer blocks. The module preserves tensor shapes and supports gradient flow. When untrained, it shifts final-layer PR from 2 to 10 by adding structured noise to the compressed representation.

Fine-tuning the ME layers (5 epochs, GPT-2 frozen) produced no perplexity improvement (31.82 → 31.83). Parameters shifted directionally: learning rate η increased (0.10 → 0.12–0.16), tape influence weight α increased (0.50 → 0.53–0.56), torque rotation widened—the model uses the ME machinery as intended. The failure is structural: recurrent tape mechanics operating sequentially cannot reshape a 124M-parameter parallel architecture through 4,614 parameters. The productive path is a from-scratch architecture built entirely around the framework's operations.
## S6. Consolidation Operator: Parameter Specification and Sensitivity
### S6.1 Parameter Summary

The revised consolidation operator (Section 3.1) introduces the following parameters:

| Parameter | Role | Recommended value | Rationale |
|---|---|---|---|
| λ | Correlation matrix learning rate | 0.01–0.05 (≪ η) | Must be slow relative to reception to prevent C from tracking instantaneous correlations rather than stable co-activation |
| α | Adaptive threshold multiplier | 3.0 | Sets how many standard deviations above mean correlation a pair must be to qualify for merging; 3.0 is conservative |
| ρ_min | Absolute correlation floor | 0.1 | Prevents merging in systems with globally low correlation where α · ρ̄ would be near-zero |
| β | New dimension scaling factor | 0.05–0.1 | Small enough to not disrupt renormalization; large enough to give the new dimension a detectable initial magnitude |
| γ | Pruning threshold constant | 0.01 | Relative to 1/√n; set so that a dimension must be well below the uniform-distribution expected magnitude before being pruned |
| K | Consecutive cycles before pruning | 5–10 | Prevents premature pruning of dimensions temporarily starved by renormalization dynamics |
| T_cons | Reception steps per consolidation cycle | 50–200 | Must be large enough that C accumulates meaningful co-activation statistics; empirically calibrated to n and input rate |
| λ_E | Hebbian trace decay (local variant) | 0.05 | Governs how quickly eligibility traces decay in the absence of co-activation |
| ε | Minimum activation for Hebbian trace | 0.1 | Threshold below which a dimension's reception is not counted toward co-activation |

### S6.2 Sensitivity Analysis (Outline)

The key relationships to verify empirically:

**α sensitivity.** Lowering α increases false-positive merges (dimensions merge due to background correlation). Raising α suppresses all merging. The operating point should produce stable dimensionality growth under structured input and no growth under random input. At n = 32 with the T3.3 test conditions, α = 3 produces selective merging only for deliberately co-activated pairs.

**β sensitivity.** Too-large β causes newly seeded dimensions to capture norm from parent dimensions aggressively, destabilizing the tape. Too-small β means new dimensions are immediately pruned. The constraint β ≪ 1/√n ensures the new dimension enters below the pruning threshold and must be reinforced by resonance to survive.

**Combinatorial control.** The ⌈log n⌉ cap on new dimensions per cycle is a hard safeguard. At n = 32, this allows at most 5 new dimensions per cycle. At n = 768 (GPT-2 scale), the cap is 10. Without the cap, a single consolidation cycle under highly correlated input could grow the basis by O(n) dimensions simultaneously, causing catastrophic renormalization collapse.

**Adaptive threshold under varying input.** The threshold θ_merge = α · max(ρ̄, ρ_min) scales with background correlation. Under random input (ρ̄ ≈ 0), it defaults to α · ρ_min = 0.3. Under structured input with high shared correlation (ρ̄ → 1), the threshold rises to α · 1 = 3.0, requiring near-perfect correlation to trigger a merge. This is the intended behavior: the system becomes more selective about merging as its dimensions become more broadly correlated.

### S6.3 Consolidation–Anticipatory Operator Coupling (Open)

The relationship between prediction error from the anticipatory operator and consolidation dynamics is not yet fully specified. Three candidate mechanisms exist:

1. **Timing trigger.** Accumulated prediction error exceeding a threshold triggers an immediate consolidation cycle outside the normal T_cons schedule. High surprise accelerates structural reorganization.
2. **Learning rate modulation.** Prediction error magnitude scales λ: high error → faster C update → faster co-activation detection. This ties the consolidation timescale to the system's learning state.
3. **Threshold modulation.** Systematic prediction error at specific dimensions lowers θ_merge locally, making those dimensions more likely to participate in merges. This directly couples novelty detection to basis growth.

These mechanisms are not mutually exclusive and may operate at different timescales. Formalizing and computationally testing this coupling is identified as a priority for future work.

### S6.4 High-n Scaling

The consolidation operator has been verified at n = 32. Three scaling concerns require investigation:

**Eigendecomposition cost.** O(n³) per cycle. At n = 768, this is approximately 1.4 × 10⁸ operations per cycle—expensive but not prohibitive for offline consolidation. The Hebbian variant reduces this to O(n²) for pairwise trace maintenance, which is tractable at n = 768. For n ≫ 10³, sparse approximations or hierarchical merging strategies would be needed.

**Adaptive threshold at high n.** As n grows, ρ̄ decreases (correlations distribute across more pairs). The floor ρ_min becomes the binding constraint rather than α · ρ̄. This means high-n systems effectively use a fixed absolute threshold, which may not scale appropriately. An alternative at high n: threshold relative to the distribution of singular values of C, which is more robust to dimensionality effects.

**Renormalization coupling.** At low n, adding one new dimension reduces all existing dimensions noticeably. At high n, the reduction per existing dimension is 1/n and is negligible. This means β can be larger at high n without destabilizing the tape. The parameter β should therefore scale as β ~ 1/√n to maintain constant norm injection per new dimension relative to the existing basis size.
## S7. Fracture: Formal Specification and Computational Conditions
### S7.1 Single-Dimension Fracture Criterion

Consider dimension i before a reception event. Its state is sᵢ = rᵢ eⁱᶠⁱ with magnitude rᵢ > 0. The update step produces the unnormalized component:

s̃ᵢ = sᵢ + η cᵢ

where cᵢ = vᵢ · sᵢ. The squared magnitude of the unnormalized result is:

|s̃ᵢ|² = rᵢ² + pᵢ² + 2rᵢpᵢ cos(ψᵢ)

where pᵢ = η|cᵢ| is the perturbation strength and ψᵢ = arg(cᵢ) − arg(sᵢ) is the relative phase between the perturbation and the current state. After renormalization, the component's magnitude is |s̃ᵢ| / ‖s̃‖.

**Definition (Fracture Event).** Dimension i undergoes a fracture event at step t if:

|s̃ᵢ|² / ‖s̃‖² < θ_frac²

where θ_frac is a small fracture threshold (suggested: θ_frac = 0.5 · γ/√n, half the slow-pruning threshold, reflecting that fracture is more severe than proximity to the pruning boundary). Unlike slow pruning—which requires K consecutive consolidation cycles below γ/√n—a fracture event occurs within a single reception step, before consolidation can intervene.

**Key geometric observations:**

- When ψᵢ ≈ π (oppositional torque, cos(ψᵢ) = −1), |s̃ᵢ|² = (rᵢ − pᵢ)². Fracture occurs when pᵢ ≈ rᵢ (the perturbation cancels the component) or pᵢ ≫ rᵢ (overshoot). This is the most dangerous configuration: a perturbation aligned against the dimension drives the magnitude toward zero.

- When ψᵢ ≈ π/2 (pure torque, cos(ψᵢ) = 0), |s̃ᵢ|² = rᵢ² + pᵢ². The magnitude cannot decrease—pure orthogonal torque rotates the dimension without shrinking it. Fracture cannot occur under pure torque alone; it requires an oppositional component.

- When ψᵢ ≈ 0 (resonance, cos(ψᵢ) = 1), |s̃ᵢ|² = (rᵢ + pᵢ)². The magnitude grows. No fracture risk.

- Emergent rigidity protects hard axes: for large rᵢ, even strong pᵢ produces small angular displacement. Fracture is predominantly a risk for soft axes (small rᵢ) receiving strong oppositional torque.

### S7.2 Outcome Phase Diagram

For a single dimension, three parameter regions determine the outcome of a torque event:

| Region | Condition | Outcome |
|---|---|---|
| Rejection | rᵢ ≫ pᵢ | Negligible angular displacement; magnitude nearly unchanged |
| Accommodation | rᵢ moderate, |s̃ᵢ|/‖s̃‖ ≥ θ_frac | Phase rotation; dimension remains functional |
| Fracture | |s̃ᵢ|²/‖s̃‖² < θ_frac² | Dimension knocked below threshold in one step |

The accommodation–fracture boundary is not fixed in (rᵢ, pᵢ) space but depends on the global norm ‖s̃‖ and the phase ψᵢ. A system with many simultaneously active dimensions (high PR) will have a larger ‖s̃‖, making fracture less likely for any given dimension—distributed activity provides collective protection. A system in near-capture (PR ≈ 1) has ‖s̃‖ ≈ |s̃_dominant|, meaning soft dimensions are especially vulnerable because the norm denominator is anchored at the dominant dimension's magnitude.

### S7.3 Correlated Fracture and Conjunctive Cluster Dissolution

When a perturbation simultaneously affects multiple dimensions belonging to a conjunctive cluster, the consequences exceed the sum of individual fractures.

Let D = {i₁, i₂, ..., iₖ} be a conjunctive cluster—a set of dimensions seeded together by the consolidation operator because their cross-correlation eigenvalue exceeded θ_merge. The cluster's cohesion is maintained by their mutual phase relationships, encoded in the correlation matrix C.

A **correlated fracture event** occurs when a subset D' ⊂ D simultaneously falls below the fracture threshold. The remaining dimensions in D continue to exist, but their phase relationships become unanchored: the co-activation pattern that generated the conjunctive dimension no longer has enough active participants to sustain its characteristic eigenvalue. At the next consolidation cycle:

1. The correlation matrix C is updated using only surviving dimensions.
2. The cluster eigenvalue may drop below θ_merge = α · max(ρ̄, ρ_min).
3. If it does, the consolidation operator does not re-merge the survivors—the structural condition for the conjunctive dimension no longer holds.

The result is conjunctive dimension dissolution: the conjunctive axis is removed from the basis, and the surviving individual dimensions continue independently, carrying phase memory of the former conjunction but lacking the dedicated axis that encoded it.

**Asymmetry with re-exposure.** After dissolution, re-exposing the system to the original co-activating stimulus does not immediately rebuild the conjunctive dimension. The system must accumulate sufficient co-activation statistics in C to again reach θ_merge. If the stimulus required a specific developmental sequence (e.g., temporal ordering, attentional state) to generate the original conjunctive correlation, simple re-exposure without that sequence may never rebuild the conjunction—the system is stuck without the structural prerequisite for recognizing the pattern as a conjunction. This is the formal mechanism underlying the clinical observation that some traumatic disruptions of integrated memory cannot be remediated by standard re-exposure paradigms.

### S7.4 Fracture Parameter and Computational Verification Sketch

**The fracture threshold θ_frac.** Suggested value: θ_frac = 0.5 · γ/√n. Setting it at half the pruning threshold ensures a gap between "fracture zone" and "slow pruning zone": dimensions that drift below θ_frac in normal operation (without a catastrophic event) are caught by slow pruning, while dimensions that cross θ_frac in a single step constitute genuine fractures. The appropriate value depends on the input distribution and η; sensitivity analysis analogous to S6.2 is required.

**Suggested computational test (T6.1—not yet implemented).** Initialize a system with a soft axis (rᵢ = 0.05) and a hard axis (rⱼ = 0.8). Apply a strong oppositional perturbation (pᵢ = pⱼ = 0.15, ψᵢ = ψⱼ = π). Expected outcomes: the soft axis fractures (|s̃ᵢ|/‖s̃‖ drops below θ_frac); the hard axis accommodates with a small phase rotation but remains far above threshold. A second phase: build a conjunctive cluster (dims 2, 3, 4 co-activated for 500 steps), then apply simultaneous oppositional torque to dims 2 and 3. Expected: correlated fracture triggers cluster eigenvalue drop at next consolidation cycle; conjunctive dimension dissolves. Compare to the slow-pruning baseline: same system, same dims, but small sustained orthogonal input rather than a large shock—dims 2 and 3 eventually pruned across K cycles rather than immediately fractured.
