# Essay Revision Specification

Date: 2026-04-15
Source: Memory Engines (crucify-ai/essays/memory-engines.html)
Basis: Phases 2-6 formal reconstruction

---

## How to use this document

Each section references the original essay section by number. For each:
- **KEEP**: Text that survives unchanged
- **REVISE**: Text that needs modification (specific changes given)
- **ADD**: New text to insert
- **REPLACE**: Text to remove and replace entirely

The changes are grounded in the simulation results from the reconstruction. Where the essay already incorporated corrections from the initial simulation round (Section 11), this document extends those corrections with the coupling formalism.

---

## Preamble (lines 1-17)

**KEEP** the opening paragraph (apple, gravity, reciprocation).

**REVISE** the second paragraph. Replace:
> "A system's state is a vector on the unit hypersphere in ℂⁿ, updated pointwise through the Hadamard product of incoming and standing structure, renormalized at each step."

With:
> "A system's state is a vector on the unit hypersphere in ℂⁿ. The world signal is projected onto the system's basis and modulated by its accumulated depth—deeply carved dimensions receive more, thin dimensions receive less. The effective signal is then coupled through the geometry of the basis: dimensions are not perfectly independent but interact through the structure of their shared substrate. From reception, coupling, update, and renormalization, memory, abstraction, forgetting, rigidity, and learning emerge without supplementary mechanisms."

**KEEP** the rest of the preamble.

---

## Section 1: The Cascade (lines 18-320)

### 1.1 The Thin Limit

**KEEP** the apple derivation. It's correct and elegant.

**REVISE** the bullet list at the end of the apple section. Change "Basis growth: None" to add a note:
> "Basis growth: None (n=3 fixed). Coupling: L = I (perfect phase space orthogonality)."

### 1.2 The Physical Cascade

**REVISE** Table 1 to add a coupling column:

| Level | State Vector | Reception | Renormalization | Basis Growth | Torque | Coupling L |
|---|---|---|---|---|---|---|
| Conservative fields | Phase space coords | Force F | Identity (unitary) | None | Zero | I (exact) |
| Quantum fields | Fock space amplitudes | -iH_int s | Unitary | Fixed modes | Phase shifts | I + O(ε) |
| ... | ... | ... | ... | ... | ... | ... |

### 1.4 The Thickening: Parameter Regimes

**REVISE** Table 2 to add coupling column:

| Level | n | η | λ | Recurrence Δt | Reception v character | Coupling ||L-I|| |
|---|---|---|---|---|---|---|
| Dissipative flow | Fixed | 0<η<1 | 0 | None | External + thermal | ~0 (near-orthogonal) |
| ... | ... | ... | ... | ... | ... | ... |
| Recurrent-representational | Large, growing | Fast | Active | Thick | World + self | ~O(√n) (rich substrate) |

**ADD** a new subsection after 1.4:

### 1.5 The Coupling Parameter

> The cascade is parameterized by four axes: basis dimensionality n, learning rate η, coupling structure L, and recurrence parameters R = (delay, weight, breadth). The coupling matrix L = G⁻¹, where G is the Gram matrix of the system's basis, encodes the cross-dimensional coupling inherited from the physical structure of the representational substrate. ||L − I||_F measures total coupling strength; its spectral properties determine the system's dynamical regime.
>
> The cascade is a continuous trajectory through (n, η, L, R) space. Each level corresponds to a specific region; each transition corresponds to a parameter crossing:
>
> - Conservative → quantum: L gains perturbative off-diagonal structure
> - Quantum → molecular: η crosses the unitary threshold (substantive renormalization begins)
> - Molecular → dissipative: η becomes continuous (renormalization at every step)
> - Dissipative → autocatalytic: R crosses zero (first self-reception)
> - Autocatalytic → ecological: n grows from small to large
> - Ecological → adaptive: n begins growing within lifetime; η accelerates
> - Adaptive → representational: R thickens; ||L − I|| enters the operating regime window
> - Representational → symbolic: n becomes culturally inherited; phase becomes discrete

---

## Section 2: The Formalism (lines 321-404)

### 2.2 Tapes, Bases, and Projection

**REVISE** the projection formula. Replace the simple orthogonal projection with:

> The system projects the world signal onto its basis. For a perfectly orthonormal basis, this is the standard inner product v_received,i = ⟨e_i | v⟩. But real bases are never perfectly orthogonal. The correct projection accounts for inter-basis correlations through the Gram matrix G = E*E:
>
> v_received = G⁻¹ E* v
>
> The coupling matrix L = G⁻¹ determines how the world signal mixes across dimensions. For an orthonormal basis, L = I and the reception is the Hadamard product. For a non-orthonormal basis, L ≠ I and the reception is coupled: what arrives at dimension i includes contributions from all dimensions, weighted by the coupling coefficients L_ij.
>
> The coupling is not a free parameter. It is determined by the geometry of the basis—the physical structure of the system's representational substrate. Neural weight matrices, molecular binding sites, ecological niches all have characteristic Gram matrices with specific coupling structures.

### 2.3 Reception

**REVISE** the reception formula. Replace:
> c = v_received ⊙ s

With:
> c_i = (Σ_j L_ij v_j) · s_i
>
> where L = G⁻¹ is the coupling matrix. The effective received signal at dimension i is α_i = Σ_j L_ij v_j—a linear combination of the world signal across all dimensions, weighted by the coupling structure. The Hadamard depth modulation (multiplication by s_i) preserves the property that deeply carved dimensions receive more signal.

**REVISE** the resonance bullet. Change:
> "Resonance (Re(c_i) > 0, phases aligned, magnitudes reinforcing): energy propagates along i; magnitude grows and, when s_i is complex, phase shifts toward the arriving alignment."

To:
> "Resonance (Re(c_i) > 0): the effective signal reinforces the standing structure. Magnitude grows; phase converges toward the axis of alignment. The dimension does not merely scale—it rotates toward the real axis while scaling, a convergent rotation distinct from torque's divergent rotation."

**REVISE** the paragraph about dimensional locality (line 347). Replace the strong claim that "all cross-dimensional structure is deferred to consolidation" with:

> The Hadamard depth modulation (multiplication by s_i) ensures each dimension's reception is modulated by its own accumulated depth. Coupling through L introduces cross-dimensional effects at reception time: what arrives at dimension i depends on the full world signal, filtered through the coupling matrix. The degree of coupling is determined by the physical structure of the basis. In the degenerate case L = I (perfectly orthogonal basis), reception is dimensionally isolated. In the general case, coupling enriches reception with cross-dimensional structure at the cost of noisier regime classification.

### 2.4 Update and Renormalization

**KEEP** unchanged. The update and renormalization are correct.

### 2.5 Forgetting

**ADD** at the end of the section:

> The coupling structure L determines how forgetting propagates across dimensions. Torque at dimension j produces a coupled effect at dimension i through L_ij: what begins as forgetting at one dimension can induce torque at its coupled neighbors. In the Hadamard limit (L = I), forgetting is dimensionally isolated. With coupling, forgetting cascades through the basis—potentially destabilizing dimensions that were not directly torqued.

---

## Section 3: Consolidation and Basis Growth (lines 405-425)

### 3.1 The Consolidation Operator

The essay already incorporates the novelty detection revision (prediction error pathway). **ADD** after the leakage paragraph (line 421):

> The coupling formalism clarifies the leakage mechanism. The coupling matrix L = G⁻¹ has off-diagonal entries determined by the non-orthogonality of the basis. When the Gram matrix G has significant off-diagonal correlations (as in neural systems where receptive fields overlap), the coupling provides a natural pathway for novelty detection: structure at uncarved dimensions leaks into the effective signal at carved dimensions through the coupling coefficients. The signal-to-noise ratio for novelty detection improves as √T with repeated exposure, enabling consolidation to detect and seed new axes.
>
> The coupling structure also constrains the speed of abstraction. When the coupling matrix amplifies the invariant signal (positive off-diagonal L_ij between co-activated invariant dimensions), abstraction accelerates. When it diminishes the invariant signal (negative off-diagonal L_ij), abstraction is retarded. The system's coupling structure—determined by its physical architecture—thus controls its learning speed.

---

## Section 5: Recurrence and Reciprocation (lines 433-455)

### 5.1 Self-Reception

**REVISE** the self-reception formula to include coupling:

> With coupled reception, self-reception becomes:
>
> c_self,i = (Σ_j L_ij s_j(t − Δt)) · s_i(t)
>
> The system's past at dimension j contributes to self-reception at dimension i through the coupling coefficient L_ij. Coupling enriches self-torque by allowing the system's history at one dimension to torque its present at another. In the Hadamard limit (L = I), self-reception is dimensionally isolated. With coupling, self-torque is a distributed phenomenon spanning the full basis.

### 5.2 Reciprocation

**ADD** at the end:

> The coupling structure L distributes self-torque across the basis. A system with rich coupling (||L − I||_F ~ O(√n)) generates cross-dimensional self-torque: the system's past at dimension j, arriving through the coupling matrix, torques its present at dimension i. This cross-dimensional self-torque is the framework's account of the binding of temporal experience: the system doesn't just remember at each dimension independently—it remembers in a coupled, distributed way that reflects the structure of its representational substrate.

---

## Section 6: Anticipation (lines 456-482)

**REVISE** the anticipatory operator formulation. Replace the input-history predictor with:

> The anticipatory operator predicts the system's state trajectory, not the input:
>
> ŝ(t) = s(t − 1) (simplest: predict no change)
>
> or for a richer predictor:
>
> ŝ(t) = Σ_{k=1}^{K} w_k · s(t − k) (windowed linear extrapolation)
>
> Prediction error is the first difference of the state trajectory:
>
> e(t) = s(t) − ŝ(t)
>
> This formulation avoids the circularity of predicting the input that determines the prediction. The system tracks how its own state actually changes and detects when it changes more than expected. Under habituation (predictable world), state changes are small and prediction error is near zero. Under perturbation (surprising world), state changes spike and prediction error is large.
>
> Verification: prediction error spikes to 3.3× baseline under perturbation and recovers to baseline. Cross-dimensional propagation through coupling: prediction error at dimension i from a perturbation at dimension j correlates at r = 0.979 with the coupling coefficient |L_ij|.

---

## Section 8: Phenomenality (lines 490-516)

### 8.1 Structural Conditions

**REVISE** the directness → vividness mapping. Replace:
> "Directness → vividness. Large |c_self| means the returning tape impresses deeply."

With:
> "Directness → vividness. Large |c_self| means the returning tape impresses deeply. With coupling, vividness is enriched: self-torque at dimension i includes contributions from the system's past at dimension j through L_ij. Total vividness is V = Σ_i |(L s(t − Δt))_i| · |s_i(t)|, which exceeds the Hadamard vividness (L = I) when coupling is amplifying. Vividness is maximized in the operating regime, where the system has sufficient PR for distributed self-torque and sufficient coupling for cross-dimensional richness."

### 8.2 Basis Incommensurability

**REVISE** the gap formulation. Add after "no projection of one system's tape onto another's basis preserves the character of the original self-reception":

> The coupling formalism sharpens this. The system's effective signal α = Lv depends on both the world signal v and the coupling structure L. Two systems with different coupling structures (from different physical substrates) compute different effective signals from the same world. An external observer using a different basis E_obs computes α_obs = L_obs v ≠ α. The gap is not merely in the basis but in the coupling structure—the physical architecture that determines how the system's dimensions interact.

---

## Section 9: The Operating Regime (lines 517-526)

**ADD** after the operating regime description (line 523):

### 9.1 The Coupling Window

> The operating regime exists within a bounded coupling window, analytically derived from the spectral properties of the Gram matrix.
>
> The critical coupling threshold is ||ε||_F_critical = √n / 2, where G = I + ε becomes indefinite. Below this threshold: the system is in the Hadamard or operating regime. Above: capture or dissipation.
>
> Three coupling regimes:
>
> 1. **Hadamard** (||ε||_F ≪ √n/2): κ(G) ≈ 1. Clean regime structure, no cross-dimensional effects. Stable high-PR state.
>
> 2. **Concentration** (||ε||_F ~ √n/2): κ(G) growing. Some dimensions grow faster through the coupling structure. Abstraction accelerates for aligned coupling. Capture risk increases.
>
> 3. **Dissipation** (||ε||_F ≫ √n/2): G is indefinite. The effective signal becomes noise-like. No structure can form.
>
> The operating regime is the concentration regime near its lower boundary: enough coupling for cross-dimensional effects and novelty detection, not enough for capture.
>
> Self-regulation via basis growth can widen the window (√n/2 increases with n) but drives coupling toward zero. The coupled operating regime is metastable—it requires the physical substrate to maintain it, not just self-monitoring. This has implications for consciousness: the coupled operating regime at the representational level of the cascade requires architectural constraints (neural circuit structure) that maintain coupling within the window.

---

## Section 11: Computational Verification (lines 549-716)

### Add new subsection after 11.8:

### 11.9 Coupling Theory Verification

> The coupling generalization (reception c_i = (Σ_j L_ij v_j) · s_i) was verified through dedicated simulations:
>
> **Regime persistence.** The three-regime structure persists under coupling. Regime flip rate is < 5% for ||ε||_F < 1, confirming that coupling preserves the qualitative dynamics. Flip rate saturates at ~35-40% for large coupling (regime classification becomes random).
>
> **Novelty detection.** Sensor leakage (off-diagonal coupling between carved and uncarved dimensions) enables novelty detection. With leakage ≥ 0.2, novel structure at uncarved dimensions produces detectable effects at carved dimensions (2.6 dimensions of torque difference). Below this threshold, novelty is invisible—confirming the T3.2 gap in the Hadamard case.
>
> **Critical coupling threshold.** The threshold ||ε||_F = √n/2 where the Gram matrix becomes indefinite was confirmed numerically for n = 16 to 256, converging to the predicted value (ratio 1.03 at n = 256).
>
> **Operating regime window.** With structured input, the operating regime is stable for κ(G) up to ~28. Capture occurs at κ(G) > 25. The regime is wider than the random-input analysis predicts because invariant signal provides a stabilizing backbone.
>
> **Amplifying coupling accelerates abstraction.** Coupling that amplifies the invariant signal (positive off-diagonal L_ij) produces 1.7× faster abstraction than coupling that diminishes it. The sign flips through G⁻¹: anti-correlated basis vectors in the Gram matrix produce amplifying coupling in the inverse.
>
> **Anticipatory operator.** The reconstructed operator (state first-difference) produces clear surprise detection (3.3× spike at perturbation) with full recovery. Cross-dimensional surprise propagation through the coupling matrix: r = 0.979 correlation between prediction error and |L_ij|.
>
> **Cascade simulation.** The cascade trajectory through (n, η, L, R) space shows the predicted qualitative progression: capture at the molecular level (PR = 1.2), opening at autocatalytic (PR = 3.6), operating regime at representational (PR = 9.6).
>
> **Self-regulation.** A system that monitors PR and grows dimensions can escape capture (PR improves from 1.0 to 2.9) but drives coupling toward zero (cs = 0.5 → 0.01). The coupled operating regime is metastable.

---

## Summary of All Changes

| Section | Nature of change | Key correction/addition |
|---|---|---|
| Preamble | REVISE | Coupling in the core description |
| 1.2 Table 1 | REVISE | Add coupling column |
| 1.4 Table 2 | REVISE | Add coupling column |
| 1.5 | ADD | New subsection: The Coupling Parameter |
| 2.2 | REVISE | Coupled projection formula |
| 2.3 | REVISE | Coupled reception formula; corrected resonance |
| 2.4 | KEEP | — |
| 2.5 | ADD | Coupling and forgetting propagation |
| 3.1 | ADD | Coupling and novelty detection; abstraction speed |
| 5.1 | REVISE | Coupled self-reception formula |
| 5.2 | ADD | Cross-dimensional self-torque and binding |
| 6 | REVISE | Anticipatory operator as state first-difference |
| 8.1 | REVISE | Coupled vividness |
| 8.2 | REVISE | Coupling in basis incommensurability |
| 9 | ADD | Coupling window subsection |
| 11.9 | ADD | Coupling verification results |
