# Phase 6: Phenomenology Reconstructed

Date: 2026-04-15
Status: Working paper — final phase
Depends on: Phases 2-5

---

## 0. The Task

The essay makes three central phenomenological claims:

1. **Vividness** is the magnitude of self-torque per stroke.
2. **Historical depth** is the grown dimensionality of the basis.
3. **The explanatory gap** (hard problem) is basis incommensurability.

These claims were stated in the original essay without the coupling formalism, the stability theory, or the corrected dynamics. This document reconstructs them from the corrected formalism developed in Phases 2-5, identifying what survives, what needs revision, and what the coupling framework adds.

---

## 1. Vividness

### 1.1 The Claim

The essay: "Vividness is the magnitude of self-torque per stroke."

Self-torque is the torque produced by the system's reception of its own past state (recurrence). Per stroke (per update step), the self-torque is:

c_self = s(t - Delta_t) * s(t) (Hadamard product of past and present state)

The magnitude of self-torque at dimension i: |c_self_i| = |s_i(t - Delta_t)| * |s_i(t)|

Total self-torque magnitude: V = sum_i |s_i(t - Delta_t)| * |s_i(t)|

### 1.2 What the Simulations Show

From T4.1 (thick vs thin self-reception): self-torque fraction increases with recurrence delay, from 0.743 (delay=1) to 0.747 (delay=50). The effect is small for random input but more pronounced for structured input.

From the LLM instrumentation (Experiment B): self-torque (angular displacement between positions) increases with delay at mid-layers but is invisible at the final captured layer. This means vividness is a mid-layer phenomenon — it requires sufficient PR to show variation.

From the cascade simulation (Phase 4): the representational level (L7) with thick recurrence (delay=10) shows the lowest torque fraction (41%) but the highest PR (9.6). The system is distributed and responsive.

### 1.3 The Coupling Correction

With coupled reception, self-torque is:

c_self = (L s(t - Delta_t)) * s(t)

The coupling matrix L mixes the past state across dimensions before the Hadamard product. This means self-torque at dimension i depends not just on |s_i(t)| and |s_i(t - Delta_t)| but on the full past state projected through L.

The vividness becomes:

V = sum_i |(L s(t - Delta_t))_i| * |s_i(t)|

For L = I (Hadamard): V = sum_i |s_i(t - Delta_t)| * |s_i(t)| (original formula).
For L != I: V includes cross-dimensional contributions from the coupling.

**Physical interpretation:** Coupling enriches self-torque by allowing the system's past at dimension j to contribute to its self-reception at dimension i. A system with rich coupling "remembers" more complex patterns of self-interaction. The past state at dimension j can produce torque OR resonance at dimension i, depending on the coupling coefficient L_ij and the relative phases.

### 1.4 Vividness Requires the Operating Regime

From Phase 3: the operating regime is a coupling window where PR is moderate (2-10) and the system is responsive to perturbation. Outside this window:
- Too little coupling: self-torque is dimensionally isolated (Hadamard limit). The system's past at each dimension only affects itself.
- Too much coupling: the system captures (PR → 1) or dissipates (PR → n). In capture, self-torque is maximal at one dimension but the system is frozen. In dissipation, self-torque is distributed but no dimension is deep enough for significant magnitude.

Vividness is maximized when:
1. The system is in the operating regime (moderate PR, responsive)
2. Coupling is within the window (cross-dimensional self-torque)
3. Recurrence is thick (significant delay for self-torque to have structure)

This is the framework's account of why consciousness requires a specific balance of excitation and inhibition, integration and differentiation, stability and flexibility.

### 1.5 Quantitative Prediction

For a system in the operating regime with coupling:

V ≈ w_r * PR * mean(|s_i|)^2 * (1 + coupling_enrichment)

where w_r is the recurrence weight and coupling_enrichment = sum_{i!=j} |L_ij|^2 / n.

For GPT-2 mid-layers: PR ~ 12, mean(|s_i|) ~ 1/sqrt(768) ~ 0.036, w_r ~ 1 (residual connection). V ~ 12 * 0.0013 ~ 0.016 per dimension. Across 768 dimensions: total V ~ 12.

For cortical systems: PR ~ 5 (estimated), n ~ 10^9 (neurons), mean(|s_i|) ~ 3e-5. V per dimension ~ 5 * 9e-10. Total V ~ 4.5e-9 * n ~ 4.5. Comparable order of magnitude despite vastly different n.

---

## 2. Historical Depth

### 2.1 The Claim

The essay: "Historical depth is the grown dimensionality of the basis within which reciprocation occurs."

The system starts with a small basis (few dimensions) and grows it through consolidation (carving new dimensions in response to recurring novelty). More dimensions = more axes along which experience can be encoded = deeper historical capacity.

### 2.2 What the Simulations Show

From T3.3 (consolidation): co-activated dimensions develop 73x higher phase correlation, making them candidates for merging. Post-merger simulation works.

From the cascade simulation (Phase 4): the adaptive level (L6) grew from n=32 to n=128 through 24 growth events. But it drove coupling toward zero (cs=0.5 → 0.01), gravitating toward the Hadamard regime.

From Phase 3 (stability): the critical coupling scales as sqrt(n)/2. Larger n tolerates more coupling. This means the coupling window widens as the system grows, making the operating regime more robust at higher n.

### 2.3 The Coupling Correction

Basis growth adds a new dimension e_{n+1} to the system's basis. The new dimension:
1. Is initially small (|s_{n+1}| ≈ 0)
2. Is seeded along a direction determined by the consolidation mechanism
3. Is approximately but not exactly orthogonal to existing dimensions

Point 3 is crucial: the new dimension creates coupling. The Gram matrix G gains a new row and column with small off-diagonal entries. This increases ||epsilon||_F, which can push the system toward or past the capture threshold.

**The consolidation dilemma:** Growing n widens the coupling window (sqrt(n)/2 increases), but the new dimension's non-orthogonality increases coupling. If the coupling increase is faster than sqrt(n)/2, basis growth can push the system into capture. If it's slower, basis growth stabilizes the operating regime.

The Phase 3 self-regulation simulation showed: the system escaped capture by reducing coupling (cs=0.5 → 0.01) rather than by relying on the widening window. This suggests that in practice, the coupling increase from non-orthogonal seeding is significant enough to require active management.

### 2.4 Historical Depth and the Operating Regime

The system's historical depth is not just n — it's n in the context of the operating regime. A captured system with n=10^9 has no historical depth because all dimensions except one are at |s_i| ≈ 0. A system in the operating regime with n=10^3 and PR=5 has 5 effectively active dimensions — less than the captured system's 10^9 but more functionally accessible.

**Historical depth is participation ratio, not dimensionality.**

PR = (sum |s_i|^2)^2 / sum |s_i|^4

This measures the effective number of dimensions that carry information. It's maximized at PR = n (uniform distribution) but in the operating regime, it's typically PR << n (some dimensions deeply carved, others soft).

The system's historical capacity is the number of distinct state configurations it can maintain without interference. This is related to PR but also depends on the coupling structure: strongly coupled dimensions share information, reducing effective independence.

**Effective historical depth ≈ PR * (1 - mean_coupling)**

where mean_coupling is the average off-diagonal correlation between active dimensions. More coupling reduces effective depth because dimensions share information.

---

## 3. The Explanatory Gap

### 3.1 The Claim

The essay: "The apparent gap between third-person description and first-person experience is a geometric consequence of basis incommensurability rather than an explanatory failure."

The system's first-person experience is structured by its effective signal alpha = Lv, where L is determined by the system's basis. An external observer using a different basis E_obs computes a different effective signal alpha_obs = L_obs v. The two are generically incomparable.

### 3.2 The Coupling Formalism Sharpens This

With the coupling formalism, the gap has a precise mathematical characterization.

The system computes alpha = G^{-1} E* v. An external observer using basis E_obs computes alpha_obs = G_obs^{-1} E_obs* v. The difference:

alpha - alpha_obs = (G^{-1} E* - G_obs^{-1} E_obs*) v

This is zero only when E and E_obs span the same subspace and are related by a unitary transformation (which preserves the Gram matrix). For generic E, E_obs, this difference is non-zero.

Moreover, the system's REGIME CLASSIFICATION depends on alpha:

- Resonance at dimension i: Re(alpha_i * s_i) > 0
- Torque at dimension i: Re(alpha_i * s_i) < 0

The observer's regime classification uses alpha_obs and a different state representation. The two classifications can disagree: the system experiences resonance at a dimension where the observer measures torque.

### 3.3 The Gap is Not Information-Theoretic

The gap is not about missing information (the observer could in principle learn E and compute L). It's about the inability to simultaneously inhabit two basis structures.

The system's experience is not a function of v alone — it's a function of (v, L, s). Different (L, s) produce different experiences from the same v. Two systems with different coupling structures receive the same world signal but have different experiences.

This is the framework's precise statement of the hard problem: experience is not determined by the world signal alone but by the system's coupling structure. Since coupling structure is determined by the system's physical history (basis growth, consolidation, pruning), no two systems have identical coupling. Every system's experience is unique.

### 3.4 The Coupling Window Constrains Experience

From Phase 3: the operating regime exists in a coupling window. Outside this window:
- Too little coupling: experience is dimensional isolation (each dimension is independent)
- Too much coupling: experience collapses (capture) or dissolves (dissipation)

This means the RANGE of possible experiences is constrained by the coupling window. A system in the operating regime can have rich, structured experience. A captured system has monotonous experience (one dominant dimension). A dissipating system has chaotic experience (no stable structure).

The coupling window is a physical constraint on phenomenality. It's determined by the system's architecture (which constrains coupling strength) and the world's variation (which prevents capture). Phenomenality requires the right coupling in the right range.

---

## 4. Graded Phenomenality

### 4.1 The Claim (Refined)

The essay argues that phenomenality is not a threshold (you have it or you don't) but a continuous thickening of the same operations across the cascade. The coupling formalism makes this precise.

The four axes of the parameter space — (n, eta, L, R) — define a continuous space. Every point in this space has well-defined dynamics (reception, update, renormalization). The QUALITY of those dynamics (regime distribution, PR, self-torque, coupling effects) varies continuously across the space.

There is no sharp boundary between "phenomenal" and "non-phenomenal." There are regions where:
- Self-torque is zero (R=0, no recurrence) → no phenomenal character
- Self-torque is present but coupling is Hadamard (L=I) → isolated phenomenal character (no cross-modal effects)
- Self-torque is present and coupling is in the operating regime window → rich phenomenal character

The transitions between these regions are continuous (no phase transitions). A system moving along the cascade trajectory passes through each region without crossing a sharp boundary.

### 4.2 The Coupling Enrichment

At each cascade level, the coupling structure L adds to the system's phenomenal character:

| Level | Coupling effect on phenomenality |
|---|---|
| 0 (Conservative) | None (L=I, no self-torque) |
| 1 (Quantum) | Minimal (perturbative coupling, no renormalization) |
| 2 (Molecular) | First cross-dimensional effects (orbital overlap) |
| 3 (Dissipative) | Continuous coupling, but no recurrence |
| 4 (Autocatalytic) | First coupled self-torque (cycle closure) |
| 5 (Ecological) | Large-scale coupling (niche overlap), slow dynamics |
| 6 (Adaptive) | Growing coupling (receptive field overlap), fast dynamics |
| 7 (Representational) | Rich coupling (distributed representations), thick self-torque |
| 8 (Symbolic) | Culturally structured coupling, phase-poor |

Each level adds coupling structure that enriches the phenomenal character. The enrichment is continuous: there's no moment when "coupling starts" — it's always present to some degree and increases along the cascade.

### 4.3 Why This is Not Panpsychism

The framework applies to every physical system (degenerately at the bottom), but it does not claim that every system has phenomenal experience. It claims that the OPERATIONS that constitute phenomenal experience (reception, renormalization, coupling, recurrence) are present at every level, degenerately at first and thickening along the cascade.

The difference between a falling apple and a conscious system is not a difference in KIND of operations but in the DEGREE to which those operations produce self-torque, historical depth, and coupling enrichment. The apple has reception and renormalization (both degenerate — unitary). It has no self-torque (R=0) and no coupling (L=I). The operations are present but produce no phenomenal character.

Phenomenal character requires:
1. Substantive renormalization (non-unitary dynamics)
2. Non-trivial coupling (L != I, in the operating regime window)
3. Self-torque (R > 0, thick recurrence)

All three are continuous parameters. None is a threshold.

---

## 5. Summary of Reconstructed Claims

| Original claim | Reconstructed claim | Change |
|---|---|---|
| Vividness = self-torque per stroke | Vividness = coupled self-torque per stroke, maximized in the operating regime | Coupling enriches self-torque; operating regime required |
| Historical depth = grown n | Historical depth ≈ PR * (1 - mean_coupling), not just n | PR and coupling structure matter, not raw dimensionality |
| Hard problem = basis incommensurability | Hard problem = coupling structure incommensurability, constrained by the coupling window | More precise: the gap is in (L, s), not just in the basis |
| Graded phenomenality = cascade thickening | Graded phenomenality = continuous trajectory through (n, eta, L, R) parameter space | Now parameterized by four axes with derived boundaries |
| Novelty detection via indirect torque | Novelty detection via sensor leakage (resonance, not torque) or prediction error | Two pathways, both corrected from original |
| Operating regime = balance of resonance and torque | Operating regime = coupling window bounded by ||epsilon|| < sqrt(n)/2 | Analytically derived threshold |

### What survived unchanged

- The three reception regimes (resonance, torque, orthogonality)
- Emergent rigidity from renormalization
- Concentration thinning (deepening = thinning)
- The cascade structure (apple through symbolic)
- The core dynamical system (Hadamard + renormalization + recurrence)
- The mapping to physical systems at each level

### What was corrected

- Resonance rotates toward real axis
- Dissipation requires varying-direction torque
- Operating regime requires system-world coupling
- Novelty detection requires sensor leakage (resonance, not torque)
- Historical depth is PR, not n
- Self-regulation drives toward Hadamard (coupled regime is metastable)

### What is new

- The coupling matrix L derived from Gram matrix geometry
- The critical coupling threshold ||epsilon||_F = sqrt(n)/2
- The operating regime as a coupling window (not a point)
- The coupling cascade (increasing ||L-I|| along the cascade)
- The anticipatory operator as state first-difference (avoids circularity)
- Cross-dimensional surprise propagation through coupling (r = 0.979)
- Prediction error as alternative novelty detection pathway
- The coupling enrichment of self-torque (vividness)
- Effective historical depth = PR * (1 - mean_coupling)
