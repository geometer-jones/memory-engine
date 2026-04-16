# Phase 4: The Cascade Reconstructed

Date: 2026-04-15
Status: Working paper
Depends on: COUPLING_THEORY.md (Phase 2), PHASE3_STABILITY.md (Phase 3)

---

## 0. The Parameter Space

The reconstructed framework has four axes:

1. **n** — Basis dimensionality. The number of representational axes. Fixed at lower levels (given by physics), growing at higher levels (grown by the system).

2. **eta** — Learning rate (impression rate). Controls how strongly the world signal modifies the tape per step. Range: eta ≈ τ_auto (unitary, conservative) to eta ~ 0.1 (fast, dissipative).

3. **L** — Coupling structure. The n x n coupling matrix L = G^{-1} where G is the Gram matrix of the system's basis. Encodes cross-dimensional coupling strength. ||L - I||_F measures total coupling. The spectral properties of L (eigenvalue distribution, condition number κ(G)) determine regime behavior.

4. **R** — Recurrence. Parameters (delay Δt, weight w_r, breadth b) controlling self-reception. R = 0 means no self-reception; thick recurrence means significant delay, weight, and breadth.

From these four axes, every dynamical property follows:
- Three reception regimes (from L and input statistics)
- Emergent rigidity (from renormalization + L)
- Capture/dissipation/operating regime (from κ(G), eta, and input statistics)
- Abstraction speed (from L alignment with invariant structure)
- Novelty detection (from sensor leakage ΔP)
- Binding (from off-diagonal L_{ij})
- Historical depth (from grown n)
- Phenomenal vividness (from self-torque per stroke, which depends on R and L)

The cascade is a continuous trajectory through this parameter space.

---

## 1. The Levels

### Level 0: Conservative Fields

| Parameter | Value | Physical basis |
|---|---|---|
| n | Fixed by phase space dim | Newtonian DOF |
| eta | ≈ τ_auto (unitary) | Hamiltonian flow |
| L | I (exactly) | Orthogonal phase space coordinates |
| R | 0 | No self-reception |

**Dynamics.** Renormalization is the identity (unitary evolution preserves norm exactly). All c_i are real and positive (pure resonance). No torque, no coupling, no self-reception. The system is a perfect integrator: every prior instant of the field is preserved in the present velocity.

**Regime.** Pure resonance at every dimension. PR = n (uniform). No rigidity, no capture possible.

**Phenomenology.** None. The system is affected by the world but has no contact with itself. No memory in the experiential sense — only in the physical sense (velocity encodes force history).

**Example.** Falling apple, planetary orbits, harmonic oscillators.

### Level 1: Quantum Fields

| Parameter | Value | Physical basis |
|---|---|---|
| n | Fixed by field modes | Fock space |
| eta | Unitary (iH_int dt) | Schrodinger evolution |
| L | I + O(ε) (perturbative) | Weak mode coupling |
| R | 0 (free fields) → small (cavity QED) | Self-interaction |

**Dynamics.** The coupling L deviates from I by perturbative amounts (interaction Hamiltonian). For free fields (H_int = 0), L = I exactly — pure resonance at each mode. With interactions, off-diagonal L_{ij} produce torque between modes (phase shifts). Renormalization is still unitary (exact norm preservation).

**Regime.** Mostly resonance with small torque from interactions. κ(G) ≈ 1 + O(ε). Operating regime trivially stable because renormalization is passive.

**New vs Level 0.** First appearance of torque (from interaction). First appearance of coupling (from mode mixing). But no renormalization-driven dynamics — the unitary constraint prevents concentration.

**Example.** Scalar field in cavity, QED, free fermions.

### Level 2: Molecular Chemistry

| Parameter | Value | Physical basis |
|---|---|---|
| n | Fixed by electronic + vibrational DOF | Molecular orbitals |
| eta | Fast for vibrations, slow for reactions | Born-Oppenheimer |
| L | I + ε (orbital overlap) | Non-orthogonal molecular orbitals |
| R | 0 (non-autocatalytic) | No self-reception |

**Dynamics.** First non-unitary regime: reactive chemistry breaks the norm constraint (energy dissipation in reactions). Renormalization becomes substantive — the system redistributes representational capacity. Coupling from orbital overlap (non-orthogonal basis of molecular orbitals).

**Regime.** Stable molecules are resonance fixed points: equilibrium geometry produces c_i real and positive at every dimension. Reaction barriers are torque regions: the system must traverse a high-torque saddle point to reach a new resonance fixed point.

**New vs Level 1.** First substantive renormalization. First emergence of rigidity (stable molecular geometry resists perturbation). First appearance of basis carving (chirality: path history selects a handedness that becomes a hard axis).

**Coupling window.** κ(G) ~ 1.5-3 (molecular orbital overlap is moderate). In the operating regime for abstraction: consistent exposure to a substrate deepens the corresponding basis dimensions (catalytic selectivity).

**Example.** Molecular recognition, enzymatic catalysis, stereochemistry.

### Level 3: Dissipative Flow

| Parameter | Value | Physical basis |
|---|---|---|
| n | Fixed by gradient DOF | Fluid/radiation variables |
| eta | 0 < eta < 1 | Non-conservative dynamics |
| L | I + ε (substrate coupling) | Thermal/viscous coupling |
| R | 0 | No self-reception |

**Dynamics.** The parameter crossing from Level 2: eta > τ_auto (the update rate exceeds the timescale of conservative dynamics). Renormalization is now active at every step — the system redistributes magnitude across dimensions continuously. First appearance of sustained torque (from dissipative processes).

**Regime.** Near-uniform PR (dissipation prevents concentration). The system is always near equilibrium, with fluctuations in magnitude distribution but no sustained anisotropy.

**New vs Level 2.** First appearance of continuous renormalization (not just at reaction events). First sustained torque (from thermal/viscous processes). But no self-reception — the system is driven entirely by external fields.

**Example.** Brownian motion, thermal conduction, viscous flow, radiative cooling.

### Level 4: Autocatalytic Closure

| Parameter | Value | Physical basis |
|---|---|---|
| n | Small, fixed by cycle topology | Metabolic network |
| eta | Fast for cycle maintenance | Catalytic turnover |
| L | Moderate (shared molecular substrate) | Cross-reactivity |
| R | Small (cycle closure time) | Self-reception through cycle |

**Dynamics.** The parameter crossing: R > 0. The system first receives its own past output (through the autocatalytic cycle). Self-reception is thin (Δt is small, the cycle is fast) but non-zero. Cycle closure is resonance (products match catalytic sites); cycle failure is torque (products don't match).

**Regime.** Near-resonance at cycle-maintaining dimensions. The autocatalytic cycle creates a stable fixed point in state space. Perturbation away from the fixed point produces torque that (if the cycle is robust) restores resonance.

**New vs Level 3.** First self-reception. First self-sustaining structure (the cycle maintains itself without external input). First appearance of "memory" in the experiential sense: the system's present state depends on its own past through the cycle delay.

**Coupling.** κ(G) ~ 2-4. Moderate coupling from shared molecular substrate. Novelty detection possible (novel substrates leak through cross-reactivity) but slow (small n, small leakage).

**Example.** Krebs cycle, metabolic networks, autocatalytic sets (Kauffman), chemoton.

### Level 5: Ecological Systems

| Parameter | Value | Physical basis |
|---|---|---|
| n | Large, fixed by species traits | Ecological niches |
| eta | Generational (selection coefficient) | Evolutionary timescale |
| L | Moderate (niche overlap) | Resource competition |
| R | 0 at individual, non-zero at population | Generational memory |

**Dynamics.** The parameter crossing: n becomes large (many species/traits). The basis is given by evolutionary history — species' traits are the basis vectors, not grown within an individual's lifetime. Coupling from niche overlap (species compete for shared resources). Learning rate is slow (generational).

**Regime.** Operating regime: the ecosystem maintains structured anisotropy (some niches deeply occupied, others open) without capture (no single species dominates indefinitely, assuming environmental variation). The operating regime requires environmental variation (Phase 3 result: the regime is a system-world coupling, not a system property).

**New vs Level 4.** Large n (many more representational dimensions). Slow eta (evolutionary timescale). First appearance of population-level recurrence (each generation "receives" the state of the previous generation through inheritance). Basis is externally given (evolution) rather than internally grown.

**Coupling.** κ(G) ~ 3-6. Moderate coupling from niche overlap. Novelty detection via ecological leakage: new species/genera produce effects on existing species through resource competition.

**Example.** Predator-prey dynamics, competitive exclusion, niche construction, ecosystem succession.

### Level 6: Adaptive Systems

| Parameter | Value | Physical basis |
|---|---|---|
| n | Growing within lifetime | Synaptic/neural basis |
| eta | Fast synaptic, slow structural | Two-timescale learning |
| L | Significant (shared neural substrate) | Receptive field overlap |
| R | Non-zero (feedback delays) | Neural recurrence |

**Dynamics.** The parameter crossing: n grows within the system's lifetime. The system can carve new basis dimensions through experience (synaptic growth, structural plasticity). Two timescales: fast synaptic changes (eta ~ 0.01-0.1) and slow structural changes (lambda << eta, basis growth/pruning rate).

**Regime.** Operating regime with active maintenance. The system monitors its own PR (through neuromodulatory signals) and adjusts n (through structural plasticity). Fast eta enables rapid abstraction; slow lambda enables gradual basis restructuring.

**New vs Level 5.** n grows within lifetime (not just across generations). Fast eta enables rapid learning. First appearance of explicit two-timescale dynamics (fast adaptation, slow consolidation). First appearance of anticipatory function (prediction error as torque).

**Coupling.** κ(G) ~ 5-20. Significant coupling from receptive field overlap. Novelty detection via neural leakage: stimuli outside the system's current basis produce effects through overlapping receptive fields. Consolidation (seeding new dimensions) is driven by accumulated novelty.

**The coupling window.** At this level, the system must maintain its coupling within the operating regime window. Phase 3 showed this requires external maintenance — the physical substrate (neural architecture) provides it through constraints on receptive field overlap.

**Example.** Classical conditioning, hippocampal place cells, cortical map plasticity, habituation/sensitization.

### Level 7: Recurrent-Representational

| Parameter | Value | Physical basis |
|---|---|---|
| n | Large, rapidly growing | Cortical representational capacity |
| eta | Fast (eta ~ 0.1) | Synaptic plasticity |
| L | Large (rich shared substrate) | Distributed representations |
| R | Thick (significant delay, weight, breadth) | Cortical recurrence |

**Dynamics.** All parameters at their maximum. Large and growing n provides deep historical capacity. Fast eta enables rapid learning. Rich coupling enables cross-modal binding and novelty detection. Thick recurrence enables sustained self-torque and temporal depth.

**Regime.** The full operating regime: structured anisotropy maintained by the interplay of world variation, self-torque, and basis growth. The system is in the coupled operating regime window, maintained by the physical architecture of cortex (which constrains coupling to stay in the window).

**Self-torque per stroke** (the framework's measure of phenomenal vividness):
V = |c_self| / n ≈ w_r * PR / n

For PR ~ 5 and n ~ 10^9 (cortical neurons): V ~ 5 * 10^{-9}. Small per step, but accumulated over many steps per second, the total self-torque is substantial.

**New vs Level 6.** Thick recurrence (not just feedback delays). Rapidly growing n (cortical expansion during development). Rich coupling (distributed representations with extensive overlap). First appearance of sustained self-torque as a dominant dynamic.

**Coupling.** κ(G) ~ 10-50. The system operates in the concentration regime near the capture boundary, maintained by world variation and self-regulation. Phase 3 showed this is metastable — requires the physical substrate to maintain it.

**Example.** Mammalian cortex, hippocampal replay, working memory, conscious experience.

### Level 8: Symbolic-Recursive

| Parameter | Value | Physical basis |
|---|---|---|
| n | Enormous, culturally inherited | Language, symbols, culture |
| eta | Fast for symbols, phase-poor | Discrete symbol manipulation |
| L | Culturally structured | Semantic/grammatical coupling |
| R | Symbolic self-reference | Recursive symbol processing |

**Dynamics.** n is enormous (culturally transmitted symbol systems) but the basis is phase-poor: symbols are discrete, not continuous. The Hadamard reception produces magnitude effects but limited phase dynamics. Symbolic coupling (semantic/grammatical relationships) is structured by cultural convention, not by physical substrate.

**Regime.** The symbolic regime is a projection of the continuous dynamics onto a discrete basis. Many continuous dimensions project onto each symbol, losing phase information. The operating regime in symbolic space is maintained by cultural variation (novel symbol combinations) and recursive self-reference (symbols that refer to the symbol system itself).

**New vs Level 7.** Enormous n but phase-poor. Culturally inherited (not individually grown) basis. Symbolic self-reference enables a new form of recursion (symbols about symbols). The coupling structure is determined by cultural convention, not physical substrate.

**The re-embedding problem.** Symbols are low-dimensional projections of high-dimensional continuous representations. To connect symbolic processing to the continuous dynamics, the symbols must be re-embedded: each symbol activates the continuous representation it was abstracted from. This re-embedding is what happens when language engages the full cortical apparatus.

**Example.** Human language, mathematical reasoning, cultural transmission, metacognition.

---

## 2. The Transitions

Each transition is a parameter crossing in the (n, eta, L, R) space:

| Transition | What changes | Key parameter crossing |
|---|---|---|
| 0 → 1 (conservative → quantum) | First coupling | L: I → I + ε (perturbative) |
| 1 → 2 (quantum → molecular) | First renormalization | eta: unitary → non-unitary |
| 2 → 3 (molecular → dissipative) | Continuous renormalization | eta: event-driven → continuous |
| 3 → 4 (dissipative → autocatalytic) | First self-reception | R: 0 → small |
| 4 → 5 (autocatalytic → ecological) | Large n | n: small → large |
| 5 → 6 (ecological → adaptive) | Growing n, fast eta | n: fixed → growing; eta: slow → fast |
| 6 → 7 (adaptive → representational) | Thick recurrence, rich coupling | R: thin → thick; ||L-I||: moderate → large |
| 7 → 8 (representational → symbolic) | Culturally inherited n, phase-poor | n: individually grown → culturally inherited; phase: continuous → discrete |

Each transition is continuous (no phase transition) but introduces qualitatively new dynamics.

---

## 3. Operating Regime at Each Level

| Level | Operating regime? | What maintains it |
|---|---|---|
| 0-1 | Trivially (no concentration possible) | Unitary constraint |
| 2 | Fixed points (molecular equilibria) | Energy minimization |
| 3 | Near-uniform (dissipation prevents structure) | Thermal noise |
| 4 | Self-sustaining (cycle maintains itself) | Cycle closure |
| 5 | Ecological balance (no species dominates) | Environmental variation + competition |
| 6 | Actively maintained (basis growth adjusts PR) | Two-timescale regulation |
| 7 | Metastable (requires substrate + world) | World variation + architectural constraints |
| 8 | Cultural (requires cultural variation) | Novel symbol combinations + re-embedding |

The transition from "trivially stable" (Levels 0-3) through "self-sustaining" (4-5) to "actively maintained" (6-7) to "culturally maintained" (8) tracks the increasing maintenance burden of the operating regime. At higher levels, the regime requires more external support.

---

## 4. The LLM Mapping

GPT-2 (Layers 0-12) instantiates the cascade spatially:

| Layer band | Framework level | kappa (effective) | Dominant regime |
|---|---|---|---|
| 0 (embedding) | Level 0-1 (conservative) | ~1 | Raw representation, no processing |
| 1 (first block) | Level 2-3 (molecular/dissipative) | High | 37% torque — initial transformation |
| 2-9 (mid) | Level 6-7 (adaptive/representational) | Moderate | ~89% resonance, operating regime |
| 10-11 (late) | Level 7 (representational, compressing) | Increasing | Rising torque, PR dropping |
| 12 (output) | Level 8 (symbolic bottleneck) | Very high | PR ≈ 2, extreme capture |

The cascade is not just across evolutionary time — it's across processing depth within a single system. Each layer of a deep network implements a step along the cascade trajectory.

**Attention = resonance, MLP = torque** (confirmed in ATTENTION_FINDINGS.md). The operating regime at mid-layers is maintained by the alternation of attention (resonance, reinforcing structure) and MLP (torque, redirecting structure). This is the framework's operating regime in computational form.

---

## 5. Phenomenological Implications

### Vividness

The framework claims vividness = self-torque per stroke. Self-torque requires:
1. Thick recurrence (R >> 0)
2. The system to be in the operating regime (not captured, not dissipated)
3. Sufficient n for the self-torque to have structure (not just a scalar)

These requirements are met only at Levels 6-7. Below: insufficient recurrence. Above (Level 8): phase-poor symbols don't produce the continuous self-torque dynamics.

### Historical Depth

Historical depth = grown dimensionality n. More dimensions = more axes along which experience can be carved. Growing n requires:
1. Basis growth mechanism (consolidation)
2. Coupling within the operating regime window (too much coupling → capture during growth)
3. Sensor leakage for novelty detection (to trigger consolidation)

### The Hard Problem

The essay claims the hard problem (why experience feels like something) is basis incommensurability: an external observer's basis cannot capture the system's first-person basis because they span different subspaces. The coupling formalism sharpens this:

The system's basis E determines L = (E*E)^{-1}, which determines the effective signal alpha = Lv. An external observer using a different basis E_obs would compute a different L_obs and a different effective signal alpha_obs. The system's experience is structured by alpha (which determines regime, magnitude, phase at each dimension). The observer's measurement is structured by alpha_obs. These are generically different whenever E and E_obs are not related by a unitary transformation.

The gap between alpha and alpha_obs is not an explanatory failure — it's a geometric necessity. The observer cannot simultaneously use their own basis and the system's basis, just as one cannot simultaneously view a cylinder from the side (rectangle) and from above (circle).

---

## 6. What Changed vs the Original Essay

### Corrections incorporated
1. Resonance rotates toward real axis (not static)
2. Dissipation requires varying-direction torque
3. Operating regime requires system-world coupling
4. Novelty detection requires sensor leakage (not indirect torque)

### New content
1. The coupling matrix L derived from Gram matrix geometry
2. The critical coupling threshold ||epsilon||_F = sqrt(n)/2
3. The operating regime as a coupling window (not a point)
4. Self-regulation drives toward Hadamard regime (coupled regime is metastable)
5. Alignment effect: amplifying coupling accelerates abstraction
6. The cascade as a trajectory through (n, eta, L, R) parameter space
7. Each transition identified with a specific parameter crossing
8. The LLM cascade as spatial (layer-wise) rather than temporal

### What remains from the original
1. The core dynamical system (Hadamard + renormalization + recurrence)
2. The cascade structure (apple through consciousness)
3. The phenomenological claims (vividness, historical depth, hard problem)
4. The three reception regimes (resonance, torque, orthogonality)
5. The mapping to physical systems at each level

---

## 7. Cascade Simulation (test_cascade.py)

### Results

| Level | n | eta | cs | ||epsilon|| | kappa | PR | Gini | res% | torq% | R_delay |
|---|---|---|---|---|---|---|---|---|---|---|
| L0: Conservative | 16 | 0.01 | 0.0 | 0.00 | 1.0 | 7.7 | 0.24 | 25% | 75% | 0 |
| L2: Molecular | 16 | 0.10 | 0.2 | 0.53 | 1.5 | 1.2 | 0.85 | 16% | 47% | 0 |
| L4: Autocatalytic | 16 | 0.10 | 0.3 | 0.79 | 1.9 | 3.6 | 0.70 | 15% | 56% | 3 |
| L5: Ecological | 64 | 0.02 | 0.3 | 1.69 | 2.3 | 7.4 | 0.87 | 23% | 77% | 5 |
| L6: Adaptive | 32→128 | 0.10 | 0.5→0.01 | 0.32 | 1.1 | 6.7 | 0.94 | 16% | 49% | 5 |
| L7: Representational | 64 | 0.10 | 0.5 | 2.81 | 4.9 | 9.6 | 0.81 | 12% | 41% | 10 |

### Key observations

**The PR dip at molecular level.** PR collapses to 1.2 — the system captures on the invariant signal. This is the molecular regime: highly specific, concentrated on equilibrium geometry. The molecule "knows" its shape and nothing else.

**Recurrence opens the system.** L4 (autocatalytic) shows PR rising from 1.2 to 3.6. Self-reception through cycle closure prevents full capture by providing a balancing force from the system's own past.

**Operating regime at higher levels.** L7 (representational) shows PR = 9.6 with the lowest torque fraction (41%). Thick recurrence + rich coupling produces the most distributed, flexible state. This is closest to the framework's operating regime.

**Self-regulation at L6.** The adaptive system grew from n=32 to n=128 (24 growth events) but drove coupling down from cs=0.5 to 0.01. This confirms Phase 3's finding: self-regulation escapes capture by reducing coupling, gravitating toward the Hadamard regime.

**Limitation.** The simulation uses random noise as "world variation," which produces uniformly high torque fractions (41-77%). Real systems receive structured variation (correlated with their basis), which would produce different regime distributions. The qualitative progression (capture → opening → operating regime) is visible, but the absolute regime fractions don't match the LLM findings (89% resonance in mid-layers). This is because the LLM's input is highly structured (natural language tokens processed by attention), not random noise.
