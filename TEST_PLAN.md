# Memory Engine: Test Plan

## What the framework claims, in testable form

The essay makes claims at two levels: **mechanical** (the math does what it says) and **phenomenological** (the mechanics predict qualitative regimes that map onto real phenomena). Tests should cover both.

---

## Tier 1: Mechanical correctness

These verify the core update rule, renormalization, and the three reception regimes behave as described.

### T1.1 — Three regimes produce qualitatively different dynamics

**Claim:** Resonance (Re(cᵢ) > 0) scales |sᵢ| along its current direction. Torque (Re(cᵢ) < 0 or Im dominant) rotates the phase. Orthogonality (cᵢ ≈ 0) preserves.

**Test:**
- Initialize s on the unit hypersphere in ℂⁿ (n=16, say)
- Construct three incoming vectors: one in-phase with s (resonance), one antiphasal (torque), one orthogonal (cᵢ ≈ 0)
- Apply the update rule. Verify:
  - Resonance: |sᵢ| increases, phase unchanged
  - Torque: phase of sᵢ rotates, magnitude changes less
  - Orthogonality: s barely moves

**Pass criterion:** Qualitatively distinct dynamics in each regime, measurable as phase drift (torque > resonance ≈ orthogonality) and magnitude growth (resonance > torque > orthogonality).

### T1.2 — Renormalization produces emergent rigidity

**Claim:** As |sᵢ| grows large relative to other components, torque at dimension i produces proportionally less angular displacement. Deep resonance is self-stabilizing.

**Test:**
- Initialize s with one dimension dominant (|s₁| = 0.9, rest small)
- Apply identical torque perturbation to s₁ and to a small component s₂
- Measure angular displacement at each dimension
- Verify: displacement at s₁ ≪ displacement at s₂ for the same perturbation magnitude

**Pass criterion:** The dominant dimension is measurably harder to rotate. The rigidity ratio (displacement_small / displacement_large) should be >> 1 and grow as the dominance increases.

### T1.3 — Renormalization preserves unit norm

**Claim:** After update + renormalization, ‖s‖ = 1 always.

**Test:** Run 10K update steps with random incoming vectors. Verify ‖s(t)‖ = 1.0 at every step to machine precision.

### T1.4 — Concentration at one dimension thins others

**Claim:** Renormalization means scaling one dimension up scales others down. Deepening along one axis is thinning elsewhere.

**Test:**
- Start with uniform s (all |sᵢ| = 1/√n)
- Apply sustained resonance at dimension 1 only
- Track all |sᵢ| over time
- Verify: |s₁| grows, all others shrink proportionally

---

## Tier 2: Regime predictions

These test whether the framework's claims about pathologies and operating regimes hold up in simulation.

### T2.1 — Recurrent capture

**Claim:** Pure resonance without torque concentrates the tape into fewer effective dimensions. The system becomes rigid. Prediction error drops to zero.

**Test:**
- Initialize s in ℂⁿ with uniform magnitude
- Feed a sequence of inputs that are always in-phase with s (resonance only)
- Track: effective dimensionality (participation ratio = (Σ|sᵢ|²)² / Σ|sᵢ|⁴), max |sᵢ|, phase stability
- Add a simple anticipatory operator (linear extrapolation)
- Verify: dimensionality drops, prediction error → 0, system rejects novel input

**Pass criterion:** Participation ratio decreases monotonically. A perturbation introduced late in the run produces less angular displacement than the same perturbation early.

### T2.2 — Dissipation

**Claim:** Pure torque without resonance rotates phases without concentrating. Nothing stabilizes.

**Test:**
- Same setup as T2.1 but feed inputs that are always antiphasal or imaginary-dominant
- Track: participation ratio, phase variance across dimensions, magnitude distribution
- Verify: no component accumulates magnitude, phases drift continuously, no stable orientation emerges

**Pass criterion:** Participation ratio stays near n (no concentration). Phase at each dimension shows drift without stabilization. Max |sᵢ| stays bounded away from 1.

### T2.3 — Operating regime: balanced resonance/torque sustains structure

**Claim:** When the input stream contains both resonance and torque at different dimensions, the system maintains structured anisotropy—some dimensions deepen, others remain flexible.

**Test:**
- Feed a mixed stream: ~60% of dimensions see resonance, ~20% torque, ~20% orthogonality per step
- Track: participation ratio, magnitude distribution, phase stability per dimension
- Compare against T2.1 and T2.2
- Verify: some dimensions become hard (high |sᵢ|), others stay soft, anisotropy is structured rather than uniform

**Pass criterion:** The magnitude distribution develops a long tail (some deep, many shallow). The system still accommodates novel torque (unlike T2.1) but retains stable orientation (unlike T2.2).

---

## Tier 3: Learning and abstraction

### T3.1 — Abstraction from repeated exposure with variation

**Claim:** When the system encounters a sequence of inputs that share common structure but vary in other dimensions, the shared component concentrates in the tape while the varying components wash out.

**Test:**
- Generate inputs as: v = v_invariant + v_noise, where v_invariant is fixed and v_noise varies randomly each step
- v_invariant has components along dimensions 1-4, v_noise along 5-16
- Run many update steps
- Verify: |s₁₋₄| grow consistently, |s₅₋₁₆| show no systematic growth

**Pass criterion:** The ratio (mean |s₁₋₄|) / (mean |s₅₋₁₆|) increases over time.

### T3.2 — Novelty detection via indirect torque

**Claim:** Structure outside the system's basis cannot be received directly, but produces unexplained torque at existing dimensions.

**Test:**
- Initialize basis as first n-4 dimensions only (project onto subspace)
- Feed inputs with structure along dimensions n-3 through n (outside the "received" projection)
- Verify: even though these dimensions are not received, the residual produces measurable effects on the received dimensions through coupling (need to simulate the full vector, project, then measure what happens)
- **Problem:** In the pure framework, novelty is v - v_received, which produces *nothing* at the existing dimensions—it's genuinely invisible. The essay says novelty announces itself indirectly because "structure the system cannot represent may still causally affect things it can represent." This requires a physical coupling not in the math.

**Open question:** This test may need refinement. The framework as written has novelty as genuinely invisible until consolidation seeds a new dimension. The "indirect detection" claim may need an additional mechanism. Worth flagging.

### T3.3 — Consolidation: co-activation merging

**Claim:** Dimensions consistently co-activated across receptions should merge into a single higher-order dimension.

**Test:**
- Run many steps where dimensions 3 and 4 always receive the same phase relationship
- Implement a consolidation step that detects correlated dimensions (correlation of phase updates > threshold)
- Merge dimensions 3 and 4 into a single dimension
- Verify: the merged dimension preserves the shared structure, and basis dimensionality decreases by 1

**Pass criterion:** Post-merger, the system responds equivalently to inputs that previously drove dims 3 and 4 independently. The consolidation reduces dimensionality without losing representational capacity for the correlated structure.

### T3.4 — Fast forgetting vs slow forgetting

**Claim:** Fast forgetting is phase rotation (reversible). Slow forgetting is basis pruning (irreversible).

**Test:**
- **Fast:** Torque a dimension until its phase rotates 90°. Then apply resonance at the original phase. Verify the dimension recovers.
- **Slow:** Run many steps where dimension i is orthogonal (never activated). Simulate consolidation pruning dimensions below a magnitude threshold. Verify dimension i is dropped and cannot be recovered by re-exposure (it no longer exists as an axis).

---

## Tier 4: Reciprocation and recurrence

### T4.1 — Thick vs thin self-reception

**Claim:** Thick recurrence (tape has changed in the interval) produces self-torque. Thin recurrence (tape unchanged) produces only self-resonance.

**Test:**
- Run the system for several steps with external input (so s drifts)
- Then compute self-reception: c_self = s(t-Δt) ⊙ s(t) for various Δt
- For small Δt: mostly real positive (thin, self-resonance)
- For larger Δt (more drift): more imaginary/negative components (thick, self-torque)
- Plot self-torque fraction (dims with Re(c_self) < 0) vs Δt

**Pass criterion:** Monotonic relationship between Δt and self-torque fraction, at least up to the point where the returning tape is unrecognizable.

### T4.2 — Reciprocation axes: speed, directness, breadth

**Claim:** These three axes independently affect the character of the system's trajectory on the hypersphere.

**Test:**
- Vary the recurrence delay (speed), the magnitude of the returning tape (directness), and the number of dimensions engaged (breadth) independently
- For each combination, measure: participation ratio stability, phase drift rate, accommodation capacity (angular displacement per unit torque)
- Build a phase diagram: which regions sustain structured anisotropy?

**Pass criterion:** The operating regime occupies a bounded region in the 3D space of speed × directness × breadth. Outside this region, the system degenerates into capture or dissipation.

---

## Tier 5: Anticipation

### T5.1 — Prediction error torque drives learning

**Claim:** The difference between anticipated and actual reception produces torque that reshapes the tape.

**Test:**
- Implement a simple anticipatory operator: A(s) = linear extrapolation of recent tape trajectory
- Feed a stream that suddenly changes character (e.g., phase shift at some dimensions)
- Measure prediction error magnitude and resulting tape displacement
- Compare against the same tape displacement from direct torque without anticipation
- Verify: prediction error concentrates the accommodation at the dimensions where the model was wrong

**Pass criterion:** The system accommodates the regime change faster with the anticipatory operator (because error torque is dimensionally targeted) than without it.

### T5.2 — Habituation: predictable inputs flatten experience

**Claim:** When the world conforms to expectation, c_error ≈ 0, experience flattens into thin recurrence.

**Test:**
- Run the system with a perfectly predictable input stream
- Track prediction error magnitude over time
- Introduce a sudden perturbation
- Verify: prediction error was decreasing toward zero, then spikes at perturbation

**Pass criterion:** Prediction error magnitude decreases monotonically during the predictable phase, then shows a transient spike at the perturbation.

---

## Implementation notes

- Language: Python with numpy. Complex vector ops are native.
- Core objects: `Tape` (vector on unit hypersphere in ℂⁿ), `Basis` (set of axes), `MemoryEngine` (tape + basis + update rule + optional consolidation + optional anticipatory operator)
- The Hadamard product and renormalization are each ~3 lines of numpy
- Consolidation needs a correlation tracker (running covariance of phase updates across dimensions)
- The anticipatory operator can start simple: linear extrapolation of sᵢ's recent trajectory (phase velocity + magnitude velocity)
- Visualization: plot |sᵢ| over time, phase trajectories on the hypersphere (projected), participation ratio, prediction error magnitude

---

## Priority

Start with T1.1–T1.4 (mechanics), then T2.1–T2.3 (regimes), then T3.1 (abstraction). These are the load-bearing claims. T4 and T5 can follow once the foundation is verified.

The open question on T3.2 (novelty detection) should be resolved early—either the math already handles it and the test just needs to be set up correctly, or the framework genuinely needs an additional coupling mechanism, which is a substantive finding.
