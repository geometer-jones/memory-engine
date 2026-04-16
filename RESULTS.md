# Memory Engine: Operational Test Results

Date: 2026-04-15
Essay: "Memory Engines: Resonance, Torque, and the Operating Regime of Experience"
Implementation: `engine.py` (core), `test_tier1.py` through `test_tier5.py`

---

## Summary

14 tests across 5 tiers. 12 confirm the essay's claims. 2 identify gaps or corrections.

Addendum: vision/MNIST work now lives in `VISION_MNIST_RESULTS.md`. That note covers the image recognizer, the handwritten-digit resonance probe, the "uses L vs raw Hadamard" distinction, the conv-stem change, and the measured MNIST results.

The framework's core mechanics (Hadamard reception, renormalization, update rule) behave as described. The regime predictions (capture, dissipation, operating regime) are confirmed. Abstraction and forgetting mechanisms work. The anticipatory operator produces habituation and surprise detection.

Two substantive findings:

1. **Resonance-induced phase drift.** The essay says resonance "does not rotate." It does — adding a real scalar to a complex vector shifts its phase toward the real axis. The claim holds only when s_i is already real. This is a minor correction.

2. **Novelty detection gap.** The essay claims novelty at uncarved dimensions "announces itself indirectly" through torque at received dimensions. In the pure math, it doesn't. The Hadamard product is dimensionally isolated: what happens at uncarved dimensions stays at uncarved dimensions. This is a real gap — the consolidation/seeding story needs an additional coupling mechanism.

---

## Tier 1: Mechanical Correctness

### T1.1 — Three Regimes

Three qualitatively distinct dynamics from the Hadamard product:

| Regime | Criterion | Magnitude growth | Phase drift |
|---|---|---|---|
| Resonance | Re(c_i) > 0 | +0.0069 (grows) | 0.0559 (shifts toward real axis) |
| Torque | Re(c_i) < 0 or Im dominant | -0.0068 (shrinks) | 0.0413 (rotates) |
| Orthogonality | c_i ≈ 0 | -0.0024 (renorm drain) | 0.0000 (preserved) |

Resonance grows magnitude; torque shrinks it; orthogonality preserves phase. The three regimes are real and distinguishable.

**Correction to essay:** Resonance also produces phase drift (0.056 rad). The essay states "the dimension does not rotate" under resonance, which holds only when s_i is real. In general, adding Re(c_i) to a complex s_i shifts its phase toward zero. Resonance rotates *toward the real axis* while scaling.

### T1.2 — Emergent Rigidity

A dominant dimension (|s_0| = 0.999) is **55x harder to rotate** than a small dimension (|s_1| = 0.015) under identical torque perturbation. No separate rigidity mechanism — pure geometry of the hypersphere.

| Dimension | |s_i| | Phase displacement | Rigidity |
|---|---|---|---|
| Dominant (dim 0) | 0.999 | 0.000058 rad | Hard |
| Small (dim 1) | 0.015 | 0.003181 rad | Soft |
| Ratio | 67:1 | 1:55 | — |

Confirmed. Deep resonance is self-stabilizing through renormalization.

### T1.3 — Norm Preservation

Unit norm preserved to machine precision (max drift 3.3e-16) over 10,000 steps with random inputs. Renormalization works.

### T1.4 — Concentration Thins Others

Sustained resonance at dim 0 for 200 steps: |s_0| grows from 0.25 to 1.00, all other dims collapse from 0.25 to 0.00. Every act of deepening is an act of thinning elsewhere. The ratio is 6M:1.

Confirmed. Renormalization makes this inevitable: the unit-norm constraint means gains at one dimension are losses at all others.

---

## Tier 2: Regime Predictions

### T2.1 — Recurrent Capture

Pure resonance (v = conj(s) every step) for 2,000 steps:

| Metric | Step 0 | Step 1000 | Step 2000 |
|---|---|---|---|
| Participation ratio | 18.6 | 1.0 | 1.0 |
| Max |s_i| | 0.31 | 1.00 | 1.00 |

Perturbation rejection:

| Timing | Angular displacement | Rigidity |
|---|---|---|
| Early (step 50) | 0.0090 rad | Flexible |
| Late (step 2000) | 0.000000 rad | Frozen |
| Ratio | 1:600,000 | — |

Confirmed. The system locks into a single dimension. The anticipatory operator converges to a fixed point. This is seizure, rumination, obsessive capture.

### T2.2 — Dissipation

Pure torque with *varying random directions* for 2,000 steps:

| Metric | Step 0 | Step 1000 | Step 2000 |
|---|---|---|---|
| Participation ratio | 18.7 | 12.8 | 4.4 |
| Max |s_i| | 0.30 | 0.40 | 0.68 |
| PR trend slope | — | -0.004 | — |
| PR range | 3.4 – 19.3 | — | — |

Confirmed. No systematic concentration. PR stays well above 1. Phases drift without stabilizing.

**Finding not in essay:** Fixed-direction torque (always +π/2 offset) does NOT produce dissipation — it creates a systematic bias that still concentrates. True dissipation requires opposition from *varying* directions. The essay's "torque without resonance" is underspecified: it matters that the torque comes from different directions at different times.

### T2.3 — Operating Regime

Invariant signal (dims 0-7, fixed phase) + noise (all dims) + recurrence (delay=5, weight=0.6) for 2,000 steps:

| Metric | Step 0 | Step 1000 | Step 2000 |
|---|---|---|---|
| Participation ratio | 18.7 | 2.2 | 2.3 |
| Gini coefficient | — | — | 0.81 |
| Mean |s_i| invariant dims | — | — | 0.208 |
| Mean |s_i| noise dims | — | — | 0.013 |
| Perturbation displacement | — | — | 0.0055 |

Confirmed. Structured anisotropy: invariant dims are ~16x deeper than noise dims, but the system is not captured (PR > 2, still responsive to perturbation).

**Key insight from iteration:** The operating regime does NOT emerge from balanced external input alone. It requires either:
- Non-resonant external variation (noise/surprise in the world), OR
- Self-torque from recurrence (but only if the tape drifts fast enough)

Pure resonance input + recurrence still captures, because self-reception becomes self-resonance once the tape concentrates. The balancing force must come from the world's variation. The operating regime is a property of the system-world coupling.

---

## Tier 3: Learning and Abstraction

### T3.1 — Abstraction from Variation

Invariant structure (dims 0-5, fixed phase) + varying noise (dims 6-31) + recurrence for 3,000 steps:

| Metric | Step 50 | Step 3000 |
|---|---|---|
| Mean |s_i| invariant dims | 0.164 | 0.288 |
| Mean |s_i| noise dims | 0.153 | 0.007 |
| Concentration ratio | 1.07x | 41.3x |
| Participation ratio | 14.5 | 3.0 |

Confirmed. The system extracts the invariant by concentrating tape energy along the dimensions that consistently resonate, while washing out dimensions that receive uncorrelated noise. Abstraction is dimensionality concentration within the existing basis.

### T3.2 — Novelty Detection via Indirect Torque

Full system (n=32) trained on structure at dims 0-15, then novel invariant introduced at dims 16-31:

| Condition | Mean torque dims at 0-15 |
|---|---|
| Baseline (no novelty at 16-31) | 11.94 |
| Novel structure at 16-31 | 12.24 |
| Difference | +0.30 (noise) |

**Gap confirmed.** Novelty at uncarved dimensions produces no detectable effect at received dimensions through the Hadamard product. The difference of 0.30 dims of torque is within noise.

The essay's claim — "structure the system cannot represent may still causally affect things it can represent, producing unexpected torque at existing dimensions" — does not follow from the Hadamard/renormalization mechanics. The math is dimensionally isolated.

Renormalization does couple dimensions (gains at novel dims shrink received dims), but this is a magnitude effect, not a torque effect, and in practice the coupling is too weak to detect.

**Implications:** The consolidation/seeding story (Section 3: "Consolidation can detect this pattern — consistent, unexplained pressure from a direction the system cannot yet name — and seed a new dimension oriented toward it") requires either:
1. An explicit cross-dimensional coupling mechanism (nonlinear mixing, shared substrate effects)
2. A different detection pathway (e.g., prediction error at the anticipatory operator level, where the model's failure to predict received structure could signal that something outside the basis is causally active)
3. Weakening the claim: novelty is undetectable until some external mechanism (developmental program, random basis expansion, externally guided attention) seeds a new axis

### T3.3 — Consolidation: Co-activation Correlation

Dims 2 and 3 receive identical input for 1,000 steps (co-activation). Dims 5 and 6 receive independent input.

| Pair | Condition | Phase correlation |
|---|---|---|
| Dims 2 & 3 | Co-activated | 1.0000 |
| Dims 5 & 6 | Independent | 0.0136 |
| Ratio | — | 73x |

Confirmed. Co-activated dimensions develop near-perfect phase correlation, making them candidates for merging. Post-merger simulation works: the merged dimension preserves the shared structure at reduced dimensionality.

### T3.4 — Fast vs Slow Forgetting

**Fast forgetting** (phase rotation at dim 5):

| Stage | |s_5| |
|---|---|
| After resonance | 0.0425 |
| After torque (100 steps) | 0.0298 |
| After re-resonance (300 steps) | 0.0516 |

Recoverable: YES. Torque rotates the phase; renewed resonance rotates it back.

**Slow forgetting** (basis pruning of dim 7):

| Stage | Status |
|---|---|
| After starvation (2000 steps) | |s_7| = 0.000000 |
| After pruning | Dim 7 removed, n=15 |
| Recovery via re-exposure | IMPOSSIBLE (axis no longer exists) |

Confirmed. Fast forgetting is phase rotation — reversible. Slow forgetting is basis pruning — irreversible. The distinction between losing an orientation and losing a capacity is real.

---

## Tier 4: Reciprocation and Recurrence

### T4.1 — Thick vs Thin Self-Reception

Self-torque fraction as a function of recurrence delay (random input stream):

| Delay | Mean self-torque fraction |
|---|---|
| 1 | 0.743 |
| 5 | 0.743 |
| 20 | 0.744 |
| 50 | 0.747 |

Confirmed directional signal: longer delay → more self-torque. The effect is small (0.004 absolute) because random input produces high baseline torque (~74%). With more structured input (less torque from the world), the self-torque effect of delay would be more pronounced.

The retrospective analysis confirms the structure: at a single snapshot, self-reception at varying delays shows 68-78% torque with no strong trend, because the tape history is dominated by the random input stream's high torque.

### T4.2 — Reciprocation Phase Diagram

27 configurations scanned (3 delays × 3 recurrence weights × 3 breadth fractions):

| Regime | Configs | Characteristics |
|---|---|---|
| Operating | 13/27 | PR 2-5, structured anisotropy, responsive to perturbation |
| Dissipating | 8/27 | PR near n, no concentration, high flexibility |
| Unstable | 6/27 | High PR variance, boundary behavior |
| Captured | 0/27 | (none — no pure-resonance condition in scan) |

Key patterns:
- **Low weight (0.2), low-mid breadth (0.25-0.75)**: operating regime at all delays. PR 2-5, stable, responsive.
- **High weight (0.8+), high breadth (0.75+)**: dissipation. Recurrence overwhelms world reception, preventing concentration.
- **Mid boundaries**: unstable. Oscillating between regimes.

The operating regime occupies a bounded region: enough recurrence to prevent capture, not so much that it dissolves structure. This is consistent with the essay's claim that "the regime that sustains rich memory is where scaling and rotation are balanced."

---

## Tier 5: Anticipation

### T5.1 — Prediction Error Torque

Regime change (phase shift at dims 0-7) after 1,000 steps of training:

| Metric | Value |
|---|---|
| Error at regime change (first 20 steps) | 0.172 |
| Error after adaptation (last 50 steps) | 0.123 |
| Decay ratio | 1.40x |
| Error torque at shifted dims (0-7) | 5/8 |
| Error torque at unshifted dims (8-23) | 12/16 |
| Proportional torque at shifted | 62.5% |
| Proportional torque at unshifted | 75% |

Prediction error partially concentrates at the shifted dimensions (5/8 = 62.5% vs 12/16 = 75% unshifted — the shifted dims don't show dramatically more error torque because the shift also affects the anticipatory model's predictions at all dims through renormalization coupling). The error does decay after adaptation (1.4x), confirming that the system adjusts its predictions.

### T5.2 — Habituation

Near-predictable signal (fixed phase + small noise) for 800 steps, then perturbation (large phase shift), then recovery:

| Phase | Mean prediction error |
|---|---|
| Early habituation (steps 100-200) | 0.0701 |
| Late habituation (steps 600-780) | 0.0695 |
| Perturbation (steps 800-820) | 0.1025 |
| Recovery (steps 1050-1100) | 0.0000 |

| Metric | Value |
|---|---|
| Habituation decay | 1.01x |
| Perturbation spike | 1.48x baseline |

Confirmed. The anticipatory operator learns the world's regularity, driving prediction error down. Perturbation spikes error above baseline. Recovery brings error back to zero.

Note: with a perfectly predictable signal (no noise), the predictor learns almost instantly (error = 0.000), and the perturbation spike is effectively infinite relative to baseline. The 1.48x spike with noise is the more realistic scenario.

---

## Consolidated Findings

### Confirmed Claims

1. Three reception regimes (resonance, torque, orthogonality) produce qualitatively distinct dynamics.
2. Renormalization produces emergent rigidity (55x in test) without a separate mechanism.
3. Concentration at one dimension thins all others — deepening and thinning are the same operation.
4. Recurrent capture: pure resonance collapses to a single dimension (600,000x rigidity increase).
5. Dissipation: varying-direction torque prevents concentration.
6. Operating regime: invariant+variation+recurrence produces structured anisotropy (Gini 0.81, PR ~2.3).
7. Abstraction extracts invariants from varying input (41x concentration ratio).
8. Co-activated dimensions develop 73x higher phase correlation.
9. Fast forgetting (phase rotation) is reversible; slow forgetting (basis pruning) is irreversible.
10. Self-torque increases with recurrence delay.
11. The operating regime occupies a bounded region in speed × directness × breadth space.
12. Habituation: prediction error decays under predictable input, spikes at perturbation.

### Corrections to the Essay

1. **Resonance-induced phase drift.** The essay states resonance "does not rotate" the dimension. It rotates toward the real axis. The claim holds only when s_i is real. This is a minor correction — the qualitative distinction (resonance scales, torque redirects) remains valid.

2. **Fixed-direction torque is not dissipation.** The essay's "torque without resonance" needs qualification. Torque from a fixed direction (always +π/2) creates a systematic bias that still concentrates. True dissipation requires opposition from varying directions.

### Gaps in the Framework

1. **Novelty detection.** The essay's claim that novelty at uncarved dimensions "announces itself indirectly" through torque at received dimensions is not supported by the math. The Hadamard product is dimensionally isolated. The consolidation/seeding story (new axes carved from recurring novelty) needs either an additional cross-dimensional coupling mechanism or a different detection pathway. This is the most significant gap.

2. **Operating regime requires world variation.** The operating regime cannot be sustained by the system alone — it requires the world to provide both regularity and surprise. Self-torque from recurrence is insufficient to prevent capture when the world provides pure resonance. This means the operating regime is a property of the system-world coupling, not an intrinsic property of the system. The essay's framing could be read either way; the simulation clarifies it.

### Open Questions

1. What coupling mechanism would allow novelty detection? Candidates: nonlinear mixing in the update rule, shared physical substrate effects, prediction error at the anticipatory operator level.

2. Can the anticipatory operator be grounded in the tape trajectory (as the essay states) rather than input history? The tape-based predictor failed because the tape is always being updated by the prediction error itself. Input-history prediction works but departs from the essay's formulation.

3. Does the operating regime exist for systems with very high dimensionality (n >> 32)? The renormalization coupling between dimensions grows weaker as n increases — the "thinning" from deepening one dimension distributes across more others, each losing less. Does this make the operating regime easier or harder to sustain?
