# Phase 5: The Anticipatory Operator

Date: 2026-04-15
Status: Working paper
Depends on: COUPLING_THEORY.md, PHASE3_STABILITY.md

---

## 0. The Problem

The essay defines the anticipatory operator A(s) as extrapolating the tape's own dynamics to predict the system's next state. The simulation (FINDINGS.md, T5.1/T5.2) showed that habituation and surprise detection work in practice, but the tape-based formulation fails: the tape is always being updated by the prediction error itself, creating a feedback loop.

The working predictor tracked input history instead of tape history — functionally correct but departing from the essay's formulation.

This document reconstructs the anticipatory operator within the coupled framework.

---

## 1. Why the Tape-Based Predictor Failed

The essay's formulation: A(s) extrapolates the tape trajectory {s(t-k), ..., s(t-1)} to predict s(t). The prediction error e(t) = s(t) - A(s) drives torque.

The circularity: s(t) is determined by the update s(t) = normalize(s(t-1) + eta * c(t-1)), where c(t-1) depends on the world signal v(t-1). The predictor tries to predict s(t) from past states, but the past states were themselves shaped by the world signals that the predictor should be learning about. The predictor can't distinguish "the state changed because the world changed" from "the state changed because my prediction was wrong."

In the simulation, this manifested as the predictor quickly converging to the identity (predict "nothing changes"), which is the optimal strategy when state changes are dominated by noise.

---

## 2. The Reconstructed Formulation

### 2.1 The Anticipatory Operator as State Extrapolation

Define the anticipatory operator as a linear predictor of the state trajectory:

**A(t) = sum_{k=1}^{K} w_k * s(t-k)**

where w_k are predictor coefficients and K is the predictor window.

The simplest non-trivial predictor uses K=1 with w_1 = 1 (predict: next state = current state):

**A(t) = s(t-1)**

The prediction error:

**e(t) = s(t) - s(t-1)**

This is the first difference of the state trajectory. It captures the change in state from one step to the next.

### 2.2 Why This Avoids Circularity

The key insight: the predictor doesn't try to predict the INPUT or the RECEPTION. It predicts the STATE. The state is a well-defined quantity that changes according to the dynamics. The predictor simply asks: "does the state continue changing at the same rate, or does something unexpected happen?"

The circularity in the essay's formulation came from trying to predict the input that drives the state change. By predicting the state itself, the predictor avoids this loop. The state changes are the system's experience of the world; the predictor tracks the regularity of that experience.

### 2.3 Prediction Error as Torque

The prediction error e(t) = s(t) - s(t-1) is a complex vector. Its Hadamard product with the current state:

**c_pred(t) = e(t) ⊙ s(t)**

This is the "prediction torque" — the surprise at each dimension. Dimensions that changed more than expected receive torque; dimensions that changed as expected receive near-zero torque.

Under habituation (predictable world): s(t) ≈ s(t-1), so e(t) ≈ 0 and c_pred(t) ≈ 0.
Under perturbation (surprising world): s(t) ≠ s(t-1), so e(t) > 0 and c_pred(t) produces torque at surprised dimensions.

### 2.4 The Enhanced Predictor

For a richer predictor, use K > 1 with learned weights:

A(t) = w_1 * s(t-1) + w_2 * s(t-2) + ... + w_K * s(t-K)

The weights can be learned by minimizing prediction error:

min_{w} sum_t ||s(t) - A(t-1)||^2

For a linear predictor, this has a closed-form solution (linear regression on past states).

With coupled dynamics, the state trajectory incorporates coupling effects. The predictor implicitly learns the effective dynamics (including coupling) without needing to represent L explicitly. It just tracks how the state actually changes and predicts continuation.

---

## 3. Habituation and Surprise Under Coupled Dynamics

### 3.1 Habituation

Under a consistent world signal (invariant structure + small noise), the state converges to a near-fixed point. The first difference e(t) = s(t) - s(t-1) → 0. Prediction error vanishes. This is habituation: the system learns the world's regularity and stops being surprised.

With coupling: the effective signal alpha = Lv includes cross-dimensional contributions. The state converges to a fixed point that reflects the coupled dynamics. The predictor learns this fixed point and stops generating prediction error.

### 3.2 Surprise Detection

When the world changes (novel structure, perturbation), the state trajectory deviates from the predicted path. The prediction error e(t) = s(t) - s(t-1) becomes large. The torque from prediction error c_pred(t) = e(t) ⊙ s(t) is concentrated at the dimensions where the surprise is strongest.

With coupling: surprise at dimension j propagates to dimension i through the coupling matrix L. The predictor learns cross-dimensional correlations (e.g., "when dimension 3 changes, dimension 7 changes next"). If the correlation breaks (coupling pattern changes), the prediction error appears at BOTH dimensions, even though the surprise originated at one.

### 3.3 The Regime Change Signature

The simulation (T5.1) showed that a regime change (phase shift at dims 0-7) produces:
- Prediction error that partially concentrates at the shifted dimensions
- Error that decays after adaptation (1.4x ratio)
- Recovery back to baseline

Under coupled dynamics, this signature is enriched:
- The coupling spreads the prediction error across correlated dimensions
- The predictor adapts by learning the new cross-dimensional correlations
- The adaptation time depends on coupling strength and predictor learning rate

---

## 4. The Role of Coupling in Prediction

### 4.1 Coupling Enables Cross-Dimensional Prediction

In the Hadamard case (L = I), the state trajectory at dimension i is independent of all other dimensions. The predictor can only predict dimension i from its own past. There are no cross-dimensional predictions.

With coupling (L ≠ I), the effective signal alpha_i depends on all dimensions. The predictor can learn:
- "When dimensions 3 and 7 resonate simultaneously, dimension 12 receives torque next step"
- "When dimension 5 rotates, dimension 9 follows one step later"

This is predictive coding in the framework's terms: the system learns statistical regularities across its representational dimensions.

### 4.2 The Prediction Matrix

For a linear predictor with window K=1:
A(t) = W * s(t-1)

The prediction matrix W is n x n. The optimal W (minimizing prediction error) is:

W_opt = E[s(t) s(t-1)*] * (E[s(t-1) s(t-1)*])^{-1}

This is the linear regression solution. For the coupled system with random input:

E[s(t) s(t-1)*] ≈ I (state changes slowly under small eta)
E[s(t-1) s(t-1)*] ≈ I (state is approximately isotropic)

So W_opt ≈ I. The best predictor is "predict no change."

But for structured input (invariant + variation), the prediction matrix develops structure:

W_opt reflects the correlation structure of the state trajectory, which depends on L and the input statistics. Dimensions that change together (coupled by L or driven by correlated input) develop off-diagonal entries in W_opt.

### 4.3 Prediction Error and Novelty

The prediction error at dimension i:
e_i(t) = s_i(t) - sum_j W_ij s_j(t-1)

For the optimal predictor, E[e_i(t)] = 0 (unbiased). The variance of the prediction error depends on how predictable the state trajectory is.

When novel structure appears (outside the system's basis), the prediction error increases because:
1. Through sensor leakage: novel structure perturbs the state at active dimensions
2. The predictor hasn't learned the correlation between the novel structure and the existing state
3. The prediction error is torque (unexpected change), which can trigger consolidation

This provides an alternative novelty detection pathway to the sensor leakage mechanism (COUPLING_THEORY.md Section 5): instead of detecting novelty through accumulated leakage torque, the system detects novelty through prediction error. The two mechanisms are complementary:
- Leakage torque: structural detection (repeated exposure accumulates)
- Prediction error: dynamical detection (unexpected state change at each step)

---

## 5. Connection to Predictive Processing

The reconstructed anticipatory operator maps onto predictive processing (PP) theories:

| Framework concept | PP concept |
|---|---|
| State s(t) | Posterior belief |
| Predicted state A(t) | Prior prediction |
| Prediction error e(t) | Prediction error (precision-weighted) |
| Torque from prediction error | Prediction error minimization |
| Coupling matrix L | Generative model structure |
| Learning rate eta | Precision / attention |
| Consolidation (basis growth) | Model complexity increase |

The key difference: in PP, prediction error minimization drives perception and action. In the framework, prediction error IS torque — it's the dynamical mechanism that redirects the state. The "minimization" happens naturally because the state update (resonance + renormalization) drives toward dimensions that reduce torque.

---

## 6. Formal Properties

### 6.1 Habituation Rate

Under a constant input v_0 with coupling L:

The state converges to a fixed point s* satisfying:
s* = normalize(s* + eta * (Lv_0) ⊙ s*)

The prediction error at step t:
||e(t)|| = ||s(t) - s(t-1)|| ~ ||s(0) - s*|| * (1 - eta * lambda_min(LL*))^t

The habituation rate is proportional to eta * lambda_min(LL*). Faster learning rates and stronger coupling produce faster habituation.

### 6.2 Surprise Response

Under a step perturbation v_0 -> v_1 at time t_0:

The prediction error peaks at:
||e(t_0)|| = ||s(t_0) - s(t_0 - 1)|| ≈ eta * ||(L(v_1 - v_0)) ⊙ s(t_0-1)||

The peak surprise is proportional to:
- eta (learning rate)
- ||L(v_1 - v_0)|| (coupled signal change)
- |s_i| at the surprised dimensions (depth modulation)

### 6.3 Cross-Dimensional Surprise Propagation

With coupling, surprise at dimension j propagates to dimension i:

e_i(t) proportional to L_ij * (v_1_j - v_0_j) * s_i

The coupling matrix determines how surprise spreads across the system. Strongly coupled dimensions share surprise; weakly coupled dimensions don't.

---

## 7. Testable Predictions

1. **Habituation rate scales with eta * kappa(G).** Faster learning and stronger coupling produce faster habituation.

2. **Surprise propagates through coupling matrix.** A perturbation at dimension j produces prediction error at dimension i proportional to |L_{ij}|.

3. **Prediction error detects novelty faster than leakage torque.** A single novel event produces prediction error immediately, while leakage torque requires repeated exposure.

4. **The optimal predictor window K scales with coupling strength.** Stronger coupling creates longer-range temporal correlations, requiring longer predictor windows.

5. **Prediction error at captured dimensions is near zero.** In recurrent capture (PR ≈ 1), the state doesn't change, so the predictor is trivially correct. The system cannot detect surprise at captured dimensions.

---

## 8. Simulation Results (test_anticipation.py)

### T1: Habituation — NUANCED

The simple predictor A(t) = s(t-1) shows prediction error stabilizing at a constant value (noise-driven steady state) rather than decaying to zero. The prediction error magnitude is constant because the noise keeps the state fluctuating around its attractor.

This is correct behavior: the predictor measures the rate of state change, which is constant under constant input + noise. True habituation (prediction error → 0) would require a predictor that learns the attractor, not just the previous state.

**Conclusion:** The first-difference predictor detects change, not surprise per se. It's a surprise detector when the input statistics change, and a noise meter when they don't.

### T2: Surprise Response — CONFIRMED

| Phase | Mean prediction error | Ratio to baseline |
|---|---|---|
| Baseline (late habituation) | 0.0178 | 1.0x |
| Perturbation peak | 0.0585 | 3.3x |
| Recovery (last 50 steps) | 0.0171 | 1.0x |

Clear spike at perturbation with full recovery. The anticipatory operator detects the regime change and adapts back.

### T3: Cross-Dimensional Surprise Propagation — CONFIRMED

Correlation between per-dimension prediction error and coupling matrix: **r = 0.979**.

The prediction error propagates through the coupling matrix as predicted. However, at moderate coupling (cs=0.5), the cross-dimensional error at individual dimensions is below noise floor. The propagation mechanism is real but typically perturbative.

### T4: Habituation Rate vs eta*kappa — NOT CONFIRMED

The decay ratio (early error / late error) shows negative correlation with eta*kappa (-0.745). Higher eta*kappa produces larger absolute errors at all phases, not faster decay. The habituation rate is better characterized as the rate of convergence to the noise-driven steady state, which is a function of the input statistics rather than coupling strength.

### Revised Predictions

1. ~~Habituation rate scales with eta * kappa.~~ **Revised:** The steady-state prediction error magnitude scales with eta * noise_level. Coupling affects the spatial distribution of error, not the temporal decay rate.

2. Surprise propagation through coupling matrix: **CONFIRMED** (r = 0.979).

3. Prediction error detects novelty immediately (single-step response): **CONFIRMED** (3.3x spike).

4. ~~Optimal predictor window K scales with coupling strength.~~ Not tested — remains open.

5. Prediction error at captured dimensions is near zero: **Not tested** — predicted but not verified.
