# The Coupling Problem: Formal Treatment

Date: 2026-04-15
Status: Working paper — simulation-verified
Context: Phase 2 of formal reconstruction, following FINDINGS.md
Simulation: test_coupling.py

---

## 0. The Problem

The Memory Engine framework defines a dynamical system on the unit hypersphere in C^n:

1. **Reception**: c_i = v_i * s_i (Hadamard product)
2. **Update**: s~_i = s_i + eta * c_i
3. **Renormalization**: s <- s~/||s~||

The Hadamard product is **dimensionally isolated**: c_i depends only on (v_i, s_i). The dynamics at dimension i are independent of all other dimensions. The simulation (RESULTS.md) confirmed this produces clean regime structure, emergent rigidity, recurrent capture, and abstraction.

It also produces two documented failures:

- **Novelty detection gap** (T3.2): Structure at dimensions outside the system's basis is invisible. The torque difference with vs. without novelty was 0.30 dimensions -- within noise.
- **Binding failure** (standalone_me_binding.py): The Hadamard product cannot associate dimensions. Cross-dimensional binding required bolt-on mechanisms that degraded performance.

The reconstruction must resolve these failures without destroying the confirmed dynamics. This document derives the minimum generalization.

---

## 1. The Isolation Theorem

**Theorem (Dimensional Isolation).** Under Hadamard reception with renormalization, the magnitude dynamics at dimension i are governed by:

|s~_i| = |s_i + eta * v_i * s_i| / ||s~|| = |s_i| * |1 + eta * v_i| / ||s~||

The phase dynamics are:

arg(s~_i) = arg(s_i) + arg(1 + eta * v_i)

Both depend only on (v_i, s_i) and the global normalization ||s~|| (which couples dimensions only through the shared norm constraint).

**Proof.** Direct computation. The Hadamard product c_i = v_i * s_i uses only the i-th components. The update s~_i = s_i + eta * c_i = s_i * (1 + eta * v_i) is multiplicative in s_i. Renormalization introduces global coupling through ||s~||, but this is a scalar that affects all dimensions equally (magnitude scaling without directional bias).

**Corollary.** The regime at dimension i (resonance, torque, orthogonality) is determined entirely by (v_i, s_i). Structure at dimension j != i cannot change the regime classification at dimension i.

This is why T3.2 failed: novelty at dimensions 16-31 cannot affect the regime at dimensions 0-15 through the Hadamard product.

---

## 2. The Coupled Reception

### 2.1 Setup

A system has basis vectors E = [e_1, ..., e_n] spanning a subspace of ambient space H (dim H >= n). The state s in C^n represents accumulated structure in basis E. The world provides signal v in H.

### 2.2 Projection

The system projects v onto its basis to receive it. The naive projection is:

w_i = <e_i, v>

For an orthonormal basis, w is the correct received signal. For a non-orthonormal basis, the correct least-squares projection accounts for inter-basis correlations through the Gram matrix G = E*E:

**v_received = G^{-1} w**    where G_{ij} = <e_i, e_j>

### 2.3 The Coupling Matrix

Define **L = G^{-1}**. The coupled reception is:

**c_i = (sum_j L_{ij} w_j) * s_i**

For an orthonormal basis, G = I, L = I, and this reduces to the Hadamard product.

For G = I + epsilon with ||epsilon|| << 1:

L = I - epsilon + epsilon^2 - ...

To first order, L_{ij} = -epsilon_{ij} = -<e_i, e_j> for i != j. Correlated basis vectors produce negative coupling in L.

**The coupling is not arbitrary.** L is determined entirely by the Gram matrix of the basis. The system inherits its coupling from the geometry of its representational substrate.

---

## 3. Two Sources of Coupling

The Gram coupling above operates within the active dimensions (between basis vectors). A second source operates between the ambient space and the active dimensions.

### 3.1 Gram Coupling (Intra-Basis)

When basis vectors are not orthogonal, the projection v_received = G^{-1} w mixes contributions across active dimensions. Signal at dimension j "leaks" into the effective reception at dimension i through L_{ij}.

**Physical motivation.** In any real system, representational dimensions are not perfectly independent:
- Neural features: correlated tuning curves (GPT-2 key matrix max cosine 0.47-0.82)
- Molecular binding sites: cross-reactivity between substrates
- Ecological niches: overlapping resource requirements

The Gram matrix encodes this overlap. Its inverse L determines how reception at one dimension is contaminated by signal at others.

**What it enables.** Cross-dimensional effects without explicit binding mechanism. Co-activation at dimensions i and j produces correlated structure through the shared coupling.

### 3.2 Sensor Leakage (Extra-Basis)

The projection P: H -> C^n is never perfectly selective. In reality:

v_received = (E^+ + DeltaP) v

where E^+ is the pseudoinverse and DeltaP represents imperfection in the physical sensor. The deviation DeltaP allows signal outside span(E) to leak into the active dimensions.

**Physical motivation.** Real sensors have finite selectivity:
- A neuron tuned to feature A also responds weakly to feature B
- A binding site for substrate X has weak affinity for substrate Y
- A species' niche overlaps with neighboring species' niches

The imperfection DeltaP is determined by the physical implementation of the projection, not by the mathematics of the basis.

**What it enables.** Novelty detection. Structure outside the system's basis enters through DeltaP and perturbs the dynamics at active dimensions. Repeated exposure accumulates the perturbation until it becomes detectable.

### 3.3 Unified Form

Both sources can be combined into a single coupling operator:

v_received_i = sum_j L_{ij} w_j + sum_a Delta_{ia} v_a

where j ranges over active dimensions (Gram coupling) and a ranges over all ambient dimensions (sensor leakage). For a = j (active dimensions), the two terms combine into the effective coupling matrix L_eff = L + Delta_active.

In practice, the dominant effects are:
- **Gram coupling**: affects cross-dimensional dynamics within the active basis
- **Sensor leakage**: affects novelty detection from outside the basis

---

## 4. Regime Persistence

### 4.1 The Multiplicative Update Under Coupling

With coupled reception, the update at dimension i is:

s~_i = s_i + eta * (sum_j L_{ij} w_j) * s_i = s_i * (1 + eta * alpha_i)

where alpha_i = sum_j L_{ij} w_j is the effective received signal.

This is still multiplicative. The state at dimension i is scaled by a complex factor (1 + eta * alpha_i). The regime at dimension i is determined by:

- **Resonance**: Re(alpha_i) > 0 and |Im(alpha_i)| < Re(alpha_i). The magnitude grows; phase shifts toward real axis.
- **Torque**: Re(alpha_i) < 0 or |Im(alpha_i)| >= |Re(alpha_i)|. The magnitude shrinks or phase rotates.
- **Orthogonality**: alpha_i ~ 0. Negligible change.

**The three regimes are preserved under coupling.** The coupling changes what alpha_i is (it now depends on all dimensions, not just i), but the regime classification based on alpha_i is unchanged.

### 4.2 When Does Coupling Flip Regimes?

The regime at dimension i flips (relative to the Hadamard classification) when the coupling contribution changes the sign of Re(alpha_i):

Re(alpha_i) = Re(w_i) + sum_{j!=i} Re(L_{ij} w_j)

The coupling perturbation is delta_i = sum_{j!=i} Re(L_{ij} w_j). For regime flip:

|delta_i| > |Re(w_i)|

This requires the coupling contribution to exceed the direct (Hadamard) contribution.

**Bound on regime flip probability.** For random input with i.i.d. components w_j ~ CN(0, sigma^2):
- Re(w_i) has variance sigma^2/2
- delta_i has variance sum_{j!=i} |L_{ij}|^2 * sigma^2/2 = (||L_{i,.}||^2 - |L_{ii}|^2) * sigma^2/2

The probability of regime flip is bounded by:
P(flip) <= (||L_{i,.}||^2 - |L_{ii}|^2) / ||L_{i,.}||^2

For L = I + epsilon with ||epsilon|| << 1:
P(flip) ~ O(||epsilon||^2)

The regime structure is robust: small coupling produces regime flips with probability O(||epsilon||^2).

### 4.3 Numerical Estimate for GPT-2

GPT-2 key matrix (proxy for coupling structure):
- Mean off-diagonal cosine: 0.039-0.060
- Effective rank: 700-724 out of 768

Estimated ||epsilon||_F per row: sqrt(n) * mean_cosine ~ sqrt(768) * 0.05 ~ 1.4

This is not small compared to 1. The GPT-2 regime flip probability per dimension is estimated at ~O(1.4^2/768) ~ 0.25%. Over 768 dimensions, ~2 regime flips per step from coupling. This is consistent with the observed ~10% torque in mid-layers: most torque comes from the MLP transformation, with coupling contributing a small fraction.

---

## 5. Novelty Detection via Sensor Leakage

### 5.1 The Detection Pathway

Consider a system with active basis e_1, ..., e_k (out of ambient dimension N > k). The world provides novel structure at direction alpha (not in span(e_1, ..., e_k)).

Through sensor leakage, the effective received signal at dimension i picks up a contribution from the novel direction:

alpha_i = sum_j L_{ij} <e_j, v> + sum_a Delta_{ia} v_a

The second term includes the contribution from v along direction alpha:

Delta_{i,alpha} * v_alpha

This is the leakage from the novel direction into dimension i.

### 5.2 Signal Accumulation

If the novel structure appears with consistent phase phi at direction alpha, the leakage contribution at dimension i after T steps is:

S_i(T) = sum_{t=1}^{T} Re(Delta_{i,alpha} * v_alpha^{(t)} * s_i^{(t)})

For consistent v_alpha (magnitude V, phase phi) and slowly varying s_i:

S_i(T) ~ T * |Delta_{i,alpha}| * V * |s_i| * cos(phi - arg(s_i))

The noise (from random fluctuations in the world signal at other directions) grows as:

N_i(T) ~ sqrt(T) * sigma_noise

The signal-to-noise ratio improves as:

SNR(T) ~ sqrt(T) * |Delta_{i,alpha}| * V * |s_i| / sigma_noise

Novelty becomes detectable when SNR exceeds some threshold (e.g., SNR > 2 for 95% confidence). The detection time is:

T_detect ~ (sigma_noise / (|Delta_{i,alpha}| * V * |s_i|))^2

### 5.3 The Minimum Leakage

For novelty to be detectable within the system's lifetime, we need:

|Delta_{i,alpha}| > sigma_noise / (V * |s_i| * sqrt(T_max))

For typical values (V ~ 1/sqrt(N), |s_i| ~ 1/sqrt(k), T_max ~ 1000 steps):

|Delta_{i,alpha}| > sigma_noise * sqrt(N) * sqrt(k) / sqrt(1000) ~ sigma_noise * sqrt(N*k) / 31.6

For a 32-dimensional system in 64-dimensional ambient space:
|Delta_min| ~ sigma_noise * sqrt(2048) / 31.6 ~ 1.4 * sigma_noise

So the minimum leakage is proportional to the noise level. In a quiet environment (low noise), even tiny leakage enables novelty detection. In a noisy environment, stronger leakage is needed.

### 5.4 The Consolidation Mechanism

Once novelty is detected (accumulated leakage signal exceeds threshold), the system can seed a new basis vector oriented toward the novel direction. This is the essay's consolidation process, now grounded in the leakage mechanism:

1. Novel structure appears at direction alpha outside span(E)
2. Sensor leakage produces small perturbations at active dimensions
3. Repeated exposure accumulates the perturbation (SNR grows as sqrt(T))
4. When accumulated signal exceeds threshold, consolidation triggers
5. A new basis vector e_{k+1} is seeded, approximately oriented toward alpha
6. The new vector is initially non-orthogonal to existing basis (by construction)
7. This increases Gram coupling (L changes), which is acceptable under Section 4

The key prediction: **consolidation time scales as 1/|Delta|^2**, where |Delta| is the leakage coefficient toward the novel direction. Systems with more imperfect sensors (larger |Delta|) consolidate faster but also have noisier reception (more cross-dimensional contamination).

---

## 6. The Coupling Cascade

Each level of the essay's cascade corresponds to increasing coupling strength, measured by ||L - I||:

| Level | Coupling ||L - I|| | Source | Novelty detection |
|---|---|---|---|
| Conservative fields | 0 | Perfect phase space coordinates | None |
| Quantum fields | ~0 | Perturbative interaction | Via interaction Hamiltonian |
| Molecular | ~epsilon | Orbital overlap | Cross-reactivity |
| Ecological | ~sqrt(k)*mean_cosine | Niche overlap | Resource competition |
| Adaptive | ~O(1) | Shared neural substrate | Receptive field overlap |
| Representational | ~O(sqrt(n)) | Rich shared substrate | Cross-modal leakage |

The cascade is now parameterized by a specific mathematical quantity (the deviation of the coupling matrix from identity), not by verbal "thickening."

**Prediction.** Systems at lower cascade levels should show:
- Cleaner regime structure (fewer regime flips)
- Slower consolidation (longer detection times)
- Less cross-dimensional binding

Systems at higher levels should show:
- Noisier regime structure (more regime flips)
- Faster consolidation (shorter detection times)
- More cross-dimensional binding

---

## 7. Open Questions

1. **Deriving DeltaP from substrate physics.** The sensor leakage DeltaP is currently a physical assumption, not a derived quantity. Can it be derived from the system's physical structure? Candidates: thermodynamic noise limits, Heisenberg uncertainty, finite precision of molecular recognition.

2. **Gram coupling and basis growth.** When a new dimension is seeded, it's approximately but not exactly orthogonal to existing dimensions. This increases ||epsilon|| over time. Does the coupling eventually become large enough to destabilize the regime structure? Or does the operating regime self-regulate the coupling strength?

3. **Coupling and the anticipatory operator.** With coupled reception, the anticipatory operator must predict the effective signal alpha_i = sum_j L_{ij} w_j, not just w_i. This is a harder prediction task but also a richer one: the anticipatory model can learn cross-dimensional correlations.

4. **Relationship to attention.** The GPT-2 analysis showed attention has zero torque (pure resonance). In the coupled framework, attention provides the projection step (computing w = E* v), and the coupling L determines how the projected signal mixes. The Hadamard depth modulation (scaling by s_i) is the step attention lacks. The MLP provides torque by modifying the residual stream (equivalent to modifying s between steps).

---

## 8. Testable Predictions

1. **Regime persistence.** For coupling ||L - I||_F < 1, the regime classification agrees with the Hadamard classification on >95% of dimensions, per step.

2. **Novelty detection threshold.** Novelty at uncarved dimensions becomes detectable when the leakage coefficient satisfies |Delta| > sigma_noise / (V * |s_i| * sqrt(T)). Below this threshold, novelty is invisible regardless of exposure time.

3. **Consolidation time scaling.** T_consolidate ~ 1/|Delta|^2. Systems with stronger leakage consolidate faster.

4. **Coupling destroys regimes at large strength.** For ||L - I||_F > sqrt(n), the regime classification becomes unreliable (>25% flips per step). The three-regime structure dissolves into a general dynamical systems description.

5. **Gram coupling enables passive binding.** Co-activated dimensions develop correlated structure through Gram coupling, even without explicit binding mechanism. The correlation strength scales with |L_{ij}|.

---

## 9. Simulation Results (test_coupling.py)

### T1: Regime Persistence — CONFIRMED

| Coupling strength | ||epsilon||_F | Flip rate |
|---|---|---|
| 0.0 | 0.00 | 0.0% |
| 0.1 | 0.38 | 2.2% |
| 0.3 | 1.15 | 6.6% |
| 0.5 | 1.92 | 12.4% |
| 1.0 | 3.83 | 36.4% |
| 2.0 | 7.67 | 35.0% |
| 5.0 | 19.17 | 37.4% |
| 10.0 | 38.35 | 39.9% |

- Flip rate < 5% for ||epsilon||_F < 1: **CONFIRMED**
- Flip rate > 25% for ||epsilon||_F > sqrt(n): **CONFIRMED**
- The O(||epsilon||^2) scaling holds for small coupling but saturates around 35-40% for large coupling (regime classification becomes essentially random).

### T2: Novelty Detection — CONFIRMED with correction

| Leakage | Torque (novel) | Torque (baseline) | Difference |
|---|---|---|---|
| 0.000 | 6.07 | 6.07 | +0.00 |
| 0.010 | 6.01 | 6.07 | -0.06 |
| 0.030 | 5.84 | 6.07 | -0.23 |
| 0.050 | 6.02 | 6.07 | -0.05 |
| 0.100 | 5.75 | 6.07 | -0.32 |
| 0.200 | 3.47 | 6.07 | -2.60 |
| 0.500 | 0.82 | 6.07 | -5.25 |

- No leakage -> undetectable: **CONFIRMED** (matches T3.2 gap)
- Leakage >= 0.200 -> detectable: **CONFIRMED**

**Correction to theory:** The essay predicts novelty produces "unexpected torque." The simulation shows novelty through leakage produces **unexpected resonance** — the leaked signal from novel dimensions reinforces active dimensions, reducing torque. The sign is opposite to the essay's claim. The detection pathway works, but the phenomenology is reduced torque (stabilization), not increased torque (disruption).

### T3: Coupling Threshold — CONFIRMED with unexpected non-monotonicity

| ||epsilon||_F | PR | Gini | Behavior |
|---|---|---|---|
| 0.04-1.49 | 12.8-13.1 | 0.29 | Stable operating regime |
| 3.11-6.47 | 5.3-5.8 | 0.50 | Concentration (partial capture) |
| 13.46 | 1.0 | 0.925 | Full capture |
| 28.01 | 13.6 | 0.31 | Dissipation (coupling too strong) |
| 58-121 | 18-18.5 | 0.24-0.25 | Near-uniform (extreme coupling = noise) |

Non-monotonic behavior: medium coupling concentrates (partial capture), strong coupling captures fully, extreme coupling dissipates (prevents any dimension from stabilizing). The operating regime exists in a bounded window of coupling strength, not at a single point.

**New prediction:** The operating regime requires coupling strength within a specific window. Below the window, dimensional isolation prevents binding and novelty detection. Above the window, coupling noise prevents concentration. The window boundaries depend on n and eta.

### T4: Passive Binding — PARTIALLY CONFIRMED

Gram coupling produces stronger co-activation correlation than the no-coupling case:

| Coupling | Co-activated phase diff | Control phase diff | Bound? |
|---|---|---|---|
| 0.0 | 1.58 | 1.63 | Marginal |
| 0.3 | 1.63 | 1.76 | Yes (weak) |
| 1.0 | 0.85 | 2.87 | Yes (strong) |
| 3.0 | 2.90 | 3.62 | Yes (moderate) |

At coupling=1.0 (||epsilon||=3.83, |L_{23}|=0.62), the co-activated pair is 3.4x more correlated than the control. Gram coupling does enable passive binding, but the effect is strongest when the direct coupling coefficient |L_{ij}| between the co-activated dimensions is large.

---

## 10. Revised Claims

Based on the simulation results, the theory's predictions are updated:

1. **Regime persistence:** CONFIRMED. Flip rate < 5% for ||epsilon||_F < 1, saturates at ~35-40% for large coupling.

2. **Novelty detection threshold:** CONFIRMED. Threshold is leakage ~0.1-0.2 for n=32, eta=0.1. Below threshold, novelty is invisible. **Correction:** novelty produces resonance (stabilization), not torque (disruption).

3. **Operating regime window:** NEW FINDING. The operating regime exists in a bounded coupling window. Too little coupling -> isolation. Too much -> capture or dissipation. This replaces the simple "more coupling = more coupling" prediction.

4. **Passive binding:** PARTIALLY CONFIRMED. Gram coupling enhances co-activation correlation, with effect size proportional to |L_{ij}|. Not as clean as the Hadamard case (T3.3 showed perfect correlation without coupling) but produces detectable binding with sufficient coupling.

5. **Coupling destroys regimes at large strength:** CONFIRMED with qualification. At ||epsilon||_F ~ sqrt(n), regime classification degrades significantly. But the system doesn't become random — it transitions through capture (medium coupling) to dissipation (extreme coupling).
