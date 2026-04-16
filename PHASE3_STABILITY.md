# Phase 3: Operating Regime Stability Theory

Date: 2026-04-15
Status: Working paper — analytically derived, numerically verified
Depends on: COUPLING_THEORY.md (Phase 2)

---

## 0. The Question

Phase 2 established that coupled reception c_i = (Lv)_i * s_i preserves the three-regime structure for small coupling. But the simulation (test_coupling.py, T3) revealed non-monotonic behavior: the operating regime exists in a bounded coupling window, with concentration and capture at medium coupling, and dissipation at extreme coupling.

This document derives the boundaries of that window analytically.

---

## 1. The Spectral Formulation

### 1.1 Setup

The coupled dynamical system:
1. Effective signal: alpha = Lv (where L = G^{-1}, G = I + epsilon)
2. Coupled reception: c_i = alpha_i * s_i
3. Update: s~_i = s_i (1 + eta * alpha_i)
4. Renormalization: s <- s~/||s~||

### 1.2 Growth Rates

The magnitude growth at dimension i per step:

|s~_i|^2 = |s_i|^2 * |1 + eta * alpha_i|^2 / ||s~||^2

For small eta:

|s~_i|^2 ≈ |s_i|^2 * (1 + 2 eta Re(alpha_i)) / ||s~||^2

The growth rate at dimension i is proportional to Re(alpha_i).

### 1.3 Expected Growth Under Random Input

For random input v with E[v v*] = sigma^2 I:

E[|alpha_i|^2] = sigma^2 (LL*)_ii = sigma^2 ||L_i,.||^2

The expected magnitude change at dimension i (second-order):

E[delta |s_i|^2] ≈ |s_i|^2 * eta^2 * sigma^2 * ||L_i,.||^2

Dimensions with larger ||L_i,.||^2 grow faster. Concentration occurs when ||L_i,.||^2 varies significantly across dimensions.

### 1.4 The Eigenvalue Connection

G is Hermitian (G = I + epsilon, epsilon Hermitian). Its eigenvalue decomposition is G = U Lambda U*, where Lambda = diag(lambda_1, ..., lambda_n) with lambda_1 >= ... >= lambda_n.

L = G^{-1} = U Lambda^{-1} U*, with eigenvalues mu_i = 1/lambda_i.

LL* = U Lambda^{-2} U*, with eigenvalues mu_i^2 = 1/lambda_i^2.

The maximum growth rate is proportional to max |mu_i| = 1/min |lambda_i|. The ratio of max to min growth rate is:

kappa(L) = max |mu_i| / min |mu_i| = max |lambda_i| / min |lambda_i| = kappa(G)

This is the condition number of G. It determines the concentration dynamics: large kappa(G) means some dimensions grow much faster than others, leading to capture.

---

## 2. The Wigner Threshold

### 2.1 Eigenvalue Distribution of epsilon

For a random Hermitian epsilon with i.i.d. off-diagonal entries epsilon_ij ~ CN(0, sigma_eps^2):

The eigenvalue distribution of epsilon follows the **Wigner semicircle law** (asymptotic in n):

rho(lambda) = (2 / (pi R^2)) sqrt(R^2 - lambda^2)  for |lambda| < R

where R = 2 sigma_eps sqrt(n) is the spectral radius.

The extreme eigenvalues approach +-R:

lambda_max(epsilon) ≈ +R = +2 sigma_eps sqrt(n)
lambda_min(epsilon) ≈ -R = -2 sigma_eps sqrt(n)

### 2.2 Critical Coupling

G = I + epsilon, so:

lambda_min(G) = 1 + lambda_min(epsilon) ≈ 1 - 2 sigma_eps sqrt(n)

G becomes **indefinite** (lambda_min < 0) when:

2 sigma_eps sqrt(n) > 1
sigma_eps > 1 / (2 sqrt(n))

In terms of the Frobenius norm:

||epsilon||_F ≈ sigma_eps * sqrt(n(n-1)) ≈ sigma_eps * n

So:

||epsilon||_F_critical ≈ n / (2 sqrt(n)) = sqrt(n) / 2

**Theorem (Critical Coupling).** For a system with random Hermitian coupling epsilon, the Gram matrix G = I + epsilon becomes indefinite at:

||epsilon||_F_critical = sqrt(n) / 2

At this coupling, L = G^{-1} has unbounded eigenvalues, and the condition number kappa(G) diverges.

### 2.3 Numerical Verification

| n | Measured ||epsilon||_F_critical | Predicted sqrt(n)/2 | Ratio |
|---|---|---|---|
| 16 | 2.62 | 2.00 | 1.31 |
| 32 | 3.13 | 2.83 | 1.11 |
| 64 | 4.30 | 4.00 | 1.08 |
| 128 | 5.90 | 5.66 | 1.04 |
| 256 | 8.24 | 8.00 | 1.03 |

Convergence to the prediction as n increases. The deviation at small n is expected: the Wigner semicircle is an asymptotic result.

---

## 3. The Operating Regime Window

### 3.1 Three Regimes of Coupling

The coupling strength ||epsilon||_F determines three qualitatively distinct dynamical regimes:

**I. Hadamard regime (||epsilon||_F << sqrt(n)/2)**

G is positive definite with kappa(G) ≈ 1. All eigenvalues of L are positive and near 1. Growth rates are nearly uniform. The system behaves like the uncoupled Hadamard case with small perturbations.

Properties: Clean regime structure, no cross-dimensional effects, no novelty detection, no binding. Stable high-PR state under random input.

**II. Concentration regime (||epsilon||_F ~ sqrt(n)/2)**

G is positive definite but with large kappa(G). Some eigenvalues of L are much larger than others. Dimensions aligned with large-L eigenvectors grow much faster, leading to concentration and eventually capture.

Properties: Regime structure perturbed (5-25% flips), cross-dimensional effects emerge, novelty detection possible, binding possible. PR decreases under sustained input.

The transition from operating regime to capture within this band depends on the interplay between signal structure and coupling structure. Invariant input aligned with large-L eigenvectors concentrates; invariant input orthogonal to them doesn't.

**III. Dissipation regime (||epsilon||_F >> sqrt(n)/2)**

G is indefinite. L has negative eigenvalues, creating always-torque dimensions. But the eigenvalues of G are spread far from zero in both directions, so the max growth rate actually decreases. The system receives noise-like input at every step and cannot concentrate.

Properties: Regime structure degraded (>25% flips), growth rates more uniform (negative eigenvalues reduce dynamic range), PR returns to high values. The system is in a dissipative state where no structure can form.

### 3.2 The Operating Regime as a Coupling Window

The operating regime (structured anisotropy without capture) exists within the concentration regime, but near its lower boundary:

**Operating regime window:**

sqrt(n) / C_lower < ||epsilon||_F < sqrt(n) / C_upper

where C_lower and C_upper depend on the system's needs:

- **Lower bound** (coupling becomes useful): When leakage is sufficient for novelty detection. Depends on noise level and exposure time (see COUPLING_THEORY.md Section 5.3).
- **Upper bound** (coupling causes capture): When kappa(G) becomes large enough that random input concentrates. Approximately ||epsilon||_F ~ sqrt(n) / 3 (where kappa(G) ~ 5).

For n = 32:
- Upper bound ≈ sqrt(32) / 3 ≈ 1.89
- Critical threshold ≈ sqrt(32) / 2 ≈ 2.83

The simulation showed PR dropping at ||epsilon|| ~ 3.1 and capture at ||epsilon|| ~ 13.5. The operating regime upper bound is conservative (captures at lower coupling than predicted for this n).

### 3.3 Why Dissipation at Extreme Coupling

At the critical coupling, kappa(G) diverges because lambda_min(G) = 0. Beyond the critical coupling, lambda_min(G) < 0 and moves further from zero. Simultaneously, lambda_max(G) increases.

The eigenvalues of L = G^{-1} are 1/lambda_i. For the eigenvalue closest to zero:

|1/lambda_min(G)| peaks near the critical coupling and then decreases as |lambda_min(G)| grows.

Numerically:

| n=32 | lambda_min(G) | max growth rate sqrt(1/lambda_min^2) |
| cs=1.0 | -0.33 | 16.8 |
| cs=2.0 | -1.56 | 25.1 |
| cs=5.0 | -5.31 | 6.0 |
| cs=10.0 | -11.93 | 2.4 |

The max growth rate peaks around cs=2 and then decreases. This means: beyond a certain coupling, the dominant dimension's growth advantage diminishes, and the system becomes more uniform.

The physical mechanism: at extreme coupling, the off-diagonal terms dominate. The effective signal alpha = Lv is a random linear combination of many independent components, which produces noise-like input regardless of the world signal structure. Noise prevents concentration.

---

## 4. Dependence on n, eta, and Input Statistics

### 4.1 Scaling with n

The critical coupling ||epsilon||_F_critical = sqrt(n) / 2 scales as sqrt(n). This means:

- Higher-dimensional systems tolerate more coupling (in absolute terms) before capture.
- The coupling window widens as sqrt(n).
- But the per-dimension coupling (||epsilon||_F / n) scales as 1/sqrt(n), so individual dimensions are LESS affected at high n.

This has implications for the cascade: systems at higher cascade levels have larger n and can sustain stronger coupling without capture. The coupling window naturally widens as representational capacity grows.

### 4.2 Scaling with eta

The learning rate eta affects the timescale of concentration but not the coupling thresholds. The rate of concentration is proportional to eta^2 (second-order growth term). Faster learning means faster capture, but the coupling boundaries are the same.

The operating regime requires that concentration (from coupling) is balanced by variation (from the world). The balance condition is:

eta^2 * sigma_signal^2 * kappa(G) ~ sigma_noise^2

where sigma_signal is the signal strength and sigma_noise is the noise strength. For larger eta, the system tolerates less coupling (smaller kappa(G)) before capture.

### 4.3 Scaling with Input Statistics

For input v = v_invariant + v_noise:

- v_invariant has fixed structure at some dimensions (correlated across time)
- v_noise is random at each step

The coupling mixes both:
alpha = L v_invariant + L v_noise

The invariant signal at dimension i:
alpha_invariant_i = sum_j L_ij * v_invariant_j

For L ≈ I + epsilon:
alpha_invariant_i ≈ v_invariant_i + sum_j epsilon_ij * v_invariant_j

The second term is "cross-dimensional invariant leakage": invariant structure at dimension j reinforces (or opposes) dimension i through the coupling.

If epsilon_ij has the right sign, this can AMPLIFY the invariant signal, causing faster abstraction. If it has the wrong sign, it can cancel the invariant signal, preventing abstraction.

**Prediction:** For systems where the coupling structure aligns with the invariant structure (epsilon_ij has the same sign as v_invariant_i * v_invariant_j), coupling accelerates abstraction. For misaligned coupling, coupling retards abstraction.

---

## 5. The Coupling Window and the Cascade

Each level of the cascade operates in a specific region of the coupling window:

| Level | Coupling | kappa(G) | Behavior |
|---|---|---|---|
| Conservative | 0 | 1 | Perfect Hadamard, no cross-talk |
| Quantum | ~0 | ~1 | Perturbative coupling, near-Hadamard |
| Molecular | small | ~1-2 | Weak coupling, slow novelty detection |
| Ecological | moderate | ~2-5 | Operating regime window |
| Adaptive | near-critical | ~5-50 | Strong coupling, fast abstraction, capture risk |
| Representational | varies | controlled | Self-regulated coupling (basis growth adjusts kappa) |

**The key insight for representational systems:** The system can regulate its own coupling strength by controlling basis growth. Adding new dimensions changes G (and hence kappa(G)). If the system monitors its own PR or regime structure, it can adjust coupling by growing or pruning dimensions.

This is the framework's account of **metacognition**: the system's ability to monitor and adjust its own operating regime through basis management. A system that detects incipient capture (PR dropping too low) can grow new dimensions (increasing n, widening the coupling window, reducing kappa(G)). A system that detects isolation (no cross-dimensional effects, no novelty detection) can allow dimensions to become more correlated (increasing epsilon, entering the operating regime window).

---

## 6. Testable Predictions

1. **Critical coupling scales as sqrt(n).** Verified numerically for n = 16 to 256.

2. **Operating regime exists for kappa(G) between ~2 and ~5.** To be verified with structured input (invariant + noise + coupling).

3. **Capture at kappa(G) >> 10.** Confirmed in test_coupling.py T3.

4. **Dissipation at very high coupling.** Confirmed in test_coupling.py T3 (PR returns to ~18 at ||epsilon|| ~120).

5. **Amplifying coupling (positive L_{ij}) accelerates abstraction; diminishing coupling (negative L_{ij}) retards it.** Verified with correction: the sign flips through G^{-1}, so anti-correlated basis vectors produce amplifying coupling.

6. **Self-regulated coupling via basis growth.** A system that monitors PR and adjusts n can partially escape capture but naturally drives toward the Hadamard regime (near-zero coupling), not the coupled operating regime.

---

## 7. Simulation Results

### T1: Operating Regime Window (structured input)

With invariant signal at dims 0-7, noise everywhere, recurrence:

| cs | ||epsilon|| | kappa(G) | PR | Status |
|---|---|---|---|---|
| 0.0 | 0.00 | 1.0 | 8.0 | Hadamard abstraction |
| 0.1 | 0.38 | 1.3 | 7.9 | Near-Hadamard |
| 0.3 | 1.15 | 2.3 | 7.6 | Operating regime |
| 0.5 | 1.92 | 4.9 | 7.1 | Operating regime |
| 0.7 | 2.68 | 28.3 | 6.5 | Operating regime (wide kappa tolerance) |
| 1.0 | 3.83 | 25.5 | 1.0 | CAPTURE |
| 1.5+ | 5.75+ | 87+ | 1.0-1.7 | Deep capture |

Capture at kappa > 25. Operating regime is stable for kappa up to ~28 — wider than predicted (5-10).

### T2: Alignment and Abstraction Speed (corrected)

Weak invariant signal (0.15) + strong noise (0.3):

| Condition | Abstraction step | CR at 500 | CR at 1000 |
|---|---|---|---|
| Amplifying (negative eps, positive L) | 122 | 12,303x | 1.4e8x |
| Random | 157 | 1,091x | 1.1e6x |
| Diminishing (positive eps, negative L) | 202 | 246x | 48,656x |

**CONFIRMED (with corrected sign).** Amplifying coupling (positive L_{ij} between invariant dims) accelerates abstraction by 1.7x relative to diminishing coupling. The sign flips through G^{-1}: anti-correlated basis vectors in the Gram matrix produce amplifying coupling in the inverse.

Physical interpretation: when basis vectors for invariant dimensions anti-correlate, the system must "work harder" to represent each one, but the coupling matrix compensates by amplifying the shared signal across dimensions. This is a form of distributed representation that the Hadamard case can't achieve.

### T3: Self-Regulation

Starting at cs=1.0 (capture regime for n=32):

| Condition | PR | n_final | cs_final |
|---|---|---|---|
| No regulation | 1.0 | 32 | 1.00 |
| With regulation (grow + reduce cs) | 2.9 | 256 | 0.01 |

**PARTIAL.** Self-regulation improves PR from 1.0 to 2.9 by growing dimensions and reducing coupling. But the system drives toward the Hadamard regime (cs=0.01) rather than maintaining moderate coupling. The coupled operating regime is not self-stabilizing — the system escapes capture by reducing coupling to near-zero.

**Implication for the framework:** The coupled operating regime (moderate coupling with cross-dimensional effects) requires external maintenance — either from the physical structure of the substrate or from explicit coupling control. A system that only monitors PR and adjusts dimensionality will naturally gravitate toward the Hadamard regime. The coupled operating regime is metastable: it exists in a window but is not an attractor of the self-regulation dynamics.
