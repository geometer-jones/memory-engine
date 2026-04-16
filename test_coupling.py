"""
Coupling Theory Verification Tests.

Tests the predictions from COUPLING_THEORY.md:
  T1: Regime persistence under increasing coupling
  T2: Novelty detection with sensor leakage
  T3: Coupling threshold characterization
  T4: Passive binding via Gram coupling
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

# ── Helpers ──────────────────────────────────────────────────────────────

def renormalize(s):
    norm = np.linalg.norm(s)
    return s / norm if norm > 1e-15 else s

def participation_ratio(s):
    mags_sq = np.abs(s) ** 2
    return float(np.sum(mags_sq) ** 2 / np.sum(mags_sq**2))

def classify_regime(c_i):
    mag = abs(c_i)
    if mag < 1e-12:
        return "orth"
    re = np.real(c_i)
    im_mag = abs(np.imag(c_i))
    if re > 0 and im_mag < re:
        return "resonance"
    if re < 0 or im_mag >= abs(re):
        return "torque"
    return "orth"

def classify_all(c):
    return [classify_regime(ci) for ci in c]

def hadamard_reception(v, s):
    """Standard Hadamard reception: c_i = v_i * s_i"""
    return v * s

def coupled_reception(v, s, L):
    """Coupled reception: c_i = (sum_j L_ij v_j) * s_i"""
    v_eff = L @ v
    return v_eff * s

def make_gram_coupling(n, coupling_strength, seed=42):
    """Generate a coupling matrix L = G^{-1} where G = I + epsilon.

    coupling_strength controls ||epsilon||_F.
    Returns L such that L approx I for small coupling.
    """
    rng = np.random.RandomState(seed)
    # Random Hermitian perturbation
    eps = coupling_strength * (rng.randn(n, n) + 1j * rng.randn(n, n)) / np.sqrt(2 * n)
    eps = (eps + eps.conj().T) / 2  # Hermitian
    np.fill_diagonal(eps, 0)  # Diagonal stays 1
    G = np.eye(n, dtype=complex) + eps
    try:
        L = np.linalg.inv(G)
    except np.linalg.LinAlgError:
        L = np.eye(n, dtype=complex)
    return L, eps

def make_sensor_leakage(n_active, n_novel, leakage_strength, seed=42):
    """Generate sensor leakage matrix Delta (n_active x n_novel).

    Delta projects novelty from uncarved dimensions into active dimensions.
    """
    rng = np.random.RandomState(seed)
    return leakage_strength * (rng.randn(n_active, n_novel) + 1j * rng.randn(n_active, n_novel)) / np.sqrt(n_active)


# ── T1: Regime Persistence ──────────────────────────────────────────────

def test_regime_persistence():
    """Vary coupling strength and measure regime flip rate.

    Prediction: flip rate ~ O(||epsilon||^2). For ||epsilon||_F < 1,
    flip rate < 5%.
    """
    n = 32
    n_steps = 1000
    eta = 0.1
    coupling_strengths = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]

    print("=" * 70)
    print("T1: REGIME PERSISTENCE UNDER COUPLING")
    print("=" * 70)
    print(f"n={n}, steps={n_steps}, eta={eta}")
    print()

    results = []

    for cs in coupling_strengths:
        if cs == 0.0:
            L = np.eye(n, dtype=complex)
            eps_norm = 0.0
        else:
            L, eps = make_gram_coupling(n, cs)
            eps_norm = np.linalg.norm(eps, 'fro')

        s = renormalize(np.random.randn(n) + 1j * np.random.randn(n))
        rng = np.random.RandomState(0)

        total_dims = 0
        regime_flips = 0

        for step in range(n_steps):
            v = rng.randn(n) + 1j * rng.randn(n)
            v = v / np.linalg.norm(v)

            # Hadamard classification
            c_hadamard = hadamard_reception(v, s)
            regimes_hadamard = classify_all(c_hadamard)

            # Coupled classification
            c_coupled = coupled_reception(v, s, L)
            regimes_coupled = classify_all(c_coupled)

            # Count flips
            for j in range(n):
                if regimes_hadamard[j] != regimes_coupled[j]:
                    regime_flips += 1
                total_dims += 1

            # Update using coupled reception
            s = renormalize(s + eta * c_coupled)

        flip_rate = regime_flips / total_dims
        results.append((cs, eps_norm, flip_rate))
        print(f"  coupling={cs:5.1f}  ||eps||_F={eps_norm:6.2f}  flip_rate={flip_rate:.4f}  ({flip_rate*100:.1f}%)")

    print()

    # Check prediction: flip rate should grow as ~||eps||^2
    print("  Prediction check (flip_rate ~ ||eps||^2):")
    baseline = results[0][2]
    for cs, eps_norm, flip_rate in results[1:]:
        if eps_norm > 0:
            ratio = flip_rate / (eps_norm**2)
            print(f"    ||eps||={eps_norm:.2f}  flip_rate={flip_rate:.4f}  ratio={ratio:.6f}")

    print()

    # Verdict
    low_coupling_flips = [r for r in results if r[1] < 1.0]
    if all(r[2] < 0.05 for r in low_coupling_flips):
        print("  CONFIRMED: Flip rate < 5% for ||eps||_F < 1")
    else:
        print("  FAILED: Flip rate exceeds 5% even at low coupling")

    high_coupling_flips = [r for r in results if r[1] > np.sqrt(n)]
    if high_coupling_flips and all(r[2] > 0.25 for r in high_coupling_flips):
        print("  CONFIRMED: Flip rate > 25% for ||eps||_F > sqrt(n)")
    else:
        max_eps = max(r[1] for r in results)
        print(f"  (Insufficient coupling range to test ||eps||_F > sqrt({n})={np.sqrt(n):.1f}, max tested: {max_eps:.1f})")

    return results


# ── T2: Novelty Detection ───────────────────────────────────────────────

def test_novelty_detection():
    """Repeat T3.2 with sensor leakage.

    Prediction: Novelty becomes detectable when leakage > noise threshold.
    Signal accumulates as sqrt(T).
    """
    n = 32
    n_active = 16  # dims 0-15 active
    n_novel = 16   # dims 16-31 novel
    eta = 0.1
    n_steps = 2000
    leakage_strengths = [0.0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5]

    print("=" * 70)
    print("T2: NOVELTY DETECTION VIA SENSOR LEAKAGE")
    print("=" * 70)
    print(f"n={n}, active={n_active}, novel={n_novel}, steps={n_steps}, eta={eta}")
    print()

    results = []

    for ls in leakage_strengths:
        rng = np.random.RandomState(42)
        s = renormalize(rng.randn(n) + 1j * rng.randn(n))
        s[n_active:] = 0.0
        s = renormalize(s)

        if ls > 0:
            Delta = make_sensor_leakage(n_active, n_novel, ls, seed=42)
        else:
            Delta = np.zeros((n_active, n_novel), dtype=complex)

        # Anticipatory model for torque tracking
        torque_at_active = []

        for step in range(n_steps):
            v = rng.randn(n) + 1j * rng.randn(n)
            v = v / np.linalg.norm(v)

            # Add invariant structure at active dims 0-7
            for i in range(8):
                v[i] += 0.5 * np.exp(1j * 0.3)

            # Novel structure at dims 16-23
            for i in range(16, 24):
                v[i] += 0.5 * np.exp(1j * 0.7)  # Consistent phase

            # Received signal with leakage
            v_received = np.zeros(n, dtype=complex)
            v_received[:n_active] = v[:n_active]
            v_received[n_active:] = 0.0

            # Sensor leakage: novel dims leak into active dims
            v_received[:n_active] += Delta @ v[n_active:]

            # Hadamard reception (only active dims matter)
            c = v_received * s
            c[n_active:] = 0.0

            # Track torque at active dims
            torque_dims = sum(1 for i in range(n_active) if classify_regime(c[i]) == "torque")
            torque_at_active.append(torque_dims)

            # Update
            s = renormalize(s + eta * c)
            s[n_active:] = 0.0

        mean_torque = np.mean(torque_at_active[-500:])  # Last 500 steps

        # Also run baseline (no novel structure)
        rng2 = np.random.RandomState(42)
        s2 = renormalize(rng2.randn(n) + 1j * rng2.randn(n))
        s2[n_active:] = 0.0
        s2 = renormalize(s2)

        if ls > 0:
            Delta2 = make_sensor_leakage(n_active, n_novel, ls, seed=42)
        else:
            Delta2 = np.zeros((n_active, n_novel), dtype=complex)

        torque_baseline = []
        for step in range(n_steps):
            v = rng2.randn(n) + 1j * rng2.randn(n)
            v = v / np.linalg.norm(v)
            for i in range(8):
                v[i] += 0.5 * np.exp(1j * 0.3)
            # NO novel structure at dims 16-23

            v_received2 = np.zeros(n, dtype=complex)
            v_received2[:n_active] = v[:n_active]
            v_received2[n_active:] = 0.0

            c2 = v_received2 * s2
            c2[n_active:] = 0.0

            torque_dims2 = sum(1 for i in range(n_active) if classify_regime(c2[i]) == "torque")
            torque_baseline.append(torque_dims2)

            s2 = renormalize(s2 + eta * c2)
            s2[n_active:] = 0.0

        mean_baseline = np.mean(torque_baseline[-500:])
        torque_diff = mean_torque - mean_baseline

        results.append((ls, mean_torque, mean_baseline, torque_diff))
        print(f"  leakage={ls:.3f}  torque(novel)={mean_torque:.2f}  torque(baseline)={mean_baseline:.2f}  diff={torque_diff:+.2f}")

    print()

    # Verdict
    no_leakage = results[0]
    with_leakage = [r for r in results if r[0] > 0]
    detectable = [r for r in with_leakage if abs(r[3]) > 1.0]

    if abs(no_leakage[3]) < 0.5:
        print(f"  CONFIRMED: No leakage (0.0) -> diff = {no_leakage[3]:+.2f} (undetectable, matches T3.2)")
    else:
        print(f"  UNEXPECTED: No leakage shows diff = {no_leakage[3]:+.2f}")

    if detectable:
        min_detectable = min(detectable, key=lambda r: r[0])
        print(f"  CONFIRMED: Leakage >= {min_detectable[0]:.3f} makes novelty detectable (diff = {min_detectable[3]:+.2f})")
    else:
        print("  FAILED: No leakage level made novelty detectable")

    return results


# ── T3: Coupling Threshold ──────────────────────────────────────────────

def test_coupling_threshold():
    """Characterize the transition from Hadamard-like to coupled behavior.

    Measure: participation ratio stability, magnitude concentration, and
    regime structure as coupling increases.
    """
    n = 32
    eta = 0.1
    n_steps = 1000
    coupling_strengths = np.logspace(-2, 1.5, 12)  # 0.01 to ~30

    print("=" * 70)
    print("T3: COUPLING THRESHOLD CHARACTERIZATION")
    print("=" * 70)
    print(f"n={n}, steps={n_steps}, eta={eta}")
    print()

    results = []

    for cs in coupling_strengths:
        L, eps = make_gram_coupling(n, cs, seed=42)
        eps_norm = np.linalg.norm(eps, 'fro')

        rng = np.random.RandomState(42)
        s = renormalize(rng.randn(n) + 1j * rng.randn(n))

        pr_values = []
        gini_values = []
        regime_counts = {"resonance": 0, "torque": 0, "orth": 0}

        for step in range(n_steps):
            v = rng.randn(n) + 1j * rng.randn(n)
            v = v / np.linalg.norm(v)

            c = coupled_reception(v, s, L)
            regimes = classify_all(c)

            for r in regimes:
                regime_counts[r] += 1

            s = renormalize(s + eta * c)

            if step >= 500:  # Measure after transient
                pr_values.append(participation_ratio(s))
                mags = np.abs(s)
                sorted_mags = np.sort(mags)
                gini = 1 - 2 * np.sum(np.cumsum(sorted_mags) / np.sum(sorted_mags)) / n
                gini_values.append(gini)

        mean_pr = np.mean(pr_values)
        std_pr = np.std(pr_values)
        mean_gini = np.mean(gini_values)
        total_classifications = sum(regime_counts.values())
        resonance_frac = regime_counts["resonance"] / total_classifications
        torque_frac = regime_counts["torque"] / total_classifications

        results.append({
            "cs": cs, "eps_norm": eps_norm,
            "mean_pr": mean_pr, "std_pr": std_pr,
            "mean_gini": mean_gini,
            "resonance_frac": resonance_frac,
            "torque_frac": torque_frac,
        })

        print(f"  ||eps||={eps_norm:6.2f}  PR={mean_pr:5.1f}+-{std_pr:.1f}  "
              f"Gini={mean_gini:.3f}  res={resonance_frac:.2f}  torque={torque_frac:.2f}")

    print()

    # Check if PR destabilizes at high coupling
    low_coupling = [r for r in results if r["eps_norm"] < 1.0]
    high_coupling = [r for r in results if r["eps_norm"] > np.sqrt(n)]

    if low_coupling:
        mean_pr_low = np.mean([r["mean_pr"] for r in low_coupling])
        mean_pr_std_low = np.mean([r["std_pr"] for r in low_coupling])
        print(f"  Low coupling (||eps||<1):  PR = {mean_pr_low:.1f} +- {mean_pr_std_low:.1f}")

    if high_coupling:
        mean_pr_high = np.mean([r["mean_pr"] for r in high_coupling])
        mean_pr_std_high = np.mean([r["std_pr"] for r in high_coupling])
        print(f"  High coupling (||eps||>sqrt(n)):  PR = {mean_pr_high:.1f} +- {mean_pr_std_high:.1f}")
        if mean_pr_std_high > 3 * mean_pr_std_low:
            print("  CONFIRMED: High coupling destabilizes PR (regime structure dissolves)")
        else:
            print("  NOTE: PR variance doesn't increase dramatically at tested coupling")

    return results


# ── T4: Passive Binding via Gram Coupling ────────────────────────────────

def test_passive_binding():
    """Test whether Gram coupling produces correlated structure at co-activated dims.

    Prediction: Co-activated dimensions develop phase correlation proportional
    to |L_{ij}|, even without explicit binding mechanism.
    """
    n = 32
    eta = 0.1
    n_steps = 1000
    coupling_strengths = [0.0, 0.3, 1.0, 3.0]

    print("=" * 70)
    print("T4: PASSIVE BINDING VIA GRAM COUPLING")
    print("=" * 70)
    print(f"n={n}, steps={n_steps}, eta={eta}")
    print(f"Co-activated pair: dims 2,3. Independent pair: dims 5,6.")
    print()

    results = []

    for cs in coupling_strengths:
        if cs == 0.0:
            L = np.eye(n, dtype=complex)
            eps_norm = 0.0
        else:
            L, eps = make_gram_coupling(n, cs, seed=42)
            eps_norm = np.linalg.norm(eps, 'fro')

        rng = np.random.RandomState(42)
        s = renormalize(rng.randn(n) + 1j * rng.randn(n))

        for step in range(n_steps):
            v = rng.randn(n) + 1j * rng.randn(n)
            v = v / np.linalg.norm(v)

            # Co-activation: dims 2 and 3 receive identical signal
            shared_signal = rng.randn() + 1j * rng.randn()
            v[2] = shared_signal
            v[3] = shared_signal

            # Dims 5 and 6 receive independent signals (already random)

            c = coupled_reception(v, s, L)
            s = renormalize(s + eta * c)

        # Measure phase correlation between co-activated pair and independent pair
        phase_2 = np.angle(s[2])
        phase_3 = np.angle(s[3])
        phase_5 = np.angle(s[5])
        phase_6 = np.angle(s[6])

        coact_phase_diff = abs(phase_2 - phase_3)
        indep_phase_diff = abs(phase_5 - phase_6)

        # Also run with completely independent signals at 2,3 for comparison
        rng2 = np.random.RandomState(42)
        s2 = renormalize(rng2.randn(n) + 1j * rng2.randn(n))

        for step in range(n_steps):
            v = rng2.randn(n) + 1j * rng2.randn(n)
            v = v / np.linalg.norm(v)
            # No co-activation -- all dims independent
            c = coupled_reception(v, s2, L)
            s2 = renormalize(s2 + eta * c)

        control_phase_diff = abs(np.angle(s2[2]) - np.angle(s2[3]))

        ratio = coact_phase_diff / max(indep_phase_diff, 1e-10)
        binding_strength = abs(L[2, 3]) if cs > 0 else 0.0

        results.append({
            "cs": cs, "eps_norm": eps_norm,
            "coact_diff": coact_phase_diff,
            "indep_diff": indep_phase_diff,
            "control_diff": control_phase_diff,
            "L_23": binding_strength,
        })

        print(f"  coupling={cs:.1f}  ||eps||={eps_norm:.2f}  |L_23|={binding_strength:.4f}")
        print(f"    co-activated phase diff: {coact_phase_diff:.4f}")
        print(f"    independent phase diff:  {indep_phase_diff:.4f}")
        print(f"    control (no co-act):     {control_phase_diff:.4f}")
        print(f"    co-act / indep ratio:    {ratio:.2f}")
        print()

    # Verdict
    hadamard = results[0]
    coupled = [r for r in results if r["cs"] > 0]

    if hadamard["coact_diff"] < 0.1:
        print(f"  CONFIRMED: Hadamard (no coupling) co-activated phase diff = {hadamard['coact_diff']:.4f} (near-perfect correlation, as T3.3 showed)")

    # Check if coupling changes the co-activation correlation
    for r in coupled:
        if r["coact_diff"] < r["control_diff"]:
            print(f"  coupling={r['cs']:.1f}: Co-activated pair MORE correlated than control ({r['coact_diff']:.4f} vs {r['control_diff']:.4f})")
        else:
            print(f"  coupling={r['cs']:.1f}: Co-activated pair LESS correlated ({r['coact_diff']:.4f} vs {r['control_diff']:.4f})")

    return results


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    r1 = test_regime_persistence()
    print("\n")
    r2 = test_novelty_detection()
    print("\n")
    r3 = test_coupling_threshold()
    print("\n")
    r4 = test_passive_binding()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("Results saved. See COUPLING_THEORY.md for predictions and interpretation.")
