"""Tier 1 tests: mechanical correctness of the memory engine."""

import numpy as np
from engine import (
    MemoryEngine,
    classify_component,
    classify_reception,
    hadamard,
    participation_ratio,
    renormalize,
    Regime,
)

np.set_printoptions(precision=4, suppress=True)


def make_resonant_input(s, dims=None):
    """Input such that Hadamard with s gives Re(c_i) > 0 at target dims.

    c_i = v_i * s_i. For Re > 0, phase(v_i) ≈ -phase(s_i), i.e. conjugate.
    """
    v = np.zeros_like(s)
    target = dims if dims is not None else range(len(s))
    for i in target:
        v[i] = np.conj(s[i])  # conjugate => c_i = |s_i|^2 (real positive)
    return v


def make_torque_input(s, dims=None):
    """Input such that Hadamard with s gives torque at target dims.

    c_i = v_i * s_i. For Re < 0 or Im dominant, offset phase by ~pi/2 from
    the conjugate direction.
    """
    v = np.zeros_like(s)
    target = dims if dims is not None else range(len(s))
    for i in target:
        # conjugate phase + pi/2 => purely imaginary Hadamard product
        v[i] = np.abs(s[i]) * np.exp(1j * (-np.angle(s[i]) + np.pi / 2))
    return v


def make_orthogonal_input(s, dims=None):
    """Input with near-zero overlap at specified dimensions."""
    v = np.zeros_like(s)
    return v  # zero vector is maximally orthogonal


# ── T1.1 ─────────────────────────────────────────────────────────────────
def test_t1_1_three_regimes():
    """Three regimes produce qualitatively different dynamics."""
    n = 16
    np.random.seed(42)
    engine = MemoryEngine(n=n, eta=0.3)

    # Partition dimensions: 0-4 resonance, 5-9 torque, 10-15 orthogonality
    v = np.zeros(n, dtype=complex)
    for i in range(5):
        # Conjugate of s_i => c_i = |s_i|^2 (real positive) => resonance
        v[i] = np.conj(engine.s[i])
    for i in range(5, 10):
        # Conjugate phase + pi/2 => c_i purely imaginary => torque
        v[i] = np.abs(engine.s[i]) * np.exp(1j * (-np.angle(engine.s[i]) + np.pi / 2))
    # dims 10-15: zero (orthogonality)

    s_before = engine.s.copy()
    phases_before = engine.phases().copy()
    mags_before = engine.magnitudes().copy()

    c, regimes = engine.receive(v)
    engine.update(c)

    phases_after = engine.phases()
    mags_after = engine.magnitudes()

    # Verify regime classification
    res_dims = [i for i, r in enumerate(regimes) if r == Regime.RESONANCE]
    tor_dims = [i for i, r in enumerate(regimes) if r == Regime.TORQUE]
    ort_dims = [i for i, r in enumerate(regimes) if r == Regime.ORTHOGONALITY]

    print("T1.1 — Three Regimes")
    print(f"  Resonance dims:  {res_dims}")
    print(f"  Torque dims:     {tor_dims}")
    print(f"  Orthogonality dims: {ort_dims}")

    # Magnitude growth: resonance > torque > orthogonality
    res_mag_growth = np.mean(mags_after[res_dims] - mags_before[res_dims])
    tor_mag_growth = np.mean(mags_after[tor_dims] - mags_before[tor_dims])
    ort_mag_growth = np.mean(mags_after[ort_dims] - mags_before[ort_dims])

    print(f"\n  Magnitude growth (resonance): {res_mag_growth:+.6f}")
    print(f"  Magnitude growth (torque):    {tor_mag_growth:+.6f}")
    print(f"  Magnitude growth (orthogonal):{ort_mag_growth:+.6f}")

    assert res_mag_growth > tor_mag_growth, "Resonance should grow more than torque"
    assert res_mag_growth > ort_mag_growth, "Resonance should grow more than orthogonality"

    # Torque dimensions should show meaningful phase drift; orthogonality should
    # show zero drift. Resonance can also drift (adding a real scalar to a complex
    # s_i shifts phase toward the real axis) — the essay's claim is about the
    # *tendency*: resonance scales magnitude, torque rotates phase.
    phase_drift = np.abs(phases_after - phases_before)
    res_phase = np.mean(phase_drift[res_dims])
    tor_phase = np.mean(phase_drift[tor_dims])
    ort_phase = np.mean(phase_drift[ort_dims])

    print(f"\n  Phase drift (resonance): {res_phase:.6f}")
    print(f"  Phase drift (torque):    {tor_phase:.6f}")
    print(f"  Phase drift (orthogonal):{ort_phase:.6f}")

    # Core qualitative checks: torque dims rotate, orth dims don't
    assert tor_phase > 0, "Torque should produce non-zero phase drift"
    assert ort_phase < 1e-10, "Orthogonality should preserve phase"
    # Resonance dims grow magnitude; torque dims don't
    assert res_mag_growth > 0, "Resonance should grow magnitude"
    assert tor_mag_growth < res_mag_growth, "Torque dims should grow less than resonance"
    print("  PASS\n")


# ── T1.2 ─────────────────────────────────────────────────────────────────
def test_t1_2_emergent_rigidity():
    """High-magnitude dimensions resist rotation more than low-magnitude ones."""
    n = 16
    np.random.seed(42)
    engine = MemoryEngine(n=n, eta=0.3)

    # Manually set s: one dominant dimension, rest small
    s = np.ones(n, dtype=complex) * 0.01
    s[0] = 0.999
    s += 1j * np.random.randn(n) * 0.01
    s = renormalize(s)
    engine.s = s

    mags = engine.magnitudes()
    print("T1.2 — Emergent Rigidity")
    print(f"  |s_0| (dominant): {mags[0]:.4f}")
    print(f"  |s_1| (small):    {mags[1]:.4f}")

    # Apply same torque perturbation to both dims
    v = np.zeros(n, dtype=complex)
    for i in [0, 1]:
        # conjugate phase + pi => c_i = |s_i|^2 * exp(i*pi) = -|s_i|^2 => torque
        v[i] = np.abs(s[i]) * np.exp(1j * (-np.angle(s[i]) + np.pi))

    s_before = engine.s.copy()
    c, _ = engine.receive(v)
    engine.update(c)
    s_after = engine.s.copy()

    drift = engine.dimension_angular_displacement(s_before, s_after)

    print(f"  Phase drift at dim 0 (dominant): {drift[0]:.6f}")
    print(f"  Phase drift at dim 1 (small):    {drift[1]:.6f}")
    print(f"  Rigidity ratio (small/large):     {drift[1]/drift[0]:.2f}x")

    assert drift[1] > drift[0], "Small dimension should rotate more than dominant"
    print("  PASS\n")


# ── T1.3 ─────────────────────────────────────────────────────────────────
def test_t1_3_norm_preservation():
    """Renormalization preserves unit norm across many steps."""
    n = 16
    np.random.seed(42)
    engine = MemoryEngine(n=n, eta=0.2)

    norms = []
    for t in range(10000):
        v = np.random.randn(n) + 1j * np.random.randn(n)
        v = renormalize(v)
        engine.step(v)
        norms.append(np.linalg.norm(engine.s))

    norms = np.array(norms)
    print("T1.3 — Norm Preservation (10K steps)")
    print(f"  Mean norm:  {norms.mean():.15f}")
    print(f"  Max drift:  {np.max(np.abs(norms - 1.0)):.2e}")
    print(f"  All within 1e-10 of 1.0: {np.all(np.abs(norms - 1.0) < 1e-10)}")

    assert np.all(np.abs(norms - 1.0) < 1e-10), "Norm should stay at 1.0"
    print("  PASS\n")


# ── T1.4 ─────────────────────────────────────────────────────────────────
def test_t1_4_concentration_thins_others():
    """Scaling one dimension up scales others down via renormalization."""
    n = 16
    np.random.seed(42)
    engine = MemoryEngine(n=n, eta=0.1)

    # Start with near-uniform magnitude
    engine.s = renormalize(np.ones(n, dtype=complex) + 1j * np.zeros(n))

    mags_initial = engine.magnitudes().copy()

    # Apply sustained resonance at dimension 0 only
    for t in range(200):
        v = np.zeros(n, dtype=complex)
        v[0] = np.conj(engine.s[0])  # resonance at dim 0
        engine.step(v)

    mags_final = engine.magnitudes()

    print("T1.4 — Concentration Thins Others")
    print(f"  |s_0| before: {mags_initial[0]:.4f}, after: {mags_final[0]:.4f}")
    print(f"  Mean |s_1-15| before: {mags_initial[1:].mean():.4f}, "
          f"after: {mags_final[1:].mean():.4f}")

    assert mags_final[0] > mags_initial[0], "Dim 0 should grow"
    assert mags_final[1:].mean() < mags_initial[1:].mean(), "Other dims should shrink"

    ratio_before = mags_final[0] / mags_initial[0]
    ratio_others = mags_final[1:].mean() / mags_initial[1:].mean()
    print(f"  Dim 0 growth factor:    {ratio_before:.2f}x")
    print(f"  Others shrink factor:   {1/ratio_others:.2f}x")
    print("  PASS\n")


if __name__ == "__main__":
    test_t1_1_three_regimes()
    test_t1_2_emergent_rigidity()
    test_t1_3_norm_preservation()
    test_t1_4_concentration_thins_others()
    print("All Tier 1 tests passed.")
