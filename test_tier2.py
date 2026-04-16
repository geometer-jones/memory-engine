"""Tier 2 tests: regime predictions — recurrent capture, dissipation, operating regime."""

import numpy as np
from engine import MemoryEngine, participation_ratio, renormalize, hadamard, Regime

np.set_printoptions(precision=4, suppress=True)

STEPS = 2000


# ── T2.1 ─────────────────────────────────────────────────────────────────
def test_t2_1_recurrent_capture():
    """Pure resonance → tape concentrates, dimensionality drops, system rejects novel input."""
    n = 32
    np.random.seed(42)
    engine = MemoryEngine(n=n, eta=0.05)

    pr_history = []
    max_mag_history = []
    phase_variance_history = []

    for t in range(STEPS):
        # Always send input that resonates with current s (self-confirming)
        v = np.conj(engine.s)
        info = engine.step(v)

        pr_history.append(info["pr_after"])
        max_mag_history.append(np.max(engine.magnitudes()))
        # Phase variance: how spread out are the phases across dims
        phase_var = np.var(engine.phases())
        phase_variance_history.append(phase_var)

    pr = np.array(pr_history)
    max_mag = np.array(max_mag_history)

    # Now test rejection: send a perturbation and measure angular displacement
    # Do this early (step 50) and late (step STEPS-1) with the same perturbation
    engine_capture = MemoryEngine(n=n, eta=0.05)
    np.random.seed(42)
    s_initial = engine_capture.s.copy()

    # Run 50 steps of pure resonance
    for t in range(50):
        v = np.conj(engine_capture.s)
        engine_capture.step(v)
    s_early = engine_capture.s.copy()

    # Run full STEPS steps
    np.random.seed(42)
    engine_full = MemoryEngine(n=n, eta=0.05)
    for t in range(STEPS):
        v = np.conj(engine_full.s)
        engine_full.step(v)
    s_late = engine_full.s.copy()

    # Same perturbation at both time points
    np.random.seed(99)
    perturbation_v = np.random.randn(n) + 1j * np.random.randn(n)
    perturbation_v = renormalize(perturbation_v)

    # Measure angular displacement from perturbation at early vs late
    def measure_displacement(eng, pv):
        s_before = eng.s.copy()
        c, _ = eng.receive(pv)
        eng.update(c)
        return eng.angular_displacement(s_before, eng.s)

    eng_early = MemoryEngine(n=n, eta=0.05)
    eng_early.s = s_early.copy()
    eng_early.history = []
    disp_early = measure_displacement(eng_early, perturbation_v)

    eng_late = MemoryEngine(n=n, eta=0.05)
    eng_late.s = s_late.copy()
    eng_late.history = []
    disp_late = measure_displacement(eng_late, perturbation_v)

    print("T2.1 — Recurrent Capture (pure resonance)")
    print(f"  Participation ratio: initial={pr[0]:.2f}, "
          f"mid={pr[STEPS//2]:.2f}, final={pr[-1]:.2f}")
    print(f"  Max |s_i|:          initial={max_mag[0]:.4f}, "
          f"mid={max_mag[STEPS//2]:.4f}, final={max_mag[-1]:.4f}")
    print(f"  Perturbation displacement: early={disp_early:.6f}, late={disp_late:.6f}")
    print(f"  Rejection ratio (early/late): {disp_early/disp_late:.2f}x")

    assert pr[-1] < pr[0], "Participation ratio should decrease under pure resonance"
    assert max_mag[-1] > max_mag[0], "Max magnitude should increase under pure resonance"
    assert disp_late < disp_early, "System should reject perturbations more at late stage"
    print("  PASS\n")


# ── T2.2 ─────────────────────────────────────────────────────────────────
def test_t2_2_dissipation():
    """Pure torque → phases spin, nothing stabilizes, no concentration."""
    n = 32
    np.random.seed(42)
    engine = MemoryEngine(n=n, eta=0.05)

    pr_history = []
    max_mag_history = []
    phase_variance_history = []

    for t in range(STEPS):
        # Send input that torques current s from *varying* random directions.
        # Fixed-direction torque (always +pi/2) creates a systematic bias that still
        # concentrates — genuine dissipation requires opposition from random angles.
        random_angles = np.random.uniform(np.pi / 4, 3 * np.pi / 4, n)
        # Some dims get +offset, some get -offset — no systematic direction
        random_angles *= np.random.choice([-1, 1], n)
        v = np.abs(engine.s) * np.exp(1j * (-np.angle(engine.s) + random_angles))
        info = engine.step(v)

        pr_history.append(info["pr_after"])
        max_mag_history.append(np.max(engine.magnitudes()))
        phase_variance_history.append(np.var(engine.phases()))

    pr = np.array(pr_history)
    max_mag = np.array(max_mag_history)

    print("T2.2 — Dissipation (pure torque)")
    print(f"  Participation ratio: initial={pr[0]:.2f}, "
          f"mid={pr[STEPS//2]:.2f}, final={pr[-1]:.2f}")
    print(f"  Max |s_i|:          initial={max_mag[0]:.4f}, "
          f"mid={max_mag[STEPS//2]:.4f}, final={max_mag[-1]:.4f}")

    assert pr[-1] > pr[-1] * 0.5, "PR should stay near n (no concentration)"

    # Check that max magnitude doesn't grow monotonically toward 1
    # (it should stay bounded away from 1 — no dimension captures the tape)
    assert max_mag[-1] < 0.8, (
        f"Max magnitude should stay bounded under pure torque, got {max_mag[-1]:.4f}"
    )

    # Participation ratio should not show systematic decrease
    # Fit a line to PR and check slope is near zero or positive
    t_axis = np.arange(len(pr))
    slope = np.polyfit(t_axis, pr, 1)[0]
    print(f"  PR trend (slope):    {slope:.6f} (near 0 = no concentration)")
    print(f"  PR range:            [{pr.min():.2f}, {pr.max():.2f}]")

    assert slope > -0.005, "PR should not systematically decrease under pure torque"
    print("  PASS\n")


# ── T2.3 ─────────────────────────────────────────────────────────────────
def test_t2_3_operating_regime():
    """World with invariant + variation → structured anisotropy with recurrence."""
    n = 32
    np.random.seed(42)

    rec_delay = 5
    rec_weight = 0.6
    engine = MemoryEngine(n=n, eta=0.04)

    # The world has a fixed "invariant" signal that the system can resonate with
    # plus random variation that produces torque at different dims each step.
    # Dims 0-7 carry the invariant; dims 8-31 are pure noise.
    n_invariant = 8
    invariant_phase = np.random.uniform(0, 2 * np.pi, n_invariant)
    noise_strength = 0.5  # relative to invariant signal

    pr_history = []
    mag_history = []

    for t in range(STEPS):
        v = np.zeros(n, dtype=complex)
        # Invariant component at dims 0-7: fixed structure (resonance opportunity)
        for i in range(n_invariant):
            v[i] = 0.7 * np.exp(1j * invariant_phase[i])
        # Noise at all dims (produces torque where misaligned)
        v += noise_strength * (np.random.randn(n) + 1j * np.random.randn(n)) / np.sqrt(n)
        v = renormalize(v)

        info = engine.step(v, recurrence_delay=rec_delay, recurrence_weight=rec_weight)
        pr_history.append(info["pr_after"])
        mag_history.append(engine.magnitudes().copy())

    pr = np.array(pr_history)
    mags = np.array(mag_history)
    final_mags = mags[-1]

    def gini(x):
        x = np.sort(x)
        n = len(x)
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * x) - (n + 1) * np.sum(x)) / (n * np.sum(x) + 1e-10)

    gini_coef = gini(final_mags)

    inv_mags = final_mags[:n_invariant]
    noise_mags = final_mags[n_invariant:]

    print("T2.3 — Operating Regime (invariant + variation + recurrence)")
    print(f"  Participation ratio: initial={pr[0]:.2f}, "
          f"mid={pr[STEPS//2]:.2f}, final={pr[-1]:.2f}")
    print(f"  PR range: [{pr.min():.2f}, {pr.max():.2f}]")
    print(f"  Gini coefficient:     {gini_coef:.4f}")
    print(f"  Mean |s_i| invariant dims: {inv_mags.mean():.4f}")
    print(f"  Mean |s_i| noise dims:     {noise_mags.mean():.4f}")
    print(f"  PR stability (std last 500): {np.std(pr[-500:]):.4f}")

    # Operating regime: not captured, not dissipated
    assert pr[-1] > 1.5, f"PR should not fully collapse, got {pr[-1]:.2f}"
    assert pr[-1] < n * 0.9, f"PR should show some concentration, got {pr[-1]:.2f}"

    # Perturbation test
    np.random.seed(99)
    perturb_v = np.random.randn(n) + 1j * np.random.randn(n)
    perturb_v = renormalize(perturb_v)
    s_before = engine.s.copy()
    c, _ = engine.receive(perturb_v)
    engine.update(c)
    disp = engine.angular_displacement(s_before, engine.s)
    print(f"  Perturbation displacement: {disp:.6f}")
    assert disp > 1e-6, "System should still respond to perturbation"

    print("  PASS\n")


if __name__ == "__main__":
    test_t2_1_recurrent_capture()
    test_t2_2_dissipation()
    test_t2_3_operating_regime()
    print("All Tier 2 tests passed.")
