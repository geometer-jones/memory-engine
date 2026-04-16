"""Tier 4 tests: reciprocation and recurrence — thick vs thin self-reception, phase diagram."""

import numpy as np
from engine import MemoryEngine, participation_ratio, renormalize, hadamard, Regime

np.set_printoptions(precision=4, suppress=True)

STEPS = 2000


# ── T4.1 ─────────────────────────────────────────────────────────────────
def test_t4_1_thick_vs_thin_self_reception():
    """Self-torque fraction increases with recurrence delay (more drift → more opposition)."""
    n = 32
    np.random.seed(42)

    # Run the engine for many steps with varied input so the tape drifts
    engine = MemoryEngine(n=n, eta=0.04)

    for t in range(500):
        v = np.random.randn(n) + 1j * np.random.randn(n)
        v = renormalize(v)
        engine.step(v)

    # Now compute self-reception at various delays
    delays = list(range(1, min(200, len(engine.history)), 5))
    self_torque_fractions = []
    self_resonance_fractions = []
    mean_c_self_mag = []

    for delay in delays:
        c_self, regimes = engine.self_reception(delay)
        n_torque = sum(1 for r in regimes if r == Regime.TORQUE)
        n_resonance = sum(1 for r in regimes if r == Regime.RESONANCE)
        self_torque_fractions.append(n_torque / n)
        self_resonance_fractions.append(n_resonance / n)
        mean_c_self_mag.append(np.mean(np.abs(c_self)))

    # Also run a forward test: run fresh engine, measure self-torque at each step
    engine2 = MemoryEngine(n=n, eta=0.04)
    torque_frac_over_time = {d: [] for d in [1, 5, 20, 50]}

    for t in range(STEPS):
        v = np.random.randn(n) + 1j * np.random.randn(n)
        v = renormalize(v)
        engine2.step(v)

        for d in torque_frac_over_time:
            if len(engine2.history) >= d:
                _, regimes = engine2.self_reception(d)
                n_torque = sum(1 for r in regimes if r == Regime.TORQUE)
                torque_frac_over_time[d].append(n_torque / n)

    print("T4.1 — Thick vs Thin Self-Reception")
    print(f"\n  Retrospective analysis (single snapshot, varying delay):")
    print(f"  {'Delay':>6} {'Torque%':>8} {'Resonance%':>11} {'Mean|c_self|':>12}")
    for i, delay in enumerate(delays[::4]):
        idx = i * 4
        if idx < len(delays):
            d = delays[idx]
            print(f"  {d:>6} {self_torque_fractions[idx]*100:>7.1f}% "
                  f"{self_resonance_fractions[idx]*100:>10.1f}% "
                  f"{mean_c_self_mag[idx]:>11.4f}")

    print(f"\n  Forward analysis (mean self-torque fraction at fixed delays):")
    for d in [1, 5, 20, 50]:
        if torque_frac_over_time[d]:
            mean_torque = np.mean(torque_frac_over_time[d])
            print(f"    delay={d:>3}: mean self-torque fraction = {mean_torque:.3f}")

    # Self-torque fraction should generally increase with delay
    # (more drift → more opposition), at least up to some point
    short_delay_torque = np.mean(torque_frac_over_time[1])
    longer_delay_torque = np.mean(torque_frac_over_time[20])

    print(f"\n  delay=1 mean torque: {short_delay_torque:.3f}")
    print(f"  delay=20 mean torque: {longer_delay_torque:.3f}")
    assert longer_delay_torque > short_delay_torque, (
        "Longer delay should produce more self-torque (thicker recurrence)"
    )
    print("  PASS\n")


# ── T4.2 ─────────────────────────────────────────────────────────────────
def test_t4_2_reciprocation_phase_diagram():
    """Map the operating region in the speed × directness × breadth space.

    Speed: recurrence delay (shorter = faster cycling)
    Directness: magnitude of self-reception perturbation
    Breadth: fraction of dimensions engaged in self-reception

    Scan parameter space, measure PR stability and accommodation capacity.
    """
    n = 24
    np.random.seed(42)
    n_steps = 1500

    # Test configurations: (delay, rec_weight, breadth_fraction)
    configs = []
    for delay in [2, 10, 50]:
        for weight in [0.2, 0.8, 2.0]:
            for breadth in [0.25, 0.75, 1.0]:
                configs.append((delay, weight, breadth))

    results = []

    for delay, weight, breadth in configs:
        np.random.seed(42)
        engine = MemoryEngine(n=n, eta=0.04)
        n_active = max(1, int(breadth * n))

        pr_history = []

        for t in range(n_steps):
            v = np.random.randn(n) + 1j * np.random.randn(n)
            v = renormalize(v)

            # Custom recurrence: only engage `breadth` fraction of dimensions
            if len(engine.history) >= delay:
                s_past = engine.history[-delay]
                # Zero out non-breadth dims in the past tape
                mask = np.zeros(n, dtype=complex)
                active_dims = np.random.choice(n, n_active, replace=False)
                mask[active_dims] = s_past[active_dims]
                c_self = hadamard(mask, engine.s)

                c_world, _ = engine.receive(v)
                c_total = c_world + weight * c_self
                s_before = engine.s.copy()
                engine.update(c_total)
                engine.history.append(s_before)
            else:
                info = engine.step(v)
                pr_history.append(info["pr_after"])
                continue

            pr_history.append(participation_ratio(engine.s))

        pr = np.array(pr_history[-500:])  # last 500 steps
        pr_mean = pr.mean()
        pr_std = pr.std()
        final_max_mag = np.max(engine.magnitudes())

        # Accommodation: apply perturbation, measure displacement
        np.random.seed(99)
        perturb = np.random.randn(n) + 1j * np.random.randn(n)
        perturb = renormalize(perturb)
        s_before = engine.s.copy()
        c, _ = engine.receive(perturb)
        engine.update(c)
        disp = engine.angular_displacement(s_before, engine.s)

        results.append({
            "delay": delay, "weight": weight, "breadth": breadth,
            "pr_mean": pr_mean, "pr_std": pr_std,
            "max_mag": final_max_mag, "displacement": disp,
        })

    print("T4.2 — Reciprocation Phase Diagram")
    print(f"\n  {'Delay':>5} {'Weight':>6} {'Breadth':>7} | "
          f"{'PR_mean':>7} {'PR_std':>6} {'MaxMag':>6} {'Disp':>8} | Regime")
    print("  " + "-" * 75)

    for r in results:
        # Classify regime
        if r["max_mag"] > 0.99:
            regime = "CAPTURED"
        elif r["pr_std"] > 3.0:
            regime = "unstable"
        elif r["displacement"] < 1e-5:
            regime = "rigid"
        elif r["pr_mean"] > n * 0.7:
            regime = "dissipating"
        else:
            regime = "operating"

        print(f"  {r['delay']:>5} {r['weight']:>6.1f} {r['breadth']:>7.2f} | "
              f"{r['pr_mean']:>7.2f} {r['pr_std']:>6.2f} {r['max_mag']:>6.3f} "
              f"{r['displacement']:>8.5f} | {regime}")

    # Count regimes
    n_operating = sum(1 for r in results if
        r["max_mag"] < 0.99 and
        r["displacement"] > 1e-5 and
        r["pr_mean"] < n * 0.7)

    print(f"\n  Operating regime configs: {n_operating}/{len(results)}")
    print(f"  Captured configs: {sum(1 for r in results if r['max_mag'] > 0.99)}/{len(results)}")

    assert n_operating > 0, "At least some configs should find the operating regime"
    print("  PASS\n")


if __name__ == "__main__":
    test_t4_1_thick_vs_thin_self_reception()
    test_t4_2_reciprocation_phase_diagram()
    print("All Tier 4 tests completed.")
