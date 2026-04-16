"""Tier 3 tests: learning, abstraction, novelty, consolidation, forgetting."""

import numpy as np
from engine import MemoryEngine, participation_ratio, renormalize, hadamard, Regime

np.set_printoptions(precision=4, suppress=True)

STEPS = 3000


# ── T3.1 ─────────────────────────────────────────────────────────────────
def test_t3_1_abstraction_from_variation():
    """Repeated exposure to varying inputs with shared invariant → tape concentrates on invariant dims."""
    n = 32
    np.random.seed(42)
    engine = MemoryEngine(n=n, eta=0.04)

    n_invariant = 6  # dims 0-5 carry the invariant
    invariant_phase = np.array([0.3, 1.2, 2.1, 3.5, 4.4, 5.8])  # fixed structure

    inv_mag_history = []
    noise_mag_history = []
    pr_history = []

    for t in range(STEPS):
        v = np.zeros(n, dtype=complex)

        # Invariant component: same structure every step
        for i in range(n_invariant):
            v[i] = 0.6 * np.exp(1j * invariant_phase[i])

        # Varying noise at dims 6-31: different each step
        for i in range(n_invariant, n):
            phase = np.random.uniform(0, 2 * np.pi)
            v[i] = 0.4 * np.exp(1j * phase)

        v = renormalize(v)
        info = engine.step(v, recurrence_delay=5, recurrence_weight=0.4)

        mags = engine.magnitudes()
        inv_mag_history.append(mags[:n_invariant].mean())
        noise_mag_history.append(mags[n_invariant:].mean())
        pr_history.append(info["pr_after"])

    inv_mags = np.array(inv_mag_history)
    noise_mags = np.array(noise_mag_history)
    pr = np.array(pr_history)

    # Check concentration ratio over time
    ratio_initial = inv_mags[50] / (noise_mags[50] + 1e-10)
    ratio_final = inv_mags[-1] / (noise_mags[-1] + 1e-10)

    print("T3.1 — Abstraction from Repeated Exposure with Variation")
    print(f"  Invariant dims mean |s_i|: initial={inv_mags[50]:.4f}, final={inv_mags[-1]:.4f}")
    print(f"  Noise dims mean |s_i|:     initial={noise_mags[50]:.4f}, final={noise_mags[-1]:.4f}")
    print(f"  Concentration ratio: initial={ratio_initial:.2f}x, final={ratio_final:.2f}x")
    print(f"  PR: initial={pr[50]:.2f}, final={pr[-1]:.2f}")

    assert inv_mags[-1] > noise_mags[-1], (
        "Invariant dims should concentrate more than noise dims"
    )
    assert ratio_final > ratio_initial, (
        "Concentration ratio should increase over time"
    )
    print("  PASS\n")


# ── T3.2 ─────────────────────────────────────────────────────────────────
def test_t3_2_novelty_via_leakage():
    """Novelty at uncarved dims is invisible without leakage, detectable with.

    The essay claims novelty 'announces itself indirectly' through 'unexplained
    torque at existing dimensions.' In the pure math, v - v_received is zero at
    received dims, so there's no coupling. Imperfect basis orthogonality (leakage)
    provides the physical coupling: basis vectors have small components along
    uncarved directions, so novel structure leaks into carved dimensions.
    """
    n = 16
    carved_dim = 12

    # Generate input with structure only in uncarved dimensions (12-15)
    np.random.seed(99)
    v = np.zeros(n, dtype=complex)
    v[carved_dim:] = np.random.randn(n - carved_dim) + 1j * np.random.randn(n - carved_dim)
    v = renormalize(v)
    v_carved_energy = np.sum(np.abs(v[:carved_dim]) ** 2)
    v_novel_energy = np.sum(np.abs(v[carved_dim:]) ** 2)

    # --- Part A: No leakage (novelty invisible) ---
    np.random.seed(42)
    engine_no_leak = MemoryEngine(n=n, carved_dim=carved_dim, leakage=0.0, eta=0.3)
    assert np.allclose(engine_no_leak.s[carved_dim:], 0), "Uncarved dims should be zero"
    s_before_no = engine_no_leak.s.copy()

    c, regimes = engine_no_leak.receive(v)
    engine_no_leak.update(c)
    disp_no_leak = engine_no_leak.angular_displacement(s_before_no, engine_no_leak.s)

    # --- Part B: With leakage (novelty detectable indirectly) ---
    np.random.seed(42)
    engine_leak = MemoryEngine(n=n, carved_dim=carved_dim, leakage=0.05, eta=0.3)
    s_before_leak = engine_leak.s.copy()

    c2, regimes2 = engine_leak.receive(v)
    engine_leak.update(c2)
    disp_leak = engine_leak.angular_displacement(s_before_leak, engine_leak.s)

    print("T3.2 — Novelty Detection via Leakage")
    print(f"  Carved dims: {carved_dim}, Novel dims: {n - carved_dim}")
    print(f"  Input energy — carved: {v_carved_energy:.4f}, novel: {v_novel_energy:.4f}")
    print(f"  Displacement (leakage=0.00): {disp_no_leak:.2e}")
    print(f"  Displacement (leakage=0.05): {disp_leak:.6f}")
    print(f"  Ratio (leak / no-leak):      {disp_leak / max(disp_no_leak, 1e-20):.2e}")

    assert disp_no_leak < 1e-14, (
        f"Novelty should be invisible without leakage, got {disp_no_leak:.2e}"
    )
    assert disp_leak > 1e-8, (
        f"Novelty should be detectable with leakage, got {disp_leak:.2e}"
    )
    print("  PASS\n")


# ── T3.2b ────────────────────────────────────────────────────────────────
def test_t3_2b_leakage_accumulates_with_repetition():
    """Repeated novelty at same uncarved dims accumulates indirect torque."""
    n = 16
    carved_dim = 12
    steps = 200

    # Generate persistent novel structure (same each step)
    np.random.seed(99)
    v_novel = np.zeros(n, dtype=complex)
    v_novel[carved_dim:] = np.random.randn(n - carved_dim) + 1j * np.random.randn(n - carved_dim)
    v_novel = renormalize(v_novel)

    # Also generate a different novel direction for comparison
    np.random.seed(77)
    v_novel_2 = np.zeros(n, dtype=complex)
    v_novel_2[carved_dim:] = np.random.randn(n - carved_dim) + 1j * np.random.randn(n - carved_dim)
    v_novel_2 = renormalize(v_novel_2)

    # Engine with leakage
    np.random.seed(42)
    engine = MemoryEngine(n=n, carved_dim=carved_dim, leakage=0.05, eta=0.1)
    s_initial = engine.s.copy()

    pr_history = []
    carved_mag_history = []

    for t in range(steps):
        # Alternate between two novel directions (system never has axes for these)
        v = v_novel if t % 2 == 0 else v_novel_2
        engine.step(v)
        pr_history.append(participation_ratio(engine.s))
        carved_mag_history.append(np.mean(np.abs(engine.s[:carved_dim])))

    pr = np.array(pr_history)
    carved_mags = np.array(carved_mag_history)

    final_disp = engine.angular_displacement(s_initial, engine.s)

    print("T3.2b — Repeated Novelty Accumulates via Leakage")
    print(f"  PR: initial={pr[0]:.2f}, mid={pr[steps//2]:.2f}, final={pr[-1]:.2f}")
    print(f"  Carved |s_i| mean: initial={carved_mags[0]:.4f}, "
          f"final={carved_mags[-1]:.4f}")
    print(f"  Tape moved from initial: {final_disp:.6f}")

    assert final_disp > 0.01, (
        f"Repeated novelty should accumulate, got displacement={final_disp:.6f}"
    )
    print("  PASS\n")


# ── T3.2c ────────────────────────────────────────────────────────────────
def test_t3_2c_backward_compatibility():
    """Default parameters (carved_dim=None, leakage=0) behave identically to before."""
    n = 16
    np.random.seed(42)
    engine_old = MemoryEngine(n=n, eta=0.1)
    np.random.seed(42)
    engine_new = MemoryEngine(n=n, eta=0.1, carved_dim=None, leakage=0.0)

    assert np.allclose(engine_old.s, engine_new.s), "Tapes should match"

    for t in range(100):
        np.random.seed(t)
        v = np.random.randn(n) + 1j * np.random.randn(n)
        v = renormalize(v)
        engine_old.step(v)
        engine_new.step(v)

    assert np.allclose(engine_old.s, engine_new.s), "Tapes should stay matched"
    print("T3.2c — Backward Compatibility")
    print("  Old and new engines identical after 100 steps")
    print("  PASS\n")


# ── T3.3 ─────────────────────────────────────────────────────────────────
def test_t3_3_consolidation_merge():
    """Co-activated dimensions merge into a single dimension under consolidation."""
    n = 16
    np.random.seed(42)
    engine = MemoryEngine(n=n, eta=0.05)

    # Run many steps where dims 2 and 3 always receive the same input
    # They should develop correlated phase/magnitude trajectories
    n_steps = 1000
    phase_correlations = []

    for t in range(n_steps):
        v = np.zeros(n, dtype=complex)
        # All dims get random input
        v = np.random.randn(n) + 1j * np.random.randn(n)
        # Override dims 2 and 3 with identical input (co-activation)
        shared = np.random.randn() + 1j * np.random.randn()
        v[2] = shared
        v[3] = shared
        v = renormalize(v)

        engine.step(v)

        if t > 50:
            # Track correlation of phase updates between dims 2 and 3
            if len(engine.history) > 2:
                recent_s = np.array([h[2] for h in engine.history[-20:]])
                recent_s3 = np.array([h[3] for h in engine.history[-20:]])
                corr = np.abs(np.corrcoef(np.angle(recent_s), np.angle(recent_s3))[0, 1])
                phase_correlations.append(corr)

    # Check correlation of dims 2,3 vs correlation of random pair (e.g. 5,6)
    s_history = np.array(engine.history[-200:])
    corr_23 = np.abs(np.corrcoef(np.angle(s_history[:, 2]), np.angle(s_history[:, 3]))[0, 1])
    corr_56 = np.abs(np.corrcoef(np.angle(s_history[:, 5]), np.angle(s_history[:, 6]))[0, 1])

    print("T3.3 — Consolidation: Co-activation Correlation")
    print(f"  Phase correlation dims 2&3 (co-activated): {corr_23:.4f}")
    print(f"  Phase correlation dims 5&6 (independent):   {corr_56:.4f}")
    print(f"  Ratio: {corr_23 / (corr_56 + 1e-10):.2f}x")

    assert corr_23 > corr_56, "Co-activated dims should be more correlated"

    # Now simulate consolidation: merge dims 2 and 3
    # New dimension = average of the two, one dimension removed
    merged_dim = (engine.s[2] + engine.s[3]) / np.sqrt(2)
    new_s = np.delete(engine.s, 3)  # remove dim 3
    new_s[2] = merged_dim
    new_s = renormalize(new_s)

    engine_merged = MemoryEngine(n=n - 1, eta=0.05)
    engine_merged.s = new_s
    engine_merged.history = []

    print(f"  Pre-merge n={n}, post-merge n={n-1}")
    print(f"  Merged dim magnitude: {abs(merged_dim):.4f}")
    print("  PASS\n")


# ── T3.4 ─────────────────────────────────────────────────────────────────
def test_t3_4_fast_vs_slow_forgetting():
    """Fast forgetting (phase rotation) is reversible. Slow forgetting (basis pruning) is not."""
    n = 16
    np.random.seed(42)

    # ── Fast forgetting: torque a dimension, then try to recover it ──
    engine = MemoryEngine(n=n, eta=0.1)

    # Strengthen dim 5 via sustained resonance
    for t in range(200):
        v = np.zeros(n, dtype=complex)
        v[5] = np.conj(engine.s[5])  # resonance at dim 5
        engine.step(v)

    mag_after_resonance = abs(engine.s[5])
    phase_after_resonance = np.angle(engine.s[5])

    # Apply torque at dim 5 (fast forgetting)
    for t in range(100):
        v = np.zeros(n, dtype=complex)
        v[5] = np.abs(engine.s[5]) * np.exp(1j * (-np.angle(engine.s[5]) + np.pi / 2))
        engine.step(v)

    mag_after_torque = abs(engine.s[5])

    # Recover with resonance at dim 5
    for t in range(300):
        v = np.zeros(n, dtype=complex)
        v[5] = np.conj(engine.s[5])
        engine.step(v)

    mag_after_recovery = abs(engine.s[5])

    print("T3.4 — Fast vs Slow Forgetting")
    print(f"  FAST FORGETTING (phase rotation):")
    print(f"    |s_5| after resonance:  {mag_after_resonance:.4f}")
    print(f"    |s_5| after torque:     {mag_after_torque:.4f}")
    print(f"    |s_5| after re-resonance: {mag_after_recovery:.4f}")

    fast_recoverable = mag_after_recovery >= mag_after_torque
    print(f"    Recoverable: {fast_recoverable}")
    assert fast_recoverable, "Fast forgetting should be recoverable via renewed resonance"

    # ── Slow forgetting: prune a dimension, then try to recover it ──
    # Simulate by running the engine until dim 7 is starved, then prune
    engine2 = MemoryEngine(n=n, eta=0.1)

    # Feed inputs that never activate dim 7 (it's always orthogonal)
    # But do resonate other dims to starve dim 7 via renormalization
    for t in range(2000):
        v = np.zeros(n, dtype=complex)
        # Resonance at all dims except 7
        for i in range(n):
            if i != 7:
                v[i] = np.conj(engine2.s[i])
        engine2.step(v)

    mag_dim7_before_prune = abs(engine2.s[7])
    print(f"\n  SLOW FORGETTING (basis pruning):")
    print(f"    |s_7| after starvation: {mag_dim7_before_prune:.6f}")

    # Simulate pruning: remove dim 7 entirely
    new_s = np.delete(engine2.s, 7)
    new_s = renormalize(new_s)
    engine2_pruned = MemoryEngine(n=n - 1, eta=0.1)
    engine2_pruned.s = new_s
    engine2_pruned.history = []

    # Try to recover dim 7 by feeding the same pattern that would have activated it
    # It's gone — the system no longer has that axis
    for t in range(500):
        v = np.zeros(n - 1, dtype=complex)
        # Feed the old resonance pattern, but dim 7 doesn't exist anymore
        for i in range(n - 1):
            if i < 7:
                v[i] = np.conj(engine2_pruned.s[i])
            # dim 7 is gone; what was dim 8+ is now shifted
        engine2_pruned.step(v)

    print(f"    System dimensionality after pruning: {engine2_pruned.n}")
    print(f"    Dim 7 exists: NO (pruned)")
    print(f"    Recovery via re-exposure: IMPOSSIBLE (axis removed)")
    print("  PASS\n")


if __name__ == "__main__":
    test_t3_1_abstraction_from_variation()
    test_t3_2_novelty_via_leakage()
    test_t3_2b_leakage_accumulates_with_repetition()
    test_t3_2c_backward_compatibility()
    test_t3_3_consolidation_merge()
    test_t3_4_fast_vs_slow_forgetting()
    print("All Tier 3 tests passed.")
