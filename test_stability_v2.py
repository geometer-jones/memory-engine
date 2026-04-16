"""
Phase 3 Stability: Targeted retests for T2 and T3.

T2-fix: Measure abstraction SPEED (steps to reach concentration ratio > 10x)
        with weak invariant signal, aligned/misaligned/random coupling.
T3-fix: Self-regulation that reduces effective coupling as n grows.
"""

import numpy as np


def renormalize(s):
    norm = np.linalg.norm(s)
    return s / norm if norm > 1e-15 else s

def participation_ratio(s):
    mags_sq = np.abs(s) ** 2
    return float(np.sum(mags_sq) ** 2 / np.sum(mags_sq**2))

def coupled_reception(v, s, L):
    return (L @ v) * s

def make_gram_coupling(n, coupling_strength, seed=42):
    rng = np.random.RandomState(seed)
    eps = coupling_strength * (rng.randn(n, n) + 1j * rng.randn(n, n)) / np.sqrt(2 * n)
    eps = (eps + eps.conj().T) / 2
    np.fill_diagonal(eps, 0)
    G = np.eye(n, dtype=complex) + eps
    L = np.linalg.inv(G)
    return L, eps, G


# ── T2-fix: Abstraction Speed ────────────────────────────────────────────

def test_abstraction_speed():
    """Weak invariant signal (amplitude 0.15) + strong noise (amplitude 0.3).

    Measure steps to reach concentration ratio > 10x (invariant dims dominate).
    """
    n = 32
    eta = 0.1
    n_invariant = 8
    n_steps = 3000
    cs = 0.3

    print("=" * 70)
    print("T2-fix: ABSTRACTION SPEED BY COUPLING ALIGNMENT")
    print("=" * 70)
    print(f"n={n}, eta={eta}, invariant signal=0.15, noise=0.3, coupling={cs}")
    print()

    invariant_phase = 0.3
    conditions = {}

    for condition in ["aligned", "misaligned", "random"]:
        if condition == "random":
            L, eps, G = make_gram_coupling(n, cs, seed=42)
        elif condition == "aligned":
            v_inv = np.zeros(n, dtype=complex)
            for i in range(n_invariant):
                v_inv[i] = np.exp(1j * invariant_phase)
            alignment = np.outer(v_inv, v_inv.conj())
            eps = cs * alignment / np.linalg.norm(alignment, 'fro')
            np.fill_diagonal(eps, 0)
            G = np.eye(n, dtype=complex) + eps
            L = np.linalg.inv(G)
        elif condition == "misaligned":
            v_inv = np.zeros(n, dtype=complex)
            for i in range(n_invariant):
                v_inv[i] = np.exp(1j * invariant_phase)
            alignment = np.outer(v_inv, v_inv.conj())
            eps = -cs * alignment / np.linalg.norm(alignment, 'fro')
            np.fill_diagonal(eps, 0)
            G = np.eye(n, dtype=complex) + eps
            L = np.linalg.inv(G)

        rng = np.random.RandomState(42)
        s = renormalize(rng.randn(n) + 1j * rng.randn(n))

        conc_ratios = []
        abstraction_step = None

        for step in range(n_steps):
            v_signal = np.zeros(n, dtype=complex)
            for i in range(n_invariant):
                v_signal[i] = 0.15 * np.exp(1j * invariant_phase)
            v_noise = 0.3 * (rng.randn(n) + 1j * rng.randn(n))
            v = v_signal + v_noise

            c = coupled_reception(v, s, L)
            s = renormalize(s + eta * c)

            mags = np.abs(s)
            mean_inv = np.mean(mags[:n_invariant])
            mean_noise = np.mean(mags[n_invariant:])
            cr = mean_inv / max(mean_noise, 1e-10)
            conc_ratios.append(cr)

            if abstraction_step is None and cr > 10:
                abstraction_step = step

        # Track concentration ratio trajectory
        cr_100 = conc_ratios[99] if len(conc_ratios) > 99 else conc_ratios[-1]
        cr_500 = conc_ratios[499] if len(conc_ratios) > 499 else conc_ratios[-1]
        cr_1000 = conc_ratios[999] if len(conc_ratios) > 999 else conc_ratios[-1]
        cr_final = conc_ratios[-1]

        conditions[condition] = {
            "abstraction_step": abstraction_step,
            "cr_100": cr_100, "cr_500": cr_500,
            "cr_1000": cr_1000, "cr_final": cr_final,
        }

        abs_str = f"step {abstraction_step}" if abstraction_step else f"not in {n_steps} steps"
        print(f"  {condition:12s}: abstracted at {abs_str}")
        print(f"    CR trajectory: 100={cr_100:.1f}x  500={cr_500:.1f}x  1000={cr_1000:.1f}x  final={cr_final:.1f}x")

    print()

    # Verdict
    aligned_step = conditions["aligned"]["abstraction_step"] or n_steps + 1
    misaligned_step = conditions["misaligned"]["abstraction_step"] or n_steps + 1
    random_step = conditions["random"]["abstraction_step"] or n_steps + 1

    if aligned_step < misaligned_step and aligned_step < random_step:
        print("  CONFIRMED: Aligned coupling abstracts fastest")
    elif misaligned_step < aligned_step:
        print(f"  UNEXPECTED: Misaligned abstracts faster ({misaligned_step}) than aligned ({aligned_step})")
    else:
        print(f"  NOTE: No clear speed ordering (aligned={aligned_step}, random={random_step}, misaligned={misaligned_step})")

    return conditions


# ── T3-fix: Self-Regulation with Coupling Reduction ──────────────────────

def test_self_regulation_v2():
    """Self-regulation that reduces effective coupling as n grows.

    Key fix: cs is reduced as n grows to keep ||eps||_F below sqrt(n)/2.
    """
    n_initial = 32
    eta = 0.1
    n_steps = 3000
    cs_initial = 1.0

    print("=" * 70)
    print("T3-fix: SELF-REGULATION WITH COUPLING REDUCTION")
    print("=" * 70)
    print(f"n_initial={n_initial}, steps={n_steps}, cs_initial={cs_initial}")
    print(f"Strategy: grow n AND reduce cs to keep ||eps||_F < sqrt(n)/3")
    print()

    # --- Without regulation ---
    rng = np.random.RandomState(42)
    L_static, _, _ = make_gram_coupling(n_initial, cs_initial, seed=42)
    s_static = renormalize(rng.randn(n_initial) + 1j * rng.randn(n_initial))

    for step in range(n_steps):
        v = rng.randn(n_initial) + 1j * rng.randn(n_initial)
        v = v / np.linalg.norm(v)
        # Add weak invariant signal
        for i in range(8):
            v[i] += 0.15 * np.exp(1j * 0.3)
        c = coupled_reception(v, s_static, L_static)
        s_static = renormalize(s_static + eta * c)

    pr_static = participation_ratio(s_static)
    mags_static = np.abs(s_static)
    print(f"  No regulation:  PR={pr_static:.1f}  |s_inv|={np.mean(mags_static[:8]):.4f}  |s_noise|={np.mean(mags_static[8:]):.4f}")

    # --- With regulation ---
    rng2 = np.random.RandomState(42)
    n_current = n_initial
    cs_current = cs_initial
    L_reg, _, _ = make_gram_coupling(n_current, cs_current, seed=42)
    s_reg = renormalize(rng2.randn(n_current) + 1j * rng2.randn(n_current))

    pr_threshold_low = 3.0
    pr_threshold_high = 0.7 * n_current
    growth_events = 0
    pr_trajectory = []

    for step in range(n_steps):
        v = rng2.randn(n_current) + 1j * rng2.randn(n_current)
        v = v / np.linalg.norm(v)
        for i in range(min(8, n_current)):
            v[i] += 0.15 * np.exp(1j * 0.3)

        c = coupled_reception(v, s_reg, L_reg)
        s_reg = renormalize(s_reg + eta * c)

        pr = participation_ratio(s_reg)
        pr_trajectory.append(pr)

        # Regulation: grow n if captured, reduce cs
        if pr < pr_threshold_low and n_current < 256:
            n_new = n_current + 8
            # Reduce cs to keep ||eps||_F below operating regime bound
            # ||eps||_F ≈ cs * sqrt(n/2), want < sqrt(n)/3
            # cs < sqrt(2) / 3 ≈ 0.47
            cs_new = min(cs_current * 0.85, 0.47)

            s_extended = np.zeros(n_new, dtype=complex)
            s_extended[:n_current] = s_reg
            s_extended[n_current:] = 0.01 * (rng2.randn(8) + 1j * rng2.randn(8))
            s_reg = renormalize(s_extended)

            L_reg, _, _ = make_gram_coupling(n_new, cs_new, seed=42 + growth_events)
            n_current = n_new
            cs_current = cs_new
            growth_events += 1

    pr_reg = participation_ratio(s_reg)
    mags_reg = np.abs(s_reg)
    n_inv = min(8, n_current)
    print(f"  With regulation: PR={pr_reg:.1f}  |s_inv|={np.mean(mags_reg[:n_inv]):.4f}  "
          f"|s_noise|={np.mean(mags_reg[n_inv:]):.4f}")
    print(f"    n: {n_initial} -> {n_current}  cs: {cs_initial:.2f} -> {cs_current:.2f}  growth_events: {growth_events}")

    # PR trajectory
    pr_early = np.mean(pr_trajectory[:300])
    pr_mid = np.mean(pr_trajectory[500:800])
    pr_late = np.mean(pr_trajectory[-300:])
    print(f"    PR trajectory: early={pr_early:.1f}  mid={pr_mid:.1f}  late={pr_late:.1f}")

    print()
    if pr_reg > pr_static and pr_reg > 3:
        print(f"  CONFIRMED: Self-regulation maintains operating regime (PR {pr_reg:.1f} vs {pr_static:.1f})")
    elif pr_reg > pr_static:
        print(f"  PARTIAL: PR improved ({pr_reg:.1f} vs {pr_static:.1f}) but not stable operating regime")
    else:
        print(f"  FAILED: Regulation didn't help ({pr_reg:.1f} vs {pr_static:.1f})")

    return {"static_pr": pr_static, "reg_pr": pr_reg, "n_final": n_current}


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    r2 = test_abstraction_speed()
    print("\n")
    r3 = test_self_regulation_v2()
