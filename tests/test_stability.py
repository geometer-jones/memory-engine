"""
Phase 3 Stability Theory Verification.

Tests predictions from docs/PHASE3_STABILITY.md:
  T1: Operating regime window with structured input (kappa bounds)
  T2: Aligned vs misaligned coupling on abstraction
  T3: Self-regulated coupling via basis monitoring
  T4: n-scaling of operating regime width
"""

import numpy as np
from dataclasses import dataclass


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
    return "torque"

def make_gram_coupling(n, coupling_strength, seed=42):
    rng = np.random.RandomState(seed)
    eps = coupling_strength * (rng.randn(n, n) + 1j * rng.randn(n, n)) / np.sqrt(2 * n)
    eps = (eps + eps.conj().T) / 2
    np.fill_diagonal(eps, 0)
    G = np.eye(n, dtype=complex) + eps
    L = np.linalg.inv(G)
    return L, eps, G

def coupled_reception(v, s, L):
    return (L @ v) * s

def condition_number(G):
    evals = np.real(np.linalg.eigvalsh(G))
    pos_evals = evals[evals > 1e-10]
    if len(pos_evals) < 2:
        return float('inf')
    return float(np.max(pos_evals) / np.min(pos_evals))


# ── T1: Operating Regime Window ──────────────────────────────────────────

def test_operating_regime_window():
    """Structured input (invariant + noise + recurrence) with varying coupling.

    Prediction: Operating regime (PR ~2-5, structured anisotropy) exists
    for kappa(G) between ~2 and ~10. Below: Hadamard isolation. Above: capture.
    """
    n = 32
    eta = 0.1
    n_steps = 2000
    n_invariant = 8

    print("=" * 70)
    print("T1: OPERATING REGIME WINDOW WITH STRUCTURED INPUT")
    print("=" * 70)
    print(f"n={n}, steps={n_steps}, eta={eta}, invariant dims=0-{n_invariant-1}")
    print()

    coupling_strengths = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
    results = []

    for cs in coupling_strengths:
        if cs == 0.0:
            L = np.eye(n, dtype=complex)
            G = np.eye(n, dtype=complex)
            kappa = 1.0
            eps_norm = 0.0
        else:
            L, eps, G = make_gram_coupling(n, cs, seed=42)
            eps_norm = np.linalg.norm(eps, 'fro')
            kappa = condition_number(G)

        rng = np.random.RandomState(42)
        s = renormalize(rng.randn(n) + 1j * rng.randn(n))

        history = []

        for step in range(n_steps):
            # Invariant signal at dims 0-7 (fixed phase)
            v_signal = np.zeros(n, dtype=complex)
            for i in range(n_invariant):
                v_signal[i] = 0.5 * np.exp(1j * 0.3)

            # Noise at all dims
            v_noise = 0.1 * (rng.randn(n) + 1j * rng.randn(n))

            # Recurrence (delay=5, weight=0.6)
            v_recurrence = np.zeros(n, dtype=complex)
            if len(history) >= 5:
                v_recurrence = 0.6 * history[-5]

            v = v_signal + v_noise + v_recurrence

            # Coupled reception
            c = coupled_reception(v, s, L)

            # Regime classification
            regimes = [classify_regime(ci) for ci in c]
            resonance_count = sum(1 for r in regimes[:n_invariant] if r == "resonance")

            s = renormalize(s + eta * c)
            history.append(s.copy())

            # Metrics (last 500 steps)
            if step >= 1500:
                pass  # Will compute after loop

        pr_final = participation_ratio(s)
        mags = np.abs(s)
        mean_invariant = np.mean(mags[:n_invariant])
        mean_noise = np.mean(mags[n_invariant:])
        concentration_ratio = mean_invariant / max(mean_noise, 1e-10)

        results.append({
            "cs": cs, "eps_norm": eps_norm, "kappa": kappa,
            "pr": pr_final, "conc_ratio": concentration_ratio,
            "mean_inv": mean_invariant, "mean_noise": mean_noise,
        })

        print(f"  cs={cs:4.1f}  ||eps||={eps_norm:6.2f}  kappa={kappa:8.1f}  "
              f"PR={pr_final:5.1f}  inv/noise={concentration_ratio:6.1f}x  "
              f"|s_inv|={mean_invariant:.4f}  |s_noise|={mean_noise:.4f}")

    print()

    # Identify operating regime window
    operating = [r for r in results if 2 < r["pr"] < 10 and r["conc_ratio"] > 2]
    if operating:
        cs_min = min(r["cs"] for r in operating)
        cs_max = max(r["cs"] for r in operating)
        kappa_min = min(r["kappa"] for r in operating if r["kappa"] < float('inf'))
        kappa_max = max(r["kappa"] for r in operating if r["kappa"] < float('inf'))
        print(f"  Operating regime window: cs in [{cs_min:.1f}, {cs_max:.1f}]")
        print(f"  Corresponding kappa(G): [{kappa_min:.1f}, {kappa_max:.1f}]")
    else:
        print("  No clear operating regime window found")

    captured = [r for r in results if r["pr"] <= 2]
    if captured:
        print(f"  Capture detected for cs >= {min(r['cs'] for r in captured):.1f} "
              f"(kappa >= {min(r['kappa'] for r in captured if r['kappa'] < float('inf')):.1f})")

    return results


# ── T2: Aligned vs Misaligned Coupling ───────────────────────────────────

def test_alignment():
    """Test whether coupling aligned with invariant structure accelerates abstraction.

    Aligned: epsilon_ij has same sign as v_invariant_i * v_invariant_j
    Misaligned: epsilon_ij has opposite sign
    Random: epsilon_ij random (control)
    """
    n = 32
    eta = 0.1
    n_steps = 1500
    n_invariant = 8
    cs = 0.3  # Moderate coupling, in operating regime window

    print("=" * 70)
    print("T2: ALIGNED VS MISALIGNED COUPLING ON ABSTRACTION")
    print("=" * 70)
    print(f"n={n}, steps={n_steps}, eta={eta}, coupling={cs}")
    print()

    invariant_phase = 0.3
    rng = np.random.RandomState(42)

    conditions = ["aligned", "misaligned", "random"]
    results = {}

    for condition in conditions:
        # Build coupling matrix
        if condition == "random":
            L, eps, G = make_gram_coupling(n, cs, seed=42)

        elif condition == "aligned":
            # eps_ij proportional to v_i * v_j (conjugate) for invariant dims
            v_inv = np.zeros(n, dtype=complex)
            for i in range(n_invariant):
                v_inv[i] = np.exp(1j * invariant_phase)
            # Outer product gives alignment
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
            eps = -cs * alignment / np.linalg.norm(alignment, 'fro')  # Negative!
            np.fill_diagonal(eps, 0)
            G = np.eye(n, dtype=complex) + eps
            L = np.linalg.inv(G)

        rng_state = np.random.RandomState(42)
        s = renormalize(rng_state.randn(n) + 1j * rng_state.randn(n))

        pr_trajectory = []

        for step in range(n_steps):
            v_signal = np.zeros(n, dtype=complex)
            for i in range(n_invariant):
                v_signal[i] = 0.5 * np.exp(1j * invariant_phase)
            v_noise = 0.1 * (rng_state.randn(n) + 1j * rng_state.randn(n))
            v = v_signal + v_noise

            c = coupled_reception(v, s, L)
            s = renormalize(s + eta * c)

            if step % 100 == 0:
                pr_trajectory.append(participation_ratio(s))

        mags = np.abs(s)
        mean_inv = np.mean(mags[:n_invariant])
        mean_noise = np.mean(mags[n_invariant:])
        pr_final = participation_ratio(s)

        results[condition] = {
            "pr_final": pr_final,
            "mean_inv": mean_inv,
            "mean_noise": mean_noise,
            "conc_ratio": mean_inv / max(mean_noise, 1e-10),
            "pr_trajectory": pr_trajectory,
        }

        print(f"  {condition:12s}: PR={pr_final:5.1f}  |s_inv|={mean_inv:.4f}  "
              f"|s_noise|={mean_noise:.4f}  inv/noise={mean_inv/max(mean_noise,1e-10):.1f}x")

    print()

    # Check prediction: aligned > random > misaligned for concentration ratio
    aligned_cr = results["aligned"]["conc_ratio"]
    misaligned_cr = results["misaligned"]["conc_ratio"]
    random_cr = results["random"]["conc_ratio"]

    if aligned_cr > random_cr > misaligned_cr:
        print("  CONFIRMED: Aligned > Random > Misaligned for concentration ratio")
    elif aligned_cr > misaligned_cr:
        print(f"  PARTIAL: Aligned ({aligned_cr:.1f}x) > Misaligned ({misaligned_cr:.1f}x), "
              f"Random ({random_cr:.1f}x)")
    else:
        print(f"  UNEXPECTED: Misaligned ({misaligned_cr:.1f}x) >= Aligned ({aligned_cr:.1f}x)")

    return results


# ── T3: Self-Regulated Coupling ──────────────────────────────────────────

def test_self_regulation():
    """System monitors PR and adjusts dimensionality to maintain operating regime.

    Start with n=32, coupling that would cause capture.
    When PR drops below threshold, grow dimensions (increase n).
    """
    n_initial = 32
    eta = 0.1
    n_steps = 2000
    cs = 1.0  # Strong coupling, should cause capture without regulation

    print("=" * 70)
    print("T3: SELF-REGULATED COUPLING VIA BASIS MONITORING")
    print("=" * 70)
    print(f"n_initial={n_initial}, steps={n_steps}, eta={eta}, coupling={cs}")
    print(f"PR threshold for growth: 3.0. Growth increment: 4 dims.")
    print()

    # --- Without regulation ---
    rng = np.random.RandomState(42)
    L, eps, G = make_gram_coupling(n_initial, cs, seed=42)

    s_no_reg = renormalize(rng.randn(n_initial) + 1j * rng.randn(n_initial))

    for step in range(n_steps):
        v = rng.randn(n_initial) + 1j * rng.randn(n_initial)
        v = v / np.linalg.norm(v)
        c = coupled_reception(v, s_no_reg, L)
        s_no_reg = renormalize(s_no_reg + eta * c)

    pr_no_reg = participation_ratio(s_no_reg)
    print(f"  Without regulation: PR = {pr_no_reg:.1f}")

    # --- With regulation ---
    rng2 = np.random.RandomState(42)
    n_current = n_initial
    L2, eps2, G2 = make_gram_coupling(n_current, cs, seed=42)
    s_reg = renormalize(rng2.randn(n_current) + 1j * rng2.randn(n_current))

    pr_threshold = 3.0
    growth_increment = 4
    n_growth_events = 0
    pr_trajectory = []

    for step in range(n_steps):
        v = rng2.randn(n_current) + 1j * rng2.randn(n_current)
        v = v / np.linalg.norm(v)
        c = coupled_reception(v, s_reg, L2)
        s_reg = renormalize(s_reg + eta * c)

        pr = participation_ratio(s_reg)
        pr_trajectory.append(pr)

        # Regulation: if PR drops, grow dimensions
        if pr < pr_threshold and n_current < 128:
            n_new = n_current + growth_increment
            # Extend state with new small dimensions
            s_extended = np.zeros(n_new, dtype=complex)
            s_extended[:n_current] = s_reg
            s_extended[n_current:] = 0.01 * (rng2.randn(growth_increment) + 1j * rng2.randn(growth_increment))
            s_reg = renormalize(s_extended)

            # Rebuild coupling matrix for new dimensionality
            L2, eps2, G2 = make_gram_coupling(n_new, cs, seed=42 + n_growth_events)
            n_current = n_new
            n_growth_events += 1

    pr_reg = participation_ratio(s_reg)
    print(f"  With regulation:    PR = {pr_reg:.1f}  (n grew {n_initial} -> {n_current}, {n_growth_events} growth events)")
    print()

    # PR trajectory summary
    pr_early = np.mean(pr_trajectory[:200])
    pr_mid = np.mean(pr_trajectory[500:700])
    pr_late = np.mean(pr_trajectory[-200:])
    print(f"  PR trajectory (regulated): early={pr_early:.1f}  mid={pr_mid:.1f}  late={pr_late:.1f}")

    if pr_reg > pr_no_reg and pr_reg > 2:
        print(f"  CONFIRMED: Self-regulation maintains operating regime (PR {pr_reg:.1f} vs {pr_no_reg:.1f})")
    elif pr_reg > pr_no_reg:
        print(f"  PARTIAL: Self-regulation improves PR ({pr_reg:.1f} vs {pr_no_reg:.1f}) but doesn't fully stabilize")
    else:
        print(f"  FAILED: Self-regulation didn't help ({pr_reg:.1f} vs {pr_no_reg:.1f})")

    return {"no_reg": pr_no_reg, "reg": pr_reg, "n_final": n_current}


# ── T4: n-Scaling of Operating Regime Width ─────────────────────────────

def test_n_scaling():
    """Measure the operating regime width (in ||epsilon||_F) for different n.

    Prediction: window width scales as sqrt(n).
    """
    print("=" * 70)
    print("T4: n-SCALING OF OPERATING REGIME WIDTH")
    print("=" * 70)
    print()

    n_values = [16, 32, 64, 128]
    results = []

    for n in n_values:
        # Find capture threshold (PR drops below 2) by sweep
        cs_values = np.linspace(0.0, 3.0, 20)
        capture_cs = None
        operating_cs = None

        for cs in cs_values:
            if cs == 0:
                L = np.eye(n, dtype=complex)
            else:
                L, eps, G = make_gram_coupling(n, cs, seed=42)

            rng = np.random.RandomState(42)
            s = renormalize(rng.randn(n) + 1j * rng.randn(n))

            for step in range(500):
                v = rng.randn(n) + 1j * rng.randn(n)
                v = v / np.linalg.norm(v)
                c = coupled_reception(v, s, L)
                s = renormalize(s + eta * c)

            pr = participation_ratio(s)

            # Operating regime: PR between 0.3*n and 0.8*n
            if operating_cs is None and 0.3 * n < pr < 0.8 * n:
                operating_cs = cs

            # Capture: PR < 2
            if capture_cs is None and pr < 2:
                capture_cs = cs

        # Compute epsilon norms
        if capture_cs is not None:
            _, eps_cap, _ = make_gram_coupling(n, capture_cs, seed=42)
            cap_eps = np.linalg.norm(eps_cap, 'fro')
        else:
            cap_eps = None

        results.append({
            "n": n, "capture_cs": capture_cs, "capture_eps": cap_eps,
        })

        print(f"  n={n:4d}  capture_cs={capture_cs}  ||eps||_capture={cap_eps}")

    print()

    # Check scaling
    if all(r["capture_eps"] is not None for r in results):
        print("  Scaling check (capture_eps ~ sqrt(n)):")
        for r in results:
            predicted = np.sqrt(r["n"]) / 2
            ratio = r["capture_eps"] / predicted
            print(f"    n={r['n']:4d}  measured={r['capture_eps']:.2f}  predicted={predicted:.2f}  ratio={ratio:.3f}")

    return results


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    eta = 0.1

    print()
    r1 = test_operating_regime_window()
    print("\n")
    r2 = test_alignment()
    print("\n")
    r3 = test_self_regulation()
    print("\n")
    r4 = test_n_scaling()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
