"""
Phase 4: Cascade Demonstration.

Simulates a single system traversing the cascade trajectory:
  Level 0: Conservative (L=I, unitary, R=0)
  Level 2: Molecular (L≈I+ε, non-unitary eta, R=0)
  Level 4: Autocatalytic (add recurrence)
  Level 6: Adaptive (growing n, two-timescale)
  Level 7: Representational (thick recurrence, rich coupling)

At each level, measures: PR, regime fractions, Gini, coupling metrics.
"""

import numpy as np

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

def coupled_reception(v, s, L):
    return (L @ v) * s

def make_gram_coupling(n, coupling_strength, seed=42):
    rng = np.random.RandomState(seed)
    eps = coupling_strength * (rng.randn(n, n) + 1j * rng.randn(n, n)) / np.sqrt(2 * n)
    eps = (eps + eps.conj().T) / 2
    np.fill_diagonal(eps, 0)
    G = np.eye(n, dtype=complex) + eps
    L = np.linalg.inv(G)
    eps_norm = np.linalg.norm(eps, 'fro')
    evals = np.real(np.linalg.eigvalsh(G))
    kappa = np.max(evals) / max(np.min(evals), 1e-10)
    return L, eps_norm, kappa


def run_level(name, n, eta, cs, recurrence_delay, recurrence_weight,
              n_steps, n_invariant=0, signal_amp=0.0, noise_amp=1.0,
              grow_n=False, grow_threshold=3.0, seed=42):
    """Run a single cascade level and return metrics."""

    rng = np.random.RandomState(seed)
    s = renormalize(rng.randn(n) + 1j * rng.randn(n))

    if cs > 0:
        L, eps_norm, kappa = make_gram_coupling(n, cs, seed=seed)
    else:
        L = np.eye(n, dtype=complex)
        eps_norm = 0.0
        kappa = 1.0

    history = []
    regime_counts = {"resonance": 0, "torque": 0, "orth": 0}
    pr_trajectory = []
    n_current = n
    cs_current = cs
    growth_events = 0

    for step in range(n_steps):
        # World signal
        v = noise_amp * (rng.randn(n_current) + 1j * rng.randn(n_current))
        if n_invariant > 0:
            for i in range(min(n_invariant, n_current)):
                v[i] += signal_amp * np.exp(1j * 0.3)

        # Coupled reception
        if n_current != n:
            # Rebuild L if n changed
            if cs_current > 0:
                L, eps_norm, kappa = make_gram_coupling(n_current, cs_current, seed=seed + growth_events)
            else:
                L = np.eye(n_current, dtype=complex)

        c_world = coupled_reception(v, s, L)

        # Recurrence
        c_self = np.zeros_like(s)
        if recurrence_delay > 0 and len(history) >= recurrence_delay:
            s_past = history[-recurrence_delay]
            # Pad if n grew
            if len(s_past) < n_current:
                s_past_padded = np.zeros(n_current, dtype=complex)
                s_past_padded[:len(s_past)] = s_past
                s_past = s_past_padded
            elif len(s_past) > n_current:
                s_past = s_past[:n_current]
            c_self = coupled_reception(s_past, s, L)

        c_total = c_world + recurrence_weight * c_self

        # Classify
        for ci in c_total:
            regime_counts[classify_regime(ci)] += 1

        # Update
        s = renormalize(s + eta * c_total)
        history.append(s.copy())

        pr = participation_ratio(s)
        pr_trajectory.append(pr)

        # Adaptive: grow n if PR drops
        if grow_n and pr < grow_threshold and n_current < 128:
            n_new = n_current + 4
            s_ext = np.zeros(n_new, dtype=complex)
            s_ext[:n_current] = s
            s_ext[n_current:] = 0.01 * (rng.randn(4) + 1j * rng.randn(4))
            s = renormalize(s_ext)
            cs_current = min(cs_current * 0.9, 0.47)
            n_current = n_new
            growth_events += 1

    # Final metrics
    pr_final = participation_ratio(s)
    mags = np.abs(s)
    sorted_mags = np.sort(mags)
    gini = 1 - 2 * np.sum(np.cumsum(sorted_mags) / np.sum(sorted_mags)) / n_current

    total_class = sum(regime_counts.values())
    res_frac = regime_counts["resonance"] / total_class
    torq_frac = regime_counts["torque"] / total_class

    pr_mean = np.mean(pr_trajectory[-200:])

    return {
        "name": name, "n": n, "n_final": n_current,
        "eta": eta, "cs": cs, "eps_norm": eps_norm, "kappa": kappa,
        "recurrence_delay": recurrence_delay,
        "pr_final": pr_final, "pr_mean": pr_mean,
        "gini": gini, "res_frac": res_frac, "torq_frac": torq_frac,
        "growth_events": growth_events,
    }


def main():
    print("=" * 80)
    print("PHASE 4: CASCADE DEMONSTRATION")
    print("A single system traversing the cascade trajectory")
    print("=" * 80)
    print()

    levels = [
        {
            "name": "L0: Conservative",
            "desc": "L=I, unitary eta, R=0. Pure resonance, no coupling.",
            "n": 16, "eta": 0.01, "cs": 0.0,
            "recurrence_delay": 0, "recurrence_weight": 0.0,
            "n_steps": 1000, "n_invariant": 0, "signal_amp": 0.0, "noise_amp": 1.0,
        },
        {
            "name": "L2: Molecular",
            "desc": "L≈I+ε, non-unitary eta. First coupling, first renormalization.",
            "n": 16, "eta": 0.1, "cs": 0.2,
            "recurrence_delay": 0, "recurrence_weight": 0.0,
            "n_steps": 1000, "n_invariant": 4, "signal_amp": 0.5, "noise_amp": 0.3,
        },
        {
            "name": "L4: Autocatalytic",
            "desc": "Add recurrence. Self-reception through cycle closure.",
            "n": 16, "eta": 0.1, "cs": 0.3,
            "recurrence_delay": 3, "recurrence_weight": 0.4,
            "n_steps": 1500, "n_invariant": 4, "signal_amp": 0.4, "noise_amp": 0.3,
        },
        {
            "name": "L5: Ecological",
            "desc": "Large n, slow eta, moderate coupling.",
            "n": 64, "eta": 0.02, "cs": 0.3,
            "recurrence_delay": 5, "recurrence_weight": 0.3,
            "n_steps": 2000, "n_invariant": 8, "signal_amp": 0.3, "noise_amp": 0.4,
        },
        {
            "name": "L6: Adaptive",
            "desc": "Growing n, fast eta, self-regulation.",
            "n": 32, "eta": 0.1, "cs": 0.5,
            "recurrence_delay": 5, "recurrence_weight": 0.5,
            "n_steps": 2000, "n_invariant": 8, "signal_amp": 0.3, "noise_amp": 0.4,
            "grow_n": True, "grow_threshold": 3.0,
        },
        {
            "name": "L7: Representational",
            "desc": "Thick recurrence, rich coupling, large n.",
            "n": 64, "eta": 0.1, "cs": 0.5,
            "recurrence_delay": 10, "recurrence_weight": 0.6,
            "n_steps": 2000, "n_invariant": 16, "signal_amp": 0.3, "noise_amp": 0.4,
        },
    ]

    results = []
    for level in levels:
        name = level.pop("name")
        desc = level.pop("desc")
        print(f"--- {name} ---")
        print(f"  {desc}")
        r = run_level(name=name, **level)
        results.append(r)

        print(f"  n={r['n']}  eta={r['eta']}  cs={r['cs']}  ||eps||={r['eps_norm']:.2f}  kappa={r['kappa']:.1f}")
        print(f"  recurrence: delay={r['recurrence_delay']}")
        print(f"  PR={r['pr_final']:.1f} (mean last 200: {r['pr_mean']:.1f})  "
              f"Gini={r['gini']:.3f}  resonance={r['res_frac']:.2f}  torque={r['torq_frac']:.2f}")
        if r["growth_events"]:
            print(f"  Growth events: {r['growth_events']}  n_final={r['n_final']}")
        print()

    # Summary table
    print("=" * 80)
    print("CASCADE SUMMARY")
    print("=" * 80)
    print()
    print(f"{'Level':<25s} {'n':>4s} {'eta':>5s} {'cs':>4s} {'||eps||':>7s} {'kappa':>6s} "
          f"{'PR':>5s} {'Gini':>5s} {'res%':>5s} {'torq%':>5s} {'R_del':>5s}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<25s} {r['n']:>4d} {r['eta']:>5.2f} {r['cs']:>4.1f} "
              f"{r['eps_norm']:>7.2f} {r['kappa']:>6.1f} "
              f"{r['pr_mean']:>5.1f} {r['gini']:>5.3f} "
              f"{r['res_frac']:>5.2f} {r['torq_frac']:>5.2f} {r['recurrence_delay']:>5d}")

    print()
    print("Key transitions:")
    print("  L0→L2: eta crosses unitary threshold, coupling appears")
    print("  L2→L4: recurrence appears (R > 0)")
    print("  L4→L5: n grows from 16 to 64")
    print("  L5→L6: eta increases, n starts growing adaptively")
    print("  L6→L7: thick recurrence, rich coupling, large n")

if __name__ == "__main__":
    main()
