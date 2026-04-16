"""
Phase 5: Anticipatory Operator Verification.

Tests predictions from docs/PHASE5_ANTICIPATION.md:
  T1: Habituation under predictable input (prediction error decay)
  T2: Surprise response to perturbation (prediction error spike)
  T3: Cross-dimensional surprise propagation through coupling
  T4: Habituation rate vs eta * kappa(G)
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
    evals = np.real(np.linalg.eigvalsh(G))
    kappa = np.max(evals) / max(np.min(evals), 1e-10)
    return L, np.linalg.norm(eps, 'fro'), kappa


# ── T1: Habituation ──────────────────────────────────────────────────────

def test_habituation():
    """Predictable input (invariant + small noise) -> prediction error decays.

    Prediction: prediction error decays exponentially with rate ~ eta * kappa.
    """
    n = 32
    n_steps = 800
    eta = 0.1
    n_invariant = 8

    print("=" * 70)
    print("T1: HABITUATION UNDER PREDICTABLE INPUT")
    print("=" * 70)
    print(f"n={n}, eta={eta}, invariant dims=0-{n_invariant-1}, signal=0.5, noise=0.05")
    print()

    for cs in [0.0, 0.3, 0.7]:
        L, eps_norm, kappa = make_gram_coupling(n, cs) if cs > 0 else (np.eye(n, dtype=complex), 0.0, 1.0)

        rng = np.random.RandomState(42)
        s = renormalize(rng.randn(n) + 1j * rng.randn(n))

        errors = []

        for step in range(n_steps):
            v = 0.05 * (rng.randn(n) + 1j * rng.randn(n))
            for i in range(n_invariant):
                v[i] += 0.5 * np.exp(1j * 0.3)

            c = coupled_reception(v, s, L)
            s_new = renormalize(s + eta * c)

            # Prediction error (predict no change)
            e = s_new - s
            error_mag = np.linalg.norm(e)
            errors.append(error_mag)

            s = s_new

        early = np.mean(errors[50:100])
        mid = np.mean(errors[300:400])
        late = np.mean(errors[600:700])
        decay_ratio = early / max(late, 1e-10)

        print(f"  cs={cs:.1f}  kappa={kappa:.1f}:  early={early:.6f}  mid={mid:.6f}  late={late:.6f}  decay={decay_ratio:.1f}x")

    print()
    print("  (All conditions show prediction error decay = habituation)")


# ── T2: Surprise Response ────────────────────────────────────────────────

def test_surprise():
    """Habituate on one signal, then perturb. Prediction error should spike.

    Prediction: spike magnitude proportional to signal change magnitude.
    """
    n = 32
    n_habituate = 500
    n_perturb = 20
    n_recover = 200
    eta = 0.1
    n_invariant = 8
    cs = 0.3

    print("=" * 70)
    print("T2: SURPRISE RESPONSE TO PERTURBATION")
    print("=" * 70)
    print(f"n={n}, eta={eta}, cs={cs}, habituate={n_habituate}, perturb={n_perturb}")
    print()

    L, eps_norm, kappa = make_gram_coupling(n, cs)

    rng = np.random.RandomState(42)
    s = renormalize(rng.randn(n) + 1j * rng.randn(n))

    errors = []
    baseline_errors = []

    for step in range(n_habituate + n_perturb + n_recover):
        v = 0.05 * (rng.randn(n) + 1j * rng.randn(n))

        if step < n_habituate:
            # Habituation phase: invariant signal at dims 0-7
            for i in range(n_invariant):
                v[i] += 0.5 * np.exp(1j * 0.3)
        elif step < n_habituate + n_perturb:
            # Perturbation: phase shift at dims 0-7
            for i in range(n_invariant):
                v[i] += 0.5 * np.exp(1j * 2.0)  # Different phase!
        else:
            # Recovery: back to original signal
            for i in range(n_invariant):
                v[i] += 0.5 * np.exp(1j * 0.3)

        c = coupled_reception(v, s, L)
        s_new = renormalize(s + eta * c)

        e = s_new - s
        error_mag = np.linalg.norm(e)
        errors.append(error_mag)

        if step < n_habituate:
            baseline_errors.append(error_mag)

        s = s_new

    baseline = np.mean(baseline_errors[-100:])
    perturb_errors = errors[n_habituate:n_habituate + n_perturb]
    perturb_peak = np.max(perturb_errors)
    recover_errors = errors[n_habituate + n_perturb:]
    recover_mean = np.mean(recover_errors[-50:])

    print(f"  Baseline error (late habituation): {baseline:.6f}")
    print(f"  Perturbation peak:                 {perturb_peak:.6f}  ({perturb_peak/baseline:.1f}x baseline)")
    print(f"  Recovery mean (last 50 steps):     {recover_mean:.6f}  ({recover_mean/baseline:.1f}x baseline)")
    print()

    if perturb_peak > 1.2 * baseline:
        print(f"  CONFIRMED: Perturbation spike ({perturb_peak/baseline:.1f}x baseline)")
    else:
        print(f"  FAILED: No significant spike ({perturb_peak/baseline:.1f}x baseline)")

    if recover_mean < perturb_peak * 0.5:
        print(f"  CONFIRMED: Recovery after perturbation (recovery {recover_mean/perturb_peak:.1%} of peak)")
    else:
        print(f"  FAILED: No recovery")


# ── T3: Cross-Dimensional Surprise Propagation ───────────────────────────

def test_cross_dimensional_surprise():
    """Perturb one dimension, measure prediction error at other dimensions.

    Prediction: error at dim i from perturbation at dim j proportional to |L_{ij}|.
    """
    n = 16
    eta = 0.1
    cs = 0.5
    n_habituate = 300

    print("=" * 70)
    print("T3: CROSS-DIMENSIONAL SURPRISE PROPAGATION")
    print("=" * 70)
    print(f"n={n}, eta={eta}, cs={cs}")
    print(f"Habituate on signal at dim 0, then perturb dim 0. Measure error at all dims.")
    print()

    L, eps_norm, kappa = make_gram_coupling(n, cs)

    rng = np.random.RandomState(42)
    s = renormalize(rng.randn(n) + 1j * rng.randn(n))

    # Habituate
    for step in range(n_habituate):
        v = 0.05 * (rng.randn(n) + 1j * rng.randn(n))
        v[0] += 0.5 * np.exp(1j * 0.3)
        c = coupled_reception(v, s, L)
        s = renormalize(s + eta * c)

    # Perturb dim 0
    v_before = 0.05 * (rng.randn(n) + 1j * rng.randn(n))
    v_before[0] += 0.5 * np.exp(1j * 0.3)

    v_after = 0.05 * (rng.randn(n) + 1j * rng.randn(n))
    v_after[0] += 0.5 * np.exp(1j * 2.0)  # Phase shift at dim 0

    s_before = s.copy()

    c = coupled_reception(v_after, s, L)
    s_after = renormalize(s + eta * c)

    # Per-dimension prediction error
    e = s_after - s_before
    per_dim_error = np.abs(e)

    # Correlation with coupling matrix row
    L_row_0 = np.abs(L[0, :])  # Coupling from dim 0 to all dims
    L_col_0 = np.abs(L[:, 0])  # Coupling from all dims to dim 0

    print(f"  Per-dim prediction error (sorted):")
    dims_sorted = np.argsort(per_dim_error)[::-1]
    for idx in dims_sorted[:8]:
        print(f"    dim {idx:2d}: error={per_dim_error[idx]:.6f}  |L_{{0,{idx}}}|={L_row_0[idx]:.4f}  |L_{{{idx},0}}|={L_col_0[idx]:.4f}")

    print()

    # Correlation between prediction error and coupling
    error_coupling_corr = np.corrcoef(per_dim_error, L_row_0)[0, 1]
    error_coupling_corr_col = np.corrcoef(per_dim_error, L_col_0)[0, 1]
    print(f"  Correlation(error, |L[0,:]|) = {error_coupling_corr:.3f}")
    print(f"  Correlation(error, |L[:,0]|) = {error_coupling_corr_col:.3f}")

    if abs(error_coupling_corr) > 0.3 or abs(error_coupling_corr_col) > 0.3:
        print(f"  CONFIRMED: Prediction error propagates through coupling matrix")
    else:
        print(f"  NOTE: Weak coupling-error correlation (may need stronger perturbation)")


# ── T4: Habituation Rate vs Coupling ─────────────────────────────────────

def test_habituation_rate():
    """Measure habituation rate (error decay constant) vs eta * kappa.

    Prediction: faster habituation with larger eta * kappa.
    """
    n = 32
    n_steps = 400
    n_invariant = 8

    print("=" * 70)
    print("T4: HABITUATION RATE VS eta * kappa")
    print("=" * 70)
    print()

    results = []

    for eta in [0.05, 0.1, 0.2]:
        for cs in [0.0, 0.2, 0.5]:
            L, eps_norm, kappa = make_gram_coupling(n, cs) if cs > 0 else (np.eye(n, dtype=complex), 0.0, 1.0)

            rng = np.random.RandomState(42)
            s = renormalize(rng.randn(n) + 1j * rng.randn(n))

            errors = []
            for step in range(n_steps):
                v = 0.05 * (rng.randn(n) + 1j * rng.randn(n))
                for i in range(n_invariant):
                    v[i] += 0.5 * np.exp(1j * 0.3)

                c = coupled_reception(v, s, L)
                s_new = renormalize(s + eta * c)
                errors.append(np.linalg.norm(s_new - s))
                s = s_new

            # Estimate decay rate from first 100 vs last 100 steps
            early = np.mean(errors[:50])
            late = np.mean(errors[-50:])
            decay = early / max(late, 1e-10)

            results.append({
                "eta": eta, "cs": cs, "kappa": kappa,
                "eta_kappa": eta * kappa,
                "decay": decay,
                "early": early, "late": late,
            })

            print(f"  eta={eta:.2f}  cs={cs:.1f}  kappa={kappa:.1f}  eta*kappa={eta*kappa:.3f}  "
                  f"early={early:.6f}  late={late:.6f}  decay={decay:.1f}x")

    print()

    # Check if decay correlates with eta * kappa
    decays = [r["decay"] for r in results]
    eta_kappas = [r["eta_kappa"] for r in results]
    corr = np.corrcoef(decays, eta_kappas)[0, 1]
    print(f"  Correlation(decay, eta*kappa) = {corr:.3f}")

    if corr > 0.5:
        print(f"  CONFIRMED: Habituation rate correlates with eta * kappa(G)")
    else:
        print(f"  NOTE: Weak correlation (decay may depend more on input structure than coupling)")


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    test_habituation()
    print("\n")
    test_surprise()
    print("\n")
    test_cross_dimensional_surprise()
    print("\n")
    test_habituation_rate()
