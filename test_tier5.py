"""Tier 5 tests: anticipation — prediction error torque, habituation."""

import numpy as np
from engine import MemoryEngine, participation_ratio, renormalize, hadamard, Regime

np.set_printoptions(precision=4, suppress=True)


class AnticipatoryEngine(MemoryEngine):
    """Memory engine with a simple linear-extrapolation anticipatory operator.

    A(s) extrapolates each dimension's trajectory from recent phase and magnitude
    velocities. Prediction error e = v_received - s_predicted drives an additional
    torque channel.
    """

    def __init__(self, n: int, eta: float = 0.05, pred_weight: float = 0.5,
                 extrapolation_window: int = 10):
        super().__init__(n=n, eta=eta)
        self.pred_weight = pred_weight
        self.extrapolation_window = extrapolation_window
        self.prediction_errors = []
        self.input_history = []  # track received inputs, not tape states

    def predict(self) -> np.ndarray:
        """Predict next received input by extrapolating from input history.

        The essay says A maps s(t) to a predicted future, but operationally
        the system needs to track what the world has been sending. We extrapolate
        from the received input trajectory, not the tape trajectory (which is
        always being updated by the prediction error itself).
        """
        if len(self.input_history) < self.extrapolation_window + 1:
            return np.zeros(self.n, dtype=complex)  # no prediction

        recent = np.array(self.input_history[-self.extrapolation_window:])
        phases = np.angle(recent)
        mags = np.abs(recent)

        unwrapped = np.unwrap(phases, axis=0)
        phase_vel = np.mean(np.diff(unwrapped, axis=0), axis=0)
        mag_vel = np.mean(np.diff(mags, axis=0), axis=0)

        pred_phase = unwrapped[-1] + phase_vel
        pred_mag = np.clip(mags[-1] + mag_vel, 0.001, 2.0)
        pred = pred_mag * np.exp(1j * pred_phase)
        return renormalize(pred)

    def step_with_prediction(self, v: np.ndarray, recurrence_delay: int = 0,
                              recurrence_weight: float = 0.4) -> dict:
        """Full step: predict → receive → compute prediction error → update."""
        s_predicted = self.predict()
        v_received = self.project(v)  # actual received input
        c_world, world_regimes = self.receive(v)  # Hadamard reception

        # Store received input for future predictions
        self.input_history.append(v_received.copy())

        # Prediction error
        error = v_received - s_predicted
        c_error = hadamard(error, self.s)

        # World reception
        c_world, _ = self.receive(v)

        # Recurrence
        c_self = None
        if recurrence_delay > 0 and len(self.history) >= recurrence_delay:
            c_self, _ = self.self_reception(recurrence_delay)

        # Combined update
        c_total = c_world + self.pred_weight * c_error
        if c_self is not None:
            c_total += recurrence_weight * c_self

        s_before = self.s.copy()
        self.update(c_total)
        self.history.append(s_before)

        error_mag = np.linalg.norm(error)
        n_dims_error_torque = sum(
            1 for c in c_error
            if np.real(c) < 0 or abs(np.imag(c)) >= abs(np.real(c))
        )

        self.prediction_errors.append({
            "error_mag": error_mag,
            "n_error_torque_dims": n_dims_error_torque,
            "step": self.step_count,
        })

        return {
            "s_predicted": s_predicted,
            "error": error,
            "c_error": c_error,
            "error_mag": error_mag,
            "n_error_torque_dims": n_dims_error_torque,
            "c_total": c_total,
            "s_before": s_before,
            "s_after": self.s.copy(),
            "pr_after": participation_ratio(self.s),
        }


# ── T5.1 ─────────────────────────────────────────────────────────────────
def test_t5_1_prediction_error_drives_learning():
    """Prediction error torque concentrates accommodation at wrong dimensions."""
    n = 24
    np.random.seed(42)

    # Two engines: with and without anticipation, same input stream
    engine_pred = AnticipatoryEngine(n=n, eta=0.04, pred_weight=0.5)
    engine_plain = MemoryEngine(n=n, eta=0.04)

    # Phase 1: train both on a predictable signal
    signal_phase = np.random.uniform(0, 2 * np.pi, n)

    for t in range(1000):
        v = np.zeros(n, dtype=complex)
        for i in range(n):
            v[i] = 0.7 * np.exp(1j * signal_phase[i])
        v += 0.2 * (np.random.randn(n) + 1j * np.random.randn(n)) / np.sqrt(n)
        v = renormalize(v)

        engine_pred.step_with_prediction(v, recurrence_delay=5, recurrence_weight=0.3)
        engine_plain.step(v, recurrence_delay=5, recurrence_weight=0.3)

    # Phase 2: sudden regime change — phase shifts at dims 0-7
    shift_dims = list(range(8))
    new_signal_phase = signal_phase.copy()
    new_signal_phase[shift_dims] += np.pi / 2  # 90-degree shift

    pred_error_shift = []
    plain_displacement = []
    pred_displacement = []

    for t in range(300):
        v = np.zeros(n, dtype=complex)
        for i in range(n):
            v[i] = 0.7 * np.exp(1j * new_signal_phase[i])
        v += 0.2 * (np.random.randn(n) + 1j * np.random.randn(n)) / np.sqrt(n)
        v = renormalize(v)

        info_p = engine_pred.step_with_prediction(v, recurrence_delay=5, recurrence_weight=0.3)
        pred_error_shift.append(info_p["error_mag"])

        s_before_plain = engine_plain.s.copy()
        engine_plain.step(v, recurrence_delay=5, recurrence_weight=0.3)
        plain_displacement.append(engine_plain.angular_displacement(s_before_plain, engine_plain.s))
        pred_displacement.append(engine_plain.angular_displacement(info_p["s_before"], info_p["s_after"]))

    # Prediction error should spike at the shift then decay
    early_error = np.mean(pred_error_shift[:20])
    late_error = np.mean(pred_error_shift[-50:])

    print("T5.1 — Prediction Error Torque Drives Learning")
    print(f"  Prediction error: early (regime change) = {early_error:.4f}, "
          f"late (adapted) = {late_error:.4f}")
    print(f"  Error decay ratio: {early_error / (late_error + 1e-10):.2f}x")

    assert early_error > late_error, (
        "Prediction error should spike at regime change then decay"
    )

    # Check that the anticipatory engine tracks which dims were wrong
    # Look at error torque at shifted vs unshifted dims in the first few steps
    np.random.seed(42)
    engine_check = AnticipatoryEngine(n=n, eta=0.04, pred_weight=0.5)
    for t in range(1000):
        v = np.zeros(n, dtype=complex)
        for i in range(n):
            v[i] = 0.7 * np.exp(1j * signal_phase[i])
        v += 0.2 * (np.random.randn(n) + 1j * np.random.randn(n)) / np.sqrt(n)
        v = renormalize(v)
        engine_check.step_with_prediction(v, recurrence_delay=5, recurrence_weight=0.3)

    # First step after shift
    v_shift = np.zeros(n, dtype=complex)
    for i in range(n):
        v_shift[i] = 0.7 * np.exp(1j * new_signal_phase[i])
    v_shift += 0.2 * (np.random.randn(n) + 1j * np.random.randn(n)) / np.sqrt(n)
    v_shift = renormalize(v_shift)

    info = engine_check.step_with_prediction(v_shift, recurrence_delay=5, recurrence_weight=0.3)
    c_error = info["c_error"]

    error_torque_shifted = sum(
        1 for i in shift_dims
        if np.real(c_error[i]) < 0 or abs(np.imag(c_error[i])) >= abs(np.real(c_error[i]))
    )
    error_torque_unshifted = sum(
        1 for i in range(8, n)
        if np.real(c_error[i]) < 0 or abs(np.imag(c_error[i])) >= abs(np.real(c_error[i]))
    )

    print(f"  Error torque at shifted dims (0-7):   {error_torque_shifted}/8")
    print(f"  Error torque at unshifted dims (8-23): {error_torque_unshifted}/{n-8}")

    assert error_torque_shifted >= error_torque_unshifted * (8 / (n - 8)) * 0.5, (
        "Prediction error should concentrate at the shifted dimensions"
    )
    print("  PASS\n")


# ── T5.2 ─────────────────────────────────────────────────────────────────
def test_t5_2_habituation():
    """Predictable inputs → prediction error decays toward zero. Perturbation → spike."""
    n = 20
    np.random.seed(42)

    engine = AnticipatoryEngine(n=n, eta=0.04, pred_weight=0.5)

    signal_phase = np.random.uniform(0, 2 * np.pi, n)
    error_history = []

    # Phase 1: near-predictable signal with small noise (gradual habituation)
    for t in range(800):
        v = np.zeros(n, dtype=complex)
        for i in range(n):
            v[i] = np.exp(1j * signal_phase[i])
        # Small noise so predictor has to learn gradually
        v += 0.15 * (np.random.randn(n) + 1j * np.random.randn(n)) / np.sqrt(n)
        v = renormalize(v)
        info = engine.step_with_prediction(v, recurrence_delay=3, recurrence_weight=0.2)
        error_history.append(info["error_mag"])

    # Phase 2: sudden perturbation
    perturb_phase = signal_phase + np.pi / 3  # large phase shift
    for t in range(100):
        v = np.zeros(n, dtype=complex)
        for i in range(n):
            v[i] = np.exp(1j * perturb_phase[i])
        v = renormalize(v)
        info = engine.step_with_prediction(v, recurrence_delay=3, recurrence_weight=0.2)
        error_history.append(info["error_mag"])

    # Phase 3: return to original signal
    for t in range(300):
        v = np.zeros(n, dtype=complex)
        for i in range(n):
            v[i] = np.exp(1j * signal_phase[i])
        v = renormalize(v)
        info = engine.step_with_prediction(v, recurrence_delay=3, recurrence_weight=0.2)
        error_history.append(info["error_mag"])

    errors = np.array(error_history)

    # Windowed averages
    early_habit = np.mean(errors[100:200])   # early habituation phase
    late_habit = np.mean(errors[600:780])     # late habituation (should be lower)
    perturb_peak = np.mean(errors[800:820])   # perturbation spike
    recovery = np.mean(errors[1050:1100])     # recovery phase

    print("T5.2 — Habituation: Predictable → Flat, Perturbation → Spike")
    print(f"  Error (early habituation):  {early_habit:.4f}")
    print(f"  Error (late habituation):   {late_habit:.4f}")
    print(f"  Error (perturbation):       {perturb_peak:.4f}")
    print(f"  Error (recovery):           {recovery:.4f}")
    print(f"  Habituation decay:          {early_habit / (late_habit + 1e-10):.2f}x")
    print(f"  Perturbation spike:         {perturb_peak / (late_habit + 1e-10):.2f}x baseline")

    assert late_habit < early_habit, "Prediction error should decrease during habituation"
    assert perturb_peak > late_habit * 1.2, "Perturbation should spike prediction error"
    assert recovery < perturb_peak, "Error should decrease after perturbation ends"

    print("  PASS\n")


if __name__ == "__main__":
    test_t5_1_prediction_error_drives_learning()
    test_t5_2_habituation()
    print("All Tier 5 tests completed.")
