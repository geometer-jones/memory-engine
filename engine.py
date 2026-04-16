"""
Memory Engine: operationalization of the resonance/torque/orthogonality framework.

Core objects:
  - Tape: vector on unit hypersphere in C^n
  - Basis: set of axes (initially canonical, grows via consolidation)
  - MemoryEngine: tape + basis + update rule + optional consolidation + optional anticipation

Operations:
  - project: map world-tape onto system's basis
  - hadamard_reception: elementwise product c = v_received * s
  - update: s_i += eta * c_i, then renormalize
  - classify: per-dimension regime (resonance / torque / orthogonality)
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum


class Regime(Enum):
    RESONANCE = "resonance"
    TORQUE = "torque"
    ORTHOGONALITY = "orthogonality"


def classify_component(c_i: complex) -> Regime:
    """Classify a single Hadamard component into its regime."""
    mag = abs(c_i)
    if mag < 1e-12:
        return Regime.ORTHOGONALITY
    if np.real(c_i) > 0 and abs(np.imag(c_i)) < np.real(c_i):
        return Regime.RESONANCE
    if np.real(c_i) < 0 or abs(np.imag(c_i)) >= abs(np.real(c_i)):
        return Regime.TORQUE
    return Regime.ORTHOGONALITY


def classify_reception(c: np.ndarray) -> list[Regime]:
    """Classify each dimension of a Hadamard product."""
    return [classify_component(c_i) for c_i in c]


def renormalize(s: np.ndarray) -> np.ndarray:
    """Project onto unit hypersphere."""
    norm = np.linalg.norm(s)
    if norm < 1e-15:
        return s
    return s / norm


def participation_ratio(s: np.ndarray) -> float:
    """Effective dimensionality: (sum |s_i|^2)^2 / sum |s_i|^4.
    For uniform |s_i| = 1/sqrt(n), PR = n.
    For one dominant component, PR -> 1.
    """
    mags_sq = np.abs(s) ** 2
    return float(np.sum(mags_sq) ** 2 / np.sum(mags_sq**2))


def hadamard(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise (Hadamard) product of two complex vectors."""
    return a * b


@dataclass
class MemoryEngine:
    """A memory engine operating on the unit hypersphere in C^n.

    Attributes:
        n: dimensionality of the tape
        eta: learning rate (impression rate)
        s: the tape vector (unit norm complex vector)
        basis: n x n complex matrix, columns are basis vectors
               (starts as identity = canonical basis)
        history: optional storage of past tape states for recurrence
    """

    n: int
    eta: float = 0.1
    carved_dim: int = field(default=None)
    leakage: float = 0.0
    s: np.ndarray = field(default=None, repr=False)
    basis: np.ndarray = field(default=None, repr=False)
    history: list[np.ndarray] = field(default_factory=list, repr=False)
    step_count: int = 0
    _coupling: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        if self.carved_dim is None:
            self.carved_dim = self.n
        if self.s is None:
            # Random initial tape on unit hypersphere
            self.s = renormalize(
                np.random.randn(self.n) + 1j * np.random.randn(self.n)
            )
        else:
            self.s = renormalize(self.s)
        if self.basis is None:
            self.basis = np.eye(self.n, dtype=complex)
        # Zero out uncarved dimensions
        if self.carved_dim < self.n:
            self.s[self.carved_dim:] = 0.0
            self.s = renormalize(self.s)
        # Coupling matrix for leakage (imperfect basis orthogonality)
        if self.leakage > 0 and self.carved_dim < self.n:
            n_novel = self.n - self.carved_dim
            self._coupling = self.leakage * (
                np.random.randn(self.carved_dim, n_novel)
                + 1j * np.random.randn(self.carved_dim, n_novel)
            ) / np.sqrt(self.carved_dim)

    def project(self, v: np.ndarray) -> np.ndarray:
        """Project world-tape v onto the system's basis.

        v_received = sum_i <e_i | v> e_i
        For canonical basis this is just v[:n].

        When carved_dim < n, only the first carved_dim dimensions are
        active. Structure along uncarved dimensions is novelty. When
        leakage > 0, imperfect basis orthogonality couples novelty
        into carved dimensions via a random coupling matrix.
        """
        if len(v) < self.n:
            padded = np.zeros(self.n, dtype=complex)
            padded[: len(v)] = v
            v = padded
        # For general basis: v_received = E @ E^H @ v
        v_received = self.basis @ (self.basis.conj().T @ v[:self.n])

        # Carved dimensions: zero out uncarved, apply leakage
        if self.carved_dim < self.n:
            v_received[self.carved_dim:] = 0.0
            if self._coupling is not None:
                novelty = v[:self.n][self.carved_dim:]
                v_received[:self.carved_dim] += self._coupling @ novelty
        return v_received

    def receive(self, v: np.ndarray) -> tuple[np.ndarray, list[Regime]]:
        """Full reception cycle: project, hadamard, classify.

        Returns (c, regimes) where c = v_received * s (Hadamard).
        """
        v_received = self.project(v)
        c = hadamard(v_received, self.s)
        regimes = classify_reception(c)
        return c, regimes

    def update(self, c: np.ndarray) -> np.ndarray:
        """Apply update rule: s_i += eta * c_i, then renormalize.

        Returns the new tape.
        """
        self.s = renormalize(self.s + self.eta * c)
        self.step_count += 1
        return self.s

    def step(self, v: np.ndarray, recurrence_delay: int = 0,
             recurrence_weight: float = 1.0) -> dict:
        """One full step: receive world-tape v, optionally apply self-reception, update tape.

        If recurrence_delay > 0, the system also receives its own past tape
        (from `delay` steps ago). The combined Hadamard is:
            c_total = c_world + recurrence_weight * c_self

        Returns dict with diagnostics.
        """
        c_world, regimes = self.receive(v)
        c_self = None
        self_regimes = None

        if recurrence_delay > 0 and len(self.history) >= recurrence_delay:
            c_self, self_regimes = self.self_reception(recurrence_delay)
            c_total = c_world + recurrence_weight * c_self
        else:
            c_total = c_world

        s_before = self.s.copy()
        self.update(c_total)

        # Store history for recurrence
        self.history.append(s_before)

        return {
            "c": c_total,
            "c_world": c_world,
            "c_self": c_self,
            "regimes": regimes,
            "self_regimes": self_regimes,
            "s_before": s_before,
            "s_after": self.s.copy(),
            "step": self.step_count,
            "pr_before": participation_ratio(s_before),
            "pr_after": participation_ratio(self.s),
        }

    def self_reception(self, delay: int = 1) -> tuple[np.ndarray, list[Regime]]:
        """Compute self-reception: c_self = s(t-delay) * s(t).

        Requires history to have enough entries.
        """
        if len(self.history) < delay:
            raise ValueError(
                f"History has {len(self.history)} entries, need delay={delay}"
            )
        s_past = self.history[-delay]
        c_self = hadamard(s_past, self.s)
        return c_self, classify_reception(c_self)

    def magnitudes(self) -> np.ndarray:
        """Per-dimension magnitudes |s_i|."""
        return np.abs(self.s)

    def phases(self) -> np.ndarray:
        """Per-dimension phases arg(s_i)."""
        return np.angle(self.s)

    def angular_displacement(self, s_before: np.ndarray, s_after: np.ndarray) -> float:
        """Angular displacement between two tape states on the hypersphere."""
        dot = np.abs(np.vdot(s_before, s_after))
        dot = np.clip(dot, 0.0, 1.0)
        return float(np.arccos(dot))

    def dimension_angular_displacement(
        self, s_before: np.ndarray, s_after: np.ndarray
    ) -> np.ndarray:
        """Per-dimension phase displacement."""
        return np.abs(np.angle(s_after) - np.angle(s_before))
