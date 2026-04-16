"""
Production-oriented Memory Engine layer with coupled reception, binding, and consolidation.

This module extends the original `me_layer.py` design rather than replacing it:

- persistent tape state updated causally across tokens
- regime-aware update with torque bias and renormalization
- residual blending back into transformer hidden states

It adds the missing machinery from the repo notes:

- Coupled reception via Gram matrix G = E^T E and coupling matrix L = G^{-1}
  (paper/root.md Section 4, COUPLING_THEORY.md Sections 3-5)
- Sensor leakage Delta for novelty detection
  (COUPLING_THEORY.md Section 5)
- Fast binding via transient conjunctive dimensions
  (standalone_me_binding.py, paper/root.md Section 7 discussion)
- Consolidation loop with merge / prune / seed
  (paper/root.md Section 4)
- Anticipatory operator with prediction-error torque
  (PHASE5_ANTICIPATION.md)

The implementation is designed for hybrid insertion into decoder-only LLMs:

- input: hidden states after self-attention / decoder block
- output: additive residual back into the hidden stream
- state: persistent per-sequence tape and basis cache

The layer is numerically stable by construction:

- hidden states are upcast to fp32 internally
- complex tape is renormalized each token
- Gram inverses use regularization
- binding is sparse and consolidation runs infrequently
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


BASE_SLOT = 0
TRANSIENT_SLOT = 1
MERGED_SLOT = 2
SEEDED_SLOT = 3


def _participation_ratio(magnitudes: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Participation ratio on |s_i|, matching engine.py but in torch."""
    mags_sq = magnitudes.pow(2)
    numer = mags_sq.sum(dim=-1).pow(2)
    denom = mags_sq.pow(2).sum(dim=-1).clamp_min(eps)
    return numer / denom


def _gini_coefficient(magnitudes: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Gini coefficient for anisotropy diagnostics."""
    sorted_mags, _ = torch.sort(magnitudes, dim=-1)
    n = sorted_mags.shape[-1]
    idx = torch.arange(1, n + 1, device=magnitudes.device, dtype=magnitudes.dtype)
    weighted = (idx * sorted_mags).sum(dim=-1)
    total = sorted_mags.sum(dim=-1).clamp_min(eps)
    return (2.0 * weighted / (n * total)) - (n + 1.0) / n


@dataclass
class MemoryEngineState:
    """Per-sequence state that persists across token chunks."""

    tape: torch.Tensor
    prev_tape: torch.Tensor
    basis: torch.Tensor
    gram: torch.Tensor
    coupling: torch.Tensor
    active_mask: torch.Tensor
    slot_kind: torch.Tensor
    lifetime: torch.Tensor
    prune_count: torch.Tensor
    binding_sources: torch.Tensor
    corr: torch.Tensor
    residual_bank: torch.Tensor
    residual_count: torch.Tensor
    residual_ptr: torch.Tensor
    step: torch.Tensor
    basis_dirty: torch.Tensor


class MemoryEngineLayer(nn.Module):
    """
    Coupled Memory Engine layer for transformer hidden states.

    The design intentionally preserves the core flow from `me_layer.py`:

    1. receive a token-conditioned signal
    2. classify resonance / torque / orthogonality
    3. update the tape
    4. renormalize on the hypersphere
    5. feed the tape delta back as a residual

    New here is that the receive step operates in a learned / grown basis E with
    coupling matrix L = (E^T E)^{-1}, plus transient binding and consolidation.

    Args:
        hidden_dim: transformer hidden width
        memory_dim: primary memory slots. Defaults to hidden_dim.
        max_aux_dims: reserve slots for transient bindings and consolidation growth
        max_transient_dims: subset of aux slots reserved for fast binding
        eta_init: impression rate
        alpha_init: residual blend gate before sigmoid
        coupling_reg: diagonal stabilizer for the Gram inverse
        leakage_init: std for sensor leakage Delta
        bind_fraction: top fraction of positive pairs to bind each step
        beta: transient seed magnitude scaling
        gamma: transient decay factor
        transient_lifetime: lifetime counter for transient conjunctive slots
        top_k_binding: only score the strongest k dimensions for binding
        consolidation_interval: run merge/prune/seed every N tokens
        corr_ema: EMA rate for the cross-correlation matrix C
        theta_merge: merge threshold on leading eigenvalue of C
        theta_prune: prune threshold on |s_i|
        prune_patience: consecutive consolidation cycles below theta_prune
        theta_seed: novelty threshold for residual-driven seeding
        seed_scale: magnitude assigned to freshly seeded slots
        merge_parent_decay: shrink parent slots after a merge
        prediction_torque_scale: scaling on c_pred torque channel
        residual_bank_size: number of recent novelty residuals kept for seeding
    """

    def __init__(
        self,
        hidden_dim: int,
        memory_dim: Optional[int] = None,
        max_aux_dims: int = 16,
        max_transient_dims: int = 8,
        eta_init: float = 0.1,
        alpha_init: float = 0.5,
        coupling_reg: float = 1e-3,
        leakage_init: float = 0.01,
        bind_fraction: float = 0.15,
        beta: float = 0.05,
        gamma: float = 0.9,
        transient_lifetime: int = 5,
        top_k_binding: int = 8,
        consolidation_interval: int = 8,
        corr_ema: float = 0.05,
        theta_merge: float = 0.4,
        theta_prune: float = 0.015,
        prune_patience: int = 2,
        theta_seed: float = 0.08,
        seed_scale: float = 0.05,
        merge_parent_decay: float = 0.85,
        prediction_torque_scale: float = 0.4,
        residual_bank_size: int = 8,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim or hidden_dim
        self.max_aux_dims = max_aux_dims
        self.max_transient_dims = min(max_transient_dims, max_aux_dims)
        self.total_slots = self.memory_dim + self.max_aux_dims

        self.coupling_reg = coupling_reg
        self.bind_fraction = bind_fraction
        self.beta = beta
        self.gamma = gamma
        self.transient_lifetime = transient_lifetime
        self.top_k_binding = top_k_binding
        self.consolidation_interval = consolidation_interval
        self.corr_ema = corr_ema
        self.theta_merge = theta_merge
        self.theta_prune = theta_prune
        self.prune_patience = prune_patience
        self.theta_seed = theta_seed
        self.seed_scale = seed_scale
        self.merge_parent_decay = merge_parent_decay
        self.prediction_torque_scale = prediction_torque_scale
        self.residual_bank_size = residual_bank_size

        # Same conceptual parameters as me_layer.py, lifted to complex tape space.
        self.tape_init_re = nn.Parameter(torch.randn(self.total_slots) / self.total_slots**0.5)
        self.tape_init_im = nn.Parameter(torch.randn(self.total_slots) / self.total_slots**0.5)
        self.eta = nn.Parameter(torch.tensor(float(eta_init)))
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.torque_bias_re = nn.Parameter(torch.randn(self.total_slots) * 0.01)
        self.torque_bias_im = nn.Parameter(torch.randn(self.total_slots) * 0.01)

        # Delta from COUPLING_THEORY.md Section 5. It maps novelty residuals in
        # ambient hidden space into active memory slots.
        self.sensor_leakage = nn.Parameter(torch.randn(self.total_slots, hidden_dim) * leakage_init)

        self.register_buffer("base_basis_template", self._make_base_basis(), persistent=False)

    def _make_base_basis(self) -> torch.Tensor:
        """
        Build the initial basis E.

        When memory_dim == hidden_dim this is exactly the identity, which matches the
        original `me_layer.py` coordinate system and makes G initially near-identity.
        For compressed memory_dim we use an orthonormal basis so G still starts near I.
        """
        basis = torch.zeros(self.hidden_dim, self.total_slots, dtype=torch.float32)
        if self.memory_dim == self.hidden_dim:
            basis[:, : self.memory_dim] = torch.eye(self.hidden_dim, dtype=torch.float32)
        else:
            q, _ = torch.linalg.qr(torch.randn(self.hidden_dim, self.memory_dim, dtype=torch.float32), mode="reduced")
            basis[:, : self.memory_dim] = q
        return basis

    @property
    def transient_start(self) -> int:
        return self.memory_dim

    @property
    def transient_end(self) -> int:
        return self.memory_dim + self.max_transient_dims

    @property
    def seed_start(self) -> int:
        return self.transient_end

    def _complex_tape_init(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        re = self.tape_init_re.to(device=device, dtype=dtype)
        im = self.tape_init_im.to(device=device, dtype=dtype)
        tape = torch.complex(re, im)
        tape[self.memory_dim :] = 0
        return tape

    def _renormalize(self, tape: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
        """Project active slots back to the unit hypersphere in C^n."""
        masked = tape * active_mask.to(dtype=tape.dtype)
        norm = masked.abs().pow(2).sum(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
        return masked / norm

    def _allocate_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> MemoryEngineState:
        basis = self.base_basis_template.to(device=device, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1)
        active_mask = torch.zeros(batch_size, self.total_slots, dtype=torch.bool, device=device)
        active_mask[:, : self.memory_dim] = True

        tape_init = self._complex_tape_init(device, torch.float32).unsqueeze(0).repeat(batch_size, 1)
        tape = self._renormalize(tape_init, active_mask)
        prev_tape = tape.clone()

        slot_kind = torch.full((batch_size, self.total_slots), -1, dtype=torch.long, device=device)
        slot_kind[:, : self.memory_dim] = BASE_SLOT

        state = MemoryEngineState(
            tape=tape,
            prev_tape=prev_tape,
            basis=basis,
            gram=torch.zeros(batch_size, self.total_slots, self.total_slots, dtype=torch.float32, device=device),
            coupling=torch.zeros(batch_size, self.total_slots, self.total_slots, dtype=torch.float32, device=device),
            active_mask=active_mask,
            slot_kind=slot_kind,
            lifetime=torch.zeros(batch_size, self.total_slots, dtype=torch.long, device=device),
            prune_count=torch.zeros(batch_size, self.total_slots, dtype=torch.long, device=device),
            binding_sources=torch.full((batch_size, self.total_slots, 2), -1, dtype=torch.long, device=device),
            corr=torch.zeros(batch_size, self.total_slots, self.total_slots, dtype=torch.complex64, device=device),
            residual_bank=torch.zeros(batch_size, self.residual_bank_size, self.hidden_dim, dtype=torch.float32, device=device),
            residual_count=torch.zeros(batch_size, dtype=torch.long, device=device),
            residual_ptr=torch.zeros(batch_size, dtype=torch.long, device=device),
            step=torch.zeros(batch_size, dtype=torch.long, device=device),
            basis_dirty=torch.ones(batch_size, dtype=torch.bool, device=device),
        )
        self._refresh_coupling(state)
        return state

    def initialize_state(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> MemoryEngineState:
        """Public constructor for persistent state caches."""
        return self._allocate_state(batch_size, device, dtype)

    def reset_state(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> MemoryEngineState:
        """Alias used by the hybrid integration example."""
        return self.initialize_state(batch_size, device, dtype)

    def _refresh_coupling(self, state: MemoryEngineState) -> None:
        """Recompute G = E^T E and L = (G + eps I)^-1 only when the basis changed."""
        if not torch.any(state.basis_dirty):
            return

        eye = torch.eye(self.total_slots, device=state.basis.device, dtype=state.basis.dtype)
        for b in torch.nonzero(state.basis_dirty, as_tuple=False).flatten():
            basis_b = state.basis[b]
            active = state.active_mask[b].to(dtype=basis_b.dtype)
            basis_active = basis_b * active.unsqueeze(0)
            gram = basis_active.transpose(0, 1) @ basis_active
            inactive_diag = (~state.active_mask[b]).to(dtype=gram.dtype)
            gram = gram + torch.diag(inactive_diag)
            gram = gram + self.coupling_reg * eye
            coupling = torch.linalg.inv(gram)
            state.gram[b] = gram
            state.coupling[b] = coupling
            state.basis_dirty[b] = False

    def _classify_regimes(self, reception: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Regime masks using the complex criterion from engine.py / the paper."""
        re = reception.real
        im = reception.imag.abs()
        mag = reception.abs()
        resonance = (re > 1e-6) & (im < re)
        torque = (re < -1e-6) | (im >= re.abs())
        orth = ~(resonance | torque) | (mag < 1e-8)
        return resonance, torque, orth

    def _project_input(
        self,
        hidden: torch.Tensor,
        state: MemoryEngineState,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Project into the current basis E and compute coupled coordinates.

        w = E^T h
        alpha = L w
        h_recon = E alpha

        The novelty residual h - h_recon is the ambient-space signal used by Delta
        and later by the residual seeding path.
        """
        self._refresh_coupling(state)
        # Basis and coupling are mutable cache tensors, not learnable parameters.
        # Detaching them keeps autograd focused on the causal tape dynamics while
        # avoiding version-counter failures when transient slots mutate the cache.
        basis = state.basis.detach().clone()
        coupling = state.coupling.detach().clone()
        coords = torch.einsum("bd,bdn->bn", hidden, basis)
        coupled = torch.bmm(coupling, coords.unsqueeze(-1)).squeeze(-1)
        coupled = coupled * state.active_mask.to(dtype=coupled.dtype)
        recon = torch.einsum("bn,bdn->bd", coupled, basis)
        residual = hidden - recon
        return coords, coupled, residual

    def _apply_sensor_leakage(self, residual: torch.Tensor, state: MemoryEngineState) -> torch.Tensor:
        """Delta residual path from COUPLING_THEORY.md Section 5."""
        leak = torch.matmul(residual, self.sensor_leakage.t())
        return leak * state.active_mask.to(dtype=leak.dtype)

    def _reserve_slot(
        self,
        state: MemoryEngineState,
        batch_index: int,
        slot_range: Tuple[int, int],
    ) -> Optional[int]:
        start, end = slot_range
        for slot in range(start, end):
            if not state.active_mask[batch_index, slot]:
                return slot
        return None

    def _release_slot(self, state: MemoryEngineState, batch_index: int, slot: int) -> None:
        state.active_mask[batch_index, slot] = False
        state.slot_kind[batch_index, slot] = -1
        state.lifetime[batch_index, slot] = 0
        state.prune_count[batch_index, slot] = 0
        state.binding_sources[batch_index, slot] = -1
        state.tape[batch_index, slot] = 0
        state.prev_tape[batch_index, slot] = 0
        state.basis[batch_index, :, slot] = 0
        state.corr[batch_index, slot, :] = 0
        state.corr[batch_index, :, slot] = 0
        state.basis_dirty[batch_index] = True

    def _decay_transients(self, state: MemoryEngineState) -> None:
        """Apply lifetime and decay to transient conjunctive slots."""
        if self.max_transient_dims == 0:
            return
        transient_mask = state.slot_kind == TRANSIENT_SLOT
        if not torch.any(transient_mask):
            return

        state.tape = state.tape.clone()
        state.tape[transient_mask] = state.tape[transient_mask] * self.gamma
        state.lifetime[transient_mask] -= 1

        for b in range(state.tape.shape[0]):
            to_release = torch.nonzero(
                transient_mask[b]
                & ((state.lifetime[b] <= 0) | (state.tape[b].abs() < 1e-6)),
                as_tuple=False,
            ).flatten()
            for slot in to_release.tolist():
                self._release_slot(state, b, slot)

        state.tape = self._renormalize(state.tape, state.active_mask)

    def _find_existing_binding(self, state: MemoryEngineState, batch_index: int, left: int, right: int) -> Optional[int]:
        transient_slots = torch.nonzero(state.slot_kind[batch_index] == TRANSIENT_SLOT, as_tuple=False).flatten()
        for slot in transient_slots.tolist():
            src_left, src_right = state.binding_sources[batch_index, slot].tolist()
            if {src_left, src_right} == {left, right}:
                return slot
        return None

    def _apply_fast_binding(
        self,
        state: MemoryEngineState,
        alpha_eff: torch.Tensor,
        reception: torch.Tensor,
    ) -> List[Dict[str, int]]:
        """
        Sparse fast binding from standalone_me_binding.py.

        New transient slots are inserted immediately so their basis contribution is
        routed into the output residual on the same step, while the Gram inverse is
        refreshed lazily for the next token.
        """
        batch_size = reception.shape[0]
        events: List[Dict[str, int]] = []
        state.tape = state.tape.clone()
        state.prev_tape = state.prev_tape.clone()
        state.basis = state.basis.clone()

        for b in range(batch_size):
            mag = reception[b].abs()
            active_idx = torch.nonzero(state.active_mask[b], as_tuple=False).flatten()
            if active_idx.numel() < 2:
                events.append({"new": 0, "refreshed": 0, "active": int((state.slot_kind[b] == TRANSIENT_SLOT).sum().item())})
                continue

            k = min(self.top_k_binding, active_idx.numel())
            masked_mag = torch.full_like(mag, -1.0)
            masked_mag[active_idx] = mag[active_idx]
            top_values, top_idx = torch.topk(masked_mag, k)
            valid_top = top_values > 0
            top_idx = top_idx[valid_top]
            if top_idx.numel() < 2:
                events.append({"new": 0, "refreshed": 0, "active": int((state.slot_kind[b] == TRANSIENT_SLOT).sum().item())})
                continue

            top_reception = reception[b, top_idx]
            top_mag = top_reception.abs()
            top_phase = torch.angle(top_reception)
            pair_i, pair_j = torch.triu_indices(top_idx.numel(), top_idx.numel(), offset=1, device=top_idx.device)
            if pair_i.numel() == 0:
                events.append({"new": 0, "refreshed": 0, "active": int((state.slot_kind[b] == TRANSIENT_SLOT).sum().item())})
                continue

            scores = top_mag[pair_i] * top_mag[pair_j] * torch.cos(top_phase[pair_i] - top_phase[pair_j])
            positive = scores > 0
            if not torch.any(positive):
                events.append({"new": 0, "refreshed": 0, "active": int((state.slot_kind[b] == TRANSIENT_SLOT).sum().item())})
                continue

            positive_scores = scores[positive]
            n_to_bind = max(1, int(torch.ceil(torch.tensor(positive_scores.numel() * self.bind_fraction)).item()))
            threshold = torch.topk(positive_scores, min(n_to_bind, positive_scores.numel())).values[-1]
            selected = positive & (scores >= threshold)

            new_count = 0
            refreshed = 0
            for idx in torch.nonzero(selected, as_tuple=False).flatten():
                left = int(top_idx[pair_i[idx]].item())
                right = int(top_idx[pair_j[idx]].item())
                existing = self._find_existing_binding(state, b, left, right)
                if existing is not None:
                    state.lifetime[b, existing] = self.transient_lifetime
                    refreshed += 1
                    continue

                slot = self._reserve_slot(state, b, (self.transient_start, self.transient_end))
                if slot is None:
                    continue

                basis_left = state.basis[b, :, left]
                basis_right = state.basis[b, :, right]
                new_basis = basis_left + basis_right
                norm = new_basis.norm().clamp_min(1e-8)
                new_basis = new_basis / norm

                phase = torch.angle(state.tape[b, left] * state.tape[b, right])
                slot_value = torch.complex(
                    torch.tensor(self.beta, device=phase.device, dtype=torch.float32) * torch.cos(phase),
                    torch.tensor(self.beta, device=phase.device, dtype=torch.float32) * torch.sin(phase),
                )

                state.active_mask[b, slot] = True
                state.slot_kind[b, slot] = TRANSIENT_SLOT
                state.lifetime[b, slot] = self.transient_lifetime
                state.binding_sources[b, slot, 0] = left
                state.binding_sources[b, slot, 1] = right
                state.basis[b, :, slot] = new_basis
                state.tape[b, slot] = slot_value.to(dtype=state.tape.dtype)
                state.prev_tape[b, slot] = state.tape[b, slot]
                alpha_eff[b, slot] = 0.5 * (alpha_eff[b, left] + alpha_eff[b, right])
                state.basis_dirty[b] = True
                new_count += 1

            state.tape[b] = self._renormalize(state.tape[b].unsqueeze(0), state.active_mask[b].unsqueeze(0)).squeeze(0)
            events.append(
                {
                    "new": new_count,
                    "refreshed": refreshed,
                    "active": int((state.slot_kind[b] == TRANSIENT_SLOT).sum().item()),
                }
            )

        return events

    def _store_residual(self, state: MemoryEngineState, batch_index: int, residual: torch.Tensor) -> None:
        ptr = int(state.residual_ptr[batch_index].item())
        state.residual_bank[batch_index, ptr] = residual
        state.residual_ptr[batch_index] = (ptr + 1) % self.residual_bank_size
        state.residual_count[batch_index] = torch.clamp(state.residual_count[batch_index] + 1, max=self.residual_bank_size)

    def _seed_from_vector(
        self,
        state: MemoryEngineState,
        batch_index: int,
        vector: torch.Tensor,
        slot_kind: int,
        tape_scale: float,
    ) -> bool:
        slot = self._reserve_slot(state, batch_index, (self.seed_start, self.total_slots))
        if slot is None:
            return False

        direction = vector / vector.norm().clamp_min(1e-8)
        state.active_mask[batch_index, slot] = True
        state.slot_kind[batch_index, slot] = slot_kind
        state.basis[batch_index, :, slot] = direction.to(dtype=state.basis.dtype)
        state.tape[batch_index, slot] = torch.complex(
            torch.tensor(tape_scale, device=vector.device, dtype=torch.float32),
            torch.tensor(0.0, device=vector.device, dtype=torch.float32),
        )
        state.prev_tape[batch_index, slot] = state.tape[batch_index, slot]
        state.basis_dirty[batch_index] = True
        state.tape[batch_index] = self._renormalize(state.tape[batch_index].unsqueeze(0), state.active_mask[batch_index].unsqueeze(0)).squeeze(0)
        return True

    def _run_consolidation(self, state: MemoryEngineState) -> List[Dict[str, int]]:
        """
        Consolidation loop from paper/root.md Section 4:

        - merge from cross-correlation eigenstructure
        - prune weak slots
        - seed from novelty residual principal components
        """
        if self.consolidation_interval <= 0:
            return []

        batch_events: List[Dict[str, int]] = []
        for b in range(state.tape.shape[0]):
            merged = 0
            pruned = 0
            seeded = 0

            active = torch.nonzero(state.active_mask[b], as_tuple=False).flatten()
            stable_active = active[state.slot_kind[b, active] != TRANSIENT_SLOT]
            if stable_active.numel() > 1:
                corr = state.corr[b][stable_active][:, stable_active]
                herm = 0.5 * (corr + corr.conj().transpose(0, 1))
                eigvals, eigvecs = torch.linalg.eigh(herm)
                top_eig = eigvals[-1].real
                if top_eig > self.theta_merge:
                    weights = eigvecs[:, -1].real
                    merged_vector = torch.einsum("k,dk->d", weights, state.basis[b, :, stable_active])
                    if self._seed_from_vector(state, b, merged_vector, MERGED_SLOT, self.seed_scale * float(top_eig.clamp_max(2.0))):
                        state.tape[b, stable_active] *= self.merge_parent_decay
                        state.tape[b] = self._renormalize(state.tape[b].unsqueeze(0), state.active_mask[b].unsqueeze(0)).squeeze(0)
                        merged = 1

            magnitudes = state.tape[b].abs()
            for slot in active.tolist():
                if state.slot_kind[b, slot].item() == BASE_SLOT:
                    continue
                if magnitudes[slot] < self.theta_prune:
                    state.prune_count[b, slot] += 1
                else:
                    state.prune_count[b, slot] = 0
                if state.prune_count[b, slot] >= self.prune_patience:
                    self._release_slot(state, b, slot)
                    pruned += 1

            count = int(state.residual_count[b].item())
            if count > 1:
                residuals = state.residual_bank[b, :count]
                centered = residuals - residuals.mean(dim=0, keepdim=True)
                # SVD on the residual bank is a stable low-rank proxy for the novelty
                # principal component without forming a full hidden_dim covariance.
                _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
                strength = singular_values[0] / max(count**0.5, 1.0)
                if strength > self.theta_seed:
                    seeded = int(self._seed_from_vector(state, b, vh[0], SEEDED_SLOT, self.seed_scale))

            batch_events.append({"merged": merged, "pruned": pruned, "seeded": seeded})

        return batch_events

    def _decode_delta(self, basis: torch.Tensor, delta_tape: torch.Tensor) -> torch.Tensor:
        """
        Map the real part of the tape delta back to hidden space.

        This keeps the external transformer interface purely real-valued while still
        allowing internal phase structure to influence output via the update dynamics.
        """
        return torch.einsum("bn,bdn->bd", delta_tape.real, basis.detach().clone())

    def forward(
        self,
        hidden_states: torch.Tensor,
        state: Optional[MemoryEngineState] = None,
        return_diagnostics: bool = True,
    ) -> Tuple[torch.Tensor, MemoryEngineState, Dict[str, torch.Tensor | List[Dict[str, int]]]]:
        """
        Process hidden states through the memory engine.

        Returns:
            output_hidden: same shape as input
            next_state: persistent state for continued decoding
            diagnostics: scalar/token metrics plus discrete binding/consolidation events
        """
        input_dtype = hidden_states.dtype
        hidden_states_fp32 = hidden_states.to(dtype=torch.float32)
        batch_size, seq_len, _ = hidden_states_fp32.shape

        if state is None:
            state = self._allocate_state(batch_size, hidden_states.device, hidden_states_fp32.dtype)
        elif state.tape.shape[0] != batch_size:
            raise ValueError(f"State batch size {state.tape.shape[0]} does not match input batch size {batch_size}.")

        outputs = []
        pr_trace = []
        gini_trace = []
        resonance_trace = []
        torque_trace = []
        orth_trace = []
        self_torque_trace = []
        coupling_trace = []
        pred_error_trace = []
        binding_events_all: List[Dict[str, int]] = []
        consolidation_events_all: List[Dict[str, int]] = []

        eta = self.eta.abs()
        alpha_gate = torch.sigmoid(self.alpha)
        torque_bias = torch.complex(self.torque_bias_re.float(), self.torque_bias_im.float()).to(hidden_states.device)

        for t in range(seq_len):
            self._decay_transients(state)

            hidden_t = hidden_states_fp32[:, t, :]
            _, coupled, novelty_residual = self._project_input(hidden_t, state)
            leakage = self._apply_sensor_leakage(novelty_residual, state)
            alpha_eff = (coupled + leakage) * state.active_mask.to(dtype=coupled.dtype)

            reception = torch.complex(alpha_eff, torch.zeros_like(alpha_eff)) * state.tape
            binding_events = self._apply_fast_binding(state, alpha_eff, reception)
            reception = torch.complex(alpha_eff, torch.zeros_like(alpha_eff)) * state.tape

            resonance_mask, torque_mask, orth_mask = self._classify_regimes(reception)

            update = eta * reception
            update = update + eta * torque_mask.to(dtype=update.dtype) * torque_bias.unsqueeze(0)
            update = update * (~orth_mask).to(dtype=update.dtype)

            tape_before = state.tape.clone()
            tape_world = self._renormalize(state.tape + update, state.active_mask)

            # PHASE5_ANTICIPATION.md: A(t) = s(t-1), e(t) = s(t) - s(t-1),
            # c_pred(t) = e(t) ⊙ s(t).
            prediction = torch.where(
                (state.step > 0).unsqueeze(-1),
                state.prev_tape,
                tape_before,
            )
            pred_error = tape_world - prediction
            c_pred = pred_error * tape_world
            _, pred_torque, _ = self._classify_regimes(c_pred)
            pred_update = eta * self.prediction_torque_scale * c_pred * pred_torque.to(dtype=c_pred.dtype)
            tape_after = self._renormalize(tape_world + pred_update, state.active_mask)

            delta_tape = tape_after - tape_before
            decoded_delta = self._decode_delta(state.basis, delta_tape)
            output_t = hidden_t + alpha_gate * decoded_delta
            outputs.append(output_t.to(dtype=input_dtype))

            # Update long-timescale statistics after the token update.
            outer = reception.unsqueeze(-1) * reception.conj().unsqueeze(-2)
            state.corr = ((1.0 - self.corr_ema) * state.corr + self.corr_ema * outer).detach()

            novelty_signal = novelty_residual + self._decode_delta(state.basis, pred_error)
            for b in range(batch_size):
                self._store_residual(state, b, novelty_signal[b].detach())

            state.prev_tape = tape_after
            state.tape = tape_after
            state.step = state.step + 1

            if self.consolidation_interval > 0 and int(state.step.max().item()) % self.consolidation_interval == 0:
                consolidation_events_all.extend(self._run_consolidation(state))

            active_magnitudes = state.tape.abs() * state.active_mask.to(dtype=state.tape.real.dtype)
            active_count = state.active_mask.sum(dim=-1).clamp_min(1)
            pr_trace.append(_participation_ratio(active_magnitudes).detach())
            gini_trace.append(_gini_coefficient(active_magnitudes).detach())
            resonance_trace.append((resonance_mask.sum(dim=-1) / active_count).detach())
            torque_trace.append((torque_mask.sum(dim=-1) / active_count).detach())
            orth_trace.append((orth_mask.sum(dim=-1) / active_count).detach())
            self_torque_trace.append(
                (pred_update.abs().sum(dim=-1) / (update.abs().sum(dim=-1).clamp_min(1e-8))).detach()
            )
            coupling_trace.append(
                ((state.gram - torch.eye(self.total_slots, device=state.gram.device).unsqueeze(0)).norm(dim=(-2, -1)) / active_count.float()).detach()
            )
            pred_error_trace.append(pred_error.abs().mean(dim=-1).detach())
            binding_events_all.extend(binding_events)

        output = torch.stack(outputs, dim=1)
        diagnostics: Dict[str, torch.Tensor | List[Dict[str, int]]] = {}
        if return_diagnostics:
            diagnostics = {
                "pr": torch.stack(pr_trace, dim=1),
                "gini": torch.stack(gini_trace, dim=1),
                "resonance_fraction": torch.stack(resonance_trace, dim=1),
                "torque_fraction": torch.stack(torque_trace, dim=1),
                "orthogonality_fraction": torch.stack(orth_trace, dim=1),
                "self_torque": torch.stack(self_torque_trace, dim=1),
                "coupling_strength": torch.stack(coupling_trace, dim=1),
                "prediction_error": torch.stack(pred_error_trace, dim=1),
                "active_slots": state.active_mask.sum(dim=-1),
                "binding_events": binding_events_all,
                "consolidation_events": consolidation_events_all,
            }
        return output, state, diagnostics
