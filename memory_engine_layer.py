"""
Production Memory Engine layer for hybrid causal LMs.

The layer keeps the hybrid insertion API used elsewhere in the repo while
compressing the trainable surface so three inserted layers stay lightweight.

Framework mapping:

- Coupled reception:
    w = E^T h
    alpha = L w
    L = (E^T E + epsilon)^-1
  where epsilon is a learned low-rank Hermitian approximation.
- Anticipatory operator:
    s_pred = W_pred s_prev
    e = s_world - s_pred
    c_pred = e ⊙ s_world
- Recurrence gates:
    w_r controls recurrent weight
    breadth_gate controls how broadly self-reception is applied
- Regime-aware update:
    resonance / torque / orthogonality masks gate the update
- Soft consolidation:
    trainable theta_merge / theta_prune thresholds drive merge/prune decisions
    every K steps, with novelty seeding from a residual bank

The basis E is stateful rather than learned, so the trainable parameter count is
dominated by low-rank coupling and prediction factors instead of dense hidden
projections. This keeps the ME insertions in the tens-of-thousands of
parameters even when the host model is much larger.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


BASE_SLOT = 0
TRANSIENT_SLOT = 1
MERGED_SLOT = 2
SEEDED_SLOT = 3


def _inverse_softplus(value: float) -> float:
    value_tensor = torch.tensor(max(float(value), 1e-6))
    return float(torch.log(torch.expm1(value_tensor)))


def _participation_ratio(magnitudes: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mags_sq = magnitudes.pow(2)
    numer = mags_sq.sum(dim=-1).pow(2)
    denom = mags_sq.pow(2).sum(dim=-1).clamp_min(eps)
    return numer / denom


def _gini_coefficient(magnitudes: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    sorted_mags, _ = torch.sort(magnitudes, dim=-1)
    n = sorted_mags.shape[-1]
    idx = torch.arange(1, n + 1, device=magnitudes.device, dtype=magnitudes.dtype)
    weighted = (idx * sorted_mags).sum(dim=-1)
    total = sorted_mags.sum(dim=-1).clamp_min(eps)
    return (2.0 * weighted / (n * total)) - (n + 1.0) / n


@dataclass
class MemoryEngineState:
    """Per-sequence state for chunked decoding."""

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
    Compact hybrid Memory Engine layer for transformer hidden states.

    The layer operates in a compressed memory space of `memory_dim + max_aux_dims`
    slots and projects back into the host hidden space through the current basis.
    Learned operators are low-rank so the trainable surface stays small:

    - low-rank Hermitian epsilon for coupling
    - low-rank prediction operator W_pred
    - vector recurrence / breadth / torque parameters
    - trainable consolidation thresholds
    """

    def __init__(
        self,
        hidden_dim: int,
        memory_dim: Optional[int] = None,
        max_aux_dims: int = 16,
        max_transient_dims: int = 8,
        eta_init: float = 0.08,
        alpha_init: float = 0.35,
        coupling_reg: float = 1e-3,
        coupling_rank: int = 10,
        prediction_rank: int = 10,
        coupling_window: float = 0.35,
        prediction_window: float = 0.25,
        bind_fraction: float = 0.15,
        beta: float = 0.08,
        gamma: float = 0.92,
        transient_lifetime: int = 5,
        top_k_binding: int = 8,
        consolidation_interval: int = 8,
        corr_ema: float = 0.05,
        theta_merge: float = 0.35,
        theta_prune: float = 0.015,
        theta_seed: float = 0.08,
        prune_patience: int = 2,
        seed_scale: float = 0.05,
        merge_parent_decay: float = 0.85,
        prediction_torque_scale: float = 0.4,
        residual_bank_size: int = 8,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_dim = min(memory_dim or hidden_dim, hidden_dim)
        self.max_aux_dims = max_aux_dims
        self.max_transient_dims = min(max_transient_dims, max_aux_dims)
        self.total_slots = self.memory_dim + self.max_aux_dims

        self.coupling_reg = coupling_reg
        self.coupling_rank = max(int(coupling_rank), 0)
        self.prediction_rank = max(int(prediction_rank), 0)
        self.coupling_window = float(coupling_window)
        self.prediction_window = float(prediction_window)
        self.bind_fraction = float(bind_fraction)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.transient_lifetime = int(transient_lifetime)
        self.top_k_binding = int(top_k_binding)
        self.consolidation_interval = int(consolidation_interval)
        self.corr_ema = float(corr_ema)
        self.prune_patience = int(prune_patience)
        self.seed_scale = float(seed_scale)
        self.merge_parent_decay = float(merge_parent_decay)
        self.prediction_torque_scale = float(prediction_torque_scale)
        self.residual_bank_size = int(residual_bank_size)

        self.tape_init_re = nn.Parameter(torch.randn(self.total_slots) / self.total_slots**0.5)
        self.tape_init_im = nn.Parameter(torch.randn(self.total_slots) / self.total_slots**0.5)
        self.eta_raw = nn.Parameter(torch.tensor(_inverse_softplus(eta_init)))
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.torque_rotation = nn.Parameter(torch.zeros(self.total_slots))
        self.w_r = nn.Parameter(torch.zeros(self.total_slots))
        self.breadth_gate = nn.Parameter(torch.zeros(self.total_slots))

        if self.coupling_rank > 0:
            self.epsilon_factor = nn.Parameter(torch.randn(self.total_slots, self.coupling_rank) * 0.02)
            self.epsilon_scale = nn.Parameter(torch.zeros(self.coupling_rank))
        else:
            self.register_parameter("epsilon_factor", None)
            self.register_parameter("epsilon_scale", None)
        self.epsilon_diag = nn.Parameter(torch.zeros(self.total_slots))

        if self.prediction_rank > 0:
            self.pred_factor = nn.Parameter(torch.randn(self.total_slots, self.prediction_rank) * 0.02)
            self.pred_scale = nn.Parameter(torch.zeros(self.prediction_rank))
        else:
            self.register_parameter("pred_factor", None)
            self.register_parameter("pred_scale", None)
        self.pred_diag = nn.Parameter(torch.zeros(self.total_slots))

        self.theta_merge_raw = nn.Parameter(torch.tensor(_inverse_softplus(theta_merge)))
        self.theta_prune_raw = nn.Parameter(torch.tensor(_inverse_softplus(theta_prune)))
        self.theta_seed_raw = nn.Parameter(torch.tensor(_inverse_softplus(theta_seed)))

        self.register_buffer("base_basis_template", self._make_base_basis(), persistent=False)

    @property
    def eta(self) -> nn.Parameter:
        return self.eta_raw

    @property
    def theta_merge(self) -> float:
        return float(F.softplus(self.theta_merge_raw).detach().cpu())

    @theta_merge.setter
    def theta_merge(self, value: float) -> None:
        self.theta_merge_raw.data.fill_(_inverse_softplus(value))

    @property
    def theta_prune(self) -> float:
        return float(F.softplus(self.theta_prune_raw).detach().cpu())

    @theta_prune.setter
    def theta_prune(self, value: float) -> None:
        self.theta_prune_raw.data.fill_(_inverse_softplus(value))

    @property
    def theta_seed(self) -> float:
        return float(F.softplus(self.theta_seed_raw).detach().cpu())

    @theta_seed.setter
    def theta_seed(self, value: float) -> None:
        self.theta_seed_raw.data.fill_(_inverse_softplus(value))

    @property
    def transient_start(self) -> int:
        return self.memory_dim

    @property
    def transient_end(self) -> int:
        return self.memory_dim + self.max_transient_dims

    @property
    def seed_start(self) -> int:
        return self.transient_end

    def eta_value(self) -> torch.Tensor:
        return F.softplus(self.eta_raw)

    def _make_base_basis(self) -> torch.Tensor:
        basis = torch.zeros(self.hidden_dim, self.total_slots, dtype=torch.float32)
        if self.memory_dim == self.hidden_dim:
            basis[:, : self.memory_dim] = torch.eye(self.hidden_dim, dtype=torch.float32)
        else:
            q, _ = torch.linalg.qr(
                torch.randn(self.hidden_dim, self.memory_dim, dtype=torch.float32),
                mode="reduced",
            )
            basis[:, : self.memory_dim] = q
        return basis

    def _complex_tape_init(self, device: torch.device) -> torch.Tensor:
        re = self.tape_init_re.to(device=device, dtype=torch.float32)
        im = self.tape_init_im.to(device=device, dtype=torch.float32)
        tape = torch.complex(re, im)
        if self.max_aux_dims > 0:
            tape[self.memory_dim :] = 0
        return tape

    def _renormalize(self, tape: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
        masked = tape * active_mask.to(dtype=tape.dtype)
        norm = masked.abs().pow(2).sum(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
        return masked / norm

    def _allocate_state(self, batch_size: int, device: torch.device) -> MemoryEngineState:
        basis = self.base_basis_template.to(device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        active_mask = torch.zeros(batch_size, self.total_slots, dtype=torch.bool, device=device)
        active_mask[:, : self.memory_dim] = True

        tape_init = self._complex_tape_init(device).unsqueeze(0).repeat(batch_size, 1)
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
            residual_bank=torch.zeros(
                batch_size,
                self.residual_bank_size,
                self.hidden_dim,
                dtype=torch.float32,
                device=device,
            ),
            residual_count=torch.zeros(batch_size, dtype=torch.long, device=device),
            residual_ptr=torch.zeros(batch_size, dtype=torch.long, device=device),
            step=torch.zeros(batch_size, dtype=torch.long, device=device),
            basis_dirty=torch.ones(batch_size, dtype=torch.bool, device=device),
        )
        self._refresh_coupling(state)
        return state

    def initialize_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> MemoryEngineState:
        del dtype
        return self._allocate_state(batch_size=batch_size, device=device)

    def reset_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> MemoryEngineState:
        return self.initialize_state(batch_size=batch_size, device=device, dtype=dtype)

    def _learned_epsilon(self, device: torch.device) -> torch.Tensor:
        diag = (self.coupling_window * torch.tanh(self.epsilon_diag)).to(device=device, dtype=torch.float32)
        epsilon = torch.diag(diag)
        if self.coupling_rank <= 0:
            return epsilon

        factor = self.epsilon_factor.to(device=device, dtype=torch.float32)
        scaled = factor * torch.tanh(self.epsilon_scale).to(device=device, dtype=torch.float32).unsqueeze(0)
        low_rank = (scaled @ factor.transpose(0, 1)) / math.sqrt(max(self.total_slots, 1))
        low_rank = 0.5 * (low_rank + low_rank.transpose(0, 1))
        low_rank = low_rank - torch.diag(torch.diagonal(low_rank))
        return epsilon + self.coupling_window * low_rank

    def _prediction_operator(self, device: torch.device) -> torch.Tensor:
        eye = torch.eye(self.total_slots, device=device, dtype=torch.float32)
        diag = (self.prediction_window * torch.tanh(self.pred_diag)).to(device=device, dtype=torch.float32)
        operator = eye + torch.diag(diag)
        if self.prediction_rank <= 0:
            return operator

        factor = self.pred_factor.to(device=device, dtype=torch.float32)
        scaled = factor * torch.tanh(self.pred_scale).to(device=device, dtype=torch.float32).unsqueeze(0)
        low_rank = (scaled @ factor.transpose(0, 1)) / math.sqrt(max(self.total_slots, 1))
        low_rank = 0.5 * (low_rank + low_rank.transpose(0, 1))
        return operator + self.prediction_window * low_rank

    def W_pred(self, device: Optional[torch.device] = None) -> torch.Tensor:
        target_device = device or self.alpha.device
        return self._prediction_operator(target_device)

    def _compute_gram_and_coupling(
        self,
        basis: torch.Tensor,
        active_mask: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        basis_active = basis * active_mask.unsqueeze(1).to(dtype=basis.dtype)
        gram = torch.matmul(basis_active.transpose(1, 2), basis_active)
        epsilon = self._learned_epsilon(device).unsqueeze(0)
        eye = torch.eye(self.total_slots, device=device, dtype=torch.float32).unsqueeze(0)
        inactive_diag = (~active_mask).to(dtype=torch.float32)
        gram = gram + epsilon + torch.diag_embed(inactive_diag) + self.coupling_reg * eye
        coupling = torch.linalg.inv(gram)
        return gram, coupling

    def _refresh_coupling(self, state: MemoryEngineState) -> None:
        if not torch.any(state.basis_dirty):
            return

        with torch.no_grad():
            dirty_indices = torch.nonzero(state.basis_dirty, as_tuple=False).flatten()
            if dirty_indices.numel() == 0:
                return

            gram, coupling = self._compute_gram_and_coupling(
                basis=state.basis[dirty_indices],
                active_mask=state.active_mask[dirty_indices],
                device=state.basis.device,
            )
            state.gram[dirty_indices] = gram
            state.coupling[dirty_indices] = coupling
            state.basis_dirty[dirty_indices] = False

    def _classify_regimes(self, reception: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        real_part = reception.real
        imag_part = reception.imag.abs()
        magnitude = reception.abs()
        resonance = (real_part > 1e-6) & (imag_part < real_part)
        torque = (real_part < -1e-6) | (imag_part >= real_part.abs())
        orthogonality = ~(resonance | torque) | (magnitude < 1e-8)
        return resonance, torque, orthogonality

    def _project_input(
        self,
        hidden: torch.Tensor,
        state: MemoryEngineState,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._refresh_coupling(state)
        basis = state.basis.to(device=hidden.device, dtype=torch.float32)
        gram, coupling = self._compute_gram_and_coupling(
            basis=basis,
            active_mask=state.active_mask,
            device=hidden.device,
        )
        with torch.no_grad():
            state.gram.copy_(gram.detach())
            state.coupling.copy_(coupling.detach())

        coords = torch.einsum("bd,bdn->bn", hidden, basis)
        coupled = torch.bmm(coupling, coords.unsqueeze(-1)).squeeze(-1)
        coupled = coupled * state.active_mask.to(dtype=coupled.dtype)
        recon = torch.einsum("bn,bdn->bd", coupled, basis)
        residual = hidden - recon
        return coords, coupled, residual

    def _apply_sensor_leakage(self, residual: torch.Tensor, state: MemoryEngineState) -> torch.Tensor:
        basis = state.basis.to(device=residual.device, dtype=residual.dtype)
        leak = torch.einsum("bd,bdn->bn", residual, basis)
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
        if self.max_transient_dims == 0:
            return
        transient_mask = state.slot_kind == TRANSIENT_SLOT
        if not torch.any(transient_mask):
            return

        state.tape = state.tape.clone()
        state.tape[transient_mask] = state.tape[transient_mask] * self.gamma
        state.lifetime[transient_mask] -= 1

        for batch_index in range(state.tape.shape[0]):
            to_release = torch.nonzero(
                transient_mask[batch_index]
                & ((state.lifetime[batch_index] <= 0) | (state.tape[batch_index].abs() < 1e-6)),
                as_tuple=False,
            ).flatten()
            for slot in to_release.tolist():
                self._release_slot(state, batch_index, slot)

        state.tape = self._renormalize(state.tape, state.active_mask)

    def _find_existing_binding(
        self,
        state: MemoryEngineState,
        batch_index: int,
        left: int,
        right: int,
    ) -> Optional[int]:
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
        batch_size = reception.shape[0]
        events: List[Dict[str, int]] = []
        state.tape = state.tape.clone()
        state.prev_tape = state.prev_tape.clone()
        state.basis = state.basis.clone()

        for batch_index in range(batch_size):
            active_idx = torch.nonzero(state.active_mask[batch_index], as_tuple=False).flatten()
            if active_idx.numel() < 2:
                events.append(
                    {
                        "new": 0,
                        "refreshed": 0,
                        "active": int((state.slot_kind[batch_index] == TRANSIENT_SLOT).sum().item()),
                    }
                )
                continue

            magnitude = reception[batch_index].abs()
            masked_mag = torch.full_like(magnitude, -1.0)
            masked_mag[active_idx] = magnitude[active_idx]
            top_count = min(self.top_k_binding, active_idx.numel())
            top_values, top_idx = torch.topk(masked_mag, top_count)
            top_idx = top_idx[top_values > 0]
            if top_idx.numel() < 2:
                events.append(
                    {
                        "new": 0,
                        "refreshed": 0,
                        "active": int((state.slot_kind[batch_index] == TRANSIENT_SLOT).sum().item()),
                    }
                )
                continue

            top_reception = reception[batch_index, top_idx]
            pair_i, pair_j = torch.triu_indices(top_idx.numel(), top_idx.numel(), offset=1, device=top_idx.device)
            if pair_i.numel() == 0:
                events.append(
                    {
                        "new": 0,
                        "refreshed": 0,
                        "active": int((state.slot_kind[batch_index] == TRANSIENT_SLOT).sum().item()),
                    }
                )
                continue

            score = top_reception.abs()[pair_i] * top_reception.abs()[pair_j]
            score = score * torch.cos(torch.angle(top_reception[pair_i]) - torch.angle(top_reception[pair_j]))
            positive = score > 0
            if not torch.any(positive):
                events.append(
                    {
                        "new": 0,
                        "refreshed": 0,
                        "active": int((state.slot_kind[batch_index] == TRANSIENT_SLOT).sum().item()),
                    }
                )
                continue

            positive_scores = score[positive]
            n_to_bind = max(1, int(math.ceil(positive_scores.numel() * self.bind_fraction)))
            threshold = torch.topk(positive_scores, min(n_to_bind, positive_scores.numel())).values[-1]
            selected = positive & (score >= threshold)

            new_count = 0
            refreshed = 0
            for idx in torch.nonzero(selected, as_tuple=False).flatten():
                left = int(top_idx[pair_i[idx]].item())
                right = int(top_idx[pair_j[idx]].item())
                existing = self._find_existing_binding(state, batch_index, left, right)
                if existing is not None:
                    state.lifetime[batch_index, existing] = self.transient_lifetime
                    refreshed += 1
                    continue

                slot = self._reserve_slot(state, batch_index, (self.transient_start, self.transient_end))
                if slot is None:
                    continue

                basis_left = state.basis[batch_index, :, left]
                basis_right = state.basis[batch_index, :, right]
                new_basis = basis_left + basis_right
                new_basis = new_basis / new_basis.norm().clamp_min(1e-8)

                phase = torch.angle(state.tape[batch_index, left] * state.tape[batch_index, right])
                slot_value = torch.complex(
                    torch.tensor(self.beta, device=phase.device, dtype=torch.float32) * torch.cos(phase),
                    torch.tensor(self.beta, device=phase.device, dtype=torch.float32) * torch.sin(phase),
                )

                state.active_mask[batch_index, slot] = True
                state.slot_kind[batch_index, slot] = TRANSIENT_SLOT
                state.lifetime[batch_index, slot] = self.transient_lifetime
                state.binding_sources[batch_index, slot, 0] = left
                state.binding_sources[batch_index, slot, 1] = right
                state.basis[batch_index, :, slot] = new_basis
                state.tape[batch_index, slot] = slot_value.to(dtype=state.tape.dtype)
                state.prev_tape[batch_index, slot] = state.tape[batch_index, slot]
                alpha_eff[batch_index, slot] = 0.5 * (alpha_eff[batch_index, left] + alpha_eff[batch_index, right])
                state.basis_dirty[batch_index] = True
                new_count += 1

            state.tape[batch_index] = self._renormalize(
                state.tape[batch_index].unsqueeze(0),
                state.active_mask[batch_index].unsqueeze(0),
            ).squeeze(0)
            events.append(
                {
                    "new": new_count,
                    "refreshed": refreshed,
                    "active": int((state.slot_kind[batch_index] == TRANSIENT_SLOT).sum().item()),
                }
            )

        return events

    def _store_residual(self, state: MemoryEngineState, batch_index: int, residual: torch.Tensor) -> None:
        ptr = int(state.residual_ptr[batch_index].item())
        state.residual_bank[batch_index, ptr] = residual
        state.residual_ptr[batch_index] = (ptr + 1) % self.residual_bank_size
        state.residual_count[batch_index] = torch.clamp(
            state.residual_count[batch_index] + 1,
            max=self.residual_bank_size,
        )

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
        state.tape[batch_index] = self._renormalize(
            state.tape[batch_index].unsqueeze(0),
            state.active_mask[batch_index].unsqueeze(0),
        ).squeeze(0)
        return True

    def _run_consolidation(self, state: MemoryEngineState) -> List[Dict[str, int]]:
        if self.consolidation_interval <= 0:
            return []

        merge_threshold = F.softplus(self.theta_merge_raw)
        prune_threshold = F.softplus(self.theta_prune_raw)
        seed_threshold = F.softplus(self.theta_seed_raw)
        batch_events: List[Dict[str, int]] = []

        for batch_index in range(state.tape.shape[0]):
            merged = 0
            pruned = 0
            seeded = 0

            active = torch.nonzero(state.active_mask[batch_index], as_tuple=False).flatten()
            stable_active = active[state.slot_kind[batch_index, active] != TRANSIENT_SLOT]
            if stable_active.numel() > 1:
                corr = state.corr[batch_index][stable_active][:, stable_active]
                herm = 0.5 * (corr + corr.conj().transpose(0, 1))
                eigvals, eigvecs = torch.linalg.eigh(herm)
                top_eig = eigvals[-1].real
                merge_gate = torch.sigmoid((top_eig - merge_threshold) * 8.0)
                if merge_gate.item() > 0.5:
                    weights = eigvecs[:, -1].real
                    merged_vector = torch.einsum(
                        "k,dk->d",
                        weights,
                        state.basis[batch_index, :, stable_active],
                    )
                    seeded_scale = self.seed_scale * float((merge_gate * top_eig.clamp_max(2.0)).item())
                    if self._seed_from_vector(state, batch_index, merged_vector, MERGED_SLOT, seeded_scale):
                        parent_decay = 1.0 - (1.0 - self.merge_parent_decay) * merge_gate.item()
                        state.tape[batch_index, stable_active] *= parent_decay
                        state.tape[batch_index] = self._renormalize(
                            state.tape[batch_index].unsqueeze(0),
                            state.active_mask[batch_index].unsqueeze(0),
                        ).squeeze(0)
                        merged = 1

            magnitudes = state.tape[batch_index].abs()
            for slot in active.tolist():
                if state.slot_kind[batch_index, slot].item() == BASE_SLOT:
                    continue
                prune_gate = torch.sigmoid((prune_threshold - magnitudes[slot]) * 8.0)
                if prune_gate.item() > 0.5:
                    state.prune_count[batch_index, slot] += 1
                else:
                    state.prune_count[batch_index, slot] = 0
                if state.prune_count[batch_index, slot] >= self.prune_patience:
                    self._release_slot(state, batch_index, slot)
                    pruned += 1

            count = int(state.residual_count[batch_index].item())
            if count > 1:
                residuals = state.residual_bank[batch_index, :count]
                centered = residuals - residuals.mean(dim=0, keepdim=True)
                _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
                strength = singular_values[0] / max(count**0.5, 1.0)
                seed_gate = torch.sigmoid((strength - seed_threshold) * 8.0)
                if seed_gate.item() > 0.5:
                    seeded = int(
                        self._seed_from_vector(
                            state,
                            batch_index,
                            vh[0],
                            SEEDED_SLOT,
                            self.seed_scale * float(seed_gate.item()),
                        )
                    )

            batch_events.append({"merged": merged, "pruned": pruned, "seeded": seeded})

        return batch_events

    def _decode_delta(self, basis: torch.Tensor, delta_tape: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bn,bdn->bd", delta_tape.real, basis.detach())

    def _predict_next_tape(self, prev_tape: torch.Tensor) -> torch.Tensor:
        operator = self._prediction_operator(prev_tape.device)
        pred_real = torch.matmul(prev_tape.real, operator.transpose(0, 1))
        pred_imag = torch.matmul(prev_tape.imag, operator.transpose(0, 1))
        return torch.complex(pred_real, pred_imag)

    def forward(
        self,
        hidden_states: torch.Tensor,
        state: Optional[MemoryEngineState] = None,
        return_diagnostics: bool = True,
    ) -> Tuple[torch.Tensor, MemoryEngineState, Dict[str, torch.Tensor | List[Dict[str, int]]]]:
        input_dtype = hidden_states.dtype
        hidden_states_fp32 = hidden_states.to(dtype=torch.float32)
        batch_size, seq_len, _ = hidden_states_fp32.shape

        if state is None:
            state = self.initialize_state(batch_size=batch_size, device=hidden_states.device)
        elif state.tape.shape[0] != batch_size:
            raise ValueError(
                f"State batch size {state.tape.shape[0]} does not match input batch size {batch_size}."
            )

        outputs = []
        pr_trace = []
        gini_trace = []
        resonance_trace = []
        torque_trace = []
        orth_trace = []
        self_torque_trace = []
        coupling_trace = []
        pred_error_trace = []
        merge_trace = []
        prune_trace = []
        binding_trace = []
        binding_events_all: List[Dict[str, int]] = []
        consolidation_events_all: List[Dict[str, int]] = []

        eta = self.eta_value()
        alpha_gate = torch.sigmoid(self.alpha)
        recurrence_gate = torch.sigmoid(self.w_r).to(device=hidden_states.device, dtype=torch.float32)
        breadth_gate = torch.sigmoid(self.breadth_gate).to(device=hidden_states.device, dtype=torch.float32)
        torque_phase = torch.exp(1j * self.torque_rotation.to(device=hidden_states.device, dtype=torch.float32))

        eye = torch.eye(self.total_slots, device=hidden_states.device, dtype=torch.float32)

        for token_index in range(seq_len):
            self._decay_transients(state)

            hidden_t = hidden_states_fp32[:, token_index, :]
            _, coupled, novelty_residual = self._project_input(hidden_t, state)
            leakage = self._apply_sensor_leakage(novelty_residual, state)
            alpha_eff = (coupled + leakage) * state.active_mask.to(dtype=coupled.dtype)

            recurrent_gain = recurrence_gate.unsqueeze(0) * breadth_gate.unsqueeze(0)
            recurrent_reception = recurrent_gain.to(dtype=state.tape.dtype) * state.prev_tape * state.tape

            world_reception = torch.complex(alpha_eff, torch.zeros_like(alpha_eff)) * state.tape
            reception = world_reception + recurrent_reception
            binding_events = self._apply_fast_binding(state, alpha_eff, reception)

            world_reception = torch.complex(alpha_eff, torch.zeros_like(alpha_eff)) * state.tape
            recurrent_reception = recurrent_gain.to(dtype=state.tape.dtype) * state.prev_tape * state.tape
            reception = world_reception + recurrent_reception

            resonance_mask, torque_mask, orth_mask = self._classify_regimes(reception)
            rotated_tape = state.tape * torque_phase.unsqueeze(0).to(dtype=state.tape.dtype)
            torque_delta = rotated_tape - state.tape

            update = eta * resonance_mask.to(dtype=reception.dtype) * reception
            update = update + eta * torque_mask.to(dtype=reception.dtype) * (reception + torque_delta)
            update = update * (~orth_mask).to(dtype=update.dtype)

            tape_before = state.tape.clone()
            tape_world = self._renormalize(state.tape + update, state.active_mask)

            predicted_tape = self._predict_next_tape(state.prev_tape)
            predicted_tape = predicted_tape * state.active_mask.to(dtype=predicted_tape.dtype)
            prediction = torch.where((state.step > 0).unsqueeze(-1), predicted_tape, tape_before)
            pred_error = tape_world - prediction
            c_pred = pred_error * tape_world
            _, pred_torque, _ = self._classify_regimes(c_pred)
            pred_update = (
                eta
                * self.prediction_torque_scale
                * pred_torque.to(dtype=c_pred.dtype)
                * c_pred
            )
            tape_after = self._renormalize(tape_world + pred_update, state.active_mask)

            delta_tape = tape_after - tape_before
            decoded_delta = self._decode_delta(state.basis, delta_tape)
            output_t = hidden_t + alpha_gate * decoded_delta
            outputs.append(output_t.to(dtype=input_dtype))

            outer = tape_after.unsqueeze(-1) * tape_after.conj().unsqueeze(-2)
            state.corr = ((1.0 - self.corr_ema) * state.corr + self.corr_ema * outer).detach()

            novelty_signal = novelty_residual + self._decode_delta(state.basis, pred_error)
            for batch_index in range(batch_size):
                self._store_residual(state, batch_index, novelty_signal[batch_index].detach())

            state.prev_tape = tape_after
            state.tape = tape_after
            state.step = state.step + 1

            merge_fraction = torch.zeros(batch_size, device=hidden_states.device, dtype=torch.float32)
            prune_fraction = torch.zeros(batch_size, device=hidden_states.device, dtype=torch.float32)
            if self.consolidation_interval > 0 and int(state.step.max().item()) % self.consolidation_interval == 0:
                consolidation_events = self._run_consolidation(state)
                consolidation_events_all.extend(consolidation_events)
                for batch_index, event in enumerate(consolidation_events):
                    active_count = state.active_mask[batch_index].sum().clamp_min(1).float()
                    merge_fraction[batch_index] = float(event["merged"]) / active_count
                    prune_fraction[batch_index] = float(event["pruned"]) / active_count

            active_magnitudes = state.tape.abs() * state.active_mask.to(dtype=state.tape.real.dtype)
            active_count = state.active_mask.sum(dim=-1).clamp_min(1)
            coupling_strength = (
                (state.coupling - eye.unsqueeze(0)).abs().mean(dim=(-2, -1))
            )
            bound_fraction = torch.tensor(
                [event["new"] + event["refreshed"] for event in binding_events],
                device=hidden_states.device,
                dtype=torch.float32,
            ) / active_count.float()

            pr_trace.append(_participation_ratio(active_magnitudes).detach())
            gini_trace.append(_gini_coefficient(active_magnitudes).detach())
            resonance_trace.append((resonance_mask.sum(dim=-1) / active_count).detach())
            torque_trace.append((torque_mask.sum(dim=-1) / active_count).detach())
            orth_trace.append((orth_mask.sum(dim=-1) / active_count).detach())
            self_torque_trace.append(
                (pred_update.abs().sum(dim=-1) / update.abs().sum(dim=-1).clamp_min(1e-8)).detach()
            )
            coupling_trace.append(coupling_strength.detach())
            pred_error_trace.append(pred_error.abs().mean(dim=-1).detach())
            merge_trace.append(merge_fraction.detach())
            prune_trace.append(prune_fraction.detach())
            binding_trace.append(bound_fraction.detach())
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
                "merge_fraction": torch.stack(merge_trace, dim=1),
                "prune_fraction": torch.stack(prune_trace, dim=1),
                "binding_fraction": torch.stack(binding_trace, dim=1),
                "active_slots": state.active_mask.sum(dim=-1),
                "binding_events": binding_events_all,
                "consolidation_events": consolidation_events_all,
            }
        return output, state, diagnostics
