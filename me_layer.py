"""
Trainable Memory Engine layer plus legacy GPT-style integration helpers.

This module now exposes two paths:

1. ``MemoryEngineLayer``: the pure, stacked ME layer used by the new
   ``MemoryEngineLLM``. It implements the coupled reception, anticipatory
   torque, recurrence gates, operating-regime masking, renormalization, and
   differentiable soft consolidation described in the repo papers.
2. ``GPT2WithMemoryEngine`` / ``create_model``: the existing frozen-base-model
   compatibility path used elsewhere in the repository.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from example_hybrid_integration import (
    PostAttentionMemoryWrapper,
    PostBlockMemoryWrapper,
    collect_memory_diagnostics,
    install_memory_engine,
    reset_memory_engine,
)


def _inverse_softplus(value: float) -> float:
    value_tensor = torch.tensor(max(float(value), 1e-6))
    return float(torch.log(torch.expm1(value_tensor)))


def _participation_ratio(magnitudes: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mags_sq = magnitudes.pow(2)
    numer = mags_sq.sum(dim=-1).pow(2)
    denom = mags_sq.pow(2).sum(dim=-1).clamp_min(eps)
    return numer / denom


@dataclass
class MemoryEngineState:
    """Persistent per-layer tape state for chunked causal processing."""

    tape: torch.Tensor
    prev_tape: torch.Tensor
    corr: torch.Tensor
    step: torch.Tensor


class _PureMemoryEngineCore(nn.Module):
    """
    Pure Memory Engine sequence layer used in the hierarchical ME-LLM.

    The implementation follows the repo papers directly:

    - docs/COUPLING_THEORY.md:
      ``w = E(x_t)``, ``alpha = L @ w``, ``c = alpha * s`` with trainable
      Hermitian ``epsilon`` defining ``L = (I + epsilon)^-1``.
    - docs/PHASE5_ANTICIPATION.md:
      ``e = s - W_pred @ s_prev`` and ``c_pred = e * s`` as prediction torque.
    - docs/PHASE3_STABILITY.md:
      regime-aware updates stay inside the operating window by keeping
      ``epsilon`` small and renormalizing after every step.

    The layer is still real-valued at its interface. The tape itself is complex;
    downstream layers consume a real projection of ``[Re(s), Im(s)]``.
    """

    def __init__(
        self,
        hidden_dim: int,
        memory_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        eta_init: float = 0.1,
        alpha_init: float = 0.5,
        coupling_reg: float = 1e-3,
        coupling_mode: str = "full",
        coupling_window: float = 0.35,
        max_aux_dims: int = 0,
        max_transient_dims: int = 0,
        bind_fraction: float = 0.15,
        beta: float = 0.5,
        gamma: float = 0.5,
        top_k_binding: int = 8,
        consolidation_interval: int = 8,
        corr_ema: float = 0.05,
        theta_merge: float = 0.35,
        theta_prune: float = 0.02,
        prediction_torque_scale: float = 0.5,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim or hidden_dim
        self.output_dim = output_dim or hidden_dim

        # Legacy kwargs are accepted so higher-level modules do not need to fork.
        self.max_aux_dims = max_aux_dims
        self.max_transient_dims = max_transient_dims
        self.top_k_binding = top_k_binding
        self.bind_fraction = bind_fraction

        self.coupling_reg = coupling_reg
        self.coupling_mode = coupling_mode
        self.coupling_window = coupling_window
        self.beta = beta
        self.gamma = gamma
        self.consolidation_interval = consolidation_interval
        self.corr_ema = corr_ema
        self.prediction_torque_scale = prediction_torque_scale

        if self.hidden_dim == self.memory_dim:
            self.input_projection: nn.Module = nn.Identity()
        else:
            self.input_projection = nn.Linear(self.hidden_dim, self.memory_dim, bias=False)

        self.output_projection = nn.Linear(2 * self.memory_dim, self.output_dim, bias=False)

        # Persistent tape and the framework hyperparameters are trainable end to end.
        self.tape_init = nn.Parameter(
            torch.randn(self.memory_dim, dtype=torch.complex64) / math.sqrt(self.memory_dim)
        )
        self.eta_raw = nn.Parameter(torch.tensor(_inverse_softplus(eta_init)))
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.torque_rotation = nn.Parameter(torch.zeros(self.memory_dim))

        # Hermitian coupling epsilon: L = (I + epsilon)^-1.
        self.epsilon = nn.Parameter(
            0.01 * torch.randn(self.memory_dim, self.memory_dim, dtype=torch.complex64)
        )
        self.epsilon_diag = nn.Parameter(torch.zeros(self.memory_dim))

        # Linear state predictor from PHASE5_ANTICIPATION.md.
        self.W_pred = nn.Parameter(torch.eye(self.memory_dim, dtype=torch.complex64))

        # Speed/directness/breadth gates.
        self.w_r = nn.Parameter(torch.zeros(self.memory_dim))
        self.breadth_gate = nn.Parameter(torch.zeros(self.memory_dim))

        # Soft consolidation thresholds are trainable through backprop.
        self.theta_merge_raw = nn.Parameter(torch.tensor(_inverse_softplus(theta_merge)))
        self.theta_prune_raw = nn.Parameter(torch.tensor(_inverse_softplus(theta_prune)))

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        if isinstance(self.input_projection, nn.Linear):
            nn.init.eye_(self.input_projection.weight)

        with torch.no_grad():
            self.output_projection.weight.zero_()
            width = min(self.output_dim, self.memory_dim)
            self.output_projection.weight[:width, :width] = torch.eye(width)

    @property
    def eta(self) -> nn.Parameter:
        return self.eta_raw

    def eta_value(self) -> torch.Tensor:
        return F.softplus(self.eta_raw)

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

    def initialize_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> MemoryEngineState:
        del dtype
        tape = self._renormalize(
            self.tape_init.to(device=device).unsqueeze(0).expand(batch_size, -1)
        )
        return MemoryEngineState(
            tape=tape,
            prev_tape=tape.clone(),
            corr=torch.zeros(
                batch_size,
                self.memory_dim,
                self.memory_dim,
                device=device,
                dtype=torch.complex64,
            ),
            step=torch.zeros(batch_size, device=device, dtype=torch.long),
        )

    def reset_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> MemoryEngineState:
        return self.initialize_state(batch_size=batch_size, device=device, dtype=dtype)

    def _renormalize(self, tape: torch.Tensor) -> torch.Tensor:
        norm = tape.abs().pow(2).sum(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
        return tape / norm

    def _bounded_hermitian_epsilon(self, device: torch.device) -> torch.Tensor:
        scale = self.coupling_window / math.sqrt(max(self.memory_dim, 1))
        epsilon = 0.5 * (self.epsilon + self.epsilon.mH)
        epsilon = epsilon - torch.diag(torch.diagonal(epsilon))
        diag = scale * torch.tanh(self.epsilon_diag).to(device=device, dtype=torch.float32)
        epsilon = scale * torch.tanh(epsilon.to(device=device))
        epsilon = epsilon + torch.diag(diag.to(dtype=torch.complex64))
        return epsilon

    def _compute_coupling(self, device: torch.device) -> torch.Tensor:
        eye = torch.eye(self.memory_dim, device=device, dtype=torch.complex64)
        if self.coupling_mode == "diagonal":
            diag = 1.0 + (self.coupling_window * torch.tanh(self.epsilon_diag)).to(
                device=device,
                dtype=torch.float32,
            )
            return torch.diag((diag + self.coupling_reg).reciprocal().to(dtype=torch.complex64))

        epsilon = self._bounded_hermitian_epsilon(device)
        gram = eye + epsilon + self.coupling_reg * eye
        return torch.linalg.inv(gram)

    def _classify_regimes(
        self,
        reception: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        real_part = reception.real
        imag_part = reception.imag.abs()
        magnitude = reception.abs()
        resonance = (real_part > 1e-6) & (imag_part < real_part)
        torque = (real_part < -1e-6) | (imag_part >= real_part.abs())
        orthogonality = ~(resonance | torque) | (magnitude < 1e-8)
        return resonance, torque, orthogonality

    def _complex_input(self, hidden: torch.Tensor) -> torch.Tensor:
        projected = self.input_projection(hidden)
        zeros = torch.zeros_like(projected)
        return torch.complex(projected, zeros)

    def _predict_next_tape(self, prev_tape: torch.Tensor) -> torch.Tensor:
        return torch.matmul(prev_tape, self.W_pred.transpose(0, 1))

    def _soft_consolidate(
        self,
        tape: torch.Tensor,
        corr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.memory_dim < 2:
            zeros = torch.zeros(tape.shape[0], self.memory_dim, device=tape.device, dtype=tape.real.dtype)
            return tape, zeros, zeros

        corr_mag = corr.abs()
        corr_mag = corr_mag - torch.diag_embed(torch.diagonal(corr_mag, dim1=-2, dim2=-1))

        partner_idx = corr_mag.argmax(dim=-1)
        partner_tape = torch.gather(tape, 1, partner_idx)

        merge_score = corr_mag.max(dim=-1).values
        merge_threshold = F.softplus(self.theta_merge_raw)
        merge_gate = torch.sigmoid((merge_score - merge_threshold) * 8.0)
        merge_gate = self.bind_fraction * merge_gate

        merged_target = 0.5 * (tape + partner_tape)
        tape = (1.0 - self.beta * merge_gate).to(dtype=tape.dtype) * tape + (
            self.beta * merge_gate
        ).to(dtype=tape.dtype) * merged_target

        prune_threshold = F.softplus(self.theta_prune_raw)
        prune_gate = torch.sigmoid((prune_threshold - tape.abs()) * 8.0)
        tape = tape * (1.0 - self.gamma * prune_gate).to(dtype=tape.dtype)
        tape = self._renormalize(tape)
        return tape, merge_gate, prune_gate

    def forward(
        self,
        hidden_states: torch.Tensor,
        state: Optional[MemoryEngineState] = None,
        return_diagnostics: bool = True,
    ) -> Tuple[torch.Tensor, MemoryEngineState, Dict[str, torch.Tensor]]:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(dtype=torch.float32)
        batch_size, seq_len, _ = hidden_states.shape

        if state is None:
            state = self.initialize_state(batch_size, hidden_states.device, hidden_states.dtype)
        elif state.tape.shape[0] != batch_size:
            raise ValueError(
                f"State batch size {state.tape.shape[0]} does not match input batch size {batch_size}."
            )

        coupling = self._compute_coupling(hidden_states.device)
        eta = self.eta_value()
        directness = torch.sigmoid(self.alpha)
        recurrence_gate = torch.sigmoid(self.w_r).to(hidden_states.device)
        breadth = torch.sigmoid(self.breadth_gate).to(hidden_states.device)
        torque_rotation = torch.exp(1j * self.torque_rotation.to(hidden_states.device))

        outputs = []
        tape_trace = []
        pr_trace = []
        resonance_trace = []
        torque_trace = []
        orth_trace = []
        prediction_error_trace = []
        merge_trace = []
        prune_trace = []

        for token_index in range(seq_len):
            world_signal = self._complex_input(hidden_states[:, token_index, :])
            # docs/COUPLING_THEORY.md: alpha = L @ w.
            coupled_signal = torch.matmul(world_signal, coupling.transpose(0, 1))

            # Recurrence gates: eta = speed, alpha = directness, breadth_gate = breadth.
            recurrent_signal = recurrence_gate.unsqueeze(0).to(dtype=torch.complex64) * state.prev_tape
            effective_signal = breadth.unsqueeze(0).to(dtype=torch.complex64) * (
                directness * coupled_signal + recurrent_signal
            )

            # Core reception from the papers: c = alpha * s.
            reception = effective_signal * state.tape
            resonance_mask, torque_mask, orth_mask = self._classify_regimes(reception)

            rotated_tape = state.tape * torque_rotation.unsqueeze(0).to(dtype=state.tape.dtype)
            torque_delta = rotated_tape - state.tape
            regime_update = resonance_mask.to(dtype=reception.dtype) * reception
            regime_update = regime_update + torque_mask.to(dtype=reception.dtype) * (reception + torque_delta)
            regime_update = regime_update * (~orth_mask).to(dtype=reception.dtype)

            tape_world = self._renormalize(state.tape + eta.to(dtype=reception.real.dtype) * regime_update)

            # docs/PHASE5_ANTICIPATION.md: e = s - W_pred @ s_prev, c_pred = e * s.
            predicted_tape = self._predict_next_tape(state.prev_tape)
            prediction_error = tape_world - predicted_tape
            prediction_reception = prediction_error * tape_world
            _, prediction_torque, _ = self._classify_regimes(prediction_reception)
            prediction_update = (
                eta.to(dtype=prediction_reception.real.dtype)
                * self.prediction_torque_scale
                * (1.0 - directness)
                * prediction_torque.to(dtype=prediction_reception.dtype)
                * prediction_reception
            )

            tape_after = self._renormalize(tape_world + prediction_update)

            # Running correlation is the co-activation signal for soft consolidation.
            outer = tape_after.unsqueeze(-1) * tape_after.conj().unsqueeze(-2)
            state.corr = (1.0 - self.corr_ema) * state.corr + self.corr_ema * outer

            merge_gate = torch.zeros_like(tape_after.real)
            prune_gate = torch.zeros_like(tape_after.real)
            next_step = state.step + 1
            if self.consolidation_interval > 0 and bool(torch.any(next_step % self.consolidation_interval == 0)):
                tape_after, merge_gate, prune_gate = self._soft_consolidate(tape_after, state.corr)

            tape_features = torch.cat([tape_after.real, tape_after.imag], dim=-1)
            output_token = self.output_projection(tape_features)

            outputs.append(output_token.to(dtype=input_dtype))
            tape_trace.append(tape_features)
            pr_trace.append(_participation_ratio(tape_after.abs()).detach())
            active_dims = torch.full(
                (batch_size,),
                float(self.memory_dim),
                device=hidden_states.device,
                dtype=torch.float32,
            )
            resonance_trace.append((resonance_mask.sum(dim=-1).float() / active_dims).detach())
            torque_trace.append((torque_mask.sum(dim=-1).float() / active_dims).detach())
            orth_trace.append((orth_mask.sum(dim=-1).float() / active_dims).detach())
            prediction_error_trace.append(prediction_error.abs().mean(dim=-1).detach())
            merge_trace.append(merge_gate.mean(dim=-1).detach())
            prune_trace.append(prune_gate.mean(dim=-1).detach())

            state.prev_tape = state.tape
            state.tape = tape_after
            state.step = next_step

        output = torch.stack(outputs, dim=1)
        diagnostics: Dict[str, torch.Tensor] = {}
        if return_diagnostics:
            coupling_strength = (
                (coupling - torch.eye(self.memory_dim, device=coupling.device, dtype=coupling.dtype))
                .abs()
                .mean()
                .expand(batch_size)
            )
            diagnostics = {
                "tape_features": torch.stack(tape_trace, dim=1),
                "pr": torch.stack(pr_trace, dim=1),
                "resonance_fraction": torch.stack(resonance_trace, dim=1),
                "torque_fraction": torch.stack(torque_trace, dim=1),
                "orthogonality_fraction": torch.stack(orth_trace, dim=1),
                "prediction_error": torch.stack(prediction_error_trace, dim=1),
                "merge_fraction": torch.stack(merge_trace, dim=1),
                "prune_fraction": torch.stack(prune_trace, dim=1),
                "coupling_strength": coupling_strength,
            }
        return output, state, diagnostics


def _resolve_decoder_layers(model: nn.Module) -> nn.ModuleList:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError("Unsupported architecture: could not locate decoder layers.")


def _infer_layer_count(model: nn.Module) -> int:
    return len(_resolve_decoder_layers(model))


def _default_insert_after(num_layers: int) -> List[int]:
    if num_layers <= 0:
        return []
    candidates = [num_layers // 4, num_layers // 2, (3 * num_layers) // 4]
    normalized = []
    for idx in candidates:
        idx = min(max(idx, 0), num_layers - 1)
        if idx not in normalized:
            normalized.append(idx)
    return normalized


def _normalize_insert_after(model: nn.Module, insert_after: Optional[Iterable[int]]) -> List[int]:
    num_layers = _infer_layer_count(model)
    requested = list(insert_after) if insert_after is not None else _default_insert_after(num_layers)
    return [idx for idx in requested if 0 <= idx < num_layers]


def _default_memory_kwargs(hidden_dim: int) -> Dict[str, int | float]:
    max_aux_dims = 16 if hidden_dim < 256 else 32
    max_transient_dims = max(4, max_aux_dims // 2)
    return {
        "memory_dim": hidden_dim,
        "max_aux_dims": max_aux_dims,
        "max_transient_dims": max_transient_dims,
        "top_k_binding": min(16, hidden_dim),
        "consolidation_interval": 8,
    }


class MemoryEngineLayer(nn.Module):
    """
    Public wrapper around the pure ME core.

    The wrapper preserves the repo's historical ergonomics:

    - ``forward(hidden_states) -> hidden_states`` by default
    - optional persistent ``state`` for chunked decoding
    - ``last_state`` and ``last_diagnostics`` caches for higher-level modules
    - ``engine`` attribute so node-based wrappers can still tune metaparameters
    """

    def __init__(
        self,
        hidden_dim: int,
        eta_init: float = 0.1,
        alpha_init: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        memory_kwargs = _default_memory_kwargs(hidden_dim)
        memory_kwargs.update(kwargs)
        self.engine = _PureMemoryEngineCore(
            hidden_dim=hidden_dim,
            eta_init=eta_init,
            alpha_init=alpha_init,
            **memory_kwargs,
        )
        self.last_state: Optional[MemoryEngineState] = None
        self.last_diagnostics: Optional[Dict[str, torch.Tensor]] = None

    @property
    def tape_init(self) -> nn.Parameter:
        return self.engine.tape_init

    @property
    def eta(self) -> nn.Parameter:
        return self.engine.eta

    @property
    def alpha(self) -> nn.Parameter:
        return self.engine.alpha

    @property
    def torque_rotation(self) -> nn.Parameter:
        return self.engine.torque_rotation

    @property
    def epsilon(self) -> nn.Parameter:
        return self.engine.epsilon

    @property
    def W_pred(self) -> nn.Parameter:
        return self.engine.W_pred

    @property
    def w_r(self) -> nn.Parameter:
        return self.engine.w_r

    @property
    def breadth_gate(self) -> nn.Parameter:
        return self.engine.breadth_gate

    def initialize_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> MemoryEngineState:
        return self.engine.initialize_state(batch_size=batch_size, device=device, dtype=dtype)

    def reset_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> MemoryEngineState:
        return self.engine.reset_state(batch_size=batch_size, device=device, dtype=dtype)

    def _renormalize(self, tape: torch.Tensor) -> torch.Tensor:
        return self.engine._renormalize(tape)

    def forward(
        self,
        hidden_states: torch.Tensor,
        state: Optional[MemoryEngineState] = None,
        return_state: bool = False,
        return_diagnostics: bool = False,
    ):
        output, next_state, diagnostics = self.engine(
            hidden_states,
            state=state,
            return_diagnostics=True,
        )
        self.last_state = next_state
        self.last_diagnostics = diagnostics

        if return_state or return_diagnostics:
            return output, next_state, diagnostics
        return output

    def get_tape_state(self, hidden_states: torch.Tensor) -> torch.Tensor:
        _, state, _ = self.engine(hidden_states, state=None, return_diagnostics=False)
        return state.tape


class GPT2WithMemoryEngine(nn.Module):
    """
    Compatibility wrapper that installs the existing hybrid memory layer into a
    frozen decoder-only Hugging Face causal LM.

    This path is kept so older scripts continue to work, but the new pure
    hierarchical LLM lives in ``memory_engine_llm.py``.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        insert_after: Optional[Iterable[int]] = None,
        **memory_kwargs,
    ) -> None:
        super().__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.config = self.base_model.config
        self.insert_after = _normalize_insert_after(self.base_model, insert_after)

        for param in self.base_model.parameters():
            param.requires_grad = False

        hidden_dim = getattr(self.config, "hidden_size", getattr(self.config, "n_embd", None))
        if hidden_dim is None:
            raise ValueError("Could not infer hidden size from model config.")

        default_kwargs = _default_memory_kwargs(hidden_dim)
        default_kwargs.update(memory_kwargs)
        install_memory_engine(self.base_model, self.insert_after, **default_kwargs)
        self.reset_memory()

    def _memory_wrappers(self) -> List[nn.Module]:
        layers = _resolve_decoder_layers(self.base_model)
        wrappers = []
        for idx in self.insert_after:
            layer = layers[idx]
            if isinstance(layer, (PostAttentionMemoryWrapper, PostBlockMemoryWrapper)):
                wrappers.append(layer)
        return wrappers

    def reset_memory(self) -> None:
        reset_memory_engine(self.base_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        reset_memory: bool = True,
        **kwargs,
    ) -> Dict:
        if reset_memory:
            self.reset_memory()

        outputs = self.base_model(
            input_ids=input_ids,
            labels=labels,
            output_hidden_states=True,
            use_cache=kwargs.pop("use_cache", False),
            **kwargs,
        )

        diagnostics = []
        for idx, wrapper in zip(self.insert_after, self._memory_wrappers()):
            diagnostics.append(
                {
                    "layer_index": idx,
                    "diagnostics": wrapper.last_memory_diagnostics or {},
                }
            )

        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
            "hidden_states": outputs.hidden_states,
            "me_diagnostics": diagnostics,
        }

    def generate(self, input_ids: torch.Tensor, reset_memory: bool = True, **kwargs) -> torch.Tensor:
        if reset_memory:
            self.reset_memory()
        return self.base_model.generate(input_ids=input_ids, **kwargs)

    def get_me_parameters(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        for wrapper in self._memory_wrappers():
            params.extend(list(wrapper.memory_layer.parameters()))
        return params

    def count_parameters(self) -> Dict[str, int]:
        trainable = sum(p.numel() for p in self.base_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.base_model.parameters())
        frozen = total - trainable
        return {"trainable": trainable, "frozen": frozen, "total": total}

    def collect_memory_diagnostics(self) -> Dict[int, Dict]:
        return collect_memory_diagnostics(self.base_model)


def create_model(
    model_name: str = "gpt2",
    insert_after: Optional[Iterable[int]] = None,
    **memory_kwargs,
) -> Tuple[GPT2WithMemoryEngine, AutoTokenizer]:
    model = GPT2WithMemoryEngine(model_name=model_name, insert_after=insert_after, **memory_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer
