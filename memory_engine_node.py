"""Composable node-based Memory Engine modules.

This module builds a graph-friendly interface on top of the existing
``me_layer.MemoryEngineLayer`` wrapper and the richer
``memory_engine_layer.MemoryEngineLayer`` runtime. Each node keeps its own
persistent ``MemoryEngineState`` with an independent tape ``s``, basis matrix
``E``, Gram matrix ``G = E^T E``, and coupling matrix ``L = G^-1``.

Design notes:
    - ``paper/supplement.md`` §S5.6 motivates the layer as a recurrent tape
      operator that can sit inside larger neural systems.
    - §S6.1–S6.4 specify the consolidation operator (binding, merge, prune,
      seed). The wrapped layer already implements those mechanics per node.
    - §S7 formalizes fracture / pruning thresholds; the wrapped layer's state
      keeps the lifetime, pruning counters, residual bank, and slot metadata
      needed to support those dynamics independently per node.

The classes below add:
    1. Explicit node identity and node-local persistent state/history.
    2. Cross-node communication through the receiving node's own coupling.
    3. Hierarchical low -> mid -> high composition with optional top-down
       predictive signals.
    4. Diagnostics grouped per node for easy integration into larger models.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from me_layer import MemoryEngineLayer as BaseMemoryEngineLayer
from memory_engine_layer import MemoryEngineState


Tensor = torch.Tensor


def _inverse_softplus(value: float) -> float:
    """Initialize a raw parameter whose softplus matches ``value``."""
    value_tensor = torch.tensor(max(float(value), 1e-6))
    return float(torch.log(torch.expm1(value_tensor)))


def _inverse_sigmoid(value: float) -> float:
    """Initialize a raw parameter whose sigmoid matches ``value``."""
    clipped = min(max(float(value), 1e-6), 1.0 - 1e-6)
    value_tensor = torch.tensor(clipped)
    return float(torch.log(value_tensor / (1.0 - value_tensor)))


def _clone_state(state: MemoryEngineState) -> MemoryEngineState:
    """Clone a MemoryEngineState without sharing storage."""
    fields: Dict[str, Any] = {}
    for name in state.__dataclass_fields__:
        value = getattr(state, name)
        fields[name] = value.clone() if torch.is_tensor(value) else value
    return MemoryEngineState(**fields)


def _as_signal_list(
    incoming_signals: Optional[Sequence[Tensor] | Tensor],
) -> List[Tensor]:
    """Normalize optional signals into a flat list."""
    if incoming_signals is None:
        return []
    if torch.is_tensor(incoming_signals):
        return [incoming_signals]
    return [signal for signal in incoming_signals if signal is not None]


class MemoryEngineNode(BaseMemoryEngineLayer):
    """A self-contained Memory Engine node with node-local state and messaging.

    The wrapped base layer already owns the full per-node Memory Engine state:
    tape ``s``, basis ``E``, Gram ``G``, coupling ``L = G^-1``, fast binding
    transients, residual bank, and consolidation loop. This subclass keeps that
    state isolated per node and adds explicit cross-node interfaces.

    Intra-sequence recurrence remains inside the wrapped layer. The node adds a
    higher-level delayed self-reception path across forward calls so state can
    persist across tokens, windows, or conversation turns.
    """

    def __init__(
        self,
        hidden_dim: int,
        input_dim: Optional[int] = None,
        node_name: Optional[str] = None,
        self_recurrence_delay: int = 0,
        self_recurrence_weight: float = 0.0,
        bottom_up_gain: float = 1.0,
        top_down_gain: float = 1.0,
        persistent_state: bool = True,
        history_size: int = 32,
        theta_bind_init: Optional[float] = None,
        **memory_kwargs: Any,
    ) -> None:
        super().__init__(hidden_dim=hidden_dim, **memory_kwargs)
        self.node_name = node_name or f"memory_node_{id(self)}"
        self.input_dim = input_dim or hidden_dim
        self.default_persistent_state = persistent_state
        self.default_recurrence_delay = int(self_recurrence_delay)
        self.gating_temperature = 8.0

        if self.input_dim == hidden_dim:
            self.input_projection = nn.Identity()
        else:
            self.input_projection = nn.Linear(self.input_dim, hidden_dim, bias=False)

        probe_state = self.initialize_state(
            batch_size=1,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.state_dim = int(probe_state.tape.shape[-1])
        self.recurrence_projection = nn.Linear(
            2 * self.state_dim,
            hidden_dim,
            bias=False,
        )

        self.bottom_up_gain = nn.Parameter(torch.tensor(float(bottom_up_gain)))
        self.top_down_gain = nn.Parameter(torch.tensor(float(top_down_gain)))
        self.self_recurrence_gain = nn.Parameter(
            torch.tensor(float(self_recurrence_weight))
        )

        # Learnable metaparams layered on top of the wrapped engine.
        lambda_init = float(getattr(self.engine, "corr_ema", 0.05))
        beta_init = float(getattr(self.engine, "beta", 0.05))
        gamma_init = float(getattr(self.engine, "gamma", 0.9))
        merge_init = float(getattr(self.engine, "theta_merge", 0.4))
        prune_init = float(getattr(self.engine, "theta_prune", 0.015))
        bind_init = (
            float(theta_bind_init)
            if theta_bind_init is not None
            else float(getattr(self.engine, "bind_fraction", 0.15))
        )

        self.consolidation_rate_raw = nn.Parameter(
            torch.tensor(_inverse_softplus(lambda_init))
        )
        self.binding_scale_raw = nn.Parameter(
            torch.tensor(_inverse_softplus(beta_init))
        )
        self.transient_decay_raw = nn.Parameter(
            torch.tensor(_inverse_sigmoid(gamma_init))
        )
        self.theta_bind_raw = nn.Parameter(torch.tensor(_inverse_softplus(bind_init)))
        self.theta_merge_raw = nn.Parameter(torch.tensor(_inverse_softplus(merge_init)))
        self.theta_prune_raw = nn.Parameter(torch.tensor(_inverse_softplus(prune_init)))

        self._tape_history: deque[Tensor] = deque(
            maxlen=max(int(history_size), int(self_recurrence_delay) + 2)
        )
        self._transient_message: Optional[Tensor] = None

    def clear_persistent_state(self) -> None:
        """Clear the wrapped layer's cached state and node-level recurrence."""
        self.last_state = None
        self.last_diagnostics = None
        self._tape_history.clear()
        self._transient_message = None

    def export_state(self) -> Optional[MemoryEngineState]:
        """Return a detached clone of the current persistent state, if present."""
        if self.last_state is None:
            return None
        return _clone_state(self.last_state)

    def load_state(self, state: MemoryEngineState) -> None:
        """Replace the persistent state with a caller-provided clone."""
        self.last_state = _clone_state(state)

    def tape_features(self, state: Optional[MemoryEngineState] = None) -> Tensor:
        """Expose a real-valued tape representation for messaging/projection."""
        working_state = state or self.last_state
        if working_state is None:
            raise RuntimeError("Node state has not been initialized yet.")
        tape = working_state.tape
        return torch.cat([tape.real, tape.imag], dim=-1)

    def metaparam_values(self) -> Dict[str, Tensor]:
        """Return positive / bounded views of the node's learnable controls."""
        return {
            "eta": F.softplus(self.eta),
            "lambda": F.softplus(self.consolidation_rate_raw),
            "w_r": self.self_recurrence_gain,
            "beta": F.softplus(self.binding_scale_raw),
            "gamma": torch.sigmoid(self.transient_decay_raw),
            "theta_bind": F.softplus(self.theta_bind_raw),
            "theta_merge": F.softplus(self.theta_merge_raw),
            "theta_prune": F.softplus(self.theta_prune_raw),
            "bottom_up_gain": self.bottom_up_gain,
            "top_down_gain": self.top_down_gain,
        }

    def summarize_diagnostics(
        self,
        diagnostics: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Reduce raw per-token diagnostics into per-node summaries."""
        summary: Dict[str, Any] = {}
        for key, value in diagnostics.items():
            if torch.is_tensor(value):
                if value.ndim >= 2:
                    summary[key] = value.mean(dim=-1)
                else:
                    summary[key] = value
            else:
                summary[key] = value
        return summary

    def coupled_reception(
        self,
        received_signal: Tensor,
        state: Optional[MemoryEngineState] = None,
    ) -> Dict[str, Tensor]:
        """Compute explicit coupled reception for diagnostics and tests.

        This is the node-local reception rule from the theory:
            w = E^T v
            alpha = L @ w
            c = alpha ⊙ s

        The wrapped layer applies the same rule internally during forward. This
        method makes it explicit so cross-node communication can be inspected
        and verified outside the engine core.
        """
        aligned = self._align_signal(received_signal, allow_input_projection=True)
        working_state = self._resolve_state(
            batch_size=aligned.shape[0],
            device=aligned.device,
            dtype=aligned.dtype,
            state=state,
        )
        basis = working_state.basis.to(device=aligned.device, dtype=aligned.dtype)
        coupling = working_state.coupling.to(
            device=aligned.device,
            dtype=aligned.dtype,
        )
        tape = working_state.tape.to(device=aligned.device)
        w = torch.einsum("btd,bdm->btm", aligned, basis)
        alpha = torch.einsum("btm,bmn->btn", w, coupling)
        c = alpha.to(dtype=tape.dtype) * tape.unsqueeze(1)
        return {
            "w": w,
            "alpha": alpha,
            "c": c,
            "basis": working_state.basis,
            "coupling": working_state.coupling,
            "tape": working_state.tape,
        }

    def forward(
        self,
        hidden_states: Optional[Tensor] = None,
        state: Optional[MemoryEngineState] = None,
        incoming_signals: Optional[Sequence[Tensor] | Tensor] = None,
        top_down_signal: Optional[Tensor] = None,
        persist_state: Optional[bool] = None,
        recurrence_delay: Optional[int] = None,
        recurrence_weight: Optional[float] = None,
        return_state: bool = False,
        return_diagnostics: bool = False,
    ):
        """Run a node update with optional bottom-up, top-down, and self input."""
        signals = _as_signal_list(incoming_signals)
        if hidden_states is None and not signals and top_down_signal is None:
            raise ValueError("A node needs external input or at least one message.")

        batch_size, seq_len, device, dtype = self._infer_sequence_meta(
            hidden_states,
            signals,
            top_down_signal,
        )
        persist = (
            self.default_persistent_state if persist_state is None else persist_state
        )
        working_state = self._resolve_state(batch_size, device, dtype, state=state)

        combined = torch.zeros(
            batch_size,
            seq_len,
            self.hidden_dim,
            device=device,
            dtype=dtype,
        )
        diagnostics: Dict[str, Any] = {"node_name": self.node_name}

        if hidden_states is not None:
            external = self._align_signal(hidden_states, allow_input_projection=True)
            combined = combined + external
            diagnostics["external_norm"] = external.norm(dim=-1)
        else:
            diagnostics["external_norm"] = torch.zeros(
                batch_size,
                seq_len,
                device=device,
                dtype=dtype,
            )

        if signals:
            aligned_messages = torch.stack(
                [self._align_signal(signal) for signal in signals],
                dim=0,
            ).sum(dim=0)
            combined = combined + self.bottom_up_gain.to(dtype) * aligned_messages
            diagnostics["incoming_norm"] = aligned_messages.norm(dim=-1)
        else:
            diagnostics["incoming_norm"] = torch.zeros(
                batch_size,
                seq_len,
                device=device,
                dtype=dtype,
            )

        if top_down_signal is not None:
            aligned_top_down = self._align_signal(top_down_signal)
            combined = combined + self.top_down_gain.to(dtype) * aligned_top_down
            diagnostics["top_down_norm"] = aligned_top_down.norm(dim=-1)
        else:
            diagnostics["top_down_norm"] = torch.zeros(
                batch_size,
                seq_len,
                device=device,
                dtype=dtype,
            )

        effective_delay = (
            self.default_recurrence_delay
            if recurrence_delay is None
            else int(recurrence_delay)
        )
        recurrence_signal = self._build_recurrence_signal(
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
            dtype=dtype,
            recurrence_delay=effective_delay,
        )
        if recurrence_signal is not None:
            recurrence_scale = (
                self.self_recurrence_gain.to(dtype)
                if recurrence_weight is None
                else torch.tensor(recurrence_weight, device=device, dtype=dtype)
            )
            combined = combined + recurrence_scale * recurrence_signal
            diagnostics["recurrent_norm"] = recurrence_signal.norm(dim=-1)
        else:
            diagnostics["recurrent_norm"] = torch.zeros(
                batch_size,
                seq_len,
                device=device,
                dtype=dtype,
            )

        combined, reception_preview, meta_diagnostics, bound_signal = self._apply_metaparam_controls(
            combined=combined,
            state=working_state,
        )
        diagnostics.update(meta_diagnostics)
        diagnostics["message_alpha_norm"] = reception_preview["alpha"].norm(dim=-1)
        diagnostics["message_c_norm"] = reception_preview["c"].abs().norm(dim=-1)
        diagnostics["self_recurrence_delay"] = torch.full(
            (batch_size,),
            effective_delay,
            device=device,
            dtype=torch.long,
        )

        previous_state = self.last_state
        previous_diagnostics = self.last_diagnostics
        self._sync_engine_metaparams()
        output, next_state, base_diagnostics = super().forward(
            combined,
            state=working_state,
            return_state=True,
            return_diagnostics=True,
        )

        diagnostics.update(base_diagnostics)
        diagnostics["active_slots"] = next_state.active_mask.sum(dim=-1)

        if persist:
            self.last_state = _clone_state(next_state)
            self.last_diagnostics = diagnostics
            self._tape_history.append(self.last_state.tape.detach().clone())
            self._update_transient_message(bound_signal)
        else:
            self.last_state = previous_state
            self.last_diagnostics = previous_diagnostics

        if return_state and return_diagnostics:
            return output, next_state, diagnostics
        if return_state:
            return output, next_state
        if return_diagnostics:
            return output, diagnostics
        return output

    def _align_signal(
        self,
        signal: Tensor,
        allow_input_projection: bool = False,
    ) -> Tensor:
        """Align an input or message tensor to the node's hidden dimension."""
        if signal.ndim != 3:
            raise ValueError(
                f"Expected a rank-3 tensor (batch, time, dim), got {signal.shape}."
            )
        if signal.shape[-1] == self.hidden_dim:
            return signal
        if allow_input_projection and signal.shape[-1] == self.input_dim:
            return self.input_projection(signal)
        raise ValueError(
            f"Signal dimension {signal.shape[-1]} does not match hidden_dim="
            f"{self.hidden_dim}."
        )

    def _infer_sequence_meta(
        self,
        hidden_states: Optional[Tensor],
        signals: Sequence[Tensor],
        top_down_signal: Optional[Tensor],
    ) -> Tuple[int, int, torch.device, torch.dtype]:
        """Infer batch/sequence metadata from whichever signal is present."""
        reference = hidden_states
        if reference is None and signals:
            reference = signals[0]
        if reference is None:
            reference = top_down_signal
        if reference is None:
            raise RuntimeError("No reference tensor available to infer sequence meta.")
        return (
            int(reference.shape[0]),
            int(reference.shape[1]),
            reference.device,
            reference.dtype,
        )

    def _resolve_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        state: Optional[MemoryEngineState],
    ) -> MemoryEngineState:
        """Use caller state, cached state, or initialize a fresh node-local state."""
        working_state = state if state is not None else self.last_state
        if working_state is None:
            return self.initialize_state(batch_size=batch_size, device=device, dtype=dtype)
        if working_state.tape.shape[0] != batch_size:
            return self.initialize_state(batch_size=batch_size, device=device, dtype=dtype)
        if working_state.tape.device != device:
            return self.initialize_state(batch_size=batch_size, device=device, dtype=dtype)
        return working_state

    def _build_recurrence_signal(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        recurrence_delay: int,
    ) -> Optional[Tensor]:
        """Project a delayed tape back into the node as an inter-call self signal."""
        if recurrence_delay <= 0 or len(self._tape_history) < recurrence_delay:
            return None
        delayed_tape = self._tape_history[-recurrence_delay]
        if delayed_tape.shape[0] != batch_size:
            return None
        delayed_tape = delayed_tape.to(device=device)
        delayed_features = torch.cat(
            [delayed_tape.real, delayed_tape.imag],
            dim=-1,
        ).to(dtype=dtype)
        projected = self.recurrence_projection(delayed_features)
        return projected.unsqueeze(1).expand(batch_size, seq_len, self.hidden_dim)

    def _apply_metaparam_controls(
        self,
        combined: Tensor,
        state: MemoryEngineState,
    ) -> Tuple[Tensor, Dict[str, Tensor], Dict[str, Tensor], Tensor]:
        """Apply differentiable meta-controls before entering the wrapped engine.

        These paths make the higher-level dynamics learnable even when the
        wrapped engine uses some scalar controls as ordinary Python floats.
        """
        meta = {
            key: value.to(device=combined.device, dtype=combined.dtype)
            for key, value in self.metaparam_values().items()
        }
        basis = state.basis.to(device=combined.device, dtype=combined.dtype)
        survival = torch.sigmoid(
            (state.tape.abs().to(device=combined.device, dtype=combined.dtype) - meta["theta_prune"])
            * self.gating_temperature
        )
        survival_feedback = torch.einsum("bm,bdm->bd", survival, basis)
        survival_feedback = survival_feedback.unsqueeze(1).expand_as(combined)

        bind_gate = torch.sigmoid(
            (combined.abs() - meta["theta_bind"]) * self.gating_temperature
        )
        bound_signal = meta["beta"] * bind_gate * combined

        reception = self.coupled_reception(bound_signal, state=state)
        merge_gate = torch.sigmoid(
            (reception["alpha"].abs() - meta["theta_merge"]) * self.gating_temperature
        )
        coupled_feedback = torch.einsum(
            "btm,bdm->btd",
            reception["alpha"] * merge_gate,
            basis,
        )

        transient_signal = torch.zeros_like(combined)
        if (
            self._transient_message is not None
            and self._transient_message.shape[0] == combined.shape[0]
            and self._transient_message.shape[-1] == combined.shape[-1]
        ):
            transient_signal = self._transient_message.to(
                device=combined.device,
                dtype=combined.dtype,
            ).unsqueeze(1).expand_as(combined)

        transient_scale = meta["gamma"]
        enriched = combined
        enriched = enriched + meta["lambda"] * coupled_feedback
        enriched = enriched + 0.1 * meta["lambda"] * survival_feedback
        enriched = enriched + transient_scale * transient_signal

        diagnostics = {
            "eta_value": meta["eta"].expand(combined.shape[0]),
            "lambda_value": meta["lambda"].expand(combined.shape[0]),
            "beta_value": meta["beta"].expand(combined.shape[0]),
            "gamma_value": meta["gamma"].expand(combined.shape[0]),
            "theta_bind_value": meta["theta_bind"].expand(combined.shape[0]),
            "theta_merge_value": meta["theta_merge"].expand(combined.shape[0]),
            "theta_prune_value": meta["theta_prune"].expand(combined.shape[0]),
            "soft_bind_fraction": bind_gate.mean(dim=-1),
            "soft_merge_fraction": merge_gate.mean(dim=-1),
            "soft_survival_fraction": survival.mean(dim=-1),
            "coupled_feedback_norm": coupled_feedback.norm(dim=-1),
            "transient_norm": transient_signal.norm(dim=-1),
        }
        return enriched, reception, diagnostics, bound_signal

    def _sync_engine_metaparams(self) -> None:
        """Keep the wrapped engine numerically aligned with the soft meta-view."""
        meta = self.metaparam_values()
        self.engine.corr_ema = float(meta["lambda"].detach().cpu())
        self.engine.beta = float(meta["beta"].detach().cpu())
        self.engine.gamma = float(meta["gamma"].detach().cpu())
        self.engine.theta_merge = float(meta["theta_merge"].detach().cpu())
        self.engine.theta_prune = float(meta["theta_prune"].detach().cpu())

    def _update_transient_message(self, bound_signal: Tensor) -> None:
        """Update the node-level transient trace with learnable decay."""
        new_trace = bound_signal.mean(dim=1).detach()
        gamma = torch.sigmoid(self.transient_decay_raw).detach().to(
            device=new_trace.device,
            dtype=new_trace.dtype,
        )
        if (
            self._transient_message is None
            or self._transient_message.shape != new_trace.shape
        ):
            self._transient_message = new_trace
            return
        previous = self._transient_message.to(device=new_trace.device, dtype=new_trace.dtype)
        self._transient_message = gamma * previous + (1.0 - gamma) * new_trace


class HierarchicalMemoryEngine(nn.Module):
    """A clean low -> mid -> high Memory Engine stack.

    Bottom-up flow is the primary path:
        low tape -> mid node -> high node

    Optional top-down flow performs one predictive refinement sweep:
        high tape -> mid node
        refined mid tape -> low node
    """

    def __init__(
        self,
        level_configs: Sequence[Mapping[str, Any]],
        enable_top_down: bool = False,
    ) -> None:
        super().__init__()
        if len(level_configs) < 2:
            raise ValueError("HierarchicalMemoryEngine needs at least two levels.")

        self.enable_top_down = enable_top_down
        self.level_names: List[str] = []
        nodes: List[MemoryEngineNode] = []

        for index, config in enumerate(level_configs):
            level_config = dict(config)
            name = str(level_config.pop("name", f"level_{index}"))
            node = MemoryEngineNode(node_name=name, **level_config)
            self.level_names.append(name)
            nodes.append(node)

        self.levels = nn.ModuleList(nodes)
        self.bottom_up_projections = nn.ModuleList(
            [
                nn.Linear(2 * self.levels[idx].state_dim, self.levels[idx + 1].hidden_dim, bias=False)
                for idx in range(len(self.levels) - 1)
            ]
        )
        self.top_down_projections = nn.ModuleList(
            [
                nn.Linear(2 * self.levels[idx + 1].state_dim, self.levels[idx].hidden_dim, bias=False)
                for idx in range(len(self.levels) - 1)
            ]
        )

    def clear_persistent_state(self) -> None:
        """Reset every node in the hierarchy."""
        for node in self.levels:
            node.clear_persistent_state()

    def metaparam_summary(self) -> Dict[str, Dict[str, float]]:
        """Return detached scalar metaparam snapshots for logging."""
        summary: Dict[str, Dict[str, float]] = {}
        for name, node in zip(self.level_names, self.levels):
            summary[name] = {
                key: float(value.detach().mean().cpu())
                for key, value in node.metaparam_values().items()
            }
        return summary

    def forward(
        self,
        hidden_states: Tensor,
        level_inputs: Optional[Mapping[str, Tensor]] = None,
        apply_top_down: Optional[bool] = None,
        return_states: bool = False,
        return_diagnostics: bool = False,
    ) -> Dict[str, Any]:
        """Run a hierarchical pass and return outputs plus diagnostics."""
        if hidden_states.ndim != 3:
            raise ValueError("Expected input of shape (batch, time, dim).")
        batch_size, seq_len, _ = hidden_states.shape
        use_top_down = self.enable_top_down if apply_top_down is None else apply_top_down
        level_inputs = dict(level_inputs or {})

        outputs: Dict[str, Tensor] = {}
        states: Dict[str, MemoryEngineState] = {}
        diagnostics: Dict[str, Dict[str, Any]] = {}
        summaries: Dict[str, Dict[str, Any]] = {}
        bottom_up_messages: Dict[str, Tensor] = {}
        top_down_messages: Dict[str, Tensor] = {}

        for idx, name in enumerate(self.level_names):
            node = self.levels[idx]
            external = hidden_states if idx == 0 else level_inputs.get(name)
            incoming = None
            if idx > 0:
                source_name = self.level_names[idx - 1]
                incoming = [bottom_up_messages[f"{source_name}->{name}"]]

            output, state, node_diag = node(
                hidden_states=external,
                incoming_signals=incoming,
                return_state=True,
                return_diagnostics=True,
            )
            outputs[name] = output
            states[name] = state
            diagnostics[name] = node_diag
            summaries[name] = node.summarize_diagnostics(node_diag)

            if idx < len(self.levels) - 1:
                target_name = self.level_names[idx + 1]
                tape_message = self.bottom_up_projections[idx](node.tape_features(state))
                bottom_up_messages[f"{name}->{target_name}"] = tape_message.unsqueeze(1).expand(
                    batch_size,
                    seq_len,
                    tape_message.shape[-1],
                )

        if use_top_down and len(self.levels) > 1:
            for idx in reversed(range(len(self.levels) - 1)):
                lower_name = self.level_names[idx]
                higher_name = self.level_names[idx + 1]
                lower_node = self.levels[idx]
                higher_node = self.levels[idx + 1]

                top_down = self.top_down_projections[idx](
                    higher_node.tape_features(states[higher_name])
                )
                top_down_signal = top_down.unsqueeze(1).expand(
                    batch_size,
                    seq_len,
                    top_down.shape[-1],
                )
                top_down_messages[f"{higher_name}->{lower_name}"] = top_down_signal

                external = hidden_states if idx == 0 else level_inputs.get(lower_name)
                incoming = None
                if idx > 0:
                    lower_source = self.level_names[idx - 1]
                    incoming = [bottom_up_messages[f"{lower_source}->{lower_name}"]]

                output, state, node_diag = lower_node(
                    hidden_states=external,
                    incoming_signals=incoming,
                    top_down_signal=top_down_signal,
                    return_state=True,
                    return_diagnostics=True,
                )
                outputs[lower_name] = output
                states[lower_name] = state
                diagnostics[lower_name] = node_diag
                summaries[lower_name] = lower_node.summarize_diagnostics(node_diag)

        result: Dict[str, Any] = {
            "final_output": outputs[self.level_names[-1]],
            "outputs": outputs,
            "summary": summaries,
            "messages": {
                "bottom_up": bottom_up_messages,
                "top_down": top_down_messages,
            },
        }
        if return_states:
            result["states"] = states
        if return_diagnostics:
            result["diagnostics"] = diagnostics
        return result


class MemoryEngineGraph(nn.Module):
    """A minimal directed graph of Memory Engine nodes.

    This class intentionally stays simple: the caller supplies an execution
    order, and only messages from already-updated source nodes are available to
    downstream targets during that pass. That keeps the graph easy to integrate
    into larger models without introducing an additional solver.
    """

    def __init__(self, nodes: Mapping[str, MemoryEngineNode]) -> None:
        super().__init__()
        self.nodes = nn.ModuleDict(dict(nodes))
        self._edges: List[Dict[str, str]] = []
        self.edge_projections = nn.ModuleDict()

    def add_edge(self, source: str, target: str, kind: str = "bottom_up") -> None:
        """Register a directed edge between two existing nodes."""
        if source not in self.nodes or target not in self.nodes:
            raise KeyError("Both source and target must already exist in the graph.")
        edge_key = f"edge_{len(self._edges)}"
        self.edge_projections[edge_key] = nn.Linear(
            2 * self.nodes[source].state_dim,
            self.nodes[target].hidden_dim,
            bias=False,
        )
        self._edges.append(
            {
                "key": edge_key,
                "source": source,
                "target": target,
                "kind": kind,
            }
        )

    def forward(
        self,
        node_inputs: Mapping[str, Tensor],
        execution_order: Optional[Sequence[str]] = None,
        return_states: bool = False,
        return_diagnostics: bool = False,
    ) -> Dict[str, Any]:
        """Execute one graph pass using the provided node order."""
        order = list(execution_order or self.nodes.keys())
        outputs: Dict[str, Tensor] = {}
        states: Dict[str, MemoryEngineState] = {}
        diagnostics: Dict[str, Dict[str, Any]] = {}
        summaries: Dict[str, Dict[str, Any]] = {}
        messages: Dict[str, Tensor] = {}

        for name in order:
            if name not in self.nodes:
                raise KeyError(f"Unknown node {name!r} in execution order.")

            external = node_inputs.get(name)
            inbound: List[Tensor] = []
            top_down: Optional[Tensor] = None

            for edge in self._edges:
                if edge["target"] != name or edge["source"] not in states:
                    continue
                source_state = states[edge["source"]]
                projected = self.edge_projections[edge["key"]](
                    self.nodes[edge["source"]].tape_features(source_state)
                )

                if external is not None:
                    seq_len = external.shape[1]
                elif inbound:
                    seq_len = inbound[0].shape[1]
                else:
                    seq_len = 1
                message = projected.unsqueeze(1).expand(
                    projected.shape[0],
                    seq_len,
                    projected.shape[-1],
                )
                messages[f"{edge['source']}->{name}"] = message
                if edge["kind"] == "top_down":
                    top_down = message if top_down is None else top_down + message
                else:
                    inbound.append(message)

            if external is None and not inbound and top_down is None:
                raise ValueError(
                    f"Node {name!r} has no external input and no available messages."
                )

            output, state, node_diag = self.nodes[name](
                hidden_states=external,
                incoming_signals=inbound or None,
                top_down_signal=top_down,
                return_state=True,
                return_diagnostics=True,
            )
            outputs[name] = output
            states[name] = state
            diagnostics[name] = node_diag
            summaries[name] = self.nodes[name].summarize_diagnostics(node_diag)

        result: Dict[str, Any] = {"outputs": outputs, "summary": summaries, "messages": messages}
        if return_states:
            result["states"] = states
        if return_diagnostics:
            result["diagnostics"] = diagnostics
        return result


__all__ = [
    "MemoryEngineGraph",
    "HierarchicalMemoryEngine",
    "MemoryEngineNode",
]
