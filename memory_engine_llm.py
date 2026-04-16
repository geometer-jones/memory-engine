"""Hybrid and pure Memory Engine language-model wrappers."""

from __future__ import annotations

import inspect
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from example_hybrid_integration import (
    PostAttentionMemoryWrapper,
    PostBlockMemoryWrapper,
    install_memory_engine,
    reset_memory_engine,
)
from me_layer import (
    MemoryEngineLayer as PureMemoryEngineLayer,
    MemoryEngineState as PureMemoryEngineState,
    _normalize_insert_after,
    _resolve_decoder_layers,
)
from memory_engine_layer import MemoryEngineLayer as RuntimeMemoryEngineLayer
from memory_engine_layer import MemoryEngineState


def _infer_hidden_size(model: nn.Module) -> int:
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("Model has no config; cannot infer hidden size.")
    for attr in ("hidden_size", "n_embd", "d_model"):
        if hasattr(config, attr):
            return int(getattr(config, attr))
    raise ValueError("Could not infer hidden size from model config.")


def _default_runtime_memory_kwargs(hidden_dim: int) -> Dict[str, int | float]:
    memory_dim = min(256, hidden_dim)
    max_aux_dims = 16 if memory_dim >= 128 else 8
    max_transient_dims = max(4, max_aux_dims // 2)
    return {
        "memory_dim": memory_dim,
        "max_aux_dims": max_aux_dims,
        "max_transient_dims": max_transient_dims,
        "top_k_binding": min(16, memory_dim),
        "coupling_rank": 10,
        "prediction_rank": 10,
        "consolidation_interval": 8,
    }


def _sanitize_memory_kwargs(memory_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    allowed = set(inspect.signature(RuntimeMemoryEngineLayer.__init__).parameters)
    allowed.discard("self")
    return {key: value for key, value in memory_kwargs.items() if key in allowed}


class MemoryEngineCausalLM(nn.Module):
    """Frozen-base causal LM with trainable Memory Engine insertions."""

    def __init__(
        self,
        model_name: str = "gpt2",
        base_model: Optional[nn.Module] = None,
        insert_after: Optional[Iterable[int]] = None,
        freeze_base_model: bool = True,
        **memory_kwargs: Any,
    ) -> None:
        super().__init__()
        self.base_model = base_model or AutoModelForCausalLM.from_pretrained(model_name)
        self.config = self.base_model.config
        self.hidden_size = _infer_hidden_size(self.base_model)
        self.insert_after = _normalize_insert_after(self.base_model, insert_after)

        if freeze_base_model:
            for param in self.base_model.parameters():
                param.requires_grad = False

        default_kwargs = _default_runtime_memory_kwargs(self.hidden_size)
        default_kwargs.update(memory_kwargs)
        install_memory_engine(
            self.base_model,
            self.insert_after,
            **_sanitize_memory_kwargs(default_kwargs),
        )
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

    @staticmethod
    def decode_memory_state(state: MemoryEngineState) -> torch.Tensor:
        """Decode the current tape back into host hidden space."""
        return torch.einsum("bn,bdn->bd", state.tape.real, state.basis.detach())

    @staticmethod
    def flatten_tape(state: MemoryEngineState) -> torch.Tensor:
        """Expose a real-valued tape representation for logging or auxiliary loss."""
        return torch.cat([state.tape.real, state.tape.imag], dim=-1)

    def collect_memory_layers(self) -> List[Dict[str, Any]]:
        layers: List[Dict[str, Any]] = []
        for layer_index, wrapper in zip(self.insert_after, self._memory_wrappers()):
            state = wrapper.memory_state
            diagnostics = wrapper.last_memory_diagnostics or {}
            layer_record: Dict[str, Any] = {
                "layer_index": layer_index,
                "diagnostics": diagnostics,
                "state": state,
            }
            if state is not None:
                layer_record["tape"] = state.tape
                layer_record["tape_features"] = self.flatten_tape(state)
                layer_record["decoded_hidden"] = self.decode_memory_state(state)
            layers.append(layer_record)
        return layers

    def get_me_parameters(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        for wrapper in self._memory_wrappers():
            params.extend(list(wrapper.memory_layer.parameters()))
        return params

    def count_parameters(self) -> Dict[str, int]:
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in self.parameters())
        frozen = total - trainable
        return {"trainable": trainable, "frozen": frozen, "total": total}

    def me_state_dict(self) -> Dict[str, torch.Tensor]:
        """Save only the trainable Memory Engine parameters."""
        trainable_names = {param_name for param_name, param in self.named_parameters() if param.requires_grad}
        return {
            name: tensor.detach().clone()
            for name, tensor in self.state_dict().items()
            if name in trainable_names
        }

    def load_me_state_dict(self, me_state_dict: Dict[str, torch.Tensor], strict: bool = True) -> None:
        model_state = self.state_dict()
        missing = []
        for name, value in me_state_dict.items():
            if name not in model_state:
                if strict:
                    missing.append(name)
                continue
            model_state[name].copy_(value)
        if strict and missing:
            raise KeyError(f"Unknown Memory Engine parameters in checkpoint: {missing}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        reset_memory: bool = True,
        output_hidden_states: bool = False,
        return_memory_features: bool = False,
        use_cache: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if reset_memory:
            self.reset_memory()

        need_hidden_states = output_hidden_states or return_memory_features
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=need_hidden_states,
            use_cache=use_cache,
            **kwargs,
        )

        result: Dict[str, Any] = {
            "loss": outputs.loss,
            "logits": outputs.logits,
        }
        if need_hidden_states:
            result["hidden_states"] = outputs.hidden_states

        memory_layers = self.collect_memory_layers()
        result["me_diagnostics"] = [
            {
                "layer_index": layer["layer_index"],
                "diagnostics": layer["diagnostics"],
            }
            for layer in memory_layers
        ]

        if return_memory_features:
            result["memory_layers"] = memory_layers
            if memory_layers and memory_layers[-1].get("state") is not None:
                result["final_memory_tape"] = memory_layers[-1]["tape"]
                result["final_memory_features"] = memory_layers[-1]["tape_features"]
                result["final_memory_hidden"] = memory_layers[-1]["decoded_hidden"]
                result["final_memory_active_slots"] = memory_layers[-1]["state"].active_mask.sum(dim=-1)

        return result


class MemoryEngineLLM(nn.Module):
    """Pure hierarchical Memory Engine language model."""

    def __init__(
        self,
        vocab_size: int,
        dim: int = 512,
        n_layers: int = 8,
        max_seq_len: int = 256,
        dropout: float = 0.0,
        **layer_kwargs: Any,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(max_seq_len, dim)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                PureMemoryEngineLayer(
                    hidden_dim=dim,
                    memory_dim=dim,
                    output_dim=dim,
                    **layer_kwargs,
                )
                for _ in range(n_layers)
            ]
        )
        self.lm_head = nn.Linear(2 * dim, vocab_size, bias=False)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def initialize_states(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> List[PureMemoryEngineState]:
        return [
            layer.initialize_state(batch_size=batch_size, device=device, dtype=dtype)
            for layer in self.layers
        ]

    def count_parameters(self) -> Dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return {"total": total, "trainable": trainable}

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        states: Optional[Sequence[PureMemoryEngineState]] = None,
        return_diagnostics: bool = True,
    ) -> Dict[str, Any]:
        if input_ids.ndim != 2:
            raise ValueError(f"Expected input_ids with shape (batch, seq_len), got {tuple(input_ids.shape)}.")

        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}. "
                "Increase max_seq_len or shorten the training block size."
            )

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        hidden = self.embed(input_ids) + self.pos_embed(positions)
        hidden = self.dropout(hidden)

        next_states: List[PureMemoryEngineState] = []
        layer_diagnostics: List[Dict[str, torch.Tensor]] = []
        final_tape_features: Optional[torch.Tensor] = None

        for layer_index, layer in enumerate(self.layers):
            layer_state = states[layer_index] if states is not None else None
            hidden, next_state, diagnostics = layer(
                hidden,
                state=layer_state,
                return_state=True,
                return_diagnostics=True,
            )
            next_states.append(next_state)
            layer_diagnostics.append(diagnostics)
            final_tape_features = diagnostics["tape_features"]

        if final_tape_features is None:
            raise RuntimeError("At least one MemoryEngineLayer is required.")

        logits = self.lm_head(final_tape_features)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), labels.reshape(-1))

        outputs: Dict[str, Any] = {
            "loss": loss,
            "logits": logits,
            "states": next_states,
        }

        if return_diagnostics:
            outputs["diagnostics"] = {
                "layer_diagnostics": layer_diagnostics,
                "final_tape_features": final_tape_features,
                "final_layer_pr": layer_diagnostics[-1]["pr"],
                "final_layer_resonance": layer_diagnostics[-1]["resonance_fraction"],
                "final_layer_torque": layer_diagnostics[-1]["torque_fraction"],
            }

        return outputs
