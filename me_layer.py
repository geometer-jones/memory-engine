"""
Backward-compatible repository entrypoint for the production Memory Engine layer.

This module keeps the legacy public surface used elsewhere in the repo:

- `MemoryEngineLayer`
- `GPT2WithMemoryEngine`
- `create_model`

Internally it delegates to the fully coupled implementation in
`memory_engine_layer.py` and the hybrid decoder wrappers in
`example_hybrid_integration.py`, so existing scripts now use the new
coupled / binding / consolidation / anticipation stack by default.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from example_hybrid_integration import (
    PostAttentionMemoryWrapper,
    PostBlockMemoryWrapper,
    collect_memory_diagnostics,
    install_memory_engine,
    reset_memory_engine,
)
from memory_engine_layer import MemoryEngineLayer as CoupledMemoryEngineLayer
from memory_engine_layer import MemoryEngineState


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
    Backward-compatible adapter around the production MemoryEngineLayer.

    Legacy callers expect `forward(hidden_states)` to return only the transformed
    hidden states. The production layer returns `(output, state, diagnostics)`.
    This adapter preserves the simple default while still exposing explicit state
    when requested.
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
        self.engine = CoupledMemoryEngineLayer(
            hidden_dim=hidden_dim,
            eta_init=eta_init,
            alpha_init=alpha_init,
            **memory_kwargs,
        )
        self.last_state: Optional[MemoryEngineState] = None
        self.last_diagnostics: Optional[Dict] = None

    @property
    def eta(self):
        return self.engine.eta

    @property
    def alpha(self):
        return self.engine.alpha

    def initialize_state(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> MemoryEngineState:
        return self.engine.initialize_state(batch_size, device, dtype)

    def reset_state(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> MemoryEngineState:
        return self.engine.reset_state(batch_size, device, dtype)

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
    Compatibility wrapper that installs the production memory layer into a
    decoder-only Hugging Face causal LM.

    The historical class name is preserved because existing training scripts use
    `create_model()` and this type name directly.
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
    """
    Create a memory-augmented causal LM plus tokenizer.

    This is the stable entrypoint used by the rest of the repo.
    """
    model = GPT2WithMemoryEngine(model_name=model_name, insert_after=insert_after, **memory_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer
