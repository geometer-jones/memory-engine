"""
Example hybrid integration for inserting MemoryEngineLayer into Hugging Face LLMs.

This file shows two integration paths:

1. Llama / Mistral style decoder layers:
   memory is applied immediately after the attention residual, before the MLP.
2. Generic decoder blocks:
   fallback wrapper that augments the block output when internals are not exposed.

Typical production usage on an 8B model:

    model = AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.bfloat16)
    install_memory_engine(model, layer_indices=range(8, 13), memory_dim=384, max_aux_dims=64)
    reset_memory_engine(model)
    tokens = model.generate(...)

The memory layers keep their own persistent tape caches across decode steps. Call
`reset_memory_engine(model)` before each new prompt / conversation.
"""

from __future__ import annotations

from typing import Dict, Iterable

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from memory_engine_layer import MemoryEngineLayer, MemoryEngineState


def _resolve_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Find the mutable decoder layer list across common HF decoder architectures."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError("Unsupported architecture: could not locate decoder layers.")


def _infer_hidden_size(model: nn.Module) -> int:
    """Get the hidden size across GPT/Llama/Mistral style configs."""
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("Model has no config; cannot infer hidden size.")
    for attr in ("hidden_size", "n_embd", "d_model"):
        if hasattr(config, attr):
            return int(getattr(config, attr))
    raise ValueError("Could not infer hidden size from model config.")


class PostAttentionMemoryWrapper(nn.Module):
    """
    Wrapper for Llama / Mistral style decoder layers.

    It preserves the original attention + MLP computation, but inserts the
    memory engine exactly at the post-attention residual point:

        x -> self_attn -> residual add -> MemoryEngine -> layernorm+MLP
    """

    def __init__(self, block: nn.Module, memory_layer: MemoryEngineLayer) -> None:
        super().__init__()
        self.block = block
        self.memory_layer = memory_layer
        self.memory_state: MemoryEngineState | None = None
        self.last_memory_diagnostics: Dict | None = None

    def reset_memory(self) -> None:
        self.memory_state = None
        self.last_memory_diagnostics = None

    def _ensure_state(self, hidden_states: torch.Tensor) -> None:
        batch_size = hidden_states.shape[0]
        if self.memory_state is None or self.memory_state.tape.shape[0] != batch_size:
            self.memory_state = self.memory_layer.reset_state(batch_size, hidden_states.device, torch.float32)

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        residual = hidden_states
        hidden_states = self.block.input_layernorm(hidden_states)

        attn_outputs = self.block.self_attn(hidden_states=hidden_states, *args, **kwargs)
        attn_hidden = attn_outputs[0]
        hidden_states = residual + attn_hidden

        self._ensure_state(hidden_states)
        hidden_states, self.memory_state, self.last_memory_diagnostics = self.memory_layer(
            hidden_states,
            state=self.memory_state,
            return_diagnostics=True,
        )

        residual = hidden_states
        hidden_states = self.block.post_attention_layernorm(hidden_states)
        hidden_states = self.block.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if len(attn_outputs) > 1:
            outputs += attn_outputs[1:]
        return outputs


class PostBlockMemoryWrapper(nn.Module):
    """
    Fallback wrapper when the block internals are not architecture-stable.

    This is less faithful than the post-attention path, but still works as an
    additive residual module at the decoder block boundary.
    """

    def __init__(self, block: nn.Module, memory_layer: MemoryEngineLayer) -> None:
        super().__init__()
        self.block = block
        self.memory_layer = memory_layer
        self.memory_state: MemoryEngineState | None = None
        self.last_memory_diagnostics: Dict | None = None

    def reset_memory(self) -> None:
        self.memory_state = None
        self.last_memory_diagnostics = None

    def _ensure_state(self, hidden_states: torch.Tensor) -> None:
        batch_size = hidden_states.shape[0]
        if self.memory_state is None or self.memory_state.tape.shape[0] != batch_size:
            self.memory_state = self.memory_layer.reset_state(batch_size, hidden_states.device, torch.float32)

    def forward(self, *args, **kwargs):
        outputs = self.block(*args, **kwargs)
        hidden_states = outputs[0]
        self._ensure_state(hidden_states)
        hidden_states, self.memory_state, self.last_memory_diagnostics = self.memory_layer(
            hidden_states,
            state=self.memory_state,
            return_diagnostics=True,
        )
        return (hidden_states, *outputs[1:])


def _make_wrapper(block: nn.Module, hidden_size: int, **memory_kwargs) -> nn.Module:
    memory_layer = MemoryEngineLayer(hidden_dim=hidden_size, **memory_kwargs)
    llama_like = all(
        hasattr(block, attr)
        for attr in ("input_layernorm", "self_attn", "post_attention_layernorm", "mlp")
    )
    if llama_like:
        return PostAttentionMemoryWrapper(block, memory_layer)
    return PostBlockMemoryWrapper(block, memory_layer)


def install_memory_engine(
    model: nn.Module,
    layer_indices: Iterable[int],
    **memory_kwargs,
) -> nn.Module:
    """
    Replace the specified decoder layers with memory-augmented wrappers.

    Example:
        install_memory_engine(
            model,
            layer_indices=range(8, 13),
            memory_dim=384,
            max_aux_dims=64,
            max_transient_dims=32,
            consolidation_interval=16,
        )
    """
    layers = _resolve_decoder_layers(model)
    hidden_size = _infer_hidden_size(model)
    for idx in layer_indices:
        layers[idx] = _make_wrapper(layers[idx], hidden_size=hidden_size, **memory_kwargs)
    return model


def reset_memory_engine(model: nn.Module) -> None:
    """Clear all persistent tape caches before a new prompt or evaluation run."""
    for module in model.modules():
        if isinstance(module, (PostAttentionMemoryWrapper, PostBlockMemoryWrapper)):
            module.reset_memory()


def collect_memory_diagnostics(model: nn.Module) -> Dict[int, Dict]:
    """Collect the most recent diagnostics from every wrapped layer."""
    diagnostics: Dict[int, Dict] = {}
    for idx, module in enumerate(model.modules()):
        if isinstance(module, (PostAttentionMemoryWrapper, PostBlockMemoryWrapper)):
            diagnostics[idx] = module.last_memory_diagnostics or {}
    return diagnostics


def main() -> None:
    """
    End-to-end example on an 8B decoder model.

    `meta-llama/Meta-Llama-3-8B-Instruct` is shown because it matches the user
    request, but any accessible Llama/Mistral-style 7B/8B checkpoint works.
    """
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    install_memory_engine(
        model,
        layer_indices=range(8, 13),
        memory_dim=384,
        max_aux_dims=64,
        max_transient_dims=32,
        eta_init=0.08,
        alpha_init=0.25,
        top_k_binding=16,
        consolidation_interval=16,
        theta_merge=0.5,
        theta_seed=0.12,
    )

    prompt = "Explain why persistent state can help a transformer track long-context motifs."
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    reset_memory_engine(model)
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            use_cache=True,
        )

    print(tokenizer.decode(generated[0], skip_special_tokens=True))

    diagnostics = collect_memory_diagnostics(model)
    for layer_id, layer_diag in diagnostics.items():
        if not layer_diag:
            continue
        pr = layer_diag["pr"][0, -1].item()
        gini = layer_diag["gini"][0, -1].item()
        coupling = layer_diag["coupling_strength"][0, -1].item()
        print(f"layer_module={layer_id} pr={pr:.3f} gini={gini:.3f} coupling={coupling:.3f}")


if __name__ == "__main__":
    main()
