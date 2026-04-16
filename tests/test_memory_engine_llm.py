from __future__ import annotations

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from memory_engine_llm import MemoryEngineCausalLM


def _make_base_model() -> GPT2LMHeadModel:
    config = GPT2Config(
        vocab_size=64,
        n_positions=32,
        n_ctx=32,
        n_embd=16,
        n_layer=2,
        n_head=2,
        bos_token_id=1,
        eos_token_id=2,
    )
    torch.manual_seed(0)
    return GPT2LMHeadModel(config)


def test_memory_engine_causal_lm_exposes_memory_features():
    model = MemoryEngineCausalLM(
        base_model=_make_base_model(),
        insert_after=[0, 1],
        memory_dim=16,
        max_aux_dims=2,
        max_transient_dims=1,
    )

    input_ids = torch.randint(0, 64, (2, 8))
    labels = input_ids.clone()
    outputs = model(
        input_ids=input_ids,
        labels=labels,
        reset_memory=True,
        output_hidden_states=True,
        return_memory_features=True,
        use_cache=False,
    )

    assert outputs["logits"].shape == (2, 8, 64)
    assert outputs["hidden_states"][-1].shape == (2, 8, 16)
    assert outputs["final_memory_hidden"].shape == (2, 16)
    assert outputs["final_memory_tape"].shape[0] == 2
    assert len(outputs["memory_layers"]) == 2


def test_only_memory_parameters_are_trainable_when_base_is_frozen():
    model = MemoryEngineCausalLM(
        base_model=_make_base_model(),
        insert_after=[0],
        freeze_base_model=True,
    )

    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    assert trainable
    assert all("memory_layer" in name for name in trainable)


def test_reset_memory_clears_wrapper_state():
    model = MemoryEngineCausalLM(
        base_model=_make_base_model(),
        insert_after=[0],
    )
    input_ids = torch.randint(0, 64, (1, 6))
    _ = model(input_ids=input_ids, reset_memory=True, return_memory_features=True, use_cache=False)

    wrappers = model._memory_wrappers()
    assert wrappers[0].memory_state is not None

    model.reset_memory()
    assert wrappers[0].memory_state is None
