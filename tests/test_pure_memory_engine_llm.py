"""Smoke tests for the pure hierarchical MemoryEngineLLM."""

from __future__ import annotations

import torch

from memory_engine_llm import MemoryEngineLLM


def test_pure_memory_engine_llm_shapes_and_loss():
    model = MemoryEngineLLM(
        vocab_size=64,
        dim=24,
        n_layers=3,
        max_seq_len=32,
        consolidation_interval=2,
    )

    input_ids = torch.randint(0, 64, (2, 12))
    labels = torch.randint(0, 64, (2, 12))

    outputs = model(input_ids, labels=labels, return_diagnostics=True)

    assert outputs["logits"].shape == (2, 12, 64)
    assert outputs["loss"].ndim == 0
    assert len(outputs["states"]) == 3

    diagnostics = outputs["diagnostics"]
    assert len(diagnostics["layer_diagnostics"]) == 3
    assert diagnostics["final_tape_features"].shape == (2, 12, 48)
    assert diagnostics["final_layer_pr"].shape == (2, 12)


def test_pure_memory_engine_llm_backward_pass():
    model = MemoryEngineLLM(vocab_size=32, dim=16, n_layers=2, max_seq_len=16)
    input_ids = torch.randint(0, 32, (2, 8))
    labels = torch.randint(0, 32, (2, 8))

    outputs = model(input_ids, labels=labels, return_diagnostics=False)
    outputs["loss"].backward()

    assert model.embed.weight.grad.norm().item() > 0.0
