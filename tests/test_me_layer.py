"""Tests for the pure trainable MemoryEngineLayer wrapper."""

from __future__ import annotations

import torch

from me_layer import MemoryEngineLayer


def test_shape_preservation_and_diagnostics():
    layer = MemoryEngineLayer(
        hidden_dim=16,
        memory_dim=16,
        output_dim=16,
        consolidation_interval=2,
    )
    inputs = torch.randn(2, 5, 16)

    outputs, state, diagnostics = layer(
        inputs,
        return_state=True,
        return_diagnostics=True,
    )

    assert outputs.shape == inputs.shape
    assert state.tape.dtype.is_complex
    assert diagnostics["tape_features"].shape == (2, 5, 32)
    assert diagnostics["pr"].shape == (2, 5)
    assert diagnostics["resonance_fraction"].shape == (2, 5)
    assert diagnostics["torque_fraction"].shape == (2, 5)


def test_recurrence_changes_persistent_tape_state():
    layer = MemoryEngineLayer(hidden_dim=12, memory_dim=12, output_dim=12)
    inputs = torch.randn(1, 6, 12)

    _, state, _ = layer(inputs, return_state=True, return_diagnostics=True)
    first_tape = layer.initialize_state(batch_size=1, device=inputs.device).tape

    assert torch.linalg.norm(state.tape - first_tape).item() > 1e-5


def test_framework_parameters_receive_gradients():
    layer = MemoryEngineLayer(
        hidden_dim=10,
        memory_dim=10,
        output_dim=10,
        consolidation_interval=1,
    )
    inputs = torch.randn(2, 4, 10)

    outputs, _, diagnostics = layer(inputs, return_state=True, return_diagnostics=True)
    loss = outputs.pow(2).mean()
    loss = loss + diagnostics["prediction_error"].mean()
    loss.backward()

    assert layer.tape_init.grad is not None
    assert layer.alpha.grad is not None
    assert layer.torque_rotation.grad is not None
    assert layer.epsilon.grad is not None
    assert layer.W_pred.grad is not None
    assert layer.w_r.grad is not None
    assert layer.breadth_gate.grad is not None
