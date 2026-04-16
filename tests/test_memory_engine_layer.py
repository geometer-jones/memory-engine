"""Unit tests for the coupled production MemoryEngineLayer."""

from __future__ import annotations

import torch

from memory_engine_layer import (
    MERGED_SLOT,
    SEEDED_SLOT,
    TRANSIENT_SLOT,
    MemoryEngineLayer,
)


def _make_layer(**overrides) -> MemoryEngineLayer:
    defaults = dict(
        hidden_dim=6,
        memory_dim=6,
        max_aux_dims=4,
        max_transient_dims=2,
        eta_init=0.15,
        alpha_init=0.5,
        top_k_binding=4,
        bind_fraction=1.0,
        consolidation_interval=1,
        theta_merge=0.15,
        theta_seed=0.05,
        theta_prune=0.05,
        prune_patience=1,
        residual_bank_size=4,
    )
    defaults.update(overrides)
    torch.manual_seed(0)
    return MemoryEngineLayer(**defaults)


def _clone_state(state):
    return state.__class__(
        tape=state.tape.detach().clone(),
        prev_tape=state.prev_tape.detach().clone(),
        basis=state.basis.detach().clone(),
        gram=state.gram.detach().clone(),
        coupling=state.coupling.detach().clone(),
        active_mask=state.active_mask.clone(),
        slot_kind=state.slot_kind.clone(),
        lifetime=state.lifetime.clone(),
        prune_count=state.prune_count.clone(),
        binding_sources=state.binding_sources.clone(),
        corr=state.corr.detach().clone(),
        residual_bank=state.residual_bank.detach().clone(),
        residual_count=state.residual_count.clone(),
        residual_ptr=state.residual_ptr.clone(),
        step=state.step.clone(),
        basis_dirty=state.basis_dirty.clone(),
    )


def test_coupling_matrix_responds_to_basis_overlap():
    """Gram overlap should induce a non-trivial inverse coupling matrix."""
    layer = _make_layer(max_aux_dims=0, max_transient_dims=0)
    state = layer.initialize_state(batch_size=1, device=torch.device("cpu"))

    state.basis[0, :, 1] = state.basis[0, :, 0]
    state.basis_dirty[0] = True
    layer._refresh_coupling(state)

    gram = state.gram[0, :6, :6]
    coupling = state.coupling[0, :6, :6]

    assert gram[0, 1] > 0.9
    assert coupling[0, 1].abs() > 0.1

    hidden = torch.zeros(1, 6)
    hidden[0, 0] = 1.0
    _, coupled, _ = layer._project_input(hidden, state)
    assert coupled[0, 1].abs() > 0.1


def test_fast_binding_creates_transient_slot():
    """Strong co-resonance should allocate a transient conjunctive dimension."""
    layer = _make_layer()
    state = layer.initialize_state(batch_size=1, device=torch.device("cpu"))

    state.tape[0, :6] = torch.complex(
        torch.tensor([0.8, 0.8, 0.2, 0.2, 0.1, 0.1]),
        torch.zeros(6),
    )
    state.tape = layer._renormalize(state.tape, state.active_mask)
    state.prev_tape = state.tape.clone()

    hidden = torch.zeros(1, 1, 6)
    hidden[0, 0, 0] = 2.0
    hidden[0, 0, 1] = 2.0
    output, state, diagnostics = layer(hidden, state=state)

    assert output.shape == hidden.shape
    assert (state.slot_kind[0] == TRANSIENT_SLOT).any()
    assert diagnostics["binding_events"][0]["new"] >= 1
    assert diagnostics["active_slots"][0].item() > layer.memory_dim


def test_consolidation_merge_allocates_merged_slot():
    """High cross-correlation should trigger the merge branch of consolidation."""
    layer = _make_layer()
    state = layer.initialize_state(batch_size=1, device=torch.device("cpu"))

    state.corr[0, 0, 1] = 1.0 + 0.0j
    state.corr[0, 1, 0] = 1.0 + 0.0j
    state.corr[0, 0, 0] = 1.0 + 0.0j
    state.corr[0, 1, 1] = 1.0 + 0.0j

    events = layer._run_consolidation(state)

    assert events[0]["merged"] == 1
    assert (state.slot_kind[0] == MERGED_SLOT).any()


def test_consolidation_seed_allocates_seeded_slot_from_novelty_bank():
    """Novelty residual principal components should trigger seeding."""
    layer = _make_layer()
    state = layer.initialize_state(batch_size=1, device=torch.device("cpu"))
    state.corr.zero_()

    residual = torch.tensor(
        [
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.8, 0.2, 0.0, 0.0],
            [0.0, 0.0, 1.1, -0.1, 0.0, 0.0],
            [0.0, 0.0, 0.9, 0.1, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    state.residual_bank[0] = residual
    state.residual_count[0] = residual.shape[0]

    events = layer._run_consolidation(state)

    assert events[0]["seeded"] == 1
    assert (state.slot_kind[0] == SEEDED_SLOT).any()


def test_prediction_error_torque_rises_on_perturbation():
    """The PHASE5 first-difference predictor should spike under perturbation."""
    layer = _make_layer(max_aux_dims=0, max_transient_dims=0, consolidation_interval=0)
    state = layer.initialize_state(batch_size=1, device=torch.device("cpu"))

    stable = torch.zeros(1, 4, 6)
    stable[:, :, 0] = 1.5
    stable[:, :, 1] = 1.5

    _, state, _ = layer(stable, state=state)
    baseline_state = _clone_state(state)

    next_stable = torch.zeros(1, 1, 6)
    next_stable[:, :, 0] = 1.5
    next_stable[:, :, 1] = 1.5
    _, _, stable_diag = layer(next_stable, state=baseline_state)
    baseline_error = stable_diag["prediction_error"][0, 0].item()

    perturbed = next_stable.clone()
    perturbed[:, :, 0] = -2.5
    perturbed[:, :, 1] = -2.5

    _, _, perturbed_diag = layer(perturbed, state=state)
    surprise_error = perturbed_diag["prediction_error"][0, 0].item()
    surprise_torque = perturbed_diag["self_torque"][0, 0].item()

    assert surprise_error > baseline_error
    assert surprise_torque > 0.0
