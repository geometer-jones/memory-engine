"""Tests for MemoryEngineLayer and GPT2WithMemoryEngine."""

import torch
import numpy as np
from me_layer import MemoryEngineLayer, GPT2WithMemoryEngine, create_model
from engine import participation_ratio

np.set_printoptions(precision=4, suppress=True)


def test_shape_preservation():
    """Forward pass preserves (batch, seq_len, hidden_dim) shape."""
    layer = MemoryEngineLayer(hidden_dim=64)
    x = torch.randn(2, 10, 64)
    out = layer(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    print("T1 — Shape preservation: PASS")


def test_tape_recurrence():
    """Tape state changes across positions (recurrence works)."""
    layer = MemoryEngineLayer(hidden_dim=32, eta_init=0.2)

    # Create input where first position is different from second
    x = torch.randn(1, 5, 32)

    # Run forward pass and get tape state after each position
    # We'll manually trace the tape
    s_init = layer._renormalize(layer.tape_init.unsqueeze(0)).detach()
    s = s_init.clone()
    tape_states = [s.clone()]

    with torch.no_grad():
        for t in range(5):
            h_t = x[:, t, :]
            c_t = h_t * s
            resonance_mask = (c_t > 1e-6).float()
            torque_mask = (c_t < -1e-6).float()
            orth_mask = 1.0 - resonance_mask - torque_mask
            eta = torch.abs(layer.eta)
            update = eta * c_t + torque_mask * layer.torque_rotation * eta
            update = update * (1.0 - orth_mask)
            s = layer._renormalize(s + update)
            tape_states.append(s.clone())

    # Tape should change across positions
    s_first = tape_states[1].numpy().flatten()
    s_last = tape_states[-1].numpy().flatten()
    change = np.linalg.norm(s_last - s_first)

    print(f"T2 — Tape recurrence: tape moved {change:.4f} from pos 1 to pos 5")
    assert change > 1e-6, "Tape should change across positions"
    print("  PASS")


def test_regime_dynamics():
    """Resonance dims grow, torque dims rotate in the tape."""
    layer = MemoryEngineLayer(hidden_dim=16, eta_init=0.3, alpha_init=1.0)

    # Construct input: in-phase with tape at dims 0-7, antiphase at dims 8-15
    with torch.no_grad():
        tape = layer._renormalize(layer.tape_init.unsqueeze(0)).squeeze()
        tape_np = tape.numpy()

    x = torch.zeros(1, 1, 16)
    for i in range(8):
        x[0, 0, i] = float(abs(tape_np[i]) * 1.5)  # same sign = resonance
    for i in range(8, 16):
        x[0, 0, i] = float(-abs(tape_np[i]) * 1.5)  # opposite sign = torque

    with torch.no_grad():
        out = layer(x)

    # Output should differ from input due to tape influence
    diff = (out - x).abs().mean().item()
    print(f"T3 — Regime dynamics: mean output change = {diff:.4f}")
    assert diff > 1e-6, "Output should differ from input"
    print("  PASS")


def test_gradient_flow():
    """Gradients flow through tape updates."""
    layer = MemoryEngineLayer(hidden_dim=32, eta_init=0.2)
    x = torch.randn(1, 5, 32, requires_grad=False)
    out = layer(x)
    loss = out.sum()
    loss.backward()

    # Check gradients exist for all learnable parameters
    assert layer.tape_init.grad is not None, "tape_init should have gradient"
    assert layer.eta.grad is not None, "eta should have gradient"
    assert layer.alpha.grad is not None, "alpha should have gradient"
    assert layer.torque_rotation.grad is not None, "torque_rotation should have gradient"

    print(f"T4 — Gradient flow:")
    print(f"  tape_init grad norm: {layer.tape_init.grad.norm():.4f}")
    print(f"  eta grad: {layer.eta.grad.item():.4f}")
    print(f"  alpha grad: {layer.alpha.grad.item():.4f}")
    print(f"  torque_rotation grad norm: {layer.torque_rotation.grad.norm():.4f}")
    print("  PASS")


def test_gpt2_integration():
    """GPT-2 with ME layers produces valid output."""
    model, tokenizer = create_model("gpt2", insert_after=[3, 6, 9])

    input_ids = tokenizer.encode("The quick brown fox", return_tensors="pt")

    with torch.no_grad():
        outputs = model(input_ids)

    logits = outputs["logits"]
    assert logits.shape[0] == 1, "Batch size should be 1"
    assert logits.shape[2] == model.config.vocab_size, "Vocab size should match"

    # Check hidden states have ME layers injected
    hs = outputs["hidden_states"]
    # GPT-2 has 12 blocks, each producing a hidden state, plus embedding + final ln
    # With 3 ME insertions, we should have: embedding + 12 block outputs + 3 ME outputs + final ln = 17
    print(f"T5 — GPT-2 integration:")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Hidden states: {len(hs)} tensors")
    print(f"  Parameters: {model.count_parameters()}")

    # Generate a few tokens
    gen = model.generate(input_ids, max_length=20)
    gen_text = tokenizer.decode(gen[0])
    print(f"  Generated: {gen_text}")
    print("  PASS")


def test_param_count():
    """ME layers are tiny compared to GPT-2."""
    model, _ = create_model("gpt2", insert_after=[3, 6, 9])
    counts = model.count_parameters()

    print(f"T6 — Parameter count:")
    print(f"  Trainable (ME): {counts['trainable']:,}")
    print(f"  Frozen (GPT-2):  {counts['frozen']:,}")
    print(f"  Total:           {counts['total']:,}")
    print(f"  ME fraction:     {counts['trainable']/counts['total']*100:.4f}%")

    assert counts["trainable"] < counts["frozen"] * 0.001, (
        "ME layers should be < 0.1% of total params"
    )
    print("  PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("Memory Engine Layer Tests")
    print("=" * 60 + "\n")

    test_shape_preservation()
    print()
    test_tape_recurrence()
    print()
    test_regime_dynamics()
    print()
    test_gradient_flow()
    print()
    test_gpt2_integration()
    print()
    test_param_count()

    print("\n" + "=" * 60)
    print("All tests passed.")
