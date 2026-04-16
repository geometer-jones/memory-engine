"""
Standalone Memory Engine model — built from scratch using framework principles.

No attention. No feed-forward network. No pretrained weights.
The core computation IS the Memory Engine:
  1. Receive: Hadamard product of input with tape
  2. Regime-aware update: resonance scales, torque rotates, orth preserves
  3. Renormalize: project back to unit hypersphere
  4. Recurrence: tape carries across positions
  5. Readout: linear projection from tape to output

Architecture:
  Input (one-hot token) → embed → [ME layer × N] → readout → logits

The ME layers are stacked: each receives the output of the previous.
Within each layer, the tape updates causally across positions.

Tasks:
  - Copy: reproduce the input sequence
  - Associative recall: given key-value pairs and a query key, return the value
  - Sequence prediction: predict next token in a simple sequence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from engine import participation_ratio

np.set_printoptions(precision=4, suppress=True)


class StandaloneMELayer(nn.Module):
    """Single Memory Engine layer: tape accumulation across positions.

    The tape is a unit-norm vector that evolves via Hadamard reception
    with the input at each position. No attention, no FFN.
    """

    def __init__(self, dim: int, eta_init: float = 0.1):
        super().__init__()
        self.dim = dim
        self.tape_init = nn.Parameter(torch.randn(dim) / np.sqrt(dim))
        self.eta = nn.Parameter(torch.tensor(eta_init))
        # Per-dimension rotation for torque handling
        self.torque_bias = nn.Parameter(torch.randn(dim) * 0.01)

    def _renorm(self, s: torch.Tensor) -> torch.Tensor:
        return s / s.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, dim) — input sequence
        Returns: (batch, seq_len, dim) — tape-updated sequence
        """
        B, T, D = x.shape
        s = self._renorm(self.tape_init.unsqueeze(0).expand(B, -1)).clone()

        eta = self.eta.abs()
        outputs = []

        for t in range(T):
            h = x[:, t, :]
            # Hadamard reception
            c = h * s

            # Regime masks
            res = (c > 1e-6).float()
            tor = (c < -1e-6).float()
            orth = 1.0 - res - tor

            # Update: resonance scales, torque rotates with learned bias
            update = eta * (c * (1.0 - orth) + tor * self.torque_bias)
            s = self._renorm(s + update)

            # Output: the updated tape state (not the input)
            outputs.append(s.clone())

        return torch.stack(outputs, dim=1)


class MemoryEngineModel(nn.Module):
    """Standalone model built entirely from Memory Engine layers.

    Token → embedding → N ME layers → readout → logits

    Args:
        vocab_size: number of tokens
        dim: hidden dimensionality (tape dimensionality)
        n_layers: number of stacked ME layers
        max_seq_len: maximum sequence length
    """

    def __init__(self, vocab_size: int = 32, dim: int = 64,
                 n_layers: int = 4, max_seq_len: int = 128):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len

        # Token embedding
        self.embed = nn.Embedding(vocab_size, dim)
        # Position embedding (learned)
        self.pos_embed = nn.Embedding(max_seq_len, dim)
        # Stacked ME layers
        self.me_layers = nn.ModuleList([
            StandaloneMELayer(dim) for _ in range(n_layers)
        ])
        # Readout: tape state → logits
        self.readout = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None) -> dict:
        """
        input_ids: (batch, seq_len) — token indices
        Returns: dict with logits, loss, hidden_states
        """
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)

        x = self.embed(input_ids) + self.pos_embed(positions)

        all_hidden = [x.detach()]
        for layer in self.me_layers:
            x = layer(x)
            all_hidden.append(x.detach())

        logits = self.readout(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1)
            )

        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": all_hidden,
        }

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


# ── Task generators ──────────────────────────────────────────────────────

def generate_copy_task(vocab_size: int = 8, seq_len: int = 10,
                       n_samples: int = 1000):
    """Copy task: reproduce the input sequence (shifted by 1).

    Input: [a, b, c, d, ...]
    Target: [a, b, c, d, ...] (same tokens)
    """
    inputs = torch.randint(0, vocab_size, (n_samples, seq_len))
    return inputs, inputs.clone()


def generate_associative_recall(vocab_size: int = 16, n_pairs: int = 3,
                                 n_samples: int = 1000):
    """Associative recall: given key-value pairs and a query, return the value.

    Format: [k1, v1, k2, v2, k3, v3, ?, query_key, target_value, pad, ...]
    Model must learn to look up the value associated with the query key.

    Simplified: input is pairs followed by a query key, target is the
    corresponding value at the end position.
    """
    seq_len = n_pairs * 2 + 2  # pairs + separator + query + answer
    inputs = torch.zeros(n_samples, seq_len, dtype=torch.long)
    targets = torch.zeros(n_samples, seq_len, dtype=torch.long)

    for i in range(n_samples):
        # Generate random key-value pairs (keys and values from different ranges)
        for p in range(n_pairs):
            key = torch.randint(0, vocab_size // 2, (1,)).item()
            val = torch.randint(vocab_size // 2, vocab_size, (1,)).item()
            inputs[i, p * 2] = key
            inputs[i, p * 2 + 1] = val

        # Pick a random key from the pairs to query
        query_idx = torch.randint(0, n_pairs, (1,)).item()
        query_key = inputs[i, query_idx * 2].item()
        target_val = inputs[i, query_idx * 2 + 1].item()

        # Separator token
        inputs[i, n_pairs * 2] = vocab_size - 1
        # Query key
        inputs[i, n_pairs * 2 + 1] = query_key

        # Target: predict the value at the last position
        targets[i, :n_pairs * 2] = inputs[i, :n_pairs * 2]
        targets[i, n_pairs * 2] = inputs[i, n_pairs * 2]
        targets[i, n_pairs * 2 + 1] = target_val

    return inputs, targets


def generate_sequence_prediction(vocab_size: int = 4, seq_len: int = 20,
                                  n_samples: int = 1000):
    """Simple sequence prediction: learn a repeating pattern.

    Pattern: [0, 1, 2, 3, 0, 1, 2, 3, ...]
    Target: shifted by 1 (predict next token).
    """
    pattern = torch.arange(vocab_size)
    full = pattern.repeat(seq_len // vocab_size + 1)[:seq_len + 1]

    inputs = full[:seq_len].unsqueeze(0).repeat(n_samples, 1)
    targets = full[1:seq_len + 1].unsqueeze(0).repeat(n_samples, 1)

    return inputs, targets


# ── Training ─────────────────────────────────────────────────────────────

def train_model(model, inputs, targets, n_epochs=50, lr=0.005,
                batch_size=32, eval_every=10, task_name="task"):
    """Train the standalone ME model on a task."""
    n_samples = inputs.shape[0]
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    print(f"\nTraining on {task_name}")
    print(f"  Model params: {model.count_parameters()}")
    print(f"  Samples: {n_samples}, Batch: {batch_size}")
    print(f"{'Epoch':>5} {'Loss':>8} {'Accuracy':>10}")
    print("-" * 30)

    for epoch in range(n_epochs):
        model.train()
        indices = torch.randperm(n_samples)
        epoch_loss = 0
        n_correct = 0
        n_total = 0

        for start in range(0, n_samples, batch_size):
            batch_idx = indices[start:start + batch_size]
            x = inputs[batch_idx]
            y = targets[batch_idx]

            outputs = model(x, labels=y)
            loss = outputs["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item() * len(batch_idx)

            # Accuracy: fraction of correct token predictions
            preds = outputs["logits"].argmax(dim=-1)
            n_correct += (preds == y).sum().item()
            n_total += y.numel()

        scheduler.step()

        if (epoch + 1) % eval_every == 0 or epoch == 0:
            avg_loss = epoch_loss / n_samples
            accuracy = n_correct / n_total
            print(f"  {epoch+1:>3} {avg_loss:>8.3f} {accuracy:>9.3f}")

    # Final eval
    model.eval()
    with torch.no_grad():
        outputs = model(inputs[:5])
        preds = outputs["logits"].argmax(dim=-1)
        final_correct = (preds == targets[:5]).float().mean().item()

    print(f"  Final sample accuracy: {final_correct:.3f}")

    # Diagnostics: PR at each ME layer
    print(f"  PR across layers:")
    for i, hs in enumerate(outputs["hidden_states"]):
        mean_pr = np.mean([
            participation_ratio(hs[0, pos].numpy().astype(complex))
            for pos in range(hs.shape[1])
        ])
        print(f"    Layer {i}: PR={mean_pr:.1f}")

    return model


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("Standalone Memory Engine Model")
    print("=" * 60)

    # Task 1: Copy
    print("\n--- TASK 1: Copy ---")
    copy_inputs, copy_targets = generate_copy_task(
        vocab_size=8, seq_len=8, n_samples=2000
    )
    model1 = MemoryEngineModel(vocab_size=8, dim=32, n_layers=3, max_seq_len=8)
    model1 = train_model(model1, copy_inputs, copy_targets,
                          n_epochs=80, lr=0.01, task_name="Copy")

    # Task 2: Sequence prediction
    print("\n--- TASK 2: Sequence Prediction ---")
    seq_inputs, seq_targets = generate_sequence_prediction(
        vocab_size=4, seq_len=16, n_samples=2000
    )
    model2 = MemoryEngineModel(vocab_size=4, dim=32, n_layers=3, max_seq_len=16)
    model2 = train_model(model2, seq_inputs, seq_targets,
                          n_epochs=80, lr=0.01, task_name="Sequence Prediction")

    # Task 3: Associative recall
    print("\n--- TASK 3: Associative Recall ---")
    assoc_inputs, assoc_targets = generate_associative_recall(
        vocab_size=16, n_pairs=3, n_samples=2000
    )
    model3 = MemoryEngineModel(vocab_size=16, dim=64, n_layers=4, max_seq_len=8)
    model3 = train_model(model3, assoc_inputs, assoc_targets,
                          n_epochs=120, lr=0.01, task_name="Associative Recall")

    print("\n" + "=" * 60)
    print("Done.")
