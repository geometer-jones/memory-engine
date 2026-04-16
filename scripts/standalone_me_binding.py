"""
Standalone Memory Engine model with Fast Binding.

Implements the full framework pipeline with complex-valued tape state:
  1. Receive: Hadamard product of real input with complex tape -> complex reception
  2. Fast binding: co-resonance scoring using complex phase structure
  3. Regime-aware update: resonance scales, torque rotates (complex)
  4. Renormalize: project back to unit hypersphere in C^n
  5. Recurrence: tape carries across positions
  6. Readout: magnitude |s_i| per dimension -> linear projection to output

Fast binding (Section 2.3.1 of the essay):
  - Co-resonance score B_ij = |c_i| * |c_j| * cos(phi_i - phi_j)
  - Adaptive threshold: bind top fraction of co-resonant pairs per step
  - Transients decay (gamma per step) and have limited lifetime (L steps)
  - Re-triggering refreshes the transient
  - Only top-k dimensions by |c_i| enter pairing (sparse check)

Tasks:
  - Copy: reproduce the input sequence
  - Associative recall: key-value lookup
  - Sequence prediction: next-token prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from engine import participation_ratio

np.set_printoptions(precision=4, suppress=True)


class StandaloneMELayerWithBinding(nn.Module):
    """Memory Engine layer with fast binding via transient conjunctive dimensions.

    Uses genuinely complex-valued tape state, as the framework requires.
    Phase structure in the complex tape gives meaningful co-resonance scores.

    At each position:
    1. Compute Hadamard reception c = input * tape (complex)
    2. Find top-k dimensions by |c_i|
    3. Compute co-resonance scores B_ij for pairs among top-k
    4. Bind top fraction of co-resonant pairs (adaptive threshold)
    5. Apply decay to all transients
    6. Augment state with transients, compute full reception
    7. Update + renormalize
    """

    def __init__(self, dim: int, eta_init: float = 0.1,
                 bind_fraction: float = 0.15, beta: float = 0.05,
                 gamma: float = 0.9, lifetime: int = 5,
                 top_k: int = 8, max_transients: int = 16):
        super().__init__()
        self.dim = dim
        # Complex tape: real + imaginary parts on the unit hypersphere in C^n
        self.tape_re = nn.Parameter(torch.randn(dim) / np.sqrt(dim))
        self.tape_im = nn.Parameter(torch.randn(dim) / np.sqrt(dim))
        self.eta = nn.Parameter(torch.tensor(eta_init))
        # Complex torque bias (learned rotation per dimension)
        self.torque_bias_re = nn.Parameter(torch.randn(dim) * 0.01)
        self.torque_bias_im = nn.Parameter(torch.randn(dim) * 0.01)

        # Fast binding parameters
        self.bind_fraction = bind_fraction  # fraction of top pairs to bind
        self.beta = beta
        self.gamma = gamma
        self.lifetime = lifetime
        self.top_k = min(top_k, dim)
        self.max_transients = max_transients

    def _renorm(self, s: torch.Tensor) -> torch.Tensor:
        """Renormalize complex vector to unit hypersphere."""
        return s / s.abs().pow(2).sum(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        x: (batch, seq_len, dim) -- real-valued input
        Returns: (output, diagnostics) where output is real (|s| per dimension)
        """
        B, T, D = x.shape
        device = x.device

        # Initialize complex tape
        s = torch.complex(self.tape_re, self.tape_im)  # (D,) complex
        s = self._renorm(s.unsqueeze(0).expand(B, -1)).clone()  # (B, D) complex

        batch_transients = [[] for _ in range(B)]
        eta = self.eta.abs()
        outputs = []
        binding_events = []

        torque_bias = torch.complex(self.torque_bias_re, self.torque_bias_im)  # (D,)

        for t in range(T):
            h = x[:, t, :]  # (B, D) real

            # 1. Complex Hadamard reception: real input * complex tape = complex c
            c = h * s  # (B, D) complex

            # 2. Fast binding check (per batch element)
            new_binding_count = 0
            refreshed_count = 0

            for b in range(B):
                c_b = c[b]  # (D,) complex
                mag_c = c_b.abs()

                # Find top-k dimensions by |c_i|
                if self.top_k < D:
                    _, top_idx = torch.topk(mag_c, self.top_k)
                else:
                    top_idx = torch.arange(D, device=device)

                # Compute co-resonance for all pairs among top-k
                top_c = c_b[top_idx]  # (top_k,) complex
                top_mag = top_c.abs()
                top_phase = torch.angle(top_c)

                n_top = len(top_idx)
                pair_scores = []
                pair_indices = []

                for ii in range(n_top):
                    for jj in range(ii + 1, n_top):
                        delta_phi = top_phase[ii] - top_phase[jj]
                        B_ij = (top_mag[ii] * top_mag[jj] * torch.cos(delta_phi)).item()
                        pair_scores.append(B_ij)
                        pair_indices.append((ii, jj))

                if not pair_scores:
                    continue

                # Adaptive threshold: bind top bind_fraction of positive-scoring pairs
                positive_scores = [sc for sc in pair_scores if sc > 0]
                if not positive_scores:
                    continue

                sorted_scores = sorted(positive_scores, reverse=True)
                n_to_bind = max(1, int(len(sorted_scores) * self.bind_fraction))
                theta = sorted_scores[min(n_to_bind - 1, len(sorted_scores) - 1)]

                for pair_idx, (ii, jj) in enumerate(pair_indices):
                    if pair_scores[pair_idx] < theta:
                        continue

                    ci = top_idx[ii].item()
                    cj = top_idx[jj].item()

                    # Check if transient already exists
                    existing = None
                    for idx, (ti, tj, tm, tc) in enumerate(batch_transients[b]):
                        if (ti == ci and tj == cj) or (ti == cj and tj == ci):
                            existing = idx
                            break

                    if existing is not None:
                        ti, tj, tm, _ = batch_transients[b][existing]
                        batch_transients[b][existing] = (ti, tj, tm, self.lifetime)
                        refreshed_count += 1
                    elif len(batch_transients[b]) < self.max_transients:
                        # Seed transient: phase from co-resonant pair
                        s_i = s[b, ci]
                        s_j = s[b, cj]
                        product = s_i * s_j
                        product_mag = product.abs().clamp(min=1e-8)
                        s_temp = self.beta * (product / product_mag)
                        batch_transients[b].append(
                            (ci, cj, complex(s_temp.item()), self.lifetime)
                        )
                        new_binding_count += 1

            binding_events.append({
                "new": new_binding_count,
                "refreshed": refreshed_count,
                "active": sum(len(bt) for bt in batch_transients),
            })

            # 3. Decay + prune transients
            for b in range(B):
                surviving = []
                for ci, cj, mag, counter in batch_transients[b]:
                    new_mag = mag * self.gamma
                    new_counter = counter - 1
                    if new_counter > 0 and abs(new_mag) > 1e-6:
                        surviving.append((ci, cj, new_mag, new_counter))
                batch_transients[b] = surviving

            # 4. Augment tape with transient contributions
            s_aug_real = s.real.clone()
            s_aug_imag = s.imag.clone()
            for b in range(B):
                for ci, cj, mag, counter in batch_transients[b]:
                    s_aug_real[b, ci] = s_aug_real[b, ci] + 0.1 * mag.real
                    s_aug_imag[b, ci] = s_aug_imag[b, ci] + 0.1 * mag.imag
                    s_aug_real[b, cj] = s_aug_real[b, cj] + 0.1 * mag.real
                    s_aug_imag[b, cj] = s_aug_imag[b, cj] + 0.1 * mag.imag
            s_aug = torch.complex(s_aug_real, s_aug_imag)

            # 5. Complex regime-aware update
            c_aug = h * s_aug  # real * complex = complex
            re_c = c_aug.real
            im_c = c_aug.imag

            # Resonance: Re(c) > 0 and |Im(c)| < Re(c)
            res_mask = ((re_c > 1e-6) & (im_c.abs() < re_c)).float()
            # Torque: Re(c) < 0 or |Im(c)| >= |Re(c)|
            tor_mask = ((re_c < -1e-6) | (im_c.abs() >= re_c.abs())).float()
            orth_mask = 1.0 - res_mask - tor_mask

            # Update: resonance and torque update from c, orthogonality skips
            # Torque gets additional learned rotational bias
            update = eta * (c_aug * (1.0 - orth_mask) + tor_mask * torque_bias)
            s = self._renorm(s + update)

            # Output: magnitude per dimension (real-valued for layer stacking)
            outputs.append(s.abs().clone())

        output = torch.stack(outputs, dim=1)  # (B, T, D) real
        return output, {"binding_events": binding_events}


class MemoryEngineModelWithBinding(nn.Module):
    """Full model: embedding -> ME layers with binding -> readout."""

    def __init__(self, vocab_size: int = 32, dim: int = 64,
                 n_layers: int = 4, max_seq_len: int = 128,
                 top_k: int = 16, max_transients: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len

        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(max_seq_len, dim)
        self.me_layers = nn.ModuleList([
            StandaloneMELayerWithBinding(
                dim, top_k=min(top_k, dim), max_transients=max_transients
            ) for _ in range(n_layers)
        ])
        self.readout = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None) -> dict:
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.embed(input_ids) + self.pos_embed(positions)

        all_hidden = [x.detach()]
        all_binding = []

        for layer in self.me_layers:
            x, bind_info = layer(x)
            all_hidden.append(x.detach())
            all_binding.append(bind_info)

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
            "binding_info": all_binding,
        }

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "trainable": total}


# -- Task generators -------------------------------------------------------

def generate_copy_task(vocab_size=8, seq_len=10, n_samples=1000):
    inputs = torch.randint(0, vocab_size, (n_samples, seq_len))
    return inputs, inputs.clone()


def generate_associative_recall(vocab_size=16, n_pairs=3, n_samples=1000):
    seq_len = n_pairs * 2 + 2
    inputs = torch.zeros(n_samples, seq_len, dtype=torch.long)
    targets = torch.zeros(n_samples, seq_len, dtype=torch.long)

    for i in range(n_samples):
        for p in range(n_pairs):
            key = torch.randint(0, vocab_size // 2, (1,)).item()
            val = torch.randint(vocab_size // 2, vocab_size, (1,)).item()
            inputs[i, p * 2] = key
            inputs[i, p * 2 + 1] = val

        query_idx = torch.randint(0, n_pairs, (1,)).item()
        query_key = inputs[i, query_idx * 2].item()
        target_val = inputs[i, query_idx * 2 + 1].item()

        inputs[i, n_pairs * 2] = vocab_size - 1
        inputs[i, n_pairs * 2 + 1] = query_key

        targets[i, :n_pairs * 2] = inputs[i, :n_pairs * 2]
        targets[i, n_pairs * 2] = inputs[i, n_pairs * 2]
        targets[i, n_pairs * 2 + 1] = target_val

    return inputs, targets


def generate_sequence_prediction(vocab_size=4, seq_len=20, n_samples=1000):
    pattern = torch.arange(vocab_size)
    full = pattern.repeat(seq_len // vocab_size + 1)[:seq_len + 1]
    inputs = full[:seq_len].unsqueeze(0).repeat(n_samples, 1)
    targets = full[1:seq_len + 1].unsqueeze(0).repeat(n_samples, 1)
    return inputs, targets


# -- Training ---------------------------------------------------------------

def train_model(model, inputs, targets, n_epochs=50, lr=0.005,
                batch_size=32, eval_every=10, task_name="task"):
    n_samples = inputs.shape[0]
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    print(f"\nTraining on {task_name}")
    print(f"  Model params: {model.count_parameters()}")
    print(f"  Samples: {n_samples}, Batch: {batch_size}")
    print(f"{'Epoch':>5} {'Loss':>8} {'Accuracy':>10} {'Bindings':>10}")
    print("-" * 40)

    for epoch in range(n_epochs):
        model.train()
        indices = torch.randperm(n_samples)
        epoch_loss = 0
        n_correct = 0
        n_total = 0
        total_bindings = 0

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
            preds = outputs["logits"].argmax(dim=-1)
            n_correct += (preds == y).sum().item()
            n_total += y.numel()

            # Track binding activity
            if outputs.get("binding_info"):
                for layer_bind in outputs["binding_info"]:
                    for evt in layer_bind["binding_events"]:
                        total_bindings += evt["active"]

        scheduler.step()

        if (epoch + 1) % eval_every == 0 or epoch == 0:
            avg_loss = epoch_loss / n_samples
            accuracy = n_correct / n_total
            avg_bindings = total_bindings / max(n_samples, 1)
            print(f"  {epoch+1:>3} {avg_loss:>8.3f} {accuracy:>9.3f} {avg_bindings:>9.1f}")

    # Final eval with diagnostics
    model.eval()
    with torch.no_grad():
        outputs = model(inputs[:5])
        preds = outputs["logits"].argmax(dim=-1)
        final_correct = (preds == targets[:5]).float().mean().item()

    print(f"  Final sample accuracy: {final_correct:.3f}")

    # PR diagnostics
    print(f"  PR across layers:")
    for i, hs in enumerate(outputs["hidden_states"]):
        mean_pr = np.mean([
            participation_ratio(hs[0, pos].numpy().astype(complex))
            for pos in range(hs.shape[1])
        ])
        print(f"    Layer {i}: PR={mean_pr:.1f}")

    # Binding diagnostics for last sample
    if outputs.get("binding_info"):
        print(f"  Binding activity (last layer, last sample):")
        last_layer = outputs["binding_info"][-1]
        events = last_layer["binding_events"]
        if events:
            active_counts = [e["active"] for e in events]
            new_counts = [e["new"] for e in events]
            refresh_counts = [e["refreshed"] for e in events]
            print(f"    Active transients: min={min(active_counts)}, "
                  f"max={max(active_counts)}, mean={np.mean(active_counts):.1f}")
            print(f"    New bindings: total={sum(new_counts)}")
            print(f"    Refreshed bindings: total={sum(refresh_counts)}")
        else:
            print(f"    No binding events recorded")

    return model


# -- Main -------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("Memory Engine Model with Fast Binding (Complex Tape)")
    print("=" * 60)

    # Task 1: Copy (small)
    print("\n--- TASK 1: Copy ---")
    copy_inputs, copy_targets = generate_copy_task(
        vocab_size=8, seq_len=8, n_samples=500
    )
    model1 = MemoryEngineModelWithBinding(
        vocab_size=8, dim=16, n_layers=2, max_seq_len=8, top_k=6, max_transients=8
    )
    model1 = train_model(model1, copy_inputs, copy_targets,
                          n_epochs=40, lr=0.01, batch_size=16, task_name="Copy")

    # Task 2: Sequence prediction (small)
    print("\n--- TASK 2: Sequence Prediction ---")
    seq_inputs, seq_targets = generate_sequence_prediction(
        vocab_size=4, seq_len=12, n_samples=500
    )
    model2 = MemoryEngineModelWithBinding(
        vocab_size=4, dim=16, n_layers=2, max_seq_len=12, top_k=6, max_transients=8
    )
    model2 = train_model(model2, seq_inputs, seq_targets,
                          n_epochs=40, lr=0.01, batch_size=16, task_name="Sequence Prediction")

    # Task 3: Associative recall (the binding test -- small)
    print("\n--- TASK 3: Associative Recall ---")
    assoc_inputs, assoc_targets = generate_associative_recall(
        vocab_size=16, n_pairs=2, n_samples=500
    )
    model3 = MemoryEngineModelWithBinding(
        vocab_size=16, dim=32, n_layers=3, max_seq_len=6, top_k=12, max_transients=16
    )
    model3 = train_model(model3, assoc_inputs, assoc_targets,
                          n_epochs=60, lr=0.01, batch_size=16, task_name="Associative Recall")

    # Summary comparison
    print("\n" + "=" * 60)
    print("Summary: Standalone Model with Fast Binding (Complex Tape)")
    print("=" * 60)
    print(f"  Compare with baselines:")
    print(f"  Copy:          no-binding 67.2% -> see above")
    print(f"  Sequence:      no-binding 100%  -> see above")
    print(f"  Recall:        no-binding 40.0% -> see above (key test for binding)")
    print("=" * 60)
