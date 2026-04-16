"""
LLM Memory Engine Instrumentation — Level 1.

Hooks into GPT-2 inference to extract per-layer, per-position diagnostics
using the Memory Engine framework's vocabulary: participation ratio, anisotropy,
regime classification (resonance/torque/orthogonality), self-torque, rigidity.

Real-valued adaptation: hidden states are real vectors, not complex.
Sign acts as binary phase:
  - same sign at a dimension = resonance (aligned)
  - sign flip = torque (opposed)
  - one near zero = orthogonality
"""

import numpy as np
import torch
from transformers import GPT2Model, GPT2Tokenizer, GPT2LMHeadModel
from engine import participation_ratio

np.set_printoptions(precision=4, suppress=True)


# ── Model loading ────────────────────────────────────────────────────────

def load_model(model_name="gpt2"):
    """Load GPT-2 and tokenizer. Returns (model, tokenizer)."""
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    return model, tokenizer


# ── Real-valued diagnostics ──────────────────────────────────────────────

def compute_pr(hidden_state: np.ndarray) -> float:
    """Participation ratio of a real-valued hidden state vector.

    Normalizes to unit norm first (the framework assumes the tape lives
    on the unit hypersphere; LLM residual streams do NOT, so we project).
    """
    norm = np.linalg.norm(hidden_state)
    if norm < 1e-10:
        return 0.0
    h_normed = hidden_state / norm
    return participation_ratio(h_normed.astype(complex))


def compute_anisotropy(hidden_state: np.ndarray) -> float:
    """Gini coefficient of |h_i| magnitudes. 0=uniform, 1=concentrated."""
    x = np.sort(np.abs(hidden_state))
    n = len(x)
    if np.sum(x) < 1e-10:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * x) - (n + 1) * np.sum(x)) / (n * np.sum(x)))


def compute_regime(h_current: np.ndarray, h_previous: np.ndarray,
                   orth_threshold: float = 0.01) -> dict:
    """Classify each dimension into resonance/torque/orthogonality.

    For real-valued vectors, sign is binary phase:
      - product > 0 → resonance (same sign, aligned)
      - product < 0 → torque (opposite sign, opposed)
      - either |h| near zero → orthogonality
    """
    product = h_current * h_previous
    mags = np.abs(h_current)
    prev_mags = np.abs(h_previous)

    n_resonance = int(np.sum(
        (product > 0) & (mags > orth_threshold) & (prev_mags > orth_threshold)
    ))
    n_torque = int(np.sum(
        (product < 0) & (mags > orth_threshold) & (prev_mags > orth_threshold)
    ))
    n_orth = int(len(product) - n_resonance - n_torque)
    n_total = len(product)

    return {
        "resonance": n_resonance,
        "torque": n_torque,
        "orthogonality": n_orth,
        "resonance_frac": n_resonance / n_total,
        "torque_frac": n_torque / n_total,
        "orth_frac": n_orth / n_total,
    }


def angular_displacement(h1: np.ndarray, h2: np.ndarray) -> float:
    """Angular displacement between two real vectors (cosine distance)."""
    dot = np.dot(h1, h2)
    norm1 = np.linalg.norm(h1)
    norm2 = np.linalg.norm(h2)
    if norm1 < 1e-10 or norm2 < 1e-10:
        return np.pi / 2
    cos_sim = np.clip(dot / (norm1 * norm2), -1.0, 1.0)
    return float(np.arccos(cos_sim))


def compute_norm(hidden_state: np.ndarray) -> float:
    """L2 norm of hidden state (check if near unit hypersphere)."""
    return float(np.linalg.norm(hidden_state))


# ── Forward pass instrumentation ─────────────────────────────────────────

def instrument_forward(model, tokenizer, text: str) -> dict:
    """Run a full forward pass and collect diagnostics at every layer and position.

    Returns dict with keys:
        hidden_states: np.ndarray (n_layers+1, seq_len, hidden_dim)
        pr: np.ndarray (n_layers+1, seq_len)
        anisotropy: np.ndarray (n_layers+1, seq_len)
        norms: np.ndarray (n_layers+1, seq_len)
        regime_by_layer: list of dicts (one per layer, comparing to previous layer)
        input_ids: token ids
        tokens: decoded tokens
    """
    input_ids = tokenizer.encode(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)

    # hidden_states: tuple of (n_layers+1) tensors, each (batch, seq_len, hidden_dim)
    hs = torch.stack(outputs.hidden_states).squeeze(1).numpy()  # (n_layers+1, seq_len, dim)
    n_layers_plus1, seq_len, hidden_dim = hs.shape

    # Per-position, per-layer diagnostics
    pr_grid = np.zeros((n_layers_plus1, seq_len))
    anisotropy_grid = np.zeros((n_layers_plus1, seq_len))
    norm_grid = np.zeros((n_layers_plus1, seq_len))

    for layer in range(n_layers_plus1):
        for pos in range(seq_len):
            h = hs[layer, pos]
            pr_grid[layer, pos] = compute_pr(h)
            anisotropy_grid[layer, pos] = compute_anisotropy(h)
            norm_grid[layer, pos] = compute_norm(h)

    # Regime classification: compare each layer to previous layer, per position
    regime_by_layer = []
    for layer in range(1, n_layers_plus1):
        layer_regimes = []
        for pos in range(seq_len):
            r = compute_regime(hs[layer, pos], hs[layer - 1, pos])
            layer_regimes.append(r)
        regime_by_layer.append(layer_regimes)

    tokens = [tokenizer.decode(t) for t in input_ids[0]]

    return {
        "hidden_states": hs,
        "pr": pr_grid,
        "anisotropy": anisotropy_grid,
        "norms": norm_grid,
        "regime_by_layer": regime_by_layer,
        "input_ids": input_ids[0].numpy(),
        "tokens": tokens,
        "n_layers": n_layers_plus1 - 1,  # excluding embedding layer
        "seq_len": seq_len,
        "hidden_dim": hidden_dim,
    }


def compute_self_torque_matrix(hs: np.ndarray, layer: int,
                                delays: list[int]) -> np.ndarray:
    """Compute angular displacement across positions at a given layer.

    hs: (n_layers+1, seq_len, dim)
    layer: which layer to analyze
    delays: list of position delays to compute

    Returns: (seq_len, len(delays)) matrix of angular displacements.
    """
    seq_len = hs.shape[1]
    result = np.full((seq_len, len(delays)), np.nan)

    for pos in range(seq_len):
        for j, delay in enumerate(delays):
            if pos - delay >= 0:
                result[pos, j] = angular_displacement(
                    hs[layer, pos], hs[layer, pos - delay]
                )
    return result


# ── Smoke test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading GPT-2...")
    model, tokenizer = load_model()

    text = "The quick brown fox jumps over the lazy dog. "
    print(f"Input: {text.strip()}")
    print(f"Tokens: {tokenizer.encode(text)}")

    print("\nRunning instrumented forward pass...")
    result = instrument_forward(model, tokenizer, text)

    print(f"Layers: {result['n_layers']} (+ embedding)")
    print(f"Sequence length: {result['seq_len']}")
    print(f"Hidden dim: {result['hidden_dim']}")

    # Norm check: is the residual stream near unit hypersphere?
    print(f"\nNorm range: [{result['norms'].min():.3f}, {result['norms'].max():.3f}]")
    print(f"  (LayerNorm should keep norms near ~1.0 per sublayer)")

    # PR at final layer across positions
    final_pr = result["pr"][-1]
    print(f"\nPR at final layer across positions:")
    for i, (tok, pr) in enumerate(zip(result["tokens"], final_pr)):
        print(f"  pos {i:>2} '{tok:>6}': PR={pr:.1f}")

    # Anisotropy at final layer
    final_gini = result["anisotropy"][-1]
    print(f"\nAnisotropy (Gini) at final layer: mean={final_gini.mean():.3f}")

    # Regime profile at final layer
    final_regime = result["regime_by_layer"][-1]
    mean_res = np.mean([r["resonance_frac"] for r in final_regime])
    mean_tor = np.mean([r["torque_frac"] for r in final_regime])
    mean_ort = np.mean([r["orth_frac"] for r in final_regime])
    print(f"\nRegime at final layer (mean across positions):")
    print(f"  Resonance:    {mean_res:.3f}")
    print(f"  Torque:       {mean_tor:.3f}")
    print(f"  Orthogonality:{mean_ort:.3f}")

    print("\nSmoke test passed.")
