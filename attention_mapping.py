"""
Attention-Framework Mapping: Empirical Test.

Tests the claim that attention IS projection (a subset of reception)
by instrumenting GPT-2's attention patterns using framework vocabulary.

Formal mapping:
  Framework:  c_j = <e_j | v> * s_j    (Hadamard reception)
  Attention:  a_ij = softmax(q_i . k_j) (inner product + normalization)

  - W_K columns = basis vectors E (what the system can receive)
  - q_i . k_j = <q_i | e_j> = projection of position i onto basis j
  - softmax = renormalization across positions (not dimensions)
  - Value vectors V = distributed tape state

Key structural difference:
  Framework has Hadamard (elementwise) reception: c_i = v_i * s_i
  Attention has inner product reception: scalar score per position pair
  Standard attention LACKS the multiplicative depth modulation (tape * projection)

Testable predictions:
  1. Attention entropy should follow the cascade pattern:
     - Early heads: low entropy (focused, high-torque)
     - Mid heads: moderate entropy (operating regime)
     - Late heads: low entropy (capture/compression)
  2. Attention pattern stability across layers should show resonance/torque structure
  3. Head diversity (attention PR) should follow the cascade
"""

import numpy as np
import torch
from transformers import GPT2Model, GPT2Tokenizer, GPT2LMHeadModel, GPT2Config

np.set_printoptions(precision=4, suppress=True)


def load_model(model_name="gpt2"):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    config = GPT2Config.from_pretrained(model_name)
    config.output_attentions = True
    model = GPT2LMHeadModel.from_pretrained(model_name, config=config)
    model.eval()
    return model, tokenizer


def attention_entropy(attn_weights: np.ndarray) -> np.ndarray:
    """Compute entropy of attention distribution for each query position.

    attn_weights: (n_heads, seq_len, seq_len)
    Returns: (n_heads, seq_len) entropy per head per position
    """
    eps = 1e-10
    probs = attn_weights + eps
    return -np.sum(probs * np.log(probs), axis=-1)


def attention_pr(attn_weights: np.ndarray) -> float:
    """Participation ratio of attention weight magnitudes across heads.

    Measures how many heads are effectively doing different things.
    attn_weights: (n_heads, seq_len, seq_len)
    Returns scalar PR across heads (averaged over positions).
    """
    n_heads, seq_len, _ = attn_weights.shape
    prs = []
    for pos in range(seq_len):
        # Each head's attention pattern at this position as a vector
        patterns = attn_weights[:, pos, :]  # (n_heads, seq_len)
        # Compute PR of the magnitude distribution across heads
        mags = np.abs(patterns).mean(axis=-1)  # (n_heads,) mean attention per head
        mags_sq = mags ** 2
        if mags_sq.sum() > 0:
            prs.append(mags_sq.sum() ** 2 / (mags_sq ** 2).sum())
    return float(np.mean(prs)) if prs else 0.0


def attention_regime(attn_current: np.ndarray, attn_previous: np.ndarray,
                     threshold: float = 0.01) -> dict:
    """Classify attention head behavior as resonance/torque/capture.

    Compares attention patterns at consecutive layers.
    - Resonance: same positions attended (cosine similarity > 0)
    - Torque: different positions attended (cosine similarity < 0)
    - Capture: attention collapsed to single position (entropy near 0)

    attn_current: (n_heads, seq_len, seq_len) for layer L
    attn_previous: (n_heads, seq_len, seq_len) for layer L-1
    """
    n_heads, seq_len, _ = attn_current.shape
    resonance_count = 0
    torque_count = 0
    capture_count = 0

    for h in range(n_heads):
        for pos in range(seq_len):
            pattern_curr = attn_current[h, pos, :]
            pattern_prev = attn_previous[h, pos, :]

            # Check capture: max attention weight > 0.9
            if pattern_curr.max() > 0.9:
                capture_count += 1
                continue

            # Cosine similarity between attention patterns
            dot = np.dot(pattern_curr, pattern_prev)
            norm_c = np.linalg.norm(pattern_curr)
            norm_p = np.linalg.norm(pattern_prev)
            if norm_c < 1e-10 or norm_p < 1e-10:
                continue

            cos_sim = dot / (norm_c * norm_p)
            if cos_sim > threshold:
                resonance_count += 1
            elif cos_sim < -threshold:
                torque_count += 1

    total = n_heads * seq_len
    return {
        "resonance": resonance_count,
        "torque": torque_count,
        "capture": capture_count,
        "resonance_frac": resonance_count / total,
        "torque_frac": torque_count / total,
        "capture_frac": capture_count / total,
    }


def head_specialization(attn_weights: np.ndarray) -> float:
    """Measure how specialized heads are (low = all doing the same, high = diverse).

    Computes mean pairwise cosine distance between head attention patterns.
    attn_weights: (n_heads, seq_len, seq_len)
    """
    n_heads, seq_len, _ = attn_weights.shape
    # Flatten each head's full attention matrix to a vector
    patterns = attn_weights.reshape(n_heads, -1)  # (n_heads, seq_len*seq_len)

    # Pairwise cosine similarity
    dists = []
    for i in range(n_heads):
        for j in range(i + 1, n_heads):
            dot = np.dot(patterns[i], patterns[j])
            ni = np.linalg.norm(patterns[i])
            nj = np.linalg.norm(patterns[j])
            if ni > 1e-10 and nj > 1e-10:
                cos_sim = dot / (ni * nj)
                dists.append(1.0 - cos_sim)  # cosine distance

    return float(np.mean(dists)) if dists else 0.0


def instrument_attention(model, tokenizer, text: str) -> dict:
    """Run forward pass collecting attention weights and framework diagnostics."""
    input_ids = tokenizer.encode(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True, output_hidden_states=True)

    # attentions: tuple of (n_layers,) tensors, each (batch, n_heads, seq_len, seq_len)
    attentions = [a.squeeze(0).numpy() for a in outputs.attentions]
    n_layers = len(attentions)
    n_heads, seq_len, _ = attentions[0].shape

    results = {
        "n_layers": n_layers,
        "n_heads": n_heads,
        "seq_len": seq_len,
        "tokens": [tokenizer.decode(t) for t in input_ids[0]],
        "entropy": [],       # per layer: (n_heads, seq_len)
        "head_pr": [],       # per layer: scalar
        "regime": [],        # per layer: dict
        "specialization": [],  # per layer: scalar
        "capture_frac": [],  # per layer: fraction of positions in capture
    }

    for layer in range(n_layers):
        attn = attentions[layer]

        # Entropy
        ent = attention_entropy(attn)
        results["entropy"].append(ent)

        # Head PR (diversity)
        pr = attention_pr(attn)
        results["head_pr"].append(pr)

        # Specialization
        spec = head_specialization(attn)
        results["specialization"].append(spec)

        # Capture fraction
        max_attn = attn.max(axis=-1)  # (n_heads, seq_len)
        capture_frac = (max_attn > 0.9).mean()
        results["capture_frac"].append(capture_frac)

        # Regime (compare to previous layer)
        if layer > 0:
            regime = attention_regime(attn, attentions[layer - 1])
            results["regime"].append(regime)

    return results


def print_report(results: dict):
    """Print attention-framework mapping report."""
    print("=" * 70)
    print("Attention-Framework Mapping: GPT-2 Analysis")
    print("=" * 70)
    print(f"Layers: {results['n_layers']}, Heads: {results['n_heads']}, "
          f"Seq len: {results['seq_len']}")
    print(f"Tokens: {results['tokens']}")
    print()

    # Cascade prediction: entropy should follow the cascade
    print("--- Attention Entropy (bits, mean over heads x positions) ---")
    print(f"{'Layer':>5} {'Mean Ent':>9} {'Head PR':>9} {'Spec':>7} {'Capture':>8}")
    print("-" * 45)
    for layer in range(results["n_layers"]):
        ent = results["entropy"][layer].mean()
        pr = results["head_pr"][layer]
        spec = results["specialization"][layer]
        cap = results["capture_frac"][layer]
        print(f"  {layer+1:>3} {ent:>9.3f} {pr:>9.1f} {spec:>7.3f} {cap:>8.3f}")

    print()

    # Regime classification
    print("--- Regime Classification (attention pattern stability) ---")
    print(f"{'Layer':>5} {'Resonance':>10} {'Torque':>8} {'Capture':>8}")
    print("-" * 35)
    for i, regime in enumerate(results["regime"]):
        print(f"  {i+2:>3} {regime['resonance_frac']:>10.3f} "
              f"{regime['torque_frac']:>8.3f} {regime['capture_frac']:>8.3f}")

    print()

    # Framework interpretation
    print("--- Framework Interpretation ---")
    mean_ents = [e.mean() for e in results["entropy"]]
    max_ent_layer = np.argmax(mean_ents) + 1
    min_ent_layer = np.argmin(mean_ents) + 1
    max_pr_layer = np.argmax(results["head_pr"]) + 1
    min_pr_layer = np.argmin(results["head_pr"]) + 1

    print(f"  Peak entropy (diverse attention): layer {max_ent_layer}")
    print(f"  Min entropy (focused attention):  layer {min_ent_layer}")
    print(f"  Peak head diversity (PR):          layer {max_pr_layer}")
    print(f"  Min head diversity (PR):           layer {min_pr_layer}")

    # Check cascade prediction
    early_ent = np.mean(mean_ents[:3])
    mid_ent = np.mean(mean_ents[3:9])
    late_ent = np.mean(mean_ents[9:])

    print(f"\n  Cascade prediction check:")
    print(f"    Early (1-3) mean entropy: {early_ent:.3f}")
    print(f"    Mid (4-9) mean entropy:   {mid_ent:.3f}")
    print(f"    Late (10-12) mean entropy:{late_ent:.3f}")

    if mid_ent > early_ent and mid_ent > late_ent:
        print("    -> Mid layers have highest entropy (operating regime confirmed)")
    elif late_ent < early_ent:
        print("    -> Late layers most captured (compression confirmed)")
    else:
        print("    -> Cascade pattern not clearly present in attention entropy")

    # Capture analysis
    capture_trend = results["capture_frac"]
    print(f"\n  Capture fraction trend: {[f'{c:.3f}' for c in capture_trend]}")
    if capture_trend[-1] > capture_trend[0]:
        print("    -> Increasing capture toward output (functional compression)")
    else:
        print("    -> No clear capture trend")


# --- Key-Value as Basis Analysis ---

def analyze_kv_as_basis(model, tokenizer, text: str):
    """Analyze key and value weight matrices as learned bases.

    The mapping claims: W_K columns = basis vectors, W_V = tape state projection.
    This function checks whether key vectors behave like an orthogonal basis
    and whether value vectors carry accumulated representation depth.
    """
    input_ids = tokenizer.encode(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True, output_hidden_states=True)

    print("\n--- Key/Value as Basis Analysis ---")
    print()

    # GPT-2 attention: c_attn projects to Q, K, V simultaneously
    # For each transformer block, get the key and value weight matrices
    for layer_idx, block in enumerate(model.transformer.h):
        # c_attn weight: (hidden_dim, 3 * hidden_dim) -> split into Q, K, V
        attn_weight = block.attn.c_attn.weight.detach().numpy()
        hidden_dim = attn_weight.shape[0]
        W_K = attn_weight[:, hidden_dim:2*hidden_dim]  # (hidden_dim, hidden_dim)
        W_V = attn_weight[:, 2*hidden_dim:]             # (hidden_dim, hidden_dim)

        # Analyze key weight columns as basis
        # Normalize columns
        col_norms = np.linalg.norm(W_K, axis=0, keepdims=True)
        W_K_normed = W_K / (col_norms + 1e-10)

        # Column-wise cosine similarity matrix (are basis vectors orthogonal?)
        cos_sim_matrix = W_K_normed.T @ W_K_normed  # (hidden_dim, hidden_dim)
        off_diag = cos_sim_matrix[np.triu_indices(hidden_dim, k=1)]
        mean_cos = np.mean(np.abs(off_diag))
        max_cos = np.max(np.abs(off_diag))

        # Singular value analysis of W_K (effective basis rank)
        svd = np.linalg.svd(W_K, compute_uv=False)
        svd_norm = svd / svd[0]
        effective_rank = np.sum(svd_norm > 0.01)

        if layer_idx in [0, 5, 11]:  # Sample layers
            print(f"  Layer {layer_idx+1}:")
            print(f"    Key basis orthogonality: mean |cos(i,j)| = {mean_cos:.4f}, "
                  f"max = {max_cos:.4f}")
            print(f"    Key effective rank (SV > 1% max): {effective_rank}/{hidden_dim}")
            print(f"    Key SV distribution: "
                  f"top3={svd[:3].round(1)}, bottom3={svd[-3:].round(1)}")

            # Value matrix: how much does it project vs. rotate?
            svd_v = np.linalg.svd(W_V, compute_uv=False)
            v_eff_rank = np.sum(svd_v / svd_v[0] > 0.01)
            print(f"    Value effective rank: {v_eff_rank}/{hidden_dim}")


if __name__ == "__main__":
    print("Loading GPT-2...")
    model, tokenizer = load_model()

    # Test with varied text
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "In the beginning God created the heaven and the earth.",
        "To be or not to be that is the question whether it is nobler",
    ]

    for text in texts:
        print(f"\n{'=' * 70}")
        print(f"Input: {text}")
        print(f"{'=' * 70}")

        results = instrument_attention(model, tokenizer, text)
        print_report(results)

    # KV basis analysis
    print("\n" + "=" * 70)
    analyze_kv_as_basis(model, tokenizer, texts[0])
