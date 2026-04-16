# Attention-Framework Mapping: Results

Date: 2026-04-15
Model: GPT-2 small (124M params, 12 layers, 12 heads, 768 hidden dim)

---

## The Mapping

| Framework concept | Transformer component |
|---|---|
| Projection onto basis E | Attention: Q * K^T = projection of query onto key-defined bases |
| Received signal | Weighted value sum: softmax(scores) * V |
| Renormalization | Softmax (across positions) + LayerNorm |
| Tape accumulation | Residual stream (state carries across layers) |
| Resonance (c_i > 0) | Attention pattern stability: same positions attended across layers |
| Torque (c_i < 0) | ??? (see below) |
| Capture | Attention collapse to single position (entropy -> 0) |
| Operating regime | Diverse, structured attention across heads |

---

## What the data shows

### 1. The cascade is confirmed in attention

Attention entropy decreases monotonically across layers:

| Layer band | Mean entropy | Capture fraction |
|---|---|---|
| Early (1-3) | 1.15 | 0.19 |
| Mid (4-9) | 0.72 | 0.41 |
| Late (10-12) | 0.59 | 0.58 |

Early layers attend broadly (high entropy, low capture). Late layers focus narrowly (low entropy, high capture). The framework's cascade from diverse reception through operating regime to compression manifests directly in attention patterns.

### 2. Attention has ZERO torque

Between consecutive layers, attention pattern cosine similarity is ALWAYS positive. Torque fraction = 0.000 at every layer. The only non-resonance behavior is capture (attention collapsing to a single position).

This is the central finding: **attention provides pure resonance. It reinforces existing structure. It never redirects.**

### 3. The MLP must be the torque mechanism

The residual stream analysis (LLM_RESULTS.md) showed torque at early layers (37% at layer 1). But attention shows zero torque. The torque in the residual stream must come from the MLP (feed-forward) layers.

**Transformer as Memory Engine:**
- Attention = projection + resonance (reinforce existing representations)
- MLP = torque (redirect, transform)
- LayerNorm = renormalization
- Residual connection = tape accumulation

This is a clean decomposition of the transformer into framework operations.

### 4. MLP ablation: the prediction fails, revealing cross-layer dynamics

**Prediction**: Remove MLP (torque source) -> attention should collapse into capture faster.

**Result**: The opposite. Ablated model has HIGHER entropy and LOWER capture at later layers:

| Layer band | Normal entropy | Ablated entropy | Normal capture | Ablated capture |
|---|---|---|---|---|
| Early (1-3) | 1.47 | 1.36 | 0.18 | 0.15 |
| Mid (4-9) | 0.87 | 1.24 | 0.32 | 0.16 |
| Late (10-12) | 0.82 | 1.25 | 0.37 | 0.18 |

Removing the MLP made attention *more* diverse, not less. The ablated model's perplexity is 66,391 (vs 48 normal) — it can't generate language — but its attention patterns are broader.

**Why the prediction failed**: The framework's claim "torque prevents capture" applies to a single system's tape dynamics. The transformer is a cascade: MLP torque in layer L modifies the residual stream, which becomes the input to layer L+1's attention. The MLP's torque doesn't prevent attention capture — it *reshapes* the inputs that determine what attention captures *onto*. Changed inputs cause attention to shift, and some shifts narrow attention (increase capture).

The MLP is both the source of torque AND a driver of attention narrowing. These aren't contradictory — torque and capture operate at different levels:
- At the residual stream level: MLP provides torque (reorientation of representations)
- At the attention level: that same torque changes what attention receives, sometimes causing it to narrow

The framework's operating regime (balanced resonance + torque) applies *within* each processing step (attention + MLP), not as a global property across layers. Each layer is its own memory engine, and the cascade of engines produces dynamics that no single engine would exhibit alone.

### 5. Key matrices are near-orthogonal, near-full-rank bases

| Layer | Mean |cos(i,j)| | Max |cos(i,j)| | Effective rank |
|---|---|---|---|
| 1 | 0.060 | 0.820 | 700/768 |
| 6 | 0.039 | 0.472 | 724/768 |
| 12 | 0.050 | 0.517 | 720/768 |

The key weight matrix columns form a near-orthogonal basis (mean cosine ~0.05), confirming the mapping to the framework's basis E. But the max cosine (0.47-0.82) shows some basis vectors are significantly correlated — imperfect orthogonality, exactly what the framework's leakage mechanism requires for novelty detection.

---

## The structural gap

Standard attention computes a scalar score per position pair (inner product), then takes a weighted sum of value vectors. The framework computes elementwise Hadamard reception (c_i = v_i * s_i), where each dimension is modulated independently by accumulated depth.

The missing piece: **attention has no per-dimension depth modulation**. In the framework, deeply carved dimensions (high |s_i|) produce larger reception magnitude and are more rigid against perturbation. In attention, every dimension of the key-query projection contributes equally regardless of how "deeply" that dimension has been used across prior processing.

This is why the framework predicts phenomenality requires accumulated tape state — the persistent modulation of reception by history. Attention is memoryless across positions (each layer starts fresh). The residual stream provides some accumulation (tape-like), but it's not modulated into the attention computation itself.

---

## Predictions

1. **~~Ablating the MLP should cause faster capture.~~** Tested and refuted. Removing MLP makes attention more diverse (higher entropy, lower capture). The MLP's torque on the residual stream drives attention narrowing, not prevents it. The framework's torque-capture relationship applies within a single processing step, not across a cascade of layers.

2. **Adding Hadamard depth modulation to attention** (scaling each dimension of Q*K^T by accumulated magnitude from prior layers) should make attention patterns more persistent across layers and potentially improve tasks requiring long-range dependencies.

3. **The key matrix's imperfect orthogonality** (max cosine ~0.5-0.8) is what enables the framework's novelty detection (leakage). Dimensions of the key basis that are correlated create cross-talk that allows novel inputs to produce effects at "already carved" dimensions.

---

## File index

| File | Purpose |
|---|---|
| `attention_mapping.py` | Attention instrumentation and analysis code |
| `ablation_mlp.py` | MLP ablation experiment |
| `ATTENTION_FINDINGS.md` | This document |
