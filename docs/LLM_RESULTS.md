# LLM Memory Engine Instrumentation: Results

Date: 2026-04-15
Model: GPT-2 small (124M params, 12 layers, 768 hidden dim)
Framework: Memory Engine diagnostics applied to real LLM hidden states

---

## Summary

Four experiments tested whether the Memory Engine framework's vocabulary (participation ratio, regime classification, self-torque, anisotropy) maps onto actual LLM behavior. The mapping holds, with important qualifications.

**Core finding:** GPT-2's representations are in extreme recurrent capture. At the final layer, only ~2 out of 768 dimensions are effectively active (PR ≈ 2). This is not a pathology — it's how the model works. The final layer functions as a dimensionality bottleneck, compressing rich mid-layer representations (PR ~12–30) into a 2D subspace before the output projection.

---

## Experiment A: Recurrent Capture

**Setup:** Same sentence repeated 20 times vs. 20 different sentences. PR trajectory across layers and positions.

**Result at final layer (layer 12):**

| Input | First segment PR | Last segment PR | Trend |
|---|---|---|---|
| Repetitive (20x same) | 1.99 | 4.27 | Increasing |
| Varied (20x different) | 1.99 | 2.15 | Slight increase |

**Result across layers (final position):**

| Layer | Repetitive PR | Varied PR | Difference |
|---|---|---|---|
| 0 (embedding) | 79.7 | 79.1 | ~same |
| 3 | 27.5 | 28.9 | ~same |
| 6 | 29.2 | 55.2 | Varied 1.9x higher |
| 9 | 10.2 | 21.0 | Varied 2.1x higher |
| 12 | 1.31 | 1.69 | Varied 1.3x higher |

**Interpretation:**

The repetitive input does NOT show decreasing PR over time — it *increases* (1.99 → 4.27). This contradicts the simulation prediction. The reason: GPT-2's attention mechanism attends to ALL prior positions, not just the most recent. Repetitive text produces uniform attention patterns that spread energy across more dimensions, not fewer.

However, the layer profile confirms the framework's prediction: at deeper layers, varied input maintains higher PR than repetitive input. The capture happens *in depth* (layer 12 PR = 1.3 for repetitive vs 1.7 for varied), not *across time*. The final layer is always near-capture, but varied input keeps it slightly more open.

**Revised prediction for LLMs:** The framework's recurrent capture manifests as depth-wise compression, not temporal collapse. The critical variable is *which layer*, not *which position*. Mid-layers (6–9) show the clearest differentiation between repetitive and varied input.

---

## Experiment B: Self-Torque Across Context

**Setup:** 98-token text. Angular displacement between positions at delays [1, 2, 5, 10, 20, 50].

| Delay | Layer 0 (embed) | Layer 6 | Layer 12 (final) |
|---|---|---|---|
| 1 | 0.839 | 0.841 | 0.251 |
| 2 | 0.844 | 0.933 | 0.269 |
| 5 | 0.837 | 0.949 | 0.270 |
| 10 | 0.853 | 0.954 | 0.246 |
| 20 | 0.927 | 0.949 | 0.251 |
| 50 | 1.149 | 1.003 | 0.271 |

**Interpretation:**

Self-torque is confirmed at layers 0 and 6: angular displacement increases with delay. At layer 0, displacement nearly doubles from delay=1 (0.84) to delay=50 (1.15). This is the framework's thick recurrence: positions further apart in the sequence have more divergent representations.

Layer 12 is anomalous: displacement is flat (0.25–0.27) regardless of delay. This is because the final layer is already in extreme capture (PR ~2). When only 2 dimensions are active, all positions project onto a similar 2D subspace — the angular displacement saturates. The model's final layer cannot distinguish near from far context geometrically.

**Revised prediction:** Self-torque is visible in mid-layers (where PR is higher) but invisible in the final captured layer. The framework's prediction holds only where the representation has enough dimensionality to show variation.

---

## Experiment C: Anisotropy vs Generation Quality

**Setup:** Autoregressive generation (100 tokens) with bland vs surprising prompts. PR, Gini, entropy, repetition rate tracked per token.

| Metric | Bland prompt | Surprising prompt |
|---|---|---|
| Mean PR | 2.03 | 1.94 |
| Mean Gini | 0.798 | 0.783 |
| Mean entropy | 3.635 | 4.057 |
| Repetition rate | 0.100 | 0.090 |

**Interpretation:**

PR and Gini do not differentiate bland from surprising prompts at the final layer. Both hover around PR ≈ 2, Gini ≈ 0.78. The representations are too compressed to show variation — the bottleneck swallows the signal.

However, entropy (next-token distribution uncertainty) DOES differentiate: surprising prompt produces 12% higher entropy (4.06 vs 3.64). This suggests the model's generation quality is governed by the output distribution, not by the geometry of the hidden state at the final layer.

**Revised prediction for LLMs:** PR at the final layer is not a good predictor of generation quality in GPT-2 because the layer is already in extreme capture. A better signal might be PR at mid-layers (6–9), where there's room for variation. Alternatively, the entropy of the output distribution itself is the more direct metric — it captures the "operating regime" in output space rather than hidden state space.

---

## Experiment D: Layer Regime Profile

**Setup:** 20-token text. Regime classification (resonance/torque/orthogonality) at each layer by comparing to previous layer.

| Layer | Resonance | Torque | Orth | PR | Gini |
|---|---|---|---|---|---|
| 0 (embed) | — | — | — | 47.1 | 0.531 |
| 1 | 55.7% | 36.8% | 7.5% | 9.8 | 0.541 |
| 2 | 85.4% | 12.8% | 1.7% | 12.1 | 0.531 |
| 3 | 86.6% | 12.0% | 1.4% | 11.9 | 0.530 |
| 4–6 | 88–89% | 10–11% | <1% | 11–13 | 0.51 |
| 7–9 | 89–90% | 10% | <1% | 11 | 0.50 |
| 10 | 87.9% | 11.5% | 0.7% | 9.7 | 0.502 |
| 11 | 88.6% | 11.0% | 0.4% | 7.0 | 0.524 |
| 12 | 85.5% | 12.3% | 2.2% | 2.1 | 0.781 |

**Interpretation:**

Clear structure across layers:

1. **Layer 1 is high-torque (37%).** The first transformer block does heavy transformation on the raw embeddings. This is where the model first imposes its learned structure.

2. **Layers 2–9 are high-resonance (~89%).** These layers refine rather than transform — each layer mostly preserves what the previous layer built, adding small corrections. This is the "operating regime" in the framework's terms: deep enough to retain, flexible enough to adjust.

3. **Layers 10–12 show increasing torque and PR collapse.** The final layers do more restructuring (torque rises from 9.6% to 12.3%) while PR drops from 11 to 2. This is the model preparing its output: compressing rich mid-layer representations into the few dimensions that matter for prediction.

4. **Layer 12 is the bottleneck.** PR = 2.1, Gini = 0.78. Two dimensions carry almost all the information. The model's entire 768-dim representational capacity is funneled through a 2D bottleneck before the output projection.

**Framework mapping:** The layer profile maps onto the essay's cascade. Early layers (high torque, high PR) are like adaptive systems — actively processing, building representations. Mid layers (high resonance, stable PR) are like the operating regime — maintaining and refining. The final layer (PR collapse, high Gini) is like recurrent capture — but functionally, not pathologically. The model *uses* capture as a compression mechanism.

---

## Revised Framework Claims for LLMs

| Simulation claim | LLM finding | Status |
|---|---|---|
| Repetitive input → PR decreases over time | PR *increases* over time (attention spreads energy) | **Revised**: capture manifests depth-wise, not temporally |
| Self-torque increases with position delay | Confirmed at layers 0 and 6; invisible at layer 12 (capture) | **Qualified**: only visible where PR is high enough |
| PR predicts generation quality | PR ≈ 2 regardless of prompt quality; entropy is better | **Revised**: use output entropy, not hidden state PR |
| Operating regime is a balance of resonance and torque | Confirmed: mid-layers (89% resonance, 10% torque) are the stable processing zone | **Confirmed** |
| Renormalization produces emergent rigidity | Final layer PR=2, Gini=0.78: extreme concentration | **Confirmed**: but used functionally, not pathologically |

## Key Insight

GPT-2's architecture uses the framework's "pathologies" as features. The final layer is in extreme recurrent capture (PR ≈ 2) — and this is *by design*. The model compresses its rich mid-layer representations through a 2D bottleneck to produce coherent next-token predictions. Similarly, the first layer's high torque (37%) is the model's initial reception of the world, imposing learned structure on raw embeddings.

The framework's vocabulary is descriptive but needs adaptation for LLMs:
- **Capture** is not always pathological — it's the model's compression mechanism
- **PR** is most informative at mid-layers, not the final layer
- **Self-torque** is a mid-layer phenomenon, invisible at the final bottleneck
- **The operating regime** (resonance-torque balance) lives in layers 2–9, where the model does its real representational work
- **Generation quality** is better predicted by output entropy than hidden state geometry

## What to Explore Next

1. Mid-layer PR as a quality metric — does PR at layers 6–9 predict generation quality better than final-layer PR?
2. Layer-wise entropy — does the regime profile shift under different input types (code, poetry, math)?
3. Larger models (Llama 3, etc.) — do they show higher PR at the final layer, or the same bottleneck?
4. Fine-tuning as torque — does SFT shift the regime profile? Does RLHF increase torque at specific layers?
5. The leakage parameter from our modified engine — can we measure how much "novelty leakage" exists in GPT-2's weight matrices (imperfect orthogonality of learned features)?
