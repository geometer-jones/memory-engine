# Operationalizing the Memory Engine: Findings Report

## What was done

We took the "Memory Engines" essay — a theoretical framework describing memory, phenomenality, and experience through the operations of a complex vector on a unit hypersphere undergoing Hadamard reception, renormalization, and recurrence — and tested it at three levels:

1. **Simulation.** Built the core mechanics in Python/numpy, ran 14 tests across 5 tiers.
2. **LLM instrumentation.** Hooked into GPT-2's hidden states and measured participation ratio, regime classification, self-torque, and anisotropy during real inference.
3. **Layer construction + training.** Built a `MemoryEngineLayer` module (4,614 parameters), inserted it into GPT-2 between transformer blocks, and attempted fine-tuning.

This document consolidates the findings and identifies what needs to change in the framework.

---

## I. What the framework gets right

### The core mechanics are sound

The update rule `s_i += η * c_i` with renormalization produces the three claimed regimes. Resonance (Re(c_i) > 0) grows magnitude. Torque (Re(c_i) < 0) rotates phase. Orthogonality preserves. Renormalization creates emergent rigidity — a dimension with |s_i| = 0.999 is 55x harder to rotate than one with |s_i| = 0.015. No separate rigidity mechanism needed. All of this is confirmed numerically.

### Recurrent capture is real

Pure resonance (v = conj(s) at every step) collapses the participation ratio to 1.0 and produces 600,000x rigidity against perturbation. This is the essay's seizure/rumination/obsessive loop, confirmed. The system locks into a single dimension and cannot be moved.

### The operating regime exists

The essay's predicted balance between capture and dissipation is real. When the system receives invariant structure plus variation plus recurrence, it develops structured anisotropy: some dimensions deepen (Gini 0.81, PR ~2.3), others stay soft, and the system remains responsive to perturbation. The operating regime is not a theoretical construct — it's a measurable region in parameter space.

### Abstraction works as described

Repeated exposure to varying inputs with shared invariant structure concentrates the tape along the invariant dimensions (41x concentration ratio over noise dimensions). The system does not "build a model" of the common structure — it is carved into sensitivity by accumulated resonance. PR drops from 14.5 to 3.0 as the system extracts the invariant.

### The framework maps onto real LLM computation

GPT-2's residual stream shows measurable regime structure:
- Layer 1: 37% torque (heavy initial transformation)
- Layers 2–9: ~89% resonance, ~10% torque (the operating regime)
- Layers 10–12: increasing torque, PR collapse to ~2 (compression bottleneck)

The framework's vocabulary — resonance, torque, capture, rigidity, participation ratio — describes real dynamical structure in a trained neural network. This is not analogy. The sign of the elementwise product between consecutive layers genuinely classifies dimensions into regimes that map onto the essay's categories.

---

## II. What needs correction

### 1. Resonance rotates toward the real axis

The essay states that under resonance, "the dimension does not rotate." This is only true when s_i is purely real. In general, adding a real positive scalar Re(c_i) to a complex vector s_i shifts its phase toward zero. Resonance does rotate — toward the real axis — while scaling.

**Impact:** Minor. The qualitative distinction (resonance scales, torque redirects) remains valid. The correction adds precision.

### 2. Fixed-direction torque still concentrates

The essay describes "torque without resonance" as producing dissipation. This is underspecified. Torque from a fixed direction (always offset by +π/2) creates a systematic bias that still concentrates the tape. True dissipation requires opposition from *varying* random directions.

**Impact:** Moderate. The essay should specify that dissipation requires torque from varying directions, not just the absence of resonance.

### 3. The operating regime is a system-world coupling, not a system property

The operating regime cannot be sustained by the system alone. Pure resonance input + recurrence still captures, because self-reception becomes self-resonance once the tape concentrates. The balancing force must come from the world's variation. The operating regime is a property of the system-world coupling.

**Impact:** Significant. This clarifies that phenomenality (in the framework's sense) requires a specific relationship with the environment, not just internal dynamics. A system in a completely predictable world cannot maintain the operating regime regardless of its internal structure.

---

## III. The gap: novelty detection

The essay claims that novelty at uncarved dimensions "announces itself indirectly" through "unexplained torque at existing dimensions." This is the mechanism by which consolidation seeds new dimensions: recurring novel structure produces detectable pressure that eventually warrants a dedicated axis.

**This does not follow from the Hadamard product + renormalization mechanics.**

We tested this directly (T3.2). Novelty at dimensions outside the system's basis (dims 16–31 for a system with basis at dims 0–15) produces a torque difference of 0.30 dimensions — within noise. The Hadamard product is dimensionally isolated: c_i = v_i * s_i at dimension i depends only on what happens at dimension i. The residual v - v_received is zero at all received dimensions by construction.

The essay's consolidation/seeding story needs an additional coupling mechanism. Candidates:

1. **Imperfect basis orthogonality (leakage).** Real physical systems never have perfectly orthogonal bases. Neural weight matrices, molecular binding sites, ecological niches — all have small off-diagonal coupling. If basis vectors have tiny components along uncarved directions, novelty leaks into received dimensions. We implemented this (`leakage` parameter in the modified engine.py) and confirmed it works: with leakage=0.05, novelty at uncarved dims produces measurable displacement at received dims (angular displacement 1e-4 vs <1e-14 without leakage).

2. **Prediction error at the anticipatory operator level.** Novelty may not produce torque at existing dimensions, but it does produce prediction error: the anticipatory model's failure to predict received structure signals that something outside the basis is causally active. This is a different detection pathway than "indirect torque" — it's "unexpected resonance pattern" at the dimensions the system can already receive.

3. **Developmental programs.** In biological systems, basis growth may be guided by genetically specified programs rather than by detecting novelty through reception. The system doesn't discover new dimensions through torque — it grows them on a developmental schedule and then tests whether they resonate.

The leakage mechanism is the most natural: it requires only the physically plausible assumption that real bases are never perfectly orthogonal, and it produces the essay's predicted behavior (repeated novelty accumulates, enabling detection). But it's an addition to the framework, not a consequence of the existing math.

---

## IV. What GPT-2 teaches about the framework

### Capture can be functional

GPT-2's final layer operates at PR ≈ 2 (out of 768). This is extreme recurrent capture — only 2 effective dimensions carry information. But it's not pathological. The model *uses* capture as a compression mechanism: the final layer funnels the rich mid-layer representations (PR ~12–30) into a 2D bottleneck before the output projection.

The framework's pathologies (capture, dissipation, fracture) are only pathologies relative to a system that needs rich phenomenality. A system that needs to produce discrete outputs (tokens, actions, signals) benefits from capture at the output layer. The framework should distinguish between *global* capture (the whole system is frozen) and *local* capture (specific layers or subsystems compress for output).

### The operating regime has a spatial structure in deep networks

In GPT-2, the operating regime doesn't live at a single point — it's distributed across layers:
- Layer 1: high-torque reception (37%)
- Layers 2–9: operating regime (89% resonance, 10% torque)
- Layers 10–12: compression toward output

This maps onto the essay's cascade (Section 5) more precisely than a single system would. The cascade isn't just across evolutionary levels — it's across processing depth within a single system.

### Self-torque is a mid-layer phenomenon

Angular displacement across positions increases with delay at layers 0 and 6 (self-torque is visible) but is flat at layer 12 (self-torque invisible). The final captured layer cannot distinguish near from far context geometrically — all positions project onto the same 2D subspace.

This means the framework's self-torque mechanism (Section 4) operates in the rich representational layers, not in the compressed output layers. Any intervention based on self-torque should target mid-layers.

---

## V. What the ME layer teaches

### The framework's operations can be implemented as a neural module

The `MemoryEngineLayer` (4,614 parameters) implements Hadamard reception, regime classification, tape update with renormalization, and causal recurrence. It:
- Preserves tensor shapes (forward pass works)
- Supports gradient flow (backprop works through tape updates)
- Integrates with a pretrained model without breaking it
- Shifts the final layer PR from 2 to 10 when untrained (adding structured noise)

### But it can't improve a pretrained model at this scale

Training the ME layers (5 epochs, frozen GPT-2) produced:
- PPL: 31.82 → 31.83 (flat)
- Parameters moved directionally: eta increased (0.10 → 0.12–0.16), alpha increased (0.50 → 0.53–0.56), torque rotation widened (0.01 → 0.03–0.04 std)
- Final layer PR dropped back to 1.9 (the untrained expansion collapsed)

**Why it doesn't work:** The ME layer is a recurrent mechanism (tape updates position by position) grafted onto a parallel architecture (transformer processes all positions simultaneously). The structural mismatch limits what 4,614 parameters can accomplish against 124M frozen parameters optimized for a different computational regime.

**The directional parameter shifts are encouraging.** The model wants faster tape updates, stronger torque rotation, and more tape influence. It's trying to use the ME machinery. But wanting isn't having — the gradient signal has to flow through hundreds of sequential tape updates, and the evaluation metric (perplexity on the frozen model's text distribution) can't be shifted by such a small intervention.

### The productive path is from scratch, not add-on

The framework's principles should be tested as a standalone computational architecture, not as an appendage to a pretrained transformer. A small model (~100K parameters) built entirely around Hadamard reception + renormalization + recurrence, trained on a simple task (copy, associative recall, sequence prediction), would test whether the framework's operations constitute a viable computational substrate on their own terms.

---

## VI. Corrections to the essay

### Minor corrections

1. **Resonance rotates toward the real axis** (Section 2, "Resonance scales"). Add: "Resonance scales magnitude while shifting phase toward zero. The dimension does not rotate freely, but it does rotate toward the real axis. The claim that resonance produces no phase change holds only when s_i is already real."

2. **Dissipation requires varying-direction torque** (Section 10, "Dissipation"). Add: "Torque from a fixed direction (always offset by +π/2) creates a systematic bias that still concentrates the tape. True dissipation — the failure of any dimension to stabilize — requires opposition from varying random directions, not merely the absence of resonance."

3. **Operating regime requires environmental variation** (Section 10, "The operating regime"). Clarify: "The operating regime is a property of the system-world coupling, not of the system alone. A system in a completely predictable world cannot maintain the operating regime regardless of its internal dynamics. The world must provide both regularity (for resonance) and surprise (for torque)."

### Substantive addition

4. **Novelty detection requires imperfect basis orthogonality** (Section 1, discussion of novelty; Section 3, consolidation). The framework's Hadamard product is dimensionally isolated — novelty at uncarved dimensions produces no effect at received dimensions. The indirect detection pathway requires an additional coupling mechanism. The most natural candidate is imperfect basis orthogonality: real physical systems have small off-diagonal coupling between basis vectors, allowing novel structure to "leak" into received dimensions. This leakage, combined with recurrence, enables the consolidation process to detect and eventually seed new axes. Without such coupling, novelty is invisible until some external mechanism (developmental program, random basis expansion) introduces a new dimension.

### Empirical observations

5. **Capture can be functional** (new subsection in Section 10). GPT-2's final layer operates at PR ≈ 2 (extreme capture), but this serves a functional purpose: compressing rich mid-layer representations into a low-dimensional bottleneck for output. The framework should distinguish between global capture (pathological freezing of the entire system) and local capture (strategic compression at specific processing stages).

6. **The cascade has spatial structure in deep networks** (Section 5). In layered systems like neural networks, the cascade manifests across processing depth, not just across evolutionary levels. GPT-2 shows: high-torque reception at early layers, operating regime at mid-layers, compression at late layers — the essay's cascade in spatial, not temporal, form.

---

## VII. The standalone model: from-scratch architecture

### The productive path is from scratch, not add-on

The ME layer's failure to improve GPT-2 (Section V) left an open question: is the problem scale or structure? A standalone model built entirely around Hadamard reception + renormalization + recurrence would test whether the framework's operations constitute a viable computational substrate.

### Real-valued baseline

`standalone_me.py` implements the simplest possible architecture: token embedding -> N ME layers -> linear readout. No attention, no FFN. Each ME layer maintains a real-valued tape on the unit hypersphere, updated via Hadamard reception with the input at each position.

Results on three tasks:

| Task | Accuracy | Notes |
|---|---|---|
| Copy (vocab=8, seq=8) | 62.5% | Partial -- learns some position-token mappings |
| Sequence prediction (vocab=4, seq=12) | 100% | Trivial repeating pattern |
| Associative recall (vocab=16, n_pairs=3) | 32.5% | Near random for the lookup component |

The model can learn simple regularities but fails on tasks requiring binding (associating specific keys with specific values). This is expected: the Hadamard product is dimensionally isolated -- what happens at dimension i depends only on what happens at dimension i. There is no mechanism for cross-dimensional binding.

### Fast binding attempt

`standalone_me_binding.py` adds the essay's fast binding mechanism (Section 2.3.1): co-resonance scoring between pairs of strongly-activated dimensions, seeding transient conjunctive dimensions when B_ij exceeds threshold.

**First attempt: binding never fired.** Two compounding bugs:

1. **Real-valued state kills phase structure.** The tape and inputs were real-valued tensors, but the framework operates in C^n. With real numbers, torch.angle() only returns 0 or pi -- the continuous phase structure that makes co-resonance scoring meaningful was absent. The framework's entire physics requires complex-valued state.

2. **Threshold unreachable at operating scale.** On the unit hypersphere in R^n, each |s_i| ~ 1/sqrt(n). The co-resonance score B_ij = |c_i| * |c_j| * cos(delta_phi) scales as O(1/n). For n=16, B_ij ~ 0.06. The fixed threshold theta_bind = 0.3 was ~5x above achievable values. The threshold must scale with dimension.

**Fix: complex tape + adaptive threshold.** The tape was made genuinely complex (separate real/imaginary parameters). The binding threshold was made adaptive: bind the top 15% of positive-scoring pairs per step rather than using a fixed absolute cutoff. This guarantees binding fires for the most co-resonant pairs regardless of absolute scale.

With the fix, binding activates (21-63 new bindings, 19-76 refreshed per task) but **performance degrades**:

| Task | No binding | With binding | Change |
|---|---|---|---|
| Copy | 62.5% | 60.4% | -2.1pp |
| Sequence prediction | 100% | 100% | flat |
| Associative recall | 32.5% | 25.0% | -7.5pp |

### Why binding hurts

The binding mechanism detects co-resonance but the transient information never reaches the readout in a usable form. Transients are stored as Python lists of (dimension_i, dimension_j, magnitude, lifetime_counter) tuples. They perturb the parent dimensions via a fixed scalar addition (0.1 * magnitude). The linear readout layer sees only |s_i| per dimension -- it has no way to recover which dimensions were co-active, what was bound, or when.

This is an architectural mismatch, not a parameter tuning problem. The binding detector works. The information channel from binding to output is broken. Fixing this would require either:
- Making transient state directly readable (augmenting the output tensor with binding information)
- A nonlinear readout that can detect the signature of transient perturbation
- A fundamentally different architecture where binding modifies the computation rather than the state

### What this tells us about the framework

The framework's operations are descriptively powerful (they map onto real neural dynamics -- Sections III-IV) but not yet constructively useful as a computational architecture. The gap between "describes what networks do" and "builds networks that work" is real. The simulation confirms the framework's dynamics; the standalone model reveals that those dynamics alone don't solve standard ML tasks without additional architectural machinery.

This is consistent with the framework's own claims: the essay describes phenomenality, not task performance. A system in the operating regime has rich internal dynamics but there's no guarantee those dynamics compute anything useful externally. The operating regime may be necessary for certain kinds of processing without being sufficient.

---

## VIII. Open questions

1. **What is the right coupling mechanism for novelty detection?** Leakage works in simulation but needs physical grounding. Is imperfect basis orthogonality the right story, or should the framework use prediction error at the anticipatory operator level?

2. **How does the framework relate to attention?** Partially resolved (see ATTENTION_FINDINGS.md). The mapping is: attention = projection + resonance (Q*K^T = projection onto key-defined basis, attention weights = resonance with existing structure). MLP = torque (the only source of reorientation in the transformer). LayerNorm = renormalization. Residual connection = tape accumulation. Key finding: attention has zero torque between consecutive layers -- it only reinforces or captures, never redirects. The MLP provides the torque that maintains the operating regime. Standard attention also lacks the framework's Hadamard depth modulation (per-dimension scaling by accumulated history).

3. **Can the anticipatory operator be grounded in tape trajectory?** The essay defines A(s) as extrapolating the tape's own dynamics. Our simulation showed this doesn't work well -- the tape is always being updated by the prediction error itself, creating a feedback loop. The working predictor tracked the world input history instead. Is there a way to make the tape-based formulation work, or should the framework use input-history prediction?

4. **What happens at higher dimensionality?** All simulations used n <= 32, and GPT-2 has n = 768. Does the operating regime become easier or harder to sustain as n grows? The renormalization coupling between dimensions weakens as n increases (deepening one dimension distributes the thinning across more others, each losing less). This might make the operating regime more robust at high n, but might also make capture harder to escape.

5. **What tasks would the framework's operations be constructive for?** The framework predicts regime maintenance, novelty detection, and hierarchical abstraction -- not sequence copying or associative recall. Testing on tasks that specifically require operating regime maintenance (e.g., continual learning without catastrophic forgetting, open-ended sensemaking) might show the framework's operations in a more favorable light.

---

## File index

| File | Purpose |
|---|---|
| `engine.py` | Core Memory Engine simulation (complex-valued, with leakage) |
| `test_tier1.py` – `test_tier5.py` | 14 simulation tests |
| `RESULTS.md` | Simulation findings |
| `llm_instrument.py` | GPT-2 instrumentation (PR, regime, self-torque) |
| `run_diagnostics.py` | LLM diagnostic experiments A–D |
| `LLM_RESULTS.md` | LLM findings |
| `me_layer.py` | MemoryEngineLayer + GPT2WithMemoryEngine |
| `test_me_layer.py` | ME layer tests |
| `run_me_diagnostics.py` | Vanilla vs ME comparison |
| `run_me_training.py` | ME layer fine-tuning attempt |
| `standalone_me.py` | Standalone ME model (real-valued, no binding) |
| `standalone_me_binding.py` | Standalone ME model with fast binding (complex tape) |
| `attention_mapping.py` | Attention-framework mapping analysis on GPT-2 |
| `ATTENTION_FINDINGS.md` | Attention mapping results |
| `FINDINGS.md` | This document |
