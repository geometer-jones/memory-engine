"""Run diagnostic experiments on GPT-2 using Memory Engine framework.

Experiments:
  A: Recurrent capture — repetitive vs varied input, PR trajectory
  B: Self-torque — angular displacement across context positions
  C: Anisotropy vs generation quality — PR/anisotropy during autoregressive generation
  D: Layer-by-layer regime profile — resonance/torque/orthogonality by layer
"""

import numpy as np
import torch
from .llm_instrument import (
    load_model,
    instrument_forward,
    compute_self_torque_matrix,
    compute_pr,
    compute_anisotropy,
    angular_displacement,
)

np.set_printoptions(precision=4, suppress=True)
RESULTS_DIR = "results"


# ── Experiment A: Recurrent Capture ──────────────────────────────────────

def experiment_a(model, tokenizer):
    """Repetitive input → PR should decrease (capture). Varied input → PR stays higher."""
    sentence = "The cat sat on the mat. "
    n_repeats = 20
    repetitive_text = sentence * n_repeats

    # Varied text: similar length, diverse content
    varied_text = (
        "The cat sat on the mat. "
        "Quantum mechanics describes nature at the smallest scales. "
        "Jazz musicians improvise over complex chord progressions. "
        "The ancient Romans built aqueducts across Europe. "
        "Machine learning models learn patterns from data. "
        "Volcanic eruptions reshape landscapes dramatically. "
        "Abstract paintings challenge conventional perception. "
        "Neural networks process information through layered transformations. "
        "The Amazon rainforest hosts extraordinary biodiversity. "
        "Chess grandmasters think many moves ahead. "
        "Glaciers carve valleys over millennia. "
        "Programming languages express computational logic. "
        "The Pacific Ocean covers a third of Earth's surface. "
        "Bacteria evolve resistance to antibiotics rapidly. "
        "Renaissance art emphasized humanism and perspective. "
        "Satellites orbit Earth at various altitudes. "
        "Philosophers debate the nature of consciousness. "
        "Electric currents generate magnetic fields. "
        "The immune system protects against pathogens. "
        "Poetry condenses meaning into precise language. "
    )

    print("Experiment A: Recurrent Capture Detection")
    print("=" * 60)

    rep_result = instrument_forward(model, tokenizer, repetitive_text)
    var_result = instrument_forward(model, tokenizer, varied_text)

    # PR at final layer across positions
    rep_pr = rep_result["pr"][-1]
    var_pr = var_result["pr"][-1]
    rep_gini = rep_result["anisotropy"][-1]
    var_gini = var_result["anisotropy"][-1]

    # Analyze by segment (each sentence = 6-7 tokens)
    seg_len = len(tokenizer.encode(sentence)) - 1  # minus BOS
    n_segs = len(rep_pr) // seg_len

    rep_seg_pr = [np.mean(rep_pr[i * seg_len:(i + 1) * seg_len]) for i in range(n_segs)]
    var_seg_pr = [np.mean(var_pr[i * seg_len:(i + 1) * seg_len]) for i in range(min(n_segs, len(var_pr) // seg_len))]

    print(f"\nRepetitive input ({n_repeats}x same sentence):")
    print(f"  PR: first seg={rep_seg_pr[0]:.2f}, last seg={rep_seg_pr[-1]:.2f}, "
          f"trend={'decreasing' if rep_seg_pr[-1] < rep_seg_pr[0] else 'stable/increasing'}")
    print(f"  Gini: first={rep_gini[:seg_len].mean():.3f}, last={rep_gini[-seg_len:].mean():.3f}")

    print(f"\nVaried input ({n_repeats}x different sentences):")
    print(f"  PR: first seg={var_seg_pr[0]:.2f}, last seg={var_seg_pr[-1]:.2f}, "
          f"trend={'decreasing' if var_seg_pr[-1] < var_seg_pr[0] else 'stable/increasing'}")
    print(f"  Gini: first={var_gini[:seg_len].mean():.3f}, last={var_gini[-seg_len:].mean():.3f}")

    # PR across all layers at final position
    print(f"\n  PR across layers (final position):")
    print(f"  {'Layer':>5} {'Repetitive':>11} {'Varied':>11}")
    for layer in [0, 3, 6, 9, 12]:
        if layer < len(rep_result["pr"]):
            print(f"  {layer:>5} {rep_result['pr'][layer, -1]:>11.2f} {var_result['pr'][layer, -1]:>11.2f}")

    # Save results
    np.savez(
        f"{RESULTS_DIR}/exp_a.npz",
        rep_pr=rep_result["pr"],
        var_pr=var_result["pr"],
        rep_anisotropy=rep_result["anisotropy"],
        var_anisotropy=var_result["anisotropy"],
        rep_seg_pr=np.array(rep_seg_pr),
        var_seg_pr=np.array(var_seg_pr),
    )

    print("  Results saved to results/exp_a.npz")
    return rep_result, var_result


# ── Experiment B: Self-Torque Across Context ─────────────────────────────

def experiment_b(model, tokenizer):
    """Angular displacement increases with position delay (thick recurrence)."""
    text = (
        "The development of artificial intelligence has been one of the most "
        "significant technological achievements of the twenty-first century. "
        "From early expert systems to modern deep learning architectures, "
        "the field has undergone remarkable transformations. Neural networks, "
        "inspired by biological computation, now power everything from image "
        "recognition to natural language processing. The attention mechanism, "
        "introduced in the transformer architecture, revolutionized how models "
        "process sequential data. Large language models demonstrate emergent "
        "capabilities that were not explicitly programmed, raising questions "
        "about the nature of understanding and intelligence."
    )

    print("\nExperiment B: Self-Torque Across Context")
    print("=" * 60)

    result = instrument_forward(model, tokenizer, text)
    delays = [1, 2, 5, 10, 20, 50]

    print(f"\n  Text length: {result['seq_len']} tokens")

    for layer_idx in [0, 6, 12]:
        torque_matrix = compute_self_torque_matrix(
            result["hidden_states"], layer_idx, delays
        )
        mean_disp = [np.nanmean(torque_matrix[:, j]) for j in range(len(delays))]

        print(f"\n  Layer {layer_idx}:")
        print(f"  {'Delay':>5} {'Mean angular disp':>18}")
        for d, disp in zip(delays, mean_disp):
            print(f"  {d:>5} {disp:>18.4f}")

    # Save: self-torque at final layer
    torque_final = compute_self_torque_matrix(
        result["hidden_states"], 12, delays
    )
    np.savez(
        f"{RESULTS_DIR}/exp_b.npz",
        torque_matrix=torque_final,
        delays=np.array(delays),
        hidden_states_last_layer=result["hidden_states"][12],
        tokens=result["tokens"],
    )

    print("  Results saved to results/exp_b.npz")
    return result


# ── Experiment C: Anisotropy vs Generation Quality ────────────────────────

def experiment_c(model, tokenizer):
    """Generate text with bland vs surprising prompts. Track PR and repetition."""
    print("\nExperiment C: Anisotropy vs Generation Quality")
    print("=" * 60)

    prompts = {
        "bland": "The weather is nice today. The sky is blue.",
        "surprising": "The archaeologist discovered that the ancient clock was running backward,",
    }

    gen_length = 100
    results = {}

    for label, prompt in prompts.items():
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        generated = input_ids[0].tolist()
        pr_track = []
        gini_track = []
        entropy_track = []

        for step in range(gen_length):
            with torch.no_grad():
                outputs = model(torch.tensor([generated[-100:]]), output_hidden_states=True)
                next_logits = outputs.logits[0, -1]

            # Entropy of next-token distribution
            probs = torch.softmax(next_logits, dim=-1).numpy()
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            entropy_track.append(entropy)

            # PR and anisotropy at final layer, last position
            h = outputs.hidden_states[-1][0, -1].numpy()
            pr_track.append(compute_pr(h))
            gini_track.append(compute_anisotropy(h))

            # Sample next token
            next_token = torch.multinomial(torch.softmax(next_logits, dim=-1), 1)
            generated.append(next_token.item())

        generated_text = tokenizer.decode(generated)

        # Repetition metric: fraction of bigrams that are repeated
        bigrams = list(zip(generated, generated[1:]))
        n_repeated_bigrams = len(bigrams) - len(set(bigrams))
        repetition_rate = n_repeated_bigrams / len(bigrams)

        results[label] = {
            "pr": np.array(pr_track),
            "gini": np.array(gini_track),
            "entropy": np.array(entropy_track),
            "repetition_rate": repetition_rate,
            "generated_text": generated_text,
            "generated_ids": np.array(generated),
        }

        print(f"\n  {label.upper()} prompt: '{prompt}'")
        print(f"  Generated ({gen_length} tokens):")
        gen_only = tokenizer.decode(generated[len(input_ids[0]):])
        print(f"    {gen_only[:200]}...")
        print(f"  PR: mean={np.mean(pr_track):.2f}, std={np.std(pr_track):.2f}")
        print(f"  Gini: mean={np.mean(gini_track):.3f}")
        print(f"  Entropy: mean={np.mean(entropy_track):.3f}")
        print(f"  Repetition rate: {repetition_rate:.3f}")

    np.savez(
        f"{RESULTS_DIR}/exp_c.npz",
        bland_pr=results["bland"]["pr"],
        bland_gini=results["bland"]["gini"],
        bland_entropy=results["bland"]["entropy"],
        surprising_pr=results["surprising"]["pr"],
        surprising_gini=results["surprising"]["gini"],
        surprising_entropy=results["surprising"]["entropy"],
    )

    print("  Results saved to results/exp_c.npz")
    return results


# ── Experiment D: Layer Regime Profile ────────────────────────────────────

def experiment_d(model, tokenizer):
    """Regime fraction by layer: early layers more resonance, later more torque."""
    text = (
        "Scientists at the research institute announced a breakthrough in "
        "quantum computing that could transform cryptography."
    )

    print("\nExperiment D: Layer-by-Layer Regime Profile")
    print("=" * 60)

    result = instrument_forward(model, tokenizer, text)

    print(f"\n  {'Layer':>5} {'Resonance%':>11} {'Torque%':>8} {'Orth%':>7} | {'PR':>5} {'Gini':>5}")
    print("  " + "-" * 50)

    layer_regimes = []
    for layer_idx in range(result["n_layers"] + 1):
        if layer_idx == 0:
            # Embedding layer — no previous to compare
            mean_pr = np.mean(result["pr"][0])
            mean_gini = np.mean(result["anisotropy"][0])
            print(f"  {0:>5} {'(embedding)':>11} {'':>8} {'':>7} | {mean_pr:>5.1f} {mean_gini:>5.3f}")
            continue

        regimes_at_layer = result["regime_by_layer"][layer_idx - 1]
        mean_res = np.mean([r["resonance_frac"] for r in regimes_at_layer])
        mean_tor = np.mean([r["torque_frac"] for r in regimes_at_layer])
        mean_ort = np.mean([r["orth_frac"] for r in regimes_at_layer])
        mean_pr = np.mean(result["pr"][layer_idx])
        mean_gini = np.mean(result["anisotropy"][layer_idx])

        layer_regimes.append({
            "layer": layer_idx,
            "resonance": mean_res,
            "torque": mean_tor,
            "orthogonality": mean_ort,
            "pr": mean_pr,
            "gini": mean_gini,
        })

        print(f"  {layer_idx:>5} {mean_res*100:>10.1f}% {mean_tor*100:>7.1f}% "
              f"{mean_ort*100:>6.1f}% | {mean_pr:>5.1f} {mean_gini:>5.3f}")

    np.savez(
        f"{RESULTS_DIR}/exp_d.npz",
        layer_regimes=np.array([(r["resonance"], r["torque"], r["orthogonality"],
                                  r["pr"], r["gini"]) for r in layer_regimes]),
    )

    print("  Results saved to results/exp_d.npz")
    return result


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading GPT-2...\n")
    model, tokenizer = load_model()

    rep_result, var_result = experiment_a(model, tokenizer)
    torque_result = experiment_b(model, tokenizer)
    gen_results = experiment_c(model, tokenizer)
    regime_result = experiment_d(model, tokenizer)

    print("\n" + "=" * 60)
    print("All experiments completed. Results in results/")
    print("Run visualize.py to generate plots.")
