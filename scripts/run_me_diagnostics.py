"""Compare GPT-2 with and without Memory Engine layers.

Runs the same diagnostics as Level 1 on both models and compares:
- PR trajectory across layers and positions
- Regime profile by layer
- Self-torque across context
- Generation quality (entropy, repetition)
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
from me_layer import create_model

np.set_printoptions(precision=4, suppress=True)


def compare_pr_profile():
    """PR across layers: does the ME layer change the capture profile?"""
    print("=" * 60)
    print("PR Profile: Vanilla GPT-2 vs GPT-2 + Memory Engine")
    print("=" * 60)

    text = (
        "The development of artificial intelligence has been one of the most "
        "significant technological achievements of the modern era. Neural networks "
        "learn patterns from data through iterative optimization."
    )

    # Vanilla
    vanilla_model, tokenizer = load_model("gpt2")
    vanilla_result = instrument_forward(vanilla_model, tokenizer, text)

    # With ME layers
    me_model, _ = create_model("gpt2", insert_after=[3, 6, 9])
    input_ids = tokenizer.encode(text, return_tensors="pt")
    with torch.no_grad():
        me_outputs = me_model(input_ids)
    # Collect hidden states from the ME model
    me_hs = torch.stack([
        h.squeeze(0) if h.dim() == 3 else h
        for h in me_outputs["hidden_states"]
    ]).numpy()

    print(f"\n{'Layer':>5} {'Vanilla PR':>11} {'ME PR':>8} {'Delta':>7}")
    print("-" * 40)

    n_layers = min(vanilla_result["pr"].shape[0], me_hs.shape[0])
    for layer in range(n_layers):
        v_pr = np.mean(vanilla_result["pr"][layer])
        # ME model hidden states include extra ME layer outputs
        m_pr_vals = []
        for pos in range(me_hs.shape[1]):
            h = me_hs[layer, pos]
            m_pr_vals.append(compute_pr(h))
        m_pr = np.mean(m_pr_vals)
        delta = m_pr - v_pr
        marker = " ← ME" if layer in [4, 7, 10] else ""  # after blocks 3,6,9
        print(f"  {layer:>3} {v_pr:>11.1f} {m_pr:>8.1f} {delta:>+7.1f}{marker}")

    return vanilla_result, me_hs


def compare_generation():
    """Generation quality: does ME change entropy and repetition?"""
    print("\n" + "=" * 60)
    print("Generation Quality: Vanilla vs ME")
    print("=" * 60)

    prompts = {
        "neutral": "The scientist examined the data carefully and",
        "creative": "The clock struck thirteen and the shadows began to",
    }

    for label, prompt in prompts.items():
        print(f"\n  --- {label.upper()} prompt ---")

        # Vanilla generation
        vanilla_model, tokenizer = load_model("gpt2")
        vanilla_ids = tokenizer.encode(prompt, return_tensors="pt")

        # ME generation
        me_model, _ = create_model("gpt2", insert_after=[3, 6, 9])
        me_ids = tokenizer.encode(prompt, return_tensors="pt")

        gen_len = 80

        for model_label, model, ids in [
            ("Vanilla", vanilla_model, vanilla_ids),
            ("ME", me_model, me_ids),
        ]:
            generated = ids[0].tolist()
            entropies = []

            for step in range(gen_len):
                input_tensor = torch.tensor([generated[-100:]])
                with torch.no_grad():
                    if model_label == "ME":
                        outputs = model(input_tensor)
                        logits = outputs["logits"]
                    else:
                        outputs = model(input_tensor, output_hidden_states=True)
                        logits = outputs.logits

                next_logits = logits[0, -1]
                probs = torch.softmax(next_logits, dim=-1).numpy()
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                entropies.append(entropy)

                next_token = torch.multinomial(torch.softmax(next_logits, dim=-1), 1)
                generated.append(next_token.item())

            text = tokenizer.decode(generated[len(ids[0]):])
            bigrams = list(zip(generated, generated[1:]))
            rep_rate = (len(bigrams) - len(set(bigrams))) / len(bigrams)

            print(f"  {model_label:>8}: entropy={np.mean(entropies):.3f}, "
                  f"repetition={rep_rate:.3f}")
            print(f"           {text[:120]}...")


def compare_regime_profile():
    """Regime fractions by layer: does ME shift the resonance/torque balance?"""
    print("\n" + "=" * 60)
    print("Regime Profile: Vanilla vs ME")
    print("=" * 60)

    text = "Scientists announced a breakthrough in quantum computing."
    vanilla_model, tokenizer = load_model("gpt2")
    me_model, _ = create_model("gpt2", insert_after=[3, 6, 9])

    vanilla_result = instrument_forward(vanilla_model, tokenizer, text)

    # ME model regime: compare consecutive hidden states
    input_ids = tokenizer.encode(text, return_tensors="pt")
    with torch.no_grad():
        me_outputs = me_model(input_ids)
    me_hs = [h.squeeze(0).numpy() if h.dim() == 3 else h.numpy() for h in me_outputs["hidden_states"]]

    print(f"\n{'Layer':>5} {'Vanilla Res%':>12} {'ME Res%':>8} | "
          f"{'Vanilla Tor%':>12} {'ME Tor%':>8}")
    print("-" * 60)

    for layer in range(1, min(len(vanilla_result["regime_by_layer"]) + 1, len(me_hs))):
        # Vanilla regime
        if layer <= len(vanilla_result["regime_by_layer"]):
            v_regimes = vanilla_result["regime_by_layer"][layer - 1]
            v_res = np.mean([r["resonance_frac"] for r in v_regimes])
            v_tor = np.mean([r["torque_frac"] for r in v_regimes])
        else:
            v_res, v_tor = 0, 0

        # ME regime: compare this layer to previous
        if layer < len(me_hs):
            h_curr = me_hs[layer]
            h_prev = me_hs[layer - 1]
            res_fracs = []
            tor_fracs = []
            for pos in range(h_curr.shape[0]):
                product = h_curr[pos] * h_prev[pos]
                n_total = len(product)
                n_res = np.sum(product > 1e-6)
                n_tor = np.sum(product < -1e-6)
                res_fracs.append(n_res / n_total)
                tor_fracs.append(n_tor / n_total)
            m_res = np.mean(res_fracs)
            m_tor = np.mean(tor_fracs)
        else:
            m_res, m_tor = 0, 0

        marker = " ← ME" if layer in [4, 7, 10] else ""
        print(f"  {layer:>3} {v_res*100:>11.1f}% {m_res*100:>7.1f}% | "
              f"{v_tor*100:>11.1f}% {m_tor*100:>7.1f}%{marker}")


if __name__ == "__main__":
    print("Loading models...\n")
    vanilla_result, me_hs = compare_pr_profile()
    compare_generation()
    compare_regime_profile()
    print("\nDone.")
