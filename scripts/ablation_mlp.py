"""
MLP Ablation Experiment: Test the framework's prediction.

Prediction: If MLP = torque and attention = resonance, then removing MLP
should cause attention to collapse into capture faster (no reorientation force).

Method: Compare attention patterns in normal GPT-2 vs GPT-2 with MLP layers
zeroed out. Measure entropy, capture fraction, and regime classification.
"""

import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from .attention_mapping import (
    attention_entropy, attention_pr, attention_regime,
    head_specialization, instrument_attention, print_report
)

np.set_printoptions(precision=4, suppress=True)


def create_mlp_ablated_model(model_name="gpt2"):
    """Create a copy of GPT-2 with MLP layers zeroed out."""
    config = GPT2Config.from_pretrained(model_name)
    config.output_attentions = True
    model = GPT2LMHeadModel.from_pretrained(model_name, config=config)
    model.eval()

    # Zero out MLP weights in each transformer block
    for block in model.transformer.h:
        # GPT-2 MLP: c_fc (768 -> 3072) + c_proj (3072 -> 768)
        # Zero both to remove all MLP computation
        nn.init.zeros_(block.mlp.c_fc.weight)
        nn.init.zeros_(block.mlp.c_fc.bias)
        nn.init.zeros_(block.mlp.c_proj.weight)
        nn.init.zeros_(block.mlp.c_proj.bias)

    return model


def compare_models(normal_results, ablated_results, text):
    """Compare normal vs MLP-ablated attention patterns."""
    print("=" * 70)
    print(f"MLP ABLATION: {text[:60]}...")
    print("=" * 70)

    n_layers = normal_results["n_layers"]

    print(f"\n{'Layer':>5} {'Normal Ent':>11} {'Ablated Ent':>12} {'Delta':>7} | "
          f"{'Normal Cap':>11} {'Ablated Cap':>12} {'Delta':>7}")
    print("-" * 75)

    normal_ents = []
    ablated_ents = []
    normal_caps = []
    ablated_caps = []

    for layer in range(n_layers):
        n_ent = normal_results["entropy"][layer].mean()
        a_ent = ablated_results["entropy"][layer].mean()
        n_cap = normal_results["capture_frac"][layer]
        a_cap = ablated_results["capture_frac"][layer]

        normal_ents.append(n_ent)
        ablated_ents.append(a_ent)
        normal_caps.append(n_cap)
        ablated_caps.append(a_cap)

        ent_delta = a_ent - n_ent
        cap_delta = a_cap - n_cap

        marker = ""
        if abs(ent_delta) > 0.05:
            marker = " ***" if abs(ent_delta) > 0.15 else " *"

        print(f"  {layer+1:>3} {n_ent:>11.3f} {a_ent:>12.3f} {ent_delta:>+7.3f} | "
              f"{n_cap:>11.3f} {a_cap:>12.3f} {cap_delta:>+7.3f}{marker}")

    print()
    print("Summary:")
    print(f"  Normal  - Early entropy: {np.mean(normal_ents[:3]):.3f}, "
          f"Late entropy: {np.mean(normal_ents[-3:]):.3f}, "
          f"Late capture: {np.mean(normal_caps[-3:]):.3f}")
    print(f"  Ablated - Early entropy: {np.mean(ablated_ents[:3]):.3f}, "
          f"Late entropy: {np.mean(ablated_ents[-3:]):.3f}, "
          f"Late capture: {np.mean(ablated_caps[-3:]):.3f}")

    # Entropy drop rate (how fast attention narrows)
    normal_drop = normal_ents[0] - normal_ents[-1]
    ablated_drop = ablated_ents[0] - ablated_ents[-1]
    print(f"\n  Entropy drop (layer 1 -> 12):")
    print(f"    Normal:  {normal_drop:+.3f}")
    print(f"    Ablated: {ablated_drop:+.3f}")

    # Capture rate
    normal_cap_rise = normal_caps[-1] - normal_caps[0]
    ablated_cap_rise = ablated_caps[-1] - ablated_caps[0]
    print(f"\n  Capture rise (layer 1 -> 12):")
    print(f"    Normal:  {normal_cap_rise:+.3f}")
    print(f"    Ablated: {ablated_cap_rise:+.3f}")

    # Test prediction
    print(f"\n  Prediction check:")
    if ablated_drop > normal_drop:
        print(f"    CONFIRMED: Ablated entropy drops faster ({ablated_drop:.3f} vs {normal_drop:.3f})")
    else:
        print(f"    NOT confirmed: Ablated entropy does NOT drop faster ({ablated_drop:.3f} vs {normal_drop:.3f})")

    if ablated_cap_rise > normal_cap_rise:
        print(f"    CONFIRMED: Ablated capture rises faster ({ablated_cap_rise:.3f} vs {normal_cap_rise:.3f})")
    else:
        print(f"    NOT confirmed: Ablated capture does NOT rise faster ({ablated_cap_rise:.3f} vs {normal_cap_rise:.3f})")

    # Perplexity comparison
    return {
        "normal_ents": normal_ents,
        "ablated_ents": ablated_ents,
        "normal_caps": normal_caps,
        "ablated_caps": ablated_caps,
    }


def compute_perplexity(model, tokenizer, text):
    """Compute perplexity of model on given text."""
    input_ids = tokenizer.encode(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    return torch.exp(outputs.loss).item()


if __name__ == "__main__":
    print("Loading models...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    normal_model = GPT2LMHeadModel.from_pretrained(
        "gpt2", config=GPT2Config.from_pretrained("gpt2", output_attentions=True)
    )
    normal_model.eval()

    ablated_model = create_mlp_ablated_model()

    # Use longer text for more stable measurements
    texts = [
        "The quick brown fox jumps over the lazy dog. The cat sat on the mat and watched.",
        "In the beginning God created the heaven and the earth. And the earth was without form.",
        "To be or not to be that is the question whether it is nobler in the mind to suffer",
    ]

    all_results = []
    for text in texts:
        print(f"\nAnalyzing: {text[:60]}...")

        normal_results = instrument_attention(normal_model, tokenizer, text)
        ablated_results = instrument_attention(ablated_model, tokenizer, text)

        comparison = compare_models(normal_results, ablated_results, text)
        all_results.append(comparison)

    # Aggregate
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS ACROSS ALL TEXTS")
    print("=" * 70)

    for text_idx, comp in enumerate(all_results):
        n_ent = comp["normal_ents"]
        a_ent = comp["ablated_ents"]
        n_cap = comp["normal_caps"]
        a_cap = comp["ablated_caps"]

        print(f"\nText {text_idx+1}:")
        print(f"  Normal  entropy drop: {n_ent[0]-n_ent[-1]:+.3f}, capture rise: {n_cap[-1]-n_cap[0]:+.3f}")
        print(f"  Ablated entropy drop: {a_ent[0]-a_ent[-1]:+.3f}, capture rise: {a_cap[-1]-a_cap[0]:+.3f}")

    # Perplexity sanity check
    print("\n--- Perplexity Sanity Check ---")
    test_text = "The cat sat on the mat and the dog ran in the park."
    n_ppl = compute_perplexity(normal_model, tokenizer, test_text)
    a_ppl = compute_perplexity(ablated_model, tokenizer, test_text)
    print(f"  Normal PPL:  {n_ppl:.1f}")
    print(f"  Ablated PPL: {a_ppl:.1f}")
    print(f"  (Ablated should be much higher — MLP removal degrades language modeling)")
