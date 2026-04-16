"""
Small-model evaluation harness for the production Memory Engine layer.

Purpose:
- compare baseline vs memory-augmented perplexity on a lightweight local corpus
- inspect Memory Engine diagnostics on an inexpensive Hugging Face checkpoint
- provide a reproducible stepping stone before moving to 7B/8B models

Recommended first run:

    python3 evaluate_small_model_memory.py --model-name sshleifer/tiny-gpt2
"""

from __future__ import annotations

import argparse
from statistics import mean
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from long_context_recall_benchmark import evaluate_long_context_recall
from me_layer import create_model


def build_eval_texts() -> List[str]:
    return [
        (
            "Transformers process tokens in parallel, but recurrence-like state can still emerge "
            "when a module accumulates structured information across positions."
        ),
        (
            "Long-context reasoning depends on preserving motifs over many steps while still "
            "adapting to surprise when the sequence changes regime."
        ),
        (
            "A useful memory layer should help retain stable structure, bind co-activated features, "
            "and detect novelty without collapsing into a single rigid axis."
        ),
        (
            "Small language models are useful for rapid iteration because they expose whether a new "
            "mechanism is numerically stable before expensive scaling experiments."
        ),
    ]


def _extract_loss(outputs) -> Optional[torch.Tensor]:
    if isinstance(outputs, dict):
        return outputs.get("loss")
    return getattr(outputs, "loss", None)


def _evaluate_perplexity(
    model,
    tokenizer,
    texts: List[str],
    block_size: int,
    device: torch.device,
    carry_memory_across_windows: bool,
) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            tokens = tokenizer.encode(text)
            if len(tokens) < 2:
                continue

            doc_reset = True
            for start in range(0, len(tokens) - 1, block_size):
                window = tokens[start : start + block_size]
                if len(window) < 2:
                    continue

                input_ids = torch.tensor([window[:-1]], dtype=torch.long, device=device)
                labels = torch.tensor([window[1:]], dtype=torch.long, device=device)

                if hasattr(model, "reset_memory"):
                    outputs = model(
                        input_ids,
                        labels=labels,
                        reset_memory=doc_reset,
                    )
                    doc_reset = not carry_memory_across_windows
                else:
                    outputs = model(input_ids=input_ids, labels=labels, use_cache=False)

                loss = _extract_loss(outputs)
                if loss is None:
                    continue
                total_loss += float(loss.item()) * labels.numel()
                total_tokens += labels.numel()

    if total_tokens == 0:
        return float("inf")
    return float(np.exp(total_loss / total_tokens))


def _sample_generation(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        if hasattr(model, "generate"):
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        else:
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
    return tokenizer.decode(generated[0], skip_special_tokens=True)


def _summarize_memory_diagnostics(model, tokenizer, texts: List[str], device: torch.device) -> List[Dict[str, float]]:
    if not hasattr(model, "reset_memory"):
        return []

    summaries: List[Dict[str, float]] = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt").to(device)
            outputs = model(**inputs, reset_memory=True)
            for layer_diag in outputs.get("me_diagnostics", []):
                diag = layer_diag["diagnostics"]
                if not diag:
                    continue
                summaries.append(
                    {
                        "layer_index": float(layer_diag["layer_index"]),
                        "pr": float(diag["pr"][0, -1].item()),
                        "gini": float(diag["gini"][0, -1].item()),
                        "resonance_fraction": float(diag["resonance_fraction"][0, -1].item()),
                        "torque_fraction": float(diag["torque_fraction"][0, -1].item()),
                        "self_torque": float(diag["self_torque"][0, -1].item()),
                        "coupling_strength": float(diag["coupling_strength"][0, -1].item()),
                        "prediction_error": float(diag["prediction_error"][0, -1].item()),
                    }
                )
    return summaries


def _group_diagnostics(rows: List[Dict[str, float]]) -> List[Dict[str, float]]:
    if not rows:
        return []

    grouped: Dict[int, Dict[str, List[float]]] = {}
    for row in rows:
        layer = int(row["layer_index"])
        grouped.setdefault(layer, {k: [] for k in row.keys() if k != "layer_index"})
        for key, value in row.items():
            if key == "layer_index":
                continue
            grouped[layer][key].append(value)

    summary = []
    for layer in sorted(grouped):
        layer_summary = {"layer_index": layer}
        for key, values in grouped[layer].items():
            layer_summary[key] = mean(values)
        summary.append(layer_summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a small model with and without Memory Engine.")
    parser.add_argument("--model-name", default="sshleifer/tiny-gpt2")
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--carry-memory-across-windows", action="store_true")
    parser.add_argument("--memory-dim", type=int, default=None)
    parser.add_argument("--max-aux-dims", type=int, default=16)
    parser.add_argument("--max-transient-dims", type=int, default=8)
    parser.add_argument("--insert-after", default=None, help="Comma-separated decoder layer indices.")
    parser.add_argument("--skip-recall-benchmark", action="store_true")
    parser.add_argument("--recall-samples", type=int, default=24)
    parser.add_argument("--recall-num-pairs", type=int, default=4)
    parser.add_argument("--recall-filler-repeats", type=int, default=48)
    return parser.parse_args()


def _parse_insert_after(raw: Optional[str]) -> Optional[List[int]]:
    if raw is None or raw.strip() == "":
        return None
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    texts = build_eval_texts()

    print(f"Loading baseline model: {args.model_name}")
    baseline_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if baseline_tokenizer.pad_token is None and baseline_tokenizer.eos_token is not None:
        baseline_tokenizer.pad_token = baseline_tokenizer.eos_token
    baseline_model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    baseline_model.eval()

    insert_after = _parse_insert_after(args.insert_after)
    memory_kwargs = {
        "max_aux_dims": args.max_aux_dims,
        "max_transient_dims": args.max_transient_dims,
    }
    if args.memory_dim is not None:
        memory_kwargs["memory_dim"] = args.memory_dim

    print(f"Loading memory model: {args.model_name}")
    memory_model, memory_tokenizer = create_model(
        args.model_name,
        insert_after=insert_after,
        **memory_kwargs,
    )
    memory_model.to(device)
    memory_model.eval()

    print("\nEvaluating perplexity...")
    baseline_ppl = _evaluate_perplexity(
        baseline_model,
        baseline_tokenizer,
        texts,
        block_size=args.block_size,
        device=device,
        carry_memory_across_windows=False,
    )
    memory_ppl = _evaluate_perplexity(
        memory_model,
        memory_tokenizer,
        texts,
        block_size=args.block_size,
        device=device,
        carry_memory_across_windows=args.carry_memory_across_windows,
    )

    print(f"  baseline perplexity: {baseline_ppl:.3f}")
    print(f"  memory perplexity:   {memory_ppl:.3f}")
    print(f"  delta:               {memory_ppl - baseline_ppl:+.3f}")

    if not args.skip_recall_benchmark:
        print("\nEvaluating long-context recall benchmark...")
        baseline_recall = evaluate_long_context_recall(
            baseline_model,
            baseline_tokenizer,
            device=device,
            block_size=args.block_size,
            num_samples=args.recall_samples,
            num_pairs=args.recall_num_pairs,
            filler_repeats=args.recall_filler_repeats,
            carry_memory_across_windows=False,
            seed=0,
        )
        memory_recall = evaluate_long_context_recall(
            memory_model,
            memory_tokenizer,
            device=device,
            block_size=args.block_size,
            num_samples=args.recall_samples,
            num_pairs=args.recall_num_pairs,
            filler_repeats=args.recall_filler_repeats,
            carry_memory_across_windows=args.carry_memory_across_windows,
            seed=0,
        )
        print(
            f"  baseline recall: accuracy={baseline_recall['accuracy']:.3f} "
            f"margin={baseline_recall['mean_margin']:+.3f}"
        )
        print(
            f"  memory recall:   accuracy={memory_recall['accuracy']:.3f} "
            f"margin={memory_recall['mean_margin']:+.3f}"
        )

    prompt = (
        "Persistent sequence state can help a transformer because"
    )
    print("\nGeneration sample:")
    baseline_text = _sample_generation(baseline_model, baseline_tokenizer, prompt, args.max_new_tokens, device)
    memory_text = _sample_generation(memory_model, memory_tokenizer, prompt, args.max_new_tokens, device)
    print(f"  baseline: {baseline_text}")
    print(f"  memory:   {memory_text}")

    diagnostics = _group_diagnostics(_summarize_memory_diagnostics(memory_model, memory_tokenizer, texts, device))
    if diagnostics:
        print("\nMemory diagnostics (mean over eval texts):")
        for row in diagnostics:
            print(
                "  layer={layer_index:>2d} pr={pr:.3f} gini={gini:.3f} "
                "res={resonance_fraction:.3f} torq={torque_fraction:.3f} "
                "self_torque={self_torque:.3f} coupling={coupling_strength:.3f} "
                "pred_err={prediction_error:.3f}".format(
                    layer_index=int(row["layer_index"]),
                    **row,
                )
            )


if __name__ == "__main__":
    main()
