"""
Synthetic long-context recall benchmark for decoder-only language models.

The task is deliberately simple and structured:

1. Present a short codebook mapping names to option letters.
2. Insert a long block of irrelevant filler text.
3. Ask which option belongs to one queried name.

Evaluation is done by candidate scoring under chunked decoding. This makes the
benchmark sensitive to persistent cross-window state while remaining deterministic
and cheap to run.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, List

import torch


@dataclass
class RecallExample:
    prompt: str
    query_name: str
    correct_option: str
    options: List[str]


NAMES = [
    "Aria",
    "Bram",
    "Cleo",
    "Dax",
    "Edda",
    "Finn",
    "Galen",
    "Hana",
    "Ivo",
    "Juno",
]

OPTION_LABELS = ["A", "B", "C", "D", "E", "F"]
FILLER_SENTENCE = "The archive note is routine and does not change the codebook. "


def build_recall_examples(
    num_samples: int = 32,
    num_pairs: int = 4,
    filler_repeats: int = 48,
    seed: int = 0,
) -> List[RecallExample]:
    rng = random.Random(seed)
    if num_pairs > len(OPTION_LABELS):
        raise ValueError(f"num_pairs={num_pairs} exceeds available option labels ({len(OPTION_LABELS)}).")
    if num_pairs > len(NAMES):
        raise ValueError(f"num_pairs={num_pairs} exceeds available names ({len(NAMES)}).")

    examples: List[RecallExample] = []
    options = OPTION_LABELS[:num_pairs]

    for _ in range(num_samples):
        names = rng.sample(NAMES, num_pairs)
        labels = options.copy()
        rng.shuffle(labels)
        mapping = list(zip(names, labels))
        query_name, correct_option = rng.choice(mapping)

        header = "You are given a small codebook.\n"
        codebook = "".join(f"{name} -> {label}.\n" for name, label in mapping)
        filler = FILLER_SENTENCE * filler_repeats
        question = (
            f"\nQuestion: Which option belongs to {query_name}?\n"
            f"Options: {', '.join(options)}.\n"
            "Answer:"
        )

        examples.append(
            RecallExample(
                prompt=header + codebook + filler + question,
                query_name=query_name,
                correct_option=correct_option,
                options=options.copy(),
            )
        )

    return examples


def _extract_logits(outputs):
    if isinstance(outputs, dict):
        return outputs["logits"]
    return outputs.logits


def _run_window(model, input_ids: torch.Tensor, reset_memory: bool):
    if hasattr(model, "reset_memory"):
        return model(input_ids, reset_memory=reset_memory)
    return model(input_ids=input_ids, use_cache=False)


def score_candidate_completion(
    model,
    tokenizer,
    prompt: str,
    candidate: str,
    block_size: int,
    device: torch.device,
    carry_memory_across_windows: bool,
) -> float:
    """
    Score a candidate completion under chunked decoding.

    The score is mean token log-probability of the candidate conditioned on the
    prompt plus any candidate prefix that already fits in previous windows.
    """
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    candidate_ids = tokenizer.encode(" " + candidate, add_special_tokens=False)
    full = prompt_ids + candidate_ids
    if len(full) < 2:
        return float("-inf")

    total_logprob = 0.0
    total_tokens = 0
    reset_flag = True

    with torch.no_grad():
        for start in range(0, len(full) - 1, block_size):
            input_slice = full[start : start + block_size]
            target_slice = full[start + 1 : start + block_size + 1]
            if not target_slice:
                continue

            input_ids = torch.tensor([input_slice], dtype=torch.long, device=device)
            outputs = _run_window(model, input_ids, reset_memory=reset_flag)
            logits = _extract_logits(outputs)
            log_probs = logits.log_softmax(dim=-1)

            for local_pos, target_id in enumerate(target_slice):
                global_target_idx = start + local_pos + 1
                if global_target_idx < len(prompt_ids):
                    continue
                total_logprob += float(log_probs[0, local_pos, target_id].item())
                total_tokens += 1

            reset_flag = not carry_memory_across_windows

    if total_tokens == 0:
        return float("-inf")
    return total_logprob / total_tokens


def evaluate_long_context_recall(
    model,
    tokenizer,
    device: torch.device,
    block_size: int = 64,
    num_samples: int = 32,
    num_pairs: int = 4,
    filler_repeats: int = 48,
    carry_memory_across_windows: bool = True,
    seed: int = 0,
) -> Dict[str, object]:
    """
    Run the multiple-choice recall benchmark and return aggregate metrics.
    """
    examples = build_recall_examples(
        num_samples=num_samples,
        num_pairs=num_pairs,
        filler_repeats=filler_repeats,
        seed=seed,
    )

    correct = 0
    margins: List[float] = []
    per_example: List[Dict[str, object]] = []

    for example in examples:
        if hasattr(model, "reset_memory"):
            model.reset_memory()

        candidate_scores = {
            option: score_candidate_completion(
                model=model,
                tokenizer=tokenizer,
                prompt=example.prompt,
                candidate=option,
                block_size=block_size,
                device=device,
                carry_memory_across_windows=carry_memory_across_windows,
            )
            for option in example.options
        }

        ranked = sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)
        predicted_option, _ = ranked[0]
        correct_score = candidate_scores[example.correct_option]
        second_score = ranked[1][1] if len(ranked) > 1 else predicted_score
        margin = correct_score - second_score

        if predicted_option == example.correct_option:
            correct += 1
        margins.append(margin)
        per_example.append(
            {
                "query_name": example.query_name,
                "correct_option": example.correct_option,
                "predicted_option": predicted_option,
                "correct": predicted_option == example.correct_option,
                "scores": candidate_scores,
                "margin": margin,
                "prompt_tokens": len(tokenizer.encode(example.prompt, add_special_tokens=False)),
            }
        )

    accuracy = correct / max(len(examples), 1)
    mean_margin = sum(margins) / max(len(margins), 1)
    return {
        "accuracy": accuracy,
        "mean_margin": mean_margin,
        "num_samples": len(examples),
        "num_pairs": num_pairs,
        "filler_repeats": filler_repeats,
        "per_example": per_example,
    }
