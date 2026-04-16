"""Phase 2 continual learning for a distilled hybrid Memory Engine student."""

from __future__ import annotations

import argparse
import math
from statistics import mean
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from distill_me_llm import (
    build_dataloaders,
    build_model_load_kwargs,
    ensure_pad_token,
    load_me_checkpoint,
    load_texts,
    prepare_causal_lm_batch,
    summarize_memory,
)
from memory_engine_llm import MemoryEngineCausalLM


def build_student_from_checkpoint(
    checkpoint: Dict[str, object],
    *,
    device: torch.device,
    student_dtype: str,
) -> MemoryEngineCausalLM:
    student_model_name = checkpoint["student_model"]
    base_model = AutoModelForCausalLM.from_pretrained(
        student_model_name,
        **build_model_load_kwargs(student_dtype, load_in_4bit=False),
    )
    memory_config = dict(checkpoint["memory_config"])
    student = MemoryEngineCausalLM(
        base_model=base_model,
        model_name=student_model_name,
        insert_after=checkpoint["insert_after"],
        freeze_base_model=True,
        **memory_config,
    ).to(device)
    student.load_me_state_dict(checkpoint["me_state_dict"], strict=False)
    return student


def run_continual_epoch(
    *,
    student_model: MemoryEngineCausalLM,
    tokenizer,
    dataloader,
    optimizer: Optional[torch.optim.Optimizer],
    sequence_length: int,
    device: torch.device,
    grad_clip: float,
) -> Dict[str, float]:
    if optimizer is None:
        student_model.eval()
    else:
        student_model.train()

    losses: List[float] = []
    memory_metrics: Dict[str, List[float]] = {}

    for texts in dataloader:
        batch = prepare_causal_lm_batch(texts, tokenizer, sequence_length, device)

        if optimizer is not None:
            optimizer.zero_grad()

        outputs = student_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            reset_memory=True,
            return_memory_features=True,
            use_cache=False,
        )
        loss = outputs["loss"]
        if loss is None:
            raise RuntimeError("Student returned no CE loss during continual learning.")

        if optimizer is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.get_me_parameters(), grad_clip)
            optimizer.step()

        losses.append(float(loss.detach().item()))
        memory_summary = summarize_memory(outputs)
        for key, value in memory_summary.items():
            memory_metrics.setdefault(key, []).append(value)

    metrics = {"ce": mean(losses), "perplexity": math.exp(mean(losses))}
    for key, values in memory_metrics.items():
        metrics[key] = mean(values)
    return metrics


def save_continual_checkpoint(
    output_path: str,
    *,
    model: MemoryEngineCausalLM,
    checkpoint: Dict[str, object],
    metrics: Dict[str, float],
) -> None:
    payload = dict(checkpoint)
    payload["me_state_dict"] = model.me_state_dict()
    payload["metrics"] = metrics
    payload["phase"] = "continual"
    torch.save(payload, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2 continual learning for the hybrid Memory Engine student.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--student-dtype", default="float32", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--text-file", default=None)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--dataset-text-key", default="text")
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--max-train-examples", type=int, default=128)
    parser.add_argument("--max-eval-examples", type=int, default=32)
    parser.add_argument("--reference-text-file", default=None)
    parser.add_argument("--reference-dataset-name", default=None)
    parser.add_argument("--reference-dataset-config", default=None)
    parser.add_argument("--reference-dataset-split", default="train")
    parser.add_argument("--reference-dataset-text-key", default="text")
    parser.add_argument("--output-path", default="continual_me_checkpoint.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = load_me_checkpoint(args.checkpoint)
    student_model = build_student_from_checkpoint(
        checkpoint,
        device=device,
        student_dtype=args.student_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint["student_model"])
    ensure_pad_token(tokenizer)

    texts = load_texts(
        text_file=args.text_file,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        dataset_text_key=args.dataset_text_key,
        max_chars=max(512, args.sequence_length * 4),
    )
    train_loader, eval_loader = build_dataloaders(
        texts=texts,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        eval_fraction=args.eval_fraction,
        max_train_examples=args.max_train_examples,
        max_eval_examples=args.max_eval_examples,
    )

    reference_loader = None
    if args.reference_text_file or args.reference_dataset_name:
        reference_texts = load_texts(
            text_file=args.reference_text_file,
            dataset_name=args.reference_dataset_name,
            dataset_config=args.reference_dataset_config,
            dataset_split=args.reference_dataset_split,
            dataset_text_key=args.reference_dataset_text_key,
            max_chars=max(512, args.sequence_length * 4),
        )
        _, reference_loader = build_dataloaders(
            texts=reference_texts,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            eval_fraction=1.0,
            max_train_examples=None,
            max_eval_examples=args.max_eval_examples,
        )

    optimizer = torch.optim.AdamW(
        student_model.get_me_parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Student model: {checkpoint['student_model']}")
    print(f"Insert after: {student_model.insert_after}")
    print(f"Parameter counts: {student_model.count_parameters()}")

    baseline_new = run_continual_epoch(
        student_model=student_model,
        tokenizer=tokenizer,
        dataloader=eval_loader,
        optimizer=None,
        sequence_length=args.sequence_length,
        device=device,
        grad_clip=args.grad_clip,
    )
    print(
        "Baseline new-data eval:"
        f" ce={baseline_new['ce']:.4f}"
        f" ppl={baseline_new['perplexity']:.2f}"
        f" pr={baseline_new.get('pr', 0.0):.4f}"
        f" self_torque={baseline_new.get('self_torque', 0.0):.4f}"
    )

    if reference_loader is not None:
        baseline_reference = run_continual_epoch(
            student_model=student_model,
            tokenizer=tokenizer,
            dataloader=reference_loader,
            optimizer=None,
            sequence_length=args.sequence_length,
            device=device,
            grad_clip=args.grad_clip,
        )
        print(
            "Baseline reference eval:"
            f" ce={baseline_reference['ce']:.4f}"
            f" ppl={baseline_reference['perplexity']:.2f}"
        )

    final_metrics = baseline_new
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_continual_epoch(
            student_model=student_model,
            tokenizer=tokenizer,
            dataloader=train_loader,
            optimizer=optimizer,
            sequence_length=args.sequence_length,
            device=device,
            grad_clip=args.grad_clip,
        )
        eval_metrics = run_continual_epoch(
            student_model=student_model,
            tokenizer=tokenizer,
            dataloader=eval_loader,
            optimizer=None,
            sequence_length=args.sequence_length,
            device=device,
            grad_clip=args.grad_clip,
        )
        final_metrics = eval_metrics
        print(
            f"Epoch {epoch}:"
            f" train_ce={train_metrics['ce']:.4f}"
            f" eval_ce={eval_metrics['ce']:.4f}"
            f" eval_ppl={eval_metrics['perplexity']:.2f}"
            f" eval_pr={eval_metrics.get('pr', 0.0):.4f}"
            f" eval_self_torque={eval_metrics.get('self_torque', 0.0):.4f}"
        )

        if reference_loader is not None:
            reference_metrics = run_continual_epoch(
                student_model=student_model,
                tokenizer=tokenizer,
                dataloader=reference_loader,
                optimizer=None,
                sequence_length=args.sequence_length,
                device=device,
                grad_clip=args.grad_clip,
            )
            print(
                f"  reference_ce={reference_metrics['ce']:.4f}"
                f" reference_ppl={reference_metrics['perplexity']:.2f}"
            )

    save_continual_checkpoint(
        args.output_path,
        model=student_model,
        checkpoint=checkpoint,
        metrics=final_metrics,
    )
    print(f"Saved continual-learning checkpoint to {args.output_path}")


if __name__ == "__main__":
    main()
