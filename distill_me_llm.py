"""Phase 1 distillation for the hybrid Memory Engine student."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from memory_engine_llm import MemoryEngineCausalLM


_FEATURE_PROJECTOR_CACHE: Dict[tuple[int, int], torch.Tensor] = {}


def resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {name}") from exc


def build_model_load_kwargs(dtype_name: str, load_in_4bit: bool) -> Dict[str, object]:
    kwargs: Dict[str, object] = {"torch_dtype": resolve_dtype(dtype_name)}
    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ValueError("4-bit loading requires bitsandbytes support in transformers.") from exc
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=resolve_dtype(dtype_name),
        )
        kwargs["device_map"] = "auto"
    return kwargs


def model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:  # pragma: no cover - defensive
        return torch.device("cpu")


def ensure_pad_token(tokenizer) -> None:
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        raise ValueError("Tokenizer must define either a pad token or an eos token.")


def tokenizers_share_vocabulary(teacher_tokenizer, student_tokenizer) -> bool:
    return teacher_tokenizer.get_vocab() == student_tokenizer.get_vocab()


def _infer_hidden_size(model) -> int:
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("Model has no config; cannot infer hidden size.")
    for attr in ("hidden_size", "n_embd", "d_model"):
        if hasattr(config, attr):
            return int(getattr(config, attr))
    raise ValueError("Could not infer hidden size from model config.")


def default_corpus() -> List[str]:
    return [
        "The history of computing spans ancient counting tools, mechanical calculators, electronic computers, and modern software ecosystems that mediate science, communication, and finance.",
        "Neural networks learn from data through gradient descent, and transformers made long-range sequence modeling practical, though persistent state remains an open design axis.",
        "The solar system includes rocky planets, gas and ice giants, dwarf planets, moons, asteroids, and cometary reservoirs shaped by orbital mechanics and planetary chemistry.",
        "Evolution by natural selection explains how heritable variation and differential reproduction produce adaptation across deep stretches of biological time.",
        "Mathematics studies quantity, proof, structure, and transformation across algebra, analysis, geometry, topology, probability, and logic.",
        "The human brain links billions of neurons through synapses whose strengths evolve with experience, giving rise to memory, planning, and language.",
        "Climate change is driven largely by greenhouse gas emissions, stressing ecosystems, infrastructure, and agriculture through rising temperatures and shifting weather patterns.",
        "Music organizes rhythm, melody, timbre, and harmony into temporal patterns that feel stable, tense, novel, or resolved through memory and prediction.",
    ]


def chunk_texts(texts: Sequence[str], max_chars: int) -> List[str]:
    chunks: List[str] = []
    for text in texts:
        words = text.split()
        if not words:
            continue
        current: List[str] = []
        current_chars = 0
        for word in words:
            extra = len(word) + (1 if current else 0)
            if current and current_chars + extra > max_chars:
                chunks.append(" ".join(current))
                current = [word]
                current_chars = len(word)
            else:
                current.append(word)
                current_chars += extra
        if current:
            chunks.append(" ".join(current))
    return chunks


def load_texts(
    *,
    text_file: Optional[str],
    dataset_name: Optional[str],
    dataset_config: Optional[str],
    dataset_split: str,
    dataset_text_key: str,
    max_chars: int,
) -> List[str]:
    if text_file:
        raw_text = Path(text_file).read_text()
        paragraphs = [part.strip() for part in raw_text.split("\n\n") if part.strip()]
        return chunk_texts(paragraphs, max_chars=max_chars)

    if dataset_name:
        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ValueError("Loading a Hugging Face dataset requires the `datasets` package.") from exc

        dataset = load_dataset(dataset_name, dataset_config, split=dataset_split)
        texts = [row[dataset_text_key].strip() for row in dataset if row.get(dataset_text_key)]
        return chunk_texts(texts, max_chars=max_chars)

    return chunk_texts(default_corpus(), max_chars=max_chars)


class TextDataset(Dataset):
    def __init__(self, texts: Sequence[str]) -> None:
        self.examples = [text for text in texts if text.strip()]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> str:
        return self.examples[idx]


def build_dataloaders(
    *,
    texts: Sequence[str],
    batch_size: int,
    eval_batch_size: int,
    eval_fraction: float,
    max_train_examples: Optional[int],
    max_eval_examples: Optional[int],
) -> tuple[DataLoader, DataLoader]:
    if len(texts) < 2:
        raise ValueError("Need at least two text examples for train/eval split.")

    eval_count = max(1, int(len(texts) * eval_fraction))
    train_texts = list(texts[:-eval_count])
    eval_texts = list(texts[-eval_count:])

    if max_train_examples is not None:
        train_texts = train_texts[:max_train_examples]
    if max_eval_examples is not None:
        eval_texts = eval_texts[:max_eval_examples]

    train_loader = DataLoader(TextDataset(train_texts), batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(TextDataset(eval_texts), batch_size=eval_batch_size, shuffle=False)
    return train_loader, eval_loader


def prepare_causal_lm_batch(
    texts: Sequence[str],
    tokenizer,
    sequence_length: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    encoded = tokenizer(
        list(texts),
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=sequence_length + 1,
    )

    input_ids = encoded["input_ids"][:, :-1]
    attention_mask = encoded["attention_mask"][:, :-1]
    labels = encoded["input_ids"][:, 1:].clone()
    label_mask = encoded["attention_mask"][:, 1:]
    labels = labels.masked_fill(label_mask == 0, -100)

    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "labels": labels.to(device),
    }


def select_last_token_hidden(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    lengths = attention_mask.long().sum(dim=-1).clamp_min(1) - 1
    batch_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    return hidden_states[batch_indices, lengths]


def project_hidden_to_dim(hidden: torch.Tensor, target_dim: int) -> torch.Tensor:
    source_dim = hidden.shape[-1]
    if source_dim == target_dim:
        return hidden

    key = (source_dim, target_dim)
    if key not in _FEATURE_PROJECTOR_CACHE:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(0)
        if source_dim >= target_dim:
            matrix = torch.randn(source_dim, target_dim, generator=generator, dtype=torch.float32)
            projector, _ = torch.linalg.qr(matrix, mode="reduced")
        else:
            matrix = torch.randn(target_dim, source_dim, generator=generator, dtype=torch.float32)
            projector, _ = torch.linalg.qr(matrix, mode="reduced")
            projector = projector.transpose(0, 1)
        _FEATURE_PROJECTOR_CACHE[key] = projector.contiguous()

    projector = _FEATURE_PROJECTOR_CACHE[key].to(device=hidden.device, dtype=hidden.dtype)
    return hidden @ projector


def compute_pr_regularization(
    me_diagnostics: Sequence[Dict[str, object]],
    target_fraction: float = 0.35,
) -> torch.Tensor:
    reg_terms: List[torch.Tensor] = []
    for layer_diag in me_diagnostics:
        diagnostics = layer_diag["diagnostics"]
        if not diagnostics:
            continue
        pr = diagnostics["pr"]
        active_slots = diagnostics["active_slots"].float().unsqueeze(-1)
        pr_fraction = pr / active_slots.clamp_min(1.0)
        reg_terms.append(F.relu(target_fraction - pr_fraction).mean())

    if not reg_terms:
        return torch.tensor(0.0)
    return torch.stack(reg_terms).mean()


def compute_distillation_losses(
    *,
    student_outputs: Dict[str, object],
    teacher_logits: Optional[torch.Tensor],
    teacher_hidden: Optional[torch.Tensor],
    labels: torch.Tensor,
    temperature: float,
    ce_weight: float,
    kl_weight: float,
    feature_weight: float,
    pr_reg_weight: float,
) -> Dict[str, torch.Tensor]:
    ce_loss = student_outputs["loss"]
    if ce_loss is None:
        ce_loss = F.cross_entropy(
            student_outputs["logits"].reshape(-1, student_outputs["logits"].shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
        )

    total_loss = ce_weight * ce_loss
    kl_loss = ce_loss.new_tensor(0.0)
    feature_loss = ce_loss.new_tensor(0.0)
    pr_loss = ce_loss.new_tensor(0.0)

    if kl_weight > 0:
        if teacher_logits is None:
            raise ValueError("teacher_logits is required when kl_weight > 0.")
        kl_loss = F.kl_div(
            F.log_softmax(student_outputs["logits"] / temperature, dim=-1),
            F.softmax(teacher_logits / temperature, dim=-1),
            reduction="batchmean",
        ) * (temperature ** 2)
        total_loss = total_loss + kl_weight * kl_loss

    if feature_weight > 0:
        if teacher_hidden is None or "final_memory_hidden" not in student_outputs:
            raise ValueError("Feature distillation requires teacher hidden states and student final memory hidden.")
        student_hidden = student_outputs["final_memory_hidden"]
        teacher_hidden = project_hidden_to_dim(teacher_hidden.to(student_hidden.device), student_hidden.shape[-1])
        feature_loss = F.mse_loss(student_hidden, teacher_hidden)
        total_loss = total_loss + feature_weight * feature_loss

    if pr_reg_weight > 0:
        pr_loss = compute_pr_regularization(student_outputs["me_diagnostics"]).to(total_loss.device)
        total_loss = total_loss + pr_reg_weight * pr_loss

    return {
        "total": total_loss,
        "ce": ce_loss,
        "kl": kl_loss,
        "feature": feature_loss,
        "pr_reg": pr_loss,
    }


def summarize_memory(outputs: Dict[str, object]) -> Dict[str, float]:
    diagnostics_rows = outputs.get("me_diagnostics", [])
    if not diagnostics_rows:
        return {}

    rows: Dict[str, List[float]] = {
        "pr": [],
        "gini": [],
        "self_torque": [],
        "prediction_error": [],
        "coupling_strength": [],
        "resonance_fraction": [],
        "torque_fraction": [],
    }
    for layer_diag in diagnostics_rows:
        diagnostics = layer_diag["diagnostics"]
        if not diagnostics:
            continue
        for key in rows:
            value = diagnostics.get(key)
            if value is None:
                continue
            if value.ndim == 1:
                rows[key].append(float(value.mean().item()))
            else:
                rows[key].append(float(value[:, -1].mean().item()))
    return {key: mean(values) for key, values in rows.items() if values}


def validate_distillation_setup(
    *,
    teacher_tokenizer,
    student_tokenizer,
    kl_weight: float,
) -> None:
    if kl_weight > 0 and not tokenizers_share_vocabulary(teacher_tokenizer, student_tokenizer):
        raise ValueError(
            "KL distillation requires teacher and student tokenizers to share a vocabulary. "
            "Use tokenizer-compatible checkpoints or set --kl-weight 0."
        )


def run_distill_epoch(
    *,
    teacher_model,
    teacher_tokenizer,
    student_model: MemoryEngineCausalLM,
    student_tokenizer,
    dataloader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    sequence_length: int,
    student_device: torch.device,
    temperature: float,
    ce_weight: float,
    kl_weight: float,
    feature_weight: float,
    pr_reg_weight: float,
    grad_clip: float,
) -> Dict[str, float]:
    teacher_model.eval()
    if optimizer is None:
        student_model.eval()
    else:
        student_model.train()

    teacher_device = model_device(teacher_model)
    losses: Dict[str, List[float]] = {"total": [], "ce": [], "kl": [], "feature": [], "pr_reg": []}
    memory_metrics: Dict[str, List[float]] = {}

    for texts in dataloader:
        student_batch = prepare_causal_lm_batch(texts, student_tokenizer, sequence_length, student_device)
        teacher_batch = prepare_causal_lm_batch(texts, teacher_tokenizer, sequence_length, teacher_device)

        with torch.no_grad():
            teacher_outputs = teacher_model(
                input_ids=teacher_batch["input_ids"],
                attention_mask=teacher_batch["attention_mask"],
                output_hidden_states=feature_weight > 0,
                use_cache=False,
            )
            teacher_hidden = None
            if feature_weight > 0:
                teacher_hidden = select_last_token_hidden(
                    teacher_outputs.hidden_states[-1],
                    teacher_batch["attention_mask"],
                ).to(student_device)
            teacher_logits = teacher_outputs.logits.to(student_device) if kl_weight > 0 else None

        if optimizer is not None:
            optimizer.zero_grad()

        student_outputs = student_model(
            input_ids=student_batch["input_ids"],
            attention_mask=student_batch["attention_mask"],
            labels=student_batch["labels"],
            reset_memory=True,
            return_memory_features=True,
            use_cache=False,
        )

        distill_losses = compute_distillation_losses(
            student_outputs=student_outputs,
            teacher_logits=teacher_logits,
            teacher_hidden=teacher_hidden,
            labels=student_batch["labels"],
            temperature=temperature,
            ce_weight=ce_weight,
            kl_weight=kl_weight,
            feature_weight=feature_weight,
            pr_reg_weight=pr_reg_weight,
        )

        if optimizer is not None:
            distill_losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(student_model.get_me_parameters(), grad_clip)
            optimizer.step()

        for key, value in distill_losses.items():
            losses[key].append(float(value.detach().item()))

        memory_summary = summarize_memory(student_outputs)
        for key, value in memory_summary.items():
            memory_metrics.setdefault(key, []).append(value)

    metrics = {key: mean(values) for key, values in losses.items() if values}
    for key, values in memory_metrics.items():
        metrics[key] = mean(values)
    metrics["perplexity"] = math.exp(metrics["ce"]) if "ce" in metrics else float("inf")
    return metrics


def parse_insert_after(raw: Optional[str]) -> Optional[List[int]]:
    if raw is None or raw.strip() == "":
        return None
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def build_student(args: argparse.Namespace, device: torch.device) -> MemoryEngineCausalLM:
    base_model = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        **build_model_load_kwargs(args.student_dtype, load_in_4bit=False),
    )
    student = MemoryEngineCausalLM(
        base_model=base_model,
        model_name=args.student_model,
        insert_after=parse_insert_after(args.insert_after),
        freeze_base_model=True,
        memory_dim=args.memory_dim,
        max_aux_dims=args.max_aux_dims,
        max_transient_dims=args.max_transient_dims,
        coupling_rank=args.coupling_rank,
        prediction_rank=args.prediction_rank,
        consolidation_interval=args.consolidation_interval,
    ).to(device)
    return student


def build_teacher(args: argparse.Namespace):
    teacher_model_name = args.teacher_model or args.student_model
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_model_name,
        **build_model_load_kwargs(args.teacher_dtype, load_in_4bit=args.teacher_load_in_4bit),
    )
    if not args.teacher_load_in_4bit:
        teacher = teacher.to(torch.device(args.device))
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    return teacher_model_name, teacher


def save_checkpoint(
    output_path: str,
    model: MemoryEngineCausalLM,
    args: argparse.Namespace,
    metrics: Dict[str, float],
) -> None:
    payload = {
        "student_model": args.student_model,
        "teacher_model": args.teacher_model or args.student_model,
        "insert_after": parse_insert_after(args.insert_after),
        "memory_config": {
            "memory_dim": args.memory_dim,
            "max_aux_dims": args.max_aux_dims,
            "max_transient_dims": args.max_transient_dims,
            "coupling_rank": args.coupling_rank,
            "prediction_rank": args.prediction_rank,
            "consolidation_interval": args.consolidation_interval,
        },
        "me_state_dict": model.me_state_dict(),
        "metrics": metrics,
    }
    torch.save(payload, output_path)


def load_me_checkpoint(path: str) -> Dict[str, object]:
    return torch.load(path, map_location="cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 distillation for the hybrid Memory Engine student.")
    parser.add_argument("--student-model", default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--teacher-model", default=None, help="Defaults to --student-model.")
    parser.add_argument("--insert-after", default="3,6,9")
    parser.add_argument("--memory-dim", type=int, default=256)
    parser.add_argument("--max-aux-dims", type=int, default=16)
    parser.add_argument("--max-transient-dims", type=int, default=8)
    parser.add_argument("--coupling-rank", type=int, default=10)
    parser.add_argument("--prediction-rank", type=int, default=10)
    parser.add_argument("--consolidation-interval", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--ce-weight", type=float, default=1.2)
    parser.add_argument("--kl-weight", type=float, default=0.5)
    parser.add_argument("--feature-weight", type=float, default=0.4)
    parser.add_argument("--pr-reg-weight", type=float, default=0.05)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--student-dtype", default="float32", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--teacher-dtype", default="bfloat16", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--teacher-load-in-4bit", action="store_true")
    parser.add_argument("--text-file", default=None)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--dataset-text-key", default="text")
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--max-train-examples", type=int, default=128)
    parser.add_argument("--max-eval-examples", type=int, default=32)
    parser.add_argument("--output-path", default="distilled_me_checkpoint.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    teacher_model_name = args.teacher_model or args.student_model

    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
    student_tokenizer = AutoTokenizer.from_pretrained(args.student_model)
    ensure_pad_token(teacher_tokenizer)
    ensure_pad_token(student_tokenizer)
    validate_distillation_setup(
        teacher_tokenizer=teacher_tokenizer,
        student_tokenizer=student_tokenizer,
        kl_weight=args.kl_weight,
    )

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

    teacher_model_name, teacher_model = build_teacher(args)
    student_model = build_student(args, device=device)
    student_hidden_size = student_model.hidden_size
    teacher_hidden_size = _infer_hidden_size(teacher_model)

    optimizer = torch.optim.AdamW(
        student_model.get_me_parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    print(f"Teacher: {teacher_model_name} (hidden={teacher_hidden_size})")
    print(f"Student: {args.student_model} (hidden={student_hidden_size})")
    print(f"Insert after: {student_model.insert_after}")
    print(f"Train/Eval examples: {len(train_loader.dataset)} / {len(eval_loader.dataset)}")
    print(f"Parameter counts: {student_model.count_parameters()}")

    baseline = run_distill_epoch(
        teacher_model=teacher_model,
        teacher_tokenizer=teacher_tokenizer,
        student_model=student_model,
        student_tokenizer=student_tokenizer,
        dataloader=eval_loader,
        optimizer=None,
        sequence_length=args.sequence_length,
        student_device=device,
        temperature=args.temperature,
        ce_weight=args.ce_weight,
        kl_weight=args.kl_weight,
        feature_weight=args.feature_weight,
        pr_reg_weight=args.pr_reg_weight,
        grad_clip=args.grad_clip,
    )
    print(
        "Baseline eval:"
        f" ce={baseline['ce']:.4f}"
        f" ppl={baseline['perplexity']:.2f}"
        f" kl={baseline['kl']:.4f}"
        f" feature={baseline['feature']:.4f}"
        f" pr_reg={baseline['pr_reg']:.4f}"
    )

    final_metrics = baseline
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_distill_epoch(
            teacher_model=teacher_model,
            teacher_tokenizer=teacher_tokenizer,
            student_model=student_model,
            student_tokenizer=student_tokenizer,
            dataloader=train_loader,
            optimizer=optimizer,
            sequence_length=args.sequence_length,
            student_device=device,
            temperature=args.temperature,
            ce_weight=args.ce_weight,
            kl_weight=args.kl_weight,
            feature_weight=args.feature_weight,
            pr_reg_weight=args.pr_reg_weight,
            grad_clip=args.grad_clip,
        )
        eval_metrics = run_distill_epoch(
            teacher_model=teacher_model,
            teacher_tokenizer=teacher_tokenizer,
            student_model=student_model,
            student_tokenizer=student_tokenizer,
            dataloader=eval_loader,
            optimizer=None,
            sequence_length=args.sequence_length,
            student_device=device,
            temperature=args.temperature,
            ce_weight=args.ce_weight,
            kl_weight=args.kl_weight,
            feature_weight=args.feature_weight,
            pr_reg_weight=args.pr_reg_weight,
            grad_clip=args.grad_clip,
        )
        final_metrics = eval_metrics
        print(
            f"Epoch {epoch}:"
            f" train_total={train_metrics['total']:.4f}"
            f" train_ce={train_metrics['ce']:.4f}"
            f" eval_total={eval_metrics['total']:.4f}"
            f" eval_ppl={eval_metrics['perplexity']:.2f}"
            f" eval_pr={eval_metrics.get('pr', 0.0):.4f}"
            f" eval_self_torque={eval_metrics.get('self_torque', 0.0):.4f}"
            f" eval_coupling={eval_metrics.get('coupling_strength', 0.0):.4f}"
        )

    save_checkpoint(args.output_path, student_model, args, final_metrics)
    print(f"Saved distilled checkpoint to {args.output_path}")


if __name__ == "__main__":
    main()
