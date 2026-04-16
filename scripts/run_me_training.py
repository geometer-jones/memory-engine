"""Train the pure hierarchical Memory Engine LLM on text windows.

The script preserves the repo's lightweight data path:

- use a built-in WikiText-style fallback corpus out of the box
- optionally read TinyStories / WikiText-style `.txt`, `.json`, or `.jsonl`
- tokenize into overlapping windows for next-token prediction

The model itself is fully trainable: embeddings, all ME layer parameters, and
the final LM head are optimized together with Adam.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from memory_engine_llm import MemoryEngineLLM

try:
    from transformers import AutoTokenizer
except Exception:  # pragma: no cover - fallback path is still tested indirectly.
    AutoTokenizer = None


class ByteTokenizer:
    """Fallback tokenizer so the script still runs without a HF tokenizer cache."""

    vocab_size = 258
    bos_token_id = 256
    pad_token_id = 257

    def encode(self, text: str) -> List[int]:
        encoded = list(text.encode("utf-8", errors="ignore"))
        return [self.bos_token_id] + encoded


def load_tokenizer(tokenizer_name: str = "gpt2"):
    if AutoTokenizer is None:
        return ByteTokenizer()

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
        return tokenizer
    except Exception:
        return ByteTokenizer()


class TextDataset(Dataset):
    """Overlapping fixed-length next-token windows."""

    def __init__(self, texts: Iterable[str], tokenizer, block_size: int = 128):
        self.examples: List[List[int]] = []
        stride = max(1, block_size // 2)

        for text in texts:
            tokens = tokenizer.encode(text)
            if len(tokens) <= block_size:
                continue
            for start in range(0, len(tokens) - block_size, stride):
                self.examples.append(tokens[start : start + block_size + 1])

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        tokens = self.examples[index]
        inputs = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        return inputs, labels


def _default_corpus() -> List[str]:
    return [
        "The history of computing spans thousands of years, from ancient counting tools to modern supercomputers. Early mechanical calculators led to electronic machines, integrated circuits, and software ecosystems that now mediate finance, communication, and science.",
        "Neural networks are layered function approximators that learn from data through gradient descent. Transformers made long-range sequence modeling practical by replacing recurrence with structured token mixing, but persistent state remains an open design axis.",
        "The solar system consists of rocky inner planets, gas and ice giants, dwarf planets, moons, asteroids, and cometary reservoirs. Orbital mechanics, planetary geology, and atmospheric chemistry reveal how worlds form and change over time.",
        "Evolution by natural selection explains how heritable variation and differential reproduction produce adaptation. Comparative anatomy, genetics, and the fossil record together show deep continuity across life on Earth.",
        "Mathematics studies structure, quantity, proof, and transformation. Algebra, analysis, geometry, topology, probability, and logic provide compact languages for describing physical systems and abstract relationships.",
        "The human brain contains billions of neurons linked by synapses whose strengths evolve with experience. Memory, planning, and language emerge from distributed dynamics rather than a single privileged location.",
        "Climate change is driven largely by greenhouse gas emissions from human activity. Its effects include rising temperatures, changing rainfall patterns, stressed ecosystems, and increasing pressure on infrastructure and agriculture.",
        "Philosophy asks what exists, what can be known, how reason works, and what should be valued. Its methods overlap with science, mathematics, law, and literature while remaining concerned with conceptual clarity itself.",
        "Architecture combines engineering constraints, human use, and aesthetic judgment. Materials, structure, climate, and circulation all shape the spaces where people gather, rest, work, and remember.",
        "Music organizes rhythm, melody, timbre, and harmony into patterns that can feel stable, tense, novel, or resolved. Its structures are temporal, embodied, and strongly tied to memory and prediction.",
    ]


def _extract_texts_from_json_record(record: object) -> Optional[str]:
    if isinstance(record, str):
        return record
    if not isinstance(record, dict):
        return None
    for key in ("text", "story", "content", "body"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def load_corpus(path: Optional[str], max_docs: int = 200) -> List[str]:
    if path is None:
        return _default_corpus()

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Corpus path does not exist: {source}")

    texts: List[str] = []
    files = [source] if source.is_file() else sorted(p for p in source.rglob("*") if p.is_file())

    for file_path in files:
        suffix = file_path.suffix.lower()
        if suffix == ".txt":
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            paragraphs = [chunk.strip() for chunk in text.splitlines() if chunk.strip()]
            texts.extend(paragraphs if paragraphs else [text])
        elif suffix == ".jsonl":
            with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    text = _extract_texts_from_json_record(json.loads(line))
                    if text:
                        texts.append(text)
        elif suffix == ".json":
            content = json.loads(file_path.read_text(encoding="utf-8", errors="ignore"))
            if isinstance(content, list):
                for item in content:
                    text = _extract_texts_from_json_record(item)
                    if text:
                        texts.append(text)
            else:
                text = _extract_texts_from_json_record(content)
                if text:
                    texts.append(text)

        if len(texts) >= max_docs:
            break

    return texts[:max_docs] if texts else _default_corpus()


def make_datasets(
    tokenizer,
    block_size: int,
    corpus_path: Optional[str],
    max_docs: int,
) -> tuple[TextDataset, TextDataset]:
    texts = load_corpus(corpus_path, max_docs=max_docs)
    if len(texts) < 3:
        texts = _default_corpus()

    n_eval = max(1, min(len(texts) // 5, 8))
    eval_texts = texts[:n_eval]
    train_texts = texts[n_eval:]

    train_dataset = TextDataset(train_texts, tokenizer, block_size=block_size)
    eval_dataset = TextDataset(eval_texts, tokenizer, block_size=block_size)

    if len(train_dataset) == 0 or len(eval_dataset) == 0:
        merged = [" ".join(texts)]
        train_dataset = TextDataset(merged, tokenizer, block_size=block_size)
        eval_dataset = TextDataset(merged, tokenizer, block_size=block_size)

    return train_dataset, eval_dataset


def _mean_tensor(metric: List[torch.Tensor]) -> float:
    if not metric:
        return 0.0
    stacked = torch.stack([value.float().mean() for value in metric])
    return float(stacked.mean().item())


def evaluate_model(
    model: MemoryEngineLLM,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    final_pr_values: List[torch.Tensor] = []
    layer_resonance: List[List[float]] = []
    layer_torque: List[List[float]] = []

    with torch.no_grad():
        for batch_index, (input_ids, labels) in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break

            input_ids = input_ids.to(device)
            labels = labels.to(device)

            outputs = model(input_ids, labels=labels, return_diagnostics=True)
            loss = outputs["loss"]
            diagnostics = outputs["diagnostics"]
            layer_diags = diagnostics["layer_diagnostics"]

            tokens = labels.numel()
            total_loss += float(loss.item()) * tokens
            total_tokens += tokens
            final_pr_values.append(layer_diags[-1]["pr"])

            if not layer_resonance:
                layer_resonance = [[] for _ in layer_diags]
                layer_torque = [[] for _ in layer_diags]

            for layer_index, layer_diag in enumerate(layer_diags):
                layer_resonance[layer_index].append(float(layer_diag["resonance_fraction"].mean().item()))
                layer_torque[layer_index].append(float(layer_diag["torque_fraction"].mean().item()))

    mean_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(mean_loss, 20.0))

    layer_summary = []
    for layer_index, (res_values, torque_values) in enumerate(zip(layer_resonance, layer_torque)):
        layer_summary.append(
            {
                "layer": layer_index,
                "resonance": sum(res_values) / max(len(res_values), 1),
                "torque": sum(torque_values) / max(len(torque_values), 1),
            }
        )

    return {
        "loss": mean_loss,
        "perplexity": perplexity,
        "final_pr": _mean_tensor(final_pr_values),
        "layer_summary": layer_summary,
    }


def train_epoch(
    model: MemoryEngineLLM,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    max_batches: Optional[int] = None,
) -> float:
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for batch_index, (input_ids, labels) in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break

        input_ids = input_ids.to(device)
        labels = labels.to(device)

        outputs = model(input_ids, labels=labels, return_diagnostics=False)
        loss = outputs["loss"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        tokens = labels.numel()
        total_loss += float(loss.item()) * tokens
        total_tokens += tokens

    return total_loss / max(total_tokens, 1)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the pure Memory Engine LLM.")
    parser.add_argument("--corpus-path", type=str, default=None, help="Optional txt/json/jsonl corpus path.")
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="HF tokenizer name; falls back to bytes.")
    parser.add_argument("--dim", type=int, default=512, help="Hidden width and tape width.")
    parser.add_argument("--n-layers", type=int, default=8, help="Number of stacked ME layers.")
    parser.add_argument("--max-seq-len", type=int, default=256, help="Maximum supported sequence length.")
    parser.add_argument("--block-size", type=int, default=128, help="Training window length.")
    parser.add_argument("--batch-size", type=int, default=2, help="Training batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=2, help="Evaluation batch size.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Adam learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping threshold.")
    parser.add_argument("--max-docs", type=int, default=200, help="Maximum number of source documents.")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Optional per-epoch batch cap.")
    parser.add_argument("--max-eval-batches", type=int, default=None, help="Optional eval batch cap.")
    parser.add_argument("--eta-init", type=float, default=0.08, help="Initial ME impression rate.")
    parser.add_argument("--alpha-init", type=float, default=0.5, help="Initial directness gate logit.")
    parser.add_argument("--coupling-mode", type=str, default="full", choices=("full", "diagonal"))
    parser.add_argument("--coupling-window", type=float, default=0.35, help="Stability window for epsilon.")
    parser.add_argument("--consolidation-interval", type=int, default=8, help="Run soft consolidation every K tokens.")
    parser.add_argument("--theta-merge", type=float, default=0.35, help="Initial soft merge threshold.")
    parser.add_argument("--theta-prune", type=float, default=0.02, help="Initial soft prune threshold.")
    parser.add_argument("--prediction-torque-scale", type=float, default=0.5, help="Blend weight on c_pred.")
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. cuda or cpu.")
    parser.add_argument("--smoke-test", action="store_true", help="Short single-GPU sanity run at the default 512x8 scale.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = load_tokenizer(args.tokenizer)
    train_dataset, eval_dataset = make_datasets(
        tokenizer=tokenizer,
        block_size=args.block_size,
        corpus_path=args.corpus_path,
        max_docs=args.max_docs,
    )

    if args.smoke_test:
        args.epochs = 1
        args.max_train_batches = 8 if args.max_train_batches is None else args.max_train_batches
        args.max_eval_batches = 4 if args.max_eval_batches is None else args.max_eval_batches

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False)

    model = MemoryEngineLLM(
        vocab_size=tokenizer.vocab_size,
        dim=args.dim,
        n_layers=args.n_layers,
        max_seq_len=args.max_seq_len,
        eta_init=args.eta_init,
        alpha_init=args.alpha_init,
        coupling_mode=args.coupling_mode,
        coupling_window=args.coupling_window,
        consolidation_interval=args.consolidation_interval,
        theta_merge=args.theta_merge,
        theta_prune=args.theta_prune,
        prediction_torque_scale=args.prediction_torque_scale,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print("Pure Memory Engine LLM training")
    print(f"  Device:      {device}")
    print(f"  Tokenizer:   {type(tokenizer).__name__}")
    print(f"  Train/Eval:  {len(train_dataset)} / {len(eval_dataset)} windows")
    print(f"  Model:       dim={args.dim}, n_layers={args.n_layers}, vocab={tokenizer.vocab_size}")
    print(f"  Parameters:  {model.count_parameters()}")
    if args.smoke_test:
        print("  Mode:        smoke test")

    baseline = evaluate_model(model, eval_loader, device=device, max_batches=args.max_eval_batches)
    print(f"Baseline perplexity: {baseline['perplexity']:.2f}")

    for epoch in range(args.epochs):
        train_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip=args.grad_clip,
            max_batches=args.max_train_batches,
        )

        evaluation = evaluate_model(
            model=model,
            dataloader=eval_loader,
            device=device,
            max_batches=args.max_eval_batches,
        )

        print(
            f"Epoch {epoch + 1:>2} | "
            f"train_loss={train_loss:.4f} | "
            f"eval_loss={evaluation['loss']:.4f} | "
            f"ppl={evaluation['perplexity']:.2f} | "
            f"final_PR={evaluation['final_pr']:.2f}"
        )

        for layer_info in evaluation["layer_summary"]:
            print(
                f"  layer {layer_info['layer']:>2}: "
                f"resonance={layer_info['resonance'] * 100:5.1f}% "
                f"torque={layer_info['torque'] * 100:5.1f}%"
            )


if __name__ == "__main__":
    main()
