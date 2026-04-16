"""Train the Vision Memory Engine on MNIST handwritten digits."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from vision_memory_engine import VisionMemoryRecognizer


MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def _renormalize_complex_rows(values: torch.Tensor) -> torch.Tensor:
    norms = values.abs().pow(2).sum(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
    return values / norms


def _resonance_fraction(reception: torch.Tensor) -> torch.Tensor:
    re = reception.real
    im = reception.imag.abs()
    mag = reception.abs()
    resonance = (re > 1e-6) & (im < re) & (mag >= 1e-8)
    return resonance.float().mean(dim=-1)


def _parse_stem_channels(raw: Optional[str | Tuple[int, ...]]) -> Tuple[int, ...]:
    if raw is None:
        return ()
    if isinstance(raw, tuple):
        return raw
    if isinstance(raw, list):
        return tuple(int(value) for value in raw)

    parts = [part.strip() for part in str(raw).split(",") if part.strip()]
    if not parts:
        return ()

    channels = tuple(int(part) for part in parts)
    if any(channel <= 0 for channel in channels):
        raise ValueError(f"stem_channels must be positive integers, got {raw!r}")
    return channels


def build_transform(image_size: int) -> transforms.Compose:
    steps = []
    if image_size != 28:
        steps.append(transforms.Resize((image_size, image_size)))
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ]
    )
    return transforms.Compose(steps)


def maybe_limit_dataset(dataset: Dataset, limit: Optional[int]) -> Dataset:
    if limit is None or limit >= len(dataset):
        return dataset
    return Subset(dataset, range(limit))


def build_datasets(
    data_dir: str,
    image_size: int,
    fake_data: bool = False,
    train_limit: Optional[int] = None,
    test_limit: Optional[int] = None,
) -> Tuple[Dataset, Dataset]:
    transform = build_transform(image_size)

    if fake_data:
        train_dataset = datasets.FakeData(
            size=train_limit or 512,
            image_size=(1, image_size, image_size),
            num_classes=10,
            transform=transform,
        )
        test_dataset = datasets.FakeData(
            size=test_limit or 128,
            image_size=(1, image_size, image_size),
            num_classes=10,
            transform=transform,
        )
        return train_dataset, test_dataset

    root = Path(data_dir)
    train_dataset = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=root, train=False, download=True, transform=transform)
    return (
        maybe_limit_dataset(train_dataset, train_limit),
        maybe_limit_dataset(test_dataset, test_limit),
    )


def build_loaders(
    train_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int,
    eval_batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, test_loader


def build_model(args: argparse.Namespace) -> VisionMemoryRecognizer:
    memory_dim = args.memory_dim if args.memory_dim is not None else args.hidden_dim
    conv_stem_channels = _parse_stem_channels(getattr(args, "stem_channels", "32,64"))
    return VisionMemoryRecognizer(
        image_size=args.image_size,
        patch_size=args.patch_size,
        in_channels=1,
        hidden_dim=args.hidden_dim,
        num_classes=10,
        dropout=args.dropout,
        conv_stem_channels=conv_stem_channels,
        memory_dim=memory_dim,
    )


def collect_digit_prototypes(
    model: VisionMemoryRecognizer,
    loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    prototype_sums = None
    counts = None

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)

            _, state, _ = model.forward_features(images, return_diagnostics=False)
            tape = state.tape.detach()

            if prototype_sums is None:
                prototype_sums = torch.zeros(10, tape.shape[-1], dtype=tape.dtype, device=device)
                counts = torch.zeros(10, dtype=torch.long, device=device)

            for digit in range(10):
                mask = labels == digit
                if torch.any(mask):
                    prototype_sums[digit] += tape[mask].sum(dim=0)
                    counts[digit] += int(mask.sum().item())

            if max_batches is not None and batch_index + 1 >= max_batches:
                break

    if prototype_sums is None or counts is None:
        raise ValueError("Could not collect digit prototypes from an empty loader.")

    prototypes = torch.zeros_like(prototype_sums)
    valid = counts > 0
    if torch.any(valid):
        prototypes[valid] = _renormalize_complex_rows(prototype_sums[valid])
    return prototypes, counts


def compute_digit_resonance_scores(
    tape: torch.Tensor,
    prototypes: torch.Tensor,
    prototype_counts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    alignment = tape.unsqueeze(1) * prototypes.conj().unsqueeze(0)
    scores = _resonance_fraction(alignment)

    if prototype_counts is not None:
        invalid = prototype_counts <= 0
        if torch.any(invalid):
            scores[:, invalid] = float("-inf")
    return scores


def evaluate_digit_resonance(
    model: VisionMemoryRecognizer,
    loader: DataLoader,
    device: torch.device,
    prototypes: torch.Tensor,
    prototype_counts: torch.Tensor,
    max_batches: Optional[int] = None,
) -> Dict[str, float | list[float]]:
    model.eval()
    total_examples = 0
    total_correct = 0
    total_true_resonance = 0.0
    total_predicted_resonance = 0.0
    per_digit_sum = torch.zeros(10, dtype=torch.float32)
    per_digit_count = torch.zeros(10, dtype=torch.long)

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)

            _, state, _ = model.forward_features(images, return_diagnostics=False)
            scores = compute_digit_resonance_scores(
                state.tape.detach(),
                prototypes=prototypes,
                prototype_counts=prototype_counts,
            )

            predicted_digits = scores.argmax(dim=-1)
            true_digit_scores = scores[torch.arange(labels.size(0), device=device), labels]
            predicted_scores = scores.max(dim=-1).values

            total_examples += labels.size(0)
            total_correct += int((predicted_digits == labels).sum().item())
            total_true_resonance += float(true_digit_scores.sum().item())
            total_predicted_resonance += float(predicted_scores.sum().item())

            for digit in range(10):
                mask = labels == digit
                if torch.any(mask):
                    per_digit_sum[digit] += true_digit_scores[mask].sum().cpu()
                    per_digit_count[digit] += int(mask.sum().item())

            if max_batches is not None and batch_index + 1 >= max_batches:
                break

    per_digit_resonance = []
    for digit in range(10):
        if per_digit_count[digit] > 0:
            per_digit_resonance.append(float((per_digit_sum[digit] / per_digit_count[digit]).item()))
        else:
            per_digit_resonance.append(float("nan"))

    return {
        "digit_probe_accuracy": total_correct / max(total_examples, 1),
        "true_digit_resonance": total_true_resonance / max(total_examples, 1),
        "predicted_digit_resonance": total_predicted_resonance / max(total_examples, 1),
        "prototype_support": [int(x) for x in prototype_counts.detach().cpu().tolist()],
        "per_digit_resonance": per_digit_resonance,
    }


def train_one_epoch(
    model: VisionMemoryRecognizer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * labels.size(0)
        total_correct += int((logits.argmax(dim=-1) == labels).sum().item())
        total_examples += labels.size(0)

    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
    }


def evaluate(
    model: VisionMemoryRecognizer,
    loader: DataLoader,
    device: torch.device,
    collect_diagnostics: bool = True,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    diagnostics_row = None

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)

            if collect_diagnostics and batch_index == 0:
                logits, _, diagnostics = model(images, return_diagnostics=True)
                diagnostics_row = {
                    "pr": float(diagnostics["pr"][:, -1].mean().item()),
                    "prediction_error": float(diagnostics["prediction_error"][:, -1].mean().item()),
                    "active_slots": float(diagnostics["active_slots"].float().mean().item()),
                }
            else:
                logits = model(images)

            loss = F.cross_entropy(logits, labels)
            total_loss += float(loss.item()) * labels.size(0)
            total_correct += int((logits.argmax(dim=-1) == labels).sum().item())
            total_examples += labels.size(0)

    metrics = {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
    }
    if diagnostics_row is not None:
        metrics.update(diagnostics_row)
    return metrics


def save_checkpoint(
    model: VisionMemoryRecognizer,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    path: str,
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": {
            "image_size": args.image_size,
            "patch_size": args.patch_size,
            "hidden_dim": args.hidden_dim,
            "memory_dim": args.memory_dim if args.memory_dim is not None else args.hidden_dim,
            "dropout": args.dropout,
            "stem_channels": list(_parse_stem_channels(getattr(args, "stem_channels", "32,64"))),
        },
    }
    torch.save(checkpoint, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Vision Memory Engine on MNIST.")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=28)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--memory-dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--stem-channels", default="32,64", help="Comma-separated conv stem widths before patchify. Use an empty string to disable.")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--digit-probe-batches", type=int, default=None)
    parser.add_argument("--report-per-digit", action="store_true")
    parser.add_argument("--fake-data", action="store_true")
    parser.add_argument("--save-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    train_dataset, test_dataset = build_datasets(
        data_dir=args.data_dir,
        image_size=args.image_size,
        fake_data=args.fake_data,
        train_limit=args.train_limit,
        test_limit=args.test_limit,
    )
    train_loader, test_loader = build_loaders(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
    )

    model = build_model(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(
        f"Training VisionMemoryRecognizer on {'FakeData' if args.fake_data else 'MNIST'} "
        f"with {len(train_dataset)} train / {len(test_dataset)} test examples"
    )

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        eval_metrics = evaluate(model, test_loader, device)
        prototypes, prototype_counts = collect_digit_prototypes(
            model,
            train_loader,
            device,
            max_batches=args.digit_probe_batches,
        )
        resonance_metrics = evaluate_digit_resonance(
            model,
            test_loader,
            device,
            prototypes=prototypes,
            prototype_counts=prototype_counts,
            max_batches=args.digit_probe_batches,
        )
        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"test_loss={eval_metrics['loss']:.4f} "
            f"test_acc={eval_metrics['accuracy']:.4f} "
            f"pr={eval_metrics.get('pr', float('nan')):.4f} "
            f"pred_err={eval_metrics.get('prediction_error', float('nan')):.4f} "
            f"active_slots={eval_metrics.get('active_slots', float('nan')):.2f} "
            f"digit_probe_acc={resonance_metrics['digit_probe_accuracy']:.4f} "
            f"true_digit_res={resonance_metrics['true_digit_resonance']:.4f}"
        )
        if args.report_per_digit:
            per_digit = " ".join(
                f"{digit}:{score:.3f}"
                for digit, score in enumerate(resonance_metrics["per_digit_resonance"])
            )
            print(f"digit_resonance {per_digit}")

    if args.save_path:
        save_checkpoint(model, optimizer, args, args.save_path)
        print(f"Saved checkpoint to {args.save_path}")


if __name__ == "__main__":
    main()
