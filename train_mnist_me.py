"""Training and evaluation for the unified hierarchical MNIST Memory Engine."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from memory_engine_hierarchy import MNISTMemoryEngine


def build_mnist_loaders(
    data_dir: str,
    batch_size: int,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    """Create standard MNIST train/test loaders."""
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


def step_metrics(logits: torch.Tensor, labels: torch.Tensor, loss: torch.Tensor) -> Dict[str, float]:
    """Compute batch-level loss/accuracy metrics."""
    predictions = logits.argmax(dim=-1)
    accuracy = (predictions == labels).float().mean().item()
    return {
        "loss": float(loss.item()),
        "accuracy": float(accuracy),
    }


def _merge_metric_sums(
    totals: Dict[str, float],
    metrics: Dict[str, Any],
    batch_size: int,
) -> None:
    """Accumulate scalar logging metrics with sample weighting."""
    totals["count"] = totals.get("count", 0.0) + batch_size
    totals["loss"] = totals.get("loss", 0.0) + metrics["loss"] * batch_size
    totals["accuracy"] = totals.get("accuracy", 0.0) + metrics["accuracy"] * batch_size

    model_metrics = metrics["model_metrics"]
    totals["mean_pr"] = totals.get("mean_pr", 0.0) + model_metrics["mean_pr"] * batch_size
    totals["mean_resonance_fraction"] = totals.get(
        "mean_resonance_fraction", 0.0
    ) + model_metrics["mean_resonance_fraction"] * batch_size
    totals["mean_torque_fraction"] = totals.get(
        "mean_torque_fraction", 0.0
    ) + model_metrics["mean_torque_fraction"] * batch_size


def _finalize_metric_sums(totals: Dict[str, float]) -> Dict[str, float]:
    """Convert running weighted sums into means."""
    count = max(totals.get("count", 0.0), 1.0)
    return {
        "loss": totals.get("loss", 0.0) / count,
        "accuracy": totals.get("accuracy", 0.0) / count,
        "mean_pr": totals.get("mean_pr", 0.0) / count,
        "mean_resonance_fraction": totals.get("mean_resonance_fraction", 0.0) / count,
        "mean_torque_fraction": totals.get("mean_torque_fraction", 0.0) / count,
    }


def train_epoch(
    model: MNISTMemoryEngine,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_interval: int = 100,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """Train for one epoch with standard cross-entropy loss."""
    model.train()
    totals: Dict[str, float] = {}
    last_model_metrics: Dict[str, Any] = {}

    for batch_idx, (images, labels) in enumerate(data_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images, return_metrics=True)
        logits = outputs["logits"]
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        batch_metrics = step_metrics(logits, labels, loss)
        batch_metrics["model_metrics"] = outputs["metrics"]
        _merge_metric_sums(totals, batch_metrics, images.shape[0])
        last_model_metrics = outputs["metrics"]

        if batch_idx % log_interval == 0:
            meta = outputs["metrics"]["metaparams"]
            print(
                f"[train] epoch={epoch:02d} batch={batch_idx:04d} "
                f"loss={batch_metrics['loss']:.4f} acc={batch_metrics['accuracy']:.4f} "
                f"pr={outputs['metrics']['mean_pr']:.3f} "
                f"res={outputs['metrics']['mean_resonance_fraction']:.3f} "
                f"torque={outputs['metrics']['mean_torque_fraction']:.3f} "
                f"eta={meta['eta']:.4f} lambda={meta['lambda']:.4f} "
                f"w_r={meta['w_r']:.4f} beta={meta['beta']:.4f} gamma={meta['gamma']:.4f}"
            )

    aggregate = _finalize_metric_sums(totals)
    aggregate["model_metrics"] = last_model_metrics
    return aggregate


@torch.no_grad()
def evaluate(
    model: MNISTMemoryEngine,
    data_loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate the hierarchy on the MNIST test set."""
    model.eval()
    totals: Dict[str, float] = {}
    last_metrics: Dict[str, Any] = {}

    for batch_idx, (images, labels) in enumerate(data_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images, return_metrics=True)
        logits = outputs["logits"]
        loss = F.cross_entropy(logits, labels)
        batch_metrics = step_metrics(logits, labels, loss)
        batch_metrics["model_metrics"] = outputs["metrics"]
        _merge_metric_sums(totals, batch_metrics, images.shape[0])
        last_metrics = outputs["metrics"]

    aggregate = _finalize_metric_sums(totals)
    aggregate["model_metrics"] = last_metrics
    return aggregate


def save_checkpoint(
    path: str | Path,
    model: MNISTMemoryEngine,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    epoch: int,
) -> None:
    """Save model, optimizer, and scheduler state."""
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    torch.save(payload, Path(path))


def load_checkpoint(
    path: str | Path,
    model: MNISTMemoryEngine,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[CosineAnnealingLR] = None,
    map_location: str | torch.device = "cpu",
) -> int:
    """Restore a checkpoint and return the next epoch index."""
    payload = torch.load(Path(path), map_location=map_location)
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])
    return int(payload.get("epoch", 0))


def print_epoch_summary(prefix: str, metrics: Dict[str, Any]) -> None:
    """Emit a compact summary of accuracy, regime metrics, and metaparams."""
    model_metrics = metrics.get("model_metrics", {})
    meta = model_metrics.get("metaparams", {})
    print(
        f"[{prefix}] "
        f"loss={metrics['loss']:.4f} "
        f"acc={metrics['accuracy']:.4f} "
        f"mean_pr={metrics['mean_pr']:.3f} "
        f"mean_res={metrics['mean_resonance_fraction']:.3f} "
        f"mean_torque={metrics['mean_torque_fraction']:.3f}"
    )
    if meta:
        print(
            f"[{prefix} metaparams] "
            f"eta={meta['eta']:.4f} lambda={meta['lambda']:.4f} "
            f"w_r={meta['w_r']:.4f} beta={meta['beta']:.4f} gamma={meta['gamma']:.4f} "
            f"theta_bind={meta['theta_bind']:.4f} "
            f"theta_merge={meta['theta_merge']:.4f} "
            f"theta_prune={meta['theta_prune']:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MNIST unified Memory Engine")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--save-path", default="mnist_memory_engine.pt")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patch-height", type=int, default=7)
    parser.add_argument("--patch-width", type=int, default=7)
    parser.add_argument("--low-nodes", type=int, default=8)
    parser.add_argument("--mid-nodes", type=int, default=3)
    parser.add_argument("--low-dim", type=int, default=128)
    parser.add_argument("--mid-dim", type=int, default=256)
    parser.add_argument("--high-dim", type=int, default=512)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = build_mnist_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )

    model = MNISTMemoryEngine(
        patch_shape=(args.patch_height, args.patch_width),
        low_nodes=args.low_nodes,
        mid_nodes=args.mid_nodes,
        low_dim=args.low_dim,
        mid_dim=args.mid_dim,
        high_dim=args.high_dim,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
        )
        print(f"Resumed from epoch {start_epoch}")

    if args.eval_only:
        metrics = evaluate(
            model=model,
            data_loader=test_loader,
            device=device,
            max_batches=args.max_eval_batches,
        )
        print_epoch_summary("eval", metrics)
        return

    best_accuracy = 0.0
    for epoch in range(start_epoch, args.epochs):
        train_metrics = train_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            max_batches=args.max_train_batches,
        )
        scheduler.step()

        eval_metrics = evaluate(
            model=model,
            data_loader=test_loader,
            device=device,
            max_batches=args.max_eval_batches,
        )
        best_accuracy = max(best_accuracy, eval_metrics["accuracy"])

        print_epoch_summary(f"train {epoch:02d}", train_metrics)
        print_epoch_summary(f"eval  {epoch:02d}", eval_metrics)
        print(f"[epoch {epoch:02d}] best_acc={best_accuracy:.4f}")

        save_checkpoint(
            path=args.save_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch + 1,
        )


if __name__ == "__main__":
    main()
