"""MNIST specialist Memory Engine training script.

This file implements a one-specialist-per-digit architecture:
    - 10 independent Memory Engine hierarchies
    - each specialist sees only its own digit during training
    - inference evaluates all specialists and picks the highest resonance score

The resonance score is intentionally interpretable. A specialist should win
when:
    1. its final tape aligns with that specialist's learned class prototype,
    2. its internal dynamics stay in a resonance-dominant regime,
    3. torque / surprise stay comparatively low,
    4. the tape does not collapse into a degenerate single-axis state.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from memory_engine_node import HierarchicalMemoryEngine


Tensor = torch.Tensor


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


class DigitSpecialist(nn.Module):
    """One dedicated Memory Engine hierarchy for one digit."""

    def __init__(
        self,
        digit: int,
        patch_size: int = 7,
        hierarchy_dims: Sequence[int] = (24, 32, 40),
        aux_dims: Sequence[int] = (2, 2, 2),
        transient_dims: Sequence[int] = (0, 0, 0),
        enable_top_down: bool = False,
    ) -> None:
        super().__init__()
        if len(hierarchy_dims) != 3 or len(aux_dims) != 3 or len(transient_dims) != 3:
            raise ValueError("Expected 3 dims for low/mid/high specialist levels.")

        self.digit = digit
        self.patch_size = patch_size
        self.num_patches = (28 // patch_size) ** 2
        self.patch_dim = patch_size * patch_size
        self.low_dim, self.mid_dim, self.high_dim = hierarchy_dims

        self.patch_embed = nn.Linear(self.patch_dim, self.low_dim, bias=False)
        self.position_embedding = nn.Parameter(
            torch.randn(1, self.num_patches, self.low_dim) * 0.02
        )
        self.input_norm = nn.LayerNorm(self.low_dim)

        level_configs = [
            {
                "name": "low",
                "input_dim": self.low_dim,
                "hidden_dim": self.low_dim,
                "memory_dim": self.low_dim,
                "max_aux_dims": aux_dims[0],
                "max_transient_dims": transient_dims[0],
                "eta_init": 0.10,
                "alpha_init": 0.50,
                "self_recurrence_delay": 1,
                "self_recurrence_weight": 0.10,
                "corr_ema": 0.05,
                "beta": 0.05,
                "gamma": 0.90,
                "theta_merge": 0.40,
                "theta_prune": 0.015,
                "theta_bind_init": 0.15,
            },
            {
                "name": "mid",
                "hidden_dim": self.mid_dim,
                "memory_dim": self.mid_dim,
                "max_aux_dims": aux_dims[1],
                "max_transient_dims": transient_dims[1],
                "eta_init": 0.08,
                "alpha_init": 0.50,
                "self_recurrence_delay": 1,
                "self_recurrence_weight": 0.12,
                "corr_ema": 0.05,
                "beta": 0.05,
                "gamma": 0.92,
                "theta_merge": 0.45,
                "theta_prune": 0.020,
                "theta_bind_init": 0.18,
            },
            {
                "name": "high",
                "hidden_dim": self.high_dim,
                "memory_dim": self.high_dim,
                "max_aux_dims": aux_dims[2],
                "max_transient_dims": transient_dims[2],
                "eta_init": 0.06,
                "alpha_init": 0.50,
                "self_recurrence_delay": 2,
                "self_recurrence_weight": 0.15,
                "corr_ema": 0.04,
                "beta": 0.05,
                "gamma": 0.94,
                "theta_merge": 0.50,
                "theta_prune": 0.025,
                "theta_bind_init": 0.20,
            },
        ]
        self.hierarchy = HierarchicalMemoryEngine(
            level_configs=level_configs,
            enable_top_down=enable_top_down,
        )

        self.prototype = nn.Parameter(torch.randn(self.high_dim) * 0.02)
        self.prototype_gain = nn.Parameter(torch.tensor(1.0))

    def clear_state(self) -> None:
        """Clear all persistent tape/history before a new image batch."""
        self.hierarchy.clear_persistent_state()

    def patchify(self, images: Tensor) -> Tensor:
        """Convert images into a sequence of non-overlapping patches."""
        patches = F.unfold(
            images,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        ).transpose(1, 2)
        return patches

    def encode(self, images: Tensor) -> Tensor:
        """Patch embedding path unique to this specialist."""
        patches = self.patchify(images)
        embedded = self.patch_embed(patches) + self.position_embedding
        return self.input_norm(embedded)

    def forward(self, images: Tensor) -> Dict[str, Any]:
        """Run one specialist hierarchy on a batch of images."""
        sequence = self.encode(images)
        hierarchy_result = self.hierarchy(
            sequence,
            return_states=True,
            return_diagnostics=True,
        )
        high_name = self.hierarchy.level_names[-1]
        high_features = hierarchy_result["outputs"][high_name].mean(dim=1)
        score, metrics = self.compute_resonance_score(high_features, hierarchy_result)
        return {
            "score": score,
            "features": high_features,
            "hierarchy": hierarchy_result,
            "metrics": metrics,
        }

    def compute_resonance_score(
        self,
        features: Tensor,
        hierarchy_result: Dict[str, Any],
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Compute the specialist's resonance score.

        Score components:
            - prototype alignment: does the final tape match the digit attractor?
            - resonance fraction: do the nodes settle into resonance-dominant flow?
            - torque penalty: specialists fighting the input should lose.
            - prediction-error penalty: high surprise implies poor fit.
            - PR bonus: reward a non-collapsed operating regime.
        """
        summaries = hierarchy_result["summary"]
        resonance_terms: List[Tensor] = []
        torque_terms: List[Tensor] = []
        prediction_terms: List[Tensor] = []
        pr_terms: List[Tensor] = []

        for name in self.hierarchy.level_names:
            node_summary = summaries[name]
            resonance_terms.append(node_summary["resonance_fraction"])
            torque_terms.append(node_summary["torque_fraction"])
            prediction_terms.append(
                node_summary.get(
                    "prediction_error",
                    torch.zeros_like(node_summary["resonance_fraction"]),
                )
            )
            pr_terms.append(node_summary["pr"])

        mean_resonance = torch.stack(resonance_terms, dim=0).mean(dim=0)
        mean_torque = torch.tanh(torch.stack(torque_terms, dim=0).mean(dim=0))
        mean_prediction_error = torch.tanh(
            torch.stack(prediction_terms, dim=0).mean(dim=0)
        )
        mean_pr = torch.stack(pr_terms, dim=0).mean(dim=0)

        normalized_features = F.normalize(features, dim=-1)
        normalized_prototype = F.normalize(self.prototype, dim=0)
        prototype_alignment = F.cosine_similarity(
            normalized_features,
            normalized_prototype.unsqueeze(0),
            dim=-1,
        )

        pr_target = 4.0
        pr_bonus = torch.exp(-((mean_pr - pr_target) / pr_target).pow(2))
        gain = F.softplus(self.prototype_gain)
        score = (
            gain * prototype_alignment
            + mean_resonance
            + 0.25 * pr_bonus
            - 0.50 * mean_torque
            - 0.25 * mean_prediction_error
        )

        return score, {
            "score": score,
            "prototype_alignment": prototype_alignment,
            "mean_resonance": mean_resonance,
            "mean_torque": mean_torque,
            "mean_prediction_error": mean_prediction_error,
            "mean_pr": mean_pr,
            "pr_bonus": pr_bonus,
        }

    def positive_loss(self, forward_result: Dict[str, Any]) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Positive-only specialist loss for that digit's own images."""
        metrics = forward_result["metrics"]
        alignment = metrics["prototype_alignment"]
        feature_energy = forward_result["features"].pow(2).mean(dim=-1)
        gain = F.softplus(self.prototype_gain)

        # Training stays on the real-valued emitted hierarchy summary, which
        # avoids the wrapped engine's in-place complex-state diagnostics during
        # backprop while still forcing metaparams to shape the specialist
        # representation through the node outputs.
        feature_reg = 0.02 * (feature_energy - 1.0).pow(2).mean()
        gain_reg = 0.01 * (gain - 1.0).pow(2)
        loss = (1.0 - gain * alignment).mean() + feature_reg + gain_reg
        return loss, {
            "loss": loss.detach(),
            "operating_reg": (feature_reg + gain_reg).detach(),
            "score": metrics["score"].detach().mean(),
            "alignment": alignment.detach().mean(),
            "mean_resonance": metrics["mean_resonance"].detach().mean(),
            "mean_torque": metrics["mean_torque"].detach().mean(),
            "mean_pr": metrics["mean_pr"].detach().mean(),
        }

    def metaparam_snapshot(self) -> Dict[str, float]:
        """Average metaparam values across the hierarchy for logging."""
        per_level = self.hierarchy.metaparam_summary()
        keys = next(iter(per_level.values())).keys()
        return {
            key: sum(level[key] for level in per_level.values()) / len(per_level)
            for key in keys
        }


class MemoryEngineSpecialist(nn.Module):
    """Top-level container holding the 10 independent digit specialists."""

    def __init__(
        self,
        patch_size: int = 7,
        hierarchy_dims: Sequence[int] = (24, 32, 40),
        aux_dims: Sequence[int] = (2, 2, 2),
        transient_dims: Sequence[int] = (0, 0, 0),
        enable_top_down: bool = False,
    ) -> None:
        super().__init__()
        self.specialists = nn.ModuleList(
            [
                DigitSpecialist(
                    digit=digit,
                    patch_size=patch_size,
                    hierarchy_dims=hierarchy_dims,
                    aux_dims=aux_dims,
                    transient_dims=transient_dims,
                    enable_top_down=enable_top_down,
                )
                for digit in range(10)
            ]
        )

    def predict(self, images: Tensor) -> Tuple[Tensor, Tensor, List[Dict[str, Tensor]]]:
        """Run all specialists and pick the digit with the highest resonance."""
        all_scores: List[Tensor] = []
        all_metrics: List[Dict[str, Tensor]] = []

        for specialist in self.specialists:
            specialist.clear_state()
            result = specialist(images)
            all_scores.append(result["score"])
            all_metrics.append(result["metrics"])
            specialist.clear_state()

        score_matrix = torch.stack(all_scores, dim=-1)
        prediction = score_matrix.argmax(dim=-1)
        return prediction, score_matrix, all_metrics

    def specialist_optimizers(
        self,
        lr: float,
        weight_decay: float = 1e-4,
        epochs: int = 10,
    ) -> Tuple[List[torch.optim.Optimizer], List[CosineAnnealingLR]]:
        """Build one optimizer and one scheduler per specialist."""
        optimizers = [
            AdamW(specialist.parameters(), lr=lr, weight_decay=weight_decay)
            for specialist in self.specialists
        ]
        schedulers = [
            CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
            for optimizer in optimizers
        ]
        return optimizers, schedulers


@dataclass
class EpochStats:
    loss: float
    mean_resonance: float
    mean_torque: float
    mean_pr: float
    specialist_counts: List[int]


def train_epoch(
    model: MemoryEngineSpecialist,
    train_loader: DataLoader,
    optimizers: Sequence[torch.optim.Optimizer],
    device: torch.device,
    epoch: int,
    log_interval: int = 100,
    max_batches: Optional[int] = None,
) -> EpochStats:
    """Train by routing each sample only to its matching specialist."""
    model.train()
    running_loss = 0.0
    running_resonance = 0.0
    running_torque = 0.0
    running_pr = 0.0
    total_examples = 0
    specialist_counts = [0 for _ in range(10)]

    for batch_idx, (images, labels) in enumerate(train_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        batch_examples = 0
        batch_loss = 0.0

        for digit, specialist in enumerate(model.specialists):
            mask = labels == digit
            if not mask.any():
                continue

            subset_images = images[mask]
            optimizer = optimizers[digit]
            optimizer.zero_grad(set_to_none=True)
            specialist.clear_state()
            forward_result = specialist(subset_images)
            loss, metrics = specialist.positive_loss(forward_result)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(specialist.parameters(), 1.0)
            optimizer.step()
            specialist.clear_state()

            count = int(mask.sum().item())
            specialist_counts[digit] += count
            batch_examples += count
            batch_loss += float(loss.item()) * count
            running_resonance += float(metrics["mean_resonance"]) * count
            running_torque += float(metrics["mean_torque"]) * count
            running_pr += float(metrics["mean_pr"]) * count

            if batch_idx % log_interval == 0:
                meta = specialist.metaparam_snapshot()
                print(
                    f"epoch={epoch:02d} batch={batch_idx:04d} digit={digit} "
                    f"count={count:03d} loss={metrics['loss']:.4f} "
                    f"score={metrics['score']:.4f} res={metrics['mean_resonance']:.4f} "
                    f"torque={metrics['mean_torque']:.4f} pr={metrics['mean_pr']:.4f} "
                    f"eta={meta['eta']:.4f} lambda={meta['lambda']:.4f} "
                    f"beta={meta['beta']:.4f} gamma={meta['gamma']:.4f}"
                )

        running_loss += batch_loss
        total_examples += batch_examples

        if batch_idx % log_interval == 0 and batch_examples > 0:
            print(
                f"[train] epoch={epoch:02d} batch={batch_idx:04d} "
                f"loss={batch_loss / batch_examples:.4f} "
                f"seen={total_examples}"
            )

    normalizer = max(total_examples, 1)
    return EpochStats(
        loss=running_loss / normalizer,
        mean_resonance=running_resonance / normalizer,
        mean_torque=running_torque / normalizer,
        mean_pr=running_pr / normalizer,
        specialist_counts=specialist_counts,
    )


def evaluate(
    model: MemoryEngineSpecialist,
    data_loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
    log_scores: bool = True,
) -> Dict[str, Any]:
    """Evaluate by running every specialist on every image."""
    model.eval()
    total_examples = 0
    total_correct = 0
    mean_score_vector = torch.zeros(10, device=device)
    first_score_rows: Optional[Tensor] = None

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(data_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            predictions, score_matrix, _ = model.predict(images)

            total_examples += labels.numel()
            total_correct += int((predictions == labels).sum().item())
            mean_score_vector += score_matrix.mean(dim=0)

            if first_score_rows is None:
                first_score_rows = score_matrix[: min(8, score_matrix.shape[0])].detach().cpu()

    accuracy = total_correct / max(total_examples, 1)
    batches = max(min(len(data_loader), max_batches or len(data_loader)), 1)
    mean_score_vector = (mean_score_vector / batches).detach().cpu()

    if log_scores:
        formatted = " ".join(
            f"{digit}:{mean_score_vector[digit]:+.3f}" for digit in range(10)
        )
        print(f"[eval] mean specialist resonance scores {formatted}")
        if first_score_rows is not None:
            print("[eval] sample resonance rows:")
            for row in first_score_rows:
                row_text = " ".join(f"{value:+.3f}" for value in row.tolist())
                print(f"  {row_text}")

    return {
        "accuracy": accuracy,
        "mean_scores": mean_score_vector,
        "examples": total_examples,
    }


def save_checkpoint(
    path: str | Path,
    model: MemoryEngineSpecialist,
    optimizers: Sequence[torch.optim.Optimizer],
    schedulers: Sequence[CosineAnnealingLR],
    epoch: int,
) -> None:
    """Persist the full specialist system."""
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizers": [optimizer.state_dict() for optimizer in optimizers],
        "schedulers": [scheduler.state_dict() for scheduler in schedulers],
    }
    torch.save(checkpoint, Path(path))


def load_checkpoint(
    path: str | Path,
    model: MemoryEngineSpecialist,
    optimizers: Optional[Sequence[torch.optim.Optimizer]] = None,
    schedulers: Optional[Sequence[CosineAnnealingLR]] = None,
    map_location: str | torch.device = "cpu",
) -> int:
    """Restore model and optional optimizer/scheduler state."""
    checkpoint = torch.load(Path(path), map_location=map_location)
    model.load_state_dict(checkpoint["model"])
    if optimizers is not None:
        for optimizer, state_dict in zip(optimizers, checkpoint.get("optimizers", [])):
            optimizer.load_state_dict(state_dict)
    if schedulers is not None:
        for scheduler, state_dict in zip(schedulers, checkpoint.get("schedulers", [])):
            scheduler.load_state_dict(state_dict)
    return int(checkpoint.get("epoch", 0))


def main() -> None:
    parser = argparse.ArgumentParser(description="MNIST specialist Memory Engines")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patch-size", type=int, default=7)
    parser.add_argument("--save-path", default="mnist_specialist_me.pt")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--top-down", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = build_mnist_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )

    model = MemoryEngineSpecialist(
        patch_size=args.patch_size,
        enable_top_down=args.top_down,
    ).to(device)
    optimizers, schedulers = model.specialist_optimizers(
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
    )

    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(
            args.resume,
            model,
            optimizers=optimizers,
            schedulers=schedulers,
            map_location=device,
        )
        print(f"Resumed from epoch {start_epoch}")

    if args.eval_only:
        metrics = evaluate(
            model,
            test_loader,
            device=device,
            max_batches=args.max_eval_batches,
            log_scores=True,
        )
        print(f"test_accuracy={metrics['accuracy']:.4f}")
        return

    best_accuracy = 0.0
    for epoch in range(start_epoch, args.epochs):
        train_stats = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizers=optimizers,
            device=device,
            epoch=epoch,
            max_batches=args.max_train_batches,
        )
        for scheduler in schedulers:
            scheduler.step()

        eval_metrics = evaluate(
            model,
            test_loader,
            device=device,
            max_batches=args.max_eval_batches,
            log_scores=True,
        )
        best_accuracy = max(best_accuracy, eval_metrics["accuracy"])

        print(
            f"[epoch {epoch:02d}] "
            f"train_loss={train_stats.loss:.4f} "
            f"train_res={train_stats.mean_resonance:.4f} "
            f"train_torque={train_stats.mean_torque:.4f} "
            f"train_pr={train_stats.mean_pr:.4f} "
            f"test_acc={eval_metrics['accuracy']:.4f} "
            f"best_acc={best_accuracy:.4f}"
        )
        print(
            "[epoch counts] "
            + " ".join(f"{digit}:{count}" for digit, count in enumerate(train_stats.specialist_counts))
        )

        save_checkpoint(
            path=args.save_path,
            model=model,
            optimizers=optimizers,
            schedulers=schedulers,
            epoch=epoch + 1,
        )


if __name__ == "__main__":
    main()
