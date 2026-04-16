"""Unified hierarchical Memory Engine for MNIST digit recognition.

This module keeps the existing Memory Engine dynamics intact at the node level
while composing them into a single low -> mid -> high hierarchy suitable for
image classification.

Architecture:
    - Low level: multiple local MemoryEngineNodes process explicit spatial patch
      groups cut from the image.
    - Mid level: a smaller set of nodes bind groups of low-level tapes.
    - High level: one node integrates the full digit and emits a final tape.
    - Readout: linear classifier on the final high-level tape.

Each node is a full Memory Engine with its own:
    - tape ``s``
    - basis ``E``
    - coupling ``L = G^-1``
    - coupled reception ``alpha = L @ w`` and ``c = alpha ⊙ s``
    - fast binding / consolidation / recurrence logic

The implementation reuses the enhanced node from ``memory_engine_node.py`` so
the learnable metaparams stay consistent across the repo.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from memory_engine_node import MemoryEngineNode as _BaseMemoryEngineNode


Tensor = torch.Tensor


class MemoryEngineNode(_BaseMemoryEngineNode):
    """Re-export the enhanced node with learnable metaparams.

    The existing implementation already makes the requested controls learnable:
    ``eta``, ``lambda``, recurrence gain, binding scale, transient decay, and
    soft thresholds for binding / merging / pruning.
    """

    pass


def _tensor_mean(value: Any) -> Optional[Tensor]:
    """Convert diagnostic values to scalar tensors when possible."""
    if not torch.is_tensor(value):
        return None
    if value.numel() == 0:
        return None
    if value.is_complex():
        value = value.real
    value = value.float()
    return value.mean()


class MNISTMemoryEngine(nn.Module):
    """Single unified hierarchical Memory Engine for MNIST.

    Default hierarchy:
        - 8 low-level nodes, dim 128
        - 3 mid-level nodes, dim 256
        - 1 high-level node, dim 512

    The low-level nodes receive genuine subsets of image patches. Mid-level
    nodes receive projected low-level tapes, and the high-level node receives
    projected mid-level tapes. The classifier reads from the final high-level
    tape.
    """

    def __init__(
        self,
        image_size: int = 28,
        patch_shape: int | Tuple[int, int] = (7, 7),
        num_classes: int = 10,
        low_nodes: int = 8,
        mid_nodes: int = 3,
        low_dim: int = 128,
        mid_dim: int = 256,
        high_dim: int = 512,
        low_aux_dims: int = 2,
        mid_aux_dims: int = 2,
        high_aux_dims: int = 2,
        low_transient_dims: int = 0,
        mid_transient_dims: int = 0,
        high_transient_dims: int = 0,
    ) -> None:
        super().__init__()
        patch_height, patch_width = self._normalize_patch_shape(patch_shape)
        if image_size % patch_height != 0 or image_size % patch_width != 0:
            raise ValueError("image_size must be divisible by the patch shape.")
        if low_nodes < mid_nodes:
            raise ValueError("low_nodes must be >= mid_nodes.")

        self.image_size = image_size
        self.patch_shape = (patch_height, patch_width)
        self.num_classes = num_classes
        self.low_node_count = low_nodes
        self.mid_node_count = mid_nodes
        self.low_dim = low_dim
        self.mid_dim = mid_dim
        self.high_dim = high_dim

        self.patch_rows = image_size // patch_height
        self.patch_cols = image_size // patch_width
        self.num_patches = self.patch_rows * self.patch_cols
        self.patch_dim = patch_height * patch_width

        # Shared image encoder before routing into low-level specialists.
        self.patch_embed = nn.Linear(self.patch_dim, low_dim, bias=False)
        self.patch_norm = nn.LayerNorm(low_dim)
        self.position_embedding = nn.Parameter(
            torch.randn(1, self.num_patches, low_dim) * 0.02
        )

        self.low_patch_groups = self._build_spatial_patch_groups(
            patch_rows=self.patch_rows,
            patch_cols=self.patch_cols,
            low_nodes=low_nodes,
        )
        self.low_input_adapters = nn.ModuleList(
            [nn.Linear(low_dim, low_dim, bias=False) for _ in range(low_nodes)]
        )

        self.low_nodes = nn.ModuleList(
            [
                MemoryEngineNode(
                    node_name=f"low_{idx}",
                    input_dim=low_dim,
                    hidden_dim=low_dim,
                    memory_dim=low_dim,
                    max_aux_dims=low_aux_dims,
                    max_transient_dims=low_transient_dims,
                    eta_init=0.10,
                    alpha_init=0.50,
                    self_recurrence_delay=1,
                    self_recurrence_weight=0.10,
                    corr_ema=0.05,
                    beta=0.05,
                    gamma=0.90,
                    theta_merge=0.40,
                    theta_prune=0.015,
                    theta_bind_init=0.15,
                )
                for idx in range(low_nodes)
            ]
        )

        self.low_to_mid = nn.ModuleList(
            [
                nn.Linear(2 * node.state_dim, mid_dim, bias=False)
                for node in self.low_nodes
            ]
        )

        self.mid_groups = self._build_groups(low_nodes=low_nodes, mid_nodes=mid_nodes)
        self.mid_nodes = nn.ModuleList(
            [
                MemoryEngineNode(
                    node_name=f"mid_{idx}",
                    input_dim=mid_dim,
                    hidden_dim=mid_dim,
                    memory_dim=mid_dim,
                    max_aux_dims=mid_aux_dims,
                    max_transient_dims=mid_transient_dims,
                    eta_init=0.08,
                    alpha_init=0.50,
                    self_recurrence_delay=1,
                    self_recurrence_weight=0.12,
                    corr_ema=0.04,
                    beta=0.05,
                    gamma=0.92,
                    theta_merge=0.45,
                    theta_prune=0.020,
                    theta_bind_init=0.18,
                )
                for idx in range(mid_nodes)
            ]
        )

        self.mid_to_high = nn.ModuleList(
            [
                nn.Linear(2 * node.state_dim, high_dim, bias=False)
                for node in self.mid_nodes
            ]
        )
        self.high_node = MemoryEngineNode(
            node_name="high",
            input_dim=high_dim,
            hidden_dim=high_dim,
            memory_dim=high_dim,
            max_aux_dims=high_aux_dims,
            max_transient_dims=high_transient_dims,
            eta_init=0.06,
            alpha_init=0.50,
            self_recurrence_delay=2,
            self_recurrence_weight=0.15,
            corr_ema=0.04,
            beta=0.05,
            gamma=0.94,
            theta_merge=0.50,
            theta_prune=0.025,
            theta_bind_init=0.20,
        )

        self.readout_norm = nn.LayerNorm(2 * self.high_node.state_dim)
        self.classifier = nn.Linear(2 * self.high_node.state_dim, num_classes)

    def reset_states(self) -> None:
        """Clear any persistent state carried by the nodes."""
        for node in self.iter_nodes():
            node.clear_persistent_state()

    def iter_nodes(self) -> Iterable[MemoryEngineNode]:
        """Iterate over every Memory Engine node in the hierarchy."""
        for node in self.low_nodes:
            yield node
        for node in self.mid_nodes:
            yield node
        yield self.high_node

    def patchify(self, images: Tensor) -> Tensor:
        """Convert images into a flat sequence of patches."""
        if images.ndim != 4:
            raise ValueError("Expected images of shape (batch, channels, height, width).")
        if images.shape[-1] != self.image_size or images.shape[-2] != self.image_size:
            raise ValueError("Unexpected image size for this hierarchy.")
        patches = F.unfold(
            images,
            kernel_size=self.patch_shape,
            stride=self.patch_shape,
        ).transpose(1, 2)
        return patches

    def encode_patches(self, images: Tensor) -> Tensor:
        """Shared patch encoder before routing to the low-level nodes."""
        patches = self.patchify(images)
        embedded = self.patch_embed(patches) + self.position_embedding
        return self.patch_norm(embedded)

    def forward(
        self,
        images: Tensor,
        return_metrics: bool = False,
    ) -> Dict[str, Any]:
        """Run the full low -> mid -> high hierarchy on a batch of images."""
        self.reset_states()
        base_sequence = self.encode_patches(images)

        node_summaries: List[Dict[str, Any]] = []
        low_messages: List[Tensor] = []

        for idx, node in enumerate(self.low_nodes):
            patch_indices = self.low_patch_groups[idx]
            routed_input = base_sequence[:, patch_indices, :]
            routed_input = self.low_input_adapters[idx](routed_input)

            if return_metrics:
                _, state, diagnostics = node(
                    hidden_states=routed_input,
                    persist_state=False,
                    return_state=True,
                    return_diagnostics=True,
                )
                node_summaries.append(
                    {
                        "level": "low",
                        "name": node.node_name,
                        "summary": node.summarize_diagnostics(diagnostics),
                    }
                )
            else:
                _, state = node(
                    hidden_states=routed_input,
                    persist_state=False,
                    return_state=True,
                )

            low_messages.append(self.low_to_mid[idx](node.tape_features(state)))

        low_message_stack = torch.stack(low_messages, dim=1)
        mid_messages: List[Tensor] = []

        for idx, node in enumerate(self.mid_nodes):
            group_indices = self.mid_groups[idx]
            incoming = low_message_stack[:, group_indices, :]
            if return_metrics:
                _, state, diagnostics = node(
                    incoming_signals=incoming,
                    persist_state=False,
                    return_state=True,
                    return_diagnostics=True,
                )
                node_summaries.append(
                    {
                        "level": "mid",
                        "name": node.node_name,
                        "summary": node.summarize_diagnostics(diagnostics),
                    }
                )
            else:
                _, state = node(
                    incoming_signals=incoming,
                    persist_state=False,
                    return_state=True,
                )

            mid_messages.append(self.mid_to_high[idx](node.tape_features(state)))

        high_incoming = torch.stack(mid_messages, dim=1)
        if return_metrics:
            _, high_state, high_diag = self.high_node(
                incoming_signals=high_incoming,
                persist_state=False,
                return_state=True,
                return_diagnostics=True,
            )
            node_summaries.append(
                {
                    "level": "high",
                    "name": self.high_node.node_name,
                    "summary": self.high_node.summarize_diagnostics(high_diag),
                }
            )
        else:
            _, high_state = self.high_node(
                incoming_signals=high_incoming,
                persist_state=False,
                return_state=True,
            )

        high_tape = self.high_node.tape_features(high_state)
        logits = self.classifier(self.readout_norm(high_tape))

        result: Dict[str, Any] = {
            "logits": logits,
            "high_tape": high_tape,
        }
        if return_metrics:
            result["metrics"] = self._aggregate_metrics(node_summaries)
        return result

    def metaparam_summary(self) -> Dict[str, float]:
        """Average learnable metaparams across the hierarchy."""
        snapshots = [node.metaparam_values() for node in self.iter_nodes()]
        keys = snapshots[0].keys()
        return {
            key: float(
                torch.stack(
                    [snapshot[key].reshape(1).float() for snapshot in snapshots],
                    dim=0,
                ).mean().detach().cpu()
            )
            for key in keys
        }

    def _aggregate_metrics(
        self,
        node_summaries: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        """Build detached logging metrics from per-node diagnostics."""
        pr_values: List[Tensor] = []
        resonance_values: List[Tensor] = []
        torque_values: List[Tensor] = []
        level_accumulator: Dict[str, Dict[str, List[Tensor]]] = {}

        for item in node_summaries:
            level = str(item["level"])
            summary = item["summary"]

            pr = _tensor_mean(summary.get("pr"))
            resonance = _tensor_mean(summary.get("resonance_fraction"))
            torque = _tensor_mean(summary.get("torque_fraction"))

            level_accumulator.setdefault(
                level,
                {"pr": [], "resonance_fraction": [], "torque_fraction": []},
            )

            if pr is not None:
                pr_values.append(pr.detach())
                level_accumulator[level]["pr"].append(pr.detach())
            if resonance is not None:
                resonance_values.append(resonance.detach())
                level_accumulator[level]["resonance_fraction"].append(resonance.detach())
            if torque is not None:
                torque_values.append(torque.detach())
                level_accumulator[level]["torque_fraction"].append(torque.detach())

        metrics: Dict[str, Any] = {
            "mean_pr": float(torch.stack(pr_values).mean().cpu()) if pr_values else 0.0,
            "mean_resonance_fraction": (
                float(torch.stack(resonance_values).mean().cpu())
                if resonance_values
                else 0.0
            ),
            "mean_torque_fraction": (
                float(torch.stack(torque_values).mean().cpu()) if torque_values else 0.0
            ),
            "metaparams": self.metaparam_summary(),
            "per_level": {},
        }

        for level, values in level_accumulator.items():
            metrics["per_level"][level] = {
                key: (
                    float(torch.stack(value_list).mean().cpu()) if value_list else 0.0
                )
                for key, value_list in values.items()
            }

        return metrics

    @staticmethod
    def _build_groups(low_nodes: int, mid_nodes: int) -> List[List[int]]:
        """Partition low nodes into ordered groups for the mid level."""
        group_size = math.ceil(low_nodes / mid_nodes)
        groups: List[List[int]] = []
        for idx in range(mid_nodes):
            start = idx * group_size
            stop = min(start + group_size, low_nodes)
            groups.append(list(range(start, stop)))
        return groups

    @staticmethod
    def _normalize_patch_shape(
        patch_shape: int | Tuple[int, int],
    ) -> Tuple[int, int]:
        """Normalize square or rectangular patch arguments."""
        if isinstance(patch_shape, int):
            return patch_shape, patch_shape
        if len(patch_shape) != 2:
            raise ValueError("patch_shape must be an int or a (height, width) pair.")
        return int(patch_shape[0]), int(patch_shape[1])

    @staticmethod
    def _build_spatial_patch_groups(
        patch_rows: int,
        patch_cols: int,
        low_nodes: int,
    ) -> List[List[int]]:
        """Assign each image patch to one spatially local low-level node."""
        if low_nodes > patch_rows * patch_cols:
            raise ValueError("low_nodes cannot exceed the number of available patches.")

        rows = max(1, int(math.sqrt(low_nodes)))
        cols = math.ceil(low_nodes / rows)
        center_rows = torch.linspace(0, patch_rows - 1, rows)
        center_cols = torch.linspace(0, patch_cols - 1, cols)

        centers: List[Tuple[float, float]] = []
        for row in center_rows.tolist():
            for col in center_cols.tolist():
                centers.append((row, col))
                if len(centers) == low_nodes:
                    break
            if len(centers) == low_nodes:
                break

        groups: List[List[int]] = [[] for _ in range(low_nodes)]
        for row in range(patch_rows):
            for col in range(patch_cols):
                distances = [
                    (row - center_row) ** 2 + (col - center_col) ** 2
                    for center_row, center_col in centers
                ]
                owner = int(torch.tensor(distances).argmin().item())
                groups[owner].append(row * patch_cols + col)

        return groups


__all__ = ["MNISTMemoryEngine", "MemoryEngineNode"]
