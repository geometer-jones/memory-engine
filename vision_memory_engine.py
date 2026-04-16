"""
Vision classifier built on top of the production Memory Engine layer.

The integration is deliberately simple:

1. patchify an image with a strided convolution
2. treat the patches as a sequence of tokens
3. run that sequence through ``MemoryEngineLayer``
4. pool the memory-conditioned token stream for classification

This keeps the image path architecture-aligned with the rest of the repository:
the Memory Engine still operates over sequences, only the token source changes.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from memory_engine_layer import MemoryEngineLayer, MemoryEngineState


def _pair(value: int | Tuple[int, int]) -> Tuple[int, int]:
    if isinstance(value, tuple):
        return value
    return (value, value)


def _make_conv_stem(in_channels: int, stem_channels: Tuple[int, ...]) -> Tuple[nn.Module, int]:
    if not stem_channels:
        return nn.Identity(), in_channels

    layers = []
    current_channels = in_channels
    for out_channels in stem_channels:
        layers.append(nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.GELU())
        current_channels = out_channels
    return nn.Sequential(*layers), current_channels


class VisionMemoryRecognizer(nn.Module):
    """Patch-based image recognizer that reuses ``MemoryEngineLayer``."""

    def __init__(
        self,
        image_size: int | Tuple[int, int] = 32,
        patch_size: int | Tuple[int, int] = 4,
        in_channels: int = 3,
        hidden_dim: int = 64,
        num_classes: int = 10,
        dropout: float = 0.0,
        conv_stem_channels: Tuple[int, ...] = (),
        **memory_kwargs,
    ) -> None:
        super().__init__()
        self.image_size = _pair(image_size)
        self.patch_size = _pair(patch_size)
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.conv_stem_channels = tuple(conv_stem_channels)

        image_h, image_w = self.image_size
        patch_h, patch_w = self.patch_size
        if image_h % patch_h != 0 or image_w % patch_w != 0:
            raise ValueError(
                "image_size must be divisible by patch_size so images map to a fixed patch sequence."
            )

        self.grid_size = (image_h // patch_h, image_w // patch_w)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.conv_stem, stem_out_channels = _make_conv_stem(in_channels, self.conv_stem_channels)
        self.patch_embed = nn.Conv2d(
            stem_out_channels,
            hidden_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.num_patches, hidden_dim))
        self.input_norm = nn.LayerNorm(hidden_dim)
        default_memory_kwargs = {
            "memory_dim": hidden_dim,
            "max_aux_dims": 0,
            "max_transient_dims": 0,
            "consolidation_interval": 0,
        }
        default_memory_kwargs.update(memory_kwargs)
        self.memory = MemoryEngineLayer(hidden_dim=hidden_dim, **default_memory_kwargs)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.conv_stem.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.kaiming_normal_(self.patch_embed.weight, nonlinearity="linear")
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)
        nn.init.trunc_normal_(self.position_embeddings, std=0.02)
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def _image_to_tokens(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(f"Expected images with shape (batch, channels, height, width), got {tuple(images.shape)}.")

        _, _, height, width = images.shape
        if (height, width) != self.image_size:
            raise ValueError(
                f"Expected image size {self.image_size}, got {(height, width)}. "
                "This module uses a fixed positional grid."
            )

        images = self.conv_stem(images)
        tokens = self.patch_embed(images)
        tokens = tokens.flatten(2).transpose(1, 2)
        tokens = tokens + self.position_embeddings
        return self.input_norm(tokens)

    def forward_features(
        self,
        images: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, MemoryEngineState, Dict]:
        tokens = self._image_to_tokens(images)
        memory_tokens, state, diagnostics = self.memory(
            tokens,
            state=None,
            return_diagnostics=return_diagnostics,
        )
        pooled = self.output_norm(memory_tokens.mean(dim=1))
        return pooled, state, diagnostics

    def forward(
        self,
        images: torch.Tensor,
        return_diagnostics: bool = False,
    ):
        pooled, state, diagnostics = self.forward_features(
            images,
            return_diagnostics=return_diagnostics,
        )
        logits = self.classifier(self.dropout(pooled))
        if return_diagnostics:
            return logits, state, diagnostics
        return logits
