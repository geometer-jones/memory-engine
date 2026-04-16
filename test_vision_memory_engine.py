from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from vision_memory_engine import VisionMemoryRecognizer


def _make_model(**overrides) -> VisionMemoryRecognizer:
    defaults = dict(
        image_size=8,
        patch_size=2,
        in_channels=1,
        hidden_dim=16,
        num_classes=2,
        dropout=0.0,
        memory_dim=16,
    )
    defaults.update(overrides)
    torch.manual_seed(0)
    return VisionMemoryRecognizer(**defaults)


def _make_stripe_batch(
    batch_size: int,
    image_size: int = 8,
    noise_scale: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    images = torch.randn(batch_size, 1, image_size, image_size) * noise_scale
    labels = torch.randint(0, 2, (batch_size,))

    for idx, label in enumerate(labels.tolist()):
        if label == 0:
            col = torch.randint(1, image_size - 1, (1,)).item()
            images[idx, 0, :, col] += 1.0
        else:
            row = torch.randint(1, image_size - 1, (1,)).item()
            images[idx, 0, row, :] += 1.0

    return images, labels


def test_vision_memory_recognizer_shapes_and_diagnostics():
    model = _make_model()
    images = torch.randn(3, 1, 8, 8)

    logits, state, diagnostics = model(images, return_diagnostics=True)

    assert logits.shape == (3, 2)
    assert state.tape.shape[0] == 3
    assert diagnostics["pr"].shape == (3, 16)
    assert diagnostics["prediction_error"].shape == (3, 16)
    assert diagnostics["active_slots"].shape == (3,)


def test_vision_memory_recognizer_backpropagates():
    model = _make_model()
    images = torch.randn(4, 1, 8, 8)
    labels = torch.randint(0, 2, (4,))

    logits = model(images)
    loss = F.cross_entropy(logits, labels)
    loss.backward()

    assert model.patch_embed.weight.grad is not None
    assert model.memory.eta.grad is not None
    assert model.memory.alpha.grad is not None
    assert model.classifier.weight.grad is not None


def test_vision_memory_recognizer_conv_stem_backpropagates():
    model = _make_model(conv_stem_channels=(8, 8))
    images = torch.randn(4, 1, 8, 8)
    labels = torch.randint(0, 2, (4,))

    logits = model(images)
    loss = F.cross_entropy(logits, labels)
    loss.backward()

    assert isinstance(model.conv_stem, nn.Sequential)
    assert model.conv_stem[0].weight.grad is not None
    assert model.patch_embed.weight.grad is not None


def test_vision_memory_recognizer_learns_simple_orientation_task():
    torch.manual_seed(0)
    model = _make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for _ in range(80):
        images, labels = _make_stripe_batch(batch_size=32)
        optimizer.zero_grad()
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        images, labels = _make_stripe_batch(batch_size=128)
        predictions = model(images).argmax(dim=-1)
        accuracy = (predictions == labels).float().mean().item()

    assert accuracy >= 0.9, f"Expected >= 0.9 accuracy on the stripe task, got {accuracy:.3f}"
