from __future__ import annotations

from types import SimpleNamespace

import torch

from scripts.train_mnist_memory_engine import (
    build_datasets,
    build_loaders,
    build_model,
    collect_digit_prototypes,
    compute_digit_resonance_scores,
    evaluate,
    evaluate_digit_resonance,
    train_one_epoch,
)


def _make_args(**overrides):
    defaults = dict(
        image_size=28,
        patch_size=4,
        hidden_dim=32,
        memory_dim=None,
        dropout=0.0,
        stem_channels="8,16",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_build_fake_mnist_pipeline_matches_model():
    train_dataset, test_dataset = build_datasets(
        data_dir="./data",
        image_size=28,
        fake_data=True,
        train_limit=32,
        test_limit=16,
    )
    train_loader, _ = build_loaders(train_dataset, test_dataset, batch_size=8, eval_batch_size=8)
    model = build_model(_make_args())

    images, labels = next(iter(train_loader))
    logits, _, diagnostics = model(images, return_diagnostics=True)

    assert logits.shape == (8, 10)
    assert labels.shape == (8,)
    assert diagnostics["pr"].shape == (8, model.num_patches)
    assert tuple(model.conv_stem_channels) == (8, 16)


def test_digit_resonance_scores_prefer_matching_prototype():
    prototypes = torch.tensor(
        [
            [1.0 + 0.0j, 1.0 + 0.0j],
            [-1.0 + 0.0j, 1.0 + 0.0j],
        ],
        dtype=torch.complex64,
    )
    prototypes = prototypes / prototypes.abs().pow(2).sum(dim=-1, keepdim=True).sqrt()

    tape = torch.tensor(
        [
            [1.0 + 0.0j, 1.0 + 0.0j],
            [-1.0 + 0.0j, 1.0 + 0.0j],
        ],
        dtype=torch.complex64,
    )
    tape = tape / tape.abs().pow(2).sum(dim=-1, keepdim=True).sqrt()

    scores = compute_digit_resonance_scores(
        tape=tape,
        prototypes=prototypes,
        prototype_counts=torch.tensor([1, 1]),
    )

    assert scores[0, 0] > scores[0, 1]
    assert scores[1, 1] > scores[1, 0]


def test_fake_mnist_training_smoke_runs():
    train_dataset, test_dataset = build_datasets(
        data_dir="./data",
        image_size=28,
        fake_data=True,
        train_limit=64,
        test_limit=32,
    )
    train_loader, test_loader = build_loaders(train_dataset, test_dataset, batch_size=16, eval_batch_size=16)
    model = build_model(_make_args())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    device = torch.device("cpu")

    train_metrics = train_one_epoch(model, train_loader, optimizer, device)
    eval_metrics = evaluate(model, test_loader, device)

    assert 0.0 <= train_metrics["accuracy"] <= 1.0
    assert 0.0 <= eval_metrics["accuracy"] <= 1.0
    assert train_metrics["loss"] >= 0.0
    assert eval_metrics["loss"] >= 0.0
    assert "pr" in eval_metrics
    assert "prediction_error" in eval_metrics
    prototypes, prototype_counts = collect_digit_prototypes(model, train_loader, device)
    resonance_metrics = evaluate_digit_resonance(
        model,
        test_loader,
        device,
        prototypes=prototypes,
        prototype_counts=prototype_counts,
    )
    assert "digit_probe_accuracy" in resonance_metrics
    assert "true_digit_resonance" in resonance_metrics
    assert len(resonance_metrics["per_digit_resonance"]) == 10
