from __future__ import annotations

import pytest
import torch

from distill_me_llm import (
    compute_distillation_losses,
    project_hidden_to_dim,
    select_last_token_hidden,
    tokenizers_share_vocabulary,
    validate_distillation_setup,
)


class _FakeTokenizer:
    def __init__(self, vocab):
        self._vocab = vocab

    def get_vocab(self):
        return self._vocab


def test_tokenizer_compatibility_check_detects_vocab_mismatch():
    teacher = _FakeTokenizer({"a": 0, "b": 1})
    student = _FakeTokenizer({"a": 0, "c": 1})
    assert not tokenizers_share_vocabulary(teacher, student)

    with pytest.raises(ValueError):
        validate_distillation_setup(
            teacher_tokenizer=teacher,
            student_tokenizer=student,
            kl_weight=0.6,
        )


def test_hidden_projection_matches_requested_dim():
    hidden = torch.randn(2, 32)
    projected = project_hidden_to_dim(hidden, target_dim=16)
    assert projected.shape == (2, 16)


def test_select_last_token_hidden_uses_attention_mask():
    hidden_states = torch.tensor(
        [
            [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            [[4.0, 0.0], [5.0, 0.0], [6.0, 0.0]],
        ]
    )
    attention_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

    selected = select_last_token_hidden(hidden_states, attention_mask)
    assert torch.equal(selected, torch.tensor([[2.0, 0.0], [6.0, 0.0]]))


def test_compute_distillation_losses_combines_all_terms():
    student_logits = torch.tensor([[[2.5, 0.5], [0.5, 2.5]]], dtype=torch.float32)
    teacher_logits = torch.tensor(
        [[[3.0, 0.0], [0.0, 3.0]]],
        dtype=torch.float32,
    )
    labels = torch.tensor([[0, 1]])
    student_memory_hidden = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
    teacher_final_hidden = torch.tensor([[1.0, 0.0, 0.5, -0.5]], dtype=torch.float32)

    losses = compute_distillation_losses(
        student_outputs={
            "loss": None,
            "logits": student_logits,
            "final_memory_hidden": student_memory_hidden,
            "me_diagnostics": [
                {
                    "diagnostics": {
                        "pr": torch.tensor([[2.0, 2.0]]),
                        "active_slots": torch.tensor([8]),
                    }
                }
            ],
        },
        teacher_logits=teacher_logits,
        labels=labels,
        teacher_hidden=teacher_final_hidden,
        temperature=2.0,
        ce_weight=1.2,
        kl_weight=0.6,
        feature_weight=0.4,
        pr_reg_weight=0.05,
    )

    assert losses["ce"].item() > 0
    assert losses["kl"].item() > 0
    assert losses["feature"].item() > 0
    assert losses["pr_reg"].item() >= 0
    assert losses["total"].item() > losses["ce"].item()
