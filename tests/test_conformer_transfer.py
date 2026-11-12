"""Behavioral checks; optional SpeechBrain environment is required for encoder tests."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch

from experiments.conformer.model import ConformerClassifier, load_source, transfer_module
from experiments.conformer.run import Clips
from src.config import Config

SOURCE = Path(__file__).resolve().parents[1] / "runs/conformer-research"


def test_reject_incomplete_or_wrong_shaped_pretrained_weights():
    layer = torch.nn.Linear(4, 3)
    with pytest.raises(ValueError, match="Missing pretrained"):
        transfer_module(layer, {}, "1.")
    with pytest.raises(ValueError, match="Incompatible pretrained"):
        transfer_module(layer, {"1.weight": torch.zeros(2, 4)}, "1.")


def test_dataset_does_not_silently_extend_short_clips(tmp_path):
    sf.write(tmp_path / "short.wav", np.zeros(16000), 16000)
    cfg = Config(labels=["command", "silence"], data_root=str(tmp_path))
    frame = pd.DataFrame([dict(path="short.wav", label=cfg.labels[0])])
    with pytest.raises(ValueError, match="1.5 s"):
        Clips(frame, cfg)


@pytest.fixture(scope="module")
def pretrained():
    pytest.importorskip("speechbrain")
    if not (SOURCE / "asr.ckpt").exists():
        pytest.skip("Pinned pretrained weights not downloaded")
    torch.set_num_threads(2)
    return load_source(SOURCE)


def test_short_inputs_and_frozen_training_preserve_encoder(pretrained):
    torch.manual_seed(7)
    model = ConformerClassifier(*pretrained, mode="frozen").train()
    before = {k: v.clone() for k, v in model.state_dict().items() if not k.startswith("fc.")}
    wav = torch.randn(2, 24000) * .02
    a, b = model(wav), model(wav)
    assert torch.equal(a, b)  # No stochastic encoder dropout in frozen mode.
    assert model.encode(wav).shape == (2,38,256)
    assert model(torch.zeros(2,16000)).shape == (2,11)
    assert torch.isfinite(a).all()
    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=5e-4)
    loss = torch.nn.functional.cross_entropy(a, torch.tensor([1,2]))
    loss.backward()
    assert model.fc.weight.grad is not None
    assert all(p.grad is None for p in model.frontend.parameters())
    assert all(p.grad is None for p in model.transformer.parameters())
    optimizer.step()
    assert all(torch.equal(model.state_dict()[k], v) for k, v in before.items())


def test_finetune_has_encoder_gradients_and_respects_no_grad(pretrained):
    model = ConformerClassifier(*pretrained, mode="finetune").train()
    mean, std = model.feature_mean.clone(), model.feature_std.clone()
    logits = model(torch.randn(2,24000) * .02)
    torch.nn.functional.cross_entropy(logits, torch.tensor([1,2])).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.frontend.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.transformer.parameters())
    assert torch.equal(model.feature_mean, mean) and torch.equal(model.feature_std, std)
    with torch.no_grad():
        encoded = model.encode(torch.zeros(2,24000))
        assert not encoded.requires_grad
