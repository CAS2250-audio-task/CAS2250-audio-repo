import pytest
import torch
import torch.nn.functional as F

from experiments.pretrain.weights import tensor_payload, transfer
from src.model import build_model


def test_speech_transfer_preserves_reference_head_and_selects_mel_channel():
    model = build_model(11, False)
    head = {k: v.clone() for k, v in model.fc.state_dict().items()}
    source = {k: v.clone() for k, v in model.state_dict().items() if not k.startswith("fc.")}
    source["conv1.weight"] = torch.randn(64, 3, 7, 7)
    source["layer4.1.bn2.running_mean"].fill_(1.25)
    audit = transfer(model, source)
    assert audit["copied_backbone_tensors"] == 120
    assert all(torch.equal(v, model.fc.state_dict()[k]) for k, v in head.items())
    assert torch.equal(model.layer4[1].bn2.running_mean, source["layer4.1.bn2.running_mean"])
    mel = torch.randn(2, 1, 32, 36)
    combined = torch.cat([torch.zeros_like(mel), torch.zeros_like(mel), mel], dim=1)
    expected = F.conv2d(combined, source["conv1.weight"], stride=2, padding=3)
    assert torch.allclose(model.conv1(mel), expected, atol=1e-5, rtol=1e-5)


def test_missing_speech_backbone_tensor_is_rejected():
    model = build_model(11, False)
    with pytest.raises(ValueError, match="missing backbone tensor"):
        transfer(model, {})


def test_unpinned_checkpoint_is_rejected_before_loading(tmp_path):
    path = tmp_path / "unexpected.ckpt"
    path.write_bytes(b"not the published checkpoint")
    with pytest.raises(ValueError, match="hash does not match"):
        tensor_payload(path)
