"""Checks that protect alignment, missingness, fair initialization, and SNR."""
import numpy as np
import pandas as pd
import pytest
import torch

from experiments.auxiliary.features import NAMES, SETS, encode, extract, fit_scaler, masked_delta
from experiments.auxiliary.model import AcousticModel, aligned_layer1
from experiments.auxiliary.data import mix_noise, _init_worker, _worker
from src.config import load_config
from src.dataset import AudioDataset
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


def test_missing_values_are_not_zero_measurements_and_never_propagate():
    raw = np.ones((2, len(NAMES), 9), dtype=np.float32)
    valid = np.ones_like(raw, dtype=bool)
    raw[0, NAMES.index("f1"), 3] = np.nan
    valid[0, NAMES.index("f1"), 3] = False
    scaler = fit_scaler(raw, valid)
    encoded = encode(raw, valid, scaler, "2")
    assert np.isfinite(encoded).all()
    assert encoded[0, 0, 3] == encoded[1, 0, 3] == 0
    assert encoded[0, 3, 3] == 0 and encoded[1, 3, 3] == 1
    assert scaler["observed_frames"][NAMES.index("f1")] == 17


def test_delta_preserves_slope_and_does_not_bridge_missing_frames():
    x = np.arange(15, dtype=float) * 0.01 * 4
    valid = np.ones(15, dtype=bool)
    valid[7] = False
    d = masked_delta(x, valid, 0.01)
    np.testing.assert_allclose(d[2:5], 4)
    assert np.isnan(d[5:10]).all()
    assert np.isnan(d[:2]).all()


def test_time_alignment_uses_centres_not_arbitrary_resize():
    x = torch.arange(151)[None, None, :]
    y = aligned_layer1(x, 38)
    assert torch.equal(y.flatten(), torch.arange(0,151,4))
    with pytest.raises(ValueError, match="centres"):
        aligned_layer1(x, 37)


def test_midfusion_starts_as_identical_reference_and_keeps_head():
    torch.manual_seed(42)
    base = AcousticModel(0, pretrained=False).eval()
    torch.manual_seed(42)
    model = AcousticModel(8, pretrained=False).eval()
    x, aux = torch.randn(2,1,128,151), torch.randn(2,8,151)
    with torch.no_grad():
        torch.testing.assert_close(base(x), model(x, aux), atol=1e-5, rtol=1e-5)
        assert model.forward_features(x,aux).shape == (2,512)
    assert model.backbone.fc.in_features == 512 and model.backbone.fc.out_features == 11
    assert all(p.requires_grad for p in model.backbone.parameters())


def test_auxiliary_branch_receives_gradients_after_identity_fusion_opens():
    torch.manual_seed(3)
    model = AcousticModel(6, pretrained=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    x, aux, y = torch.randn(2,1,128,151), torch.randn(2,6,151), torch.tensor([0,1])
    for _ in range(2):
        optimizer.zero_grad()
        torch.nn.functional.cross_entropy(model(x,aux),y).backward()
        optimizer.step()
    assert model.aux.layers[0].weight.grad.abs().sum() > 0


def test_snr_and_peak_scaling_are_controlled():
    rng = np.random.default_rng(5)
    speech, noise = rng.normal(0,0.01,24000), rng.normal(0,0.2,24000)
    mixed, info = mix_noise(speech,noise,10)
    measured = 10*np.log10(np.mean(speech**2)/np.mean((mixed-speech)**2))
    assert abs(measured-10) < 1e-5
    assert abs(info["reference_snr_db"]-10) < 1e-8
    large, info = mix_noise(speech*100,noise,0)
    assert np.max(np.abs(large)) <= 0.990001
    assert info["common_peak_gain"] < 1


def test_silence_has_zero_rms_but_no_invented_pitch_or_formants():
    raw, valid = extract(np.zeros(24000))
    assert raw.shape == (len(NAMES),151)
    for name in ["log_f0","f1","f2","f3","flatness","centroid"]:
        assert not valid[NAMES.index(name)].any()
    assert np.all(raw[NAMES.index("log_rms")] == -160)
    assert valid[NAMES.index("log_rms")].all()
    assert np.all(raw[NAMES.index("voiced_prob")] == 0)


def test_set7_is_union_of_all_other_sets():
    assert set(SETS["7"]) == set().union(*(set(SETS[str(i)]) for i in range(1,7)))


def test_partial_digital_silence_cannot_be_a_valid_formant():
    from experiments.auxiliary.features import apply_support
    raw=np.ones((len(NAMES),151),dtype=np.float32)*500
    wav=np.zeros(24000)
    wav[8000:16000]=0.1
    corrected, valid=apply_support(wav,raw)
    assert not valid[NAMES.index("f1"),:40].any()
    assert valid[NAMES.index("f1"),60:90].all()
    assert not valid[NAMES.index("delta_f1"),:40].any()


def test_cached_spectrogram_is_exact_reference_frontend():
    cfg = load_config(ROOT / "configs/task2_real.json")
    path = sorted((Path(cfg.data_root)/"audio").glob("*/*.wav"))[0]
    from experiments.auxiliary.features import FeatureConfig
    _init_worker(FeatureConfig().to_dict(), cfg.to_dict())
    spec, _, _, _ = _worker(dict(wav=str(path), noise_only=False))
    expected = AudioDataset(pd.DataFrame(),cfg).transform_file(path)
    torch.testing.assert_close(torch.from_numpy(spec), expected, rtol=0, atol=0)
