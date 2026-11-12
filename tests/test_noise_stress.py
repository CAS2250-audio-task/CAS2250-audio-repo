"""Signal-level safeguards for the paired noise evaluation."""
import numpy as np
import pytest
import torch

from experiments.noise_stress.mixing import crop_noise, mix_at_snr, SAMPLES
from experiments.noise_stress.run import mel_batch
from src.config import load_config
from src.dataset import AudioDataset, read_metadata


def test_crop_is_repeatable_contiguous_and_uses_different_draws():
    recording=np.random.default_rng(42).normal(0,.1,SAMPLES*8).astype(np.float32)
    noise,a=crop_noise(recording,"same-file","traffic",0)
    again,b=crop_noise(recording,"same-file","traffic",0)
    other,c=crop_noise(recording,"same-file","traffic",1)
    assert a==b and np.array_equal(noise,again) and a["start_sample"]!=c["start_sample"]
    expected=recording[a["start_sample"]:a["end_sample"]].astype(np.float64)
    assert np.array_equal(noise,expected-expected.mean()) and len(noise)==24000
    with pytest.raises(ValueError):crop_noise(np.zeros(SAMPLES*2),"file","traffic",0)


def test_snr_is_measured_from_components_and_joint_scaling_preserves_it():
    rng=np.random.default_rng(11)
    wave=rng.normal(0,.4,SAMPLES).astype(np.float32)
    noise=rng.normal(0,.1,SAMPLES)
    power=float(np.mean(wave.astype(np.float64)**2))
    for level in (10,0,-5,-10):
        mixture,audit=mix_at_snr(wave,noise,level,power)
        clean=wave.astype(np.float64)*audit["joint_attenuation"]
        added=noise*audit["noise_gain"]*audit["joint_attenuation"]
        measured=10*np.log10(np.mean(clean**2)/np.mean(added**2))
        assert abs(measured-level)<1e-10 and np.max(np.abs(mixture))<=.990001
        assert np.allclose(mixture,clean+added,atol=1e-7)
    assert audit["joint_attenuation"]<1  # Loudest stress case exercises anti-clipping.


def test_noise_only_uses_training_speech_reference_instead_of_zero_silence_power():
    noise=np.random.default_rng(3).normal(0,.1,SAMPLES)
    mixed,audit=mix_at_snr(np.zeros(SAMPLES),noise,0,.01)
    assert abs(np.mean(mixed.astype(np.float64)**2)-.01)<1e-8
    with pytest.raises(ValueError):mix_at_snr(np.zeros(SAMPLES),noise,0,0)


def test_batched_reference_mel_does_not_share_db_ceiling_between_clips():
    cfg=load_config("configs/task2_real.json")
    frame=read_metadata(cfg).query("split == 'val'").reset_index(drop=True)
    transform=AudioDataset(frame,cfg,train=False)
    rng=np.random.default_rng(19)
    waves=torch.from_numpy(np.stack([rng.normal(0,amplitude,SAMPLES).astype(np.float32) for amplitude in (.00001,.01,.9)]))
    actual=mel_batch(waves,transform)
    for i in range(3):
        one=mel_batch(waves[i:i+1],transform)[0]
        assert torch.allclose(actual[i],one,atol=1e-6,rtol=1e-6)
