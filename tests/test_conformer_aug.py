"""Verify training-only transforms, native masks and preserved signal content."""
import json

import numpy as np
import torch

from experiments.conformer_aug.augment import speed_crop, speed_wave, mask_features, TrainingClips
from experiments.conformer_aug.model import AugmentedConformer
from experiments.conformer_aug.sources import ROOT, SOURCES
from experiments.conformer.model import ConformerClassifier, load_source
from src.config import load_config
from src.dataset import read_metadata
from src.engine import seed_all


def test_speed_protects_reviewed_interval_and_keeps_pitch():
    assert speed_crop(round(24000/.9),.9,(0,24000)) is None
    for rate in (.9,.95,1.05,1.1):
        crop=speed_crop(round(24000/rate),rate,(4000,18000))
        assert crop is not None
        assert crop[0]<=int(np.floor(4000/rate)) and crop[1]>=int(np.ceil(18000/rate))
    t=np.arange(24000)/16000
    wave=(.1*np.sin(2*np.pi*220*t)).astype(np.float32)
    unchanged,trace=speed_wave(wave,.9,(0,24000))
    assert trace["skipped"] and np.array_equal(unchanged,wave)
    result,trace=speed_wave(wave,1.1,(0,24000))
    assert result.shape==(24000,) and trace["padding"]==24000-round(24000/1.1)
    peak=np.fft.rfftfreq(24000,1/16000)[np.abs(np.fft.rfft(result)).argmax()]
    assert abs(peak-220)<2


def test_masks_use_native_silence_floor_and_preserve_unmasked_values():
    features=torch.ones(32,151,80)
    floor=torch.arange(80,dtype=torch.float32)-100
    actual,traces=mask_features(features,floor,42,"spk02",1,0)
    again,repeat=mask_features(features,floor,42,"spk02",1,0)
    assert traces==repeat and torch.equal(actual,again) and 0<len(traces)<32
    expected=features.clone()
    for row in traces:
        assert 2<=row["time_width"]<=5 and 3<=row["freq_width"]<=5
        item,ts,tw,fs,fw=[row[k] for k in ("item","time_start","time_width","freq_start","freq_width")]
        expected[item,ts:ts+tw]=floor
        expected[item,:,fs:fs+fw]=floor[fs:fs+fw]
    assert torch.equal(actual,expected) and torch.equal(features,torch.ones_like(features))


def test_training_noise_sources_are_original_recordings_disjoint_from_validation():
    evaluation=json.loads((ROOT/"reports/noise-stress-v1/noise_sources.json").read_text())
    assert len(SOURCES)==6 and not {s[3] for s in SOURCES}&{s["original_filename"] for s in evaluation}
    assert [s[1] for s in SOURCES].count("road")==2
    assert [s[1] for s in SOURCES].count("crowd")==2
    assert [s[1] for s in SOURCES].count("machine")==2


def test_online_noise_is_reproducible_and_preserves_targets_and_duration():
    cfg=load_config(ROOT/"configs/task2_real.json")
    frame=read_metadata(cfg).query("split == 'train'")
    source=[dict(noise_id="synthetic")]
    noise={"synthetic":np.random.default_rng(7).normal(0,.05,100000).astype(np.float32)}
    data=TrainingClips(frame,cfg,"spk05",None,{},source,noise,noise=True)
    data.set_epoch(1)
    first=[data[i] for i in range(20)]
    traces=data.trace.copy()
    assert any(row["noise_selected"] for row in traces) and not all(row["noise_selected"] for row in traces)
    data.set_epoch(1);data.trace=[]
    repeated=[data[i] for i in range(20)]
    assert traces==data.trace
    for a,b in zip(first,repeated):
        assert torch.equal(a[0],b[0]) and a[1:]==b[1:] and a[0].shape==(24000,)
    for row in traces:
        if row["noise_selected"]:
            assert -5<=row["snr_db"]<=20 and abs(row["snr_db"]-row["snr_db_measured"])<1e-8


def test_mask_wrapper_keeps_checkpoint_compatibility_and_eval_inputs_unchanged():
    state,stats=load_source(ROOT/"runs/conformer-research")
    seed_all(42);base=ConformerClassifier(state,stats,"finetune").eval()
    seed_all(42);aug=AugmentedConformer(state,stats,"spk02",mask=True).eval()
    assert list(base.state_dict())==list(aug.state_dict())
    assert all(torch.equal(tensor,aug.state_dict()[name]) for name,tensor in base.state_dict().items())
    wave=torch.randn(2,24000)*.01
    with torch.no_grad():
        assert torch.equal(base.features(wave),aug.features(wave))
        assert torch.allclose(base(wave),aug(wave),atol=1e-7,rtol=1e-7)
    aug.train();aug.set_epoch(1)
    with torch.no_grad():masked=aug.features(wave.repeat(10,1))
    assert aug.mask_trace and not torch.equal(masked,base.features(wave.repeat(10,1)))
