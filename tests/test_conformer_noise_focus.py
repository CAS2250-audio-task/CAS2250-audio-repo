"""Same original draws, independent speech/SNR policies, and unchanged sample count."""
import numpy as np
import torch

from src.config import load_config
from src.dataset import read_metadata
from experiments.conformer_aug.augment import TrainingClips
from experiments.conformer_noise_focus.augment import FocusTrainingClips
from experiments.conformer_noise_focus.sources import ROOT


def datasets():
    cfg=load_config(ROOT/"configs/task2_real.json")
    frame=read_metadata(cfg).query("split == 'train'")
    old=[dict(noise_id="old",category="road")]
    added=[dict(noise_id="added",category="crowd",family="babble")]
    rng=np.random.default_rng(721)
    recordings={name:rng.normal(0,.05,100000).astype(np.float32) for name in ("old","added")}
    plain=TrainingClips(frame,cfg,"spk05",None,{},old,recordings,noise=True)
    variants=[FocusTrainingClips(frame,cfg,"spk05",old,added,recordings,add_speech=a,focus_zero=b)
              for a,b in ((False,False),(True,False),(False,True),(True,True))]
    return plain,variants


def test_unmodified_focus_policy_exactly_reproduces_old_training_inputs():
    old,variants=datasets()
    base=variants[0]
    old.set_epoch(3);base.set_epoch(3)
    for i in range(50):
        x,y,path=old[i];actual,label,p=base[i]
        assert torch.equal(x,actual) and (y,path)==(label,p)
    assert len(base)==609


def test_speech_and_snr_overrides_are_independent_and_keep_wide_range():
    _,variants=datasets()
    for data in variants:
        data.set_epoch(2)
        for i in range(200):data[i]
    base,speech,focus,both=[d.trace for d in variants]
    assert all([r["noise_selected"] for r in trace]==[r["noise_selected"] for r in base] for trace in (speech,focus,both))
    assert [r["added_speech_selected"] for r in speech]==[r["added_speech_selected"] for r in both]
    assert [r["near_zero_selected"] for r in focus]==[r["near_zero_selected"] for r in both]
    for a,b,c,d in zip(base,speech,focus,both):
        if not a["noise_selected"]:
            assert a["input_sha256"]==b["input_sha256"]==c["input_sha256"]==d["input_sha256"]
        else:
            assert a["snr_db"]==b["snr_db"]
            assert c["snr_db"]==d["snr_db"]
            assert a["noise_id"]==c["noise_id"] and b["noise_id"]==d["noise_id"]
            assert -5<=d["snr_db"]<=20
            if d["near_zero_selected"]:assert -3<=d["snr_db"]<=3
    assert any(r.get("snr_db",0)>3 for r in both)
    assert any(r["near_zero_selected"] for r in both)
    assert any(r["added_speech_selected"] for r in both)


def test_new_policy_replays_same_epoch_draw_and_preserves_length_and_peak():
    _,variants=datasets()
    data=variants[-1]
    data.set_epoch(4)
    actual=[data[i][0] for i in range(50)]
    traces=data.trace.copy()
    data.set_epoch(4);data.trace=[]
    again=[data[i][0] for i in range(50)]
    assert traces==data.trace and all(torch.equal(a,b) for a,b in zip(actual,again))
    for wave,trace in zip(actual,traces):
        assert wave.shape==(24000,) and torch.isfinite(wave).all()
        if trace["noise_selected"]:
            assert wave.abs().max()<=.990001
            assert abs(trace["snr_db"]-trace["snr_db_measured"])<1e-8
