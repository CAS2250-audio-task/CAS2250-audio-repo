"""Selection and independent formant/noise streams must preserve the protocol."""
import copy
import numpy as np
import pandas as pd

from experiments.conformer_noise_focus.augment import FocusTrainingClips
from experiments.overnight.data import Training
from experiments.overnight import run


def test_selection_rejects_mean_gain_that_harms_the_weakest_speaker(monkeypatch):
    values={"base":(.89,.76),"harmful":(.91,.74),"tiny":(.892,.77),"useful":(.90,.77)}
    monkeypatch.setattr(run,"scores",lambda spec:values[spec["name"]])
    assert not run.better(dict(name="harmful"),dict(name="base"))
    assert not run.better(dict(name="tiny"),dict(name="base"))
    assert run.better(dict(name="useful"),dict(name="base"))


def training_fixture(cls,formants=None):
    data=object.__new__(cls)
    data.frame=pd.DataFrame(dict(path=["audio/test/spk03_001.wav"]))
    data.wave=(np.sin(np.arange(24000)*.04)*.1).astype(np.float32)[None,:]
    data.targets=[0];data.sources=[dict(noise_id="n",category="road")]
    data.recordings={"n":np.random.default_rng(7).normal(0,.01,48000).astype(np.float32)}
    data.speaker="spk02";data.add_speech=False;data.focus_zero=False
    data.reference_power=.005;data.trace=[];data.set_epoch(21)
    if cls is Training:data.formants=formants
    return data


def test_disabled_formants_reproduce_every_original_noise_input():
    old=training_fixture(FocusTrainingClips);new=training_fixture(Training)
    for draw in range(30):
        a,_,_=old[0];b,_,_=new[0]
        assert np.array_equal(a.numpy(),b.numpy())
        for key,value in old.trace[-1].items():assert new.trace[-1][key]==value


def test_formants_can_cooccur_with_noise_without_changing_its_random_stream():
    old=training_fixture(FocusTrainingClips);new=training_fixture(Training)
    original=new.wave.copy();path=new.frame.path.iloc[0]
    altered=(np.sin(np.arange(24000)*.07)*.1).astype(np.float32)
    new.formants={(path,v):(altered,1.1) for v in (0,1)}
    both=False;inactive=False
    for draw in range(40):
        a,_,_=old[0];b,_,_=new[0]
        x,y=old.trace[-1],new.trace[-1]
        assert x['noise_selected']==y['noise_selected']
        for key in ('base_noise_id','base_start_sample','base_snr_db'):
            if key in x:assert x[key]==y[key]
        if not y['formant_selected']:
            inactive=True;assert np.array_equal(a.numpy(),b.numpy())
        if y['noise_selected'] and y['formant_selected']:both=True
        assert np.array_equal(new.wave,original)
    assert both and inactive


def test_A_B_and_AB_select_only_their_enabled_waveform_components():
    path="audio/test/spk03_001.wav"
    for augmented,formant in ((True,False),(False,True),(True,True)):
        d=training_fixture(Training)
        d.augmented=augmented;d.formants={} if formant else None
        d.variants={}
        for flags in range(1,8):
            for variant in (0,1):
                w=np.full(24000,.01*flags,np.float32)
                d.variants[(path,flags,variant)]=(w,dict(formant_ratio=1.05 if flags&4 else 1.,speed=dict(applied_rate=1.05)))
        original=d.wave.copy();observed=set()
        for n in range(80):
            d[0];t=d.trace[-1];flags=t['waveform_flags'];observed.add(flags)
            assert bool(flags&1)==t['pitch_selected'] and bool(flags&2)==t['speed_selected']
            assert bool(flags&4)==t['formant_selected']
            if not augmented:assert flags in (0,4)
            if not formant:assert flags in (0,1,2,3)
            assert np.array_equal(d.wave,original)
        assert 0 in observed
        if augmented and formant:assert 7 in observed
