import numpy as np
import pandas as pd
import pytest
import torch

from experiments.conformer_sanity import Counterfactual, SPEAKERS, speaker_fold, shuffled_training_labels
from src.config import load_config
from src.dataset import read_metadata


def test_cv_partitions_every_development_group_once_without_test_use():
    cfg=load_config("configs/task2_real.json")
    frame=read_metadata(cfg)
    seen=set()
    test=set(frame[frame.split=="test"].path)
    for speaker in SPEAKERS:
        fold=speaker_fold(frame,speaker,cfg)
        train=set(fold[fold.split=="train"].path)
        val=set(fold[fold.split=="val"].path)
        assert len(train)==609 and len(val)==203
        assert not val&seen and not train&val and not val&test and not train&test
        assert fold[fold.split=="val"].group.eq("speaker:"+speaker).all()
        seen.update(val)
    assert seen==set(frame[frame.split!="test"].path)
    with pytest.raises(ValueError):speaker_fold(frame,"spk01",cfg)


def test_negative_control_changes_only_training_labels_and_preserves_counts():
    cfg=load_config("configs/task2_real.json")
    frame=read_metadata(cfg)
    original=frame.copy(deep=True)
    changed=shuffled_training_labels(frame)
    assert frame.equals(original)
    assert changed.drop(columns="label").equals(frame.drop(columns="label"))
    assert changed[changed.split!="train"].equals(frame[frame.split!="train"])
    assert int((changed.label!=frame.label).sum())>400


def test_counterfactual_controls_change_exactly_the_intended_input():
    dataset=[(torch.tensor([1.,2.]),0,"a"),(torch.tensor([3.,4.]),1,"b")]
    permutation=np.array([1,0])
    x,y,path=Counterfactual(dataset,"zero_audio",permutation)[0]
    assert torch.equal(x,torch.zeros(2)) and (y,path)==(0,"a")
    x,y,path=Counterfactual(dataset,"permuted_audio",permutation)[0]
    assert torch.equal(x,dataset[1][0]) and (y,path)==(0,"a")
    x,y,path=Counterfactual(dataset,"permuted_targets",permutation)[0]
    assert torch.equal(x,dataset[0][0]) and (y,path)==(1,"a")
