import json

import numpy as np
import pandas as pd
import pytest
import torch

from experiments.auxiliary.features import NAMES, fit_scaler
from experiments.auxiliary.model import AcousticModel
from experiments.auxiliary_cv.data import SPEAKERS, FoldDataset, Inputs, fold_indices, mask_inputs
from experiments.auxiliary_cv.study import CVModel
from src.engine import file_hash


def frame():
    return pd.DataFrame([dict(path=f'{s}/{label}.wav', label=label, group=f'speaker:{s}',
        speaker_id=s if label == 'word' else '', room_session_id=f'{s}:room', split='train')
        for s in SPEAKERS for label in ('word', 'silence')])


def test_cv_moves_linked_silence_with_held_out_speaker_and_rejects_test():
    data = frame()
    for speaker in SPEAKERS:
        train, val = fold_indices(data, speaker)
        assert len(train) == 6 and len(val) == 2
        assert data.iloc[val].group.unique().tolist() == [f'speaker:{speaker}']
        assert not set(data.iloc[train].group) & set(data.iloc[val].group)
    with pytest.raises(ValueError):
        fold_indices(data, 'spk01')
    data.loc[0, 'group'] = 'speaker:spk01'
    with pytest.raises(ValueError):
        fold_indices(data, 'spk05')


def test_fold_encoding_uses_train_scaler_and_validation_is_never_masked(tmp_path):
    data = frame()
    raw = np.ones((8, len(NAMES), 151), np.float32)
    raw[:2] = 100
    valid = np.ones_like(raw, bool)
    original = tmp_path / 'cache/original'
    original.mkdir(parents=True)
    data.to_csv(original / 'metadata.csv', index=False)
    np.savez(original / 'inputs.npz', raw=raw, valid=valid,
             spec=np.ones((8, 1, 128, 151), np.float32), paths=data.path.to_numpy(dtype=str))
    (original / 'manifest.json').write_text(json.dumps({'inputs_sha256':file_hash(original / 'inputs.npz')}))
    train, _ = fold_indices(data, 'spk02')
    scaler = fit_scaler(raw[train], valid[train])
    assert scaler['mean'][NAMES.index('f1')] == 1
    folder = tmp_path / 'folds/spk02'
    folder.mkdir(parents=True)
    (folder / 'scaler.json').write_text(json.dumps(scaler))
    inputs = Inputs(tmp_path)
    val = FoldDataset(inputs, 'spk02', '4', ['word', 'silence'], policies=('mask',))
    spec, aux, _, _ = val[0]
    assert torch.all(spec == 1)
    assert aux[0, 0] > 1e6
    train0 = FoldDataset(inputs, 'spk02', '0', ['word','silence'], True, ('mask',), 42)
    train4 = FoldDataset(inputs, 'spk02', '4', ['word','silence'], True, ('mask',), 42)
    masked = 0
    for i in range(12):
        a, b = train0[i % len(train0)], train4[i % len(train4)]
        assert torch.equal(a[0], b[0])
        masked += bool((a[0] == -2).any())
    assert masked > 0


def test_time_masks_hide_both_value_and_availability_channels_at_same_positions():
    spec, aux = mask_inputs(np.ones((1,128,151),np.float32), np.ones((12,151),np.float32),
                            np.random.default_rng(42))
    temporal = np.all(spec == -2, axis=(0,1))
    assert 2 <= temporal.sum() <= 5
    assert np.all(aux[:,temporal] == 0)
    assert np.all(aux[:,~temporal] == 1)


def test_cv_head_preserves_initial_baseline_and_frozen_bn_statistics():
    torch.manual_seed(5)
    original = AcousticModel(0, pretrained=False).eval()
    torch.manual_seed(5)
    model = CVModel(0, pretrained=False, dropout=0).eval()
    spec = torch.randn(2,1,128,151)
    with torch.no_grad():
        torch.testing.assert_close(model(spec), original(spec))
        assert model.forward_features(spec).shape == (2,512)
    frozen = CVModel(0, pretrained=False, freeze_front=True).train()
    assert not frozen.backbone.bn1.training
    assert not frozen.backbone.layer1.training
    assert frozen.backbone.layer2.training
    assert not frozen.backbone.conv1.weight.requires_grad


def test_variable_pitch_preserves_length_alignment_and_changes_within_clip():
    from scipy.signal import correlate
    from experiments.auxiliary_cv.augment import transform

    pulse = np.zeros(24000,np.float32)
    pulse[8000:8004] = .1
    pulse[16000:16004] = .1
    identity = transform(pulse,curve=[0,0,0,0])
    lag = int(np.argmax(correlate(identity,pulse,method='fft')) - len(pulse) + 1)
    assert abs(lag) <= 1
    t = np.arange(24000) / 16000
    tone = (.1 * np.sin(2*np.pi*160*t)).astype(np.float32)
    shifted = transform(tone,curve=[0,1,-1,0])
    assert shifted.shape == tone.shape and np.isfinite(shifted).all()
    def frequency(center):
        start = round((center-.05)*16000)
        window = shifted[start:start+1600] * np.hanning(1600)
        spectrum = np.abs(np.fft.rfft(window,n=32768))
        return np.fft.rfftfreq(32768,1/16000)[np.argmax(spectrum)]
    assert frequency(.5) - frequency(1.) > 8


def test_speed_protects_reviewed_interval_and_pads_fast_audio_with_zeros():
    from experiments.auxiliary_cv.augment import speed_wave, protected_speed_crop

    tone = (.1 * np.sin(2*np.pi*160*np.arange(24000)/16000)).astype(np.float32)
    rejected, info = speed_wave(tone,.9,(0,24000))
    assert info['applied_rate'] == 1
    np.testing.assert_array_equal(rejected,tone)
    assert protected_speed_crop(26667,.9,(0,24000)) is None
    fast, info = speed_wave(tone,1.1,(0,24000))
    assert fast.shape == tone.shape and info['zero_padding'] > 0
    padding = info['zero_padding']//2
    assert not np.any(fast[:padding]) and not np.any(fast[-padding:])
    slow, info = speed_wave(tone,.9,(5000,18000))
    assert slow.shape == tone.shape and info['applied_rate'] == .9
    a,b = 5000/.9,18000/.9
    assert info['crop'][0] <= a and info['crop'][1] >= b


def test_augmented_features_are_recomputed_from_the_stored_final_waveform(tmp_path):
    import soundfile as sf
    from experiments.auxiliary import data as legacy
    from experiments.auxiliary.features import FeatureConfig, extract
    from experiments.auxiliary_cv.augment import augmented_worker
    from src.config import load_config

    cfg = load_config('configs/task2_real_context.json')
    legacy._init_worker(FeatureConfig().to_dict(),cfg.to_dict())
    source = tmp_path / 'tone.wav'
    sf.write(source,.1*np.sin(2*np.pi*160*np.arange(24000)/16000),16000,subtype='PCM_16')
    task = (dict(path='tone.wav',label='word',wav=str(source),variant=0,
                 policies=['pitch'],protected=[5000,18000],noise_files=[]),str(tmp_path/'row.npz'))
    spec,raw,valid,_ = augmented_worker(task)
    with np.load(tmp_path/'row.npz') as saved:
        wave = saved['wave']
    expected_raw,expected_valid = extract(wave)
    np.testing.assert_allclose(raw,expected_raw,equal_nan=True)
    np.testing.assert_array_equal(valid,expected_valid)
    with torch.no_grad():
        expected_spec = ((legacy._DB(legacy._MEL(torch.from_numpy(wave)[None]))+80)/80).clamp(0,1)
        expected_spec = ((expected_spec-.5)/.25).numpy()
    np.testing.assert_array_equal(spec,expected_spec)


def test_condition_selection_requires_mean_gain_and_speaker_stability():
    from experiments.auxiliary_cv.pipeline import improves

    reference = dict(mean_f1=.85,worst_speaker_f1=.75,per_set={'0':.86,'4':.84})
    good = dict(mean_f1=.86,worst_speaker_f1=.75,per_set={'0':.87,'4':.85})
    assert improves(good,reference)
    assert not improves(dict(good,mean_f1=.852),reference)
    assert not improves(dict(good,worst_speaker_f1=.73),reference)
    assert not improves(dict(good,per_set={'0':.88,'4':.83}),reference)
