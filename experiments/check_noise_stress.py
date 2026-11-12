"""Independent artifact checks and fresh waveform/checkpoint replay."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from sklearn.metrics import accuracy_score, f1_score

from src.config import Config
from src.engine import file_hash, write_json, seed_all
from src.model import build_model
from src.dataset import AudioDataset
from experiments.conformer.model import ConformerClassifier, load_source
from experiments.noise_stress.mixing import crop_noise
from experiments.noise_stress.run import predict_pair, read_wave

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/"reports/noise-stress-v1"
STUDY=ROOT/"runs/noise-stress-v1"


def main():
    torch.set_num_threads(2)
    seed_all(20261112)
    import speechbrain
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    protocol=json.loads((REPORT/"protocol.json").read_text())
    pins={**protocol["checkpoint_and_metadata_sha256"],**protocol["source_sha256"]}
    assert all(file_hash(ROOT/p)==digest for p,digest in pins.items())
    p=pd.read_csv(REPORT/"predictions.csv.gz")
    a=pd.read_csv(REPORT/"mixtures.csv.gz",dtype={"crop_seed":str})
    m=pd.read_csv(REPORT/"condition_metrics.csv")
    keys=["heldout_speaker","model","condition","noise_id","category","snr_db","draw"]
    grouped=p.groupby(keys,sort=False)
    for row in m.itertuples():
        part=grouped.get_group(tuple(getattr(row,key) for key in keys))
        assert len(part)==row.n
        assert abs(accuracy_score(part.true_idx,part.pred_idx)-row.accuracy)<1e-12
        assert abs(f1_score(part.true_idx,part.pred_idx,labels=list(range(11)),average="macro",zero_division=0)-row.macro_f1)<1e-12
    mixed=a[a.path.str.startswith("audio/")].copy()
    crop_consistency=mixed.groupby(["heldout_speaker","path","noise_id","draw"])[["start_sample","end_sample","crop_seed","noise_power","reference_power"]].nunique()
    assert crop_consistency.eq(1).all().all() and len(crop_consistency)==19488
    snr=10*np.log10(a.reference_power/(a.noise_power*a.noise_gain**2))
    assert np.max(np.abs(snr-a.snr_db))<1e-8
    assert len(p)==162648 and len(a)==80512
    recordings={src["noise_id"]:sf.read(ROOT/src["prepared_path"],dtype="float32")[0] for src in protocol["noise_sources"]}
    assert all(file_hash(ROOT/src["prepared_path"])==src["prepared_sha256"] for src in protocol["noise_sources"])
    state,stats=load_source(ROOT/"runs/conformer-research")
    cv=pd.read_csv(ROOT/"reports/conformer-sanity-v1/speaker_cv.csv")
    replay_rows=[]
    checked_audios=set()
    for speaker in protocol["folds"]:
        models={};frames={}
        for row in cv[cv.heldout_speaker.eq(speaker)].itertuples():
            run=ROOT/row.source_run
            cfg=Config(**json.loads((run/("config.json" if row.model=="imagenet" else "reference_config.json")).read_text()))
            frame=pd.read_csv(run/"metadata.csv").fillna("")
            frames[row.model]=frame[frame.split.eq("val")].set_index("path")
            model=build_model(11,False) if row.model=="imagenet" else ConformerClassifier(state,stats,"finetune")
            checkpoint=torch.load(run/"best.pt",map_location="cpu",weights_only=True)
            model.load_state_dict(checkpoint["model"],strict=True)
            models[row.model]=model.cuda().eval()
            del checkpoint
        val=frames["imagenet"]
        assert len(val)==203 and not val.speaker_id.eq("spk01").any()
        for path in val.index:
            assert file_hash(Path(cfg.data_root)/path)==protocol["development_audio_sha256"][path]
            checked_audios.add(path)
        truth=val.label.map(dict(zip(cfg.labels,range(11))))
        selected_p=p[p.heldout_speaker.eq(speaker)&p.condition.ne("noise_only")]
        assert np.array_equal(truth.loc[selected_p.path].to_numpy(),selected_p.true_idx.to_numpy())
        ref=json.loads((STUDY/(speaker+"_reference.json")).read_text())["silence_reference_power"]
        training=frame[frame.split.eq("train") & frame.label.ne("silence")]
        assert len(training)==600 and not training.speaker_id.isin((speaker,"spk01")).any()
        rms=[np.sqrt(np.mean(read_wave(Path(cfg.data_root)/path).astype(np.float64)**2)) for path in training.path]
        assert abs(float(np.median(rms)**2)-ref)<1e-12
        assert np.max(np.abs(mixed[mixed.heldout_speaker.eq(speaker)&mixed.silence_reference.eq(True)].reference_power-ref))<1e-12
        chosen=[]
        for (_,_,_),part in mixed[mixed.heldout_speaker.eq(speaker)&mixed.draw.eq(0)].groupby(["noise_id","snr_db","draw"]):
            chosen.append(part[part.silence_reference.eq(False)].iloc[37])
            chosen.append(part[part.silence_reference.eq(True)].iloc[0])
        chosen=pd.DataFrame(chosen)
        waves=[]
        for row in chosen.itertuples():
            original=read_wave(Path(cfg.data_root)/row.path)
            recording=recordings[row.noise_id]
            noise=recording[row.start_sample:row.end_sample].astype(np.float64)
            noise-=noise.mean()
            from_rng,details=crop_noise(recording,row.path,row.noise_id,row.draw)
            assert details["start_sample"]==row.start_sample and np.array_equal(from_rng,noise)
            power=ref if row.silence_reference else float(np.mean(original.astype(np.float64)**2))
            gain=np.sqrt(power/(np.mean(noise**2)*10**(row.snr_db/10)))
            raw=original.astype(np.float64)+gain*noise
            attenuation=min(1.,.99/np.max(np.abs(raw)))
            wave=(raw*attenuation).astype(np.float32)
            assert np.max(np.abs(wave))<=.990001
            assert hashlib.sha256(wave.tobytes()).hexdigest()==row.mixture_sha256
            assert abs(power-row.reference_power)<1e-12
            waves.append(wave)
        transform=AudioDataset(val.reset_index(),cfg,False)
        actual=predict_pair(models,np.stack(waves),transform)
        for model,indices in actual.items():
            saved=p[p.heldout_speaker.eq(speaker)&p.model.eq(model)&p.condition.eq("mixed")]
            wanted=chosen[["path","noise_id","snr_db","draw"]].merge(saved,on=["path","noise_id","snr_db","draw"],how="left",validate="one_to_one")
            assert np.array_equal(wanted.pred_idx.to_numpy(),indices),(speaker,model)
            replay_rows.append(dict(speaker=speaker,model=model,mixed_predictions_replayed=len(indices)))
        del models,model
        torch.cuda.empty_cache()
    assert len(checked_audios)==812
    assert all(file_hash(ROOT/path)==digest for path,digest in pins.items())
    audit=dict(metric_conditions_recomputed=len(m),prediction_rows=len(p),mixture_rows=len(a),
               identical_crops_across_levels=19488,independent_waveforms_reconstructed=256,
               noisy_predictions_replayed=sum(row["mixed_predictions_replayed"] for row in replay_rows),
               all_predictions_identical=True,original_development_wav_hashes_checked=len(checked_audios),
               original_checkpoint_hashes_unchanged=True,training_rms_references_independently_recomputed=4,
               test_evaluated=False,details=replay_rows,
               verifier_sha256=file_hash(Path(__file__)))
    write_json(REPORT/"independent_replay.json",audit)
    print("VERIFIED",json.dumps(audit),flush=True)


if __name__=="__main__":main()
