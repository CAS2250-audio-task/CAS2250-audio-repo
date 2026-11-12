"""Replay eight fixed development-fold models on identical noisy waveforms."""
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from src.config import Config, load_config
from src.dataset import AudioDataset
from src.engine import file_hash, metrics, seed_all, write_json
from src.model import build_model
from experiments.conformer.model import ConformerClassifier, load_source
from .mixing import crop_noise, mix_at_snr, LEVELS, DRAWS, SEED, SAMPLES
from .sources import ROOT, STUDY, REPORT


def read_wave(path):
    wav, rate = sf.read(path, dtype="float32")
    if rate != 16000 or wav.shape != (SAMPLES,) or not np.isfinite(wav).all():
        raise ValueError("Reference waveform must remain exactly mono 16k/1.5s: " + str(path))
    return wav


def mel_batch(waves, transform):
    # 4D dB input gives each item its own top_db maximum, as in transform_file.
    spec = ((transform.to_db(transform.mel(waves[:, None])) + 80) / 80).clamp(0, 1)
    return (spec - .5) / .25


@torch.inference_mode()
def predict_pair(models, waves, transform):
    predictions = {key: [] for key in models}
    before = hashlib.sha256(waves.tobytes()).hexdigest()
    for start in range(0, len(waves), 32):
        batch = torch.from_numpy(waves[start:start+32])
        inputs = {"imagenet": mel_batch(batch, transform).cuda(), "conformer": batch.cuda()}
        for key, model in models.items():
            predictions[key].extend(model(inputs[key]).argmax(1).cpu().tolist())
    assert hashlib.sha256(waves.tobytes()).hexdigest() == before
    return predictions


def freeze_protocol():
    sources = json.loads((REPORT/"noise_sources.json").read_text())
    rows = pd.read_csv(ROOT/"reports/conformer-sanity-v1/speaker_cv.csv")
    pins = {}
    for row in rows.itertuples():
        for name in ("best.pt", "metadata.csv", "config.json" if row.model=="imagenet" else "reference_config.json",
                     "validation/predictions.csv"):
            path = ROOT/row.source_run/name
            pins[str(path.relative_to(ROOT))] = file_hash(path)
    source_pins = {str(p.relative_to(ROOT)): file_hash(p) for p in Path(__file__).parent.glob("*.py")}
    cfg = load_config(ROOT/"configs/task2_real.json")
    audio_audit = pd.read_csv(ROOT/"reports/conformer-sanity-v1/audio_identity_audit.csv")
    audio_pins = dict(zip(audio_audit.path,audio_audit.file_sha256))
    assert len(audio_pins)==812
    assert all(file_hash(Path(cfg.data_root)/relative)==digest for relative,digest in audio_pins.items())
    protocol = dict(study="noise-stress-v1", seed=SEED, snr_db=list(LEVELS), crop_draws=DRAWS,
                    crop_seconds=1.5, noise_only_windows_per_source=20,
                    folds=["spk02","spk03","spk04","spk05"], model_training_seed=42,
                    model_selection="previous clean-validation best.pt, frozen before noisy inference; no tuning",
                    scope="development evaluation only; no spk01 audio loaded or inferred",
                    signal_reference="original full 1.5s clip power, including background and padding; not clean speech power",
                    silence_reference="median original keyword RMS across 600 fold-training clips, squared; nominal SNR",
                    noise_dc_removed=True, noise_window_min_power=1e-8,
                    anti_clipping="joint linear attenuation of speech and added noise to peak .99; no clipping/independent normalization",
                    crops="SHA256-derived RNG key uses path/noise/draw, no label/model/level; identical crop across levels/models",
                    independence="one original recording per noise ID; crop repeats/microphones are not independent recordings; reserve these original sources from future training/test noises",
                    checkpoint_and_metadata_sha256=pins, source_sha256=source_pins, noise_sources=sources,
                    development_audio_sha256=audio_pins,
                    versions={name:version(name) for name in ("torch","torchaudio","speechbrain","numpy")},
                    cuda_matmul_allow_tf32=True, cudnn_allow_tf32=True)
    target = REPORT/"protocol.json"
    if target.exists():
        if json.loads(target.read_text()) != protocol:
            raise ValueError("Frozen protocol changed; use a new study directory")
    else:
        write_json(target,protocol)
    return protocol, rows


def main():
    torch.set_num_threads(2)
    seed_all(SEED)
    # SpeechBrain enables TF32; set explicitly for both model families.
    import speechbrain
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    REPORT.mkdir(parents=True, exist_ok=True)
    partials=STUDY/"parts"
    partials.mkdir(parents=True, exist_ok=True)
    protocol, cv = freeze_protocol()
    state, statistics = load_source(ROOT/"runs/conformer-research")
    recordings = {}
    for source in protocol["noise_sources"]:
        path = ROOT/source["prepared_path"]
        assert file_hash(path) == source["prepared_sha256"]
        recordings[source["noise_id"]] = sf.read(path,dtype="float32")[0]
    started=time.time()
    done=0
    for speaker in protocol["folds"]:
        selected=cv[cv.heldout_speaker.eq(speaker)]
        models={}
        frames={}
        for row in selected.itertuples():
            run=ROOT/row.source_run
            cfg=Config(**json.loads((run/("config.json" if row.model=="imagenet" else "reference_config.json")).read_text()))
            frame=pd.read_csv(run/"metadata.csv").fillna("")
            val=frame[frame.split.eq("val")].reset_index(drop=True)
            assert len(val)==203 and val.group.eq("speaker:"+speaker).all()
            assert not val.speaker_id.eq("spk01").any()
            frames[row.model]=val
            model=(build_model(11,use_pretrained=False) if row.model=="imagenet" else
                   ConformerClassifier(state,statistics,mode="finetune"))
            checkpoint=torch.load(run/"best.pt",map_location="cpu",weights_only=True)
            model.load_state_dict(checkpoint["model"],strict=True)
            models[row.model]=model.cuda().eval()
            del checkpoint
        assert frames["imagenet"].path.equals(frames["conformer"].path)
        val=frames["imagenet"]
        assert set(frame[frame.split.eq("train")].path).isdisjoint(val.path)
        original=np.stack([read_wave(Path(cfg.data_root)/p) for p in val.path])
        lookup={label:idx for idx,label in enumerate(cfg.labels)}
        targets=np.array([lookup[label] for label in val.label])
        keyword_training=frame[frame.split.eq("train") & frame.label.ne("silence")]
        assert len(keyword_training)==600 and not keyword_training.speaker_id.isin((speaker,"spk01")).any()
        training_rms=[float(np.sqrt(np.mean(read_wave(Path(cfg.data_root)/p).astype(np.float64)**2))) for p in keyword_training.path]
        reference_power=float(np.median(training_rms)**2)
        write_json(STUDY/(speaker+"_reference.json"),dict(heldout_speaker=speaker,training_keyword_clips=600,
                   silence_reference_power=reference_power,training_speakers=sorted(keyword_training.speaker_id.unique().tolist())))
        transform=AudioDataset(val,cfg,train=False)
        for i in (0,100,202):
            assert torch.allclose(mel_batch(torch.from_numpy(original[[i]]),transform)[0],
                                  transform.transform_file(Path(cfg.data_root)/val.iloc[i].path),atol=1e-6,rtol=1e-6)
        clean_preds=predict_pair(models,original,transform)
        clean_rows=[]
        for row in selected.itertuples():
            saved=pd.read_csv(ROOT/row.source_run/"validation/predictions.csv").set_index("path")
            expected=saved.loc[val.path].pred_idx.to_numpy()
            assert np.array_equal(expected,clean_preds[row.model]), (speaker,row.model,"clean replay mismatch")
            score=metrics(targets,clean_preds[row.model],11)
            assert abs(score["macro_f1"]-row.macro_f1)<1e-12
            clean_rows.extend(dict(heldout_speaker=speaker,model=row.model,condition="clean",noise_id="clean",category="clean",
                                  snr_db=999,draw=0,path=p,true_idx=int(y),pred_idx=int(pred))
                              for p,y,pred in zip(val.path,targets,clean_preds[row.model]))
        pd.DataFrame(clean_rows).to_csv(partials/(speaker+"_clean.csv.gz"),index=False)
        for source in protocol["noise_sources"]:
            noise_id=source["noise_id"]
            recording=recordings[noise_id]
            for draw in range(DRAWS):
                crops=[crop_noise(recording,path,noise_id,draw) for path in val.path]
                for level in LEVELS:
                    stem=f"{speaker}_{noise_id}_{draw}_{level}"
                    pred_path=partials/(stem+"_pred.csv.gz")
                    audit_path=partials/(stem+"_mix.csv.gz")
                    if pred_path.exists() and audit_path.exists():
                        done+=1
                        continue
                    mixtures=[]; audits=[]
                    for i,(noise,crop) in enumerate(crops):
                        power=(reference_power if targets[i]==lookup["silence"] else float(np.mean(original[i].astype(np.float64)**2)))
                        mixed,audit=mix_at_snr(original[i],noise,level,power)
                        mixtures.append(mixed)
                        audits.append(dict(heldout_speaker=speaker,path=val.iloc[i].path,noise_id=noise_id,
                                           **crop,**audit,silence_reference=bool(targets[i]==lookup["silence"])))
                    mixed=np.stack(mixtures)
                    preds=predict_pair(models,mixed,transform)
                    out=[]
                    for model,indices in preds.items():
                        out.extend(dict(heldout_speaker=speaker,model=model,condition="mixed",noise_id=noise_id,
                                        category=source["category"],snr_db=level,draw=draw,path=p,true_idx=int(y),pred_idx=int(pred))
                                   for p,y,pred in zip(val.path,targets,indices))
                    pd.DataFrame(audits).to_csv(audit_path,index=False)
                    pd.DataFrame(out).to_csv(pred_path,index=False)
                    done+=1
                    progress=dict(conditions_done=done,conditions_total=384,heldout_speaker=speaker,
                                  latest_condition=stem,elapsed_seconds=round(time.time()-started),test_evaluated=False)
                    write_json(STUDY/"progress.json",progress)
                    if done%8==0:print("PROGRESS",json.dumps(progress),flush=True)
                    # Listening examples use one predetermined spk02 / class 0 file, draw 0.
                    if speaker=="spk02" and draw==0:
                        example=int(np.flatnonzero(targets==0)[0])
                        examples=STUDY/"examples"; examples.mkdir(exist_ok=True)
                        sf.write(examples/f"{noise_id}_{level}dB.wav",mixed[example],16000,subtype="FLOAT")
                        sf.write(examples/"original.wav",original[example],16000,subtype="FLOAT")
                        write_json(examples/"voice.json",dict(path=val.iloc[example].path,label=cfg.labels[0],speaker=speaker))
            for level in LEVELS:
                target=partials/f"{speaker}_{noise_id}_{level}_noiseonly.csv.gz"
                if target.exists():continue
                mixtures=[]
                audits=[]
                for index in range(20):
                    noise,crop=crop_noise(recording,"noise-only",noise_id,index)
                    mixed,audit=mix_at_snr(np.zeros(SAMPLES,dtype=np.float32),noise,level,reference_power)
                    mixtures.append(mixed)
                    audits.append(dict(heldout_speaker=speaker,noise_id=noise_id,path=f"noise-only/{index:02d}",**crop,**audit))
                preds=predict_pair(models,np.stack(mixtures),transform)
                out=[]
                for model,indices in preds.items():
                    out.extend(dict(heldout_speaker=speaker,model=model,condition="noise_only",noise_id=noise_id,
                                    category=source["category"],snr_db=level,draw=0,path=f"noise-only/{i:02d}",
                                    true_idx=lookup["silence"],pred_idx=int(pred)) for i,pred in enumerate(indices))
                pd.DataFrame(out).to_csv(target,index=False)
                pd.DataFrame(audits).to_csv(partials/f"{speaker}_{noise_id}_{level}_noiseonly_mix.csv.gz",index=False)
        del models,model
        torch.cuda.empty_cache()
        print("FOLD_DONE",speaker,flush=True)
    # Protocol recheck includes all checkpoint hashes: no training or mutation permitted.
    assert freeze_protocol()[0]==protocol
    write_json(STUDY/"completed.json",dict(conditions_done=done,elapsed_seconds=time.time()-started,
               protocol_sha256=file_hash(REPORT/"protocol.json"),clean_predictions_checked=1624,
               test_evaluated=False,weights_unchanged=True))
    print("COMPLETE",done,flush=True)


if __name__=="__main__":main()
