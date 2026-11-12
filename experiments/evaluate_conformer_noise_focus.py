"""All training conditions x fixed DFN3 off/on on identical frozen stress inputs."""
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from src import engine
from src.config import load_config
from src.dataset import read_metadata
from experiments.conformer.model import ConformerClassifier, load_source
from experiments.noise_stress.mixing import crop_noise,mix_at_snr,LEVELS,DRAWS,SAMPLES
from experiments.noise_stress.run import read_wave
from experiments.denoise_eval.filters import make_filter
from experiments.conformer_noise_focus.sources import ROOT,STUDY,REPORT
from experiments.conformer_noise_focus.run import CONDITIONS,protocol

TRAINING=("baseline",*CONDITIONS)


def checkpoint(condition,speaker):
    if condition=="baseline":return ROOT/f"runs/conformer-aug-v1/folds/{speaker}/noise/best.pt"
    return STUDY/"folds"/speaker/condition/"best.pt"


@torch.inference_mode()
def predict(models,waves):
    result={name:[] for name in models}
    for offset in range(0,len(waves),32):
        x=torch.from_numpy(waves[offset:offset+32]).cuda()
        for name,model in models.items():
            result[name].extend(model(x).argmax(1).cpu().tolist())
    return result


def records(predictions,frame,speaker,denoiser,kind,noise_id,category,level,draw):
    return pd.DataFrame([dict(training_condition=name,denoiser=denoiser,heldout_speaker=speaker,
        kind=kind,noise_id=noise_id,category=category,snr_db=level,draw=draw,path=path,
        true_idx=int(label),pred_idx=int(pred)) for name,indices in predictions.items()
        for path,label,pred in zip(frame.path,frame.true_idx,indices)])


def main():
    torch.set_num_threads(2);engine.seed_all(42)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    assert (STUDY/"training_complete.json").exists()
    cfg=load_config(ROOT/"configs/task2_real.json");metadata=read_metadata(cfg)
    frozen=protocol(cfg,metadata)
    eval_protocol=json.loads((ROOT/"reports/noise-stress-v1/protocol.json").read_text())
    sources=eval_protocol["noise_sources"]
    recordings={}
    for source in sources:
        path=ROOT/source["prepared_path"]
        assert engine.file_hash(path)==source["prepared_sha256"]
        recordings[source["noise_id"]]=sf.read(path,dtype="float32")[0]
    old=pd.read_csv(ROOT/"reports/conformer-aug-v1/predictions.csv.gz")
    old=old[old.condition.eq("noise")].set_index(["heldout_speaker","kind","noise_id","snr_db","draw","path"])
    old_dfn=pd.read_csv(ROOT/"reports/denoise-eval-v1/predictions.csv.gz")
    old_dfn=old_dfn[old_dfn["filter"].eq("deepfilternet3")].set_index(["heldout_speaker","kind","noise_id","snr_db","draw","path"])
    manifests=pd.read_csv(ROOT/"reports/noise-stress-v1/mixtures.csv.gz")
    prior_hash=manifests.set_index(["heldout_speaker","path","noise_id","snr_db","draw"]).mixture_sha256.to_dict()
    state,statistics=load_source(ROOT/"runs/conformer-research")
    partials=STUDY/"evaluation_parts";partials.mkdir(exist_ok=True)
    audit_parts=STUDY/"evaluation_audits";audit_parts.mkdir(exist_ok=True)
    dfn=make_filter("deepfilternet3")
    checkpoint_pins={}
    for primary in (True,False):
        for speaker in frozen["folds"]:
            frame=metadata[metadata.group.eq("speaker:"+speaker)].reset_index(drop=True).copy()
            frame["true_idx"]=frame.label.map({v:i for i,v in enumerate(cfg.labels)})
            assert len(frame)==203 and speaker!="spk01"
            original=np.stack([read_wave(Path(cfg.data_root)/p) for p in frame.path])
            models={}
            for condition in TRAINING:
                path=checkpoint(condition,speaker)
                expected=json.loads((path.parent/"summary.json").read_text())["checkpoint_sha256"]
                assert engine.file_hash(path)==expected
                checkpoint_pins[str(path.relative_to(ROOT))]=expected
                model=ConformerClassifier(state,statistics,"finetune")
                model.load_state_dict(torch.load(path,map_location="cpu",weights_only=True)["model"],strict=True)
                models[condition]=model.cuda().eval()
            old_cases=pd.read_csv(ROOT/f"runs/denoise-eval-v1/{speaker}_cases.csv.gz")
            old_map=old_cases.set_index(["kind","noise_id","snr_db","draw","path"])
            old_waves=np.load(ROOT/f"runs/denoise-eval-v1/filtered/deepfilternet3/{speaker}.npy",mmap_mode="r")

            def evaluate_unit(waves,unit_frame,kind,noise_id,category,level,draw):
                name=f"{speaker}_{kind}_{noise_id}_{level}_{draw}"
                target=partials/(name+".csv.gz")
                audit_path=audit_parts/(name+".json")
                if target.exists() and audit_path.exists():return
                original_hashes=[hashlib.sha256(w.tobytes()).hexdigest() for w in waves]
                cached=kind=="clean" or level==0
                if cached:
                    indices=[int(old_map.loc[(kind,noise_id,level,draw,p)].case_index) for p in unit_frame.path]
                    assert original_hashes==old_cases.set_index("case_index").loc[indices].input_sha256.tolist()
                    filtered=np.array(old_waves[indices])
                else:
                    cache=STUDY/"dfn_cache"/(name+".npy")
                    cache.parent.mkdir(exist_ok=True)
                    if cache.exists():
                        filtered=np.load(cache)
                        assert filtered.shape==waves.shape and filtered.dtype==np.float32 and np.isfinite(filtered).all()
                    else:
                        filtered=np.concatenate([dfn(waves[i:i+16]) for i in range(0,len(waves),16)])
                        with cache.with_suffix(".partial.npy").open("wb") as stream:np.save(stream,filtered)
                        cache.with_suffix(".partial.npy").replace(cache)
                pieces=[]
                for denoiser,inputs in (("off",waves),("dfn3",filtered)):
                    result=predict(models,inputs)
                    baseline=old if denoiser=="off" else old_dfn if cached else None
                    if baseline is not None:
                        expected=[int(baseline.loc[(speaker,kind,noise_id,level,draw,p)].pred_idx) for p in unit_frame.path]
                        assert result["baseline"]==expected,(name,denoiser,"old baseline differs")
                    pieces.append(records(result,unit_frame,speaker,denoiser,kind,noise_id,category,level,draw))
                combined=pd.concat(pieces,ignore_index=True)
                combined.to_csv(target.with_suffix(".partial.gz"),index=False)
                target.with_suffix(".partial.gz").replace(target)
                engine.write_json(audit_path,dict(heldout_speaker=speaker,kind=kind,noise_id=noise_id,snr_db=level,draw=draw,
                    paths=unit_frame.path.tolist(),input_sha256=original_hashes,
                    dfn3_output_sha256=[hashlib.sha256(w.tobytes()).hexdigest() for w in filtered],
                    previous_dfn_cache_reused=cached,prediction_rows=len(combined),
                    prediction_file_sha256=engine.file_hash(target),output_peak=float(np.abs(filtered).max())))

            evaluate_unit(original,frame,"clean","clean","clean",999,0)
            training=metadata[~metadata.group.isin(["speaker:spk01","speaker:"+speaker])&metadata.label.ne("silence")]
            assert len(training)==600
            reference=float(np.median([float(np.sqrt(np.mean(read_wave(Path(cfg.data_root)/p).astype(np.float64)**2)))
                                       for p in training.path])**2)
            for source in sources:
                noise_id=source["noise_id"];recording=recordings[noise_id]
                for draw in range(DRAWS):
                    crops=[crop_noise(recording,path,noise_id,draw)[0] for path in frame.path]
                    for level in LEVELS:
                        if (level==0)!=primary:continue
                        mixtures=[]
                        for i,noise in enumerate(crops):
                            power=reference if frame.true_idx.iloc[i]==10 else float(np.mean(original[i].astype(np.float64)**2))
                            wave,audit=mix_at_snr(original[i],noise,level,power)
                            assert audit["mixture_sha256"]==prior_hash[(speaker,frame.path.iloc[i],noise_id,level,draw)]
                            mixtures.append(wave)
                        evaluate_unit(np.stack(mixtures),frame,"mixed",noise_id,source["category"],level,draw)
                only_frame=pd.DataFrame(dict(path=[f"noise-only/{i:02d}" for i in range(20)],true_idx=[10]*20))
                for level in LEVELS:
                    if (level==0)!=primary:continue
                    mixtures=[]
                    for i in range(20):
                        noise,_=crop_noise(recording,"noise-only",noise_id,i)
                        wave,audit=mix_at_snr(np.zeros(SAMPLES,dtype=np.float32),noise,level,reference)
                        assert audit["mixture_sha256"]==prior_hash[(speaker,f"noise-only/{i:02d}",noise_id,level,i)]
                        mixtures.append(wave)
                    evaluate_unit(np.stack(mixtures),only_frame,"noise_only",noise_id,source["category"],level,0)
                print("FOCUS_EVAL_SOURCE_DONE",speaker,noise_id,len(list(partials.glob("*.csv.gz"))),516,flush=True)
            del model,models,old_waves
            gc.collect();torch.cuda.empty_cache()
    predictions=pd.concat([pd.read_csv(p) for p in sorted(partials.glob("*.csv.gz"))],ignore_index=True)
    assert len(predictions)==650592
    predictions.to_csv(REPORT/"predictions.csv.gz",index=False)
    audits=[json.loads(p.read_text()) for p in sorted(audit_parts.glob("*.json"))]
    assert len(audits)==516
    reused=sum(len(a["paths"]) for a in audits if a["previous_dfn_cache_reused"])
    assert reused==20940
    engine.write_json(REPORT/"evaluation_audit.json",dict(prediction_rows=len(predictions),units=len(audits),
        input_cases=81324,old_mixture_sha256_matched=80512,previous_dfn_waveforms_reused=reused,
        new_dfn_waveforms=81324-reused,old_unfiltered_baseline_replayed=81324,old_dfn_baseline_replayed=20940,
        checkpoint_sha256=checkpoint_pins,inference_batch_size=32,dfn_batch_size=16,
        source_sha256=engine.file_hash(Path(__file__)),test_evaluated=False))
    engine.write_json(STUDY/"evaluation_complete.json",dict(predictions=len(predictions),test_evaluated=False))
    print("FOCUS_EVALUATION_COMPLETE",len(predictions),flush=True)


if __name__=="__main__":main()
