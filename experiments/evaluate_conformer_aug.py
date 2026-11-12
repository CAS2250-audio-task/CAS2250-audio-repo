"""Fixed stress evaluation for augmentation models; original waveform hashes match."""
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
from experiments.conformer_aug.run import CONDITIONS
from experiments.conformer_aug.sources import ROOT,STUDY,REPORT


@torch.inference_mode()
def predict(models,waves):
    result={name:[] for name in models}
    for start in range(0,len(waves),32):
        batch=torch.from_numpy(waves[start:start+32]).cuda()
        for name,model in models.items():result[name].extend(model(batch).argmax(1).cpu().tolist())
    return result


def rows(predictions,frame,speaker,kind,noise_id,category,level,draw):
    result=[]
    for condition,indices in predictions.items():
        result.extend(dict(condition=condition,heldout_speaker=speaker,kind=kind,noise_id=noise_id,
                           category=category,snr_db=level,draw=draw,path=path,true_idx=int(y),pred_idx=int(pred))
                      for path,y,pred in zip(frame.path,frame.true_idx,indices))
    return pd.DataFrame(result)


def main():
    torch.set_num_threads(2)
    import speechbrain
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    assert (STUDY/"training_complete.json").exists()
    protocol=json.loads((REPORT/"protocol.json").read_text())
    assert all(engine.file_hash(ROOT/p)==h for p,h in {**protocol["source_sha256"],**protocol["baseline_pins"]}.items())
    cfg=load_config(ROOT/"configs/task2_real.json")
    metadata=read_metadata(cfg)
    lookup={label:i for i,label in enumerate(cfg.labels)}
    state,statistics=load_source(ROOT/"runs/conformer-research")
    eval_protocol=json.loads((ROOT/"reports/noise-stress-v1/protocol.json").read_text())
    assert engine.file_hash(ROOT/"reports/noise-stress-v1/protocol.json")==protocol["evaluation_protocol_sha256"]
    sources=eval_protocol["noise_sources"]
    recordings={}
    for src in sources:
        path=ROOT/src["prepared_path"]
        assert engine.file_hash(path)==src["prepared_sha256"]
        recordings[src["noise_id"]]=sf.read(path,dtype="float32")[0]
    manifests=pd.read_csv(ROOT/"reports/noise-stress-v1/mixtures.csv.gz")
    prior_hashes=manifests.set_index(["heldout_speaker","path","noise_id","snr_db","draw"]).mixture_sha256.to_dict()
    partials=STUDY/"evaluation_parts";partials.mkdir(exist_ok=True)
    hashes_checked=0;done=0
    checkpoint_pins={}
    for speaker in protocol["folds"]:
        frame=metadata[metadata.group.eq("speaker:"+speaker)].reset_index(drop=True).copy()
        frame["true_idx"]=frame.label.map(lookup)
        assert len(frame)==203 and speaker!="spk01"
        original=np.stack([read_wave(Path(cfg.data_root)/p) for p in frame.path])
        models={}
        for condition in CONDITIONS:
            run=STUDY/"folds"/speaker/condition
            summary=json.loads((run/"summary.json").read_text())
            assert engine.file_hash(run/"best.pt")==summary["checkpoint_sha256"]
            checkpoint_pins[str((run/"best.pt").relative_to(ROOT))]=summary["checkpoint_sha256"]
            model=ConformerClassifier(state,statistics,"finetune")
            checkpoint=torch.load(run/"best.pt",map_location="cpu",weights_only=True)
            model.load_state_dict(checkpoint["model"],strict=True)
            models[condition]=model.cuda().eval()
            del checkpoint
        actual=predict(models,original)
        for condition,indices in actual.items():
            saved=pd.read_csv(STUDY/"folds"/speaker/condition/"predictions.csv").set_index("path")
            assert np.array_equal(saved.loc[frame.path].pred_idx,indices)
        rows(actual,frame,speaker,"clean","clean","clean",999,0).to_csv(partials/f"{speaker}_clean.csv.gz",index=False)
        training=metadata[metadata.group.ne("speaker:"+speaker)&metadata.group.ne("speaker:spk01")&metadata.label.ne("silence")]
        assert len(training)==600
        rms=[float(np.sqrt(np.mean(read_wave(Path(cfg.data_root)/p).astype(np.float64)**2))) for p in training.path]
        reference=float(np.median(rms)**2)
        for source in sources:
            noise_id=source["noise_id"];recording=recordings[noise_id]
            for draw in range(DRAWS):
                crops=[crop_noise(recording,path,noise_id,draw)[0] for path in frame.path]
                for level in LEVELS:
                    target=partials/f"{speaker}_{noise_id}_{draw}_{level}.csv.gz"
                    if target.exists():
                        done+=1;hashes_checked+=203
                        continue
                    mixtures=[]
                    for i,noise in enumerate(crops):
                        power=reference if frame.true_idx.iloc[i]==10 else float(np.mean(original[i].astype(np.float64)**2))
                        wave,audit=mix_at_snr(original[i],noise,level,power)
                        assert audit["mixture_sha256"]==prior_hashes[(speaker,frame.path.iloc[i],noise_id,level,draw)]
                        hashes_checked+=1;mixtures.append(wave)
                    result=predict(models,np.stack(mixtures))
                    rows(result,frame,speaker,"mixed",noise_id,source["category"],level,draw).to_csv(target,index=False)
                    done+=1
                    if done%16==0:print("AUG_STRESS_PROGRESS",done,384,flush=True)
            only_frame=pd.DataFrame(dict(path=[f"noise-only/{i:02d}" for i in range(20)],true_idx=[10]*20))
            for level in LEVELS:
                target=partials/f"{speaker}_{noise_id}_{level}_noiseonly.csv.gz"
                if target.exists():hashes_checked+=20;continue
                mixtures=[]
                for i in range(20):
                    noise,_=crop_noise(recording,"noise-only",noise_id,i)
                    wave,audit=mix_at_snr(np.zeros(SAMPLES,dtype=np.float32),noise,level,reference)
                    assert audit["mixture_sha256"]==prior_hashes[(speaker,f"noise-only/{i:02d}",noise_id,level,i)]
                    hashes_checked+=1;mixtures.append(wave)
                result=predict(models,np.stack(mixtures))
                rows(result,only_frame,speaker,"noise_only",noise_id,source["category"],level,0).to_csv(target,index=False)
        del model,models
        torch.cuda.empty_cache()
        print("AUG_STRESS_FOLD_DONE",speaker,flush=True)
    assert hashes_checked==80512
    predictions=pd.concat([pd.read_csv(path) for path in partials.glob("*.csv.gz")],ignore_index=True)
    assert len(predictions)==243972
    baseline=pd.read_csv(ROOT/"reports/noise-stress-v1/predictions.csv.gz")
    baseline=baseline[baseline.model.eq("conformer")].drop(columns="model").rename(columns={"condition":"kind"})
    baseline["condition"]="baseline"
    assert len(baseline)==81324
    all_predictions=pd.concat([baseline,predictions],ignore_index=True)
    all_predictions.to_csv(REPORT/"predictions.csv.gz",index=False)
    engine.write_json(REPORT/"evaluation_audit.json",dict(prediction_rows=len(all_predictions),
                      new_prediction_rows=len(predictions),baseline_rows=len(baseline),
                      old_stress_mixture_hashes_matched=hashes_checked,clean_predictions_replayed=2436,
                      new_checkpoint_sha256=checkpoint_pins,all_models_evaluation_mode=True,
                      augmentation_disabled_in_validation=True,test_evaluated=False))
    print("AUG_EVALUATION_COMPLETE",len(all_predictions),flush=True)


if __name__=="__main__":main()
