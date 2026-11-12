"""Independently verify training, split isolation, traces and checkpoint replay."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from sklearn.metrics import accuracy_score,f1_score
from torch.utils.data import DataLoader

from src import engine
from src.config import load_config
from experiments.conformer.model import ConformerClassifier,load_source
from experiments.conformer.run import Clips
from experiments.conformer_aug.sources import ROOT,STUDY,REPORT
from experiments.conformer_aug.run import CONDITIONS
from experiments.noise_stress.mixing import crop_noise
from experiments.noise_stress.run import read_wave


def score(frame):
    return dict(accuracy=accuracy_score(frame.true_idx,frame.pred_idx),
                macro_f1=f1_score(frame.true_idx,frame.pred_idx,labels=list(range(11)),average="macro",zero_division=0))


def main():
    torch.set_num_threads(2);engine.seed_all(42)
    import speechbrain
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    protocol=json.loads((REPORT/"protocol.json").read_text())
    cfg=load_config(ROOT/"configs/task2_real.json")
    pins={**protocol["source_sha256"],**protocol["baseline_pins"]}
    assert all(engine.file_hash(ROOT/p)==h for p,h in pins.items())
    assert all(engine.file_hash(Path(cfg.data_root)/p)==h for p,h in protocol["development_wav_sha256"].items())
    train_sources=protocol["training_noise"]["sources"]
    eval_sources=protocol["evaluation_sources"]
    for key in ("original_filename","original_sha256","prepared_sha256"):
        assert not {s[key] for s in train_sources}&{s[key] for s in eval_sources}
    assert all(engine.file_hash(ROOT/s["prepared_path"])==s["prepared_sha256"] for s in train_sources+eval_sources)
    predictions=pd.read_csv(REPORT/"predictions.csv.gz")
    units=pd.read_csv(REPORT/"condition_metrics.csv")
    keys=["condition","heldout_speaker","kind","noise_id","category","snr_db","draw"]
    groups=predictions.groupby(keys,sort=False)
    for row in units.itertuples():
        part=groups.get_group(tuple(getattr(row,k) for k in keys))
        assert len(part)==row.n
        actual=score(part)
        assert abs(actual["accuracy"]-row.accuracy)<1e-12 and abs(actual["macro_f1"]-row.macro_f1)<1e-12
    assert not predictions.path.str.contains("spk01").any()
    state,statistics=load_source(ROOT/"runs/conformer-research")
    recordings={s["noise_id"]:sf.read(ROOT/s["prepared_path"],dtype="float32")[0] for s in eval_sources}
    manifest=pd.read_csv(ROOT/"reports/noise-stress-v1/mixtures.csv.gz")
    details=[];epoch_rows=0;original_rows=0;stress_rows=0;reconstructed=0
    for speaker in protocol["folds"]:
        run=STUDY/"folds"/speaker/"legacy"
        frame=pd.read_csv(run/"metadata.csv").fillna("")
        train=frame[frame.split.eq("train")]
        val=frame[frame.split.eq("val")]
        assert len(train)==609 and len(val)==203
        assert not set(train.path)&set(val.path) and not train.group.isin(("speaker:spk01","speaker:"+speaker)).any()
        rms=[np.sqrt(np.mean(read_wave(Path(cfg.data_root)/p).astype(np.float64)**2)) for p in train[train.label.ne("silence")].path]
        reference=float(np.median(rms)**2)
        candidates=manifest[manifest.heldout_speaker.eq(speaker)&manifest.path.str.startswith("audio/")&manifest.draw.eq(0)]
        selected=[]
        for _,g in candidates.groupby(["noise_id","snr_db"]):
            selected.append(g[g.silence_reference.eq(False)].iloc[37])
            selected.append(g[g.silence_reference.eq(True)].iloc[0])
        selected=pd.DataFrame(selected)
        waves=[]
        for row in selected.itertuples():
            original=read_wave(Path(cfg.data_root)/row.path)
            recording=recordings[row.noise_id]
            noise=recording[row.start_sample:row.end_sample].astype(np.float64);noise-=noise.mean()
            expected,trace=crop_noise(recording,row.path,row.noise_id,row.draw)
            assert trace["start_sample"]==row.start_sample and np.array_equal(noise,expected)
            power=reference if row.silence_reference else float(np.mean(original.astype(np.float64)**2))
            gain=np.sqrt(power/(np.mean(noise**2)*10**(row.snr_db/10)))
            mixed=original.astype(np.float64)+gain*noise
            mixed=(mixed*min(1.,.99/np.max(np.abs(mixed)))).astype(np.float32)
            assert hashlib.sha256(mixed.tobytes()).hexdigest()==row.mixture_sha256
            waves.append(mixed);reconstructed+=1
        waves=np.stack(waves)
        for condition in CONDITIONS:
            run=STUDY/"folds"/speaker/condition
            init=json.loads((run/"initialization.json").read_text())
            assert abs(init["silence_reference_power"]-reference)<1e-12
            summary=json.loads((run/"summary.json").read_text())
            assert engine.file_hash(run/"best.pt")==summary["checkpoint_sha256"]
            history=pd.read_csv(run/"history.csv")
            assert history.epoch.tolist()==list(range(1,11))
            assert int(history.iloc[int(np.argmax(history.val_macro_f1))].epoch)==summary["best_epoch"]
            for epoch in range(1,11):
                old=pd.read_csv(run/"epochs"/f"epoch_{epoch:02d}.csv")
                assert set(old.path)==set(val.path)
                actual=score(old);h=history[history.epoch.eq(epoch)].iloc[0]
                assert abs(actual["macro_f1"]-h.val_macro_f1)<1e-12
                epoch_rows+=len(old)
                trace=pd.read_csv(run/"epochs"/f"augmentation_{epoch:02d}.csv.gz")
                assert len(trace)==609 and trace.draw.tolist()==list(range(609))
                assert set(trace.path)<=set(train.path) and trace.epoch.eq(epoch).all()
                selected_noise=trace[trace.noise_selected]
                if condition=="legacy":assert selected_noise.empty
                else:
                    assert set(selected_noise.noise_id)<={s["noise_id"] for s in train_sources}
                    assert selected_noise.snr_db.between(-5,20).all()
                    assert np.max(np.abs(selected_noise.snr_db-selected_noise.snr_db_measured))<1e-8
            model=ConformerClassifier(state,statistics,"finetune").cuda().eval()
            checkpoint=torch.load(run/"best.pt",map_location="cpu",weights_only=True)
            model.load_state_dict(checkpoint["model"],strict=True)
            assert torch.equal(model.feature_mean.cpu(),statistics["glob_mean"])
            assert torch.equal(model.feature_std.cpu(),statistics["glob_std"])
            for kind,part,target_name in (("validation",val,"predictions.csv"),("original_train",train,"original_train_predictions.csv")):
                actual,p=engine.evaluate_model(model,DataLoader(Clips(part,cfg),batch_size=32),"cuda",11)
                saved=pd.read_csv(run/target_name).set_index("path")
                assert np.array_equal(saved.loc[p.path].pred_idx,p.pred_idx)
                assert abs(actual["macro_f1"]-summary[kind]["macro_f1"])<1e-12
                original_rows+=len(p)
            with torch.inference_mode():
                indices=[]
                for start in range(0,len(waves),32):indices.extend(model(torch.from_numpy(waves[start:start+32]).cuda()).argmax(1).cpu().tolist())
            old=predictions[predictions.condition.eq(condition)&predictions.heldout_speaker.eq(speaker)&predictions.kind.eq("mixed")]
            wanted=selected[["path","noise_id","snr_db","draw"]].merge(old,on=["path","noise_id","snr_db","draw"],validate="one_to_one")
            assert np.array_equal(wanted.pred_idx,indices)
            stress_rows+=len(indices)
            details.append(dict(condition=condition,speaker=speaker,original_predictions_replayed=812,stress_predictions_replayed=len(indices),
                                checkpoint_sha256=summary["checkpoint_sha256"]))
            del model,checkpoint;torch.cuda.empty_cache()
    assert original_rows==9744 and epoch_rows==24360 and stress_rows==768 and reconstructed==256
    assert all(engine.file_hash(ROOT/p)==h for p,h in pins.items())
    audit=dict(metric_conditions_recomputed=len(units),prediction_rows=len(predictions),epoch_prediction_rows_recomputed=epoch_rows,
               original_predictions_replayed=original_rows,stress_predictions_replayed=stress_rows,
               stress_waveforms_reconstructed=reconstructed,all_predictions_identical=True,
               training_noise_originals_disjoint=True,source_and_baseline_hashes_match=True,
               normalization_unchanged=True,test_evaluated=False,details=details,
               verifier_sha256=engine.file_hash(Path(__file__)))
    engine.write_json(REPORT/"independent_audit.json",audit)
    print("AUG_VERIFIED",json.dumps(audit),flush=True)


if __name__=="__main__":main()
