"""Independent training-trace, metric, original-prediction and stress replay checks."""
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score
import soundfile as sf
import torch

from src import engine
from src.config import load_config
from src.dataset import read_metadata
from experiments.conformer.model import ConformerClassifier,load_source
from experiments.conformer.run import Clips
from experiments.conformer_sanity import speaker_fold
from experiments.noise_stress.mixing import crop_noise,mix_at_snr
from experiments.noise_stress.run import read_wave
from experiments.denoise_eval.filters import make_filter
from experiments.conformer_noise_focus.sources import ROOT,STUDY,REPORT
from experiments.conformer_noise_focus.run import CONDITIONS,protocol
from experiments.conformer_noise_focus.augment import FocusTrainingClips
from experiments.evaluate_conformer_noise_focus import TRAINING,checkpoint,predict


@torch.inference_mode()
def main():
    torch.set_num_threads(2);engine.seed_all(42)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    cfg=load_config(ROOT/"configs/task2_real.json");metadata=read_metadata(cfg)
    frozen=protocol(cfg,metadata)
    for group in ("source_sha256","baseline_pins","dependency_sha256"):
        for path,expected in frozen[group].items():assert engine.file_hash(ROOT/path)==expected,path
    for path,expected in frozen["development_wav_sha256"].items():
        assert engine.file_hash(Path(cfg.data_root)/path)==expected
    p=pd.read_csv(REPORT/"predictions.csv.gz")
    assert len(p)==650592 and set(p.heldout_speaker)==set(frozen["folds"])
    assert not p.path.str.contains("spk01").any()
    keys=["training_condition","denoiser","heldout_speaker","kind","noise_id","category","snr_db","draw"]
    assert not p.duplicated(keys+["path"]).any()
    scores=pd.read_csv(REPORT/"condition_metrics.csv").set_index(keys)
    for key,g in p.groupby(keys):
        row=scores.loc[key]
        assert abs(row.accuracy-accuracy_score(g.true_idx,g.pred_idx))<1e-12
        assert abs(row.macro_f1-f1_score(g.true_idx,g.pred_idx,labels=list(range(11)),average="macro",zero_division=0))<1e-12
        s=g[g.true_idx.eq(10)]
        assert abs(row.false_command_rate-float(s.pred_idx.ne(10).mean()))<1e-12
    expected_rows=0;audited_inputs=0
    for file in (STUDY/"evaluation_audits").glob("*.json"):
        audit=json.loads(file.read_text())
        prediction_file=STUDY/"evaluation_parts"/(file.stem+".csv.gz")
        assert engine.file_hash(prediction_file)==audit["prediction_file_sha256"]
        assert len(audit["input_sha256"])==len(audit["paths"])==len(audit["dfn3_output_sha256"])
        expected_rows+=audit["prediction_rows"];audited_inputs+=len(audit["paths"])
    assert expected_rows==650592 and audited_inputs==81324
    traces=pd.read_csv(REPORT/"training_draws.csv.gz",low_memory=False)
    assert len(traces)==97440
    idx=["heldout_speaker","epoch","draw"]
    base=traces[traces.training_condition.eq("baseline")].set_index(idx).sort_index()
    for condition in CONDITIONS:
        current=traces[traces.training_condition.eq(condition)].set_index(idx).sort_index()
        assert current.index.equals(base.index) and current.path.equals(base.path)
        assert current.noise_selected.equals(base.noise_selected)
        unchanged=~current.added_speech_selected&~current.near_zero_selected
        assert current[unchanged].input_sha256.equals(base[unchanged].input_sha256)
    speech=traces[traces.training_condition.eq("speech")].set_index(idx).sort_index()
    focus=traces[traces.training_condition.eq("focus")].set_index(idx).sort_index()
    both=traces[traces.training_condition.eq("speech_focus")].set_index(idx).sort_index()
    assert speech.added_speech_selected.equals(both.added_speech_selected)
    assert focus.near_zero_selected.equals(both.near_zero_selected)
    mask=both.noise_selected
    assert speech[mask].noise_id.equals(both[mask].noise_id)
    assert np.array_equal(focus[mask].snr_db,both[mask].snr_db)
    recordings={}
    for s in frozen["original_noise_sources"]+frozen["added_speech_sources"]:
        assert engine.file_hash(ROOT/s["prepared_path"])==s["prepared_sha256"]
        recordings[s["noise_id"]]=sf.read(ROOT/s["prepared_path"],dtype="float32")[0]
    state,stats=load_source(ROOT/"runs/conformer-research")
    original_predictions=0;epoch_metric_groups=0;trace_replay=0;stress_replay=0;filter_replay=0
    dfn=make_filter("deepfilternet3")
    eval_sources={s["noise_id"]:s for s in json.loads((ROOT/"reports/noise-stress-v1/noise_sources.json").read_text())}
    for speaker in frozen["folds"]:
        fold=speaker_fold(metadata,speaker,cfg)
        train=fold[fold.split.eq("train")]
        val=fold[fold.split.eq("val")].reset_index(drop=True)
        all_original=pd.concat([train,val],ignore_index=True)
        dataset=Clips(all_original,cfg)
        models={}
        for condition in TRAINING:
            file=checkpoint(condition,speaker)
            summary=json.loads((file.parent/"summary.json").read_text())
            assert engine.file_hash(file)==summary["checkpoint_sha256"]
            model=ConformerClassifier(state,stats,"finetune")
            model.load_state_dict(torch.load(file,map_location="cpu",weights_only=True)["model"],strict=True)
            models[condition]=model.cuda().eval()
            assert torch.equal(model.feature_mean.cpu(),stats["glob_mean"])
            assert torch.equal(model.feature_std.cpu(),stats["glob_std"])
            if condition=="baseline":continue
            run=file.parent
            saved=pd.concat([pd.read_csv(run/"original_train_predictions.csv"),pd.read_csv(run/"predictions.csv")]).set_index("path")
            original_waves=np.stack([w.numpy() for w in dataset.wavs])
            # Preserve the original609/203 loader boundaries, including final batch sizes.
            actual=(predict({condition:model},original_waves[:609])[condition]
                    +predict({condition:model},original_waves[609:])[condition])
            assert actual==saved.loc[all_original.path].pred_idx.tolist()
            original_predictions+=len(actual)
            history=pd.read_csv(run/"history.csv",float_precision="round_trip")
            best_epoch=int(history.loc[history.val_macro_f1.idxmax()].epoch)
            assert summary["best_epoch"]==best_epoch
            initial=json.loads((run/"initialization.json").read_text())
            old_init=json.loads((ROOT/f"runs/conformer-aug-v1/folds/{speaker}/noise/initialization.json").read_text())
            assert initial==old_init
            for row in history.itertuples():
                pred=pd.read_csv(run/"epochs"/f"epoch_{row.epoch:02d}.csv")
                assert abs(f1_score(pred.true_idx,pred.pred_idx,labels=list(range(11)),average="macro",zero_division=0)-row.val_macro_f1)<1e-12
                assert abs(accuracy_score(pred.true_idx,pred.pred_idx)-row.val_accuracy)<1e-12
                epoch_metric_groups+=1
            data=FocusTrainingClips(train,cfg,speaker,frozen["original_noise_sources"],frozen["added_speech_sources"],recordings,**CONDITIONS[condition])
            sample=traces[traces.training_condition.eq(condition)&traces.heldout_speaker.eq(speaker)].sample(24,random_state=42)
            for row in sample.itertuples():
                data.set_epoch(int(row.epoch));data.draw=int(row.draw)
                index=int(np.flatnonzero(data.frame.path.eq(row.path))[0])
                wave,_,_=data[index]
                assert hashlib.sha256(wave.numpy().tobytes()).hexdigest()==row.input_sha256
                trace_replay+=1
        # Rebuild several full-batch stress windows, including a very noisy level.
        for kind,noise_id,level,draw in (("clean","clean",999,0),("mixed","restaurant",0,0),("mixed","compressor",-10,1),("noise_only","cafeteria",0,0)):
            name=f"{speaker}_{kind}_{noise_id}_{level}_{draw}"
            audit=json.loads((STUDY/"evaluation_audits"/(name+".json")).read_text())
            paths=audit["paths"]
            if kind=="clean":
                waves=np.stack([read_wave(Path(cfg.data_root)/path) for path in paths])
            else:
                recording=sf.read(ROOT/eval_sources[noise_id]["prepared_path"],dtype="float32")[0]
                keyword=np.stack([read_wave(Path(cfg.data_root)/path) for path in train[train.label.ne("silence")].path])
                reference=float(np.median(np.sqrt(np.mean(keyword.astype(np.float64)**2,axis=1)))**2)
                waves=[]
                for path in paths:
                    is_only=kind=="noise_only"
                    noise,_=crop_noise(recording,"noise-only" if is_only else path,noise_id,int(path.split("/")[-1]) if is_only else draw)
                    original=np.zeros(24000,np.float32) if is_only else read_wave(Path(cfg.data_root)/path)
                    power=reference if is_only or "/silence/" in path else float(np.mean(original.astype(np.float64)**2))
                    waves.append(mix_at_snr(original,noise,level,power)[0])
                waves=np.stack(waves)
            assert [hashlib.sha256(w.tobytes()).hexdigest() for w in waves]==audit["input_sha256"]
            if audit["previous_dfn_cache_reused"]:
                cases=pd.read_csv(ROOT/f"runs/denoise-eval-v1/{speaker}_cases.csv.gz").set_index(["kind","noise_id","snr_db","draw","path"])
                bank=np.load(ROOT/f"runs/denoise-eval-v1/filtered/deepfilternet3/{speaker}.npy",mmap_mode="r")
                inputs=np.load(ROOT/f"runs/denoise-eval-v1/{speaker}_inputs.npy",mmap_mode="r")
                indices=[int(cases.loc[(kind,noise_id,level,draw,path)].case_index) for path in paths]
                # Replay the exact old16-clip boundaries, including surrounding clips;
                # changing GPU batch shape can change a few near-zero output samples.
                blocks={start:dfn(np.array(inputs[start:start+16])) for start in sorted({i//16*16 for i in indices})}
                filtered=np.stack([blocks[i//16*16][i%16] for i in indices])
                saved=np.stack([bank[i] for i in indices])
                assert np.array_equal(filtered,saved),(speaker,name,float(np.abs(filtered-saved).max()))
            else:
                filtered=np.concatenate([dfn(waves[i:i+16]) for i in range(0,len(waves),16)])
            assert [hashlib.sha256(w.tobytes()).hexdigest() for w in filtered]==audit["dfn3_output_sha256"]
            filter_replay+=len(waves)
            for denoiser,inputs in (("off",waves),("dfn3",filtered)):
                actual=predict(models,inputs)
                selected=p[p.heldout_speaker.eq(speaker)&p.kind.eq(kind)&p.noise_id.eq(noise_id)&p.snr_db.eq(level)&p.draw.eq(draw)&p.denoiser.eq(denoiser)]
                for condition in TRAINING:
                    expected=selected[selected.training_condition.eq(condition)].set_index("path")
                    assert actual[condition]==expected.loc[paths].pred_idx.tolist(),(speaker,name,denoiser,condition)
                    stress_replay+=len(paths)
            del waves,filtered
        del model,models,dataset
        gc.collect();torch.cuda.empty_cache()
        print("FOCUS_INDEPENDENT_FOLD_OK",speaker,flush=True)
    engine.write_json(REPORT/"independent_audit.json",dict(prediction_rows=len(p),metric_groups_recomputed=len(scores),
        evaluation_input_cases_with_hashes=audited_inputs,epoch_metric_groups_recomputed=epoch_metric_groups,
        original_predictions_replayed=original_predictions,training_input_waveforms_replayed=trace_replay,
        stress_predictions_replayed=stress_replay,dfn_waveforms_replayed=filter_replay,
        matched_sampler_paths_and_independent_policies=True,unchanged_inputs_match_original_baseline=True,
        original_initializations_and_normalization_preserved=True,all_replayed_predictions_equal=True,
        source_checkpoint_dataset_pins_valid=True,test_evaluated=False,
        code_sha256=engine.file_hash(Path(__file__))))
    print("FOCUS_INDEPENDENT_AUDIT_COMPLETE",stress_replay,flush=True)


if __name__=="__main__":main()
