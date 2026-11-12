"""Independently recompute metrics, pairing, LR choices and selected predictions."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src import engine
from experiments.conformer.model import ConformerClassifier,load_source
from experiments.conformer_sanity import SPEAKERS
from experiments.conformer_long_zero.policy import CONDITIONS,BATCHES,LEARNING_RATES,select_learning_rate,update_count,overfit_status
from experiments.conformer_long_zero.run import ROOT,STUDY,REPORT,branch,ValidationBank,write_json


def assert_close(actual,expected):
    assert abs(float(actual)-float(expected))<1e-10,(actual,expected)


def recompute_predictions(predictions):
    scores=[]
    for keys,g in predictions.groupby(["kind","noise_id","snr_db","draw"],sort=False):
        scores.append(dict(kind=keys[0],**engine.metrics(g.true_idx,g.pred_idx,11),loss=float(g.loss.mean())))
    frame=pd.DataFrame(scores)
    zero=frame[frame.kind.eq("mixed")];clean=frame[frame.kind.eq("clean")].iloc[0]
    result=dict(zero_f1=float(zero.macro_f1.mean()),zero_loss=float(zero.loss.mean()),
        zero_accuracy=float(zero.accuracy.mean()),clean_f1=float(clean.macro_f1),clean_loss=float(clean.loss))
    only=predictions[predictions.kind.eq("noise_only")]
    if len(only):result["noise_only_fpr"]=float(only.pred_idx.ne(10).mean())
    return result


def check():
    torch.set_num_threads(2);engine.seed_all(42)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    frozen=json.loads((REPORT/"protocol.json").read_text())
    for mapping in ("source_sha256","initial_checkpoint_sha256","validation_bank_sha256"):
        for relative,sha in frozen[mapping].items():assert engine.file_hash(ROOT/relative)==sha,relative
    for relative,sha in frozen["development_wav_sha256"].items():
        assert "spk01_" not in relative
        assert engine.file_hash(Path(frozen["data_root"])/relative)==sha
    choices=json.loads((REPORT/"lr_selection.json").read_text())
    count=0;trace_count=0;epoch_count=0
    for condition in CONDITIONS:
        for batch in BATCHES:
            pilot_rows=[]
            for lr in LEARNING_RATES:
                for speaker in SPEAKERS:
                    run=branch(condition,batch,lr,speaker)
                    history=pd.read_csv(run/"history.csv",float_precision="round_trip")
                    frozen_pilot=pd.read_csv(run/"pilot_history.csv")
                    marker=json.loads((run/"pilot_complete.json").read_text())
                    assert engine.file_hash(run/"pilot_history.csv")==marker["history_sha256"]
                    pilot_rows.extend(frozen_pilot.to_dict("records"))
                    for row in history.to_dict("records"):
                        epoch=int(row["epoch"])
                        path=run/"source_epoch10_predictions.csv.gz" if epoch==10 else run/"epochs"/f"epoch_{epoch:03d}.csv.gz"
                        predictions=pd.read_csv(path)
                        assert len(predictions)==1827 and not predictions.path.str.contains("spk01_").any()
                        scores=recompute_predictions(predictions)
                        for key,value in scores.items():assert_close(row[key],value)
                        assert row["updates"]==update_count(epoch,batch)
                        count+=len(predictions);epoch_count+=1
                        if epoch>10:
                            traces=pd.read_csv(run/"epochs"/f"augmentation_{epoch:03d}.csv.gz")
                            assert len(traces)==609 and traces.draw.tolist()==list(range(609))
                            assert not traces.path.str.contains("spk01_|"+speaker+"_").any()
                            trace_count+=len(traces)
                            # Every batch/LR branch uses exactly the same input draws for a given condition/epoch.
                            references=[branch(condition,b,l,speaker)/"epochs"/f"augmentation_{epoch:03d}.csv.gz"
                                        for b in BATCHES for l in LEARNING_RATES]
                            reference=next((p for p in references if p.exists()),None)
                            if reference is not None:
                                prior=pd.read_csv(reference)
                                for k in ("draw","path","input_sha256"):
                                    assert prior[k].tolist()==traces[k].tolist(),(condition,batch,lr,speaker,epoch,k)
            selected=next(r for r in choices if r["condition"]==condition and r["batch"]==batch)
            expected=select_learning_rate(pilot_rows)
            for key,value in expected.items():assert_close(selected[key],value)
    state,statistics=load_source(ROOT/"runs/conformer-research")
    replayed=0;final_rows=0
    for speaker in SPEAKERS:
        bank=ValidationBank(speaker)
        for selection in choices:
            run=branch(selection["condition"],selection["batch"],selection["encoder_lr"],speaker)
            summary=json.loads((run/"summary.json").read_text())
            history=pd.read_csv(run/"history.csv",float_precision="round_trip")
            decision=overfit_status(history.to_dict("records"))
            for key in ("stop","reason","confirmation_epoch","latest_bad_streak"):
                assert decision[key]==summary["stop"][key]
            if decision["evidence"]:
                for point in ("reference","current"):
                    for key,value in decision["evidence"][point].items():
                        assert_close(value,summary["stop"]["evidence"][point][key])
            assert summary["stop"]["stop"]
            assert summary["best_epoch"]==int(history.loc[history.zero_f1.idxmax(),"epoch"])
            assert engine.file_hash(run/"best.pt")==summary["checkpoint_sha256"]
            assert engine.file_hash(run/"last.pt")==summary["last_checkpoint_sha256"]
            continuation=torch.load(run/"last.pt",map_location="cpu",weights_only=True)
            assert continuation["epoch"]==int(history.epoch.iloc[-1])
            expected_updates=update_count(continuation["epoch"],selection["batch"])
            assert continuation["optimizer"]["state"]
            for value in continuation["optimizer"]["state"].values():
                assert int(value["step"])==expected_updates
            for key in ("torch_rng","cuda_rng","loader_rng","val_loader_rng"):
                assert key in continuation
            del continuation
            for selected in ("best","last"):
                for denoiser in ("off","dfn3"):
                    preds=pd.read_csv(run/f"{selected}_{denoiser}_full_predictions.csv.gz")
                    assert len(preds)==5235
                    actual=recompute_predictions(preds)
                    for key,value in actual.items():assert_close(value,summary["final_results"][f"{selected}_{denoiser}"][key])
                    final_rows+=len(preds)
            model=ConformerClassifier(state,statistics,"finetune").cuda()
            saved=torch.load(run/"best.pt",map_location="cpu",weights_only=True)
            model.load_state_dict(saved["model"],strict=True);del saved
            assert torch.equal(model.feature_mean.cpu(),statistics["glob_mean"])
            assert torch.equal(model.feature_std.cpu(),statistics["glob_std"])
            for denoiser in ("off","dfn3"):
                scores,actual,_=bank.evaluate(model,denoiser=denoiser)
                expected=pd.read_csv(run/f"best_{denoiser}_full_predictions.csv.gz")
                joined=actual.merge(expected,on="case_index",suffixes=("_new","_old"),validate="one_to_one")
                assert joined.pred_idx_new.eq(joined.pred_idx_old).all()
                assert np.max(np.abs(joined.loss_new-joined.loss_old))<1e-6
                replayed+=len(joined)
            del model;torch.cuda.empty_cache()
            print("LONG_ZERO_AUDIT_REPLAY",speaker,selection["condition"],selection["batch"],flush=True)
        del bank
    write_json(REPORT/"independent_audit.json",dict(validation_epoch_rows_recomputed=count,
        epoch_groups_recomputed=epoch_count,training_draws_checked=trace_count,
        final_prediction_rows_recomputed=final_rows,selected_checkpoint_predictions_replayed=replayed,
        all_initial_checkpoints_and_frozen_sources_preserved=True,paired_inputs_across_batch_and_lr=True,
        learning_rate_selections_recomputed=True,stopping_decisions_recomputed=True,
        optimizer_update_counters_and_resume_states_verified=True,
        test_audio_read=False,test_evaluated=False,checker_sha256=engine.file_hash(Path(__file__))))
    print("LONG_ZERO_INDEPENDENT_AUDIT_COMPLETE",flush=True)


if __name__=="__main__":check()
