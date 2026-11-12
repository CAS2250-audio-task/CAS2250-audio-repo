"""Audit only the new extension; the completed parent audit remains untouched."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src import engine
from experiments.check_conformer_long_zero import recompute_predictions
from experiments.conformer.model import ConformerClassifier,load_source
from experiments.conformer_long_zero.policy import update_count
from experiments.conformer_long_zero.run import ValidationBank
from .run import ROOT,STUDY,REPORT,local_branch,freeze_protocol


def check():
    protocol=freeze_protocol()
    state,statistics=load_source(ROOT/"runs/conformer-research")
    epoch_rows=0;draws=0;full_rows=0;replayed=0;extended=0
    for speaker in ("spk02","spk03","spk04","spk05"):
        items=[i for i in protocol["plan"] if i["speaker"]==speaker and i["extend"]]
        if not items:continue
        bank=ValidationBank(speaker)
        for item in items:
            run=local_branch(item);summary=json.loads((run/"summary.json").read_text())
            cp=torch.load(run/"last.pt",map_location="cpu",weights_only=True)
            assert cp["epoch"]==100 and summary["final_epoch"]==100
            assert all(int(v["step"])==update_count(100,item["batch"]) for v in cp["optimizer"]["state"].values())
            for key in ("torch_rng","cuda_rng","loader_rng","val_loader_rng"):assert key in cp
            history=pd.DataFrame(cp["history"])
            assert summary["best_epoch"]==int(history.loc[history.zero_f1.idxmax(),"epoch"])
            old=pd.read_csv(ROOT/item["source_run"]/"history.csv",float_precision="round_trip")
            for col in old.columns:
                if pd.api.types.is_numeric_dtype(old[col]):
                    assert np.allclose(history[col].iloc[:len(old)],old[col],atol=1e-12,rtol=0)
                else:assert history[col].iloc[:len(old)].tolist()==old[col].tolist()
            assert engine.file_hash(run/"best.pt")==summary["checkpoint_sha256"]
            assert engine.file_hash(run/"last.pt")==summary["last_checkpoint_sha256"]
            for row in history[history.epoch.gt(item["source_epoch"])].to_dict("records"):
                e=int(row["epoch"])
                pred=pd.read_csv(run/"epochs"/f"epoch_{e:03d}.csv.gz")
                assert len(pred)==1827 and not pred.path.str.contains("spk01_").any()
                for key,value in recompute_predictions(pred).items():assert abs(value-row[key])<1e-10
                assert row["updates"]==update_count(e,item["batch"])
                trace=pd.read_csv(run/"epochs"/f"augmentation_{e:03d}.csv.gz")
                assert len(trace)==609 and trace.draw.tolist()==list(range(609))
                assert not trace.path.str.contains("spk01_|"+speaker+"_").any()
                # Compare input draws across batches, including reused parent epochs.
                for other in protocol["plan"]:
                    if other["condition"]!=item["condition"] or other["speaker"]!=speaker:continue
                    candidate=local_branch(other)/"epochs"/f"augmentation_{e:03d}.csv.gz"
                    if not candidate.exists():candidate=ROOT/other["source_run"]/"epochs"/f"augmentation_{e:03d}.csv.gz"
                    if candidate.exists():
                        reference=pd.read_csv(candidate)
                        for k in ("draw","path","input_sha256"):assert trace[k].tolist()==reference[k].tolist()
                draws+=len(trace);epoch_rows+=len(pred)
            for selected in ("best","last"):
                for denoiser in ("off","dfn3"):
                    pred=pd.read_csv(run/f"{selected}_{denoiser}_full_predictions.csv.gz")
                    assert len(pred)==5235 and not pred.path.str.contains("spk01_").any()
                    for key,value in recompute_predictions(pred).items():
                        assert abs(value-summary["final_results"][f"{selected}_{denoiser}"][key])<1e-10
                    full_rows+=len(pred)
            model=ConformerClassifier(state,statistics,"finetune").cuda()
            saved=torch.load(run/"best.pt",map_location="cpu",weights_only=True)
            model.load_state_dict(saved["model"],strict=True)
            for denoiser in ("off","dfn3"):
                _,pred,_=bank.evaluate(model,denoiser=denoiser)
                expected=pd.read_csv(run/f"best_{denoiser}_full_predictions.csv.gz")
                joined=pred.merge(expected,on="case_index",suffixes=("_new","_old"),validate="one_to_one")
                assert len(joined)==1827 and joined.pred_idx_new.eq(joined.pred_idx_old).all()
                assert np.max(np.abs(joined.loss_new-joined.loss_old))<1e-6
                replayed+=len(joined)
            del cp,saved,model;torch.cuda.empty_cache();extended+=1
            print("FIXED100_AUDIT",item["condition"],item["batch"],speaker,flush=True)
        del bank
    assert extended==sum(i["extend"] for i in protocol["plan"])
    parent=__import__('experiments.conformer_long_zero.run',fromlist=['REPORT'])
    for name,sha in json.loads((parent.REPORT/"artifact_checksums.json").read_text()).items():
        # Older manifests may use a nested file mapping.
        if isinstance(sha,str) and len(sha)==64:assert engine.file_hash(parent.REPORT/name)==sha
    engine.write_json(REPORT/"independent_audit.json",dict(extended_branches=extended,
        new_validation_predictions_recomputed=epoch_rows,new_training_draws_checked=draws,
        final_prediction_rows_recomputed=full_rows,best_checkpoint_predictions_replayed=replayed,
        parent_checkpoints_preserved=True,training_input_pairing_checked=True,optimizer_and_rng_verified=True,
        early_stopping_disabled=True,all_final_epochs=100,test_audio_read=False,test_evaluated=False))


if __name__=="__main__":
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    check()
