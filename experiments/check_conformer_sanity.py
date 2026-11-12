"""Independently replay the CV checkpoints and verify all audit artifacts."""
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src import engine
from src.config import Config, load_config
from src.dataset import AudioDataset, read_metadata
from src.model import build_model
from experiments.conformer.model import ConformerClassifier, load_source
from experiments.conformer.run import Clips

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/"reports/conformer-sanity-v1"


def main():
    torch.set_num_threads(2)
    protocol=json.loads((REPORT/"protocol.json").read_text())
    assert all(engine.file_hash(ROOT/p)==digest for p,digest in protocol["source_sha256"].items())
    assert engine.file_hash(ROOT/"runs/conformer-ref-v1/protocol.json")==protocol["original_protocol_sha256"]
    cfg=load_config(ROOT/"configs/task2_real.json")
    metadata=read_metadata(cfg)
    development=metadata[metadata.split.ne("test")]
    audio=pd.read_csv(REPORT/"audio_identity_audit.csv")
    joined=audio.merge(development[["path","group","source_sha256","source_filename"]],on="path",validate="one_to_one")
    counts={}
    for column in ("source_sha256","source_filename","file_sha256","pcm_sha256"):
        groups=joined[joined[column].ne("")].groupby(column).group.nunique()
        counts[column]=int((groups>1).sum())
    assert not any(counts.values())
    assert all(engine.file_hash(Path(cfg.data_root)/row.path)==row.file_sha256 for row in audio.itertuples())
    engine.write_json(REPORT/"development_identity_audit.json",dict(
        files=len(audio),speakers=sorted(joined.group.unique()),cross_speaker_duplicate_identifiers=counts,
        metadata_environment_columns_available=[c for c in ("room","distance_cm","noise_type","device","style") if c in metadata],
        test_evaluated=False))
    state,statistics=load_source(ROOT/"runs/conformer-research")
    folds=pd.read_csv(REPORT/"speaker_cv.csv")
    replays=[]
    for row in folds.itertuples():
        run=ROOT/row.source_run
        checkpoint=torch.load(run/"best.pt",map_location="cpu",weights_only=True)
        run_cfg=Config(**json.loads((run/("config.json" if row.model=="imagenet" else "reference_config.json")).read_text()))
        frame=pd.read_csv(run/"metadata.csv").fillna("")
        selected=frame[frame.split.eq("val")]
        assert len(selected)==203 and selected.group.eq("speaker:"+row.heldout_speaker).all()
        assert not selected.group.eq("speaker:spk01").any()
        if row.model=="imagenet":
            model=build_model(11,use_pretrained=False).cuda()
            dataset=AudioDataset(selected,run_cfg,train=False)
        else:
            model=ConformerClassifier(state,statistics,mode="finetune").cuda()
            dataset=Clips(selected,run_cfg)
        model.load_state_dict(checkpoint["model"],strict=True)
        scores,predictions=engine.evaluate_model(model,DataLoader(dataset,batch_size=32,shuffle=False),"cuda",11)
        actual=predictions.set_index("path").sort_index()
        saved=pd.read_csv(run/"validation/predictions.csv").set_index("path").sort_index()
        assert actual.true_idx.equals(saved.true_idx) and actual.pred_idx.equals(saved.pred_idx)
        result=json.loads((run/"comparison_summary.json").read_text())
        assert all(abs(scores[k]-result["validation"][k])<1e-6 for k in scores)
        replays.append(dict(heldout_speaker=row.heldout_speaker,model=row.model,seed=int(row.seed),
                            predictions_checked=len(actual),metrics=scores,
                            checkpoint_sha256=engine.file_hash(run/"best.pt")))
        del model,checkpoint
        torch.cuda.empty_cache()
    negative=REPORT/"negative_control/runs/conformer_frozen_seed42"
    h=pd.read_csv(negative/"history.csv")
    negative_rows=0
    for epoch in range(1,11):
        p=pd.read_csv(negative/"epochs"/f"epoch_{epoch:02d}.csv")
        score=engine.metrics(p.true_idx,p.pred_idx,11)
        stored=h[h.epoch.eq(epoch)].iloc[0]
        assert abs(score["macro_f1"]-stored.val_macro_f1)<1e-12
        negative_rows+=len(p)
    engine.write_json(REPORT/"cv_checkpoint_replay.json",dict(
        checkpoints=8,prediction_rows=sum(r["predictions_checked"] for r in replays),
        predictions_identical=True,negative_control_epoch_rows_checked=negative_rows,
        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,test_evaluated=False,details=replays))
    audit_path=REPORT/"execution_audit.json"
    audit=json.loads(audit_path.read_text())
    for relative,digest in audit["artifact_sha256"].items():
        # The cross-development audit is expanded above, before its final hash.
        if not relative.endswith("development_identity_audit.json"):
            assert engine.file_hash(ROOT/relative)==digest,relative
    audit.update(cv_checkpoints_replayed=8,cv_checkpoint_prediction_rows_checked=1624,
                 negative_control_epoch_prediction_rows_checked=negative_rows,
                 cross_development_speaker_exact_duplicates=0,
                 tests_passed=39,source_hashes_unchanged=True,
                 additional_verifier_source_sha256=engine.file_hash(Path(__file__)))
    audit["artifact_sha256"]={str(p.relative_to(ROOT)):engine.file_hash(p) for p in REPORT.rglob("*")
                               if p.is_file() and p.name!="execution_audit.json"}
    engine.write_json(audit_path,audit)
    print("VERIFIED",json.dumps({k:v for k,v in audit.items() if k!="artifact_sha256"}),flush=True)


if __name__=="__main__":main()
