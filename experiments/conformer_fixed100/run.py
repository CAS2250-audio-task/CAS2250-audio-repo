"""Resume the frozen long0dB study without its early stopping rule."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F

from src import engine
from src.config import load_config
from src.dataset import read_metadata
from experiments.conformer.model import load_source
from experiments.conformer.run import Clips, loaders, atomic_save
from experiments.conformer_sanity import SPEAKERS, speaker_fold
from experiments.conformer_aug.model import AugmentedConformer
from experiments.conformer_noise_focus.augment import FocusTrainingClips
from experiments.conformer_long_zero import run as parent
from experiments.conformer_long_zero.policy import update_count
from .report import render

ROOT = parent.ROOT
STUDY = ROOT / "runs/conformer-fixed100-v1"
REPORT = ROOT / "reports/conformer-fixed100-v1"
MAX_EPOCH = 100


def local_branch(item):
    return STUDY/"branches"/item["condition"]/f"bs{item['batch']}"/f"lr{item['encoder_lr']:.0e}"/item["speaker"]


def freeze_protocol():
    old = json.loads((parent.REPORT/"protocol.json").read_text())
    choices = json.loads((parent.REPORT/"lr_selection.json").read_text())
    items = []
    for choice in choices:
        for speaker in SPEAKERS:
            source = parent.branch(choice["condition"],choice["batch"],choice["encoder_lr"],speaker)
            summary = json.loads((source/"summary.json").read_text())
            items.append(dict(condition=choice["condition"],batch=choice["batch"],
                encoder_lr=choice["encoder_lr"],speaker=speaker,source_run=str(source.relative_to(ROOT)),
                source_epoch=summary["final_epoch"],source_best_epoch=summary["best_epoch"],
                extend=summary["final_epoch"]<MAX_EPOCH,
                last_sha256=engine.file_hash(source/"last.pt"),best_sha256=engine.file_hash(source/"best.pt"),
                summary_sha256=engine.file_hash(source/"summary.json")))
    pins = dict(old["source_sha256"])
    pins.update({str(p.relative_to(ROOT)):engine.file_hash(p) for p in Path(__file__).parent.glob("*.py")})
    for p,sha in pins.items():
        assert engine.file_hash(ROOT/p)==sha,p
    for p,sha in old["validation_bank_sha256"].items():
        assert engine.file_hash(ROOT/p)==sha,p
    for p,sha in old["development_wav_sha256"].items():
        assert "spk01_" not in p
        assert engine.file_hash(Path(old["data_root"])/p)==sha,p
    result = dict(study="conformer-fixed100-v1",parent_protocol_sha256=engine.file_hash(parent.REPORT/"protocol.json"),
        maximum_total_epoch=MAX_EPOCH,early_stopping=False,
        change="Only remove early stopping; continue each stopped branch from its final LAST checkpoint to epoch100",
        resume="Restore model, AdamW moments and all saved Torch/CUDA/sampler/validation RNG; same constant LR",
        checkpoint_selection=old["checkpoint_selection"],seed=42,plan=items,source_sha256=pins,
        data_root=old["data_root"],development_wav_sha256=old["development_wav_sha256"],
        validation_bank_sha256=old["validation_bank_sha256"],
        original_training_sources=old["original_training_sources"],added_training_sources=old["added_training_sources"],
        baseline_validation="Original203 and fixed0dB draw0x8x203 per fold; final full3draw and noise-only, DFN off/on",
        reused_completed_branches="Branches already at100 are referenced read-only, not retrained",
        test_audio_read=False,test_evaluated=False)
    path=REPORT/"protocol.json"
    if path.exists():assert json.loads(path.read_text())==result,"Frozen fixed100 protocol changed"
    else:parent.write_json(path,result)
    return result


def progress(**values):
    values.update(pid=os.getpid(),updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
        completed=len(list((REPORT/"runs").glob("*.json"))))
    parent.write_json(STUDY/"progress.json",values)
    parent.write_json(REPORT/"progress.json",values)
    print("PROGRESS",json.dumps(values),flush=True)


def save_history(item,history):
    frame=pd.DataFrame(history)
    frame.to_csv(local_branch(item)/"history.csv",index=False)
    target=REPORT/"histories"/(parent.branch_id(item["condition"],item["batch"],item["encoder_lr"],item["speaker"])+".csv")
    target.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(target,index=False)


def run_branch(item,cfg,metadata,state,statistics,frozen,recordings,bank):
    name=parent.branch_id(item["condition"],item["batch"],item["encoder_lr"],item["speaker"])
    summary_path=REPORT/"runs"/(name+".json")
    if summary_path.exists():return
    source=ROOT/item["source_run"]
    if not item["extend"]:
        summary=json.loads((source/"summary.json").read_text())
        summary.update(extended=False,source_epoch=item["source_epoch"],run_dir=str(source.relative_to(ROOT)))
        parent.write_json(summary_path,summary)
        target=REPORT/"histories"/(name+".csv");target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source/"history.csv",target)
        progress(stage="reusing_epoch100",speaker=item["speaker"],condition=item["condition"],batch=item["batch"])
        return
    run=local_branch(item);run.mkdir(parents=True,exist_ok=True)
    latest=run/"last.pt"
    fresh=not latest.exists()
    engine.seed_all(42)
    frame=speaker_fold(metadata,item["speaker"],cfg)
    train_frame=frame[frame.split.eq("train")];val_frame=frame[frame.split.eq("val")]
    assert len(train_frame)==609 and len(val_frame)==203
    assert not train_frame.group.isin(("speaker:spk01","speaker:"+item["speaker"])).any()
    train=FocusTrainingClips(train_frame,cfg,item["speaker"],frozen["original_training_sources"],
        frozen["added_training_sources"],recordings,add_speech=item["condition"]=="speech_focus",focus_zero=item["condition"]=="speech_focus")
    val=Clips(val_frame,cfg)
    train_loader,val_loader,_=loaders(train,val,42,item["batch"])
    model=AugmentedConformer(state,statistics,item["speaker"],mask=False).cuda()
    lr=item["encoder_lr"]
    optimizer=torch.optim.AdamW([
        dict(params=list(model.frontend.parameters())+list(model.transformer.parameters()),lr=lr),
        dict(params=model.fc.parameters(),lr=50*lr)],weight_decay=1e-4)
    cp=torch.load(source/"last.pt" if fresh else latest,map_location="cpu",weights_only=True)
    model.load_state_dict(cp["model"],strict=True);optimizer.load_state_dict(cp["optimizer"])
    assert [g["lr"] for g in optimizer.param_groups]==[lr,50*lr]
    assert all(int(v["step"])==update_count(cp["epoch"],item["batch"]) for v in optimizer.state.values())
    history,best_f1,best_epoch=cp["history"],cp["best_f1"],cp["best_epoch"]
    if fresh:
        assert cp["epoch"]==item["source_epoch"]
        shutil.copy2(source/"best.pt",run/"best.pt")
    scores,predictions,_=bank.evaluate(model)
    for key,value in scores.items():assert abs(value-history[-1][key])<1e-6,(name,key,value,history[-1][key])
    prediction_path=(source if fresh else run)/"epochs"/f"epoch_{cp['epoch']:03d}.csv.gz"
    prior=pd.read_csv(prediction_path)
    joined=predictions.merge(prior,on="case_index",suffixes=("_new","_old"),validate="one_to_one")
    assert len(joined)==1827 and joined.pred_idx_new.eq(joined.pred_idx_old).all()
    assert np.max(np.abs(joined.loss_new-joined.loss_old))<1e-6
    if fresh:
        parent.write_json(run/"initialization.json",dict(source_epoch=cp["epoch"],
            source_checkpoint=str((source/"last.pt").relative_to(ROOT)),source_sha256=item["last_sha256"],
            source_best_sha256=item["best_sha256"],canonical_predictions_replayed=len(joined),
            optimizer_moments_restored=True,learning_rate_unchanged=True))
    torch.set_rng_state(cp["torch_rng"]);torch.cuda.set_rng_state_all(cp["cuda_rng"])
    train_loader.sampler.generator.set_state(cp["loader_rng"])
    val_loader.generator.set_state(cp["val_loader_rng"])
    del cp,prior,joined,predictions
    save_history(item,history)
    for epoch in range(history[-1]["epoch"]+1,MAX_EPOCH+1):
        train.set_epoch(epoch);train.trace=[];model.set_epoch(epoch);model.train()
        total=0.;steps=0;started=time.monotonic()
        for x,y,_ in train_loader:
            x,y=x.cuda(),y.cuda();optimizer.zero_grad(set_to_none=True)
            loss=F.cross_entropy(model(x),y);assert torch.isfinite(loss)
            loss.backward();optimizer.step();total+=float(loss.detach())*len(y);steps+=1
        assert len(train.trace)==609 and steps==math.ceil(609/item["batch"])
        scores,predictions,units=bank.evaluate(model)
        history.append(dict(epoch=epoch,train_loss=total/609,encoder_lr=lr,head_lr=50*lr,
            updates=update_count(epoch,item["batch"]),stage="fixed100",**scores))
        epochs=run/"epochs";epochs.mkdir(exist_ok=True)
        predictions.to_csv(epochs/f"epoch_{epoch:03d}.csv.gz",index=False)
        units.to_csv(epochs/f"units_{epoch:03d}.csv",index=False)
        pd.DataFrame(train.trace).to_csv(epochs/f"augmentation_{epoch:03d}.csv.gz",index=False)
        if scores["zero_f1"]>best_f1:
            best_f1,best_epoch=scores["zero_f1"],epoch
            atomic_save(dict(model=model.state_dict(),epoch=epoch,monitor=scores),run/"best.pt")
        atomic_save(parent.capture(model,optimizer,history,best_f1,best_epoch,train_loader,val_loader),latest)
        save_history(item,history)
        progress(stage="training",condition=item["condition"],batch=item["batch"],speaker=item["speaker"],
            epoch=epoch,best_epoch=best_epoch,zero_f1=scores["zero_f1"],zero_accuracy=scores["zero_accuracy"],
            train_loss=total/609,zero_loss=scores["zero_loss"],seconds=time.monotonic()-started)
        if epoch%10==0:render()
    assert history[-1]["epoch"]==MAX_EPOCH
    assert torch.equal(model.feature_mean.cpu(),statistics["glob_mean"])
    assert torch.equal(model.feature_std.cpu(),statistics["glob_std"])
    final_results={}
    for selected in ("best","last"):
        saved=torch.load(run/f"{selected}.pt",map_location="cpu",weights_only=True)
        model.load_state_dict(saved["model"],strict=True);epoch=saved["epoch"];del saved
        for denoiser in ("off","dfn3"):
            scores,predictions,units=bank.evaluate(model,full=True,denoiser=denoiser)
            clean=units[units.kind.eq("clean")].iloc[0]
            scores["clean_accuracy"]=float(clean.accuracy)
            predictions.to_csv(run/f"{selected}_{denoiser}_full_predictions.csv.gz",index=False)
            units.to_csv(run/f"{selected}_{denoiser}_full_units.csv",index=False)
            final_results[f"{selected}_{denoiser}"]=dict(epoch=epoch,**scores)
    summary=dict(condition=item["condition"],batch=item["batch"],encoder_lr=lr,speaker=item["speaker"],
        source_epoch=item["source_epoch"],best_epoch=best_epoch,final_epoch=MAX_EPOCH,best_monitor_f1=best_f1,
        final_results=final_results,checkpoint_sha256=engine.file_hash(run/"best.pt"),
        last_checkpoint_sha256=engine.file_hash(latest),extended=True,run_dir=str(run.relative_to(ROOT)),
        stop_reason="fixed_epoch100_without_early_stopping",test_evaluated=False)
    parent.write_json(run/"summary.json",summary);parent.write_json(summary_path,summary)
    progress(stage="branch_complete",condition=item["condition"],batch=item["batch"],speaker=item["speaker"],epoch=MAX_EPOCH)
    render()
    del model,optimizer,train,val,train_loader,val_loader
    gc.collect();torch.cuda.empty_cache()


def main():
    import fcntl
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only",action="store_true")
    args=parser.parse_args()
    STUDY.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    with (STUDY/"driver.lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        frozen=freeze_protocol();render()
        if args.prepare_only:return
        if (STUDY/"complete.json").exists():return
        parent.write_json(STUDY/"process.json",dict(pid=os.getpid(),started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())))
        assert torch.cuda.is_available()
        torch.set_num_threads(2);engine.seed_all(42)
        torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
        cfg=load_config(ROOT/"configs/task2_real.json");metadata=read_metadata(cfg)
        state,statistics=load_source(ROOT/"runs/conformer-research")
        recordings={}
        for source in frozen["original_training_sources"]+frozen["added_training_sources"]:
            path=ROOT/source["prepared_path"]
            assert engine.file_hash(path)==source["prepared_sha256"]
            recordings[source["noise_id"]]=sf.read(path,dtype="float32")[0]
        for item in frozen["plan"]:
            if not item["extend"]:run_branch(item,cfg,metadata,state,statistics,frozen,recordings,None)
        for speaker in SPEAKERS:
            pending=[i for i in frozen["plan"] if i["speaker"]==speaker and i["extend"]]
            if not pending:continue
            bank=parent.ValidationBank(speaker)
            for item in pending:run_branch(item,cfg,metadata,state,statistics,frozen,recordings,bank)
            del bank
        extended=sum(i["extend"] for i in frozen["plan"])
        parent.write_json(STUDY/"training_complete.json",dict(extended_branches=extended,
            reused_epoch100_branches=len(frozen["plan"])-extended,test_evaluated=False))
        progress(stage="auditing");render()
        from .check import check
        check()
        assert freeze_protocol()==frozen
        parent.write_json(STUDY/"complete.json",dict(training_and_independent_audit_complete=True,test_evaluated=False))
        progress(stage="complete");render()
        print("FIXED100_COMPLETE",flush=True)


if __name__=="__main__":main()
