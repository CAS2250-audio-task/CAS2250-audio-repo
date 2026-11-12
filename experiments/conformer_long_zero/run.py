"""Resumable batch/LR pilots and long continuation selected on fixed0dB dev audio."""
import argparse
from contextlib import contextmanager
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F

from src import engine
from src.config import load_config
from src.dataset import read_metadata
from experiments.conformer.model import ConformerClassifier, load_source
from experiments.conformer.run import Clips, loaders, atomic_save
from experiments.conformer_sanity import SPEAKERS, speaker_fold
from experiments.conformer_aug.model import AugmentedConformer
from experiments.conformer_noise_focus.augment import FocusTrainingClips
from .policy import (CONDITIONS, BATCHES, LEARNING_RATES, PILOT_EPOCHS,
                     MAX_EPOCH, select_learning_rate, overfit_status, update_count)

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "runs/conformer-long-zero-v1"
REPORT = ROOT / "reports/conformer-long-zero-v1"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+".partial")
    engine.write_json(tmp, value)
    tmp.replace(path)


def source_checkpoint(condition, speaker):
    if condition == "baseline":
        return ROOT/f"runs/conformer-aug-v1/folds/{speaker}/noise/last.pt"
    return ROOT/f"runs/conformer-noise-focus-v1/folds/{speaker}/speech_focus/last.pt"


def branch(condition, batch, lr, speaker):
    return STUDY/"branches"/condition/f"bs{batch}"/f"lr{lr:.0e}"/speaker


def branch_id(condition, batch, lr, speaker):
    return f"{condition}_bs{batch}_lr{lr:.0e}_{speaker}"


def freeze_protocol():
    parent = json.loads((ROOT/"reports/conformer-noise-focus-v1/protocol.json").read_text())
    pins = dict(parent["source_sha256"])
    for relative,sha in pins.items():assert engine.file_hash(ROOT/relative)==sha,relative
    for relative in ("src/engine.py","src/config.py","src/dataset.py"):
        pins[relative]=engine.file_hash(ROOT/relative)
    pins.update({str(p.relative_to(ROOT)): engine.file_hash(p)
                 for p in Path(__file__).parent.glob("*.py")})
    starts = {str(source_checkpoint(c,s).relative_to(ROOT)): engine.file_hash(source_checkpoint(c,s))
              for c in CONDITIONS for s in SPEAKERS}
    banks = {}
    for s in SPEAKERS:
        for relative in (f"runs/denoise-eval-v1/{s}_cases.csv.gz",
                         f"runs/denoise-eval-v1/{s}_inputs.npy",
                         f"runs/denoise-eval-v1/filtered/deepfilternet3/{s}.npy"):
            banks[relative] = engine.file_hash(ROOT/relative)
    result = dict(study="conformer-long-zero-v1", seed=42, folds=list(SPEAKERS),
        data_root=load_config(ROOT/"configs/task2_real.json").data_root,
        reference_config_sha256=engine.file_hash(ROOT/"configs/task2_real.json"),
        conditions=list(CONDITIONS), training_batches=list(BATCHES), encoder_lr_candidates=list(LEARNING_RATES),
        head_lr_ratio=50, pilot_additional_epochs=PILOT_EPOCHS, maximum_total_epoch=MAX_EPOCH,
        learning_rate="constant from epoch11; reset LR only, retain AdamW moments; do not resume old cosine schedule",
        lr_selection="per condition and training batch: largest mean0dB macroF1 across4folds and epochs13..15; ties lower loss then lower LR",
        start="each previous condition's epoch10 LAST model, optimizer, sampler and torch/CUDA RNG; not clean-best or fresh ASR",
        training_draws_per_epoch=609, previous_updates=200,
        updates_per_epoch={str(b):math.ceil(609/b) for b in BATCHES},
        budget_comparison="same sampled clips/epoch; smaller batch makes more optimizer updates; batch/LR separately tuned, not an isolated batch-only causal claim",
        normalization="unchanged ASR global statistics", weight_decay=1e-4,
        training_augmentation="exact old policies; baseline original50%+wide50%; speech_focus original50%+wide25%+near[-3,3]25%, speech pool on half noisy draws",
        monitor="original203 plus8 original noise recordings x fixed0dB draw0 x203 per heldout speaker; same inputs every epoch",
        checkpoint_selection="unfiltered fixed0dB mean of8 per-noise macroF1 scores, strict improvement, earliest tie; includes epoch10 starting state",
        final_evaluation="selected-best, final-last and source-epoch10: all3draw0dB, original, noise-only; same model with DFN3 off/on",
        classifier_evaluation_batch=32, classifier_batch_boundaries="reset at each203-clip/noise/draw unit;20 per noise-only unit",
        dfn3="reuse previously verified full-bankB16 filtered waves; no denoising during training and no DFN-driven epoch/LR selection",
        overfit_indicator=dict(minimum_epoch=25, averaging_window=3, consecutive_bad_windows=5,
            minimum_train_loss_fall=.02, minimum_validation_loss_rise=.03, minimum_f1_fall=.005,
            reference="lowest historical smoothed validation loss at least5epochs earlier; F1 compared to earlier smoothed maximum",
            additional_epochs_after_confirmation=5,
            plateau_is_not_overfitting=True, cap_without_indicator_is_not_overfitting=True),
        source_sha256=pins, initial_checkpoint_sha256=starts, validation_bank_sha256=banks,
        parent_protocol_sha256=engine.file_hash(ROOT/"reports/conformer-noise-focus-v1/protocol.json"),
        training_sources=parent["original_noise_sources"]+parent["added_speech_sources"],
        original_training_sources=parent["original_noise_sources"], added_training_sources=parent["added_speech_sources"],
        development_wav_sha256=parent["development_wav_sha256"],
        cuda_matmul_allow_tf32=True, cudnn_allow_tf32=True,
        test_audio_read=False, test_evaluated=False,
        limitations=["development speakers/noises now explicitly tune epochs and LR; scores are not independent test estimates",
            "draw0 monitoring overlaps all3draw final evaluation; extra draws measure stability, not a heldout test",
            "one training seed; four speakers; earlier source/session overlap limitations remain",
            "batch16 and32 have the same exposures but different update counts and dropout/normalization dynamics",
            "a finite run may not show sustained overfitting; do not force or claim an overfit pattern"])
    path = REPORT/"protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == result, "Frozen long-run protocol changed"
    else:
        write_json(path,result)
    return result


class ValidationBank:
    def __init__(self, speaker):
        self.cases = pd.read_csv(ROOT/f"runs/denoise-eval-v1/{speaker}_cases.csv.gz")
        assert len(self.cases)==5235 and self.cases.heldout_speaker.eq(speaker).all()
        assert not self.cases.path.str.contains("spk01_").any()
        self.raw = np.load(ROOT/f"runs/denoise-eval-v1/{speaker}_inputs.npy",mmap_mode="r")
        self.dfn = np.load(ROOT/f"runs/denoise-eval-v1/filtered/deepfilternet3/{speaker}.npy",mmap_mode="r")
        assert self.raw.shape == self.dfn.shape == (5235,24000)
        for row in self.cases.itertuples():
            assert hashlib.sha256(self.raw[row.case_index].tobytes()).hexdigest()==row.input_sha256

    @torch.inference_mode()
    def evaluate(self, model, full=False, denoiser="off"):
        model.eval()
        frame = self.cases if full else self.cases[
            self.cases.kind.eq("clean") | (self.cases.kind.eq("mixed") & self.cases.draw.eq(0))]
        waves = self.raw if denoiser=="off" else self.dfn
        pieces, units = [], []
        # Preserve canonical per203/20 unit batch32 boundaries from the preceding study.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            for keys, unit in frame.groupby(["kind","noise_id","snr_db","draw"],sort=False):
                preds, losses = [], []
                for offset in range(0,len(unit),32):
                    chunk=unit.iloc[offset:offset+32]
                    x=torch.from_numpy(np.array(waves[chunk.case_index.to_numpy()])).cuda()
                    y=torch.tensor(chunk.true_idx.to_numpy(),device="cuda")
                    logits=model(x)
                    assert torch.isfinite(logits).all()
                    preds.extend(logits.argmax(1).cpu().tolist())
                    losses.extend(F.cross_entropy(logits,y,reduction="none").cpu().tolist())
                result=unit.copy()
                result["pred_idx"],result["loss"],result["denoiser"]=preds,losses,denoiser
                pieces.append(result)
                units.append(dict(kind=keys[0],noise_id=keys[1],snr_db=int(keys[2]),draw=int(keys[3]),
                    **engine.metrics(result.true_idx,result.pred_idx,11),loss=float(np.mean(losses))))
        predictions=pd.concat(pieces,ignore_index=True)
        metrics=pd.DataFrame(units)
        zero=metrics[metrics.kind.eq("mixed")]
        original=metrics[metrics.kind.eq("clean")].iloc[0]
        summary=dict(zero_f1=float(zero.macro_f1.mean()),zero_loss=float(zero.loss.mean()),
            zero_accuracy=float(zero.accuracy.mean()),clean_f1=float(original.macro_f1),clean_loss=float(original.loss))
        only=predictions[predictions.kind.eq("noise_only")]
        if len(only):summary["noise_only_fpr"]=float(only.pred_idx.ne(10).mean())
        return summary,predictions,metrics


def progress(**values):
    values.update(updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
        pid=os.getpid(),pilot_completed=len(list((STUDY/"branches").glob("*/*/*/*/pilot_complete.json"))),
        long_completed=len(list((STUDY/"branches").glob("*/*/*/*/summary.json"))))
    write_json(STUDY/"progress.json",values)
    write_json(REPORT/"progress.json",values)
    print("PROGRESS",json.dumps(values),flush=True)


def save_history(run, history, condition,batch,lr,speaker):
    frame=pd.DataFrame(history)
    frame.to_csv(run/"history.csv",index=False)
    target=REPORT/"histories"/(branch_id(condition,batch,lr,speaker)+".csv")
    target.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(target,index=False)


def capture(model, optimizer, history, best_f1, best_epoch, train_loader, val_loader):
    return dict(model=model.state_dict(),optimizer=optimizer.state_dict(),history=history,
        epoch=history[-1]["epoch"],best_f1=best_f1,best_epoch=best_epoch,
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
        loader_rng=train_loader.sampler.generator.get_state(),val_loader_rng=val_loader.generator.get_state())


def run_branch(condition,batch,lr,speaker,target,cfg,metadata,state,statistics,frozen,recordings,bank):
    run=branch(condition,batch,lr,speaker)
    run.mkdir(parents=True,exist_ok=True)
    if target==15 and (run/"pilot_complete.json").exists():return
    if target>15 and (run/"summary.json").exists():return
    engine.seed_all(42)
    frame=speaker_fold(metadata,speaker,cfg)
    train_frame=frame[frame.split.eq("train")]
    val_frame=frame[frame.split.eq("val")]
    assert len(train_frame)==609 and len(val_frame)==203
    assert not train_frame.group.isin(("speaker:spk01","speaker:"+speaker)).any()
    train=FocusTrainingClips(train_frame,cfg,speaker,frozen["original_training_sources"],
        frozen["added_training_sources"],recordings,add_speech=condition=="speech_focus",focus_zero=condition=="speech_focus")
    val=Clips(val_frame,cfg)
    train_loader,val_loader,_=loaders(train,val,42,batch)
    model=AugmentedConformer(state,statistics,speaker,mask=False).cuda()
    optimizer=torch.optim.AdamW([
        dict(params=list(model.frontend.parameters())+list(model.transformer.parameters()),lr=lr),
        dict(params=model.fc.parameters(),lr=50*lr)],weight_decay=1e-4)
    latest=run/"last.pt"
    fresh=not latest.exists()
    source=source_checkpoint(condition,speaker) if fresh else latest
    cp=torch.load(source,map_location="cpu",weights_only=True)
    model.load_state_dict(cp["model"],strict=True)
    optimizer.load_state_dict(cp["optimizer"])
    # Old cosine reached zero; deliberately choose a new constant continuation LR.
    for group,value in zip(optimizer.param_groups,(lr,50*lr)):
        group["lr"]=value
        group["initial_lr"]=value
    if fresh:
        assert cp["epoch"]==10
        scores,predictions,units=bank.evaluate(model)
        history=[dict(epoch=10,train_loss=cp["history"][-1]["train_loss"],encoder_lr=lr,
            head_lr=50*lr,updates=200,stage="source_epoch10",**scores)]
        predictions.to_csv(run/"source_epoch10_predictions.csv.gz",index=False)
        best_f1,best_epoch=scores["zero_f1"],10
        atomic_save(dict(model=model.state_dict(),epoch=10,monitor=scores),run/"best.pt")
        write_json(run/"initialization.json",dict(condition=condition,training_batch=batch,encoder_lr=lr,
            source_checkpoint=str(source.relative_to(ROOT)),source_sha256=engine.file_hash(source),
            optimizer_moments_restored=True,old_cosine_not_resumed=True))
    else:
        history,best_f1,best_epoch=cp["history"],cp["best_f1"],cp["best_epoch"]
    torch.set_rng_state(cp["torch_rng"]);torch.cuda.set_rng_state_all(cp["cuda_rng"])
    train_loader.sampler.generator.set_state(cp["loader_rng"])
    val_loader.generator.set_state(cp["val_loader_rng"])
    del cp
    # A crash during final evaluation must not train an already stopped branch again.
    end_target=history[-1]["epoch"] if target>15 and overfit_status(history)["stop"] else target
    for epoch in range(history[-1]["epoch"]+1,end_target+1):
        train.set_epoch(epoch);train.trace=[]
        model.set_epoch(epoch);model.train()
        total=0.;started=time.monotonic();steps=0
        for x,y,_ in train_loader:
            x,y=x.cuda(),y.cuda();optimizer.zero_grad(set_to_none=True)
            loss=F.cross_entropy(model(x),y)
            assert torch.isfinite(loss)
            loss.backward();optimizer.step()
            total+=float(loss.detach())*len(y);steps+=1
        assert len(train.trace)==609 and steps==math.ceil(609/batch)
        scores,predictions,units=bank.evaluate(model)
        history.append(dict(epoch=epoch,train_loss=total/609,encoder_lr=lr,head_lr=50*lr,
            updates=update_count(epoch,batch),stage="pilot" if epoch<=15 else "long",**scores))
        epochs=run/"epochs";epochs.mkdir(exist_ok=True)
        predictions.to_csv(epochs/f"epoch_{epoch:03d}.csv.gz",index=False)
        units.to_csv(epochs/f"units_{epoch:03d}.csv",index=False)
        pd.DataFrame(train.trace).to_csv(epochs/f"augmentation_{epoch:03d}.csv.gz",index=False)
        if scores["zero_f1"]>best_f1:
            best_f1,best_epoch=scores["zero_f1"],epoch
            atomic_save(dict(model=model.state_dict(),epoch=epoch,monitor=scores),run/"best.pt")
        atomic_save(capture(model,optimizer,history,best_f1,best_epoch,train_loader,val_loader),latest)
        save_history(run,history,condition,batch,lr,speaker)
        stop=overfit_status(history)
        progress(stage="pilot" if target==15 else "long",condition=condition,batch=batch,encoder_lr=lr,
            speaker=speaker,epoch=epoch,updates=update_count(epoch,batch),best_epoch=best_epoch,
            zero_f1=scores["zero_f1"],zero_loss=scores["zero_loss"],train_loss=total/609,
            seconds=time.monotonic()-started,overfit_confirmation_epoch=stop["confirmation_epoch"])
        if epoch==15 and not (run/"pilot_complete.json").exists():
            pd.DataFrame(history).to_csv(run/"pilot_history.csv",index=False)
            write_json(run/"pilot_complete.json",dict(epoch=15,
                history_sha256=engine.file_hash(run/"pilot_history.csv"),test_evaluated=False))
        if target>15 and stop["stop"]:break
    if target==15 and not (run/"pilot_complete.json").exists():
        assert history[-1]["epoch"]==15
        pd.DataFrame(history).to_csv(run/"pilot_history.csv",index=False)
        write_json(run/"pilot_complete.json",dict(epoch=15,
            history_sha256=engine.file_hash(run/"pilot_history.csv"),test_evaluated=False))
    assert torch.equal(model.feature_mean.cpu(),statistics["glob_mean"])
    assert torch.equal(model.feature_std.cpu(),statistics["glob_std"])
    if target>15:
        final_results={}
        for selected in ("best","last"):
            saved=torch.load(run/f"{selected}.pt",map_location="cpu",weights_only=True)
            model.load_state_dict(saved["model"],strict=True)
            epoch=saved["epoch"];del saved
            for denoiser in ("off","dfn3"):
                scores,predictions,units=bank.evaluate(model,full=True,denoiser=denoiser)
                predictions.to_csv(run/f"{selected}_{denoiser}_full_predictions.csv.gz",index=False)
                units.to_csv(run/f"{selected}_{denoiser}_full_units.csv",index=False)
                final_results[f"{selected}_{denoiser}"]=dict(epoch=epoch,**scores)
        summary=dict(condition=condition,batch=batch,encoder_lr=lr,speaker=speaker,best_epoch=best_epoch,
            final_epoch=history[-1]["epoch"],best_monitor_f1=best_f1,stop=overfit_status(history),
            final_results=final_results,checkpoint_sha256=engine.file_hash(run/"best.pt"),
            last_checkpoint_sha256=engine.file_hash(latest),test_evaluated=False)
        write_json(run/"summary.json",summary)
        write_json(REPORT/"runs"/(branch_id(condition,batch,lr,speaker)+".json"),summary)
        print("LONG_BRANCH_COMPLETE",json.dumps(summary),flush=True)
    del model,optimizer,train,val,train_loader,val_loader
    gc.collect();torch.cuda.empty_cache()


def preflight(state,statistics):
    """Replay canonical historical predictions, and exercise batch16 and batch1."""
    historical=pd.read_csv(ROOT/"reports/conformer-noise-focus-v1/predictions.csv.gz")
    bank=ValidationBank("spk02")
    model=ConformerClassifier(state,statistics,"finetune").cuda()
    count=0
    for condition in CONDITIONS:
        path=source_checkpoint(condition,"spk02").parent/"best.pt"
        cp=torch.load(path,map_location="cpu",weights_only=True)
        model.load_state_dict(cp["model"],strict=True);del cp
        scores,predictions,_=bank.evaluate(model)
        expected=historical[(historical.training_condition==condition)&historical.heldout_speaker.eq("spk02")
            &historical.denoiser.eq("off") & (historical.kind.eq("clean")|
            (historical.kind.eq("mixed")&historical.snr_db.eq(0)&historical.draw.eq(0)))]
        joined=predictions.merge(expected,on=["kind","noise_id","snr_db","draw","path"],suffixes=("_new","_old"),validate="one_to_one")
        assert len(joined)==1827 and joined.pred_idx_new.eq(joined.pred_idx_old).all()
        count+=len(joined)
    model.train()
    for batch in (16,1):
        model.zero_grad(set_to_none=True)
        x=torch.from_numpy(np.array(bank.raw[:batch])).cuda()
        y=torch.tensor(bank.cases.true_idx.iloc[:batch].to_numpy(),device="cuda")
        loss=F.cross_entropy(model(x),y);loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(model.fc.weight.grad).all()
    write_json(REPORT/"preflight.json",dict(canonical_predictions_replayed=count,batch16_and_last_batch1_backward_finite=True,
        validation_samples_per_epoch=1827,zero_mixed_monitor_samples=1624,test_evaluated=False))
    del model,bank,historical
    gc.collect();torch.cuda.empty_cache()


def evaluate_source_epoch10(condition,speaker,state,statistics,bank):
    target=REPORT/"source_epoch10"/f"{condition}_{speaker}.json"
    if target.exists():return
    model=ConformerClassifier(state,statistics,"finetune").cuda()
    cp=torch.load(source_checkpoint(condition,speaker),map_location="cpu",weights_only=True)
    model.load_state_dict(cp["model"],strict=True);del cp
    results={}
    folder=STUDY/"source_epoch10";folder.mkdir(exist_ok=True)
    for denoiser in ("off","dfn3"):
        scores,predictions,units=bank.evaluate(model,full=True,denoiser=denoiser)
        predictions.to_csv(folder/f"{condition}_{speaker}_{denoiser}.csv.gz",index=False)
        units.to_csv(folder/f"{condition}_{speaker}_{denoiser}_units.csv",index=False)
        results[denoiser]=scores
    write_json(target,dict(condition=condition,speaker=speaker,epoch=10,results=results,test_evaluated=False))
    del model;torch.cuda.empty_cache()


@contextmanager
def process_lock():
    with (STUDY/"driver.lock").open("w") as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        handle.write(str(os.getpid()));handle.flush()
        yield


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only",action="store_true")
    args=parser.parse_args()
    STUDY.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);engine.seed_all(42)
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    with process_lock():
        frozen=freeze_protocol()
        cfg=load_config(ROOT/"configs/task2_real.json");metadata=read_metadata(cfg)
        state,statistics=load_source(ROOT/"runs/conformer-research")
        if not (REPORT/"preflight.json").exists():preflight(state,statistics)
        if args.preflight_only:return
        recordings={}
        for source in frozen["training_sources"]:
            path=ROOT/source["prepared_path"]
            assert engine.file_hash(path)==source["prepared_sha256"]
            recordings[source["noise_id"]]=sf.read(path,dtype="float32")[0]
        # Cache only one held-out bank object at a time. Pilot choices are made on ALL4folds.
        for speaker in SPEAKERS:
            bank=ValidationBank(speaker)
            for condition in CONDITIONS:
                for batch in BATCHES:
                    for lr in LEARNING_RATES:
                        run_branch(condition,batch,lr,speaker,15,cfg,metadata,state,statistics,frozen,recordings,bank)
            del bank
        all_rows=[]
        selections=[]
        for condition in CONDITIONS:
            for batch in BATCHES:
                rows=[]
                for lr in LEARNING_RATES:
                    for speaker in SPEAKERS:
                        p=branch(condition,batch,lr,speaker)/"pilot_history.csv"
                        for r in pd.read_csv(p).to_dict("records"):
                            r.update(condition=condition,batch=batch,speaker=speaker)
                            rows.append(r)
                all_rows.extend(rows)
                choice=select_learning_rate(rows)
                selections.append(dict(condition=condition,batch=batch,**choice))
        pd.DataFrame(all_rows).to_csv(REPORT/"pilot_history.csv",index=False)
        selection_path=REPORT/"lr_selection.json"
        if selection_path.exists():assert json.loads(selection_path.read_text())==selections
        else:write_json(selection_path,selections)
        progress(stage="pilot_complete",selected_learning_rates=selections)
        from experiments.report_conformer_long_zero import render
        render()
        for speaker in SPEAKERS:
            bank=ValidationBank(speaker)
            for condition in CONDITIONS:evaluate_source_epoch10(condition,speaker,state,statistics,bank)
            for selection in selections:
                run_branch(selection["condition"],selection["batch"],selection["encoder_lr"],speaker,MAX_EPOCH,
                    cfg,metadata,state,statistics,frozen,recordings,bank)
                render()
            del bank
        assert freeze_protocol()==frozen
        write_json(STUDY/"training_complete.json",dict(pilot_branches=48,long_branches=16,test_evaluated=False))
        progress(stage="training_complete")
        render()
        from experiments.check_conformer_long_zero import check
        check()
        write_json(STUDY/"complete.json",dict(training_and_independent_audit_complete=True,test_evaluated=False))
        render()
        print("LONG_ZERO_TRAINING_COMPLETE",flush=True)


if __name__=="__main__":main()
