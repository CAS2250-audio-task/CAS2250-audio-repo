"""Detached seven-hour study; frozen training protocol and automatic stage selection."""
import argparse
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F

from src import engine
from src.config import load_config
from src.dataset import read_metadata
from experiments.conformer.model import load_source
from experiments.conformer.run import Clips,loaders,atomic_save
from experiments.conformer_sanity import SPEAKERS,speaker_fold
from experiments.conformer_long_zero.run import ValidationBank,capture,write_json
from experiments.conformer_long_zero.policy import update_count
from experiments.conformer_aug.augment import LIBDIR,LIBFILES
from experiments.check_conformer_long_zero import recompute_predictions
from .data import Training,FeatureCache,formant_variant
from .features import fit_scaler,SETS
from .model import AcousticConformer
from .report import render

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/"runs/overnight-v1"
REPORT=ROOT/"reports/overnight-v1"
LR=1e-5


def candidate(name,wd=1e-4,dropout=.1,formant=None,set_id="0",augmented=False):
    return dict(name=name,wd=wd,dropout=dropout,formant=formant,set_id=str(set_id),augmented=augmented)


def progress(**kw):
    kw.update(pid=os.getpid(),updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()))
    write_json(STUDY/"progress.json",kw);write_json(REPORT/"progress.json",kw)
    print("PROGRESS",json.dumps(kw),flush=True)


def freeze():
    old=json.loads((ROOT/"reports/conformer-long-zero-v1/protocol.json").read_text())
    pins=dict(old["source_sha256"])
    pins.update({str(p.relative_to(ROOT)):engine.file_hash(p) for p in Path(__file__).parent.glob("*.py")})
    pins.update({str(LIBDIR/name):engine.file_hash(LIBDIR/name) for name in LIBFILES})
    for p,sha in pins.items():assert engine.file_hash(ROOT/p)==sha,p
    starts={f"runs/conformer-aug-v1/folds/{s}/noise/last.pt":engine.file_hash(ROOT/f"runs/conformer-aug-v1/folds/{s}/noise/last.pt") for s in SPEAKERS}
    for mapping in (old["validation_bank_sha256"],):
        for p,sha in mapping.items():assert engine.file_hash(ROOT/p)==sha,p
    for p,sha in old["development_wav_sha256"].items():
        assert "spk01_" not in p
        assert engine.file_hash(Path(old["data_root"])/p)==sha
    value=dict(study="overnight-v1",seed=42,batch=16,epochs=100,encoder_lr=LR,head_lr=50*LR,
        init="Same condition-specific baseline-noise epoch10 LAST; restore model/AdamW/RNG; change candidate settings from epoch11 only",
        no_freezing=True,no_lr_search=True,early_stopping=False,draws_per_epoch=609,
        baseline="Reuse conformer-fixed100-v1 baseline bs16 best, same epoch10 initialization/continuation schedule",
        regularization=[candidate("wd_1e-3",wd=1e-3),candidate("dropout_0p2",dropout=.2)],
        combine="Only if both single changes pass the same selection gate; compare combination with best single",
        augmentation="Common noise p0.5 U[-5,20]; A=SpecAugment+speed+within-file pitch, B=formants, A+B; each component independently p0.5;2fixed variants for each active waveform combination",
        augmentation_candidates=[dict(name="augment_A",augmented=True,formant=None),
            dict(name="formant_B",augmented=False,formant="weak"),
            dict(name="augment_AB",augmented=True,formant="weak")],
        waveform_order="within-file pitch -> protected speed -> formant -> online noise -> native Fbank masks",
        pitch="[0,U[-1,1],U[-1,1],0] semitone anchors, internal separation>=.35, formant-preserved Rubber Band2 R2; latency compensated",
        speed="pitch-preserved0.9..1.1; preserve entire reviewed nonpadding interval; skip overflowing slowdown; centre-zero-pad acceleration",
        specaugment="time20..50ms and3..5/80mel bins, native silence floor; time mask also zeros auxiliary values+availability; train only",
        formants={"weak":[1/1.1,1.1]},
        formant_method="Praat Change gender, floor65 ceiling600, new pitch median0 (preserve), range1 duration1; symmetric log-uniform formant ratio; RMS match then peak<=.99",
        formant_reference="https://fon.hum.uva.nl/praat/manual/Sound__Change_gender___.html",
        auxiliary_required=["3","7"],auxiliary_optional=[],
        auxiliary_sets=SETS,auxiliary_fusion="151-frame descriptors+availability, Conv1d151->76->38, zero-initialized projection to256 residual on encoder output; existing256->11head retained",
        scaling="Fit only original train valid measurements per fold; extract all descriptors from actual augmented/filtered waveform; no clean-side auxiliary input",
        acoustic_ceiling="Fixed5500Hz for all speakers, unchanged legacy extractor; no sex/speaker-conditioned ceiling",
        checkpoint_selection=old["checkpoint_selection"],
        condition_selection="all3draw0dB 4fold meanF1 gain>=0.005, minimum speaker F1 decrease<=0.01; earliest tie; same gate each stage",
        final_filter="DFN3 off/on on the same selected checkpoints; hash-pinned historical B16 waveform cache; recompute auxiliary from filtered waveform",
        retention="Keep best models and all epoch prediction/trace files; remove only newly generated completed LAST optimizer state after audit; checkpoint every epoch for incomplete folds",
        budget="Aim7hours; only optional auxiliary candidates gated by elapsed time, reserve45minutes for final evaluation/audit; never truncate a fold",
        source_sha256=pins,initial_checkpoint_sha256=starts,validation_bank_sha256=old["validation_bank_sha256"],
        development_wav_sha256=old["development_wav_sha256"],data_root=str(ROOT/"datasets/task2/fixed-1p5-v1"),
        original_training_sources=old["original_training_sources"],added_training_sources=old["added_training_sources"],
        parent_protocol_sha256=engine.file_hash(ROOT/"reports/conformer-fixed100-v1/protocol.json"),
        test_audio_read=False,test_evaluated=False)
    p=REPORT/"protocol.json"
    if p.exists():assert json.loads(p.read_text())==value,"Frozen overnight protocol changed"
    else:write_json(p,value)
    return value


def prepare_waveforms(metadata,cfg):
    from .waveforms import variants
    folder=STUDY/"waveforms";folder.mkdir(parents=True,exist_ok=True)
    audio=folder/"waves.npy";marker=folder/"complete.json"
    frame=metadata[metadata.group.ne("speaker:spk01")&metadata.label.ne("silence")].reset_index(drop=True)
    assert len(frame)==800
    if not marker.exists():
        progress(stage="preparing_waveforms",candidate="A_B_AB")
        recipe=pd.read_csv(Path(cfg.data_root)/"preprocessing_recipe.csv").set_index("path")
        output=np.lib.format.open_memmap(folder/"waves.partial.npy",mode="w+",dtype="float32",shape=(800,7,2,24000))
        records=[]
        for n,row in enumerate(frame.itertuples()):
            x=sf.read(Path(cfg.data_root)/row.path,dtype="float32")[0]
            rec=recipe.loc[row.path]
            protected=(int(rec.pad_left_samples),24000-int(rec.pad_right_samples))
            for variant in (0,1):
                for flags,(y,details) in variants(x,row.path,variant,protected).items():
                    output[n,flags-1,variant]=y
                    records.append(dict(path=row.path,variant=variant,flags=flags,details=details,
                        output_sha256=hashlib.sha256(y.tobytes()).hexdigest()))
            if n%100==0:print("WAVEFORM_BANK",n,800,flush=True)
        output.flush();del output
        (folder/"waves.partial.npy").replace(audio)
        write_json(folder/"rows.json",records)
        write_json(marker,dict(count=11200,sha256=engine.file_hash(audio),test_included=False))
    assert engine.file_hash(audio)==json.loads(marker.read_text())["sha256"]
    waves=np.load(audio,mmap_mode="r")
    records=json.loads((folder/"rows.json").read_text())
    lookup={};indices={p:i for i,p in enumerate(frame.path)}
    for record in records:
        n=indices[record["path"]]
        y=waves[n,record["flags"]-1,record["variant"]]
        assert hashlib.sha256(y.tobytes()).hexdigest()==record["output_sha256"]
        lookup[(record["path"],record["flags"],record["variant"])]=(y,record["details"])
    return lookup


def make_model(spec,speaker,state,statistics,cache,scaler):
    return AcousticConformer(state,statistics,speaker,dropout=spec["dropout"],set_id=spec["set_id"],cache=cache,scaler=scaler,augmented=spec["augmented"]).cuda()


def audit_fold(run,spec,speaker,history,model,bank,optimizer):
    assert history[-1]["epoch"]==100
    for group in optimizer.param_groups[:2]:
        for parameter in group["params"]:
            assert int(optimizer.state[parameter]["step"])==update_count(100,16)
    # Auxiliary parameters begin at epoch11 rather than the source's first10 epochs.
    if model.auxiliary is not None:
        for parameter in model.auxiliary.parameters():
            assert int(optimizer.state[parameter]["step"])==90*39
    epochs=0;draws=0
    for row in history:
        e=row["epoch"]
        if e<=10:continue
        pred=pd.read_csv(run/"epochs"/f"epoch_{e:03d}.csv.gz")
        assert len(pred)==1827 and not pred.path.str.contains("spk01_").any()
        for k,v in recompute_predictions(pred).items():assert abs(v-row[k])<1e-10,(spec["name"],speaker,e,k)
        trace=pd.read_csv(run/"epochs"/f"augmentation_{e:03d}.csv.gz")
        assert len(trace)==609 and trace.draw.tolist()==list(range(609))
        assert not trace.path.str.contains("spk01_|"+speaker+"_").any()
        assert not trace.added_speech_selected.any() and not trace.near_zero_selected.any()
        if not spec["augmented"]:assert not trace.pitch_selected.any() and not trace.speed_selected.any()
        if not spec["formant"]:assert not trace.formant_selected.any()
        if spec["augmented"]:
            masks=pd.read_csv(run/"epochs"/f"masks_{e:03d}.csv.gz")
            assert len(masks)>0 and masks.time_width.between(2,5).all() and masks.freq_width.between(3,5).all()
        epochs+=len(pred);draws+=len(trace)
    return dict(epoch_predictions_recomputed=epochs,training_draws_checked=draws,all_epochs_to100=True)


def run_fold(spec,speaker,cfg,metadata,state,statistics,frozen,recordings,formants,cache):
    name=spec["name"];run=STUDY/"candidates"/name/speaker;run.mkdir(parents=True,exist_ok=True)
    out=REPORT/"folds"/name/(speaker+".json")
    if out.exists():return json.loads(out.read_text())
    assert shutil.disk_usage(STUDY).free>1_500_000_000,"Less than1.5GB free; stop safely with checkpoints intact"
    engine.seed_all(42)
    frame=speaker_fold(metadata,speaker,cfg)
    train_frame=frame[frame.split.eq("train")];val_frame=frame[frame.split.eq("val")]
    train=Training(train_frame,cfg,speaker,frozen["original_training_sources"],frozen["added_training_sources"],
        recordings,add_speech=False,focus_zero=False,formants=formants if spec["formant"] else None,
        augmented=spec["augmented"])
    train.variants=formants
    val=Clips(val_frame,cfg)
    train_loader,val_loader,_=loaders(train,val,42,16)
    scaler=None
    if spec["set_id"]!="0":
        progress(stage="fitting_train_scaler",candidate=name,speaker=speaker)
        raw,valid=cache.raw_many(train.wave)
        scaler=fit_scaler(raw,valid);write_json(run/"scaler.json",scaler)
        del raw,valid
    model=make_model(spec,speaker,state,statistics,cache,scaler)
    opt=torch.optim.AdamW([
        dict(params=list(model.frontend.parameters())+list(model.transformer.parameters()),lr=LR),
        dict(params=model.fc.parameters(),lr=50*LR)],weight_decay=spec["wd"])
    latest=run/"last.pt";fresh=not latest.exists()
    source=ROOT/f"runs/conformer-aug-v1/folds/{speaker}/noise/last.pt"
    cp=torch.load(source if fresh else latest,map_location="cpu",weights_only=True)
    missing,unexpected=model.load_state_dict(cp["model"],strict=False)
    assert not unexpected and all(k.startswith("auxiliary.") for k in missing)
    if not fresh and model.auxiliary is not None:opt.add_param_group(dict(params=model.auxiliary.parameters(),lr=50*LR,weight_decay=spec["wd"]))
    opt.load_state_dict(cp["optimizer"])
    if fresh and model.auxiliary is not None:opt.add_param_group(dict(params=model.auxiliary.parameters(),lr=50*LR,weight_decay=spec["wd"]))
    for j,group in enumerate(opt.param_groups):
        group["lr"]=LR if j==0 else 50*LR;group["initial_lr"]=group["lr"];group["weight_decay"]=spec["wd"]
    bank=ValidationBank(speaker)
    if fresh:
        assert cp["epoch"]==10
        metrics,_,_=bank.evaluate(model)
        history=[dict(epoch=10,train_loss=cp["history"][-1]["train_loss"],updates=200,**metrics)]
        best_f1,best_epoch=metrics["zero_f1"],10
        atomic_save(dict(model=model.state_dict(),epoch=10,monitor=metrics),run/"best.pt")
        write_json(run/"initialization.json",dict(source_sha256=engine.file_hash(source),source_epoch=10,
            spec=spec,optimizer_moments_restored=True,dropout_modules=len(model.dropout_changes),test_included=False))
    else:history,best_f1,best_epoch=cp["history"],cp["best_f1"],cp["best_epoch"]
    torch.set_rng_state(cp["torch_rng"]);torch.cuda.set_rng_state_all(cp["cuda_rng"])
    train_loader.sampler.generator.set_state(cp["loader_rng"]);val_loader.generator.set_state(cp["val_loader_rng"])
    del cp
    for epoch in range(history[-1]["epoch"]+1,101):
        train.set_epoch(epoch);train.trace=[];model.set_epoch(epoch);model.mask_trace=[];model.train()
        total=0.;started=time.monotonic();steps=0
        for x,y,_ in train_loader:
            x,y=x.cuda(),y.cuda();opt.zero_grad(set_to_none=True)
            loss=F.cross_entropy(model(x),y);assert torch.isfinite(loss)
            loss.backward();opt.step();total+=float(loss.detach())*len(y);steps+=1
        assert steps==39 and len(train.trace)==609
        scores,pred,units=bank.evaluate(model)
        history.append(dict(epoch=epoch,train_loss=total/609,updates=update_count(epoch,16),**scores))
        folder=run/"epochs";folder.mkdir(exist_ok=True)
        pred.to_csv(folder/f"epoch_{epoch:03d}.csv.gz",index=False)
        pd.DataFrame(train.trace).to_csv(folder/f"augmentation_{epoch:03d}.csv.gz",index=False)
        if spec["augmented"]:pd.DataFrame(model.mask_trace).to_csv(folder/f"masks_{epoch:03d}.csv.gz",index=False)
        if scores["zero_f1"]>best_f1:
            best_f1,best_epoch=scores["zero_f1"],epoch
            atomic_save(dict(model=model.state_dict(),epoch=epoch,monitor=scores),run/"best.pt")
        atomic_save(capture(model,opt,history,best_f1,best_epoch,train_loader,val_loader),latest)
        pd.DataFrame(history).to_csv(run/"history.csv",index=False)
        progress(stage="training",candidate=name,speaker=speaker,epoch=epoch,best_epoch=best_epoch,
            zero_f1=scores["zero_f1"],zero_accuracy=scores["zero_accuracy"],seconds=time.monotonic()-started)
        if epoch%20==0:render()
    audit=audit_fold(run,spec,speaker,history,model,bank,opt)
    saved=torch.load(run/"best.pt",map_location="cpu",weights_only=True)
    assert saved["epoch"]==best_epoch==max(history,key=lambda h:h["zero_f1"])["epoch"]
    model.load_state_dict(saved["model"],strict=True);del saved
    scores,pred,units=bank.evaluate(model,full=True)
    scores["clean_accuracy"]=float(units[units.kind.eq("clean")].accuracy.iloc[0])
    for k,v in recompute_predictions(pred).items():assert abs(v-scores[k])<1e-10
    pred.to_csv(run/"best_off_predictions.csv.gz",index=False);units.to_csv(run/"best_off_units.csv",index=False)
    _,replay,_=bank.evaluate(model)
    joined=replay.merge(pred,on="case_index",suffixes=("_new","_saved"),validate="one_to_one")
    assert len(joined)==1827 and joined.pred_idx_new.eq(joined.pred_idx_saved).all()
    assert np.max(np.abs(joined.loss_new-joined.loss_saved))<1e-6
    audit.update(best_predictions_replayed=1827,final_predictions_recomputed=len(pred))
    result=dict(candidate=name,spec=spec,speaker=speaker,best_epoch=best_epoch,final_epoch=100,
        metrics=scores,audit=audit,run_dir=str(run.relative_to(ROOT)),best_sha256=engine.file_hash(run/"best.pt"),test_evaluated=False)
    write_json(out,result);write_json(run/"summary.json",result)
    # This is exclusively a newly generated optimizer checkpoint, after a completed fold audit.
    latest.unlink()
    progress(stage="fold_complete",candidate=name,speaker=speaker,epoch=100);render()
    del model,opt,train,val,train_loader,val_loader,bank
    gc.collect();torch.cuda.empty_cache()
    return result


def baseline():
    for speaker in SPEAKERS:
        path=REPORT/"folds/baseline"/(speaker+".json")
        if path.exists():continue
        parent=json.loads((ROOT/f"reports/conformer-fixed100-v1/runs/baseline_bs16_lr1e-05_{speaker}.json").read_text())
        run=ROOT/parent["run_dir"]
        units=pd.read_csv(run/"best_off_full_units.csv")
        scores=dict(parent["final_results"]["best_off"]);scores.pop("epoch",None)
        scores["clean_accuracy"]=float(units[units.kind.eq("clean")].accuracy.iloc[0])
        assert engine.file_hash(run/"best.pt")==parent["checkpoint_sha256"]
        write_json(path,dict(candidate="baseline",spec=candidate("baseline"),speaker=speaker,
            best_epoch=parent["best_epoch"],final_epoch=100,metrics=scores,reused=True,
            run_dir=str(run.relative_to(ROOT)),best_sha256=parent["checkpoint_sha256"],test_evaluated=False))
    render()


def evaluate_candidate(spec,context):
    started=time.monotonic()
    formants=prepare_waveforms(context["metadata"],context["cfg"]) if spec["formant"] or spec["augmented"] else None
    results=[run_fold(spec,s,formants=formants,**context) for s in SPEAKERS]
    write_json(REPORT/"candidates"/(spec["name"]+".json"),dict(spec=spec,seconds=time.monotonic()-started,
        f1=float(np.mean([r["metrics"]["zero_f1"] for r in results]))))
    return spec


def scores(spec):
    rows=[json.loads((REPORT/"folds"/spec["name"]/(s+".json")).read_text()) for s in SPEAKERS]
    f1=[r["metrics"]["zero_f1"] for r in rows]
    return float(np.mean(f1)),min(f1)


def better(spec,reference):
    a,w=scores(spec);b,z=scores(reference)
    return a>=b+.005 and w>=z-.01


def select(stage,reference,options):
    path=REPORT/f"selection_{stage}.json"
    choices=[reference]+[s for s in options if better(s,reference)]
    chosen=max(choices,key=lambda s:scores(s)[0])
    value=dict(selected=chosen["name"],spec=chosen,reference=reference["name"],
        results=[dict(name=s["name"],f1=scores(s)[0],worst_f1=scores(s)[1],passes_gate=s==reference or better(s,reference)) for s in [reference]+options])
    if path.exists():assert json.loads(path.read_text())==value
    else:write_json(path,value)
    render();return chosen


def final_filter(spec,context):
    output=[]
    for speaker in SPEAKERS:
        fold=json.loads((REPORT/"folds"/spec["name"]/(speaker+".json")).read_text())
        run=ROOT/fold["run_dir"]
        scaler=json.loads((run/"scaler.json").read_text()) if spec["set_id"]!="0" else None
        model=make_model(spec,speaker,context["state"],context["statistics"],context["cache"],scaler)
        saved=torch.load(run/"best.pt",map_location="cpu",weights_only=True);model.load_state_dict(saved["model"],strict=True);del saved
        bank=ValidationBank(speaker)
        for denoiser in ("off","dfn3"):
            p=REPORT/"final"/f"{speaker}_{denoiser}.json"
            if p.exists():output.append(json.loads(p.read_text()));continue
            progress(stage="final_filter",candidate=spec["name"],speaker=speaker,denoiser=denoiser)
            m,pred,units=bank.evaluate(model,full=True,denoiser=denoiser)
            m["clean_accuracy"]=float(units[units.kind.eq("clean")].accuracy.iloc[0])
            for k,v in recompute_predictions(pred).items():assert abs(v-m[k])<1e-10
            row=dict(candidate=spec["name"],speaker=speaker,denoiser=denoiser,**m)
            write_json(p,row);pred.to_csv(p.with_suffix(".csv.gz"),index=False);output.append(row)
        del model,bank;torch.cuda.empty_cache()
    d=pd.DataFrame(output);d.to_csv(REPORT/"final_filter_speakers.csv",index=False)
    d.groupby("denoiser",as_index=False).mean(numeric_only=True).to_csv(REPORT/"final_filter_summary.csv",index=False)


def smoke(context):
    from .data import shift_formants
    from .features import extract
    from experiments.conformer.model import ConformerClassifier
    speaker="spk02";bank=ValidationBank(speaker)
    source=ROOT/f"runs/conformer-aug-v1/folds/{speaker}/noise/last.pt"
    cp=torch.load(source,map_location="cpu",weights_only=True)
    original=ConformerClassifier(context["state"],context["statistics"],"finetune").cuda()
    original.load_state_dict(cp["model"],strict=True);original.eval()
    wave=torch.from_numpy(np.array(bank.raw[:2])).cuda()
    with torch.inference_mode():expected=original(wave)
    frame=speaker_fold(context["metadata"],speaker,context["cfg"])
    assert len(frame[frame.split.eq("train")])==609 and not frame[frame.split.eq("train")].path.str.contains("spk01_|spk02_").any()
    words=frame[frame.split.eq("train")&frame.label.ne("silence")].iloc[:8]
    originals=np.stack([sf.read(Path(context["cfg"].data_root)/p,dtype="float32")[0] for p in words.path])
    raw,valid=context["cache"].raw_many(originals);scaler=fit_scaler(raw,valid)
    parameter_counts={}
    for set_id in ("0","1","2","3","7"):
        model=make_model(candidate("smoke",set_id=set_id),speaker,context["state"],context["statistics"],context["cache"],scaler)
        missing,unexpected=model.load_state_dict(cp["model"],strict=False)
        assert not unexpected and all(k.startswith("auxiliary.") for k in missing)
        model.eval()
        with torch.inference_mode():actual=model(wave)
        assert torch.equal(actual,expected),(set_id,torch.max(torch.abs(actual-expected)))
        model.train();model.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model(wave),torch.tensor(bank.cases.true_idx.iloc[:2].to_numpy(),device="cuda"));loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(model.fc.weight.grad).all()
        if model.auxiliary is not None:
            grad=model.auxiliary[-1].weight.grad
            assert torch.isfinite(grad).all() and grad.abs().sum()>0
        parameter_counts[set_id]=sum(p.numel() for p in model.parameters())
        del model;torch.cuda.empty_cache()
    # Synthetic vowel: prescribed F0 and three formants, independent of speaker labels.
    from scipy.signal import lfilter
    t=np.arange(24000)/16000
    x=sum(np.sin(2*np.pi*120*k*t)/k for k in range(1,50))
    for freq,bw in ((600,80),(1300,100),(2500,120)):
        radius=np.exp(-np.pi*bw/16000);x=lfilter([1-radius],[1,-2*radius*np.cos(2*np.pi*freq/16000),radius**2],x)
    x=(x/np.max(np.abs(x))*.5).astype(np.float32)
    y=shift_formants(x,1.1)
    import parselmouth
    a=parselmouth.Sound(x,16000).to_pitch_ac(pitch_floor=65,pitch_ceiling=600).selected_array["frequency"]
    b=parselmouth.Sound(y,16000).to_pitch_ac(pitch_floor=65,pitch_ceiling=600).selected_array["frequency"]
    assert y.shape==x.shape and np.isfinite(y).all() and np.max(np.abs(y))<=.990001
    assert abs(np.median(a[a>0])/np.median(b[b>0])-1)<.02
    fa=parselmouth.Sound(x,16000).to_formant_burg(maximum_formant=5500)
    fb=parselmouth.Sound(y,16000).to_formant_burg(maximum_formant=5500)
    formant_ratio=fb.get_value_at_time(1,.75)/fa.get_value_at_time(1,.75)
    assert 1.03<formant_ratio<1.2,formant_ratio
    from .waveforms import variants
    altered=variants(x,"synthetic-vowel",0,(4000,18000))
    assert set(altered)==set(range(1,8))
    for flags,(audio,detail) in altered.items():
        assert audio.shape==(24000,) and np.isfinite(audio).all() and np.abs(audio).max()<=.990001
        if flags&1:assert len(detail["pitch_anchors"])==4 and max(map(abs,detail["pitch_anchors"]))<=1
    augmented_model=make_model(candidate("smoke_aug",set_id="3",augmented=True),speaker,
        context["state"],context["statistics"],context["cache"],scaler)
    augmented_model.load_state_dict(cp["model"],strict=False);augmented_model.set_epoch(11)
    captured=[]
    hook=augmented_model.auxiliary.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().cpu().clone()))
    augmented_model.train();logits=augmented_model(wave.repeat(8,1))
    assert torch.isfinite(logits).all() and augmented_model.mask_trace
    for row in augmented_model.mask_trace:
        assert captured[0][row["item"],:,row["time_start"]:row["time_start"]+row["time_width"]].eq(0).all()
    hook.remove();del augmented_model
    write_json(REPORT/"preflight.json",dict(native_logits_preserved_all_aux_sets=True,auxiliary_backward_finite=True,
        synthetic_formant_ratio=formant_ratio,synthetic_pitch_preserved=True,parameter_counts=parameter_counts,
        seven_waveform_combinations_finite=True,auxiliary_time_masks_aligned=True,
        sources={p.name:engine.file_hash(p) for p in Path(__file__).parent.glob("*.py")},
        train_validation_disjoint=True,test_audio_read=False,test_evaluated=False))
    del original,cp,bank;torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-only",action="store_true")
    args=parser.parse_args()
    STUDY.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    with (STUDY/"driver.lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(2);engine.seed_all(42)
        torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
        assert torch.cuda.is_available()
        cfg=load_config(ROOT/"configs/task2_real.json");metadata=read_metadata(cfg)
        state,statistics=load_source(ROOT/"runs/conformer-research")
        cache=FeatureCache(STUDY/"feature_cache",workers=8)
        context=dict(cfg=cfg,metadata=metadata,state=state,statistics=statistics,cache=cache)
        if args.smoke_only:
            smoke(context);cache.close();return
        frozen=freeze();context["frozen"]=frozen
        assert (REPORT/"preflight.json").exists(),"Run --smoke-only before detaching"
        checked=json.loads((REPORT/"preflight.json").read_text())["sources"]
        assert checked=={p.name:engine.file_hash(p) for p in Path(__file__).parent.glob("*.py")},"Run preflight after source changes"
        if (STUDY/"complete.json").exists():cache.close();return
        (STUDY/"failure.json").unlink(missing_ok=True)
        process_path=STUDY/"process.json"
        started=json.loads(process_path.read_text())["started_unix"] if process_path.exists() else time.time()
        write_json(process_path,dict(pid=os.getpid(),started_unix=started,deadline_unix=started+7*3600,
            started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime(started)),session_id=os.getsid(0)))
        recordings={}
        for source in frozen["original_training_sources"]:
            path=ROOT/source["prepared_path"];assert engine.file_hash(path)==source["prepared_sha256"]
            recordings[source["noise_id"]]=sf.read(path,dtype="float32")[0]
        context["recordings"]=recordings
        baseline();base=candidate("baseline")
        singles=[evaluate_candidate(spec,context) for spec in frozen["regularization"]]
        options=singles.copy()
        if all(better(s,base) for s in singles):
            options.append(evaluate_candidate(candidate("wd_dropout",wd=1e-3,dropout=.2),context))
        regular=select("regularization",base,options)
        variants=[evaluate_candidate(dict(regular,**setting),context) for setting in frozen["augmentation_candidates"]]
        augmented=select("augmentation",regular,variants)
        auxiliaries=[]
        for set_id in frozen["auxiliary_required"]:
            auxiliaries.append(evaluate_candidate(dict(augmented,name=f"aux_set{set_id}",set_id=set_id),context))
        for set_id in frozen["auxiliary_optional"]:
            recent=json.loads((REPORT/"candidates"/(auxiliaries[-1]["name"]+".json")).read_text())["seconds"]
            remaining=started+7*3600-time.time()
            if remaining<max(recent,35*60)+45*60:
                write_json(REPORT/f"optional_set{set_id}_skipped.json",dict(reason="Reserve final evaluation and audit within7hour target",remaining_seconds=remaining))
                continue
            auxiliaries.append(evaluate_candidate(dict(augmented,name=f"aux_set{set_id}",set_id=set_id),context))
        selected=select("auxiliary",augmented,auxiliaries)
        final_filter(selected,context)
        progress(stage="auditing",candidate=selected["name"])
        audit_rows=[json.loads(p.read_text()) for p in (REPORT/"folds").glob("*/*.json")]
        checks=[r["audit"] for r in audit_rows if not r.get("reused")]
        assert checks and all(r["final_epoch"]==100 for r in audit_rows)
        for row in audit_rows:assert engine.file_hash(ROOT/row["run_dir"]/"best.pt")==row["best_sha256"]
        assert freeze()==frozen
        write_json(REPORT/"independent_audit.json",dict(completed_folds=len(audit_rows),trained_folds=len(checks),
            **{k:sum(c[k] for c in checks) for k in ("epoch_predictions_recomputed","training_draws_checked","best_predictions_replayed","final_predictions_recomputed")},
            source_hashes_preserved=True,all_final_epochs100=True,test_audio_read=False,test_evaluated=False))
        write_json(STUDY/"complete.json",dict(selected=selected,all_stages_and_audits_complete=True,test_evaluated=False))
        progress(stage="complete",candidate=selected["name"]);render();cache.close()
        print("OVERNIGHT_COMPLETE",flush=True)


if __name__=="__main__":
    try:main()
    except Exception:
        if STUDY.exists():write_json(STUDY/"failure.json",dict(error=traceback.format_exc(),timestamp=time.time(),test_evaluated=False))
        raise
