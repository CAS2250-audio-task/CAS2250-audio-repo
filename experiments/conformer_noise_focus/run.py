"""Paired 2x2 training comparison; reuse the old original-noise control."""
import json
from importlib.metadata import version
from pathlib import Path
import time

import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src import engine
from src.config import load_config
from src.dataset import read_metadata
from experiments.conformer.model import load_source, SOURCE_HASHES
from experiments.conformer.run import Clips, loaders, atomic_save
from experiments.conformer_sanity import SPEAKERS, speaker_fold
from experiments.conformer_aug.model import AugmentedConformer
from experiments.pretrain.run import tensor_hash
from .augment import FocusTrainingClips
from .sources import ROOT, STUDY, REPORT

CONDITIONS = {"speech":dict(add_speech=True,focus_zero=False),
              "focus":dict(add_speech=False,focus_zero=True),
              "speech_focus":dict(add_speech=True,focus_zero=True)}


def protocol(cfg, metadata):
    old=json.loads((ROOT/"reports/conformer-aug-v1/protocol.json").read_text())
    added=json.loads((REPORT/"added_speech_sources.json").read_text())
    originals=old["training_noise"]["sources"]
    for key in ("original_filename","original_sha256","prepared_sha256"):
        assert not {s[key] for s in originals+added}&{s[key] for s in old["evaluation_sources"]}
    assert len(added)>=30 and len({s["pcm_sha256"] for s in added})==len(added)
    pins={str(p.relative_to(ROOT)):engine.file_hash(p) for p in Path(__file__).parent.glob("*.py")}
    for folder in ("experiments/conformer", "experiments/conformer_aug"):
        pins.update({str(p.relative_to(ROOT)):engine.file_hash(p) for p in (ROOT/folder).glob("*.py")})
    for name in ("experiments/conformer_sanity.py","experiments/noise_stress/mixing.py","experiments/denoise_eval/filters.py"):
        pins[name]=engine.file_hash(ROOT/name)
    bases={str((ROOT/f"runs/conformer-aug-v1/folds/{s}/noise"/name).relative_to(ROOT)):
           engine.file_hash(ROOT/f"runs/conformer-aug-v1/folds/{s}/noise"/name)
           for s in SPEAKERS for name in ("best.pt","initialization.json","summary.json","predictions.csv","metadata.csv")}
    result=dict(study="conformer-noise-focus-v1",seed=42,folds=list(SPEAKERS),new_runs=12,epochs=10,batch_size=32,
        conditions={"baseline":"reuse existing noise-trained four folds",**CONDITIONS},
        initialization="original Korean ASR weights and identical seed42 head; no continuation of a previous fine-tuned model",
        source_weights=SOURCE_HASHES,reference_config=cfg.to_dict(),model=old["model"],
        encoder_lr=1e-5,classifier_lr=5e-4,weight_decay=1e-4,dropout=old["dropout"],
        scheduler="cosine 10epochs",sampler="same inverse-class weighted sampler,609 draws/epoch,seed42",
        original_probability=.5,noise_probability=.5,
        source_policy="speech flag: .5 of noise draws use added sources; uniform family then uniform file. Otherwise original six sources uniformly",
        snr_policy="focus flag: .5 of noise draws U[-3,3], otherwise original U[-5,20]. No focus: original U[-5,20]",
        focused_input_proportions=dict(original=.5,wide_snr_noise=.25,near_zero_noise=.25),
        total_training_budget_unchanged=True,source_and_snr_random_streams_independent=True,
        original_noise_sources=originals,added_speech_sources=added,
        source_preparation_audit_sha256=engine.file_hash(REPORT/"source_preparation_audit.json"),
        source_overlap_audit_sha256=engine.file_hash(REPORT/"source_overlap_audit.json"),
        signal_power="same full-clip power; silence uses fold-training600 keyword RMS median squared",
        clipping="joint speech/noise attenuation to.99; no clipping or peak normalization after filtering",
        speed=False,mask=False,auxiliary_features=False,
        checkpoint_selection="original clean-validation macro-F1, earliest tie; no noisy/filtered epoch selection",
        evaluation="same 8 noises x4 SNRs x3 draws and original/noise-only; all four training conditions with and without fixed DFN3",
        dfn3="fixed official inference from denoise-eval-v1; mixture only, no training or fitted dry blend",
        source_sha256=pins,baseline_pins=bases,
        dependency_sha256={p:engine.file_hash(ROOT/p) for p in (
            "reports/conformer-aug-v1/protocol.json","reports/conformer-aug-v1/predictions.csv.gz",
            "reports/noise-stress-v1/protocol.json","reports/noise-stress-v1/mixtures.csv.gz",
            "reports/denoise-eval-v1/protocol.json","reports/denoise-eval-v1/predictions.csv.gz")},
        development_wav_sha256={p:engine.file_hash(Path(cfg.data_root)/p) for p in metadata[metadata.split.ne("test")].path},
        cuda_matmul_allow_tf32=True,cudnn_allow_tf32=True,
        versions={p:version(p) for p in ("torch","torchaudio","speechbrain","numpy","deepfilterlib")},
        test_audio_read=False,test_evaluated=False,
        limitation="one training seed and four development speakers; numbered MS-SNSD files are unique PCM, not proven independent source sessions")
    target=REPORT/"protocol.json"
    if target.exists():
        assert json.loads(target.read_text())==result,"Frozen focus protocol changed"
    else:
        engine.write_json(target,result)
    return result


def run_one(condition,speaker,cfg,frame,state,statistics,sources,added,recordings):
    run=STUDY/"folds"/speaker/condition
    run.mkdir(parents=True,exist_ok=True)
    if (run/"summary.json").exists():return
    engine.seed_all(42)
    train_frame=frame[frame.split.eq("train")]
    val_frame=frame[frame.split.eq("val")]
    assert len(train_frame)==609 and len(val_frame)==203
    assert not train_frame.group.isin(("speaker:spk01","speaker:"+speaker)).any()
    train=FocusTrainingClips(train_frame,cfg,speaker,sources,added,recordings,**CONDITIONS[condition])
    val=Clips(val_frame,cfg)
    original_train=Clips(train_frame,cfg)
    train_loader,val_loader,_=loaders(train,val,42,32)
    model=AugmentedConformer(state,statistics,speaker,mask=False).cuda()
    initial=dict(classifier_initial_sha256=tensor_hash(model.fc.state_dict()),
        encoder_initial_sha256=tensor_hash({k:v for k,v in model.state_dict().items() if not k.startswith("fc.")}),
        total_parameters=sum(p.numel() for p in model.parameters()),
        source_normalizer_count=int(statistics["count"]),mask_floor=model.mask_floor.cpu().tolist(),
        silence_reference_power=train.reference_power)
    baseline=json.loads((ROOT/f"runs/conformer-aug-v1/folds/{speaker}/noise/initialization.json").read_text())
    assert initial==baseline
    optimizer=torch.optim.AdamW([
        dict(params=list(model.frontend.parameters())+list(model.transformer.parameters()),lr=1e-5),
        dict(params=model.fc.parameters(),lr=5e-4)],weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=10)
    history=[];best_f1=-1.;best_epoch=None;first=1
    latest=run/"last.pt"
    if latest.exists():
        cp=torch.load(latest,map_location="cpu",weights_only=True)
        model.load_state_dict(cp["model"],strict=True)
        optimizer.load_state_dict(cp["optimizer"]);scheduler.load_state_dict(cp["scheduler"])
        history=cp["history"];best_f1=cp["best_f1"];best_epoch=cp["best_epoch"];first=cp["epoch"]+1
        torch.set_rng_state(cp["torch_rng"]);torch.cuda.set_rng_state_all(cp["cuda_rng"])
        train_loader.sampler.generator.set_state(cp["loader_rng"])
        val_loader.generator.set_state(cp["val_loader_rng"])
        del cp
    else:
        engine.write_json(run/"initialization.json",initial)
        engine.write_json(run/"config.json",cfg.to_dict())
        frame.to_csv(run/"metadata.csv",index=False)
    print("RUN_START",speaker,condition,first,flush=True)
    started=time.monotonic()
    for epoch in range(first,11):
        train.set_epoch(epoch);train.trace=[]
        model.set_epoch(epoch);model.train();total=0.
        for x,y,_ in train_loader:
            x,y=x.cuda(),y.cuda();optimizer.zero_grad(set_to_none=True)
            loss=F.cross_entropy(model(x),y)
            assert torch.isfinite(loss)
            loss.backward();optimizer.step();total+=float(loss.detach())*len(y)
        scores,predictions=engine.evaluate_model(model,val_loader,"cuda",11)
        history.append(dict(epoch=epoch,train_loss=total/609,encoder_lr=optimizer.param_groups[0]["lr"],
            head_lr=optimizer.param_groups[1]["lr"],**{"val_"+k:v for k,v in scores.items()}))
        directory=run/"epochs";directory.mkdir(exist_ok=True)
        predictions.to_csv(directory/f"epoch_{epoch:02d}.csv",index=False)
        assert len(train.trace)==609
        pd.DataFrame(train.trace).to_csv(directory/f"augmentation_{epoch:02d}.csv.gz",index=False)
        if scores["macro_f1"]>best_f1:
            best_f1,best_epoch=scores["macro_f1"],epoch
            atomic_save(dict(model=model.state_dict(),epoch=epoch,validation=scores,condition=condition),run/"best.pt")
        pd.DataFrame(history).to_csv(run/"history.csv",index=False)
        scheduler.step()
        atomic_save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            epoch=epoch,history=history,best_f1=best_f1,best_epoch=best_epoch,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
            loader_rng=train_loader.sampler.generator.get_state(),val_loader_rng=val_loader.generator.get_state()),latest)
        progress=dict(condition=condition,speaker=speaker,epoch=epoch,clean_val_macro_f1=scores["macro_f1"],
            completed_runs=len(list((STUDY/"folds").glob("*/*/summary.json"))),planned_new_runs=12)
        engine.write_json(STUDY/"progress.json",progress)
        print("EPOCH",json.dumps(progress),flush=True)
    best=torch.load(run/"best.pt",map_location="cpu",weights_only=True)
    model.load_state_dict(best["model"],strict=True)
    scores,predictions=engine.evaluate_model(model,val_loader,"cuda",11)
    train_scores,train_predictions=engine.evaluate_model(model,DataLoader(original_train,batch_size=32),"cuda",11)
    predictions.to_csv(run/"predictions.csv",index=False)
    train_predictions.to_csv(run/"original_train_predictions.csv",index=False)
    assert torch.equal(model.feature_mean.cpu(),statistics["glob_mean"])
    assert torch.equal(model.feature_std.cpu(),statistics["glob_std"])
    engine.write_json(run/"summary.json",dict(condition=condition,speaker=speaker,seed=42,best_epoch=best_epoch,
        validation=scores,original_train=train_scores,checkpoint_sha256=engine.file_hash(run/"best.pt"),
        normalization_unchanged=True,elapsed_seconds=time.monotonic()-started,test_evaluated=False))
    del model,optimizer,best
    torch.cuda.empty_cache()
    print("RUN_COMPLETE",speaker,condition,flush=True)


def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    STUDY.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    cfg=load_config(ROOT/"configs/task2_real.json");metadata=read_metadata(cfg)
    frozen=protocol(cfg,metadata)
    sources=frozen["original_noise_sources"];added=frozen["added_speech_sources"]
    recordings={}
    for source in sources+added:
        path=ROOT/source["prepared_path"]
        assert engine.file_hash(path)==source["prepared_sha256"]
        recordings[source["noise_id"]]=sf.read(path,dtype="float32")[0]
    state,statistics=load_source(ROOT/"runs/conformer-research")
    for condition in CONDITIONS:
        for speaker in SPEAKERS:
            run_one(condition,speaker,cfg,speaker_fold(metadata,speaker,cfg),state,statistics,sources,added,recordings)
    assert protocol(cfg,metadata)==frozen
    engine.write_json(STUDY/"training_complete.json",dict(new_runs=12,epochs=120,
        protocol_sha256=engine.file_hash(REPORT/"protocol.json"),test_evaluated=False))
    print("FOCUS_TRAINING_COMPLETE",flush=True)


if __name__=="__main__":main()
