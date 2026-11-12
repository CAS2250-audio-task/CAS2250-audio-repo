"""Four development folds: baseline, legacy augmentation, noise, combined."""
import json
from importlib.metadata import version
from pathlib import Path
import time

import numpy as np
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
from experiments.pretrain.run import tensor_hash
from .augment import TrainingClips, speed_bank, LIBDIR, LIBFILES
from .model import AugmentedConformer
from .sources import ROOT, STUDY, REPORT

CONDITIONS={"legacy":dict(legacy=True,noise=False),"noise":dict(legacy=False,noise=True),
            "combined":dict(legacy=True,noise=True)}


def protocol(cfg,frame):
    sources=json.loads((REPORT/"training_noise_sources.json").read_text())
    evaluation=json.loads((ROOT/"reports/noise-stress-v1/noise_sources.json").read_text())
    for key in ("original_filename","original_sha256","prepared_sha256"):
        assert not {s[key] for s in sources}&{s[key] for s in evaluation}
    cv=pd.read_csv(ROOT/"reports/conformer-sanity-v1/speaker_cv.csv")
    baseline=cv[cv.model.eq("conformer")]
    base_pins={str((Path(row.source_run)/name)):engine.file_hash(ROOT/row.source_run/name)
               for row in baseline.itertuples() for name in ("best.pt","metadata.csv","initialization.json","comparison_summary.json")}
    originals={p:engine.file_hash(Path(cfg.data_root)/p) for p in frame[frame.split.ne("test")].path}
    pins={str(p.relative_to(ROOT)):engine.file_hash(p) for p in Path(__file__).parent.glob("*.py")}
    pins.update({p:engine.file_hash(ROOT/p) for p in ("experiments/conformer/model.py","experiments/conformer/run.py",
                 "experiments/noise_stress/mixing.py","experiments/conformer_sanity.py")})
    result=dict(study="conformer-aug-v1",seed=42,folds=list(SPEAKERS),epochs=10,batch_size=32,
                conditions={"baseline":"reuse clean 4-fold seed42 checkpoints",**CONDITIONS},
                reference_config=cfg.to_dict(),encoder_lr=1e-5,classifier_lr=5e-4,weight_decay=1e-4,
                dropout="original encoder dropout .1; no additional classifier dropout",scheduler="cosine 10 epochs",
                sampler="inverse-class WeightedRandomSampler, 609 draws/epoch; seed42, paired across conditions",
                model="same pretrained Korean 12-layer Conformer and 256->11 head; no auxiliary acoustic inputs",
                source_weights=SOURCE_HASHES,initialization="original ASR weights and fresh identical seed42 head for every run, not a previously fine-tuned checkpoint",
                normalization="original ASR global mean/std frozen; masks bypassed in evaluation",
                speed=dict(probability=.5,range=[.9,1.1],pitch_preserved=True,variants_per_file=2,
                           protection="entire ref reviewed non-padding interval; skip rates that would cut it; not new VAD annotation",
                           padding="center zero pad only missing length after acceleration"),
                mask=dict(probability=.5,time_frames=[2,5],time_ms=[20,50],frequency_bins=[3,5],native_mel_bins=80,
                          adaptation="frequency width scaled from prior 4..8 of 128 bins; not an exact old input representation",
                          fill="per-frequency native Fbank output of a zero waveform, before fixed ASR normalization"),
                training_noise=dict(probability=.5,snr_uniform_db=[-5,20],sources=sources,
                                    sampling="uniform source (2 per category), uniform contiguous 1.5s crop per training draw; new each epoch",
                                    signal_power="current post-speed full clip, including existing background/padding",
                                    silence_reference="median RMS of 600 unaugmented fold-training keyword clips, squared; nominal SNR",
                                    dc_removed=True,near_silence_power_threshold=1e-8,
                                    clipping="joint attenuation to .99 peak, same gain on speech/noise, no clipping"),
                combined="independent probability .5 per policy; waveform speed then noise then native-feature masking",
                checkpoint_selection="highest original clean validation macro-F1, earliest tie, same as baseline; noisy scores not used for checkpoint selection",
                evaluation="original validation plus exactly the existing 8-source/4-level/3-draw stress mixtures and noise-only windows",
                evaluation_protocol_sha256=engine.file_hash(ROOT/"reports/noise-stress-v1/protocol.json"),
                evaluation_sources=evaluation,baseline_pins=base_pins,development_wav_sha256=originals,source_sha256=pins,
                speed_library_sha256={str(LIBDIR/name):engine.file_hash(LIBDIR/name) for name in LIBFILES},
                cuda_matmul_allow_tf32=True,cudnn_allow_tf32=True,
                versions={name:version(name) for name in ("torch","torchaudio","speechbrain","numpy")},
                test_audio_read=False,test_evaluated=False)
    target=REPORT/"protocol.json"
    if target.exists():assert json.loads(target.read_text())==result,"Frozen augmentation protocol changed"
    else:engine.write_json(target,result)
    return result,baseline


def run_one(condition,speaker,cfg,frame,state,statistics,bank,mapping,sources,recordings,baseline_run):
    run=STUDY/"folds"/speaker/condition
    run.mkdir(parents=True,exist_ok=True)
    if (run/"summary.json").exists():return json.loads((run/"summary.json").read_text())
    flags=CONDITIONS[condition]
    engine.seed_all(42)
    train_frame=frame[frame.split.eq("train")]
    val_frame=frame[frame.split.eq("val")]
    assert len(train_frame)==609 and len(val_frame)==203
    assert not train_frame.group.isin(("speaker:spk01","speaker:"+speaker)).any()
    train=TrainingClips(train_frame,cfg,speaker,bank,mapping,sources,recordings,**flags)
    val=Clips(val_frame,cfg)
    original_train=Clips(train_frame,cfg)
    train_loader,val_loader,_=loaders(train,val,42,32)
    model=AugmentedConformer(state,statistics,speaker,mask=flags["legacy"]).cuda()
    initial=dict(classifier_initial_sha256=tensor_hash(model.fc.state_dict()),
                 encoder_initial_sha256=tensor_hash({k:v for k,v in model.state_dict().items() if not k.startswith("fc.")}),
                 total_parameters=sum(p.numel() for p in model.parameters()),
                 source_normalizer_count=int(statistics["count"]),mask_floor=model.mask_floor.cpu().tolist(),
                 silence_reference_power=train.reference_power)
    baseline_init=json.loads((baseline_run/"initialization.json").read_text())
    assert initial["classifier_initial_sha256"]==baseline_init["classifier_initial_sha256"]
    assert initial["encoder_initial_sha256"]==baseline_init["encoder_initial_sha256"]
    assert initial["total_parameters"]==baseline_init["total_parameters"]
    optimizer=torch.optim.AdamW([
        {"params":list(model.frontend.parameters())+list(model.transformer.parameters()),"lr":1e-5},
        {"params":model.fc.parameters(),"lr":5e-4}],weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=10)
    history=[];best_f1=-1.;best_epoch=None;first=1
    latest=run/"last.pt"
    if latest.exists():
        checkpoint=torch.load(latest,map_location="cpu",weights_only=True)
        model.load_state_dict(checkpoint["model"],strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"]);scheduler.load_state_dict(checkpoint["scheduler"])
        history=checkpoint["history"];best_f1=checkpoint["best_f1"];best_epoch=checkpoint["best_epoch"]
        first=checkpoint["epoch"]+1
        torch.set_rng_state(checkpoint["torch_rng"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        train_loader.sampler.generator.set_state(checkpoint["loader_rng"])
        val_loader.generator.set_state(checkpoint["val_loader_rng"])
        del checkpoint
    else:
        engine.write_json(run/"initialization.json",initial)
        engine.write_json(run/"config.json",cfg.to_dict())
        frame.to_csv(run/"metadata.csv",index=False)
    started=time.monotonic()
    print("RUN_START",speaker,condition,"epoch",first,flush=True)
    for epoch in range(first,11):
        train.set_epoch(epoch);train.trace=[]
        model.set_epoch(epoch);model.mask_trace=[];model.train()
        total=0.
        for x,y,_ in train_loader:
            x,y=x.cuda(),y.cuda()
            optimizer.zero_grad(set_to_none=True)
            loss=F.cross_entropy(model(x),y)
            assert torch.isfinite(loss)
            loss.backward();optimizer.step()
            total+=float(loss.detach())*len(y)
        scores,predictions=engine.evaluate_model(model,val_loader,"cuda",11)
        history.append(dict(epoch=epoch,train_loss=total/609,encoder_lr=optimizer.param_groups[0]["lr"],
                            head_lr=optimizer.param_groups[-1]["lr"],**{"val_"+k:v for k,v in scores.items()}))
        directory=run/"epochs";directory.mkdir(exist_ok=True)
        predictions.to_csv(directory/f"epoch_{epoch:02d}.csv",index=False)
        pd.DataFrame(train.trace).to_csv(directory/f"augmentation_{epoch:02d}.csv.gz",index=False)
        engine.write_json(directory/f"masks_{epoch:02d}.json",model.mask_trace)
        assert len(train.trace)==609
        if scores["macro_f1"]>best_f1:
            best_f1,best_epoch=scores["macro_f1"],epoch
            atomic_save(dict(model=model.state_dict(),epoch=epoch,validation=scores,condition=condition),run/"best.pt")
        pd.DataFrame(history).to_csv(run/"history.csv",index=False)
        scheduler.step()
        atomic_save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                         epoch=epoch,history=history,best_f1=best_f1,best_epoch=best_epoch,
                         torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                         loader_rng=train_loader.sampler.generator.get_state(),val_loader_rng=val_loader.generator.get_state()),latest)
        progress=dict(speaker=speaker,condition=condition,epoch=epoch,clean_val_macro_f1=scores["macro_f1"],
                      completed_runs=len(list((STUDY/"folds").glob("*/*/summary.json"))),planned_new_runs=12,
                      elapsed_run_seconds=round(time.monotonic()-started),test_evaluated=False)
        engine.write_json(STUDY/"progress.json",progress)
        print("EPOCH",json.dumps(progress),flush=True)
    best=torch.load(run/"best.pt",map_location="cpu",weights_only=True)
    model.load_state_dict(best["model"],strict=True)
    scores,predictions=engine.evaluate_model(model,val_loader,"cuda",11)
    original_scores,original_predictions=engine.evaluate_model(model,DataLoader(original_train,batch_size=32),"cuda",11)
    assert torch.equal(model.feature_mean.cpu(),statistics["glob_mean"]) and torch.equal(model.feature_std.cpu(),statistics["glob_std"])
    predictions.to_csv(run/"predictions.csv",index=False)
    original_predictions.to_csv(run/"original_train_predictions.csv",index=False)
    result=dict(speaker=speaker,condition=condition,seed=42,best_epoch=best_epoch,validation=scores,
                original_train=original_scores,generalization_gap=original_scores["macro_f1"]-scores["macro_f1"],
                checkpoint_sha256=engine.file_hash(run/"best.pt"),normalization_unchanged=True,
                elapsed_seconds=time.monotonic()-started,test_evaluated=False)
    engine.write_json(run/"summary.json",result)
    print("RUN_COMPLETE",json.dumps(result),flush=True)
    del model,optimizer,best
    torch.cuda.empty_cache()
    return result


def main():
    torch.set_num_threads(2)
    import speechbrain
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    cfg=load_config(ROOT/"configs/task2_real.json")
    frame=read_metadata(cfg)
    STUDY.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    frozen,baseline=protocol(cfg,frame)
    sources=frozen["training_noise"]["sources"]
    recordings={}
    for src in sources:
        path=ROOT/src["prepared_path"]
        assert engine.file_hash(path)==src["prepared_sha256"]
        recordings[src["noise_id"]]=sf.read(path,dtype="float32")[0]
    development=frame[frame.split.ne("test")].reset_index(drop=True)
    bank,bank_manifest=speed_bank(development,cfg)
    mapping={path:i for i,path in enumerate(development.path)}
    state,statistics=load_source(ROOT/"runs/conformer-research")
    results=[]
    # Each condition finishes all four folds before moving to the next condition.
    for condition in CONDITIONS:
        for speaker in SPEAKERS:
            fold=speaker_fold(frame,speaker,cfg)
            old=ROOT/baseline[baseline.heldout_speaker.eq(speaker)].iloc[0].source_run
            results.append(run_one(condition,speaker,cfg,fold,state,statistics,bank,mapping,sources,recordings,old))
    assert protocol(cfg,frame)[0]==frozen
    engine.write_json(STUDY/"training_complete.json",dict(new_runs=12,epochs=120,
                      protocol_sha256=engine.file_hash(REPORT/"protocol.json"),test_evaluated=False))
    print("TRAINING_COMPLETE",len(results),flush=True)


if __name__=="__main__":main()
