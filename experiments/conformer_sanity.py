"""Leakage checks, counterfactual inputs and fixed-protocol speaker CV."""
import copy
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from src import engine
from src.config import load_config
from src.dataset import read_metadata, validate_metadata
from experiments.pretrain.run import export_run
from experiments.conformer.model import ConformerClassifier, load_source
from experiments.conformer.run import Clips, run_one

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT/"runs/conformer-sanity-v1"
REPORT = ROOT/"reports/conformer-sanity-v1"
SPEAKERS = ("spk02", "spk03", "spk04", "spk05")


def speaker_fold(frame, heldout, cfg):
    if heldout not in SPEAKERS:
        raise ValueError("Only development speakers may be held out")
    result=frame.copy()
    group_speaker=result.group.str.removeprefix("speaker:")
    if set(group_speaker) != {"spk01", *SPEAKERS}:
        raise ValueError("Unexpected reference groups")
    result["split"]=np.where(group_speaker.eq("spk01"),"test",
                             np.where(group_speaker.eq(heldout),"val","train"))
    validate_metadata(result,cfg)
    assert result.groupby("split").size().to_dict()=={"test":203,"train":609,"val":203}
    assert set(result[result.split=="test"].path)==set(frame[frame.split=="test"].path)
    return result


def shuffled_training_labels(frame):
    result=frame.copy()
    mask=result.split.eq("train")
    labels=result.loc[mask,"label"].to_numpy()
    result.loc[mask,"label"]=np.random.default_rng(7421).permutation(labels)
    assert result.loc[mask,"label"].value_counts().sort_index().equals(frame.loc[mask,"label"].value_counts().sort_index())
    assert result.loc[~mask].equals(frame.loc[~mask])
    return result


class Counterfactual(Dataset):
    def __init__(self,dataset,kind,permutation=None):
        self.dataset,self.kind,self.permutation=dataset,kind,permutation

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,index):
        x,y,path=self.dataset[index]
        if self.kind=="zero_audio":
            x=torch.zeros_like(x)
        elif self.kind=="permuted_audio":
            x=self.dataset[int(self.permutation[index])][0]
        elif self.kind=="permuted_targets":
            y=self.dataset[int(self.permutation[index])][1]
        else:
            raise ValueError(self.kind)
        return x,y,path


def overlap_audit(cfg,frame):
    overlaps={}
    for column in ("path","group","speaker_id","source_sha256","source_filename"):
        subset=frame[frame[column].ne("")]
        if column=="speaker_id":subset=subset[subset.label.ne("silence")]
        overlap=set(subset[subset.split=="train"][column])&set(subset[subset.split=="val"][column])
        overlaps[column]=len(overlap)
        assert not overlap,(column,overlap)
    rows=[]
    for row in frame[frame.split.isin(["train","val"])].itertuples():
        path=Path(cfg.data_root)/row.path
        wav,sr=sf.read(path,dtype="float32")
        assert sr==16000 and wav.shape==(24000,) and np.isfinite(wav).all()
        rows.append(dict(path=row.path,split=row.split,label=row.label,
                         file_sha256=engine.file_hash(path),pcm_sha256=hashlib.sha256(wav.tobytes()).hexdigest(),
                         rms=float(np.sqrt(np.mean(wav**2))),nonzero_samples=int(np.count_nonzero(wav))))
    audio=pd.DataFrame(rows)
    audio.to_csv(REPORT/"audio_identity_audit.csv",index=False,encoding="utf-8-sig")
    duplicate_counts={}
    for column in ("file_sha256","pcm_sha256"):
        shared=set(audio[audio.split=="train"][column])&set(audio[audio.split=="val"][column])
        duplicate_counts[column]=len(shared)
        assert not shared
    result=dict(train_speakers=sorted(frame[(frame.split=="train")&frame.label.ne("silence")].speaker_id.unique()),
                validation_speakers=sorted(frame[(frame.split=="val")&frame.label.ne("silence")].speaker_id.unique()),
                train_validation_metadata_overlap=overlaps,exact_audio_duplicates=duplicate_counts,
                zero_waveforms=int((audio.nonzero_samples==0).sum()),checked_train_and_validation_files=len(audio),
                model_input="waveform tensor only; paths and labels are not model arguments",
                source_normalization="fixed pretrained global statistics; no validation statistics fitted",
                pretraining_corpus_overlap="not independently checked against the full KsponSpeech corpus",
                transformed_near_duplicates="not ruled out by exact hashes",test_evaluated=False)
    engine.write_json(REPORT/"leakage_audit.json",result)
    return result


def input_controls(cfg,frame,state,statistics):
    path=REPORT/"counterfactual_summary.json"
    if path.exists():return json.loads(path.read_text())
    val=Clips(frame[frame.split=="val"],cfg)
    checkpoint_path=ROOT/"runs/conformer-ref-v1/runs/conformer_finetune_seed44/best.pt"
    model=ConformerClassifier(state,statistics,mode="finetune").cuda()
    checkpoint=torch.load(checkpoint_path,map_location="cpu",weights_only=True)
    model.load_state_dict(checkpoint["model"],strict=True)
    scores,reference=engine.evaluate_model(model,DataLoader(val,batch_size=32),"cuda",11)
    saved=pd.read_csv(ROOT/"reports/conformer-ref-v1/runs/conformer_finetune_seed44/validation/predictions.csv")
    assert reference.set_index("path").pred_idx.equals(saved.set_index("path").pred_idx)
    rows=[dict(condition="normal_audio",**scores)]
    permutation=np.random.default_rng(3957).permutation(len(val))
    assert (permutation!=np.arange(len(val))).sum()>190
    for kind in ("zero_audio","permuted_audio","permuted_targets"):
        dataset=Counterfactual(val,kind,permutation)
        score,predictions=engine.evaluate_model(model,DataLoader(dataset,batch_size=32),"cuda",11)
        predictions.to_csv(REPORT/f"control_{kind}.csv",index=False,encoding="utf-8-sig")
        rows.append(dict(condition=kind,**score))
    score,reordered=engine.evaluate_model(model,DataLoader(Subset(val,permutation.tolist()),batch_size=17),"cuda",11)
    expected=reference.set_index("path").sort_index()
    actual=reordered.set_index("path").sort_index()
    assert expected.pred_idx.equals(actual.pred_idx) and expected.true_idx.equals(actual.true_idx)
    rows.append(dict(condition="reordered_files_batch17",**score))
    pd.DataFrame(rows).to_csv(REPORT/"counterfactual_summary.csv",index=False)
    result=dict(checkpoint_sha256=engine.file_hash(checkpoint_path),seed=44,
                normal_correct=int((reference.true_idx==reference.pred_idx).sum()),
                validation_files=203,permutation_seed=3957,reordered_predictions_identical=True,
                conditions=rows,test_evaluated=False)
    engine.write_json(path,result)
    del model,checkpoint
    torch.cuda.empty_cache()
    print("COUNTERFACTUALS",json.dumps(result),flush=True)
    return result


def resnet_fold(cfg,frame,fold_study,fold_report):
    run=fold_study/"runs/imagenet_seed42"
    if (run/"comparison_summary.json").exists():return run
    if run.exists():run.rename(run.with_name(run.name+f"_interrupted_{time.time_ns()}"))
    fold_study.mkdir(parents=True,exist_ok=True)
    metadata=fold_study/"fold_metadata.csv"
    frame.to_csv(metadata,index=False)
    run_cfg=copy.deepcopy(cfg)
    run_cfg.seed=42
    run_cfg.device="cuda"
    run_cfg.metadata_csv=str(metadata.resolve())
    saved=engine.evaluate_model
    counter=0
    def evaluate(model,loader,device,num_classes):
        nonlocal counter
        scores,predictions=saved(model,loader,device,num_classes)
        counter+=1
        if counter<=10:
            (run/"epochs").mkdir(exist_ok=True)
            predictions.to_csv(run/"epochs"/f"epoch_{counter:02d}.csv",index=False,encoding="utf-8-sig")
        return scores,predictions
    engine.evaluate_model=evaluate
    try:engine.train(run_cfg,run)
    finally:engine.evaluate_model=saved
    summary=json.loads((run/"summary.json").read_text())
    result=dict(best_epoch=summary["best_epoch"],validation=json.loads((run/"validation/metrics.json").read_text()),
                checkpoint_sha256=engine.file_hash(run/"best.pt"),test_evaluated=False)
    engine.write_json(run/"comparison_summary.json",result)
    export_run(run,fold_report/"runs"/run.name)
    torch.cuda.empty_cache()
    return run


def cv(cfg,frame,state,statistics):
    rows=[]
    for speaker in SPEAKERS:
        print("CV_HOLDOUT",speaker,flush=True)
        fold=speaker_fold(frame,speaker,cfg)
        directory=STUDY/"folds"/speaker
        report=REPORT/"folds"/speaker
        directory.mkdir(parents=True,exist_ok=True)
        report.mkdir(parents=True,exist_ok=True)
        cfg_fold=copy.deepcopy(cfg)
        cfg_fold.split_version=f"conformer-audit-dev4fold-holdout-{speaker}"
        baseline=resnet_fold(cfg_fold,fold,directory,report)
        if speaker=="spk05":
            conformer=ROOT/"runs/conformer-ref-v1/runs/conformer_finetune_seed42"
            original=pd.read_csv(conformer/"metadata.csv").fillna("")
            assert original[["path","label","group","split"]].equals(fold[["path","label","group","split"]])
        else:
            train=Clips(fold[fold.split=="train"],cfg_fold)
            val=Clips(fold[fold.split=="val"],cfg_fold)
            run_one("finetune",42,state,statistics,cfg_fold,fold,train,val,directory,report)
            conformer=directory/"runs/conformer_finetune_seed42"
        for kind,run in (("imagenet",baseline),("conformer",conformer)):
            result=json.loads((run/"comparison_summary.json").read_text())
            assert not result["test_evaluated"]
            assert engine.file_hash(run/"best.pt")==result["checkpoint_sha256"]
            metadata=pd.read_csv(run/"metadata.csv").fillna("")
            expected=set(fold[fold.split=="val"].path)
            assert set(metadata[metadata.split=="val"].path)==expected
            history=pd.read_csv(run/"history.csv")
            assert history.epoch.tolist()==list(range(1,11))
            assert int(history.loc[history.val_macro_f1.idxmax(),"epoch"])==result["best_epoch"]
            for e in range(1,11):
                p=pd.read_csv(run/"epochs"/f"epoch_{e:02d}.csv")
                assert set(p.path)==expected and len(p)==203
                scores=engine.metrics(p.true_idx,p.pred_idx,11)
                h=history[history.epoch==e].iloc[0]
                assert abs(h.val_macro_f1-scores["macro_f1"])<1e-12
                assert abs(h.val_accuracy-scores["accuracy"])<1e-12
            p=pd.read_csv(run/"validation/predictions.csv")
            score=engine.metrics(p.true_idx,p.pred_idx,11)
            assert all(abs(score[k]-result["validation"][k])<1e-12 for k in score)
            rows.append(dict(heldout_speaker=speaker,model=kind,seed=42,best_epoch=result["best_epoch"],
                             **result["validation"],correct=int((p.true_idx==p.pred_idx).sum()),
                             final_epoch_macro_f1=float(history.iloc[-1].val_macro_f1),
                             source_run=str(run.relative_to(ROOT)),reused=(speaker=="spk05" and kind=="conformer")))
        pd.DataFrame(rows).to_csv(REPORT/"speaker_cv.csv",index=False)
    return pd.DataFrame(rows)


def report(cfg,leakage,controls,folds,shuffled):
    means=folds.groupby("model")[["macro_f1","accuracy","final_epoch_macro_f1"]].mean()
    means.to_csv(REPORT/"speaker_cv_means.csv")
    fig,ax=plt.subplots(figsize=(8,4))
    x=np.arange(4)
    for kind,offset,color in (("imagenet",-.18,"#527ea6"),("conformer",.18,"#b85a91")):
        data=folds[folds.model==kind].set_index("heldout_speaker").loc[list(SPEAKERS)]
        ax.bar(x+offset,data.macro_f1*100,width=.35,label=kind,color=color)
    ax.set(xticks=x,xticklabels=SPEAKERS,ylabel="Validation macro F1 (%)",ylim=(0,105),
           title="Reference WAVs, seed 42, leave-one-speaker-out over 4 development speakers")
    ax.legend()
    ax.grid(axis="y",alpha=.2)
    fig.tight_layout()
    fig.savefig(REPORT/"speaker_cv.png",dpi=180)
    plt.close(fig)
    lines=["# Conformer 성능 과대평가 점검", "",
           "기존99.85%는spk05 한 명의203개 검증 파일을3시드로 평가한 평균 macro-F1이다. "
           "화자 일반화 확인을 위해 ref의 동일1.5초 WAV를 개발 화자02–05에서4-fold로 재분할했다. "
           "각fold는 학습609·검증203이며 test spk01은 추론하지 않았다.", "",
           "## 먼저 모델 설명 정정", "",
           "이전에 제시한 Conv2개+Transformer6층·512→11·약2,060만이라는 표는 **Whisper-base** 설명이다. "
           "실제99.85% 모델은 CNN3블록+Conformer12층·256→11·32,122,699개 파라미터다.", "",
           "## 누수와 평가 검사", "",
           f"- 학습·검증{leakage['checked_train_and_validation_files']}개: 화자·group·원본SHA·파일명 교집합0, 파일/PCM 정확 중복0.",
           "- 모델은 파형만 받는다. 라벨·파일 경로는 모델 입력이 아니다. 정규화는 원래 사전학습 통계로 고정돼 있다.",
           "- 기존6개 checkpoint 재추론4,872개는 저장 예측과 일치했다. 이번 CV8개 실행의80epoch 예측16,240개도 재계산했다.",
           "- 정확 해시 검사는 변형·근사 중복까지 배제하지 않는다. 실제 화자 identity는 기존 metadata를 신뢰한다.",
           "- KsponSpeech 전체 원음과의 직접 중복 대조는 하지 못했다. 사전학습 데이터 오염을 완전히 배제했다는 결론은 아니다.", "",
           "## 입력 대조 검사: 원래seed44 최고checkpoint", "",
           "| 조건 | 정확도 | macro-F1 |", "| --- | ---: | ---: |"]
    for r in controls["conditions"]:
        lines.append(f"| {r['condition']} | {r['accuracy']*100:.2f}% | {r['macro_f1']*100:.2f}% |")
    lines += ["", "입력 순서와batch size를 바꿔도 파일별 예측은 같았다. "
              "오디오/정답을 섞는 대조는 모델이 정답 대신 오디오에 반응하는지 확인하는 검사이며 성능 벤치마크가 아니다.", "",
              f"훈련 라벨만 섞고 분류층을10epoch 학습한 고정 인코더의 최고 검증 F1은 "
              f"**{shuffled['validation']['macro_f1']*100:.2f}%**, 정확도는{shuffled['validation']['accuracy']*100:.2f}%였다. "
              "검증 라벨은 원래대로 유지했다. 대조군에도 최고epoch 선택을 적용했다.", "",
              "## 개발 화자4-fold: seed42 한 개, 각10epoch", "",
              "| 검증 화자 | ImageNet F1 | Conformer F1 | ImageNet 정답 | Conformer 정답 |", "| --- | ---: | ---: | ---: | ---: |"]
    for speaker in SPEAKERS:
        a,b=[folds[(folds.model==kind)&(folds.heldout_speaker==speaker)].iloc[0] for kind in ("imagenet","conformer")]
        lines.append(f"| {speaker} | {a.macro_f1*100:.2f}% | {b.macro_f1*100:.2f}% | {a.correct}/203 | {b.correct}/203 |")
    lines += [f"| **평균** | **{means.loc['imagenet','macro_f1']*100:.2f}%** | **{means.loc['conformer','macro_f1']*100:.2f}%** | — | — |", "",
              "최고epoch 선택 효과를 줄여보기 위해 마지막10epoch도 비교했다. 평균F1은 "
              f"ImageNet {means.loc['imagenet','final_epoch_macro_f1']*100:.2f}%, "
              f"Conformer {means.loc['conformer','final_epoch_macro_f1']*100:.2f}%다.", "",
              "Conformer는 원래 미세조정 조건(LR1e-5/5e-4,WD1e-4,무증강)을 고정했다. "
              "spk05 Conformer는 같은 분할·seed42의 기존 실행을 재사용했다. 나머지는 매번 원래 사전학습에서 새로 학습했다. "
              "ImageNet은4fold 모두 새로 학습했으며 이번에는 양 모델 모두 SpeechBrain 환경의matmul TF32 on을 사용했다.", "",
              "이 CV는 고정 조건의 개발 결과이며 한 시드다. fold끼리 학습 화자가 겹치고 검증으로epoch를 골랐으므로 "
              "독립 최종test나 실제 배포 성능으로 해석하면 안 된다. 최종test spk01은 계속 남겨뒀다.", "",
              "![화자별 비교](speaker_cv.png)", "",
              "[누수 검사](leakage_audit.json), [입력 대조](counterfactual_summary.json), "
              "[화자별 결과](speaker_cv.csv), [프로토콜](protocol.json).", ""]
    (REPORT/"README.md").write_text("\n".join(lines))
    artifacts=[p for p in REPORT.rglob("*") if p.is_file() and p.name!="execution_audit.json"]
    engine.write_json(REPORT/"execution_audit.json",dict(
        cv_runs=8,cv_epochs=80,new_cv_runs=7,reused_cv_runs=1,shuffled_label_runs=1,
        epoch_prediction_rows_checked=16240,input_counterfactuals_verified=True,
        test_evaluated=False,artifact_sha256={str(p.relative_to(ROOT)):engine.file_hash(p) for p in artifacts}))
    print("CV_MEANS",means.to_string(),flush=True)
    print("SANITY_AUDIT_COMPLETE",flush=True)


def main():
    torch.set_num_threads(2)
    STUDY.mkdir(parents=True,exist_ok=True)
    REPORT.mkdir(parents=True,exist_ok=True)
    cfg=load_config(ROOT/"configs/task2_real.json")
    frame=read_metadata(cfg)
    state,statistics=load_source(ROOT/"runs/conformer-research")
    sources=[Path(__file__),*sorted((ROOT/"experiments/conformer").glob("*.py")),*sorted((ROOT/"src").glob("*.py"))]
    protocol=dict(reference_config=cfg.to_dict(),development_speakers=list(SPEAKERS),seed=42,epochs=10,
                  conditions=["imagenet","conformer_finetune"],frozen_training_labels_permutation_seed=7421,
                  test_excluded=True,cv_purpose="audit robustness to validation-speaker choice, not retune hyperparameters",
                  source_sha256={str(p.relative_to(ROOT)):engine.file_hash(p) for p in sources},
                  original_protocol_sha256=engine.file_hash(ROOT/"runs/conformer-ref-v1/protocol.json"))
    if (STUDY/"protocol.json").exists():
        assert json.loads((STUDY/"protocol.json").read_text())==protocol,"Protocol changed"
    engine.write_json(STUDY/"protocol.json",protocol)
    engine.write_json(REPORT/"protocol.json",protocol)
    leakage=overlap_audit(cfg,frame)
    controls=input_controls(cfg,frame,state,statistics)
    shuffled=shuffled_training_labels(frame)
    neg_study=STUDY/"negative_control"
    neg_report=REPORT/"negative_control"
    neg_study.mkdir(exist_ok=True)
    neg_report.mkdir(exist_ok=True)
    negative=run_one("frozen",42,state,statistics,copy.deepcopy(cfg),shuffled,
                     Clips(shuffled[shuffled.split=="train"],cfg),Clips(shuffled[shuffled.split=="val"],cfg),
                     neg_study,neg_report)
    folds=cv(cfg,frame,state,statistics)
    report(cfg,leakage,controls,folds,negative)
    engine.write_json(STUDY/"complete.json",dict(complete=True,test_evaluated=False))


if __name__=="__main__":main()
