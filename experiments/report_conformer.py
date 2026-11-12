"""Recompute all epoch metrics and compare Korean Conformer with ref controls."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.engine import file_hash, metrics, write_json
from experiments.pretrain.run import tensor_hash

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "runs/conformer-ref-v1"
REPORT = ROOT / "reports/conformer-ref-v1"
BASELINE = ROOT / "runs/speech-pretrain-ref-v1"
KINDS = ("imagenet", "speech_sc09", "conformer_frozen", "conformer_finetune")
NAMES = dict(imagenet="ImageNet ResNet18", speech_sc09="CLAR English SC09",
             conformer_frozen="Korean Conformer frozen", conformer_finetune="Korean Conformer fine-tuned")


def replay_checkpoints(protocol):
    """Reconstruct the actual network and independently repeat best-model inference."""
    from src.config import Config
    from src.dataset import read_metadata
    from src.engine import evaluate_model, seed_all
    from experiments.conformer.model import ConformerClassifier, load_source
    from experiments.conformer.run import Clips
    from torch.utils.data import DataLoader

    torch.set_num_threads(2)
    cfg = Config(**protocol["reference_config"])
    frame = read_metadata(cfg)
    loaders = {split:DataLoader(Clips(frame[frame.split==split],cfg),batch_size=32,shuffle=False)
               for split in ("train","val")}
    state,statistics = load_source(ROOT/"runs/conformer-research")
    replays = []
    for kind in KINDS[2:]:
        for seed in (42,43,44):
            seed_all(seed)
            run=STUDY/"runs"/f"{kind}_seed{seed}"
            model=ConformerClassifier(state,statistics,mode=kind.removeprefix("conformer_")).cuda()
            checkpoint=torch.load(run/"best.pt",map_location="cpu",weights_only=True)
            model.load_state_dict(checkpoint["model"],strict=True)
            result=json.loads((run/"comparison_summary.json").read_text())
            for split,relative,key in (("val","validation/predictions.csv","validation"),
                                       ("train","original_train_predictions.csv","original_train")):
                scores,predictions=evaluate_model(model,loaders[split],"cuda",len(cfg.labels))
                saved=pd.read_csv(run/relative).set_index("path")
                actual=predictions.set_index("path").loc[saved.index]
                assert saved.true_idx.equals(actual.true_idx) and saved.pred_idx.equals(actual.pred_idx)
                assert all(abs(scores[k]-result[key][k])<1e-6 for k in scores)
                replays.append(dict(condition=kind,seed=seed,split=split,n=len(saved),metrics=scores))
            del model,checkpoint
            torch.cuda.empty_cache()
    audit=dict(replayed_checkpoints=6,replayed_train_and_validation_predictions=sum(r["n"] for r in replays),
               predictions_identical=True,validation_speaker="spk05",test_evaluated=False,
               cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
               cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
               details=replays)
    write_json(REPORT/"checkpoint_replay.json",audit)
    return audit


def paired_table(frame, labels, data_root):
    lines = ["| 파일 | 정답 | ImageNet | Conformer 고정 | Conformer 미세조정 |",
             "| --- | --- | --- | --- | --- |"]
    for row in frame.itertuples():
        path = Path(data_root) / row.path
        lines.append(f"| [{Path(row.path).name}]({path}) | {labels[row.true_idx]} | "
                     f"{labels[row.imagenet]} | {labels[row.conformer_frozen]} | {labels[row.conformer_finetune]} |")
    return lines


def main():
    complete = json.loads((STUDY / "complete.json").read_text())
    assert complete["complete"] and complete["runs"] == 6 and complete["epochs"] == 60
    protocol = json.loads((STUDY / "protocol.json").read_text())
    assert all(file_hash(ROOT/p) == digest for p,digest in protocol["source_sha256"].items())
    base_protocol = json.loads((BASELINE / "protocol.json").read_text())
    assert protocol["reference_config"] == base_protocol["config"]
    cfg = protocol["reference_config"]
    labels = cfg["labels"]
    audio_hashes = protocol["data"]["audio_sha256"]
    assert all(file_hash(Path(cfg["data_root"])/p)==digest for p,digest in audio_hashes.items())
    from experiments.conformer.model import SOURCE_HASHES
    assert all(file_hash(ROOT/"runs/conformer-research"/p)==digest for p,digest in SOURCE_HASHES.items())
    probe = json.loads((REPORT / "input_probe.json").read_text())
    assert probe["input_shape"] == [2,24000] and probe["encoded_shape"] == [2,38,256]
    assert probe["no_extra_audio_padding"] and probe["one_second_input_accepted"]
    selected, initializations, rows, histories = {}, {}, [], []
    predictions_checked = 0
    validation_paths = None
    for kind in KINDS:
        for seed in (42,43,44):
            directory = STUDY if kind.startswith("conformer") else BASELINE
            run = directory / "runs" / f"{kind}_seed{seed}"
            result = json.loads((run / "comparison_summary.json").read_text())
            assert not result["test_evaluated"]
            saved_cfg = json.loads((run / ("reference_config.json" if kind.startswith("conformer") else "config.json")).read_text())
            assert {k:v for k,v in saved_cfg.items() if k not in ("seed","device")} == {k:v for k,v in cfg.items() if k not in ("seed","device")}
            assert saved_cfg["device"] in ("auto","cuda")
            frame = pd.read_csv(run/"metadata.csv")
            expected = set(frame[frame.split=="val"].path)
            assert len(expected)==203
            if validation_paths is None:
                validation_paths = expected
            assert expected == validation_paths
            if not kind.startswith("conformer"):
                assert json.loads((run/"manifest.json").read_text())["audio_sha256"] == audio_hashes
            history = pd.read_csv(run/"history.csv")
            assert history.epoch.tolist() == list(range(1,11))
            assert int(history.loc[history.val_macro_f1.idxmax(),"epoch"]) == result["best_epoch"]
            for epoch in range(1,11):
                p = pd.read_csv(run/"epochs"/f"epoch_{epoch:02d}.csv")
                assert len(p)==203 and p.path.is_unique and set(p.path)==expected
                score = metrics(p.true_idx,p.pred_idx,len(labels))
                stored = history[history.epoch==epoch].iloc[0]
                assert abs(score["macro_f1"]-stored.val_macro_f1)<1e-12
                assert abs(score["accuracy"]-stored.val_accuracy)<1e-12
                predictions_checked += len(p)
            predictions = pd.read_csv(run/"validation/predictions.csv").set_index("path")
            selected[(kind,seed)] = predictions
            score = metrics(predictions.true_idx,predictions.pred_idx,len(labels))
            assert all(abs(score[k]-result["validation"][k])<1e-12 for k in score)
            epoch_predictions = pd.read_csv(run/"epochs"/f"epoch_{result['best_epoch']:02d}.csv").set_index("path")
            assert predictions.pred_idx.equals(epoch_predictions.loc[predictions.index].pred_idx)
            assert file_hash(run/"best.pt") == result["checkpoint_sha256"]
            checkpoint = torch.load(run/"best.pt",map_location="cpu",weights_only=True)
            assert checkpoint["epoch"] == result["best_epoch"]
            assert checkpoint["validation"] == result["validation"]
            if kind.startswith("conformer"):
                audit = json.loads((run/"initialization.json").read_text())
                initializations[(kind,seed)] = audit
                assert audit["total_parameters"] == 32122699 and audit["copied_encoder_tensors"] == 406
                assert audit["classifier_parameters"] == 2827
                assert checkpoint["model"]["fc.weight"].shape == (11,256)
                assert result["source_normalization_unchanged"]
                encoded = {k:v for k,v in checkpoint["model"].items() if not k.startswith("fc.")}
                final_hash = tensor_hash(encoded)
                assert final_hash == result["encoder_final_sha256"]
                if kind=="conformer_frozen":
                    assert audit["trainable_parameters"] == 2827
                    assert final_hash == audit["encoder_initial_sha256"]
                else:
                    assert audit["trainable_parameters"] == audit["total_parameters"]
                    assert final_hash != audit["encoder_initial_sha256"]
                train_predictions = pd.read_csv(run/"original_train_predictions.csv")
                assert len(train_predictions)==609 and train_predictions.path.is_unique
                assert set(train_predictions.path) == set(frame[frame.split=="train"].path)
                train_score = metrics(train_predictions.true_idx,train_predictions.pred_idx,len(labels))
                assert all(abs(train_score[k]-result["original_train"][k])<1e-12 for k in train_score)
            del checkpoint
            rows.append(dict(initialization=kind,seed=seed,best_epoch=result["best_epoch"],
                             val_macro_f1=result["validation"]["macro_f1"],
                             val_accuracy=result["validation"]["accuracy"], val_loss=result["validation"]["loss"],
                             train_macro_f1=result["original_train"]["macro_f1"],
                             train_val_gap=result["original_train"]["macro_f1"]-result["validation"]["macro_f1"]))
            histories.append(history.assign(initialization=kind,seed=seed))
    for seed in (42,43,44):
        a,b = [initializations[(kind,seed)] for kind in KINDS[2:]]
        assert a["classifier_initial_sha256"] == b["classifier_initial_sha256"]
        assert a["encoder_initial_sha256"] == b["encoder_initial_sha256"]
    replay = replay_checkpoints(protocol)
    runs = pd.DataFrame(rows)
    runs.to_csv(REPORT/"run_summary.csv",index=False)
    columns = ["val_macro_f1","val_accuracy","val_loss","train_macro_f1","train_val_gap"]
    summary = runs.groupby("initialization")[columns].mean().reindex(KINDS)
    summary.to_csv(REPORT/"mean_summary.csv")
    history = pd.concat(histories,ignore_index=True)
    history.to_csv(REPORT/"all_epoch_history.csv",index=False)
    mean_history = history.groupby(["initialization","epoch"])[["train_loss","val_macro_f1","val_accuracy","val_loss"]].mean()
    mean_history.to_csv(REPORT/"mean_epoch_history.csv")
    common_epochs = {kind:int(mean_history.loc[kind].val_macro_f1.idxmax()) for kind in KINDS}
    paired_rows = []
    for seed in (42,43,44):
        base = selected[("imagenet",seed)]
        paired = base[["true_idx"]].copy()
        for kind in KINDS:
            p = selected[(kind,seed)].loc[base.index]
            assert p.true_idx.equals(base.true_idx)
            paired[kind] = p.pred_idx
            paired[kind+"_correct"] = p.true_idx == p.pred_idx
        paired = paired.reset_index()
        paired.to_csv(REPORT/f"seed{seed}_paired_predictions.csv",index=False,encoding="utf-8-sig")
        for kind in KINDS[2:]:
            gains = paired[~paired.imagenet_correct & paired[kind+"_correct"]]
            losses = paired[paired.imagenet_correct & ~paired[kind+"_correct"]]
            paired_rows.append(dict(seed=seed,condition=kind,conformer_only_correct=len(gains),
                                    imagenet_only_correct=len(losses),
                                    imagenet_correct=int(paired.imagenet_correct.sum()),
                                    conformer_correct=int(paired[kind+"_correct"].sum())))
        different = paired[(paired.imagenet != paired.conformer_frozen) | (paired.imagenet != paired.conformer_finetune)]
        lines = [f"# seed{seed}: ImageNet와 한국어 Conformer 예측 차이", "",
                 "각 실행의 검증 F1 최고 checkpoint를 같은203개 파일에서 비교했다.", ""]
        lines += paired_table(different,labels,cfg["data_root"])
        (REPORT/f"seed{seed}_files.md").write_text("\n".join(lines)+"\n")
    pd.DataFrame(paired_rows).to_csv(REPORT/"paired_summary.csv",index=False)
    fig,axes = plt.subplots(1,3,figsize=(14,4))
    colors = ("#527ea6","#c6a24a","#32a17b","#b85a91")
    for kind,color in zip(KINDS,colors):
        for ax,key,factor in zip(axes,("train_loss","val_loss","val_macro_f1"),(1,1,100)):
            h=history[history.initialization==kind].pivot(index="epoch",columns="seed",values=key)*factor
            avg,sd=h.mean(axis=1),h.std(axis=1,ddof=1)
            ax.plot(h.index,avg,"o-",color=color,label=NAMES[kind],markersize=3)
            ax.fill_between(h.index,avg-sd,avg+sd,color=color,alpha=.12)
    for ax,title in zip(axes,("Training loss","Validation loss","Validation macro F1 (%)")):
        ax.set(title=title,xlabel="Epoch",xticks=range(1,11))
        ax.grid(alpha=.2)
    axes[2].set_ylim(0,100)
    axes[0].legend(fontsize=7)
    fig.suptitle("Ref clips and speaker split; 3 seeds; native features per encoder; shading: seed SD")
    fig.tight_layout(rect=(0,0,1,.94))
    fig.savefig(REPORT/"learning_curves.png",dpi=180)
    plt.close(fig)
    best_kind=max(KINDS[2:],key=lambda k:summary.loc[k,"val_macro_f1"])
    delta=(summary.loc[best_kind,"val_macro_f1"]-summary.loc["imagenet","val_macro_f1"])*100
    lines = ["# ref 기준: 한국어 Conformer 전이 실험", "",
        "ref fixed-1p5-v1의 원래16kHz·1.5초 음성과 train609(spk02/03/04) / validation203(spk05)를 사용했다. "
        "test203(spk01)은 추론하지 않았다. 보조 특징·증강 없이3시드42/43/44, 각10epoch다.", "",
        "## 구조와 실제 입력", "",
        "[공개 한국어 모델](https://huggingface.co/speechbrain/asr-conformer-transformerlm-ksponspeech)의 "
        "고정 버전 `ecb74648052c89618cd1ce53ccbc909d1141f164`다. KsponSpeech965.2시간으로 학습된 "
        "CNN 앞단·12층 Conformer 인코더를 사용하고 ASR 디코더·언어 모델 대신 평균 풀링과 새256→11층을 붙였다. "
        "본체의406개 텐서를 모두 strict load했다. 전체32,122,699개 파라미터이며 동결 조건은2,827개만 학습한다.", "",
        "원래80-bin Fbank·FFT400·25ms창·10mshop·f_min0과 사전학습 global 정규화 통계를 사용한다. "
        "타깃 데이터로 정규화 통계를 갱신하지 않는다. 실제 `[2,24000]` 입력은 "
        "`[2,151,80]` 특징 → `[2,38,256]` 인코더 출력 → `[2,11]` logits가 된다. "
        "추가 오디오 패딩은0이며1초 입력도 통과했다.", "",
        "## 학습 조건", "",
        "- 고정: CNN·인코더 eval+동결, 분류층 LR5e-4.",
        "- 미세조정: CNN·인코더 LR1e-5, 분류층 LR5e-4, 원 모델 Dropout0.1 활성.",
        "- 공통: AdamW WD1e-4, cosine10epoch, batch32, 역빈도 weighted replacement sampler로 epoch당609개.",
        "- 같은 시드 두 조건의 초기 인코더와 분류층은 동일하다. 각 실행의 검증 macro-F1 최고 epoch를 선택한다.",
        "- ImageNet와 CLAR 값은 이전 `speech-pretrain-ref-v1`에서 같은 데이터·분할·3시드로 측정한 대조군이다.", "",
        "## 실행별 최고 checkpoint의3시드 평균", "",
        "| 모델 | 검증 F1 | 정확도 | 학습 원본 F1 | 학습–검증 F1 격차 |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for kind in KINDS:
        r=summary.loc[kind]
        lines.append(f"| {NAMES[kind]} | {r.val_macro_f1*100:.2f}% | {r.val_accuracy*100:.2f}% | "
                     f"{r.train_macro_f1*100:.2f}% | {r.train_val_gap*100:.2f}%p |")
    lines += ["",f"Conformer 중 평균 최고 조건은 **{NAMES[best_kind]}**, ImageNet 대비 **{delta:+.2f}%p**다.","",
              "| 시드 | ImageNet | Conformer 고정 | Conformer 미세조정 |", "| --- | ---: | ---: | ---: |"]
    for seed in (42,43,44):
        parts=[]
        for kind in ("imagenet",*KINDS[2:]):
            row=runs[(runs.initialization==kind)&(runs.seed==seed)].iloc[0]
            parts.append(f"{row.val_macro_f1*100:.2f}% (e{int(row.best_epoch)})")
        lines.append(f"| [{seed}](seed{seed}_files.md) | "+" | ".join(parts)+" |")
    lines += ["", "## 3시드 평균에서 공통 epoch 하나를 선택", "",
              "| 모델 | 공통 epoch | 평균 검증 F1 |", "| --- | ---: | ---: |"]
    for kind in KINDS:
        e=common_epochs[kind]
        lines.append(f"| {NAMES[kind]} | {e} | {mean_history.loc[(kind,e),'val_macro_f1']*100:.2f}% |")
    lines += ["", "## 해석 범위", "",
        "- 본체 구조·모델 크기·원래 입력 특징·사전학습 언어·미세조정 학습률이 함께 바뀐 시스템 비교다. "
        "차이를 한국어라는 조건 하나의 효과로 해석할 수 없다.",
        "- 단일 검증 화자203개를 조건·epoch 선택에 사용한 개발 결과다.3시드는 같은 파일을 반복하며 독립 화자3명이 아니다. "
        "이 결과만으로 새로운 화자 전체에 같은 성능을 보장할 수 없다.",
        "- 고정 조건에서는 인코더 Dropout이 비활성이고 미세조정에서는 활성이다. 두 조건의 차이는 학습률 하나만의 효과가 아니다.",
        "- 기존 ref의 원래 단일seed42 F1 92.58%와 이번 대조군3시드 평균89.26%는 집계가 다르다.", "",
        "- SpeechBrain1.0.3은 CUDA matmul TF32를 활성화한다. 기존 대조군은 PyTorch 기본 matmul TF32 off였다. "
        "양쪽 CuDNN TF32는 on이며 모델 교체 외 수치 설정도 완전히 같은 통제는 아니다.", "",
        "## 검증 및 산출물", "",
        f"새6회·60epoch와 기존6회·60epoch, 총{predictions_checked:,}개 epoch 예측 행의 지표를 다시 계산했다. "
        "원본1,015개 해시·선택 checkpoint 해시·동결 본체 보존·정규화 통계 보존·초기 분류층 일치를 확인했다. "
        f"새6개 최고 checkpoint를 실제 모델로 재구성해 학습·검증 {replay['replayed_train_and_validation_predictions']:,}개 예측을 재추론했고 모두 일치했다.", "",
        "[프로토콜](protocol.json), [실제 입력 확인](input_probe.json), [실행별 요약](run_summary.csv), "
        "[짝 비교](paired_summary.csv), [학습 기록](all_epoch_history.csv), "
        "[체크포인트 재추론](checkpoint_replay.json), [검증 기록](execution_audit.json).", "",
        "![학습곡선](learning_curves.png)", ""]
    (REPORT/"README.md").write_text("\n".join(lines))
    artifacts=sorted(p for p in REPORT.rglob("*") if p.is_file() and p.name!="execution_audit.json")
    write_json(REPORT/"execution_audit.json",dict(new_runs=6,new_training_epochs=60,control_runs=6,
        epoch_prediction_rows_checked=predictions_checked,all_epoch_prediction_metrics_reproduced=True,
        same_reference_files_and_split=True,source_hashes_unchanged=True,normalization_fixed=True,
        same_initial_weights_between_adaptation_modes=True,frozen_encoder_unchanged=True,
        one_second_and_one_point_five_second_inputs_verified=True,test_evaluated=False,
        best_checkpoint_replay_predictions_checked=replay["replayed_train_and_validation_predictions"],
        artifact_sha256={str(p.relative_to(ROOT)):file_hash(p) for p in artifacts}))
    print(summary.to_string())
    print("COMMON_EPOCHS",common_epochs)
    print(pd.DataFrame(paired_rows).to_string(index=False))


if __name__ == "__main__":
    main()
