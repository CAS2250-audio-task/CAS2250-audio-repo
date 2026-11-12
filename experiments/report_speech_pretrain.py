"""Audit and report the paired speech-pretraining experiment on ref data."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.engine import file_hash, metrics, write_json

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "runs/speech-pretrain-ref-v1"
REPORT = ROOT / "reports/speech-pretrain-ref-v1"
KINDS = ("imagenet", "speech_sc09")


def table_files(frame):
    lines = ["| 정답 | 파일 | ImageNet 예측 | 음성 사전학습 예측 |", "| --- | --- | --- | --- |"]
    for row in frame.sort_values(["true_label", "path"]).itertuples():
        lines.append(f"| {row.true_label} | [{Path(row.path).name}]({row.audio_path}) | "
                     f"{row.imagenet_prediction} | {row.speech_prediction} |")
    return lines if len(frame) else ["해당 파일 없음."]


def main():
    complete = json.loads((STUDY / "complete.json").read_text())
    assert complete["complete"] and not complete["test_evaluated"]
    protocol = json.loads((STUDY / "protocol.json").read_text())
    assert all(file_hash(ROOT / p) == digest for p, digest in protocol["source_sha256"].items())
    rows, histories, configs, audio_hashes = [], [], [], []
    selected_predictions, initializations = {}, {}
    for seed in (42, 43, 44):
        for kind in KINDS:
            run = STUDY / "runs" / f"{kind}_seed{seed}"
            result = json.loads((run / "comparison_summary.json").read_text())
            assert not result["test_evaluated"]
            cfg = json.loads((run / "config.json").read_text())
            configs.append({k: v for k, v in cfg.items() if k != "seed"})
            manifest = json.loads((run / "manifest.json").read_text())
            audio_hashes.append(manifest["audio_sha256"])
            history = pd.read_csv(run / "history.csv")
            assert history.epoch.tolist() == list(range(1, 11))
            assert int(history.loc[history.val_macro_f1.idxmax(), "epoch"]) == result["best_epoch"]
            for epoch in range(1, 11):
                f = pd.read_csv(run / "epochs" / f"epoch_{epoch:02d}.csv")
                assert len(f) == 203 and f.path.is_unique
                score = metrics(f.true_idx, f.pred_idx, len(cfg["labels"]))
                stored = history[history.epoch == epoch].iloc[0]
                assert abs(score["accuracy"]-stored.val_accuracy) < 1e-12
                assert abs(score["macro_f1"]-stored.val_macro_f1) < 1e-12
                assert set(f.path) == set(pd.read_csv(run / "metadata.csv").query("split == 'val'").path)
            pred = pd.read_csv(run / "validation/predictions.csv").set_index("path")
            score = metrics(pred.true_idx, pred.pred_idx, len(cfg["labels"]))
            assert all(abs(score[k]-result["validation"][k]) < 1e-12 for k in score)
            assert file_hash(run / "best.pt") == result["checkpoint_sha256"]
            checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
            assert checkpoint["epoch"] == result["best_epoch"]
            assert checkpoint["validation"] == result["validation"]
            assert checkpoint["model"]["fc.weight"].shape == (11, 512)
            initializations[(kind, seed)] = json.loads((run / "initialization.json").read_text())
            if kind == "speech_sc09":
                assert initializations[(kind, seed)]["copied_backbone_tensors"] == 120
                assert not initializations[(kind, seed)]["classifier_loaded_from_pretraining"]
            selected_predictions[(kind, seed)] = pred
            rows.append(dict(initialization=kind, seed=seed, best_epoch=result["best_epoch"],
                val_macro_f1=result["validation"]["macro_f1"], val_accuracy=result["validation"]["accuracy"],
                val_loss=result["validation"]["loss"], train_macro_f1=result["original_train"]["macro_f1"],
                train_val_gap=result["original_train"]["macro_f1"]-result["validation"]["macro_f1"]))
            histories.append(history.assign(initialization=kind, seed=seed))
    assert all(c == configs[0] for c in configs)
    assert all(h == audio_hashes[0] for h in audio_hashes)
    assert all(file_hash(Path(configs[0]["data_root"]) / path) == digest for path, digest in audio_hashes[0].items())
    for seed in (42, 43, 44):
        assert initializations[(KINDS[0], seed)]["classifier_initial_sha256"] == initializations[(KINDS[1], seed)]["classifier_initial_sha256"]
        assert initializations[(KINDS[0], seed)]["parameters"] == initializations[(KINDS[1], seed)]["parameters"] == 11175883
    runs = pd.DataFrame(rows)
    runs.to_csv(REPORT / "run_summary.csv", index=False)
    summary = runs.groupby("initialization")[["val_macro_f1", "val_accuracy", "val_loss", "train_macro_f1", "train_val_gap"]].mean()
    summary.to_csv(REPORT / "mean_summary.csv")
    history = pd.concat(histories, ignore_index=True)
    history.to_csv(REPORT / "all_epoch_history.csv", index=False)
    mean_history = history.groupby(["initialization", "epoch"])[["train_loss", "val_macro_f1", "val_accuracy", "val_loss"]].mean()
    mean_history.to_csv(REPORT / "mean_epoch_history.csv")
    common = {kind: int(mean_history.loc[kind].val_macro_f1.idxmax()) for kind in KINDS}
    paired_rows = []
    for seed in (42, 43, 44):
        a, b = [selected_predictions[(kind, seed)] for kind in KINDS]
        b = b.loc[a.index]
        assert a.true_idx.equals(b.true_idx)
        f = a[["true_label"]].copy()
        f["imagenet_prediction"], f["speech_prediction"] = a.pred_label, b.pred_label
        f["imagenet_correct"], f["speech_correct"] = a.true_idx == a.pred_idx, b.true_idx == b.pred_idx
        f["audio_path"] = [str(Path(configs[0]["data_root"]) / p) for p in f.index]
        f = f.reset_index()
        gains, losses = f[~f.imagenet_correct & f.speech_correct], f[f.imagenet_correct & ~f.speech_correct]
        f.to_csv(REPORT / f"seed{seed}_paired_predictions.csv", index=False, encoding="utf-8-sig")
        gains.to_csv(REPORT / f"seed{seed}_speech_only_correct.csv", index=False, encoding="utf-8-sig")
        losses.to_csv(REPORT / f"seed{seed}_imagenet_only_correct.csv", index=False, encoding="utf-8-sig")
        paired_rows.append(dict(seed=seed, speech_only_correct=len(gains), imagenet_only_correct=len(losses),
                                imagenet_correct=int(f.imagenet_correct.sum()), speech_correct=int(f.speech_correct.sum())))
        lines = [f"# seed{seed}: ref 조건의 사전학습 비교", "",
            f"같은203개 검증 파일에서 음성 사전학습만 정답 {len(gains)}개, ImageNet만 정답 {len(losses)}개다.", "",
            "## 음성 사전학습만 맞힌 파일", ""] + table_files(gains)
        lines += ["", "## ImageNet만 맞힌 파일", ""] + table_files(losses)
        (REPORT / f"seed{seed}_files.md").write_text("\n".join(lines)+"\n")
    pd.DataFrame(paired_rows).to_csv(REPORT / "paired_summary.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for kind, color, label in (("imagenet", "#527ea6", "ImageNet"), ("speech_sc09", "#dc7832", "Speech Commands")):
        for ax, key, factor in ((axes[0], "train_loss", 1), (axes[1], "val_loss", 1), (axes[2], "val_macro_f1", 100)):
            h = history[history.initialization == kind].pivot(index="epoch", columns="seed", values=key)*factor
            avg, sd = h.mean(axis=1), h.std(axis=1, ddof=1)
            ax.plot(h.index, avg, "o-", color=color, label=label, markersize=3)
            ax.fill_between(h.index, avg-sd, avg+sd, color=color, alpha=.12)
    for ax, title in zip(axes, ("Training loss", "Validation loss", "Validation macro F1 (%)")):
        ax.set(title=title, xlabel="Epoch", xticks=range(1, 11))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[2].set_ylim(0, 100)
    fig.suptitle("Ref fixed-1.5s data, ResNet18, identical training; shading: SD over 3 seeds")
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(REPORT / "learning_curves.png", dpi=180)
    plt.close(fig)
    lines = ["# ref 기준: 스펙트로그램 음성 사전학습 비교", "",
        "브랜치 `pretrain`은 `origin/reference/ai-baseline`의 `6d208bc`에서 시작했다. "
        "`jaehun`의 새 데이터·보조 특징·CV 설정을 가져오지 않았다. "
        "ref의 fixed-1p5-v1 데이터, train609(spk02/03/04) / validation203(spk05), "
        "16kHz·1.5초·128-bin log-mel·ResNet18·512→11 분류층·10epoch·무증강·LR5e-4·WD1e-4를 유지했다.", "",
        "## 사용한 음성 사전학습", "",
        "[CLAR 공식 코드](https://github.com/haideraltahan/CLAR)의 공개 `sp_2d_100_fd_tm/sc09/epoch=1000.ckpt`다. "
        "Speech Commands SC09의 짧은 영어 숫자 발화를 분류한 supervised 2D ResNet18이다. "
        "메타데이터의 저장 epoch는1002다. 한국어 ASR나 원시 파형 Transformer를 사용한 실험은 아니다.", "",
        "원래 입력은 STFT magnitude(dB)·STFT phase·log-mel의3채널이다. "
        "ref 입력을 유지하기 위해 첫 conv에서 log-mel인 채널2의 가중치를 선택했고, "
        "본체의120개 가중치/BN 통계 텐서를 모두 옮겼다. 원래 분류층은 제외하고 ref의512→11층을 새로 학습했다. "
        "같은 시드의 두 모델은 새 분류층 초기값이 동일하다.", "",
        "## 결과: ref와 같은 실행별 최고 checkpoint 기준", "",
        "| 초기화 | 3시드 평균 검증 F1 | 평균 정확도 | 학습 원본 F1 | 학습–검증 F1 격차 |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for kind in KINDS:
        r = summary.loc[kind]
        lines.append(f"| {kind} | {r.val_macro_f1*100:.2f}% | {r.val_accuracy*100:.2f}% | "
                     f"{r.train_macro_f1*100:.2f}% | {r.train_val_gap*100:.2f}%p |")
    delta = (summary.loc["speech_sc09", "val_macro_f1"]-summary.loc["imagenet", "val_macro_f1"])*100
    lines += ["", f"이 전이 방식의 음성 사전학습은 ImageNet 대비 평균 F1 **{delta:+.2f}%p**였다. "
        "현재 ref 설정에서 개선을 확인하지 못했으므로 기존 ImageNet 기준을 유지한다.", "",
        "| 시드 | ImageNet 최고 epoch / F1 | 음성 최고 epoch / F1 | 차이 |",
        "| --- | ---: | ---: | ---: |"]
    for seed in (42, 43, 44):
        a, b = [runs[(runs.seed == seed) & (runs.initialization == kind)].iloc[0] for kind in KINDS]
        lines.append(f"| [{seed}](seed{seed}_files.md) | {a.best_epoch} / {a.val_macro_f1*100:.2f}% | "
                     f"{b.best_epoch} / {b.val_macro_f1*100:.2f}% | {(b.val_macro_f1-a.val_macro_f1)*100:+.2f}%p |")
    lines += ["", "## 전체 평균으로 공통 epoch 하나를 골랐을 때", "",
        "| 초기화 | 공통 epoch | 3시드 평균 검증 F1 |", "| --- | ---: | ---: |"]
    for kind in KINDS:
        lines.append(f"| {kind} | {common[kind]} | {mean_history.loc[(kind, common[kind]), 'val_macro_f1']*100:.2f}% |")
    lines += ["", "각 시드 링크에 두 모델의 다른 예측·음성 파일 링크를 넣었다. "
        "원래 ref의 단일seed42 기록 F1 92.58%와 이번 재학습92.38%는 완전히 같은 재현값은 아니다. "
        "이번 비교에는 같은 환경에서 새로 측정한 ImageNet 대조군을 사용했다.", "",
        "## 해석 범위", "",
        "- 짧은 단어 음성·스펙트로그램·ResNet18이라는 조건은 가깝지만 영어 숫자와 한국어 명령어, "
        "원래3채널 스펙트로그램과 ref의1채널 log-mel은 차이가 있다. 사전학습 도메인·입력 채널 적응·BN 통계 차이가 결과에 영향을 줬을 수 있으며 원인을 분리한 실험은 아니다.",
        "- 음성 사전학습 전체가 ImageNet보다 나쁘다는 결론은 아니다. 선택한 공개 가중치를 ref 입력으로 옮긴 이번10epoch 설정의 결과다.",
        "- test spk01은 추론하지 않았다. 조건·epoch 선택에 쓰인 단일 검증 화자203개의 개발 결과이며 3시드는 같은 파일을 반복한다.", "",
        "## 검증과 재현", "",
        "- 총6회·60epoch. 원래 ref의 `src/`와 기본 설정을 변경하지 않고 별도 runner에서 초기화만 바꿨다.",
        "- 실제 checkpoint·코드·데이터1,015개 해시, 동일 분류층 초기값, 전체 본체120개 텐서, "
        "모델 파라미터 수11,175,883개를 확인했다. 60개epoch의12,180개 검증 예측에서 정확도/F1을 재계산했다.",
        "- 기존 ref 테스트와 전이 검증 테스트 합계32개 통과. 기존 jaehun 브랜치 커밋 전55개 테스트도 통과했다.",
        "- [사전 프로토콜](protocol.json), [공개 모델 출처](pretraining_source.json), "
        "[실행별 요약](run_summary.csv), [전체 학습 기록](all_epoch_history.csv), [짝 비교](paired_summary.csv).", "",
        "```bash", "cd /root/audio/.worktrees/pretrain",
        "OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 /root/audio/.venv/bin/python -m experiments.pretrain.run",
        "/root/audio/.venv/bin/python -m experiments.report_speech_pretrain", "```", "",
        "![학습곡선](learning_curves.png)", ""]
    (REPORT / "README.md").write_text("\n".join(lines))
    artifacts = sorted(p for p in REPORT.rglob("*") if p.is_file() and p.name != "execution_audit.json")
    write_json(REPORT / "execution_audit.json", dict(runs=6, training_epochs=60, tests_passed=32,
        epoch_predictions_checked=12180, all_prediction_metrics_reproduced=True,
        same_reference_configuration=True, same_audio_hashes=True, same_classifier_initialization=True,
        copied_speech_backbone_tensors=120, source_hashes_unchanged=True, test_evaluated=False,
        artifact_sha256={str(p.relative_to(ROOT)): file_hash(p) for p in artifacts}))
    print(summary.to_string())
    print("COMMON_EPOCHS", common)
    print(pd.DataFrame(paired_rows).to_string(index=False))


if __name__ == "__main__":
    main()
