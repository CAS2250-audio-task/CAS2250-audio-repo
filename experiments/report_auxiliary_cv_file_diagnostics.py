"""List per-file epoch regressions from verified diagnostic replays."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from src.engine import metrics, write_json

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/auxiliary-cv-v1/file_diagnostics"
STUDY = ROOT / "runs/auxiliary-cv-v1"
DATA = ROOT / "datasets/task2/real-context-1p5-v1"
SET_ID = "3"


def table_files(frame, include_source=False):
    columns = ["정답", "파일", "epoch10 예측"]
    if include_source:
        columns += ["원본 파일명"]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"]*len(columns)) + " |"]
    for row in frame.sort_values(["true_label", "path"]).itertuples():
        values = [row.true_label, f"[{Path(row.path).name}]({row.audio_path})", row.pred_e10]
        if include_source:
            values.append(str(row.source_filename).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return lines if len(frame) else ["해당 파일 없음."]


def diagnose(speaker, metadata, recipe, labels):
    folder = "spk01_external_model_spk02_seed44" if speaker == "spk01" else f"c2ed04a871308_set{SET_ID}_seed44_{speaker}"
    directory = REPORT / folder
    frames = {e:pd.read_csv(directory / f"epoch_{e:02d}.csv").set_index("path") for e in range(1,11)}
    paths = frames[3].index
    assert len(paths) == 203 and paths.is_unique
    ledger = pd.DataFrame(index=paths)
    ledger["true_label"] = frames[3].true_label
    ledger["speaker"] = speaker
    ledger = ledger.join(metadata[["source_filename", "source_seconds"]])
    ledger = ledger.join(recipe[["pad_left_samples", "pad_right_samples", "gain_db", "peak_gain_reduced"]])
    ledger["padding_ms"] = (ledger.pad_left_samples + ledger.pad_right_samples)/16
    ledger["audio_path"] = [str(DATA / p) for p in paths]
    assert all(Path(p).exists() for p in ledger.audio_path)
    probability_columns = [f"p_{i}" for i in range(len(labels))]
    histories = []
    for epoch in range(1,11):
        frame = frames[epoch].loc[paths]
        assert frame.true_label.equals(frames[3].true_label)
        computed = metrics(frame.true_idx, frame.pred_idx, len(labels))
        stored = json.loads((directory / f"epoch_{epoch:02d}_metrics.json").read_text())
        assert all(abs(computed[k]-stored[k]) < 1e-12 for k in computed)
        p = frame[probability_columns].to_numpy()[np.arange(len(frame)), frame.true_idx.to_numpy()]
        assert (p > 0).all() and np.isfinite(p).all()
        nll = -np.log(p)
        assert abs(nll.mean()-stored["loss"]) < 1e-5
        ledger[f"pred_e{epoch:02d}"] = frame.pred_label
        ledger[f"correct_e{epoch:02d}"] = frame.true_idx == frame.pred_idx
        ledger[f"true_probability_e{epoch:02d}"] = p
        ledger[f"nll_e{epoch:02d}"] = nll
        histories.append(dict(epoch=epoch,**stored))
    if speaker == "spk01":
        source_run = f"c2ed04a871308_set{SET_ID}_seed44_spk02"
        selected_epoch = json.loads((STUDY / "runs" / source_run / "summary.json").read_text())["best_epoch"]
    else:
        selected_epoch = json.loads((STUDY / "runs" / folder / "summary.json").read_text())["best_epoch"]
        audit = json.loads((directory / "replay_audit.json").read_text())
        assert all(v for k,v in audit.items() if k not in ("speaker", "original_run"))
    ledger["correct_epochs_count"] = ledger[[f"correct_e{e:02d}" for e in range(1,11)]].sum(axis=1)
    ledger["correct_then_wrong_e03_e10"] = ledger.correct_e03 & ~ledger.correct_e10
    ledger["wrong_then_correct_e03_e10"] = ~ledger.correct_e03 & ledger.correct_e10
    ledger["nll_change_e03_e10"] = ledger.nll_e10 - ledger.nll_e03
    ledger["correct_then_wrong_selected_e10"] = ledger[f"correct_e{selected_epoch:02d}"] & ~ledger.correct_e10
    ledger = ledger.reset_index()
    lost = ledger[ledger.correct_then_wrong_e03_e10]
    recovered = ledger[ledger.wrong_then_correct_e03_e10]
    selected_lost = ledger[ledger.correct_then_wrong_selected_e10]
    for suffix, frame in [("all_files", ledger), ("regressed_e03_e10", lost),
                          ("recovered_e03_e10", recovered), ("regressed_selected_e10", selected_lost),
                          ("loss_worsened_e03_e10", ledger[ledger.nll_change_e03_e10>1e-6])]:
        frame.to_csv(REPORT / f"{speaker}_{suffix}.csv", index=False, encoding="utf-8-sig")
    classes = []
    for epoch in [3,10]:
        frame = frames[epoch]
        result = classification_report(frame.true_idx, frame.pred_idx, labels=list(range(len(labels))),
            target_names=labels, output_dict=True, zero_division=0)
        for label in labels:
            classes.append(dict(epoch=epoch,label=label,**result[label]))
    pd.DataFrame(classes).to_csv(REPORT / f"{speaker}_class_metrics.csv",index=False)
    history = pd.DataFrame(histories).set_index("epoch")
    summary = dict(speaker=speaker, comparison_epochs=[3,10], selected_epoch=selected_epoch,
        correct_e03=int(ledger.correct_e03.sum()), correct_e10=int(ledger.correct_e10.sum()),
        regressed=int(len(lost)), recovered=int(len(recovered)),
        loss_worsened_files=int((ledger.nll_change_e03_e10>1e-6).sum()),
        f1_e03=float(history.loc[3,"macro_f1"]), f1_e10=float(history.loc[10,"macro_f1"]),
        loss_e03=float(history.loc[3,"loss"]), loss_e10=float(history.loc[10,"loss"]),
        regressed_selected_to10=int(len(selected_lost)),
        mean_padding_ms_regressed=float(lost.padding_ms.mean()),
        zero_padding_files_regressed=int((lost.padding_ms>0).sum()),
        evaluation="external_spk01_with_spk02_fold_model" if speaker=="spk01" else "own_speaker_held_out_fold")
    assert summary["correct_e03"]-summary["regressed"]+summary["recovered"] == summary["correct_e10"]
    if speaker == "spk01":
        scope = ("spk02 검증 fold의 모델(학습 spk03/04/05)을 spk01에 별도 평가했다. "
                 f"선택 epoch{selected_epoch}은 원래 spk02 기준이며 spk01 결과로 고른 것이 아니다. "
                 "주 비교 epoch3→10은 두 모델 모두 고정했다.")
    else:
        scope = f"{speaker}를 학습에서 제외한 고유 CV fold의 seed44 실행이다. 원래 결과와 학습 history·최적 가중치 텐서·최적 예측이 동일한 재현을 확인했다."
    lines = [f"# {speaker} 파일별 진단", "", scope, "",
        "기준은 모든 화자에 동일한 **epoch3 정답→epoch10 오답**이다. 파일이 원래부터 불량했다는 판정이 아니며, 같은 파일 입력에 대한 모델 예측이 바뀐 경우다.", "",
        f"203개 중 정답 {summary['correct_e03']}→{summary['correct_e10']}, 악화 {len(lost)}개 / 회복 {len(recovered)}개. "
        f"Macro F1 {summary['f1_e03']*100:.2f}%→{summary['f1_e10']*100:.2f}%, "
        f"loss {summary['loss_e03']:.4f}→{summary['loss_e10']:.4f}.", "",
        f"[전체 203개·epoch별 예측·확률·전처리](./{speaker}_all_files.csv) · "
        f"[악화 목록 CSV](./{speaker}_regressed_e03_e10.csv) · [회복 목록 CSV](./{speaker}_recovered_e03_e10.csv) · "
        f"[정답 확률이 낮아져 loss가 증가한 목록](./{speaker}_loss_worsened_e03_e10.csv)", "",
        "## epoch3 정답→epoch10 오답 전체 파일", ""] + table_files(lost, include_source=True)
    lines += ["", f"## 원래 선택 epoch{selected_epoch}→10의 악화", "",
        f"원래 선택 시점부터 비교하면 정답→오답은 {len(selected_lost)}개다. "
        "spk01은 고정된 원래 모델의 선택 시점을 썼다. 화자마다 선택 epoch가 달라 이 개수로 화자를 직접 순위 매기지 않는다.", "",
        f"[이 기준의 전체 목록 CSV](./{speaker}_regressed_selected_e10.csv)", ""]
    if selected_epoch != 3:
        lines += table_files(selected_lost, include_source=True) + [""]
    lines += ["## 원인 해석 범위", "",
        f"epoch3→10 악화 파일 {len(lost)}개 중 원본 부족분의 0 패딩이 있는 파일은 {summary['zero_padding_files_regressed']}개다. "
        "추론 입력은 epoch마다 동일하다. 패딩·원본 길이·gain과 파일명 태그는 관찰 자료이며 오류 원인을 증명하지 않는다.", "",
        "확률은 모델의 예측값이며 음성의 품질 점수가 아니다. 사람의 청취 검수는 하지 않았다. "
        "같은 화자의 203개와 한 seed의 진단이므로 화자 전체의 성능으로 일반화하지 않는다.", ""]
    (REPORT / f"{speaker}.md").write_text("\n".join(lines))
    return summary, history, ledger


def main():
    assert json.loads((REPORT / "complete.json").read_text())["complete"]
    metadata = pd.read_csv(DATA / "metadata.csv", keep_default_na=False).set_index("path")
    recipe = pd.read_csv(DATA / "preprocessing_recipe.csv").set_index("path")
    labels = json.loads((STUDY / "protocol.json").read_text())["config"]["labels"]
    rows, histories, ledgers = [], {}, {}
    for speaker in ("spk01", "spk02", "spk03", "spk04"):
        result, history, ledger = diagnose(speaker, metadata, recipe, labels)
        rows.append(result);histories[speaker]=history;ledgers[speaker]=ledger
    pd.DataFrame(rows).to_csv(REPORT / "speaker_summary.csv",index=False)
    sp2 = ledgers["spk02"].copy()
    # Preserve exact filename tags; do not invent room or distance metadata.
    def tag(row):
        stem = Path(row.source_filename).stem
        i = stem.find(row.true_label)
        return stem[:i] if i>=0 and stem[:i] else "(prefix 없음)"
    sp2["filename_prefix"] = sp2.apply(tag, axis=1)
    sp2["filename_distance_tag"] = sp2.source_filename.str.extract(r"(\d+\s*cm)",expand=False).fillna("미기록")
    tags=[]
    for field in ["filename_prefix", "filename_distance_tag"]:
        for key,a in sp2[sp2.true_label!="silence"].groupby(field):
            tags.append(dict(field=field,tag=key,total_files=len(a),correct_e03=int(a.correct_e03.sum()),
                regressed=int(a.correct_then_wrong_e03_e10.sum()),recovered=int(a.wrong_then_correct_e03_e10.sum())))
    pd.DataFrame(tags).to_csv(REPORT / "spk02_filename_tags.csv",index=False)
    extra = ["", "## 파일명 태그로 본 분포", "",
        "아래 분모는 해당 태그의 epoch3 정답 파일 수이며, 분자는 그중 epoch10에 틀린 수다. "
        "원본 파일명에서 읽은 태그이고 환경·거리의 실제 측정이나 원인 검증은 아니다. 단어·다른 조건과의 혼재가 남는다.", "",
        "| 파일명 항목 | 태그 | epoch3 정답 | 그중 악화 | 비율 |", "| --- | --- | ---: | ---: | ---: |"]
    for row in tags:
        extra.append(f"| {row['field']} | {row['tag']} | {row['correct_e03']} | {row['regressed']} | "
            f"{100*row['regressed']/row['correct_e03']:.1f}% |")
    extra += ["", "## 단어별 정답 수 변화", "", "| 단어 | epoch3 정답 /20 | epoch10 정답 /20 |",
              "| --- | ---: | ---: |"]
    for label in labels:
        if label == "silence":
            continue
        group = sp2[sp2.true_label==label]
        extra.append(f"| {label} | {int(group.correct_e03.sum())} | {int(group.correct_e10.sum())} |")
    with (REPORT / "spk02.md").open("a") as handle:
        handle.write("\n".join(extra)+"\n")
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True,sharey=True)
    for ax,(speaker,h) in zip(axes.flat,histories.items()):
        ax.plot(h.index,h.macro_f1*100,"o-",markersize=4)
        ax.axvline(3,color="#dc8b35",linestyle="--",alpha=.7)
        ax.set(title=speaker+(" (external; spk02-fold model)" if speaker=="spk01" else " (own held-out fold)"),
            xlabel="Epoch",ylabel="Validation/diagnostic macro F1 (%)",ylim=(0,100),xticks=range(1,11))
        ax.grid(alpha=.2)
    fig.suptitle("Final set 3, seed 44: speaker-specific epoch curves")
    fig.text(.5,.015,"spk01 is a separate post-selection evaluation. No model was trained on its clips or selected using its scores.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.035,1,.96));fig.savefig(REPORT / "speaker_epoch_curves.png",dpi=180);plt.close(fig)
    lines=["# 최종 세트3: 화자별 악화 파일 조사", "",
        "범위: seed44, 최종 공통 조건 mask+speed, 세트3. 비교 시점은 모두 epoch3→10이다. "
        "spk02/03/04의 재현은 원래 학습 history·최적 모델 텐서·최적 예측과 모두 일치했다.", "",
        "spk01은 사용자의 별도 조사 요청에 따라 spk02 fold 모델(학습 spk03/04/05)로 평가했다. "
        "기존 184회 CV와 최종 조건 선택은 변경하지 않았다. 이번 spk01 분석을 기존 CV 결과나 새로운 미공개 test 점수로 섞지 않는다.", "",
        "| 화자 | 평가 방식 | F1 epoch3→10 | 정답 수 /203 | 정답→오답 | 오답→정답 |",
        "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        s=row["speaker"];scope="spk02 모델의 별도 외부 평가" if s=="spk01" else "각자 미학습 화자로 둔 CV"
        lines.append(f"| [{s}]({s}.md) | {scope} | {row['f1_e03']*100:.2f}%→{row['f1_e10']*100:.2f}% | "
            f"{row['correct_e03']}→{row['correct_e10']} | {row['regressed']} | {row['recovered']} |")
    lines += ["", "spk01/02는 이 비교에서 전체 점수가 내려갔다. spk03/04는 일부 악화 파일이 있어도 회복 파일이 더 많아 전체 점수는 올라갔다. "
        "동일한 후반 악화가 모든 화자에게 나타난 것은 아니다.", "",
        "## 바로 열 파일 목록", "",
        "- [spk02 악화 43개 전체·음성 링크](spk02.md)",
        "- [spk01 별도 진단](spk01.md)", "- [spk03 진단](spk03.md)", "- [spk04 진단](spk04.md)",
        "- [화자별 요약 CSV](speaker_summary.csv)", "",
        "각 목록에는 정답·후반 예측·실제 WAV 링크·원본 파일명이 있다. 전체 CSV에는 10개 epoch의 예측과 정답 확률, 원본 길이·패딩·gain을 포함한다.", "",
        "원래 최고 epoch부터 끝까지의 악화도 별도 목록으로 제공한다. spk03은 epoch7→10에 7개, "
        "spk04는 epoch9→10에 4개이며, epoch3→10 목록과 기준이 다르다.", "",
        "## 해석과 검증", "",
        "- 정답→오답은 모델의 예측 변화다. 그 파일이 불량 녹음이거나 라벨이 틀렸다는 뜻이 아니다. 청취 기반 원인 판정은 하지 않았다.",
        "- 정답 수 감소는 정답→오답에서 오답→정답을 뺀 값과 일치한다. Macro F1은 파일별로 단순 가산되지 않으므로 파일 수만으로 F1 하락을 분해하지 않는다.",
        "- 매 epoch의 CSV에서 정확도·macro F1과 정답 확률의 NLL을 다시 계산해 저장 지표와 비교했다.",
        "- 진단용 재현 30 epochs는 원래 정식 실험 184회에 추가해 조건을 다시 선택한 실험이 아니다.", "",
        "![화자별 곡선](speaker_epoch_curves.png)", ""]
    (REPORT / "README.md").write_text("\n".join(lines))
    write_json(REPORT / "analysis_audit.json",dict(speakers=4,clips_per_speaker=203,epochs_per_speaker=10,
        prediction_rows_checked=8120,all_prediction_metrics_reproduced=True,replays_bit_identical=3,
        same_epoch_comparison=[3,10],original_selection_unchanged=True,
        user_authorized_spk01_post_selection_evaluation=True,human_listening_performed=False))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
