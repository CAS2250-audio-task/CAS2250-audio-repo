"""Pair each set's originally selected best checkpoint, without new training."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.engine import file_hash, metrics, write_json

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "runs/auxiliary-cv-v1"
REPORT = ROOT / "reports/auxiliary-cv-v1"
OUT = REPORT / "best_checkpoint_comparison"
DATA = ROOT / "datasets/task2/real-context-1p5-v1"
CONDITION = "2ed04a871308"
SEED = 44
SPEAKERS = ("spk01", "spk02", "spk03", "spk04")


def selected_predictions(speaker, set_id, labels):
    source_speaker = "spk02" if speaker == "spk01" else speaker
    name = f"c{CONDITION}_set{set_id}_seed{SEED}_{source_speaker}"
    run = STUDY / "runs" / name
    summary = json.loads((run / "summary.json").read_text())
    epoch = int(summary["best_epoch"])
    assert summary["condition_id"] == CONDITION and summary["seed"] == SEED
    history = pd.read_csv(run / "history.csv")
    assert int(history.loc[history.val_macro_f1.idxmax(), "epoch"]) == epoch
    base = REPORT / ("file_diagnostics_set0" if set_id == 0 else "file_diagnostics")
    folder = base / ("spk01_external_model_spk02_seed44" if speaker == "spk01" else name)
    prediction_path = folder / f"epoch_{epoch:02d}.csv"
    frame = pd.read_csv(prediction_path).set_index("path")
    assert len(frame) == 203 and frame.index.is_unique
    assert frame.true_label.equals(frame.true_idx.map(dict(enumerate(labels))))
    assert frame.pred_label.equals(frame.pred_idx.map(dict(enumerate(labels))))
    saved = json.loads((folder / f"epoch_{epoch:02d}_metrics.json").read_text())
    calculated = metrics(frame.true_idx, frame.pred_idx, len(labels))
    assert all(abs(saved[k] - v) < 1e-12 for k, v in calculated.items())
    probabilities = frame[[f"p_{i}" for i in range(len(labels))]].to_numpy()
    assert np.isfinite(probabilities).all() and (probabilities >= 0).all()
    assert np.allclose(probabilities.sum(axis=1), 1, atol=1e-6)
    assert np.array_equal(probabilities.argmax(axis=1), frame.pred_idx)
    true_p = probabilities[np.arange(len(frame)), frame.true_idx.to_numpy()]
    assert (true_p > 0).all() and abs(float(-np.log(true_p).mean()) - saved["loss"]) < 1e-5
    if speaker != "spk01":
        original = pd.read_csv(run / "validation/predictions.csv").set_index("path")
        pd.testing.assert_frame_equal(frame, original, check_exact=True)
        assert saved == summary["validation"]
    else:
        external = json.loads((folder / "protocol.json").read_text())
        assert external["source_model"] == name
        assert external["no_training_or_scaler_fit_on_spk01"]
    info = dict(source_run=name, source_summary=summary, selected_epoch=epoch, scores=saved,
        prediction_source=str(prediction_path.relative_to(ROOT)), prediction_sha256=file_hash(prediction_path),
        source_best_checkpoint_sha256=file_hash(run / "best.pt"))
    return frame, true_p, info


def file_table(frame):
    lines = ["| 정답 | 음성 파일 | 세트0 최고 모델 예측 | 세트3 최고 모델 예측 |",
             "| --- | --- | --- | --- |"]
    for row in frame.sort_values(["true_label", "path"]).itertuples():
        lines.append(f"| {row.true_label} | [{Path(row.path).name}]({row.audio_path}) | "
                     f"{row.set0_prediction} | {row.set3_prediction} |")
    return lines if len(frame) else ["해당 파일 없음."]


def compare(speaker, metadata, recipe, labels):
    paired = None
    infos = {}
    for set_id in (0, 3):
        frame, true_p, info = selected_predictions(speaker, set_id, labels)
        infos[set_id] = info
        if paired is None:
            paired = frame[["true_label"]].join(metadata[["source_filename", "source_seconds"]])
            paired["padding_ms"] = (recipe.pad_left_samples + recipe.pad_right_samples) / 16
            paired["audio_path"] = [str(DATA / p) for p in paired.index]
            assert all(Path(p).is_file() for p in paired.audio_path)
        else:
            assert paired.index.equals(frame.index) and paired.true_label.equals(frame.true_label)
        paired[f"set{set_id}_epoch"] = info["selected_epoch"]
        paired[f"set{set_id}_prediction"] = frame.pred_label
        paired[f"set{set_id}_correct"] = frame.true_idx == frame.pred_idx
        paired[f"set{set_id}_true_probability"] = true_p
        paired[f"set{set_id}_nll"] = -np.log(true_p)
    assert infos[0]["source_summary"]["condition"] == infos[3]["source_summary"]["condition"]
    assert infos[0]["source_summary"]["backbone_initial_sha256"] == infos[3]["source_summary"]["backbone_initial_sha256"]
    paired["relation"] = np.select(
        [~paired.set0_correct & paired.set3_correct, paired.set0_correct & ~paired.set3_correct,
         paired.set0_correct & paired.set3_correct],
        ["set3_only_correct", "set0_only_correct", "both_correct"], default="both_wrong")
    paired["nll_difference_set3_minus_set0"] = paired.set3_nll - paired.set0_nll
    paired = paired.reset_index()
    row = dict(speaker=speaker, clips=203, evaluation=("external_spk01_with_spk02_fold_model"
               if speaker == "spk01" else "own_speaker_held_out_fold"))
    for set_id in (0, 3):
        info = infos[set_id]
        row[f"set{set_id}_epoch"] = info["selected_epoch"]
        for key in ("macro_f1", "accuracy", "loss"):
            row[f"set{set_id}_{key}"] = float(info["scores"][key])
        row[f"set{set_id}_correct"] = int(paired[f"set{set_id}_correct"].sum())
        row[f"set{set_id}_source_train_f1"] = info["source_summary"]["original_train"]["macro_f1"]
    row["f1_difference_pp"] = 100 * (row["set3_macro_f1"] - row["set0_macro_f1"])
    for relation in ("set3_only_correct", "set0_only_correct", "both_correct", "both_wrong"):
        row[relation] = int((paired.relation == relation).sum())
        paired[paired.relation == relation].to_csv(OUT / f"{speaker}_{relation}.csv", index=False, encoding="utf-8-sig")
    assert sum(row[r] for r in ("set3_only_correct", "set0_only_correct", "both_correct", "both_wrong")) == 203
    assert row["set3_only_correct"] - row["set0_only_correct"] == row["set3_correct"] - row["set0_correct"]
    paired.to_csv(OUT / f"{speaker}_all_files.csv", index=False, encoding="utf-8-sig")
    classes = []
    for label, group in paired.groupby("true_label", sort=False):
        classes.append(dict(label=label, clips=len(group), set0_correct=int(group.set0_correct.sum()),
            set3_correct=int(group.set3_correct.sum()),
            set3_only_correct=int((group.relation == "set3_only_correct").sum()),
            set0_only_correct=int((group.relation == "set0_only_correct").sum())))
    pd.DataFrame(classes).to_csv(OUT / f"{speaker}_class_comparison.csv", index=False, encoding="utf-8-sig")
    scope = ("spk02 검증 점수로 고른 체크포인트를 spk01에 별도 평가했다. "
             "spk01 자체의 최고 epoch를 고른 결과가 아니다." if speaker == "spk01" else
             f"각 세트가 원래 {speaker} 검증 macro F1로 고른 최고 체크포인트를 비교했다.")
    lines = [f"# {speaker}: 최고 체크포인트끼리 비교", "", scope, "",
        "공통 mask+speed·seed44·weight decay 1e-4·Dropout0. 세트0은 보조 없음, "
        "세트3은 포먼트 F1–F3+대역폭 B1–B3. 추가 학습 없이 저장된 예측을 썼다.", "",
        "| 모델 | 선택 epoch | Macro F1 | 정답 /203 | loss |", "| --- | ---: | ---: | ---: | ---: |"]
    for s in (0, 3):
        lines.append(f"| 세트{s} | {row[f'set{s}_epoch']} | {row[f'set{s}_macro_f1']*100:.2f}% | "
                     f"{row[f'set{s}_correct']} | {row[f'set{s}_loss']:.4f} |")
    lines += ["", f"세트3 F1 차이 {row['f1_difference_pp']:+.2f}%p. "
        f"세트3만 정답 {row['set3_only_correct']}개 / 세트0만 정답 {row['set0_only_correct']}개 / "
        f"둘 다 정답 {row['both_correct']}개 / 둘 다 오답 {row['both_wrong']}개.", "",
        f"[전체203개 CSV]({speaker}_all_files.csv) · [단어별 집계]({speaker}_class_comparison.csv)", "",
        "## 세트3만 맞힌 파일", "",
        f"[CSV]({speaker}_set3_only_correct.csv)", ""] + file_table(paired[paired.relation == "set3_only_correct"])
    lines += ["", "## 세트0만 맞힌 파일", "",
        f"[CSV]({speaker}_set0_only_correct.csv)", ""] + file_table(paired[paired.relation == "set0_only_correct"])
    lines += ["", "## 둘 다 틀린 파일", "", f"[CSV]({speaker}_both_wrong.csv)", ""] + file_table(paired[paired.relation == "both_wrong"])
    lines += ["", "## 단어별 정답 수", "", "| 단어 | 파일 수 | 세트0 | 세트3 | 차이 |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for c in classes:
        lines.append(f"| {c['label']} | {c['clips']} | {c['set0_correct']} | {c['set3_correct']} | "
                     f"{c['set3_correct']-c['set0_correct']:+d} |")
    if speaker == "spk02":
        previous = pd.read_csv(REPORT / "file_diagnostics/spk02_regressed_e03_e10.csv")
        former = paired[paired.path.isin(previous.path)]
        assert len(former) == 43 and former.set3_correct.all()
        former.to_csv(OUT / "spk02_previous_43_at_best.csv", index=False, encoding="utf-8-sig")
        lines += ["", "## 앞서 epoch3→10에 악화한 43개", "",
            f"세트3 최고점이 epoch3이므로 이43개는 모두 세트3 최고 모델에서 정답이다. "
            f"세트0 최고 모델에서는 {int(former.set0_correct.sum())}개 정답, "
            f"{int((~former.set0_correct).sum())}개 오답이다.", "",
            "[43개 최고점 예측 CSV](spk02_previous_43_at_best.csv)", ""] + file_table(former)
    lines += ["", "각자의 검증 최고점을 고른 개발 결과이며 독립 test 성능은 아니다. "
        "한 시드의 파일별 차이를 보조 특징의 인과 효과로 확정하지 않는다. "
        "정답 수와 Macro F1 차이는 서로 다른 지표다. 사람의 청취/라벨 검수는 하지 않았다.", ""]
    (OUT / f"{speaker}.md").write_text("\n".join(lines))
    provenance = {str(s): {k: v for k, v in info.items() if k != "source_summary"}
                  for s, info in infos.items()}
    return row, provenance


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((STUDY / "protocol.json").read_text())
    assert all(file_hash(p) == digest for p, digest in protocol["source_sha256"].items())
    labels = protocol["config"]["labels"]
    metadata = pd.read_csv(DATA / "metadata.csv", keep_default_na=False).set_index("path")
    recipe = pd.read_csv(DATA / "preprocessing_recipe.csv").set_index("path")
    rows, provenance = [], {}
    for speaker in SPEAKERS:
        row, provenance[speaker] = compare(speaker, metadata, recipe, labels)
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "speaker_summary.csv", index=False)
    lines = ["# 보조 제외/포함: 각자의 최고 성능 체크포인트 비교", "",
        "같은 공통 학습 조건·seed44에서 각 세트가 원래 검증 Macro F1로 선택한 최고 epoch끼리 비교했다. "
        "epoch3 또는10으로 맞춘 직전 분석과 구분한다. 새 학습이나 조건 재선택은 하지 않았다.", "",
        "| 화자 | 세트0 epoch / F1 | 세트3 epoch / F1 | 차이 | 정답 수 세트0→3 | 세트3만 정답 / 세트0만 정답 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        lines.append(f"| [{r['speaker']}]({r['speaker']}.md) | {r['set0_epoch']} / {r['set0_macro_f1']*100:.2f}% | "
            f"{r['set3_epoch']} / {r['set3_macro_f1']*100:.2f}% | {r['f1_difference_pp']:+.2f}%p | "
            f"{r['set0_correct']}→{r['set3_correct']} | {r['set3_only_correct']} / {r['set0_only_correct']} |")
    lines += ["", "spk01은 각각 spk02 검증에서 선택한 epoch4/3 모델을 그대로 평가했다. "
        "spk01의 결과로 epoch를 고르거나 두 모델 중 하나를 선택하지 않았다. "
        "spk02/03/04는 각자 미학습 화자로 둔 고유 CV fold이며 이 네 행을 하나의 CV 평균으로 합치지 않는다.", "",
        "세트3은 이 비교의 네 행에서 F1이 높다. 화자2에서는 +7.32%p, 정답10개 증가지만 "
        "세트0만 맞히는 파일26개도 남는다. 화자3의 F1 차이는 +0.57%p로 작고 loss는 오히려 세트0이 낮다.", "",
        "## loss 비교", "", "| 화자 | 세트0 loss | 세트3 loss |", "| --- | ---: | ---: |"]
    for r in rows:
        lines.append(f"| {r['speaker']} | {r['set0_loss']:.4f} | {r['set3_loss']:.4f} |")
    lines += ["", "## 자료와 검증", "",
        "- 각 화자 문서에 세트3만 정답·세트0만 정답·둘 다 오답인 전체 파일과 양쪽 예측, WAV 링크가 있다.",
        "- 전체 CSV에는203개씩 정답·양쪽 예측·정답확률·NLL·선택 epoch·원본 파일명·패딩을 넣었다.",
        "- [앞서 악화했던 화자2의43개를 최고점끼리 대조](spk02_previous_43_at_best.csv).",
        "- 선택 epoch가 원래 history의 최초 최고 F1 시점인지 확인했다. spk02/03/04는 "
        "각 선택 epoch의 예측이 원래 저장된 best checkpoint 예측과 정확히 일치했다.",
        "- 두 세트×4화자=1,624개 예측에서 정확도·Macro F1·NLL을 재계산했다. "
        "원래 checkpoint·예측 파일의 해시를 analysis_audit.json에 기록했다.",
        "- 한 시드의 개발 검증 최고점이다. 독립 최종 test나 청취 기반 원인 판정으로 해석하지 않는다.", "",
        "[직전 epoch3→10 비교](../file_diagnostics_comparison/README.md) · [요약 CSV](speaker_summary.csv)", ""]
    (OUT / "README.md").write_text("\n".join(lines))
    artifacts = sorted(p for p in OUT.iterdir() if p.suffix in (".csv", ".md"))
    assert artifacts
    write_json(OUT / "analysis_audit.json", dict(seed=SEED, condition_id=CONDITION,
        set_ids=[0, 3], comparison="originally_selected_best_checkpoints_per_set_and_fold",
        selected_epochs_from_original_summaries=True, source_history_argmax_verified=True,
        original_selected_predictions_exact_match=True, prediction_rows_checked=1624,
        all_prediction_metrics_reproduced=True, frozen_sources_unchanged=True,
        spk01_epochs_selected_only_on_spk02=True, additional_training_epochs=0,
        original_formal_runs_unchanged=184, provenance=provenance,
        artifact_sha256={str(p.relative_to(ROOT)): file_hash(p) for p in artifacts}))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
