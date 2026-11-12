"""Compare fixed-seed set0/set3 epoch transitions on identical audio clips."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments import report_auxiliary_cv_file_diagnostics as diagnostic
from src.engine import file_hash, metrics, write_json

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/auxiliary-cv-v1"
OUT = REPORT / "file_diagnostics_comparison"
SPEAKERS = ("spk01", "spk02", "spk03", "spk04")
CONDITION = "2ed04a871308"
MODEL_NAMES = {0: "보조 없음(세트0)", 3: "보조 포함(세트3)"}
STATUS_NAMES = {"regressed": "정답→오답", "recovered": "오답→정답",
                "stable_correct": "정답 유지", "stable_wrong": "오답 유지"}


def directory(set_id, speaker):
    root = REPORT / ("file_diagnostics_set0" if set_id == 0 else "file_diagnostics")
    folder = ("spk01_external_model_spk02_seed44" if speaker == "spk01"
              else f"c{CONDITION}_set{set_id}_seed44_{speaker}")
    return root / folder


def verify_predictions(set_id, speaker, labels):
    folder = directory(set_id, speaker)
    ledger = pd.read_csv(folder.parent / f"{speaker}_all_files.csv").set_index("path")
    history, reference = [], None
    for epoch in range(1, 11):
        frame = pd.read_csv(folder / f"epoch_{epoch:02d}.csv").set_index("path")
        assert len(frame) == 203 and frame.index.is_unique
        if reference is None:
            reference = frame[["true_idx", "true_label"]]
        else:
            assert frame[["true_idx", "true_label"]].equals(reference)
        assert frame.true_label.equals(frame.true_idx.map(dict(enumerate(labels))))
        assert frame.pred_label.equals(frame.pred_idx.map(dict(enumerate(labels))))
        probabilities = frame[[f"p_{i}" for i in range(len(labels))]].to_numpy()
        assert np.isfinite(probabilities).all() and (probabilities >= 0).all()
        assert np.allclose(probabilities.sum(axis=1), 1, atol=1e-6)
        assert np.array_equal(probabilities.argmax(axis=1), frame.pred_idx)
        true_p = probabilities[np.arange(len(frame)), frame.true_idx.to_numpy()]
        assert (true_p > 0).all()
        paired_rows = ledger.loc[frame.index]
        assert paired_rows.true_label.equals(frame.true_label)
        assert paired_rows[f"pred_e{epoch:02d}"].equals(frame.pred_label)
        assert np.array_equal(paired_rows[f"correct_e{epoch:02d}"], frame.true_idx == frame.pred_idx)
        assert np.allclose(paired_rows[f"true_probability_e{epoch:02d}"], true_p, rtol=1e-12, atol=1e-15)
        assert np.allclose(paired_rows[f"nll_e{epoch:02d}"], -np.log(true_p), rtol=1e-12, atol=1e-12)
        scores = json.loads((folder / f"epoch_{epoch:02d}_metrics.json").read_text())
        calculated = metrics(frame.true_idx, frame.pred_idx, len(labels))
        assert all(abs(scores[k] - v) < 1e-12 for k, v in calculated.items())
        assert abs(float(-np.log(true_p).mean()) - scores["loss"]) < 1e-5
        history.append(dict(epoch=epoch, **scores))
    if speaker != "spk01":
        replay = json.loads((folder / "replay_audit.json").read_text())
        assert all(v for k, v in replay.items() if k not in ("speaker", "original_run"))
    return pd.DataFrame(history).set_index("epoch")


def status(frame, prefix):
    early, late = frame[f"set{prefix}_correct_e03"], frame[f"set{prefix}_correct_e10"]
    return np.select([early & ~late, ~early & late, early & late],
                     ["regressed", "recovered", "stable_correct"], default="stable_wrong")


def file_table(frame):
    lines = ["| 정답 | 파일 | 세트0 epoch3→10 | 세트3 epoch3→10 | 세트0 변화 |",
             "| --- | --- | --- | --- | --- |"]
    for row in frame.sort_values(["true_label", "path"]).itertuples():
        lines.append(f"| {row.true_label} | [{Path(row.path).name}]({row.audio_path}) | "
                     f"{row.set0_pred_e03}→{row.set0_pred_e10} | "
                     f"{row.set3_pred_e03}→{row.set3_pred_e10} | {STATUS_NAMES[row.set0_status]} |")
    return lines if len(frame) else ["해당 파일 없음."]


def paired_speaker(speaker, base_summary, histories):
    base = pd.read_csv(REPORT / "file_diagnostics_set0" / f"{speaker}_all_files.csv").set_index("path")
    aux = pd.read_csv(REPORT / "file_diagnostics" / f"{speaker}_all_files.csv").set_index("path")
    assert set(base.index) == set(aux.index) and len(base) == 203
    aux = aux.loc[base.index]
    meta = ["true_label", "speaker", "source_filename", "source_seconds", "padding_ms", "audio_path"]
    assert base[meta].equals(aux[meta])
    paired = base[meta].copy()
    for set_id, frame in [(0, base), (3, aux)]:
        for epoch in range(1, 11):
            for stem in ("pred", "correct", "true_probability", "nll"):
                col = f"{stem}_e{epoch:02d}"
                paired[f"set{set_id}_{col}"] = frame[col]
        paired[f"set{set_id}_status"] = status(paired, set_id)
    paired = paired.copy()  # Consolidate the wide ledger before adding comparisons.
    paired["both_correct_e03"] = paired.set0_correct_e03 & paired.set3_correct_e03
    paired["both_regressed"] = (paired.set0_status == "regressed") & (paired.set3_status == "regressed")
    paired["set3_regressed"] = paired.set3_status == "regressed"
    paired["set0_regressed"] = paired.set0_status == "regressed"
    paired["set3_gain_at_e10"] = ~paired.set0_correct_e10 & paired.set3_correct_e10
    paired["set3_loss_at_e10"] = paired.set0_correct_e10 & ~paired.set3_correct_e10
    paired["nll_difference_set3_minus_set0_e10"] = paired.set3_nll_e10 - paired.set0_nll_e10
    paired = paired.reset_index()
    previous = paired[paired.set3_regressed]
    common = paired[paired.both_correct_e03]
    row = dict(speaker=speaker, clips=203,
        set0_f1_e03=histories[0].loc[3, "macro_f1"], set0_f1_e10=histories[0].loc[10, "macro_f1"],
        set3_f1_e03=histories[3].loc[3, "macro_f1"], set3_f1_e10=histories[3].loc[10, "macro_f1"],
        set0_correct_e03=int(paired.set0_correct_e03.sum()), set0_correct_e10=int(paired.set0_correct_e10.sum()),
        set3_correct_e03=int(paired.set3_correct_e03.sum()), set3_correct_e10=int(paired.set3_correct_e10.sum()),
        set0_regressed=int(paired.set0_regressed.sum()), set3_regressed=len(previous),
        set0_recovered=int((paired.set0_status == "recovered").sum()),
        set3_recovered=int((paired.set3_status == "recovered").sum()),
        both_regressed=int(paired.both_regressed.sum()),
        set3_regressed_but_set0_stable_correct=int((previous.set0_status == "stable_correct").sum()),
        set3_regressed_but_set0_already_wrong=int((previous.set0_status == "stable_wrong").sum()),
        set3_regressed_but_set0_recovered=int((previous.set0_status == "recovered").sum()),
        common_correct_e03=len(common), common_set0_regressed=int(common.set0_regressed.sum()),
        common_set3_regressed=int(common.set3_regressed.sum()),
        e10_set3_gains=int(paired.set3_gain_at_e10.sum()), e10_set3_losses=int(paired.set3_loss_at_e10.sum()),
        set0_selected_epoch=int(base_summary["selected_epoch"]))
    source_speaker = "spk02" if speaker == "spk01" else speaker
    original = diagnostic.STUDY / "runs" / f"c{CONDITION}_set3_seed44_{source_speaker}" / "summary.json"
    row["set3_selected_epoch"] = json.loads(original.read_text())["best_epoch"]
    for set_id in (0, 3):
        epoch = row[f"set{set_id}_selected_epoch"]
        row[f"set{set_id}_f1_selected"] = float(histories[set_id].loc[epoch, "macro_f1"])
        for e in (3, 10):
            row[f"set{set_id}_loss_e{e:02d}"] = float(histories[set_id].loc[e, "loss"])
        row[f"set{set_id}_selected_to10_regressed"] = int(
            (paired[f"set{set_id}_correct_e{epoch:02d}"] & ~paired[f"set{set_id}_correct_e10"]).sum())
        assert row[f"set{set_id}_correct_e03"] - row[f"set{set_id}_regressed"] + row[f"set{set_id}_recovered"] == row[f"set{set_id}_correct_e10"]
    assert row["e10_set3_gains"] - row["e10_set3_losses"] == row["set3_correct_e10"] - row["set0_correct_e10"]
    assert len(previous) == sum(row[k] for k in ("both_regressed", "set3_regressed_but_set0_stable_correct",
        "set3_regressed_but_set0_already_wrong", "set3_regressed_but_set0_recovered"))
    subsets = {"all_files": paired, "set3_regressed_paired": previous,
        "set0_regressed_paired": paired[paired.set0_regressed], "both_regressed": paired[paired.both_regressed],
        "common_correct_e03": common, "set3_gains_e10": paired[paired.set3_gain_at_e10],
        "set3_losses_e10": paired[paired.set3_loss_at_e10]}
    for name, frame in subsets.items():
        frame.to_csv(OUT / f"{speaker}_{name}.csv", index=False, encoding="utf-8-sig")
    class_rows = []
    for label, group in paired.groupby("true_label", sort=False):
        counts = dict(speaker=speaker, label=label, clips=len(group))
        for set_id in (0, 3):
            for epoch in (3, 10):
                counts[f"set{set_id}_correct_e{epoch:02d}"] = int(group[f"set{set_id}_correct_e{epoch:02d}"].sum())
        counts["set3_gains_e10"] = int(group.set3_gain_at_e10.sum())
        counts["set3_losses_e10"] = int(group.set3_loss_at_e10.sum())
        class_rows.append(counts)
    pd.DataFrame(class_rows).to_csv(OUT / f"{speaker}_class_comparison.csv", index=False)
    scope = ("spk02 fold 모델(학습 spk03/04/05)의 별도 평가. spk01에 학습·scaler 적합·epoch 선택을 하지 않았다."
             if speaker == "spk01" else f"{speaker}를 학습에서 제외한 고유 CV fold.")
    lines = [f"# {speaker}: 보조 특징 제외/포함 비교", "", scope, "",
        "seed44, mask+speed, weight decay 1e-4, Dropout 0, 나머지 조건·데이터·분할 동일. "
        "세트0은 보조 없음, 세트3은 F1–F3 포먼트와 B1–B3 대역폭 보조다.", "",
        "| 모델 | F1 epoch3→10 | 정답 /203 | 정답→오답 | 오답→정답 | 선택 epoch의 F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for s in (0, 3):
        lines.append(f"| {MODEL_NAMES[s]} | {row[f'set{s}_f1_e03']*100:.2f}%→{row[f'set{s}_f1_e10']*100:.2f}% | "
            f"{row[f'set{s}_correct_e03']}→{row[f'set{s}_correct_e10']} | {row[f'set{s}_regressed']} | "
            f"{row[f'set{s}_recovered']} | epoch{row[f'set{s}_selected_epoch']}: {row[f'set{s}_f1_selected']*100:.2f}% |")
    lines += ["", f"검증/진단 loss epoch3→10: 세트0 {row['set0_loss_e03']:.4f}→{row['set0_loss_e10']:.4f}, "
        f"세트3 {row['set3_loss_e03']:.4f}→{row['set3_loss_e10']:.4f}.", "",
        f"원래 선택 epoch→10 악화: 세트0 {row['set0_selected_to10_regressed']}개, "
        f"세트3 {row['set3_selected_to10_regressed']}개. 선택 시점이 달라 epoch3→10과 구분한다.", "",
        "## 같은 파일끼리 비교", "",
        f"두 모델이 epoch3에 모두 맞힌 {len(common)}개 중 epoch10에 틀린 파일은 "
        f"세트0 {row['common_set0_regressed']}개({100*row['common_set0_regressed']/len(common):.1f}%), "
        f"세트3 {row['common_set3_regressed']}개({100*row['common_set3_regressed']/len(common):.1f}%). "
        f"둘 다 악화한 파일은 {row['both_regressed']}개다.", "",
        f"epoch10 전체203개에서 세트3만 맞힌 파일 {row['e10_set3_gains']}개, "
        f"세트0만 맞힌 파일 {row['e10_set3_losses']}개. 정답 수 차이는 "
        f"{row['set3_correct_e10']-row['set0_correct_e10']:+d}개다.", "",
        f"[전체203개 짝 비교 CSV]({speaker}_all_files.csv) · "
        f"[공통 epoch3 정답 집단]({speaker}_common_correct_e03.csv) · "
        f"[세트3만 맞힌 파일]({speaker}_set3_gains_e10.csv) · "
        f"[세트0만 맞힌 파일]({speaker}_set3_losses_e10.csv) · "
        f"[단어별 비교]({speaker}_class_comparison.csv)", "",
        f"## 앞서 세트3에서 악화한 {len(previous)}개를 세트0과 대조", "",
        f"세트0에서도 악화 {row['both_regressed']}개 / 세트0은 정답 유지 {row['set3_regressed_but_set0_stable_correct']}개 / "
        f"세트0은 처음부터 오답 유지 {row['set3_regressed_but_set0_already_wrong']}개 / "
        f"세트0은 회복 {row['set3_regressed_but_set0_recovered']}개.", "",
        f"[이 목록 CSV]({speaker}_set3_regressed_paired.csv)", ""] + file_table(previous)
    lines += ["", "## 세트0에서 악화한 전체 파일", "",
        f"[CSV]({speaker}_set0_regressed_paired.csv)", ""] + file_table(paired[paired.set0_regressed])
    lines += ["", "## 단어별 정답 수", "",
        "| 단어 | 파일 수 | 세트0 epoch3→10 | 세트3 epoch3→10 |",
        "| --- | ---: | ---: | ---: |"]
    for c in class_rows:
        lines.append(f"| {c['label']} | {c['clips']} | {c['set0_correct_e03']}→{c['set0_correct_e10']} | "
                     f"{c['set3_correct_e03']}→{c['set3_correct_e10']} |")
    lines += ["", "## 해석 범위", "",
        "한 seed의 사후 진단이다. 같은 epoch라도 두 모델의 수렴 진행 정도는 다르다. "
        "두 모델 모두 epoch3에 맞힌 집단의 비율과 최종 정답/오답의 짝 비교도 함께 보아야 한다. "
        "이 자료만으로 보조 특징이 특정 파일의 오류를 유발했다고 인과를 확정하지 않는다. "
        "파일은 모든 epoch에 동일한 검증 파형이며, 청취 검수는 하지 않았다.", ""]
    (OUT / f"{speaker}.md").write_text("\n".join(lines))
    return row, paired


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(diagnostic.DATA / "metadata.csv", keep_default_na=False).set_index("path")
    recipe = pd.read_csv(diagnostic.DATA / "preprocessing_recipe.csv").set_index("path")
    protocol = json.loads((diagnostic.STUDY / "protocol.json").read_text())
    assert all(file_hash(p) == digest for p, digest in protocol["source_sha256"].items())
    labels = protocol["config"]["labels"]
    scopes = [json.loads((REPORT / root / "scope.json").read_text())
              for root in ("file_diagnostics_set0", "file_diagnostics")]
    assert scopes[0]["condition"] == scopes[1]["condition"] and scopes[0]["seed"] == scopes[1]["seed"] == 44
    assert [s["set_id"] for s in scopes] == ["0", "3"]
    for root in ("file_diagnostics_set0", "file_diagnostics"):
        assert json.loads((REPORT / root / "complete.json").read_text())["complete"]
    diagnostic.SET_ID = "0"
    diagnostic.REPORT = REPORT / "file_diagnostics_set0"
    rows, base_rows, histories, paired = [], [], {}, {}
    for speaker in SPEAKERS:
        base_summary, _, _ = diagnostic.diagnose(speaker, metadata, recipe, labels)
        base_rows.append(base_summary)
        histories[speaker] = {s: verify_predictions(s, speaker, labels) for s in (0, 3)}
        row, paired[speaker] = paired_speaker(speaker, base_summary, histories[speaker])
        rows.append(row)
    pd.DataFrame(base_rows).to_csv(diagnostic.REPORT / "speaker_summary.csv", index=False)
    write_json(diagnostic.REPORT / "analysis_audit.json", dict(set_id=0, seed=44,
        speakers=4, clips_per_speaker=203, epochs_per_speaker=10,
        prediction_rows_checked=8120, all_prediction_metrics_reproduced=True,
        replays_bit_identical=3, same_epoch_comparison=[3, 10], original_selection_unchanged=True,
        user_authorized_spk01_post_selection_evaluation=True, human_listening_performed=False))
    (diagnostic.REPORT / "README.md").write_text(
        "# 세트0 보조 없는 모델의 파일별 진단\n\n"
        "동일 공통 조건·seed44·epoch3→10 기준. [세트3과 짝 비교](../file_diagnostics_comparison/README.md).\n\n" +
        "\n".join(f"- [{s}]({s}.md)" for s in SPEAKERS) + "\n")
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "speaker_summary.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True, sharey=True)
    for ax, speaker in zip(axes.flat, SPEAKERS):
        for set_id, color in ((0, "#527ea6"), (3, "#dc7832")):
            h = histories[speaker][set_id]
            ax.plot(h.index, h.macro_f1*100, "o-", color=color, markersize=4,
                    label=f"set {set_id}: " + ("no auxiliary" if set_id == 0 else "formants + bandwidths"))
        ax.axvline(3, color="gray", linestyle="--", alpha=.5)
        ax.set(title=speaker + (" (external; spk02-fold model)" if speaker == "spk01" else " (own held-out fold)"),
               xlabel="Epoch", ylabel="Macro F1 (%)", ylim=(0, 100), xticks=range(1, 11))
        ax.grid(alpha=.2)
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("Same mask+speed condition and seed 44: set 0 versus set 3")
    fig.text(.5, .014, "spk01: separate post-selection evaluation; no training or epoch selection on its clips.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .035, 1, .96))
    for suffix in ("png", "svg"):
        fig.savefig(OUT / f"speaker_epoch_comparison.{suffix}", dpi=180)
    plt.close(fig)
    lines = ["# 보조 특징 제외/포함: 같은 파일·같은 epoch 비교", "",
        "세트0(보조 없음)과 세트3(포먼트 F1–F3+대역폭 B1–B3)을 동일한 mask+speed·seed44·"
        "weight decay 1e-4·Dropout 0·화자 분할로 비교했다. 원래 정식 실행은 변경하지 않았다.", "",
        "| 화자 | 세트0 F1 epoch3→10 | 세트3 F1 epoch3→10 | 악화 개수 세트0 / 세트3 | 회복 개수 세트0 / 세트3 |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for r in rows:
        lines.append(f"| [{r['speaker']}]({r['speaker']}.md) | {r['set0_f1_e03']*100:.2f}%→{r['set0_f1_e10']*100:.2f}% | "
            f"{r['set3_f1_e03']*100:.2f}%→{r['set3_f1_e10']*100:.2f}% | "
            f"{r['set0_regressed']} / {r['set3_regressed']} | {r['set0_recovered']} / {r['set3_recovered']} |")
    lines += ["", "spk02/03/04는 각자 학습에서 제외한 CV fold다. spk01은 두 세트 모두 "
        "spk02 fold 모델(학습 spk03/04/05)에 넣은 별도 평가이며 CV 평균에 넣지 않는다.", "",
        "## 초반 성능 차이를 반영한 파일 비교", "",
        "| 화자 | 둘 다 epoch3에 정답 | 그중 세트0 악화 | 그중 세트3 악화 | 둘 다 악화 | epoch10 세트3만 정답 / 세트0만 정답 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        n = r["common_correct_e03"]
        lines.append(f"| {r['speaker']} | {n} | {r['common_set0_regressed']} ({100*r['common_set0_regressed']/n:.1f}%) | "
            f"{r['common_set3_regressed']} ({100*r['common_set3_regressed']/n:.1f}%) | {r['both_regressed']} | "
            f"{r['e10_set3_gains']} / {r['e10_set3_losses']} |")
    r = next(r for r in rows if r["speaker"] == "spk02")
    lines += ["", f"앞서 spk02 세트3에서 악화한 {r['set3_regressed']}개 중 세트0에서도 악화한 파일은 "
        f"{r['both_regressed']}개다. 세트0에서 정답 유지 {r['set3_regressed_but_set0_stable_correct']}개, "
        f"오답 유지 {r['set3_regressed_but_set0_already_wrong']}개, 회복 {r['set3_regressed_but_set0_recovered']}개다. "
        "[43개 파일의 양쪽 예측과 WAV 링크](spk02.md).", "",
        "## 해석", "",
        "세트3의 spk02 하락 폭만으로 보조를 빼면 더 좋다고 결론내릴 수 없다. "
        f"세트0의 epoch3 점수가 {r['set0_f1_e03']*100:.2f}%로 더 낮고, epoch10도 "
        f"{r['set0_f1_e10']*100:.2f}%로 세트3 {r['set3_f1_e10']*100:.2f}%보다 낮다. "
        f"spk02의 원래 선택점도 세트0 epoch{r['set0_selected_epoch']} {r['set0_f1_selected']*100:.2f}%, "
        f"세트3 epoch{r['set3_selected_epoch']} {r['set3_f1_selected']*100:.2f}%다. "
        "다만 보조가 각 파일의 오류에 미친 인과는 이 한 시드 비교로 분리할 수 없다.", "",
        f"spk02 loss는 epoch3→10에 세트0 {r['set0_loss_e03']:.4f}→{r['set0_loss_e10']:.4f}, "
        f"세트3 {r['set3_loss_e03']:.4f}→{r['set3_loss_e10']:.4f}다. "
        "epoch10에 세트3의 F1이 더 높아도 평균 정답 NLL은 더 나쁘다. "
        "분류 정답 여부와 예측 확신의 변화는 별도로 보아야 한다.", "",
        "spk01은 세트3의 큰 초반 우위가 후반에 사라지고 epoch10에는 세트0이 소폭 높다. "
        "spk03/04는 두 모델 모두 epoch3보다 epoch10 점수가 높다. "
        "같은 epoch는 같은 학습 진척을 보장하지 않으며, 더 많은 시드의 파일 진단이나 "
        "사전에 고정한 정지 정책의 별도 검증 없이 이 사례로 학습 조건을 다시 선택하지 않는다.", "",
        "## 검증과 산출물", "",
        "- 세트0 재현3회도 원래 history CSV·최적 가중치 텐서·최적 예측 CSV·지표와 모두 일치했다.",
        "- 두 세트·4화자·10epoch의 16,240개 예측에서 정확도/F1/NLL을 재계산했다. 동일 파일·정답·전처리 metadata도 확인했다.",
        "- 기존 정식184회/1,840epochs는 변경하지 않았다. 후속 파일 진단 재현은 세트3 30epochs+세트0 30epochs다.",
        "- [화자 요약 CSV](speaker_summary.csv). 각 화자 Markdown에 기존 세트3 악화 파일과 세트0 악화 파일 전체를 WAV 링크로 제공한다. "
        "각 화자 CSV는10epoch 예측·정답확률·NLL·원본파일명·패딩을 함께 담는다.",
        "- 사람의 청취나 라벨 재검수는 하지 않았다. 파일명 태그·패딩의 오류 원인도 확정하지 않았다.", "",
        "![동일 조건에서 보조 제외/포함 곡선](speaker_epoch_comparison.png)", ""]
    (OUT / "README.md").write_text("\n".join(lines))
    artifacts = sorted(p for p in OUT.iterdir() if p.suffix in (".csv", ".md", ".png", ".svg"))
    assert artifacts, "No comparison artifacts found for the hash audit"
    write_json(OUT / "analysis_audit.json", dict(set_ids=[0, 3], seed=44, epochs=[3, 10],
        speakers=list(SPEAKERS), prediction_rows_checked=16240, identical_audio_and_labels=True,
        all_prediction_metrics_reproduced=True, exact_replays_per_set=3, frozen_sources_unchanged=True,
        original_formal_runs_unchanged=184, diagnostic_replay_epochs_across_both_sets=60,
        spk01_separate_post_selection_evaluation=True, human_listening_performed=False,
        artifact_sha256={str(p.relative_to(ROOT)): file_hash(p) for p in artifacts}))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
