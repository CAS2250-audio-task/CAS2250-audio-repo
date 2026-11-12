"""Select one epoch per set using all seed/fold means, then compare clips."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.engine import file_hash, metrics, write_json

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "runs/auxiliary-cv-v1"
REPORT = ROOT / "reports/auxiliary-cv-v1"
OUT = REPORT / "common_epoch_comparison"
DATA = ROOT / "datasets/task2/real-context-1p5-v1"
CONDITION = "2ed04a871308"
CV_SPEAKERS = ("spk02", "spk03", "spk04", "spk05")
SEEDS = (42, 43, 44)
PAIRED_SEED = 44


def histories():
    frames, hashes = [], {}
    for set_id in (0, 3):
        for seed in SEEDS:
            for speaker in CV_SPEAKERS:
                path = STUDY / "runs" / f"c{CONDITION}_set{set_id}_seed{seed}_{speaker}" / "history.csv"
                h = pd.read_csv(path)
                assert h.epoch.tolist() == list(range(1, 11)) and h.val_n.eq(203).all()
                h = h.assign(set_id=set_id, seed=seed, speaker=speaker)
                frames.append(h)
                hashes[str(path.relative_to(ROOT))] = file_hash(path)
    frame = pd.concat(frames, ignore_index=True)
    means = frame.groupby(["set_id", "epoch"])[["val_macro_f1", "val_accuracy", "val_loss"]].mean()
    selected = {s: int(means.loc[s].val_macro_f1.idxmax()) for s in (0, 3)}
    per_seed = frame.groupby(["set_id", "seed", "epoch"])[["val_macro_f1", "val_accuracy", "val_loss"]].mean()
    rows = []
    for s, epoch in selected.items():
        f1 = [per_seed.loc[(s, seed, epoch), "val_macro_f1"] for seed in SEEDS]
        rows.append(dict(set_id=s, selected_common_epoch=epoch,
            cv_macro_f1=means.loc[(s, epoch), "val_macro_f1"], seed_cv_f1_std=np.std(f1, ddof=1),
            mean_accuracy=means.loc[(s, epoch), "val_accuracy"], mean_loss=means.loc[(s, epoch), "val_loss"]))
    frame.to_csv(OUT / "all_seed_fold_epoch_histories.csv", index=False)
    means.to_csv(OUT / "all_seed_common_epoch_means.csv")
    per_seed.to_csv(OUT / "per_seed_common_epoch_means.csv")
    pd.DataFrame(rows).to_csv(OUT / "aggregate_summary.csv", index=False)
    frame[frame.apply(lambda r: r.epoch == selected[r.set_id], axis=1)].to_csv(
        OUT / "fold_seed_at_selected_epoch.csv", index=False)
    return selected, frame, means, per_seed, rows, hashes


def load_epoch(set_id, speaker, epoch, labels, history):
    base = REPORT / ("file_diagnostics_set0" if set_id == 0 else "file_diagnostics")
    folder = base / ("spk01_external_model_spk02_seed44" if speaker == "spk01" else
                     f"c{CONDITION}_set{set_id}_seed{PAIRED_SEED}_{speaker}")
    source = folder / f"epoch_{epoch:02d}.csv"
    f = pd.read_csv(source).set_index("path")
    assert len(f) == 203 and f.index.is_unique
    assert f.true_label.equals(f.true_idx.map(dict(enumerate(labels))))
    assert f.pred_label.equals(f.pred_idx.map(dict(enumerate(labels))))
    p = f[[f"p_{i}" for i in range(len(labels))]].to_numpy()
    assert np.isfinite(p).all() and (p >= 0).all() and np.allclose(p.sum(axis=1), 1, atol=1e-6)
    assert np.array_equal(p.argmax(axis=1), f.pred_idx)
    true_p = p[np.arange(len(f)), f.true_idx.to_numpy()]
    assert (true_p > 0).all()
    scores = json.loads((folder / f"epoch_{epoch:02d}_metrics.json").read_text())
    calculated = metrics(f.true_idx, f.pred_idx, len(labels))
    assert all(abs(scores[k] - v) < 1e-12 for k, v in calculated.items())
    assert abs(float(-np.log(true_p).mean()) - scores["loss"]) < 1e-5
    if speaker in CV_SPEAKERS:
        h = history.query("set_id == @set_id and seed == @PAIRED_SEED and speaker == @speaker and epoch == @epoch").iloc[0]
        assert all(abs(scores[k] - h[v]) < 1e-12 for k, v in
                   (("macro_f1", "val_macro_f1"), ("accuracy", "val_accuracy"), ("loss", "val_loss")))
        audit = json.loads((folder / "replay_audit.json").read_text())
        assert all(v for k, v in audit.items() if k not in ("speaker", "original_run"))
    else:
        protocol = json.loads((folder / "protocol.json").read_text())
        assert protocol["no_training_or_scaler_fit_on_spk01"]
    f["true_probability"] = true_p
    f["nll"] = -np.log(true_p)
    return f, scores, {str(source.relative_to(ROOT)): file_hash(source)}


def file_table(frame):
    lines = ["| 정답 | 파일 | 세트0 공통 최고점 예측 | 세트3 공통 최고점 예측 |",
             "| --- | --- | --- | --- |"]
    for row in frame.sort_values(["true_label", "path"]).itertuples():
        lines.append(f"| {row.true_label} | [{Path(row.path).name}]({row.audio_path}) | "
                     f"{row.set0_prediction} | {row.set3_prediction} |")
    return lines if len(frame) else ["해당 파일 없음."]


def paired_speaker(speaker, selected, history, labels, metadata, recipe):
    loaded = {s: load_epoch(s, speaker, selected[s], labels, history) for s in (0, 3)}
    a, b = loaded[0][0], loaded[3][0]
    assert a.index.equals(b.index) and a.true_label.equals(b.true_label)
    paired = a[["true_idx", "true_label"]].join(metadata[["source_filename", "source_seconds"]])
    paired["speaker"] = speaker
    paired["padding_ms"] = (recipe.pad_left_samples + recipe.pad_right_samples) / 16
    paired["audio_path"] = [str(DATA / p) for p in paired.index]
    assert all(Path(p).is_file() for p in paired.audio_path)
    row = dict(speaker=speaker, seed=PAIRED_SEED, clips=203, included_in_cv=speaker in CV_SPEAKERS)
    for s, (f, scores, _) in loaded.items():
        paired[f"set{s}_epoch"] = selected[s]
        paired[f"set{s}_prediction"] = f.pred_label
        paired[f"set{s}_pred_idx"] = f.pred_idx
        paired[f"set{s}_correct"] = f.pred_idx == f.true_idx
        for key in ("true_probability", "nll"):
            paired[f"set{s}_{key}"] = f[key]
        row.update({f"set{s}_{k}": scores[k] for k in ("macro_f1", "accuracy", "loss")})
        row[f"set{s}_correct"] = int(paired[f"set{s}_correct"].sum())
    paired["relation"] = np.select(
        [~paired.set0_correct & paired.set3_correct, paired.set0_correct & ~paired.set3_correct,
         paired.set0_correct & paired.set3_correct],
        ["set3_only_correct", "set0_only_correct", "both_correct"], default="both_wrong")
    paired = paired.reset_index()
    for relation in ("set3_only_correct", "set0_only_correct", "both_correct", "both_wrong"):
        row[relation] = int((paired.relation == relation).sum())
        paired[paired.relation == relation].to_csv(OUT / f"{speaker}_{relation}.csv", index=False, encoding="utf-8-sig")
    assert row["set3_only_correct"] - row["set0_only_correct"] == row["set3_correct"] - row["set0_correct"]
    paired.to_csv(OUT / f"{speaker}_all_files.csv", index=False, encoding="utf-8-sig")
    scope = ("화자1은 전체 평균의 epoch 선택에서 제외했다. 선택된 epoch를 기존 spk02-fold 모델에 적용한 별도 진단이다."
             if speaker == "spk01" else f"{speaker}를 학습에서 제외한 CV fold, seed44의 같은203개 파일이다.")
    lines = [f"# {speaker}: 총합 기준의 공통 최고 epoch 비교", "", scope, "",
        f"전체3시드·4-fold 평균 Macro F1로 고른 세트0 epoch{selected[0]}, 세트3 epoch{selected[3]}을 모든 화자에 동일하게 적용했다.", "",
        "| 모델 | F1 | 정답 /203 | loss |", "| --- | ---: | ---: | ---: |"]
    for s in (0, 3):
        lines.append(f"| 세트{s} | {row[f'set{s}_macro_f1']*100:.2f}% | {row[f'set{s}_correct']} | {row[f'set{s}_loss']:.4f} |")
    for relation, title in (("set3_only_correct", "세트3만 맞힌 파일"), ("set0_only_correct", "세트0만 맞힌 파일"), ("both_wrong", "둘 다 틀린 파일")):
        lines += ["", f"## {title}: {row[relation]}개", "", f"[CSV]({speaker}_{relation}.csv)", ""]
        lines += file_table(paired[paired.relation == relation])
    lines += ["", f"[전체203개와 확률·NLL·원본명·패딩]({speaker}_all_files.csv)", "",
        "원래 개발 CV에서 얻은 사후 비교다. 독립 test나 파일 오류의 인과 판정은 아니다. 청취 검수는 하지 않았다.", ""]
    (OUT / f"{speaker}.md").write_text("\n".join(lines))
    sources = {k: v for _, _, hashes in loaded.values() for k, v in hashes.items()}
    return row, paired, sources


def render_report(selected, means, per_seed, aggregates, rows, pooled_rows, pooled, fold_means):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for s, color in ((0, "#527ea6"), (3, "#dc7832")):
        for ax, key, factor in ((axes[0], "val_macro_f1", 100), (axes[1], "val_loss", 1)):
            h = per_seed.loc[s][key].unstack("seed")
            avg, sd = h.mean(axis=1)*factor, h.std(axis=1, ddof=1)*factor
            ax.plot(h.index, avg, "o-", color=color, markersize=3, label=f"set {s}")
            ax.fill_between(h.index, avg-sd, avg+sd, color=color, alpha=.12)
            e = selected[s]
            ax.plot(e, avg.loc[e], "*", color=color, markersize=14)
            ax.axvline(e, color=color, alpha=.35, linestyle="--")
        axes[0].annotate(f"set {s}: e{selected[s]} / {means.loc[(s, selected[s]), 'val_macro_f1']*100:.2f}%",
            (selected[s], means.loc[(s, selected[s]), "val_macro_f1"]*100),
            xytext=(-110, -32 if s == 0 else 18), textcoords="offset points", color=color, fontsize=9)
    for ax in axes:
        ax.set(xlabel="Shared epoch across all folds and seeds", xticks=range(1, 11))
        ax.grid(alpha=.2)
        ax.legend()
    axes[0].set(ylabel="Mean CV macro F1 (%)", ylim=(0, 100))
    axes[1].set(ylabel="Mean validation loss", ylim=(0, None))
    fig.suptitle("One shared epoch per set: 3 seeds x 4 held-out speakers")
    fig.text(.5, .015, "Shading: SD across 3 seed-level CV means. Stars: shared epochs selected by mean CV F1.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, .96))
    for suffix in ("png", "svg"):
        fig.savefig(OUT / f"common_epoch_curves.{suffix}", dpi=180)
    plt.close(fig)
    lines = ["# 총합 기준의 공통 최고 epoch 비교", "",
        "원래 실험과 같은 mask+speed·weight decay1e-4·Dropout0에서 세트0과3을 비교했다. "
        "세트마다 **3시드(42/43/44)×4-fold(spk02~05) 전체 평균 Macro F1이 최고인 공통 epoch 하나**를 골랐다. "
        "모든 화자·시드에 같은 epoch를 적용하며 화자1은 선택에서 제외했다.", "",
        "## 전체3시드·4-fold 결과", "",
        "| 모델 | 공통 최고 epoch | 평균 CV F1 | 시드별 CV F1 표준편차 | 평균 정확도 | 평균 loss |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in aggregates:
        lines.append(f"| 세트{r['set_id']} | {r['selected_common_epoch']} | {r['cv_macro_f1']*100:.2f}% | "
            f"{r['seed_cv_f1_std']*100:.2f}%p | {r['mean_accuracy']*100:.2f}% | {r['mean_loss']:.4f} |")
    delta = (aggregates[1]["cv_macro_f1"] - aggregates[0]["cv_macro_f1"])*100
    lines += ["", f"보조 포함 세트3의 평균 F1 차이는 **{delta:+.2f}%p**다. "
        "앞서 화자별 최고 epoch를 각각 선택한82.51%/84.62%와 집계 방식이 다르다. "
        "CV의 각 fold 모델은 학습 화자 구성이 다르며, 여기서는 그4개 모델에 같은 학습 시점을 적용한다.", "",
        "[전체240행 학습 기록](all_seed_fold_epoch_histories.csv) · "
        "[epoch별 총합](all_seed_common_epoch_means.csv) · "
        "[선택 epoch의24개 fold/seed 점수](fold_seed_at_selected_epoch.csv)", "",
        "같은 epoch에서 양쪽을 직접 비교하면 다음과 같다.", "",
        "| 공통 epoch | 세트0 평균 F1 | 세트3 평균 F1 |", "| --- | ---: | ---: |"]
    for epoch in sorted(set(selected.values())):
        lines.append(f"| {epoch} | {means.loc[(0, epoch), 'val_macro_f1']*100:.2f}% | "
                     f"{means.loc[(3, epoch), 'val_macro_f1']*100:.2f}% |")
    lines += ["",
        "## 공통 epoch에서 화자별3시드 평균", "",
        "| 화자 | 세트0 F1 | 세트3 F1 | 차이 |", "| --- | ---: | ---: | ---: |"]
    for speaker in CV_SPEAKERS:
        a, b = [float(fold_means.loc[(s, speaker), "val_macro_f1"]) for s in (0, 3)]
        lines.append(f"| {speaker} | {a*100:.2f}% | {b*100:.2f}% | {(b-a)*100:+.2f}%p |")
    lines += ["", "## 같은 seed44의 파일별 비교", "",
        f"위에서 고른 공통 epoch{selected[0]}/{selected[3]}을 그대로 썼다. "
        "이 비교는3시드 평균과 구분한다. seed44만으로 공통 최고 epoch를 골라도 같은8/9가 선택된다.", "",
        "| 화자 | 세트0 F1 | 세트3 F1 | 정답 세트0→3 | 세트3만 정답 / 세트0만 정답 |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for r in rows:
        lines.append(f"| [{r['speaker']}]({r['speaker']}.md) | {r['set0_macro_f1']*100:.2f}% | {r['set3_macro_f1']*100:.2f}% | "
            f"{r['set0_correct']}→{r['set3_correct']} | {r['set3_only_correct']} / {r['set0_only_correct']} |")
    gains = int((pooled.relation == "set3_only_correct").sum())
    losses = int((pooled.relation == "set0_only_correct").sum())
    assert gains-losses == pooled_rows[1]["correct"]-pooled_rows[0]["correct"]
    lines += ["", "spk01은spk02-fold 모델(학습03/04/05)의 별도 평가이며 어떤 총합에도 넣지 않았다. "
        "각 화자 문서에는 한쪽만 정답·둘 다 오답인 모든 파일의 양쪽 예측과 WAV 링크가 있다.", "",
        f"seed44의 CV812개에서 세트3만 정답 {gains}개, 세트0만 정답 {losses}개다. "
        f"총 정답은 {pooled_rows[0]['correct']}→{pooled_rows[1]['correct']}개다.", "",
        "### 화자별 평균과812개를 합친 지표의 구분", "",
        "| seed44 지표 | 세트0 | 세트3 |", "| --- | ---: | ---: |"]
    for key, title in (("val_macro_f1", "4-fold 평균 Macro F1"), ("val_accuracy", "평균 정확도")):
        values = [float(per_seed.loc[(s, PAIRED_SEED, selected[s]), key])*100 for s in (0, 3)]
        lines.append(f"| {title} | {values[0]:.2f}% | {values[1]:.2f}% |")
    lines.append(f"| 812개 예측을 합친 Macro F1 | {pooled_rows[0]['macro_f1']*100:.2f}% | {pooled_rows[1]['macro_f1']*100:.2f}% |")
    lines.append(f"| 평균 loss | {pooled_rows[0]['loss']:.4f} | {pooled_rows[1]['loss']:.4f} |")
    lines += ["", "화자별 F1을 평균하는 방식과 예측을 합쳐 F1을 계산하는 방식은 다르다. "
        "원래 연구의 주 지표인4-fold 평균 F1로 epoch를 선택했고,812개 합산 F1이나 정확도로 다시 고르지 않았다. "
        "seed44에서는 F1이 조금 높아도 정답 수와loss는 보조 포함 쪽이 나빴다.", "",
        "[812개 전체 짝 비교](seed44_all_812_paired.csv) · "
        "[세트3만 정답](seed44_812_set3_only_correct.csv) · "
        "[세트0만 정답](seed44_812_set0_only_correct.csv) · "
        "[단어별 정답 수](seed44_class_correct_counts.csv)", "",
        "## 검증 범위", "",
        "- 원래 학습 기록24개/240행에서 전체 평균과 선택epoch를 재계산했다. 학습 소스19개는 변경하지 않았다.",
        "- 파일별 예측이 없던 spk05의 세트0/3만 각10epoch 별도 재현했다. "
        "원래 history·최적 가중치 텐서·최적 예측·점수가 모두 일치했다. 이번 추가20epoch, 누적 파일 진단80epoch이며 정식184회와 구분한다.",
        "- 공통 최고점의 두 세트×(CV812개+별도spk01 203개)=2,030개 예측에서 정확도/F1/NLL을 재계산했다.",
        "- 공통epoch도 같은 개발 CV로 선택한 사후 분석이다. 독립 일반화 성능이 아니며 화자1은 선택에 사용하지 않았다. 청취 검수는 하지 않았다.", "",
        "![전체 공통 epoch 곡선](common_epoch_curves.png)", "",
        "[각자 최고점 비교](../best_checkpoint_comparison/README.md) · "
        "[전체 실험](../README.md)", ""]
    (OUT / "README.md").write_text("\n".join(lines))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((STUDY / "protocol.json").read_text())
    assert all(file_hash(p) == digest for p, digest in protocol["source_sha256"].items())
    selected, history, means, per_seed, aggregates, source_hashes = histories()
    metadata = pd.read_csv(DATA / "metadata.csv", keep_default_na=False).set_index("path")
    recipe = pd.read_csv(DATA / "preprocessing_recipe.csv").set_index("path")
    rows, paired_frames = [], []
    for speaker in (*CV_SPEAKERS, "spk01"):
        row, paired, sources = paired_speaker(speaker, selected, history, protocol["config"]["labels"], metadata, recipe)
        rows.append(row)
        source_hashes.update(sources)
        if speaker in CV_SPEAKERS:
            paired_frames.append(paired)
    pd.DataFrame(rows).to_csv(OUT / "seed44_speaker_summary.csv", index=False)
    pooled = pd.concat(paired_frames, ignore_index=True)
    assert len(pooled) == 812 and pooled.path.is_unique
    pooled.to_csv(OUT / "seed44_all_812_paired.csv", index=False, encoding="utf-8-sig")
    pooled_rows = []
    for s in (0, 3):
        scores = metrics(pooled.true_idx, pooled[f"set{s}_pred_idx"], len(protocol["config"]["labels"]))
        pooled_rows.append(dict(set_id=s, epoch=selected[s], **scores,
            loss=float(pooled[f"set{s}_nll"].mean()), correct=int(pooled[f"set{s}_correct"].sum())))
    pd.DataFrame(pooled_rows).to_csv(OUT / "seed44_pooled_metrics.csv", index=False)
    for relation in ("set3_only_correct", "set0_only_correct", "both_wrong"):
        pooled[pooled.relation == relation].to_csv(OUT / f"seed44_812_{relation}.csv", index=False, encoding="utf-8-sig")
    classes = pooled.groupby("true_label")[["set0_correct", "set3_correct"]].sum()
    classes.to_csv(OUT / "seed44_class_correct_counts.csv")
    fold_means = history[history.apply(lambda r: r.epoch == selected[r.set_id], axis=1)].groupby(
        ["set_id", "speaker"])[["val_macro_f1", "val_accuracy", "val_loss"]].mean()
    fold_means.to_csv(OUT / "three_seed_speaker_means.csv")
    render_report(selected, means, per_seed, aggregates, rows, pooled_rows, pooled, fold_means)
    artifacts = sorted(p for p in OUT.iterdir() if p.suffix in (".csv", ".md", ".png", ".svg"))
    assert artifacts
    write_json(OUT / "analysis_audit.json", dict(selection_metric="mean macro F1 over 3 seeds and 4 held-out folds at a shared epoch",
        selected_epochs=selected, source_histories=24, source_history_rows=240,
        paired_seed=44, seed44_cv_clips=812, separate_spk01_clips=203, prediction_rows_checked=2030,
        spk01_excluded_from_epoch_selection=True, individual_fold_best_epochs_not_used=True,
        cv_fold_models_are_separate=True, original_formal_runs_unchanged=184,
        new_diagnostic_replay_epochs=20, cumulative_diagnostic_replay_epochs=80,
        all_prediction_metrics_reproduced=True, frozen_sources_unchanged=True,
        source_sha256=source_hashes, artifact_sha256={str(p.relative_to(ROOT)): file_hash(p) for p in artifacts}))
    print(pd.DataFrame(aggregates).to_string(index=False))
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(pooled_rows).to_string(index=False))


if __name__ == "__main__":
    main()
