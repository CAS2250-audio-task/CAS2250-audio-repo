"""Independently audit completed CV runs and make standalone scientific figures."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.engine import file_hash, metrics, write_json


def write_interpretation(study, report, selection, frame, selected, pairs, run_count):
    names = ["스펙트로그램 기준", "F0·변화량·유성 정보", "포먼트 F1~F3",
             "포먼트+대역폭", "포먼트+변화량", "flatness·flux·centroid", "RMS", "전체 보조 특징"]
    settings = selection["common_training"]
    chosen = selection["selected_set"]
    control = frame[(frame.augmentation == "none") & (frame.weight_decay == 1e-4)
        & (frame.dropout == 0) & ~frame.freeze_front & (frame.backbone_lr_ratio == 1)]
    before = control.groupby("set_id").f1.mean()
    after = selected[selected.set_id.isin(["0", "4"])].groupby("set_id").f1.mean()
    lines = ["# 결과 해석", "", f"사전 절차에 따른 정식 학습 **{run_count}회**를 완료했다. "
        f"공통 조건의 최종 선택은 **세트 {chosen}**다. 주 지표는 개발 화자 4-fold 평균 macro F1이며, "
        "조건과 최적 epoch 선택에 사용한 개발 점수다.", "",
        "## 지난 실험에서 바뀐 점", "",
        "이전 16회 실험은 시간 정렬 보조 입력과 동일 초기값 비교, 512차원 표현 보존을 구현했다. "
        "하지만 단일 검증 화자의 3시드 F1이 기준 89.26%, 세트4 89.09%로 일관된 개선이 없었다. "
        "이번에는 원본 주변음을 더 보존한 새 데이터에서 화자별 CV, 개별 증강 선별과 확인, "
        "정규화 비교를 거쳐 동일 학습 조건으로 전체 특징 세트를 다시 비교했다. 데이터와 평가 방식이 "
        "달라 이전 89%와 이번 CV 점수의 높낮이를 직접 비교하지 않는다.", "",
        f"같은 새 데이터·CV의 3시드 평균에서 무증강→선택 조건은 세트0 "
        f"{before['0']*100:.2f}%→{after['0']*100:.2f}%, 세트4 "
        f"{before['4']*100:.2f}%→{after['4']*100:.2f}%다. 이 변화와 보조 입력의 효과를 구별해 해석한다.", "",
        "## 학습 조건 선택", "",
        "모든 선별은 seed42, 유망 조건 확인은 seed42/43/44를 사용했다. 공통 조건 채택 기준은 "
        "세트0·4 평균 F1 +0.5%p 이상, 두 모델을 평균한 최저 화자 F1 하락 1%p 이내, "
        "각 세트 평균 하락 0.5%p 이내다. 이는 사전 의사결정 기준이며 통계 검정이 아니다.", "",
        "| 단계·후보 | 비교 시드 수 | 세트0·4 평균 F1 | 최저 화자 F1 |",
        "| --- | ---: | ---: | ---: |"]
    for filename, stage in [("augmentation_selection.json", "개별 증강"),
                             ("regularization_selection.json", "정규화"),
                             ("freeze_lr_selection.json", "앞단·학습률")]:
        path = study / filename
        if not path.exists():
            continue
        for candidate in json.loads(path.read_text())["screens"]:
            c, m = candidate["condition"], candidate["metrics"]
            if stage == "개별 증강":
                label = "+".join(c["augmentation"])
            elif stage == "정규화":
                label = f"weight decay {c['weight_decay']:g}, dropout {c['dropout']:g}"
            else:
                label = f"앞단 고정 {c['freeze_front']}, 본체 LR 비율 {c['backbone_lr_ratio']:g}"
            lines.append(f"| {stage}: {label} | {len(m['seeds'])} | {m['mean_f1']*100:.2f}% | {m['worst_speaker_f1']*100:.2f}% |")
    augmentation = json.loads((study / "augmentation_final.json").read_text())
    lines += ["", "개별 증강은 무증강을, 정규화·앞단·학습률은 그 단계에서 선택한 증강 조건을 기준으로 비교했다. "
        "피치·잡음의 미채택은 이 선별 범위의 결과이며 일반적인 무효 판정이 아니다.", "",
        f"결합 증강 3시드 평균: {augmentation['metrics']['mean_f1']*100:.2f}%. "
        "Weight decay 1e-3은 평균이 높아져도 최저 화자 하락 제한을 넘어 채택하지 않았다.", "",
        f"최종 공통 조건: `{json.dumps(settings,ensure_ascii=False,sort_keys=True)}`.", "",
        "## 공통 조건에서 0~7 전체 비교", "",
        "아래는 모든 세트를 동일하게 비교한 seed42의 4-fold 결과다. 세트마다 유리한 학습 조건을 따로 쓰지 않았다.", "",
        "| 세트 | 추가 입력 | 평균 F1 | 최저 화자 F1 |", "| --- | --- | ---: | ---: |"]
    for i in range(8):
        s = selected[(selected.set_id == str(i)) & (selected.seed == 42)]
        lines.append(f"| {i} | {names[i]} | {s.f1.mean()*100:.2f}% | {s.f1.min()*100:.2f}% |")
    lines += ["", "## 유망 세트의 3시드 확인", "",
        "seed42 상위 두 세트와 세트0·4를 3시드로 확인했다. 최저 화자는 각 화자의 3시드 평균 중 최솟값이며, "
        "표준편차는 시드별 4-fold 평균 F1 세 개의 모집단 표준편차다. 화자 모집단의 신뢰구간은 아니다.", "",
        "| 세트 | 평균 F1 | 시드 표준편차 | 최저 화자 F1 | 기준 대비 평균 차이 |",
        "| --- | ---: | ---: | ---: | ---: |"]
    ranked = selection["confirmed_ranking"]
    reference = next(r["metrics"] for r in ranked if r["set_id"] == "0")
    for r in ranked:
        m = r["metrics"]
        lines.append(f"| {r['set_id']} | {m['mean_f1']*100:.2f}% | {m['seed_std']*100:.2f}%p | "
            f"{m['worst_speaker_f1']*100:.2f}% | {(m['mean_f1']-reference['mean_f1'])*100:+.2f}%p |")
    lines += ["", "보조 입력은 기준 대비 평균 F1이 0.5%p 이상 높고 최저 화자 하락이 1%p 이내일 때만 선택한다. "
        f"이 기준에 따라 세트 {chosen} 선택으로 결정했다. 3시드와 네 화자로 통계적 우월성이나 보편적인 개선을 주장하지 않는다.", "",
        "## 동일 발화의 정답·오답 변화", "",
        "공통 학습 조건의 세트0과 최종 선택 세트를 같은 화자·시드·발화끼리 짝지었다. "
        "각 시드는 네 검증 화자의 812개 클립를 한 번씩 포함한다. 세 시드는 같은 발화를 재사용한다.", "",
        "| 시드 | 기준 오답→선택 정답 | 기준 정답→선택 오답 |", "| --- | ---: | ---: |"]
    for seed in (42,43,44):
        p = pairs[(pairs.set_id == chosen) & (pairs.seed == seed)]
        lines.append(f"| {seed} | {int((~p.baseline_correct & p.correct).sum())} | {int((p.baseline_correct & ~p.correct).sum())} |")
    lines += ["", "[개별 짝지은 예측](paired_predictions.csv), [혼동 쌍](error_pairs.csv), "
        "[반복 오답](persistent_errors.csv)에 근거를 남겼다. 예측만으로 발음·거리·잡음 중 원인을 확정하지 않는다.", "",
        "## 속도 변형과 여백 확인", "",
        "고정 캐시의 단어 변형 1,600개 중 797개는 가속 후 부족한 길이만 0 패딩했고, "
        "784개는 감속 후 기존 검수 구간을 포함하도록 잘랐다. 19개는 보호 구간이 넘쳐 감속을 건너뛰었다. "
        "추가 0 패딩은 최대 약 135.8ms이며 silence 변형 24개는 원본을 유지했다. "
        "이는 학습 중 실제 추출 횟수가 아닌 고정 변형 수다. "
        "기존 검수 구간 보존은 확인했지만 실제 발화 경계의 정답을 새로 검수한 것은 아니다. "
        "[증강 적용 감사](augmentation_application_audit.json)에 기록했다.", "",
        "## 남는 한계와 검증", "",
        "- 개발 화자는 네 명이며 조건과 epoch를 같은 개발 CV로 선택했다. 독립 최종 평가가 아니다.",
        "- spk01은 이번 학습·특징 추출·모델 평가에서 제외했다. 이전 실험에서 이미 확인한 test이므로 새로운 미공개 test로 주장하지 않는다.",
        "- 짧은 원본 350개에는 부족한 길이만 0을 채웠다. 패딩 비율·녹음 환경·거리의 화자별 차이를 분리 통제하지 못했다.",
        "- 파형 증강은 클립당 고정 변형 2개다. 현재 주 지표는 원본 검증 성능이며 새 잡음·공간·거리의 강건성 향상을 직접 평가하지 않았다.",
        "- F0/포먼트의 availability는 측정 가능 표시이며 정확도의 정답이 아니다. 추출 품질이 추가 입력의 효과에 영향을 줄 수 있다.",
        "- 피크 예외 6개는 원본 검수 구간·수치를 확인했지만 사람의 청취 검수는 하지 않았다.", "",
        f"정식 {run_count}회×10 epochs의 예측·점수·최적 epoch·분할·scaler·초기값·소스 해시를 재검증했다. "
        "원본·이전·새 WAV 총 3,045개 해시 보존과 Linear(512,11)을 확인했다. 복구 검증 4 epochs는 선택에 쓰지 않았다. "
        "[실행 감사](execution_audit.json), [실행별 감사](run_audit.csv), [고정 프로토콜](protocol.json)에 기록했다.", "",
        "![전체 세트 비교](feature_sets_cv.png)", "", "![화자별 변화](speaker_stability.png)", ""]
    (report / "interpretation.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--study", default="runs/auxiliary-cv-v1")
    parser.add_argument("--reports", default="reports/auxiliary-cv-v1")
    parser.add_argument("--partial", action="store_true", help="Audit completed runs without a final selection")
    args = parser.parse_args()
    study, report = Path(args.study), Path(args.reports)
    selection_path = study / "final_selection.json"
    selection = json.loads(selection_path.read_text()) if selection_path.exists() else None
    if selection is None and not args.partial:
        parser.error("The study has no final selection; use --partial to audit its completed runs")
    protocol = json.loads((study / "protocol.json").read_text())
    data = pd.read_csv(study / "cache/original/metadata.csv", keep_default_na=False).set_index("path")
    labels = protocol["config"]["labels"]
    mapping = {label:i for i,label in enumerate(labels)}
    source_errors = [p for p,h in protocol["source_sha256"].items() if file_hash(p) != h]
    assert not source_errors, source_errors
    rows, checks, hashes = [], [], {}
    errors = []
    for summary_file in sorted((study / "runs").glob("*/summary.json")):
        summary = json.loads(summary_file.read_text())
        run = summary_file.parent
        config = json.loads((run / "config.json").read_text())
        manifest = json.loads((run / "manifest.json").read_text())
        fold = study / "folds" / summary["validation_speaker"]
        split = json.loads((fold / "split.json").read_text())
        history = pd.read_csv(run / "history.csv")
        pred = pd.read_csv(run / "validation/predictions.csv")
        best_checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
        scores = metrics(pred.true_idx, pred.pred_idx, len(labels))
        probabilities = pred[[f"p_{i}" for i in range(len(labels))]].to_numpy()
        checks_one = dict(run=run.name,
            epochs_complete=history.epoch.tolist() == list(range(1,11)),
            predictions_cover_exact_validation=set(pred.path) == set(split["val_paths"]) and len(pred) == 203,
            train_val_disjoint=not set(split["train_paths"]) & set(split["val_paths"]) and len(split["train_paths"]) == 609,
            true_labels_match=pred.true_idx.tolist() == [mapping[data.loc[p,"label"]] for p in pred.path],
            score_reproduced=all(abs(scores[k]-summary["validation"][k]) < 1e-12 for k in scores),
            probabilities_valid=bool(np.isfinite(probabilities).all() and np.allclose(probabilities.sum(1),1,atol=1e-6)
                and np.array_equal(probabilities.argmax(1),pred.pred_idx.to_numpy())),
            earliest_best_epoch=int(history.iloc[history.val_macro_f1.to_numpy().argmax()].epoch) == summary["best_epoch"],
            checkpoint_matches=best_checkpoint["epoch"] == summary["best_epoch"]
                and best_checkpoint["validation"] == summary["validation"]
                and tuple(best_checkpoint["model"]["backbone.fc.weight"].shape) == (11,512),
            config_matches_summary=all(config[k] == summary[k]
                for k in ("condition", "set_id", "seed", "validation_speaker")),
            protocol_matches=manifest["protocol_sha256"] == file_hash(study / "protocol.json"),
            manifest_sources_match=manifest["source_sha256"] == protocol["source_sha256"],
            scaler_matches=manifest["scaler_sha256"] == file_hash(fold / "scaler.json"),
            test_excluded=not summary["test_evaluated"] and not config["test_evaluated"]
                and all("spk01" not in p for p in split["train_paths"] + split["val_paths"]))
        if not all(v for k,v in checks_one.items() if k != "run"):
            errors.append(checks_one)
        checks.append(dict(checks_one, best_checkpoint_sha256=file_hash(run / "best.pt")))
        del best_checkpoint
        hashes.setdefault(summary["seed"],set()).add(summary["backbone_initial_sha256"])
        rows.append(dict(run=run.name, condition_id=summary["condition_id"], set_id=summary["set_id"],
            seed=summary["seed"], speaker=summary["validation_speaker"],
            f1=scores["macro_f1"], accuracy=scores["accuracy"],
            augmentation="+".join(summary["condition"]["augmentation"]) or "none",
            weight_decay=summary["condition"]["weight_decay"], dropout=summary["condition"]["dropout"],
            freeze_front=summary["condition"]["freeze_front"], backbone_lr_ratio=summary["condition"]["backbone_lr_ratio"]))
    assert not errors, errors[:2]
    assert all(len(values)==1 for values in hashes.values())
    frame = pd.DataFrame(rows)
    pd.DataFrame(checks).to_csv(report / "run_audit.csv", index=False)
    if args.partial:
        sample = study / "runs" / frame.run.iloc[0]
        checkpoint = torch.load(sample / "best.pt", map_location="cpu", weights_only=True)
        assert tuple(checkpoint["model"]["backbone.fc.weight"].shape) == (11,512)
        recipe = pd.read_csv(Path(protocol["config"]["data_root"]) / "preprocessing_recipe.csv")
        for r in recipe.itertuples():
            assert file_hash(Path(protocol["data"]["raw_root"])/r.path) == r.raw_wav_sha256
            assert file_hash(Path(protocol["data"]["reviewed_root"])/r.path) == r.previous_output_sha256
            assert file_hash(Path(protocol["config"]["data_root"])/r.path) == r.output_sha256
        audit = dict(formal_training_runs=len(rows), completed_epochs=len(rows)*10,
            study_complete=selection is not None, all_completed_run_checks_passed=True,
            same_initial_backbone_per_seed={str(k):len(v)==1 for k,v in hashes.items()},
            source_hashes_match_frozen_protocol=True, head_shape=[11,512], test_evaluated=False,
            raw_previous_and_new_wavs_unchanged=True, wav_hashes_verified_after_training=len(recipe)*3,
            engineering_recovery_probe=json.loads((study / "recovery_check.json").read_text()))
        write_json(report / "interim_execution_audit.json", audit)
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return
    settings = selection["common_training"]
    selected = frame[(frame.augmentation == ("+".join(settings["augmentation"]) or "none"))
        & (frame.weight_decay == settings["weight_decay"]) & (frame.dropout == settings["dropout"])
        & (frame.freeze_front == settings["freeze_front"]) & (frame.backbone_lr_ratio == settings["backbone_lr_ratio"])]
    common_id = selected.condition_id.iloc[0]
    assert selected.condition_id.nunique() == 1
    for set_id in map(str, range(8)):
        screen = selected[(selected.set_id == set_id) & (selected.seed == 42)]
        assert sorted(screen.speaker.tolist()) == ["spk02", "spk03", "spk04", "spk05"]
    actual = {}
    for entry in selection["confirmed_ranking"]:
        group = selected[selected.set_id == entry["set_id"]]
        assert len(group) == 12 and sorted(group.seed.unique().tolist()) == [42,43,44]
        assert group.groupby("seed").speaker.nunique().eq(4).all()
        actual[entry["set_id"]] = dict(mean_f1=float(group.f1.mean()),
            worst_speaker_f1=float(group.groupby("speaker").f1.mean().min()))
        assert all(abs(value-entry["metrics"][key]) < 1e-12
                   for key,value in actual[entry["set_id"]].items())
    ref = actual["0"]
    eligible = [(s,m) for s,m in actual.items() if m["mean_f1"] >= ref["mean_f1"]+.005
                and m["worst_speaker_f1"] >= ref["worst_speaker_f1"]-.01]
    expected = max(eligible, key=lambda item:item[1]["mean_f1"])[0] if eligible else "0"
    assert selection["selected_set"] == expected
    selected.to_csv(report / "selected_common_training_runs.csv",index=False)
    baseline = frame[(frame.augmentation == "none") & (frame.weight_decay == 1e-4)
        & (frame.dropout == 0) & ~frame.freeze_front & (frame.backbone_lr_ratio == 1)]
    paired = []
    for set_id in sorted({"0",selection["selected_set"]}):
        for seed in (42,43,44):
            for speaker in ("spk02","spk03","spk04","spk05"):
                chosen = selected[(selected.set_id==set_id)&(selected.seed==seed)&(selected.speaker==speaker)].iloc[0]
                control = selected[(selected.set_id=="0")&(selected.seed==seed)&(selected.speaker==speaker)].iloc[0]
                a = pd.read_csv(study / "runs" / control.run / "validation/predictions.csv").set_index("path")
                b = pd.read_csv(study / "runs" / chosen.run / "validation/predictions.csv").set_index("path")
                assert set(a.index)==set(b.index)
                b = b.loc[a.index]
                for path in a.index:
                    paired.append(dict(set_id=set_id,seed=seed,speaker=speaker,path=path,
                        true_label=labels[int(a.loc[path,"true_idx"])],
                        baseline_prediction=labels[int(a.loc[path,"pred_idx"])],
                        prediction=labels[int(b.loc[path,"pred_idx"])],
                        baseline_correct=bool(a.loc[path,"true_idx"]==a.loc[path,"pred_idx"]),
                        correct=bool(b.loc[path,"true_idx"]==b.loc[path,"pred_idx"])))
    pairs = pd.DataFrame(paired)
    pairs.to_csv(report / "paired_predictions.csv",index=False)
    wrong = pairs[~pairs.correct]
    wrong.groupby(["set_id","speaker","true_label","prediction"]).size().rename("wrong_predictions_across_seeds").reset_index().sort_values(
        "wrong_predictions_across_seeds",ascending=False).to_csv(report / "error_pairs.csv",index=False)
    pairs.groupby(["set_id","speaker","path","true_label"]).agg(
        errors_in_three_seeds=("correct",lambda x:int((~x).sum())),
        baseline_errors_in_three_seeds=("baseline_correct",lambda x:int((~x).sum()))).reset_index().sort_values(
            "errors_in_three_seeds",ascending=False).to_csv(report / "persistent_errors.csv",index=False)
    fig,ax = plt.subplots(figsize=(9,4.8))
    original = selected[selected.seed == 42]
    means = original.groupby("set_id").f1.mean()
    for i in range(8):
        scores = original[original.set_id == str(i)].sort_values("speaker").f1.to_numpy()*100
        ax.bar(i,means[str(i)]*100,color="#527ca5" if i != int(selection["selected_set"]) else "#dc8b35",alpha=.85)
        ax.scatter(i+np.linspace(-.15,.15,len(scores)),scores,s=26,color="#202a35",zorder=3)
    ax.set(xticks=range(8),xticklabels=[f"Set {i}" for i in range(8)],ylabel="Validation macro F1 (%)",
           title="All feature sets under the selected training condition (seed 42)",ylim=(0,100))
    ax.grid(axis="y",alpha=.2)
    fig.text(.5,.01,"Bars: mean of four held-out speakers. Dots: individual speakers; not confidence intervals.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1))
    fig.savefig(report / "feature_sets_cv.png",dpi=180)
    plt.close(fig)
    fig,ax = plt.subplots(figsize=(9,4.8))
    comparisons = [(baseline,"Initial"),(selected,"Selected")]
    if baseline.condition_id.iloc[0] == common_id:
        comparisons = [(baseline,"Initial / selected")]
    groups = [(df[df.set_id == str(s)],f"{name} set{s}") for df,name in comparisons for s in (0,4)]
    x = np.arange(4)
    width=.8/len(groups)
    for j,(group,label) in enumerate(groups):
        stats=group.groupby("speaker").f1.agg(["mean","std"]).reindex(["spk02","spk03","spk04","spk05"])
        ax.bar(x-.4+width*(j+.5),stats["mean"]*100,width,yerr=stats["std"].fillna(0)*100,
               capsize=3,label=label)
    ax.set(xticks=x,xticklabels=["spk02","spk03","spk04","spk05"],ylabel="Validation macro F1 (%)",
           title="Initial and selected common training: speaker stability",ylim=(0,100))
    ax.legend(fontsize=9)
    ax.grid(axis="y",alpha=.2)
    fig.text(.5,.01,"Error bars: standard deviation across seeds 42/43/44 within each speaker, not population uncertainty.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1))
    fig.savefig(report / "speaker_stability.png",dpi=180)
    plt.close(fig)
    sample = next(p.parent for p in (study / "runs").glob("*/summary.json")
                  if json.loads(p.read_text())["condition_id"] == common_id
                  and json.loads(p.read_text())["set_id"] == selection["selected_set"])
    checkpoint = torch.load(sample / "best.pt",map_location="cpu",weights_only=True)
    assert tuple(checkpoint["model"]["backbone.fc.weight"].shape) == (11,512)
    audit = dict(formal_training_runs=len(rows),completed_epochs=len(rows)*10,all_run_checks_passed=True,
        same_initial_backbone_per_seed={str(k):len(v)==1 for k,v in hashes.items()},
        source_hashes_match_frozen_protocol=True,head_shape=[11,512],test_evaluated=False,
        final_selection_reproduced_from_predictions=True,
        selected_condition_id=common_id,selected_set=selection["selected_set"],
        engineering_recovery_probe=json.loads((study / "recovery_check.json").read_text()))
    original_recipe = pd.read_csv(Path(protocol["config"]["data_root"]) / "preprocessing_recipe.csv")
    raw_root = Path(protocol["data"]["raw_root"])
    old_root = Path(protocol["data"]["reviewed_root"])
    for r in original_recipe.itertuples():
        assert file_hash(raw_root/r.path)==r.raw_wav_sha256
        assert file_hash(old_root/r.path)==r.previous_output_sha256
        assert file_hash(Path(protocol["config"]["data_root"])/r.path)==r.output_sha256
    audit["raw_previous_and_new_wavs_unchanged"] = True
    audit["wav_hashes_verified_after_training"] = len(original_recipe)*3
    write_json(report / "execution_audit.json",audit)
    write_interpretation(study, report, selection, frame, selected, pairs, len(rows))
    common = selected[selected.set_id.isin(["0","4"])].groupby("set_id").f1.mean()
    old = baseline.groupby("set_id").f1.mean()
    note = ("\n**완료 결과**\n\n"
        f"정식 학습 {len(rows)}회({len(rows)*10} epochs)를 완료했다. 별도의 복구 검증 4 epochs는 조건 선택에 쓰지 않았다.\n\n"
        f"초기 무증강 3시드 CV F1: set0 {old['0']*100:.2f}%, set4 {old['4']*100:.2f}%.\n"
        f"선택한 공통 조건의 3시드 CV F1: set0 {common['0']*100:.2f}%, set4 {common['4']*100:.2f}%.\n"
        f"공통 조건: `{json.dumps(settings,ensure_ascii=False,sort_keys=True)}`. 최종 선택 세트: {selection['selected_set']}.\n\n"
        "전체 세트의 공정한 한 시드 비교는 feature_sets_cv.png, 초기/선택 조건의 화자별 3시드 비교는 "
        "speaker_stability.png에 있다. 체크포인트 선택·예측·초기값·분할·표준화·프로토콜 해시를 "
        "독립적으로 다시 확인했고 run_audit.csv와 execution_audit.json에 기록했다.\n\n"
        "전체 세트 표·선택 근거·오답 변화와 한계는 [결과 해석](interpretation.md)에 정리했다.\n")
    with (report / "README.md").open("a") as handle:
        handle.write(note)
    print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
