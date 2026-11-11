"""Export compact, reviewable records; large caches/checkpoints stay in runs/."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .features import NAMES, SETS
from .study import collect
from src.engine import write_json


def markdown_table(frame):
    columns = list(frame.columns)
    return "| " + " | ".join(columns) + " |\n| " + " | ".join(["---"]*len(columns)) + " |\n" + "\n".join(
        "| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None))


def feature_qa(study, output):
    root = study / "cache/original"
    frame = pd.read_csv(root / "metadata.csv")
    with np.load(root / "inputs.npz") as a:
        selected = frame[frame.split == "val"].groupby("label", sort=True).head(1).index[:6]
        fig, axes = plt.subplots(len(selected), 2, figsize=(12, 2.2*len(selected)), squeeze=False)
        times = np.arange(a["raw"].shape[-1]) * 0.01
        for axrow, i in zip(axes, selected):
            label_id = json.loads((study / "protocol.json").read_text())["baseline_config"]["labels"].index(frame.loc[i,"label"])
            for k in [1,2,3]:
                axrow[0].plot(times,a["raw"][i,NAMES.index(f"f{k}")],label=f"F{k}",lw=1)
            axrow[0].set(ylabel="Hz",title=f"{Path(frame.loc[i,'path']).stem}, class ID {label_id}")
            axrow[0].legend(loc="upper right",ncol=3,fontsize=8)
            axrow[1].plot(times,a["raw"][i,NAMES.index("log_f0")],label="F0 semitones",lw=1)
            axrow[1].set(ylabel="ST relative to 100 Hz")
            probability = axrow[1].twinx()
            probability.plot(times,a["raw"][i,NAMES.index("voiced_prob")],color="gray",alpha=.4,label="voicing probability")
            probability.set(ylim=(0,1),ylabel="voicing probability")
            for ax in axrow:
                ax.set_xlim(0,1.5)
        axes[-1,0].set_xlabel("seconds")
        axes[-1,1].set_xlabel("seconds")
        fig.tight_layout()
        fig.savefig(output/"feature_qa.png",dpi=140)
        plt.close(fig)


def validation_cases(study, output, result, screening):
    """Describe observed errors; never invent acoustic causes or listening notes."""
    regular = result[(result.fusion == "middle") & (result.control_extra == 0) & (result.set_id != "0")]
    confirmed = regular.groupby("set_id").filter(lambda x: set(x.seed) >= {42,43,44})
    if len(confirmed):
        candidate = confirmed.groupby("set_id").val_macro_f1.mean().idxmax()
    else:
        candidate = regular.sort_values("val_macro_f1", ascending=False).iloc[0].set_id
    base = pd.read_csv(study/"runs/set0_seed42_middle/validation/predictions.csv")
    aux = pd.read_csv(study/f"runs/set{candidate}_seed42_middle/validation/predictions.csv").set_index("path").loc[base.path]
    cases = base[["path","true_idx","true_label","pred_label"]].rename(columns={"pred_label":"baseline_pred"}).copy()
    cases["auxiliary_set"] = candidate
    cases["auxiliary_pred"] = aux.pred_label.to_numpy()
    cases["baseline_confidence"] = base[[f"p_{i}" for i in range(11)]].max(axis=1)
    cases["auxiliary_confidence"] = aux[[f"p_{i}" for i in range(11)]].max(axis=1).to_numpy()
    cases["baseline_correct"] = base.pred_idx == base.true_idx
    cases["auxiliary_correct"] = aux.pred_idx.to_numpy() == base.true_idx.to_numpy()
    cases["effect"] = np.select([
        ~cases.baseline_correct & cases.auxiliary_correct,
        cases.baseline_correct & ~cases.auxiliary_correct,
        ~cases.baseline_correct & ~cases.auxiliary_correct],
        ["fixed_baseline_error","introduced_error","both_incorrect"], default="both_correct")
    cases["screening_error_count"] = 0
    cases["max_wrong_softmax"] = 0.0
    for row in screening.itertuples():
        pred = pd.read_csv(study/"runs"/row.run/"validation/predictions.csv").set_index("path").loc[base.path]
        wrong = pred.pred_idx.to_numpy() != base.true_idx.to_numpy()
        confidence = pred[[f"p_{i}" for i in range(11)]].max(axis=1).to_numpy()
        cases["screening_error_count"] += wrong.astype(int)
        cases["max_wrong_softmax"] = np.maximum(cases.max_wrong_softmax, np.where(wrong, confidence, 0))
    cases = cases[cases.screening_error_count > 0].copy()
    cases["pair_has_error"] = ~cases.baseline_correct | ~cases.auxiliary_correct
    cases = cases.sort_values(["pair_has_error","max_wrong_softmax","path"], ascending=[False,False,True])
    # Select diverse true labels, prioritizing baseline/candidate errors within each label.
    groups = [list(group.index) for _, group in cases.groupby("true_label",sort=True)]
    picked = []
    while len(picked) < min(20,len(cases)):
        for group in groups:
            if group and len(picked) < 20:
                picked.append(group.pop(0))
    cases = cases.loc[picked].copy()
    metadata = pd.read_csv(study/"cache/original/metadata.csv")
    lookup = dict(zip(metadata.path,metadata.index))
    with np.load(study/"cache/original/inputs.npz") as arrays:
        indices = [lookup[path] for path in cases.path]
        for name in ["log_f0","f1","f2","f3"]:
            cases[f"{name}_available_fraction"] = arrays["valid"][indices,NAMES.index(name),:].mean(axis=-1)
        cases["mean_voicing_probability"] = arrays["raw"][indices,NAMES.index("voiced_prob"),:].mean(axis=-1)
    notes = {
        "fixed_baseline_error":"기준 오답·보조 정답. 개선 원인은 미확정.",
        "introduced_error":"기준 정답·보조 오답. 악화 원인은 미확정.",
        "both_incorrect":"기준과 보조 모두 오답. 혼동 쌍과 추출 상태 검토 필요.",
        "both_correct":"비교 두 모델은 정답이나 다른 screening 세트에서 오류 발생."}
    cases["observation"] = cases.effect.map(notes)
    cases["audio_listened"] = False
    cases["acoustic_cause_verified"] = False
    cases.drop(columns=["pair_has_error"]).to_csv(output/"validation_case_review.csv",index=False,encoding="utf-8-sig")
    table = cases[["path","true_label","baseline_pred","auxiliary_pred","effect","f1_available_fraction"]].copy()
    table["path"] = table.path.map(lambda p:Path(p).stem)
    table["f1_available_fraction"] = table.f1_available_fraction.map(lambda v:f"{100*v:.1f}%")
    table.columns = ["파일","정답","기준 예측",f"세트 {candidate} 예측","관찰","F1 측정 가능 프레임"]
    text = "# Validation 오류 사례 점검\n\n"
    text += f"seed42 기준과, 반복 검증에서 가장 높았던 보조 세트 {candidate}를 비교한다. "
    text += "전체 8세트 중 오류가 나온 원본 validation 파일에서 정답 단어별로 순환 선택한 최대 20개다. "
    text += "아래는 예측과 추출 상태로 확인한 관찰이며, 청취에 근거한 오류 원인 판정은 하지 않았다.\n\n"
    text += "F1 측정 가능 비율은 정확도나 발화 비율이 아니다. Softmax 값도 보정된 정답 확률이 아니다.\n\n"
    text += markdown_table(table) + "\n"
    (output/"validation_case_review.md").write_text(text,encoding="utf-8")


def generate(study, output):
    study, output = Path(study), Path(output)
    output.mkdir(parents=True,exist_ok=True)
    result = collect(study)
    if not len(result):
        raise ValueError("No completed runs")
    result.to_csv(output/"all_runs.csv",index=False)
    for filename in ["protocol.json","dataset_audit.json","scaler.json","environment.json",
                     "confirmation_selection.json","final_selection.json","noise_reference.json",
                     "preflight_support_repair.json","cache_recomputation_check.json",
                     "fairness_audit.csv","execution_audit.json"]:
        if (study/filename).exists():
            shutil.copy2(study/filename,output/filename)
    attribution = output/"noise_sources"
    attribution.mkdir(exist_ok=True)
    for filename in ["manifest.json","LICENSE.txt"]:
        shutil.copy2(study/"noise_sources"/filename,attribution/filename)
    screening = result[(result.seed==42)&(result.fusion=="middle")&(result.control_extra==0)].sort_values("set_id")
    screening.to_csv(output/"screening.csv",index=False)
    for row in result.itertuples():
        src = study/"runs"/row.run
        dst = output/"runs"/row.run
        dst.mkdir(parents=True,exist_ok=True)
        for filename in ["config.json","manifest.json","summary.json","history.csv","epochs.jsonl",
                         "val_stress_summary.json","test_stress_summary.json"]:
            if (src/filename).exists():
                shutil.copy2(src/filename,dst/filename)
        if (src/"validation").exists():
            shutil.copytree(src/"validation",dst/"validation",dirs_exist_ok=True)
        if (src/"evaluation/test_original").exists():
            shutil.copytree(src/"evaluation/test_original",dst/"test_original",dirs_exist_ok=True)
        for condition in (src/"evaluation").glob("val_*"):
            compact = dst/"noise_evaluation"/condition.name
            compact.mkdir(parents=True,exist_ok=True)
            for filename in ["metrics.json","predictions.csv","confusion_counts.csv"]:
                shutil.copy2(condition/filename,compact/filename)
    # Aggregate by original speech examples, not by treating noisy copies as new independent speakers.
    regular = result[(result.fusion=="middle")&(result.control_extra==0)]
    aggregates = regular.groupby("set_id").agg(
        seeds=("seed","count"),val_accuracy_mean=("val_accuracy","mean"),val_accuracy_std=("val_accuracy","std"),
        val_macro_f1_mean=("val_macro_f1","mean"),val_macro_f1_std=("val_macro_f1","std"),
        noisy_f1_mean=("mean_noisy_macro_f1","mean"),noisy_f1_std=("mean_noisy_macro_f1","std"))
    aggregates.to_csv(output/"seed_summary.csv")
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for row in screening.itertuples():
        h=pd.read_csv(study/"runs"/row.run/"history.csv")
        axes[0].plot(h.epoch,h.train_loss,label=f"set {row.set_id}")
        axes[1].plot(h.epoch,h.val_macro_f1,label=f"set {row.set_id}")
    axes[0].set(xlabel="epoch",ylabel="train CE loss")
    axes[1].set(xlabel="epoch",ylabel="original validation macro F1")
    axes[1].legend(ncol=2,fontsize=8)
    fig.tight_layout();fig.savefig(output/"learning_curves.png",dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    x=np.arange(len(screening))
    axes[0].bar(x-.18,screening.val_macro_f1,width=.36,label="original val")
    axes[0].bar(x+.18,screening.mean_noisy_macro_f1,width=.36,label="noisy val mean")
    axes[0].set(xticks=x,xticklabels=screening.set_id,ylim=(0,1),xlabel="set",ylabel="macro F1")
    axes[0].legend()
    axes[1].bar(x,screening.noise_only_false_keyword_rate)
    axes[1].set(xticks=x,xticklabels=screening.set_id,ylim=(0,1),xlabel="set",ylabel="noise-only false keyword rate")
    fig.tight_layout();fig.savefig(output/"screening_comparison.png",dpi=150);plt.close(fig)
    noise_rows=[]
    availability=[]
    for row in screening.itertuples():
        stress=json.loads((study/"runs"/row.run/"val_stress_summary.json").read_text())
        noise_rows += [dict(set_id=row.set_id,**v) for v in stress["conditions"]]
    noise=pd.DataFrame(noise_rows)
    noise.to_csv(output/"noise_by_condition.csv",index=False)
    if len(noise):
        table=noise.pivot(index="set_id",columns="condition",values="macro_f1")
        fig,ax=plt.subplots(figsize=(13,5))
        im=ax.imshow(table,vmin=0,vmax=1,cmap="viridis",aspect="auto")
        ax.set(yticks=np.arange(len(table)),yticklabels=table.index,
               xticks=np.arange(len(table.columns)),xticklabels=[c.replace("val_","") for c in table.columns],
               ylabel="set",title="Validation macro F1 by added-noise condition")
        plt.setp(ax.get_xticklabels(),rotation=50,ha="right")
        fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(output/"noise_heatmap.png",dpi=150);plt.close(fig)
    for p in sorted((study/"cache").glob("*/availability.csv")):
        if p.parent.name.startswith("test"):
            continue
        a=pd.read_csv(p);a.insert(0,"condition",p.parent.name);availability.append(a)
    pd.concat(availability,ignore_index=True).to_csv(output/"feature_availability.csv",index=False)
    feature_qa(study,output)
    baseline=pd.read_csv(study/"runs/set0_seed42_middle/validation/predictions.csv")
    error_rows=[]
    for row in screening.itertuples():
        pred=pd.read_csv(study/"runs"/row.run/"validation/predictions.csv")
        merged=baseline[["path","true_idx","pred_idx"]].rename(columns={"pred_idx":"baseline_pred"}).merge(
            pred[["path","pred_idx"]],on="path",validate="one_to_one")
        merged.insert(0,"set_id",row.set_id)
        merged["baseline_correct"]=merged.baseline_pred==merged.true_idx
        merged["candidate_correct"]=merged.pred_idx==merged.true_idx
        merged["effect"]=np.select([~merged.baseline_correct&merged.candidate_correct,
                                   merged.baseline_correct&~merged.candidate_correct],
                                  ["fixed_baseline_error","introduced_error"],default="unchanged")
        error_rows.append(merged)
    errors=pd.concat(error_rows,ignore_index=True)
    errors.to_csv(output/"paired_validation_predictions.csv",index=False)
    errors[errors.effect!="unchanged"].to_csv(output/"changed_validation_predictions.csv",index=False)
    validation_cases(study, output, result, screening)
    labels=json.loads((study/"protocol.json").read_text())["baseline_config"]["labels"]
    table=screening[["set_id","val_accuracy","val_macro_f1","mean_noisy_macro_f1","noise_only_false_keyword_rate","parameters"]].copy()
    for c in table.columns[1:5]:table[c]=table[c].map(lambda v:f"{v*100:.2f}%")
    table.columns=["세트","원본 val 정확도","원본 val macro F1","소음 val 평균 F1","잡음 단독 단어 오인율","파라미터 수"]
    text="# 보조 음향 특징 실험 결과\n\n"
    text+="학습은 고정 1.5초 WAV 그대로, 증강 OFF. **원본 validation macro F1이 주 평가·체크포인트 선택 기준**이다.\n\n"
    if (output/"interpretation.md").exists():
        text += "관찰한 효과와 한계: [결과 해석](interpretation.md).\n\n"
    if (study/"final_selection.json").exists():
        selection=json.loads((study/"final_selection.json").read_text())
        text += f"0~7세트 screening 8회, 추가 시드 6회, 결합 방식/용량 대조 2회로 총 {len(result)}회 학습했다. "
        if selection["selected_set"] == "0":
            text += "이번 설정에서는 보조 입력의 성능 개선을 확인하지 못해 **기준 모델(세트 0)을 유지**한다. "
        else:
            text += f"반복 검증 기준으로 **세트 {selection['selected_set']}를 선택**했다. "
        text += "이는 고정된 10-epoch 학습·한 명의 validation 화자에서 얻은 결론이다.\n\n"
    text += "아래 첫 표는 seed42의 결과다. 최종 선택에는 다음 반복 검증 표를 사용했다.\n\n"
    text+=markdown_table(table)+"\n\n"
    repeated = aggregates[aggregates.seeds >= 3].reset_index()
    if len(repeated):
        table = repeated[["set_id","seeds","val_accuracy_mean","val_macro_f1_mean","val_macro_f1_std","noisy_f1_mean"]].copy()
        for column in table.columns[2:]:
            suffix = "%p" if column == "val_macro_f1_std" else "%"
            table[column] = table[column].map(lambda v:f"{100*v:.2f}{suffix}")
        table.columns = ["세트","시드 수","원본 val 평균 정확도","원본 val 평균 F1","F1 시드 표준편차","소음 val 평균 F1"]
        text += "## 반복 검증\n\n" + markdown_table(table) + "\n\n"
        text += "시드는 42/43/44이며 표준편차는 같은 validation 화자에서의 학습 변동이다. 새 화자 간 불확실성이나 신뢰구간이 아니다.\n\n"
    controls = result[(result.fusion == "late") | (result.control_extra != 0)]
    if len(controls):
        table = controls[["run","val_accuracy","val_macro_f1","mean_noisy_macro_f1","parameters"]].copy()
        for column in ["val_accuracy","val_macro_f1","mean_noisy_macro_f1"]:
            table[column] = table[column].map(lambda v:f"{100*v:.2f}%")
        table.columns = ["대조군","원본 val 정확도","원본 val F1","소음 val 평균 F1","전체 파라미터 수"]
        text += "## 결합 방식과 용량 대조\n\n" + markdown_table(table) + "\n\n"
        text += "두 대조군은 seed42 한 번의 보조 비교다. late fusion의 전체 파라미터 수에는 forward에서 쓰지 않는 기존 fc의 5,643개도 포함된다. "
        text += "용량 대조군은 spectral-only residual이며 세트 3과 전체 파라미터 수가 32개 차이 난다.\n\n"
    text+="## 구성\n\n"+markdown_table(pd.DataFrame([dict(세트=k,추가입력=", ".join(v) or "없음") for k,v in SETS.items()]))+"\n\n"
    text+="## 해석과 한계\n\n"
    text+="- train 609 / val 203 / test 203, 화자 분리 유지. val은 한 화자이므로 새 화자 전체의 성능을 보장하지 않는다.\n"
    text+="- 203개에서 한 파일 차이는 정확도 약 0.49%p다. 반복 seed 평균/표준편차는 `seed_summary.csv`를 참고한다.\n"
    text+="- 소음 평가의 복사본들은 독립적인 새 화자/원본 예제가 아니다. 조건별 강건성 분석용이다.\n"
    text+="- 측정 불가는 오측정과 다르다. 포먼트/피치가 수치로 나와도 틀릴 수 있다. availability는 정답 품질이 아니다.\n"
    text+="- RMS <= 1e-8(-160 dBFS) 검사는 PCM16 원본에서는 사실상 디지털 0 분석 창 검사다. 약한 발화의 음량 문턱이나 포먼트의 물리적 부재 판정이 아니다.\n"
    text+="- 소음 SNR은 기존 녹음 full-clip 파워 대 추가 잡음이며, 원래 무잡음 음성의 SNR이 아니다.\n"
    text+="- silence/noise-only의 명목 소음 레벨은 train keyword RMS 중앙값 기준이다. 잡음 단독은 144개다.\n"
    text+="- 주 실험의 원래 ResNet 512→11 head와 학습 설정을 유지했다. 보조 경로 때문에 늘어난 용량은 capacity control로 추가 확인한다.\n\n"
    if (study/"final_selection.json").exists():
        selection=json.loads((study/"final_selection.json").read_text())
        text+=f"## 최종 선택 및 test\n\n반복 seed의 원본 val 평균 macro F1로 선택한 세트: **{selection['selected_set']}**. Test에 사용할 seed는 사전에 42로 고정했다.\n\n"
        test_rows=[]
        for p in sorted((study/"runs").glob("*/summary.json")):
            s=json.loads(p.read_text())
            if s.get("test_evaluated"):
                test_rows.append(dict(세트=s["set_id"],정확도=f"{s['test']['accuracy']*100:.2f}%",macro_F1=f"{s['test']['macro_f1']*100:.2f}%"))
        test_noise=[]
        for p in sorted((study/"runs").glob("*/test_stress_summary.json")):
            s=json.loads(p.read_text())
            test_noise += [dict(run=p.parent.name,**v) for v in s["conditions"]]
        if test_noise:
            pd.DataFrame(test_noise).to_csv(output/"test_noise_by_condition.csv",index=False)
        if test_rows:text+=markdown_table(pd.DataFrame(test_rows))+"\n\n"
        else:text+="최종 test 실행 전 상태.\n\n"
        if selection["selected_set"] == "0":
            text += "기준 모델 자체가 선택되어 보조 모델의 test는 열지 않았다. Test 점수는 이후 실험의 설정 선택에 사용하지 않는다.\n\n"
    text+="## 파일\n\n- `all_runs.csv`: 모든 반복/대조 실험의 지표·시간·GPU 메모리.\n- `noise_by_condition.csv`: 잡음 종류/SNR별 지표.\n- `feature_availability.csv`, `feature_qa.png`: 추출 상태와 원본 val 6개 예시.\n- `changed_validation_predictions.csv`: 기준에서 개선/악화된 파일.\n- `validation_case_review.md`, `validation_case_review.csv`: 단어별로 고른 오류 사례 20개의 관찰과 추출 상태. 청취 원인은 미확정.\n- `runs/`: 실행별 설정·소스 해시·학습 기록·혼동표·예측. 가중치/캐시는 로컬 `runs/auxiliary-v1/`.\n- `noise_sources/manifest.json`: ESC-50 고정 commit/출처/source ID/hash 및 라이선스.\n\n"
    text+="재현 명령과 상세 정의: [실험 코드 설명](../../experiments/auxiliary/README.md).\n\n"
    text+="클래스 ID: "+", ".join(f"{i}={v}" for i,v in enumerate(labels))+".\n"
    (output/"README.md").write_text(text,encoding="utf-8")
    print(output/"README.md")


if __name__=="__main__":
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--study",default="runs/auxiliary-v1")
    parser.add_argument("--output",default="reports/auxiliary-v1")
    args=parser.parse_args()
    generate(args.study,args.output)
