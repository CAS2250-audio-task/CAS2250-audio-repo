"""Render partial/final long0dB results without changing frozen training code."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/"reports/conformer-long-zero-v1"
STUDY=ROOT/"runs/conformer-long-zero-v1"


def render():
    REPORT.mkdir(exist_ok=True)
    progress=json.loads((REPORT/"progress.json").read_text()) if (REPORT/"progress.json").exists() else {}
    if (STUDY/"complete.json").exists():progress["stage"]="complete"
    choices=json.loads((REPORT/"lr_selection.json").read_text()) if (REPORT/"lr_selection.json").exists() else []
    lines=["# 0dB 기준 연장 학습·배치16/32·학습률 비교", "",
        f"상태: `{progress.get('stage','preparing')}`. pilot {progress.get('pilot_completed',0)}/48, 긴 학습 {progress.get('long_completed',0)}/16 완료.","",
        "기존 noise와 speech_focus의 epoch10 마지막 모델·AdamW 모멘텀·sampler/RNG에서 이어서 학습한다. 원래10epoch cosine을 연장하지 않고 epoch11부터 고정LR을 쓴다.",
        "배치32/16 모두609draw/epoch이며 업데이트는20/39번이다. 첫10epoch는 모두batch32로200update였다. 배치마다LR을 선택하므로 최종비교는 튜닝된 설정 간 비교다.","",
        "LR 후보: 인코더1e-6/3e-6/1e-5, head는50배. 각조건·배치에서4fold의epoch13~15 평균0dB F1로 선택한다(동점은낮은손실→낮은LR).",
        "매epoch0dB 모니터는 평가8원본×고정draw0×203개/화자(1,624혼합)이며 원본203개도 별도로 기록한다. 최고epoch는 DFN없는0dB F1로 고른다.",
        "최종은 선택best·마지막last·시작epoch10 모두 기존전체3draw 0dB·원본·잡음단독과 DFN3 off/on으로 평가한다. 추가draw도개발평가이며독립test가아니다.","",
        "과적합 표시: 최소25epoch 이후3epoch평균에서 과거최저검증손실 대비 검증손실≥3%증가·훈련손실≥2%감소·과거최고F1대비≥0.5%p하락이5회연속이면확인시점을기록하고5epoch더남긴다. 단순정체로중단하지않는다. 최대100epoch에서도조건이안나오면‘과적합미확인’이다.","",
        "0dB 소음과 개발화자02~05를 epoch/LR 선택에 사용한다. 아래점수는튜닝에사용한개발결과다. test01오디오/추론은사용하지않는다. 기존실험파일은보존한다.","",
        "실시간 상태: [progress.json](progress.json), [실행 로그](../../runs/conformer-long-zero-v1/execution.log), [고정 설정](protocol.json).",""]
    pilots=[]
    for path in sorted((STUDY/"branches").glob("*/*/*/*/pilot_history.csv")):
        data=pd.read_csv(path)
        data["condition"]=path.parents[3].name
        data["batch"]=int(path.parents[2].name.removeprefix("bs"))
        data["candidate_lr"]=float(path.parents[1].name.removeprefix("lr"))
        data["speaker"]=path.parent.name
        pilots.append(data)
    if pilots:
        pilot=pd.concat(pilots,ignore_index=True)
        pilot.to_csv(REPORT/"pilot_progress.csv",index=False)
        fig,axes=plt.subplots(2,2,figsize=(12,8))
        for ax,(condition,batch) in zip(axes.flat,[(c,b) for c in ("baseline","speech_focus") for b in (32,16)]):
            subset=pilot[pilot.condition.eq(condition)&pilot.batch.eq(batch)]
            for lr,g in subset.groupby("candidate_lr"):
                m=g.groupby("epoch",as_index=False).zero_f1.mean()
                ax.plot(m.epoch,m.zero_f1*100,label=f"LR{lr:.0e}, n={g.speaker.nunique()}/4 folds")
            ax.set(title=f"{condition} / batch{batch}",xlabel="epoch",ylabel="0 dB F1, draw0 (%)")
            ax.grid(alpha=.2)
            if len(subset):ax.legend(fontsize=8)
        fig.tight_layout();fig.savefig(REPORT/"pilot_curves.png",dpi=150);plt.close(fig)
        lines += ["## 짧은 학습률 비교", "", "![LR후보pilot](pilot_curves.png)", "",
            "진행중에는완료화자수가후보마다다를수있으므로범례n/4를확인한다. 최종LR은모두4fold가끝난뒤선택한다.", ""]
    if choices:
        table=pd.DataFrame(choices)
        table.to_csv(REPORT/"selected_learning_rates.csv",index=False)
        lines += ["## 선택 학습률", "", "| 증강 | 배치 | 인코더LR | headLR | pilot 0dB F1 |", "|---|---:|---:|---:|---:|"]
        for r in choices:lines.append(f"| {r['condition']} | {r['batch']} | {r['encoder_lr']:.0e} | {r['encoder_lr']*50:.1e} | {r['score']*100:.2f}% |")
        lines += ["", "pilot F1은4fold×마지막3epoch 평균이며 전체3draw 최종평가와 구간수가 다르다.", ""]
    histories=[]
    for choice in choices:
        prefix=f"{choice['condition']}_bs{choice['batch']}_lr{choice['encoder_lr']:.0e}_"
        for p in sorted((REPORT/"histories").glob(prefix+"*.csv")):
            data=pd.read_csv(p);data["speaker"]=p.stem.split("_")[-1]
            data["condition"],data["batch"]=choice["condition"],choice["batch"]
            histories.append(data)
    if histories:
        history=pd.concat(histories,ignore_index=True)
        history.to_csv(REPORT/"selected_history.csv",index=False)
        fig,axes=plt.subplots(2,3,figsize=(16,8))
        specs=[("epoch","zero_f1","0 dB F1 (fixed draw0)",100),
               ("epoch","zero_loss","0 dB validation loss",1),
               ("epoch","train_loss","Training loss (augmented)",1),
               ("updates","zero_f1","0 dB F1 vs cumulative updates",100),
               ("epoch","clean_f1","Original validation F1",100),
               ("epoch","clean_loss","Original validation loss",1)]
        means=[]
        for (condition,batch),frame in history.groupby(["condition","batch"]):
            counts=frame.groupby("epoch").speaker.nunique()
            # Never silently change the speaker denominator in a mean curve.
            usable=frame[frame.epoch.isin(counts[counts.eq(4)].index)]
            mean=usable.groupby("epoch",as_index=False).mean(numeric_only=True)
            mean["condition"],mean["batch"]=condition,int(batch);means.append(mean)
            for ax,(x,y,title,scale) in zip(axes.flat,specs):
                ax.plot(mean[x],mean[y]*scale,label=f"{condition} / B{int(batch)}")
                ax.set(xlabel=x,ylabel=title,title=title);ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=8)
        fig.tight_layout();fig.savefig(REPORT/"learning_curves.png",dpi=150);plt.close(fig)
        pd.concat(means,ignore_index=True).to_csv(REPORT/"learning_curve_means.csv",index=False)
        lines += ["## 학습곡선", "", "![4fold가모두있는epoch만평균](learning_curves.png)", "",
            "평균곡선은4fold가모두있는epoch만사용한다. 진행중이거나조기종료된fold가있으면개별곡선을함께본다. epoch10의훈련손실은이전실험값이다.", ""]
        fig,axes=plt.subplots(3,4,figsize=(18,11),sharex=True)
        for j,speaker in enumerate(sorted(history.speaker.unique())):
            for (condition,batch),frame in history[history.speaker.eq(speaker)].groupby(["condition","batch"]):
                frame=frame.sort_values("epoch");label=f"{condition} B{int(batch)}"
                axes[0,j].plot(frame.epoch,frame.zero_f1*100,label=label)
                axes[1,j].plot(frame.epoch,frame.train_loss,label=label)
                axes[2,j].plot(frame.epoch,frame.zero_loss,label=label)
                best=frame.loc[frame.zero_f1.idxmax()]
                axes[0,j].plot(best.epoch,best.zero_f1*100,"x",color=axes[0,j].lines[-1].get_color())
            axes[0,j].set(title=speaker,ylabel="0 dB F1 (%)")
            axes[1,j].set(ylabel="Augmented training loss")
            axes[2,j].set(xlabel="epoch",ylabel="0 dB validation loss")
        axes[0,0].legend(fontsize=7)
        fig.tight_layout();fig.savefig(REPORT/"speaker_curves.png",dpi=150);plt.close(fig)
        lines += ["![화자별곡선](speaker_curves.png)", "",
            "화자별 전체 에포크의 0dB F1·증강 훈련 손실·0dB 검증 손실이다. ×는 최고 F1 지점이다. 훈련과 검증의 입력 분포가 다르므로 손실 절댓값의 차이보다 각 곡선의 변화 방향을 확인한다.", ""]
    summaries=[json.loads(p.read_text()) for p in sorted((REPORT/"runs").glob("*.json"))]
    source_rows=[]
    for p in sorted((REPORT/"source_epoch10").glob("*.json")):
        item=json.loads(p.read_text())
        for denoiser,metrics in item["results"].items():
            source_rows.append(dict(condition=item["condition"],speaker=item["speaker"],denoiser=denoiser,**metrics))
    if source_rows:
        sources=pd.DataFrame(source_rows)
        sources.to_csv(REPORT/"source_epoch10.csv",index=False)
        lines += ["## 이어 학습하기 전 epoch10", "",
            "이전공식표는원본검증clean-best였고여기는epoch10 LAST다. 시작점차이와연장효과를구분한다.", "",
            "| 증강 | 필터 | 전체0dB F1 | 원본 F1 |", "|---|---|---:|---:|"]
        for (condition,denoiser),frame in sources.groupby(["condition","denoiser"]):
            if frame.speaker.nunique()==4:
                lines.append(f"| {condition} | {denoiser} | {frame.zero_f1.mean()*100:.2f}% | {frame.clean_f1.mean()*100:.2f}% |")
        lines.append("")
    if summaries:
        rows=[]
        diagnostics=[]
        for r in summaries:
            name=f"{r['condition']}_bs{r['batch']}_lr{r['encoder_lr']:.0e}_{r['speaker']}"
            history=pd.read_csv(REPORT/"histories"/(name+".csv"))
            confirmed=r["stop"]["confirmation_epoch"]
            new_peak=False
            if confirmed is not None:
                before=history[history.epoch.le(confirmed)]
                after=history[history.epoch.gt(confirmed)]
                new_peak=len(after)>0 and float(after.zero_f1.max())>float(before.zero_f1.max())
            note=("탐지후F1최고점재갱신" if new_peak else
                  "후속창에서악화신호비지속" if confirmed is not None and r["stop"]["latest_bad_streak"]==0 else
                  "악화신호관측" if confirmed is not None else "지속과적합미확인")
            diagnostics.append(dict(condition=r["condition"],batch=r["batch"],speaker=r["speaker"],
                confirmation_epoch=confirmed,new_f1_peak_after_confirmation=bool(new_peak),
                final_bad_streak=r["stop"]["latest_bad_streak"],interpretation=note))
            for pipeline,m in r["final_results"].items():
                rows.append(dict(condition=r["condition"],batch=r["batch"],encoder_lr=r["encoder_lr"],
                    speaker=r["speaker"],pipeline=pipeline,final_epoch=r["final_epoch"],
                    stop_reason=r["stop"]["reason"],overfit_confirmation_epoch=r["stop"]["confirmation_epoch"],**m))
        results=pd.DataFrame(rows);results.to_csv(REPORT/"runs.csv",index=False)
        pd.DataFrame(diagnostics).to_csv(REPORT/"overfit_diagnostics.csv",index=False)
        lines += ["## 전체3draw 결과(완료된4fold만 평균)", "",
            "| 증강 | 배치 | 체크포인트/필터 | 0dB F1 | 원본 F1 | 잡음단독 오반응 |", "|---|---:|---|---:|---:|---:|"]
        groups=[]
        for (condition,batch,pipeline),frame in results.groupby(["condition","batch","pipeline"]):
            if frame.speaker.nunique()!=4:continue
            means={k:float(frame[k].mean()) for k in ("zero_f1","zero_loss","clean_f1","noise_only_fpr")}
            groups.append(dict(condition=condition,batch=int(batch),pipeline=pipeline,**means))
            lines.append(f"| {condition} | {int(batch)} | {pipeline} | {means['zero_f1']*100:.2f}% | {means['clean_f1']*100:.2f}% | {means['noise_only_fpr']*100:.2f}% |")
        pd.DataFrame(groups).to_csv(REPORT/"summary.csv",index=False)
        lines += ["", "best는DFN없는고정draw0로고른동일checkpoint이며off/DFN3를각자최적epoch로고르지않는다. last는중단시점이다.", ""]
        lines += ["## 종료 시점", "", "| 증강 | 배치 | 화자 | 최고epoch | 마지막epoch | 악화신호탐지epoch | 후속 관찰 |", "|---|---:|---|---:|---:|---:|---|"]
        for r,d in zip(summaries,diagnostics):lines.append(f"| {r['condition']} | {r['batch']} | {r['speaker']} | {r['best_epoch']} | {r['final_epoch']} | {r['stop']['confirmation_epoch']} | {d['interpretation']} |")
        lines += ["", "탐지는미리정한운영상종료기준이다. 후속5epoch에서회복하거나새최고점을찍을수있으므로탐지epoch를‘이후계속과적합’의증거로단정하지않는다. 후속회복여부와전체곡선을함께해석한다.", ""]
    if (REPORT/"independent_audit.json").exists():lines += ["", "독립검사: [independent_audit.json](independent_audit.json)."]
    if (REPORT/"RESULTS.md").exists():lines += ["", "최종기준비교·추가구간확인: [RESULTS.md](RESULTS.md)."]
    (REPORT/"README.md").write_text("\n".join(lines)+"\n")


if __name__=="__main__":render()
