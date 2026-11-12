"""Report common epoch100 endpoints, preserving the original study."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/"reports/conformer-fixed100-v1"
STUDY=ROOT/"runs/conformer-fixed100-v1"


def render():
    REPORT.mkdir(parents=True,exist_ok=True)
    status=json.loads((REPORT/"progress.json").read_text()) if (REPORT/"progress.json").exists() else {}
    protocol=json.loads((REPORT/"protocol.json").read_text())
    extended=sum(i["extend"] for i in protocol["plan"])
    lines=["# 조기 종료 없이 100epoch까지 연장", "",
        f"상태: {status.get('stage','prepared')}, 결과 정리 {status.get('completed',0)}/16분기.","",
        f"이전 조기 종료 {extended}분기를 마지막 checkpoint의 모델·AdamW·RNG에서 이어 학습한다. 이미100epoch인 {len(protocol['plan'])-extended}분기는 재사용한다.",
        "변경은 조기 종료 제거 하나다. 학습률·배치·증강·609draw/epoch·고정 검증 파형은 유지한다.",
        "최고본 선택은 기존과 같은 무필터0dB 고정draw0 F1이며 이전 최고본도 후보에 포함한다. 마지막 모델은 모두100epoch다.",
        "최종 평가는 원본·0dB 전체3draw·잡음단독, DFN3 off/on. 개발 검증 결과이며 test01은 사용하지 않는다.","",
        "[진행상태](progress.json) · [실행 로그](../../runs/conformer-fixed100-v1/execution.log) · [고정 설정](protocol.json)",""]
    histories=[]
    for item in protocol["plan"]:
        name=f"{item['condition']}_bs{item['batch']}_lr{item['encoder_lr']:.0e}_{item['speaker']}"
        path=REPORT/"histories"/(name+".csv")
        if not path.exists():path=ROOT/item["source_run"]/"history.csv"
        frame=pd.read_csv(path);frame["condition"]=item["condition"];frame["batch"]=item["batch"];frame["speaker"]=item["speaker"]
        histories.append(frame)
    history=pd.concat(histories,ignore_index=True);history.to_csv(REPORT/"selected_history.csv",index=False)
    fig,axes=plt.subplots(3,4,figsize=(18,11),sharex=True)
    for j,speaker in enumerate(("spk02","spk03","spk04","spk05")):
        for (condition,batch),frame in history[history.speaker.eq(speaker)].groupby(["condition","batch"]):
            frame=frame.sort_values("epoch");label=f"{condition} B{batch}"
            line=axes[0,j].plot(frame.epoch,frame.zero_f1*100,label=label)[0]
            color=line.get_color()
            axes[1,j].plot(frame.epoch,frame.train_loss,color=color)
            axes[2,j].plot(frame.epoch,frame.zero_loss,color=color)
            old=next(i for i in protocol["plan"] if i["speaker"]==speaker and i["condition"]==condition and i["batch"]==batch)
            best=frame.loc[frame.zero_f1.idxmax()]
            axes[0,j].plot(best.epoch,best.zero_f1*100,"x",color=color)
            if old["extend"]:
                point=frame[frame.epoch.eq(old["source_epoch"])].iloc[0]
                axes[0,j].plot(point.epoch,point.zero_f1*100,"o",mfc="none",color=color)
        axes[0,j].set(title=speaker,ylabel="0 dB F1, draw0 (%)")
        axes[1,j].set(ylabel="Augmented training loss")
        axes[2,j].set(xlabel="epoch",ylabel="0 dB validation loss",xlim=(10,100))
    axes[0,0].legend(fontsize=7)
    fig.tight_layout();fig.savefig(REPORT/"speaker_curves.png",dpi=150);plt.close(fig)
    lines += ["![화자별 학습곡선](speaker_curves.png)","", "○는 이전 종료점, ×는 현재까지 최고 F1이다. 미완료 분기는 진행된 epoch까지 표시한다.",""]
    summaries=[json.loads(p.read_text()) for p in sorted((REPORT/"runs").glob("*.json"))]
    rows=[];endpoints=[]
    for s in summaries:
        old=next(i for i in protocol["plan"] if i["condition"]==s["condition"] and i["batch"]==s["batch"] and i["speaker"]==s["speaker"])
        endpoints.append(dict(condition=s["condition"],batch=s["batch"],speaker=s["speaker"],
            old_end_epoch=old["source_epoch"],old_best_epoch=old["source_best_epoch"],new_best_epoch=s["best_epoch"],new_end_epoch=s["final_epoch"],extended=s["extended"]))
        for pipeline,metrics in s["final_results"].items():
            row=dict(condition=s["condition"],batch=s["batch"],speaker=s["speaker"],pipeline=pipeline,**metrics)
            if "clean_accuracy" not in row:
                units=pd.read_csv(ROOT/s["run_dir"]/f"{pipeline}_full_units.csv")
                row["clean_accuracy"]=float(units[units.kind.eq("clean")].accuracy.iloc[0])
            rows.append(row)
    pd.DataFrame(endpoints).to_csv(REPORT/"endpoints.csv",index=False)
    results=pd.DataFrame(rows);results.to_csv(REPORT/"runs.csv",index=False)
    comparisons=[]
    if len(results):
        lines += ["| 증강 | 배치 | 모델/필터 | 원본 F1 | 원본 정확도 | 0dB F1 | 0dB 정확도 | 이전 대비0dB F1 |", "|---|---:|---|---:|---:|---:|---:|---:|"]
        old=pd.read_csv(ROOT/"reports/conformer-long-zero-v1/summary.csv")
        for (condition,batch,pipeline),g in results.groupby(["condition","batch","pipeline"]):
            if g.speaker.nunique()!=4:continue
            metrics={k:float(g[k].mean()) for k in ("clean_f1","clean_accuracy","zero_f1","zero_accuracy","zero_loss","noise_only_fpr")}
            previous=old[old.condition.eq(condition)&old.batch.eq(batch)&old.pipeline.eq(pipeline)].iloc[0]
            metrics["zero_f1_delta_pp"]=100*(metrics["zero_f1"]-previous.zero_f1)
            comparisons.append(dict(condition=condition,batch=int(batch),pipeline=pipeline,**metrics))
            lines.append(f"| {condition} | {batch} | {pipeline} | {metrics['clean_f1']*100:.2f}% | {metrics['clean_accuracy']*100:.2f}% | {metrics['zero_f1']*100:.2f}% | {metrics['zero_accuracy']*100:.2f}% | {metrics['zero_f1_delta_pp']:+.2f}%p |")
    pd.DataFrame(comparisons).to_csv(REPORT/"summary.csv",index=False)
    lines += ["", "표는 4fold가 모두 완료된 조건만 표시한다. 추가 학습으로 최고점 탐색 범위도 늘어나므로 개발 점수 상승이 독립 test 개선을 보장하지 않는다.",""]
    (REPORT/"README.md").write_text("\n".join(lines)+"\n")


if __name__=="__main__":render()
