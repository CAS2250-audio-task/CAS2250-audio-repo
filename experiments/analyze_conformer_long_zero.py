"""Compare completed long runs against identical prior development cases."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import engine
from experiments.conformer_long_zero.run import ROOT,STUDY,REPORT,branch
from experiments.conformer_long_zero.policy import update_count

KEYS=["heldout_speaker","kind","noise_id","snr_db","draw","path"]


def metrics(frame):
    """Equal-weight noise/draw/fold macroF1, matching preceding reports."""
    units=[]
    for keys,g in frame.groupby(["heldout_speaker","kind","noise_id","draw"],sort=False):
        units.append(dict(speaker=keys[0],kind=keys[1],noise_id=keys[2],draw=int(keys[3]),
            **engine.metrics(g.true_idx,g.pred_idx,11)))
    unit=pd.DataFrame(units)
    zero=unit[unit.kind.eq("mixed")]
    clean=unit[unit.kind.eq("clean")]
    noisy=frame[frame.kind.eq("mixed")]
    words=noisy[noisy.true_idx.ne(10)]
    only=frame[frame.kind.eq("noise_only")]
    result=dict(zero_f1=float(zero.macro_f1.mean()),zero_accuracy=float(zero.accuracy.mean()),
        clean_f1=float(clean.macro_f1.mean()),noise_only_fpr=float(only.pred_idx.ne(10).mean()),
        keyword_accuracy=float(words.pred_idx.eq(words.true_idx).mean()),
        keyword_silence_miss=float(words.pred_idx.eq(10).mean()),
        keyword_other_wrong=float((words.pred_idx.ne(10)&words.pred_idx.ne(words.true_idx)).mean()))
    return result,unit


def analyze():
    assert (STUDY/"complete.json").exists(),"Training and independent audit must finish first"
    assert (REPORT/"independent_audit.json").exists()
    choices=json.loads((REPORT/"lr_selection.json").read_text())
    prior=pd.read_csv(ROOT/"reports/conformer-noise-focus-v1/predictions.csv.gz")
    prior=prior[prior.training_condition.isin(("baseline","speech_focus")) &
        (prior.kind.eq("clean")|prior.snr_db.eq(0))].copy()
    collections=[]
    for (condition,denoiser),frame in prior.groupby(["training_condition","denoiser"]):
        collections.append((dict(condition=condition,stage="previous_clean_best",batch=32,denoiser=denoiser),frame))
    for condition in ("baseline","speech_focus"):
        for denoiser in ("off","dfn3"):
            frame=pd.concat([pd.read_csv(STUDY/"source_epoch10"/f"{condition}_{s}_{denoiser}.csv.gz")
                             for s in ("spk02","spk03","spk04","spk05")],ignore_index=True)
            collections.append((dict(condition=condition,stage="source_epoch10",batch=32,denoiser=denoiser),frame))
    for choice in choices:
        for selected in ("best","last"):
            for denoiser in ("off","dfn3"):
                frame=pd.concat([pd.read_csv(branch(choice["condition"],choice["batch"],choice["encoder_lr"],s)/
                    f"{selected}_{denoiser}_full_predictions.csv.gz") for s in ("spk02","spk03","spk04","spk05")],ignore_index=True)
                collections.append((dict(condition=choice["condition"],stage="long_"+selected,
                    batch=choice["batch"],denoiser=denoiser),frame))
    comparisons=[];splits=[];noise_rows=[];speaker_rows=[];error_rows=[]
    reference_inputs={};source_predictions={}
    for info,frame in collections:
        assert len(frame)==20940 and not frame.path.str.contains("spk01_").any()
        assert not frame.duplicated(KEYS).any()
        identity=frame.set_index(KEYS).true_idx.sort_index()
        identity_key=(info["condition"],info["denoiser"])
        if identity_key in reference_inputs:
            assert identity.equals(reference_inputs[identity_key])
        else:reference_inputs[identity_key]=identity
        result,units=metrics(frame)
        comparisons.append({**info,**result})
        zero=units[units.kind.eq("mixed")]
        assert len(zero)==96
        for split,draws in (("monitor_draw0",(0,)),("additional_draw1_2",(1,2))):
            subset=zero[zero.draw.isin(draws)]
            splits.append(dict(**info,draw_subset=split,zero_f1=float(subset.macro_f1.mean()),
                metric_units=len(subset),interpretation="same development speakers/noise recordings; not independent test"))
        for noise,g in zero.groupby("noise_id"):
            noise_rows.append(dict(**info,noise_id=noise,zero_f1=float(g.macro_f1.mean()),zero_accuracy=float(g.accuracy.mean())))
        for speaker,g in frame.groupby("heldout_speaker"):
            m,_=metrics(g)
            speaker_rows.append(dict(**info,speaker=speaker,**m))
        if info["stage"]=="source_epoch10":source_predictions[identity_key]=frame
        if info["stage"]=="long_best":
            joined=frame.merge(source_predictions[identity_key],on=KEYS,suffixes=("_new","_source"),validate="one_to_one")
            joined=joined[joined.kind.eq("mixed")]
            assert joined.true_idx_new.eq(joined.true_idx_source).all()
            old_ok=joined.pred_idx_source.eq(joined.true_idx_source)
            new_ok=joined.pred_idx_new.eq(joined.true_idx_new)
            error_rows.append(dict(**info,number_of_mixed_cases=len(joined),
                wrong_to_correct=int((~old_ok&new_ok).sum()),correct_to_wrong=int((old_ok&~new_ok).sum()),
                both_wrong=int((~old_ok&~new_ok).sum()),both_correct=int((old_ok&new_ok).sum()),
                count_unit="repeated speaker clip x noise recording x crop, not unique utterances"))
    compare=pd.DataFrame(comparisons)
    for i,row in compare.iterrows():
        match=compare.condition.eq(row.condition)&compare.denoiser.eq(row.denoiser)
        for stage,label in (("previous_clean_best","previous_clean_best"),("source_epoch10","source_epoch10")):
            value=float(compare[match&compare.stage.eq(stage)].zero_f1.iloc[0])
            compare.loc[i,"zero_f1_delta_vs_"+label]=float(row.zero_f1)-value
    compare.to_csv(REPORT/"comparison.csv",index=False)
    pd.DataFrame(splits).to_csv(REPORT/"draw_split_summary.csv",index=False)
    pd.DataFrame(noise_rows).to_csv(REPORT/"noise_summary.csv",index=False)
    pd.DataFrame(speaker_rows).to_csv(REPORT/"speaker_comparison.csv",index=False)
    pd.DataFrame(error_rows).to_csv(REPORT/"paired_error_changes.csv",index=False)
    common=[];endpoints=[]
    for choice in choices:
        histories=[]
        for speaker in ("spk02","spk03","spk04","spk05"):
            run=branch(choice["condition"],choice["batch"],choice["encoder_lr"],speaker)
            history=pd.read_csv(run/"history.csv");history["speaker"]=speaker;histories.append(history)
            summary=json.loads((run/"summary.json").read_text())
            endpoints.append(dict(condition=choice["condition"],batch=choice["batch"],speaker=speaker,
                best_epoch=summary["best_epoch"],final_epoch=summary["final_epoch"],
                best_updates=update_count(summary["best_epoch"],choice["batch"]),
                final_updates=update_count(summary["final_epoch"],choice["batch"]),
                sampled_clips_at_best=609*summary["best_epoch"],sampled_clips_at_end=609*summary["final_epoch"]))
        history=pd.concat(histories,ignore_index=True)
        for epoch in (10,15,20,25,30):
            group=history[history.epoch.eq(epoch)]
            assert len(group)==4
            common.append(dict(condition=choice["condition"],batch=choice["batch"],epoch=epoch,
                updates=update_count(epoch,choice["batch"]),zero_f1=float(group.zero_f1.mean()),
                zero_loss=float(group.zero_loss.mean()),clean_f1=float(group.clean_f1.mean()),
                scope="fixed draw0 monitor; same609 sampled clips per epoch, different number of optimizer updates"))
    pd.DataFrame(common).to_csv(REPORT/"common_epoch_summary.csv",index=False)
    pd.DataFrame(endpoints).to_csv(REPORT/"endpoint_table.csv",index=False)

    # Means with individual speaker points show the spread hidden by a single average.
    plotted=compare[compare.stage.isin(("previous_clean_best","long_best"))]
    configurations=[("previous_clean_best","baseline",32),("previous_clean_best","speech_focus",32),
                    ("long_best","baseline",32),("long_best","baseline",16),
                    ("long_best","speech_focus",32),("long_best","speech_focus",16)]
    labels=["old/base","old/expanded","long/base/B32","long/base/B16","long/expanded/B32","long/expanded/B16"]
    fig,axes=plt.subplots(1,3,figsize=(17,5))
    speakers=pd.DataFrame(speaker_rows)
    for ax,metric,title,limits in zip(axes,("zero_f1","clean_f1","noise_only_fpr"),
        ("0 dB macro-F1","Original macro-F1","Noise-only false activation"),((50,100),(90,100.5),(0,35))):
        for denoiser,color,offset in (("off","tab:blue",-.12),("dfn3","tab:orange",.12)):
            values=[]
            for x,(stage,condition,batch) in enumerate(configurations):
                sel=(plotted.stage.eq(stage)&plotted.condition.eq(condition)&plotted.batch.eq(batch)&plotted.denoiser.eq(denoiser))
                values.append(float(plotted[sel][metric].iloc[0])*100)
                g=speakers[speakers.stage.eq(stage)&speakers.condition.eq(condition)&speakers.batch.eq(batch)&speakers.denoiser.eq(denoiser)]
                ax.scatter(np.full(len(g),x+offset),g[metric]*100,s=15,alpha=.25,color=color)
            ax.plot(np.arange(len(configurations))+offset,values,"o",color=color,label=denoiser,markersize=7)
        ax.set(xticks=range(len(labels)),xticklabels=labels,title=title,ylabel="%",ylim=limits)
        ax.tick_params(axis="x",rotation=35);ax.grid(alpha=.2)
    axes[0].axhline(90,color="gray",ls="--",lw=.8)
    axes[0].legend()
    fig.suptitle("4 development speakers, all 3 crops at 0 dB; small points = individual speakers")
    fig.tight_layout();fig.savefig(REPORT/"final_comparison.png",dpi=160)
    svg=REPORT/"final_comparison.svg"
    fig.savefig(svg)
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines())+"\n")
    plt.close(fig)

    lines=["# 0dB 기준 연장 학습 최종 비교", "",
        "기존 10에포크의 원본 검증 최고본, 재개 출발점인 10에포크 마지막 모델, 0dB 기준 연장 학습 최고본을 같은 4화자·8소음·3구간에서 비교했다.",
        "에포크 수와 함께 학습률 스케줄, 최고본 선택 기준도 바뀌었다. 배치 16은 에포크당 업데이트가 더 많다. 아래 점수 차이는 각 설정을 함께 적용한 결과다.","",
        "| 학습 | 배치 | 0dB F1 | +DFN3 | 원본 F1 | 원본+DFN3 | 소음단독 오반응 | +DFN3 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    names={"baseline":"기존 증강","speech_focus":"말소리·0dB 강화"}
    for stage,condition,batch in [("previous_clean_best","baseline",32),("previous_clean_best","speech_focus",32),
        ("source_epoch10","baseline",32),("source_epoch10","speech_focus",32),*configurations[2:]]:
        group=compare[compare.stage.eq(stage)&compare.condition.eq(condition)&compare.batch.eq(batch)].set_index("denoiser")
        a,b=group.loc["off"],group.loc["dfn3"]
        title=("10에포크 원본 최고본 · " if stage=="previous_clean_best" else "10에포크 마지막 · " if stage=="source_epoch10" else "연장 최고본 · ")+names[condition]
        lines.append(f"| {title} | {batch} | {a.zero_f1*100:.2f}% | {b.zero_f1*100:.2f}% | {a.clean_f1*100:.2f}% | {b.clean_f1*100:.2f}% | {a.noise_only_fpr*100:.2f}% | {b.noise_only_fpr*100:.2f}% |")
    lines += ["", "![최종비교](final_comparison.png)", "",
        "## 종료 체크포인트의 보조 비교", "",
        "종료 시점은 무필터 0dB 추세와 100에포크 상한으로 정했다. 아래는 그 종료본을 평가한 결과다.", "",
        "| 학습 | 배치 | 0dB F1 | +DFN3 |", "|---|---:|---:|---:|"]
    for condition,batch in (("baseline",32),("baseline",16),("speech_focus",32),("speech_focus",16)):
        group=compare[compare.stage.eq("long_last")&compare.condition.eq(condition)&compare.batch.eq(batch)].set_index("denoiser")
        lines.append(f"| {names[condition]} | {batch} | {group.loc['off','zero_f1']*100:.2f}% | {group.loc['dfn3','zero_f1']*100:.2f}% |")
    observed=compare.loc[compare.zero_f1.idxmax()]
    lines += ["", f"전체 비교표의 최고 관측값은 {names[observed.condition]}·배치 {observed.batch}·{observed.stage}·{observed.denoiser}의 **{observed.zero_f1*100:.2f}%**다. 이 값도 개발 데이터 비교 결과이며 DFN3 기준으로 에포크를 별도 최적화한 실험은 아니다.", "",
        "첫 표의 ‘연장 최고본’은 DFN3 없이 고정된 첫 구간의 0dB F1로 선택했다. DFN3 열은 그 동일한 모델에 필터를 적용한 결과다.",
        "배치 32·16 모두 한 에포크에 609개를 추출하며, 업데이트는 각각 20·39회다. 분기마다 종료 에포크도 다르다. [같은 에포크 비교](common_epoch_summary.csv)와 [누적 업데이트 수](endpoint_table.csv)를 함께 확인할 수 있다.","",
        "[선택 구간과 추가 두 구간](draw_split_summary.csv)은 같은 개발 화자·소음 원본의 다른 구간이다. 이번 LR·에포크 선택에는 첫 구간만 사용했으며, 추가 구간도 독립 시험셋은 아니다. 이번 실험에서 test01 오디오는 읽거나 추론하지 않았다.","",
        "[화자별 결과](speaker_comparison.csv), [소음별 결과](noise_summary.csv), [출발점 대비 정답·오답 변화](paired_error_changes.csv), [모든 비교 지표](comparison.csv). 정답·오답 개수는 반복 혼합 평가 건수이며 고유 발화 개수가 아니다.","",
        "[학습곡선과 종료 후 회복 해석](README.md), [독립 검사](independent_audit.json), [검증·재개 방법](VALIDATION.md)."]
    (REPORT/"RESULTS.md").write_text("\n".join(lines)+"\n")
    print(compare.to_string(index=False))


if __name__=="__main__":analyze()
