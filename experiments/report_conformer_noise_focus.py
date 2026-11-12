"""Separate speech-source, near-zero SNR, and fixed DFN3 effects in development CV."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import engine
from experiments.conformer_noise_focus.sources import ROOT,STUDY,REPORT
from experiments.conformer_noise_focus.run import CONDITIONS

TRAINING=("baseline",*CONDITIONS)
NAMES={"baseline":"기존 소음 혼합","speech":"말소리 소음 추가","focus":"0dB 부근 비중 증가","speech_focus":"둘 다 적용"}


def main():
    p=pd.read_csv(REPORT/"predictions.csv.gz")
    keys=["training_condition","denoiser","heldout_speaker","kind","noise_id","category","snr_db","draw"]
    assert len(p)==650592 and not p.duplicated(keys+["path"]).any()
    units=[]
    for key,g in p.groupby(keys):
        assert len(g)==(20 if key[3]=="noise_only" else 203)
        words=g[g.true_idx.ne(10)];silence=g[g.true_idx.eq(10)]
        units.append(dict(zip(keys,key),**engine.metrics(g.true_idx,g.pred_idx,11),n=len(g),
            keyword_accuracy=float(words.true_idx.eq(words.pred_idx).mean()) if len(words) else 0.,
            keyword_silence_rate=float(words.pred_idx.eq(10).mean()) if len(words) else 0.,
            false_command_rate=float(silence.pred_idx.ne(10).mean())))
    units=pd.DataFrame(units)
    assert len(units)==4128
    units.to_csv(REPORT/"condition_metrics.csv",index=False)
    clean=units[units.kind.eq("clean")].groupby(["training_condition","denoiser"])[["accuracy","macro_f1"]].mean()
    snr=units[units.kind.eq("mixed")].groupby(["training_condition","denoiser","snr_db"])[["accuracy","macro_f1","keyword_accuracy","keyword_silence_rate","false_command_rate"]].mean()
    false=units[units.kind.eq("noise_only")].groupby(["training_condition","denoiser","snr_db"]).false_command_rate.mean()
    noise=units[units.kind.eq("mixed")].groupby(["training_condition","denoiser","noise_id","snr_db"])[["accuracy","macro_f1"]].mean()
    speaker=units[units.kind.eq("mixed")].groupby(["training_condition","denoiser","heldout_speaker","snr_db"])[["accuracy","macro_f1"]].mean()
    summary=snr.xs(0,level="snr_db").rename(columns={"accuracy":"mixed_accuracy","macro_f1":"mixed_macro_f1"}).join(
        clean.rename(columns={"accuracy":"original_accuracy","macro_f1":"original_macro_f1"})).join(
        false.xs(0,level="snr_db").rename("noise_only_false_command_rate"))
    summary["delta_vs_existing_same_pipeline_pp"]=[(row.mixed_macro_f1-summary.loc[("baseline",idx[1]),"mixed_macro_f1"])*100
                                                   for idx,row in summary.iterrows()]
    for name,value in (("summary",summary),("clean_summary",clean),("snr_summary",snr),
                       ("noise_only_summary",false),("noise_summary",noise),("speaker_summary",speaker)):
        value.to_csv(REPORT/(name+".csv"))
    assert abs(summary.loc[("baseline","off"),"mixed_macro_f1"]-.7847484723252243)<1e-12
    assert abs(summary.loc[("baseline","dfn3"),"mixed_macro_f1"]-.8379844921800265)<1e-12
    traces=[];runs=[];histories=[]
    protocol=json.loads((REPORT/"protocol.json").read_text())
    for condition in TRAINING:
        for spk in protocol["folds"]:
            run=ROOT/f"runs/conformer-aug-v1/folds/{spk}/noise" if condition=="baseline" else STUDY/"folds"/spk/condition
            result=json.loads((run/"summary.json").read_text())
            trace=pd.concat([pd.read_csv(run/"epochs"/f"augmentation_{e:02d}.csv.gz") for e in range(1,11)])
            assert len(trace)==6090
            trace["training_condition"]=condition;trace["heldout_speaker"]=spk
            if condition=="baseline":
                trace["added_speech_selected"]=False;trace["near_zero_selected"]=False
            traces.append(trace)
            h=pd.read_csv(run/"history.csv");h["training_condition"]=condition;h["heldout_speaker"]=spk;histories.append(h)
            runs.append(dict(training_condition=condition,heldout_speaker=spk,best_epoch=result["best_epoch"],
                original_train_f1=result["original_train"]["macro_f1"],original_validation_f1=result["validation"]["macro_f1"],
                checkpoint_sha256=result["checkpoint_sha256"],source_run=str(run.relative_to(ROOT))))
    pd.DataFrame(runs).to_csv(REPORT/"runs.csv",index=False)
    trace=pd.concat(traces,ignore_index=True)
    assert len(trace)==97440
    trace.to_csv(REPORT/"training_draws.csv.gz",index=False)
    rates=[]
    for condition,g in trace.groupby("training_condition"):
        n=g[g.noise_selected]
        rates.append(dict(training_condition=condition,draws=len(g),original_fraction=float((~g.noise_selected).mean()),
            noise_fraction=float(g.noise_selected.mean()),added_speech_fraction=float(g.added_speech_selected.mean()),
            near_zero_component_fraction=float(g.near_zero_selected.mean()),
            snr_le_zero_fraction=float(n.snr_db.le(0).sum()/len(g)),
            within_2db_fraction=float(n.snr_db.between(-2,2).sum()/len(g))))
    pd.DataFrame(rates).to_csv(REPORT/"augmentation_rates.csv",index=False)
    curves=pd.concat(histories).groupby(["training_condition","epoch"])[["train_loss","val_loss","val_macro_f1"]].mean()
    curves.to_csv(REPORT/"learning_curve_means.csv")
    # All eight pipelines, no hidden selection of a denoiser per noise recording.
    fig,axes=plt.subplots(1,3,figsize=(16,4.8))
    for condition in TRAINING:
        for denoiser,style in (("off","-"),("dfn3","--")):
            idx=(condition,denoiser)
            y=[clean.loc[idx,"macro_f1"]]+[snr.loc[(*idx,s),"macro_f1"] for s in (10,0,-5,-10)]
            axes[0].plot(range(5),np.array(y)*100,"o"+style,label=condition+" / "+denoiser)
            axes[1].plot(range(4),[false.loc[(*idx,s)]*100 for s in (10,0,-5,-10)],"o"+style)
        c=curves.loc[condition]
        axes[2].plot(c.index,c.val_macro_f1*100,label=condition)
    axes[0].set(xticks=range(5),xticklabels=["Original","10","0","-5","-10"],ylim=(0,102),ylabel="Macro-F1 (%)",title="Training policy and fixed DFN3 off/on")
    axes[1].set(xticks=range(4),xticklabels=["10","0","-5","-10"],ylim=(0,102),ylabel="False command rate (%)",title="Noise-only input")
    axes[2].set(xlabel="Epoch",ylabel="Original validation macro-F1 (%)",title="Same clean-best checkpoint rule")
    axes[0].legend(fontsize=7);axes[2].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(REPORT/"comparison.png",dpi=170);plt.close(fig)
    lines=["# 말소리 소음 확대·0dB 학습 비중·DFN3 비교", "",
        "사용자 요청에 따라 기존 원본/넓은 SNR 학습을 유지하면서 말소리 소음과0dB 부근 입력 비중을 늘렸다. "
        "두 변경을 각각/함께 적용한 모델을 새로 학습하고, 기존 소음 혼합 모델과 모두 DFN3 off/on으로 비교했다.","",
        "## 0dB 결과", "", "| 학습 조건 | DFN3 | 원본 F1 | 0dB F1 | 기존 동일 경로 대비 | 잡음 단독 오반응 |",
        "| --- | --- | ---: | ---: | ---: | ---: |"]
    for condition in TRAINING:
        for denoiser in ("off","dfn3"):
            r=summary.loc[(condition,denoiser)]
            lines.append(f"| {NAMES[condition]} | {denoiser} | {r.original_macro_f1*100:.2f}% | {r.mixed_macro_f1*100:.2f}% | {r.delta_vs_existing_same_pipeline_pp:+.2f}%p | {r.noise_only_false_command_rate*100:.2f}% |")
    best=summary.mixed_macro_f1.idxmax()
    lines += ["",f"이번 개발 검증의 0dB 평균 macro-F1 최고는 **{NAMES[best[0]]} / {best[1]}** "
        f"({summary.loc[best,'mixed_macro_f1']*100:.2f}%)다. 원본 성능과 잡음 단독 오반응도 함께 판단한다. "
        "가장 높은 한 수치만으로 실제 운용 조건을 확정하지 않는다.","",
        "이번 비율에서는 새 학습3조건 모두0dB F1이 기존 동일 경로보다 높아지지 않았다. "
        "말소리 추가+DFN3는83.74%로 기존+DFN3의83.80%와 비슷했고, 잡음 단독 오반응은12.81→10.16%로 낮았다. "
        "DFN3 없이 말소리만 추가하면 실제 명령어 무음 누락이3.16→8.77%로 늘어 F1 하락과 동반됐다. "
        "기본 학습 조건을 새 결합 조건으로 자동 교체하지 않았다.","",
        "![SNR·오반응·학습곡선](comparison.png)","",
        "원본은 개발4-fold 평균, 혼합 점수는 fold4×검증 소음 원본8×추출3의 11-class macro-F1을 같은 가중치로 평균했다. "
        "서로 다른 발화 원본은812개이며 여러 소음 조건으로 반복 평가했다. 이번 비교에 test01 오디오를 읽거나 추론하지 않았다.","",
        "## 0dB의 명령어 누락", "",
        "| 학습 조건 | DFN3 | 실제 명령어 정답률 | 실제 명령어를 무음으로 놓침 |", "| --- | --- | ---: | ---: |"]
    for condition in TRAINING:
        for denoiser in ("off","dfn3"):
            r=summary.loc[(condition,denoiser)]
            lines.append(f"| {NAMES[condition]} | {denoiser} | {r.keyword_accuracy*100:.2f}% | {r.keyword_silence_rate*100:.2f}% |")
    lines += ["", "잡음 단독 오반응 감소와 실제 명령어 누락 증가가 함께 나타나는지 구분한다. "
        "현재 수치만으로 필터나 증강이 개선됐다고 한 방향으로 판단하지 않는다. "
        "누락은 실제 명령어 입력 중 silence로 예측한 비율이며, 잡음 단독 입력의 오반응과 다른 지표다.","",
        "## 학습 분포", "",
        "- 모든 조건의 원본 입력 확률50% 유지. 소음 확률50%,609draw/epoch,10epoch로 학습량도 같다.",
        "- 기존: 소음일 때SNR U[-5,20]dB. 훈련 원본6개에서 균등 선택.",
        "- 말소리 추가: 소음 draw의50%는 기존6개,50%는 추가 말소리 파일. 웅성거림/옆 사람 말소리/방송 유형을 균등 선택한 뒤 파일 균등 선택.",
        "- 0dB 비중 증가: 소음 draw의50%는 기존U[-5,20],50%는U[-3,3]. 전체 입력으로는 원본50/넓은SNR25/0dB부근25%의 혼합 분포.",
        "- 둘 다: 말소리 선택과SNR 선택을 독립 적용. 기존과 같은 학습 샘플·소음 적용 여부·기본 소음/SNR 랜덤 스트림을 짝지었다.",
        "- 총학습량 고정 비교이므로 추가 말소리/near-zero가 일부 기존 소음 draw를 대체한다. 기존 소음 노출 횟수까지 보존한 추가 학습의 효과는 따로 측정하지 않았다.",
        "- ref의16kHz·1.5초, 초기 한국어ASR 인코더/동일seed42 head, LR1e-5/5e-4,WD1e-4,encoder dropout.1,cosine10epoch 유지. 속도·마스킹·보조 특징 없음.",
        "- epoch는 기존과 같은 원본 validation F1로 선택. 각 학습 조건에서 DFN3 off/on에 같은 checkpoint를 사용했다. DFN3는 훈련에서 사용하지 않았으며, 고정된 범용 전처리의 추론 효과만 비교했다.","",
        "실제 적용 비율은 [augmentation_rates.csv](augmentation_rates.csv), 모든 훈련 draw는 [training_draws.csv.gz](training_draws.csv.gz)에 기록했다.","",
        "## SNR별 F1", "", "| 학습 조건 | DFN3 | 10dB | 0dB | −5dB | −10dB |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for condition in TRAINING:
        for denoiser in ("off","dfn3"):
            values=[snr.loc[(condition,denoiser,s),"macro_f1"] for s in (10,0,-5,-10)]
            lines.append("| "+NAMES[condition]+" | "+denoiser+" | "+" | ".join(f"{v*100:.2f}%" for v in values)+" |")
    lines += ["", "## 자료 분리와 해석 범위", "",
        "추가 파일은 [공식 MS-SNSD](https://github.com/microsoft/MS-SNSD)의train Babble/NeighborSpeaking/AirportAnnouncements37개,약45.75분이다. "
        "MS-SNSD에 포함된 CafeTeria/Restaurant/Station 등의DEMAND 이름변경 원본은 사용하지 않았다. Git revision/blob과원본/준비파일/PCM 해시를 고정했다. "
        "번호가 다른 파일들의 녹음 세션 독립성은 확인되지 않았으므로37개 독립 원본 세션이라고 주장하지 않는다.","",
        "검증8원본과 파일명·원본/준비해시가 겹치지 않으며, 처음/중간/끝8초를 상관도 검색해 뚜렷한 재인코딩 복제를 점검했다. "
        "높은 저주파 상관 사례는 음성 대역에서 재확인했다. 이 검사가 모든 부분 중복이나 외부 사전학습 자료 중복 부재를 증명하지는 않는다.","",
        "기존 혼합80,512개는 SHA256까지 일치한다. 원본812개를 포함한 입력81,324개에 학습4조건×DFN3 off/on,총650,592예측이다. "
        "이전DFN3 원본/0dB 출력20,940개를 그대로 재사용했다. DFN3 입력은 혼합 음성뿐이며 깨끗한 원본/분리된 잡음/라벨을 주지 않았다.","",
        "0dB는 원래1.5초 전체 파형 파워 대비 추가 잡음이다. silence/잡음 단독은 fold 훈련600개 명령어 RMS 중앙값 기준이다. "
        "DFN3는 공식48k 왕복 리샘플·지연 보정을 유지했고, 임의 피크 정규화/클리핑/원본과의 혼합을 추가하지 않았다.","",
        "훈련seed1개·개발 화자4명에서 조건을 고른 개발 결과다. 최종test나새녹음 일반화를 확정하지 않는다. "
        "프로젝트 과거 브랜치의 test 사용 이력과 이번 실행의 test 미사용은 구분한다.","",
        "원본 validation F1로 epoch를 선택하므로0dB의 최적 epoch를 직접 선택한 실험은 아니다. "
        "0dB 비중과 새로운 소음의 적정 비율·필요 학습량도 이번 한 설정으로 확정하지 않는다. "
        "추가 배경 음성의 전사와 동일 한국어 키워드 포함 여부는 검수하지 않았다.","",
        "[고정 프로토콜](protocol.json) · [소음별 결과](noise_summary.csv) · [화자별 결과](speaker_summary.csv) · "
        "[독립 검사](independent_audit.json) · [검증·재개 방법](VALIDATION.md) · [듣기 샘플](../../runs/conformer-noise-focus-v1/listening/README.md)",""]
    (REPORT/"README.md").write_text("\n".join(lines),encoding="utf-8")
    print("FOCUS_REPORT",summary.to_string(),flush=True)


if __name__=="__main__":main()
