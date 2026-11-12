"""Aggregate the noise study without pooling correlated crops as new subjects."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.engine import file_hash, metrics, write_json

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/"runs/noise-stress-v1"
REPORT=ROOT/"reports/noise-stress-v1"
NAMES={"traffic":"도로 교차로","car":"차량 내부","cafeteria":"구내식당","restaurant":"식당",
       "station":"역","washing":"세탁기","compressor":"공기압축기","fan":"환풍기"}
KEYS=["heldout_speaker","model","condition","noise_id","category","snr_db","draw"]


def main():
    completion=json.loads((STUDY/"completed.json").read_text())
    protocol=json.loads((REPORT/"protocol.json").read_text())
    assert file_hash(REPORT/"protocol.json")==completion["protocol_sha256"]
    files=list((STUDY/"parts").glob("*.csv.gz"))
    predictions=pd.concat([pd.read_csv(p) for p in files if not p.name.endswith("_mix.csv.gz")],ignore_index=True)
    mixes=pd.concat([pd.read_csv(p,dtype={"crop_seed":str}) for p in files if p.name.endswith("_mix.csv.gz")],ignore_index=True)
    assert len(predictions)==1624+155904+5120
    assert len(mixes)==77952+2560
    assert set(predictions.heldout_speaker)==set(protocol["folds"])
    assert not predictions.path.str.contains("spk01").any()
    assert not predictions.duplicated(KEYS+["path"]).any()
    assert np.isfinite(mixes.noise_gain).all() and mixes.joint_attenuation.between(0,1).all()
    assert np.max(np.abs(mixes.snr_db-mixes.snr_db_measured))<1e-7
    units=[]
    for key,part in predictions.groupby(KEYS,sort=True):
        assert len(part)==(20 if key[2]=="noise_only" else 203)
        score=metrics(part.true_idx,part.pred_idx,11)
        command=part.true_idx.ne(10)
        silence=part.true_idx.eq(10)
        units.append(dict(zip(KEYS,key),n=len(part),**score,
                          keyword_accuracy=float((part[command].true_idx==part[command].pred_idx).mean()) if command.any() else None,
                          silence_false_command_rate=float(part[silence].pred_idx.ne(10).mean()) if silence.any() else None))
    units=pd.DataFrame(units)
    units.to_csv(REPORT/"condition_metrics.csv",index=False)
    clean=units[units.condition.eq("clean")].groupby("model")[["accuracy","macro_f1","keyword_accuracy"]].mean()
    mixed=units[units.condition.eq("mixed")]
    levels=mixed.groupby(["model","snr_db"])[["accuracy","macro_f1","keyword_accuracy","silence_false_command_rate"]].mean()
    noise=mixed.groupby(["model","noise_id","category","snr_db"])[["accuracy","macro_f1","keyword_accuracy","silence_false_command_rate"]].mean()
    folds=mixed.groupby(["model","heldout_speaker","snr_db"])[["accuracy","macro_f1"]].mean()
    families=mixed.groupby(["model","category","snr_db"])[["accuracy","macro_f1"]].mean()
    false=units[units.condition.eq("noise_only")].groupby(["model","noise_id","category","snr_db"])["silence_false_command_rate"].mean().to_frame()
    for name,frame in (("clean_summary",clean),("snr_summary",levels),("noise_summary",noise),
                       ("speaker_summary",folds),("family_summary",families),("noise_only_summary",false)):
        frame.to_csv(REPORT/(name+".csv"))
    predictions.sort_values(KEYS+["path"]).to_csv(REPORT/"predictions.csv.gz",index=False)
    mixes.sort_values(["heldout_speaker","noise_id","snr_db","path","draw"]).to_csv(REPORT/"mixtures.csv.gz",index=False)
    commands=predictions[predictions.condition.eq("mixed") & predictions.true_idx.ne(10)].copy()
    commands["correct"]=commands.true_idx.eq(commands.pred_idx)
    commands["missed_as_silence"]=commands.pred_idx.eq(10)
    commands["wrong_command"]=commands.true_idx.ne(commands.pred_idx) & commands.pred_idx.ne(10)
    commands.groupby(["model","noise_id","snr_db"])[["correct","missed_as_silence","wrong_command"]].mean().to_csv(REPORT/"keyword_error_modes.csv")

    fig,axs=plt.subplots(2,2,figsize=(13,10))
    x=[15,10,0,-5,-10]
    for model,color in (("imagenet","#d27c22"),("conformer","#226ac4")):
        y=[clean.loc[model,"macro_f1"]]+[levels.loc[(model,s),"macro_f1"] for s in (10,0,-5,-10)]
        axs[0,0].plot(x,np.array(y)*100,"o-",color=color,label=model)
    axs[0,0].set(xticks=x,xticklabels=["Original","10","0","-5","-10"],ylim=(0,102),ylabel="Macro-F1 (%)",
                 xlabel="Added-noise SNR (dB)",title="Mean of 4 folds x 8 sources x 3 crops")
    axs[0,0].invert_xaxis();axs[0,0].legend();axs[0,0].grid(alpha=.2)
    ids=list(NAMES)
    heat=np.array([[noise.loc[("conformer",n,category,s),"macro_f1"]*100 for s in (10,0,-5,-10)]
                   for n,category in ((v,next(src["category"] for src in protocol["noise_sources"] if src["noise_id"]==v)) for v in ids)])
    im=axs[0,1].imshow(heat,vmin=0,vmax=100,cmap="YlGnBu",aspect="auto")
    axs[0,1].set(yticks=range(8),yticklabels=ids,xticks=range(4),xticklabels=[10,0,-5,-10],xlabel="Added-noise SNR (dB)",title="Conformer macro-F1 (%)")
    for i in range(8):
        for j in range(4):axs[0,1].text(j,i,f"{heat[i,j]:.1f}",ha="center",va="center",color="white" if heat[i,j]>55 else "black")
    fig.colorbar(im,ax=axs[0,1])
    for i,(model,color) in enumerate((("imagenet","#d27c22"),("conformer","#226ac4"))):
        yy=[noise.xs((model,n),level=("model","noise_id")).xs(0,level="snr_db").iloc[0].macro_f1*100 for n in ids]
        axs[1,0].barh(np.arange(8)+(i-.5)*.36,yy,height=.36,label=model,color=color)
    axs[1,0].set(yticks=range(8),yticklabels=ids,xlim=(0,100),xlabel="Macro-F1 (%)",title="Equal original/noise power (0 dB)")
    axs[1,0].legend()
    for model,color in (("imagenet","#d27c22"),("conformer","#226ac4")):
        rate=false.xs(model).groupby("snr_db").silence_false_command_rate.mean()
        axs[1,1].plot([10,0,-5,-10],[rate.loc[s]*100 for s in (10,0,-5,-10)],"o-",color=color,label=model)
    axs[1,1].set(xticks=[10,0,-5,-10],ylim=(0,102),xlabel="Nominal level vs training keyword power (dB)",
                 ylabel="False command rate (%)",title="Noise only: 20 shared windows/source; no target speech")
    axs[1,1].invert_xaxis();axs[1,1].legend();axs[1,1].grid(alpha=.2)
    fig.tight_layout();fig.savefig(REPORT/"noise_stress.png",dpi=170);plt.close(fig)

    lines=["# 실제 환경 소음 스트레스 평가", "",
           "개발 화자 4-fold의 ResNet18과 미세조정 Conformer를 고정하고 실제 환경 소음 8종을 추가했다. "
           "새 학습이나 설정 탐색 없이 각 모델의 기존 최고 checkpoint를 사용했다.","",
           "| 조건 | ResNet macro-F1 | Conformer macro-F1 |", "| --- | ---: | ---: |",
           f"| 원본 | {clean.loc['imagenet','macro_f1']*100:.2f}% | {clean.loc['conformer','macro_f1']*100:.2f}% |"]
    for s in (10,0,-5,-10):
        lines.append(f"| {s}dB | {levels.loc[('imagenet',s),'macro_f1']*100:.2f}% | {levels.loc[('conformer',s),'macro_f1']*100:.2f}% |")
    lines += ["", "원본은 4개 fold 평균, 추가 소음은 4개 fold × 8개 원본 × 3회 구간의 macro-F1 평균이다. "
              "소음 원본마다 같은 가중치를 준다(도로 2·군중 3·기계 3). 명령어 10개와 silence의 11클래스 macro-F1이며, 전체 예측을 합쳐 계산한 F1과는 다르다.","",
              "Conformer가 모든 평균 조건에서 ResNet보다 높았지만, 강한 소음에서는 성능이 크게 떨어졌다. "
              "특히 식당과 공기압축기 녹음에 취약했다. 같은 −10dB라도 차량 내부 96.97%, 식당 3.31%로 차이가 커서 SNR만으로 난도를 설명하기 어렵다.","",
              "![소음 강도·종류별 비교](noise_stress.png)","", "## 소음별 Conformer macro-F1", "",
              "| 환경 | 원본 길이 | 10dB | 0dB | −5dB | −10dB |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for src in protocol["noise_sources"]:
        n=src["noise_id"];category=src["category"]
        values=" | ".join(f"{noise.loc[('conformer',n,category,s),'macro_f1']*100:.2f}%" for s in (10,0,-5,-10))
        lines.append(f"| {NAMES[n]} | {src['seconds']/60:.1f}분 | {values} |")
    lines += ["", "## 음성 없이 소음만 넣었을 때", "",
              "정답을 silence로 두고 명령어로 반응한 비율을 측정했다. 소음마다 20개 구간을 뽑아 4개 fold 모델에 동일하게 넣었다. "
              "각 fold의 훈련 키워드 600개 RMS 중앙값을 기준으로 잡음 수준을 맞췄다.","",
              "| 명목 수준 | ResNet 오반응률 | Conformer 오반응률 |", "| --- | ---: | ---: |"]
    for s in (10,0,-5,-10):
        vals=[false.xs(m).xs(s,level="snr_db").silence_false_command_rate.mean()*100 for m in ("imagenet","conformer")]
        lines.append(f"| {s}dB | {vals[0]:.2f}% | {vals[1]:.2f}% |")
    lines += ["", "## 평가와 해석", "",
              "- 긴 녹음에서 연속 1.5초를 파일별 3회 무작위 추출했다. 모든 모델과 SNR에 같은 구간을 사용했다. 경로·소음 ID·반복 번호로 추출 시드를 정한다.",
              "- SNR은 원본 전체 1.5초의 파워와 추가 잡음 파워의 비다. 기존 배경음·패딩을 포함하므로 깨끗한 발화만의 SNR은 아니다. 0dB는 두 파워가 같고, −10dB는 추가 잡음 파워가 10배다. silence는 훈련 키워드 RMS 중앙값을 기준으로 한 명목 SNR이다.",
              "- 잡음의 DC만 제거하고 거의 무음인 구간은 재추출했다. 클리핑을 막을 때 혼합 전체에 같은 선형 감쇠를 적용해 SNR을 유지했다. 이에 따른 절대 음량 변화도 현재 시스템 평가에 포함된다.",
              "- 군중 녹음에는 배경 사람 음성이 포함된다. 목표 발화는 기존 파일의 명령어이며 배경 발화를 별도 명령으로 주석하지 않았다. 배경 언어도 통제하지 않았다.",
              "- 원본 8개에서 구간을 반복한 결과다. 수만 개 혼합이 수만 개 독립 환경을 뜻하지 않는다. fold의 학습 화자도 겹친다. 학습 시드 42 하나의 개발 평가다.",
              "- 혼합 평가로 거리·잔향·마이크 왜곡·실제 환경의 비선형 녹음까지 재현하지는 못한다. unknown 단어도 평가하지 않았다.",
              "- 이번에는 test spk01을 불러오거나 추론하지 않았다. Conformer의 최종 test 결과는 아직 없다. 프로젝트 전체에서는 이전 jaehun의 선택된 모델이 test를 평가한 이력이 있다.",
              "- 이 소음 원본 8개 전체를 개발 평가용으로 예약했다. 향후 증강 학습과 최종 test용 소음에는 다른 원본을 사용한다. 같은 녹음의 다른 구간이나 마이크는 독립 원본으로 취급하지 않는다.","",
              "실제 명령어가 있는 −10dB 입력에서 Conformer는 33.69%를 맞히고, 46.92%를 silence로 놓쳤으며, "
              "19.39%를 다른 명령어로 혼동했다. [명령어 오류 유형](keyword_error_modes.csv)에 소음별 비율을 저장했다.","",
              "## 직접 듣기", "",
              "미리 정한 spk02의 ‘출발’ 1개에 첫 번째 추출 구간을 혼합한 예시다. 결과를 보고 예시를 고르지 않았다. "
              "현재 서버의 캐시 파일 링크이며, 듣기 파일에 혼합 후 별도 음량 정규화를 적용하지 않았다.","",
              "[원본](../../runs/noise-stress-v1/examples/original.wav)","",
              "| 소음 | 10dB | 0dB | −10dB |", "| --- | --- | --- | --- |",
              *[f"| {NAMES[n]} | [듣기](../../runs/noise-stress-v1/examples/{n}_10dB.wav) | [듣기](../../runs/noise-stress-v1/examples/{n}_0dB.wav) | [듣기](../../runs/noise-stress-v1/examples/{n}_-10dB.wav) |" for n in ("traffic","cafeteria","compressor","fan")],
              "", "## 출처와 재현", "",
              "[DEMAND](https://zenodo.org/records/1227121): Joachim Thiemann, Nobutaka Ito, Emmanuel Vincent. "
              "16kHz 원본의 ch01만 사용했다. 설명은 CC BY-SA 3.0, 현재 API 필드는 CC BY 4.0으로 달라 둘 다 기록했다.","",
              "[RPi 환경 녹음](https://zenodo.org/records/7493820): Gaurav Govilkar, Kader BT Shaikh, N Gopalkrishnan. "
              "API에 CC BY 4.0이 명시돼 있다. 공기압축기 60분·환풍기 30분 원본을 mono 16k로 변환했다. "
              "원본을 보존하고 평가용 사본을 만들었다. 두 자료 모두 원배포 MD5와 준비 파일 SHA256을 확인했다.","",
              "[고정 프로토콜](protocol.json) · [소음 출처·해시](noise_sources.json) · [SNR별 결과](snr_summary.csv) · "
              "[소음별 결과](noise_summary.csv) · [화자별 결과](speaker_summary.csv) · [조건별 지표](condition_metrics.csv) · "
              "[전체 파일 예측](predictions.csv.gz) · [추출위치·혼합해시](mixtures.csv.gz)","",
              "```bash", "OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 runs/conformer-env/bin/python -m experiments.noise_stress.sources",
              "OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 runs/conformer-env/bin/python -m experiments.noise_stress.run",
              "runs/conformer-env/bin/python -m experiments.report_noise_stress",
              "OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 runs/conformer-env/bin/python -m experiments.check_noise_stress", "```", "",
              "모델·평가 캐시·듣기 예시는 `runs/noise-stress-v1/`, 실행 로그는 `execution.log`에 있다. "
              "완료 조건은 재개 시 건너뛰며 동결 프로토콜이 달라지면 거부한다.","",
              "[집계 검사](aggregation_audit.json) · [독립 재추론 검사](independent_replay.json) · [검증 기록](VALIDATION.md)",""]
    (REPORT/"README.md").write_text("\n".join(lines))
    write_json(REPORT/"aggregation_audit.json",dict(prediction_rows=len(predictions),mixture_rows=len(mixes),
               clean_rows=1624,mixed_rows=155904,noise_only_rows=5120,conditions=len(units),
               mixed_conditions=768,source_count=8,fold_count=4,seed_count=1,
               max_snr_error_db=float(np.max(np.abs(mixes.snr_db-mixes.snr_db_measured))),
               jointly_attenuated_fraction=float(mixes.joint_attenuation.lt(1).mean()),
               checkpoint_and_source_pins_match=all(file_hash(ROOT/p)==h for p,h in {**protocol["checkpoint_and_metadata_sha256"],**protocol["source_sha256"]}.items()),
               test_evaluated=False,protocol_sha256=file_hash(REPORT/"protocol.json")))
    print("SUMMARY",levels.to_string(),flush=True)
    print("NOISE_ONLY",false.groupby(["model","snr_db"]).mean().to_string(),flush=True)


if __name__=="__main__":main()
