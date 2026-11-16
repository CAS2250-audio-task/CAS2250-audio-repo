# 최종 분류기 실험 인수인계 — 먼저 읽을 문서

2026-11-16. **실제 학습을 마친16모델 결과**와원코드·입력·가중치의전달문서다. 후속실험은아직실행하지않았다.

## 바로 내릴 수 있는 판단

- 현재 실제검출기연결의 우선후보: **Mean+보조72.79%, Mean70.89%**. Pooled67.08%는대조로남긴다.
- 수동경계oracle에서는Mean84.33%, Mean+보조83.10%. 이점수를현실파이프라인의배포성능으로쓰지않는다.
- 기존 실제학습Mean+보조를고정하고입력만oracle crop으로교체하면72.79→82.67%. 입력crop개선의우선순위가높다.
- Oracle학습Mean+보조를실제crop에넣으면67.39%다. Oracle전용모델을그대로배포하지않는다.
- 현재세로CNN/lowfocus추가격자는우선보류한다. 이후예산은최소Mean/Mean+보조/Pooled대조의동일예산확인·반복seed/view검사에우선배정할수있다. 새학습을중복실행하지않는다.

## 이번 두 단계의 결과

**0dB 명령어10종 macroF1을8소음별로계산후평균**,각모델200epoch의선택best다.

| 구조 | 실제검출crop | 수동경계oracle |
|---|---:|---:|
| Mean+보조 |72.79%|83.10%|
| Mean |70.89%|84.33%|
| Pooled |67.08%|77.13%|
| 축분리+Pooled |65.76%|79.37%|
| Pooled+보조 |64.70%|78.33%|
| 2구역+세로attention |56.18%|65.56%|
| 2구역CNN |52.04%|64.02%|
| 2구역+세로attention+lowfocus |50.39%|58.10%|

전체보고서와소음/epoch/FPR표는[RESULTS_KO.md](final-classifier-experiment-20261116/RESULTS_KO.md),[actual_vs_oracle.csv](final-classifier-experiment-20261116/actual_vs_oracle.csv),[all_results.csv](final-classifier-experiment-20261116/all_results.csv),[교차평가](final-classifier-experiment-20261116/cross_evaluation.csv),[완료검증](final-classifier-experiment-20261116/completion_audit.json).

## 공통 검출/crop 계약 — 사용자 확정

1. v2 미래문맥 단독lookahead/best.pt, **epoch152**. checkpoint SHA256=`0626b8c1573db6b68fb0adb4340c573238733cd4307939ee637f158e14452055`.
2. EOF에**마지막실제mel16frame(80ms)**유지. 정규화0으로채우지않는다. 샘플별실제길이기준이며마지막점수frame을버리지않는다.
3. 기존speech decoder(.6/.4,20/50ms확인후backdate). boundary head로구간생성하지않는다.
4. 복수검출은**첫구간시작~마지막구간끝**으로감싼다. 가장긴구간선택/정답위치선택이아니다.
5. margin **앞50ms/뒤30ms**를원본실제샘플에서확보한다. wav길이고정/zero padding과구분한다.
6. 미검출은삭제하지않고**silence**로고정한다. 양성실패도지표에포함한다.
7. raw detected_segments와실제crop_segments/samplebounds·margin·정책·checkpoint식별을보존한다.

## 데이터·학습·해석 범위

- D3 manual-clipped-v1 정리원본. S03/04/05학습600,S02검증200,S01음원미접근.
- 화자대응:S01=spk01,S02=spk03,S03=spk02,S04=spk05,S05=spk04. S03(여성화자)는이번학습에들어있다. 여성미학습일반화평가가아니다.
- 기존훈련6+ESC88=94소음원본. 기존검증8은학습미사용. 새audit22는이번분류기결과평가에사용하지않았다.
- 학습600reference/noisy쌍+200noise-only쌍. 고정view bank/seed20261116. 소음family는명령어순서와독립적인균형random permutation.
- noisy합성은전체원본에먼저수행,발화부전력기준SNR[-5,20]uniform. 검출/crop후그실제noisy파형에서mel/보조추출.
- 화자증강identity. 검출기쪽검증구현을받기전중복구현하지않았다.
- 두단계3480개합성전체파형SHA가같다. 바뀐것은경계/gate와해당학습reference에서다시fit한scaler다.
- seed42,32pairs/64views,paired CE+0.1same_command SupCon/temp.1,silence CE only.
- AdamW lr.001→.00001cosine1000,WD.0001,clip5. **실제200epoch/고정계획1000**. 200cosine완주가아니다.
- 모든CNN중간출력mask,유효frame만nativeBN,aux는maskedGN,Mean/Pooled/Self의유효frame집계. 고정길이기존출력과학습padding/gradient불변성검사통과.
- 초기모델/프로젝터tensor를공유했다. Oracle의보조scaler buffer는새학습reference에다시맞췄다.
- 검증1880=원본200+8소음0dB1600+noise-only80. 선택과평가에같은bank를사용했다. 독립test가아니다.
- 기록30,080예측의F1재계산과16checkpoint/200epoch/양서버공통코드·입력·초기값검증을통과했다.

## Oracle 주의

사용자가제공한clip경로는`/root/CAS2250-detector-diversity-v2/datasets/task2/manual-clipped-v1/clips/`다.
이서버에는그경로가없어같은원격데이터의개발clip800개를가져왔고현재실험원본PCM과대조해offset0을확인했다.
제공clip에는이미30/30ms여유가있다. 이를중복확장하지않고CSV의speech_*_in_trimmed_s에서원본50/30ms를다시추출했다.
Oracle의noise-only구간없음→silence/FPR0은정의상결과이며,학습된잡음거절능력개선이아니다.
Oracle는사용자수동경계를이상적검출기로근사한진단조건이며,독립음소정렬정답이나증명된최대성능이아니다.

## 원코드와 입력

- [실제검출 학습 릴리스](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/releases/tag/classifier-post-detector-v1): `classifier-study-v1-source-nogh.tar.gz`, `classifier-study-v1-inputs.tar.gz`와각sha256. gh없는환경은수정source-nogh를쓴다.
- [Oracle 학습 릴리스](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/releases/tag/classifier-oracle-v1): `classifier-oracle-v1-source.tar.gz`, `classifier-oracle-v1-inputs.tar.gz`와각sha256.
- 최종배정:실제local4/Jupyter4,oraclelocal6/Jupyter2. 각모델전체학습이며epoch분할없음.
- 묶음내`study/train.py`가실제분류기학습기다. `compare_margins.py`만실행하는것은분류기학습이아니다.
- `bank/metadata.json`과SOURCE_SHA256/INPUT_SHA256,initialization.pt,scaler를함께보존한다. 현재bank는고정view이므로과거매epoch재합성과다르다.
- `study/models.make_model`+`state['model']`로읽고저장mel_scaler/aux_scaler를그대로쓴다. forward에는실제frame lengths가필요하다. 모델부분GPU시간만을전체Burg/파형처리포함지연으로쓰지않는다.
- 향후학습연장시같은코드/입력/프로토콜의last.pt로resume하고runtime stop-after만변경할수있다. 새bank/새seed는별도run으로기록한다.

## 권장 checkpoint 위치

### 실제 검출 Mean+보조

- 0dB command10 F1:72.79%,best epoch196.
- [archive](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/releases/download/classifier-post-detector-v1/classifier-study-local-e200-results.tar.gz)와`.sha256`를다운로드한다.
- archive내경로:`results/local/mean_aux/best.pt`.
- best.pt SHA256:`8f892ebff1b2c981e44679d7eca3db486ce713862508675e599a294be6e7bd16`.

### 실제 검출 Mean

- 0dB command10 F1:70.89%,best epoch150.
- [archive](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/releases/download/classifier-post-detector-v1/classifier-study-jupyter-e200-results.tar.gz)와`.sha256`를다운로드한다.
- archive내경로:`results/jupyter/mean/best.pt`.
- best.pt SHA256:`b6771be566873e1c488cc13ef8ed26929e3882e56b1a6150606cb9862ad2b447`.

### 실제 검출 Pooled 기준

- 0dB command10 F1:67.08%,best epoch170.
- [archive](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/releases/download/classifier-post-detector-v1/classifier-study-local-e200-results.tar.gz)와`.sha256`를다운로드한다.
- archive내경로:`results/local/base/best.pt`.
- best.pt SHA256:`cfdf7168b5fc0d03452ec4e655d0e494ecaadda83beadd8b78729984cb200be6`.

### Oracle Mean

- 0dB command10 F1:84.33%,best epoch184.
- [archive](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/releases/download/classifier-oracle-v1/classifier-study-local-e200-results.tar.gz)와`.sha256`를다운로드한다.
- archive내경로:`results/local/mean/best.pt`.
- best.pt SHA256:`489c6037c2dcb390ec4bf8204da467c1e567584a64b4f9adbe8571bbbfdffb45`.

## 전체 프로젝트 0dB 최고 기록과 구분

이번실험점수를과거D1/전체1.5초/500~1000epoch/11종macroF1과섞지않는다.
확인된완료표에서 **개발4fold평균raw 최고는 사전학습Conformer+약한포먼트/formant_B+VAD연동attention gate의91.6418%**다.
약3212만params,100epoch,0dB11종macroF1이며별도대형모델계열이다. 새v2정밀검출기방식이아니다.
이조건은기존채택기준+0.5%p에미달해서선택되지는않았고,실제채택기준선formant_B는91.2974%다.
단일화자점수를4fold평균과섞어전체최고라고쓰지않는다.
근거:`reports/attention-pitch-v1/summary.csv`,`selection.json`,`fold_results.csv`(attention-pitch-v1브랜치),`reports/overnight-v1/summary.csv`.
소형무사전학습과거D1/500epoch계열은frequency_attn+lowfocus87.68%,axis+standard87.56%(spk03)였으며그차이로확정우열을주장하지않는다.

한seed·개발한화자·고정view의현재선별결과로성별일반화완성/전체testSOTA/경계하나의독립인과효과를주장하지않는다.
