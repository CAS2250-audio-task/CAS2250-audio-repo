# 실제 검출 crop / 수동 경계 oracle 분류기 비교 완료

**결론:** 현재 실제 검출 조건의 우선 후보는 Mean+보조(72.79%)와 Mean(70.89%)이다. oracle에서는 Mean(84.33%)이 최고다. 입력 경계·불필요한 문맥이 중요한 병목이며, 더 복잡한 주파수 CNN/세로 attention/lowfocus를 일괄 채택할 근거는 이번 조건에서 없다.

## 1. 완료 및 공통 조건

- 두 단계 각각8모델, 총16모델의 실제200epoch 학습 완료. 고정1000epoch cosine의 앞200이다.
- 실제검출: local4/Jupyter4. Oracle: local6/Jupyter2. 같은 모델의 epoch를 서버간 나누지 않았다.
- archive체크섬, 200행history, 선택checkpoint/예측, 단계내 공통코드·입력·초기값을 검증했다.
- 모델별1880개, 총30,080개 저장예측에서 지표를 재계산해 학습기록과 일치했다.
- 고정seed42, batch32paired examples/64views, paired CE+0.1same_command SupCon,temp0.1. silence는CE만.
- D3 manual-clipped-v1의정리원본: S03/04/05학습600, S02검증200. S01음원미접근.
- 학습소음94원본, 검증소음8원본은훈련미사용. 원본200+0dB1600+noise-only80으로검증했다.
- 학습은고정reference/noisy view bank와200 noise-only pairs를사용한초기선별이다. 과거매epoch재합성과다르다.
- noisy는전체원본에발화부전력기준소음을먼저합성한뒤검출/crop했다. noisy보조특징도그실제noisycrop에서추출했다.
- 현재두단계의3480개합성전체파형SHA256이모두같다. 차이는경계/gate 및각학습입력에서fit한scaler다.

실제검출은lookahead best152/EOF마지막mel16frame유지, 첫시작~마지막끝crop, 미검출silence다. 양쪽모두margin앞50/뒤30ms. Oracle는수동speech경계를사용하며, 기존30/30ms clip에margin을중복해서붙이지않았다. 개발clip800개의PCM/offset을확인했고시작샘플차이는모두0이었다.

## 2. 주 결과: 0dB 명령어10종 macro F1

소음별200조건의command10 macroF1을계산한뒤8소음평균이다. detector eventF1이나과거11종macroF1과다른지표다. 아래각행은그단계의검증최고checkpoint이며, 실제검출에서학습한모델에oracle입력만준결과와구분한다.

| 분류기 | 실제검출 F1 | Oracle F1 | 차이 %p | best epoch 실제/oracle |
|---|---:|---:|---:|---:|
| local + Pooled | 67.08% | 77.13% | +10.05 | 170/191 |
| 축분리2D + Pooled | 65.76% | 79.37% | +13.61 | 115/176 |
| 2구역 CNN + Pooled | 52.04% | 64.02% | +11.98 | 132/182 |
| 2구역 CNN + 세로 attention | 56.18% | 65.56% | +9.38 | 171/182 |
| 위 조건 + lowfocus | 50.39% | 58.10% | +7.71 | 195/173 |
| local + Pooled + 보조 | 64.70% | 78.33% | +13.63 | 200/197 |
| local + Mean | 70.89% | 84.33% | +13.44 | 150/184 |
| local + Mean + 보조 | 72.79% | 83.10% | +10.30 | 196/149 |

## 3. 보조 입력과 구조의 해석

- 실제검출에서Mean보조추가는70.89→72.79%(+1.90%p). 원본명령어F1은88.44→92.76%, noise-only명령오반응은21/80→19/80.
- 실제검출Pooled에서는보조추가67.08→64.70%(-2.38%p). 보조가모든집계에서좋지는않다.
- Oracle에서는Mean보조84.33→83.10%(-1.24%p), Pooled보조77.13→78.33%(+1.20%p). 경계조건과집계에따라효과가달랐다.
- 2구역CNN의세로attention추가는실제52.04→56.18%, oracle64.02→65.56%로그CNN안에서일부개선됐다. local Mean계열은넘지못했다.
- 그조건에lowfocus를추가하면실제56.18→50.39%, oracle65.56→58.10%로하락했다. 현조건에서는우선보류후보다.
- 축분리는실제65.76%로기본Pooled67.08%보다낮지만oracle79.37%로기본Pooled77.13%보다높았다. 이전순위가유지된다고가정할수없다.

## 4. 고정 모델 교차평가: crop 효과와 재학습 구분

추가32회평가(자기조건재현16회포함)를실행했다. 같은조건의예측1880개는저장값과전부일치했다. 교차평가에서는해당모델의원래학습mel/aux scaler를유지하고targetcrop의실제특징을넣었다. checkpoint를새로선택하거나재학습하지않았다.

| 구조 | 실제학습→실제crop | 실제학습→oraclecrop | oracle학습→실제crop | oracle학습→oraclecrop |
|---|---:|---:|---:|---:|
| local + Pooled | 67.08% | 74.20% | 67.24% | 77.13% |
| local + Mean | 70.89% | 81.12% | 68.43% | 84.33% |
| local + Mean + 보조 | 72.79% | 82.67% | 67.39% | 83.10% |

현재Mean+보조를그대로두고oracle입력으로만바꾸면+9.88%p(72.79→82.67). Oracle전용재학습·scaler재적합·checkpoint선택을포함한추가차이는82.67→83.10%(+0.42%p)다. Mean은고정모델입력변경으로70.89→81.12%(+10.23%p), oracle재학습후84.33%다. **경계/양성미검출/주변문맥의영향이크다는직접근거**이며어떤음소가잘렸는지까지입증한것은아니다.

Oracle전용Mean/Mean+보조를현실검출crop에연결하면68.43/67.39%로, 실제crop학습모델70.89/72.79%보다낮다. Oracle모델을현재파이프라인에그대로배포하지않는다.

## 5. 잡음과 실패 유형

Oracle는noise-only에구간을주지않고silence로처리하므로FPR0은정의상결과다. 분류기의잡음거절능력개선으로쓰면안된다. Oracle학습모델을실제검출crop에넣으면8조건모두noise-only28/80에서명령을출력했다. 이는실제검출기가구간을낸28창전체다.

실제조건FPR은Pooled기본18/80, Mean+보조19/80, Pooled+보조20/80, Mean21/80이다. 표본차이가작고같은8소음원본의crop들이므로유의한거절성능차이로단정하지않는다.

상위Mean+보조도cafeteria35.87%, restaurant43.54%, compressor53.61%로어려운소음이남아있다. 차량97.49%와차이가크다. Mean대비보조이득은8소음중5개에서만양수였다. 가장많은오답은전진→정지32혼합조건이며,후진→대기20,오른쪽→왼쪽18,시작→재시작18이다. 이수는독립녹음수가아니다.

## 6. 다음 우선순위와 한계

현재실제입력후보는Mean+보조/Mean을남기고Pooled를기준으로유지한다. 세로CNN·lowfocus추가격자는우선보류한다. 다음확인은동일예산연장또는새학습view/seed에서최소후보를반복하는방식이적절하다. 보조채택에는Burg추출을포함한실제지연도필요하다. 본분석중새학습은시작하지않았다.

한개검증화자·한seed·고정view bank이고최고checkpoint선택과평가가같은개발bank다. 200epoch는1000cosine완주가아니며, 일부모델은마지막20epoch에서최고점을선택했다. 예컨대oracle축분리는best79.37%/last66.59%로차이가커서선택최고점만으로안정적우월성을주장하지않는다.

S03(과거spk02,여성화자)는학습에들어있다. 여성미학습일반화개선은평가하지않았다. 최근실험의S01미접근을프로젝트전체에서처음미사용test라고확장하지않는다.

과거D1/전체1.5초/500~1000epoch/다른SNR기준·증강·지표와직접적인개선폭으로섞지않는다. Oracle는사용자수동경계를이상적검출기로근사한진단조건이며,배포성능이나증명된최대성능이아니다. 두단계차이에는다른crop의재학습/scaler와완벽한negativegate가포함되므로전체차이를검출경계하나의독립인과효과로해석하지않는다.

## 근거 파일

- all_results.csv:16모델원본/소음/FPR/best/last/선택epoch.
- actual_vs_oracle.csv:8구조쌍비교.
- cross_evaluation.csv:고정모델교차평가32개.
- completion_audit.json:공통코드/입력/초기값·완료·예측재계산검증.
- details/:모델별소음·클래스F1/혼동행렬.
- analyze.py,cross_evaluate.py:검증/추가평가실행코드.

원격결과는classifier-post-detector-v1와classifier-oracle-v1릴리스의local/jupyter-e200-results archive에보존돼있다.
