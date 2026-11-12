# 화자별 4-fold 개발 CV와 증강 실험

새 데이터 `datasets/task2/real-context-1p5-v1`에서 세트 0·4로 공통 학습 조건을 선택한 뒤
세트 0~7을 재비교한다. 기존 `src/`, `experiments/auxiliary`, 이전 실험 결과는 보존한다.
spk01은 모델 선택과 평가에 사용하지 않는다.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -u -m experiments.auxiliary_cv.pipeline \
  --config configs/task2_real_context.json \
  --study runs/auxiliary-cv-v1 \
  --reports reports/auxiliary-cv-v1
```

실행 로그는 `runs/auxiliary-cv-v1/driver.log`, 현재 작업은 `active.json`에 있다.
중단하면 같은 명령으로 재개한다. 완료 fold는 건너뛰고, 미완료 학습은 마지막 epoch의
모델·optimizer·scheduler·샘플러·Python/NumPy/Torch/CUDA 난수 상태에서 재개한다.
특징 추출도 파일별 row cache를 저장한다. 두 driver가 동시에 실행되지 않도록 파일 잠금을 쓴다.
재개 결과가 연속 학습과 가중치까지 일치하는지 실제 GPU에서 별도 검증했다.

**데이터와 초기값**

- spk02~05와 연결된 silence 812개로만 개발한다. 각 fold는 train 609 / val 203이며
  validation 화자를 한 명씩 바꾼다. 라벨·화자·원래 파일 정체성은 유지한다.
- scaler는 해당 fold의 원본 train 측정 가능 프레임에서만 적합한다. 검증과 증강 입력에도
  이 scaler를 사용한다. 라벨/화자를 입력 특징으로 제공하지 않는다.
- 모든 fold는 같은 seed의 ImageNet ResNet18 초기값에서 독립적으로 시작한다.
  이전에 fine-tune한 체크포인트를 초기값으로 재사용하지 않는다.
- 10 epochs, batch 32, AdamW LR 5e-4, cosine, 역빈도 WeightedRandomSampler를 유지한다.
  원본 validation macro F1이 가장 높은 epoch를 선택하며 동률이면 먼저 나온 epoch를 쓴다.
- 모델 뒤쪽 512차원 표현과 Linear(512,11)을 유지한다. 분류기 앞 dropout은 별도 비교하고,
  앞단 고정 시 stem/layer1의 BatchNorm running 통계도 고정한다.

**증강 구현**

파형 정책마다 클립당 2개의 고정 변형을 준비한다. 각 training draw에서 각 정책을 확률 0.5로
독립 적용하며, 변형과 적용 순서는 모델/학습 seed 사이에 공유한다. 검증에는 원본만 사용한다.
증강된 validation 화자의 복사본이 캐시에 있더라도 해당 fold의 train index로 선택하지 않는다.
파형 변형 뒤 스펙트로그램과 17개 보조 특징을 같은 최종 파형에서 다시 추출한다.

| 정책 | 구현 |
|---|---|
| 피치 | 0 / U[-1,1] / U[-1,1] / 0 반음 anchor를 연결해 파일 안에서 변화. 내부 두 anchor 차이는 최소 0.35반음. Rubber Band 2.0 R2, formant-preserved, high-consistency 모드. 입력 앞 padding과 출력 startup delay를 함께 보정. |
| 속도 | 피치를 유지하는 U[0.9,1.1]. 기존 검수 구간 전체를 보존할 수 있는 crop만 허용하며 불가능한 감속은 원본 유지. 짧아지면 좌우 0 padding. |
| 잡음 | ESC-50 rain/engine/church_bells/keyboard_typing, 16개 학습 source. 이전 평가 fold4/5 source ID와 중복 없음. 추가 SNR U[10,30]dB. |
| 마스킹 | 시간 20~50ms는 스펙트로그램과 보조 값·availability 채널을 함께 가림. 주파수 4~8 mel bin은 스펙트로그램에 적용. 스펙트로그램 최저 에너지값 -2로 채움. |

잡음 SNR은 **변형 후 원본 full-clip 파워 대 추가 잡음 파워**다. 원본에 포함된 잡음을 제거한
깨끗한 발화의 SNR은 아니다. silence는 자신의 원본 배경음 clip 파워를 기준으로 한다.
전체 혼합 파형의 peak가 0.99를 넘으면 공통 배율로 줄인다. 평가 잡음을 학습에 재사용하지 않는다.

이 첫 비교는 클립당 유한한 2개 변형을 재사용한다. 매 epoch 무한히 새로운 파형을 만드는
증강과 동일한 결과라고 해석하지 않는다. 파형 조합은 pitch → speed → noise 순서로 처리한다.
독립적인 정책 적용 조합별로 최종 파형과 특징을 준비한다.

Rubber Band와 의존 라이브러리는 수치 학습 환경을 재설치하지 않고
`runs/auxiliary-cv-v1/runtime/unpacked`에 Ubuntu Jammy 패키지를 풀어 사용했다.
원본 다운로드 URL과 SHA-256은 `runtime/sources.json`, 환경은 `environment.json`에 기록한다.
다른 머신에서는 이 파일의 공식 패키지 URL로 같은 라이브러리를 확보하고 같은 경로에 풀어야 한다.

**사전 선택 기준**

1. 세트 0·4 무증강 seed42 CV 8회.
2. 피치/속도/잡음/마스킹 개별 seed42 비교. 유망한 최대 두 조건을 seed42/43/44로 확인.
   확인된 개별 조건만 결합 후보로 시험하고, 기존 최선보다 개선될 때 조합을 채택한다.
3. 선택한 증강 위에서 weight decay 1e-3과 본체 dropout 0.2를 각각 비교한다.
   유망 조건은 3시드 확인하고, 둘 다 유효하면 결합도 비교한다.
4. train minus validation F1이 8%p 이상이고 validation 평균 F1이 95% 미만이면
   앞단 고정과 본체 LR 비율 0.1을 각각 비교한다. 가장 유망한 조건을 3시드 확인한다.
5. 선택한 공통 조건으로 세트 0~7의 seed42 CV를 실행하고, 상위 두 세트 및 세트 0·4를
   3시드로 확인한다. 동일 조건의 기존 완료 run은 재사용한다.

공통 학습 조건의 개선은 두 모델을 평균한 원본 CV macro F1이 최소 0.5%p 상승하고,
최저 화자 F1 하락이 1%p 이내이며, 어느 한 모델의 평균 하락도 0.5%p 이내일 때로 정의한다.
동일 기준을 선별과 3시드 확인에 적용한다. 최종 세트는 3시드 평균이 세트 0보다 0.5%p 이상
좋고 최저 화자 하락이 1%p 이내일 때만 보조 입력을 채택한다. 이 임계값은 통계 검정이 아니다.
조건/epoch 선택에 사용한 CV 점수는 개발용 점수다. 네 화자를 수백 명의 독립 평가처럼 해석하지 않는다.

`protocol.json`과 코드 해시는 첫 실행 때 고정한다. 실행 도중 코드를 바꾸면 같은 study를
덮어써 재개하지 않는다. 각 단계 선택은 JSON으로 고정하며 test를 자동 실행하는 명령은 없다.
체크포인트와 row cache는 로컬 `runs/`에, CSV·설정·예측·보고서는 `reports/auxiliary-cv-v1/`에 저장한다.

사용자용 진행 요약은 [INTERIM.md](../../reports/auxiliary-cv-v1/INTERIM.md), 단계별 근거는
[진행 보고서](../../reports/auxiliary-cv-v1/README.md)에서 확인한다. 전체 완료 후 다음 명령으로
예측·최적 epoch·분할·표준화·초기값·코드/데이터 해시를 독립 검증하고 그래프와 해석을 만든다.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m experiments.report_auxiliary_cv
```

최종 선택 전 완료된 실행만 검증하려면 같은 명령에 `--partial`을 붙인다. 이때 결과는
`interim_execution_audit.json`이며 최종 완료 보고서와 구별한다.

데이터의 피크 예외 6개는 신호 수치와 기존 검수 구간 보존을 확인했지만 사람의 청취 검수는
아직 수행하지 않았다. 자동 검증을 청취 완료로 표현하지 않는다.
