# local CNN 기반 주파수축 실험

검출기 없이 전체1.5초, 사전학습 없이 화자03 검증, 학습 화자02/04/05.
기존 증강 캐시·소음·seed42·1000epoch cosine·CE+0.1 SupCon(same_command)을 유지한다.
각 그룹은 Mean/Pooled/Self 세 모델에 같은 CNN 초기값, 분류기 초기값, 증강 입력을 주고 독립 학습한다.

## 어텐션 제거 비교

`local + Mean`이 이미 어텐션 없는 비교군이다. 시간 self-attention과 attention pooling 모두 없다.
완료 결과 원본/0dB F1은 Mean91.97/85.44%, Pooled95.42/86.77%, Self96.30/85.55%.
최종 특징의 평균에 중간 특징의 평균을 더하므로 중간 경로는 Mean에서도 유효하다.
Linear projection과 시간 평균은 교환되지만, 앞단 CNN의 서로 다른 깊이·비선형 특징은 같아지지 않는다.
같은 소음+화자 SupCon baseline Mean과 local Mean의 0dB F1은 반올림하면 둘 다85.44%다.
이 한 조건만으로 모든 readout에서 중간 경로가 불필요하다고 결론내리지는 않는다.

## 18조건

- CNN: basic(local 그대로), lowcost(248파라미터 residual2D 전처리+local), frequency(주파수축을 유지하는 분리형 F/T CNN+중간 전달).
- 세로 self-attention: 미적용/적용. 64주파수 token, 매 시간프레임 독립 처리, d16/2heads/FF32, 주파수 위치인코딩, zero-init 출력 residual. SDPA 사용.
- 시간 집계: 기존 Mean/Pooled/Self 유지.

세로 어텐션은 주파수축이 섞이기 **전** backbone 입구에 둔다. 세로 CNN 없이도 적용 가능하다.
lowcost에서만 2D 전처리 뒤에 둔다. 주파수 해상도 축소 없이64밴드를 유지한다.
어텐션을 backbone 내부에 포함해 `paired_loss`에서도 실제로 학습되게 한다.

frequency CNN은 주파수 해상도64→32→16→8, 시간151프레임 유지.
두 번째 axis block의48채널 특징을 주파수 평균→48→32 투영하여 마지막32D 특징에 더한다.
시간 kernel은11/13/15/17/29(dilation2), stem 시간kernel1이므로 시간범위109프레임, 중간23프레임이다.
lowcost의3×3 두 층은 시간범위를4프레임 더하므로113/27프레임이다. 이 차이는 통제되지 않은 구조 차이로 명시한다.
주파수 attention을 쓰면 주파수 방향 support는 전역이다.

| CNN | 세로 없음 Mean/Pooled/Self | 세로 있음 Mean/Pooled/Self |
|---|---:|---:|
| basic(local) | 79,915 / 80,939 / 88,395 | 82,140 / 83,164 / 90,620 |
| lowcost | 80,163 / 81,187 / 88,643 | 82,388 / 83,412 / 90,868 |
| frequency | 81,291 / 82,315 / 89,771 | 83,516 / 84,540 / 91,996 |

대조학습 projector4,192개는 훈련 전용으로 위 수에서 제외한다.
기존basic/세로없음3모델은 완료 결과를 재사용하고 나머지15모델을 새로 학습한다.
저비용은 추가 파라미터가 적다는 뜻이며 latency가 줄어든다는 뜻은 아니다.

## 실행과 상태

```bash
python -m experiments.frequency_local.check
python -m experiments.frequency_local.launch
# 단일 그룹
python -m experiments.frequency_local.run --cnn frequency --vertical \
  --objective paired_supcon --positive-mode same_command \
  --supcon-weight 0.1 --temperature 0.1 --folds spk03 --epochs 1000 \
  --output runs/frequency-local-frequency-v1-spk03-e1000
```

실행 순서는 lowcost→frequency→basic+세로→lowcost+세로→frequency+세로, GPU 작업은 한 그룹씩이다.
상태: `runs/frequency-local-v1/status.json`, 로그: 해당 폴더 `*.log`.
기존 trainer와 checkpoint 경로를 수정하지 않는다. 새 그룹은 전용 runs/reports 폴더에 저장한다.
그룹 완료마다 `reports/frequency-local-v1/comparison.csv`를 갱신한다.
선택은0dB draw0 최고점, 최종 평가는8소음×3draw. 독립test01은 읽지 않는다.

사전검사는18조건 shape, 동일CNN 초기값, zero-init 전처리의 local 출력 일치,
paired-loss 유한 역전파와 세로Q projection까지 도달하는 실제 gradient를 확인한다.
GPU step 비용은 `reports/frequency-local-v1/preflight.json`에 저장한다.

## 설계 맥락과 해석

[평균/어텐션 비교와 주파수축 도입의 검증 가설](../../docs/COMPACT_MODEL_DESIGN_HANDOFF.md)을 함께 전달한다. 전체macro F1뿐 아니라 명령어10종F1·silence·정답률·개별 혼동을 분리해 평가한다. 모델별CNN도 독립 학습됐고lowcost의시간support도 달라 순수어텐션/주파수축의 인과효과로 단정하지 않는다.
