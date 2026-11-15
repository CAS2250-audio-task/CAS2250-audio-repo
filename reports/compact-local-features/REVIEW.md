# 완료 후 검토

화자03 검증, 소음+화자 SupCon, seed42, 각1000epoch. 새9모델 정상 종료. baseline3모델 재사용. 전체 종료 2026-11-16 00:36:52 KST.

고정 검증 draw0의 최고점에서 선택한 체크포인트를 8소음×3draw로 평가한 0dB macro F1이다. 괄호는 선택 epoch다.

| 조건 | Mean | Pooled attention | Self-attention |
|---|---:|---:|---:|
| baseline | 85.44% (585) | 84.68% (895) | 83.92% (331) |
| short: 1.105→0.625초 | 82.99% (579) | 77.92% (436) | 82.74% (538) |
| local: 기존 범위+0.245초 중간 경로 | 85.44% (806) | **86.77% (565)** | 85.55% (508) |
| both: 축소+중간 경로 | 84.93% (590) | 76.23% (408) | 83.49% (573) |

local pooled는 같은 head baseline 대비 +2.09%p, 기존 전체 최고 baseline mean 대비 +1.33%p. local self는 같은 head 대비 +1.63%p, local mean은 사실상 동일. short/both는 모든 head에서 baseline보다 낮다.

local pooled의 draw별 F1은86.20/87.21/86.90%, baseline pooled는85.09/84.36/84.60%. 선택에 쓰이지 않은 draw1/2에서도 개선되나 화자·녹음·소음종류는 같으므로 독립 검증은 아니다.

## 오답 변화

local pooled 대 baseline pooled: 혼합 오답718→651/4872. 기존 오답355개 수정, 기존 정답288개에서 새 오답. 오른쪽→왼쪽46→17, 전진→정지87→51은 개선. 재시작→시작43→102, 재시작 전체 오답53→127/480으로 악화. 전체 점수 개선이 모든 명령어 개선을 뜻하지 않는다.

local self 대 baseline self: 오른쪽→왼쪽103→37, 정지→전진98→52. 반면 전진 전체 오답96→126, 후진122→149.

local mean 대 baseline mean: 혼합 오답679→611이나 macro F1은 거의 같다. macro F1은 클래스별 precision/recall을 반영하므로 총오답수와 다르게 움직일 수 있다. 원본 오답9→15.

short mean 재시작→시작127→69, both mean은57. 범위 축소가 특정 혼동에는 유리할 가능성은 있지만 전체 성능 개선으로 이어지지는 않았다. 이번 결과만으로 CNN이 특정 음절을 지웠다거나 attention 용량이 원인이라고 확정할 수 없다.

## 비용과 후보

기존 동일 세션 batch1, CPU2threads, 1.5초 waveform의 frontend 포함 추론 중앙값. WAV 읽기와 녹음시간 제외.

| 후보 | 파라미터 | GPU ms | CPU ms | 잡음 단독 오반응 |
|---|---:|---:|---:|---:|
| baseline mean | 77,835 | 1.323 | 2.290 | 4.38% |
| local mean | 79,915 | 1.396 | 2.887 | 4.58% |
| local pooled | 80,939 | 1.525 | 2.944 | 11.88% |
| local self | 88,395 | 1.933 | 3.481 | 12.08% |

사용자 결정에 따라 local 구조의 Mean/Pooled/Self 세 head를 유지한다. CPU latency는 선택 기준에서 제외한다. latency 파일의 both는 학습 중 체크포인트 측정이며 최종 체크포인트 측정으로 부르지 않는다.

local mean/pooled/self의 선택점 F1→1000epoch F1은85.44→84.44,86.77→82.76,85.55→80.55%. 마지막 학습 정확도99.84/99.67/100.00%. 후반 일반화 악화와 일치하지만 원인을 LR 또는 구조 하나로 단정할 수 없다. best 체크포인트를 보존·비교한다.

다음 비교 대상은 local 구조의 Mean/Pooled/Self 세 head다. 다른 화자 분할에서 먼저 순위 확인 후 LR·배치를 따로 비교할 수 있다. 추가 학습은 시작하지 않았다.

## 검증

독립 검토:12개1000epoch history,유한손실,최고epoch선택,60,900개 예측의 중복/개수,12개최종점수재계산,현재소스해시39개 확인. launcher는 새3000개 증강trace가baseline과 동일함도 검증했다. 단일화자·단일seed 개발검증이며 test01은 이번 실험에서 읽지 않았다. RF 변경시 커널크기·초기화도 변하고 local은2080파라미터가 증가하므로 시간범위 하나만의 인과효과로 해석하지 않는다.

[전체비교](comparison.csv) · [클래스별오답](review_class_errors.csv) · [정오답교체](review_paired_changes.csv) · [소음별비교](review_noise_comparison.csv) · [검토기록](review_audit.json) · [학습곡선](learning_curves.png) · [latency](LATENCY.md)

## 최종 사용자 결정

중간 특징 전달만(local)을 채택한다. Mean/Pooled/Self 세 head를 유지한다. CPU latency는 선택 기준에서 제외하므로 앞의 CPU 비용에 근거한 후보 축소 제안은 적용하지 않는다. 다른 서버의 대상 브랜치에 병합할 내용은 MERGE_HANDOFF.md를 따른다.
