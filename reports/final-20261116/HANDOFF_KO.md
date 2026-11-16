# 최종 실험: 보고서 담당자 인계

2026-11-16 23:09 KST 기준. 최종 보고서 본문이 아닌 결과·근거·한계 인계다. 이 디렉터리에는 오디오/가중치를 포함하지 않는다. role0는 로컬 증거 파일, role1~3은 **사용자가 붙여준 REPORT_SUMMARY_V2 출력**을 구조화한 것으로 원격 원본 파일을 직접 검증한 자료가 아니다.

## 선정 방법과 후보

S02 개발 active-power0dB의 command10 F1을 우선하고 동률이면 noise-only FP, 원본 F1 순으로 선정했다. S01 test 최고값으로 후보를 바꾸지 않는다. 500/1000epoch 완주 예산과 최고 checkpoint의 선택 epoch는 다르다. 배포 후보의 500/1000 비교는 가능하지만 LS 효과는 **동일500epoch** 대조로 해석한다.

| 역할 | 개발 선정 후보 | 학습 예산 | 개발 F1 % | noise FP/80 | 원본 개발 F1 % |
|---|---|---:|---:|---:|---:|
| 0 | LS500_actual_mean_aux_eps0p1__soft_time | 500 | 77.516 | 21 | 97.495 |
| 1 | baseline1000_actual_mean__command_ce | 1000 | 75.310 | 18 | 95.979 |
| 2 | LS500_off_mean_eps0p1 | 500 | 73.497 | 6 | 98.169 |
| 3 | LS500_off_mean_aux_eps0p1 | 500 | 71.062 | 16 | 93.745 |

개발 지표로 남길 3개 유력 후보는 role0 **LS MeanAux+soft_time**(정확도), role1 **Mean+commandCE**(Burg 보조 연산 없음), role2 **LS OFF Mean**(검출 대기 없음·FP 적음)이다. 이것은 실제 게임 결과가 아니라 개발자료 기준 shortlist다. role3 OFFMeanAux는 개발 F1·FP 모두 OFFMean보다 불리하고 보조 연산 비용도 있으나 test 원본 F1 **87.605%**를 포함한 결과는 숨기지 않는다.

## S01 고정 후보 평가: 모든 조건

원본도 이미 배경음이 있는 녹음이다. S01은 별도 test 화자이며 개발 여성 화자를 대체하는 독립 여성 test가 아니다. 아래 test 값은 후보 선정에 사용하지 않는다.

| 역할 | 조건 | command10 F1 % | resident p50 ms | resident p95 ms |
|---|---|---:|---:|---:|
| 0 | original_already_contains_background | 85.829 | 미측정 | 미측정 |
| 0 | ambient_mean | 82.150 | 21.957844961434603 | 26.097198086790737 |
| 0 | background_p90_lower_snr | 80.695 | 15.821086475625634 | 25.161820556968447 |
| 1 | original_already_contains_background | 87.468 | 미측정 | 미측정 |
| 1 | ambient_mean | 84.241 | 15.124904573895037 | 15.684939560014755 |
| 1 | background_p90_lower_snr | 82.261 | 7.5372596038505435 | 14.971782127395272 |
| 2 | original_already_contains_background | 86.128 | 미측정 | 미측정 |
| 2 | ambient_mean | 81.708 | 8.187873056158423 | 10.59438152005896 |
| 2 | background_p90_lower_snr | 79.235 | 5.712444894015789 | 5.9838864603079855 |
| 3 | original_already_contains_background | 87.605 | 미측정 | 미측정 |
| 3 | ambient_mean | 82.441 | 34.42920558154583 | 49.84188891248777 |
| 3 | background_p90_lower_snr | 80.214 | 28.701690607704222 | 35.55464894743636 |

잡음 조건은 임의의0/5/10dB가 아니다. 파일별 수동 발화 경계±50ms를 제외한 구간에서 완전히 비발화인 비중첩20ms 프레임을10개 이상 확보하고, 비발화 전체 DC를 한 번 제거한 프레임 전력의 평균과 p90을 계산했다. 추가 외부잡음 목표는 **평균**과 **max(평균,p90)**이다. 두 조건의 외부 소음 원본·offset은 같고 gain=sqrt(목표전력/소음전력), peak attenuation을 공통 적용한다. p90 조건은 보수적인 낮은 SNR 스트레스이며 **95% 신뢰구간이나 최솟값 추정이 아니다**. 배경 부족이면 원본 평가를 남기고 추가 잡음만 unavailable로 기록하며, 평균≤1e-12면0추가한다. 실제 전력·coverage는 로컬 evidence의 protocol/metrics와 원격 원본 수집 시 확인한다. noise-only80은 기존 학습 화자 전력 기준으로 고정한다. 이80개는 개발검증에서 사용한 고정 잡음 파형을 재사용하므로 새로운 독립 잡음test라고 주장하지 않는다. 추가 소음8종도 개발에서 사용한 검증 소음이며, S01의 독립성은 주로 화자/원음에 관한 것이다.

모든 서버가 RTX3090이라도 CPU·환경이 다르므로 위 latency는 엄밀한 동일 게임 PC 비교가 아니다. CPU mel/Burg·실제 검출·crop·전송·분류를 포함하며 파일 IO/모델 load/오디오 수집 대기는 제외한다. 검출 ON은 별도 lookahead80ms와 frontend12.5ms가 있다. 이 ms와 검출 시간오차 cost의 ms-equivalent를 혼동하지 않는다.

## LS 효과를 과장하지 말 것

- Mean+commandCE, 동일500: baseline75.0825 → LS74.9534%, **-0.1291%p**. LS가 모든 모델에서 개선됐다는 주장은 틀리다. 1000baseline75.3104와500LS 차이를 LS 단독 효과로 쓰지 않는다.
- OFFMean, 동일500: baseline65.9700 → LS73.4967%, **+7.5266%p**, noise FP8→6.
- OFFMeanAux, 동일500: baseline69.7505 → LS71.0621%, **+1.3116%p**이지만 noise FP3→16으로 악화.
- 실제 crop MeanAux+soft_time, 동일500: baseline74.6999 → LS77.5159%, +2.8160%p; noise FP23→21. 기존 BCE 및 다른 검출기 결과도 local_results.json의 cases 전체를 제시한다.

## 같은 가중치로 OFF / oracle / actual 진단

분류기 LS MeanAux500(best475), 고정 체크포인트와 동일1880 개발 파형으로 검출/crop 효과를 분리했다. actual은 기존 reference152이며 최종 soft_time 후보와 구분한다.

| 조건 | 개발 noisy F1 % | 원본 F1 % | noise FP/80 |
|---|---:|---:|---:|
| off | 20.159 | 10.493 | 79 |
| oracle | 84.378 | 97.491 | 0 |
| actual: reference152 | 74.310 | 95.422 | 20 |
| actual: 선정 soft_time | 77.516 | 97.495 | 21 |

OFF는 **crop 학습 가중치에 전체 native 입력을 곧바로 준 분포 변화 진단**이며, 따로 전체 입력으로 학습한 OFF 대조군과 다르다. oracle는 수동 정답 경계를 쓰는 진단이며 배포 후보가 아니다. oracle negative는 정답으로 gate하므로 실용적인 잡음 거절 성능으로 해석하지 않는다.

## 화자 일반화 / CV 한계

현재 CV는 사전 지정한 **검출 OFF 기본 Mean**의 개발용 화자 분리 실험이다. 최종 선정 검출기 결합·LS 후보의4fold 완료로 표현하면 안 된다. fold별3화자600으로 학습하고 제외 화자200으로 검증하며, 해당 제외 화자의 점수로 최고 epoch를 선택했다. 따라서 독립 outer-CV 추정도 아니다. 전체4fold 완료 원본이 없는 경우4fold 평균을 꾸며내지 않는다.

확인된 여성 제외 fold는 **현재 S03 = legacy spk02 = 곽민채**, 학습 S02/S04/S05, OFFMean1000(best157)이다. 원본 F1 **56.9148%**, noisy F1 **27.1726%**, noise-only FPR15%. 같은 OFFMean1000의 남성 S02(legacy spk03, 노정민) fold는 원본 **84.193%**, noisy **65.9700%**이다. 이 차이는 해당 모델의 화자 취약성을 보여주지만, 성별의 인과 효과나 선정 검출기 후보의 여성 일반화 성능을 측정한 결과가 아니다. S01 test가 독립 여성 검증을 제공한다고 쓰지 않는다.

## 이전 Seoj 실험 참고 시 주의

Seoj 제공 원격 자료는 [commit82fe 고정 원문](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/82feeb6c78ac2d70597e8a5b6142f7c5be559cf3/reports/seojoon-feature-aug-v1/README.md)을 인용한다. 참고값은 original83.70→90.15%, legacy spk02 여성70.39→84.69%. 이는 증강/LS 등 조합 효과이며 LS 단독 효과가 아니다. 후진 혼동7→0도 증강 단독에서 이미0이었다. D1 및20epoch 실험으로 현재 D3·500/1000 후보와 직접 순위를 매기면 안 된다.

## 증거와 남은 업무

- local_results.json: 로컬 후보 전체 결과/최고epoch/완주예산/S01 결과. evidence/에는 로컬 선택 protocol, winner, development_scores, S01 protocol·metrics, 검출기 요약이 있다.
- remote_results_user_supplied.json: 원격3서버 사용자 출력 전사이며 직접 원본 검증은 미완료. 원격 체크포인트/상세 예측/coverage/protocol 원본 수집 필요.
- same_classifier_crop_diagnostic.json: 동일 가중치 OFF/oracle/actual 진단 원본. comparison.csv는 정리용 수치 표다.
- **실제 게임 비교는 실행하지 않았다.** 현재 작업환경에 게임 실행 진입점이 없어 같은 게임 조건 승자를 검증했다고 쓰지 않는다.
- 23:11 1차 원격 커밋 이후 로컬3변형의 분류기 FP16 비교를 수행했고 모두 FP32 유지로 판정했다. [추론 최적화·지연 해석 추가 인계](OPTIMIZATION_AND_LATENCY.md)를 참조한다. 원격후보 전체의 동일 PC 최적화나 실제게임 비교가 완료된 것은 아니다.
- 최종 유력 후보의 전체 파이프라인 화자 분리 CV와 독립 여성 검증은 남는다. 이미 화자를 본 기존 검출기를 해당 fold에 재사용하면 누수가 된다.
- test 결과를 본 뒤 후보/threshold/증강을 다시 고르지 않는다. 추가 개발을 하면 S01은 이미 관찰한 test라는 사실을 남긴다.

## 모델 묶음 준비 상태

로컬 검출기 결합 후보의 추론용 가중치 묶음은 exports/final-candidates-20261116-local.tar.gz에 준비했다. 약0.6MB이며 오디오와 optimizer를 제외했다. 원본tensor완전일치와 S02 예측확률 재현3건을 확인했다. 파라미터 수 감소나 FP16최적화 결과가 아니라 배포용 파일정리다. 원격의 OFFMean과Mean+commandCE 가중치는 이서버로 회수하지 않았으므로 이파일을최종3후보전체라고 부르지 않는다. 공개 가중치 업로드는 아직 하지 않았다.
