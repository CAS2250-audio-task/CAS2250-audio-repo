# 화자03: CE 1000epoch와 대조학습 1000epoch 비교

**선택 체크포인트의 0dB F1은 세 모델 모두 대조학습 쪽이 높았다. 평균 풀링+화자 대조가85.44%로 가장 높았다.** 원격CE 예측 파일을 릴리스에서 받아 점수도 독립 재계산했다.

| 모델 | CE | 소음 대조 | 소음+화자 | 소음−CE | 화자포함−CE |
|---|---:|---:|---:|---:|---:|
| 평균 풀링 | 79.65% | 84.01% | 85.44% | +4.36%p | +5.79%p |
| Pooled attention | 78.76% | 83.76% | 84.68% | +5.00%p | +5.92%p |
| Self-attention + CLS | 82.89% | 83.98% | 83.92% | +1.09%p | +1.03%p |

| 모델 | CE 잡음단독 오반응 | 소음 대조 | 소음+화자 |
|---|---:|---:|---:|
| 평균 풀링 | 20.62% | 6.04% | 4.38% |
| Pooled attention | 23.54% | 10.21% | 10.42% |
| Self-attention + CLS | 20.42% | 8.75% | 16.46% |

CE도 기존 소음·포먼트 증강을 사용했다. 모델 구조·원본812개 해시·학습/평가 소음 manifest가 같으며, 검증 화자03·seed42·총1000epoch·최종8소음×3crop 평가 기준을 맞췄다.

**남은 차이:** CE는100epoch LAST에서AdamW 상태를 복원하고 epoch101에LR1e-4로 재시작했다. 대조학습은 처음부터LR1e-3의cosine1000이다. CE는1뷰/소음p0.5, 대조학습은원본+소음2뷰이며 두뷰에CE를 적용한다. torch/CUDA환경도다르다. 따라서관측점수차이는분명하지만순수로스효과를분리한통제실험은아니다.

Self의1000epoch 종료점0dB F1은CE81.11%, 소음대조80.90%, 화자포함79.95%다. 선택된 최고점 우위와 최종 가중치 우위는 구분한다.

[CE 원격 완료 보고서](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/56cfb504afa5fe768cb7c35f5e09ea3fe67bd649/reports/compact-pooling-spk03-1000-v1/README.md)
[비교CSV](ce1000_comparison.csv) · [CE 원자료](ce1000_reference_summary.json) · [CE 프로토콜](ce1000_reference_protocol.json) · [세 모델 전체 오답 분석](spk03-errors/README.md) · [대조학습 전체 결과](README.md)
