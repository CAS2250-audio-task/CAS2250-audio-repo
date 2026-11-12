# 0dB 요약 — 전체 비교 완료

학습12회/120epoch·전체SNR의650,592예측·독립검사·프로젝트54tests까지완료했다. [최종 보고서](README.md)와 [검증 기록](independent_audit.json)을 확인한다.

| 학습 조건 | DFN3 | 원본 F1 | 0dB F1 | 잡음 단독 오반응 |
| --- | --- | ---: | ---: | ---: |
| baseline | dfn3 | 95.68% | 83.80% | 12.81% |
| baseline | off | 99.20% | 78.47% | 4.22% |
| focus | dfn3 | 95.33% | 83.32% | 12.34% |
| focus | off | 99.20% | 77.67% | 3.75% |
| speech | dfn3 | 95.68% | 83.74% | 10.16% |
| speech | off | 99.20% | 75.84% | 1.41% |
| speech_focus | dfn3 | 94.85% | 82.95% | 10.62% |
| speech_focus | off | 99.09% | 75.03% | 2.19% |

seed42개발4fold이며test01은미사용. 총학습량고정,원본50%유지,focus는넓은SNR25%+근처25%. 최종보고서는README.md에정리했다.
