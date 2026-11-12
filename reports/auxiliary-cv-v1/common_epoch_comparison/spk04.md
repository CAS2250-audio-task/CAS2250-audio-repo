# spk04: 총합 기준의 공통 최고 epoch 비교

spk04를 학습에서 제외한 CV fold, seed44의 같은203개 파일이다.

전체3시드·4-fold 평균 Macro F1로 고른 세트0 epoch8, 세트3 epoch9을 모든 화자에 동일하게 적용했다.

| 모델 | F1 | 정답 /203 | loss |
| --- | ---: | ---: | ---: |
| 세트0 | 84.02% | 175 | 0.4892 |
| 세트3 | 87.42% | 180 | 0.4474 |

## 세트3만 맞힌 파일: 9개

[CSV](spk04_set3_only_correct.csv)

| 정답 | 파일 | 세트0 공통 최고점 예측 | 세트3 공통 최고점 예측 |
| --- | --- | --- | --- |
| 시작 | [spk04_045.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk04_045.wav) | 재시작 | 시작 |
| 시작 | [spk04_047.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk04_047.wav) | 재시작 | 시작 |
| 재시작 | [spk04_118.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/재시작/spk04_118.wav) | 정지 | 재시작 |
| 전진 | [spk04_140.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk04_140.wav) | 후진 | 전진 |
| 정지 | [spk04_153.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk04_153.wav) | 전진 | 정지 |
| 정지 | [spk04_155.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk04_155.wav) | 후진 | 정지 |
| 정지 | [spk04_156.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk04_156.wav) | 전진 | 정지 |
| 정지 | [spk04_158.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk04_158.wav) | 전진 | 정지 |
| 후진 | [spk04_195.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk04_195.wav) | 전진 | 후진 |

## 세트0만 맞힌 파일: 4개

[CSV](spk04_set0_only_correct.csv)

| 정답 | 파일 | 세트0 공통 최고점 예측 | 세트3 공통 최고점 예측 |
| --- | --- | --- | --- |
| 왼쪽 | [spk04_087.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk04_087.wav) | 왼쪽 | 오른쪽 |
| 왼쪽 | [spk04_092.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk04_092.wav) | 왼쪽 | 오른쪽 |
| 전진 | [spk04_135.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk04_135.wav) | 전진 | 오른쪽 |
| 출발 | [spk04_174.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/출발/spk04_174.wav) | 출발 | 시작 |

## 둘 다 틀린 파일: 19개

[CSV](spk04_both_wrong.csv)

| 정답 | 파일 | 세트0 공통 최고점 예측 | 세트3 공통 최고점 예측 |
| --- | --- | --- | --- |
| silence | [spk04_023.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/silence/spk04_023.wav) | 출발 | 대기 |
| 왼쪽 | [spk04_084.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk04_084.wav) | 오른쪽 | 오른쪽 |
| 재시작 | [spk04_116.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/재시작/spk04_116.wav) | 정지 | 전진 |
| 재시작 | [spk04_122.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/재시작/spk04_122.wav) | 시작 | 정지 |
| 정지 | [spk04_144.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk04_144.wav) | 전진 | 전진 |
| 정지 | [spk04_145.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk04_145.wav) | 전진 | 전진 |
| 정지 | [spk04_146.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk04_146.wav) | 전진 | 전진 |
| 정지 | [spk04_147.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk04_147.wav) | 전진 | 전진 |
| 정지 | [spk04_148.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk04_148.wav) | 전진 | 전진 |
| 정지 | [spk04_149.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk04_149.wav) | 출발 | 전진 |
| 정지 | [spk04_150.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk04_150.wav) | 전진 | 전진 |
| 정지 | [spk04_151.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk04_151.wav) | 전진 | 전진 |
| 정지 | [spk04_152.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk04_152.wav) | 출발 | 전진 |
| 정지 | [spk04_154.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk04_154.wav) | 복귀 | 복귀 |
| 정지 | [spk04_157.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk04_157.wav) | 전진 | 전진 |
| 정지 | [spk04_159.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk04_159.wav) | 전진 | 전진 |
| 정지 | [spk04_163.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk04_163.wav) | 전진 | 전진 |
| 후진 | [spk04_189.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk04_189.wav) | 전진 | 전진 |
| 후진 | [spk04_203.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk04_203.wav) | 왼쪽 | 왼쪽 |

[전체203개와 확률·NLL·원본명·패딩](spk04_all_files.csv)

원래 개발 CV에서 얻은 사후 비교다. 독립 test나 파일 오류의 인과 판정은 아니다. 청취 검수는 하지 않았다.
