# spk04: 최고 체크포인트끼리 비교

각 세트가 원래 spk04 검증 macro F1로 고른 최고 체크포인트를 비교했다.

공통 mask+speed·seed44·weight decay 1e-4·Dropout0. 세트0은 보조 없음, 세트3은 포먼트 F1–F3+대역폭 B1–B3. 추가 학습 없이 저장된 예측을 썼다.

| 모델 | 선택 epoch | Macro F1 | 정답 /203 | loss |
| --- | ---: | ---: | ---: | ---: |
| 세트0 | 9 | 84.73% | 173 | 0.5087 |
| 세트3 | 9 | 87.42% | 180 | 0.4474 |

세트3 F1 차이 +2.70%p. 세트3만 정답 12개 / 세트0만 정답 5개 / 둘 다 정답 168개 / 둘 다 오답 18개.

[전체203개 CSV](spk04_all_files.csv) · [단어별 집계](spk04_class_comparison.csv)

## 세트3만 맞힌 파일

[CSV](spk04_set3_only_correct.csv)

| 정답 | 음성 파일 | 세트0 최고 모델 예측 | 세트3 최고 모델 예측 |
| --- | --- | --- | --- |
| 시작 | [spk04_045.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk04_045.wav) | 재시작 | 시작 |
| 시작 | [spk04_047.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk04_047.wav) | 재시작 | 시작 |
| 시작 | [spk04_052.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk04_052.wav) | 재시작 | 시작 |
| 재시작 | [spk04_118.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/재시작/spk04_118.wav) | 정지 | 재시작 |
| 전진 | [spk04_133.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk04_133.wav) | 후진 | 전진 |
| 전진 | [spk04_140.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk04_140.wav) | 후진 | 전진 |
| 정지 | [spk04_153.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk04_153.wav) | 전진 | 정지 |
| 정지 | [spk04_155.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk04_155.wav) | 후진 | 정지 |
| 정지 | [spk04_156.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk04_156.wav) | 전진 | 정지 |
| 정지 | [spk04_158.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk04_158.wav) | 전진 | 정지 |
| 출발 | [spk04_173.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/출발/spk04_173.wav) | 시작 | 출발 |
| 후진 | [spk04_201.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk04_201.wav) | 전진 | 후진 |

## 세트0만 맞힌 파일

[CSV](spk04_set0_only_correct.csv)

| 정답 | 음성 파일 | 세트0 최고 모델 예측 | 세트3 최고 모델 예측 |
| --- | --- | --- | --- |
| silence | [spk04_023.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/silence/spk04_023.wav) | silence | 대기 |
| 왼쪽 | [spk04_087.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk04_087.wav) | 왼쪽 | 오른쪽 |
| 왼쪽 | [spk04_092.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk04_092.wav) | 왼쪽 | 오른쪽 |
| 재시작 | [spk04_116.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/재시작/spk04_116.wav) | 재시작 | 전진 |
| 전진 | [spk04_135.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk04_135.wav) | 전진 | 오른쪽 |

## 둘 다 틀린 파일

[CSV](spk04_both_wrong.csv)

| 정답 | 음성 파일 | 세트0 최고 모델 예측 | 세트3 최고 모델 예측 |
| --- | --- | --- | --- |
| 왼쪽 | [spk04_084.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk04_084.wav) | 오른쪽 | 오른쪽 |
| 재시작 | [spk04_122.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/재시작/spk04_122.wav) | 시작 | 정지 |
| 정지 | [spk04_144.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk04_144.wav) | 전진 | 전진 |
| 정지 | [spk04_145.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk04_145.wav) | 전진 | 전진 |
| 정지 | [spk04_146.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk04_146.wav) | 전진 | 전진 |
| 정지 | [spk04_147.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk04_147.wav) | 전진 | 전진 |
| 정지 | [spk04_148.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk04_148.wav) | 전진 | 전진 |
| 정지 | [spk04_149.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk04_149.wav) | 전진 | 전진 |
| 정지 | [spk04_150.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk04_150.wav) | 전진 | 전진 |
| 정지 | [spk04_151.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk04_151.wav) | 전진 | 전진 |
| 정지 | [spk04_152.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk04_152.wav) | 전진 | 전진 |
| 정지 | [spk04_154.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk04_154.wav) | 복귀 | 복귀 |
| 정지 | [spk04_157.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk04_157.wav) | 전진 | 전진 |
| 정지 | [spk04_159.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk04_159.wav) | 전진 | 전진 |
| 정지 | [spk04_163.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk04_163.wav) | 복귀 | 전진 |
| 출발 | [spk04_174.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/출발/spk04_174.wav) | 시작 | 시작 |
| 후진 | [spk04_189.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk04_189.wav) | 전진 | 전진 |
| 후진 | [spk04_203.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk04_203.wav) | 왼쪽 | 왼쪽 |

## 단어별 정답 수

| 단어 | 파일 수 | 세트0 | 세트3 | 차이 |
| --- | ---: | ---: | ---: | ---: |
| 대기 | 20 | 20 | 20 | +0 |
| silence | 3 | 3 | 2 | -1 |
| 복귀 | 20 | 20 | 20 | +0 |
| 시작 | 20 | 17 | 20 | +3 |
| 오른쪽 | 20 | 20 | 20 | +0 |
| 왼쪽 | 20 | 19 | 17 | -2 |
| 재시작 | 20 | 18 | 18 | +0 |
| 전진 | 20 | 18 | 19 | +1 |
| 정지 | 20 | 3 | 7 | +4 |
| 출발 | 20 | 18 | 19 | +1 |
| 후진 | 20 | 17 | 18 | +1 |

각자의 검증 최고점을 고른 개발 결과이며 독립 test 성능은 아니다. 한 시드의 파일별 차이를 보조 특징의 인과 효과로 확정하지 않는다. 정답 수와 Macro F1 차이는 서로 다른 지표다. 사람의 청취/라벨 검수는 하지 않았다.
