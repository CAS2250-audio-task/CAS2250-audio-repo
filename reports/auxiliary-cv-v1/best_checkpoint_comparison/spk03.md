# spk03: 최고 체크포인트끼리 비교

각 세트가 원래 spk03 검증 macro F1로 고른 최고 체크포인트를 비교했다.

공통 mask+speed·seed44·weight decay 1e-4·Dropout0. 세트0은 보조 없음, 세트3은 포먼트 F1–F3+대역폭 B1–B3. 추가 학습 없이 저장된 예측을 썼다.

| 모델 | 선택 epoch | Macro F1 | 정답 /203 | loss |
| --- | ---: | ---: | ---: | ---: |
| 세트0 | 8 | 87.02% | 175 | 0.4534 |
| 세트3 | 7 | 87.59% | 176 | 0.5132 |

세트3 F1 차이 +0.57%p. 세트3만 정답 13개 / 세트0만 정답 12개 / 둘 다 정답 163개 / 둘 다 오답 15개.

[전체203개 CSV](spk03_all_files.csv) · [단어별 집계](spk03_class_comparison.csv)

## 세트3만 맞힌 파일

[CSV](spk03_set3_only_correct.csv)

| 정답 | 음성 파일 | 세트0 최고 모델 예측 | 세트3 최고 모델 예측 |
| --- | --- | --- | --- |
| 복귀 | [spk03_041.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk03_041.wav) | 대기 | 복귀 |
| 오른쪽 | [spk03_082.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk03_082.wav) | 전진 | 오른쪽 |
| 왼쪽 | [spk03_088.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk03_088.wav) | 오른쪽 | 왼쪽 |
| 왼쪽 | [spk03_102.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk03_102.wav) | 후진 | 왼쪽 |
| 왼쪽 | [spk03_103.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk03_103.wav) | 후진 | 왼쪽 |
| 재시작 | [spk03_104.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/재시작/spk03_104.wav) | 시작 | 재시작 |
| 재시작 | [spk03_113.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/재시작/spk03_113.wav) | 시작 | 재시작 |
| 정지 | [spk03_145.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk03_145.wav) | 전진 | 정지 |
| 정지 | [spk03_146.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk03_146.wav) | 전진 | 정지 |
| 정지 | [spk03_158.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk03_158.wav) | 전진 | 정지 |
| 정지 | [spk03_161.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk03_161.wav) | 전진 | 정지 |
| 정지 | [spk03_162.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk03_162.wav) | 전진 | 정지 |
| 정지 | [spk03_163.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk03_163.wav) | 전진 | 정지 |

## 세트0만 맞힌 파일

[CSV](spk03_set0_only_correct.csv)

| 정답 | 음성 파일 | 세트0 최고 모델 예측 | 세트3 최고 모델 예측 |
| --- | --- | --- | --- |
| 오른쪽 | [spk03_077.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk03_077.wav) | 오른쪽 | 왼쪽 |
| 오른쪽 | [spk03_081.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk03_081.wav) | 오른쪽 | 왼쪽 |
| 전진 | [spk03_130.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk03_130.wav) | 전진 | 정지 |
| 전진 | [spk03_132.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk03_132.wav) | 전진 | 정지 |
| 전진 | [spk03_134.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk03_134.wav) | 전진 | 정지 |
| 전진 | [spk03_136.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk03_136.wav) | 전진 | 정지 |
| 전진 | [spk03_143.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk03_143.wav) | 전진 | 후진 |
| 후진 | [spk03_187.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk03_187.wav) | 후진 | 오른쪽 |
| 후진 | [spk03_191.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk03_191.wav) | 후진 | 시작 |
| 후진 | [spk03_192.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk03_192.wav) | 후진 | 시작 |
| 후진 | [spk03_193.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk03_193.wav) | 후진 | 왼쪽 |
| 후진 | [spk03_194.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk03_194.wav) | 후진 | 시작 |

## 둘 다 틀린 파일

[CSV](spk03_both_wrong.csv)

| 정답 | 음성 파일 | 세트0 최고 모델 예측 | 세트3 최고 모델 예측 |
| --- | --- | --- | --- |
| 복귀 | [spk03_025.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk03_025.wav) | 전진 | 정지 |
| 복귀 | [spk03_038.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk03_038.wav) | 대기 | 정지 |
| 복귀 | [spk03_042.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk03_042.wav) | 후진 | 정지 |
| 재시작 | [spk03_106.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/재시작/spk03_106.wav) | 후진 | 정지 |
| 재시작 | [spk03_107.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/재시작/spk03_107.wav) | 시작 | 시작 |
| 재시작 | [spk03_108.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/재시작/spk03_108.wav) | 왼쪽 | 왼쪽 |
| 재시작 | [spk03_109.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/재시작/spk03_109.wav) | 시작 | 시작 |
| 재시작 | [spk03_119.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/재시작/spk03_119.wav) | 대기 | 시작 |
| 재시작 | [spk03_123.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/재시작/spk03_123.wav) | 시작 | 시작 |
| 전진 | [spk03_124.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk03_124.wav) | 출발 | 정지 |
| 전진 | [spk03_133.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk03_133.wav) | 정지 | 정지 |
| 전진 | [spk03_135.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk03_135.wav) | 정지 | 정지 |
| 전진 | [spk03_137.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk03_137.wav) | 정지 | 정지 |
| 전진 | [spk03_142.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk03_142.wav) | 정지 | 정지 |
| 정지 | [spk03_159.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk03_159.wav) | 전진 | 전진 |

## 단어별 정답 수

| 단어 | 파일 수 | 세트0 | 세트3 | 차이 |
| --- | ---: | ---: | ---: | ---: |
| 대기 | 20 | 20 | 20 | +0 |
| silence | 3 | 3 | 3 | +0 |
| 복귀 | 20 | 16 | 17 | +1 |
| 시작 | 20 | 20 | 20 | +0 |
| 오른쪽 | 20 | 19 | 18 | -1 |
| 왼쪽 | 20 | 17 | 20 | +3 |
| 재시작 | 20 | 12 | 14 | +2 |
| 전진 | 20 | 15 | 10 | -5 |
| 정지 | 20 | 13 | 19 | +6 |
| 출발 | 20 | 20 | 20 | +0 |
| 후진 | 20 | 20 | 15 | -5 |

각자의 검증 최고점을 고른 개발 결과이며 독립 test 성능은 아니다. 한 시드의 파일별 차이를 보조 특징의 인과 효과로 확정하지 않는다. 정답 수와 Macro F1 차이는 서로 다른 지표다. 사람의 청취/라벨 검수는 하지 않았다.
