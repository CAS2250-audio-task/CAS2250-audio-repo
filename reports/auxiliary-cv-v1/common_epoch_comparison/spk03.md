# spk03: 총합 기준의 공통 최고 epoch 비교

spk03를 학습에서 제외한 CV fold, seed44의 같은203개 파일이다.

전체3시드·4-fold 평균 Macro F1로 고른 세트0 epoch8, 세트3 epoch9을 모든 화자에 동일하게 적용했다.

| 모델 | F1 | 정답 /203 | loss |
| --- | ---: | ---: | ---: |
| 세트0 | 87.02% | 175 | 0.4534 |
| 세트3 | 86.47% | 174 | 0.4817 |

## 세트3만 맞힌 파일: 14개

[CSV](spk03_set3_only_correct.csv)

| 정답 | 파일 | 세트0 공통 최고점 예측 | 세트3 공통 최고점 예측 |
| --- | --- | --- | --- |
| 복귀 | [spk03_038.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk03_038.wav) | 대기 | 복귀 |
| 복귀 | [spk03_041.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk03_041.wav) | 대기 | 복귀 |
| 오른쪽 | [spk03_082.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk03_082.wav) | 전진 | 오른쪽 |
| 왼쪽 | [spk03_088.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk03_088.wav) | 오른쪽 | 왼쪽 |
| 왼쪽 | [spk03_102.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk03_102.wav) | 후진 | 왼쪽 |
| 왼쪽 | [spk03_103.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk03_103.wav) | 후진 | 왼쪽 |
| 재시작 | [spk03_104.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/재시작/spk03_104.wav) | 시작 | 재시작 |
| 재시작 | [spk03_113.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/재시작/spk03_113.wav) | 시작 | 재시작 |
| 재시작 | [spk03_119.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/재시작/spk03_119.wav) | 대기 | 재시작 |
| 정지 | [spk03_145.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk03_145.wav) | 전진 | 정지 |
| 정지 | [spk03_146.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk03_146.wav) | 전진 | 정지 |
| 정지 | [spk03_161.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk03_161.wav) | 전진 | 정지 |
| 정지 | [spk03_162.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk03_162.wav) | 전진 | 정지 |
| 정지 | [spk03_163.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk03_163.wav) | 전진 | 정지 |

## 세트0만 맞힌 파일: 15개

[CSV](spk03_set0_only_correct.csv)

| 정답 | 파일 | 세트0 공통 최고점 예측 | 세트3 공통 최고점 예측 |
| --- | --- | --- | --- |
| 오른쪽 | [spk03_076.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk03_076.wav) | 오른쪽 | 왼쪽 |
| 오른쪽 | [spk03_077.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk03_077.wav) | 오른쪽 | 왼쪽 |
| 오른쪽 | [spk03_081.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk03_081.wav) | 오른쪽 | 왼쪽 |
| 오른쪽 | [spk03_083.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk03_083.wav) | 오른쪽 | 왼쪽 |
| 전진 | [spk03_125.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk03_125.wav) | 전진 | 정지 |
| 전진 | [spk03_130.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk03_130.wav) | 전진 | 정지 |
| 전진 | [spk03_132.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk03_132.wav) | 전진 | 정지 |
| 전진 | [spk03_134.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk03_134.wav) | 전진 | 대기 |
| 전진 | [spk03_136.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk03_136.wav) | 전진 | 정지 |
| 전진 | [spk03_143.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk03_143.wav) | 전진 | 후진 |
| 정지 | [spk03_152.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk03_152.wav) | 정지 | 대기 |
| 출발 | [spk03_168.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/출발/spk03_168.wav) | 출발 | 왼쪽 |
| 후진 | [spk03_192.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk03_192.wav) | 후진 | 시작 |
| 후진 | [spk03_193.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk03_193.wav) | 후진 | 왼쪽 |
| 후진 | [spk03_194.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk03_194.wav) | 후진 | 시작 |

## 둘 다 틀린 파일: 14개

[CSV](spk03_both_wrong.csv)

| 정답 | 파일 | 세트0 공통 최고점 예측 | 세트3 공통 최고점 예측 |
| --- | --- | --- | --- |
| 복귀 | [spk03_025.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk03_025.wav) | 전진 | 정지 |
| 복귀 | [spk03_042.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk03_042.wav) | 후진 | 후진 |
| 재시작 | [spk03_106.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/재시작/spk03_106.wav) | 후진 | 복귀 |
| 재시작 | [spk03_107.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/재시작/spk03_107.wav) | 시작 | 시작 |
| 재시작 | [spk03_108.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/재시작/spk03_108.wav) | 왼쪽 | 왼쪽 |
| 재시작 | [spk03_109.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/재시작/spk03_109.wav) | 시작 | 시작 |
| 재시작 | [spk03_123.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/재시작/spk03_123.wav) | 시작 | 시작 |
| 전진 | [spk03_124.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk03_124.wav) | 출발 | 정지 |
| 전진 | [spk03_133.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk03_133.wav) | 정지 | 정지 |
| 전진 | [spk03_135.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk03_135.wav) | 정지 | 정지 |
| 전진 | [spk03_137.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk03_137.wav) | 정지 | 정지 |
| 전진 | [spk03_142.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk03_142.wav) | 정지 | 정지 |
| 정지 | [spk03_158.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk03_158.wav) | 전진 | 전진 |
| 정지 | [spk03_159.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk03_159.wav) | 전진 | 전진 |

[전체203개와 확률·NLL·원본명·패딩](spk03_all_files.csv)

원래 개발 CV에서 얻은 사후 비교다. 독립 test나 파일 오류의 인과 판정은 아니다. 청취 검수는 하지 않았다.
