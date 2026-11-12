# spk05: 총합 기준의 공통 최고 epoch 비교

spk05를 학습에서 제외한 CV fold, seed44의 같은203개 파일이다.

전체3시드·4-fold 평균 Macro F1로 고른 세트0 epoch8, 세트3 epoch9을 모든 화자에 동일하게 적용했다.

| 모델 | F1 | 정답 /203 | loss |
| --- | ---: | ---: | ---: |
| 세트0 | 88.38% | 177 | 0.3830 |
| 세트3 | 83.65% | 167 | 0.5202 |

## 세트3만 맞힌 파일: 10개

[CSV](spk05_set3_only_correct.csv)

| 정답 | 파일 | 세트0 공통 최고점 예측 | 세트3 공통 최고점 예측 |
| --- | --- | --- | --- |
| 대기 | [spk05_002.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/대기/spk05_002.wav) | 재시작 | 대기 |
| 대기 | [spk05_010.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/대기/spk05_010.wav) | 재시작 | 대기 |
| 복귀 | [spk05_025.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk05_025.wav) | 정지 | 복귀 |
| 복귀 | [spk05_032.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk05_032.wav) | 재시작 | 복귀 |
| 오른쪽 | [spk05_073.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk05_073.wav) | 왼쪽 | 오른쪽 |
| 정지 | [spk05_157.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk05_157.wav) | 후진 | 정지 |
| 후진 | [spk05_186.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk05_186.wav) | 시작 | 후진 |
| 후진 | [spk05_192.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk05_192.wav) | 시작 | 후진 |
| 후진 | [spk05_196.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk05_196.wav) | 출발 | 후진 |
| 후진 | [spk05_200.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk05_200.wav) | 재시작 | 후진 |

## 세트0만 맞힌 파일: 20개

[CSV](spk05_set0_only_correct.csv)

| 정답 | 파일 | 세트0 공통 최고점 예측 | 세트3 공통 최고점 예측 |
| --- | --- | --- | --- |
| 시작 | [spk05_052.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk05_052.wav) | 시작 | 재시작 |
| 시작 | [spk05_054.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk05_054.wav) | 시작 | 재시작 |
| 시작 | [spk05_055.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk05_055.wav) | 시작 | 왼쪽 |
| 왼쪽 | [spk05_095.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk05_095.wav) | 왼쪽 | 오른쪽 |
| 전진 | [spk05_124.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk05_124.wav) | 전진 | 대기 |
| 전진 | [spk05_125.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk05_125.wav) | 전진 | 정지 |
| 전진 | [spk05_126.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk05_126.wav) | 전진 | 정지 |
| 전진 | [spk05_127.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk05_127.wav) | 전진 | 대기 |
| 전진 | [spk05_129.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk05_129.wav) | 전진 | 대기 |
| 전진 | [spk05_131.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk05_131.wav) | 전진 | 재시작 |
| 전진 | [spk05_132.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk05_132.wav) | 전진 | 대기 |
| 전진 | [spk05_133.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk05_133.wav) | 전진 | 대기 |
| 전진 | [spk05_142.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk05_142.wav) | 전진 | 대기 |
| 정지 | [spk05_145.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk05_145.wav) | 정지 | 대기 |
| 정지 | [spk05_149.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk05_149.wav) | 정지 | 대기 |
| 정지 | [spk05_151.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk05_151.wav) | 정지 | 대기 |
| 정지 | [spk05_160.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk05_160.wav) | 정지 | 대기 |
| 후진 | [spk05_184.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk05_184.wav) | 후진 | 대기 |
| 후진 | [spk05_189.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk05_189.wav) | 후진 | 대기 |
| 후진 | [spk05_193.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk05_193.wav) | 후진 | 대기 |

## 둘 다 틀린 파일: 16개

[CSV](spk05_both_wrong.csv)

| 정답 | 파일 | 세트0 공통 최고점 예측 | 세트3 공통 최고점 예측 |
| --- | --- | --- | --- |
| 대기 | [spk05_004.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/대기/spk05_004.wav) | 재시작 | 재시작 |
| 복귀 | [spk05_038.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk05_038.wav) | 왼쪽 | 대기 |
| 시작 | [spk05_047.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk05_047.wav) | 재시작 | 재시작 |
| 시작 | [spk05_048.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk05_048.wav) | 재시작 | 재시작 |
| 시작 | [spk05_059.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk05_059.wav) | 재시작 | 재시작 |
| 시작 | [spk05_063.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk05_063.wav) | 재시작 | 재시작 |
| 전진 | [spk05_130.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk05_130.wav) | 대기 | 대기 |
| 전진 | [spk05_141.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk05_141.wav) | 재시작 | 대기 |
| 정지 | [spk05_148.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk05_148.wav) | 대기 | 대기 |
| 정지 | [spk05_150.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk05_150.wav) | 대기 | 대기 |
| 정지 | [spk05_154.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk05_154.wav) | 후진 | 재시작 |
| 정지 | [spk05_155.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk05_155.wav) | 대기 | 대기 |
| 정지 | [spk05_161.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk05_161.wav) | 재시작 | 대기 |
| 정지 | [spk05_162.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk05_162.wav) | 재시작 | 재시작 |
| 출발 | [spk05_174.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/출발/spk05_174.wav) | 재시작 | 재시작 |
| 후진 | [spk05_202.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk05_202.wav) | 출발 | 오른쪽 |

[전체203개와 확률·NLL·원본명·패딩](spk05_all_files.csv)

원래 개발 CV에서 얻은 사후 비교다. 독립 test나 파일 오류의 인과 판정은 아니다. 청취 검수는 하지 않았다.
