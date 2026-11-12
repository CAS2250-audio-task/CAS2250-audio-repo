# spk01: 최고 체크포인트끼리 비교

spk02 검증 점수로 고른 체크포인트를 spk01에 별도 평가했다. spk01 자체의 최고 epoch를 고른 결과가 아니다.

공통 mask+speed·seed44·weight decay 1e-4·Dropout0. 세트0은 보조 없음, 세트3은 포먼트 F1–F3+대역폭 B1–B3. 추가 학습 없이 저장된 예측을 썼다.

| 모델 | 선택 epoch | Macro F1 | 정답 /203 | loss |
| --- | ---: | ---: | ---: | ---: |
| 세트0 | 4 | 58.46% | 115 | 1.9459 |
| 세트3 | 3 | 71.86% | 140 | 1.1088 |

세트3 F1 차이 +13.40%p. 세트3만 정답 40개 / 세트0만 정답 15개 / 둘 다 정답 100개 / 둘 다 오답 48개.

[전체203개 CSV](spk01_all_files.csv) · [단어별 집계](spk01_class_comparison.csv)

## 세트3만 맞힌 파일

[CSV](spk01_set3_only_correct.csv)

| 정답 | 음성 파일 | 세트0 최고 모델 예측 | 세트3 최고 모델 예측 |
| --- | --- | --- | --- |
| 복귀 | [spk01_002.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk01_002.wav) | 대기 | 복귀 |
| 복귀 | [spk01_022.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk01_022.wav) | 후진 | 복귀 |
| 복귀 | [spk01_042.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk01_042.wav) | 후진 | 복귀 |
| 복귀 | [spk01_073.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk01_073.wav) | 후진 | 복귀 |
| 복귀 | [spk01_093.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk01_093.wav) | 후진 | 복귀 |
| 복귀 | [spk01_103.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk01_103.wav) | 후진 | 복귀 |
| 복귀 | [spk01_115.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk01_115.wav) | 시작 | 복귀 |
| 복귀 | [spk01_144.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk01_144.wav) | 후진 | 복귀 |
| 복귀 | [spk01_194.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk01_194.wav) | 대기 | 복귀 |
| 시작 | [spk01_056.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk01_056.wav) | 재시작 | 시작 |
| 오른쪽 | [spk01_004.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk01_004.wav) | 왼쪽 | 오른쪽 |
| 오른쪽 | [spk01_014.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk01_014.wav) | 후진 | 오른쪽 |
| 오른쪽 | [spk01_024.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk01_024.wav) | 후진 | 오른쪽 |
| 오른쪽 | [spk01_044.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk01_044.wav) | 대기 | 오른쪽 |
| 오른쪽 | [spk01_075.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk01_075.wav) | 후진 | 오른쪽 |
| 오른쪽 | [spk01_085.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk01_085.wav) | 후진 | 오른쪽 |
| 오른쪽 | [spk01_146.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk01_146.wav) | 후진 | 오른쪽 |
| 오른쪽 | [spk01_176.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk01_176.wav) | 시작 | 오른쪽 |
| 왼쪽 | [spk01_106.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk01_106.wav) | 대기 | 왼쪽 |
| 왼쪽 | [spk01_197.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk01_197.wav) | 시작 | 왼쪽 |
| 재시작 | [spk01_107.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/재시작/spk01_107.wav) | 시작 | 재시작 |
| 재시작 | [spk01_122.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/재시작/spk01_122.wav) | 시작 | 재시작 |
| 재시작 | [spk01_148.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/재시작/spk01_148.wav) | 시작 | 재시작 |
| 재시작 | [spk01_158.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/재시작/spk01_158.wav) | 대기 | 재시작 |
| 전진 | [spk01_017.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk01_017.wav) | 후진 | 전진 |
| 전진 | [spk01_027.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk01_027.wav) | 정지 | 전진 |
| 전진 | [spk01_063.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk01_063.wav) | 정지 | 전진 |
| 전진 | [spk01_098.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk01_098.wav) | 후진 | 전진 |
| 전진 | [spk01_108.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk01_108.wav) | 대기 | 전진 |
| 전진 | [spk01_124.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk01_124.wav) | 대기 | 전진 |
| 전진 | [spk01_149.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk01_149.wav) | 대기 | 전진 |
| 전진 | [spk01_159.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk01_159.wav) | 대기 | 전진 |
| 전진 | [spk01_169.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk01_169.wav) | 대기 | 전진 |
| 전진 | [spk01_199.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk01_199.wav) | 대기 | 전진 |
| 정지 | [spk01_150.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk01_150.wav) | 대기 | 정지 |
| 정지 | [spk01_200.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk01_200.wav) | 대기 | 정지 |
| 출발 | [spk01_080.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/출발/spk01_080.wav) | 후진 | 출발 |
| 출발 | [spk01_201.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/출발/spk01_201.wav) | 시작 | 출발 |
| 후진 | [spk01_050.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk01_050.wav) | 대기 | 후진 |
| 후진 | [spk01_081.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk01_081.wav) | 시작 | 후진 |

## 세트0만 맞힌 파일

[CSV](spk01_set0_only_correct.csv)

| 정답 | 음성 파일 | 세트0 최고 모델 예측 | 세트3 최고 모델 예측 |
| --- | --- | --- | --- |
| 대기 | [spk01_021.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/대기/spk01_021.wav) | 대기 | 후진 |
| 대기 | [spk01_031.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/대기/spk01_031.wav) | 대기 | 전진 |
| 대기 | [spk01_072.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/대기/spk01_072.wav) | 대기 | 복귀 |
| 대기 | [spk01_082.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/대기/spk01_082.wav) | 대기 | 복귀 |
| 시작 | [spk01_043.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk01_043.wav) | 시작 | 복귀 |
| 시작 | [spk01_074.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk01_074.wav) | 시작 | 후진 |
| 시작 | [spk01_094.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk01_094.wav) | 시작 | 재시작 |
| 시작 | [spk01_165.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk01_165.wav) | 시작 | 출발 |
| 재시작 | [spk01_046.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/재시작/spk01_046.wav) | 재시작 | 후진 |
| 전진 | [spk01_088.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk01_088.wav) | 전진 | 정지 |
| 정지 | [spk01_028.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk01_028.wav) | 정지 | 전진 |
| 정지 | [spk01_066.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk01_066.wav) | 정지 | 오른쪽 |
| 출발 | [spk01_151.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/출발/spk01_151.wav) | 출발 | 복귀 |
| 후진 | [spk01_030.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk01_030.wav) | 후진 | 정지 |
| 후진 | [spk01_070.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk01_070.wav) | 후진 | 전진 |

## 둘 다 틀린 파일

[CSV](spk01_both_wrong.csv)

| 정답 | 음성 파일 | 세트0 최고 모델 예측 | 세트3 최고 모델 예측 |
| --- | --- | --- | --- |
| 복귀 | [spk01_114.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk01_114.wav) | 대기 | 대기 |
| 복귀 | [spk01_164.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk01_164.wav) | 대기 | 후진 |
| 복귀 | [spk01_184.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk01_184.wav) | 대기 | 후진 |
| 시작 | [spk01_055.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk01_055.wav) | 재시작 | 왼쪽 |
| 오른쪽 | [spk01_105.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk01_105.wav) | 대기 | 대기 |
| 오른쪽 | [spk01_118.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk01_118.wav) | 후진 | 후진 |
| 오른쪽 | [spk01_119.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk01_119.wav) | 시작 | 후진 |
| 오른쪽 | [spk01_156.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk01_156.wav) | 왼쪽 | 왼쪽 |
| 오른쪽 | [spk01_166.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk01_166.wav) | 대기 | 후진 |
| 오른쪽 | [spk01_186.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk01_186.wav) | 후진 | 후진 |
| 오른쪽 | [spk01_196.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk01_196.wav) | 대기 | 대기 |
| 왼쪽 | [spk01_045.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk01_045.wav) | 대기 | 대기 |
| 왼쪽 | [spk01_060.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk01_060.wav) | 오른쪽 | 오른쪽 |
| 왼쪽 | [spk01_076.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk01_076.wav) | 후진 | 후진 |
| 왼쪽 | [spk01_086.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk01_086.wav) | 복귀 | 복귀 |
| 왼쪽 | [spk01_120.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk01_120.wav) | 후진 | 복귀 |
| 왼쪽 | [spk01_121.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk01_121.wav) | 후진 | 복귀 |
| 왼쪽 | [spk01_147.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk01_147.wav) | 대기 | 복귀 |
| 왼쪽 | [spk01_167.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk01_167.wav) | 대기 | 복귀 |
| 왼쪽 | [spk01_187.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk01_187.wav) | 후진 | 복귀 |
| 재시작 | [spk01_087.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/재시작/spk01_087.wav) | 후진 | 후진 |
| 재시작 | [spk01_137.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/재시작/spk01_137.wav) | 시작 | 후진 |
| 재시작 | [spk01_168.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/재시작/spk01_168.wav) | 대기 | 후진 |
| 재시작 | [spk01_188.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/재시작/spk01_188.wav) | 대기 | 대기 |
| 전진 | [spk01_007.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk01_007.wav) | 시작 | 복귀 |
| 전진 | [spk01_037.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk01_037.wav) | 후진 | 오른쪽 |
| 전진 | [spk01_047.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk01_047.wav) | 대기 | 후진 |
| 전진 | [spk01_078.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk01_078.wav) | 시작 | 후진 |
| 전진 | [spk01_125.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk01_125.wav) | 재시작 | 복귀 |
| 전진 | [spk01_179.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk01_179.wav) | 대기 | 대기 |
| 전진 | [spk01_189.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk01_189.wav) | 대기 | 후진 |
| 정지 | [spk01_008.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk01_008.wav) | 대기 | 후진 |
| 정지 | [spk01_038.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk01_038.wav) | 대기 | 전진 |
| 정지 | [spk01_048.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk01_048.wav) | 후진 | 후진 |
| 정지 | [spk01_079.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk01_079.wav) | 후진 | 복귀 |
| 정지 | [spk01_099.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk01_099.wav) | 후진 | 후진 |
| 정지 | [spk01_109.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk01_109.wav) | 대기 | 대기 |
| 정지 | [spk01_126.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk01_126.wav) | 재시작 | 복귀 |
| 정지 | [spk01_127.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk01_127.wav) | 대기 | 후진 |
| 정지 | [spk01_160.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk01_160.wav) | 대기 | 복귀 |
| 정지 | [spk01_170.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk01_170.wav) | 대기 | 복귀 |
| 정지 | [spk01_180.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk01_180.wav) | 대기 | 전진 |
| 정지 | [spk01_190.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk01_190.wav) | 후진 | 후진 |
| 출발 | [spk01_110.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/출발/spk01_110.wav) | 시작 | 후진 |
| 출발 | [spk01_128.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/출발/spk01_128.wav) | 후진 | 후진 |
| 출발 | [spk01_129.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/출발/spk01_129.wav) | 후진 | 후진 |
| 출발 | [spk01_171.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/출발/spk01_171.wav) | 후진 | 후진 |
| 출발 | [spk01_191.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/출발/spk01_191.wav) | 대기 | 후진 |

## 단어별 정답 수

| 단어 | 파일 수 | 세트0 | 세트3 | 차이 |
| --- | ---: | ---: | ---: | ---: |
| 대기 | 20 | 20 | 16 | -4 |
| 복귀 | 20 | 8 | 17 | +9 |
| 시작 | 20 | 18 | 15 | -3 |
| 오른쪽 | 20 | 5 | 13 | +8 |
| 왼쪽 | 20 | 9 | 11 | +2 |
| 재시작 | 20 | 12 | 15 | +3 |
| 전진 | 20 | 3 | 12 | +9 |
| 정지 | 20 | 6 | 6 | +0 |
| 출발 | 20 | 13 | 14 | +1 |
| 후진 | 20 | 18 | 18 | +0 |
| silence | 3 | 3 | 3 | +0 |

각자의 검증 최고점을 고른 개발 결과이며 독립 test 성능은 아니다. 한 시드의 파일별 차이를 보조 특징의 인과 효과로 확정하지 않는다. 정답 수와 Macro F1 차이는 서로 다른 지표다. 사람의 청취/라벨 검수는 하지 않았다.
