# spk02: 최고 체크포인트끼리 비교

각 세트가 원래 spk02 검증 macro F1로 고른 최고 체크포인트를 비교했다.

공통 mask+speed·seed44·weight decay 1e-4·Dropout0. 세트0은 보조 없음, 세트3은 포먼트 F1–F3+대역폭 B1–B3. 추가 학습 없이 저장된 예측을 썼다.

| 모델 | 선택 epoch | Macro F1 | 정답 /203 | loss |
| --- | ---: | ---: | ---: | ---: |
| 세트0 | 4 | 62.82% | 126 | 1.2559 |
| 세트3 | 3 | 70.14% | 136 | 1.1441 |

세트3 F1 차이 +7.32%p. 세트3만 정답 36개 / 세트0만 정답 26개 / 둘 다 정답 100개 / 둘 다 오답 41개.

[전체203개 CSV](spk02_all_files.csv) · [단어별 집계](spk02_class_comparison.csv)

## 세트3만 맞힌 파일

[CSV](spk02_set3_only_correct.csv)

| 정답 | 음성 파일 | 세트0 최고 모델 예측 | 세트3 최고 모델 예측 |
| --- | --- | --- | --- |
| 복귀 | [spk02_008.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_008.wav) | 대기 | 복귀 |
| 복귀 | [spk02_009.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_009.wav) | 대기 | 복귀 |
| 복귀 | [spk02_010.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_010.wav) | 대기 | 복귀 |
| 복귀 | [spk02_011.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_011.wav) | 대기 | 복귀 |
| 복귀 | [spk02_012.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_012.wav) | 후진 | 복귀 |
| 복귀 | [spk02_013.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_013.wav) | 재시작 | 복귀 |
| 복귀 | [spk02_070.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_070.wav) | 대기 | 복귀 |
| 복귀 | [spk02_071.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_071.wav) | 재시작 | 복귀 |
| 복귀 | [spk02_088.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_088.wav) | 대기 | 복귀 |
| 복귀 | [spk02_089.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_089.wav) | 재시작 | 복귀 |
| 복귀 | [spk02_090.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_090.wav) | 대기 | 복귀 |
| 복귀 | [spk02_091.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_091.wav) | 대기 | 복귀 |
| 복귀 | [spk02_102.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_102.wav) | 대기 | 복귀 |
| 복귀 | [spk02_103.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_103.wav) | 대기 | 복귀 |
| 복귀 | [spk02_104.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_104.wav) | 대기 | 복귀 |
| 복귀 | [spk02_106.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_106.wav) | 대기 | 복귀 |
| 시작 | [spk02_107.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk02_107.wav) | 출발 | 시작 |
| 시작 | [spk02_108.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk02_108.wav) | 출발 | 시작 |
| 시작 | [spk02_111.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk02_111.wav) | 출발 | 시작 |
| 시작 | [spk02_156.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk02_156.wav) | 후진 | 시작 |
| 오른쪽 | [spk02_020.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk02_020.wav) | 후진 | 오른쪽 |
| 오른쪽 | [spk02_021.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk02_021.wav) | 후진 | 오른쪽 |
| 오른쪽 | [spk02_022.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk02_022.wav) | 후진 | 오른쪽 |
| 오른쪽 | [spk02_025.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk02_025.wav) | 후진 | 오른쪽 |
| 오른쪽 | [spk02_114.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk02_114.wav) | 후진 | 오른쪽 |
| 오른쪽 | [spk02_115.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk02_115.wav) | 후진 | 오른쪽 |
| 오른쪽 | [spk02_118.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk02_118.wav) | 재시작 | 오른쪽 |
| 왼쪽 | [spk02_119.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk02_119.wav) | 재시작 | 왼쪽 |
| 왼쪽 | [spk02_169.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk02_169.wav) | 오른쪽 | 왼쪽 |
| 전진 | [spk02_134.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk02_134.wav) | 후진 | 전진 |
| 전진 | [spk02_182.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk02_182.wav) | 정지 | 전진 |
| 정지 | [spk02_045.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk02_045.wav) | 재시작 | 정지 |
| 정지 | [spk02_048.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk02_048.wav) | 후진 | 정지 |
| 정지 | [spk02_186.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk02_186.wav) | 전진 | 정지 |
| 정지 | [spk02_187.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk02_187.wav) | 전진 | 정지 |
| 정지 | [spk02_189.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk02_189.wav) | 전진 | 정지 |

## 세트0만 맞힌 파일

[CSV](spk02_set0_only_correct.csv)

| 정답 | 음성 파일 | 세트0 최고 모델 예측 | 세트3 최고 모델 예측 |
| --- | --- | --- | --- |
| 대기 | [spk02_004.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/대기/spk02_004.wav) | 대기 | 복귀 |
| 대기 | [spk02_005.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/대기/spk02_005.wav) | 대기 | 복귀 |
| 대기 | [spk02_007.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/대기/spk02_007.wav) | 대기 | 복귀 |
| 대기 | [spk02_067.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/대기/spk02_067.wav) | 대기 | 복귀 |
| 대기 | [spk02_099.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/대기/spk02_099.wav) | 대기 | 복귀 |
| 대기 | [spk02_100.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/대기/spk02_100.wav) | 대기 | 복귀 |
| 시작 | [spk02_159.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk02_159.wav) | 시작 | 출발 |
| 왼쪽 | [spk02_171.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk02_171.wav) | 왼쪽 | 오른쪽 |
| 재시작 | [spk02_034.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/재시작/spk02_034.wav) | 재시작 | 후진 |
| 재시작 | [spk02_035.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/재시작/spk02_035.wav) | 재시작 | 후진 |
| 재시작 | [spk02_128.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/재시작/spk02_128.wav) | 재시작 | 후진 |
| 재시작 | [spk02_130.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/재시작/spk02_130.wav) | 재시작 | 시작 |
| 재시작 | [spk02_173.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/재시작/spk02_173.wav) | 재시작 | 오른쪽 |
| 전진 | [spk02_080.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk02_080.wav) | 전진 | 복귀 |
| 전진 | [spk02_081.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk02_081.wav) | 전진 | 정지 |
| 전진 | [spk02_132.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk02_132.wav) | 전진 | 복귀 |
| 전진 | [spk02_180.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk02_180.wav) | 전진 | 복귀 |
| 출발 | [spk02_051.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/출발/spk02_051.wav) | 출발 | 정지 |
| 출발 | [spk02_053.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/출발/spk02_053.wav) | 출발 | 후진 |
| 출발 | [spk02_054.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/출발/spk02_054.wav) | 출발 | 정지 |
| 출발 | [spk02_146.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/출발/spk02_146.wav) | 출발 | 복귀 |
| 출발 | [spk02_193.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/출발/spk02_193.wav) | 출발 | 정지 |
| 출발 | [spk02_194.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/출발/spk02_194.wav) | 출발 | 전진 |
| 출발 | [spk02_195.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/출발/spk02_195.wav) | 출발 | 오른쪽 |
| 후진 | [spk02_058.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk02_058.wav) | 후진 | 오른쪽 |
| 후진 | [spk02_059.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk02_059.wav) | 후진 | 복귀 |

## 둘 다 틀린 파일

[CSV](spk02_both_wrong.csv)

| 정답 | 음성 파일 | 세트0 최고 모델 예측 | 세트3 최고 모델 예측 |
| --- | --- | --- | --- |
| 대기 | [spk02_002.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/대기/spk02_002.wav) | 시작 | 후진 |
| 대기 | [spk02_006.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/대기/spk02_006.wav) | 시작 | 복귀 |
| 대기 | [spk02_068.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/대기/spk02_068.wav) | 재시작 | 복귀 |
| 대기 | [spk02_069.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/대기/spk02_069.wav) | 재시작 | 복귀 |
| 복귀 | [spk02_101.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_101.wav) | 대기 | 대기 |
| 복귀 | [spk02_105.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_105.wav) | 대기 | 대기 |
| 시작 | [spk02_014.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk02_014.wav) | 후진 | 후진 |
| 시작 | [spk02_016.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk02_016.wav) | 후진 | 후진 |
| 시작 | [spk02_017.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk02_017.wav) | 후진 | 후진 |
| 시작 | [spk02_018.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk02_018.wav) | 출발 | 후진 |
| 시작 | [spk02_072.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk02_072.wav) | 출발 | 후진 |
| 시작 | [spk02_073.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk02_073.wav) | 출발 | 오른쪽 |
| 시작 | [spk02_110.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk02_110.wav) | 후진 | 후진 |
| 시작 | [spk02_155.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk02_155.wav) | 출발 | 출발 |
| 시작 | [spk02_157.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk02_157.wav) | 후진 | 출발 |
| 오른쪽 | [spk02_023.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk02_023.wav) | 후진 | 후진 |
| 오른쪽 | [spk02_116.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk02_116.wav) | 재시작 | 복귀 |
| 왼쪽 | [spk02_028.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk02_028.wav) | 대기 | 복귀 |
| 왼쪽 | [spk02_029.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk02_029.wav) | 후진 | 복귀 |
| 왼쪽 | [spk02_076.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk02_076.wav) | 후진 | 오른쪽 |
| 왼쪽 | [spk02_120.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk02_120.wav) | 재시작 | 복귀 |
| 왼쪽 | [spk02_121.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk02_121.wav) | 오른쪽 | 복귀 |
| 왼쪽 | [spk02_122.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk02_122.wav) | 재시작 | 복귀 |
| 왼쪽 | [spk02_123.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk02_123.wav) | 복귀 | 복귀 |
| 왼쪽 | [spk02_124.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk02_124.wav) | 복귀 | 복귀 |
| 재시작 | [spk02_078.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/재시작/spk02_078.wav) | 후진 | 복귀 |
| 재시작 | [spk02_079.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/재시작/spk02_079.wav) | 오른쪽 | 복귀 |
| 재시작 | [spk02_175.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/재시작/spk02_175.wav) | 후진 | 오른쪽 |
| 전진 | [spk02_039.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk02_039.wav) | 정지 | 정지 |
| 전진 | [spk02_040.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk02_040.wav) | 후진 | 오른쪽 |
| 전진 | [spk02_041.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk02_041.wav) | 후진 | 후진 |
| 전진 | [spk02_042.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk02_042.wav) | 재시작 | 정지 |
| 전진 | [spk02_043.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk02_043.wav) | 재시작 | 후진 |
| 전진 | [spk02_133.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk02_133.wav) | 정지 | 정지 |
| 전진 | [spk02_136.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk02_136.wav) | 정지 | 정지 |
| 정지 | [spk02_046.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk02_046.wav) | 후진 | 후진 |
| 정지 | [spk02_047.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk02_047.wav) | 후진 | 후진 |
| 정지 | [spk02_049.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk02_049.wav) | 재시작 | 복귀 |
| 정지 | [spk02_082.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk02_082.wav) | 대기 | 복귀 |
| 정지 | [spk02_083.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk02_083.wav) | 대기 | 후진 |
| 후진 | [spk02_087.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk02_087.wav) | 재시작 | 복귀 |

## 단어별 정답 수

| 단어 | 파일 수 | 세트0 | 세트3 | 차이 |
| --- | ---: | ---: | ---: | ---: |
| silence | 3 | 3 | 3 | +0 |
| 대기 | 20 | 16 | 10 | -6 |
| 복귀 | 20 | 2 | 18 | +16 |
| 시작 | 20 | 7 | 10 | +3 |
| 오른쪽 | 20 | 11 | 18 | +7 |
| 왼쪽 | 20 | 10 | 11 | +1 |
| 재시작 | 20 | 17 | 12 | -5 |
| 전진 | 20 | 11 | 9 | -2 |
| 정지 | 20 | 10 | 15 | +5 |
| 출발 | 20 | 20 | 13 | -7 |
| 후진 | 20 | 19 | 17 | -2 |

## 앞서 epoch3→10에 악화한 43개

세트3 최고점이 epoch3이므로 이43개는 모두 세트3 최고 모델에서 정답이다. 세트0 최고 모델에서는 22개 정답, 21개 오답이다.

[43개 최고점 예측 CSV](spk02_previous_43_at_best.csv)

| 정답 | 음성 파일 | 세트0 최고 모델 예측 | 세트3 최고 모델 예측 |
| --- | --- | --- | --- |
| 복귀 | [spk02_008.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_008.wav) | 대기 | 복귀 |
| 복귀 | [spk02_009.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_009.wav) | 대기 | 복귀 |
| 복귀 | [spk02_010.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_010.wav) | 대기 | 복귀 |
| 복귀 | [spk02_013.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_013.wav) | 재시작 | 복귀 |
| 복귀 | [spk02_070.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_070.wav) | 대기 | 복귀 |
| 복귀 | [spk02_088.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_088.wav) | 대기 | 복귀 |
| 복귀 | [spk02_090.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_090.wav) | 대기 | 복귀 |
| 복귀 | [spk02_102.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_102.wav) | 대기 | 복귀 |
| 복귀 | [spk02_104.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_104.wav) | 대기 | 복귀 |
| 복귀 | [spk02_106.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_106.wav) | 대기 | 복귀 |
| 시작 | [spk02_015.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk02_015.wav) | 시작 | 시작 |
| 시작 | [spk02_019.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk02_019.wav) | 시작 | 시작 |
| 시작 | [spk02_108.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk02_108.wav) | 출발 | 시작 |
| 시작 | [spk02_112.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk02_112.wav) | 시작 | 시작 |
| 시작 | [spk02_156.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk02_156.wav) | 후진 | 시작 |
| 오른쪽 | [spk02_022.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk02_022.wav) | 후진 | 오른쪽 |
| 오른쪽 | [spk02_114.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk02_114.wav) | 후진 | 오른쪽 |
| 오른쪽 | [spk02_115.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk02_115.wav) | 후진 | 오른쪽 |
| 오른쪽 | [spk02_118.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk02_118.wav) | 재시작 | 오른쪽 |
| 오른쪽 | [spk02_164.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk02_164.wav) | 오른쪽 | 오른쪽 |
| 왼쪽 | [spk02_119.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk02_119.wav) | 재시작 | 왼쪽 |
| 왼쪽 | [spk02_167.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk02_167.wav) | 왼쪽 | 왼쪽 |
| 재시작 | [spk02_037.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/재시작/spk02_037.wav) | 재시작 | 재시작 |
| 전진 | [spk02_038.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk02_038.wav) | 전진 | 전진 |
| 전진 | [spk02_131.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk02_131.wav) | 전진 | 전진 |
| 전진 | [spk02_134.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk02_134.wav) | 후진 | 전진 |
| 전진 | [spk02_135.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk02_135.wav) | 전진 | 전진 |
| 전진 | [spk02_182.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk02_182.wav) | 정지 | 전진 |
| 전진 | [spk02_184.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk02_184.wav) | 전진 | 전진 |
| 정지 | [spk02_044.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk02_044.wav) | 정지 | 정지 |
| 정지 | [spk02_045.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk02_045.wav) | 재시작 | 정지 |
| 정지 | [spk02_138.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk02_138.wav) | 정지 | 정지 |
| 정지 | [spk02_139.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk02_139.wav) | 정지 | 정지 |
| 정지 | [spk02_140.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk02_140.wav) | 정지 | 정지 |
| 정지 | [spk02_141.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk02_141.wav) | 정지 | 정지 |
| 정지 | [spk02_186.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk02_186.wav) | 전진 | 정지 |
| 후진 | [spk02_061.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk02_061.wav) | 후진 | 후진 |
| 후진 | [spk02_086.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk02_086.wav) | 후진 | 후진 |
| 후진 | [spk02_149.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk02_149.wav) | 후진 | 후진 |
| 후진 | [spk02_150.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk02_150.wav) | 후진 | 후진 |
| 후진 | [spk02_152.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk02_152.wav) | 후진 | 후진 |
| 후진 | [spk02_153.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk02_153.wav) | 후진 | 후진 |
| 후진 | [spk02_203.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk02_203.wav) | 후진 | 후진 |

각자의 검증 최고점을 고른 개발 결과이며 독립 test 성능은 아니다. 한 시드의 파일별 차이를 보조 특징의 인과 효과로 확정하지 않는다. 정답 수와 Macro F1 차이는 서로 다른 지표다. 사람의 청취/라벨 검수는 하지 않았다.
