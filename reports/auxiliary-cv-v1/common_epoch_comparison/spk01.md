# spk01: 총합 기준의 공통 최고 epoch 비교

화자1은 전체 평균의 epoch 선택에서 제외했다. 선택된 epoch를 기존 spk02-fold 모델에 적용한 별도 진단이다.

전체3시드·4-fold 평균 Macro F1로 고른 세트0 epoch8, 세트3 epoch9을 모든 화자에 동일하게 적용했다.

| 모델 | F1 | 정답 /203 | loss |
| --- | ---: | ---: | ---: |
| 세트0 | 67.00% | 135 | 1.4381 |
| 세트3 | 62.18% | 130 | 1.5258 |

## 세트3만 맞힌 파일: 11개

[CSV](spk01_set3_only_correct.csv)

| 정답 | 파일 | 세트0 공통 최고점 예측 | 세트3 공통 최고점 예측 |
| --- | --- | --- | --- |
| 시작 | [spk01_055.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk01_055.wav) | 재시작 | 시작 |
| 오른쪽 | [spk01_004.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk01_004.wav) | 왼쪽 | 오른쪽 |
| 오른쪽 | [spk01_135.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk01_135.wav) | 왼쪽 | 오른쪽 |
| 전진 | [spk01_159.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk01_159.wav) | 대기 | 전진 |
| 정지 | [spk01_200.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk01_200.wav) | 대기 | 정지 |
| 출발 | [spk01_128.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/출발/spk01_128.wav) | 재시작 | 출발 |
| 출발 | [spk01_129.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/출발/spk01_129.wav) | 시작 | 출발 |
| 출발 | [spk01_191.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/출발/spk01_191.wav) | 복귀 | 출발 |
| 후진 | [spk01_130.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk01_130.wav) | 시작 | 후진 |
| 후진 | [spk01_141.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk01_141.wav) | 복귀 | 후진 |
| 후진 | [spk01_172.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk01_172.wav) | 복귀 | 후진 |

## 세트0만 맞힌 파일: 16개

[CSV](spk01_set0_only_correct.csv)

| 정답 | 파일 | 세트0 공통 최고점 예측 | 세트3 공통 최고점 예측 |
| --- | --- | --- | --- |
| 대기 | [spk01_072.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/대기/spk01_072.wav) | 대기 | 복귀 |
| 복귀 | [spk01_164.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk01_164.wav) | 복귀 | silence |
| 복귀 | [spk01_184.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk01_184.wav) | 복귀 | silence |
| 시작 | [spk01_056.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk01_056.wav) | 시작 | 왼쪽 |
| 시작 | [spk01_074.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk01_074.wav) | 시작 | 후진 |
| 시작 | [spk01_104.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk01_104.wav) | 시작 | 출발 |
| 시작 | [spk01_165.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk01_165.wav) | 시작 | 출발 |
| 시작 | [spk01_185.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk01_185.wav) | 시작 | 후진 |
| 오른쪽 | [spk01_085.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk01_085.wav) | 오른쪽 | 복귀 |
| 왼쪽 | [spk01_197.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk01_197.wav) | 왼쪽 | 대기 |
| 전진 | [spk01_063.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk01_063.wav) | 전진 | 후진 |
| 전진 | [spk01_064.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk01_064.wav) | 전진 | 후진 |
| 정지 | [spk01_089.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk01_089.wav) | 정지 | 복귀 |
| 정지 | [spk01_160.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk01_160.wav) | 정지 | 대기 |
| 정지 | [spk01_190.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk01_190.wav) | 정지 | 후진 |
| 출발 | [spk01_171.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/출발/spk01_171.wav) | 출발 | 후진 |

## 둘 다 틀린 파일: 57개

[CSV](spk01_both_wrong.csv)

| 정답 | 파일 | 세트0 공통 최고점 예측 | 세트3 공통 최고점 예측 |
| --- | --- | --- | --- |
| 대기 | [spk01_082.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/대기/spk01_082.wav) | 복귀 | 복귀 |
| 시작 | [spk01_043.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk01_043.wav) | 재시작 | 재시작 |
| 오른쪽 | [spk01_044.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk01_044.wav) | 대기 | 대기 |
| 오른쪽 | [spk01_075.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk01_075.wav) | 복귀 | 복귀 |
| 오른쪽 | [spk01_105.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk01_105.wav) | 대기 | 대기 |
| 오른쪽 | [spk01_118.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk01_118.wav) | 후진 | 후진 |
| 오른쪽 | [spk01_119.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk01_119.wav) | 대기 | 대기 |
| 오른쪽 | [spk01_146.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk01_146.wav) | 복귀 | 후진 |
| 오른쪽 | [spk01_156.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk01_156.wav) | 왼쪽 | 복귀 |
| 오른쪽 | [spk01_166.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk01_166.wav) | 대기 | 대기 |
| 오른쪽 | [spk01_186.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk01_186.wav) | 복귀 | 후진 |
| 오른쪽 | [spk01_196.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk01_196.wav) | 대기 | 대기 |
| 왼쪽 | [spk01_045.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk01_045.wav) | 대기 | 대기 |
| 왼쪽 | [spk01_060.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk01_060.wav) | 오른쪽 | 오른쪽 |
| 왼쪽 | [spk01_076.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk01_076.wav) | 대기 | 출발 |
| 왼쪽 | [spk01_086.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk01_086.wav) | 복귀 | 복귀 |
| 왼쪽 | [spk01_106.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk01_106.wav) | 대기 | 대기 |
| 왼쪽 | [spk01_120.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk01_120.wav) | 대기 | 대기 |
| 왼쪽 | [spk01_121.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk01_121.wav) | 재시작 | 출발 |
| 왼쪽 | [spk01_147.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk01_147.wav) | 복귀 | 복귀 |
| 왼쪽 | [spk01_167.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk01_167.wav) | 복귀 | 복귀 |
| 왼쪽 | [spk01_187.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk01_187.wav) | 복귀 | 복귀 |
| 재시작 | [spk01_107.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/재시작/spk01_107.wav) | 시작 | 시작 |
| 재시작 | [spk01_137.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/재시작/spk01_137.wav) | 시작 | 시작 |
| 재시작 | [spk01_168.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/재시작/spk01_168.wav) | 대기 | 대기 |
| 재시작 | [spk01_188.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/재시작/spk01_188.wav) | 대기 | 대기 |
| 전진 | [spk01_007.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk01_007.wav) | 정지 | 복귀 |
| 전진 | [spk01_017.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk01_017.wav) | 후진 | 정지 |
| 전진 | [spk01_027.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk01_027.wav) | 정지 | 왼쪽 |
| 전진 | [spk01_037.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk01_037.wav) | 후진 | 후진 |
| 전진 | [spk01_047.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk01_047.wav) | 대기 | 대기 |
| 전진 | [spk01_078.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk01_078.wav) | 대기 | 후진 |
| 전진 | [spk01_088.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk01_088.wav) | 정지 | 정지 |
| 전진 | [spk01_098.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk01_098.wav) | 후진 | 정지 |
| 전진 | [spk01_108.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk01_108.wav) | 대기 | 대기 |
| 전진 | [spk01_124.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk01_124.wav) | 대기 | 복귀 |
| 전진 | [spk01_125.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk01_125.wav) | 재시작 | 복귀 |
| 전진 | [spk01_138.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk01_138.wav) | 정지 | 대기 |
| 전진 | [spk01_149.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk01_149.wav) | 대기 | 대기 |
| 전진 | [spk01_169.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk01_169.wav) | 대기 | 대기 |
| 전진 | [spk01_179.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk01_179.wav) | 대기 | 대기 |
| 전진 | [spk01_189.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk01_189.wav) | 대기 | 복귀 |
| 전진 | [spk01_199.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk01_199.wav) | 대기 | 대기 |
| 정지 | [spk01_008.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk01_008.wav) | 대기 | 대기 |
| 정지 | [spk01_038.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk01_038.wav) | 대기 | 대기 |
| 정지 | [spk01_048.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk01_048.wav) | 대기 | 대기 |
| 정지 | [spk01_079.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk01_079.wav) | 재시작 | 복귀 |
| 정지 | [spk01_099.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk01_099.wav) | 후진 | 후진 |
| 정지 | [spk01_109.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk01_109.wav) | 대기 | 대기 |
| 정지 | [spk01_126.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk01_126.wav) | 재시작 | 복귀 |
| 정지 | [spk01_127.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk01_127.wav) | 대기 | 후진 |
| 정지 | [spk01_150.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk01_150.wav) | 대기 | 복귀 |
| 정지 | [spk01_170.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk01_170.wav) | 대기 | 대기 |
| 정지 | [spk01_180.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk01_180.wav) | 대기 | 대기 |
| 후진 | [spk01_050.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk01_050.wav) | 대기 | 대기 |
| 후진 | [spk01_081.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk01_081.wav) | 시작 | 복귀 |
| 후진 | [spk01_192.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk01_192.wav) | 복귀 | 복귀 |

[전체203개와 확률·NLL·원본명·패딩](spk01_all_files.csv)

원래 개발 CV에서 얻은 사후 비교다. 독립 test나 파일 오류의 인과 판정은 아니다. 청취 검수는 하지 않았다.
