# spk02: 총합 기준의 공통 최고 epoch 비교

spk02를 학습에서 제외한 CV fold, seed44의 같은203개 파일이다.

전체3시드·4-fold 평균 Macro F1로 고른 세트0 epoch8, 세트3 epoch9을 모든 화자에 동일하게 적용했다.

| 모델 | F1 | 정답 /203 | loss |
| --- | ---: | ---: | ---: |
| 세트0 | 55.13% | 108 | 1.5684 |
| 세트3 | 59.10% | 111 | 1.9907 |

## 세트3만 맞힌 파일: 26개

[CSV](spk02_set3_only_correct.csv)

| 정답 | 파일 | 세트0 공통 최고점 예측 | 세트3 공통 최고점 예측 |
| --- | --- | --- | --- |
| 복귀 | [spk02_012.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_012.wav) | 대기 | 복귀 |
| 복귀 | [spk02_070.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_070.wav) | 대기 | 복귀 |
| 복귀 | [spk02_089.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_089.wav) | 재시작 | 복귀 |
| 복귀 | [spk02_090.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_090.wav) | 대기 | 복귀 |
| 복귀 | [spk02_091.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_091.wav) | 대기 | 복귀 |
| 복귀 | [spk02_092.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_092.wav) | 대기 | 복귀 |
| 복귀 | [spk02_093.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_093.wav) | 대기 | 복귀 |
| 복귀 | [spk02_104.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_104.wav) | 대기 | 복귀 |
| 복귀 | [spk02_106.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_106.wav) | 대기 | 복귀 |
| 시작 | [spk02_107.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk02_107.wav) | 출발 | 시작 |
| 시작 | [spk02_109.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk02_109.wav) | 출발 | 시작 |
| 시작 | [spk02_111.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk02_111.wav) | 출발 | 시작 |
| 시작 | [spk02_158.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk02_158.wav) | 출발 | 시작 |
| 오른쪽 | [spk02_020.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk02_020.wav) | 복귀 | 오른쪽 |
| 오른쪽 | [spk02_021.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk02_021.wav) | 재시작 | 오른쪽 |
| 오른쪽 | [spk02_025.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk02_025.wav) | 복귀 | 오른쪽 |
| 오른쪽 | [spk02_117.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk02_117.wav) | 복귀 | 오른쪽 |
| 오른쪽 | [spk02_164.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk02_164.wav) | 전진 | 오른쪽 |
| 왼쪽 | [spk02_119.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk02_119.wav) | 출발 | 왼쪽 |
| 왼쪽 | [spk02_167.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk02_167.wav) | 출발 | 왼쪽 |
| 재시작 | [spk02_174.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/재시작/spk02_174.wav) | 출발 | 재시작 |
| 재시작 | [spk02_175.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/재시작/spk02_175.wav) | 출발 | 재시작 |
| 재시작 | [spk02_177.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/재시작/spk02_177.wav) | 출발 | 재시작 |
| 정지 | [spk02_142.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk02_142.wav) | 출발 | 정지 |
| 후진 | [spk02_057.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk02_057.wav) | 복귀 | 후진 |
| 후진 | [spk02_198.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk02_198.wav) | 오른쪽 | 후진 |

## 세트0만 맞힌 파일: 23개

[CSV](spk02_set0_only_correct.csv)

| 정답 | 파일 | 세트0 공통 최고점 예측 | 세트3 공통 최고점 예측 |
| --- | --- | --- | --- |
| 대기 | [spk02_004.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/대기/spk02_004.wav) | 대기 | 복귀 |
| 대기 | [spk02_069.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/대기/spk02_069.wav) | 대기 | 복귀 |
| 복귀 | [spk02_013.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_013.wav) | 복귀 | 대기 |
| 시작 | [spk02_015.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk02_015.wav) | 시작 | 출발 |
| 왼쪽 | [spk02_121.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk02_121.wav) | 왼쪽 | 복귀 |
| 재시작 | [spk02_035.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/재시작/spk02_035.wav) | 재시작 | 출발 |
| 재시작 | [spk02_037.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/재시작/spk02_037.wav) | 재시작 | 출발 |
| 재시작 | [spk02_128.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/재시작/spk02_128.wav) | 재시작 | 복귀 |
| 전진 | [spk02_038.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk02_038.wav) | 전진 | 복귀 |
| 전진 | [spk02_042.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk02_042.wav) | 전진 | 복귀 |
| 전진 | [spk02_080.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk02_080.wav) | 전진 | 복귀 |
| 전진 | [spk02_081.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk02_081.wav) | 전진 | 정지 |
| 전진 | [spk02_131.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk02_131.wav) | 전진 | 정지 |
| 전진 | [spk02_180.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk02_180.wav) | 전진 | 복귀 |
| 전진 | [spk02_182.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk02_182.wav) | 전진 | 정지 |
| 전진 | [spk02_184.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk02_184.wav) | 전진 | 복귀 |
| 정지 | [spk02_139.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk02_139.wav) | 정지 | 대기 |
| 정지 | [spk02_141.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk02_141.wav) | 정지 | 복귀 |
| 출발 | [spk02_193.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/출발/spk02_193.wav) | 출발 | 정지 |
| 출발 | [spk02_195.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/출발/spk02_195.wav) | 출발 | 복귀 |
| 후진 | [spk02_149.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk02_149.wav) | 후진 | 정지 |
| 후진 | [spk02_153.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk02_153.wav) | 후진 | 대기 |
| 후진 | [spk02_203.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk02_203.wav) | 후진 | 복귀 |

## 둘 다 틀린 파일: 69개

[CSV](spk02_both_wrong.csv)

| 정답 | 파일 | 세트0 공통 최고점 예측 | 세트3 공통 최고점 예측 |
| --- | --- | --- | --- |
| 대기 | [spk02_005.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/대기/spk02_005.wav) | 복귀 | 복귀 |
| 대기 | [spk02_068.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/대기/spk02_068.wav) | 복귀 | 복귀 |
| 복귀 | [spk02_008.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_008.wav) | 대기 | 대기 |
| 복귀 | [spk02_009.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_009.wav) | 대기 | 대기 |
| 복귀 | [spk02_010.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_010.wav) | 대기 | 대기 |
| 복귀 | [spk02_088.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_088.wav) | 대기 | 대기 |
| 복귀 | [spk02_101.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_101.wav) | 대기 | 대기 |
| 복귀 | [spk02_102.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_102.wav) | 대기 | 대기 |
| 복귀 | [spk02_105.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_105.wav) | 대기 | 대기 |
| 시작 | [spk02_014.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk02_014.wav) | 출발 | 출발 |
| 시작 | [spk02_016.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk02_016.wav) | 출발 | 출발 |
| 시작 | [spk02_017.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk02_017.wav) | 복귀 | 출발 |
| 시작 | [spk02_018.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk02_018.wav) | 출발 | 후진 |
| 시작 | [spk02_019.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk02_019.wav) | 출발 | 출발 |
| 시작 | [spk02_072.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk02_072.wav) | 출발 | 출발 |
| 시작 | [spk02_073.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk02_073.wav) | 복귀 | 출발 |
| 시작 | [spk02_108.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk02_108.wav) | 출발 | 출발 |
| 시작 | [spk02_110.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk02_110.wav) | 복귀 | 복귀 |
| 시작 | [spk02_112.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk02_112.wav) | 출발 | 후진 |
| 시작 | [spk02_155.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk02_155.wav) | 출발 | 출발 |
| 시작 | [spk02_156.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk02_156.wav) | 후진 | 후진 |
| 시작 | [spk02_157.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk02_157.wav) | 출발 | 출발 |
| 시작 | [spk02_159.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk02_159.wav) | 출발 | 출발 |
| 시작 | [spk02_160.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk02_160.wav) | 출발 | 후진 |
| 오른쪽 | [spk02_022.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk02_022.wav) | silence | 복귀 |
| 오른쪽 | [spk02_023.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk02_023.wav) | 복귀 | 복귀 |
| 오른쪽 | [spk02_114.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk02_114.wav) | 복귀 | 복귀 |
| 오른쪽 | [spk02_115.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk02_115.wav) | 복귀 | 복귀 |
| 오른쪽 | [spk02_116.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk02_116.wav) | 복귀 | 복귀 |
| 오른쪽 | [spk02_118.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk02_118.wav) | 복귀 | 복귀 |
| 왼쪽 | [spk02_028.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk02_028.wav) | 복귀 | 복귀 |
| 왼쪽 | [spk02_029.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk02_029.wav) | 복귀 | 복귀 |
| 왼쪽 | [spk02_076.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk02_076.wav) | 복귀 | 후진 |
| 왼쪽 | [spk02_120.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk02_120.wav) | 복귀 | 복귀 |
| 왼쪽 | [spk02_122.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk02_122.wav) | 복귀 | 복귀 |
| 왼쪽 | [spk02_123.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk02_123.wav) | 복귀 | 복귀 |
| 왼쪽 | [spk02_124.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk02_124.wav) | 복귀 | 복귀 |
| 왼쪽 | [spk02_171.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk02_171.wav) | 오른쪽 | 오른쪽 |
| 재시작 | [spk02_078.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/재시작/spk02_078.wav) | 복귀 | 출발 |
| 재시작 | [spk02_079.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/재시작/spk02_079.wav) | 오른쪽 | 복귀 |
| 재시작 | [spk02_173.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/재시작/spk02_173.wav) | 출발 | 복귀 |
| 전진 | [spk02_039.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk02_039.wav) | 정지 | 복귀 |
| 전진 | [spk02_040.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk02_040.wav) | 복귀 | 복귀 |
| 전진 | [spk02_041.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk02_041.wav) | 복귀 | 복귀 |
| 전진 | [spk02_043.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk02_043.wav) | 재시작 | 복귀 |
| 전진 | [spk02_132.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk02_132.wav) | 출발 | 대기 |
| 전진 | [spk02_133.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk02_133.wav) | 복귀 | 복귀 |
| 전진 | [spk02_134.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk02_134.wav) | 복귀 | 복귀 |
| 전진 | [spk02_135.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk02_135.wav) | 복귀 | 복귀 |
| 전진 | [spk02_136.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk02_136.wav) | 복귀 | 복귀 |
| 정지 | [spk02_044.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk02_044.wav) | 전진 | 복귀 |
| 정지 | [spk02_045.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk02_045.wav) | 재시작 | 복귀 |
| 정지 | [spk02_046.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk02_046.wav) | 복귀 | 복귀 |
| 정지 | [spk02_047.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk02_047.wav) | 복귀 | 복귀 |
| 정지 | [spk02_049.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk02_049.wav) | 재시작 | 복귀 |
| 정지 | [spk02_082.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk02_082.wav) | 복귀 | 복귀 |
| 정지 | [spk02_083.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk02_083.wav) | 대기 | 대기 |
| 정지 | [spk02_138.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk02_138.wav) | 재시작 | 대기 |
| 정지 | [spk02_140.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk02_140.wav) | 복귀 | 복귀 |
| 정지 | [spk02_185.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk02_185.wav) | 전진 | 복귀 |
| 정지 | [spk02_186.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk02_186.wav) | 전진 | 복귀 |
| 정지 | [spk02_189.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk02_189.wav) | 전진 | 복귀 |
| 후진 | [spk02_058.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk02_058.wav) | 오른쪽 | 복귀 |
| 후진 | [spk02_059.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk02_059.wav) | 복귀 | 복귀 |
| 후진 | [spk02_061.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk02_061.wav) | 재시작 | 복귀 |
| 후진 | [spk02_086.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk02_086.wav) | 복귀 | 복귀 |
| 후진 | [spk02_087.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk02_087.wav) | 복귀 | 복귀 |
| 후진 | [spk02_150.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk02_150.wav) | 재시작 | 대기 |
| 후진 | [spk02_152.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk02_152.wav) | 복귀 | 복귀 |

[전체203개와 확률·NLL·원본명·패딩](spk02_all_files.csv)

원래 개발 CV에서 얻은 사후 비교다. 독립 test나 파일 오류의 인과 판정은 아니다. 청취 검수는 하지 않았다.
