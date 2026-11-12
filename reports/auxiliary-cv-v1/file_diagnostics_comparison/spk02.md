# spk02: 보조 특징 제외/포함 비교

spk02를 학습에서 제외한 고유 CV fold.

seed44, mask+speed, weight decay 1e-4, Dropout 0, 나머지 조건·데이터·분할 동일. 세트0은 보조 없음, 세트3은 F1–F3 포먼트와 B1–B3 대역폭 보조다.

| 모델 | F1 epoch3→10 | 정답 /203 | 정답→오답 | 오답→정답 | 선택 epoch의 F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 보조 없음(세트0) | 50.13%→52.55% | 99→104 | 30 | 35 | epoch4: 62.82% |
| 보조 포함(세트3) | 70.14%→57.26% | 136→110 | 43 | 17 | epoch3: 70.14% |

검증/진단 loss epoch3→10: 세트0 1.7256→1.6870, 세트3 1.1441→1.8016.

원래 선택 epoch→10 악화: 세트0 33개, 세트3 43개. 선택 시점이 달라 epoch3→10과 구분한다.

## 같은 파일끼리 비교

두 모델이 epoch3에 모두 맞힌 83개 중 epoch10에 틀린 파일은 세트0 24개(28.9%), 세트3 23개(27.7%). 둘 다 악화한 파일은 14개다.

epoch10 전체203개에서 세트3만 맞힌 파일 25개, 세트0만 맞힌 파일 19개. 정답 수 차이는 +6개다.

[전체203개 짝 비교 CSV](spk02_all_files.csv) · [공통 epoch3 정답 집단](spk02_common_correct_e03.csv) · [세트3만 맞힌 파일](spk02_set3_gains_e10.csv) · [세트0만 맞힌 파일](spk02_set3_losses_e10.csv) · [단어별 비교](spk02_class_comparison.csv)

## 앞서 세트3에서 악화한 43개를 세트0과 대조

세트0에서도 악화 14개 / 세트0은 정답 유지 9개 / 세트0은 처음부터 오답 유지 19개 / 세트0은 회복 1개.

[이 목록 CSV](spk02_set3_regressed_paired.csv)

| 정답 | 파일 | 세트0 epoch3→10 | 세트3 epoch3→10 | 세트0 변화 |
| --- | --- | --- | --- | --- |
| 복귀 | [spk02_008.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_008.wav) | 대기→대기 | 복귀→대기 | 오답 유지 |
| 복귀 | [spk02_009.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_009.wav) | 복귀→대기 | 복귀→대기 | 정답→오답 |
| 복귀 | [spk02_010.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_010.wav) | 대기→대기 | 복귀→대기 | 오답 유지 |
| 복귀 | [spk02_013.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_013.wav) | 복귀→복귀 | 복귀→대기 | 정답 유지 |
| 복귀 | [spk02_070.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_070.wav) | 복귀→대기 | 복귀→대기 | 정답→오답 |
| 복귀 | [spk02_088.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_088.wav) | 대기→대기 | 복귀→대기 | 오답 유지 |
| 복귀 | [spk02_090.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_090.wav) | 대기→대기 | 복귀→대기 | 오답 유지 |
| 복귀 | [spk02_102.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_102.wav) | 대기→대기 | 복귀→대기 | 오답 유지 |
| 복귀 | [spk02_104.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_104.wav) | 대기→대기 | 복귀→대기 | 오답 유지 |
| 복귀 | [spk02_106.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_106.wav) | 복귀→대기 | 복귀→대기 | 정답→오답 |
| 시작 | [spk02_015.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk02_015.wav) | 후진→시작 | 시작→출발 | 오답→정답 |
| 시작 | [spk02_019.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk02_019.wav) | 후진→출발 | 시작→출발 | 오답 유지 |
| 시작 | [spk02_108.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk02_108.wav) | 출발→출발 | 시작→출발 | 오답 유지 |
| 시작 | [spk02_112.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk02_112.wav) | 후진→출발 | 시작→후진 | 오답 유지 |
| 시작 | [spk02_156.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk02_156.wav) | 후진→후진 | 시작→후진 | 오답 유지 |
| 오른쪽 | [spk02_022.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk02_022.wav) | 오른쪽→silence | 오른쪽→silence | 정답→오답 |
| 오른쪽 | [spk02_114.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk02_114.wav) | 후진→복귀 | 오른쪽→복귀 | 오답 유지 |
| 오른쪽 | [spk02_115.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk02_115.wav) | 오른쪽→복귀 | 오른쪽→복귀 | 정답→오답 |
| 오른쪽 | [spk02_118.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk02_118.wav) | 오른쪽→복귀 | 오른쪽→복귀 | 정답→오답 |
| 오른쪽 | [spk02_164.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk02_164.wav) | 오른쪽→전진 | 오른쪽→전진 | 정답→오답 |
| 왼쪽 | [spk02_119.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk02_119.wav) | 전진→대기 | 왼쪽→재시작 | 오답 유지 |
| 왼쪽 | [spk02_167.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk02_167.wav) | 오른쪽→복귀 | 왼쪽→복귀 | 오답 유지 |
| 재시작 | [spk02_037.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/재시작/spk02_037.wav) | 재시작→재시작 | 재시작→출발 | 정답 유지 |
| 전진 | [spk02_038.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk02_038.wav) | 전진→전진 | 전진→정지 | 정답 유지 |
| 전진 | [spk02_131.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk02_131.wav) | 전진→전진 | 전진→정지 | 정답 유지 |
| 전진 | [spk02_134.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk02_134.wav) | 전진→복귀 | 전진→복귀 | 정답→오답 |
| 전진 | [spk02_135.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk02_135.wav) | 전진→복귀 | 전진→복귀 | 정답→오답 |
| 전진 | [spk02_182.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk02_182.wav) | 전진→전진 | 전진→정지 | 정답 유지 |
| 전진 | [spk02_184.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk02_184.wav) | 전진→전진 | 전진→복귀 | 정답 유지 |
| 정지 | [spk02_044.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk02_044.wav) | 전진→전진 | 정지→복귀 | 오답 유지 |
| 정지 | [spk02_045.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk02_045.wav) | 재시작→재시작 | 정지→복귀 | 오답 유지 |
| 정지 | [spk02_138.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk02_138.wav) | 정지→재시작 | 정지→대기 | 정답→오답 |
| 정지 | [spk02_139.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk02_139.wav) | 전진→대기 | 정지→대기 | 오답 유지 |
| 정지 | [spk02_140.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk02_140.wav) | 복귀→복귀 | 정지→복귀 | 오답 유지 |
| 정지 | [spk02_141.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk02_141.wav) | 정지→대기 | 정지→복귀 | 정답→오답 |
| 정지 | [spk02_186.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk02_186.wav) | 전진→전진 | 정지→복귀 | 오답 유지 |
| 후진 | [spk02_061.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk02_061.wav) | 후진→재시작 | 후진→복귀 | 정답→오답 |
| 후진 | [spk02_086.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk02_086.wav) | 복귀→복귀 | 후진→복귀 | 오답 유지 |
| 후진 | [spk02_149.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk02_149.wav) | 후진→후진 | 후진→정지 | 정답 유지 |
| 후진 | [spk02_150.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk02_150.wav) | 후진→재시작 | 후진→대기 | 정답→오답 |
| 후진 | [spk02_152.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk02_152.wav) | 후진→복귀 | 후진→복귀 | 정답→오답 |
| 후진 | [spk02_153.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk02_153.wav) | 후진→후진 | 후진→대기 | 정답 유지 |
| 후진 | [spk02_203.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk02_203.wav) | 후진→후진 | 후진→복귀 | 정답 유지 |

## 세트0에서 악화한 전체 파일

[CSV](spk02_set0_regressed_paired.csv)

| 정답 | 파일 | 세트0 epoch3→10 | 세트3 epoch3→10 | 세트0 변화 |
| --- | --- | --- | --- | --- |
| 대기 | [spk02_005.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/대기/spk02_005.wav) | 대기→복귀 | 복귀→복귀 | 정답→오답 |
| 복귀 | [spk02_009.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_009.wav) | 복귀→대기 | 복귀→대기 | 정답→오답 |
| 복귀 | [spk02_011.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_011.wav) | 복귀→대기 | 복귀→복귀 | 정답→오답 |
| 복귀 | [spk02_070.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_070.wav) | 복귀→대기 | 복귀→대기 | 정답→오답 |
| 복귀 | [spk02_089.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_089.wav) | 복귀→재시작 | 복귀→복귀 | 정답→오답 |
| 복귀 | [spk02_092.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_092.wav) | 복귀→대기 | 복귀→복귀 | 정답→오답 |
| 복귀 | [spk02_093.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_093.wav) | 복귀→대기 | 복귀→복귀 | 정답→오답 |
| 복귀 | [spk02_106.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_106.wav) | 복귀→대기 | 복귀→대기 | 정답→오답 |
| 시작 | [spk02_109.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk02_109.wav) | 시작→출발 | 시작→시작 | 정답→오답 |
| 오른쪽 | [spk02_020.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk02_020.wav) | 오른쪽→복귀 | 오른쪽→오른쪽 | 정답→오답 |
| 오른쪽 | [spk02_022.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk02_022.wav) | 오른쪽→silence | 오른쪽→silence | 정답→오답 |
| 오른쪽 | [spk02_115.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk02_115.wav) | 오른쪽→복귀 | 오른쪽→복귀 | 정답→오답 |
| 오른쪽 | [spk02_116.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk02_116.wav) | 오른쪽→복귀 | 복귀→복귀 | 정답→오답 |
| 오른쪽 | [spk02_117.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk02_117.wav) | 오른쪽→복귀 | 오른쪽→오른쪽 | 정답→오답 |
| 오른쪽 | [spk02_118.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk02_118.wav) | 오른쪽→복귀 | 오른쪽→복귀 | 정답→오답 |
| 오른쪽 | [spk02_164.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk02_164.wav) | 오른쪽→전진 | 오른쪽→전진 | 정답→오답 |
| 전진 | [spk02_039.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk02_039.wav) | 전진→정지 | 정지→복귀 | 정답→오답 |
| 전진 | [spk02_132.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk02_132.wav) | 전진→재시작 | 복귀→대기 | 정답→오답 |
| 전진 | [spk02_133.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk02_133.wav) | 전진→복귀 | 정지→재시작 | 정답→오답 |
| 전진 | [spk02_134.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk02_134.wav) | 전진→복귀 | 전진→복귀 | 정답→오답 |
| 전진 | [spk02_135.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk02_135.wav) | 전진→복귀 | 전진→복귀 | 정답→오답 |
| 전진 | [spk02_136.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk02_136.wav) | 전진→복귀 | 정지→복귀 | 정답→오답 |
| 정지 | [spk02_138.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk02_138.wav) | 정지→재시작 | 정지→대기 | 정답→오답 |
| 정지 | [spk02_141.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk02_141.wav) | 정지→대기 | 정지→복귀 | 정답→오답 |
| 정지 | [spk02_142.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk02_142.wav) | 정지→재시작 | 정지→정지 | 정답→오답 |
| 후진 | [spk02_057.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk02_057.wav) | 후진→재시작 | 후진→후진 | 정답→오답 |
| 후진 | [spk02_061.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk02_061.wav) | 후진→재시작 | 후진→복귀 | 정답→오답 |
| 후진 | [spk02_150.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk02_150.wav) | 후진→재시작 | 후진→대기 | 정답→오답 |
| 후진 | [spk02_152.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk02_152.wav) | 후진→복귀 | 후진→복귀 | 정답→오답 |
| 후진 | [spk02_154.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk02_154.wav) | 후진→재시작 | 후진→후진 | 정답→오답 |

## 단어별 정답 수

| 단어 | 파일 수 | 세트0 epoch3→10 | 세트3 epoch3→10 |
| --- | ---: | ---: | ---: |
| silence | 3 | 3→3 | 3→3 |
| 대기 | 20 | 13→18 | 10→16 |
| 복귀 | 20 | 10→3 | 18→8 |
| 시작 | 20 | 1→2 | 10→5 |
| 오른쪽 | 20 | 16→9 | 18→13 |
| 왼쪽 | 20 | 3→9 | 11→9 |
| 재시작 | 20 | 11→16 | 12→16 |
| 전진 | 20 | 16→11 | 9→3 |
| 정지 | 20 | 4→3 | 15→8 |
| 출발 | 20 | 8→20 | 13→19 |
| 후진 | 20 | 14→10 | 17→10 |

## 해석 범위

한 seed의 사후 진단이다. 같은 epoch라도 두 모델의 수렴 진행 정도는 다르다. 두 모델 모두 epoch3에 맞힌 집단의 비율과 최종 정답/오답의 짝 비교도 함께 보아야 한다. 이 자료만으로 보조 특징이 특정 파일의 오류를 유발했다고 인과를 확정하지 않는다. 파일은 모든 epoch에 동일한 검증 파형이며, 청취 검수는 하지 않았다.
