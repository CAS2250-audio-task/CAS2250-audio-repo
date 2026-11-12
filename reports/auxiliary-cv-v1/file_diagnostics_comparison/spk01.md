# spk01: 보조 특징 제외/포함 비교

spk02 fold 모델(학습 spk03/04/05)의 별도 평가. spk01에 학습·scaler 적합·epoch 선택을 하지 않았다.

seed44, mask+speed, weight decay 1e-4, Dropout 0, 나머지 조건·데이터·분할 동일. 세트0은 보조 없음, 세트3은 F1–F3 포먼트와 B1–B3 대역폭 보조다.

| 모델 | F1 epoch3→10 | 정답 /203 | 정답→오답 | 오답→정답 | 선택 epoch의 F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 보조 없음(세트0) | 56.36%→65.23% | 113→131 | 19 | 37 | epoch4: 58.46% |
| 보조 포함(세트3) | 71.86%→64.12% | 140→133 | 24 | 17 | epoch3: 71.86% |

검증/진단 loss epoch3→10: 세트0 1.6819→1.6001, 세트3 1.1088→1.4608.

원래 선택 epoch→10 악화: 세트0 12개, 세트3 24개. 선택 시점이 달라 epoch3→10과 구분한다.

## 같은 파일끼리 비교

두 모델이 epoch3에 모두 맞힌 102개 중 epoch10에 틀린 파일은 세트0 13개(12.7%), 세트3 13개(12.7%). 둘 다 악화한 파일은 9개다.

epoch10 전체203개에서 세트3만 맞힌 파일 13개, 세트0만 맞힌 파일 11개. 정답 수 차이는 +2개다.

[전체203개 짝 비교 CSV](spk01_all_files.csv) · [공통 epoch3 정답 집단](spk01_common_correct_e03.csv) · [세트3만 맞힌 파일](spk01_set3_gains_e10.csv) · [세트0만 맞힌 파일](spk01_set3_losses_e10.csv) · [단어별 비교](spk01_class_comparison.csv)

## 앞서 세트3에서 악화한 24개를 세트0과 대조

세트0에서도 악화 9개 / 세트0은 정답 유지 4개 / 세트0은 처음부터 오답 유지 9개 / 세트0은 회복 2개.

[이 목록 CSV](spk01_set3_regressed_paired.csv)

| 정답 | 파일 | 세트0 epoch3→10 | 세트3 epoch3→10 | 세트0 변화 |
| --- | --- | --- | --- | --- |
| 시작 | [spk01_104.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk01_104.wav) | 시작→시작 | 시작→출발 | 정답 유지 |
| 오른쪽 | [spk01_044.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk01_044.wav) | 오른쪽→대기 | 오른쪽→대기 | 정답→오답 |
| 오른쪽 | [spk01_075.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk01_075.wav) | 오른쪽→복귀 | 오른쪽→복귀 | 정답→오답 |
| 오른쪽 | [spk01_085.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk01_085.wav) | 오른쪽→오른쪽 | 오른쪽→복귀 | 정답 유지 |
| 오른쪽 | [spk01_146.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk01_146.wav) | 오른쪽→복귀 | 오른쪽→후진 | 정답→오답 |
| 왼쪽 | [spk01_106.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk01_106.wav) | 대기→대기 | 왼쪽→대기 | 오답 유지 |
| 왼쪽 | [spk01_157.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk01_157.wav) | 전진→왼쪽 | 왼쪽→대기 | 오답→정답 |
| 왼쪽 | [spk01_197.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk01_197.wav) | 복귀→왼쪽 | 왼쪽→대기 | 오답→정답 |
| 재시작 | [spk01_107.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/재시작/spk01_107.wav) | 시작→시작 | 재시작→시작 | 오답 유지 |
| 전진 | [spk01_017.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk01_017.wav) | 전진→후진 | 전진→정지 | 정답→오답 |
| 전진 | [spk01_027.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk01_027.wav) | 전진→정지 | 전진→정지 | 정답→오답 |
| 전진 | [spk01_064.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk01_064.wav) | 전진→전진 | 전진→정지 | 정답 유지 |
| 전진 | [spk01_098.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk01_098.wav) | 오른쪽→후진 | 전진→정지 | 오답 유지 |
| 전진 | [spk01_108.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk01_108.wav) | 대기→대기 | 전진→대기 | 오답 유지 |
| 전진 | [spk01_124.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk01_124.wav) | 대기→대기 | 전진→대기 | 오답 유지 |
| 전진 | [spk01_138.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk01_138.wav) | 전진→정지 | 전진→대기 | 정답→오답 |
| 전진 | [spk01_149.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk01_149.wav) | 대기→대기 | 전진→대기 | 오답 유지 |
| 전진 | [spk01_169.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk01_169.wav) | 대기→대기 | 전진→대기 | 오답 유지 |
| 전진 | [spk01_199.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk01_199.wav) | 대기→대기 | 전진→대기 | 오답 유지 |
| 정지 | [spk01_089.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk01_089.wav) | 정지→정지 | 정지→복귀 | 정답 유지 |
| 정지 | [spk01_150.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk01_150.wav) | 대기→대기 | 정지→대기 | 오답 유지 |
| 후진 | [spk01_050.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk01_050.wav) | 후진→대기 | 후진→대기 | 정답→오답 |
| 후진 | [spk01_081.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk01_081.wav) | 후진→시작 | 후진→복귀 | 정답→오답 |
| 후진 | [spk01_192.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk01_192.wav) | 후진→복귀 | 후진→복귀 | 정답→오답 |

## 세트0에서 악화한 전체 파일

[CSV](spk01_set0_regressed_paired.csv)

| 정답 | 파일 | 세트0 epoch3→10 | 세트3 epoch3→10 | 세트0 변화 |
| --- | --- | --- | --- | --- |
| 시작 | [spk01_165.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk01_165.wav) | 시작→재시작 | 출발→출발 | 정답→오답 |
| 오른쪽 | [spk01_044.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk01_044.wav) | 오른쪽→대기 | 오른쪽→대기 | 정답→오답 |
| 오른쪽 | [spk01_075.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk01_075.wav) | 오른쪽→복귀 | 오른쪽→복귀 | 정답→오답 |
| 오른쪽 | [spk01_105.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk01_105.wav) | 오른쪽→대기 | 대기→대기 | 정답→오답 |
| 오른쪽 | [spk01_135.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk01_135.wav) | 오른쪽→복귀 | 오른쪽→오른쪽 | 정답→오답 |
| 오른쪽 | [spk01_146.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk01_146.wav) | 오른쪽→복귀 | 오른쪽→후진 | 정답→오답 |
| 왼쪽 | [spk01_120.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk01_120.wav) | 왼쪽→대기 | 복귀→대기 | 정답→오답 |
| 전진 | [spk01_017.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk01_017.wav) | 전진→후진 | 전진→정지 | 정답→오답 |
| 전진 | [spk01_027.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk01_027.wav) | 전진→정지 | 전진→정지 | 정답→오답 |
| 전진 | [spk01_047.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk01_047.wav) | 전진→대기 | 후진→대기 | 정답→오답 |
| 전진 | [spk01_088.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk01_088.wav) | 전진→정지 | 정지→정지 | 정답→오답 |
| 전진 | [spk01_138.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk01_138.wav) | 전진→정지 | 전진→대기 | 정답→오답 |
| 전진 | [spk01_159.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk01_159.wav) | 전진→대기 | 전진→전진 | 정답→오답 |
| 정지 | [spk01_066.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk01_066.wav) | 정지→대기 | 오른쪽→정지 | 정답→오답 |
| 후진 | [spk01_050.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk01_050.wav) | 후진→대기 | 후진→대기 | 정답→오답 |
| 후진 | [spk01_081.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk01_081.wav) | 후진→시작 | 후진→복귀 | 정답→오답 |
| 후진 | [spk01_141.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk01_141.wav) | 후진→복귀 | 후진→후진 | 정답→오답 |
| 후진 | [spk01_172.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk01_172.wav) | 후진→복귀 | 후진→후진 | 정답→오답 |
| 후진 | [spk01_192.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk01_192.wav) | 후진→복귀 | 후진→복귀 | 정답→오답 |

## 단어별 정답 수

| 단어 | 파일 수 | 세트0 epoch3→10 | 세트3 epoch3→10 |
| --- | ---: | ---: | ---: |
| 대기 | 20 | 18→19 | 16→18 |
| 복귀 | 20 | 13→20 | 17→18 |
| 시작 | 20 | 15→17 | 15→16 |
| 오른쪽 | 20 | 14→9 | 13→9 |
| 왼쪽 | 20 | 4→9 | 11→8 |
| 재시작 | 20 | 8→16 | 15→16 |
| 전진 | 20 | 8→2 | 12→2 |
| 정지 | 20 | 3→6 | 6→7 |
| 출발 | 20 | 7→15 | 14→19 |
| 후진 | 20 | 20→15 | 18→17 |
| silence | 3 | 3→3 | 3→3 |

## 해석 범위

한 seed의 사후 진단이다. 같은 epoch라도 두 모델의 수렴 진행 정도는 다르다. 두 모델 모두 epoch3에 맞힌 집단의 비율과 최종 정답/오답의 짝 비교도 함께 보아야 한다. 이 자료만으로 보조 특징이 특정 파일의 오류를 유발했다고 인과를 확정하지 않는다. 파일은 모든 epoch에 동일한 검증 파형이며, 청취 검수는 하지 않았다.
