# spk03: 보조 특징 제외/포함 비교

spk03를 학습에서 제외한 고유 CV fold.

seed44, mask+speed, weight decay 1e-4, Dropout 0, 나머지 조건·데이터·분할 동일. 세트0은 보조 없음, 세트3은 F1–F3 포먼트와 B1–B3 대역폭 보조다.

| 모델 | F1 epoch3→10 | 정답 /203 | 정답→오답 | 오답→정답 | 선택 epoch의 F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 보조 없음(세트0) | 66.33%→84.22% | 128→169 | 6 | 47 | epoch8: 87.02% |
| 보조 포함(세트3) | 72.02%→85.61% | 149→172 | 10 | 33 | epoch7: 87.59% |

검증/진단 loss epoch3→10: 세트0 1.5866→0.4705, 세트3 0.8648→0.5253.

원래 선택 epoch→10 악화: 세트0 8개, 세트3 7개. 선택 시점이 달라 epoch3→10과 구분한다.

## 같은 파일끼리 비교

두 모델이 epoch3에 모두 맞힌 113개 중 epoch10에 틀린 파일은 세트0 5개(4.4%), 세트3 4개(3.5%). 둘 다 악화한 파일은 2개다.

epoch10 전체203개에서 세트3만 맞힌 파일 13개, 세트0만 맞힌 파일 10개. 정답 수 차이는 +3개다.

[전체203개 짝 비교 CSV](spk03_all_files.csv) · [공통 epoch3 정답 집단](spk03_common_correct_e03.csv) · [세트3만 맞힌 파일](spk03_set3_gains_e10.csv) · [세트0만 맞힌 파일](spk03_set3_losses_e10.csv) · [단어별 비교](spk03_class_comparison.csv)

## 앞서 세트3에서 악화한 10개를 세트0과 대조

세트0에서도 악화 2개 / 세트0은 정답 유지 2개 / 세트0은 처음부터 오답 유지 2개 / 세트0은 회복 4개.

[이 목록 CSV](spk03_set3_regressed_paired.csv)

| 정답 | 파일 | 세트0 epoch3→10 | 세트3 epoch3→10 | 세트0 변화 |
| --- | --- | --- | --- | --- |
| 복귀 | [spk03_038.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk03_038.wav) | 복귀→대기 | 복귀→대기 | 정답→오답 |
| 오른쪽 | [spk03_074.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk03_074.wav) | 왼쪽→오른쪽 | 오른쪽→왼쪽 | 오답→정답 |
| 오른쪽 | [spk03_076.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk03_076.wav) | 왼쪽→왼쪽 | 오른쪽→왼쪽 | 오답 유지 |
| 오른쪽 | [spk03_077.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk03_077.wav) | 왼쪽→오른쪽 | 오른쪽→왼쪽 | 오답→정답 |
| 오른쪽 | [spk03_081.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk03_081.wav) | 왼쪽→오른쪽 | 오른쪽→왼쪽 | 오답→정답 |
| 오른쪽 | [spk03_083.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk03_083.wav) | 왼쪽→오른쪽 | 오른쪽→왼쪽 | 오답→정답 |
| 전진 | [spk03_132.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk03_132.wav) | 전진→정지 | 전진→정지 | 정답→오답 |
| 후진 | [spk03_192.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk03_192.wav) | 시작→시작 | 후진→시작 | 오답 유지 |
| 후진 | [spk03_193.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk03_193.wav) | 후진→후진 | 후진→왼쪽 | 정답 유지 |
| 후진 | [spk03_194.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk03_194.wav) | 후진→후진 | 후진→시작 | 정답 유지 |

## 세트0에서 악화한 전체 파일

[CSV](spk03_set0_regressed_paired.csv)

| 정답 | 파일 | 세트0 epoch3→10 | 세트3 epoch3→10 | 세트0 변화 |
| --- | --- | --- | --- | --- |
| 복귀 | [spk03_038.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk03_038.wav) | 복귀→대기 | 복귀→대기 | 정답→오답 |
| 복귀 | [spk03_041.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk03_041.wav) | 복귀→대기 | 복귀→복귀 | 정답→오답 |
| 오른쪽 | [spk03_068.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk03_068.wav) | 오른쪽→왼쪽 | 오른쪽→오른쪽 | 정답→오답 |
| 오른쪽 | [spk03_082.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk03_082.wav) | 오른쪽→전진 | 오른쪽→오른쪽 | 정답→오답 |
| 왼쪽 | [spk03_102.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk03_102.wav) | 왼쪽→후진 | 오른쪽→왼쪽 | 정답→오답 |
| 전진 | [spk03_132.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk03_132.wav) | 전진→정지 | 전진→정지 | 정답→오답 |

## 단어별 정답 수

| 단어 | 파일 수 | 세트0 epoch3→10 | 세트3 epoch3→10 |
| --- | ---: | ---: | ---: |
| 대기 | 20 | 18→20 | 19→20 |
| silence | 3 | 3→3 | 3→3 |
| 복귀 | 20 | 13→16 | 17→17 |
| 시작 | 20 | 19→20 | 19→20 |
| 오른쪽 | 20 | 10→16 | 19→14 |
| 왼쪽 | 20 | 19→18 | 12→20 |
| 재시작 | 20 | 6→11 | 8→15 |
| 전진 | 20 | 6→12 | 10→10 |
| 정지 | 20 | 5→14 | 7→16 |
| 출발 | 20 | 11→20 | 16→20 |
| 후진 | 20 | 18→19 | 19→17 |

## 해석 범위

한 seed의 사후 진단이다. 같은 epoch라도 두 모델의 수렴 진행 정도는 다르다. 두 모델 모두 epoch3에 맞힌 집단의 비율과 최종 정답/오답의 짝 비교도 함께 보아야 한다. 이 자료만으로 보조 특징이 특정 파일의 오류를 유발했다고 인과를 확정하지 않는다. 파일은 모든 epoch에 동일한 검증 파형이며, 청취 검수는 하지 않았다.
