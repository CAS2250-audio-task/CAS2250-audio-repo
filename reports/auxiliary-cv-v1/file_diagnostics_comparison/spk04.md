# spk04: 보조 특징 제외/포함 비교

spk04를 학습에서 제외한 고유 CV fold.

seed44, mask+speed, weight decay 1e-4, Dropout 0, 나머지 조건·데이터·분할 동일. 세트0은 보조 없음, 세트3은 F1–F3 포먼트와 B1–B3 대역폭 보조다.

| 모델 | F1 epoch3→10 | 정답 /203 | 정답→오답 | 오답→정답 | 선택 epoch의 F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 보조 없음(세트0) | 76.55%→84.17% | 154→175 | 7 | 28 | epoch9: 84.73% |
| 보조 포함(세트3) | 75.78%→86.23% | 153→177 | 10 | 34 | epoch9: 87.42% |

검증/진단 loss epoch3→10: 세트0 0.6605→0.4800, 세트3 0.7370→0.4044.

원래 선택 epoch→10 악화: 세트0 2개, 세트3 4개. 선택 시점이 달라 epoch3→10과 구분한다.

## 같은 파일끼리 비교

두 모델이 epoch3에 모두 맞힌 134개 중 epoch10에 틀린 파일은 세트0 5개(3.7%), 세트3 3개(2.2%). 둘 다 악화한 파일은 3개다.

epoch10 전체203개에서 세트3만 맞힌 파일 10개, 세트0만 맞힌 파일 8개. 정답 수 차이는 +2개다.

[전체203개 짝 비교 CSV](spk04_all_files.csv) · [공통 epoch3 정답 집단](spk04_common_correct_e03.csv) · [세트3만 맞힌 파일](spk04_set3_gains_e10.csv) · [세트0만 맞힌 파일](spk04_set3_losses_e10.csv) · [단어별 비교](spk04_class_comparison.csv)

## 앞서 세트3에서 악화한 10개를 세트0과 대조

세트0에서도 악화 3개 / 세트0은 정답 유지 0개 / 세트0은 처음부터 오답 유지 7개 / 세트0은 회복 0개.

[이 목록 CSV](spk04_set3_regressed_paired.csv)

| 정답 | 파일 | 세트0 epoch3→10 | 세트3 epoch3→10 | 세트0 변화 |
| --- | --- | --- | --- | --- |
| 정지 | [spk04_144.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk04_144.wav) | 정지→전진 | 정지→전진 | 정답→오답 |
| 정지 | [spk04_145.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk04_145.wav) | 전진→전진 | 정지→전진 | 오답 유지 |
| 정지 | [spk04_147.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk04_147.wav) | 전진→전진 | 정지→전진 | 오답 유지 |
| 정지 | [spk04_148.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk04_148.wav) | 전진→전진 | 정지→전진 | 오답 유지 |
| 정지 | [spk04_151.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk04_151.wav) | 전진→전진 | 정지→전진 | 오답 유지 |
| 정지 | [spk04_154.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk04_154.wav) | 정지→복귀 | 정지→전진 | 정답→오답 |
| 정지 | [spk04_157.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk04_157.wav) | 정지→전진 | 정지→전진 | 정답→오답 |
| 정지 | [spk04_163.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk04_163.wav) | 전진→전진 | 정지→전진 | 오답 유지 |
| 후진 | [spk04_189.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk04_189.wav) | 전진→전진 | 후진→전진 | 오답 유지 |
| 후진 | [spk04_203.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk04_203.wav) | 전진→왼쪽 | 후진→전진 | 오답 유지 |

## 세트0에서 악화한 전체 파일

[CSV](spk04_set0_regressed_paired.csv)

| 정답 | 파일 | 세트0 epoch3→10 | 세트3 epoch3→10 | 세트0 변화 |
| --- | --- | --- | --- | --- |
| 정지 | [spk04_144.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk04_144.wav) | 정지→전진 | 정지→전진 | 정답→오답 |
| 정지 | [spk04_152.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk04_152.wav) | 정지→전진 | 출발→전진 | 정답→오답 |
| 정지 | [spk04_154.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk04_154.wav) | 정지→복귀 | 정지→전진 | 정답→오답 |
| 정지 | [spk04_157.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk04_157.wav) | 정지→전진 | 정지→전진 | 정답→오답 |
| 정지 | [spk04_158.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk04_158.wav) | 정지→전진 | 정지→정지 | 정답→오답 |
| 후진 | [spk04_195.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk04_195.wav) | 후진→전진 | 후진→후진 | 정답→오답 |
| 후진 | [spk04_201.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk04_201.wav) | 후진→전진 | 시작→후진 | 정답→오답 |

## 단어별 정답 수

| 단어 | 파일 수 | 세트0 epoch3→10 | 세트3 epoch3→10 |
| --- | ---: | ---: | ---: |
| 대기 | 20 | 14→20 | 10→20 |
| silence | 3 | 2→2 | 2→2 |
| 복귀 | 20 | 20→20 | 17→18 |
| 시작 | 20 | 18→18 | 20→20 |
| 오른쪽 | 20 | 19→20 | 20→20 |
| 왼쪽 | 20 | 15→19 | 10→16 |
| 재시작 | 20 | 15→18 | 15→17 |
| 전진 | 20 | 13→19 | 11→20 |
| 정지 | 20 | 8→3 | 14→7 |
| 출발 | 20 | 12→20 | 17→19 |
| 후진 | 20 | 18→16 | 17→18 |

## 해석 범위

한 seed의 사후 진단이다. 같은 epoch라도 두 모델의 수렴 진행 정도는 다르다. 두 모델 모두 epoch3에 맞힌 집단의 비율과 최종 정답/오답의 짝 비교도 함께 보아야 한다. 이 자료만으로 보조 특징이 특정 파일의 오류를 유발했다고 인과를 확정하지 않는다. 파일은 모든 epoch에 동일한 검증 파형이며, 청취 검수는 하지 않았다.
