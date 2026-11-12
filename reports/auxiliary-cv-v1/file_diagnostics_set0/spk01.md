# spk01 파일별 진단

spk02 검증 fold의 모델(학습 spk03/04/05)을 spk01에 별도 평가했다. 선택 epoch4은 원래 spk02 기준이며 spk01 결과로 고른 것이 아니다. 주 비교 epoch3→10은 두 모델 모두 고정했다.

기준은 모든 화자에 동일한 **epoch3 정답→epoch10 오답**이다. 파일이 원래부터 불량했다는 판정이 아니며, 같은 파일 입력에 대한 모델 예측이 바뀐 경우다.

203개 중 정답 113→131, 악화 19개 / 회복 37개. Macro F1 56.36%→65.23%, loss 1.6819→1.6001.

[전체 203개·epoch별 예측·확률·전처리](./spk01_all_files.csv) · [악화 목록 CSV](./spk01_regressed_e03_e10.csv) · [회복 목록 CSV](./spk01_recovered_e03_e10.csv) · [정답 확률이 낮아져 loss가 증가한 목록](./spk01_loss_worsened_e03_e10.csv)

## epoch3 정답→epoch10 오답 전체 파일

| 정답 | 파일 | epoch10 예측 | 원본 파일명 |
| --- | --- | --- | --- |
| 시작 | [spk01_165.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk01_165.wav) | 재시작 | 야2o시작.m4a |
| 오른쪽 | [spk01_044.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk01_044.wav) | 대기 | 방5o오른쪽.m4a |
| 오른쪽 | [spk01_075.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk01_075.wav) | 복귀 | 복15o오른쪽.m4a |
| 오른쪽 | [spk01_105.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk01_105.wav) | 대기 | 복2x오른쪽.m4a |
| 오른쪽 | [spk01_135.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk01_135.wav) | 복귀 | 복5x오른쪽.m4a |
| 오른쪽 | [spk01_146.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk01_146.wav) | 복귀 | 야15o오른쪽.m4a |
| 왼쪽 | [spk01_120.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk01_120.wav) | 대기 | 복5o왼쪽.m4a |
| 전진 | [spk01_017.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk01_017.wav) | 후진 | 방15x전진.m4a |
| 전진 | [spk01_027.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk01_027.wav) | 정지 | 방2o전진.m4a |
| 전진 | [spk01_047.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk01_047.wav) | 대기 | 방5o전진.m4a |
| 전진 | [spk01_088.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk01_088.wav) | 정지 | 복15x전진.m4a |
| 전진 | [spk01_138.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk01_138.wav) | 정지 | 복5x전진.m4a |
| 전진 | [spk01_159.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk01_159.wav) | 대기 | 야15x전진.m4a |
| 정지 | [spk01_066.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk01_066.wav) | 대기 | 방6x정지_.m4a |
| 후진 | [spk01_050.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk01_050.wav) | 대기 | 방5o후진.m4a |
| 후진 | [spk01_081.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk01_081.wav) | 시작 | 복15o후진.m4a |
| 후진 | [spk01_141.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk01_141.wav) | 복귀 | 복5x후진.m4a |
| 후진 | [spk01_172.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk01_172.wav) | 복귀 | 야2o후진.m4a |
| 후진 | [spk01_192.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk01_192.wav) | 복귀 | 야5o후진.m4a |

## 원래 선택 epoch4→10의 악화

원래 선택 시점부터 비교하면 정답→오답은 12개다. spk01은 고정된 원래 모델의 선택 시점을 썼다. 화자마다 선택 epoch가 달라 이 개수로 화자를 직접 순위 매기지 않는다.

[이 기준의 전체 목록 CSV](./spk01_regressed_selected_e10.csv)

| 정답 | 파일 | epoch10 예측 | 원본 파일명 |
| --- | --- | --- | --- |
| 대기 | [spk01_082.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/대기/spk01_082.wav) | 복귀 | 복15x대기.m4a |
| 시작 | [spk01_043.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk01_043.wav) | 재시작 | 방5o시작.m4a |
| 시작 | [spk01_165.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk01_165.wav) | 재시작 | 야2o시작.m4a |
| 오른쪽 | [spk01_135.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk01_135.wav) | 복귀 | 복5x오른쪽.m4a |
| 왼쪽 | [spk01_035.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk01_035.wav) | 오른쪽 | 방2x왼쪽.m4a |
| 전진 | [spk01_088.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk01_088.wav) | 정지 | 복15x전진.m4a |
| 전진 | [spk01_138.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk01_138.wav) | 정지 | 복5x전진.m4a |
| 정지 | [spk01_066.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk01_066.wav) | 대기 | 방6x정지_.m4a |
| 출발 | [spk01_151.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/출발/spk01_151.wav) | 복귀 | 야15o출발.m4a |
| 후진 | [spk01_141.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk01_141.wav) | 복귀 | 복5x후진.m4a |
| 후진 | [spk01_172.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk01_172.wav) | 복귀 | 야2o후진.m4a |
| 후진 | [spk01_192.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk01_192.wav) | 복귀 | 야5o후진.m4a |

## 원인 해석 범위

epoch3→10 악화 파일 19개 중 원본 부족분의 0 패딩이 있는 파일은 9개다. 추론 입력은 epoch마다 동일하다. 패딩·원본 길이·gain과 파일명 태그는 관찰 자료이며 오류 원인을 증명하지 않는다.

확률은 모델의 예측값이며 음성의 품질 점수가 아니다. 사람의 청취 검수는 하지 않았다. 같은 화자의 203개와 한 seed의 진단이므로 화자 전체의 성능으로 일반화하지 않는다.
