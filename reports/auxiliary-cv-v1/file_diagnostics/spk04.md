# spk04 파일별 진단

spk04를 학습에서 제외한 고유 CV fold의 seed44 실행이다. 원래 결과와 학습 history·최적 가중치 텐서·최적 예측이 동일한 재현을 확인했다.

기준은 모든 화자에 동일한 **epoch3 정답→epoch10 오답**이다. 파일이 원래부터 불량했다는 판정이 아니며, 같은 파일 입력에 대한 모델 예측이 바뀐 경우다.

203개 중 정답 153→177, 악화 10개 / 회복 34개. Macro F1 75.78%→86.23%, loss 0.7370→0.4044.

[전체 203개·epoch별 예측·확률·전처리](./spk04_all_files.csv) · [악화 목록 CSV](./spk04_regressed_e03_e10.csv) · [회복 목록 CSV](./spk04_recovered_e03_e10.csv) · [정답 확률이 낮아져 loss가 증가한 목록](./spk04_loss_worsened_e03_e10.csv)

## epoch3 정답→epoch10 오답 전체 파일

| 정답 | 파일 | epoch10 예측 | 원본 파일명 |
| --- | --- | --- | --- |
| 정지 | [spk04_144.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk04_144.wav) | 전진 | 이서준_정지_실내_1.m4a |
| 정지 | [spk04_145.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk04_145.wav) | 전진 | 이서준_정지_실내_2.m4a |
| 정지 | [spk04_147.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk04_147.wav) | 전진 | 이서준_정지_실내_4.m4a |
| 정지 | [spk04_148.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk04_148.wav) | 전진 | 이서준_정지_실내_5.m4a |
| 정지 | [spk04_151.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk04_151.wav) | 전진 | 이서준_정지_실내_8.m4a |
| 정지 | [spk04_154.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk04_154.wav) | 전진 | 이서준_정지_야외_2.m4a |
| 정지 | [spk04_157.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk04_157.wav) | 전진 | 이서준_정지_야외_5.m4a |
| 정지 | [spk04_163.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk04_163.wav) | 전진 | 이서준_정지_자택_2.m4a |
| 후진 | [spk04_189.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk04_189.wav) | 전진 | 이서준_후진_실내_6.m4a |
| 후진 | [spk04_203.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk04_203.wav) | 전진 | 이서준_후진_자택_2.m4a |

## 원래 선택 epoch9→10의 악화

원래 선택 시점부터 비교하면 정답→오답은 4개다. spk01은 고정된 원래 모델의 선택 시점을 썼다. 화자마다 선택 epoch가 달라 이 개수로 화자를 직접 순위 매기지 않는다.

[이 기준의 전체 목록 CSV](./spk04_regressed_selected_e10.csv)

| 정답 | 파일 | epoch10 예측 | 원본 파일명 |
| --- | --- | --- | --- |
| 복귀 | [spk04_038.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk04_038.wav) | 정지 | 이서준_복귀_야외_6.m4a |
| 복귀 | [spk04_041.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk04_041.wav) | 정지 | 이서준_복귀_야외_9.m4a |
| 왼쪽 | [spk04_103.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk04_103.wav) | 출발 | 이서준_왼쪽_자택_2.m4a |
| 재시작 | [spk04_119.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/재시작/spk04_119.wav) | 시작 | 이서준_재시작_야외_7.m4a |

## 원인 해석 범위

epoch3→10 악화 파일 10개 중 원본 부족분의 0 패딩이 있는 파일은 7개다. 추론 입력은 epoch마다 동일하다. 패딩·원본 길이·gain과 파일명 태그는 관찰 자료이며 오류 원인을 증명하지 않는다.

확률은 모델의 예측값이며 음성의 품질 점수가 아니다. 사람의 청취 검수는 하지 않았다. 같은 화자의 203개와 한 seed의 진단이므로 화자 전체의 성능으로 일반화하지 않는다.
