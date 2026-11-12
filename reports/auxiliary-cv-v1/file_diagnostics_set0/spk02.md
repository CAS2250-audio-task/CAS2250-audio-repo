# spk02 파일별 진단

spk02를 학습에서 제외한 고유 CV fold의 seed44 실행이다. 원래 결과와 학습 history·최적 가중치 텐서·최적 예측이 동일한 재현을 확인했다.

기준은 모든 화자에 동일한 **epoch3 정답→epoch10 오답**이다. 파일이 원래부터 불량했다는 판정이 아니며, 같은 파일 입력에 대한 모델 예측이 바뀐 경우다.

203개 중 정답 99→104, 악화 30개 / 회복 35개. Macro F1 50.13%→52.55%, loss 1.7256→1.6870.

[전체 203개·epoch별 예측·확률·전처리](./spk02_all_files.csv) · [악화 목록 CSV](./spk02_regressed_e03_e10.csv) · [회복 목록 CSV](./spk02_recovered_e03_e10.csv) · [정답 확률이 낮아져 loss가 증가한 목록](./spk02_loss_worsened_e03_e10.csv)

## epoch3 정답→epoch10 오답 전체 파일

| 정답 | 파일 | epoch10 예측 | 원본 파일명 |
| --- | --- | --- | --- |
| 대기 | [spk02_005.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/대기/spk02_005.wav) | 복귀 | 거리대기작게50cm.m4a |
| 복귀 | [spk02_009.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_009.wav) | 대기 | 거리복귀빠르게50cm.m4a |
| 복귀 | [spk02_011.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_011.wav) | 대기 | 거리복귀작게50cm.m4a |
| 복귀 | [spk02_070.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_070.wav) | 대기 | 백양복귀빠르게20cm.m4a |
| 복귀 | [spk02_089.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_089.wav) | 재시작 | 복귀빠르게50cm.m4a |
| 복귀 | [spk02_092.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_092.wav) | 대기 | 복귀정상20cm.m4a |
| 복귀 | [spk02_093.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_093.wav) | 대기 | 복귀정상50cm.m4a |
| 복귀 | [spk02_106.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_106.wav) | 대기 | 복도복귀정상50cm.m4a |
| 시작 | [spk02_109.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk02_109.wav) | 출발 | 복도시작작게20cm.m4a |
| 오른쪽 | [spk02_020.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk02_020.wav) | 복귀 | 거리오른쪽빠르게20cm.m4a |
| 오른쪽 | [spk02_022.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk02_022.wav) | silence | 거리오른쪽작게20cm.m4a |
| 오른쪽 | [spk02_115.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk02_115.wav) | 복귀 | 복도오른쪽작게20cm.m4a |
| 오른쪽 | [spk02_116.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk02_116.wav) | 복귀 | 복도오른쪽작게50cm.m4a |
| 오른쪽 | [spk02_117.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk02_117.wav) | 복귀 | 복도오른쪽정상20cm.m4a |
| 오른쪽 | [spk02_118.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk02_118.wav) | 복귀 | 복도오른쪽정상50cm.m4a |
| 오른쪽 | [spk02_164.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk02_164.wav) | 전진 | 오른쪽작게50cm.m4a |
| 전진 | [spk02_039.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk02_039.wav) | 정지 | 거리전진빠르게50cm.m4a |
| 전진 | [spk02_132.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk02_132.wav) | 재시작 | 복도전진빠르게50cm.m4a |
| 전진 | [spk02_133.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk02_133.wav) | 복귀 | 복도전진작게20cm.m4a |
| 전진 | [spk02_134.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk02_134.wav) | 복귀 | 복도전진작게50cm.m4a |
| 전진 | [spk02_135.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk02_135.wav) | 복귀 | 복도전진정상20cm.m4a |
| 전진 | [spk02_136.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk02_136.wav) | 복귀 | 복도전진정상50cm.m4a |
| 정지 | [spk02_138.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk02_138.wav) | 재시작 | 복도정지빠르게50cm.m4a |
| 정지 | [spk02_141.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk02_141.wav) | 대기 | 복도정지정상20cm.m4a |
| 정지 | [spk02_142.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk02_142.wav) | 재시작 | 복도정지정상50cm.m4a |
| 후진 | [spk02_057.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk02_057.wav) | 재시작 | 거리후진빠르게50cm.m4a |
| 후진 | [spk02_061.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk02_061.wav) | 재시작 | 거리후진정상50cm.m4a |
| 후진 | [spk02_150.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk02_150.wav) | 재시작 | 복도후진빠르게50cm.m4a |
| 후진 | [spk02_152.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk02_152.wav) | 복귀 | 복도후진작게50cm.m4a |
| 후진 | [spk02_154.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk02_154.wav) | 재시작 | 복도후진정상50cm.m4a |

## 원래 선택 epoch4→10의 악화

원래 선택 시점부터 비교하면 정답→오답은 33개다. spk01은 고정된 원래 모델의 선택 시점을 썼다. 화자마다 선택 epoch가 달라 이 개수로 화자를 직접 순위 매기지 않는다.

[이 기준의 전체 목록 CSV](./spk02_regressed_selected_e10.csv)

| 정답 | 파일 | epoch10 예측 | 원본 파일명 |
| --- | --- | --- | --- |
| 대기 | [spk02_005.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/대기/spk02_005.wav) | 복귀 | 거리대기작게50cm.m4a |
| 복귀 | [spk02_092.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_092.wav) | 대기 | 복귀정상20cm.m4a |
| 복귀 | [spk02_093.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_093.wav) | 대기 | 복귀정상50cm.m4a |
| 시작 | [spk02_019.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk02_019.wav) | 출발 | 거리시작정상50cm.m4a |
| 시작 | [spk02_109.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk02_109.wav) | 출발 | 복도시작작게20cm.m4a |
| 시작 | [spk02_112.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk02_112.wav) | 출발 | 복도시작정상50cm.m4a |
| 시작 | [spk02_158.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk02_158.wav) | 후진 | 시작작게50cm.m4a |
| 시작 | [spk02_159.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk02_159.wav) | 출발 | 시작정상20cm.m4a |
| 시작 | [spk02_160.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk02_160.wav) | 후진 | 시작정상50cm.m4a |
| 오른쪽 | [spk02_117.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk02_117.wav) | 복귀 | 복도오른쪽정상20cm.m4a |
| 오른쪽 | [spk02_164.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk02_164.wav) | 전진 | 오른쪽작게50cm.m4a |
| 왼쪽 | [spk02_167.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk02_167.wav) | 복귀 | 왼쪽빠르게20cm.m4a |
| 왼쪽 | [spk02_171.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk02_171.wav) | 오른쪽 | 왼쪽정상20cm.m4a |
| 재시작 | [spk02_177.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/재시작/spk02_177.wav) | 전진 | 재시작정상20cm.m4a |
| 전진 | [spk02_132.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk02_132.wav) | 재시작 | 복도전진빠르게50cm.m4a |
| 전진 | [spk02_135.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk02_135.wav) | 복귀 | 복도전진정상20cm.m4a |
| 정지 | [spk02_044.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk02_044.wav) | 전진 | 거리정지빠르게20cm.m4a |
| 정지 | [spk02_137.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk02_137.wav) | 재시작 | 복도정지빠르게20cm.m4a |
| 정지 | [spk02_138.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk02_138.wav) | 재시작 | 복도정지빠르게50cm.m4a |
| 정지 | [spk02_139.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk02_139.wav) | 대기 | 복도정지작게20cm.m4a |
| 정지 | [spk02_140.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk02_140.wav) | 복귀 | 복도정지작게50cm.m4a |
| 정지 | [spk02_141.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk02_141.wav) | 대기 | 복도정지정상20cm.m4a |
| 정지 | [spk02_142.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk02_142.wav) | 재시작 | 복도정지정상50cm.m4a |
| 정지 | [spk02_185.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk02_185.wav) | 전진 | 정지빠르게20cm.m4a |
| 후진 | [spk02_057.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk02_057.wav) | 재시작 | 거리후진빠르게50cm.m4a |
| 후진 | [spk02_058.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk02_058.wav) | 복귀 | 거리후진작게20cm.m4a |
| 후진 | [spk02_059.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk02_059.wav) | 복귀 | 거리후진작게50cm.m4a |
| 후진 | [spk02_061.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk02_061.wav) | 재시작 | 거리후진정상50cm.m4a |
| 후진 | [spk02_086.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk02_086.wav) | 복귀 | 백양후진빠르게20cm.m4a |
| 후진 | [spk02_150.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk02_150.wav) | 재시작 | 복도후진빠르게50cm.m4a |
| 후진 | [spk02_152.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk02_152.wav) | 복귀 | 복도후진작게50cm.m4a |
| 후진 | [spk02_154.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk02_154.wav) | 재시작 | 복도후진정상50cm.m4a |
| 후진 | [spk02_198.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk02_198.wav) | 오른쪽 | 후진빠르게20cm.m4a |

## 원인 해석 범위

epoch3→10 악화 파일 30개 중 원본 부족분의 0 패딩이 있는 파일은 0개다. 추론 입력은 epoch마다 동일하다. 패딩·원본 길이·gain과 파일명 태그는 관찰 자료이며 오류 원인을 증명하지 않는다.

확률은 모델의 예측값이며 음성의 품질 점수가 아니다. 사람의 청취 검수는 하지 않았다. 같은 화자의 203개와 한 seed의 진단이므로 화자 전체의 성능으로 일반화하지 않는다.
