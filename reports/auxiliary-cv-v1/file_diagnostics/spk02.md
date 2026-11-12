# spk02 파일별 진단

spk02를 학습에서 제외한 고유 CV fold의 seed44 실행이다. 원래 결과와 학습 history·최적 가중치 텐서·최적 예측이 동일한 재현을 확인했다.

기준은 모든 화자에 동일한 **epoch3 정답→epoch10 오답**이다. 파일이 원래부터 불량했다는 판정이 아니며, 같은 파일 입력에 대한 모델 예측이 바뀐 경우다.

203개 중 정답 136→110, 악화 43개 / 회복 17개. Macro F1 70.14%→57.26%, loss 1.1441→1.8016.

[전체 203개·epoch별 예측·확률·전처리](./spk02_all_files.csv) · [악화 목록 CSV](./spk02_regressed_e03_e10.csv) · [회복 목록 CSV](./spk02_recovered_e03_e10.csv) · [정답 확률이 낮아져 loss가 증가한 목록](./spk02_loss_worsened_e03_e10.csv)

## epoch3 정답→epoch10 오답 전체 파일

| 정답 | 파일 | epoch10 예측 | 원본 파일명 |
| --- | --- | --- | --- |
| 복귀 | [spk02_008.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_008.wav) | 대기 | 거리복귀빠르게20cm.m4a |
| 복귀 | [spk02_009.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_009.wav) | 대기 | 거리복귀빠르게50cm.m4a |
| 복귀 | [spk02_010.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_010.wav) | 대기 | 거리복귀작게20cm.m4a |
| 복귀 | [spk02_013.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_013.wav) | 대기 | 거리복귀정상50cm.m4a |
| 복귀 | [spk02_070.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_070.wav) | 대기 | 백양복귀빠르게20cm.m4a |
| 복귀 | [spk02_088.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_088.wav) | 대기 | 복귀빠르게20cm.m4a |
| 복귀 | [spk02_090.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_090.wav) | 대기 | 복귀작게20cm.m4a |
| 복귀 | [spk02_102.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_102.wav) | 대기 | 복도복귀빠르게50cm.m4a |
| 복귀 | [spk02_104.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_104.wav) | 대기 | 복도복귀작게50cm.m4a |
| 복귀 | [spk02_106.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk02_106.wav) | 대기 | 복도복귀정상50cm.m4a |
| 시작 | [spk02_015.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk02_015.wav) | 출발 | 거리시작빠르게50cm.m4a |
| 시작 | [spk02_019.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk02_019.wav) | 출발 | 거리시작정상50cm.m4a |
| 시작 | [spk02_108.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk02_108.wav) | 출발 | 복도시작빠르게50cm.m4a |
| 시작 | [spk02_112.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk02_112.wav) | 후진 | 복도시작정상50cm.m4a |
| 시작 | [spk02_156.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/시작/spk02_156.wav) | 후진 | 시작빠르게50cm.m4a |
| 오른쪽 | [spk02_022.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk02_022.wav) | silence | 거리오른쪽작게20cm.m4a |
| 오른쪽 | [spk02_114.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk02_114.wav) | 복귀 | 복도오른쪽빠르게50cm.m4a |
| 오른쪽 | [spk02_115.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk02_115.wav) | 복귀 | 복도오른쪽작게20cm.m4a |
| 오른쪽 | [spk02_118.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk02_118.wav) | 복귀 | 복도오른쪽정상50cm.m4a |
| 오른쪽 | [spk02_164.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk02_164.wav) | 전진 | 오른쪽작게50cm.m4a |
| 왼쪽 | [spk02_119.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk02_119.wav) | 재시작 | 복도왼쪽빠르게20cm.m4a |
| 왼쪽 | [spk02_167.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/왼쪽/spk02_167.wav) | 복귀 | 왼쪽빠르게20cm.m4a |
| 재시작 | [spk02_037.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/재시작/spk02_037.wav) | 출발 | 거리재시작정상50cm.m4a |
| 전진 | [spk02_038.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk02_038.wav) | 정지 | 거리전진빠르게20cm.m4a |
| 전진 | [spk02_131.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk02_131.wav) | 정지 | 복도전진빠르게20cm.m4a |
| 전진 | [spk02_134.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk02_134.wav) | 복귀 | 복도전진작게50cm.m4a |
| 전진 | [spk02_135.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk02_135.wav) | 복귀 | 복도전진정상20cm.m4a |
| 전진 | [spk02_182.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk02_182.wav) | 정지 | 전진작게50cm.m4a |
| 전진 | [spk02_184.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk02_184.wav) | 복귀 | 전진정상50cm.m4a |
| 정지 | [spk02_044.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk02_044.wav) | 복귀 | 거리정지빠르게20cm.m4a |
| 정지 | [spk02_045.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk02_045.wav) | 복귀 | 거리정지빠르게50cm.m4a |
| 정지 | [spk02_138.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk02_138.wav) | 대기 | 복도정지빠르게50cm.m4a |
| 정지 | [spk02_139.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk02_139.wav) | 대기 | 복도정지작게20cm.m4a |
| 정지 | [spk02_140.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk02_140.wav) | 복귀 | 복도정지작게50cm.m4a |
| 정지 | [spk02_141.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk02_141.wav) | 복귀 | 복도정지정상20cm.m4a |
| 정지 | [spk02_186.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk02_186.wav) | 복귀 | 정지빠르게50cm.m4a |
| 후진 | [spk02_061.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk02_061.wav) | 복귀 | 거리후진정상50cm.m4a |
| 후진 | [spk02_086.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk02_086.wav) | 복귀 | 백양후진빠르게20cm.m4a |
| 후진 | [spk02_149.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk02_149.wav) | 정지 | 복도후진빠르게20cm.m4a |
| 후진 | [spk02_150.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk02_150.wav) | 대기 | 복도후진빠르게50cm.m4a |
| 후진 | [spk02_152.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk02_152.wav) | 복귀 | 복도후진작게50cm.m4a |
| 후진 | [spk02_153.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk02_153.wav) | 대기 | 복도후진정상20cm.m4a |
| 후진 | [spk02_203.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk02_203.wav) | 복귀 | 후진정상50cm.m4a |

## 원래 선택 epoch3→10의 악화

원래 선택 시점부터 비교하면 정답→오답은 43개다. spk01은 고정된 원래 모델의 선택 시점을 썼다. 화자마다 선택 epoch가 달라 이 개수로 화자를 직접 순위 매기지 않는다.

[이 기준의 전체 목록 CSV](./spk02_regressed_selected_e10.csv)

## 원인 해석 범위

epoch3→10 악화 파일 43개 중 원본 부족분의 0 패딩이 있는 파일은 0개다. 추론 입력은 epoch마다 동일하다. 패딩·원본 길이·gain과 파일명 태그는 관찰 자료이며 오류 원인을 증명하지 않는다.

확률은 모델의 예측값이며 음성의 품질 점수가 아니다. 사람의 청취 검수는 하지 않았다. 같은 화자의 203개와 한 seed의 진단이므로 화자 전체의 성능으로 일반화하지 않는다.

## 파일명 태그로 본 분포

아래 분모는 해당 태그의 epoch3 정답 파일 수이며, 분자는 그중 epoch10에 틀린 수다. 원본 파일명에서 읽은 태그이고 환경·거리의 실제 측정이나 원인 검증은 아니다. 단어·다른 조건과의 혼재가 남는다.

| 파일명 항목 | 태그 | epoch3 정답 | 그중 악화 | 비율 |
| --- | --- | ---: | ---: | ---: |
| filename_prefix | (prefix 없음) | 49 | 9 | 18.4% |
| filename_prefix | 거리 | 33 | 12 | 36.4% |
| filename_prefix | 백양 | 8 | 2 | 25.0% |
| filename_prefix | 복도 | 43 | 20 | 46.5% |
| filename_distance_tag | 20cm | 72 | 18 | 25.0% |
| filename_distance_tag | 50cm | 61 | 25 | 41.0% |

## 단어별 정답 수 변화

| 단어 | epoch3 정답 /20 | epoch10 정답 /20 |
| --- | ---: | ---: |
| 출발 | 13 | 19 |
| 정지 | 15 | 8 |
| 전진 | 9 | 3 |
| 후진 | 17 | 10 |
| 왼쪽 | 11 | 9 |
| 오른쪽 | 18 | 13 |
| 시작 | 10 | 5 |
| 복귀 | 18 | 8 |
| 대기 | 10 | 16 |
| 재시작 | 12 | 16 |
