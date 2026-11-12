# spk03 파일별 진단

spk03를 학습에서 제외한 고유 CV fold의 seed44 실행이다. 원래 결과와 학습 history·최적 가중치 텐서·최적 예측이 동일한 재현을 확인했다.

기준은 모든 화자에 동일한 **epoch3 정답→epoch10 오답**이다. 파일이 원래부터 불량했다는 판정이 아니며, 같은 파일 입력에 대한 모델 예측이 바뀐 경우다.

203개 중 정답 149→172, 악화 10개 / 회복 33개. Macro F1 72.02%→85.61%, loss 0.8648→0.5253.

[전체 203개·epoch별 예측·확률·전처리](./spk03_all_files.csv) · [악화 목록 CSV](./spk03_regressed_e03_e10.csv) · [회복 목록 CSV](./spk03_recovered_e03_e10.csv) · [정답 확률이 낮아져 loss가 증가한 목록](./spk03_loss_worsened_e03_e10.csv)

## epoch3 정답→epoch10 오답 전체 파일

| 정답 | 파일 | epoch10 예측 | 원본 파일명 |
| --- | --- | --- | --- |
| 복귀 | [spk03_038.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/복귀/spk03_038.wav) | 대기 | 노정민.복귀.복도1.m4a |
| 오른쪽 | [spk03_074.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk03_074.wav) | 왼쪽 | 노정민.오른쪽.방4.m4a |
| 오른쪽 | [spk03_076.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk03_076.wav) | 왼쪽 | 노정민.오른쪽.방6.m4a |
| 오른쪽 | [spk03_077.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk03_077.wav) | 왼쪽 | 노정민.오른쪽.방7.m4a |
| 오른쪽 | [spk03_081.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk03_081.wav) | 왼쪽 | 노정민.오른쪽.복도4.m4a |
| 오른쪽 | [spk03_083.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk03_083.wav) | 왼쪽 | 노정민.오른쪽.복도6.m4a |
| 전진 | [spk03_132.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/전진/spk03_132.wav) | 정지 | 노정민.전진.방2.m4a |
| 후진 | [spk03_192.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk03_192.wav) | 시작 | 노정민.후진.방2.m4a |
| 후진 | [spk03_193.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk03_193.wav) | 왼쪽 | 노정민.후진.방3.m4a |
| 후진 | [spk03_194.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/후진/spk03_194.wav) | 시작 | 노정민.후진.방4.m4a |

## 원래 선택 epoch7→10의 악화

원래 선택 시점부터 비교하면 정답→오답은 7개다. spk01은 고정된 원래 모델의 선택 시점을 썼다. 화자마다 선택 epoch가 달라 이 개수로 화자를 직접 순위 매기지 않는다.

[이 기준의 전체 목록 CSV](./spk03_regressed_selected_e10.csv)

| 정답 | 파일 | epoch10 예측 | 원본 파일명 |
| --- | --- | --- | --- |
| 오른쪽 | [spk03_066.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk03_066.wav) | 복귀 | 노정민.오른쪽.밖3.m4a |
| 오른쪽 | [spk03_074.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk03_074.wav) | 왼쪽 | 노정민.오른쪽.방4.m4a |
| 오른쪽 | [spk03_076.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk03_076.wav) | 왼쪽 | 노정민.오른쪽.방6.m4a |
| 오른쪽 | [spk03_083.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/오른쪽/spk03_083.wav) | 왼쪽 | 노정민.오른쪽.복도6.m4a |
| 정지 | [spk03_152.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk03_152.wav) | 대기 | 노정민.정지.방2.m4a |
| 정지 | [spk03_158.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk03_158.wav) | 전진 | 노정민.정지.복도1.m4a |
| 정지 | [spk03_161.wav](/root/audio/datasets/task2/real-context-1p5-v1/audio/정지/spk03_161.wav) | 대기 | 노정민.정지.복도4.m4a |

## 원인 해석 범위

epoch3→10 악화 파일 10개 중 원본 부족분의 0 패딩이 있는 파일은 0개다. 추론 입력은 epoch마다 동일하다. 패딩·원본 길이·gain과 파일명 태그는 관찰 자료이며 오류 원인을 증명하지 않는다.

확률은 모델의 예측값이며 음성의 품질 점수가 아니다. 사람의 청취 검수는 하지 않았다. 같은 화자의 203개와 한 seed의 진단이므로 화자 전체의 성능으로 일반화하지 않는다.
