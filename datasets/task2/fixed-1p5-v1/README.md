# 최종 1.5초 음성 데이터 — 1,015개

모든 WAV는 **mono 16 kHz / PCM16 / 정확히 24,000샘플 = 1.5초**입니다.
원본 파일당 결과 하나이며 라벨·화자·split을 유지합니다.
원본은 `../raw-v1/`과 `../source/`에 보존합니다.

| 처리 방식 | 파일 수 |
|---|---:|
| 기존 통과 처리본 | 703 |
| 기존 보류 처리본 | 295 |
| VAD 구간을 보존하며 앞뒤 여유 축소 | 9 |
| 사용자가 고른 1.5초 윈도우 | 8 |
| 합계 | 1,015 |

보류 사유는 오류 확정이 아닌 검토 기록이며 `preprocessing_recipe.csv`에 보존했습니다.
Silence 15개는 증폭하지 않았고, 발화 속도도 바꾸지 않았습니다.

## 긴 8개의 최종 선택

사용자의 `11111611`을 앞서 안내한 청취 순서대로 적용했습니다.

| 순서 | 단어 | 파일 | 선택 후보 |
|---|---|---|---:|
| 1 | 대기 | spk03_003 | 1 |
| 2 | 복귀 | spk03_028 | 1 |
| 3 | 시작 | spk01_155 | 1 |
| 4 | 재시작 | spk03_108 | 1 |
| 5 | 정지 | spk02_188 | 1 |
| 6 | 정지 | spk04_150 | 6 |
| 7 | 출발 | spk01_009 | 1 |
| 8 | 후진 | spk03_187 | 1 |

후보와 중간 음성은 정리했고, 선택 번호·실제 자른 위치는 기록으로 남겼습니다.

## 학습

Train 609 / validation 203 / test 203입니다.
학습 기본값은 LR 5e-4, 10 epochs, pretrained=True, augmentation=False입니다.

```bash
python -m src validate --config configs/task2_real.json
python -m src train --config configs/task2_real.json --output runs/task2-fixed-1p5-v1
```

노트북: `notebooks/task2_real_preprocessed.ipynb`.
클리핑은 별도 `preprocessing/clip_audio.py`에 있습니다.
원본에서 이 최종본을 정확히 재생성하려면 [클리핑 안내](../../../preprocessing/README.md)의 `--recipe` 명령을 사용합니다.

## 기록

- `metadata.csv`: 원본 라벨·화자·split을 유지한 1,015개 학습 목록.
- `preprocessing_recipe.csv`: 원본 정보, VAD 구간, 최종 자른 샘플 범위, 좌우 패딩, 증폭량,
  기존 검토 사유, 선택 번호, 원본 WAV와 최종 WAV의 SHA-256.
- `window_selections.csv`: 사용자의 8개 선택.
- `manifest.json`: 규격·개수·분할·전처리 설정·해시.

## 실행 결과

기존 baseline 설정으로 10 epochs 학습을 완료했습니다.
최적 epoch 10에서 validation accuracy **92.12%**, macro-F1 **0.925813**입니다.
Test 203개는 추론하지 않았습니다.
상세 지표와 학습 곡선은 [실행 기록](../../../reports/task2-fixed-1p5-v1/README.md)에 있습니다.
