# 실제 음성 데이터 — jaehun 개인 실험

완료한 CV 실험의 데이터는 **`real-context-1p5-v1`의 1,015개 1.5초 WAV**입니다.
원본이 충분한 665개는 실제 연속 1.5초, 짧은 350개는 원본 전체에 부족한 길이만 0 패딩을 적용했습니다.
기존 baseline과 `auxiliary-v1`의 기준인 `fixed-1p5-v1`도 그대로 보존합니다.
10개 단어 × 100개 = 1,000개와 silence 15개입니다.
원본 음성은 보존하고 중간 처리본·후보·청취용 합본은 정리했습니다.

- `source/audio_real_dataset.zip`: 업로드 원본 ZIP.
- `raw-v1/originals/`: 원본 M4A 1,015개.
- `raw-v1/audio/`: 음량이나 길이를 바꾸지 않고 변환한 원본 WAV 1,015개.
- `raw-v1/metadata.csv`, `raw-v1/import_manifest.json`: 파일명·화자·원본 해시·변환과 split 기록.
- `fixed-1p5-v1/audio/`: 최종 mono 16 kHz / PCM16 / 24,000샘플 WAV 1,015개.
- `fixed-1p5-v1/metadata.csv`: 최종 학습 목록. 원본 라벨·화자·split 유지.
- `fixed-1p5-v1/preprocessing_recipe.csv`: 원본에서 최종본까지의 자르기·패딩·증폭·검토 기록.
- `fixed-1p5-v1/window_selections.csv`: 긴 8개의 선택 번호.
- `fixed-1p5-v1/manifest.json`: 데이터 규격·개수·버전·설정·해시.
- `real-context-1p5-v1/`: 다음 실험용 데이터와 전체 metadata·재현 recipe·manifest.
  사용 설정은 `configs/task2_real_context.json`이며, [조사와 검증 기록](../../reports/real-context-1p5-v1/README.md)을 참고하세요.

원본 ZIP의 SHA-256은 `07101815f0570cdff13dcf2be8b6cfa17d2988ed792f33f1c30e28c430760fa4`입니다.
원본 M4A/WAV는 로컬 생성 파일이라 Git에 중복 저장하지 않고, 원본 ZIP과 최종 WAV를 관리합니다.
새로 클론한 경우 원본은 보관된 ZIP에서 `python -m src.import_recordings`로 새 출력 폴더에 재생성할 수 있습니다.

## 분할과 학습

**train 609 / validation 203 / test 203**, 화자 기준 3명 / 1명 / 1명입니다.
Train은 spk02·spk03·spk04, validation은 spk05, test는 spk01입니다.
단어별 각 split의 개수는 60 / 20 / 20, silence는 9 / 3 / 3입니다.
`room_session_id`는 silence 원본 파일별 식별자이며 실제 세션이 검증된 것은 아닙니다.
라벨은 원본 파일명에서 추론한 값을 유지합니다.

새 데이터셋은 기존 라벨·화자·분할을 유지합니다. `configs/task2_real_context.json`으로 검증할 수 있으며,
이 설정 자체는 기존 단일 분할입니다. 완료한 4-fold CV의 fold별 분할·학습 실행기는
[실행 설명](../../experiments/auxiliary_cv/README.md), 결과는 [CV 보고서](../../reports/auxiliary-cv-v1/README.md)에 있습니다.

```bash
python -m src validate --config configs/task2_real.json
python -m src train --config configs/task2_real.json --output runs/task2-fixed-1p5-v1
```

Baseline은 LR 5e-4, 10 epochs, ImageNet pretrained=True, augmentation=False입니다.
실행 노트북은 `notebooks/task2_real_preprocessed.ipynb`입니다.
자세한 규격과 8개 선택은 [최종 데이터 안내](fixed-1p5-v1/README.md),
별도 클리핑 코드는 [preprocessing](../../preprocessing/README.md)을 참고하세요.

이 데이터와 전처리 실험은 `study/jaehun` 브랜치 전용입니다.
