# 클리핑 코드

모델 학습과 분리한 `clip_audio.py`에서 VAD 검출, 발화 주변 자르기, 음량 조절과 zero padding을 처리합니다.
학습 노트북은 이미 확정된 1.5초 WAV만 읽습니다.

현재 최종 데이터는 `datasets/task2/fixed-1p5-v1/`이며,
`preprocessing_recipe.csv`에 1,015개 원본과 결과의 대응, 샘플 단위 자르기 위치,
좌우 패딩, 증폭량, 검토 사유, 선택 번호 및 파일 해시를 보존합니다.
긴 8개는 사용자의 `11111611` 선택을 반영했고, silence는 증폭하지 않았습니다.

## 현재 확정본을 원본에서 재생성

```bash
python -m preprocessing.clip_audio \
  --data-root datasets/task2/raw-v1 \
  --recipe datasets/task2/fixed-1p5-v1/preprocessing_recipe.csv \
  --output-root processed_datasets/task2-rebuilt
```

다시 VAD로 판단하지 않고 검수된 위치와 증폭량을 그대로 적용합니다.
모든 결과의 SHA-256이 현재 확정본과 일치해야 완료됩니다.
원본, 라벨, 화자 및 split을 검사하고 기존 출력 폴더는 덮어쓰지 않습니다.

## 새로운 녹음에 VAD 적용

```bash
python -m preprocessing.clip_audio \
  --data-root /path/to/new-raw-dataset \
  --output-root processed_datasets/new-vad-review
```

기본값은 Silero VAD, 앞뒤 200ms, 1.5초, 발화 RMS 목표 -20 dBFS,
최대 증폭 +12 dB, 피크 상한 -1 dBFS입니다. silence는 중앙 구간을 사용하고 증폭하지 않습니다.
새로운 녹음에서 검출이 여러 개이거나 너무 길거나 애매하면 `review.csv`에 남기고 자동 학습 목록에서 제외합니다.
현재 데이터의 최종 결정은 위의 `--recipe`를 사용해야 동일하게 재현됩니다.

## 원본 주변 녹음을 보존한 새 1.5초 데이터

`real_context.py`는 기존 검수 구간 전체를 포함하면서, 원본 주변 녹음에서 연속 1.5초를
가져오는 별도 경로입니다. 짧은 원본은 전체를 보존하고 부족한 길이만 좌우 0값으로 채웁니다.
VAD를 다시 실행하거나 기존 데이터와 기존 학습 설정을 바꾸지 않습니다.

```bash
python -m preprocessing.real_context \
  --raw-root datasets/task2/raw-v1 \
  --reviewed-root datasets/task2/fixed-1p5-v1 \
  --output-root datasets/task2/real-context-1p5-v1 \
  --short-policy zero
```

새 데이터는 665개에 원본 연속 1.5초를 사용하고, 짧은 350개는 원본 전체에 0 패딩을 넣어
총 1,015개를 유지합니다. 새 설정은 `configs/task2_real_context.json`입니다.
배경음 반복·속도 변환은 적용하지 않습니다. 새 주변음 때문에 기존 gain에서 피크가 커지면
전체 gain을 낮춰 검수 대상으로 남기며, silence는 증폭하지 않습니다.

이전의 기본 모드 `--short-policy pending`은 짧은 파일을 보류하고 후보 665개만 만듭니다.
그 모드의 `candidate_metadata.csv`는 전체 학습 목록이 아닙니다.

근거 자료, 처리 선택지와 검증 결과는 [조사 보고서](../reports/real-context-1p5-v1/README.md)에 있습니다.
