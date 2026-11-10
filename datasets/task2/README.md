# 실제 명령어 녹음 v1 — jaehun 개인 클리핑 실험

이 브랜치의 `clipped-v1` 및 검토 목록은 `jaehun` 전용입니다.
공유 `main`/`reference/ai-baseline`은 원본 1,015개와 공통 학습 설정을 사용합니다.

업로드된 `audio_real_dataset.zip`은 화자 5명 × 203개 = **1,015개 M4A**입니다.
10개 명령어마다 100개씩 총 1,000개와 silence 15개를 포함합니다.

- `source/audio_real_dataset.zip`: 업로드 원본 그대로 보관한 단일 ZIP.
- `raw-v1/metadata.csv`, `raw-v1/import_manifest.json`: 원본 파일명·SHA-256·화자·split과 변환 기록.
- `raw-v1/originals/`, `raw-v1/audio/`: 로컬에서 재생성하는 M4A/WAV, Git에는 중복 저장하지 않음.
- `clipped-v1/audio/`, `clipped-v1/metadata.csv`: 자동 통과한 **703개**, mono 16 kHz, PCM16, 정확히 1.5초.
- `clipped-v1/review.csv`: 원본 확인이 필요한 **312개**, 자동 학습 목록에서 제외.
- `clipped-v1/preprocess_report.csv`: 전체 1,015개 처리 기록 및 상대 경로, 검출/자르기 시간.
- `clipped-v1/preprocess_options.json`, `clipped-v1/summary.json`: 설정과 집계.

원본 ZIP SHA-256: `07101815f0570cdff13dcf2be8b6cfa17d2988ed792f33f1c30e28c430760fa4`.
`documents/`와 더미 폴더에 업로드된 ZIP은 이 파일과 동일합니다. 그 사본은 로컬에 보존하고 Git에서는 제외합니다.

## 전처리

Silero VAD 6.2.3의 발화 시작/끝을 사용하고 앞뒤 200ms를 보존합니다.
단어 속도는 바꾸지 않고 양쪽 zero padding으로 1.5초를 맞춥니다.
발화 RMS -20 dBFS, 증폭 한도 +12 dB, 피크 상한 -1 dBFS를 적용합니다.
Silence는 중앙 1.5초를 추출하고 증폭하지 않습니다. 원본은 수정하지 않습니다.

검토 사유별 건수는 중복을 포함합니다.

| 사유 | 파일 수 |
|---|---:|
| 원본 시작/끝 50ms 안의 발화 | 163 |
| 목표 음량에 필요한 증폭이 +12 dB 초과 | 131 |
| 낮은 발화/배경 에너지 대비 | 47 |
| 여러 발화 구간 | 23 |
| 발화와 앞뒤 여유가 1.5초 초과 | 17 |
| 원본 진폭 clipping 의심 | 4 |
| silence에서 음성 검출 | 1 |

이는 오류 확정이나 실제 SNR 측정이 아닌 보수적인 검토 표시입니다.
파일 경계의 발화는 원래 잘렸을 가능성이 있어 자동으로 승인하지 않았습니다.
검출된 소리를 잘라서 강제로 1.5초에 넣거나, 검출 실패를 silence로 재라벨링하지 않습니다.

## 라벨과 split

파일명에서 명령어 라벨을 추론했습니다. CP949 ZIP 파일명과 분해된 한글 Unicode를 정규화하고
`재시작`과 `시작`을 구별합니다. 장소/거리/소음 정도는 추정으로 채우지 않았습니다.
`room_session_id`는 silence 원본 파일마다 부여한 식별자이며 실제 녹음 세션이 검증된 것은 아닙니다.

화자 목록을 고정하고 seed 42로 **3명 train / 1명 validation / 1명 test**로 분리했습니다.
화자 및 원본 파일이 split 사이에서 겹치지 않습니다. 원래 70% 보고서의 split은 제공되지 않아
동일한 성능을 재현했다고 주장할 수 없습니다.
자동 통과 파일은 train **457개**, validation **137개**, test **109개**입니다.
모든 split에 11개 클래스가 남아 있으며, 검토 파일을 제외하면서 클래스별 개수는 달라졌습니다.

## 실행

저장소 루트에서 실행합니다. 학습용 전처리 WAV는 이미 포함되어 있습니다.

```bash
python -m src validate --config configs/task2_real.json
python -m src train --config configs/task2_real.json --output runs/task2-v1
```

공통 baseline 설정은 업로드된 `notebooks/1차보고서_제출_70%.ipynb`에서 가져온 **LR 5e-4,
10 epochs, ImageNet pretrained=True, augmentation=False**입니다.
이 학습 기본값은 기본 config·더미 config·실제 config와 실행 노트북에 모두 동일하게 적용됩니다.
실행용 노트북은 `notebooks/task2_real_preprocessed.ipynb`입니다.
보고서 노트북에는 학습 출력이 없으므로 70%는 파일명이며 검증된 수치가 아닙니다.
이번 통합 작업에서는 학습을 실행하지 않았습니다.

검토 파일을 들으려면 무변형 WAV를 먼저 재생성하세요. 새 폴더만 허용하며 덮어쓰지 않습니다.

```bash
python -m src.import_recordings \
  --archive datasets/task2/source/audio_real_dataset.zip \
  --output-root datasets/task2/raw-v1-rebuilt
```

같은 경로의 `raw-v1`을 재생성하려면 기존 metadata/manifest를 별도 폴더로 옮겨
`raw-v1`이 없는 상태에서 `--output-root datasets/task2/raw-v1`을 사용합니다.
또는 노트북의 `REVIEW_SOURCE_ROOT`를 `raw-v1-rebuilt`의 절대 경로로 지정합니다.
현재 작업 환경에는 `raw-v1`의 원본 WAV가 이미 있습니다.
검토 후 원본/라벨을 수정하거나 train/validation에서 설정을 정한 뒤 새 버전으로 다시 처리합니다.

```bash
python -m src.preprocess \
  --data-root datasets/task2/raw-v1 \
  --output-root datasets/task2/clipped-v2
```

단순 입력 crop을 사용하는 기존 `predict` 명령에는 이 VAD 정책이 자동 적용되지 않습니다.
실제 배포 추론에도 동일한 VAD/여유/padding/음량 처리를 연결해야 합니다.

## 전체 처리본 연속 청취 (jaehun 전용)

기존 통과 703개는 그대로 두고 `review-candidates-v1/audio/`에 보류 312개의
클리핑·정규화 후보를 추가했습니다. 상태는 계속 review이며 학습 metadata에는 포함하지 않습니다.
`review-candidates-v1/candidates.csv`에 기존 사유, 후보 길이, 실제 증폭량과 자른 범위를 기록합니다.
증폭 한도는 +12 dB이고 silence는 증폭하지 않습니다. 긴 후보 17개는 1.5초로 강제 절단하지 않습니다.

현재 작업 환경의 `processed_datasets/manual-review/listening-v1/`에서
`accepted_all.wav`(703개, 21:05)와 `pending_all.wav`(312개, 09:28)를 각각 연속 재생할 수 있습니다.
원본 재생 없이 처리본만 포함하고, 음성 사이 0.3초 무음을 넣었습니다.
각 `*_index.csv`의 시작/종료 시각과 파일명으로 문제 파일을 찾고 decision/notes를 기록하세요.
연속 재생 파일은 로컬 검수 산출물이며 Git에서는 제외합니다.

새 후보/출력 폴더에 다음과 같이 재생성합니다. 이미 있는 폴더는 덮어쓰지 않습니다.

```bash
python -m src.review_audio \
  --data-root datasets/task2/raw-v1 \
  --processed-root datasets/task2/clipped-v1 \
  --candidate-root datasets/task2/review-candidates-v2 \
  --output-root processed_datasets/manual-review/listening-v2
```
