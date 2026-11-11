# 전체 1,015개 · 1.5초 규격화 — jaehun 개인 실험

현재 **1,007개 준비 완료 / 긴 8개 윈도우 선택 대기**입니다.
선택을 전부 반영한 뒤 학습용 `metadata.csv`를 생성합니다.
기존 `clipped-v1`과 `review-candidates-v1` 및 원본은 별도 버전으로 보관합니다.

| 처리 | 파일 수 |
|---|---:|
| 기존 통과 처리본 그대로 사용 | 703 |
| 길이가 맞는 기존 보류 처리본 사용 | 295 |
| 검출 구간 전체를 보존하고 앞뒤 여유 축소 | 9 |
| 사용자 선택 윈도우 적용 대기 | 8 |
| 최종 목표 | 1,015 |

모든 준비 파일과 윈도우 후보는 **mono 16 kHz / PCM16 / 24,000샘플 = 1.5초**입니다.
기존 처리본의 증폭량을 유지합니다. silence 15개도 기존 음량 그대로 사용합니다.
과거 review 사유는 `preparation_report.csv`에 남겨 두며, 자동 통과로 재판정한 것은 아닙니다.
이번에는 사용자의 전체 1,015개 활용 지시에 따라 기존 보류 처리본도 포함합니다.

## 8개 윈도우 고르기

로컬 `processed_datasets/manual-review/windows-1p5-v1/listen.html`을 브라우저에서 엽니다.
페이지 안에 음성이 포함되어 있어 이 HTML 하나만 내려받아도 재생할 수 있습니다.
파일마다 **단어의 처음과 끝이 모두 들리는 후보 하나**를 고르고, 선택 결과를 복사해 채팅에 전달하세요.
후보가 모두 부적절하면 해당 파일은 번호 대신 "없음"으로 알려 주세요.

전체 연속 재생은 같은 폴더의 `windows_all.wav`이며 **1분 17.5초**입니다.
아래 시각부터 해당 파일의 후보 1, 2, 3… 순서로 나옵니다.
후보는 각 1.5초이고 후보 사이 간격은 0.5초입니다.

| 단어 | 파일 | 후보 번호 | 전체 재생 시작 |
|---|---|---|---|
| 대기 | spk03_003 | 1–3 | 00:00.000 |
| 복귀 | spk03_028 | 1–4 | 00:06.000 |
| 시작 | spk01_155 | 1–5 | 00:14.000 |
| 재시작 | spk03_108 | 1–4 | 00:24.000 |
| 정지 | spk02_188 | 1–6 | 00:32.000 |
| 정지 | spk04_150 | 1–6 | 00:44.000 |
| 출발 | spk01_009 | 1–5 | 00:56.000 |
| 후진 | spk03_187 | 1–6 | 01:06.000 |

개별 후보 WAV는 `../window-candidates-v1/audio/<단어>/<파일>_w01.wav` 형식입니다.
번호는 원본에서 창이 시작하는 시각 순입니다. 전체 범위의 양 끝, 각 VAD 구간을 중심에 둔 창,
200ms 간격의 창을 만들고 시작점 차이가 50ms 미만인 중복 후보는 줄였습니다.
실제 원본 자르기 시각과 후보 해시는 `../window-candidates-v1/windows.csv`에 있습니다.
정지 파일이 2개이므로 답변에 **파일명과 후보 번호**를 함께 적어 주세요. 예: `spk03_003=2`.

## 선택 확정 후 학습

선택 번호를 `../window-candidates-v1/selections.csv`의 `window_number`에 채웁니다.
8개 전부 유효한 번호일 때만 확정되며, 재확정하려면 새 데이터셋 버전을 만듭니다.

```bash
python -m src.fixed_length finalize \
  --output-root datasets/task2/fixed-1p5-v1 \
  --selections datasets/task2/window-candidates-v1/selections.csv
python -m src.fixed_length verify --output-root datasets/task2/fixed-1p5-v1
python -m src validate --config configs/task2_real.json
python -m src train --config configs/task2_real.json --output runs/task2-fixed-1p5-v1
```

사용자가 번호를 주면 이 확정과 학습을 이어서 실행합니다. 선택 대기 중에는 학습하지 않습니다.
원본 파일당 최종 WAV 하나씩이며 후보를 여러 학습 표본으로 중복 등록하지 않습니다.
원래 화자 분할을 유지해 **train 609 / validation 203 / test 203**입니다.
실제 가중치 학습에는 train 609개만 사용하고, validation으로 checkpoint를 선택합니다.
Test 추론은 최종 평가 명령에서 별도로 실행합니다.

학습 기본값은 공통 baseline의 **LR 5e-4 / 10 epochs / pretrained=True / augmentation=False**입니다.
설정은 `configs/task2_real.json`, 실행 노트북은 `notebooks/task2_real_preprocessed.ipynb`입니다.

## 파일과 재생성

- `audio/`: 현재 준비된 1,007개. 선택 완료 후 1,015개.
- `metadata_draft.csv`: 원본 1,015개의 라벨·화자·split을 그대로 보존한 초안.
- `preparation_report.csv`: 원래 검토 사유, 처리 방식, 줄인 여유, 자른 위치와 해시.
- `state.json`: 준비 개수, 선택 상태 및 길이 규격.
- `metadata.csv`, `window_selections.csv`: 선택 확정 후 생성하는 학습 목록과 선택 기록.

원본/기존 처리 폴더가 로컬에 있을 때 새 폴더로 재생성할 수 있습니다.

```bash
python -m src.fixed_length prepare \
  --raw-root datasets/task2/raw-v1 \
  --accepted-root datasets/task2/clipped-v1 \
  --pending-root datasets/task2/review-candidates-v1 \
  --output-root datasets/task2/fixed-1p5-v2 \
  --candidate-root datasets/task2/window-candidates-v2 \
  --listening-root processed_datasets/manual-review/windows-1p5-v2
```

기존 출력 폴더는 덮어쓰지 않습니다. 클리핑 실험과 이번 데이터는 개인 `study/jaehun` 브랜치에서만 관리합니다.
