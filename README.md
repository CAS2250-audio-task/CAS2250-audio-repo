# Task 2 — Keyword Spotting Reference Baseline

교수 제공 노트북의 baseline을 모듈과 CLI로 옮긴 **완성형 참고 구현**입니다.
`reference/ai-baseline` 브랜치에서 관리합니다. 팀원용 skeleton과 실제 데이터는 별도 단계입니다.

현재 목표는 **출발 / 정지 / 전진 / 후진 / 왼쪽 / 오른쪽 / 시작 / 복귀 / 대기 / 재시작 + silence**입니다.
단어 목록과 class ID 순서는 `configs/task2.json`의 `labels`로 관리합니다.
실제 녹음은 Google Drive에서 수합합니다. 내려받은 폴더 경로를 실행 시 지정합니다.

팀원 시작 절차는 [VESSL · GitHub 팀 가이드](documents/vessl_github_team_guide.md)를 참고하세요.

## 환경과 설치

현재 서버: Python 3.10, NVIDIA RTX 3090, 드라이버 550.107.02.
`requirements.txt`는 CUDA 12.4용 PyTorch 2.6.0 / torchvision 0.21.0 / torchaudio 2.6.0과
프로젝트의 직접 의존성을 고정합니다. 아래 명령은 저장소 루트에서 실행합니다.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python -m ipykernel install --user --name audio-ai-jgb --display-name 'Python (audio-ai-jgb)'
```

노트북에서는 `Python (audio-ai-jgb)` 커널을 선택합니다.
CPU 검증은 `--device cpu`, GPU를 필수로 쓰려면 `--device cuda`를 지정합니다.
`auto`는 CUDA가 가능하면 GPU, 아니면 CPU를 사용하며 실행 결과에 device를 기록합니다.

## 제공 dummy 데이터로 실행

```bash
.venv/bin/python -m src validate --config configs/task2_dummy.json
.venv/bin/python -m src train --config configs/task2_dummy.json --output runs/dummy-v1 --device cuda
.venv/bin/python -m src evaluate --run runs/dummy-v1 --split val
.venv/bin/python -m src evaluate --run runs/dummy-v1 --split test
.venv/bin/python -m src predict --run runs/dummy-v1 --wav dummy_datasets/keyword_spotting/audio/keyword_01/speaker_01.wav
```

33개 합성 WAV의 학습·평가 점수는 **실제 성능을 의미하지 않습니다**.
Dummy 설정은 제공 데이터의 원래 단어를 사용합니다. `출발` 등 새로운 단어로 label을 바꾸거나
실제 팀 녹음에 섞지 않습니다. Dummy 실행은 설정·manifest·checkpoint와 평가 실행 기록에도 `dummy: true`로 기록됩니다.
같은 출력 디렉터리에 재학습하지 않으며, 재실행하려면 `runs/dummy-v2`처럼 새 이름을 사용합니다.

## Baseline 조건

- WAV를 mono, 16 kHz, 1.5초로 변환: 짧으면 zero padding, 길면 학습 random crop / 평가 center crop.
- `n_fft=512`, `win_length=512`, `hop_length=160`, `n_mels=128`, `f_min=40`, `f_max=8000`.
- Power mel → dB (`top_db=80`) → `(dB+80)/80` clamp → `(x-0.5)/0.25`.
- Resize 없이 `[1, 128, 151]` 입력, 1채널 ResNet-18, 11개 출력.
- 기본 ImageNet pretrained **off**, augmentation **on**.
- 학습 augmentation: gain, circular time shift, additive noise, frequency/time masking. 평가에는 적용하지 않음.
- 역빈도 WeightedRandomSampler, CrossEntropy, AdamW (`lr=3e-4`, `weight_decay=1e-4`), cosine schedule.
- 기본 1 epoch는 교수 dummy 실행 조건. 실제 실험의 epoch는 설정이나 `--epochs`로 지정.
- Validation macro-F1 최고 epoch를 `best.pt`에 저장. 동점이면 앞선 epoch 유지.

처리와 학습 설정은 교수 노트북을 따릅니다. 경로/label 관리, identity 검사, 재현성 기록,
평가 명령 분리 등은 reference의 관리 기능입니다. Fixed dB scaling을 다른 normalization으로
바꾸거나 모델·augmentation을 조정하는 일은 baseline 이후 실험으로 분리합니다.

## 팀 녹음 수집 형식 (Drive / GitHub 공통)

`templates/task2_collection_plan.csv`는 모든 참여자의 음성을 합친 통합 수집표입니다.
5명 × 10단어 × 단어당 20회 = 음성 1,000개이며, 단어별 100행을 제공합니다.
사람별 폴더나 CSV는 만들지 않고 `audio/<label>/clip_001.wav`부터 `clip_100.wav`까지
서로 겹치지 않는 파일명을 사용합니다. 파일별 실제 화자는 `speaker_id`에 기록합니다.
장소·거리·소음 조건은 실제 녹음값으로 채우며 자동 배정하지 않습니다.
Silence는 추가분(+α)이며 시작용으로 장소 3개 × 소음 유무 2개의 6행을 제공합니다.
수합한 silence 개수에 맞게 행을 추가하고 파일명을 중복 없이 지정합니다.

Drive에 `metadata.csv`와 `audio/<label>/*.wav`를 같은 데이터셋 폴더 아래 올립니다.
수집 계획 CSV를 복사해 `metadata.csv`로 사용하며, `path`와 실제 파일명을 일치시킵니다.
파일별 `speaker_id`, `room_session_id`, 장비와 환경을 기록합니다.
`room`은 장소 종류, `noise_level`은 quiet/noisy, `noise_type`은 traffic/conversation/fan 등
실제 소음 종류입니다. silence의 `distance_cm`과 `speaker_id`는 빈칸으로 둡니다.
환경 필드는 보조 정보이며 분류 대상은 10단어와 silence입니다.

`group`과 `split`은 수집 계획에서 비워 둡니다. 수집 후 실제 화자/세션 기준으로 배정하고
아래 validate 명령을 통과해야 학습할 수 있습니다. 한 화자의 녹음을 여러 split으로
나누는 방식은 현재 코드가 허용하지 않으므로 train/val/test에 각각 다른 화자가 필요합니다.
전체 데이터셋을 하나로 수합하되, 학습 시에는 CSV의 split으로 train/val/test를 구분합니다.

나중에 별도 GitHub 데이터 저장소로 옮겨도 이 상대 경로와 CSV를 그대로 유지합니다.
README에 데이터 버전과 수집 방법을 함께 기록합니다. WAV를 Git LFS로 관리할 경우
데이터 저장소에서 다음을 실행하고, 내려받을 때 `git lfs pull`로 실제 WAV를 받습니다.

```bash
git lfs install
git lfs track "*.wav"
git add .gitattributes metadata.csv audio README.md
```

이 프로젝트에서는 다운로드한 데이터 루트를 지정하면 되므로 Drive/GitHub 전용 로더는
필요하지 않습니다. `prepare-metadata`는 환경 라벨을 추론하지 않으므로 이미 작성한
수집 CSV가 있다면 그 CSV를 유지합니다. 녹음은 WAV로, 단어 전체가 1.5초 구간에
들어가도록 맞추는 것을 권장합니다.

## 실제 데이터가 준비되면

`configs/task2.json`의 `data_root`는 의도적으로 `null`입니다.
로컬 디렉터리나 AWS에서 내려받거나 마운트한 디렉터리를 `--data-root`로 전달합니다.
현재 CLI는 WAV 파일시스템을 읽으며 `s3://` URL을 직접 읽지는 않습니다.
원격 저장소 연결 방식이 정해져도 Dataset/모델 코드를 바꿀 필요가 없습니다.

```text
<DATA_ROOT>/
    metadata.csv
    audio/
        출발/*.wav
        정지/*.wav
        ...
        재시작/*.wav
        silence/*.wav
```

위 폴더 구조는 metadata 초안 생성용입니다. 학습에는 metadata의 상대 경로를 사용하므로
다른 폴더 구조도 가능합니다. WAV path는 데이터 루트 안의 상대 경로여야 합니다.

```bash
.venv/bin/python -m src prepare-metadata --data-root /path/to/data --output /path/to/metadata_draft.csv
```

초안은 label과 path만 채우고 실제 identity와 split은 비워 둡니다. 기존 annotation을 덮어쓰지 않습니다.
`templates/task2_metadata_template.csv`는 작성 예시이며 학습 가능한 데이터셋이 아닙니다.

### Metadata 계약

| 필드 | 의미 |
| --- | --- |
| `path` | WAV 상대 경로; 저장 위치가 바뀌어도 동일하게 유지 |
| `label` | 설정의 정확한 label; silence는 `silence` |
| `group` | keyword: `speaker:spk01`, silence: `room_session:session01` 등 실제 묶음 ID |
| `split` | `train`, `val`, `test` 중 하나; 전 행에 명시 |
| `speaker_id` | keyword 행에 필수; 익명 화자 ID |
| `room_session_id` | silence 행에 필수; 배경음 녹음 세션 ID. keyword 행에도 기록 권장 |
| `room`, `distance_cm`, `noise_level`, `noise_type`, `device`, `style` | 환경별 분석을 위한 권장 필드 |
| `source_url`, `license`, `annotator`, `notes` | 출처·동의·검수 기록 |

5명 중 train 3명, validation 1명, test 1명을 먼저 정하고 **모든 keyword 녹음에 동일하게 적용**합니다.
Silence는 room session 전체를 한 split에 넣습니다. 화자와 환경이 항상 일대일로 묶이지 않도록 수집합니다.
자동 random clip split은 제공하지 않습니다. 그룹·화자·silence 세션의 split 중복, 빈 필드,
중복 파일 경로, label 불일치, split별 class 누락은 실행 전에 오류로 처리합니다.
화자 필드를 함께 검사하므로 group을 세션별로 다르게 적어도 화자 누출을 놓치지 않습니다.

데이터 수합 후에는 설정을 복사해서 dataset/split 버전, epochs를 기록합니다.
키워드가 바뀌면 `labels`를 수정하고 metadata와 일치시킵니다. Class ID는 labels 순서이며
checkpoint에 보존됩니다. 기존 run 설정을 수정해서 checkpoint의 의미를 바꾸면 오류가 납니다.

```bash
.venv/bin/python -m src validate --config configs/task2.json --data-root /path/to/data
.venv/bin/python -m src train --config configs/task2.json --data-root /path/to/data --epochs 20 --output runs/baseline-v1
.venv/bin/python -m src evaluate --run runs/baseline-v1 --split val
# 데이터·모델·설정 동결 후 마지막에만 실행
.venv/bin/python -m src evaluate --run runs/baseline-v1 --split test --final-test
```

별도 metadata 파일은 `--metadata`로 전달합니다. 상대 경로면 data root 기준, 절대 경로면 그대로 사용합니다.
Config 안의 data root 상대 경로는 **config 파일 위치** 기준이고, CLI의 `--data-root`는 현재 실행 위치 기준입니다.
저장 위치가 달라진 동일한 데이터는 평가 시 `--data-root /new/location`으로 다시 지정합니다.
데이터/annotation이 수정되면 새 dataset/split 버전과 새 run으로 학습해야 합니다.

## 실행 결과와 평가 규칙

```text
runs/<run>/
    config.json           # 실제 적용한 경로, labels, hyperparameters, seed
    metadata.csv          # 고정된 split snapshot
    split_counts.csv
    manifest.json         # dataset/split version, SHA-256, Git 상태, command, device/환경
    history.csv
    best.pt               # 최고 validation macro-F1 checkpoint
    summary.json
    validation/           # 학습 종료 시 best checkpoint 평가
    evaluation/val/       # 필요할 때 다시 평가
    evaluation/test/      # 최종 test 1회
```

평가 폴더에는 metrics, class별 report, confusion matrix CSV/PNG, 모든 prediction,
`error_analysis.csv`, identity/환경별 `group_metrics.json`이 저장됩니다.
PNG 축은 서버 폰트에 의존하지 않는 class ID이며, 한글 label 순서는 config와 CSV에서 확인합니다.
전체 macro-F1은 11개 클래스를 모두 포함합니다. 환경별 macro-F1은 해당 slice의 실제 class를 기준으로 하며
`class_ids`와 표본 수를 함께 기록하므로 class 구성이 다른 slice의 점수를 직접 비교하지 않습니다.

학습 명령은 test 추론을 하지 않습니다. 무결성 확인을 위한 test 파일의 SHA-256만 함께 기록합니다.
평가는 저장된 split을 사용하고 파일 해시를 검사합니다. 실제 test에는 `--final-test`가 필요하며
같은 run의 test 결과 디렉터리는 한 번만 생성할 수 있습니다. 이는 실수 방지 장치입니다.
새 run을 만들어 test 점수를 보고 튜닝하는 행동까지 자동으로 막을 수는 없으므로 팀 전체에서
최종 test 정책을 지켜야 합니다. 결과를 본 뒤 설정을 다시 선택하면 test가 validation이 됩니다.
최종 보고서에서는 오분류 20개를 직접 듣고 원인을 기록합니다. 오분류가 20개 미만이면 전부 분석합니다.

`predict`는 고정 길이 단일 clip 분류입니다. 확률은 softmax 값이며 보정된 confidence가 아닙니다.
Continuous streaming, unknown class, AWS 업로드, 팀원 skeleton, 실험 branch는 다음 단계입니다.

## 검증

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Regression checks: group/identity leakage, split별 class 누락, label 불일치, 경로 탈출/중복,
평가 전처리의 결정성, stereo/resampling/padding, 11-class macro-F1.
Dummy end-to-end GPU 학습·checkpoint 재평가·test·단일 WAV 추론도 개발 시 확인합니다.
