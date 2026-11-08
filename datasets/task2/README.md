# 실제 녹음 원본 데이터와 공통 baseline

`source/audio_real_dataset.zip`은 업로드 원본 그대로이며, 화자 5명 × 203개 = **1,015개 M4A**입니다.
명령어별 100개씩 총 1,000개와 silence 15개를 포함합니다.
`raw-v1/metadata.csv`와 `raw-v1/import_manifest.json`에 파일명, SHA-256, 화자, split을 기록했습니다.

ZIP SHA-256: `07101815f0570cdff13dcf2be8b6cfa17d2988ed792f33f1c30e28c430760fa4`.
`documents/`와 더미 폴더의 같은 ZIP은 로컬 사본으로 유지하고 Git에는 중복 저장하지 않습니다.

## 실행

저장소 루트에서 원본 M4A를 mono 16 kHz PCM16 WAV로 변환합니다.
변환은 sample rate/채널/파일 형식만 맞추며 발화 검출, 발화 구간 자르기,
시간 늘이기/줄이기, 별도 음량 정규화를 하지 않습니다.

```bash
python -m src.import_recordings \
  --archive datasets/task2/source/audio_real_dataset.zip \
  --output-root datasets/task2/decoded-v1
python -m src validate --config configs/task2_real.json
python -m src train --config configs/task2_real.json --output runs/task2-baseline-v1
```

생성한 `decoded-v1/`은 Git에서 제외하며 원본 ZIP과 변환 코드로 재생성할 수 있습니다.
기존 출력 폴더는 덮어쓰지 않습니다.
실행용 노트북은 `notebooks/task2_real_baseline.ipynb`입니다.

공통 학습 설정은 `notebooks/1차보고서_제출_70%.ipynb`의 **LR 5e-4, 10 epochs,
ImageNet pretrained=True, augmentation=False**를 모든 config와 실행 노트북에 적용했습니다.
기존 baseline 로더는 1.5초 입력을 위해 짧으면 padding, 길면 학습 random crop /
평가 center crop을 사용합니다.
보고서의 실행 출력과 split은 제공되지 않아 70% 성능을 재현했다고 주장할 수 없습니다.

## Split

| 구분 | 화자 | 녹음 수 |
|---|---|---:|
| Train | spk02, spk03, spk04 | 609 |
| Validation | spk05 | 203 |
| Test | spk01 | 203 |

seed 42로 화자 단위로 고정했으며 split 사이 화자와 동일 원본 파일이 겹치지 않습니다.
각 split에 11개 클래스가 모두 존재합니다. 원래 70% 보고서의 split과 동일하다고 가정하지 않습니다.
라벨은 파일명에서 추론했으며 CP949와 분해된 한글 Unicode를 정규화했습니다.
실제 장소/거리/소음 정도는 추정으로 채우지 않았고, silence의 세션 ID는 파일별 식별자입니다.
실제 녹음 세션이나 장소까지 완전히 분리됐는지는 추가 기록이 필요합니다.
