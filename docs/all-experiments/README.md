# 모든 원격 브랜치·실험 인수인계

대상: https://github.com/CAS2250-audio-task/CAS2250-audio-repo. UTC 스냅샷: 2026-11-15T16:44:01.301435+00:00.

이 문서는 모든 원격 브랜치를 목록화한 탐색 지도다. 내용 전체를 다시 검증한 통합 결론은 아니다. 각 링크는 commit으로 고정했으며 브랜치가 달라지면 최신 내용과 차이가 날 수 있다.

## 팀원 또는 다른 AI에게 전달할 지시문

> 이 저장소의 모든 원격 브랜치를 확인하고 실험을 인수인계받아 주세요. 아래 브랜치별 문서 전체를 읽고, reports의 README/RESULTS/VALIDATION 및 연결된 CSV·JSON·그림을 확인하세요. 보고서가 없는 실험은 각 브랜치의 실험 코드·설정·커밋 기록을 확인하세요. 공통 문서는 blob hash로 중복 제거해 한 번만 읽고, 같은 경로라도 내용이 다른 버전은 모두 비교하세요. 데이터 분할·사전학습·증강·손실함수·평가지표·선택 checkpoint가 다른 결과를 같은 조건으로 간주하지 마세요. 완료/실패/중간/계획을 구분하고 아직 학습·수정·업로드하지 말고 먼저 전체 현황을 파악해 주세요.

```bash
git clone --no-single-branch https://github.com/CAS2250-audio-task/CAS2250-audio-repo.git
cd CAS2250-audio-repo
git fetch --all --tags --prune
git branch -r
```

원격 브랜치는 clone으로 모두 받아지며 각각 checkout 또는 git worktree로 볼 수 있다. 기본 checkout의 문서만 읽으면 다른 브랜치 실험을 놓친다.

## 우선 확인할 현재 결정

- `experiment/compact-local-features`의 `reports/compact-local-features/MERGE_HANDOFF.md`, `README.md`, `REVIEW.md`부터 확인한다. local 중간 특징 전달 구조와 Mean/Pooled/Self 세 집계를 채택했다.
- 소음+화자 지도대조학습 결과는 같은 브랜치의 `reports/compact-contrastive-launch/`에 있다.
- `compact-pooling-v1`의 `reports/compact-pooling-spk03-1000-v1/`에는 기존 CE 1000epoch와 Conformer 비교가 있다.
- 화자03 개발검증 결과와 독립 test 성능을 구분한다. 추가 소음을 섞기 전 원본 녹음도 완전한 clean이라고 보장하지 않는다.
- 최근 대화의 소형2D전처리/세로CNN/세로어텐션 비교 준비는 서버 로컬 작업이다. 정식 비교 실험은 사용자 요청으로 중단됐고 이 문서 브랜치에는 그 미업로드 코드를 포함하지 않는다.

## 모든 브랜치

| 브랜치 | 스냅샷 commit | 문서 수 | 전체 추적 파일 수 |
|---|---|---:|---:|
| [attention-pitch-v1](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/tree/5a98cef77e0a669076b5f6925e1570b1473b8f85) | `5a98cef77e` | 51 | 2411 |
| [compact-pooling-v1](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/tree/56cfb504afa5fe768cb7c35f5e09ea3fe67bd649) | `56cfb504af` | 56 | 2444 |
| [experiment/compact-local-features](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/tree/b301e3f51da37128db71ce611b8bb379773d38da) | `b301e3f51d` | 72 | 2597 |
| [main](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/tree/aaa535dad6b35ce3de85ebdb621ae765ebcc74a4) | `aaa535dad6` | 7 | 186 |
| [overnight-regularization-formant-aux](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/tree/b041d6f0bae94feec6cf84333c93d06d119aacf6) | `b041d6f0ba` | 45 | 2317 |
| [pretrain](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/tree/9fbdad4b1a5255491975dbc219863109cef371d4) | `9fbdad4b1a` | 38 | 2154 |
| [reference/ai-baseline](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/tree/6d208bc0eead31d179e63d050a26ff4511a8fee9) | `6d208bc0ee` | 10 | 1225 |
| [stduy/jeongmin](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/tree/80860cf693d36e4ee08b6062620a098685bdc2d2) | `80860cf693` | 4 | 166 |
| [study/jaehun](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/tree/8af36bcf1a4d5004c0ccb48bfecc00fe77cf89eb) | `8af36bcf1a` | 49 | 5085 |
| [study/minchae](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/tree/4153c30a595780328977e23450403cf5aba71fd5) | `4153c30a59` | 7 | 186 |

## 브랜치별 전체 문서 목록

동일 문서가 여러 브랜치에 반복된다. `inventory.json`의 blob 값이 같으면 내용도 같다. 보고서가 없는 코드 실험도 누락하지 않도록 inventory에는 모든 추적 파일을 기록했다.

### attention-pitch-v1

[전체 파일](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/tree/5a98cef77e0a669076b5f6925e1570b1473b8f85) · [커밋 기록](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/commits/attention-pitch-v1)

- [ATTENTION_PITCH_HANDOFF.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/5a98cef77e0a669076b5f6925e1570b1473b8f85/ATTENTION_PITCH_HANDOFF.md)
- [CHAT_HANDOFF_ATTENTION_PITCH.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/5a98cef77e0a669076b5f6925e1570b1473b8f85/CHAT_HANDOFF_ATTENTION_PITCH.md)
- [EXPERIMENT_BRANCHES.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/5a98cef77e0a669076b5f6925e1570b1473b8f85/EXPERIMENT_BRANCHES.md)
- [OVERNIGHT_HANDOFF.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/5a98cef77e0a669076b5f6925e1570b1473b8f85/OVERNIGHT_HANDOFF.md)
- [PRETRAIN_HANDOFF.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/5a98cef77e0a669076b5f6925e1570b1473b8f85/PRETRAIN_HANDOFF.md)
- [README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/5a98cef77e0a669076b5f6925e1570b1473b8f85/README.md)
- [datasets/task2/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/5a98cef77e0a669076b5f6925e1570b1473b8f85/datasets/task2/README.md)
- [datasets/task2/fixed-1p5-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/5a98cef77e0a669076b5f6925e1570b1473b8f85/datasets/task2/fixed-1p5-v1/README.md)
- [documents/notion.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/5a98cef77e0a669076b5f6925e1570b1473b8f85/documents/notion.md)
- [documents/workflow.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/5a98cef77e0a669076b5f6925e1570b1473b8f85/documents/workflow.md)
- [dummy_datasets/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/5a98cef77e0a669076b5f6925e1570b1473b8f85/dummy_datasets/README.md)
- [experiments/conformer/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/5a98cef77e0a669076b5f6925e1570b1473b8f85/experiments/conformer/README.md)
- [experiments/pretrain/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/5a98cef77e0a669076b5f6925e1570b1473b8f85/experiments/pretrain/README.md)
- [notebooks/audio_classification_resnet18_dummy_dataset.ipynb](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/5a98cef77e0a669076b5f6925e1570b1473b8f85/notebooks/audio_classification_resnet18_dummy_dataset.ipynb)
- [notebooks/task2_real_preprocessed.ipynb](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/5a98cef77e0a669076b5f6925e1570b1473b8f85/notebooks/task2_real_preprocessed.ipynb)
- [preprocessing/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/5a98cef77e0a669076b5f6925e1570b1473b8f85/preprocessing/README.md)
- [reports/attention-pitch-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/5a98cef77e0a669076b5f6925e1570b1473b8f85/reports/attention-pitch-v1/README.md)
- [reports/attention-pitch-v1/listening/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/5a98cef77e0a669076b5f6925e1570b1473b8f85/reports/attention-pitch-v1/listening/README.md)
- [reports/attention-unseen-noise-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/5a98cef77e0a669076b5f6925e1570b1473b8f85/reports/attention-unseen-noise-v1/README.md)
- [reports/conformer-aug-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/5a98cef77e0a669076b5f6925e1570b1473b8f85/reports/conformer-aug-v1/README.md)
- [reports/conformer-aug-v1/VALIDATION.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/5a98cef77e0a669076b5f6925e1570b1473b8f85/reports/conformer-aug-v1/VALIDATION.md)
- [reports/conformer-fixed100-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/5a98cef77e0a669076b5f6925e1570b1473b8f85/reports/conformer-fixed100-v1/README.md)
- [reports/conformer-fixed100-v1/RESULTS.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/5a98cef77e0a669076b5f6925e1570b1473b8f85/reports/conformer-fixed100-v1/RESULTS.md)
- [reports/conformer-latency-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/5a98cef77e0a669076b5f6925e1570b1473b8f85/reports/conformer-latency-v1/README.md)
- [reports/conformer-long-zero-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/5a98cef77e0a669076b5f6925e1570b1473b8f85/reports/conformer-long-zero-v1/README.md)
- [reports/conformer-long-zero-v1/RESULTS.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/5a98cef77e0a669076b5f6925e1570b1473b8f85/reports/conformer-long-zero-v1/RESULTS.md)
- [reports/conformer-long-zero-v1/VALIDATION.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/5a98cef77e0a669076b5f6925e1570b1473b8f85/reports/conformer-long-zero-v1/VALIDATION.md)
- [reports/conformer-noise-focus-v1/INTERIM.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/5a98cef77e0a669076b5f6925e1570b1473b8f85/reports/conformer-noise-focus-v1/INTERIM.md)
- [reports/conformer-noise-focus-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/5a98cef77e0a669076b5f6925e1570b1473b8f85/reports/conformer-noise-focus-v1/README.md)
- [reports/conformer-noise-focus-v1/VALIDATION.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/5a98cef77e0a669076b5f6925e1570b1473b8f85/reports/conformer-noise-focus-v1/VALIDATION.md)
- [reports/conformer-ref-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/5a98cef77e0a669076b5f6925e1570b1473b8f85/reports/conformer-ref-v1/README.md)
- [reports/conformer-ref-v1/seed42_files.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/5a98cef77e0a669076b5f6925e1570b1473b8f85/reports/conformer-ref-v1/seed42_files.md)
- [reports/conformer-ref-v1/seed43_files.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/5a98cef77e0a669076b5f6925e1570b1473b8f85/reports/conformer-ref-v1/seed43_files.md)
- [reports/conformer-ref-v1/seed44_files.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/5a98cef77e0a669076b5f6925e1570b1473b8f85/reports/conformer-ref-v1/seed44_files.md)
- [reports/conformer-sanity-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/5a98cef77e0a669076b5f6925e1570b1473b8f85/reports/conformer-sanity-v1/README.md)
- [reports/conformer-sanity-v1/VALIDATION.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/5a98cef77e0a669076b5f6925e1570b1473b8f85/reports/conformer-sanity-v1/VALIDATION.md)
- [reports/denoise-eval-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/5a98cef77e0a669076b5f6925e1570b1473b8f85/reports/denoise-eval-v1/README.md)
- [reports/denoise-eval-v1/VALIDATION.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/5a98cef77e0a669076b5f6925e1570b1473b8f85/reports/denoise-eval-v1/VALIDATION.md)
- [reports/noise-stress-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/5a98cef77e0a669076b5f6925e1570b1473b8f85/reports/noise-stress-v1/README.md)
- [reports/noise-stress-v1/VALIDATION.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/5a98cef77e0a669076b5f6925e1570b1473b8f85/reports/noise-stress-v1/VALIDATION.md)
- [reports/overnight-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/5a98cef77e0a669076b5f6925e1570b1473b8f85/reports/overnight-v1/README.md)
- [reports/overnight-v1/RESULTS.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/5a98cef77e0a669076b5f6925e1570b1473b8f85/reports/overnight-v1/RESULTS.md)
- [reports/overnight-v1/listening/difficult-noises/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/5a98cef77e0a669076b5f6925e1570b1473b8f85/reports/overnight-v1/listening/difficult-noises/README.md)
- [reports/overnight-v1/listening/formants/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/5a98cef77e0a669076b5f6925e1570b1473b8f85/reports/overnight-v1/listening/formants/README.md)
- [reports/overnight-v1/listening/pitch-formant-diversity/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/5a98cef77e0a669076b5f6925e1570b1473b8f85/reports/overnight-v1/listening/pitch-formant-diversity/README.md)
- [reports/overnight-v1/preparation_v0/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/5a98cef77e0a669076b5f6925e1570b1473b8f85/reports/overnight-v1/preparation_v0/README.md)
- [reports/speech-pretrain-ref-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/5a98cef77e0a669076b5f6925e1570b1473b8f85/reports/speech-pretrain-ref-v1/README.md)
- [reports/speech-pretrain-ref-v1/seed42_files.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/5a98cef77e0a669076b5f6925e1570b1473b8f85/reports/speech-pretrain-ref-v1/seed42_files.md)
- [reports/speech-pretrain-ref-v1/seed43_files.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/5a98cef77e0a669076b5f6925e1570b1473b8f85/reports/speech-pretrain-ref-v1/seed43_files.md)
- [reports/speech-pretrain-ref-v1/seed44_files.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/5a98cef77e0a669076b5f6925e1570b1473b8f85/reports/speech-pretrain-ref-v1/seed44_files.md)
- [reports/task2-fixed-1p5-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/5a98cef77e0a669076b5f6925e1570b1473b8f85/reports/task2-fixed-1p5-v1/README.md)

### compact-pooling-v1

[전체 파일](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/tree/56cfb504afa5fe768cb7c35f5e09ea3fe67bd649) · [커밋 기록](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/commits/compact-pooling-v1)

- [ATTENTION_PITCH_HANDOFF.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/56cfb504afa5fe768cb7c35f5e09ea3fe67bd649/ATTENTION_PITCH_HANDOFF.md)
- [CHAT_HANDOFF_ATTENTION_PITCH.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/56cfb504afa5fe768cb7c35f5e09ea3fe67bd649/CHAT_HANDOFF_ATTENTION_PITCH.md)
- [EXPERIMENT_BRANCHES.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/56cfb504afa5fe768cb7c35f5e09ea3fe67bd649/EXPERIMENT_BRANCHES.md)
- [OVERNIGHT_HANDOFF.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/56cfb504afa5fe768cb7c35f5e09ea3fe67bd649/OVERNIGHT_HANDOFF.md)
- [PRETRAIN_HANDOFF.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/56cfb504afa5fe768cb7c35f5e09ea3fe67bd649/PRETRAIN_HANDOFF.md)
- [README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/56cfb504afa5fe768cb7c35f5e09ea3fe67bd649/README.md)
- [datasets/task2/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/56cfb504afa5fe768cb7c35f5e09ea3fe67bd649/datasets/task2/README.md)
- [datasets/task2/fixed-1p5-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/56cfb504afa5fe768cb7c35f5e09ea3fe67bd649/datasets/task2/fixed-1p5-v1/README.md)
- [documents/notion.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/56cfb504afa5fe768cb7c35f5e09ea3fe67bd649/documents/notion.md)
- [documents/workflow.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/56cfb504afa5fe768cb7c35f5e09ea3fe67bd649/documents/workflow.md)
- [dummy_datasets/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/56cfb504afa5fe768cb7c35f5e09ea3fe67bd649/dummy_datasets/README.md)
- [experiments/compact_pooling/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/56cfb504afa5fe768cb7c35f5e09ea3fe67bd649/experiments/compact_pooling/README.md)
- [experiments/compact_pooling_long/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/56cfb504afa5fe768cb7c35f5e09ea3fe67bd649/experiments/compact_pooling_long/README.md)
- [experiments/conformer/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/56cfb504afa5fe768cb7c35f5e09ea3fe67bd649/experiments/conformer/README.md)
- [experiments/pretrain/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/56cfb504afa5fe768cb7c35f5e09ea3fe67bd649/experiments/pretrain/README.md)
- [notebooks/audio_classification_resnet18_dummy_dataset.ipynb](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/56cfb504afa5fe768cb7c35f5e09ea3fe67bd649/notebooks/audio_classification_resnet18_dummy_dataset.ipynb)
- [notebooks/task2_real_preprocessed.ipynb](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/56cfb504afa5fe768cb7c35f5e09ea3fe67bd649/notebooks/task2_real_preprocessed.ipynb)
- [preprocessing/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/56cfb504afa5fe768cb7c35f5e09ea3fe67bd649/preprocessing/README.md)
- [reports/attention-pitch-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/56cfb504afa5fe768cb7c35f5e09ea3fe67bd649/reports/attention-pitch-v1/README.md)
- [reports/attention-pitch-v1/listening/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/56cfb504afa5fe768cb7c35f5e09ea3fe67bd649/reports/attention-pitch-v1/listening/README.md)
- [reports/attention-unseen-noise-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/56cfb504afa5fe768cb7c35f5e09ea3fe67bd649/reports/attention-unseen-noise-v1/README.md)
- [reports/compact-pooling-spk03-1000-v1/CONFORMER_COMPARISON.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/56cfb504afa5fe768cb7c35f5e09ea3fe67bd649/reports/compact-pooling-spk03-1000-v1/CONFORMER_COMPARISON.md)
- [reports/compact-pooling-spk03-1000-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/56cfb504afa5fe768cb7c35f5e09ea3fe67bd649/reports/compact-pooling-spk03-1000-v1/README.md)
- [reports/compact-pooling-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/56cfb504afa5fe768cb7c35f5e09ea3fe67bd649/reports/compact-pooling-v1/README.md)
- [reports/conformer-aug-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/56cfb504afa5fe768cb7c35f5e09ea3fe67bd649/reports/conformer-aug-v1/README.md)
- [reports/conformer-aug-v1/VALIDATION.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/56cfb504afa5fe768cb7c35f5e09ea3fe67bd649/reports/conformer-aug-v1/VALIDATION.md)
- [reports/conformer-fixed100-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/56cfb504afa5fe768cb7c35f5e09ea3fe67bd649/reports/conformer-fixed100-v1/README.md)
- [reports/conformer-fixed100-v1/RESULTS.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/56cfb504afa5fe768cb7c35f5e09ea3fe67bd649/reports/conformer-fixed100-v1/RESULTS.md)
- [reports/conformer-latency-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/56cfb504afa5fe768cb7c35f5e09ea3fe67bd649/reports/conformer-latency-v1/README.md)
- [reports/conformer-long-zero-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/56cfb504afa5fe768cb7c35f5e09ea3fe67bd649/reports/conformer-long-zero-v1/README.md)
- [reports/conformer-long-zero-v1/RESULTS.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/56cfb504afa5fe768cb7c35f5e09ea3fe67bd649/reports/conformer-long-zero-v1/RESULTS.md)
- [reports/conformer-long-zero-v1/VALIDATION.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/56cfb504afa5fe768cb7c35f5e09ea3fe67bd649/reports/conformer-long-zero-v1/VALIDATION.md)
- [reports/conformer-noise-focus-v1/INTERIM.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/56cfb504afa5fe768cb7c35f5e09ea3fe67bd649/reports/conformer-noise-focus-v1/INTERIM.md)
- [reports/conformer-noise-focus-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/56cfb504afa5fe768cb7c35f5e09ea3fe67bd649/reports/conformer-noise-focus-v1/README.md)
- [reports/conformer-noise-focus-v1/VALIDATION.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/56cfb504afa5fe768cb7c35f5e09ea3fe67bd649/reports/conformer-noise-focus-v1/VALIDATION.md)
- [reports/conformer-ref-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/56cfb504afa5fe768cb7c35f5e09ea3fe67bd649/reports/conformer-ref-v1/README.md)
- [reports/conformer-ref-v1/seed42_files.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/56cfb504afa5fe768cb7c35f5e09ea3fe67bd649/reports/conformer-ref-v1/seed42_files.md)
- [reports/conformer-ref-v1/seed43_files.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/56cfb504afa5fe768cb7c35f5e09ea3fe67bd649/reports/conformer-ref-v1/seed43_files.md)
- [reports/conformer-ref-v1/seed44_files.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/56cfb504afa5fe768cb7c35f5e09ea3fe67bd649/reports/conformer-ref-v1/seed44_files.md)
- [reports/conformer-sanity-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/56cfb504afa5fe768cb7c35f5e09ea3fe67bd649/reports/conformer-sanity-v1/README.md)
- [reports/conformer-sanity-v1/VALIDATION.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/56cfb504afa5fe768cb7c35f5e09ea3fe67bd649/reports/conformer-sanity-v1/VALIDATION.md)
- [reports/denoise-eval-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/56cfb504afa5fe768cb7c35f5e09ea3fe67bd649/reports/denoise-eval-v1/README.md)
- [reports/denoise-eval-v1/VALIDATION.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/56cfb504afa5fe768cb7c35f5e09ea3fe67bd649/reports/denoise-eval-v1/VALIDATION.md)
- [reports/noise-stress-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/56cfb504afa5fe768cb7c35f5e09ea3fe67bd649/reports/noise-stress-v1/README.md)
- [reports/noise-stress-v1/VALIDATION.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/56cfb504afa5fe768cb7c35f5e09ea3fe67bd649/reports/noise-stress-v1/VALIDATION.md)
- [reports/overnight-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/56cfb504afa5fe768cb7c35f5e09ea3fe67bd649/reports/overnight-v1/README.md)
- [reports/overnight-v1/RESULTS.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/56cfb504afa5fe768cb7c35f5e09ea3fe67bd649/reports/overnight-v1/RESULTS.md)
- [reports/overnight-v1/listening/difficult-noises/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/56cfb504afa5fe768cb7c35f5e09ea3fe67bd649/reports/overnight-v1/listening/difficult-noises/README.md)
- [reports/overnight-v1/listening/formants/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/56cfb504afa5fe768cb7c35f5e09ea3fe67bd649/reports/overnight-v1/listening/formants/README.md)
- [reports/overnight-v1/listening/pitch-formant-diversity/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/56cfb504afa5fe768cb7c35f5e09ea3fe67bd649/reports/overnight-v1/listening/pitch-formant-diversity/README.md)
- [reports/overnight-v1/preparation_v0/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/56cfb504afa5fe768cb7c35f5e09ea3fe67bd649/reports/overnight-v1/preparation_v0/README.md)
- [reports/speech-pretrain-ref-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/56cfb504afa5fe768cb7c35f5e09ea3fe67bd649/reports/speech-pretrain-ref-v1/README.md)
- [reports/speech-pretrain-ref-v1/seed42_files.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/56cfb504afa5fe768cb7c35f5e09ea3fe67bd649/reports/speech-pretrain-ref-v1/seed42_files.md)
- [reports/speech-pretrain-ref-v1/seed43_files.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/56cfb504afa5fe768cb7c35f5e09ea3fe67bd649/reports/speech-pretrain-ref-v1/seed43_files.md)
- [reports/speech-pretrain-ref-v1/seed44_files.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/56cfb504afa5fe768cb7c35f5e09ea3fe67bd649/reports/speech-pretrain-ref-v1/seed44_files.md)
- [reports/task2-fixed-1p5-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/56cfb504afa5fe768cb7c35f5e09ea3fe67bd649/reports/task2-fixed-1p5-v1/README.md)

### experiment/compact-local-features

[전체 파일](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/tree/b301e3f51da37128db71ce611b8bb379773d38da) · [커밋 기록](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/commits/experiment%2Fcompact-local-features)

- [ATTENTION_PITCH_HANDOFF.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/ATTENTION_PITCH_HANDOFF.md)
- [CHAT_HANDOFF_ATTENTION_PITCH.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/CHAT_HANDOFF_ATTENTION_PITCH.md)
- [EXPERIMENT_BRANCHES.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/EXPERIMENT_BRANCHES.md)
- [OVERNIGHT_HANDOFF.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/OVERNIGHT_HANDOFF.md)
- [PRETRAIN_HANDOFF.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/PRETRAIN_HANDOFF.md)
- [README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/README.md)
- [datasets/task2/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/datasets/task2/README.md)
- [datasets/task2/fixed-1p5-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/datasets/task2/fixed-1p5-v1/README.md)
- [documents/notion.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/documents/notion.md)
- [documents/workflow.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/documents/workflow.md)
- [dummy_datasets/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/dummy_datasets/README.md)
- [experiments/compact_pooling/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/experiments/compact_pooling/README.md)
- [experiments/conformer/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/experiments/conformer/README.md)
- [experiments/local_features/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/experiments/local_features/README.md)
- [experiments/pretrain/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/experiments/pretrain/README.md)
- [notebooks/audio_classification_resnet18_dummy_dataset.ipynb](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/notebooks/audio_classification_resnet18_dummy_dataset.ipynb)
- [notebooks/task2_real_preprocessed.ipynb](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/notebooks/task2_real_preprocessed.ipynb)
- [preprocessing/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/preprocessing/README.md)
- [reports/attention-pitch-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/reports/attention-pitch-v1/README.md)
- [reports/attention-pitch-v1/listening/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/reports/attention-pitch-v1/listening/README.md)
- [reports/attention-unseen-noise-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/reports/attention-unseen-noise-v1/README.md)
- [reports/compact-contrastive-launch/CE_1000_COMPARISON.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/reports/compact-contrastive-launch/CE_1000_COMPARISON.md)
- [reports/compact-contrastive-launch/CE_BRANCH_COMPARISON.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/reports/compact-contrastive-launch/CE_BRANCH_COMPARISON.md)
- [reports/compact-contrastive-launch/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/reports/compact-contrastive-launch/README.md)
- [reports/compact-contrastive-launch/spk03-errors/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/reports/compact-contrastive-launch/spk03-errors/README.md)
- [reports/compact-contrastive-launch/spk03-errors/deep/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/reports/compact-contrastive-launch/spk03-errors/deep/README.md)
- [reports/compact-contrastive-launch/spk03-errors/mechanisms/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/reports/compact-contrastive-launch/spk03-errors/mechanisms/README.md)
- [reports/compact-local-both-spk03-e1000/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/reports/compact-local-both-spk03-e1000/README.md)
- [reports/compact-local-features/LATENCY.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/reports/compact-local-features/LATENCY.md)
- [reports/compact-local-features/MERGE_HANDOFF.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/reports/compact-local-features/MERGE_HANDOFF.md)
- [reports/compact-local-features/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/reports/compact-local-features/README.md)
- [reports/compact-local-features/REVIEW.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/reports/compact-local-features/REVIEW.md)
- [reports/compact-local-features/attention-vs-mean/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/reports/compact-local-features/attention-vs-mean/README.md)
- [reports/compact-local-local-spk03-e1000/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/reports/compact-local-local-spk03-e1000/README.md)
- [reports/compact-local-short-spk03-e1000/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/reports/compact-local-short-spk03-e1000/README.md)
- [reports/compact-noise-speaker-spk02-e100/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/reports/compact-noise-speaker-spk02-e100/README.md)
- [reports/compact-noise-speaker-spk03-e1000/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/reports/compact-noise-speaker-spk03-e1000/README.md)
- [reports/compact-noise-spk02-e100/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/reports/compact-noise-spk02-e100/README.md)
- [reports/compact-noise-spk03-e1000/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/reports/compact-noise-spk03-e1000/README.md)
- [reports/compact-pooling-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/reports/compact-pooling-v1/README.md)
- [reports/conformer-aug-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/reports/conformer-aug-v1/README.md)
- [reports/conformer-aug-v1/VALIDATION.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/reports/conformer-aug-v1/VALIDATION.md)
- [reports/conformer-fixed100-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/reports/conformer-fixed100-v1/README.md)
- [reports/conformer-fixed100-v1/RESULTS.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/reports/conformer-fixed100-v1/RESULTS.md)
- [reports/conformer-latency-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/reports/conformer-latency-v1/README.md)
- [reports/conformer-long-zero-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/reports/conformer-long-zero-v1/README.md)
- [reports/conformer-long-zero-v1/RESULTS.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/reports/conformer-long-zero-v1/RESULTS.md)
- [reports/conformer-long-zero-v1/VALIDATION.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/reports/conformer-long-zero-v1/VALIDATION.md)
- [reports/conformer-noise-focus-v1/INTERIM.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/reports/conformer-noise-focus-v1/INTERIM.md)
- [reports/conformer-noise-focus-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/reports/conformer-noise-focus-v1/README.md)
- [reports/conformer-noise-focus-v1/VALIDATION.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/reports/conformer-noise-focus-v1/VALIDATION.md)
- [reports/conformer-ref-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/reports/conformer-ref-v1/README.md)
- [reports/conformer-ref-v1/seed42_files.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/reports/conformer-ref-v1/seed42_files.md)
- [reports/conformer-ref-v1/seed43_files.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/reports/conformer-ref-v1/seed43_files.md)
- [reports/conformer-ref-v1/seed44_files.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/reports/conformer-ref-v1/seed44_files.md)
- [reports/conformer-sanity-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/reports/conformer-sanity-v1/README.md)
- [reports/conformer-sanity-v1/VALIDATION.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/reports/conformer-sanity-v1/VALIDATION.md)
- [reports/denoise-eval-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/reports/denoise-eval-v1/README.md)
- [reports/denoise-eval-v1/VALIDATION.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/reports/denoise-eval-v1/VALIDATION.md)
- [reports/noise-stress-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/reports/noise-stress-v1/README.md)
- [reports/noise-stress-v1/VALIDATION.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/reports/noise-stress-v1/VALIDATION.md)
- [reports/overnight-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/reports/overnight-v1/README.md)
- [reports/overnight-v1/RESULTS.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/reports/overnight-v1/RESULTS.md)
- [reports/overnight-v1/listening/difficult-noises/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/reports/overnight-v1/listening/difficult-noises/README.md)
- [reports/overnight-v1/listening/formants/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/reports/overnight-v1/listening/formants/README.md)
- [reports/overnight-v1/listening/pitch-formant-diversity/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/reports/overnight-v1/listening/pitch-formant-diversity/README.md)
- [reports/overnight-v1/preparation_v0/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/reports/overnight-v1/preparation_v0/README.md)
- [reports/speech-pretrain-ref-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/reports/speech-pretrain-ref-v1/README.md)
- [reports/speech-pretrain-ref-v1/seed42_files.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/reports/speech-pretrain-ref-v1/seed42_files.md)
- [reports/speech-pretrain-ref-v1/seed43_files.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/reports/speech-pretrain-ref-v1/seed43_files.md)
- [reports/speech-pretrain-ref-v1/seed44_files.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/reports/speech-pretrain-ref-v1/seed44_files.md)
- [reports/task2-fixed-1p5-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b301e3f51da37128db71ce611b8bb379773d38da/reports/task2-fixed-1p5-v1/README.md)

### main

[전체 파일](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/tree/aaa535dad6b35ce3de85ebdb621ae765ebcc74a4) · [커밋 기록](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/commits/main)

- [README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/aaa535dad6b35ce3de85ebdb621ae765ebcc74a4/README.md)
- [datasets/task2/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/aaa535dad6b35ce3de85ebdb621ae765ebcc74a4/datasets/task2/README.md)
- [documents/notion.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/aaa535dad6b35ce3de85ebdb621ae765ebcc74a4/documents/notion.md)
- [documents/workflow.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/aaa535dad6b35ce3de85ebdb621ae765ebcc74a4/documents/workflow.md)
- [dummy_datasets/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/aaa535dad6b35ce3de85ebdb621ae765ebcc74a4/dummy_datasets/README.md)
- [notebooks/audio_classification_resnet18_dummy_dataset.ipynb](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/aaa535dad6b35ce3de85ebdb621ae765ebcc74a4/notebooks/audio_classification_resnet18_dummy_dataset.ipynb)
- [notebooks/task2_real_baseline.ipynb](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/aaa535dad6b35ce3de85ebdb621ae765ebcc74a4/notebooks/task2_real_baseline.ipynb)

### overnight-regularization-formant-aux

[전체 파일](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/tree/b041d6f0bae94feec6cf84333c93d06d119aacf6) · [커밋 기록](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/commits/overnight-regularization-formant-aux)

- [OVERNIGHT_HANDOFF.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b041d6f0bae94feec6cf84333c93d06d119aacf6/OVERNIGHT_HANDOFF.md)
- [PRETRAIN_HANDOFF.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b041d6f0bae94feec6cf84333c93d06d119aacf6/PRETRAIN_HANDOFF.md)
- [README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b041d6f0bae94feec6cf84333c93d06d119aacf6/README.md)
- [datasets/task2/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b041d6f0bae94feec6cf84333c93d06d119aacf6/datasets/task2/README.md)
- [datasets/task2/fixed-1p5-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b041d6f0bae94feec6cf84333c93d06d119aacf6/datasets/task2/fixed-1p5-v1/README.md)
- [documents/notion.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b041d6f0bae94feec6cf84333c93d06d119aacf6/documents/notion.md)
- [documents/workflow.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b041d6f0bae94feec6cf84333c93d06d119aacf6/documents/workflow.md)
- [dummy_datasets/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b041d6f0bae94feec6cf84333c93d06d119aacf6/dummy_datasets/README.md)
- [experiments/conformer/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b041d6f0bae94feec6cf84333c93d06d119aacf6/experiments/conformer/README.md)
- [experiments/pretrain/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b041d6f0bae94feec6cf84333c93d06d119aacf6/experiments/pretrain/README.md)
- [notebooks/audio_classification_resnet18_dummy_dataset.ipynb](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b041d6f0bae94feec6cf84333c93d06d119aacf6/notebooks/audio_classification_resnet18_dummy_dataset.ipynb)
- [notebooks/task2_real_preprocessed.ipynb](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b041d6f0bae94feec6cf84333c93d06d119aacf6/notebooks/task2_real_preprocessed.ipynb)
- [preprocessing/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b041d6f0bae94feec6cf84333c93d06d119aacf6/preprocessing/README.md)
- [reports/conformer-aug-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b041d6f0bae94feec6cf84333c93d06d119aacf6/reports/conformer-aug-v1/README.md)
- [reports/conformer-aug-v1/VALIDATION.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b041d6f0bae94feec6cf84333c93d06d119aacf6/reports/conformer-aug-v1/VALIDATION.md)
- [reports/conformer-fixed100-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b041d6f0bae94feec6cf84333c93d06d119aacf6/reports/conformer-fixed100-v1/README.md)
- [reports/conformer-fixed100-v1/RESULTS.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b041d6f0bae94feec6cf84333c93d06d119aacf6/reports/conformer-fixed100-v1/RESULTS.md)
- [reports/conformer-latency-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b041d6f0bae94feec6cf84333c93d06d119aacf6/reports/conformer-latency-v1/README.md)
- [reports/conformer-long-zero-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b041d6f0bae94feec6cf84333c93d06d119aacf6/reports/conformer-long-zero-v1/README.md)
- [reports/conformer-long-zero-v1/RESULTS.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b041d6f0bae94feec6cf84333c93d06d119aacf6/reports/conformer-long-zero-v1/RESULTS.md)
- [reports/conformer-long-zero-v1/VALIDATION.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b041d6f0bae94feec6cf84333c93d06d119aacf6/reports/conformer-long-zero-v1/VALIDATION.md)
- [reports/conformer-noise-focus-v1/INTERIM.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b041d6f0bae94feec6cf84333c93d06d119aacf6/reports/conformer-noise-focus-v1/INTERIM.md)
- [reports/conformer-noise-focus-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b041d6f0bae94feec6cf84333c93d06d119aacf6/reports/conformer-noise-focus-v1/README.md)
- [reports/conformer-noise-focus-v1/VALIDATION.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b041d6f0bae94feec6cf84333c93d06d119aacf6/reports/conformer-noise-focus-v1/VALIDATION.md)
- [reports/conformer-ref-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b041d6f0bae94feec6cf84333c93d06d119aacf6/reports/conformer-ref-v1/README.md)
- [reports/conformer-ref-v1/seed42_files.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b041d6f0bae94feec6cf84333c93d06d119aacf6/reports/conformer-ref-v1/seed42_files.md)
- [reports/conformer-ref-v1/seed43_files.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b041d6f0bae94feec6cf84333c93d06d119aacf6/reports/conformer-ref-v1/seed43_files.md)
- [reports/conformer-ref-v1/seed44_files.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b041d6f0bae94feec6cf84333c93d06d119aacf6/reports/conformer-ref-v1/seed44_files.md)
- [reports/conformer-sanity-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b041d6f0bae94feec6cf84333c93d06d119aacf6/reports/conformer-sanity-v1/README.md)
- [reports/conformer-sanity-v1/VALIDATION.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b041d6f0bae94feec6cf84333c93d06d119aacf6/reports/conformer-sanity-v1/VALIDATION.md)
- [reports/denoise-eval-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b041d6f0bae94feec6cf84333c93d06d119aacf6/reports/denoise-eval-v1/README.md)
- [reports/denoise-eval-v1/VALIDATION.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b041d6f0bae94feec6cf84333c93d06d119aacf6/reports/denoise-eval-v1/VALIDATION.md)
- [reports/noise-stress-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b041d6f0bae94feec6cf84333c93d06d119aacf6/reports/noise-stress-v1/README.md)
- [reports/noise-stress-v1/VALIDATION.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b041d6f0bae94feec6cf84333c93d06d119aacf6/reports/noise-stress-v1/VALIDATION.md)
- [reports/overnight-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b041d6f0bae94feec6cf84333c93d06d119aacf6/reports/overnight-v1/README.md)
- [reports/overnight-v1/RESULTS.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b041d6f0bae94feec6cf84333c93d06d119aacf6/reports/overnight-v1/RESULTS.md)
- [reports/overnight-v1/listening/difficult-noises/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b041d6f0bae94feec6cf84333c93d06d119aacf6/reports/overnight-v1/listening/difficult-noises/README.md)
- [reports/overnight-v1/listening/formants/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b041d6f0bae94feec6cf84333c93d06d119aacf6/reports/overnight-v1/listening/formants/README.md)
- [reports/overnight-v1/listening/pitch-formant-diversity/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b041d6f0bae94feec6cf84333c93d06d119aacf6/reports/overnight-v1/listening/pitch-formant-diversity/README.md)
- [reports/overnight-v1/preparation_v0/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b041d6f0bae94feec6cf84333c93d06d119aacf6/reports/overnight-v1/preparation_v0/README.md)
- [reports/speech-pretrain-ref-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b041d6f0bae94feec6cf84333c93d06d119aacf6/reports/speech-pretrain-ref-v1/README.md)
- [reports/speech-pretrain-ref-v1/seed42_files.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b041d6f0bae94feec6cf84333c93d06d119aacf6/reports/speech-pretrain-ref-v1/seed42_files.md)
- [reports/speech-pretrain-ref-v1/seed43_files.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b041d6f0bae94feec6cf84333c93d06d119aacf6/reports/speech-pretrain-ref-v1/seed43_files.md)
- [reports/speech-pretrain-ref-v1/seed44_files.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b041d6f0bae94feec6cf84333c93d06d119aacf6/reports/speech-pretrain-ref-v1/seed44_files.md)
- [reports/task2-fixed-1p5-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/b041d6f0bae94feec6cf84333c93d06d119aacf6/reports/task2-fixed-1p5-v1/README.md)

### pretrain

[전체 파일](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/tree/9fbdad4b1a5255491975dbc219863109cef371d4) · [커밋 기록](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/commits/pretrain)

- [PRETRAIN_HANDOFF.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/9fbdad4b1a5255491975dbc219863109cef371d4/PRETRAIN_HANDOFF.md)
- [README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/9fbdad4b1a5255491975dbc219863109cef371d4/README.md)
- [datasets/task2/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/9fbdad4b1a5255491975dbc219863109cef371d4/datasets/task2/README.md)
- [datasets/task2/fixed-1p5-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/9fbdad4b1a5255491975dbc219863109cef371d4/datasets/task2/fixed-1p5-v1/README.md)
- [documents/notion.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/9fbdad4b1a5255491975dbc219863109cef371d4/documents/notion.md)
- [documents/workflow.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/9fbdad4b1a5255491975dbc219863109cef371d4/documents/workflow.md)
- [dummy_datasets/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/9fbdad4b1a5255491975dbc219863109cef371d4/dummy_datasets/README.md)
- [experiments/conformer/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/9fbdad4b1a5255491975dbc219863109cef371d4/experiments/conformer/README.md)
- [experiments/pretrain/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/9fbdad4b1a5255491975dbc219863109cef371d4/experiments/pretrain/README.md)
- [notebooks/audio_classification_resnet18_dummy_dataset.ipynb](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/9fbdad4b1a5255491975dbc219863109cef371d4/notebooks/audio_classification_resnet18_dummy_dataset.ipynb)
- [notebooks/task2_real_preprocessed.ipynb](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/9fbdad4b1a5255491975dbc219863109cef371d4/notebooks/task2_real_preprocessed.ipynb)
- [preprocessing/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/9fbdad4b1a5255491975dbc219863109cef371d4/preprocessing/README.md)
- [reports/conformer-aug-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/9fbdad4b1a5255491975dbc219863109cef371d4/reports/conformer-aug-v1/README.md)
- [reports/conformer-aug-v1/VALIDATION.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/9fbdad4b1a5255491975dbc219863109cef371d4/reports/conformer-aug-v1/VALIDATION.md)
- [reports/conformer-fixed100-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/9fbdad4b1a5255491975dbc219863109cef371d4/reports/conformer-fixed100-v1/README.md)
- [reports/conformer-fixed100-v1/RESULTS.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/9fbdad4b1a5255491975dbc219863109cef371d4/reports/conformer-fixed100-v1/RESULTS.md)
- [reports/conformer-latency-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/9fbdad4b1a5255491975dbc219863109cef371d4/reports/conformer-latency-v1/README.md)
- [reports/conformer-long-zero-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/9fbdad4b1a5255491975dbc219863109cef371d4/reports/conformer-long-zero-v1/README.md)
- [reports/conformer-long-zero-v1/RESULTS.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/9fbdad4b1a5255491975dbc219863109cef371d4/reports/conformer-long-zero-v1/RESULTS.md)
- [reports/conformer-long-zero-v1/VALIDATION.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/9fbdad4b1a5255491975dbc219863109cef371d4/reports/conformer-long-zero-v1/VALIDATION.md)
- [reports/conformer-noise-focus-v1/INTERIM.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/9fbdad4b1a5255491975dbc219863109cef371d4/reports/conformer-noise-focus-v1/INTERIM.md)
- [reports/conformer-noise-focus-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/9fbdad4b1a5255491975dbc219863109cef371d4/reports/conformer-noise-focus-v1/README.md)
- [reports/conformer-noise-focus-v1/VALIDATION.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/9fbdad4b1a5255491975dbc219863109cef371d4/reports/conformer-noise-focus-v1/VALIDATION.md)
- [reports/conformer-ref-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/9fbdad4b1a5255491975dbc219863109cef371d4/reports/conformer-ref-v1/README.md)
- [reports/conformer-ref-v1/seed42_files.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/9fbdad4b1a5255491975dbc219863109cef371d4/reports/conformer-ref-v1/seed42_files.md)
- [reports/conformer-ref-v1/seed43_files.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/9fbdad4b1a5255491975dbc219863109cef371d4/reports/conformer-ref-v1/seed43_files.md)
- [reports/conformer-ref-v1/seed44_files.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/9fbdad4b1a5255491975dbc219863109cef371d4/reports/conformer-ref-v1/seed44_files.md)
- [reports/conformer-sanity-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/9fbdad4b1a5255491975dbc219863109cef371d4/reports/conformer-sanity-v1/README.md)
- [reports/conformer-sanity-v1/VALIDATION.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/9fbdad4b1a5255491975dbc219863109cef371d4/reports/conformer-sanity-v1/VALIDATION.md)
- [reports/denoise-eval-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/9fbdad4b1a5255491975dbc219863109cef371d4/reports/denoise-eval-v1/README.md)
- [reports/denoise-eval-v1/VALIDATION.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/9fbdad4b1a5255491975dbc219863109cef371d4/reports/denoise-eval-v1/VALIDATION.md)
- [reports/noise-stress-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/9fbdad4b1a5255491975dbc219863109cef371d4/reports/noise-stress-v1/README.md)
- [reports/noise-stress-v1/VALIDATION.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/9fbdad4b1a5255491975dbc219863109cef371d4/reports/noise-stress-v1/VALIDATION.md)
- [reports/speech-pretrain-ref-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/9fbdad4b1a5255491975dbc219863109cef371d4/reports/speech-pretrain-ref-v1/README.md)
- [reports/speech-pretrain-ref-v1/seed42_files.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/9fbdad4b1a5255491975dbc219863109cef371d4/reports/speech-pretrain-ref-v1/seed42_files.md)
- [reports/speech-pretrain-ref-v1/seed43_files.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/9fbdad4b1a5255491975dbc219863109cef371d4/reports/speech-pretrain-ref-v1/seed43_files.md)
- [reports/speech-pretrain-ref-v1/seed44_files.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/9fbdad4b1a5255491975dbc219863109cef371d4/reports/speech-pretrain-ref-v1/seed44_files.md)
- [reports/task2-fixed-1p5-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/9fbdad4b1a5255491975dbc219863109cef371d4/reports/task2-fixed-1p5-v1/README.md)

### reference/ai-baseline

[전체 파일](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/tree/6d208bc0eead31d179e63d050a26ff4511a8fee9) · [커밋 기록](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/commits/reference%2Fai-baseline)

- [README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/6d208bc0eead31d179e63d050a26ff4511a8fee9/README.md)
- [datasets/task2/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/6d208bc0eead31d179e63d050a26ff4511a8fee9/datasets/task2/README.md)
- [datasets/task2/fixed-1p5-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/6d208bc0eead31d179e63d050a26ff4511a8fee9/datasets/task2/fixed-1p5-v1/README.md)
- [documents/notion.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/6d208bc0eead31d179e63d050a26ff4511a8fee9/documents/notion.md)
- [documents/workflow.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/6d208bc0eead31d179e63d050a26ff4511a8fee9/documents/workflow.md)
- [dummy_datasets/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/6d208bc0eead31d179e63d050a26ff4511a8fee9/dummy_datasets/README.md)
- [notebooks/audio_classification_resnet18_dummy_dataset.ipynb](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/6d208bc0eead31d179e63d050a26ff4511a8fee9/notebooks/audio_classification_resnet18_dummy_dataset.ipynb)
- [notebooks/task2_real_preprocessed.ipynb](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/6d208bc0eead31d179e63d050a26ff4511a8fee9/notebooks/task2_real_preprocessed.ipynb)
- [preprocessing/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/6d208bc0eead31d179e63d050a26ff4511a8fee9/preprocessing/README.md)
- [reports/task2-fixed-1p5-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/6d208bc0eead31d179e63d050a26ff4511a8fee9/reports/task2-fixed-1p5-v1/README.md)

### stduy/jeongmin

[전체 파일](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/tree/80860cf693d36e4ee08b6062620a098685bdc2d2) · [커밋 기록](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/commits/stduy%2Fjeongmin)

- [documents/notion.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/80860cf693d36e4ee08b6062620a098685bdc2d2/documents/notion.md)
- [documents/workflow.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/80860cf693d36e4ee08b6062620a098685bdc2d2/documents/workflow.md)
- [dummy_datasets/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/80860cf693d36e4ee08b6062620a098685bdc2d2/dummy_datasets/README.md)
- [notebooks/audio_classification_resnet18_dummy_dataset.ipynb](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/80860cf693d36e4ee08b6062620a098685bdc2d2/notebooks/audio_classification_resnet18_dummy_dataset.ipynb)

### study/jaehun

[전체 파일](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/tree/8af36bcf1a4d5004c0ccb48bfecc00fe77cf89eb) · [커밋 기록](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/commits/study%2Fjaehun)

- [CHAT_HANDOFF_ATTENTION_PITCH.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/8af36bcf1a4d5004c0ccb48bfecc00fe77cf89eb/CHAT_HANDOFF_ATTENTION_PITCH.md)
- [NEXT_EXPERIMENT_HANDOFF.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/8af36bcf1a4d5004c0ccb48bfecc00fe77cf89eb/NEXT_EXPERIMENT_HANDOFF.md)
- [README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/8af36bcf1a4d5004c0ccb48bfecc00fe77cf89eb/README.md)
- [datasets/task2/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/8af36bcf1a4d5004c0ccb48bfecc00fe77cf89eb/datasets/task2/README.md)
- [datasets/task2/fixed-1p5-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/8af36bcf1a4d5004c0ccb48bfecc00fe77cf89eb/datasets/task2/fixed-1p5-v1/README.md)
- [datasets/task2/real-context-1p5-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/8af36bcf1a4d5004c0ccb48bfecc00fe77cf89eb/datasets/task2/real-context-1p5-v1/README.md)
- [documents/notion.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/8af36bcf1a4d5004c0ccb48bfecc00fe77cf89eb/documents/notion.md)
- [documents/workflow.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/8af36bcf1a4d5004c0ccb48bfecc00fe77cf89eb/documents/workflow.md)
- [dummy_datasets/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/8af36bcf1a4d5004c0ccb48bfecc00fe77cf89eb/dummy_datasets/README.md)
- [experiments/auxiliary/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/8af36bcf1a4d5004c0ccb48bfecc00fe77cf89eb/experiments/auxiliary/README.md)
- [experiments/auxiliary_cv/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/8af36bcf1a4d5004c0ccb48bfecc00fe77cf89eb/experiments/auxiliary_cv/README.md)
- [notebooks/audio_classification_resnet18_dummy_dataset.ipynb](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/8af36bcf1a4d5004c0ccb48bfecc00fe77cf89eb/notebooks/audio_classification_resnet18_dummy_dataset.ipynb)
- [notebooks/task2_real_preprocessed.ipynb](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/8af36bcf1a4d5004c0ccb48bfecc00fe77cf89eb/notebooks/task2_real_preprocessed.ipynb)
- [preprocessing/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/8af36bcf1a4d5004c0ccb48bfecc00fe77cf89eb/preprocessing/README.md)
- [reports/auxiliary-cv-v1/INTERIM.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/8af36bcf1a4d5004c0ccb48bfecc00fe77cf89eb/reports/auxiliary-cv-v1/INTERIM.md)
- [reports/auxiliary-cv-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/8af36bcf1a4d5004c0ccb48bfecc00fe77cf89eb/reports/auxiliary-cv-v1/README.md)
- [reports/auxiliary-cv-v1/best_checkpoint_comparison/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/8af36bcf1a4d5004c0ccb48bfecc00fe77cf89eb/reports/auxiliary-cv-v1/best_checkpoint_comparison/README.md)
- [reports/auxiliary-cv-v1/best_checkpoint_comparison/spk01.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/8af36bcf1a4d5004c0ccb48bfecc00fe77cf89eb/reports/auxiliary-cv-v1/best_checkpoint_comparison/spk01.md)
- [reports/auxiliary-cv-v1/best_checkpoint_comparison/spk02.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/8af36bcf1a4d5004c0ccb48bfecc00fe77cf89eb/reports/auxiliary-cv-v1/best_checkpoint_comparison/spk02.md)
- [reports/auxiliary-cv-v1/best_checkpoint_comparison/spk03.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/8af36bcf1a4d5004c0ccb48bfecc00fe77cf89eb/reports/auxiliary-cv-v1/best_checkpoint_comparison/spk03.md)
- [reports/auxiliary-cv-v1/best_checkpoint_comparison/spk04.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/8af36bcf1a4d5004c0ccb48bfecc00fe77cf89eb/reports/auxiliary-cv-v1/best_checkpoint_comparison/spk04.md)
- [reports/auxiliary-cv-v1/common_epoch_comparison/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/8af36bcf1a4d5004c0ccb48bfecc00fe77cf89eb/reports/auxiliary-cv-v1/common_epoch_comparison/README.md)
- [reports/auxiliary-cv-v1/common_epoch_comparison/spk01.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/8af36bcf1a4d5004c0ccb48bfecc00fe77cf89eb/reports/auxiliary-cv-v1/common_epoch_comparison/spk01.md)
- [reports/auxiliary-cv-v1/common_epoch_comparison/spk02.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/8af36bcf1a4d5004c0ccb48bfecc00fe77cf89eb/reports/auxiliary-cv-v1/common_epoch_comparison/spk02.md)
- [reports/auxiliary-cv-v1/common_epoch_comparison/spk03.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/8af36bcf1a4d5004c0ccb48bfecc00fe77cf89eb/reports/auxiliary-cv-v1/common_epoch_comparison/spk03.md)
- [reports/auxiliary-cv-v1/common_epoch_comparison/spk04.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/8af36bcf1a4d5004c0ccb48bfecc00fe77cf89eb/reports/auxiliary-cv-v1/common_epoch_comparison/spk04.md)
- [reports/auxiliary-cv-v1/common_epoch_comparison/spk05.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/8af36bcf1a4d5004c0ccb48bfecc00fe77cf89eb/reports/auxiliary-cv-v1/common_epoch_comparison/spk05.md)
- [reports/auxiliary-cv-v1/file_diagnostics/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/8af36bcf1a4d5004c0ccb48bfecc00fe77cf89eb/reports/auxiliary-cv-v1/file_diagnostics/README.md)
- [reports/auxiliary-cv-v1/file_diagnostics/spk01.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/8af36bcf1a4d5004c0ccb48bfecc00fe77cf89eb/reports/auxiliary-cv-v1/file_diagnostics/spk01.md)
- [reports/auxiliary-cv-v1/file_diagnostics/spk02.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/8af36bcf1a4d5004c0ccb48bfecc00fe77cf89eb/reports/auxiliary-cv-v1/file_diagnostics/spk02.md)
- [reports/auxiliary-cv-v1/file_diagnostics/spk03.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/8af36bcf1a4d5004c0ccb48bfecc00fe77cf89eb/reports/auxiliary-cv-v1/file_diagnostics/spk03.md)
- [reports/auxiliary-cv-v1/file_diagnostics/spk04.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/8af36bcf1a4d5004c0ccb48bfecc00fe77cf89eb/reports/auxiliary-cv-v1/file_diagnostics/spk04.md)
- [reports/auxiliary-cv-v1/file_diagnostics_comparison/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/8af36bcf1a4d5004c0ccb48bfecc00fe77cf89eb/reports/auxiliary-cv-v1/file_diagnostics_comparison/README.md)
- [reports/auxiliary-cv-v1/file_diagnostics_comparison/spk01.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/8af36bcf1a4d5004c0ccb48bfecc00fe77cf89eb/reports/auxiliary-cv-v1/file_diagnostics_comparison/spk01.md)
- [reports/auxiliary-cv-v1/file_diagnostics_comparison/spk02.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/8af36bcf1a4d5004c0ccb48bfecc00fe77cf89eb/reports/auxiliary-cv-v1/file_diagnostics_comparison/spk02.md)
- [reports/auxiliary-cv-v1/file_diagnostics_comparison/spk03.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/8af36bcf1a4d5004c0ccb48bfecc00fe77cf89eb/reports/auxiliary-cv-v1/file_diagnostics_comparison/spk03.md)
- [reports/auxiliary-cv-v1/file_diagnostics_comparison/spk04.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/8af36bcf1a4d5004c0ccb48bfecc00fe77cf89eb/reports/auxiliary-cv-v1/file_diagnostics_comparison/spk04.md)
- [reports/auxiliary-cv-v1/file_diagnostics_set0/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/8af36bcf1a4d5004c0ccb48bfecc00fe77cf89eb/reports/auxiliary-cv-v1/file_diagnostics_set0/README.md)
- [reports/auxiliary-cv-v1/file_diagnostics_set0/spk01.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/8af36bcf1a4d5004c0ccb48bfecc00fe77cf89eb/reports/auxiliary-cv-v1/file_diagnostics_set0/spk01.md)
- [reports/auxiliary-cv-v1/file_diagnostics_set0/spk02.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/8af36bcf1a4d5004c0ccb48bfecc00fe77cf89eb/reports/auxiliary-cv-v1/file_diagnostics_set0/spk02.md)
- [reports/auxiliary-cv-v1/file_diagnostics_set0/spk03.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/8af36bcf1a4d5004c0ccb48bfecc00fe77cf89eb/reports/auxiliary-cv-v1/file_diagnostics_set0/spk03.md)
- [reports/auxiliary-cv-v1/file_diagnostics_set0/spk04.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/8af36bcf1a4d5004c0ccb48bfecc00fe77cf89eb/reports/auxiliary-cv-v1/file_diagnostics_set0/spk04.md)
- [reports/auxiliary-cv-v1/interpretation.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/8af36bcf1a4d5004c0ccb48bfecc00fe77cf89eb/reports/auxiliary-cv-v1/interpretation.md)
- [reports/auxiliary-cv-v1/overfitting.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/8af36bcf1a4d5004c0ccb48bfecc00fe77cf89eb/reports/auxiliary-cv-v1/overfitting.md)
- [reports/auxiliary-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/8af36bcf1a4d5004c0ccb48bfecc00fe77cf89eb/reports/auxiliary-v1/README.md)
- [reports/auxiliary-v1/interpretation.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/8af36bcf1a4d5004c0ccb48bfecc00fe77cf89eb/reports/auxiliary-v1/interpretation.md)
- [reports/auxiliary-v1/validation_case_review.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/8af36bcf1a4d5004c0ccb48bfecc00fe77cf89eb/reports/auxiliary-v1/validation_case_review.md)
- [reports/real-context-1p5-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/8af36bcf1a4d5004c0ccb48bfecc00fe77cf89eb/reports/real-context-1p5-v1/README.md)
- [reports/task2-fixed-1p5-v1/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/8af36bcf1a4d5004c0ccb48bfecc00fe77cf89eb/reports/task2-fixed-1p5-v1/README.md)

### study/minchae

[전체 파일](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/tree/4153c30a595780328977e23450403cf5aba71fd5) · [커밋 기록](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/commits/study%2Fminchae)

- [README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/4153c30a595780328977e23450403cf5aba71fd5/README.md)
- [datasets/task2/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/4153c30a595780328977e23450403cf5aba71fd5/datasets/task2/README.md)
- [documents/notion.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/4153c30a595780328977e23450403cf5aba71fd5/documents/notion.md)
- [documents/workflow.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/4153c30a595780328977e23450403cf5aba71fd5/documents/workflow.md)
- [dummy_datasets/README.md](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/4153c30a595780328977e23450403cf5aba71fd5/dummy_datasets/README.md)
- [notebooks/audio_classification_resnet18_dummy_dataset.ipynb](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/4153c30a595780328977e23450403cf5aba71fd5/notebooks/audio_classification_resnet18_dummy_dataset.ipynb)
- [notebooks/task2_real_baseline.ipynb](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/4153c30a595780328977e23450403cf5aba71fd5/notebooks/task2_real_baseline.ipynb)

## 데이터·캐시·가중치 릴리스

Git clone만으로 gitignore된 가중치·음원·캐시가 복원되지는 않는다. 아래 릴리스의 자산 목록과 manifest/SHA256을 확인하고 해당 실험 문서의 복원 절차를 따른다.

- [compact-local-features-selected-v1](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/releases/tag/compact-local-features-selected-v1): Selected intermediate-feature models (speaker03) (2026-11-15T16:00:45Z)
- [compact-pooling-spk03-1000-v1-results](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/releases/tag/compact-pooling-spk03-1000-v1-results): spk03: verified 1000-epoch results and checkpoints (2026-11-15T13:38:08Z)
- [compact-pooling-v1-assets](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/releases/tag/compact-pooling-v1-assets): Compact pooling experiment: exact assets and completed results (2026-11-15T12:15:05Z)

## 기계 판독용 목록

[inventory.json](inventory.json): 각 브랜치 commit, 전체 파일 경로·blob hash, 릴리스 목록. 공유되지 않은 다른 서버의 로컬 기록과 이 스냅샷 이후 변경은 포함하지 않는다.
