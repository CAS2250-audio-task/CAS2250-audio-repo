# 다른 브랜치 관리 서버 전달 사항

## 확정 결정

- 소스 브랜치: `experiment/compact-local-features`.
- 구조는 **local(중간 특징 전달만)** 채택. CNN 시간 범위1.105초를 유지하고 두 번째 블록의 약0.245초 특징을64→32 Linear로 투영해 최종32D 특징에 residual addition.
- `short`(0.625초로 축소), `both`(축소+중간 전달)는 채택하지 않는다. 비교 재현용 코드는 유지한다.
- Mean/Pooled/Self 세 head를 유지한다. Self를 제외하지 않는다. CPU latency는 의사결정 기준에서 제외한다.
- LR/batch 튜닝은 후속 작업이며 현재 학습 설정은 유지했다.

## 완료 결과

화자03 검증203개, 화자02/04/05 학습609개, seed42,각1000epoch. 사전학습 없이 CE+0.1SupCon,temperature0.1,positive_mode=same_command(소음+화자). 원본은 추가잡음 합성 전 녹음이며 기존 잡음이 남아 있을 수 있다.

| head | local 원본 F1 | local 0dB F1 | best epoch | 기존 0dB F1 | 파라미터 |
|---|---:|---:|---:|---:|---:|
| mean | 91.97% | 85.44% | 806 | 85.44% | 79,915 |
| pooled | 95.42% | 86.77% | 565 | 84.68% | 80,939 |
| self | 96.30% | 85.55% | 508 | 83.92% | 88,395 |

0dB draw0 최고점으로 checkpoint를 선택하고8소음×3draw에서 평가했다. 단일 화자/seed의 개발검증이며 독립test가 아니다. test01은 이번 실행에서 읽지 않았다. 선택 checkpoint를 사용하고 epoch1000 last로 대체하지 않는다.

Pooled의 오른쪽→왼쪽46→17,전진→정지87→51은 개선됐으나 재시작→시작43→102로 악화. local Self의 재시작→시작은49건이고 원본 성능이 더 높다. 전체 F1만으로 Self를 탈락시키지 않는다. 자세한 오답은REVIEW.md와review_class_errors.csv에 기록했다.

## 병합할 코드와 충돌 처리

새 구조/진입점:

- `experiments/local_features/models.py`: FeatureRoutes,variant_models,variant_description.
- `experiments/local_features/run.py`: 기존 trainer에 모델 factory와 고정 메타데이터를 전달.
- `experiments/local_features/README.md`: 채택 구조와 실행 방법.

공용 코드 변경:

- `experiments/compact_pooling/run.py`: `main(argv=None, model_factory=None, experiment_config=None)` 및 `parse_args(argv)`, `(model_factory or matched_models)(42)` 지원. 추가 구조 메타데이터와 추가 소스파일 hash를protocol에 기록. 기본 모델 factory는 기존 동작을 유지.
- `experiments/compact_pooling/objectives.py`: `positive_mode=same_recording|same_command` 지원. same_recording에서는 다른 녹음의 같은 명령어를 negative로 쓰지 않고 제외한다. 기존same_command 기본값을 유지.
- 이 브랜치는 앞선 대조학습 실험과 오답 분석 커밋도 포함한다. 전체 병합 시 관련 보고서도 들어온다.
- 공통 `experiments/compact_pooling/models.py`의 baseline 정의를 local로 일괄 교체하지 않았다. 병합만 하고 기존 entrypoint를 실행하면 baseline이다. **local을 사용할 caller가 variant_models(...,variant='local')을 호출하거나 wrapper에 --variant local을 명시해야 한다.**
- 대상 브랜치에서 trainer/objective를 수정했다면 파일 전체를 ours/theirs로 덮지 말고 해당 API와 positive_mode 전달을 대상 학습 흐름에 통합한다. formant cache의 identity/source hash가 일치해야 한다.

다른 서버에서 대상 브랜치를 checkout하고 미커밋 작업을 정리한 뒤:

```bash
git fetch origin
git merge origin/experiment/compact-local-features
```

대상 브랜치가 아직 지정되지 않아 이 서버에서는 실제 병합하지 않았다. 구조 도입은 손실함수 도입과 분리할 수 있다. 대상 브랜치의 손실함수를 유지하려면 local factory만 통합하되, 보고된 점수는 paired_supcon 결과임을 유지한다.

재현 명령:

```bash
python -m experiments.local_features.run \
  --variant local \
  --objective paired_supcon --positive-mode same_command \
  --supcon-weight 0.1 --temperature 0.1 \
  --folds spk03 --epochs 1000 \
  --output runs/compact-local-local-spk03-e1000
```

## 입력·가중치·산출물

- 공통 입력은 기존 GitHub release `compact-pooling-v1-assets`의 `compact-pooling-v1-assets.tar.gz`를 재사용한다. manifest의 개발812음성과 formant/noise/cache hash를 확인한다.
- 학습률AdamW1e-3→1e-5 cosine1000,WD1e-4,batch32원본/64paired views,clip5. 소음SNR[-5,20],formant p0.5. train-only per-band normalization,64mel,16kHz,1.5초,noVAD.
- 이 서버의 선택 가중치: `runs/compact-local-local-spk03-e1000/spk03/{cnn_mean,cnn_pooled_query,cnn_self_cls}_best.pt`. 각각806/565/508epoch. model/mean/std와protocol hash가 포함된다.
- `runs/`는gitignore이므로 branch push에는 가중치·cache·raw audio가 포함되지 않는다. 다른 서버에서 재학습 없이 사용할 때는 아래 선택 가중치 패키지를 별도로 받아야 한다.
- 게시된 선택 가중치 패키지 자산명: `compact-local-features-selected-v1.tar.gz`,릴리스 태그 `compact-local-features-selected-v1`. 게시 완료. 원격 asset digest와 로컬SHA256 일치 확인. archive manifest와SHA256을 함께 제공한다.
- `reports/compact-local-features/`:12조건 비교표,전체epoch압축CSV,학습곡선,latency,독립 검토,선택 결정/본 문서.
- `reports/compact-local-{short,local,both}-spk03-e1000/`:각 protocol/summary.
- 검토:12개1000epoch기록,60,900개예측,12개최종점수재계산,현재소스해시39개,baseline과새조건3000개증강trace일치.
- `launch_local_features.py`,benchmark/check 스크립트 일부는 당시 worktree의 절대경로를 사용한다. 이들을 이식용 기본 launcher로 사용하지 않는다.

후속 작업은 다른 검증 화자에서 세 head를 비교하는 것과 LR/batch 탐색이다. 아직 추가 실행하지 않았다.

선택 가중치 다운로드: https://github.com/CAS2250-audio-task/CAS2250-audio-repo/releases/tag/compact-local-features-selected-v1

다른 서버의 해당 checkout 루트에서 아래 명령으로 다운로드·검증·복원한다. `gh`는 해당 서버에서 설치/인증되어 있어야 한다.

```bash
mkdir -p /tmp/compact-local-selected
gh release download compact-local-features-selected-v1 \
  --repo CAS2250-audio-task/CAS2250-audio-repo \
  --dir /tmp/compact-local-selected
(cd /tmp/compact-local-selected && sha256sum -c compact-local-features-selected-v1.tar.gz.sha256)
python -m experiments.compact_pooling.bundle restore /tmp/compact-local-selected/compact-local-features-selected-v1.tar.gz
```

릴리스의 보고서는 패키지 제작 당시 스냅샷이다. 최신 병합 전달 문서는 브랜치의 이 파일을 따른다.

## 풀링·주파수축·보조 음향의 설계 맥락

[병합 시 함께 전달할 설계/분석 문서](../../docs/COMPACT_MODEL_DESIGN_HANDOFF.md)를 함께 읽는다. 원본/소음 정답률과macro F1의 순위가 다르고, 전체F1 상승의silence 의존성을 확인했다. 주파수축 도입은 이웃 대역/대역 간 관계를 보완하려는 가설이며 특정 음절 구별 개선이 증명된 결과는 아니다. 보조음향set3는 그와 별도로 실행 중인 실험이다.
