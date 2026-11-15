# 채택 구조: 중간 특징 전달(local)

사용자 결정: CNN의 기존 넓은 시간 범위는 유지하고, 약 0.245초 범위의 중간 특징을 최종 특징에 더한다. `--variant local`을 사용한다. `short`/`both`는 비교 실험 재현용으로 보존한다. Mean/Pooled/Self 세 readout을 유지하며 CPU latency는 모델 선택 기준에서 제외한다. LR·batch 재탐색은 아직 수행하지 않았다.

```python
from experiments.local_features.models import variant_models
models = variant_models(seed=42, variant='local')
# keys: cnn_mean, cnn_pooled_query, cnn_self_cls
# input: [batch, 64 mel bands, time frames]
```

encoder index1(두 번째 블록) 출력 `[B,T,64]`를 Linear(64,32)로 투영해 기존 최종 `[B,T,32]` 특징에 더한다. 새 projection의 weight/bias는 0으로 초기화하므로 초기 출력은 baseline과 같다. 나머지 CNN/readout은 유지한다. 추론 파라미터는 head별79,915/80,939/88,395개이며 local 경로는2,080개를 추가한다.

## 완료된 실험 재현

저장소 루트에서 실행한다. 기존 compact-pooling-v1 입력 자산(formant cache, sources.json, noise, fixed-1p5-v1 개발 음성)을 먼저 준비해야 한다. 자산 manifest의 경로가 현재 checkout에서 유효해야 한다.

```bash
python -m experiments.local_features.run \
  --variant local \
  --objective paired_supcon --positive-mode same_command \
  --supcon-weight 0.1 --temperature 0.1 \
  --folds spk03 --epochs 1000 \
  --output runs/compact-local-local-spk03-e1000
```

원본·추가잡음 쌍에 CE와0.1 SupCon을 적용한다. same_command는 같은 명령어의 다른 학습 화자도 positive로 묶는 **소음+화자** 조건이다. 화자-only 실험은 아니다. 구조만 도입할 경우 기존 학습 objective를 보존할 수 있으며 이 wrapper는 `--objective ce`도 지원한다. 위 수치는 paired_supcon 조건에서 얻었으므로 CE 결과로 인용하지 않는다.

기존 baseline best.pt에는 local_projection이 없으므로 local 구조로 strict load할 수 없다. local 체크포인트는 반드시 variant='local'로 생성한 모델에 로드하고 체크포인트의 훈련-fold mean/std를 그대로 사용한다. projection head32→64→32는 대조학습 전용이며 추론에서 제거한다.

`experiments/launch_local_features.py`는 이번 세 조건을 실행했던 서버 전용 ablation supervisor다. baseline 경로가 `/root/CAS2250-compact-contrastive-100`으로 고정돼 있으므로 다른 서버에서 그대로 재실행하지 말고 위의 portable wrapper를 사용한다. `experiments/check_local_features.py`에도 이전 worktree/cache 경로 의존성이 있다.

[실험 결과](../../reports/compact-local-features/README.md) · [병합 전달 문서](../../reports/compact-local-features/MERGE_HANDOFF.md)
