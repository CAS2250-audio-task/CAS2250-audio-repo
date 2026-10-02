ㅇㅇ. **큰 방향은 상당히 좋음.** 특히 “AI가 먼저 완성본을 만들어주고 → 팀원들은 skeleton에서 직접 구현하며 학습 → 다시 합치고 → 2차에서 실험적으로 갈라진다”는 구조가 **개발 속도와 학습 둘 다 챙기기 좋음.** 다만 지금 안 그대로 하면 Git/data/test 관리가 조금 꼬일 수 있어서 몇 군데만 다듬는 게 좋음.

내가 추천하는 최종 형태는 이거임.

1. **Phase 0 — 기준선 만들기**
   실제 데이터 없이 제공 dummy data로 **Codex가 교수 제공 baseline을 최대한 충실하게 구현/정리**함. 여기서 새로운 모델을 만드는 게 아니라, PDF의 `WAV → log-mel → 1-channel ResNet-18 → validation macro-F1 기준 checkpoint` 파이프라인이 끝까지 재현되는 코드가 목표임. 교수 자료도 처음에는 설정을 바꾸지 않고 baseline을 끝까지 실행하라고 함. [Team Project] Audio task (4)

   그런데 이 AI 완성본을 바로 `main`에 넣어서 모든 팀원이 보는 것보다는:

   ```text
   main
   └── skeleton / 공통 구조

   reference/ai-baseline
   └── Codex가 완성한 reference implementation
   ```

   식이 더 좋음.

   왜냐면 “AI 코드 보고 싶지 않은 사람은 자기 branch에서 삭제”보다 **처음부터 reference branch를 따로 두는 게 훨씬 깔끔함.** Git은 삭제해도 과거 commit에 코드가 남아 있으니까 진짜로 안 보려는 사람한테는 분리 branch가 맞음.

2. **Phase 1 — 실제 데이터 수집 + 학습형 구현**
   실제 WAV 수집과 annotation을 동시에 시작.

   `main`에는:

   ```text
   src/
       dataset.py        # skeleton
       model.py
       train.py
       evaluate.py
   configs/
   notebooks/
   README.md
   ```

   정도의 **interface와 TODO만 있는 skeleton**을 둠.

   각자:

   ```text
   learn/A
   learn/B
   learn/C
   ...
   ```

   branch를 따서 자기 Workspace에서 구현.

   원하면 `reference/ai-baseline`을 참고하고, 원하지 않으면 아예 안 보면 됨.

   이 단계 목적은 **누가 더 좋은 모델을 만드는가가 아니라, 전원이 전체 pipeline을 이해하는 것**임.

3. **Phase 1 integration — 1차 보고서용 baseline 확정**
   여기서 중요한 수정 하나.

   각자 만든 코드와 AI 코드를 **전부 덕지덕지 merge하는 건 비추천**임.

   대신 비교해서:

   ```text
   dataset 구현 → 가장 명확한 버전
   preprocessing → 검증된 버전
   model → 교수 baseline과 가장 충실한 버전
   train loop → 검증된 버전
   evaluation → 검증된 버전
   ```

   을 골라 하나의 **integration branch**에서 합침.

   ```text
   integration/baseline-v1
   ```

   여기서 실제 데이터로 한 번 정상적으로:

   ```text
   metadata
   ↓
   train/val split
   ↓
   log-mel
   ↓
   ResNet-18
   ↓
   train
   ↓
   validation
   ```

   이 돌아가면:

   ```text
   tag: baseline-v1
   dataset: dataset-v1
   ```

   로 **동결**.

   이게 1차 보고서의 기준점이 됨.

   교수 자료도 dummy data 단계에서는 accuracy 자체가 의미 있는 게 아니라 end-to-end 실행 확인이 목적이고, 실제 pipeline에서는 metadata/group split → spectrogram → baseline 학습으로 이어짐. [Team Project] Audio task (4)

4. **1차 보고서**
   여기서 보고서에는 “우리끼리 만든 여러 코드”보다 **최종 baseline-v1 하나를 정확하게 설명**하는 게 좋음.

   특히 기록할 것:

   ```text
   Git commit hash
   dataset version
   split version
   config
   random seed
   실행 command
   validation 결과
   ```

   그래야 나중에 2차 결과와 정확히 비교 가능함.

5. **Phase 2 — 본격적인 탐구 / benchmark**
   여기서 네가 말한 구조가 딱 좋음.

   예를 들면 팀원별로:

   ```text
   exp/augmentation
   exp/pretrained
   exp/mel-params
   exp/model
   exp/optimizer
   ```

   같이 branch를 나눔.

   여러 VESSL Workspace에서 개발하고, **정식 benchmark는 가능하면 VESSL Run으로 같은 조건에서 실행**.

   여기서는:

   ```text
   baseline-v1
        ↓
   실험 하나 변경
        ↓
   validation benchmark
        ↓
   결과 기록
   ```

   을 반복.

   **한 번에 여러 요소를 바꾸지 않는 것**이 중요함. 그래야 뭐가 효과 있었는지 알 수 있음.

6. **데이터셋도 2차에서 수정 가능하긴 함**
   이것도 네 생각대로 가능함. 다만 그냥 같은 폴더를 계속 수정하면 안 됨.

   ```text
   dataset-v1  ← 1차 보고서 기준
   dataset-v2  ← annotation 수정 / 데이터 추가
   dataset-v3  ← 최종
   ```

   식으로 버전 남겨야 함.

   특히 source/group split이 이 과제 핵심이라 dataset 변경으로 group이 달라지면 **split도 versioning**해야 함. PDF도 같은 speaker/source video/dog ID가 여러 split에 들어가는 group leakage를 경고함. [Team Project] Audio task (4)

7. **최종 integration**
   2차 실험에서 잘 된 것들을:

   ```text
   exp/* branches
        ↓ PR / review
   integration/final
        ↓
   validation
   ```

   으로 하나씩 합침.

   여기서도 합칠 때마다 validation을 다시 돌려야 함. A 개선 + B 개선이 각각 좋았다고 해서 **A+B가 반드시 좋은 건 아니기 때문**임.

8. **그리고 마지막에 Test 개봉**
   이게 진짜 중요.

   ```text
   dataset-vFinal
   code-vFinal
   hyperparameters 확정
   augmentation 확정
   모델 확정
               ↓
            TEST
   ```

   test 결과가 마음에 안 들어도 다시 모델 고치는 용도로 사용하면 안 됨. 교수 자료도 test 결과를 보고 설정을 바꾸면 test가 validation이 된다고 직접 적어둠. [Team Project] Audio task (4)

   마지막으로:

   ```text
   accuracy
   macro-F1
   confusion matrix
   misclassified audio 20개 분석
   ```

   하고 최종 보고서 + 발표. [Team Project] Audio task (4)

그래서 네 최초안을 조금 압축해서 고치면:

```text
Codex reference baseline
        ↓
Skeleton 공개
        ↓
각자 구현하며 학습
        ↓
Integration
        ↓
실제 데이터 baseline
        ↓
baseline-v1 + dataset-v1 freeze
        ↓
1차 보고서
────────────────────────
개인/주제별 experiment branches
        ↓
validation benchmark
        ↓
좋은 것 integration
        ↓
필요하면 dataset v2/v3
        ↓
재실험
        ↓
final code + final data freeze
        ↓
TEST 1회
        ↓
error analysis
        ↓
2차 보고서 + 발표
```

이 구조면 **교육용 팀플로도 상당히 좋고, 실제 연구 workflow 맛도 꽤 남.**

특히 나는 `AI 완성코드를 main에 넣었다가 각자 지우기`만 **`reference/ai-baseline` 별도 branch로 분리**하는 걸 강하게 추천함. 그리고 **1차 제출 시점에 `baseline-v1` 태그를 박는 것**도 해두면, 2차에서 코드가 아무리 난장판이 되어도 언제든 기준점으로 돌아갈 수 있음.

##부가정보
팀원은 총 5명. 정민 재훈 서준 환준 민채. 브랜치 만들 때 참고.