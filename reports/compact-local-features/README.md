# CNN 시간범위·중간 특징 전달 비교

검증 화자03,다른 개발화자3명으로 학습. seed42,소음+화자 대조,각1000epoch/cosine1000. 세 집계 모델(mean/pooled-query/self-CLS)을 모두 비교한다. 입력 전체1.5초는 유지하며 spk01 음원은 읽지 않는다.

| 조건 | 최종 시간 수용영역 | 중간 경로 |
|---|---:|---|
| baseline,기존 실행 재사용 | 1.105초 | 없음 |
| short | 0.625초 | 없음 |
| local | 1.105초 | 약0.245초 특징64→32,zero-init 잔차 연결 |
| both | 0.625초 | 약0.245초 특징64→32,zero-init 잔차 연결 |

short는 CNN의마지막큰depthwise커널만29/dilation2→9/dilation1로바꾼다. local은두번째블록출력의별도투영을최종32차원특징에더한다. 중간투영은0으로초기화해 local의시작출력을baseline과같게한다. 나머지공통초기가중치를유지하며짧은커널은기존중앙9tap을초기화분산에맞게스케일해사용한다.

기준baseline은이전완료한 compact-noise-speaker-spk03-e1000 결과를재사용한다. 새로학습하는것은3조건×3모델×1000epoch=9000model-epochs다. 동일한원본·포먼트캐시·소음·crop/SNR·학습스케줄을사용한다.

진행은 runs/compact-local-features/status.json,short.log,local.log,both.log에저장한다. 최대2조건을같은GPU에서동시에실행한다. 완료후선택모델의예측과기준대비증강trace를검증하고최종비교표·학습곡선으로이문서를교체한다.

이번시험은0.245초가최적음절길이라는가정을검증하는것이아니라,기존중간층의짧은특징을직접전달하는효과를보는시험이다. 파라미터수가약1~3%달라지며단일개발화자·seed1개의선별실험이다.

실행: `python -u -m experiments.launch_local_features`
