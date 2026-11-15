# Compact CNN 소음·화자 대조학습 — 실행 중

사용자 지정: 소음만 대조 / 다른 화자의 같은 명령어까지 대조, 모델3종(mean / pooled-query / self-CLS), 사전학습 없음. 검증 화자03은1000epoch, 화자02는100epoch. 각 fold의 다른 개발화자3명으로만 학습한다. spk01 음원은 읽지 않는다.

공유 릴리스 compact-pooling-v1-assets.tar.gz(998,749,839 bytes)의 전체 SHA256 및1069개 내부 파일 해시를 검증했다. 포먼트/소음 캐시는 재생성하지 않는다. 실행환경은 기존 서버의 torch2.6+CUDA12.4이며 원 서버torch2.3+CUDA12.1과 다르다. 현재 서버의 두 조건끼리 같은 환경에서 비교한다.

소음만 조건에서는 같은 녹음의 원본·추가잡음 두 뷰만positive, 다른 명령어만negative. 다른 녹음의 같은 명령어는 제외한다. 화자 포함 조건은 같은 명령어의 다른 녹음·학습화자도positive. 두 조건 모두 CE+0.1 SupCon, temperature0.1, 학습전용 투영부32→64→32, silence는 CE만 사용한다.

두 조건은 같은 GPU에서 별도 프로세스로 실행한다. 각 조건에서 화자03/1000을 먼저, 화자02/100을 다음에 실행한다. 같은 초기 모델·배치순서·파형 증강 난수를 사용한다. 100epoch 결과에서 연장한 것이 아니라 각 예산의 cosine 스케줄로 무작위 초기화에서 시작한다.

진행: runs/compact-contrastive-launch/status.json 및 noise.log / noise-speaker.log. 각 작업의 progress.json에 실제epoch·소요시간을 기록한다. 완료 후 이 문서를 최종 비교표로 교체하고 저장 예측에서 점수를 재계산한다.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -u -m experiments.launch_compact_contrastive
```
