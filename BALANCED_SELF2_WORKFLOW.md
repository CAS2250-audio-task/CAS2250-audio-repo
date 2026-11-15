# 시간 기준 균형 분담: 다른 서버는 Self 두 작업

**이 분담이 이전 `experiment/frequency-attention-frequency2` 분담을 대체한다.**
이전 브랜치의 세로 CNN Pooled/Self 두 모델을 동시에 실행하지 않는다.

| 서버 | 남은 작업 |
|---|---|
| 기존 서버 | 현재 Mean 마무리 + lowcost Pooled671→1000 + 세로CNN Pooled1→1000 |
| 다른 서버 | lowcost Self671→1000 + 세로CNN Self1→1000 |

동일 RTX3090 기준 각 서버 잔여 연산을 대략30~40분대로 맞춘 분담이다.
GPU 성능 차이, 서버 준비·약1GB 입력 다운로드·최종 평가 시간은 별도다.

## 서버에서 받기

```bash
mkdir -p ~/audio-balanced-self2
cd ~/audio-balanced-self2
git clone --branch experiment/frequency-attention-balanced-self2 \
  https://github.com/CAS2250-audio-task/CAS2250-audio-repo.git
cd CAS2250-audio-repo
```

Jupyter에서 **[`notebooks/frequency_attention_balanced_self2.ipynb`](notebooks/frequency_attention_balanced_self2.ipynb)**만 실행한다.
이전 team4/frequency2 노트북과 전체실험 launcher는 사용하지 않는다.

노트북은 동일 원본·포먼트 캐시·소음을 받아 검증한 뒤, 공개된 partial670epoch 패키지에서
Self 모델/optimizer/projector/정규화/학습 이력만 분리하여 별도 출력 경로에 복원한다.
학습률1000epoch 일정은 그대로이며671epoch부터 재개한다. 세로CNN Self는 처음부터1000epoch까지 학습한다.
모델별 초기화는 전체3head 실행과 같다. GPU 환경 차이는 메타데이터와 복원 기록에 남긴다.

각 모델 결과:

```text
runs/frequency-local-lowcost-v1-self-spk03-e1000/
runs/frequency-local-frequency-v1-self-spk03-e1000/
reports/위와 같은 실험명/
```

기존 서버는 별도의 `-pooled-` 경로를 사용하므로 결과 파일이 충돌하지 않는다.

## 완료 후 원격 업로드

본인 계정으로 `gh auth login`, `gh auth setup-git` 후 노트북 마지막 업로드 셀을 실행한다.
저장소 쓰기 권한이 필요하다. 토큰을 노트북·채팅·Git에 저장하지 않는다.
보고서는 이 전용 브랜치로 push하고, 가중치·학습 이력은
`frequency-attention-balanced-self2-results-v1` 릴리스에 업로드한다.
완료 후 기존 서버 담당자에게 commit SHA와 릴리스 링크를 전달한다.

학습 중 소스 수정·git pull·패키지 변경은 하지 않는다.
실행 셀은 background supervisor만 시작한다. 진행·ETA 확인, 중단·재개 방법은 노트북에 있다.
