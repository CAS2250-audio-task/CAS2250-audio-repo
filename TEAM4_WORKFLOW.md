# 팀원 GPU 작업: 세로 어텐션 4개, 각1000epoch

작업 브랜치: **`experiment/frequency-attention-team4`**.
기존 서버와 별도 브랜치에서 학습하고 완료 결과를 이 브랜치에 올린다.

## 1. 서버에 코드 받기

```bash
mkdir -p ~/audio-team4
cd ~/audio-team4
git clone --branch experiment/frequency-attention-team4 \
  https://github.com/CAS2250-audio-task/CAS2250-audio-repo.git
cd CAS2250-audio-repo
```

Python3.10/3.11 NVIDIA GPU 환경에서 Jupyter를 열고
[`notebooks/frequency_attention_team4.ipynb`](notebooks/frequency_attention_team4.ipynb)를 실행한다.
노트북 기본 REPO 경로는 위 clone 경로와 같다. 다른 위치를 쓰면 노트북의 WORK/REPO를 조정한다.
Jupyter가 없다면 선택한 Python 환경에서 `python -m pip install jupyterlab ipykernel`로 설치한다.
Jupyter 접근은 해당 서버가 제공하는 UI 또는 SSH 터널을 사용한다.

## 2. 담당 범위

| CNN | 추가 세로 어텐션 | 시간 집계 | 신규 학습 수 |
|---|---|---|---:|
| 소형2D 전처리 + local CNN (`lowcost`) | 있음 | Pooled / Self | 2 |
| 세로 CNN + 중간 특징 (`frequency`) | 있음 | Pooled / Self | 2 |

총4개를 각1000epoch 학습한다. 두 그룹은 순차 실행한다.
기존 서버 담당은 basic+세로의3head, lowcost+세로의Mean, frequency+세로의Mean 총5개다.
기존 서버의 어텐션 없는6개 학습과 중복 실행하지 않는다.
`experiments.frequency_local.launch`는 전체 조건 launcher이므로 실행하지 않는다.

## 3. 노트북 실행 순서

1. 전용 커널에서 노트북에 적힌 패키지를 준비한다. torch/torchaudio2.3.1+cu121을 기준으로 한다.
2. 전용 브랜치와 기준 학습 소스가 일치하는지 검사한다.
3. 기존 공개 릴리스의 원본812개·포먼트 캐시·소음을 다운로드하고 SHA256을 검증한다.
4. `frequency_local_subset` 진입점으로 담당4개만 백그라운드 실행한다.
5. JSON 기반 모니터로 epoch·현재 그룹 ETA를 확인한다.
6. 완료 후 summary/history를 검사하고 가중치·epoch기록·예측·증강trace를 tar.gz로 묶는다.
7. reports를 Git에 push하고 tar.gz와 SHA256을 GitHub release로 게시한다.

별도 subset 진입점은 부분 모델 선택에도 전체3head 실행과 같은 분류기 및 SupCon projector 초기값을 보존한다.
일반 진입점에 `--models`만 전달하면 초기화 조건이 달라질 수 있으므로 노트북 명령을 그대로 사용한다.
소음/화자 대조학습, 증강, seed42, 화자03 검증, LR/batch/epoch 수를 바꾸지 않는다.
독립test01은 사용하지 않는다. 원본은 추가잡음 합성 전 녹음이며 완전한 clean을 뜻하지 않는다.

## 4. 완료 후 원격 게시

GitHub 저장소 쓰기 권한이 있는 **본인 계정**으로 서버 터미널에서 인증한다.

```bash
gh auth login
gh auth setup-git
```

GitHub CLI가 없으면 서버 환경에 맞게 먼저 설치한다. 토큰을 코드·노트북·채팅에 적지 않는다.
Git author가 없다면 전용 checkout에서 본인 정보로 설정한다.

```bash
git config user.name "본인 이름"
git config user.email "본인 이메일"
```

노트북 마지막 업로드 셀을 실행하면:

- `reports/frequency-local-{lowcost,frequency}-v1-teamPS-spk03-e1000/`를 이 브랜치에 commit/push한다.
- 가중치·optimizer·전체 학습기록은 `frequency-attention-team4-results-v1` 릴리스에 업로드한다.
- 원본·캐시는 기존 `compact-pooling-v1-assets` 릴리스를 참조하며 다시 올리지 않는다.
- 선택 best.pt와 마지막1000epoch last.pt를 모두 보존한다.

업로드 완료 후 기존 서버 담당자에게 **브랜치 commit SHA와 릴리스 링크**를 전달한다.
팀원이 기존 서버 브랜치에 직접 병합할 필요는 없다.

## 재개 및 문제 발생

노트북 브라우저를 닫아도 서버 프로세스가 유지되는 환경이면 계속 학습한다.
workspace 자체 종료/유휴 정지는 별개이며 Jupyter 셀 중지는 백그라운드 학습을 중단하지 않는다.
중단 명령은 노트북 끝에 있다. 같은 소스·경로·옵션으로 실행하면10epoch 단위 last.pt에서 재개한다.
학습 중 `git pull`, 소스 수정, 패키지 교체를 하지 않는다. 소스 hash가 바뀌면 재개가 거부된다.
실패하면 해당 그룹 log 끝부분을 먼저 전달한다. `force push`나 기존 checkpoint 삭제로 해결하지 않는다.
