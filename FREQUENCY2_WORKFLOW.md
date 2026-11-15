# 다른 서버 담당: 세로 CNN + 세로 어텐션, Pooled/Self 2개

전용 브랜치: **`experiment/frequency-attention-frequency2`**.
이 브랜치에서는 아래 전용 노트북만 사용한다. 이전 team4 전체4개 노트북이나 전체조건 launcher는 실행하지 않는다.

## 서버에서 받기

```bash
mkdir -p ~/audio-frequency2
cd ~/audio-frequency2
git clone --branch experiment/frequency-attention-frequency2 \
  https://github.com/CAS2250-audio-task/CAS2250-audio-repo.git
cd CAS2250-audio-repo
```

Jupyter에서 [`notebooks/frequency_attention_frequency2.ipynb`](notebooks/frequency_attention_frequency2.ipynb)를 열고 위에서 아래로 실행한다.
노트북의 기본 REPO는 위 경로와 같으며 다른 경로를 사용하면 WORK/REPO를 수정한다.
Python3.10/3.11 Linux NVIDIA GPU 환경을 사용한다. 패키지 설치 셀은 필요할 때 실행한 뒤 커널을 재시작한다.
초기 환경은 torch/torchaudio2.3.1+cu121을 기준으로 하며 세부 버전은 노트북에 있다.

## 학습 분담

- **이 서버:** frequency CNN + vertical attention의 `cnn_pooled_query`, `cnn_self_cls`, 각1000epoch.
- **기존 서버:** 해당 조건의 Mean, 그리고 lowcost 조건의 Mean/Pooled/Self.
- 전달된 lowcost670epoch 체크포인트는 이 서버에서 받거나 재개할 필요 없다.
- 이 두 모델은 처음부터 시작한다. 두 모델을 한 그룹에서 함께 학습하며, 서로 가중치는 공유하지 않는다.

입력은 기존 `compact-pooling-v1-assets` 공개 릴리스의 개발812음성·동일 포먼트 캐시·소음을 사용한다.
노트북에서 약1GB 다운로드와 SHA256·cache identity 검증을 수행한다.
화자03 검증203개, 화자02/04/05 학습609개, seed42,1000epoch,
CE+0.1SupCon(same_command),temperature0.1, 기존 LR/batch/증강을 유지한다.
사전학습/검출기 없이 전체1.5초 입력. 독립test01은 읽지 않는다.
시간 집계 외의 CNN·분류기·투영기 초기값은 전체3head 실행과 일치한다.

결과 경로:

```text
runs/frequency-local-frequency-v1-teamPS-spk03-e1000/
reports/frequency-local-frequency-v1-teamPS-spk03-e1000/
~/audio-frequency2/frequency2-control/  # 로그·실행 상태
```

## 완료·원격 업로드

진행·ETA는 노트북5번 셀에서 JSON만 읽는다.1000epoch와 최종 평가가 끝나면6번 결과 패키징 셀을 실행한다.
7번 업로드 셀은 보고서를 이 전용 브랜치에 commit/push하고, 가중치·optimizer·학습 이력은
`frequency-attention-frequency2-results-v1` 릴리스에 올린다.

서버 터미널에서 본인 GitHub 계정으로 먼저 인증한다. 저장소 쓰기 권한과 GitHub CLI가 필요하다.

```bash
gh auth login
gh auth setup-git
```

Git author가 없으면 checkout에서 본인 정보로 설정한다. 토큰은 노트북·채팅·Git에 적지 않는다.
업로드 후 **브랜치 commit SHA와 릴리스 링크**를 기존 서버 담당자에게 전달한다.
선택 best.pt와 종료점 last.pt는 모두 보존하며 원본 데이터·캐시는 다시 업로드하지 않는다.

학습 중 소스 수정·git pull·패키지 교체는 하지 않는다. 소스 hash가 바뀌면 재개가 거부된다.
브라우저를 닫아도 서버 프로세스가 유지되는 환경이면 백그라운드 학습은 계속되나 workspace 종료는 별개다.
중단이 필요하면 노트북 끝의 프로세스 그룹 중단 절차를 사용한다.
