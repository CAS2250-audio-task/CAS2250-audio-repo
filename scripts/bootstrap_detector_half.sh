#!/usr/bin/env bash
set -euo pipefail
TASK_REPO='CAS2250-audio-task/CAS2250-audio-repo'
TASK_BRANCH='experiment/detector-diversity-v2'
TASK_TAG='detector-diversity-v2-distributed-assets-v1'
TASK_DIR="${DET_V2_DIR:-$HOME/CAS2250-detector-diversity-v2-jupyter}"
export PATH="$HOME/.local/bin:$TASK_DIR/.tools/bin:$PATH"
mkdir -p "$TASK_DIR/.tools/bin"
printf '\n[1/6] GitHub CLI 및 쓰기 인증 확인\n'
if ! command -v gh >/dev/null 2>&1; then
    case "$(uname -m)" in x86_64) TASK_ARCH=amd64;; aarch64|arm64) TASK_ARCH=arm64;; *) echo '지원하지 않는 CPU 구조'; exit 1;; esac
    TASK_GH=2.104.0
    curl -fL --retry 3 "https://github.com/cli/cli/releases/download/v$TASK_GH/gh_${TASK_GH}_linux_${TASK_ARCH}.tar.gz" -o "$TASK_DIR/.tools/gh.tar.gz"
    tar -xzf "$TASK_DIR/.tools/gh.tar.gz" -C "$TASK_DIR/.tools"
    cp "$TASK_DIR/.tools/gh_${TASK_GH}_linux_${TASK_ARCH}/bin/gh" "$TASK_DIR/.tools/bin/gh"
fi
if ! gh auth status --hostname github.com >/dev/null 2>&1; then
    printf '이 서버의 GitHub 인증이 필요합니다. 표시되는 기기 코드로 웹 승인을 한 번 진행하세요.\n'
    gh auth login --hostname github.com --git-protocol https --web
fi
gh api "repos/$TASK_REPO" --jq '.permissions.push' | python3 -c 'import sys; assert sys.stdin.read().strip()=="true", "이 계정에는 저장소 쓰기 권한이 없습니다"'
printf '\n[2/6] 실험 브랜치 받기\n'
if [ -d "$TASK_DIR/repo/.git" ]; then
    if [ -n "$(git -C "$TASK_DIR/repo" status --porcelain)" ]; then echo '기존 repo에 수정 내용이 있습니다. 덮어쓰지 않고 중단합니다. DET_V2_DIR에 새 경로를 지정하세요.'; exit 1; fi
    git -C "$TASK_DIR/repo" fetch origin "$TASK_BRANCH"
    git -C "$TASK_DIR/repo" checkout --detach FETCH_HEAD
else
    git clone --single-branch --branch "$TASK_BRANCH" "https://github.com/$TASK_REPO.git" "$TASK_DIR/repo"
fi
cd "$TASK_DIR/repo"
printf '\n[3/6] Python/CUDA 환경 준비\n'
TASK_PY=''
for candidate in "${DET_V2_PYTHON:-}" "$TASK_DIR/repo/.venv/bin/python" python3.11 python3.10 python3; do
    [ -n "$candidate" ] || continue
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import torch,torchaudio,numpy,scipy,pandas,soundfile; assert torch.__version__.split("+")[0]=="2.6.0"; assert torch.cuda.is_available()' >/dev/null 2>&1; then TASK_PY="$(command -v "$candidate")"; break; fi
done
if [ -z "$TASK_PY" ]; then
    TASK_BASE=''
    for candidate in python3.11 python3.10 python3; do
        if command -v "$candidate" >/dev/null 2>&1; then TASK_BASE="$candidate"; break; fi
    done
    [ -n "$TASK_BASE" ] || { echo 'Python3가 필요합니다'; exit 1; }
    "$TASK_BASE" -m venv "$TASK_DIR/repo/.venv"
    TASK_PY="$TASK_DIR/repo/.venv/bin/python"
    "$TASK_PY" -m pip install --upgrade pip
    "$TASK_PY" -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
    if "$TASK_PY" -c 'import sys; assert sys.version_info<(3,12)' >/dev/null 2>&1; then TASK_NUMPY='numpy==1.24.4'; else TASK_NUMPY='numpy>=1.26,<2'; fi
    "$TASK_PY" -m pip install "$TASK_NUMPY" scipy==1.14.0 pandas==2.2.2 soundfile==0.14.0
fi
"$TASK_PY" -c 'import torch; assert torch.cuda.is_available(), "CUDA를 사용할 수 없습니다"; print(torch.__version__,torch.cuda.get_device_name(0))'
printf '\n[4/6] 공통 소음/이어학습 체크포인트 받기 및 검증\n'
mkdir -p "$TASK_DIR/downloads"
for task_file in detector-diversity-v2-assets.tar.gz detector-diversity-v2-resume.tar.gz detector-diversity-v2-resume.tar.gz.sha256; do
    if [ ! -f "$TASK_DIR/downloads/$task_file" ]; then gh release download "$TASK_TAG" --repo "$TASK_REPO" --pattern "$task_file" --dir "$TASK_DIR/downloads"; fi
done
"$TASK_PY" - "$TASK_DIR/downloads" <<'PY'
import hashlib,json,sys
from pathlib import Path
folder=Path(sys.argv[1])
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for part in iter(lambda:f.read(1048576),b''):h.update(part)
 return h.hexdigest()
metadata=json.loads(Path('reports/detector-diversity-v2/assets.json').read_text());assert sha(folder/metadata['filename'])==metadata['sha256'],'Noise archive checksum mismatch'
expected=(folder/'detector-diversity-v2-resume.tar.gz.sha256').read_text().split()[0];assert sha(folder/'detector-diversity-v2-resume.tar.gz')==expected,'Resume archive checksum mismatch'
PY
if [ ! -f runs/detector-diversity-v2/.remote_prepared ]; then
    tar -xzf "$TASK_DIR/downloads/detector-diversity-v2-assets.tar.gz"
    tar -xzf "$TASK_DIR/downloads/detector-diversity-v2-resume.tar.gz"
    "$TASK_PY" -m experiments.detector_diversity.assets --rebase
    "$TASK_PY" scripts/detector_diversity_remote.py --prepare-only
    touch runs/detector-diversity-v2/.remote_prepared
fi
printf '\n[5/6] 경계 헤드 + 2D CNN 실험 시작 (각300epoch, 전송된20/21epoch부터 재개)\n'
if [ -f runs/detector-diversity-v2/remote.pid ] && kill -0 "$(cat runs/detector-diversity-v2/remote.pid)" 2>/dev/null; then
    echo '이미 실행 중입니다.'
else
    nohup env PYTHONUTF8=1 PYTHONIOENCODING=utf-8 CUBLAS_WORKSPACE_CONFIG=:4096:8 "$TASK_PY" scripts/detector_diversity_remote.py > runs/detector-diversity-v2/remote.log 2>&1 < /dev/null &
    echo $! > runs/detector-diversity-v2/remote.pid
fi
printf '\n[6/6] 완료되면 결과 tar와 체크섬을 같은 GitHub 릴리스에 자동 업로드합니다.\n'
printf '작업 경로: %s\nPID: %s\n진행 확인: tail -f %q\n' "$PWD" "$(cat runs/detector-diversity-v2/remote.pid)" "$PWD/runs/detector-diversity-v2/remote.log"
