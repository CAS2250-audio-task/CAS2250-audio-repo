"""Launch the frozen study in a separate session, independent of SSH/VSCode."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "runs/attention-pitch-v1"
REPORT = ROOT / "reports/attention-pitch-v1"
PYTHON = "/root/audio/.worktrees/pretrain/runs/denoise-env/bin/python"


def main():
    protocol = json.loads((REPORT / "protocol.json").read_text(encoding="utf-8"))
    checked = json.loads((REPORT / "preflight.json").read_text(encoding="utf-8"))
    assert checked["source_sha256"] == protocol["source_sha256"]
    assert checked["acoustic_pitch_contours_verified"]
    for name, digest in protocol["source_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    previous = STUDY / "launch.json"
    if previous.exists():
        old = json.loads(previous.read_text(encoding="utf-8"))
        command = Path(f"/proc/{old['pid']}/cmdline")
        if command.exists() and b"experiments.attention_pitch.run" in command.read_bytes():
            print(json.dumps(dict(already_running=True, **old)))
            return
    environment = dict(os.environ, PYTHONUTF8="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
    log = (STUDY / "execution.log").open("ab", buffering=0)
    process = subprocess.Popen([PYTHON, "-u", "-m", "experiments.attention_pitch.run"],
        cwd=ROOT, env=environment, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True, close_fds=True,
        preexec_fn=lambda: signal.signal(signal.SIGHUP, signal.SIG_IGN))
    log.close()
    time.sleep(.3)
    assert process.poll() is None, "Launch exited; inspect execution.log"
    assert os.getsid(process.pid) == process.pid
    value = dict(pid=process.pid, session_id=os.getsid(process.pid),
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        command=[PYTHON, "-u", "-m", "experiments.attention_pitch.run"], cwd=str(ROOT),
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        detached=True, stdin="DEVNULL", sighup="ignored", log=str(STUDY / "execution.log"))
    for path in (STUDY / "launch.json", REPORT / "launch.json"):
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(value, ensure_ascii=False))


if __name__ == "__main__":
    main()
