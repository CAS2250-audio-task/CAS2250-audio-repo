"""Detached, single-instance launcher; waits for restored assets before training."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
from .run import ROOT,OUT,write

def alive(pid):
    path=Path(f'/proc/{pid}/stat')
    return path.exists() and path.read_text().split()[2]!='Z'

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-pid',type=int,required=True)
    parser.add_argument('--timeout-minutes',type=int,default=90)
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'launcher.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.monotonic()
    while not ((OUT/'sources.json').exists() and (OUT/'formants.json').exists()):
        if not (OUT/'sources.json').exists() and not alive(args.source_pid):
            raise RuntimeError('Source preparation exited without sources.json; inspect sources.log')
        if time.monotonic()-start>args.timeout_minutes*60:
            raise TimeoutError('Asset preparation timeout; inspect sources.log and formants.log')
        write(OUT/'launcher_status.json',dict(stage='waiting_for_assets',elapsed_minutes=(time.monotonic()-start)/60))
        time.sleep(10)
    with (OUT/'training.log').open('a') as log:
        job=subprocess.Popen([sys.executable,'-u','-m','experiments.compact_pooling.run'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        write(OUT/'launcher_status.json',dict(stage='training',pid=job.pid))
        status=job.wait()
    write(OUT/'launcher_status.json',dict(stage='complete' if status==0 else 'failed',exit_code=status))
    raise SystemExit(status)

if __name__=='__main__':main()
