"""Wait for this GPU's host5 supervisor, then finish imported team4 checkpoints."""
import csv
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'runs/frequency-attention-team4-resume'
REPORT = ROOT/'reports/team4-resume'
HEADS = ['cnn_pooled_query', 'cnn_self_cls']


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save(state):
    temp = OUT/'status.tmp'
    temp.write_text(json.dumps(state, indent=2)+'\n')
    temp.replace(OUT/'status.json')


def summarize():
    rows = []
    for cnn in ['lowcost', 'frequency']:
        path = ROOT/'reports'/f'frequency-local-{cnn}-v1-teamPS-spk03-e1000/summary.json'
        if path.exists():
            for row in json.loads(path.read_text()):
                rows.append(dict(cnn=cnn, vertical=True, model=row['model'],
                    best_epoch=row['best_epoch'], original_f1=row['best']['original_f1'],
                    zero_f1=row['best']['zero_f1'], noise_only_fpr=row['noise_only_fpr']))
    if rows:
        with (REPORT/'comparison.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cnns', choices=['lowcost', 'frequency'], nargs='+', default=['lowcost', 'frequency'])
    parser.add_argument('--models', choices=HEADS, nargs='+', default=HEADS)
    args = parser.parse_args()
    assert len(set(args.cnns)) == len(args.cnns)
    heads=args.models
    suffix='teamPS' if len(heads)==2 else ('pooled' if heads[0]=='cnn_pooled_query' else 'self')
    OUT.mkdir(exist_ok=True, parents=True);REPORT.mkdir(exist_ok=True, parents=True)
    assert not (OUT/'status.json').exists(), 'Inspect existing supervisor before duplicate launch'
    state = dict(status='waiting_for_host5', supervisor_pid=os.getpid(), started_utc=now(),
        epochs=1000, imported_release='team4-partial-20261115T192001Z', groups={},
        externally_assigned_cnns=[c for c in ['lowcost', 'frequency'] if c not in args.cnns])
    for cnn in args.cnns:
        name = f'frequency-local-{cnn}-v1-{suffix}-spk03-e1000'
        state['groups'][cnn+'-v1'] = dict(status='queued', models=heads,
            progress=str(ROOT/'runs'/name/'progress.json'))
    state['source_sha256'] = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ['frequency_local','local_features','compact_pooling']
        for p in (ROOT/'experiments'/folder).glob('*.py')}
    helper = ROOT/'experiments/frequency_local_subset.py'
    state['source_sha256'][str(helper.relative_to(ROOT))] = hashlib.sha256(helper.read_bytes()).hexdigest()
    save(state)
    try:
        while True:
            host = json.loads((ROOT/'runs/frequency-attention-host5/status.json').read_text())
            if host['status']=='complete':
                break
            if host['status']!='running':
                raise RuntimeError('Host5 did not complete: '+host['status'])
            time.sleep(15)
        state.update(status='running', training_started_utc=now());save(state)
        for cnn in args.cnns:
            for path, digest in state['source_sha256'].items():
                assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest, path
            key = cnn+'-v1';name = f'frequency-local-{cnn}-v1-{suffix}-spk03-e1000'
            command = [sys.executable,'-u','-m','experiments.frequency_local_subset',
                '--cnn',cnn,'--vertical','--models',*heads,
                '--objective','paired_supcon','--positive-mode','same_command',
                '--supcon-weight','0.1','--temperature','0.1','--folds','spk03',
                '--epochs','1000','--output',str(ROOT/'runs'/name)]
            with (OUT/(key+'.log')).open('a',buffering=1) as log:
                child = subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                state['groups'][key].update(status='running',pid=child.pid,command=command,started_utc=now())
                save(state);code = child.wait()
            state['groups'][key].update(status='complete' if code==0 else 'failed',returncode=code,finished_utc=now());save(state)
            if code:
                raise RuntimeError(f'{key} failed; inspect {OUT/(key+".log")}')
            summarize()
        state.update(status='complete',finished_utc=now());save(state)
    except Exception as error:
        state.update(status='failed',error=str(error),finished_utc=now());save(state)
        raise


if __name__ == '__main__':
    main()
