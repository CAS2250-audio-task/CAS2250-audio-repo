"""This GPU: basic+vertical three heads, lowcost/frequency+vertical Mean only."""
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'runs/frequency-attention-host5'
REPORT = ROOT/'reports/frequency-attention-host5'
JOBS = [('basic', ['cnn_mean', 'cnn_pooled_query', 'cnn_self_cls']),
        ('lowcost', ['cnn_mean']), ('frequency', ['cnn_mean'])]


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save(state):
    temp = OUT/'status.tmp'
    temp.write_text(json.dumps(state, indent=2)+'\n')
    temp.replace(OUT/'status.json')


def summarize():
    rows = []
    for cnn, _ in JOBS:
        path = ROOT/'reports'/f'frequency-local-{cnn}-v1-host-spk03-e1000/summary.json'
        if not path.exists():
            continue
        for row in json.loads(path.read_text()):
            rows.append(dict(cnn=cnn, vertical=True, model=row['model'],
                parameters=row['parameters'], best_epoch=row['best_epoch'],
                original_f1=row['best']['original_f1'], zero_f1=row['best']['zero_f1'],
                noise_only_fpr=row['noise_only_fpr']))
    if rows:
        with (REPORT/'comparison.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True);REPORT.mkdir(parents=True, exist_ok=True)
    assert not (OUT/'status.json').exists(), 'Inspect existing supervisor before duplicate launch'
    state = dict(status='running', supervisor_pid=os.getpid(), started_utc=now(),
        epochs=1000, speaker='spk03', groups={}, team4_excluded=True)
    files = [ROOT/'experiments/frequency_local_subset.py', Path(__file__)]
    for directory in ['frequency_local', 'local_features', 'compact_pooling']:
        files.extend((ROOT/'experiments'/directory).glob('*.py'))
    state['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    for cnn, heads in JOBS:
        name = f'frequency-local-{cnn}-v1-host-spk03-e1000'
        state['groups'][cnn+'-v1'] = dict(status='queued', models=heads,
            progress=str(ROOT/'runs'/name/'progress.json'))
    save(state)
    try:
        for cnn, heads in JOBS:
            key = cnn+'-v1'
            name = f'frequency-local-{cnn}-v1-host-spk03-e1000'
            command = [sys.executable, '-u', '-m', 'experiments.frequency_local_subset',
                '--cnn', cnn, '--vertical', '--models', *heads,
                '--objective', 'paired_supcon', '--positive-mode', 'same_command',
                '--supcon-weight', '0.1', '--temperature', '0.1',
                '--folds', 'spk03', '--epochs', '1000', '--output', str(ROOT/'runs'/name)]
            with (OUT/(key+'.log')).open('a', buffering=1) as log:
                child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                state['groups'][key].update(status='running', pid=child.pid, command=command, started_utc=now())
                save(state);code = child.wait()
            state['groups'][key].update(status='complete' if code==0 else 'failed', returncode=code, finished_utc=now())
            save(state)
            if code:
                raise RuntimeError(f'{key} failed; inspect {OUT/(key+".log")}')
            summarize()
        state.update(status='complete', finished_utc=now());save(state)
    except Exception as error:
        state.update(status='failed', error=str(error), finished_utc=now());save(state)
        raise


if __name__ == '__main__':
    main()
