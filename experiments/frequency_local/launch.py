"""Sequential GPU supervisor; reuse local reference and train five new groups."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'runs/frequency-local-v1'
REPORT = ROOT/'reports/frequency-local-v1'
GROUPS = [('lowcost', False), ('frequency', False), ('basic', True),
          ('lowcost', True), ('frequency', True)]


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(state):
    path = OUT/'status.json'
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(state, indent=2)+'\n')
    temp.replace(path)


def summarize():
    import pandas as pd
    rows = []
    for cnn, vertical in [('basic', False)] + GROUPS:
        name = f'frequency-local-{cnn}-v{int(vertical)}-spk03-e1000'
        source = ROOT/'reports'/name/'summary.json'
        if cnn == 'basic' and not vertical:
            source = ROOT/'reports/compact-local-local-spk03-e1000/summary.json'
        if source.exists():
            for row in json.loads(source.read_text()):
                rows.append(dict(cnn=cnn, vertical=vertical, model=row['model'],
                    parameters=row['parameters'], best_epoch=row['best_epoch'],
                    original_f1=row['best']['original_f1'], zero_f1=row['best']['zero_f1'],
                    noise_only_fpr=row['noise_only_fpr'], reused=cnn=='basic' and not vertical))
    pd.DataFrame(rows).to_csv(REPORT/'comparison.csv', index=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True);REPORT.mkdir(parents=True, exist_ok=True)
    if (OUT/'status.json').exists():
        raise RuntimeError('Existing supervisor status: inspect before starting a duplicate launcher')
    state = dict(status='running', supervisor_pid=os.getpid(), started_utc=now(),
        epochs=1000, speaker='spk03', groups={})
    state['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ('frequency_local', 'local_features', 'compact_pooling')
        for p in (ROOT/'experiments'/folder).glob('*.py')}
    save(state);summarize()
    try:
        for cnn, vertical in GROUPS:
            key = f'{cnn}-v{int(vertical)}'
            name = f'frequency-local-{key}-spk03-e1000'
            command = [sys.executable, '-u', '-m', 'experiments.frequency_local.run',
                '--cnn', cnn, '--objective', 'paired_supcon', '--positive-mode', 'same_command',
                '--supcon-weight', '0.1', '--temperature', '0.1', '--folds', 'spk03',
                '--epochs', '1000', '--output', str(ROOT/'runs'/name)]
            if vertical: command.append('--vertical')
            with (OUT/(key+'.log')).open('a', buffering=1) as log:
                child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                state['groups'][key] = dict(status='running', pid=child.pid, command=command,
                    started_utc=now(), progress=str(ROOT/'runs'/name/'progress.json'))
                save(state)
                code = child.wait()
            state['groups'][key].update(status='complete' if code==0 else 'failed', returncode=code, finished_utc=now())
            save(state)
            if code: raise RuntimeError(f'{key} failed; see {OUT/(key+".log")}')
            summarize()
        state.update(status='complete', finished_utc=now());save(state)
    except Exception as error:
        state.update(status='failed', error=str(error), finished_utc=now());save(state)
        raise


if __name__ == '__main__':
    main()
