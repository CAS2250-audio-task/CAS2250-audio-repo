"""Read saved JSON only: no torch, GPU queries, model loads or evaluations."""
import argparse
import datetime as dt
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def duration(seconds):
    if seconds is None:
        return '계산 중'
    return f'{max(0, round(seconds/60))}분'


def show(status_path):
    state = read(status_path)
    if not state:
        print(f'상태 파일 없음: {status_path}', flush=True)
        return
    split = read(status_path.parent/'split.json')
    keys = split.get('this_server', list(state.get('groups', {})))
    preflight = read(ROOT/'reports/frequency-local-v1/preflight.json')
    total = 0.; estimated = False
    print(f"상태: {state.get('status')} | {dt.datetime.now().astimezone():%Y-%m-%d %H:%M:%S %Z}")
    for key in keys:
        group = state.get('groups', {}).get(key, {})
        progress = read(Path(group.get('progress', ROOT/'runs'/f'frequency-local-{key}-spk03-e1000/progress.json')))
        status = group.get('status', 'queued')
        epoch = progress.get('epoch', 0)
        epochs = progress.get('epochs', state.get('epochs', 1000))
        remaining = None
        if status == 'complete':
            epoch = epochs
            remaining = 0.
        elif status == 'running' and progress.get('stage') == 'training':
            minutes = progress.get('estimated_training_minutes_remaining')
            if minutes is not None:
                remaining = minutes*60
        elif status == 'queued':
            run_name = Path(group.get('progress', '')).parent.name
            provenance = read(ROOT/'reports'/run_name/'resume_provenance.json')
            epoch = provenance.get('resume_epoch', epoch)
            cnn, vertical = key.rsplit('-v', 1)
            row = next((r for r in preflight.get('rows', [])
                        if r['cnn'] == cnn and r['vertical'] == bool(int(vertical))), {})
            cost = row.get('estimated_group_epoch_compute_seconds')
            if cost is not None:
                remaining = (epochs-epoch)*cost*len(group.get('models', [None]*3))/3
                estimated = True
        if remaining is not None:
            total += remaining
        else:
            estimated = True
        stage = progress.get('stage', '-') if status == 'running' else '-'
        print(f'{key}: {status} {epoch}/{epochs} ({epoch/epochs:.1%}) | {stage} | 남음 {duration(remaining)}')
        if status == 'running' and progress.get('updated_utc'):
            updated = dt.datetime.fromisoformat(progress['updated_utc'].replace('Z', '+00:00'))
            age = (dt.datetime.now(dt.timezone.utc)-updated).total_seconds()
            print(f'  기록 갱신 {age:.0f}초 전')
    active = state.get('status') in ('running', 'waiting_for_host5')
    if state.get('status') == 'waiting_for_host5':
        host = read(ROOT/'runs/frequency-attention-host5/status.json')
        ahead = 0.
        for group in host.get('groups', {}).values():
            if group.get('status') == 'running':
                progress = read(Path(group['progress']))
                ahead += 60*progress.get('estimated_training_minutes_remaining', 0.)
        total += ahead
        print(f'선행 작업 남음: 약 {duration(ahead)}')
    if active:
        print(f'전체 남은 시간: 약 {duration(total)}' + (' (대기 그룹은 사전 측정치)' if estimated else ''))
        print('최종 평가·저장 시간은 추가될 수 있음.')
    print(flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--status', type=Path, default=ROOT/'runs/frequency-local-cnn-half-v1/status.json')
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--interval', type=float, default=30.)
    args = parser.parse_args()
    while True:
        show(args.status)
        if not args.watch:
            break
        try:
            time.sleep(max(5., args.interval))
        except KeyboardInterrupt:
            break


if __name__ == '__main__':
    main()
