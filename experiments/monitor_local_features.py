"""Read-only terminal monitor for the local-feature experiment."""
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
KST = timezone(timedelta(hours=9), name='KST')


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def timestamp(value):
    if not value:
        return None
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def duration(seconds):
    seconds = max(0, int(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f'{hours}시간 {minutes:02d}분 {seconds:02d}초' if hours else f'{minutes}분 {seconds:02d}초'


def render():
    now = datetime.now(timezone.utc)
    state = read_json(ROOT / 'runs/compact-local-features/status.json')
    lines = [f'CNN 실험 실시간 현황 | {now.astimezone(KST):%m/%d %H:%M:%S} KST',
             'ETA는 학습 기준 추정치입니다. 최종 평가·결과 정리 시간은 추가됩니다.', '']
    variants = state.get('variants', {})
    if not variants:
        lines.append('진행 정보를 기다리는 중입니다.')
    for name in ('short', 'local', 'both'):
        info = variants.get(name, {})
        progress_path = Path(info.get('progress', ROOT / f'runs/compact-local-{name}-spk03-e1000/progress.json'))
        progress = read_json(progress_path)
        status = info.get('status', 'queued')
        if status == 'complete':
            lines.append(f'{name:6} [완료] 1000/1000 epoch')
            continue
        if status == 'failed':
            lines.append(f'{name:6} [실패] 로그: runs/compact-local-features/{name}.log')
            continue
        if status == 'queued':
            lines.append(f'{name:6} [대기]')
            continue
        epoch = progress.get('epoch', 0)
        total = progress.get('epochs', 1000)
        ratio = min(1, epoch / max(1, total))
        filled = int(24 * ratio)
        bar = '#' * filled + '-' * (24 - filled)
        stage = progress.get('stage', 'starting')
        lines.append(f'{name:6} [{bar}] {epoch}/{total} ({ratio:.1%}) | {stage}')
        updated = timestamp(progress.get('updated_utc'))
        remaining = progress.get('estimated_training_minutes_remaining')
        if stage == 'complete':
            lines.append('       학습·평가 완료, 상위 프로세스의 결과 정리를 기다리는 중')
        elif stage == 'training' and remaining is not None and updated:
            finish = updated + timedelta(minutes=float(remaining))
            seconds = (finish - now).total_seconds()
            stale = (now - updated).total_seconds()
            if seconds > 0:
                lines.append(f'       ETA {duration(seconds)} | 예상 학습 종료 {finish.astimezone(KST):%m/%d %H:%M:%S} KST')
            else:
                lines.append('       기존 ETA 경과 — 다음 진행 정보 갱신을 기다리는 중')
            if stale > 60:
                lines.append(f'       주의: 진행 정보가 {duration(stale)} 동안 갱신되지 않았습니다.')
        else:
            lines.append('       ETA 계산 대기 / 최종 평가 중')
        for row in progress.get('results', []):
            if 'zero_f1' in row:
                lines.append(f'       {row["model"]:17} 현재 0dB F1 {row["zero_f1"] * 100:6.2f}%')
    lines.extend(['', f'전체 상태: {state.get("status", "대기")} | 종료: Ctrl+C'])
    if state.get('error'):
        lines.append(f'오류: {state["error"]}')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', action='store_true', help='Print one snapshot and exit')
    parser.add_argument('--interval', type=float, default=2, help='Refresh interval in seconds')
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error('--interval must be positive')
    try:
        while True:
            if sys.stdout.isatty() and not args.once:
                print('\033[2J\033[H', end='')
            print(render(), flush=True)
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
