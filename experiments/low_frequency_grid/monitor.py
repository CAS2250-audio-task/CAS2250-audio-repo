"""Read saved JSON only; no GPU, checkpoints or model evaluation."""
import argparse,datetime,json,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def read(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return {}


def show():
    state=read(ROOT/'runs/low-frequency-grid/status.json')
    preflight=read(ROOT/'reports/low-frequency-grid/preflight.json')
    costs={r['architecture']:r['estimated_epoch_compute_seconds'] for r in preflight.get('rows',[])}
    costs.update(preflight.get('estimated_actual_epoch_seconds',{}))
    budget=state.get('epochs',500);total=0.
    print('\n상태:',state.get('status','not_started'),'목표:',budget,'epoch')
    for key,group in state.get('groups',{}).items():
        path=Path(group['progress']);p=read(path)
        summary=read(ROOT/'reports'/path.parent.name/'summary.json')
        completed=summary[0].get('completed_epochs',0) if isinstance(summary,list) and summary else 0
        epoch=max(completed,p.get('epoch',0) or 0)
        status=group['status'];remaining=max(0,budget-epoch)*costs.get(group['architecture'],1.)/60
        if status=='complete':remaining=0.
        elif status=='running' and p.get('stage')=='training':remaining=p.get('estimated_training_minutes_remaining',remaining)
        total+=remaining
        print(f'{key}: {status} {epoch}/{budget} | 약 {remaining:.1f}분')
    print(f'전체 추정: 약 {total:.1f}분 + 남은 초기화·최종 평가/저장',flush=True)
    return state.get('status')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch',action='store_true')
    parser.add_argument('--interval',type=float,default=30.)
    args=parser.parse_args()
    try:
        while True:
            status=show()
            if not args.watch or status in ('complete','failed','time_limit'):break
            time.sleep(max(5,args.interval))
    except KeyboardInterrupt:pass


if __name__=='__main__':main()
