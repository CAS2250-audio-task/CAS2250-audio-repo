"""One read-only snapshot; watch handles refresh without touching training."""
from datetime import datetime,timedelta,timezone
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
p=root/'runs/compact-acoustic-local-spk03-e1000/progress.json'
try:
    d=json.loads(p.read_text())
    stage=d.get('stage','preparing')
    print('상태:',stage)
    if stage=='complete':
        print('학습·평가 완료')
    else:
        print('Epoch: {}/{}'.format(d.get('epoch','준비 중'),d.get('epochs',1000)))
        m=d.get('estimated_training_minutes_remaining')
        if m is not None and stage=='training':
            updated=datetime.fromisoformat(d['updated_utc'].replace('Z','+00:00'))
            now=datetime.now(timezone.utc)
            finish=updated+timedelta(minutes=m)
            remaining=max(0,int((finish-now).total_seconds()))
            minutes,seconds=divmod(remaining,60)
            print('ETA: {}분 {:02d}초'.format(minutes,seconds))
            print('예상 학습 종료:',finish.astimezone(timezone(timedelta(hours=9))).strftime('%m/%d %H:%M:%S'),'KST')
            print('최종 평가 시간은 별도 추가')
            if (now-updated).total_seconds()>60:print('주의: 진행 정보 갱신이 60초 이상 지연됨')
        else:print('ETA 계산 대기 / 최종 평가 중')
except (OSError,ValueError) as e:
    print('진행 정보 갱신 대기:',type(e).__name__)
