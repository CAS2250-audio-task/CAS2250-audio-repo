"""Launch the two authorized contrastive conditions; preserve fixed epoch budgets."""
import concurrent.futures
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/compact-contrastive-launch'
REPORT=ROOT/'reports/compact-contrastive-launch'
LOCK=threading.Lock()
STATE={}


def write_state():
    temporary=OUT/'status.tmp'
    temporary.write_text(json.dumps(STATE,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(OUT/'status.json')


def worker(condition,mode):
    # Requested held-out speaker03 receives 1000 epochs; speaker02 receives100.
    for speaker,epochs in [('spk03',1000),('spk02',100)]:
        name=f'compact-{condition}-{speaker}-e{epochs}'
        command=[sys.executable,'-u','-m','experiments.compact_pooling.run',
                 '--objective','paired_supcon','--positive-mode',mode,
                 '--folds',speaker,'--epochs',str(epochs),
                 '--output',str(ROOT/'runs'/name),
                 '--supcon-weight','0.1','--temperature','0.1']
        with (OUT/f'{condition}.log').open('a',buffering=1) as log:
            child=subprocess.Popen(command,cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
            with LOCK:
                STATE[condition]=dict(status='running',pid=child.pid,speaker=speaker,epochs=epochs,
                        command=command,progress=str(ROOT/'runs'/name/'progress.json'))
                write_state()
            code=child.wait()
        with LOCK:
            STATE[condition]['returncode']=code
            STATE[condition]['status']='fold_complete' if code==0 else 'failed'
            write_state()
        if code:raise RuntimeError(f'{condition}/{speaker} exited{code}; see log')
    with LOCK:
        STATE[condition]['status']='complete'
        write_state()


def summarize():
    import numpy as np
    import pandas as pd
    rows=[];checked=0
    for condition in ('noise','noise-speaker'):
        for speaker,epochs in [('spk03',1000),('spk02',100)]:
            name=f'compact-{condition}-{speaker}-e{epochs}'
            summary=json.loads((ROOT/'reports'/name/'summary.json').read_text())
            for row in summary:
                path=ROOT/'runs'/name/speaker/(row['model']+'_best_predictions.csv.gz')
                frame=pd.read_csv(path)
                def f1(part):
                    cm=np.bincount(part.true_idx.to_numpy()*11+part.pred_idx.to_numpy(),minlength=121).reshape(11,11)
                    denom=cm.sum(0)+cm.sum(1)
                    return np.divide(2.*cm.diagonal(),denom,out=np.zeros(11),where=denom!=0).mean()
                original=f1(frame[frame.kind=='original'])
                noisy=np.mean([f1(part) for _,part in frame[frame.kind=='mixed_0db'].groupby(['noise_id','draw'])])
                assert abs(original-row['best']['original_f1'])<1e-6
                assert abs(noisy-row['best']['zero_f1'])<1e-6
                checked+=1
                rows.append(dict(condition=condition,speaker=speaker,epochs=epochs,model=row['model'],
                    parameters=row['parameters'],best_epoch=row['best_epoch'],
                    original_f1=original,zero_f1=noisy,last_zero_f1=row['last']['zero_f1'],noise_only_fpr=row['noise_only_fpr']))
    pd.DataFrame(rows).to_csv(REPORT/'comparison.csv',index=False)
    (REPORT/'result_checks.json').write_text(json.dumps(dict(selected_predictions_recomputed=checked,test01_audio_read=False),indent=2)+'\n')
    lines=['# Compact CNN 소음·화자 대조학습 완료','',
           '동일 모델3종에서 소음만 대조 / 같은 명령어의 다른 학습화자까지 대조. 화자03 1000epoch, 화자02 100epoch. 초기화와 입력·캐시는 같고 두 조건의 positive 구성만 다르다.',
           '', '| 조건 | 검증화자 | 모델 | 선택epoch | 원본F1 | 0dB F1 | 종료점0dB F1 | 잡음단독 오반응 |',
           '|---|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["condition"]} | {r["speaker"]} | {r["model"]} | {r["best_epoch"]} | {r["original_f1"]*100:.2f}% | {r["zero_f1"]*100:.2f}% | {r["last_zero_f1"]*100:.2f}% | {r["noise_only_fpr"]*100:.2f}% |')
    lines += ['', '개발 두 화자·seed42의 결과다. 화자별 epoch 예산이 다르므로 동일 예산의 CV 평균으로 표현하지 않는다. 동일0dB draw0로 최고epoch를 선택하고 최종3draw에도 포함했다. 독립test01은 읽지 않았다.',
              '', '사전학습 없음. 캐시는 compact-pooling-v1-assets 릴리스와 해시 일치. 현재 서버 torch2.6/CUDA12.4이며 원 서버torch2.3/CUDA12.1과 달라 두 서버의 차이를 순수 로스 효과로 해석하지 않는다. 이 서버의 두 대조조건끼리 같은 환경에서 비교한다.',
              '', '소음만 대조 조건은 같은 녹음의 두 뷰만positive로 두고, 다른 녹음의 같은 명령어는negative로 쓰지 않고 제외한다. 화자 포함 조건은 같은 명령어의 다른 녹음·학습화자도positive다. silence는CE만 적용한다.',
              '', '실행별 원본 보고서는 reports/compact-{noise,noise-speaker}-{spk03-e1000,spk02-e100}/, 체크포인트·epoch기록·증강추적은 대응하는 runs/ 경로에 있다.']
    (REPORT/'README.md').write_text('\n'.join(lines)+'\n')


def main():
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    STATE.update(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),supervisor_pid=os.getpid(),
                 status='running',gpu='shared RTX3090; two condition workers, each CPU2threads',
                 planned_model_epochs=6600,reference='origin/compact-pooling-v1 d0f8531; exact release cache',
                 noise=dict(status='queued'),**{'noise-speaker':dict(status='queued')})
    write_state()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(worker,'noise','same_recording'),pool.submit(worker,'noise-speaker','same_command')]
            for future in concurrent.futures.as_completed(futures):future.result()
        summarize()
        with LOCK:STATE['status']='complete';write_state()
    except Exception as error:
        with LOCK:STATE['status']='failed';STATE['error']=str(error);write_state()
        raise


if __name__=='__main__':main()
