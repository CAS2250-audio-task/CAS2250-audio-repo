import argparse,datetime,json,os,signal,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/low-frequency-grid'
REPORT=ROOT/'reports/low-frequency-grid'
JOBS=[(arch,mode,'spk03') for arch in ['axis','basic','frequency','frequency_attn','axis_attn'] for mode in ['standard','lowfocus']]
JOBS += [('basic','standard','spk02')]


def save(state):
    p=OUT/'status.tmp';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(OUT/'status.json')


def summarize():
    import pandas as pd
    import numpy as np
    rows=[]
    for arch,mode,fold in JOBS:
        name=f'lowfreq-{arch}-{mode}-{fold}-e1000';report=ROOT/'reports'/name
        if not (report/'summary.json').exists():continue
        row=json.loads((report/'summary.json').read_text())[0]
        p=pd.read_csv(ROOT/'runs'/name/fold/'cnn_pooled_query_best_predictions.csv.gz')
        f1=[]
        for _,g in p[p.kind=='mixed_0db'].groupby(['noise_id','draw']):
            cm=np.bincount(g.true_idx.to_numpy()*11+g.pred_idx.to_numpy(),minlength=121).reshape(11,11)
            den=cm.sum(0)+cm.sum(1)
            f1.append(np.divide(2.*cm.diagonal(),den,out=np.zeros(11),where=den!=0))
        f=np.stack(f1);assert abs(f.mean()-row['best']['zero_f1'])<1e-6
        rows.append(dict(architecture=arch,frontend=mode,fold=fold,parameters=row['parameters'],
            completed_epochs=row['completed_epochs'],planned_epochs=row['planned_epochs'],
            best_epoch=row['best_epoch'],original_f1=row['best']['original_f1'],zero_f1=row['best']['zero_f1'],
            command10_f1=f[:,:10].mean(),silence_f1=f[:,10].mean(),noise_only_fpr=row['noise_only_fpr']))
    if rows:pd.DataFrame(rows).to_csv(REPORT/'comparison.csv',index=False)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stop-after',type=int,default=500)
    parser.add_argument('--limit-minutes',type=float,default=180)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    assert 0<args.stop_after<=1000 and args.limit_minutes>0
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    if (OUT/'status.json').exists():
        old=json.loads((OUT/'status.json').read_text())
        assert args.resume and old['status']!='running','Duplicate supervisor; inspect existing job'
        (OUT/f'previous-status-{int(time.time())}.json').write_text(json.dumps(old,indent=2)+'\n')
    start=time.time();deadline=start+args.limit_minutes*60
    state=dict(status='running',supervisor_pid=os.getpid(),started_utc=datetime.datetime.fromtimestamp(start,datetime.timezone.utc).isoformat(),
        deadline_utc=datetime.datetime.fromtimestamp(deadline,datetime.timezone.utc).isoformat(),epochs=args.stop_after,planned_epochs=1000,groups={})
    for arch,mode,fold in JOBS:
        key=f'{arch}-{mode}-{fold}';name=f'lowfreq-{key}-e1000'
        state['groups'][key]=dict(status='queued',architecture=arch,frontend=mode,fold=fold,
            progress=str(ROOT/'runs'/name/'progress.json'))
    save(state)
    try:
        for arch,mode,fold in JOBS:
            key=f'{arch}-{mode}-{fold}';name=f'lowfreq-{key}-e1000'
            if time.time()>=deadline:state['status']='time_limit';save(state);return
            cmd=[sys.executable,'-u','-m','experiments.low_frequency_grid.run',
                '--architecture',arch,'--frontend',mode,'--folds',fold,
                '--objective','paired_supcon','--positive-mode','same_command',
                '--supcon-weight','0.1','--temperature','0.1',
                '--epochs','1000','--schedule-epochs','1000','--stop-after',str(args.stop_after),'--output',str(ROOT/'runs'/name)]
            with (OUT/(key+'.log')).open('a',buffering=1) as log:
                child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                state['groups'][key].update(status='running',pid=child.pid,command=cmd);save(state)
                try:code=child.wait(timeout=max(1,deadline-time.time()))
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=5)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    state['groups'][key]['status']='time_limit';state['status']='time_limit';save(state);summarize();return
            state['groups'][key].update(status='complete' if code==0 else 'failed',returncode=code);save(state)
            if code:raise RuntimeError(key+' failed; see log')
            summarize()
        state['status']='complete';save(state)
    except Exception as error:state.update(status='failed',error=str(error));save(state);raise


if __name__=='__main__':main()
