import csv,datetime,json,os,signal,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/quick-frequency40'
REPORT=ROOT/'reports/quick-frequency40'
ORDER=['low8','low16','freq_avg','freq_band2','freq_capacity','low_axis8']


def save(s):
    p=OUT/'status.tmp';p.write_text(json.dumps(s,indent=2)+'\n');p.replace(OUT/'status.json')


def summarize():
    import pandas as pd
    import numpy as np
    rows=[]
    for variant in ORDER:
        study=f'quick-frequency40-{variant}-spk03-e300';folder=ROOT/'reports'/study
        if not (folder/'summary.json').exists():continue
        r=json.loads((folder/'summary.json').read_text())[0]
        p=pd.read_csv(ROOT/'runs'/study/'spk03/cnn_pooled_query_best_predictions.csv.gz')
        values=[]
        for _,g in p[p.kind=='mixed_0db'].groupby(['noise_id','draw']):
            cm=np.bincount(g.true_idx.to_numpy()*11+g.pred_idx.to_numpy(),minlength=121).reshape(11,11)
            den=cm.sum(0)+cm.sum(1);values.append(np.divide(2.*cm.diagonal(),den,out=np.zeros(11),where=den!=0))
        f=np.stack(values)
        assert abs(f.mean()-r['best']['zero_f1'])<1e-6
        rows.append(dict(variant=variant,parameters=r['parameters'],best_epoch=r['best_epoch'],
            original_f1=r['best']['original_f1'],zero_f1=r['best']['zero_f1'],
            command10_f1=f[:,:10].mean(),silence_f1=f[:,10].mean(),noise_only_fpr=r['noise_only_fpr']))
    if rows:
        with (REPORT/'comparison.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True,parents=True);REPORT.mkdir(exist_ok=True,parents=True)
    assert not (OUT/'status.json').exists(),'Duplicate supervisor'
    # Deadline starts at preparation, not at the end of setup; leave a safety margin.
    deadline=(ROOT/'experiments/quick_frequency40/__init__.py').stat().st_mtime+36*60
    s=dict(status='running',pid=os.getpid(),deadline_utc=datetime.datetime.fromtimestamp(deadline,datetime.timezone.utc).isoformat(),
        epochs=300,cosine_schedule_epochs=1000,groups={v:dict(status='queued') for v in ORDER})
    save(s)
    try:
        for v in ORDER:
            remaining=deadline-time.time()
            if remaining<=0:s['status']='time_limit';save(s);return
            study=f'quick-frequency40-{v}-spk03-e300'
            cmd=[sys.executable,'-u','-m','experiments.quick_frequency40.run','--variant',v,
                '--models','cnn_pooled_query','--objective','paired_supcon','--positive-mode','same_command',
                '--supcon-weight','0.1','--temperature','0.1','--folds','spk03',
                '--epochs','300','--schedule-epochs','1000','--output',str(ROOT/'runs'/study)]
            with (OUT/(v+'.log')).open('a',buffering=1) as log:
                child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                s['groups'][v].update(status='running',pid=child.pid,command=cmd,progress=str(ROOT/'runs'/study/'progress.json'));save(s)
                try:code=child.wait(timeout=max(1,deadline-time.time()))
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=5)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    s['groups'][v]['status']='time_limit';s['status']='time_limit';save(s);summarize();return
            s['groups'][v].update(status='complete' if code==0 else 'failed',returncode=code);save(s)
            if code:raise RuntimeError(v+' failed; inspect log')
            summarize()
        s['status']='complete';save(s)
    except Exception as e:s.update(status='failed',error=str(e));save(s);raise


if __name__=='__main__':main()
