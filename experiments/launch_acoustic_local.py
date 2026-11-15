"""Detached single-group supervisor, final score replay and portable report export."""
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
NAME='compact-acoustic-local-spk03-e1000'
OUT=ROOT/'runs/compact-acoustic-local-v1'
REPORT=ROOT/'reports/compact-acoustic-local-v1'


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def summarize():
    import numpy as np
    import pandas as pd
    name=NAME;folder=ROOT/'runs'/name/'spk03';report=ROOT/'reports'/name
    for model in ['cnn_mean','cnn_pooled_query','cnn_self_cls']:
        for suffix in ['_best_predictions.csv.gz','_final_metrics.json']:
            shutil.copyfile(folder/(model+suffix),report/(model+suffix))
    history=pd.DataFrame(json.loads((folder/'history.json').read_text()))
    assert len(history)==3000
    history.to_csv(REPORT/'epoch_histories.csv.gz',index=False)
    rows=[]
    for condition,source in [('local',ROOT/'reports/compact-local-local-spk03-e1000'),('local_aux_set3',report)]:
        for record in json.loads((source/'summary.json').read_text()):
            model=record['model'];p=pd.read_csv(source/(model+'_best_predictions.csv.gz'));assert len(p)==5075
            noisy=p[p.kind=='mixed_0db'];scores=[]
            for _,g in noisy.groupby(['noise_id','draw']):
                cm=np.bincount(g.true_idx.to_numpy()*11+g.pred_idx.to_numpy(),minlength=121).reshape(11,11)
                den=cm.sum(0)+cm.sum(1);scores.append(np.divide(2*cm.diagonal(),den,out=np.zeros(11),where=den!=0).mean())
            assert abs(np.mean(scores)-record['best']['zero_f1'])<1e-6
            if condition=='local_aux_set3':
                h=history[history.model==model];assert len(h)==1000 and int(h.loc[h.zero_f1.idxmax(),'epoch'])==record['best_epoch']
            rows.append(dict(condition=condition,model=model,parameters=record['parameters'],best_epoch=record['best_epoch'],
                original_f1=record['best']['original_f1'],zero_f1=record['best']['zero_f1'],last_zero_f1=record['last']['zero_f1'],noise_only_fpr=record['noise_only_fpr'],noise_errors=int(noisy.true_idx.ne(noisy.pred_idx).sum())))
    pd.DataFrame(rows).to_csv(REPORT/'comparison.csv',index=False)
    (REPORT/'completion_audit.json').write_text(json.dumps(dict(prediction_sets_recomputed=6,model_epochs=3000,baseline_reused=True,test01_audio_read=False),indent=2)+'\n')


def main():
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    state_path=OUT/'status.json'
    if state_path.exists():raise RuntimeError('Existing supervisor status; inspect before starting duplicate')
    state=dict(status='starting',supervisor_pid=os.getpid(),started_utc=now(),epochs=1000,new_model_epochs=3000,variant='local',auxiliary='legacy set3')
    def save():
        tmp=state_path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(state_path)
    save()
    try:
        command=[sys.executable,'-u','-m','experiments.acoustic_local.run','--aux-workers','4',
            '--objective','paired_supcon','--positive-mode','same_command','--supcon-weight','0.1','--temperature','0.1',
            '--folds','spk03','--epochs','1000','--output',str(ROOT/'runs'/NAME)]
        with (OUT/'training.log').open('a',buffering=1) as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            state.update(status='running',training_pid=child.pid,command=command,progress=str(ROOT/'runs'/NAME/'progress.json'));save()
            code=child.wait()
        state['returncode']=code
        if code:raise RuntimeError(f'Training exited{code}; see{OUT}/training.log')
        state.update(status='summarizing');save();summarize()
        state.update(status='complete',finished_utc=now());save()
    except Exception as error:
        state.update(status='failed',error=str(error),finished_utc=now());save();raise


if __name__=='__main__':main()
