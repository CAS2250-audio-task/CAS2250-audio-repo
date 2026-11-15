"""Three new variants,2 concurrent workers; reuse the completed identical baseline."""
from pathlib import Path
import concurrent.futures
import datetime
import hashlib
import json
import os
import subprocess
import sys
import threading

from experiments.local_features.models import variant_description

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/compact-local-features'
REPORT=ROOT/'reports/compact-local-features'
BASE=Path('/root/CAS2250-compact-contrastive-100')
BASE_NAME='compact-noise-speaker-spk03-e1000'
VARIANTS=('short','local','both')
LOCK=threading.Lock();STATE={}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_state():
    p=OUT/'status.tmp';p.write_text(json.dumps(STATE,indent=2)+'\n');p.replace(OUT/'status.json')


def worker(variant):
    name=f'compact-local-{variant}-spk03-e1000'
    command=[sys.executable,'-u','-m','experiments.local_features.run','--variant',variant,
        '--objective','paired_supcon','--positive-mode','same_command','--folds','spk03',
        '--epochs','1000','--output',str(ROOT/'runs'/name),'--supcon-weight','0.1','--temperature','0.1']
    with (OUT/(variant+'.log')).open('a',buffering=1) as log:
        child=subprocess.Popen(command,cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
        with LOCK:
            STATE['variants'][variant]=dict(status='running',pid=child.pid,command=command,
                progress=str(ROOT/'runs'/name/'progress.json'),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            save_state()
        code=child.wait()
    with LOCK:
        STATE['variants'][variant].update(status='complete' if code==0 else 'failed',returncode=code,
            finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save_state()
    if code:raise RuntimeError(f'{variant} exited{code}; see{OUT/(variant+".log")}')


def summarize():
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=('cnn_mean','cnn_pooled_query','cnn_self_cls')
    rows=[];checked=0;traces=0;all_history=[]
    for variant in ('baseline',)+VARIANTS:
        origin=BASE if variant=='baseline' else ROOT
        name=BASE_NAME if variant=='baseline' else f'compact-local-{variant}-spk03-e1000'
        study=origin/'runs'/name/'spk03'
        summary=json.loads((study/'summary.json').read_text())
        protocol=json.loads((origin/'reports'/name/'protocol.json').read_text())
        old=json.loads((BASE/'reports'/BASE_NAME/'protocol.json').read_text())
        assert protocol['audio_sha256']==old['audio_sha256'] and protocol['sources']==old['sources']
        assert protocol['epochs']==1000 and protocol['folds']==['spk03']
        history=pd.DataFrame(json.loads((study/'history.json').read_text())).assign(variant=variant)
        assert len(history)==3000 and history.epoch.max()==1000
        all_history.append(history)
        for model in names:
            record=next(r for r in summary if r['model']==model)
            p=pd.read_csv(study/(model+'_best_predictions.csv.gz'))
            def f1(frame):
                cm=np.bincount(frame.true_idx.to_numpy()*11+frame.pred_idx.to_numpy(),minlength=121).reshape(11,11)
                den=cm.sum(0)+cm.sum(1)
                return np.divide(2.*cm.diagonal(),den,out=np.zeros(11),where=den!=0).mean()
            original=p[p.kind=='original'];noisy=p[p.kind=='mixed_0db']
            clean=f1(original);zero=float(np.mean([f1(g) for _,g in noisy.groupby(['noise_id','draw'])]))
            assert abs(clean-record['best']['original_f1'])<1e-6 and abs(zero-record['best']['zero_f1'])<1e-6
            bad=noisy.true_idx.ne(noisy.pred_idx)
            def confusion(a,b):return int((noisy.true_idx.eq(a)&noisy.pred_idx.eq(b)).sum())
            rows.append(dict(variant=variant,model=model,parameters=record['parameters'],best_epoch=record['best_epoch'],
                original_f1=clean,zero_f1=zero,last_zero_f1=record['last']['zero_f1'],noise_only_fpr=record['noise_only_fpr'],
                original_errors=int(original.true_idx.ne(original.pred_idx).sum()),noise_errors=int(bad.sum()),
                restart_to_start=confusion(9,6),forward_to_stop=confusion(2,1),stop_to_forward=confusion(1,2),
                right_to_left=confusion(5,4),left_to_right=confusion(4,5)))
            checked+=1
        if variant!='baseline':
            for epoch in range(1,1001):
                assert digest(study/f'augmentation_{epoch:03d}.json')==digest(BASE/'runs'/BASE_NAME/'spk03'/f'augmentation_{epoch:03d}.json'),(variant,epoch)
                traces+=1
    result=pd.DataFrame(rows);result.to_csv(REPORT/'comparison.csv',index=False)
    all_history=pd.concat(all_history,ignore_index=True);all_history.to_csv(REPORT/'epoch_histories.csv.gz',index=False)
    fig,axes=plt.subplots(1,3,figsize=(15,4),sharey=True)
    for ax,model in zip(axes,names):
        for variant in ('baseline',)+VARIANTS:
            part=all_history[(all_history.model==model)&(all_history.variant==variant)].sort_values('epoch')
            ax.plot(part.epoch,part.zero_f1.rolling(20,min_periods=1).mean()*100,label=variant)
        ax.set(title=model,xlabel='Epoch',ylabel='0dB macro F1 (%)');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(REPORT/'learning_curves.png',dpi=160);plt.close(fig)
    lines=['# CNN 시간범위·중간 특징 전달 비교 완료','',
        '화자03 검증, 다른 개발화자3명 학습. 소음+화자 대조, seed42,각1000epoch cosine. 기준baseline은 기존 완료 실행을 재사용했고 새3조건만 학습했다.', '',
        '| 조건 | 모델 | 파라미터 | 최고epoch | 원본F1 | 0dB F1 | 종료점0dB F1 | 잡음단독 오반응 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f'| {row["variant"]} | {row["model"]} | {row["parameters"]:,} | {row["best_epoch"]} | {row["original_f1"]*100:.2f}% | {row["zero_f1"]*100:.2f}% | {row["last_zero_f1"]*100:.2f}% | {row["noise_only_fpr"]*100:.2f}% |')
    lines += ['',
        'short=최종 시간범위1.105→0.625초. local=약0.245초 중간 특징을64→32 zero-init residual로 최종특징에 더함. both=둘다. 모든경로에서32차원 readout입력을 유지했다. 파라미터가약1~3%변하므로 완전히동일한용량의실험은아니다.',
        '', '모든조건의원본·평가소음·학습순서·포먼트·소음/crop/SNR을맞췄다. 선택checkpoint는0dB draw0 최고, 최종표는8소음×3draw평균이다. 같은개발검증에서선택했고 최종평가에도draw0가포함된다. 독립test결과로해석하지않는다.',
        '', '![학습곡선](learning_curves.png)', '',
        '원본예측에서최종F1을재계산했고 새3조건의3000epoch 증강trace가기준과같음을확인했다. 전체조건/모델별명령어혼동은 comparison.csv에저장했다.',
        '', '[비교표](comparison.csv) · [전체epoch](epoch_histories.csv.gz) · [감사](completion_audit.json) · [사전검사](preflight.json) · [고정설정](launch_protocol.json)']
    (REPORT/'README.md').write_text('\n'.join(lines)+'\n')
    (REPORT/'completion_audit.json').write_text(json.dumps(dict(prediction_sets_recomputed=checked,shared_trace_pairs_verified=traces,
        newly_trained_model_epochs=9000,reused_baseline_model_epochs=3000,test01_audio_read=False),indent=2)+'\n')


def main():
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    protocol=dict(seed=42,speaker='spk03',epochs=1000,new_variants=list(VARIANTS),all_variants=[variant_description(v) for v in ('baseline',)+VARIANTS],
        objective='CE+0.1SupCon,same_command positive mode,temperature0.1; no other training changes',
        baseline_root=str(BASE),baseline_protocol_sha256=digest(BASE/'reports'/BASE_NAME/'protocol.json'),
        baseline_summary_sha256=digest(BASE/'runs'/BASE_NAME/'spk03/summary.json'),
        baseline_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=BASE,text=True).strip(),
        new_source_sha256={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'experiments/local_features').glob('*.py')},
        workers=2,planned_new_model_epochs=9000,reused_baseline_model_epochs=3000)
    destination=REPORT/'launch_protocol.json'
    if destination.exists():
        previous=json.loads(destination.read_text())
        protocol['baseline_git_commit']=previous['baseline_git_commit']
        assert previous==protocol,'Launch protocol changed'
    destination.write_text(json.dumps(protocol,indent=2)+'\n')
    STATE.update(status='running',supervisor_pid=os.getpid(),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        variants={v:dict(status='queued') for v in VARIANTS})
    save_state()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(worker,v) for v in VARIANTS]
            for future in concurrent.futures.as_completed(futures):future.result()
        summarize()
        with LOCK:
            STATE.update(status='complete',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save_state()
    except Exception as error:
        with LOCK:STATE.update(status='failed',error=str(error));save_state()
        raise


if __name__=='__main__':main()
