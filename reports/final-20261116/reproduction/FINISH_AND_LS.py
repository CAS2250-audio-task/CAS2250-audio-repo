"""Finish only useful existing tails, then run one fresh label-smoothing arm.
No S01 training/evaluation and no result uploads. Standard-library launcher.
"""
import argparse,hashlib,importlib.util,json,os,shutil,subprocess,sys,time,urllib.request
from pathlib import Path
from datetime import datetime,timezone,timedelta

BASE='https://github.com/CAS2250-audio-task/CAS2250-audio-repo/releases/download/detector-diversity-v2-distributed-assets-v1/'
TRAINER_SHA='bb75031c74ded5b6b8d9ddc9cf84e7af99d685380b30ec2a0ef1f96efc5a40b2'
TRAIN_END=datetime.fromisoformat('2026-11-16T14:05:00+00:00').timestamp()
EVAL_END=datetime.fromisoformat('2026-11-16T14:20:00+00:00').timestamp()
DEFAULTS={0:('actual','mean_aux'),1:('actual','mean'),2:('off','mean'),3:('off','mean_aux')}

def read(path):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return {}

def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.nextstage.tmp');temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');temp.replace(path)

def alive(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
    except (OSError,IndexError):return False

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def discover():
    override=os.environ.get('DETECTOR_REPO')
    if override:return Path(override).expanduser().resolve()
    for path in sorted(Path('/tmp').glob('start-final-*-v3.py'),key=lambda p:p.stat().st_mtime,reverse=True):
        helper=load(path,'nextstage_start_helper');root=helper.find_root()
        if (root/'runs/final-classifier-2215/classifier-study-20261116/bank/metadata.json').is_file():return root
    for base in [Path.cwd(),Path.home(),Path('/workspace'),Path('/work'),Path('/data')]:
        if not base.is_dir():continue
        for directory,dirs,_ in os.walk(base):
            root=Path(directory)
            if (root/'runs/final-classifier-2215/classifier-study-20261116/bank/metadata.json').is_file():return root.resolve()
            dirs[:]=[n for n in dirs if not n.startswith('.') and n not in ['datasets','runs','reports','venv','env','node_modules','site-packages','__pycache__']] if len(root.relative_to(base).parts)<6 else []
    raise RuntimeError('Existing classifier study not found; set DETECTOR_REPO')

def choose_python(raw,gpu):
    for path in sorted(Path('/tmp').glob('start-final-*-v3.py'),key=lambda p:p.stat().st_mtime,reverse=True):
        helper=load(path,'nextstage_cuda_helper')
        return helper.cuda_python(raw,gpu)
    choices=[os.environ.get('DETECTOR_PYTHON'),str(raw/'.venv/bin/python'),sys.executable,'/root/CAS2250-resnet18-pretrain-100/.venv/bin/python','/opt/conda/bin/python']
    for candidate in choices:
        if candidate and Path(candidate).is_file():
            result=subprocess.run([candidate,'-c','import torch,torchaudio,numpy;assert torch.cuda.is_available()'],env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            if result.returncode==0:return candidate
    raise RuntimeError('No CUDA Python found; set DETECTOR_PYTHON')

def blockers(raw,gpu):
    run=raw/'runs/final-four-gpu-2225';active=[]
    for role in range(4):
        pidfile=run/f'role{role}_launcher.pid'
        try:pid=int(pidfile.read_text())
        except (OSError,ValueError):continue
        if not alive(pid) or read(run/f'role{role}_status.json').get('stage')=='complete':continue
        try:
            values=Path(f'/proc/{pid}/environ').read_bytes().split(b'\0')
            device=next((v.split(b'=',1)[1].decode() for v in values if v.startswith(b'CUDA_VISIBLE_DEVICES=')),'0')
        except OSError:device=gpu
        if device==gpu:active.append(dict(role=role,pid=pid))
    return active

def tail_for(role,work):
    if role==0:return ('mean_aux',work/'bank',work/'results/extended_actual/mean_aux','study.train')
    if role==1:return ('mean',work/'bank',work/'results/extended_actual/mean','study.train')
    if role==2:return ('mean',work/'bank_cv_S03',work/'results/cv/S03/mean','study.train_off')
    return None

def run_job(cmd,work,out,logpath,status,role,label,target=1000):
    if time.time()>=TRAIN_END:return False
    with logpath.open('ab') as log:child=subprocess.Popen(cmd,cwd=work,stdout=log,stderr=subprocess.STDOUT)
    while child.poll() is None:
        progress=read(out/'status.json')
        write(status,dict(stage='training',role=role,current=label,pid=os.getpid(),child_pid=child.pid,
            epoch=progress.get('epoch',0),target_epochs=target,planned_epochs=1000,epoch_seconds=progress.get('epoch_seconds'),
            current_eta_seconds=progress.get('eta_seconds'),training_stop_kst='23:05',evaluation_stop_kst='23:20'))
        if time.time()>=TRAIN_END:
            child.terminate()
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            return False
        time.sleep(2)
    if child.returncode:raise RuntimeError(label+' exited'+str(child.returncode)+'; log='+str(logpath))
    return True

def summarize(out,bank,label):
    state=read(out/'status.json');history=read(out/'history.json')
    history=history if isinstance(history,list) else []
    best=max(history,key=lambda r:r.get('zero_command10_f1',-1)) if history else {}
    meta=read(bank/'metadata.json')
    return dict(label=label,completed_epochs=history[-1]['epoch'] if history else state.get('epoch',0),
        selected_best_epoch=best.get('epoch'),noisy_F1_pct=100*best['zero_command10_f1'] if 'zero_command10_f1' in best else None,
        original_F1_pct=100*best['original']['command10_f1'] if best.get('original') else None,
        noise_only_FPR_pct=100*best['noise_only_fpr'] if best.get('noise_only_fpr') is not None else None,
        by_speaker=best.get('by_speaker'),train_speakers=meta.get('train_speakers'),validation_speakers=meta.get('validation_speakers'),
        checkpoint=str(out/'best.pt'),last_checkpoint=str(out/'last.pt'),bank=str(bank),
        note='Development validation; best checkpoint selected on these validation speakers. Not S01 test or independent outer-CV.')

def controller(args,raw):
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu,CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
    work=raw/'runs/final-classifier-2215/classifier-study-20261116';run=raw/'runs/final-four-gpu-2225'
    status=run/f'nextstage_role{args.role}_status.json';lock=run/f'nextstage_role{args.role}.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    summaries=[]
    try:
        while True:
            active=blockers(raw,args.gpu)
            if not active:break
            write(status,dict(stage='waiting_for_existing_training',role=args.role,active=active))
            if time.time()>=TRAIN_END:raise TimeoutError('Existing GPU work did not finish before23:05; nothing stopped')
            time.sleep(5)
        tail=tail_for(args.role,work)
        if tail:
            condition,bank,out,module=tail;state=read(out/'status.json');history=read(out/'history.json')
            epoch=history[-1]['epoch'] if isinstance(history,list) and history else state.get('epoch',0)
            if epoch<1000:
                assert (out/'last.pt').is_file(),'Missing tail checkpoint: '+str(out)
                cmd=[sys.executable,'-u','-m',module,'--condition',condition,'--bank',str(bank),'--output',str(out),
                    '--epochs','1000','--schedule-epochs','1000','--stop-after','1000','--resume']
                run_job(cmd,work,out,run/f'nextstage_role{args.role}_tail.log',status,args.role,'finish_existing_'+condition)
            if read(out/'status.json').get('epoch')==1000 and (out/'last.pt').is_file():
                snapshot=out/'fixed_epoch1000_last.pt'
                if not snapshot.exists():shutil.copy2(out/'last.pt',snapshot)
            summaries.append(summarize(out,bank,'existing_tail'))
        kind=args.bank_kind or DEFAULTS[args.role][0];condition=args.condition or DEFAULTS[args.role][1]
        bank=work/('bank' if kind=='actual' else 'bank_off')
        assert (bank/'COMPLETE.json').is_file(),'Missing completed bank: '+str(bank)
        assert set(read(bank/'metadata.json').get('train_speakers',[]))=={'S03','S04','S05'},'LS split changed; refusing to include S01 or new validation split'
        assert read(bank/'metadata.json').get('validation_speakers')==['S02'],'LS uses existing S02 validation, not S01'
        source=urllib.request.urlopen(BASE+'TRAIN_SMOOTHING.py',timeout=30).read()
        assert hashlib.sha256(source).hexdigest()==TRAINER_SHA,'Smoothing source SHA256 mismatch'
        compile(source,'train_smoothing.py','exec');target=work/'study/train_smoothing.py'
        target.write_bytes(source)
        token=str(args.epsilon).replace('.','p');out=work/'results/label_smoothing'/(kind+'_'+condition+'_eps'+token)
        cmd=[sys.executable,'-u','-m','study.train_smoothing','--condition',condition,'--bank',str(bank),'--output',str(out),
            '--epochs','1000','--schedule-epochs','1000','--stop-after','500','--input-mode',kind,'--label-smoothing',str(args.epsilon)]
        if (out/'last.pt').is_file():cmd.append('--resume')
        run_job(cmd,work,out,run/f'nextstage_role{args.role}_smoothing.log',status,args.role,'LS_'+kind+'_'+condition,target=500)
        if read(out/'status.json').get('epoch')==500 and (out/'last.pt').is_file():
            snapshot=out/'fixed_epoch500_last.pt'
            if not snapshot.exists():shutil.copy2(out/'last.pt',snapshot)
        summaries.append(summarize(out,bank,'label_smoothing_'+str(args.epsilon)))
        report=raw/'reports/final-nextstage';report.mkdir(parents=True,exist_ok=True)
        write(report/f'role{args.role}_summary.json',summaries)
        # S02 diagnostic only. The parent final evaluation owns S01 and SNR sweep.
        if args.evaluate and (out/'best.pt').is_file() and time.time()<EVAL_END:
            destination=report/f'role{args.role}'/'threeway';destination.parent.mkdir(parents=True,exist_ok=True)
            write(status,dict(stage='pipeline_evaluation',role=args.role,current=str(out/'best.pt')))
            cmd=[sys.executable,'-u','pipeline_eval.py','--condition',condition,'--checkpoint',str(out/'best.pt'),'--raw-root',str(raw),'--output',str(destination)]
            with (run/f'nextstage_role{args.role}_evaluation.log').open('ab') as log:
                child=subprocess.Popen(cmd,cwd=work,stdout=log,stderr=subprocess.STDOUT)
                try:child.wait(timeout=max(1,EVAL_END-time.time()))
                except subprocess.TimeoutExpired:child.terminate();child.wait(timeout=15)
        write(status,dict(stage='complete',role=args.role,summaries=summaries,training_stop_kst='23:05',evaluation_stop_kst='23:20',S01_evaluated=False))
        print('NEXTSTAGE_SUMMARY',json.dumps(summaries,ensure_ascii=False),flush=True)
    except BaseException as exc:
        write(status,dict(stage='failed',role=args.role,error=repr(exc),summaries=summaries));raise
    finally:
        if lock.exists() and lock.read_text().strip()==str(os.getpid()):lock.unlink()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--role',type=int,choices=range(4),required=True)
    parser.add_argument('--gpu',default='0');parser.add_argument('--condition',choices=['mean','mean_aux'])
    parser.add_argument('--bank-kind',choices=['actual','off']);parser.add_argument('--epsilon',type=float,default=.1)
    parser.add_argument('--raw-root',type=Path);parser.add_argument('--controller',action='store_true');parser.add_argument('--evaluate',action='store_true')
    args=parser.parse_args()
    if not 0<=args.epsilon<=1:parser.error('--epsilon must be between0 and1')
    raw=args.raw_root.resolve() if args.raw_root else discover()
    if args.controller:return controller(args,raw)
    assert time.time()<TRAIN_END,'Training cutoff23:05 already passed'
    work=raw/'runs/final-classifier-2215/classifier-study-20261116';run=raw/'runs/final-four-gpu-2225';run.mkdir(parents=True,exist_ok=True)
    lock=run/f'nextstage_role{args.role}.lock'
    if lock.exists():
        pid=int(lock.read_text())
        if alive(pid):print('NEXTSTAGE ALREADY RUNNING',pid);return
        lock.unlink()
    python=choose_python(raw,args.gpu)
    for name in ['baseline200','front5','soft_time','extended_actual','command_ce','off_trained']:
        (raw/'reports/final-four-gpu-2225'/name).mkdir(parents=True,exist_ok=True)
    cmd=[python,'-u',str(Path(__file__).resolve()),'--controller','--role',str(args.role),'--raw-root',str(raw),'--gpu',args.gpu,'--epsilon',str(args.epsilon)]
    if args.condition:cmd+=['--condition',args.condition]
    if args.bank_kind:cmd+=['--bank-kind',args.bank_kind]
    if args.evaluate:cmd+=['--evaluate']
    with (run/f'nextstage_role{args.role}_launcher.log').open('ab') as log:
        process=subprocess.Popen(cmd,cwd=work,env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu,CUBLAS_WORKSPACE_CONFIG=':4096:8'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (run/f'nextstage_role{args.role}_launcher.pid').write_text(str(process.pid)+'\n')
    time.sleep(2)
    if process.poll() is not None and process.returncode:
        raise RuntimeError('Nextstage failed; read '+str(run/f'nextstage_role{args.role}_launcher.log'))
    print('NEXTSTAGE STARTED ROLE',args.role,'PID',process.pid,'STATUS',run/f'nextstage_role{args.role}_status.json',flush=True)
    print('Useful tails to1000; NEW LS eps0.1 to500 (fixed1000 LR schedule); train cutoff23:05, final eval deadline23:20 KST. No S01 access or uploads.',flush=True)

if __name__=='__main__':main()
