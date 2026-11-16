"""Queue frozen deployable-pool selection, freeze winner, then run S01 ambient-noise test.
No training edits, no training signals, and no model/results publication.
"""
import argparse,hashlib,json,os,subprocess,sys,time,urllib.request
from pathlib import Path
from datetime import datetime

END=datetime.fromisoformat('2026-11-16T14:20:00+00:00').timestamp()
BASE='https://github.com/CAS2250-audio-task/CAS2250-audio-repo/releases/download/detector-diversity-v2-distributed-assets-v1/'
HELPERS={'pipeline_eval_extended.py':'7f1ca8bdda34efe54bd12a6615bee5cd51c8075856d69e2ccf2de118932a9268','EVAL_AMBIENT_TEST.py':'82f36e3c0272fa81204e792810342ad4b4a8651c78dccf473a19928c8a802839','EVAL_SNR_TEST.py':'d65cca75fe7aad77ea57a4e2818aedcd8a54ea79a5b0feea5fe17b3f96a7680d'}
RULE=dict(primary='maximum S02 dev active-power0dB mean command10F1',tie_tolerance=1e-12,
          tie1='minimum noise-only FP files',tie2='maximum original command10F1',tie3='lexicographic candidate_id',
          eligible='completed baseline500/baseline1000 and LS500 deployment pool; diagnostic200 excluded; report budgets separately',global_selection='same DEV rule across host-local winners; never use S01 results')

class BudgetExpired(RuntimeError):pass

def read(path):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return {}

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.finaleval.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1<<20),b''):h.update(part)
    return h.hexdigest()

def alive(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
    except (OSError,IndexError):return False

def discover(explicit=None):
    if explicit or os.environ.get('DETECTOR_REPO'):return Path(explicit or os.environ['DETECTOR_REPO']).expanduser().resolve()
    seen=set()
    for base in [Path.cwd(),Path.home(),Path('/workspace'),Path('/work'),Path('/data')]:
        if not base.is_dir():continue
        for directory,dirs,_ in os.walk(base):
            p=Path(directory)
            if p in seen:dirs[:]=[];continue
            seen.add(p)
            if (p/'runs/final-classifier-2215/classifier-study-20261116/bank/metadata.json').is_file():return p.resolve()
            dirs[:]=[n for n in dirs if not n.startswith('.') and n not in ['datasets','runs','reports','venv','env','node_modules','site-packages','__pycache__']] if len(p.relative_to(base).parts)<6 else []
    raise RuntimeError('Prepared classifier repository not found; set DETECTOR_REPO')

def choose_python(raw,role,gpu="0"):
    run=raw/'runs/final-four-gpu-2225';choices=[os.environ.get('DETECTOR_PYTHON')]
    for file in [run/f'nextstage_role{role}_launcher.pid',run/f'role{role}_launcher.pid']:
        try:
            pid=int(file.read_text());args=Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
            if alive(pid) and args:choices.append(args[0])
        except (OSError,ValueError):pass
    choices.extend([str(raw/'.venv/bin/python'),'/root/CAS2250-resnet18-pretrain-100/.venv/bin/python',sys.executable])
    for candidate in dict.fromkeys(choices):
        if not candidate or not Path(candidate).is_file():continue
        if time.time()>=END:raise BudgetExpired('Deadline before CUDA Python probe')
        try:
            probe=subprocess.run([candidate,'-c','import torch;assert torch.cuda.is_available()'],env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=max(1,min(12,END-time.time())))
            if probe.returncode==0:return candidate
        except subprocess.TimeoutExpired:continue
    return None

def ensure_helpers(directory):
    directory.mkdir(parents=True,exist_ok=True)
    for name,expected in HELPERS.items():
        target=directory/name
        if target.is_file() and sha(target)==expected:continue
        if time.time()>=END:raise BudgetExpired('Deadline before helper download')
        data=urllib.request.urlopen(BASE+name+'?sha256='+expected,timeout=max(1,min(30,END-time.time()))).read()
        assert hashlib.sha256(data).hexdigest()==expected,'Helper source SHA mismatch: '+name
        compile(data,name,'exec');target.write_bytes(data)
    return directory

def inventory(raw):
    work=raw/'runs/final-classifier-2215/classifier-study-20261116';classifiers=[];missing=[]
    for condition in ['mean','mean_aux']:
        for mode,family in [('actual','extended_actual'),('off','off')]:
            folder=work/'results'/family/condition
            for budget in [500,1000]:
                milestone=folder/'milestones'/f'e{budget}'
                p=milestone/'best.pt'
                if p.is_file() and read(milestone/'status.json').get('epoch')==budget:
                    classifiers.append(dict(id=f'baseline{budget}_'+mode+'_'+condition,condition=condition,mode=mode,budget=budget,checkpoint=str(p),eligible=True))
                elif read(folder/'status.json').get('epoch')==budget and (folder/'best.pt').is_file():
                    classifiers.append(dict(id=f'baseline{budget}_'+mode+'_'+condition,condition=condition,mode=mode,budget=budget,checkpoint=str(folder/'best.pt'),eligible=True))
            for ls in sorted((work/'results/label_smoothing').glob(mode+'_'+condition+'_eps*')):
                if read(ls/'status.json').get('epoch')==500 and (ls/'best.pt').is_file():
                    classifiers.append(dict(id='LS500_'+ls.name,condition=condition,mode=mode,budget=500,checkpoint=str(ls/'best.pt'),eligible=True))
        if not any(x['condition']==condition and x['mode']=='actual' for x in classifiers):
            p=work/'frozen200'/condition/'best.pt'
            if p.is_file():classifiers.append(dict(id='diagnostic200_actual_'+condition,condition=condition,mode='actual',budget=200,checkpoint=str(p),eligible=False))
    paths=[('speaker_BCE',raw/'runs/detector-front-aux-speaker-v1/speaker/best.pt'),
           ('auxiliary',raw/'runs/detector-front-aux-speaker-v1/auxiliary/best.pt'),
           ('auxiliary_speaker',raw/'runs/detector-front-aux-speaker-v1/auxiliary_speaker/best.pt'),
           ('front5',raw/'runs/detector-final-2215-v2/front5/best.pt'),
           ('soft_time',raw/'runs/detector-final-2215-v2/soft_time/best.pt'),
           ('command_ce',raw/'runs/detector-final-2215-v2/command_ce/best.pt'),
           ('reference152',work/'reference_detector.pt')]
    if not (work/'reference_detector.pt').is_file():paths.append(('reference',raw/'runs/detector-front-aux-speaker-v1/reference/best.pt'))
    detectors=[];seen=set()
    for name,p in paths:
        if not p.is_file():missing.append(name);continue
        digest=sha(p)
        if digest not in seen:detectors.append(dict(name=name,path=str(p),sha256=digest));seen.add(digest)
    candidates=[]
    for c in classifiers:
        c['checkpoint_selection']='best_within_completed_budget'
        c['checkpoint_sha256']=sha(c['checkpoint'])
        if c['mode']=='off':candidates.append(dict(c,detector=None))
        else:
            for d in detectors:candidates.append(dict(c,id=c['id']+'__'+d['name'],detector=d))
    # Deployable completed budgets first. Frozen200 is diagnostic only and cannot win.

    return sorted(candidates,key=lambda c:(not c['eligible'],c['id'])),missing

def pick(rows):
    best=None
    for row in rows:
        if not row['candidate']['eligible']:continue
        if best is None or row['dev_f1']>best['dev_f1']+1e-12:best=row;continue
        if abs(row['dev_f1']-best['dev_f1'])<=1e-12:
            a=(row['noise_fp'],-row['original_f1'],row['candidate']['id']);b=(best['noise_fp'],-best['original_f1'],best['candidate']['id'])
            if a<b:best=row
    return best

def run_child(command,work,env,log,status,detail,limit=None):
    deadline=END if limit is None else min(END,limit)
    if time.time()>=deadline:
        if deadline<END:raise RuntimeError('Development phase reserve reached; preserve time for frozen test')
        raise BudgetExpired('23:20 deadline reached')
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('ab') as stream:child=subprocess.Popen(command,cwd=work,env=env,stdout=stream,stderr=subprocess.STDOUT)
    while child.poll() is None:
        write(status,dict(stage=detail['stage'],current=detail.get('current'),child_pid=child.pid,deadline_kst='23:20'))
        if time.time()>=deadline:
            child.terminate()
            try:child.wait(timeout=2)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            if deadline<END:raise RuntimeError('Stopped own development child at reserve boundary; training untouched')
            raise BudgetExpired('Stopped own evaluation child at23:20; training untouched')
        time.sleep(.5)
    if child.returncode:raise RuntimeError('Evaluation failed; see '+str(log))

def controller(args,raw):
    work=raw/'runs/final-classifier-2215/classifier-study-20261116';run=raw/'runs/final-four-gpu-2225'
    status=run/f'finaleval_role{args.role}_status.json';lock=run/f'finaleval_role{args.role}.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    output=raw/'reports/final-selected-test'/f'role{args.role}';output.mkdir(parents=True,exist_ok=True)
    scores=[];failed=[];remaining=[]
    try:
        while True:
            stage=read(run/f'nextstage_role{args.role}_status.json')
            if stage.get('stage')=='failed':raise RuntimeError('NEXTSTAGE failed: '+str(stage.get('error')))
            if stage.get('stage')=='complete':break
            if time.time()>=END:raise BudgetExpired('NEXTSTAGE did not finish before23:20; no training stopped')
            write(status,dict(stage='waiting_for_nextstage',role=args.role,nextstage=stage.get('stage'),epoch=stage.get('epoch'),deadline_kst='23:20'))
            time.sleep(5)
        helpers=ensure_helpers(run/'final_evaluation_source');candidates,missing=inventory(raw)
        protocol=dict(selector_source_sha256=sha(__file__),rule=RULE,development_speaker='S02',development_snr_reference='active',development_snr_db=0,
            candidate_scope='available same-host baseline500/1000 and LS500 deployment pool; no CV models; missing models explicit; no LS1000 claim',
            missing_detectors=missing,candidates=candidates,bank_metadata_sha256=sha(work/'bank/metadata.json'),
            manifest_sha256=sha(raw/'runs/detector-diversity-v2/manifest.jsonl'),helpers=HELPERS,
            checkpoint_selection_note='Product selection may mix declared completed budgets; isolate budget500 baseline-vs-LS effects in matched500_effect_track; best-selection criterion identical; no fixed-last fallback',
            budget_rule='deterministic eligible-first order; reserve8minutes for frozen S01 test; unfinished scope explicitly reported',
            S01_used_for_selection=False)
        freeze=output/'selection_protocol.json'
        if freeze.exists():assert read(freeze)==protocol,'Frozen candidate set/protocol changed; inspect before reusing this output'
        else:write(freeze,protocol)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu,CUBLAS_WORKSPACE_CONFIG=':4096:8',CLASSIFIER_WORK=str(work))
        seen={}
        for number,c in enumerate(candidates):
            if not c['eligible'] and pick(scores) is not None:continue
            # Keep a fixed budget for the held-out test; never silently claim the full grid.
            if time.time()>END-480:
                remaining=[v['id'] for v in candidates[number:] if v['eligible'] or pick(scores) is None];break
            if not c['eligible'] and pick(scores) is not None:continue
            key=(c['mode'],c['condition'],c['checkpoint_sha256'],c['detector']['sha256'] if c['detector'] else None)
            if key in seen:
                source=seen[key];scores.append(dict(source,candidate=c,alias_of=source['candidate']['id']));continue
            case=output/'dev'/c['id'];case.mkdir(parents=True,exist_ok=True)
            if c['mode']=='actual':
                result=case/(c['condition']+'_threeway.json')
                command=[sys.executable,'-u',str(helpers/'pipeline_eval_extended.py'),'--condition',c['condition'],'--checkpoint',c['checkpoint'],'--detector-checkpoint',c['detector']['path'],'--raw-root',str(raw),'--output',str(case),'--actual-only']
            else:
                result=case/'metrics.json'
                command=[sys.executable,'-u',str(helpers/'EVAL_SNR_TEST.py'),'--condition',c['condition'],'--checkpoint',c['checkpoint'],'--mode','off','--split','dev','--snr-reference','active','--snr-db','0','--raw-root',str(raw),'--output',str(case)]
            try:
                previous=read(result)
                if c['mode']=='actual':
                    done=isinstance(previous,list) and len(previous)==1 and previous[0].get('classifier_checkpoint_sha256')==c['checkpoint_sha256'] and previous[0].get('detector_checkpoint_sha256')==c['detector']['sha256']
                else:
                    child_protocol=read(case/'candidate_protocol_freeze.json')
                    done=isinstance(previous,dict) and previous.get('stage')=='complete' and child_protocol.get('classifier_checkpoint_sha256')==c['checkpoint_sha256'] and child_protocol.get('snr_reference')=='active'
                if not done:run_child(command,work,env,case/'evaluation.log',status,dict(stage='development_selection',current=c['id']),limit=END-480)
                value=read(result)
                row=value[0] if c['mode']=='actual' else next(r for r in value['results'] if r.get('condition')=='added_noise' and r['nominal_snr_db']==0)
                score=dict(candidate=c,dev_f1=row['zero_command10_f1'] if c['mode']=='actual' else row['command10_f1'],
                    noise_fp=row['noise_fp_files'],original_f1=row['original']['command10_f1'],selected_best_epoch=row.get('classifier_epoch',row.get('selected_classifier_epoch')),result=str(result))
                scores.append(score);seen[key]=score
            except BudgetExpired:raise
            except Exception as exc:failed.append(dict(candidate=c['id'],error=repr(exc)))
            write(output/'development_scores.json',dict(scores=scores,matched500_effect_track=[r for r in scores if r['candidate']['budget']==500],failed=failed,missing_detectors=missing))
        winner=pick(scores)
        if winner is None:raise RuntimeError('No eligible completed deployment-pool candidate; diagnostic200 cannot win')
        winner=dict(winner,selection_rule=RULE,selection_protocol_sha256=sha(freeze),selection_complete=not remaining and not failed,
            unscored_candidates=remaining,failed_candidates=failed,missing_detectors=missing,S01_used_for_selection=False,
            global_selection_instruction='Parent selects among these local winners by the SAME DEV rule, never by test score')
        winner_path=output/'winner.json'
        if winner_path.exists():assert read(winner_path)==winner,'Frozen winner changed; refusing to reopen S01 with a different selection'
        else:write(winner_path,winner)
        print('DEV_WINNER_FROZEN',winner['candidate']['id'],'F1',winner['dev_f1'],flush=True)
        c=winner['candidate'];destination=output/'S01_test'
        command=[sys.executable,'-u',str(helpers/'EVAL_AMBIENT_TEST.py'),'--condition',c['condition'],'--checkpoint',c['checkpoint'],'--mode',c['mode'],'--split','test','--raw-root',str(raw),'--output',str(destination),'--latency']
        if c['detector']:command+=['--detector-checkpoint',c['detector']['path']]
        test=read(destination/'metrics.json')
        if test.get('stage')!='complete':run_child(command,work,env,output/'S01_test.log',status,dict(stage='S01_frozen_candidate_test',current=c['id']))
        final=dict(stage='complete',role=args.role,winner=winner,test_metrics=str(destination/'metrics.json'),
            test_results=read(destination/'metrics.json'),deadline_kst='23:20',training=False,S01_used_for_selection=False)
        write(output/'final_status.json',final);write(status,final)
        print('FINAL_TEST_COMPLETE ROLE',args.role,'WINNER',c['id'],'DEV_F1',winner['dev_f1'],'REPORT',output,flush=True)
    except BaseException as exc:
        final=dict(stage='budget_exhausted' if isinstance(exc,BudgetExpired) else 'failed',role=args.role,error=repr(exc),scores_completed=len(scores),failed_candidates=failed,unscored_candidates=remaining,deadline_kst='23:20')
        write(output/'final_status.json',final);write(status,final);raise
    finally:
        if lock.exists() and lock.read_text().strip()==str(os.getpid()):lock.unlink()

def main():
    p=argparse.ArgumentParser();p.add_argument('--role',type=int,choices=range(4));p.add_argument('--raw-root',type=Path);p.add_argument('--gpu',default='0');p.add_argument('--controller',action='store_true');p.add_argument('--status',action='store_true')
    args=p.parse_args();raw=discover(args.raw_root)
    if args.role is None:
        roles=sorted({int(path.name.split('role',1)[1].split('_',1)[0]) for path in (raw/'runs/final-four-gpu-2225').glob('nextstage_role*_status.json')})
        if len(roles)!=1:raise RuntimeError('Cannot infer one role; use --role0/1/2/3 with a space before the number')
        args.role=roles[0]
    if args.status:
        run=raw/'runs/final-four-gpu-2225'
        current=read(run/f'nextstage_role{args.role}_status.json');final=read(run/f'finaleval_role{args.role}_status.json')
        print(json.dumps(dict(role=args.role,training_stage=current.get('stage'),training_job=current.get('current'),epoch=current.get('epoch'),target=current.get('target_epochs'),eta_seconds=current.get('current_eta_seconds'),evaluation_stage=final.get('stage','not_queued'),evaluation_job=final.get('current'),deadline='23:20 KST'),ensure_ascii=False,indent=2));return
    if args.controller:return controller(args,raw)
    if time.time()>=END:raise BudgetExpired('23:20 deadline passed')
    run=raw/'runs/final-four-gpu-2225';run.mkdir(parents=True,exist_ok=True);pidfile=run/f'finaleval_role{args.role}_launcher.pid'
    if pidfile.exists():
        try:
            pid=int(pidfile.read_text())
            if alive(pid):print('FINAL_EVAL ALREADY QUEUED ROLE',args.role,'PID',pid);return
        except ValueError:pass
    python=choose_python(raw,args.role,args.gpu)
    if not python:raise RuntimeError('No existing Python; set DETECTOR_PYTHON')
    cmd=[python,'-u',str(Path(__file__).resolve()),'--controller','--role',str(args.role),'--raw-root',str(raw),'--gpu',args.gpu]
    with (run/f'finaleval_role{args.role}_launcher.log').open('ab') as log:
        child=subprocess.Popen(cmd,cwd=raw,env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu,CUBLAS_WORKSPACE_CONFIG=':4096:8'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    pidfile.write_text(str(child.pid)+'\n');time.sleep(1)
    if child.poll() is not None and child.returncode:raise RuntimeError('Final evaluator startup failed; inspect '+str(run/f'finaleval_role{args.role}_launcher.log'))
    print('FINAL_EVAL QUEUED ROLE',args.role,'PID',child.pid,'STATUS',run/f'finaleval_role{args.role}_status.json','DEADLINE23:20KST',flush=True)

if __name__=='__main__':main()
