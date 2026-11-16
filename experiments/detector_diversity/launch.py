"""Run four independent ablations, compare, verify useful changes and publish local results."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from experiments.boundary_detector.train import write
from .report import phase_comparison
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/detector-diversity-v2'
NAMES=['baseline','lookahead','boundary','spectral']

def command(name,seed=42,enabled=None):
    cmd=[sys.executable,'-m','experiments.detector_diversity.train','--variant',name,'--output',str(OUT/name),'--epochs','300','--seed',str(seed)]
    if enabled is not None:cmd+=['--enabled']+enabled
    if (OUT/name/'last.pt').exists():cmd+=['--resume']
    return cmd

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--max-parallel',type=int,default=4);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    write(OUT/'supervisor_pid.json',dict(pid=os.getpid(),started_unix=time.time(),physical_gpu_count=1,max_parallel=args.max_parallel))
    pending=NAMES.copy();active={};done=[];started=time.time()
    while pending or active:
        while pending and len(active)<args.max_parallel:
            name=pending.pop(0);progress=OUT/name/'progress.json'
            if progress.exists() and json.loads(progress.read_text()).get('stage')=='complete':done.append(name);continue
            log=(OUT/(name+'.log')).open('a',encoding='utf-8');p=subprocess.Popen(command(name),cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);log.close();active[name]=p
        failed=[]
        for name,p in list(active.items()):
            result=p.poll()
            if result is not None:
                del active[name]
                if result:failed.append(dict(name=name,returncode=result))
                else:done.append(name)
        if failed:
            for p in active.values():p.terminate()
            write(OUT/'pipeline.json',dict(stage='failed',errors=failed,finished=done));raise SystemExit(1)
        write(OUT/'pipeline.json',dict(stage='phase1_training',running={k:p.pid for k,p in active.items()},queued=pending,finished=done,elapsed_minutes=(time.time()-started)/60))
        if active:time.sleep(10)
    comparison=phase_comparison();inputs=[json.loads((OUT/n/'inputs.json').read_text()) for n in NAMES]
    assert len({x['epoch1_first64_augmented_sha256'] for x in inputs})==1,'Models did not receive identical augmented inputs'
    assert len({x['normalization_sha256'] for x in inputs})==1,'Training normalization mismatch'
    write(OUT/'pipeline.json',dict(stage='phase1_evaluation',elapsed_minutes=(time.time()-started)/60))
    # Report before selecting combination; selection uses ONLY existing validation,
    # never new environmental audit performance.
    base=comparison[0]['cost_ms'];useful=[r['name'] for r in comparison[1:] if r['cost_ms']<base*.99]
    write(OUT/'combination_decision.json',dict(criterion='>1% reduction in legacy validation asymmetric cost vs common new-augmentation baseline',baseline_cost_ms=base,useful=useful,seed=42 if len(useful)>1 else 43,source='phase1_comparison.json; new environmental audit not yet opened'))
    # Stage 3: full error/retention/latency comparison before any combination run.
    with (OUT/'evaluation.log').open('a',encoding='utf-8') as log:
        result=subprocess.run([sys.executable,'-m','experiments.detector_diversity.report'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:write(OUT/'pipeline.json',dict(stage='failed',name='phase1_evaluation',returncode=result.returncode));raise SystemExit(result.returncode)
    if useful:
        write(OUT/'pipeline.json',dict(stage='combination_training',enabled=useful,seed=42 if len(useful)>1 else 43))
        with (OUT/'combined.log').open('a',encoding='utf-8') as log:
            result=subprocess.run(command('combined',42 if len(useful)>1 else 43,useful),cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:write(OUT/'pipeline.json',dict(stage='failed',name='combined',returncode=result.returncode));raise SystemExit(result.returncode)
    else:write(OUT/'combination_skipped.json',dict(reason='No individual change improved the validation cost by >1%; no unsupported combination launched'))
    write(OUT/'pipeline.json',dict(stage='final_evaluation',elapsed_minutes=(time.time()-started)/60))
    with (OUT/'evaluation.log').open('a',encoding='utf-8') as log:
        result=subprocess.run([sys.executable,'-m','experiments.detector_diversity.report'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:write(OUT/'pipeline.json',dict(stage='failed',name='evaluation',returncode=result.returncode));raise SystemExit(result.returncode)
    write(OUT/'pipeline.json',dict(stage='complete',elapsed_minutes=(time.time()-started)/60,phase1=NAMES,combination=useful))

if __name__=='__main__':main()
