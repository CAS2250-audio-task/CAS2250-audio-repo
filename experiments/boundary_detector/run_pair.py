"""Detached sequential 300-epoch paired experiment and validation margin sweep."""
import json,os,subprocess,sys,time
from pathlib import Path
from .segments import asymmetric_cost
from .train import write
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/boundary-300-v1'

def margins(directory):
    rows=json.loads((directory/'best_predictions.json').read_text(encoding='utf-8'))
    scores=[]
    for before in [0,10,20,30]:
        for after in [0,10,20,30]:
            costs=[]
            for row in rows:
                expanded=[]
                for a,b in row['predicted']:
                    a=max(0,a-before/1000);b=min(row['duration_seconds'],b+after/1000)
                    if expanded and a<=expanded[-1][1]:expanded[-1][1]=max(expanded[-1][1],b)
                    else:expanded.append([a,b])
                costs.append(asymmetric_cost(row['gold'],expanded)['cost_seconds'])
            scores.append(dict(before_ms=before,after_ms=after,validation_cost_ms=1000*sum(costs)/len(costs)))
    write(directory/'margin_sweep.json',dict(selection='validation only; deploy explicitly with selected margins',recommended=min(scores,key=lambda x:x['validation_cost_ms']),all_settings=scores))

def main():
    OUT.mkdir(exist_ok=True,parents=True)
    env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    started=time.time()
    for arch in ['tcn','cnn_gru']:
        directory=OUT/arch
        write(OUT/'pair_progress.json',dict(stage='running',architecture=arch,total_epochs=600,started_unix=started))
        command=[sys.executable,'-m','experiments.boundary_detector.train',
                 '--manifest',str(ROOT/'runs/manual-clipped-v1-import/manifest.jsonl'),
                 '--train-speakers','S03','S04','S05','--val-speakers','S02',
                 '--noise-manifest',str(OUT/'noise.json'),'--output',str(directory),
                 '--architecture',arch,'--width','64','--hop','80','--epochs','300','--batch','32']
        if (directory/'last.pt').exists():command+=['--resume']
        with (OUT/f'{arch}.log').open('a',encoding='utf-8') as log:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            write(OUT/'pair_progress.json',dict(stage='failed',architecture=arch,returncode=result.returncode));raise SystemExit(result.returncode)
        margins(directory)
    write(OUT/'pair_progress.json',dict(stage='complete',elapsed_minutes=(time.time()-started)/60))

if __name__=='__main__':main()
