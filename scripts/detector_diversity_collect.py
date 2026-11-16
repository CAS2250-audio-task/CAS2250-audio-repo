"""Wait for the local and Jupyter halves, compare and run the predeclared verification."""
import hashlib,json,os,subprocess,sys,tarfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/detector-diversity-v2'
sys.path.insert(0,str(ROOT))
from experiments.boundary_detector.train import write
from experiments.detector_diversity.report import phase_comparison
from experiments.detector_diversity.launch import command
TAG='detector-diversity-v2-distributed-assets-v1';REPO='CAS2250-audio-task/CAS2250-audio-repo'
ENV=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8',CUBLAS_WORKSPACE_CONFIG=':4096:8',PATH='/root/.local/bin:'+os.environ['PATH'])

def status(name):
 p=OUT/name/'progress.json';return json.loads(p.read_text()) if p.exists() else {}

def main():
    started=time.time();download=OUT/'remote_download';download.mkdir(exist_ok=True);received=False
    while True:
        local={n:status(n) for n in ['baseline','lookahead']}
        if not received:
            result=subprocess.run(['gh','release','download',TAG,'--repo',REPO,'--pattern','jupyter-boundary-spectral-results.tar.gz*','--dir',str(download),'--clobber'],env=ENV,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            if result.returncode==0:
                archive=download/'jupyter-boundary-spectral-results.tar.gz';expected=Path(str(archive)+'.sha256').read_text().split()[0];assert hashlib.sha256(archive.read_bytes()).hexdigest()==expected
                with tarfile.open(archive,'r:gz') as tar:
                    for member in tar.getmembers():
                        assert not member.issym() and not member.islnk() and '..' not in Path(member.name).parts
                        assert any(member.name.startswith('runs/detector-diversity-v2/'+n) for n in ['boundary','spectral'])
                    tar.extractall(ROOT)
                assert all(status(n).get('stage')=='complete' and status(n).get('epoch')==300 for n in ['boundary','spectral']);received=True
        write(OUT/'distributed.json',dict(stage='waiting_training',local={n:dict(stage=d.get('stage'),epoch=d.get('epoch')) for n,d in local.items()},remote_received=received,elapsed_minutes=(time.time()-started)/60))
        if received and all(d.get('stage')=='complete' for d in local.values()):break
        time.sleep(30)
    inputs=[json.loads((OUT/n/'inputs.json').read_text()) for n in ['baseline','lookahead','boundary','spectral']]
    assert all(x['noise_hashes']==inputs[0]['noise_hashes'] for x in inputs)
    wave_match=len({x['epoch1_first64_augmented_sha256'] for x in inputs})==1
    write(ROOT/'reports/detector-diversity-v2/distributed_audit.json',dict(same_waveform_fingerprint=wave_match,same_noise_contents=True,actual_gpu_allocation='local baseline/lookahead; Jupyter boundary/spectral',normalization_note='Inference uses each checkpoint own statistics, with tolerance1e-5 cross-hardware equality check'))
    comparison=phase_comparison();base=comparison[0]['cost_ms'];useful=[r['name'] for r in comparison[1:] if r['cost_ms']<base*.99]
    write(OUT/'combination_decision.json',dict(criterion='>1% lower legacy validation asymmetric cost',baseline_cost_ms=base,useful=useful,seed=42 if len(useful)>1 else 43,noise_audit_not_used_for_decision=True))
    write(OUT/'distributed.json',dict(stage='phase1_evaluation',useful=useful))
    with (OUT/'evaluation.log').open('a') as log:subprocess.run([sys.executable,'-m','experiments.detector_diversity.report'],cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT,check=True)
    if useful:
        write(OUT/'distributed.json',dict(stage='combination_training',enabled=useful))
        with (OUT/'combined.log').open('a') as log:subprocess.run(command('combined',42 if len(useful)>1 else 43,useful),cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT,check=True)
        with (OUT/'evaluation.log').open('a') as log:subprocess.run([sys.executable,'-m','experiments.detector_diversity.report'],cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT,check=True)
    write(OUT/'distributed.json',dict(stage='complete',elapsed_minutes=(time.time()-started)/60,combined=useful))
    subprocess.run(['git','add','reports/detector-diversity-v2'],cwd=ROOT,check=True)
    changed=subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT)
    if changed.returncode:
        subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@openai.com','commit','-m','Report distributed detector diversity results and combination verification'],cwd=ROOT,check=True)
        subprocess.run(['git','push','origin','experiment/detector-diversity-v2'],cwd=ROOT,env=ENV,check=True)

if __name__=='__main__':
 try:main()
 except BaseException as exc:
  write(OUT/'distributed.json',dict(stage='failed',error=repr(exc)));raise
