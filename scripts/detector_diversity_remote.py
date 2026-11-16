"""Resume the Jupyter half and upload its compact result archive automatically."""
import argparse,hashlib,json,os,subprocess,sys,tarfile,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/detector-diversity-v2'
REPO='CAS2250-audio-task/CAS2250-audio-repo';TAG='detector-diversity-v2-distributed-assets-v1';NAMES=['boundary','spectral']

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def write(path,value):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(path)

def migrate():
    manifest=OUT/'manifest.jsonl';noise=OUT/'noise.json';metadata=json.loads(noise.read_text());actual_noise={k:[digest(r['path']) for r in rows] for k,rows in metadata.items()}
    rows=[json.loads(x) for x in manifest.read_text(encoding='utf-8').splitlines()];selected=[r for r in rows if r['speaker'] in ['S02','S03','S04','S05']]
    for r in selected:assert digest(r['audio_path'])==r['audio_sha256']
    critical=('train.py','augment.py','models.py','data.py','model.py','segments.py')
    for name in NAMES:
        folder=OUT/name;old=json.loads((folder/'config.json').read_text());inputs=json.loads((folder/'inputs.json').read_text());assert actual_noise==inputs['noise_hashes'],'Noise content differs from transferred checkpoint'
        new=dict(old);new.update(manifest=str(manifest),noise_manifest=str(noise),output=str(folder),manifest_sha256=digest(manifest),noise_manifest_sha256=digest(noise))
        sources={str(p.relative_to(ROOT)):digest(p) for package in ['experiments/detector_diversity','experiments/boundary_detector'] for p in (ROOT/package).glob('*.py')}
        for path,expected in old['source_sha256'].items():
            if Path(path).name in critical:assert sources[path]==expected,f'Training source changed: {path}'
        new['source_sha256']=sources;write(folder/'transfer_migration.json',dict(old_path=old['output'],new_path=new['output'],noise_hashes_verified=True,audio_hashes_verified=True,training_sources_unchanged=True,python=sys.version,torch=str(torch.__version__)))
        write(folder/'config.json',new)
        for file in ['last.pt','best.pt']:
            path=folder/file;state=torch.load(path,map_location='cpu',weights_only=False);assert state['config']==old or state['config']==new;state['config']=new;torch.save(state,path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');a=p.parse_args();migrate()
    if a.prepare_only:print('Transfer verification/config migration passed');return
    assert torch.cuda.is_available(),'CUDA is not available in the selected Python environment'
    env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8',CUBLAS_WORKSPACE_CONFIG=':4096:8');active={};started=time.time()
    for name in NAMES:
        progress=OUT/name/'progress.json'
        if progress.exists() and json.loads(progress.read_text()).get('stage')=='complete':continue
        log=(OUT/(name+'.log')).open('a',encoding='utf-8')
        active[name]=subprocess.Popen([sys.executable,'-m','experiments.detector_diversity.train','--variant',name,'--output',str(OUT/name),'--epochs','300','--resume'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);log.close()
    while active:
        progress={}
        for name,process in list(active.items()):
            result=process.poll()
            if result is not None:
                if result:write(OUT/'remote_status.json',dict(stage='failed',name=name,returncode=result));raise SystemExit(result)
                del active[name]
            file=OUT/name/'progress.json'
            if file.exists():progress[name]=json.loads(file.read_text())
        write(OUT/'remote_status.json',dict(stage='training' if active else 'packing',elapsed_minutes=(time.time()-started)/60,models=progress))
        if active:time.sleep(15)
    payload=OUT.parent/'jupyter-boundary-spectral-results.tar.gz'
    with tarfile.open(payload,'w:gz',compresslevel=2) as archive:
        for name in NAMES:archive.add(OUT/name,arcname=str((OUT/name).relative_to(ROOT)))
    checksum=Path(str(payload)+'.sha256');checksum.write_text(digest(payload)+'  '+payload.name+'\n')
    for attempt in range(12):
        result=subprocess.run(['gh','release','upload',TAG,str(payload),str(checksum),'--clobber','--repo',REPO],cwd=ROOT,env=env)
        if result.returncode==0:break
        write(OUT/'remote_status.json',dict(stage='upload_retry',attempt=attempt+1));time.sleep(min(120,10*(attempt+1)))
    else:write(OUT/'remote_status.json',dict(stage='upload_failed',archive=str(payload)));raise SystemExit('Training finished; GitHub upload failed, result archive retained')
    write(OUT/'remote_status.json',dict(stage='complete',elapsed_minutes=(time.time()-started)/60,release=TAG,asset=payload.name));print('두 실험 완료 및 원격 업로드 완료: '+TAG,flush=True)

if __name__=='__main__':main()
