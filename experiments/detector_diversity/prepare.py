"""Pin/download a compact source-disjoint environmental noise expansion."""
import csv,hashlib,io,json,urllib.request,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import soundfile as sf
from experiments.boundary_detector.data import load_audio,sha
from experiments.boundary_detector.train import write
ROOT=Path(__file__).resolve().parents[2]
PIN='33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6'
BASE=f'https://raw.githubusercontent.com/karolpiczak/ESC-50/{PIN}'
CATEGORIES=['rain','sea_waves','crackling_fire','crickets','water_drops','pouring_water','wind','toilet_flush','washing_machine','vacuum_cleaner','clock_tick','engine','helicopter','chainsaw','hand_saw','train','airplane','car_horn','siren','church_bells','door_wood_knock','keyboard_typing']
OUT=ROOT/'runs/detector-diversity-v2'

def fetch(url,path):
    if path.exists():return
    path.parent.mkdir(parents=True,exist_ok=True)
    for attempt in range(4):
        try:
            data=urllib.request.urlopen(url,timeout=45).read();tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_bytes(data);tmp.replace(path);return
        except Exception:
            if attempt==3:raise
            time.sleep(1+attempt)

def main():
    OUT.mkdir(parents=True,exist_ok=True);raw=OUT/'esc50/raw';prepared=OUT/'esc50/mono16k'
    fetch(BASE+'/meta/esc50.csv',OUT/'esc50/esc50.csv');fetch(BASE+'/LICENSE',OUT/'esc50/LICENSE')
    records=list(csv.DictReader((OUT/'esc50/esc50.csv').read_text().splitlines()));selected=[];groups=set()
    for category in CATEGORIES:
        rows=sorted((r for r in records if r['category']==category),key=lambda r:hashlib.sha256(('detector-v2|'+r['filename']).encode()).hexdigest())
        for split,folds,count in [('train',{1,2,3},4),('audit',{4,5},1)]:
            chosen=[]
            for r in rows:
                if int(r['fold']) not in folds or r['src_file'] in groups:continue
                groups.add(r['src_file']);chosen.append(dict(r,split=split))
                if len(chosen)==count:break
            if len(chosen)!=count:raise ValueError(f'Insufficient independent recordings: {category}, {split}, {len(chosen)}')
            selected.extend(chosen)
    def download(r):
        original=raw/r['filename'];fetch(BASE+'/audio/'+r['filename'],original)
        target=prepared/r['filename'];target.parent.mkdir(parents=True,exist_ok=True)
        wave=load_audio(original)
        if np.mean(wave.astype(np.float64)**2)<1e-9:raise ValueError('Silent selected noise')
        sf.write(target,wave,16000,subtype='PCM_16')
        return dict(**r,path=str(target),original_group='esc50:'+r['src_file'],raw_sha256=sha(original),prepared_sha256=sha(target),seconds=len(wave)/16000,source_url=BASE+'/audio/'+r['filename'],freesound_url='https://freesound.org/s/'+r['src_file']+'/',license='ESC-50 CC BY-NC 3.0; see upstream LICENSE')
    with ThreadPoolExecutor(max_workers=8) as pool:downloaded=list(pool.map(download,selected))
    prior=json.loads((ROOT.parent/'CAS2250-speech-boundary-detector/runs/boundary-300-v1/noise.json').read_text())
    # Copy/link original prepared recordings locally so a bundle is portable.
    existing=OUT/'legacy_noise';existing.mkdir(exist_ok=True);noise={}
    for split in ['train','validation']:
        noise[split]=[]
        for entry in prior[split]:
            source=Path(entry['path']);dest=existing/source.name
            if not dest.exists():dest.symlink_to(source)
            noise[split].append(dict(path=str(dest),original_group=entry['original_group'],family='legacy:'+source.stem))
    for r in downloaded:
        if r['split']=='train':noise['train'].append(dict(path=r['path'],original_group=r['original_group'],family='esc50:'+r['category']))
    write(OUT/'noise.json',noise)
    write(OUT/'noise_audit.json',dict(audit=[dict(path=r['path'],original_group=r['original_group'],category=r['category']) for r in downloaded if r['split']=='audit']))
    # Rebase the manual manifest to this worktree; no audio from holdout is opened.
    source=ROOT.parent/'CAS2250-speech-boundary-detector/runs/manual-clipped-v1-import/manifest.jsonl';rows=[]
    for line in source.read_text(encoding='utf-8').splitlines():
        r=json.loads(line);r['audio_path']=str(ROOT/'datasets/task2/manual-clipped-v1'/Path(r['audio_path']).relative_to(ROOT.parent/'CAS2250-speech-boundary-detector/datasets/task2/manual-clipped-v1'));rows.append(r)
    (OUT/'manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    provenance=dict(dataset='ESC-50 environmental subset',commit=PIN,categories=CATEGORIES,records=downloaded,train_originals=len(noise['train']),legacy_train_originals=6,new_train_originals=sum(r['split']=='train' for r in downloaded),new_audit_originals=sum(r['split']=='audit' for r in downloaded),validation_legacy_sources=8,split='ESC folds1-3 train,4-5 final noise audit; src_file groups disjoint',labels_note='Non-human/non-speech category filter, incidental speech not perceptually audited; no intentional babble added',license_path=str(OUT/'esc50/LICENSE'))
    write(OUT/'noise_provenance.json',provenance)
    write(ROOT/'reports/detector-diversity-v2/noise_provenance.json',provenance)
    print(json.dumps({k:v for k,v in provenance.items() if k not in ['records','categories']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
