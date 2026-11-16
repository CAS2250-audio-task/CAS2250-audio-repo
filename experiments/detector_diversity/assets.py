"""Build portable noise assets and rebase their manifests on another checkout."""
import argparse,json,tarfile
from pathlib import Path
from experiments.boundary_detector.train import write
from experiments.boundary_detector.data import sha
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/detector-diversity-v2'

def rebase():
    p=OUT/'noise.json';d=json.loads(p.read_text())
    for key,rows in d.items():
        for r in rows:
            old=Path(r['path']);r['path']=str(OUT/('esc50/mono16k' if r['original_group'].startswith('esc50:') else 'legacy_noise')/old.name)
    write(p,d);p=OUT/'noise_audit.json';d=json.loads(p.read_text())
    for r in d['audit']:r['path']=str(OUT/'esc50/mono16k'/Path(r['path']).name)
    write(p,d);p=OUT/'manifest.jsonl';rows=[]
    for line in p.read_text(encoding='utf-8').splitlines():
        r=json.loads(line);suffix=r['audio_path'].split('/datasets/task2/manual-clipped-v1/',1)[1];r['audio_path']=str(ROOT/'datasets/task2/manual-clipped-v1'/suffix);rows.append(r)
    p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')

def pack():
    target=ROOT/'runs/detector-diversity-v2-assets.tar.gz'
    paths=['legacy_noise','esc50/mono16k','esc50/LICENSE','noise.json','noise_audit.json','manifest.jsonl','noise_provenance.json']
    with tarfile.open(target,'w:gz',compresslevel=2,dereference=True) as archive:
        for path in paths:archive.add(OUT/path,arcname=str((OUT/path).relative_to(ROOT)))
    write(ROOT/'reports/detector-diversity-v2/assets.json',dict(filename=target.name,sha256=sha(target),bytes=target.stat().st_size,note='Extract in matching repo checkout then python -m experiments.detector_diversity.assets --rebase; audio uses upstream ESC-50 and legacy source licenses'))
    print(target,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--rebase',action='store_true');a=p.parse_args();rebase() if a.rebase else pack()
