"""Pack/restore exact inputs and experiment artifacts; excludes secrets and test01 audio."""
import argparse
import hashlib
import io
import json
from pathlib import Path,PurePosixPath
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'runs/compact-pooling-v1'
DEFAULT=Path('/root/CAS2250-artifacts/compact-pooling-v1-assets.tar.gz')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(1<<20),b''):h.update(data)
    return h.hexdigest()


def pack(destination,cache_only=False):
    import pandas as pd
    sources=json.loads((STUDY/'sources.json').read_text())
    frame=pd.read_csv(ROOT/'datasets/task2/fixed-1p5-v1/metadata.csv')
    frame=frame[frame.speaker_id.isin(['spk02','spk03','spk04','spk05'])]
    paths=[STUDY/'sources.json',STUDY/'formants.npy',STUDY/'formants.json']
    paths += [ROOT/s['prepared_path'] for rows in sources.values() for s in rows]
    paths += [ROOT/'datasets/task2/fixed-1p5-v1'/p for p in frame.path]
    paths += [ROOT/'datasets/task2/fixed-1p5-v1'/p for p in ['metadata.csv','preprocessing_recipe.csv','manifest.json','README.md']]
    paths += [ROOT/'configs/task2_real.json']
    if not cache_only:
        assert (ROOT/'reports/compact-pooling-v1/summary.json').exists(),'Finish training first or use --cache-only'
        paths += list((ROOT/'reports/compact-pooling-v1').glob('*'))
        for speaker in ('spk02','spk03'):
            paths += [p for p in (STUDY/speaker).glob('*') if p.is_file()]
        paths += [p for p in [STUDY/'progress.json',STUDY/'timing_pilot.json',STUDY/'branch_artifact_audit.json'] if p.exists()]
    paths=sorted(set(paths))
    manifest=dict(format='compact-pooling-v1-bundle-1',cache_only=cache_only,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        contains_test01_audio=False,files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in paths])
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_suffix(destination.suffix+'.partial')
    with tarfile.open(temporary,'w:gz',compresslevel=1) as tar:
        payload=(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode()
        info=tarfile.TarInfo('_BUNDLE_MANIFEST.json');info.size=len(payload)
        tar.addfile(info,io.BytesIO(payload))
        for p in paths:tar.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
    temporary.replace(destination)
    digest=sha(destination)
    destination.with_suffix(destination.suffix+'.sha256').write_text(f'{digest}  {destination.name}\n')
    destination.with_suffix(destination.suffix+'.manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(archive=str(destination),bytes=destination.stat().st_size,sha256=digest,files=len(paths)),indent=2),flush=True)


def restore(source):
    with tarfile.open(source,'r:gz') as tar:
        manifest=json.load(tar.extractfile('_BUNDLE_MANIFEST.json'))
        assert manifest['format']=='compact-pooling-v1-bundle-1'
        allowed={r['path']:r for r in manifest['files']}
        for member in tar:
            if member.name=='_BUNDLE_MANIFEST.json':continue
            name=PurePosixPath(member.name)
            if name.is_absolute() or '..' in name.parts or not member.isfile() or member.name not in allowed:
                raise ValueError('Unsafe archive member: '+member.name)
            target=ROOT/str(name)
            assert target.resolve().is_relative_to(ROOT.resolve())
            row=allowed[member.name]
            assert member.size==row['bytes']
            target.parent.mkdir(parents=True,exist_ok=True)
            temporary=target.with_suffix(target.suffix+'.restore-partial')
            with tar.extractfile(member) as f,temporary.open('wb') as out:
                for block in iter(lambda:f.read(1<<20),b''):out.write(block)
            assert sha(temporary)==row['sha256'],member.name
            temporary.replace(target)
        for name,row in allowed.items():assert sha(ROOT/name)==row['sha256'],name
    print('RESTORED_AND_VERIFIED',len(allowed),'files',flush=True)


def main():
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('pack');p.add_argument('--output',type=Path,default=DEFAULT);p.add_argument('--cache-only',action='store_true')
    p=sub.add_parser('restore');p.add_argument('archive',type=Path)
    args=parser.parse_args()
    if args.action=='pack':pack(args.output,args.cache_only)
    else:restore(args.archive)

if __name__=='__main__':main()
