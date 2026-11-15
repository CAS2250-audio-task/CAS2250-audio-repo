"""Split a verified two-head resume snapshot without resetting optimizer or schedule."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import torch
import torchaudio


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def import_snapshot(root, archive):
    with tarfile.open(archive) as tar:
        manifest_name = next(m.name for m in tar.getmembers() if m.name.endswith('/manifest.json'))
        prefix = manifest_name[:-len('manifest.json')]
        manifest = json.load(tar.extractfile(manifest_name))
        for row in manifest['files']:
            rel = Path(row['path'])
            assert not rel.is_absolute() and '..' not in rel.parts
            member = tar.getmember(prefix+row['path']);assert member.isfile()
            data = tar.extractfile(member).read()
            assert hashlib.sha256(data).hexdigest()==row['sha256']
            dest = root/rel
            if dest.exists():
                assert digest(dest)==row['sha256'], str(dest)
            else:
                dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)


def prepare(root, head, destination):
    source = 'frequency-local-lowcost-v1-teamPS-spk03-e1000'
    src = root/'runs'/source/'spk03';dst = root/'runs'/destination/'spk03'
    protocol_path = root/'reports'/source/'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    for path, value in protocol['source_sha256'].items():
        assert digest(root/path)==value, path
    checkpoint = torch.load(src/'last.pt',map_location='cpu',weights_only=True)
    assert checkpoint['protocol_sha256']==digest(protocol_path)
    assert checkpoint['epoch']==670 and head in checkpoint['models']
    assert not dst.exists(), 'Destination exists; inspect and resume it instead of preparing again'
    original_hash = digest(protocol_path)
    protocol['study'] = destination
    protocol['environment'] = dict(torch=torch.__version__,torchaudio=torchaudio.__version__,gpu=torch.cuda.get_device_name(0))
    report = root/'reports'/destination;report.mkdir(parents=True,exist_ok=True)
    target_protocol = report/'protocol.json'
    target_protocol.write_text(json.dumps(protocol,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    new_hash = digest(target_protocol)
    dst.mkdir(parents=True)
    for key in ['models','optimizers','projectors','best']:
        checkpoint[key] = {head:checkpoint[key][head]}
    checkpoint['protocol_sha256'] = new_hash
    torch.save(checkpoint,dst/'last.pt')
    best = torch.load(src/(head+'_best.pt'),map_location='cpu',weights_only=True)
    best['protocol_sha256'] = new_hash
    torch.save(best,dst/(head+'_best.pt'))
    history = [r for r in json.loads((src/'history.json').read_text()) if r['model']==head and r['epoch']<=670]
    assert len(history)==670
    (dst/'history.json').write_text(json.dumps(history,ensure_ascii=False,indent=2)+'\n')
    (report/'resume_provenance.json').write_text(json.dumps(dict(
        source=source,source_protocol_sha256=original_hash,source_last_sha256=digest(src/'last.pt'),
        head=head,resume_epoch=670,optimizer_and_schedule_preserved=True,
        changed_metadata=['study output name','current GPU environment'],
        destination_protocol_sha256=new_hash),indent=2)+'\n')
    print('PREPARED',head,'epoch670 ->',destination,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--head',choices=['cnn_pooled_query','cnn_self_cls'],required=True)
    parser.add_argument('--destination',required=True)
    parser.add_argument('--archive',type=Path)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    if args.archive:import_snapshot(root,args.archive)
    prepare(root,args.head,args.destination)
