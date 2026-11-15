"""Restore canonical noise assets without changing historical report manifests."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import torch
import hashlib
from .download import download
import experiments.noise_stress.sources as eval_module
import experiments.conformer_aug.sources as train_module
from experiments.conformer_aug.sources import SOURCES as TRAIN, prepare as training
from experiments.noise_stress.sources import SOURCES as EVAL, prepare as evaluation

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/compact-pooling-v1'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def restore(kind,fn,item):
    manifest=ROOT/('reports/conformer-aug-v1/training_noise_sources.json' if kind=='train' else 'reports/noise-stress-v1/noise_sources.json')
    row=next(r for r in json.loads(manifest.read_text()) if r['noise_id']==item[0])
    prepared=ROOT/row['prepared_path']
    if prepared.exists():
        row['prepared_sha256']=sha(prepared)
        row['regenerated_from_pinned_original']=True
        return row
    return fn(item)


def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    results = {'train': [], 'evaluation': []}
    eval_module.download=download
    train_module.download=download
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(restore, kind, fn, item): kind for kind, fn, entries in
                   [('train', training, TRAIN), ('evaluation', evaluation, EVAL)] for item in entries}
        for future in as_completed(futures):
            row = future.result()
            results[futures[future]].append(row)
            print('DOWNLOADED', futures[future], row['noise_id'], flush=True)
    for kind, rows in results.items():
        rows.sort(key=lambda x: x['noise_id'])
    assert not {r['original_sha256'] for r in results['train']} & {r['original_sha256'] for r in results['evaluation']}
    (OUT/'sources.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print('SOURCES_COMPLETE', flush=True)

if __name__ == '__main__':
    main()
