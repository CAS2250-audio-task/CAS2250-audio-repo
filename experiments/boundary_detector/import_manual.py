"""Join the manual-clipped-v1 export by pair_id; use speech, not padded output bounds."""
import argparse,csv,json
from pathlib import Path
from .prepare import prepare


def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--speakers',nargs='+',required=True);a=p.parse_args()
    data=a.dataset.resolve();a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'manifest.jsonl').exists():raise FileExistsError('Use a fresh import output')
    ranges=read_csv(data/'clipping_ranges_1000.csv');index=read_csv(data/'file_index.csv')
    lookup={r['pair_id']:r for r in ranges}
    if len(lookup)!=len(ranges) or len({r['pair_id'] for r in index})!=len(index):raise ValueError('Duplicate pair IDs')
    if set(lookup)!={r['pair_id'] for r in index}:raise ValueError('Index/range IDs differ')
    joined=[];speakers={}
    for row in index:
        r=lookup[row['pair_id']];sid=row['pair_id'].split('-')[0]
        if sid not in a.speakers:continue
        if r['status']!='confirmed':raise ValueError('Unconfirmed boundary: '+r['pair_id'])
        if row['speaker']!=r['speaker'] or row['word']!=r['word']:raise ValueError('ID metadata mismatch')
        if sid in speakers and speakers[sid]!=r['speaker']:raise ValueError('Conflicting speaker ID')
        speakers[sid]=r['speaker']
        joined.append(dict(original_id=r['pair_id'],original_path=row['trimmed_original_path'],speaker_id=sid,
            speech_start=r['speech_start_in_trimmed_s'],speech_end=r['speech_end_in_trimmed_s'],
            boundary_source='user_confirmed',is_negative='false'))
    table=a.output/'detector_annotations.csv'
    if not joined:raise ValueError('No selected data')
    with table.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(joined[0]));w.writeheader();w.writerows(joined)
    mapping=dict(path_col='original_path',speaker_col='speaker_id',id_col='original_id',start_col='speech_start',end_col='speech_end',negative_col='is_negative',annotation_source_col='boundary_source',unit='seconds')
    config=a.output/'mapping.json';config.write_text(json.dumps(mapping,indent=2)+'\n',encoding='utf-8')
    records=prepare(table,config,a.output/'manifest.jsonl',data,a.speakers)
    (a.output/'speaker_ids.json').write_text(json.dumps(speakers,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(originals=len(records),speakers=speakers,manifest=str(a.output/'manifest.jsonl'),labels='speech_*_in_trimmed_s; no 30ms output margin',training_started=False),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
