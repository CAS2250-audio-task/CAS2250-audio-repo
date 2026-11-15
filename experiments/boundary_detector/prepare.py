"""Adapt arbitrary column names and timestamp units to original-audio JSONL."""
import argparse,csv,json
from pathlib import Path
from .data import audio_info,sha


def prepare(csv_path,mapping_path,output,audio_root,speakers):
    cfg=json.loads(Path(mapping_path).read_text(encoding='utf-8'));unit=cfg['unit']
    if unit not in ['seconds','milliseconds','samples']:raise ValueError('Explicit unit: seconds/milliseconds/samples')
    output=Path(output)
    if output.exists():raise FileExistsError(f'Use fresh manifest path: {output}')
    root=Path(audio_root).resolve();records={}
    with Path(csv_path).open(encoding=cfg.get('encoding','utf-8-sig'),newline='') as f:
        reader=csv.DictReader(f,delimiter=cfg.get('delimiter',','))
        required=[cfg[k] for k in ['path_col','speaker_col']]
        required += [cfg['segments_col']] if cfg.get('segments_col') else [cfg['start_col'],cfg['end_col']]
        if not set(required)<=set(reader.fieldnames or []):raise ValueError(f'Missing columns: {set(required)-set(reader.fieldnames or [])}')
        for line,row in enumerate(reader,2):
            speaker=row[cfg['speaker_col']].strip()
            # Filter before touching audio, including test/held-out files.
            if speaker not in speakers:continue
            value=row[cfg['path_col']].strip();path=(root/value).resolve()
            if not path.is_relative_to(root):raise ValueError(f'Path outside audio-root at row{line}')
            if path not in records:
                rate,duration=audio_info(path)
                records[path]=dict(source_id=row.get(cfg.get('id_col',''),value),audio_path=str(path),speaker=speaker,
                    native_sample_rate=rate,duration_seconds=duration,audio_sha256=sha(path),segments=[],
                    annotation_source=row.get(cfg.get('annotation_source_col',''),cfg.get('annotation_source','unspecified')))
            r=records[path]
            if cfg.get('id_col') and r['source_id']!=row[cfg['id_col']]:raise ValueError('Conflicting IDs for same original')
            if r['speaker']!=speaker:raise ValueError('Same original assigned multiple speakers')
            negative=row.get(cfg.get('negative_col',''),'').strip().lower() in ['1','true','yes']
            if negative:
                if r['segments']:raise ValueError('Conflicting negative and positive annotations')
                r['_negative_declared']=True;segments=[]
            elif cfg.get('segments_col'):segments=json.loads(row[cfg['segments_col']])
            else:
                if not row[cfg['start_col']].strip() or not row[cfg['end_col']].strip():raise ValueError(f'Missing boundary at row{line}; explicit negative flag required')
                segments=[[float(row[cfg['start_col']]),float(row[cfg['end_col']])]]
            factor={'seconds':1.,'milliseconds':1000.,'samples':float(cfg.get('annotation_sample_rate',r['native_sample_rate']))}[unit]
            if segments and r.get('_negative_declared'):raise ValueError('Conflicting negative and positive annotations')
            if factor<=0:raise ValueError('Invalid annotation sample rate')
            for pair in segments:
                if isinstance(pair,dict):a,b=pair['start'],pair['end']
                else:a,b=pair
                a,b=float(a)/factor,float(b)/factor
                if not (0<=a<b<=r['duration_seconds']+1/r['native_sample_rate']):raise ValueError(f'Out-of-range/reversed boundary at row{line}: {(a,b)}')
                r['segments'].append([a,b])
            if negative and r['segments']:raise ValueError('Original has both negative and positive annotation')
    if not records:raise ValueError('No selected records')
    ids=set()
    for r in records.values():
        r.pop('_negative_declared',None)
        r['segments'].sort()
        if any(a[1]>b[0] for a,b in zip(r['segments'],r['segments'][1:])):raise ValueError(f'Overlapping/duplicate annotations: {r["source_id"]}')
        if r['source_id'] in ids:raise ValueError('source_id must identify original, not clip/row')
        ids.add(r['source_id'])
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records.values()),encoding='utf-8')
    output.with_suffix(output.suffix+'.provenance.json').write_text(json.dumps(dict(csv_sha256=sha(csv_path),mapping=cfg,selected_speakers=speakers,originals=len(records)),ensure_ascii=False,indent=2),encoding='utf-8')
    return list(records.values())


def main():
    p=argparse.ArgumentParser();p.add_argument('--csv',required=True);p.add_argument('--mapping',required=True);p.add_argument('--audio-root',required=True);p.add_argument('--output',required=True);p.add_argument('--speakers',nargs='+',required=True)
    a=p.parse_args();r=prepare(a.csv,a.mapping,a.output,a.audio_root,a.speakers);print('Prepared originals:',len(r))


if __name__=='__main__':main()
