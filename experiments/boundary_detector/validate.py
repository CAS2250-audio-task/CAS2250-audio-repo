"""Inspect imported labels without starting training or reading other speakers."""
import argparse,json
from collections import Counter
from .data import read_manifest,sha,audio_info


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--speakers',nargs='+',required=True);a=p.parse_args()
    rows=read_manifest(a.manifest,a.speakers);flags=[]
    for r in rows:
        if sha(r['audio_path'])!=r['audio_sha256']:raise ValueError('Audio changed: '+r['source_id'])
        _,duration=audio_info(r['audio_path'])
        if abs(duration-r['duration_seconds'])>.01:raise ValueError('Duration changed: '+r['source_id'])
        seconds=sum(b-a for a,b in r['segments'])
        if seconds>duration-.02:flags.append(dict(source_id=r['source_id'],reason='Almost entire original is labelled positive; verify boundaries, do not automatically reject'))
        if r['annotation_source']!='manual':flags.append(dict(source_id=r['source_id'],reason='Not explicitly marked manual; reference quality must be audited'))
    print(json.dumps(dict(originals=len(rows),speakers=dict(Counter(r['speaker'] for r in rows)),negative_originals=sum(not r['segments'] for r in rows),total_seconds=sum(r['duration_seconds'] for r in rows),files_under1p5_seconds=sum(r['duration_seconds']<1.5 for r in rows),review_flags=flags),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
