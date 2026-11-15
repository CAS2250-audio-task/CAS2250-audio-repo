"""Audio I/O and one canonical manifest contract, independent of CSV layout."""
import hashlib,json,math,subprocess
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SR=16000

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def audio_info(path):
    try:
        i=sf.info(path);return i.samplerate,i.duration
    except (RuntimeError,sf.LibsndfileError):
        try:
            d=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','a:0','-show_entries','stream=sample_rate,duration:format=duration','-of','json',str(path)],text=True))
            stream=d['streams'][0];return int(stream['sample_rate']),float(stream.get('duration',d['format']['duration']))
        except Exception as e:raise ValueError(f'Cannot inspect {path}; WAV/FLAC or ffprobe is required') from e


def load_audio(path):
    try:
        x,rate=sf.read(path,dtype='float32',always_2d=True);x=x.mean(1)
        if rate!=SR:
            g=math.gcd(rate,SR);x=resample_poly(x,SR//g,rate//g).astype(np.float32)
    except (RuntimeError,sf.LibsndfileError):
        raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(path),'-ac','1','-ar',str(SR),'-f','f32le','pipe:1'])
        x=np.frombuffer(raw,dtype='<f4').copy()
    if not len(x) or not np.isfinite(x).all():raise ValueError(f'Empty/nonfinite audio: {path}')
    return x


def read_manifest(path,speakers):
    rows=[json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    selected=[r for r in rows if r['speaker'] in speakers]
    found={r['speaker'] for r in selected}
    if found!=set(speakers):raise ValueError(f'Missing speaker(s): {set(speakers)-found}')
    seen=set();ids=set()
    for r in selected:
        p=Path(r['audio_path']).resolve()
        if p in seen or r['source_id'] in ids:raise ValueError('Duplicate original/source_id in canonical manifest')
        seen.add(p);ids.add(r['source_id'])
        last=-1.
        for start,end in r['segments']:
            if not (0<=start<end<=r['duration_seconds']+1/SR and start>=last):raise ValueError(f'Invalid segments: {r["source_id"]}')
            last=end
    return selected


def frame_labels(record,centers):
    labels=np.zeros(len(centers),np.float32)
    for start,end in record['segments']:labels[(centers>=start)&(centers<end)]=1
    return labels


def noise_manifest(path):
    d=json.loads(Path(path).read_text(encoding='utf-8'));base=Path(path).resolve().parent
    result={}
    for kind in ['train','validation']:
        entries=d.get(kind,[])
        paths=[e['path'] if isinstance(e,dict) else e for e in entries]
        result[kind]=[(base/p).resolve() if not Path(p).is_absolute() else Path(p) for p in paths]
        if not result[kind]:raise ValueError(f'Noise manifest requires nonempty {kind} list')
    groups={k:{e['original_group'] for e in d.get(k,[]) if isinstance(e,dict) and e.get('original_group')} for k in ['train','validation']}
    if groups['train']&groups['validation']:raise ValueError('Train/validation noise recording groups overlap')
    hashes={k:{sha(p) for p in paths} for k,paths in result.items()}
    if hashes['train']&hashes['validation']:raise ValueError('Train/validation noise source hashes overlap')
    return result
