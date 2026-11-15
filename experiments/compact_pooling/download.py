"""Range-parallel recovery of publisher-pinned original files, with resumable prefixes."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import math
from pathlib import Path
import shutil
import urllib.request


def md5(path):
    h=hashlib.md5()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def download(url,target,expected):
    target=Path(target)
    if target.exists() and md5(target)==expected:return
    def request(start,end):
        return urllib.request.Request(url,headers={'User-Agent':'audio-research/1.0','Range':f'bytes={start}-{end}','Accept-Encoding':'identity'})
    with urllib.request.urlopen(request(0,0),timeout=60) as response:
        if response.status!=206:
            raise RuntimeError('Server does not support verified range downloads')
        total=int(response.headers['Content-Range'].split('/')[-1])
        response.read()
    prefix=target.with_suffix(target.suffix+'.partial')
    start=prefix.stat().st_size if prefix.exists() and prefix.stat().st_size<total else 0
    parts=[]
    width=math.ceil((total-start)/6)
    def part(bounds):
        begin,end=bounds
        destination=target.with_name(target.name+f'.range-{begin}-{end}')
        expected_length=end-begin+1
        if destination.exists() and destination.stat().st_size==expected_length:return destination
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request(begin,end),timeout=120) as response,destination.open('wb') as f:
                    assert response.status==206 and response.headers['Content-Range']==f'bytes {begin}-{end}/{total}'
                    shutil.copyfileobj(response,f,length=1<<20)
                assert destination.stat().st_size==expected_length
                print('RANGE_COMPLETE',target.name,begin,end,flush=True)
                return destination
            except Exception:
                if attempt==2:raise
    bounds=[(begin,min(begin+width,total)-1) for begin in range(start,total,width)]
    print('RANGE_START',target.name,'prefix',start,'total',total,flush=True)
    with ThreadPoolExecutor(max_workers=6) as pool:parts=list(pool.map(part,bounds))
    assembled=target.with_name(target.name+'.assembled')
    with assembled.open('wb') as output:
        if start:
            with prefix.open('rb') as f:shutil.copyfileobj(f,output,length=1<<20)
        for p in parts:
            with p.open('rb') as f:shutil.copyfileobj(f,output,length=1<<20)
    assert assembled.stat().st_size==total and md5(assembled)==expected,'Publisher MD5 mismatch'
    assembled.replace(target)
    for p in parts:p.unlink()
    if prefix.exists():prefix.unlink()
    print('VERIFIED_ORIGINAL',target.name,flush=True)
