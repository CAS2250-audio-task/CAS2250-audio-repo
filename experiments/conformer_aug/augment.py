"""Protected speed perturbation, native-feature masks and online training noise."""
import ctypes as ct
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset

from src.engine import file_hash, write_json
from experiments.noise_stress.mixing import mix_at_snr
from .sources import ROOT, STUDY, REPORT

LIBDIR=Path("/root/audio/runs/auxiliary-cv-v1/runtime/unpacked/usr/lib/x86_64-linux-gnu")
LIBFILES=("libfftw3.so.3","libsamplerate.so.0","librubberband.so.2")
_LIB=None


def library():
    global _LIB
    if _LIB is not None:return _LIB
    for name in LIBFILES[:2]:ct.CDLL(str(LIBDIR/name),mode=ct.RTLD_GLOBAL)
    lib=ct.CDLL(str(LIBDIR/LIBFILES[-1]),mode=ct.RTLD_GLOBAL)
    pointer=ct.POINTER(ct.POINTER(ct.c_float))
    signatures={"new":(ct.c_void_p,[ct.c_uint,ct.c_uint,ct.c_int,ct.c_double,ct.c_double]),
                "delete":(None,[ct.c_void_p]),"set_max_process_size":(None,[ct.c_void_p,ct.c_uint]),
                "set_expected_input_duration":(None,[ct.c_void_p,ct.c_uint]),
                "study":(None,[ct.c_void_p,pointer,ct.c_uint,ct.c_int]),
                "process":(None,[ct.c_void_p,pointer,ct.c_uint,ct.c_int]),
                "available":(ct.c_int,[ct.c_void_p]),"retrieve":(ct.c_uint,[ct.c_void_p,pointer,ct.c_uint])}
    for name,(result,args) in signatures.items():
        function=getattr(lib,"rubberband_"+name)
        function.restype,function.argtypes=result,args
    _LIB=lib
    return lib


def pointer(array):
    return (ct.POINTER(ct.c_float)*1)(array.ctypes.data_as(ct.POINTER(ct.c_float)))


def speed_crop(length,rate,protected):
    a,b=int(np.floor(protected[0]/rate)),int(np.ceil(protected[1]/rate))
    if b-a>24000:return None
    if length<=24000:return 0,length
    low,high=max(0,b-24000),min(a,length-24000)
    if low>high:return None
    start=int(np.clip((length-24000)//2,low,high))
    return start,start+24000


def speed_wave(wave,rate,protected):
    expected=round(len(wave)/rate)
    crop=speed_crop(expected,rate,protected)
    if crop is None:return wave.copy(),dict(rate=rate,applied_rate=1.,skipped=True,padding=0)
    lib=library()
    state=lib.rubberband_new(16000,1,0x00010000|0x01000000,1/rate,1.)
    if not state:raise RuntimeError("Rubber Band initialization failed")
    output=[]
    try:
        lib.rubberband_set_max_process_size(state,512)
        lib.rubberband_set_expected_input_duration(state,len(wave))
        for offset in range(0,len(wave),512):
            chunk=np.ascontiguousarray(wave[offset:offset+512])
            lib.rubberband_study(state,pointer(chunk),len(chunk),int(offset+512>=len(wave)))
        for offset in range(0,len(wave),512):
            chunk=np.ascontiguousarray(wave[offset:offset+512])
            lib.rubberband_process(state,pointer(chunk),len(chunk),int(offset+512>=len(wave)))
            while (count:=lib.rubberband_available(state))>0:
                result=np.empty(count,dtype=np.float32)
                received=lib.rubberband_retrieve(state,pointer(result),count)
                output.append(result[:received])
        result=np.concatenate(output)
        assert len(result)>=expected
        result=result[:expected]
    finally:lib.rubberband_delete(state)
    crop=speed_crop(len(result),rate,protected)
    assert crop is not None
    result=result[crop[0]:crop[1]]
    missing=24000-len(result)
    result=np.pad(result,(missing//2,missing-missing//2))
    gain=min(1.,.99/max(float(np.abs(result).max()),1e-9))
    result=(result*gain).astype(np.float32)
    assert result.shape==(24000,) and np.isfinite(result).all()
    return result,dict(rate=rate,applied_rate=rate,skipped=False,padding=missing,crop=list(crop),peak_gain=gain)


def speed_bank(frame,cfg):
    directory=STUDY/"speed_bank";directory.mkdir(parents=True,exist_ok=True)
    target=directory/"waves.npy";manifest=directory/"manifest.json"
    recipes=pd.read_csv(Path(cfg.data_root)/"preprocessing_recipe.csv").set_index("path")
    pins={p:file_hash(Path(cfg.data_root)/p) for p in frame.path}
    code=file_hash(Path(__file__))
    if manifest.exists():
        saved=json.loads(manifest.read_text())
        assert saved["audio_sha256"]==pins and saved["source_sha256"]==code
        assert saved["bank_sha256"]==file_hash(target)
        assert saved["paths"]==frame.path.tolist()
        return np.load(target,mmap_mode="r"),saved
    bank=np.lib.format.open_memmap(target,mode="w+",dtype=np.float32,shape=(2,len(frame),24000))
    traces=[]
    for i,row in enumerate(frame.itertuples()):
        wave,rate=sf.read(Path(cfg.data_root)/row.path,dtype="float32")
        assert rate==16000 and wave.shape==(24000,)
        rec=recipes.loc[row.path]
        protected=(int(rec.pad_left_samples),24000-int(rec.pad_right_samples))
        for variant in range(2):
            if row.label=="silence":
                result,trace=wave.copy(),dict(skipped=True,reason="silence",rate=1.,applied_rate=1.,padding=0)
            else:
                token=f"waveform-first-bank-v1|20261111|{row.path}|{variant}|speed"
                rng=np.random.default_rng(int.from_bytes(hashlib.sha256(token.encode()).digest()[:8],"little"))
                result,trace=speed_wave(wave,float(rng.uniform(.9,1.1)),protected)
            bank[variant,i]=result
            traces.append(dict(path=row.path,variant=variant,protected=list(protected),**trace,
                               waveform_sha256=hashlib.sha256(result.tobytes()).hexdigest()))
        if (i+1)%100==0:print("SPEED_BANK",i+1,len(frame),flush=True)
    bank.flush()
    saved=dict(paths=frame.path.tolist(),audio_sha256=pins,source_sha256=code,
               protection="entire reviewed non-padding interval in ref; conservative proxy, not newly annotated speech boundary",
               variants=2,bank_sha256=file_hash(target),traces=traces)
    write_json(manifest,saved)
    return np.load(target,mmap_mode="r"),saved


def draw_rng(seed,speaker,epoch,draw,component):
    return np.random.default_rng(np.random.SeedSequence([seed,int(speaker[-2:]),epoch,draw,component]))


class TrainingClips(Dataset):
    def __init__(self,frame,cfg,speaker,speed,mapping,noise_sources,recordings,legacy=False,noise=False):
        self.frame=frame.reset_index(drop=True)
        self.labels={v:i for i,v in enumerate(cfg.labels)}
        self.targets=[self.labels[label] for label in self.frame.label]
        self.wave=np.stack([sf.read(Path(cfg.data_root)/p,dtype="float32")[0] for p in self.frame.path])
        self.speed,self.mapping=speed,mapping
        self.sources,self.recordings=noise_sources,recordings
        self.speaker,self.legacy,self.noise=speaker,legacy,noise
        rms=np.sqrt(np.mean(self.wave[np.array(self.targets)!=10].astype(np.float64)**2,axis=1))
        assert len(rms)==600
        self.reference_power=float(np.median(rms)**2)
        self.trace=[]
        self.set_epoch(0)

    def set_epoch(self,epoch):self.epoch,self.draw=epoch,0

    def __len__(self):return len(self.frame)

    def __getitem__(self,index):
        draw=self.draw;self.draw+=1
        path=self.frame.iloc[index].path
        wave=self.wave[index].copy()
        speed_rng=draw_rng(42,self.speaker,self.epoch,draw,1)
        speed_on=bool(self.legacy and speed_rng.random()<.5)
        variant=int(speed_rng.integers(2))
        if speed_on:wave=self.speed[variant,self.mapping[path]].copy()
        noise_rng=draw_rng(42,self.speaker,self.epoch,draw,2)
        noise_on=bool(self.noise and noise_rng.random()<.5)
        trace=dict(epoch=self.epoch,draw=draw,path=path,speed_selected=speed_on,speed_variant=variant,noise_selected=noise_on)
        if noise_on:
            source=self.sources[int(noise_rng.integers(len(self.sources)))]
            recording=self.recordings[source["noise_id"]]
            for retry in range(1000):
                start=int(noise_rng.integers(0,len(recording)-24000+1))
                noise=recording[start:start+24000].astype(np.float64)
                noise-=noise.mean()
                if np.mean(noise**2)>=1e-8:break
            else:raise ValueError("No energetic training-noise crop")
            level=float(noise_rng.uniform(-5,20))
            power=self.reference_power if self.targets[index]==10 else float(np.mean(wave.astype(np.float64)**2))
            wave,audit=mix_at_snr(wave,noise,level,power)
            trace.update(noise_id=source["noise_id"],start_sample=start,crop_retry=retry,**audit)
        trace["input_sha256"]=hashlib.sha256(wave.tobytes()).hexdigest()
        self.trace.append(trace)
        return torch.from_numpy(wave),self.targets[index],path


def mask_features(features,floor,seed,speaker,epoch,batch):
    masked=features.clone()
    rng=draw_rng(seed,speaker,epoch,batch,3)
    trace=[]
    for item in range(len(features)):
        if rng.random()>=.5:continue
        tw=int(rng.integers(2,6));ts=int(rng.integers(features.shape[1]-tw+1))
        fw=int(rng.integers(3,6));fs=int(rng.integers(features.shape[2]-fw+1))
        masked[item,ts:ts+tw]=floor
        masked[item,:,fs:fs+fw]=floor[fs:fs+fw]
        trace.append(dict(item=item,time_start=ts,time_width=tw,freq_start=fs,freq_width=fw))
    return masked,trace
