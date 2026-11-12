"""Train-only formant changes and waveform-keyed, label-free acoustic caching."""
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch

from experiments.conformer_aug.augment import draw_rng
from experiments.conformer_noise_focus.augment import FocusTrainingClips
from .features import FeatureConfig,extract,encode


def shift_formants(wave,ratio):
    import parselmouth
    from parselmouth.praat import call
    x=np.asarray(wave,dtype=np.float32)
    sound=call(parselmouth.Sound(x,16000),"Change gender",65.,600.,float(ratio),0.,1.,1.)
    y=sound.values[0]
    assert sound.sampling_frequency==16000 and abs(len(y)-24000)<=1
    y=np.pad(y,(0,max(0,24000-len(y))))[:24000].astype(np.float64)
    assert np.isfinite(y).all()
    p0=np.mean(x.astype(np.float64)**2);p1=np.mean(y*y)
    if p0>0 and p1>0:y*=np.sqrt(p0/p1)
    peak=np.max(np.abs(y))
    if peak>.99:y*=.99/peak
    return y.astype(np.float32)


def formant_variant(wave,path,strength,variant):
    token=f"formant-only-v1|42|{path}|{strength}|{variant}"
    seed=int.from_bytes(hashlib.sha256(token.encode()).digest()[:8],"little")
    rng=np.random.default_rng(seed)
    maximum={"weak":1.1,"wide":1.2}[strength]
    ratio=float(np.exp(rng.uniform(-np.log(maximum),np.log(maximum))))
    return shift_formants(wave,ratio),ratio


class Training(FocusTrainingClips):
    def __init__(self,*args,formants=None,augmented=False,**kwargs):
        super().__init__(*args,**kwargs)
        self.formants=formants
        self.augmented=augmented

    def __getitem__(self,index):
        draw=self.draw
        word=self.targets[index]!=10
        rng=draw_rng(42,self.speaker,self.epoch,draw,6)
        active=self.formants is not None and word and rng.random()<.5
        variant=int(draw_rng(42,self.speaker,self.epoch,draw,8).integers(2))
        bundled=getattr(self,"augmented",False)
        pitch=bool(bundled and word and draw_rng(42,self.speaker,self.epoch,draw,7).random()<.5)
        speed=bool(bundled and word and draw_rng(42,self.speaker,self.epoch,draw,1).random()<.5)
        flags=int(pitch)+2*int(speed)+4*int(active)
        path=self.frame.iloc[index].path
        changed=bool(flags)
        original=self.wave[index].copy() if changed else None
        ratio=1.
        details={}
        try:
            if changed:
                if hasattr(self,"variants") and self.variants is not None:
                    wave,details=self.variants[(path,flags,variant)]
                    ratio=details["formant_ratio"]
                else:
                    wave,ratio=self.formants[(path,variant)]
                self.wave[index]=wave
            result=super().__getitem__(index)
        finally:
            if changed:self.wave[index]=original
        self.trace[-1].update(formant_selected=bool(active),formant_variant=variant,formant_ratio=ratio,
            pitch_selected=pitch,speed_selected=speed,waveform_flags=flags,
            pitch_anchors=str(details.get("pitch_anchors")),speed_applied_rate=details.get("speed",{}).get("applied_rate",1.))
        return result


def _extract_cached(task):
    wave,path=task
    path=Path(path)
    if not path.exists():
        raw,valid=extract(wave)
        tmp=path.with_name(path.stem+f".{os.getpid()}.partial.npz")
        np.savez_compressed(tmp,raw=raw,valid=valid)
        tmp.replace(path)
    with np.load(path,allow_pickle=False) as saved:
        return saved["raw"].copy(),saved["valid"].copy()


class FeatureCache:
    def __init__(self,directory,workers=8):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.version=hashlib.sha256(json.dumps(FeatureConfig().to_dict(),sort_keys=True).encode()).hexdigest()[:12]
        self.pool=ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context("spawn"))

    def path(self,wave):
        key=hashlib.sha256(np.asarray(wave,dtype=np.float32).tobytes()).hexdigest()
        folder=self.directory/key[:2];folder.mkdir(exist_ok=True)
        return folder/f"{self.version}_{key}.npz"

    def raw_many(self,waves):
        tasks=[(np.asarray(w,dtype=np.float32),str(self.path(w))) for w in waves]
        values=list(self.pool.map(_extract_cached,tasks,chunksize=1))
        return np.stack([v[0] for v in values]),np.stack([v[1] for v in values])

    def encoded(self,waves,scaler,set_id):
        raw,valid=self.raw_many(waves)
        return encode(raw,valid,scaler,set_id)

    def close(self):self.pool.shutdown(wait=True)
