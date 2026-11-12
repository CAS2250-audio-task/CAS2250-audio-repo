"""Reuse protected speed changes and the previous within-file pitch policy."""
import ctypes as ct
import hashlib

import numpy as np

from experiments.conformer_aug.augment import library,pointer,speed_wave


def pitch_wave(wave,curve):
    lib=library()
    lib.rubberband_get_latency.restype=ct.c_uint
    lib.rubberband_get_latency.argtypes=[ct.c_void_p]
    lib.rubberband_set_pitch_scale.restype=None
    lib.rubberband_set_pitch_scale.argtypes=[ct.c_void_p,ct.c_double]
    state=lib.rubberband_new(16000,1,0x00010000|0x01000000|0x00000001|0x04000000,1.,1.)
    if not state:raise RuntimeError("Pitch Rubber Band initialization failed")
    output=[]
    try:
        lib.rubberband_set_max_process_size(state,128)
        delay=int(lib.rubberband_get_latency(state))
        signal=np.r_[np.zeros(delay,np.float32),wave,np.zeros(delay,np.float32)]
        for offset in range(0,len(signal),128):
            position=np.clip((offset-delay)/max(1,len(wave)-1),0,1)
            shift=float(np.interp(position,[0,1/3,2/3,1],curve))
            lib.rubberband_set_pitch_scale(state,2**(shift/12))
            chunk=np.ascontiguousarray(signal[offset:offset+128])
            lib.rubberband_process(state,pointer(chunk),len(chunk),int(offset+128>=len(signal)))
            while (count:=lib.rubberband_available(state))>0:
                part=np.empty(count,dtype=np.float32)
                received=lib.rubberband_retrieve(state,pointer(part),count)
                output.append(part[:received])
        y=np.concatenate(output)
        assert len(y)>=delay+len(wave)
        return y[delay:delay+len(wave)].copy()
    finally:lib.rubberband_delete(state)


def settings(path,variant):
    def rng(component):
        token=f"waveform-first-bank-v1|20261111|{path}|{variant}|{component}"
        return np.random.default_rng(int.from_bytes(hashlib.sha256(token.encode()).digest()[:8],"little"))
    middle=rng("pitch").uniform(-1,1,size=2)
    if abs(middle[1]-middle[0])<.35:middle[1]=-np.sign(middle[0] or 1)*.75
    rate=float(rng("speed").uniform(.9,1.1))
    formant=float(np.exp(rng("formant").uniform(-np.log(1.1),np.log(1.1))))
    return [0.,*middle.tolist(),0.],rate,formant


def variants(wave,path,variant,protected):
    from .data import shift_formants
    curve,rate,ratio=settings(path,variant)
    results={}
    for p in (0,1):
        pitched=pitch_wave(wave,curve) if p else wave
        for s in (0,1):
            if s:x,speed=speed_wave(pitched,rate,protected)
            else:x,speed=pitched,dict(applied_rate=1.,skipped=False,padding=0)
            for f in (0,1):
                flags=p+2*s+4*f
                if not flags:continue
                y=shift_formants(x,ratio) if f else x.copy()
                gain=min(1.,.99/max(float(np.abs(y).max()),1e-9))
                y=np.asarray(y*gain,dtype=np.float32)
                assert y.shape==(24000,) and np.isfinite(y).all()
                results[flags]=(y,dict(pitch_anchors=curve if p else None,speed=speed,
                    formant_ratio=ratio if f else 1.,peak_gain=gain))
    return results
