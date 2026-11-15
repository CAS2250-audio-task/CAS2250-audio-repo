"""Legacy set3 Burg formants/bandwidths, actual-input extraction, no pitch search."""
from concurrent.futures import ProcessPoolExecutor
from collections import OrderedDict
import hashlib
import multiprocessing as mp
import numpy as np

NAMES=['f1','f2','f3','b1','b2','b3']
CONFIG=dict(sample_rate=16000,samples=24000,hop=160,formant_ceiling=5500.,formant_window=.025,maximum_formants=5,support_rms_threshold=1e-8)


def extract(wave):
    import parselmouth
    x=np.asarray(wave,dtype=np.float64)
    assert x.shape==(24000,) and np.isfinite(x).all()
    out=np.full((6,151),np.nan,dtype=np.float32)
    if np.max(np.abs(x))>1e-8 and np.std(x)>1e-8:
        formants=parselmouth.Sound(x,16000).to_formant_burg(time_step=.01,max_number_of_formants=5,maximum_formant=5500.,window_length=.025,pre_emphasis_from=50.)
        times=np.arange(151)*.01
        for k in (1,2,3):
            f=np.array([formants.get_value_at_time(k,float(t)) for t in times]);b=np.array([formants.get_bandwidth_at_time(k,float(t)) for t in times])
            good=np.isfinite(f)&(f>50)&(f<5450)
            out[k-1,good]=f[good]
            good=good&np.isfinite(b)&(b>0);out[k+2,good]=b[good]
    # Same 800-sample centered zero-padded support window as legacy apply_support.
    padded=np.pad(x,(400,400));starts=np.arange(151)*160
    support=np.array([np.sqrt(np.mean(padded[t:t+800]**2))>1e-8 for t in starts])
    out[:,~support]=np.nan
    valid=np.isfinite(out)
    return np.concatenate([np.where(valid,out,0),valid.astype(np.float32)],axis=0)


def fit_scaler(encoded):
    means=[];stds=[];counts=[]
    for j in range(6):
        values=encoded[:,j,:][encoded[:,j+6,:]>0]
        assert len(values)>0
        means.append(float(values.astype(np.float64).mean()));stds.append(max(float(values.astype(np.float64).std()),1e-6));counts.append(len(values))
    return dict(names=NAMES,mean=means,std=stds,observed_frames=counts,fitted_on='609 original training clips only; valid frames')


class Extractor:
    def __init__(self,workers=4):
        self.pool=ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'))
        self.cache=OrderedDict();self.hits=0;self.misses=0

    def __call__(self,waves):
        keys=[hashlib.sha256(np.asarray(x,dtype=np.float32).tobytes()).hexdigest() for x in waves]
        pending={k:x for k,x in zip(keys,waves) if k not in self.cache}
        fresh=dict(zip(pending,self.pool.map(extract,pending.values(),chunksize=8)))
        self.hits+=len(keys)-len(pending);self.misses+=len(pending)
        result=np.stack([self.cache[k] if k in self.cache else fresh[k] for k in keys])
        for k in keys:
            if k in self.cache:self.cache.move_to_end(k)
            else:self.cache[k]=fresh[k]
        while len(self.cache)>8192:self.cache.popitem(last=False)
        return result

    def close(self):self.pool.shutdown()
