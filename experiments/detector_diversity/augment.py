"""Length-preserving shared waveform augmentation; no speaker transforms."""
import math
import numpy as np
import torch
from scipy.signal import lfilter
from torch.utils.data import Dataset
from experiments.boundary_detector.data import SR,frame_labels
from experiments.boundary_detector.train import mix

class Windows(Dataset):
    def __init__(self,records,noises,families,reference,hop=80,seconds=1.5,seed=42):
        self.records=records;self.noises=noises;self.reference=reference;self.hop=hop;self.samples=round(seconds*SR);self.seed=seed;self.epoch=0;self.context_samples=round(.08*SR)
        self.families=[np.array([i for i,f in enumerate(families) if f==family]) for family in sorted(set(families))]
    def __len__(self):return 4*len(self.records)
    def noise(self,length,rng):
        family=self.families[int(rng.integers(len(self.families)))];index=int(rng.choice(family));n=self.noises[index]
        if len(n)<length:n=np.tile(n,math.ceil(length/len(n)))
        for _ in range(50):
            start=int(rng.integers(len(n)-length+1));out=n[start:start+length].astype(np.float64,copy=True);out-=out.mean()
            if np.mean(out*out)>1e-9:break
        else:raise ValueError('No energetic train noise crop')
        if rng.random()<.3:
            envelope=np.interp(np.arange(length),np.linspace(0,length-1,3),rng.uniform(-3,3,3));out*=10**(envelope/20)
        return out
    def snr(self,rng):
        # Positive-window overall rates: original20%, 5..20dB40%, 0..5dB30%, -5..0dB10%.
        bucket=int(rng.choice(3,p=[.5,.375,.125]));return float(rng.uniform(*[(5,20),(0,5),(-5,0)][bucket]))
    def __getitem__(self,index):
        r,original=self.records[index//4];kind=index%4;rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch,index]))
        segments=r['segments'];wave=original
        negative=kind==3
        if negative:wave=np.zeros(max(len(original),self.samples),np.float32);segments=[]
        if negative or rng.random()<.8:
            support=np.ones(len(wave),bool) if not segments else np.zeros(len(wave),bool)
            for a,b in segments:support[round(a*SR):round(b*SR)]=True
            # Transient recordings may be exactly silent where speech occurs.
            # Resample rather than inventing an infinite gain for a nominal SNR.
            for attempt in range(100):
                n=self.noise(len(wave),rng)
                if rng.random()<.2:
                    n2=self.noise(len(wave),rng);n+=n2*np.sqrt(np.mean(n*n)/np.mean(n2*n2))*10**(rng.uniform(-6,0)/20)
                if np.mean(n[support]**2)>1e-10:break
            else:raise ValueError('No train noise energy over annotated speech region')
            wave=mix(wave,n,segments,rng,self.snr(rng),self.reference)
        if rng.random()<.5:
            alpha=math.exp(-2*math.pi*1500/SR);low=lfilter([1-alpha],[1,-alpha],wave)
            gains=10**(rng.uniform(-3,3,2)/20);wave=(gains[0]*low+gains[1]*(wave-low)).astype(np.float32)
        if rng.random()<.3:
            envelope=10**(np.interp(np.arange(len(wave)),np.linspace(0,len(wave)-1,3),rng.uniform(-2,2,3))/20);wave=(wave*envelope).astype(np.float32)
        if segments and kind in [0,1]:
            seg=segments[int(rng.integers(len(segments)))];start=round(seg[kind]*SR)-int(rng.uniform(.2,.8)*self.samples)
        else:start=int(rng.integers(min(0,len(wave)-self.samples),max(0,len(wave)-self.samples)+1))
        x=np.zeros(self.samples+self.context_samples,np.float32);lo=max(0,start);hi=min(len(wave),start+len(x))
        if hi>lo:x[lo-start:hi-start]=wave[lo:hi]
        gain=10**(float(rng.uniform(-6,3))/20);x*=min(gain,.99/max(float(np.max(np.abs(x))),1e-12))
        centers=(start+np.arange(self.samples//self.hop+1)*self.hop)/SR;y=frame_labels(dict(segments=segments),centers)
        valid=(centers>=0)&(centers<len(wave)/SR);weights=np.ones(len(centers),np.float32);boundaries=np.zeros((2,len(centers)),np.float32)
        for a,b in segments:
            weights[(centers>=a)&(centers<min(b,a+.05))]=3;weights[(centers>=max(a,b-.05))&(centers<b)]=5
            for j,t in enumerate([a,b]):
                # Only supervise a boundary when its annotated time is inside this window and recording.
                if centers[0]<=t<=centers[-1]:boundaries[j]=np.maximum(boundaries[j],np.exp(-.5*((centers-t)/.01)**2).astype(np.float32))
        return tuple(torch.from_numpy(z) for z in [x,y,valid,weights,boundaries])
