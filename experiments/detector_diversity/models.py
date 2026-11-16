"""Matched small detectors: causal baseline, bounded lookahead, heads, spectral CNN."""
import torch
from torch import nn
from torch.nn import functional as F
from experiments.boundary_detector.model import CNNGRU

class SpectralStem(nn.Module):
    def __init__(self,width):
        super().__init__();self.c1=nn.Conv2d(1,8,3,stride=(2,1),bias=False);self.c2=nn.Conv2d(8,8,3,stride=(2,1),bias=False)
        self.n1=nn.LayerNorm(8*32);self.n2=nn.LayerNorm(8*16);self.proj=nn.Conv1d(8*16,width,1,bias=False)
    def stage(self,x,conv,norm):
        x=conv(F.pad(x,(2,0,1,1)))
        b,c,f,t=x.shape
        x=norm(x.permute(0,3,1,2).reshape(b,t,c*f)).reshape(b,t,c,f).permute(0,2,3,1)
        return F.silu(x)
    def forward(self,x):
        x=self.stage(x[:,None],self.c1,self.n1);x=self.stage(x,self.c2,self.n2)
        b,c,f,t=x.shape
        return F.silu(self.proj(x.reshape(b,c*f,t)))

class DiversityDetector(nn.Module):
    def __init__(self,width=64,lookahead_frames=0,boundary=False,spectral=False):
        super().__init__();self.core=CNNGRU(width);self.lookahead_frames=lookahead_frames
        if spectral:self.core.stem=SpectralStem(width)
        self.boundaries=nn.Linear(32,2) if boundary else None
    def forward_heads(self,x):
        length=x.shape[-1];k=self.lookahead_frames
        if k:x=F.pad(x,(0,k))
        x=self.core.local(self.core.stem(x)).transpose(1,2);x,_=self.core.gru(x);x=x[:,k:k+length]
        return self.core.head(x)[:,:,0],self.boundaries(x).transpose(1,2) if self.boundaries is not None else None
    def forward(self,x):return self.forward_heads(x)[0]

def make_model(variant='baseline',width=64,hop=80,enabled=None):
    flags=dict(lookahead=variant=='lookahead',boundary=variant=='boundary',spectral=variant=='spectral')
    if enabled is not None:flags={k:k in enabled for k in flags}
    return DiversityDetector(width,round(.08*16000/hop) if flags['lookahead'] else 0,flags['boundary'],flags['spectral'])
