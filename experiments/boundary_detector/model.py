"""Causal temporal CNN with channel-only normalization and no temporal pooling."""
import torch
from torch import nn
from torch.nn import functional as F
import torchaudio


class Frontend(nn.Module):
    def __init__(self,hop=160):
        super().__init__();self.hop=hop
        self.mel=torchaudio.transforms.MelSpectrogram(sample_rate=16000,n_fft=512,win_length=400,hop_length=hop,n_mels=64,f_min=20.,f_max=7800.,center=True,pad_mode='constant',power=2.)
    def forward(self,x):return self.mel(x).clamp_min(1e-10).log10()


class FrameNorm(nn.Module):
    def __init__(self,width):super().__init__();self.norm=nn.LayerNorm(width)
    def forward(self,x):return self.norm(x.transpose(1,2)).transpose(1,2)


class Block(nn.Module):
    def __init__(self,width,dilation):
        super().__init__();self.left=4*dilation
        self.dw=nn.Conv1d(width,width,5,dilation=dilation,groups=width,bias=False)
        self.pw=nn.Conv1d(width,width,1,bias=False);self.norm=FrameNorm(width)
    def forward(self,x):
        y=self.pw(F.silu(self.dw(F.pad(x,(self.left,0)))))
        return F.silu(x+self.norm(y))


class Detector(nn.Module):
    def __init__(self,width=64):
        super().__init__();self.stem=nn.Sequential(nn.Conv1d(64,width,1,bias=False),FrameNorm(width),nn.SiLU())
        self.blocks=nn.Sequential(*[Block(width,d) for d in [1,2,4,8]])
        self.head=nn.Conv1d(width,1,1)
    def forward(self,x):return self.head(self.blocks(self.stem(x)))[:,0]
