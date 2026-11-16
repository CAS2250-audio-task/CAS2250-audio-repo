"""Length-aware finite-record inference with explicit unavailable-future policy.

The trained core appends normalized zeros internally. Those mean average log-mel
energy, NOT acoustic silence. For lookahead models we fill each item's missing
future explicitly, then crop logits back to original frames. No frames or
predicted events are dropped, no labels/margins/thresholds are changed.
"""
import torch
from torch.nn import functional as F

def infer_logits(model,x,lengths=None,policy='hold',silence=None):
    if policy not in ['legacy_zero','hold','silence']:raise ValueError('Unknown EOF policy')
    k=getattr(model,'lookahead_frames',0)
    if not k or policy=='legacy_zero':return model(x)
    b,c,t=x.shape;lengths=[t]*b if lengths is None else [int(n) for n in lengths]
    if len(lengths)!=b or any(n<1 or n>t for n in lengths):raise ValueError('Invalid sequence lengths')
    extended=F.pad(x,(0,k))
    for i,n in enumerate(lengths):
        fill=x[i,:,n-1:n] if policy=='hold' else silence
        if fill is None:raise ValueError('Provide standardized acoustic silence')
        extended[i,:,n:n+k]=fill.reshape(c,1).expand(c,k)
    return model(extended)[:,:t]
