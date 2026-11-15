"""Confirmation delays are backdated; default added crop margin is zero."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def decode(probabilities,hop_seconds,duration,on=.6,off=.4,on_ms=20.,off_ms=50.):
    if not 0<=off<=on<=1:raise ValueError('Require 0 <= off <= on <= 1')
    need_on=max(1,int(np.ceil(on_ms/1000/hop_seconds)));need_off=max(1,int(np.ceil(off_ms/1000/hop_seconds)))
    result=[];start=None;candidate=None;end=None
    for i,p in enumerate(probabilities):
        if i*hop_seconds>=duration:break
        if start is None:
            if p>=on:
                if candidate is None:candidate=i
                if i-candidate+1>=need_on:start=candidate;candidate=None
            else:candidate=None
        else:
            if p<off:
                if end is None:end=i
                if i-end+1>=need_off:result.append([start*hop_seconds,min(end*hop_seconds,duration)]);start=None;end=None;candidate=None
            else:end=None
    if start is not None:
        # Unconfirmed trailing dip remains inside speech; confirmed dip above is backdated.
        result.append([start*hop_seconds,duration])
    return [p for p in result if p[1]>p[0]]


def boundary_metrics(gold,pred,iou_threshold=.5,tolerance_ms=10.):
    if not gold or not pred:return dict(gold=len(gold),pred=len(pred),matched=0,start_errors=[],end_errors=[],start_cuts=0,end_cuts=0)
    iou=np.zeros((len(gold),len(pred)))
    for i,(a,b) in enumerate(gold):
        for j,(c,d) in enumerate(pred):iou[i,j]=max(0,min(b,d)-max(a,c))/(max(b,d)-min(a,c))
    ii,jj=linear_sum_assignment(-iou);pairs=[(gold[i],pred[j]) for i,j in zip(ii,jj) if iou[i,j]>=iou_threshold]
    return dict(gold=len(gold),pred=len(pred),matched=len(pairs),start_errors=[abs(p[0]-g[0]) for g,p in pairs],end_errors=[abs(p[1]-g[1]) for g,p in pairs],start_cuts=sum(p[0]-g[0]>tolerance_ms/1000 for g,p in pairs),end_cuts=sum(g[1]-p[1]>tolerance_ms/1000 for g,p in pairs))
