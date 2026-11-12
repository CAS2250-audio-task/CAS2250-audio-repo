"""Unchanged Conformer parameters; optional masks only during training."""
import torch

from experiments.conformer.model import ConformerClassifier
from .augment import mask_features


class AugmentedConformer(ConformerClassifier):
    def __init__(self,state,statistics,speaker,mask=False):
        super().__init__(state,statistics,mode="finetune")
        self.speaker,self.mask_enabled=speaker,mask
        self.epoch,self.mask_batch=0,0
        self.mask_trace=[]
        with torch.no_grad():
            zero=self.features(torch.zeros(1,24000))
        assert torch.equal(zero,zero[:,0:1].expand_as(zero))
        self.register_buffer("mask_floor",zero[0,0].clone(),persistent=False)
        self.features.register_forward_hook(self._training_mask)

    def set_epoch(self,epoch):self.epoch,self.mask_batch=epoch,0

    def _training_mask(self,module,inputs,features):
        if not self.training or not self.mask_enabled:return features
        masked,trace=mask_features(features,self.mask_floor,42,self.speaker,self.epoch,self.mask_batch)
        self.mask_trace.extend(dict(epoch=self.epoch,batch=self.mask_batch,**row) for row in trace)
        self.mask_batch+=1
        return masked
