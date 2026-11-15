"""Change final CNN context and/or add a zero-initialized local-feature residual."""
import math
import torch
from torch import nn
from experiments.compact_pooling.models import matched_models as original_models

VARIANTS=('baseline','short','local','both')


class FeatureRoutes(nn.Module):
    def __init__(self,base,short=False,local=False):
        super().__init__()
        self.encoder=base.encoder
        self.projection=base.projection
        if short:
            old=self.encoder[4].main[0]
            assert old.kernel_size==(29,) and old.dilation==(2,) and old.groups==64
            new=nn.Conv1d(64,64,9,padding=4,dilation=1,groups=64,bias=False)
            # Preserve the exact common initial weights. Center taps are rescaled
            # to the initialization variance for fan-in9 instead of fan-in29.
            with torch.no_grad():new.weight.copy_(old.weight[:,:,10:19]*math.sqrt(29/9))
            self.encoder[4].main[0]=new
        self.local_projection=nn.Linear(64,32) if local else None
        if self.local_projection is not None:
            nn.init.zeros_(self.local_projection.weight)
            nn.init.zeros_(self.local_projection.bias)

    def forward(self,x):
        local=None
        for index,block in enumerate(self.encoder):
            x=block(x)
            if index==1 and self.local_projection is not None:
                local=self.local_projection(x.transpose(1,2))
        result=self.projection(x.transpose(1,2))
        return result if local is None else result+local


def variant_models(seed=42,variant='baseline'):
    assert variant in VARIANTS
    models=original_models(seed)
    if variant=='baseline':return models
    for model in models.values():
        model.backbone=FeatureRoutes(model.backbone,short=variant in ('short','both'),local=variant in ('local','both'))
    return models


def variant_description(variant):
    return dict(name=variant,global_receptive_field_frames=61 if variant in ('short','both') else 109,
        global_receptive_field_ms=625 if variant in ('short','both') else 1105,
        local_receptive_field_frames=23 if variant in ('local','both') else None,
        local_receptive_field_ms=245 if variant in ('local','both') else None,
        local_connection='after CNN block2,64->32, add to final32D features before readout; projection initialized to zero' if variant in ('local','both') else None,
        short_change='Only block5 depthwise kernel29,dilation2,padding28 -> kernel9,dilation1,padding4' if variant in ('short','both') else None,
        kernel_initialization='Identical common weights; shortened kernel uses center9 taps of the original29, rescaled sqrt(29/9)',
        output_width=32,baseline='Previously completed same_command/paired_supcon spk03 1000epoch; reused after input/init checks')
