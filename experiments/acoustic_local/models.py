import torch
from torch import nn
from experiments.local_features.models import variant_models


class AcousticRoute(nn.Module):
    def __init__(self,base,scaler):
        super().__init__();self.main=base
        self.register_buffer('aux_mean',torch.tensor(scaler['mean'],dtype=torch.float32)[None,:,None])
        self.register_buffer('aux_std',torch.tensor(scaler['std'],dtype=torch.float32)[None,:,None])
        self.auxiliary=nn.Sequential(nn.Conv1d(12,16,5,padding=2),nn.GroupNorm(4,16),nn.GELU(),
            nn.Conv1d(16,32,5,padding=2),nn.GroupNorm(4,32),nn.GELU(),nn.Conv1d(32,32,1))
        nn.init.zeros_(self.auxiliary[-1].weight);nn.init.zeros_(self.auxiliary[-1].bias)

    def forward(self,x):
        assert x.shape[1]==76
        valid=x[:,70:76];values=((x[:,64:70]-self.aux_mean)/self.aux_std)*valid
        aux=torch.cat([values,valid],dim=1)
        return self.main(x[:,:64])+self.auxiliary(aux).transpose(1,2)


def acoustic_models(seed,scaler):
    models=variant_models(seed,'local')
    for i,model in enumerate(models.values()):
        # Keep old base initialization, give all three auxiliary branches identical tensors.
        torch.manual_seed(seed+700)
        model.backbone=AcousticRoute(model.backbone,scaler)
    return models
