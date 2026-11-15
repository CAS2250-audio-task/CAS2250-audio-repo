import copy
import torch
from torch import nn
from experiments.frequency_local_subset import model_factory

VARIANTS=('low8','low16','low_axis8','freq_avg','freq_band2','freq_capacity')


class Frontend(nn.Module):
    def __init__(self,width,kernels=((3,3),(3,3))):
        super().__init__()
        a,b=kernels
        self.layers=nn.Sequential(nn.Conv2d(1,width,a,padding=tuple(k//2 for k in a),bias=False),
            nn.BatchNorm2d(width),nn.ReLU(),
            nn.Conv2d(width,width,b,padding=tuple(k//2 for k in b),groups=width,bias=False),
            nn.Conv2d(width,width,1,bias=False),nn.BatchNorm2d(width),nn.ReLU())
        self.output=nn.Conv2d(width,1,1,bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self,x):return x+self.output(self.layers(x[:,None]))[:,0]


class BandRoutes(nn.Module):
    def __init__(self,base):
        super().__init__()
        self.stem,self.blocks,self.final=base.stem,base.blocks,base.final
        self.projection=nn.Linear(256,32)
        self.local_projection=nn.Linear(96,32)
        with torch.no_grad():
            for original,new in [(base.projection,self.projection),(base.local_projection,self.local_projection)]:
                new.weight.copy_(original.weight.repeat(1,2)/2)
                new.bias.copy_(original.bias)

    @staticmethod
    def pool(x):
        b,c,f,t=x.shape
        return x.reshape(b,c,2,f//2,t).mean(3).permute(0,3,2,1).reshape(b,t,2*c)

    def forward(self,x):
        x=self.stem(x[:,None])
        for i,block in enumerate(self.blocks):
            x=block(x)
            if i==1:local=self.local_projection(self.pool(x))
        return self.projection(self.pool(self.final(x)))+local


class CapacityControl(nn.Module):
    def __init__(self,base):
        super().__init__();self.base=base
        self.extra=nn.Sequential(nn.Linear(32,86),nn.ReLU(),nn.Linear(86,32))
        nn.init.zeros_(self.extra[-1].weight);nn.init.zeros_(self.extra[-1].bias)

    def forward(self,x):
        x=self.base(x);return x+self.extra(x)


def make_models(seed,variant):
    assert variant in VARIANTS
    cnn='lowcost' if variant.startswith('low') else 'frequency'
    models=model_factory(seed,cnn,False)
    backbone=copy.deepcopy(models['cnn_mean'].backbone)
    torch.manual_seed(seed+2000)
    if variant=='low16':backbone[0]=Frontend(16)
    if variant=='low_axis8':backbone[0]=Frontend(8,((5,1),(1,3)))
    if variant=='freq_band2':backbone[2]=BandRoutes(backbone[2])
    if variant=='freq_capacity':backbone[2]=CapacityControl(backbone[2])
    for model in models.values():model.backbone=copy.deepcopy(backbone)
    return models
