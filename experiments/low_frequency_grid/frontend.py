import torch
from experiments.compact_pooling.run import Frontend as OriginalFrontend

ALPHAS={'standard':0.,'lowfocus':.2}


def configuration(mode):
    alpha=ALPHAS[mode]
    u=torch.linspace(0.,1.,66)
    lo=2595*torch.log10(torch.tensor(1+20/700))
    hi=2595*torch.log10(torch.tensor(1+7800/700))
    warped=(1-alpha)*u+alpha*u.square()
    points=700*(10**((lo+(hi-lo)*warped)/2595)-1)
    return dict(mode=mode,alpha=alpha,bands=64,f_min=20.,f_max=7800.,
        mapping='normalized mel coordinate u -> (1-alpha)*u+alpha*u^2',
        centers_below_1000=int((points[1:-1]<1000).sum()),
        centers_below_2000=int((points[1:-1]<2000).sum()),
        centers_above_4000=int((points[1:-1]>4000).sum()),
        filter_edges_hz=points.tolist(),manual_attention_prior=False,
        normalization='Per-band mean/std fitted from original training-fold audio only')


class Frontend(OriginalFrontend):
    def __init__(self,mode='standard'):
        super().__init__()
        if mode=='standard':return
        points=torch.tensor(configuration(mode)['filter_edges_hz'])
        frequencies=torch.linspace(0,8000,257)
        differences=points[1:]-points[:-1]
        slopes=points[None,:]-frequencies[:,None]
        bank=torch.maximum(torch.zeros(1),torch.minimum(
            -slopes[:,:-2]/differences[:-1],slopes[:,2:]/differences[1:]))
        assert bank.shape==(257,64) and (bank.max(0).values>0).all()
        self.mel.mel_scale.fb.copy_(bank)
