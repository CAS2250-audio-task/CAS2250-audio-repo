"""Optional supervised contrastive training, with a paired-CE control.

Original recordings may already contain noise. 'Reference' denotes the same
formant-transformed recording before additional synthetic noise, not clean truth.
The projection head is used only during training and discarded for inference.
"""
import torch
from torch import nn
from torch.nn import functional as F


class ProjectionHead(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Linear(32,64),nn.ReLU(),nn.Linear(64,32))


def supervised_contrastive(projected, targets, temperature=.1, positive_mode="same_command"):
    """Class-aware positives, self excluded; silence stays in CE only."""
    assert positive_mode in ("same_recording", "same_command")
    selected=targets.ne(10)
    z=F.normalize(projected[selected],dim=1)
    labels=targets[selected]
    if len(z)<2:
        return projected.sum()*0.
    sim=(z@z.T)/temperature
    eye=torch.eye(len(z),dtype=torch.bool,device=z.device)
    same_label=labels[:,None].eq(labels[None,:])
    if positive_mode=="same_recording":
        assert len(targets)%2==0, "Expected [reference batch, noisy batch]"
        origins=torch.arange(len(targets)//2,device=targets.device).repeat(2)[selected]
        positive=origins[:,None].eq(origins[None,:]) & ~eye
    else:
        positive=same_label & ~eye
    # Other recordings of the same command are ignored in noise-only mode,
    # never treated as negatives. Command mode explicitly attracts them.
    denominator=((~same_label)|positive) & ~eye
    valid=positive.any(1)
    if not valid.any():
        return projected.sum()*0.
    sim=sim-sim.max(1,keepdim=True).values.detach()
    log_denom=torch.logsumexp(sim.masked_fill(~denominator,-torch.inf),dim=1,keepdim=True)
    log_prob=sim-log_denom
    mean_positive=(log_prob*positive).sum(1)/positive.sum(1).clamp_min(1)
    return -mean_positive[valid].mean()


def paired_loss(model, projector, reference, noisy, labels, weight=.1, temperature=.1, positive_mode="same_command"):
    batch=torch.cat((reference,noisy))
    target=torch.cat((labels,labels))
    embedding=model.readout(model.backbone(batch))
    logits=model.classifier(embedding)
    ce=F.cross_entropy(logits,target)
    contrast=supervised_contrastive(projector(embedding),target,temperature,positive_mode) if projector is not None else ce*0.
    return ce+weight*contrast,logits[len(labels):],ce.detach(),contrast.detach()
