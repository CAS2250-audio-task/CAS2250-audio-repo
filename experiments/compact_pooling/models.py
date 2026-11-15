"""Common small CNN with independently trained mean, pooled-query and self-CLS readouts.

Accepts 64-channel acoustic features. No detector or pretrained weights.
The ST-inspired and Transformer readouts are compact custom configurations.
"""
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F


class TemporalBlock(nn.Module):
    def __init__(self, ci, co, kernel, dilation=1, residual=False, separable=True):
        super().__init__()
        if separable:
            layers = [nn.Conv1d(ci, ci, kernel, padding=dilation*(kernel-1)//2,
                                dilation=dilation, groups=ci, bias=False),
                      nn.Conv1d(ci, co, 1, bias=False)]
        else:
            layers = [nn.Conv1d(ci, co, kernel, padding=(kernel-1)//2, bias=False)]
        self.main = nn.Sequential(*layers, nn.BatchNorm1d(co))
        self.skip = nn.Sequential(nn.Conv1d(ci, co, 1, bias=False), nn.BatchNorm1d(co)) if residual else None

    def forward(self, x):
        y = self.main(x)
        if self.skip is not None:
            y = y + self.skip(x)
        return F.relu(y)


class CommonCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            TemporalBlock(64, 128, 11),
            TemporalBlock(128, 64, 13, residual=True),
            TemporalBlock(64, 64, 15, residual=True),
            TemporalBlock(64, 64, 17, residual=True),
            TemporalBlock(64, 128, 29, dilation=2),
            TemporalBlock(128, 128, 1, separable=False),
        )
        self.projection = nn.Linear(128, 32)

    def forward(self, x):
        return self.projection(self.encoder(x).transpose(1, 2))


def position_encoding(length, width, device, dtype):
    position = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    scale = torch.exp(torch.arange(0, width, 2, device=device, dtype=torch.float32) * (-math.log(10000.) / width))
    pe = torch.zeros(length, width, device=device)
    pe[:, 0::2] = torch.sin(position * scale)
    pe[:, 1::2] = torch.cos(position * scale)
    return pe.to(dtype)[None]


class SmallAttention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        assert width % heads == 0
        self.heads, self.head_width = heads, width // heads
        self.q = nn.Linear(width, width, bias=False)
        self.k = nn.Linear(width, width, bias=False)
        self.v = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width)

    def forward(self, x):
        batch, frames, width = x.shape
        def split(layer):
            return layer(x).view(batch, frames, self.heads, self.head_width).transpose(1, 2)
        q, k, v = split(self.q), split(self.k), split(self.v)
        weights = torch.softmax((q @ k.transpose(-2, -1)) / math.sqrt(self.head_width), dim=-1)
        result = (weights @ v).transpose(1, 2).reshape(batch, frames, width)
        return self.output(result)


class SmallTransformerBlock(nn.Module):
    def __init__(self, width, ff_width, heads):
        super().__init__()
        self.attention = SmallAttention(width, heads)
        self.norm1 = nn.LayerNorm(width, eps=1e-6)
        self.norm2 = nn.LayerNorm(width, eps=1e-6)
        self.ffn = nn.Sequential(nn.Linear(width, ff_width), nn.GELU(), nn.Linear(ff_width, width))

    def forward(self, x):
        x = self.norm1(x + self.attention(x))
        return self.norm2(x + self.ffn(x))


class MeanReadout(nn.Module):
    def forward(self, x):
        return x.mean(dim=1)


class PooledQueryReadout(nn.Module):
    """Single-head ST-inspired shared-projection pooled query, not exact paper reproduction."""
    def __init__(self, width=32):
        super().__init__()
        self.shared = nn.Linear(width, width, bias=False)

    def forward(self, x):
        kv = self.shared(x)
        q = self.shared(x.mean(dim=1))[:, None]
        weights = torch.softmax((q @ kv.transpose(-2, -1)) / math.sqrt(x.shape[-1]), dim=-1)
        return (weights @ kv)[:, 0]


class SelfCLSReadout(nn.Module):
    def __init__(self, width=32, ff_width=64, heads=2, depth=1):
        super().__init__()
        self.cls = nn.Parameter(torch.empty(1, 1, width))
        nn.init.trunc_normal_(self.cls, std=.02)
        self.blocks = nn.Sequential(*[SmallTransformerBlock(width, ff_width, heads) for _ in range(depth)])

    def forward(self, x):
        x = torch.cat((self.cls.expand(len(x), -1, -1), x), dim=1)
        x = x + position_encoding(x.shape[1], x.shape[2], x.device, x.dtype)
        return self.blocks(x)[:, 0]


class CNNComparator(nn.Module):
    def __init__(self, backbone, readout, classes=11):
        super().__init__()
        self.backbone = backbone
        self.readout = readout
        self.classifier = nn.Linear(32, classes)

    def forward(self, x):
        return self.classifier(self.readout(self.backbone(x)))


def matched_models(seed=42):
    torch.manual_seed(seed)
    backbone = CommonCNN()
    result = {}
    for index, (name, readout_type) in enumerate([
        ('cnn_mean', MeanReadout), ('cnn_pooled_query', PooledQueryReadout), ('cnn_self_cls', SelfCLSReadout)
    ]):
        torch.manual_seed(seed + 100 + index)
        result[name] = CNNComparator(copy.deepcopy(backbone), readout_type())
    return result


if __name__ == '__main__':
    torch.set_num_threads(2)
    models = matched_models()
    states = [m.backbone.state_dict() for m in models.values()]
    assert all(torch.equal(states[0][key], s[key]) for s in states[1:] for key in states[0])
    with torch.no_grad():
        for name, model in models.items():
            model.eval()
            for frames in (60, 151):
                output = model(torch.randn(2, 64, frames))
                assert output.shape == (2, 11) and torch.isfinite(output).all()
            print(name, sum(p.numel() for p in model.parameters()))
