"""Matched readouts; frequency processing lives inside backbone for paired_loss."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from experiments.compact_pooling.models import (
    SmallAttention, SmallTransformerBlock, position_encoding,
)
from experiments.local_features.models import variant_models

CNNS = ('basic', 'lowcost', 'frequency')


class SDPAttention(SmallAttention):
    """Same projections as SmallAttention; fused/memory-efficient GPU dispatch."""
    def forward(self, x):
        b, n, d = x.shape
        def split(layer):
            return layer(x).view(b, n, self.heads, self.head_width).transpose(1, 2)
        value = F.scaled_dot_product_attention(split(self.q), split(self.k), split(self.v))
        return self.output(value.transpose(1, 2).reshape(b, n, d))


class FrequencyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(1, 16)
        self.block = SmallTransformerBlock(16, 32, 2)
        # Preserve projection values and state keys when switching attention kernels.
        attention = SDPAttention(16, 2)
        attention.load_state_dict(self.block.attention.state_dict())
        self.block.attention = attention
        self.output = nn.Linear(16, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        b, f, t = x.shape
        tokens = self.embedding(x.transpose(1, 2).reshape(b*t, f, 1))
        tokens = tokens + position_encoding(f, 16, x.device, x.dtype)
        delta = self.output(self.block(tokens)).reshape(b, t, f).transpose(1, 2)
        return x + delta


class FrequencyFrontend(nn.Module):
    """248-parameter residual 2D frontend; low parameter cost, not free latency."""
    def __init__(self):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8), nn.ReLU(),
            nn.Conv2d(8, 8, 3, padding=1, groups=8, bias=False),
            nn.Conv2d(8, 8, 1, bias=False), nn.BatchNorm2d(8), nn.ReLU())
        self.project = nn.Conv2d(8, 1, 1, bias=False)
        nn.init.zeros_(self.project.weight)

    def forward(self, x):
        return x + self.project(self.local(x[:, None]))[:, 0]


class AxisBlock(nn.Module):
    def __init__(self, ci, co, kernel, stride=1, dilation=1):
        super().__init__()
        self.freq = nn.Sequential(nn.Conv2d(ci, ci, (3, 1), stride=(stride, 1),
            padding=(1, 0), groups=ci, bias=False), nn.BatchNorm2d(ci), nn.ReLU())
        self.mix = nn.Sequential(nn.Conv2d(ci, co, 1, bias=False), nn.BatchNorm2d(co), nn.ReLU())
        self.time = nn.Sequential(nn.Conv2d(co, co, (1, kernel),
            padding=(0, dilation*(kernel-1)//2), dilation=(1, dilation), groups=co,
            bias=False), nn.BatchNorm2d(co))
        self.skip = nn.Sequential(nn.Conv2d(ci, co, 1, stride=(stride, 1), bias=False), nn.BatchNorm2d(co))

    def forward(self, x):
        return F.relu(self.time(self.mix(self.freq(x))) + self.skip(x))


class FrequencyTimeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Temporal kernel1 keeps the same 109-frame temporal support as local.
        self.stem = nn.Sequential(nn.Conv2d(1, 24, (5, 1), stride=(2, 1),
            padding=(2, 0), bias=False), nn.BatchNorm2d(24), nn.ReLU())
        self.blocks = nn.ModuleList([AxisBlock(24, 32, 11, 2), AxisBlock(32, 48, 13, 2),
            AxisBlock(48, 64, 15), AxisBlock(64, 96, 17), AxisBlock(96, 128, 29, dilation=2)])
        self.final = nn.Sequential(nn.Conv2d(128, 128, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU())
        self.projection = nn.Linear(128, 32)
        self.local_projection = nn.Linear(48, 32)
        nn.init.zeros_(self.local_projection.weight)
        nn.init.zeros_(self.local_projection.bias)

    def forward(self, x):
        x = self.stem(x[:, None])
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == 1:
                local = self.local_projection(x.mean(2).transpose(1, 2))
        return self.projection(self.final(x).mean(2).transpose(1, 2)) + local


def grid_models(seed=42, cnn='basic', vertical=False):
    if cnn not in CNNS:
        raise ValueError(cnn)
    models = variant_models(seed, 'local')
    if cnn == 'basic' and not vertical:
        return models  # Exact existing state layout/checkpoint compatibility.
    torch.manual_seed(seed + 1000)
    if cnn == 'frequency':
        core = FrequencyTimeCNN()
    else:
        core = copy.deepcopy(next(iter(models.values())).backbone)
    torch.manual_seed(seed + 1001)
    frontend = FrequencyFrontend() if cnn == 'lowcost' else nn.Identity()
    torch.manual_seed(seed + 1002)
    attention = FrequencyAttention() if vertical else nn.Identity()
    backbone = nn.Sequential(frontend, attention, core)
    for model in models.values():
        model.backbone = copy.deepcopy(backbone)
    return models
