"""Preserve the reference ResNet tail and 512 -> 11 head; fuse at layer1."""
import hashlib
import torch
from torch import nn
from src.model import build_model


class AuxiliaryCNN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        layers = []
        for cin, cout, dilation in [(channels, 32, 1), (32, 64, 2), (64, 64, 4)]:
            layers += [nn.Conv1d(cin, cout, 5, padding=2*dilation, dilation=dilation),
                       nn.GroupNorm(4 if cout == 32 else 8, cout), nn.GELU()]
        self.layers = nn.Sequential(*layers)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        return self.dropout(self.layers(x))


def aligned_layer1(aux, time_steps):
    """Symmetric ResNet conv7/s2/p3 + maxpool3/s2/p1: centre j = input 4*j.

    layer1 changes neither stride nor frame-centre offset. No resize/interpolation
    across missing frames and no arbitrary four-bin pooling are used.
    """
    result = aux[..., ::4]
    if result.shape[-1] != time_steps:
        raise ValueError("ResNet and auxiliary time centres disagree")
    return result


class AcousticModel(nn.Module):
    def __init__(self, channels=0, pretrained=True, fusion="middle", control_extra=0):
        super().__init__()
        # Construct this FIRST in every set to keep identical backbone initialization.
        self.backbone = build_model(11, pretrained)
        self.fusion = fusion
        self.channels = channels
        self.aux = AuxiliaryCNN(channels) if channels else None
        if channels and fusion == "middle":
            self.mix = nn.Conv2d(128, 64, 1)
            # The initial network is exactly the baseline; auxiliary influence is learned.
            nn.init.zeros_(self.mix.weight)
            nn.init.zeros_(self.mix.bias)
            with torch.no_grad():
                self.mix.weight[:, :64, 0, 0].copy_(torch.eye(64))
        elif channels and fusion == "late":
            # Global pooling is the late-fusion control, not the rejected 4-bin default.
            self.aux_projection = nn.Linear(64, 64)
            self.late_head = nn.Linear(576, 11)
            with torch.no_grad():
                self.late_head.weight.zero_()
                self.late_head.weight[:, :512].copy_(self.backbone.fc.weight)
                self.late_head.bias.copy_(self.backbone.fc.bias)
        elif channels:
            raise ValueError(f"Unknown fusion {fusion}")
        if control_extra:
            # A trainable spectral-only temporal residual, matched to extra parameter budget.
            width = max(1, round((control_extra - 64) / 129))
            self.spectral_control = nn.Sequential(nn.Conv2d(64, width, 1), nn.GELU(),
                                                  nn.Conv2d(width, 64, 1))
            nn.init.zeros_(self.spectral_control[-1].weight)
            nn.init.zeros_(self.spectral_control[-1].bias)
        else:
            self.spectral_control = None

    def forward_features(self, spec, auxiliary=None):
        m = self.backbone
        x = m.layer1(m.maxpool(m.relu(m.bn1(m.conv1(spec)))))
        if self.aux is not None and self.fusion == "middle":
            a = aligned_layer1(self.aux(auxiliary), x.shape[-1])
            a = a[:, :, None, :].expand(-1, -1, x.shape[-2], -1)
            x = self.mix(torch.cat([x, a], dim=1))
        if self.spectral_control is not None:
            x = x + self.spectral_control(x)
        x = m.layer4(m.layer3(m.layer2(x)))
        return torch.flatten(m.avgpool(x), 1)

    def forward(self, spec, auxiliary=None):
        x = self.forward_features(spec, auxiliary)
        if self.aux is not None and self.fusion == "late":
            a = self.aux_projection(self.aux(auxiliary).mean(-1))
            return self.late_head(torch.cat([x, a], dim=1))
        return self.backbone.fc(x)


def state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
