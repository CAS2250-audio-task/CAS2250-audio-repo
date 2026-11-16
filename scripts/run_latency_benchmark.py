import json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import soundfile as sf
import torchaudio
from sklearn.metrics import accuracy_score, f1_score

# 1. Patch config.py, model.py, engine.py to support backbone="resnet18" | "mobilenet_v3_small"
cfg_p = Path("src/config.py")
cfg_txt = cfg_p.read_text(encoding="utf-8")
if "backbone:" not in cfg_txt:
    cfg_txt = cfg_txt.replace(
        '    label_smoothing: float = 0.0\n',
        '    label_smoothing: float = 0.0\n    backbone: str = "resnet18"\n',
        1,
    )
    cfg_p.write_text(cfg_txt, encoding="utf-8")

Path("src/model.py").write_text('''"""ResNet-18 or MobileNetV3-Small (1-channel or 3-channel); optional ImageNet initialization."""
import torch.nn as nn
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    mobilenet_v3_small,
    resnet18,
)


def build_model(num_classes=11, use_pretrained=True, in_channels=1, backbone="resnet18"):
    if backbone == "mobilenet_v3_small":
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT if use_pretrained else None)
        if in_channels != 3:
            first_conv = model.features[0][0]
            old_w = first_conv.weight.detach().mean(dim=1, keepdim=True)
            new_conv = nn.Conv2d(in_channels, first_conv.out_channels, kernel_size=3, stride=2, padding=1, bias=False)
            if use_pretrained:
                new_conv.weight.data.copy_(old_w.repeat(1, in_channels, 1, 1) / in_channels)
            model.features[0][0] = new_conv
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model

    model = resnet18(weights=ResNet18_Weights.DEFAULT if use_pretrained else None)
    if in_channels != 3:
        old_weight = model.conv1.weight.detach().mean(dim=1, keepdim=True)
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if use_pretrained:
            model.conv1.weight.data.copy_(old_weight.repeat(1, in_channels, 1, 1) / in_channels)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
''', encoding="utf-8")

eng_p = Path("src/engine.py")
eng_txt = eng_p.read_text(encoding="utf-8")
if "cfg.backbone" not in eng_txt:
    eng_txt = eng_txt.replace(
        "build_model(len(cfg.labels), cfg.use_pretrained, in_channels=in_ch)",
        "build_model(len(cfg.labels), cfg.use_pretrained, in_channels=in_ch, backbone=cfg.backbone)",
        1,
    ).replace(
        "build_model(len(cfg.labels), use_pretrained=False, in_channels=in_ch)",
        "build_model(len(cfg.labels), use_pretrained=False, in_channels=in_ch, backbone=cfg.backbone)",
        1,
    )
    eng_p.write_text(eng_txt, encoding="utf-8")

print("Patched backbone support successfully.")
