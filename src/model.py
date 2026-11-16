"""ResNet-18 or MobileNetV3-Small (1-channel or 3-channel); optional ImageNet initialization."""
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
