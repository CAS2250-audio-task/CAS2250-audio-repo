"""One-channel ResNet-18; optional ImageNet initialization."""
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def build_model(num_classes=11, use_pretrained=False):
    model = resnet18(weights=ResNet18_Weights.DEFAULT if use_pretrained else None)
    old_weight = model.conv1.weight.detach().mean(dim=1, keepdim=True)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    if use_pretrained:
        model.conv1.weight.data.copy_(old_weight)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
