import json
from pathlib import Path

p = Path("src/config.py")
txt = p.read_text(encoding="utf-8")
if "pre_emphasis" not in txt:
    txt = txt.replace(
        '    device: str = "auto"\n',
        '    device: str = "auto"\n    f_min: int = 40\n    pre_emphasis: float = 0.0\n    use_delta: bool = False\n    label_smoothing: float = 0.0\n',
        1,
    )
    p.write_text(txt, encoding="utf-8")

Path("src/model.py").write_text('''"""ResNet-18 (1-channel or 3-channel); optional ImageNet initialization."""
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def build_model(num_classes=11, use_pretrained=True, in_channels=1):
    model = resnet18(weights=ResNet18_Weights.DEFAULT if use_pretrained else None)
    if in_channels != 3:
        old_weight = model.conv1.weight.detach().mean(dim=1, keepdim=True)
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if use_pretrained:
            model.conv1.weight.data.copy_(old_weight.repeat(1, in_channels, 1, 1) / in_channels)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
''', encoding="utf-8")

p = Path("src/dataset.py")
txt = p.read_text(encoding="utf-8")
txt = txt.replace("f_min=40,", "f_min=cfg.f_min,", 1)
if "pre_emphasis" not in txt:
    txt = txt.replace(
        '        spec = ((self.to_db(self.mel(wav)) + 80.0) / 80.0).clamp(0, 1)\n',
        '        if self.cfg.pre_emphasis > 0:\n'
        '            wav = torch.cat([wav[..., :1], wav[..., 1:] - self.cfg.pre_emphasis * wav[..., :-1]], dim=-1)\n'
        '        spec = ((self.to_db(self.mel(wav)) + 80.0) / 80.0).clamp(0, 1)\n',
        1,
    )
    txt = txt.replace(
        '        return (spec - 0.5) / 0.25\n',
        '        norm = (spec - 0.5) / 0.25\n'
        '        if self.cfg.use_delta:\n'
        '            d1 = torchaudio.functional.compute_deltas(norm)\n'
        '            d2 = torchaudio.functional.compute_deltas(d1)\n'
        '            norm = torch.cat([norm, d1 * 2.0, d2 * 4.0], dim=0)\n'
        '        return norm\n',
        1,
    )
    p.write_text(txt, encoding="utf-8")

p = Path("src/engine.py")
txt = p.read_text(encoding="utf-8")
if "in_channels" not in txt:
    txt = txt.replace(
        '    model = build_model(len(cfg.labels), cfg.use_pretrained).to(device)\n',
        '    in_ch = 3 if cfg.use_delta else 1\n'
        '    model = build_model(len(cfg.labels), cfg.use_pretrained, in_channels=in_ch).to(device)\n',
        1,
    )
    txt = txt.replace(
        '            loss = F.cross_entropy(model(x), y)\n',
        '            loss = F.cross_entropy(model(x), y, label_smoothing=cfg.label_smoothing)\n',
        1,
    )
    txt = txt.replace(
        '    model = build_model(len(cfg.labels), use_pretrained=False).to(selected)\n',
        '    in_ch = 3 if cfg.use_delta else 1\n'
        '    model = build_model(len(cfg.labels), use_pretrained=False, in_channels=in_ch).to(selected)\n',
        1,
    )
    p.write_text(txt, encoding="utf-8")

base = json.loads(Path("configs/task2_real.json").read_text(encoding="utf-8"))
base["data_root"] = "../../datasets/task2/fixed-1p5-v1"

exps = {
    "exp1_aug": {**base, "use_augmentation": True},
    "exp2_filter": {**base, "pre_emphasis": 0.97, "f_min": 80},
    "exp3_delta3ch": {**base, "use_delta": True},
    "exp4_combo": {**base, "pre_emphasis": 0.97, "f_min": 80, "use_delta": True, "use_augmentation": True, "label_smoothing": 0.1, "epochs": 20},
}
for name, cfg in exps.items():
    Path(f"configs/seojoon/{name}.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
print("Patch & configs created successfully.")
