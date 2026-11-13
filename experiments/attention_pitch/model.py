"""Native encoder plus fixed soft key priors and speech-weighted temporal pooling."""
import numpy as np
import torch
from torch import nn

from experiments.overnight.model import AcousticConformer


def attention_bias(gates, heads=4):
    batch, time = gates.shape
    return gates.clamp(min=.05, max=1).log()[:, None, None, :].expand(
        batch, heads, time, time).reshape(batch * heads, time, time)


class MaskedConformer(AcousticConformer):
    def __init__(self, state, statistics, speaker, detector=None, masking=False):
        super().__init__(state, statistics, speaker, dropout=.2, set_id="0", augmented=False)
        self.detector = detector
        self.masking = masking
        self.pooling_only = False
        self.gate_override = None
        self.last_gates = None
        self.kept_layers = list(range(12))

    def prune(self, count):
        assert len(self.transformer.encoder.layers) == 12
        self.kept_layers = np.rint(np.linspace(0, 11, count)).astype(int).tolist()
        self.transformer.encoder.layers = nn.ModuleList(
            [self.transformer.encoder.layers[index] for index in self.kept_layers])

    def get_gates(self, wave):
        if self.gate_override is not None:
            result = self.gate_override(wave)
        else:
            assert self.detector is not None
            result = self.detector.gates(wave)
        assert result.shape == (len(wave), 38) and torch.isfinite(result).all()
        return result.clamp(.05, 1)

    def forward(self, wave):
        if not self.masking:
            return self.fc(self.encode(wave).mean(dim=1))
        gates = self.get_gates(wave)
        self.last_gates = gates.detach()
        if self.pooling_only:
            encoded = self.encode(wave)
        else:
            with torch.no_grad():
                mel = self.features(wave)
                normalized = (mel - self.feature_mean) / self.feature_std
            front = self.frontend(normalized)
            batch, time, frequency, channels = front.shape
            x = self.transformer.custom_src_module(front.reshape(batch, time, frequency * channels))
            pos = self.transformer.positional_encoding(x)
            encoded, _ = self.transformer.encoder(
                src=x, src_mask=attention_bias(gates), src_key_padding_mask=None,
                pos_embs=pos)
        pooled = (encoded * gates[:, :, None]).sum(dim=1) / gates.sum(dim=1, keepdim=True)
        return self.fc(pooled)


def optimizer_for(model):
    return torch.optim.AdamW([
        dict(params=list(model.frontend.parameters()) + list(model.transformer.parameters()), lr=1e-5),
        dict(params=model.fc.parameters(), lr=5e-4)], weight_decay=1e-4)


def initialize(model, checkpoint, blocks=12, resume=False):
    """Retain exact AdamW states for every surviving parameter, including pruned models."""
    if resume:
        if blocks != 12:
            model.prune(blocks)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer = optimizer_for(model)
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        model.load_state_dict(checkpoint["model"], strict=True)
        original = optimizer_for(model)
        original.load_state_dict(checkpoint["optimizer"])
        if blocks != 12:
            model.prune(blocks)
            optimizer = optimizer_for(model)
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    optimizer.state[parameter] = original.state[parameter]
            del original
        else:
            optimizer = original
    for index, group in enumerate(optimizer.param_groups):
        group["lr"] = 1e-5 if index == 0 else 5e-4
        group["initial_lr"] = group["lr"]
        group["weight_decay"] = 1e-4
    return optimizer
