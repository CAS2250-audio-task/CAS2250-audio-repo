"""Frozen, batch-independent Silero probabilities from the actual classifier input."""
from collections import OrderedDict
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def wave_hash(wave):
    return hashlib.sha256(np.asarray(wave, dtype=np.float32).tobytes()).hexdigest()


def frame_gates(probabilities):
    """32ms VAD windows -> CNN centers at 0,40,...1480ms; 80ms boundary margin."""
    positions = (torch.arange(38, device=probabilities.device) * 640 / 512 - .5).clamp(0, 46)
    lower = positions.floor().long()
    upper = (lower + 1).clamp(max=46)
    fraction = positions - lower
    p = probabilities[:, lower] * (1 - fraction) + probabilities[:, upper] * fraction
    p = F.avg_pool1d(p[:, None], 3, stride=1, padding=1, count_include_pad=False)
    p = F.max_pool1d(p, 5, stride=1, padding=2)[:, 0]
    # No confident speech must not create an empty attention/mean or force silence.
    fallback = p.max(dim=1).values < .3
    p = torch.where(fallback[:, None], torch.ones_like(p), p)
    return .05 + .95 * p, fallback


class Detector:
    """Plain object: its frozen parameters are not part of the classifier optimizer."""
    def __init__(self, path, device="cuda"):
        self.path = Path(path)
        self.device = torch.device(device)
        self.model = torch.jit.load(str(path), map_location=self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.evaluation_cache = {}
        self.recent = OrderedDict()
        self.calls = 0

    @torch.no_grad()
    def raw(self, waves):
        assert waves.ndim == 2 and waves.shape[1] == 24000
        x = F.pad(waves.to(self.device, dtype=torch.float32), (0, 64))
        self.model.reset_states()
        values = [self.model(x[:, offset:offset + 512], 16000)
                  for offset in range(0, 24064, 512)]
        self.calls += len(waves)
        return torch.cat(values, dim=1)

    @torch.no_grad()
    def gates(self, waves):
        cpu = waves.detach().cpu().numpy()
        keys = [wave_hash(x) for x in cpu]
        values = [self.evaluation_cache.get(k, self.recent.get(k)) for k in keys]
        missing = [i for i, value in enumerate(values) if value is None]
        if missing:
            computed, _ = frame_gates(self.raw(waves[missing]))
            for index, value in zip(missing, computed.cpu().numpy()):
                values[index] = value.copy()
                self.recent[keys[index]] = value.copy()
            while len(self.recent) > 8192:
                self.recent.popitem(last=False)
        return torch.from_numpy(np.stack(values)).to(waves.device)

    def add_evaluation_bank(self, cases, waves, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        speaker = cases.heldout_speaker.iloc[0]
        target = folder / f"{speaker}.npy"
        if target.exists():
            gates = np.load(target)
        else:
            chunks = []
            for offset in range(0, len(waves), 32):
                x = torch.from_numpy(np.array(waves[offset:offset + 32])).to(self.device)
                g, _ = frame_gates(self.raw(x))
                chunks.append(g.cpu().numpy())
            gates = np.concatenate(chunks)
            partial = folder / f"{speaker}.partial.npy"
            np.save(partial, gates)
            partial.replace(target)
        assert gates.shape == (len(cases), 38) and np.isfinite(gates).all()
        for row in cases.itertuples():
            assert "spk01_" not in row.path
            assert wave_hash(waves[row.case_index]) == row.input_sha256
            self.evaluation_cache[row.input_sha256] = gates[row.case_index].copy()
        return gates
