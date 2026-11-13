"""Keep the previous formant/noise draws; add an independent speech-local pitch stream."""
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.conformer_aug.augment import draw_rng
from experiments.conformer_noise_focus.augment import FocusTrainingClips


class LegacyFormants:
    def __init__(self, root, metadata):
        root = Path(root)
        frame = metadata[metadata.group.ne("speaker:spk01") & metadata.label.ne("silence")].reset_index(drop=True)
        self.indices = {path: i for i, path in enumerate(frame.path)}
        self.waves = np.load(root / "waves.npy", mmap_mode="r")
        self.records = {(r["path"], r["variant"]): r for r in json.loads((root / "rows.json").read_text(encoding="utf-8")) if r["flags"] == 4}

    def get(self, path, variant):
        wave = self.waves[self.indices[path], 3, variant]
        assert hashlib.sha256(wave.tobytes()).hexdigest() == self.records[(path, variant)]["output_sha256"]
        return wave.copy()


class Training(FocusTrainingClips):
    def __init__(self, *args, formants, pitch_bank=None, pitch=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.formants = formants
        self.pitch_bank = pitch_bank
        self.pitch_enabled = pitch

    def __getitem__(self, index):
        draw = self.draw
        path = self.frame.iloc[index].path
        word = self.targets[index] != 10
        formant_on = bool(word and draw_rng(42, self.speaker, self.epoch, draw, 6).random() < .5)
        formant_variant = int(draw_rng(42, self.speaker, self.epoch, draw, 8).integers(2))
        pitch_on = bool(word and self.pitch_enabled and draw_rng(42, self.speaker, self.epoch, draw, 7).random() < .5)
        pitch_variant = int(draw_rng(42, self.speaker, self.epoch, draw, 9).integers(8))
        changed = pitch_on or formant_on
        original = self.wave[index].copy() if changed else None
        try:
            if pitch_on:
                self.wave[index] = self.pitch_bank.get(path, pitch_variant, formant_variant + 1 if formant_on else 0)
            elif formant_on:
                self.wave[index] = self.formants.get(path, formant_variant)
            output = super().__getitem__(index)
        finally:
            if changed:
                self.wave[index] = original
        self.trace[-1].update(pitch_selected=pitch_on, pitch_variant=pitch_variant,
                             formant_selected=formant_on, formant_variant=formant_variant,
                             speed_selected=False)
        return output
