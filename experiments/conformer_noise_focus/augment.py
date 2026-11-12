"""Keep original draws and wide SNR coverage; optionally add speech and near-zero draws."""
import hashlib

import numpy as np
import torch

from experiments.conformer_aug.augment import TrainingClips, draw_rng
from experiments.noise_stress.mixing import mix_at_snr


def energetic_crop(recording, rng):
    for retry in range(1000):
        start = int(rng.integers(0, len(recording)-24000+1))
        noise = recording[start:start+24000].astype(np.float64)
        noise -= noise.mean()
        if np.mean(noise**2) >= 1e-8:
            return noise, start, retry
    raise ValueError("No energetic speech-noise crop")


class FocusTrainingClips(TrainingClips):
    def __init__(self, frame, cfg, speaker, original_sources, added_sources, recordings,
                 add_speech=False, focus_zero=False):
        super().__init__(frame, cfg, speaker, None, {}, original_sources, recordings, legacy=False, noise=True)
        self.add_speech, self.focus_zero = add_speech, focus_zero
        self.added = added_sources
        self.families = sorted(set(s["family"] for s in added_sources))

    def __getitem__(self, index):
        draw = self.draw
        self.draw += 1
        path = self.frame.iloc[index].path
        wave = self.wave[index].copy()
        rng = draw_rng(42, self.speaker, self.epoch, draw, 2)
        noise_on = bool(rng.random() < .5)
        speech_rng = draw_rng(42, self.speaker, self.epoch, draw, 4)
        snr_rng = draw_rng(42, self.speaker, self.epoch, draw, 5)
        speech_selected = bool(speech_rng.random() < .5 and self.add_speech and noise_on)
        near_selected = bool(snr_rng.random() < .5 and self.focus_zero and noise_on)
        trace = dict(epoch=self.epoch, draw=draw, path=path, noise_selected=noise_on,
                     added_speech_selected=speech_selected, near_zero_selected=near_selected)
        if noise_on:
            # Always reproduce the original source/crop/SNR stream before independent overrides.
            source = self.sources[int(rng.integers(len(self.sources)))]
            noise, start, retry = energetic_crop(self.recordings[source["noise_id"]], rng)
            level = float(rng.uniform(-5,20))
            trace.update(base_noise_id=source["noise_id"], base_start_sample=start, base_snr_db=level)
            if speech_selected:
                family = self.families[int(speech_rng.integers(len(self.families)))]
                members = [s for s in self.added if s["family"] == family]
                source = members[int(speech_rng.integers(len(members)))]
                noise, start, retry = energetic_crop(self.recordings[source["noise_id"]], speech_rng)
            if near_selected:
                level = float(snr_rng.uniform(-3,3))
            power = self.reference_power if self.targets[index] == 10 else float(np.mean(wave.astype(np.float64)**2))
            wave, audit = mix_at_snr(wave, noise, level, power)
            trace.update(noise_id=source["noise_id"], source_family=source.get("family", source["category"]),
                         start_sample=start, crop_retry=retry, **audit)
        trace["input_sha256"] = hashlib.sha256(wave.tobytes()).hexdigest()
        self.trace.append(trace)
        return torch.from_numpy(wave), self.targets[index], path
