"""Speech-local smooth pitch contours; legacy formants/noise remain common factors."""
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.interpolate import PchipInterpolator

from experiments.overnight.data import shift_formants
from experiments.overnight.waveforms import settings
from experiments.conformer_long_zero.run import write_json
from .vad import wave_hash


def source_regions(row):
    crop_start, crop_end = int(row.crop_start_sample), int(row.crop_end_sample)
    left = int(row.pad_left_samples)
    intervals = []
    for segment in json.loads(row.vad_segments):
        start, end = max(crop_start, segment["start"]), min(crop_end, segment["end"])
        if end > start:
            intervals.append((start - crop_start + left, end - crop_start + left))
    return intervals


def pitch_curve(path, variant, regions):
    token = f"speech-local-pitch-v1|42|{path}|{variant}"
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "little"))
    curve = np.zeros(24000, np.float32)
    envelope = np.zeros(24000, np.float32)
    records = []
    for start, end in regions:
        length = end - start
        if length < 1600:
            continue
        # At least two turns inside speech, with random knot locations and smooth interpolation.
        interior = 3 if length < 8000 else 4
        locations = np.r_[0., np.arange(1, interior + 1) / (interior + 1)
                          + rng.uniform(-.055, .055, interior), 1.]
        maximum = (1., 1.5, 2., 2.5)[variant % 4]
        direction = -1 if variant < 4 else 1
        values = np.r_[0., [direction * (-1) ** k * rng.uniform(.45, 1.) * maximum
                           for k in range(interior)], 0.]
        points = np.linspace(0, 1, length)
        smooth = PchipInterpolator(locations, values)(points)
        # Limit the largest instantaneous slope; preserve variation instead of stretching it into padding.
        max_slope = np.max(np.abs(np.diff(smooth))) * 16000
        scale = min(1., 32. / max(max_slope, 1e-9))
        smooth *= scale
        curve[start:end] = smooth
        taper = min(320, length // 8)
        weight = np.ones(length, np.float32)
        weight[:taper] = .5 - .5 * np.cos(np.linspace(0, np.pi, taper))
        weight[-taper:] = weight[:taper][::-1]
        envelope[start:end] = weight
        records.append(dict(start_sample=start, end_sample=end, knot_positions=locations.tolist(),
                            semitones=(values * scale).tolist(), maximum_slope_st_per_second=min(32., float(max_slope))))
    return curve, envelope, records


def apply_curve(wave, curve):
    import parselmouth
    from parselmouth.praat import call
    sound = parselmouth.Sound(wave, 16000)
    manipulation = call(sound, "To Manipulation", .01, 65., 600.)
    original = call(manipulation, "Extract pitch tier")
    count = call(original, "Get number of points")
    if count == 0:
        return wave.copy(), dict(voiced_pitch_points=0, applied=False)
    altered = call("Create PitchTier", "dynamic", 0., 1.5)
    for index in range(1, count + 1):
        time = call(original, "Get time from index", index)
        frequency = call(original, "Get value at index", index)
        shift = float(np.interp(time * 16000, np.arange(24000), curve))
        call(altered, "Add point", time, frequency * 2 ** (shift / 12))
    call([altered, manipulation], "Replace pitch tier")
    result = call(manipulation, "Get resynthesis (overlap-add)").values[0]
    assert result.shape == wave.shape
    return result.astype(np.float32), dict(voiced_pitch_points=int(count), applied=True)


def local_pitch(wave, path, variant, regions):
    curve, envelope, record = pitch_curve(path, variant, regions)
    transformed, tracking = apply_curve(wave, curve)
    inside = envelope > 0
    if inside.any():
        p0 = np.mean(wave[inside].astype(np.float64) ** 2)
        p1 = np.mean(transformed[inside].astype(np.float64) ** 2)
        gain = np.sqrt(p0 / p1) if p0 > 0 and p1 > 0 else 1.
        peak = np.max(np.abs(transformed[inside]))
        gain = min(gain, .99 / max(float(peak), 1e-9))
        transformed *= gain
    result = np.asarray(wave * (1 - envelope) + transformed * envelope, dtype=np.float32)
    assert np.array_equal(result[~inside], wave[~inside])
    assert result.shape == (24000,) and np.isfinite(result).all() and np.abs(result).max() <= 1.
    return result, dict(regions=record, background_outside_speech_bitwise_unchanged=True,
                        curve_sha256=wave_hash(curve), pitch_method="Praat pitch-tier overlap-add", **tracking)


def _prepare_one(task):
    path, source, regions, target = task
    target = Path(target)
    x, rate = sf.read(source, dtype="float32")
    assert rate == 16000 and x.shape == (24000,)
    result = np.empty((8, 3, 24000), np.float32)
    records = []
    ratios = [settings(path, variant)[2] for variant in (0, 1)]
    for variant in range(8):
        pitched, details = local_pitch(x, path, variant, regions)
        for slot in range(3):
            y = pitched if slot == 0 else shift_formants(pitched, ratios[slot - 1])
            result[variant, slot] = y
        records.append(dict(variant=variant, **details))
    partial = target.with_suffix(".partial.npy")
    np.save(partial, result)
    partial.replace(target)
    write_json(target.with_suffix(".json"), dict(path=path, source_float32_sha256=wave_hash(x),
        formant_ratios=ratios, pitch=records, output_sha256=hashlib.sha256(result.tobytes()).hexdigest()))
    return str(target)


class PitchBank:
    def __init__(self, folder, paths):
        self.folder = Path(folder)
        self.indices = {path: i for i, path in enumerate(paths)}
        self.recent = {}

    def get(self, path, variant, slot):
        index = self.indices[path]
        if index not in self.recent:
            if len(self.recent) >= 24:
                self.recent.pop(next(iter(self.recent)))
            self.recent[index] = np.load(self.folder / f"{index:04d}.npy", mmap_mode="r")
        return self.recent[index][variant, slot].copy()


def prepare(metadata, data_root, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    frame = metadata[metadata.group.ne("speaker:spk01") & metadata.label.ne("silence")].reset_index(drop=True)
    assert len(frame) == 800
    recipe = pd.read_csv(Path(data_root) / "preprocessing_recipe.csv").set_index("path")
    regions = {path: source_regions(recipe.loc[path]) for path in frame.path}
    assert all(regions.values())
    tasks = []
    for index, path in enumerate(frame.path):
        target = folder / f"{index:04d}.npy"
        if not target.exists() or not target.with_suffix(".json").exists():
            tasks.append((path, str(Path(data_root) / path), regions[path], str(target)))
    if tasks:
        with ProcessPoolExecutor(max_workers=8, mp_context=mp.get_context("spawn")) as pool:
            for count, _ in enumerate(pool.map(_prepare_one, tasks, chunksize=1), 1):
                if count % 50 == 0:
                    print("PITCH_BANK", count, len(tasks), flush=True)
    for index in range(800):
        target = folder / f"{index:04d}.npy"
        record = json.loads(target.with_suffix(".json").read_text(encoding="utf-8"))
        assert record["path"] == frame.path.iloc[index]
        assert hashlib.sha256(np.load(target, mmap_mode="r").tobytes()).hexdigest() == record["output_sha256"]
    write_json(folder / "complete.json", dict(files=800, variants_per_file=8,
        formant_slots=3, test_audio_included=False, source_regions=regions,
        note="Original automatic VAD boundaries are used only to place training augmentation; classifier gates use actual mixed inputs"))
    return PitchBank(folder, frame.path.tolist()), regions
