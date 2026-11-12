"""Waveform-first augmentation with reproducible, resumable variant banks."""
import ctypes as ct
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from experiments.auxiliary import data as legacy
from experiments.auxiliary.features import FeatureConfig, extract
from src.engine import file_hash, write_json
from .data import subsets, wave_key

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = 2
VERSION = "waveform-first-bank-v1"
_LIB = None


def rubberband():
    global _LIB
    if _LIB is not None:
        return _LIB
    root = ROOT / "runs/auxiliary-cv-v1/runtime/unpacked/usr/lib/x86_64-linux-gnu"
    for name in ("libfftw3.so.3", "libsamplerate.so.0"):
        ct.CDLL(str(root / name), mode=ct.RTLD_GLOBAL)
    lib = ct.CDLL(str(root / "librubberband.so.2"), mode=ct.RTLD_GLOBAL)
    state, pointer = ct.c_void_p, ct.POINTER(ct.POINTER(ct.c_float))
    signatures = {
        "new": (state, [ct.c_uint, ct.c_uint, ct.c_int, ct.c_double, ct.c_double]),
        "delete": (None, [state]),
        "get_latency": (ct.c_uint, [state]),
        "set_pitch_scale": (None, [state, ct.c_double]),
        "set_max_process_size": (None, [state, ct.c_uint]),
        "set_expected_input_duration": (None, [state, ct.c_uint]),
        "study": (None, [state, pointer, ct.c_uint, ct.c_int]),
        "process": (None, [state, pointer, ct.c_uint, ct.c_int]),
        "available": (ct.c_int, [state]),
        "retrieve": (ct.c_uint, [state, pointer, ct.c_uint]),
    }
    for name, (result, args) in signatures.items():
        function = getattr(lib, "rubberband_" + name)
        function.restype, function.argtypes = result, args
    _LIB = lib
    return lib


def pointers(array):
    return (ct.POINTER(ct.c_float) * 1)(array.ctypes.data_as(ct.POINTER(ct.c_float)))


def transform(wave, rate=1.0, curve=None):
    """Rubber Band 2.0 R2; dynamic pitch preserves formants and nominal duration."""
    lib = rubberband()
    dynamic = curve is not None
    options = 0x00010000 | 0x01000000  # Never create internal threads; preserve formants.
    if dynamic:
        options |= 0x00000001 | 0x04000000  # Real-time + consistent variable pitch.
    state = lib.rubberband_new(16000, 1, options, 1.0 / rate, 1.0)
    if not state:
        raise RuntimeError("Rubber Band initialization failed")
    wave = np.asarray(wave, dtype=np.float32)
    block = 128 if dynamic else 512
    output = []
    try:
        lib.rubberband_set_max_process_size(state, block)
        delay = int(lib.rubberband_get_latency(state)) if dynamic else 0
        # Match the 2.0 CLI: explicitly prepend getLatency() zeros in real-time
        # mode, then discard that many output samples. Tail zeros flush the end.
        signal = np.r_[np.zeros(delay, np.float32), wave, np.zeros(delay, np.float32)] if dynamic else wave
        if not dynamic:
            lib.rubberband_set_expected_input_duration(state, len(signal))
            for offset in range(0, len(signal), block):
                chunk = np.ascontiguousarray(signal[offset:offset + block])
                lib.rubberband_study(state, pointers(chunk), len(chunk), int(offset + block >= len(signal)))
        for offset in range(0, len(signal), block):
            if dynamic:
                semitones = float(np.interp(np.clip((offset - delay) / max(1, len(wave) - 1), 0, 1),
                                           [0, 1 / 3, 2 / 3, 1], curve))
                lib.rubberband_set_pitch_scale(state, 2 ** (semitones / 12))
            chunk = np.ascontiguousarray(signal[offset:offset + block])
            lib.rubberband_process(state, pointers(chunk), len(chunk), int(offset + block >= len(signal)))
            while (available := lib.rubberband_available(state)) > 0:
                result = np.empty(available, dtype=np.float32)
                received = lib.rubberband_retrieve(state, pointers(result), available)
                output.append(result[:received])
        result = np.concatenate(output) if output else np.empty(0, np.float32)
        expected = len(wave) if dynamic else round(len(wave) / rate)
        if len(result) < delay + expected:
            raise ValueError(f"Rubber Band returned too few samples: {len(result)}, expected {delay + expected}")
        result = result[delay:delay + expected].copy()
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite transformed waveform")
        return result
    finally:
        lib.rubberband_delete(state)


def protected_speed_crop(length, rate, protected):
    a, b = int(np.floor(protected[0] / rate)), int(np.ceil(protected[1] / rate))
    if b - a > 24000:
        return None
    if length <= 24000:
        return (0, length)
    low, high = max(0, b - 24000), min(a, length - 24000)
    if low > high:
        return None
    start = int(np.clip((length - 24000) // 2, low, high))
    return start, start + 24000


def speed_wave(wave, rate, protected):
    expected = round(len(wave) / rate)
    crop = protected_speed_crop(expected, rate, protected)
    if crop is None:
        return wave.copy(), dict(requested_rate=rate, applied_rate=1.0, skipped="protected_interval_would_be_cut")
    result = transform(wave, rate=rate)
    crop = protected_speed_crop(len(result), rate, protected)
    if crop is None:
        raise ValueError("Transformed length violates the planned protected interval")
    result = result[crop[0]:crop[1]]
    missing = 24000 - len(result)
    result = np.pad(result, (missing // 2, missing - missing // 2))
    return result, dict(requested_rate=rate, applied_rate=rate, crop=list(crop), zero_padding=missing)


def component_rng(path, variant, component):
    key = f"{VERSION}|20261111|{path}|{variant}|{component}"
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little"))


def augmented_worker(task):
    job, stored = task
    stored = Path(stored)
    if stored.exists():
        with np.load(stored, allow_pickle=False) as data:
            return tuple(data[key].copy() for key in ("spec", "raw", "valid")) + (json.loads(str(data["trace"])),)
    wave, _ = sf.read(job["wav"], dtype="float32")
    trace = dict(path=job["path"], variant=job["variant"], policies=job["policies"])
    if "pitch" in job["policies"] and job["label"] != "silence":
        rng = component_rng(job["path"], job["variant"], "pitch")
        middle = rng.uniform(-1, 1, size=2)
        if abs(middle[1] - middle[0]) < 0.35:
            middle[1] = -np.sign(middle[0] or 1) * 0.75
        curve = [0.0, *middle.tolist(), 0.0]
        wave = transform(wave, curve=curve)
        trace["pitch_semitone_anchors"] = curve
    if "speed" in job["policies"] and job["label"] != "silence":
        rng = component_rng(job["path"], job["variant"], "speed")
        rate = float(rng.uniform(0.9, 1.1))
        wave, details = speed_wave(wave, rate, job["protected"])
        trace["speed"] = details
    if "noise" in job["policies"]:
        rng = component_rng(job["path"], job["variant"], "noise")
        entry = job["noise_files"][int(rng.integers(len(job["noise_files"])))]
        noise = legacy.noise_wave(entry)
        candidates = rng.integers(0, len(noise) - 24000 + 1, size=30)
        start = next((int(s) for s in candidates if np.mean(noise[s:s + 24000] ** 2) > 1e-9), None)
        if start is None:
            raise ValueError("No energetic noise segment")
        snr = float(rng.uniform(10, 30))
        wave, details = legacy.mix_noise(wave, noise, snr, start)
        trace["noise"] = dict(filename=Path(entry).name, start=start, snr_db=snr, **details)
    peak = float(np.max(np.abs(wave)))
    peak_gain = min(1.0, 0.99 / max(peak, 1e-9))
    wave = np.asarray(wave * peak_gain, dtype=np.float32)
    trace["final_peak_gain"] = peak_gain
    if wave.shape != (24000,) or not np.isfinite(wave).all():
        raise ValueError("Invalid augmented waveform")
    with torch.no_grad():
        mel = ((legacy._DB(legacy._MEL(torch.from_numpy(wave)[None])) + 80) / 80).clamp(0, 1)
        spec = ((mel - 0.5) / 0.25).numpy()
    raw, valid = extract(wave, FeatureConfig())
    trace["waveform_sha256"] = hashlib.sha256(wave.tobytes()).hexdigest()
    partial = stored.with_suffix(".partial.npz")
    np.savez(partial, spec=spec, raw=raw, valid=valid, wave=wave, trace=json.dumps(trace))
    partial.replace(stored)
    return spec, raw, valid, trace


def prepare_bank(study, cfg, policies, workers=8):
    key = wave_key(policies)
    if not key:
        return
    study = Path(study)
    directory = study / "cache" / key
    directory.mkdir(parents=True, exist_ok=True)
    archive, manifest_file = directory / "inputs.npz", directory / "manifest.json"
    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text())
        if (manifest["version"] != VERSION or manifest["policies"] != sorted(policies)
                or manifest["code_sha256"] != file_hash(__file__)
                or manifest["original_cache_sha256"] != file_hash(study / "cache/original/inputs.npz")):
            raise ValueError("Augmentation protocol or source changed")
        if file_hash(archive) != manifest["inputs_sha256"]:
            raise ValueError("Augmentation archive changed")
        return
    frame = pd.read_csv(study / "cache/original/metadata.csv", keep_default_na=False)
    recipe = pd.read_csv(Path(cfg.data_root) / "preprocessing_recipe.csv").set_index("path")
    noise_manifest = json.loads((study / "noise_sources/manifest.json").read_text())
    noise_files = [str((study / "noise_sources/audio" / e["filename"]).resolve()) for e in noise_manifest["files"]]
    for entry, path in zip(noise_manifest["files"], noise_files):
        if file_hash(path) != entry["sha256"]:
            raise ValueError("Training noise checksum differs")
    rows = directory / "rows"
    rows.mkdir(exist_ok=True)
    tasks = []
    for variant in range(VARIANTS):
        for row in frame.itertuples():
            rec = recipe.loc[row.path]
            a = int(rec.previous_crop_start_sample - rec.crop_start_sample + rec.pad_left_samples)
            b = int(rec.previous_crop_end_sample - rec.crop_start_sample + rec.pad_left_samples)
            job = dict(path=row.path, label=row.label, wav=str(Path(cfg.data_root) / row.path),
                       variant=variant, policies=sorted(policies), protected=[a, b], noise_files=noise_files)
            stamp = dict(job=job, source_wave_sha256=rec.output_sha256, version=VERSION,
                         code_sha256=file_hash(__file__), feature_config=FeatureConfig().to_dict())
            identity = hashlib.sha256(json.dumps(stamp, sort_keys=True).encode()).hexdigest()
            tasks.append((job, str(rows / f"{identity}.npz")))
    print(f"BANK {key}: {sum(Path(p).exists() for _, p in tasks)}/{len(tasks)} reusable", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
            initializer=legacy._init_worker, initargs=(FeatureConfig().to_dict(), cfg.to_dict())) as pool:
        for i, result in enumerate(pool.map(augmented_worker, tasks, chunksize=2), 1):
            results.append(result)
            if i % 100 == 0 or i == len(tasks):
                print(f"BANK {key} {i}/{len(tasks)}", flush=True)
    arrays = [np.stack([r[k] for r in results]) for k in range(3)]
    spec, raw, valid = [array.reshape(VARIANTS, len(frame), *array.shape[1:]) for array in arrays]
    partial = directory / "inputs.partial.npz"
    np.savez(partial, spec=spec, raw=raw, valid=valid, paths=frame.path.to_numpy(dtype=str))
    partial.replace(archive)
    write_json(manifest_file, dict(version=VERSION, policies=sorted(policies), variants=VARIANTS,
        code_sha256=file_hash(__file__), original_cache_sha256=file_hash(study / "cache/original/inputs.npz"),
        inputs_sha256=file_hash(archive), traces=[r[3] for r in results],
        raw_features_reextracted_after_waveform=True, validation_uses_original_only=True,
        snr_definition="post-transform original full-clip power (including original noise) versus added noise; silence uses its own original clip power"))


def prepare_banks(study, cfg, policies, workers=8):
    for combo in subsets(policies):
        prepare_bank(study, cfg, combo, workers)
