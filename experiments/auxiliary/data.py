"""Frozen data/noise manifests and cached inputs shared by every model."""
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
import io
import json
import math
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from scipy.signal import resample_poly

from src.dataset import audio_path, read_metadata
from src.engine import file_hash, write_json
from .features import FeatureConfig, NAMES, extract, fit_scaler

NOISE_CATEGORIES = ["rain", "engine", "church_bells", "keyboard_typing"]
SNRS = [20, 10, 0]
_MEL = _DB = _FC = None


def audit_dataset(cfg):
    frame = read_metadata(cfg)
    root = Path(cfg.data_root)
    manifest = json.loads((root / "manifest.json").read_text())
    for filename, field in [("metadata.csv", "metadata_sha256"),
                            ("preprocessing_recipe.csv", "recipe_sha256")]:
        if file_hash(root / filename) != manifest[field]:
            raise ValueError(f"Reviewed dataset changed: {filename}")
    recipe = pd.read_csv(root / "preprocessing_recipe.csv").set_index("path")
    hashes = {}
    for row in frame.itertuples():
        path = audio_path(root, row.path)
        info = sf.info(path)
        if (info.samplerate, info.channels, info.frames, info.subtype) != (16000, 1, 24000, "PCM_16"):
            raise ValueError(f"Unexpected reviewed WAV geometry: {row.path}")
        h = file_hash(path)
        if h != recipe.loc[row.path, "output_sha256"]:
            raise ValueError(f"Reviewed WAV hash mismatch: {row.path}")
        hashes[row.path] = h
    hashed = frame.assign(sha256=frame.path.map(hashes))
    if (hashed.groupby("sha256").split.nunique() > 1).any():
        raise ValueError("Identical audio leaks across splits")
    return frame, {"dataset": manifest, "audio_sha256": hashes,
                   "split_counts": frame.split.value_counts().to_dict(),
                   "speakers": {s: sorted(frame.loc[(frame.split == s) & (frame.label != "silence"),
                                                    "speaker_id"].unique().tolist())
                                for s in ["train", "val", "test"]},
                   "originals_modified": False, "test_predictions_made": False}


def download_noise(root):
    import requests
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    existing = root / "manifest.json"
    if existing.exists():
        result = json.loads(existing.read_text())
        for item in result["files"]:
            if file_hash(root / "audio" / item["filename"]) != item["sha256"]:
                raise ValueError("External noise file changed")
        return result

    def get(url):
        r = requests.get(url, timeout=90)
        r.raise_for_status()
        return r

    commit = get("https://api.github.com/repos/karolpiczak/ESC-50/commits/master").json()["sha"]
    prefix = f"https://raw.githubusercontent.com/karolpiczak/ESC-50/{commit}"
    metadata_bytes = get(prefix + "/meta/esc50.csv").content
    (root / "esc50.csv").write_bytes(metadata_bytes)
    metadata = pd.read_csv(io.BytesIO(metadata_bytes))
    files = []
    (root / "audio").mkdir(exist_ok=True)
    for split, fold in [("val", 4), ("test", 5)]:
        for category in NOISE_CATEGORIES:
            rows = (metadata[(metadata.fold == fold) & (metadata.category == category)]
                    .sort_values("filename").drop_duplicates("src_file").head(2))
            if len(rows) != 2:
                raise ValueError(f"Missing independent ESC-50 sources: {category}/{split}")
            for row in rows.itertuples():
                url = prefix + "/audio/" + row.filename
                data = get(url).content
                path = root / "audio" / row.filename
                path.write_bytes(data)
                files.append(dict(filename=row.filename, category=category, split=split,
                                  fold=fold, source_id=int(row.src_file), url=url,
                                  freesound_url=f"https://freesound.org/s/{row.src_file}/",
                                  sha256=file_hash(path)))
    for split in ["val", "test"]:
        if len({f["source_id"] for f in files if f["split"] == split}) != 8:
            raise ValueError("Noise source IDs overlap within the selected categories")
    if {f["source_id"] for f in files if f["split"] == "val"} & {f["source_id"] for f in files if f["split"] == "test"}:
        raise ValueError("Noise source leakage")
    (root / "LICENSE.txt").write_bytes(get(prefix + "/LICENSE").content)
    result = dict(dataset="ESC-50", repository="https://github.com/karolpiczak/ESC-50",
                  commit=commit, license="CC BY-NC 3.0 (dataset); original source attribution via Freesound IDs",
                  use="evaluation only; no noise augmentation during the baseline ablation",
                  files=files)
    write_json(existing, result)
    return result


@lru_cache(maxsize=32)
def noise_wave(path):
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    y = y.mean(1).astype(np.float64)
    if sr != 16000:
        gcd = math.gcd(sr, 16000)
        y = resample_poly(y, 16000 // gcd, sr // gcd)
    y -= y.mean()
    if not np.isfinite(y).all() or np.mean(y*y) <= 1e-12:
        raise ValueError("Empty/nonfinite external noise")
    return y


def mix_noise(wav, noise, snr_db, start=0, reference_rms=None):
    wav = np.asarray(wav, dtype=np.float64)
    n = np.asarray(noise, dtype=np.float64)
    if len(n) < len(wav):
        n = np.tile(n, math.ceil(len(wav) / len(n)))
    n = n[start:start + len(wav)].copy()
    if len(n) != len(wav):
        raise ValueError("Noise segment is too short")
    power = np.mean(wav ** 2) if reference_rms is None else reference_rms ** 2
    if power <= 1e-12:
        raise ValueError("SNR is undefined for zero-power speech without a reference")
    noise_power = np.mean(n ** 2)
    if noise_power <= 1e-12:
        raise ValueError("Selected noise segment contains no energy")
    n *= np.sqrt(power / noise_power / (10 ** (snr_db / 10)))
    mixed = wav + n
    gain = min(1.0, 0.99 / max(np.max(np.abs(mixed)), 1e-12))
    measured = float(10 * np.log10(power / np.mean(n*n)))
    return (mixed * gain).astype(np.float32), dict(reference_snr_db=measured,
                                                 common_peak_gain=gain,
                                                 reference_rms=float(np.sqrt(power)))


def _init_worker(fc_dict, cfg_dict):
    global _MEL, _DB, _FC
    torch.set_num_threads(1)
    _FC = FeatureConfig(**fc_dict)
    _MEL = torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg_dict["sample_rate"], n_fft=cfg_dict["n_fft"],
        win_length=cfg_dict["win_length"], hop_length=cfg_dict["hop_length"],
        n_mels=cfg_dict["n_mels"], f_min=40, f_max=8000, power=2.0)
    _DB = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)


def render_job(job):
    if job["noise_only"]:
        wav = np.zeros(24000, dtype=np.float32)
    else:
        wav, sr = sf.read(job["wav"], dtype="float32")
        if sr != 16000 or wav.shape != (24000,):
            raise ValueError("Cache only accepts the frozen reviewed WAVs")
    details = {}
    if job.get("noise"):
        wav, details = mix_noise(wav, noise_wave(job["noise"]), job["snr"], job["start"],
                                 job.get("reference_rms"))
    return wav, details


def _worker(job):
    wav, details = render_job(job)
    with torch.no_grad():
        spec = ((_DB(_MEL(torch.from_numpy(wav).unsqueeze(0))) + 80) / 80).clamp(0, 1)
        spec = ((spec - 0.5) / 0.25).numpy()
    raw, valid = extract(wav, _FC)
    return spec, raw, valid, details


def make_jobs(frame, cfg, noise_root=None, noise_manifest=None, category=None, snr=None,
              split="val", reference_rms=None, noise_only=False):
    entries = None
    if category:
        entries = [x for x in noise_manifest["files"] if x["split"] == split and x["category"] == category]
    jobs = []
    for i, row in enumerate(frame.to_dict("records")):
        key = f"aux-study-v1|{split}|{category}|{row['path']}"
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little"))
        job = dict(path=row["path"], label=row["label"], noise_only=noise_only,
                   wav=str(audio_path(cfg.data_root, row["path"])) if not noise_only else None)
        if entries:
            entry = entries[int(rng.integers(len(entries)))]
            noise_path = Path(noise_root) / "audio" / entry["filename"]
            n = noise_wave(str(noise_path))
            # Choose an energetic segment once; keep the same file/start at every SNR.
            candidates = rng.integers(0, max(1, len(n)-24000+1), size=20)
            start = next((int(v) for v in candidates if np.mean(n[v:v+24000]**2) > 1e-9), None)
            if start is None:
                raise ValueError("No usable deterministic noise crop")
            job.update(noise=str(noise_path), noise_filename=entry["filename"],
                       start=start, snr=int(snr))
            if noise_only or row["label"] == "silence":
                job["reference_rms"] = reference_rms
        jobs.append(job)
    return jobs


def build_cache(directory, frame, jobs, cfg, fc, workers=4):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / "inputs.npz"
    manifest_path = directory / "manifest.json"
    if archive.exists() and manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing["feature_config"] != fc.to_dict() or existing["jobs"] != jobs:
            raise ValueError("Cache protocol differs; use a new study directory")
        if file_hash(archive) != existing["inputs_sha256"]:
            raise ValueError("Input cache changed")
        return
    results = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                             initializer=_init_worker, initargs=(fc.to_dict(), cfg.to_dict())) as pool:
        for i, result in enumerate(pool.map(_worker, jobs, chunksize=4), 1):
            results.append(result)
            if i % 100 == 0 or i == len(jobs):
                print(f"cache={directory.name} {i}/{len(jobs)}", flush=True)
    spec, raw, valid = [np.stack([r[j] for r in results]) for j in range(3)]
    if not np.isfinite(spec).all():
        raise ValueError("Nonfinite spectral input")
    tmp = directory / "inputs.partial.npz"
    np.savez(tmp, spec=spec, raw=raw, valid=valid,
             paths=frame.path.to_numpy(dtype=str), labels=frame.label.to_numpy(dtype=str))
    tmp.replace(archive)
    frame.to_csv(directory / "metadata.csv", index=False, encoding="utf-8-sig")
    diagnostics = [{"feature": name, "frames": int(valid[:, j].size),
                    "unavailable_frames": int((~valid[:, j]).sum()),
                    "unavailable_fraction": float((~valid[:, j]).mean())}
                   for j, name in enumerate(NAMES)]
    pd.DataFrame(diagnostics).to_csv(directory / "availability.csv", index=False)
    write_json(manifest_path, dict(feature_config=fc.to_dict(), inputs_sha256=file_hash(archive),
                                   jobs=jobs, mixing=[r[3] for r in results],
                                   unavailable_is_not_incorrect=True))


def prepare(study, cfg, workers=4, split="val"):
    study = Path(study)
    study.mkdir(parents=True, exist_ok=True)
    frame, audit = audit_dataset(cfg)
    write_json(study / "dataset_audit.json", audit)
    noise_root = study / "noise_sources"
    noise = download_noise(noise_root)
    fc = FeatureConfig()
    cache = study / "cache"
    if split == "val":
        selected = frame[frame.split.isin(["train", "val"])].reset_index(drop=True)
        jobs = make_jobs(selected, cfg)
        build_cache(cache / "original", selected, jobs, cfg, fc, workers)
        with np.load(cache / "original" / "inputs.npz") as data:
            train = selected.split.to_numpy() == "train"
            scaler = fit_scaler(data["raw"][train], data["valid"][train])
        write_json(study / "scaler.json", scaler)
        train_rms = []
        for row in selected[(selected.split == "train") & (selected.label != "silence")].itertuples():
            y, _ = sf.read(audio_path(cfg.data_root, row.path))
            train_rms.append(np.sqrt(np.mean(y*y)))
        write_json(study / "noise_reference.json", dict(rms=float(np.median(train_rms)),
                                                        fitted_on="train_keyword_full_clip_RMS"))
    else:
        selected = frame[frame.split == "test"].reset_index(drop=True)
        build_cache(cache / "test_original", selected, make_jobs(selected, cfg), cfg, fc, workers)
    reference = json.loads((study / "noise_reference.json").read_text())["rms"]
    evaluation = frame[frame.split == split].reset_index(drop=True)
    for category in NOISE_CATEGORIES:
        for snr in SNRS:
            name = f"{split}_{category}_{snr}db"
            jobs = make_jobs(evaluation, cfg, noise_root, noise, category, snr, split, reference)
            build_cache(cache / name, evaluation, jobs, cfg, fc, workers)
    # Noise-only negatives at the same nominal levels, 12 independent crops/category/level.
    negative_frames, negative_jobs = [], []
    for category in NOISE_CATEGORIES:
        neg = pd.DataFrame([dict(path=f"noise/{category}/{i:02d}", label="silence", split=split,
                                 group=f"noise:{category}", speaker_id="", room_session_id=category)
                            for i in range(12)])
        for snr in SNRS:
            jobs = make_jobs(neg, cfg, noise_root, noise, category, snr, split, reference, True)
            tagged = neg.copy()
            tagged["path"] = tagged.path + f"/{snr}db"
            tagged["noise_category"], tagged["nominal_snr"] = category, snr
            for j, path in zip(jobs, tagged.path):
                j["path"] = path
            negative_frames.append(tagged)
            negative_jobs.extend(jobs)
    neg = pd.concat(negative_frames, ignore_index=True)
    build_cache(cache / f"{split}_noise_only", neg, negative_jobs, cfg, fc, workers)
    return frame
