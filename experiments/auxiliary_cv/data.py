"""Fold-local scaling and paired, train-only augmentation inputs."""
import itertools
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from experiments.auxiliary.data import audit_dataset, make_jobs, _init_worker, _worker
from experiments.auxiliary.features import FeatureConfig, encode, fit_scaler
from src.engine import file_hash, write_json

SPEAKERS = ("spk02", "spk03", "spk04", "spk05")


def speaker_groups(frame):
    groups = frame.group.astype(str).str.removeprefix("speaker:")
    if not groups.isin(SPEAKERS).all():
        raise ValueError("Development CV permits only spk02..05, including linked silence")
    words = frame.label != "silence"
    if not (frame.loc[words, "speaker_id"].to_numpy() == groups[words].to_numpy()).all():
        raise ValueError("Speaker identity and group disagree")
    return groups.to_numpy()


def fold_indices(frame, validation_speaker):
    if validation_speaker not in SPEAKERS:
        raise ValueError("Test speaker cannot be a development fold")
    groups = speaker_groups(frame)
    train, val = np.flatnonzero(groups != validation_speaker), np.flatnonzero(groups == validation_speaker)
    if not len(train) or not len(val) or set(groups[train]) & set(groups[val]):
        raise ValueError("Invalid or overlapping speaker fold")
    return train, val


def cached_original_worker(task):
    job, row_path = task
    row_path = Path(row_path)
    if row_path.exists():
        with np.load(row_path, allow_pickle=False) as saved:
            return tuple(saved[name].copy() for name in ("spec", "raw", "valid"))
    spec, raw, valid, _ = _worker(job)
    partial = row_path.with_suffix(".partial.npz")
    np.savez(partial, spec=spec, raw=raw, valid=valid)
    partial.replace(row_path)
    return spec, raw, valid


def original_cache(directory, frame, jobs, cfg, workers):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path, archive = directory / "manifest.json", directory / "inputs.npz"
    fc = FeatureConfig()
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if saved["jobs"] != jobs or saved["feature_config"] != fc.to_dict():
            raise ValueError("Original cache protocol differs")
        if file_hash(archive) != saved["inputs_sha256"]:
            raise ValueError("Original cache checksum differs")
        return
    rows = directory / "rows"
    rows.mkdir(exist_ok=True)
    tasks = []
    for job in jobs:
        key = hashlib.sha256(json.dumps(dict(job=job, wav_sha256=file_hash(job["wav"]),
            features=fc.to_dict(), frontend=cfg.to_dict()), sort_keys=True).encode()).hexdigest()
        tasks.append((job, str(rows / f"{key}.npz")))
    print(f"Original row cache: {sum(Path(path).exists() for _, path in tasks)}/{len(tasks)} reusable", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
            initializer=_init_worker, initargs=(fc.to_dict(), cfg.to_dict())) as pool:
        for i, result in enumerate(pool.map(cached_original_worker, tasks, chunksize=2), 1):
            results.append(result)
            if i % 50 == 0 or i == len(tasks):
                print(f"original_features {i}/{len(tasks)}", flush=True)
    spec, raw, valid = [np.stack([row[k] for row in results]) for k in range(3)]
    partial = directory / "inputs.partial.npz"
    np.savez(partial, spec=spec, raw=raw, valid=valid, paths=frame.path.to_numpy(dtype=str))
    partial.replace(archive)
    frame.to_csv(directory / "metadata.csv", index=False)
    write_json(manifest_path, dict(feature_config=fc.to_dict(), jobs=jobs,
                                  inputs_sha256=file_hash(archive)))


def prepare(study, cfg, workers=8):
    study = Path(study)
    frame, audit = audit_dataset(cfg)
    frame = frame[frame.split != "test"].reset_index(drop=True)
    speaker_groups(frame)
    write_json(study / "dataset_audit.json", audit)
    original = study / "cache" / "original"
    original_cache(original, frame, make_jobs(frame, cfg), cfg, workers)
    with np.load(original / "inputs.npz", allow_pickle=False) as arrays:
        for speaker in SPEAKERS:
            train, val = fold_indices(frame, speaker)
            folder = study / "folds" / speaker
            folder.mkdir(parents=True, exist_ok=True)
            scaler = fit_scaler(arrays["raw"][train], arrays["valid"][train])
            write_json(folder / "scaler.json", scaler)
            write_json(folder / "split.json", dict(validation_speaker=speaker,
                train_paths=frame.iloc[train].path.tolist(), val_paths=frame.iloc[val].path.tolist(),
                train_groups=sorted(set(speaker_groups(frame)[train])), val_groups=[speaker],
                original_cache_sha256=file_hash(original / "inputs.npz"), test_included=False))
    return frame


def wave_key(policies):
    return "+".join(sorted(set(policies) - {"mask"}))


def subsets(policies):
    waves = sorted(set(policies) - {"mask"})
    return [combo for n in range(1, len(waves) + 1) for combo in itertools.combinations(waves, n)]


class Inputs:
    def __init__(self, study):
        self.study = Path(study)
        root = self.study / "cache" / "original"
        self.frame = pd.read_csv(root / "metadata.csv", keep_default_na=False)
        speaker_groups(self.frame)
        manifest = json.loads((root / "manifest.json").read_text())
        if file_hash(root / "inputs.npz") != manifest["inputs_sha256"]:
            raise ValueError("Original input cache checksum differs")
        with np.load(root / "inputs.npz", allow_pickle=False) as arrays:
            if not np.array_equal(arrays["paths"], self.frame.path.to_numpy()):
                raise ValueError("Original cache path order differs")
            self.spec, self.raw, self.valid = [arrays[k].copy() for k in ("spec", "raw", "valid")]


def mask_inputs(spec, auxiliary, rng):
    """Time masks hide both paths; spectral frequency masks have no aux-bin analogue."""
    spec, auxiliary = spec.copy(), auxiliary.copy()
    width = int(rng.integers(2, 6))  # 20..50 ms at the 10-ms grid.
    start = int(rng.integers(0, spec.shape[-1] - width + 1))
    spec[..., start:start + width] = -2.0  # Scaled-mel floor, not normalized mean.
    auxiliary[..., start:start + width] = 0  # Values and availability channels both hidden.
    width = int(rng.integers(4, 9))
    start = int(rng.integers(0, spec.shape[-2] - width + 1))
    spec[..., start:start + width, :] = -2.0
    return spec, auxiliary


class FoldDataset(Dataset):
    def __init__(self, inputs, speaker, set_id, labels, training=False, policies=(), seed=42):
        self.inputs = inputs
        train, val = fold_indices(inputs.frame, speaker)
        self.indices = train if training else val
        self.frame = inputs.frame.iloc[self.indices].reset_index(drop=True)
        self.training, self.policies, self.seed = training, tuple(policies), int(seed)
        self.epoch, self.draw = 0, 0
        scaler = json.loads((inputs.study / "folds" / speaker / "scaler.json").read_text())
        self.aux = encode(inputs.raw, inputs.valid, scaler, set_id)
        self.banks = {}
        if training:
            for combo in subsets(policies):
                key = wave_key(combo)
                root = inputs.study / "cache" / key
                manifest = json.loads((root / "manifest.json").read_text())
                if file_hash(root / "inputs.npz") != manifest["inputs_sha256"]:
                    raise ValueError("Augmented cache hash differs")
                with np.load(root / "inputs.npz", allow_pickle=False) as bank:
                    if not np.array_equal(bank["paths"], inputs.frame.path.to_numpy()):
                        raise ValueError("Augmented cache path order differs")
                    spec, raw, valid = [bank[k].copy() for k in ("spec", "raw", "valid")]
                variants, count, features, times = raw.shape
                aux = encode(raw.reshape(-1, features, times), valid.reshape(-1, features, times),
                             scaler, set_id).reshape(variants, count, self.aux.shape[1], times)
                self.banks[key] = (spec, aux)
        mapping = {label: i for i, label in enumerate(labels)}
        self.targets = np.array([mapping[label] for label in self.frame.label])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = self.indices[index]
        spec, aux = self.inputs.spec[row], self.aux[row]
        if self.training and self.policies:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, self.draw]))
            self.draw += 1
            active = [p for p in self.policies if rng.random() < 0.5]
            key = wave_key(active)
            if key:
                bank = self.banks[key]
                variant = int(rng.integers(bank[0].shape[0]))
                spec, aux = bank[0][variant, row], bank[1][variant, row]
            if "mask" in active:
                spec, aux = mask_inputs(spec, aux, rng)
        return torch.from_numpy(spec), torch.from_numpy(aux), int(self.targets[index]), self.frame.path.iloc[index]
