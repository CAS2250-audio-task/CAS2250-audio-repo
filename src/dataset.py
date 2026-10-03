"""Metadata validation and the professor's WAV -> log-mel baseline."""
from pathlib import Path
import random

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .config import Config

SPLITS = ("train", "val", "test")


def audio_path(root, relative):
    root = Path(root).resolve()
    rel = Path(relative)
    path = (root / rel).resolve()
    if rel.is_absolute() or not path.is_relative_to(root) or path.suffix.lower() != ".wav":
        raise ValueError(f"Expected a relative WAV path inside data_root: {relative}")
    if not path.is_file():
        raise ValueError(f"Missing audio: {path}")
    return path


def validate_metadata(frame, cfg):
    required = {"path", "label", "group", "split"}
    if not cfg.dummy:
        required |= {"speaker_id", "room_session_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing metadata columns: {sorted(missing)}")
    frame = frame.copy().fillna("")
    for column in ("path", "label", "group", "split"):
        if frame[column].map(lambda x: not isinstance(x, str) or not x.strip() or x != x.strip()).any():
            raise ValueError(f"Empty or whitespace-padded metadata field: {column}")
    if frame.path.duplicated().any():
        raise ValueError("Duplicate audio paths in metadata.")
    if set(frame.label) != set(cfg.labels):
        raise ValueError(f"Label mismatch: missing={sorted(set(cfg.labels)-set(frame.label))}, "
                         f"unexpected={sorted(set(frame.label)-set(cfg.labels))}")
    if set(frame.split) != set(SPLITS):
        raise ValueError("Explicit train/val/test splits are required on every row.")
    if (frame.groupby("group").split.nunique() > 1).any():
        raise ValueError("Group leakage: a group appears in multiple splits.")
    # Identity checks also catch leakage hidden by different group IDs.
    for column, subset in (("speaker_id", frame[frame.label != "silence"]),
                           ("room_session_id", frame[frame.label == "silence"])):
        if column in frame:
            values = subset[column].astype(str)
            if not cfg.dummy and values.map(lambda x: not x.strip() or x != x.strip()).any():
                raise ValueError(f"Real metadata requires {column} for its applicable rows.")
            known = subset[values.str.strip() != ""]
            if (known.groupby(column).split.nunique() > 1).any():
                raise ValueError(f"Identity leakage: {column} appears in multiple splits.")
    table = pd.crosstab(frame.label, frame.split).reindex(index=cfg.labels, columns=SPLITS, fill_value=0)
    if (table == 0).any().any():
        raise ValueError("Every split must contain every class.\n" + table.to_string())
    resolved = [audio_path(cfg.data_root, p) for p in frame.path]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Different metadata paths resolve to the same audio file.")
    return frame.reset_index(drop=True)


def read_metadata(cfg):
    path = Path(cfg.metadata_csv).expanduser()
    if not path.is_absolute():
        path = Path(cfg.data_root) / path
    return validate_metadata(pd.read_csv(path, dtype=str, keep_default_na=False), cfg)


def metadata_draft(root):
    rows = []
    for path in sorted(Path(root).glob("audio/*/*.wav")):
        rows.append(dict(path=path.relative_to(root).as_posix(), label=path.parent.name,
                         group="", split="", speaker_id="", room_session_id="", room="",
                         distance_cm="", device="", style="", source_url="", license="",
                         annotator="", notes="Fill actual speaker/session IDs and split before training"))
    if not rows:
        raise ValueError("No audio/<label>/*.wav files found.")
    return pd.DataFrame(rows)


class AudioDataset(Dataset):
    def __init__(self, frame, cfg: Config, train=False):
        self.frame = frame.reset_index(drop=True)
        self.cfg, self.train = cfg, train
        self.label_to_idx = {label: i for i, label in enumerate(cfg.labels)}
        self.samples = int(cfg.sample_rate * cfg.seconds)
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate, n_fft=cfg.n_fft, win_length=cfg.win_length,
            hop_length=cfg.hop_length, n_mels=cfg.n_mels, f_min=40,
            f_max=cfg.sample_rate // 2, power=2.0)
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
        self.freq_mask = torchaudio.transforms.FrequencyMasking(12)
        self.time_mask = torchaudio.transforms.TimeMasking(18)

    def __len__(self):
        return len(self.frame)

    def transform_file(self, path):
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        if audio.shape[0] == 0 or not np.isfinite(audio).all():
            raise ValueError(f"Empty or nonfinite audio: {path}")
        wav = torch.from_numpy(audio.T).mean(0, keepdim=True)
        if sr != self.cfg.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.cfg.sample_rate)
        if wav.shape[-1] < self.samples:
            wav = F.pad(wav, (0, self.samples - wav.shape[-1]))
        elif wav.shape[-1] > self.samples:
            start = (random.randint(0, wav.shape[-1] - self.samples) if self.train
                     else (wav.shape[-1] - self.samples) // 2)
            wav = wav[..., start:start + self.samples]
        if self.train and self.cfg.use_augmentation:
            wav = wav * random.uniform(0.7, 1.3)
            wav = torch.roll(wav, random.randint(-1600, 1600), dims=-1)
            if random.random() < 0.5:
                wav = wav + torch.randn_like(wav) * random.uniform(0.001, 0.015)
            wav = wav.clamp(-1, 1)
        spec = ((self.to_db(self.mel(wav)) + 80.0) / 80.0).clamp(0, 1)
        if self.train and self.cfg.use_augmentation:
            spec = self.time_mask(self.freq_mask(spec))
        return (spec - 0.5) / 0.25

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        spec = self.transform_file(audio_path(self.cfg.data_root, row.path))
        return spec, self.label_to_idx[row.label], row.path


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def make_loader(frame, cfg, split, device):
    frame = frame[frame.split == split]
    dataset = AudioDataset(frame, cfg, train=(split == "train"))
    generator = torch.Generator().manual_seed(cfg.seed)
    sampler = None
    if split == "train":
        counts = frame.label.value_counts()
        weights = [1.0 / counts[label] for label in frame.label]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True, generator=generator)
    return DataLoader(dataset, batch_size=cfg.batch_size, sampler=sampler, shuffle=False,
                      num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"),
                      worker_init_fn=seed_worker, generator=generator)
