"""Portable JSON configuration; metadata paths are relative to the data root."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass
class Config:
    labels: list[str]
    data_root: str | None = None
    metadata_csv: str = "metadata.csv"
    dataset_version: str = "uncollected"
    split_version: str = "unassigned"
    dummy: bool = False
    sample_rate: int = 16000
    seconds: float = 1.5
    n_fft: int = 512
    win_length: int = 512
    hop_length: int = 160
    n_mels: int = 128
    batch_size: int = 32
    epochs: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 0
    use_pretrained: bool = False
    use_augmentation: bool = True
    seed: int = 42
    device: str = "auto"

    def validate(self):
        if len(self.labels) != 11 or len(set(self.labels)) != 11 or "silence" not in self.labels:
            raise ValueError("Task 2 requires ten unique keywords + silence (11 labels).")
        if any(not isinstance(x, str) or not x.strip() or x != x.strip() for x in self.labels):
            raise ValueError("Labels must be nonempty strings without surrounding whitespace.")
        for key in ("sample_rate", "seconds", "n_fft", "win_length", "hop_length", "n_mels",
                    "batch_size", "epochs", "learning_rate"):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive.")
        if self.win_length > self.n_fft or self.num_workers < 0 or self.weight_decay < 0:
            raise ValueError("Invalid window, worker count or weight decay.")
        if self.sample_rate <= 80 or int(self.seconds * self.sample_rate) < 1:
            raise ValueError("Invalid audio length/sample rate.")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu or cuda.")

    def to_dict(self):
        return asdict(self)


def load_config(path, data_root=None, metadata_csv=None, device=None, epochs=None):
    cfg = Config(**json.loads(Path(path).read_text(encoding="utf-8")))
    if data_root is not None:
        cfg.data_root = str(Path(data_root).expanduser().resolve())
    elif cfg.data_root is not None:
        root = Path(cfg.data_root).expanduser()
        # Config-relative paths work regardless of the shell's working directory.
        cfg.data_root = str((Path(path).resolve().parent / root).resolve())
    for key, value in (("metadata_csv", metadata_csv), ("device", device), ("epochs", epochs)):
        if value is not None:
            setattr(cfg, key, value)
    cfg.validate()
    if not cfg.data_root:
        raise ValueError("Real data is not collected yet. Supply --data-root when available.")
    return cfg
