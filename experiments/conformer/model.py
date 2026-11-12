"""Native SpeechBrain features and encoder, with a new keyword classifier."""
from pathlib import Path
from contextlib import nullcontext
import urllib.request

import torch
from torch import nn

from src.engine import file_hash

REPOSITORY = "speechbrain/asr-conformer-transformerlm-ksponspeech"
REVISION = "ecb74648052c89618cd1ce53ccbc909d1141f164"
SOURCE_HASHES = {
    "asr.ckpt": "8e718dc29b403dfaa8d2604c43c3666be3fa99e958b77e3c6ff387e94d4a174c",
    "normalizer.ckpt": "1da2ced935d955c014177591249e5db497d0c5dc7143e64378da0cb5590fe77a",
    "hyperparams.yaml": "e6a2f1f86947ebcd72e3421d4b84e306ff823eda1dd5c44074c41bf21d28e44e",
    "README.md": "8714c4c3c2edbf9d07ab25c2c1f386e28443b53a35e2af6496d1cc2160dfc380",
}


def source_files(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, expected in SOURCE_HASHES.items():
        target = directory / name
        if not target.exists():
            partial = target.with_suffix(target.suffix + ".partial")
            url = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}"
            with urllib.request.urlopen(url, timeout=90) as response, partial.open("wb") as stream:
                while data := response.read(1024 * 1024):
                    stream.write(data)
            if file_hash(partial) != expected:
                raise ValueError(f"Downloaded {name} has the wrong hash")
            partial.replace(target)
        if file_hash(target) != expected:
            raise ValueError(f"Pinned source hash mismatch: {name}")
    return directory


def load_source(directory):
    directory = source_files(directory)
    state = torch.load(directory / "asr.ckpt", map_location="cpu", weights_only=True)
    statistics = torch.load(directory / "normalizer.ckpt", map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("Expected a plain tensor state dictionary")
    for key in ("glob_mean", "glob_std"):
        value = statistics[key]
        if value.shape != (80,) or not torch.isfinite(value).all():
            raise ValueError(f"Invalid normalization statistics: {key}")
    if not (statistics["glob_std"] > 0).all():
        raise ValueError("Normalization standard deviations must be positive")
    return state, statistics


def transfer_module(module, state, prefix):
    selected = {}
    for name, target in module.state_dict().items():
        key = prefix + name
        if key not in state:
            raise ValueError(f"Missing pretrained encoder tensor: {key}")
        if target.shape != state[key].shape:
            raise ValueError(f"Incompatible pretrained encoder tensor: {key}")
        selected[name] = state[key]
    module.load_state_dict(selected, strict=True)
    return len(selected)


class ConformerClassifier(nn.Module):
    """All supplied samples are real 1.5-second clips; no long-window padding."""

    def __init__(self, state, statistics, mode="frozen", num_classes=11):
        super().__init__()
        if mode not in ("frozen", "finetune"):
            raise ValueError(f"Unknown adaptation mode: {mode}")
        # Optional dependency stays out of the reference installation/imports.
        from speechbrain.lobes.features import Fbank
        from speechbrain.lobes.models.convolution import ConvolutionFrontEnd
        from speechbrain.lobes.models.transformer.TransformerASR import TransformerASR

        self.mode = mode
        self.features = Fbank(sample_rate=16000, n_fft=400, n_mels=80)
        self.register_buffer("feature_mean", statistics["glob_mean"].clone())
        self.register_buffer("feature_std", statistics["glob_std"].clone())
        self.frontend = ConvolutionFrontEnd(
            input_shape=(8, 10, 80), num_blocks=3, num_layers_per_block=1,
            out_channels=(64, 64, 64), kernel_sizes=(5, 5, 1),
            strides=(2, 2, 1), residuals=(False, False, True))
        self.transformer = TransformerASR(
            tgt_vocab=5000, input_size=1280, d_model=256, nhead=4,
            num_encoder_layers=12, num_decoder_layers=0, d_ffn=2048,
            dropout=0.1, activation=nn.GELU, encoder_module="conformer",
            attention_type="RelPosMHAXL", normalize_before=True, causal=False)
        # num_decoder_layers=0 omits the decoder and its target-language embedding.
        self.copied_tensors = transfer_module(self.frontend, state, "0.")
        self.copied_tensors += transfer_module(self.transformer, state, "1.")
        self.fc = nn.Linear(256, num_classes)
        for module in (self.features, self.frontend, self.transformer):
            for parameter in module.parameters():
                parameter.requires_grad_(mode == "finetune" and module is not self.features)
        self.train()

    def train(self, mode=True):
        super().train(mode)
        self.features.eval()
        if self.mode == "frozen":
            self.frontend.eval()
            self.transformer.eval()
        return self

    def encode(self, wav):
        if wav.ndim != 2 or wav.shape[1] < 400:
            raise ValueError("Expected batch × samples of 16 kHz mono audio")
        with torch.no_grad():
            mel = self.features(wav)
            # Equivalent to the source global InputNormalization in eval mode.
            normalized = (mel - self.feature_mean) / self.feature_std
        context = torch.no_grad() if self.mode == "frozen" else nullcontext()
        with context:
            encoded = self.frontend(normalized)
            lengths = torch.ones(wav.shape[0], device=wav.device)
            return self.transformer.encode(encoded, wav_len=lengths)

    def forward(self, wav):
        return self.fc(self.encode(wav).mean(dim=1))
