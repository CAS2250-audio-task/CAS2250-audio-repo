"""Import tensor weights from the public CLAR Speech Commands checkpoint."""
import pickle
import re
from pathlib import Path
import urllib.request

import torch

from src.engine import file_hash

EXPECTED_SHA256 = "ac36f2f4d6f25dfe792a2ea4699cc046630903decc5c948a700554741bbc5b7f"
SOURCE_URL = ("https://drive.usercontent.google.com/download?"
              "id=1xZP8eaFoW7aOiVU5p8hN-cfCTaKKrREg&export=download&confirm=t")


def download_checkpoint(path):
    path = Path(path)
    if path.exists():
        if file_hash(path) != EXPECTED_SHA256:
            raise ValueError("Existing checkpoint has a different hash")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=45) as response:
        if "text/html" in response.headers.get("Content-Type", ""):
            raise ValueError("Checkpoint download returned an HTML page")
        with temporary.open("wb") as stream:
            while block := response.read(1024 * 1024):
                stream.write(block)
    if file_hash(temporary) != EXPECTED_SHA256:
        raise ValueError("Downloaded checkpoint does not match the pinned hash")
    temporary.replace(path)
    return path


class InertMetadata(dict):
    """Consume legacy Lightning metadata without importing its saved callables."""
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __setstate__(self, value):
        self.saved_state = value


def tensor_payload(path):
    if file_hash(path) != EXPECTED_SHA256:
        raise ValueError("Speech pretraining checkpoint hash does not match the pinned source")
    aliases = []
    for _ in range(40):
        try:
            with torch.serialization.safe_globals([(InertMetadata, name) for name in aliases]):
                payload = torch.load(path, map_location="cpu", weights_only=True)
            return payload, aliases
        except pickle.UnpicklingError as error:
            match = re.search(r"Unsupported global: GLOBAL ([\w.]+)", str(error))
            if not match or match[1] in aliases:
                raise
            aliases.append(match[1])
    raise ValueError("Too many legacy metadata types in checkpoint")


def encoder_weights(path):
    payload, aliases = tensor_payload(path)
    state = payload["state_dict"]
    if not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("Checkpoint state_dict contains non-tensor values")
    params = payload.get("hyper_parameters", {})
    info = {k: params[k] for k in ("sampling_rate", "dataset", "model", "n_classes") if k in params}
    info.update(checkpoint_epoch=payload.get("epoch"), inert_metadata_aliases=aliases)
    if params.get("dataset") != "sc09" or params.get("n_classes") != 10:
        raise ValueError(f"Unexpected source dataset/task: {info}")
    encoded = {k.removeprefix("model."): v for k, v in state.items() if k.startswith("model.")}
    return encoded, info


def transfer(model, encoded):
    """Keep ref architecture and 512→11 head; select source's log-mel channel."""
    expected = {k: v for k, v in model.state_dict().items() if not k.startswith("fc.")}
    copied = {}
    for key, target in expected.items():
        if key not in encoded:
            raise ValueError(f"Speech checkpoint is missing backbone tensor {key}")
        source = encoded[key]
        if key == "conv1.weight":
            if tuple(source.shape) != (64, 3, 7, 7):
                raise ValueError(f"Unexpected source convolution: {source.shape}")
            # CLAR's channels are STFT magnitude, STFT phase, log-mel, in that order.
            source = source[:, 2:3].clone()
        if source.shape != target.shape:
            raise ValueError(f"Incompatible tensor {key}: {source.shape} vs {target.shape}")
        copied[key] = source
    result = model.load_state_dict(copied, strict=False)
    if set(result.missing_keys) != {"fc.weight", "fc.bias"} or result.unexpected_keys:
        raise ValueError(f"Unexpected transfer result: {result}")
    return dict(copied_backbone_tensors=len(copied),
                copied_backbone_elements=sum(v.numel() for v in copied.values()),
                retained_reference_classifier=[512, 11], source_channel_used="log_mel_index_2",
                classifier_loaded_from_pretraining=False)
