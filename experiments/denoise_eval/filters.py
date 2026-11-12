"""Mixture-only 16 kHz adapters; no clean signal, label, or noise reference input."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "runs/denoise-eval-v1"
REPORT = ROOT / "reports/denoise-eval-v1"
FILTERS = ("none", "deepfilternet3", "dns64", "nsnet2")


class Filter:
    def __call__(self, mixtures):
        x = np.asarray(mixtures)
        if x.dtype != np.float32 or x.ndim != 2 or x.shape[1] != 24000 or not np.isfinite(x).all():
            raise ValueError("Expected finite float32 batch of 16 kHz, 1.5-second mixtures")
        with torch.inference_mode():
            out = np.ascontiguousarray(self.process(x), dtype=np.float32)
        if out.shape != x.shape or not np.isfinite(out).all():
            raise ValueError("Invalid denoiser output")
        # No peak normalization, clipping, fitted gain, or oracle alignment.
        return out


class Identity(Filter):
    def process(self, x):
        return x.copy()


class DeepFilterNet3(Filter):
    def __init__(self):
        sys.path.insert(0, str(STUDY / "vendor/DeepFilterNet/DeepFilterNet"))
        from df.enhance import init_df, enhance
        from torchaudio.transforms import Resample
        self.model, self.state, _, self.epoch = init_df(
            str(STUDY / "models/DeepFilterNet3"), post_filter=False,
            log_level="ERROR", log_file=None, config_allow_defaults=True)
        self.enhance = enhance
        # Official df.io sinc_fast settings, cached kernels.
        self.up = Resample(16000, 48000, lowpass_filter_width=16, dtype=torch.float32)
        self.down = Resample(48000, 16000, lowpass_filter_width=16, dtype=torch.float32)

    def process(self, x):
        audio = self.up(torch.from_numpy(x))
        # Official Rust analysis/synthesis reset per channel; neural state resets per call.
        enhanced = self.enhance(self.model, self.state, audio, pad=True, atten_lim_db=None)
        return self.down(enhanced).numpy()


class DNS64(Filter):
    def __init__(self):
        sys.path.insert(0, str(STUDY / "vendor/denoiser"))
        from denoiser.demucs import Demucs
        self.model = Demucs(hidden=64, sample_rate=16000)
        state = torch.load(STUDY / "models/dns64/dns64-a7761ff99a7d5bb6.th",
                           map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.cuda().eval()

    def process(self, x):
        # Official forward restores its mixture-derived scale; dry=0, fully enhanced.
        return self.model(torch.from_numpy(x).cuda()).squeeze(1).cpu().numpy()


class NSNet2(Filter):
    """Batched equivalent of Microsoft's pinned 16 kHz featurelib + ONNX path."""
    def __init__(self):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.model = ort.InferenceSession(
            str(STUDY / "vendor/nsnet2/nsnet2-20ms-baseline.onnx"),
            sess_options=options, providers=["CPUExecutionProvider"])
        self.input_name = self.model.get_inputs()[0].name
        self.window = np.sqrt(np.hanning(320))

    def process(self, x):
        # 20 ms sqrt symmetric Hann, 10 ms hop, right zero flush; official nodelay=True.
        padded = np.pad(x.astype(np.float64), ((0, 0), (0, 160)))
        framed = np.lib.stride_tricks.sliding_window_view(padded, 320, axis=1)[:, ::160]
        spectrum = np.fft.rfft(framed * self.window, n=320, axis=-1)
        features = np.log10(np.maximum(np.abs(spectrum)**2, 1e-12)).astype(np.float32)
        gains = self.model.run(None, {self.input_name: features})[0]
        gains = np.clip(gains, 1e-4, 1.0)  # Official mingain=-80 dB.
        frames = np.fft.irfft(spectrum * gains, n=320, axis=-1) * self.window
        result = np.zeros((len(x), 24160), dtype=np.float64)
        for frame in range(150):
            result[:, frame*160:frame*160+320] += frames[:, frame]
        # Official synthesis has a 160-sample right flush. Preserve original time origin.
        return result[:, :24000]


def make_filter(name):
    return {"none": Identity, "deepfilternet3": DeepFilterNet3,
            "dns64": DNS64, "nsnet2": NSNet2}[name]()
