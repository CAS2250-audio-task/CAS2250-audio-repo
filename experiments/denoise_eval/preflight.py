"""Check native-reference agreement and independence before scoring filters."""
import gc
import importlib.util
import sys
import time

import numpy as np
import scipy.signal
import torch

from src import engine
from .filters import ROOT, STUDY, make_filter
from experiments.noise_stress.run import read_wave


def nsnet_reference(wave):
    # Removed SciPy aliases are unused by the pinned featurelib but imported there.
    scipy.signal.blackman = scipy.signal.windows.blackman
    scipy.signal.blackmanharris = scipy.signal.windows.blackmanharris
    sys.path.insert(0, str(STUDY / "vendor/nsnet2"))
    from enhance_onnx import NSnet2Enhancer
    import onnxruntime as ort
    # Reference constructor creates an unbounded default ORT thread pool. Set the same
    # reference object fields directly, using the official __call__ + featurelib path.
    ref = NSnet2Enhancer.__new__(NSnet2Enhancer)
    ref.cfg = dict(winlen=.02, hopfrac=.5, fs=16000, mingain=-80, feattype="LogPow", nfft=320)
    ref.mingain = 1e-4
    ref.dtype = np.float32
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    ref.ort = ort.InferenceSession(str(STUDY / "vendor/nsnet2/nsnet2-20ms-baseline.onnx"),
                                   sess_options=options, providers=["CPUExecutionProvider"])
    return ref(wave, 16000)[:24000].astype(np.float32)


def main():
    torch.set_num_threads(2)
    engine.seed_all(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    paths = [ROOT / "datasets/task2/fixed-1p5-v1/audio/출발/spk02_052.wav",
             next((ROOT / "datasets/task2/fixed-1p5-v1/audio/정지").glob("spk03_*"))]
    original = np.stack([read_wave(p) for p in paths])
    # Different amplitudes, silence, and additive noise exercise per-clip state.
    rng = np.random.default_rng(601)
    x = np.concatenate([original, original*.1, np.zeros_like(original),
                        original+rng.normal(0,.06,original.shape).astype(np.float32)], axis=0)
    results = {}
    for name in ("deepfilternet3", "dns64", "nsnet2"):
        filt = make_filter(name)
        started = time.perf_counter()
        batched = filt(x)
        elapsed = time.perf_counter()-started
        single = np.concatenate([filt(w[None]) for w in x])
        reverse = filt(x[::-1].copy())[::-1]
        replay = filt(x)
        rms = float(np.sqrt(np.mean((batched-single)**2)/max(np.mean(batched**2), 1e-20)))
        order_rms = float(np.sqrt(np.mean((batched-reverse)**2)/max(np.mean(batched**2), 1e-20)))
        assert rms < .002 and order_rms < .002, (name, rms, order_rms)
        assert np.array_equal(batched, replay), name
        result = dict(shape=list(batched.shape), dtype=str(batched.dtype), finite=bool(np.isfinite(batched).all()),
                      batch_relative_rms=rms, order_relative_rms=order_rms,
                      repeated_call_bitwise_equal=True, first_batch_seconds=elapsed)
        if name == "nsnet2":
            reference = np.stack([nsnet_reference(w) for w in x[:2]])
            error = float(np.max(np.abs(reference-batched[:2])))
            assert error < 1e-6, error
            result["official_reference_max_abs_error"] = error
        if name == "deepfilternet3":
            from df.io import resample
            official_up = resample(torch.from_numpy(x), 16000, 48000)
            assert torch.equal(official_up, filt.up(torch.from_numpy(x)))
            result["resampling_matches_official_sinc_fast"] = True
        results[name] = result
        print("PREFLIGHT_OK", name, result, flush=True)
        del filt
        gc.collect()
        torch.cuda.empty_cache()
    engine.write_json(STUDY / "preflight.json", results)


if __name__ == "__main__":
    main()
