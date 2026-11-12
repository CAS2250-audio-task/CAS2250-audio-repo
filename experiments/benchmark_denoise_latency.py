"""Batch-one CPU-waveform to CPU-waveform filter time and end-to-end KWS time."""
import gc
import json
import platform
import time

import numpy as np
import pandas as pd
import torch

from src import engine
from experiments.conformer.model import ConformerClassifier, load_source
from experiments.denoise_eval.filters import ROOT, STUDY, REPORT, FILTERS, make_filter


@torch.inference_mode()
def main():
    assert (STUDY / "complete.json").exists()
    engine.seed_all(42)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cases = pd.read_csv(STUDY / "spk02_cases.csv.gz")
    row = cases[cases.kind.eq("mixed") & cases.noise_id.eq("traffic") & cases.true_idx.eq(0)].iloc[0]
    wave = np.load(STUDY / "spk02_inputs.npy", mmap_mode="r")[int(row.case_index):int(row.case_index)+1].copy()
    state, statistics = load_source(ROOT / "runs/conformer-research")
    model = ConformerClassifier(state, statistics, "finetune")
    model.load_state_dict(torch.load(ROOT / "runs/conformer-aug-v1/folds/spk02/noise/best.pt",
                                    map_location="cpu", weights_only=True)["model"], strict=True)
    model.cuda().eval()
    samples, summary = [], []
    for name in FILTERS:
        filt = make_filter(name)
        for phase in ("filter_only", "filter_plus_classifier"):
            def call():
                y = filt(wave)
                if phase == "filter_plus_classifier":
                    return model(torch.from_numpy(y).cuda()).argmax(1).cpu().tolist()
                return y
            for _ in range(20):
                call()
            ms = []
            for repeat in range(100):
                torch.cuda.synchronize()
                start = time.perf_counter()
                call()
                torch.cuda.synchronize()
                elapsed = (time.perf_counter()-start)*1000
                ms.append(elapsed)
                samples.append(dict(filter=name, phase=phase, repeat=repeat, milliseconds=elapsed))
            result = dict(filter=name, phase=phase, median_ms=float(np.median(ms)),
                          p95_ms=float(np.percentile(ms,95)), mean_ms=float(np.mean(ms)))
            summary.append(result)
            print("LATENCY", result, flush=True)
        del filt
        gc.collect()
        torch.cuda.empty_cache()
    pd.DataFrame(samples).to_csv(REPORT / "latency_samples.csv", index=False)
    pd.DataFrame(summary).to_csv(REPORT / "latency_summary.csv", index=False)
    engine.write_json(REPORT / "latency_protocol.json", dict(
        gpu=torch.cuda.get_device_name(), python=platform.python_version(), torch=torch.__version__,
        batch_size=1, input="CPU float32 mixture, mono16k,24000 samples", output="CPU waveform or CPU predicted index",
        warmup=20, repetitions=100, torch_threads=2, tf32=True,
        included="resampling, state reset, enhancement, output crop, CPU/GPU transfers; plus native Fbank and classifier in total",
        excluded="model loading, WAV decode, input acquisition/window wait, networking",
        streaming=False, window_seconds=1.5, fixture=row.to_dict(),
        note="Processing time for an available whole window; not streaming algorithmic delay or time from speech onset"))


if __name__ == "__main__":
    main()
