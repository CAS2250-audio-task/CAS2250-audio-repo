"""Resumable fixed-model inference; filtered wave caches are local ignored artifacts."""
import gc
import hashlib
import json
import time

import numpy as np
import pandas as pd
import torch

from src import engine
from experiments.conformer.model import ConformerClassifier, load_source
from .filters import ROOT, STUDY, REPORT, FILTERS, make_filter


def verify_protocol():
    protocol = json.loads((REPORT / "protocol.json").read_text())
    for group in ("source_sha256", "checkpoint_sha256", "vendor_sha256", "input_sha256", "dependency_sha256", "dataset_sha256", "source_noise_sha256"):
        for path, expected in protocol[group].items():
            assert engine.file_hash(ROOT / path) == expected, path
    assert engine.file_hash(STUDY / "preflight.json") == protocol["preflight_sha256"]
    return protocol


def main():
    torch.set_num_threads(2)
    engine.seed_all(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    protocol = verify_protocol()
    state, statistics = load_source(ROOT / "runs/conformer-research")
    partials = STUDY / "parts"
    partials.mkdir(exist_ok=True)
    times = []
    for name in FILTERS:
        print("FILTER_START", name, flush=True)
        filt = make_filter(name)
        for speaker in protocol["folds"]:
            target = partials / f"{name}_{speaker}.csv.gz"
            if target.exists():
                print("REUSE", name, speaker, flush=True)
                continue
            cases = pd.read_csv(STUDY / f"{speaker}_cases.csv.gz")
            x = np.load(STUDY / f"{speaker}_inputs.npy", mmap_mode="r")
            cache = STUDY / "filtered" / name
            cache.mkdir(parents=True, exist_ok=True)
            cache_path = cache / f"{speaker}.npy"
            y = np.lib.format.open_memmap(cache_path.with_suffix(".partial.npy"), mode="w+", dtype=np.float32, shape=x.shape)
            hashes = []
            started = time.perf_counter()
            for offset in range(0, len(x), 16):
                out = filt(np.array(x[offset:offset+16]))
                y[offset:offset+len(out)] = out
                hashes.extend(hashlib.sha256(w.tobytes()).hexdigest() for w in out)
                if offset % 1024 == 0:
                    print("FILTER_PROGRESS", name, speaker, offset, len(x), flush=True)
            y.flush()
            seconds = time.perf_counter() - started
            del y
            cache_path.with_suffix(".partial.npy").replace(cache_path)
            times.append(dict(filter=name, heldout_speaker=speaker, n=len(x), filter_seconds=seconds))
            y = np.load(cache_path, mmap_mode="r")
            model = ConformerClassifier(state, statistics, "finetune")
            checkpoint = ROOT / f"runs/conformer-aug-v1/folds/{speaker}/noise/best.pt"
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"], strict=True)
            model.cuda().eval()
            predictions = []
            with torch.inference_mode():
                for offset in range(0, len(y), 32):
                    batch = torch.from_numpy(np.array(y[offset:offset+32])).cuda()
                    predictions.extend(model(batch).argmax(1).cpu().tolist())
            if name == "none":
                assert np.array_equal(predictions, cases.prior_pred_idx), "Unfiltered baseline replay differs"
                assert hashes == cases.input_sha256.tolist()
            cases["filter"] = name
            cases["pred_idx"] = predictions
            cases["output_sha256"] = hashes
            cases.to_csv(target.with_suffix(".partial.gz"), index=False)
            target.with_suffix(".partial.gz").replace(target)
            engine.write_json(cache / f"{speaker}_audit.json", dict(
                input_cases=len(x), output_shape=list(y.shape), output_dtype=str(y.dtype),
                peak=float(np.max(np.abs(y))), samples_outside_unit_range=int((np.abs(y)>1).sum()),
                cache_sha256=engine.file_hash(cache_path), filter_seconds=seconds))
            del model, x, y
            gc.collect()
            torch.cuda.empty_cache()
            print("FOLD_DONE", name, speaker, "filter_seconds", round(seconds, 2), flush=True)
        del filt
        gc.collect()
        torch.cuda.empty_cache()
    predictions = pd.concat([pd.read_csv(partials / f"{name}_{speaker}.csv.gz")
                             for name in FILTERS for speaker in protocol["folds"]], ignore_index=True)
    assert len(predictions) == 83760
    predictions.to_csv(REPORT / "predictions.csv.gz", index=False)
    pd.DataFrame(times).to_csv(STUDY / "throughput_times.csv", index=False)
    engine.write_json(STUDY / "complete.json", dict(prediction_rows=len(predictions), test_evaluated=False))
    print("DENOISE_EVALUATION_COMPLETE", len(predictions), flush=True)


if __name__ == "__main__":
    main()
