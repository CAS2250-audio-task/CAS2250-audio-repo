"""Recreate the existing 0 dB validation mixtures exactly, then pin the protocol."""
import json
import platform
import importlib.metadata
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from src.config import load_config
from src.dataset import read_metadata
from src.engine import file_hash, write_json
from experiments.noise_stress.mixing import mix_at_snr
from experiments.noise_stress.run import read_wave
from .filters import ROOT, STUDY, REPORT, FILTERS

FOLDS = ("spk02", "spk03", "spk04", "spk05")


def main():
    STUDY.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "configs/task2_real.json")
    metadata = read_metadata(cfg)
    expected = pd.read_csv(ROOT / "reports/conformer-aug-v1/predictions.csv.gz")
    expected = expected[expected.condition.eq("noise") &
                        (expected.kind.eq("clean") | expected.snr_db.eq(0))].copy()
    audit = pd.read_csv(ROOT / "reports/noise-stress-v1/mixtures.csv.gz", float_precision="round_trip")
    audit = audit[audit.snr_db.eq(0)].set_index(["heldout_speaker", "path", "noise_id", "draw"])
    sources = json.loads((ROOT / "reports/noise-stress-v1/noise_sources.json").read_text())
    recordings = {s["noise_id"]: s for s in sources}
    for source in sources:
        assert file_hash(ROOT / source["prepared_path"]) == source["prepared_sha256"]
    assert len(expected) == 20940 and len(audit) == 20128
    all_cases, dataset_pins, checkpoints = [], {}, {}
    development = metadata[metadata.group.isin(["speaker:"+s for s in FOLDS])]
    assert len(development) == 812
    originals = {}
    for path in development.path:
        filename = Path(cfg.data_root) / path
        originals[path] = read_wave(filename)
        dataset_pins[str(filename.relative_to(ROOT))] = file_hash(filename)
    for speaker in FOLDS:
        subset = expected[expected.heldout_speaker.eq(speaker)].reset_index(drop=True)
        assert len(subset) == 5235
        waves = np.empty((len(subset), 24000), dtype=np.float32)
        training = development[development.group.ne("speaker:"+speaker) & development.label.ne("silence")]
        assert len(training) == 600
        reference = float(np.median([float(np.sqrt(np.mean(originals[p].astype(np.float64)**2)))
                                    for p in training.path])**2)
        for i, row in enumerate(subset.itertuples()):
            if row.kind == "clean":
                wave = originals[row.path]
            else:
                draw = int(row.path.split("/")[-1]) if row.kind == "noise_only" else int(row.draw)
                detail = audit.loc[(speaker, row.path, row.noise_id, draw)]
                source = recordings[row.noise_id]
                noise, sr = sf.read(ROOT / source["prepared_path"], start=int(detail.start_sample),
                                    frames=24000, dtype="float32")
                assert sr == 16000 and noise.shape == (24000,)
                noise = noise.astype(np.float64)
                noise -= noise.mean()
                original = np.zeros(24000, dtype=np.float32) if row.kind == "noise_only" else originals[row.path]
                # Historical CSV powers were rounded; recompute from the exact original
                # waves, as in the frozen evaluation, rather than use rounded CSV floats.
                power = reference if row.true_idx == 10 else float(np.mean(original.astype(np.float64)**2))
                assert np.isclose(power, detail.reference_power, rtol=1e-10, atol=1e-15)
                wave, rebuilt = mix_at_snr(original, noise, 0, power)
                assert rebuilt["mixture_sha256"] == detail.mixture_sha256
            waves[i] = wave
        subset = subset.rename(columns={"pred_idx": "prior_pred_idx"}).drop(columns="condition")
        subset["case_index"] = np.arange(len(subset))
        import hashlib
        subset["input_sha256"] = [hashlib.sha256(w.tobytes()).hexdigest() for w in waves]
        subset.to_csv(STUDY / f"{speaker}_cases.csv.gz", index=False)
        np.save(STUDY / f"{speaker}_inputs.npy", waves)
        all_cases.append(subset)
        path = ROOT / f"runs/conformer-aug-v1/folds/{speaker}/noise/best.pt"
        summary = json.loads((path.parent / "summary.json").read_text())
        assert file_hash(path) == summary["checkpoint_sha256"]
        checkpoints[str(path.relative_to(ROOT))] = file_hash(path)
        print("INPUTS_PREPARED", speaker, len(subset), flush=True)
    cases = pd.concat(all_cases, ignore_index=True)
    assert set(cases.heldout_speaker) == set(FOLDS) and not cases.path.str.contains("spk01").any()
    cases.to_csv(REPORT / "cases.csv.gz", index=False)
    vendor_pins = {}
    for folder in (STUDY / "vendor/DeepFilterNet/DeepFilterNet/df", STUDY / "vendor/denoiser/denoiser", STUDY / "vendor/nsnet2", STUDY / "models"):
        for path in folder.rglob("*"):
            if path.is_file() and path.suffix not in (".pyc",) and "__pycache__" not in path.parts:
                vendor_pins[str(path.relative_to(ROOT))] = file_hash(path)
    source_pins = {str(p.relative_to(ROOT)): file_hash(p) for p in (ROOT / "experiments/denoise_eval").glob("*.py")}
    for name in ("experiments/conformer/model.py", "experiments/noise_stress/mixing.py", "src/engine.py"):
        source_pins[name] = file_hash(ROOT / name)
    dependencies = ("reports/noise-stress-v1/protocol.json", "reports/noise-stress-v1/mixtures.csv.gz",
                    "reports/conformer-aug-v1/predictions.csv.gz", "reports/conformer-aug-v1/protocol.json")
    input_pins = {str(p.relative_to(ROOT)): file_hash(p) for p in STUDY.glob("spk*_*")}
    revisions = {name: json.loads((STUDY / f"{name}-revision.json").read_text()) for name in ("DeepFilterNet", "denoiser", "nsnet2")}
    protocol = dict(version="denoise-eval-v1", classifier="noise-trained Conformer only", folds=list(FOLDS),
        filters=list(FILTERS), primary="0 dB macro-F1, equal mean over fold x noise x draw",
        secondary=["original validation macro-F1", "0 dB nominal noise-only false command rate", "batch-one added processing time"],
        snr_db=[0], mixed_cases=19488, clean_cases=812, noise_only_cases=640,
        data="ref fixed-1p5-v1, mono 16 kHz 24000 samples", classifier_training=False, denoiser_training=False,
        checkpoint_selection="existing clean-best epochs; no denoised/noisy epoch selection",
        oracle_input=False, filter_input="mixture waveform only; no clean/noise reference, label, noise ID, or SNR",
        parameters="official full enhancement; no postfilter, dry blend, attenuation limit, fitted gain or peak normalization",
        batching="16 independent clips per call; all recurrent/feature state resets; order invariance tested",
        analysis="paired development CV, 4 speakers and one training seed; filter selection is development tuning",
        pretrained_denoiser_noise_overlap="Not fully audited against external DNS pretraining corpora; sources are held out from our classifier training only",
        deepfilternet3=dict(native_rate=48000, resample="torchaudio sinc_interp_hann, width16", pad=True,
                           stft_delay_compensation_samples_48k=480, output_rate=16000, output_samples=24000),
        nsnet2=dict(native_rate=16000, fft=320, hop=160, window="sqrt symmetric Hann", min_gain_db=-80,
                    synthesis_right_flush_removed=160, output_samples=24000),
        dns64=dict(native_rate=16000, dry=0, output_samples=24000),
        official_revisions=revisions, python=platform.python_version(),
        packages={p: importlib.metadata.version(p) for p in ("torch", "torchaudio", "speechbrain", "numpy", "onnxruntime", "deepfilterlib")},
        preflight_sha256=file_hash(STUDY / "preflight.json"), source_sha256=source_pins,
        checkpoint_sha256=checkpoints, vendor_sha256=vendor_pins, input_sha256=input_pins,
        dataset_sha256=dataset_pins, dependency_sha256={p: file_hash(ROOT / p) for p in dependencies},
        source_noise_sha256={s["prepared_path"]: s["prepared_sha256"] for s in sources}, test_evaluated=False,
        dns64_download="Official public URL had expired TLS certificate; isolated public-file download verified against official 64-bit SHA256 filename prefix before weights_only loading; full SHA256 pinned here")
    target = REPORT / "protocol.json"
    if target.exists():
        assert json.loads(target.read_text()) == protocol, "Frozen protocol differs; use a new study"
    else:
        write_json(target, protocol)
    print("PROTOCOL_FROZEN", len(cases), flush=True)


if __name__ == "__main__":
    main()
