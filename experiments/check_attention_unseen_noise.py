"""Recompute F1 and regenerate new mixtures independently from saved crop traces."""
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from src import engine
from experiments.conformer.model import load_source
from experiments.attention_pitch.model import MaskedConformer
from experiments.attention_pitch.vad import Detector, wave_hash

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/attention-unseen-noise-v1"
STUDY = ROOT / "runs/attention-unseen-noise-v1"
KEYS = ["suite", "heldout_speaker", "kind", "noise_id", "family", "snr_db", "draw"]


@torch.inference_mode()
def main():
    started = time.monotonic()
    assert (STUDY / "complete.json").exists()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    engine.seed_all(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    predictions = pd.read_csv(REPORT / "predictions.csv.gz")
    assert len(predictions) == 38552 and not predictions.path.str.contains("spk01_").any()
    expected = pd.read_csv(REPORT / "units.csv").set_index(KEYS)
    count = 0
    for keys, group in predictions.groupby(KEYS, sort=False):
        confusion = np.zeros((11, 11), np.int64)
        np.add.at(confusion, (group.true_idx.to_numpy(), group.pred_idx.to_numpy()), 1)
        denominator = confusion.sum(0) + confusion.sum(1)
        f1 = np.divide(2*np.diag(confusion), denominator,
                       out=np.zeros(11, dtype=np.float64), where=denominator != 0).mean()
        actual = expected.loc[keys]
        assert abs(float(actual.macro_f1)-f1) < 1e-12
        assert abs(float(actual.accuracy)-np.trace(confusion)/confusion.sum()) < 1e-12
        count += 1
    assert count == 244
    existing = predictions[predictions.suite.eq("existing")]
    replayed = 0
    originals = {}
    for speaker in ("spk02", "spk03", "spk04", "spk05"):
        prior = pd.read_csv(ROOT / f"runs/attention-pitch-v1/candidates/attention/{speaker}/best_predictions.csv.gz")
        actual = existing[existing.heldout_speaker.eq(speaker)]
        prior = prior.set_index("case_index").loc[actual.case_index]
        assert actual.pred_idx.tolist() == prior.pred_idx.tolist()
        assert actual.input_sha256.tolist() == prior.input_sha256.tolist()
        replayed += len(actual)
        cases = pd.read_csv(ROOT / f"runs/denoise-eval-v1/{speaker}_cases.csv.gz")
        raw = np.load(ROOT / f"runs/denoise-eval-v1/{speaker}_inputs.npy", mmap_mode="r")
        clean = cases[cases.kind.eq("clean")]
        originals[speaker] = {row.path: (int(row.true_idx), np.array(raw[int(row.case_index)]))
                              for row in clean.itertuples()}
    sources = json.loads((REPORT / "noise_sources.json").read_text(encoding="utf-8"))
    recordings = {s["noise_id"]: sf.read(ROOT / s["prepared_path"], dtype="float32")[0] for s in sources}
    regenerated = 0
    replay_samples = {}
    new = predictions[predictions.suite.eq("new")]
    for speaker, group in new.groupby("heldout_speaker", sort=False):
        rms = [float(np.sqrt(np.mean(w.astype(np.float64)**2)))
               for other, entries in originals.items() if other != speaker
               for label, w in entries.values() if label != 10]
        assert len(rms) == 600
        reference = float(np.median(rms)**2)
        selected_source = group.noise_id.iloc[0]
        sample_waves, sample_predictions = [], []
        for row in group.itertuples():
            recording = recordings[row.noise_id]
            noise = recording[int(row.start_sample):int(row.end_sample)].astype(np.float64)
            assert len(noise) == 24000
            noise -= noise.mean()
            if row.kind == "noise_only":
                wave = np.zeros(24000, np.float32)
                assert int(row.true_idx) == 10
                power = reference
            else:
                label, wave = originals[speaker][row.path]
                assert label == int(row.true_idx)
                power = reference if label == 10 else float(np.mean(wave.astype(np.float64)**2))
            noise_power = float(np.mean(noise**2))
            gain = float(np.sqrt(power/noise_power))
            mixed = wave.astype(np.float64) + gain*noise
            peak = float(np.abs(mixed).max())
            attenuation = min(1., .99/peak) if peak else 1.
            mixed = (mixed*attenuation).astype(np.float32)
            assert wave_hash(mixed) == row.input_sha256
            assert abs(float(row.reference_power)-power) < 1e-12
            assert abs(10*np.log10(power/(noise_power*gain**2))) < 1e-8
            regenerated += 1
            if row.kind == "mixed" and row.noise_id == selected_source and int(row.draw) == 0:
                sample_waves.append(mixed)
                sample_predictions.append(int(row.pred_idx))
        assert len(sample_waves) == 203
        replay_samples[speaker] = (np.stack(sample_waves), sample_predictions)
        print("MIXTURE_AUDIT_OK", speaker, len(group), flush=True)
    assert regenerated == 17612 and replayed == 20940
    state, statistics = load_source(ROOT / "runs/conformer-research")
    replay_count = 0
    for speaker, (waves, wanted) in replay_samples.items():
        detector = Detector(ROOT / "runs/attention-pitch-v1/assets/silero_vad.jit")
        model = MaskedConformer(state, statistics, speaker, detector, masking=True).cuda().eval()
        checkpoint = ROOT / f"runs/attention-pitch-v1/candidates/attention/{speaker}/best.pt"
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"], strict=True)
        actual = []
        for offset in range(0, len(waves), 32):
            actual.extend(model(torch.from_numpy(waves[offset:offset+32].copy()).cuda()).argmax(1).cpu().tolist())
        assert actual == wanted
        replay_count += len(actual)
        del model, detector
        torch.cuda.empty_cache()
    audit = dict(prediction_rows_checked=len(predictions), independent_confusion_matrix_units=count,
                 new_mixtures_independently_regenerated=regenerated,
                 previous_attention_predictions_matched=replayed,
                 fresh_vad_and_classifier_predictions_replayed=replay_count,
                 source_sha256=engine.file_hash(Path(__file__)),
                 filters_used=False, training_performed=False, test_audio_read=False, test_evaluated=False,
                 elapsed_seconds=time.monotonic()-started)
    engine.write_json(REPORT / "independent_audit.json", audit)
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()
