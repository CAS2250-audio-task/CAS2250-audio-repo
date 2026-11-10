"""Offline keyword preprocessing. Originals and metadata identities are preserved."""
import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

from .dataset import audio_path


@dataclass(frozen=True)
class Options:
    seconds: float = 1.5
    sample_rate: int = 16000
    margin_ms: int = 200
    threshold: float = 0.5
    target_dbfs: float = -20.0
    max_gain_db: float = 12.0
    peak_dbfs: float = -1.0
    min_contrast_db: float = 6.0

    def __post_init__(self):
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError("Options must be finite")
        if self.sample_rate != 16000 or self.seconds <= 0 or self.margin_ms < 0:
            raise ValueError("Use 16 kHz, positive seconds and nonnegative margin")
        if not 0 < self.threshold < 1 or self.max_gain_db < 0:
            raise ValueError("Invalid threshold/gain")
        if self.target_dbfs > 0 or self.peak_dbfs > 0:
            raise ValueError("dBFS targets must be <= 0")


class SileroDetector:
    def __init__(self, options):
        from silero_vad import get_speech_timestamps, load_silero_vad
        self.model = load_silero_vad()
        self.get_timestamps = get_speech_timestamps
        self.options = options

    def __call__(self, wav):
        return self.get_timestamps(
            wav, self.model, sampling_rate=16000, threshold=self.options.threshold,
            min_speech_duration_ms=100, min_silence_duration_ms=200,
            speech_pad_ms=0, return_seconds=False,
        )


def rms(wav):
    return float(wav.square().mean().sqrt()) if wav.numel() else 0.0


def prepare_clip(wav, segments, label, options, *, keep_review=False):
    """Return a fixed-length clip or None plus auditable review diagnostics.

    keep_review exports a flagged listening candidate without approving it.
    Overlong candidates retain their length instead of losing speech.
    Detector timestamps use unpadded 16-kHz sample indices. Contrast is only a
    foreground/background energy heuristic, not an estimated true speech SNR.
    """
    n, sr = wav.numel(), options.sample_rate
    target = round(options.seconds * sr)
    flags = []
    info = dict(segment_count=len(segments), source_seconds=n / sr,
                speech_seconds=0.0, crop_start_seconds=0.0, crop_end_seconds=n / sr,
                contrast_db=None, gain_db=0.0)
    if n == 0 or not torch.isfinite(wav).all():
        return None, dict(info, review_reasons="empty_or_nonfinite")
    if float(wav.abs().max()) >= 0.999:
        flags.append("possible_source_clipping")
    if label == "silence":
        if segments:
            flags.append("speech_in_silence")
        start = max(0, (n - target) // 2)
        clip = wav[start:start + target]
        info.update(crop_start_seconds=start / sr, crop_end_seconds=(start + clip.numel()) / sr)
        # Preserve the noise level of negative examples; never amplify silence.
    else:
        if not segments:
            return None, dict(info, review_reasons=";".join(flags + ["no_speech"]))
        start, end = segments[0]["start"], segments[-1]["end"]
        mask = torch.zeros(n, dtype=torch.bool)
        for segment in segments:
            mask[segment["start"]:segment["end"]] = True
        info["speech_seconds"] = int(mask.sum()) / sr
        if len(segments) > 1:
            flags.append("multiple_speech_regions")
        if start < int(0.05 * sr) or end > n - int(0.05 * sr):
            flags.append("speech_at_source_boundary")
        speech_rms = rms(wav[mask])
        margin = round(options.margin_ms * sr / 1000)
        # Exclude the boundary margin from background estimation.
        bg = torch.cat((wav[:max(0, start - margin)], wav[min(n, end + margin):]))
        if bg.numel() >= int(0.1 * sr):
            contrast = 20 * np.log10(max(speech_rms, 1e-9) / max(rms(bg), 1e-9))
            info["contrast_db"] = float(contrast)
            if contrast < options.min_contrast_db:
                flags.append("low_energy_contrast")
        left, right = max(0, start - margin), min(n, end + margin)
        info.update(crop_start_seconds=left / sr, crop_end_seconds=right / sr)
        clip = wav[left:right]
        if clip.numel() > target:
            flags.append("speech_and_margin_too_long")
        if speech_rms < 1e-5:
            flags.append("very_quiet_speech")
        desired = options.target_dbfs - 20 * np.log10(max(speech_rms, 1e-9))
        if desired > options.max_gain_db:
            flags.append("gain_limit_reached")
        peak = float(clip.abs().max())
        gain_db = min(desired, options.max_gain_db,
                      options.peak_dbfs - 20 * np.log10(max(peak, 1e-9)))
        info["gain_db"] = float(gain_db)
        clip = clip * (10 ** (gain_db / 20))
    info["review_reasons"] = ";".join(flags)
    if flags and not keep_review:
        return None, info
    remaining = target - clip.numel()
    if remaining > 0:
        clip = F.pad(clip, (remaining // 2, remaining - remaining // 2))
    return clip, info


def preprocess_dataset(data_root, metadata_csv, output_root, options=None, detector=None):
    options = options or Options()
    root, out = Path(data_root).resolve(), Path(output_root).resolve()
    if out == root or out.is_relative_to(root) or root.is_relative_to(out):
        raise ValueError("Output must be separate from the source dataset")
    if out.exists():
        raise FileExistsError(f"Use a fresh output directory: {out}")
    metadata_csv = Path(metadata_csv)
    if not metadata_csv.is_absolute():
        metadata_csv = root / metadata_csv
    frame = pd.read_csv(metadata_csv, dtype=str, keep_default_na=False)
    if frame.empty or not {"path", "label"} <= set(frame.columns) or frame.path.duplicated().any():
        raise ValueError("Unique path and label columns are required")
    detector = detector if detector is not None else SileroDetector(options)
    out.mkdir(parents=True)
    (out / "preprocess_options.json").write_text(json.dumps(asdict(options), indent=2) + "\n")
    accepted, reports = [], []
    for row in frame.to_dict("records"):
        report = dict(row, source_path=Path(os.path.relpath(root / row["path"], out)).as_posix(), status="review", review_reasons="")
        try:
            source = audio_path(root, row["path"])
            audio, sr = sf.read(source, dtype="float32", always_2d=True)
            if not audio.size or not np.isfinite(audio).all():
                raise ValueError("Empty or nonfinite audio")
            wav = torch.from_numpy(audio.T.copy()).mean(0)
            if sr != options.sample_rate:
                wav = torchaudio.functional.resample(wav, sr, options.sample_rate)
            segments = detector(wav)
            clip, diagnostics = prepare_clip(wav, segments, row["label"], options)
            report.update(diagnostics, speech_segments=json.dumps(segments))
            if clip is not None:
                destination = out / row["path"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                sf.write(destination, clip.numpy(), options.sample_rate, subtype="PCM_16")
                accepted.append(row)
                report["status"] = "accepted"
        except (ValueError, RuntimeError, OSError) as exc:
            report.update(status="error", review_reasons="processing_error", error=str(exc))
        reports.append(report)
    report = pd.DataFrame(reports)
    pd.DataFrame(accepted, columns=frame.columns).to_csv(out / "metadata.csv", index=False)
    report.to_csv(out / "preprocess_report.csv", index=False)
    report[report.status != "accepted"].to_csv(out / "review.csv", index=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--metadata-csv", default="metadata.csv")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seconds", type=float, default=1.5)
    parser.add_argument("--margin-ms", type=int, default=200)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    report = preprocess_dataset(args.data_root, args.metadata_csv, args.output_root,
                                Options(seconds=args.seconds, margin_ms=args.margin_ms,
                                        threshold=args.threshold))
    print(report.status.value_counts().to_string())
    print(f"Review: {Path(args.output_root) / 'review.csv'}")


if __name__ == "__main__":
    main()
