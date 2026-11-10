"""Export pending processing candidates and separate continuous listening WAVs."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio

from .dataset import audio_path
from .preprocess import Options, prepare_clip


def timecode(seconds):
    milliseconds = round(seconds * 1000)
    hours, milliseconds = divmod(milliseconds, 3600000)
    minutes, milliseconds = divmod(milliseconds, 60000)
    seconds, milliseconds = divmod(milliseconds, 1000)
    return f'{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}'


def export_pending(data_root, processed_root, candidate_root):
    raw, processed, out = map(lambda p: Path(p).resolve(), (data_root, processed_root, candidate_root))
    if out.exists():
        raise FileExistsError(f'Use a fresh candidate directory: {out}')
    if any(out == root or out.is_relative_to(root) for root in (raw, processed)):
        raise ValueError('Candidates must be outside the source and accepted folders')
    options = Options(**json.loads((processed / 'preprocess_options.json').read_text()))
    report = pd.read_csv(processed / 'preprocess_report.csv', keep_default_na=False)
    if report.path.duplicated().any() or not set(report.status) <= {'accepted', 'review'}:
        raise ValueError('Expected unique paths with accepted/review statuses')
    pending = report[report.status == 'review']
    out.mkdir(parents=True)
    rows = []
    for row in pending.to_dict('records'):
        source = audio_path(raw, row['path'])
        signal, rate = sf.read(source, dtype='float32', always_2d=True)
        if not signal.size or not np.isfinite(signal).all():
            raise ValueError(f'Invalid source audio: {source}')
        wav = torch.from_numpy(signal.T.copy()).mean(0)
        if rate != options.sample_rate:
            wav = torchaudio.functional.resample(wav, rate, options.sample_rate)
        if abs(wav.numel() / options.sample_rate - float(row['source_seconds'])) > 1 / options.sample_rate:
            raise ValueError(f'Source duration changed since VAD analysis: {source}')
        segments = json.loads(row['speech_segments'])
        clip, info = prepare_clip(wav, segments, row['label'], options, keep_review=True)
        if clip is None:
            # No speech / no usable timestamps: preserve context; do not invent a crop or boost noise.
            clip = wav
            info.update(gain_db=0.0, crop_start_seconds=0.0,
                        crop_end_seconds=wav.numel() / options.sample_rate)
            kind = 'untrimmed_fallback'
        else:
            kind = 'vad_candidate'
        destination = out / row['path']
        destination.parent.mkdir(parents=True, exist_ok=True)
        sf.write(destination, clip.numpy(), options.sample_rate, subtype='PCM_16')
        rows.append(dict(row, candidate_kind=kind, candidate_seconds=len(clip) / options.sample_rate,
                         candidate_gain_db=info['gain_db'],
                         candidate_crop_start_seconds=info['crop_start_seconds'],
                         candidate_crop_end_seconds=info['crop_end_seconds'],
                         candidate_review_reasons=info['review_reasons']))
    candidates = pd.DataFrame(rows)
    candidates.to_csv(out / 'candidates.csv', index=False)
    (out / 'preprocess_options.json').write_text(json.dumps(options.__dict__, indent=2) + '\n')
    return report, candidates, options


def write_reel(frame, audio_root, output_root, name, sample_rate=16000, gap_seconds=0.3):
    """Concatenate PCM samples without volume changes; keep a per-file time index."""
    root, out = Path(audio_root).resolve(), Path(output_root).resolve()
    if not np.isfinite(gap_seconds) or gap_seconds < 0:
        raise ValueError('Gap must be finite and nonnegative')
    gap = np.zeros(round(sample_rate * gap_seconds), dtype=np.int16)
    rows, cursor = [], 0
    path = out / f'{name}_all.wav'
    ordered = frame.sort_values(['label', 'path']).to_dict('records')
    # Exclusive creation guards completed listening and annotation files.
    with path.open('xb') as stream:
        with sf.SoundFile(stream, mode='w', samplerate=sample_rate, channels=1, subtype='PCM_16', format='WAV') as reel:
            for number, row in enumerate(ordered, 1):
                source = audio_path(root, row['path'])
                signal, rate = sf.read(source, dtype='int16')
                if rate != sample_rate or signal.ndim != 1 or not signal.size:
                    raise ValueError(f'Expected nonempty mono {sample_rate}-Hz audio: {source}')
                start = cursor / sample_rate
                reel.write(signal)
                cursor += len(signal)
                end = cursor / sample_rate
                rows.append(dict(number=number, start_time=timecode(start), end_time=timecode(end),
                                 start_seconds=start, end_seconds=end, path=row['path'], label=row['label'],
                                 split=row.get('split', ''), status=row['status'],
                                 review_reasons=row.get('review_reasons', ''),
                                 audio_path=Path(os.path.relpath(source, out)).as_posix(),
                                 decision='', notes=''))
                if number < len(ordered):
                    reel.write(gap)
                    cursor += len(gap)
    index = pd.DataFrame(rows)
    index.to_csv(out / f'{name}_index.csv', index=False, encoding='utf-8-sig')
    return dict(files=len(rows), seconds=cursor / sample_rate, duration=timecode(cursor / sample_rate))


def build_listening(data_root, processed_root, candidate_root, output_root, gap_seconds=0.3):
    out = Path(output_root).resolve()
    if out.exists():
        raise FileExistsError(f'Use a fresh listening directory: {out}')
    report, candidates, options = export_pending(data_root, processed_root, candidate_root)
    out.mkdir(parents=True)
    accepted = write_reel(report[report.status == 'accepted'], processed_root, out, 'accepted',
                          options.sample_rate, gap_seconds)
    pending = write_reel(candidates, candidate_root, out, 'pending', options.sample_rate, gap_seconds)
    summary = dict(accepted=accepted, pending=pending, gap_seconds=gap_seconds,
                   total_files=accepted['files'] + pending['files'],
                   long_pending=int((candidates.candidate_seconds > options.seconds).sum()),
                   note='Listening only. Pending files remain excluded from training. Silence gain is zero.')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--processed-root', required=True)
    parser.add_argument('--candidate-root', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--gap-seconds', type=float, default=0.3)
    args = parser.parse_args()
    print(json.dumps(build_listening(args.data_root, args.processed_root, args.candidate_root,
                                    args.output_root, args.gap_seconds), indent=2))


if __name__ == '__main__':
    main()
