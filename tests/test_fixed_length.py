"""Check speech preservation and the human-selection gate for the full dataset."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import soundfile as sf

from src.fixed_length import (SAMPLES, finalize, prepare, read_pcm,
                              reduced_margin_start, verify_dataset, window_starts)


class FixedLengthTests(unittest.TestCase):
    def test_reduced_margin_preserves_the_entire_vad_span(self):
        # The tightest real case has 1.408 s of detected speech and only 92 ms left.
        first, last, length = 3200, 25728, 28928
        start = reduced_margin_start(length, first, last)
        self.assertEqual(first - start, 736)
        self.assertEqual(start + SAMPLES - last, 736)
        with self.assertRaises(ValueError):
            reduced_margin_start(32000, 1000, 25001)

    def test_windows_cover_both_separated_regions_with_full_length(self):
        length = 40192
        regions = [dict(start=3200, end=5248), dict(start=27264, end=36992)]
        starts = window_starts(length, regions)
        self.assertEqual(starts[0], 0)
        self.assertEqual(starts[-1] + SAMPLES, length)
        for segment in regions:
            self.assertTrue(any(start <= segment['start'] < segment['end'] <= start + SAMPLES
                                for start in starts))
        self.assertTrue(all(0 <= start <= length - SAMPLES for start in starts))
        self.assertEqual(len(starts), len(set(starts)))

    def fixture(self, directory):
        base = Path(directory)
        raw, accepted, pending, out, review, listening = [base / name for name in
                                                        ('raw', 'accepted', 'pending', 'fixed', 'windows', 'listening')]
        for p in (raw, accepted, pending):
            p.mkdir()
        rows, reports, candidates, signals = [], [], [], {}
        for name, label, split, status, length, segments in [
            ('a', '출발', 'train', 'accepted', SAMPLES, []),
            ('b', 'silence', 'val', 'review', SAMPLES, []),
            ('c', '전진', 'train', 'review', 24320, [dict(start=3200, end=21120)]),
            ('d', '정지', 'test', 'review', 32000,
             [dict(start=3200, end=10000), dict(start=27000, end=30000)]),
        ]:
            path = f'audio/{label}/{name}.wav'
            row = dict(path=path, label=label, split=split, group=name, speaker_id=name,
                       room_session_id='', source_seconds=length / 16000)
            rows.append(row)
            reports.append(dict(row, status=status, review_reasons='test_flag' if status == 'review' else '',
                                speech_segments=json.dumps(segments)))
            source = (accepted if status == 'accepted' else pending) / path
            source.parent.mkdir(parents=True, exist_ok=True)
            signal = (np.arange(length) % 1000).astype(np.int16)
            signals[path] = signal
            sf.write(source, signal, 16000, subtype='PCM_16')
            if status == 'review':
                candidates.append(dict(row, candidate_crop_start_seconds=0.0))
        pd.DataFrame(rows).to_csv(raw / 'metadata.csv', index=False)
        pd.DataFrame(reports).to_csv(accepted / 'preprocess_report.csv', index=False)
        pd.DataFrame(candidates).to_csv(pending / 'candidates.csv', index=False)
        prepare(raw, accepted, pending, out, review, listening)
        return out, review, listening, signals

    def test_finalize_requires_choices_and_preserves_identity_split_and_pcm(self):
        with tempfile.TemporaryDirectory() as directory:
            out, review, listening, signals = self.fixture(directory)
            self.assertFalse((out / 'metadata.csv').exists())
            with self.assertRaises(ValueError):
                verify_dataset(out)
            with self.assertRaises(ValueError):
                finalize(out, review / 'selections.csv')
            self.assertEqual(len(list((out / 'audio').glob('*/*.wav'))), 3)
            np.testing.assert_array_equal(read_pcm(out / 'audio/silence/b.wav'), signals['audio/silence/b.wav'])
            np.testing.assert_array_equal(read_pcm(out / 'audio/전진/c.wav'), signals['audio/전진/c.wav'][160:24160])
            choices = pd.read_csv(review / 'selections.csv')
            choices['window_number'] = 1
            choices.to_csv(review / 'selections.csv', index=False)
            windows = pd.read_csv(review / 'windows.csv')
            expected = read_pcm(review / windows.iloc[0]['path'])
            result = finalize(out, review / 'selections.csv')
            self.assertEqual(result['files'], 4)
            self.assertEqual(result['split_counts'], {'train': 2, 'val': 1, 'test': 1})
            np.testing.assert_array_equal(read_pcm(out / 'audio/정지/d.wav'), expected)
            source_meta = pd.read_csv(out / 'metadata_draft.csv').fillna('')
            final_meta = pd.read_csv(out / 'metadata.csv').fillna('')
            pd.testing.assert_frame_equal(source_meta, final_meta)
            # The page contains embedded audio and no dependency on a network service.
            self.assertIn('data:audio/wav;base64,', (listening / 'listen.html').read_text())
            with self.assertRaises(FileExistsError):
                finalize(out, review / 'selections.csv')

    def test_modified_candidate_is_rejected_before_any_pending_audio_is_written(self):
        with tempfile.TemporaryDirectory() as directory:
            out, review, _, _ = self.fixture(directory)
            choices = pd.read_csv(review / 'selections.csv')
            choices['window_number'] = 1
            choices.to_csv(review / 'selections.csv', index=False)
            selected = pd.read_csv(review / 'windows.csv').iloc[0]
            source = review / selected['path']
            signal = read_pcm(source)
            signal[0] += 1
            sf.write(source, signal, 16000, subtype='PCM_16')
            with self.assertRaisesRegex(ValueError, 'modified'):
                finalize(out, review / 'selections.csv')
            self.assertFalse((out / 'metadata.csv').exists())
            self.assertFalse((out / choices.iloc[0]['path']).exists())


if __name__ == '__main__':
    unittest.main()
