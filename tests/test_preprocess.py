"""Preprocessing checks independent of neural VAD predictions."""
import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from src.preprocess import Options, prepare_clip, preprocess_dataset


class PreprocessTests(unittest.TestCase):
    def setUp(self):
        self.options = Options()
        self.wav = torch.zeros(48000)
        self.wav[16000:24000] = torch.sin(torch.arange(8000) * 0.2) * 0.1
        self.segments = [{"start": 16000, "end": 24000}]

    def test_speech_is_preserved_padded_and_normalized(self):
        clip, info = prepare_clip(self.wav, self.segments, "전진", self.options)
        self.assertEqual(clip.numel(), 24000)
        self.assertEqual(info['review_reasons'], '')
        self.assertEqual(int(torch.count_nonzero(clip)), int(torch.count_nonzero(self.wav)))
        self.assertAlmostEqual(float(clip[8000:16000].square().mean().sqrt()), 0.1, places=4)
        self.assertLessEqual(float(clip.abs().max()), 10 ** (-1 / 20))

    def test_no_speech_is_reviewed(self):
        clip, info = prepare_clip(self.wav, [], "전진", self.options)
        self.assertIsNone(clip)
        self.assertIn('no_speech', info['review_reasons'])

    def test_long_speech_is_not_cropped(self):
        clip, info = prepare_clip(self.wav, [{"start": 8000, "end": 40000}], "전진", self.options)
        self.assertIsNone(clip)
        self.assertIn('too_long', info['review_reasons'])

    def test_multiple_and_boundary_speech_are_reviewed(self):
        clip, info = prepare_clip(self.wav, [{"start": 0, "end": 4000}, *self.segments], "전진", self.options)
        self.assertIsNone(clip)
        self.assertIn('multiple_speech_regions', info['review_reasons'])
        self.assertIn('speech_at_source_boundary', info['review_reasons'])

    def test_low_contrast_is_reviewed(self):
        clip, info = prepare_clip(self.wav + 0.08, self.segments, '전진', self.options)
        self.assertIsNone(clip)
        self.assertIn('low_energy_contrast', info['review_reasons'])

    def test_silence_is_preserved_without_amplification(self):
        clip, info = prepare_clip(torch.ones(48000) * 0.01, [], 'silence', self.options)
        self.assertEqual(clip.numel(), 24000)
        self.assertAlmostEqual(float(clip.mean()), 0.01, places=6)
        self.assertEqual(info['gain_db'], 0)
        clip, info = prepare_clip(self.wav, self.segments, 'silence', self.options)
        self.assertIsNone(clip)
        self.assertIn('speech_in_silence', info['review_reasons'])

    def test_review_candidate_keeps_long_speech_and_review_flag(self):
        segments = [{"start": 8000, "end": 40000}]
        clip, info = prepare_clip(self.wav, segments, "전진", self.options, keep_review=True)
        self.assertGreater(clip.numel(), 24000)
        self.assertIn("speech_and_margin_too_long", info["review_reasons"])
        self.assertIsNone(prepare_clip(self.wav, segments, "전진", self.options)[0])

    def test_quiet_review_candidate_uses_capped_gain(self):
        clip, info = prepare_clip(self.wav * 0.01, self.segments, "전진", self.options, keep_review=True)
        self.assertIsNotNone(clip)
        self.assertEqual(clip.numel(), 24000)
        self.assertAlmostEqual(info["gain_db"], 12.0)
        self.assertIn("gain_limit_reached", info["review_reasons"])
        self.assertIsNone(prepare_clip(self.wav * 0.01, self.segments, "전진", self.options)[0])

    def test_batch_keeps_identities_and_reviews_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root, out = Path(directory) / 'raw', Path(directory) / 'processed'
            root.mkdir()
            sf.write(root / 'ok.wav', self.wav.numpy(), 16000)
            frame = pd.DataFrame([
                dict(path='ok.wav', label='전진', split='test', group='speaker-1'),
                dict(path='missing.wav', label='전진', split='train', group='speaker-2')])
            frame.to_csv(root / 'metadata.csv', index=False)
            report = preprocess_dataset(root, 'metadata.csv', out, self.options, lambda _: self.segments)
            self.assertEqual(report.status.tolist(), ['accepted', 'error'])
            accepted = pd.read_csv(out / 'metadata.csv')
            self.assertEqual(accepted.group.tolist(), ['speaker-1'])
            self.assertEqual(accepted.split.tolist(), ['test'])
            self.assertEqual(len(pd.read_csv(out / 'review.csv')), 1)
            self.assertEqual(sf.info(out / 'ok.wav').frames, 24000)
            with self.assertRaises(FileExistsError):
                preprocess_dataset(root, 'metadata.csv', out, self.options, lambda _: [])
            self.assertEqual(sf.info(root / 'ok.wav').frames, 48000)


if __name__ == '__main__':
    unittest.main()
