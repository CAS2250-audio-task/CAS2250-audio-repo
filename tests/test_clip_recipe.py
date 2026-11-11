"""A reviewed crop recipe must preserve identity, silence level and exact PCM."""
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import soundfile as sf

from preprocessing.clip_audio import replay_dataset


class RecipeTests(unittest.TestCase):
    def fixture(self, folder):
        base = Path(folder)
        raw = base / 'raw'
        raw.mkdir()
        metadata, recipe = [], []
        for name, label, gain, multiplier in [('word', '전진', 20 * np.log10(2), 2),
                                               ('background', 'silence', 0, 1)]:
            relative = f'audio/{label}/{name}.wav'
            source = raw / relative
            source.parent.mkdir(parents=True)
            original = ((np.arange(32000) % 1000) - 500).astype(np.int16)
            sf.write(source, original, 16000, subtype='PCM_16')
            expected = np.pad(original[8000:24000] * multiplier, (4000, 4000))
            reference = base / f'{name}_expected.wav'
            sf.write(reference, expected, 16000, subtype='PCM_16')
            row = dict(path=relative, label=label, group=name, split='train',
                       speaker_id=name if label != 'silence' else '',
                       room_session_id=name if label == 'silence' else '')
            metadata.append(row)
            recipe.append(dict(row, crop_start_sample=8000, crop_end_sample=24000,
                               pad_left_samples=4000, pad_right_samples=4000, gain_db=gain,
                               raw_wav_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                               output_sha256=hashlib.sha256(reference.read_bytes()).hexdigest()))
        pd.DataFrame(metadata).to_csv(raw / 'metadata.csv', index=False)
        path = base / 'recipe.csv'
        pd.DataFrame(recipe).to_csv(path, index=False)
        return raw, path, base / 'output'

    def test_replay_exact_samples_metadata_and_originals(self):
        with tempfile.TemporaryDirectory() as folder:
            raw, recipe, out = self.fixture(folder)
            before = {p: p.read_bytes() for p in raw.rglob('*.wav')}
            result = replay_dataset(raw, recipe, out)
            self.assertTrue(result['all_output_hashes_match'])
            for p, content in before.items():
                self.assertEqual(p.read_bytes(), content)
            for p in out.glob('audio/*/*.wav'):
                signal, rate = sf.read(p, dtype='int16')
                self.assertEqual((len(signal), rate), (24000, 16000))
                np.testing.assert_array_equal(signal[:4000], np.zeros(4000))
                np.testing.assert_array_equal(signal[-4000:], np.zeros(4000))
            pd.testing.assert_frame_equal(pd.read_csv(raw / 'metadata.csv'), pd.read_csv(out / 'metadata.csv'))

    def test_silence_gain_and_label_changes_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as folder:
            raw, recipe, out = self.fixture(folder)
            frame = pd.read_csv(recipe, keep_default_na=False)
            frame.loc[frame.label == 'silence', 'gain_db'] = 1
            frame.to_csv(recipe, index=False)
            with self.assertRaisesRegex(ValueError, 'silence'):
                replay_dataset(raw, recipe, out)
            self.assertFalse(out.exists())
            frame.loc[frame.label == 'silence', 'gain_db'] = 0
            frame.loc[frame.label == '전진', 'label'] = '후진'
            frame.to_csv(recipe, index=False)
            with self.assertRaisesRegex(ValueError, 'label'):
                replay_dataset(raw, recipe, out)
            self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()
