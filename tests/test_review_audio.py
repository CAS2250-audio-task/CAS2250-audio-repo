"""A listening export must preserve sample values and its file/time mapping."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import soundfile as sf

from src.review_audio import write_reel


class ListeningTests(unittest.TestCase):
    def test_reel_preserves_samples_and_records_exact_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'audio').mkdir()
            first = np.array([100, -200, 300, 0], dtype=np.int16)
            second = np.array([-100, 200], dtype=np.int16)
            sf.write(root / 'audio/a.wav', first, 16000, subtype='PCM_16')
            sf.write(root / 'audio/b.wav', second, 16000, subtype='PCM_16')
            frame = pd.DataFrame([dict(path='audio/a.wav', label='시작', status='accepted'),
                                  dict(path='audio/b.wav', label='시작', status='accepted')])
            result = write_reel(frame, root, root, 'check', gap_seconds=0.001)
            audio, rate = sf.read(root / 'check_all.wav', dtype='int16')
            np.testing.assert_array_equal(audio, np.concatenate([first, np.zeros(16, dtype=np.int16), second]))
            self.assertEqual(rate, 16000)
            index = pd.read_csv(root / 'check_index.csv')
            self.assertEqual(index.path.tolist(), ['audio/a.wav', 'audio/b.wav'])
            self.assertAlmostEqual(index.iloc[1].start_seconds, 20/16000)
            self.assertAlmostEqual(result['seconds'], 22/16000)
            with self.assertRaises(FileExistsError):
                write_reel(frame, root, root, 'check')


if __name__ == '__main__':
    unittest.main()
