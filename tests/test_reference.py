"""Regression checks for label, split and audio invariants; no network required."""
from dataclasses import replace
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from src.config import load_config
from src.dataset import AudioDataset, validate_metadata
from src.engine import metrics

ROOT = Path(__file__).resolve().parents[1]


class MetadataTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(ROOT / "configs/task2_dummy.json")
        self.frame = pd.read_csv(Path(self.cfg.data_root) / "metadata.csv", dtype=str, keep_default_na=False)

    def test_provided_dummy_has_all_classes_without_group_leakage(self):
        self.assertEqual(len(validate_metadata(self.frame, self.cfg)), 33)

    def test_group_leakage_is_rejected(self):
        self.frame.loc[1, "group"] = self.frame.loc[0, "group"]
        with self.assertRaisesRegex(ValueError, "Group leakage"):
            validate_metadata(self.frame, self.cfg)

    def real_frame(self):
        frame = self.frame.copy()
        frame["speaker_id"] = np.where(frame.label != "silence", frame.group, "")
        frame["room_session_id"] = np.where(frame.label == "silence", frame.group, "")
        return frame

    def test_speaker_leakage_is_rejected_even_with_distinct_groups(self):
        frame = self.real_frame()
        frame.loc[1, "speaker_id"] = frame.loc[0, "speaker_id"]
        with self.assertRaisesRegex(ValueError, "speaker_id"):
            validate_metadata(frame, replace(self.cfg, dummy=False))

    def test_silence_session_leakage_is_rejected(self):
        frame = self.real_frame()
        silence = frame.index[frame.label == "silence"]
        frame.loc[silence[1], "room_session_id"] = frame.loc[silence[0], "room_session_id"]
        with self.assertRaisesRegex(ValueError, "room_session_id"):
            validate_metadata(frame, replace(self.cfg, dummy=False))

    def test_real_metadata_requires_identity_fields(self):
        with self.assertRaisesRegex(ValueError, "Missing metadata columns"):
            validate_metadata(self.frame, replace(self.cfg, dummy=False))

    def test_missing_class_in_one_split_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Every split"):
            validate_metadata(self.frame.drop(index=0), self.cfg)

    def test_actual_keywords_are_not_assigned_to_dummy_recordings(self):
        actual = load_config(ROOT / "configs/task2.json", data_root=self.cfg.data_root)
        with self.assertRaisesRegex(ValueError, "Label mismatch"):
            validate_metadata(self.real_frame(), actual)

    def test_blank_split_is_rejected(self):
        self.frame.loc[0, "split"] = ""
        with self.assertRaisesRegex(ValueError, "Empty"):
            validate_metadata(self.frame, self.cfg)

    def test_duplicate_audio_is_rejected(self):
        self.frame.loc[1, "path"] = self.frame.loc[0, "path"]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_metadata(self.frame, self.cfg)

    def test_path_cannot_escape_dataset_root(self):
        self.frame.loc[0, "path"] = "../../outside.wav"
        with self.assertRaisesRegex(ValueError, "relative WAV"):
            validate_metadata(self.frame, self.cfg)

    def test_real_data_root_is_unset_until_collection(self):
        with self.assertRaisesRegex(ValueError, "--data-root"):
            load_config(ROOT / "configs/task2.json")


class AudioTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(ROOT / "configs/task2_dummy.json")
        self.dataset = AudioDataset(pd.DataFrame(), self.cfg, train=False)

    def test_short_stereo_audio_is_resampled_padded_and_finite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "short.wav"
            sf.write(path, np.zeros((4000, 2), dtype=np.float32), 8000)
            x = self.dataset.transform_file(path)
            self.assertEqual(tuple(x.shape), (1, 128, 151))
            self.assertTrue(torch.isfinite(x).all())

    def test_validation_cropping_has_no_random_augmentation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "long.wav"
            wav = np.sin(np.arange(64000) * 0.03).astype(np.float32) * 0.2
            sf.write(path, wav, 16000)
            random.seed(1)
            first = self.dataset.transform_file(path)
            random.seed(999)
            second = self.dataset.transform_file(path)
            self.assertTrue(torch.equal(first, second))

    def test_macro_f1_includes_all_configured_classes(self):
        # A slice containing one correctly predicted class must not report macro-F1=1 for 11 classes.
        result = metrics([0], [0], 11)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertAlmostEqual(result["macro_f1"], 1 / 11)


if __name__ == "__main__":
    unittest.main()
