"""Guard reviewed speech, raw continuity and input integrity during expansion."""
import numpy as np
import pytest
import soundfile as sf

from preprocessing.real_context import (
    continuous_window, digest, prepare_candidates, read_rows, write_rows,
)


def fixture(tmp_path, short_frames=18000):
    raw, reviewed = tmp_path / "raw", tmp_path / "reviewed"
    metadata, recipe = [], []
    for name, frames, start, end, gain, label in (
        ("normal", 32000, 8000, 24000, 0, "전진"),
        ("left_edge", 32000, 0, 16000, 0, "전진"),
        ("right_edge", 32000, 16000, 32000, 0, "전진"),
        ("short", short_frames, 1000, 17000, 0, "전진"),
        ("loud_context", 32000, 8000, 24000, 6, "전진"),
        ("silence", 32000, 4000, 28000, 0, "silence"),
        ("selected", 32000, 6000, 30000, 0, "전진"),
    ):
        relative = f"audio/{label}/{name}.wav"
        signal = np.full(frames, 300, dtype=np.int16)
        if name == "loud_context":
            signal[4000:8000] = 24000
        if name == "short":
            signal[:1000] = 100
            signal[17000:] = 200
        source = raw / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        sf.write(source, signal, 16000, subtype="PCM_16")
        left = (24000 - (end - start)) // 2
        right = 24000 - (end - start) - left
        decoded, _ = sf.read(source, dtype="float32")
        previous = np.pad(decoded[start:end] * (10 ** (gain / 20)), (left, right))
        reference = reviewed / relative
        reference.parent.mkdir(parents=True, exist_ok=True)
        sf.write(reference, previous, 16000, subtype="PCM_16")
        row = dict(path=relative, label=label, group="speaker:one", split="train",
                   speaker_id="one" if label != "silence" else "",
                   room_session_id="session-one")
        metadata.append(row)
        recipe.append(dict(row, source_frames=frames, crop_start_sample=start,
                           crop_end_sample=end, pad_left_samples=left,
                           pad_right_samples=right, gain_db=gain, method=name,
                           raw_wav_sha256=digest(source), output_sha256=digest(reference)))
    write_rows(raw / "metadata.csv", metadata)
    write_rows(reviewed / "metadata.csv", metadata)
    write_rows(reviewed / "preprocessing_recipe.csv", recipe)
    return raw, reviewed, tmp_path / "output"


def test_boundary_shifts_preserve_the_whole_reviewed_interval():
    assert continuous_window(32000, 0, 16000, 4000) == (0, 24000)
    assert continuous_window(32000, 16000, 32000, 4000) == (8000, 32000)
    assert continuous_window(32000, 6000, 30000, 0) == (6000, 30000)
    assert continuous_window(18000, 1000, 17000, 4000) is None
    with pytest.raises(ValueError):
        continuous_window(32000, 0, 25000, 0)


def test_candidates_are_contiguous_raw_samples_and_short_sources_stay_pending(tmp_path):
    raw, reviewed, output = fixture(tmp_path)
    before = {p: digest(p) for root in (raw, reviewed) for p in root.rglob("*") if p.is_file()}
    result = prepare_candidates(raw, reviewed, output)
    assert (result["continuous_candidates"], result["pending_short_originals"]) == (6, 1)
    assert result["status"] == "pending_short_source_policy"
    assert not (output / "metadata.csv").exists()
    assert not (output / "audio/전진/short.wav").exists()
    assert len(read_rows(output / "candidate_metadata.csv")) == 6
    for row in read_rows(output / "context_recipe.csv"):
        if row["status"] != "continuous_candidate":
            continue
        signal, sr = sf.read(output / row["path"], dtype="int16")
        assert sr == 16000 and signal.shape == (24000,)
        original, _ = sf.read(raw / row["path"], dtype="float32")
        expected = original[int(row["crop_start_sample"]):int(row["crop_end_sample"])]
        expected = np.floor(expected * 10 ** (float(row["gain_db"]) / 20) * 32768).astype(np.int16)
        np.testing.assert_array_equal(signal, expected)
        assert np.all(signal != 0)  # Background came from raw samples, not zero padding.
    assert all(digest(p) == expected for p, expected in before.items())
    for name, label in (("selected", "전진"), ("silence", "silence")):
        relative = f"audio/{label}/{name}.wav"
        assert digest(output / relative) == digest(reviewed / relative)


def test_new_loud_background_reduces_whole_clip_gain_without_hard_clipping(tmp_path):
    raw, reviewed, output = fixture(tmp_path)
    result = prepare_candidates(raw, reviewed, output)
    assert result["gain_reduced_files"] == ["audio/전진/loud_context.wav"]
    row = next(r for r in read_rows(output / "context_recipe.csv") if r["peak_gain_reduced"] == "True")
    assert row["review_required"] == "True"
    assert float(row["gain_db"]) < float(row["previous_gain_db"])
    signal, _ = sf.read(output / row["path"], dtype="float32")
    assert float(np.abs(signal).max()) <= 10 ** (-1 / 20) + 1 / 32768


def test_changed_original_and_changed_labels_fail_before_output_is_created(tmp_path):
    raw, reviewed, output = fixture(tmp_path)
    source = raw / "audio/전진/normal.wav"
    with source.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        prepare_candidates(raw, reviewed, output)
    assert not output.exists()
    rows = read_rows(reviewed / "preprocessing_recipe.csv")
    rows[0]["label"] = "후진"
    write_rows(reviewed / "preprocessing_recipe.csv", rows)
    with pytest.raises(ValueError, match="identity label"):
        prepare_candidates(raw, reviewed, output)
    assert not output.exists()


def test_existing_or_overlapping_outputs_are_rejected(tmp_path):
    raw, reviewed, output = fixture(tmp_path)
    output.mkdir()
    with pytest.raises(FileExistsError):
        prepare_candidates(raw, reviewed, output)
    with pytest.raises(ValueError, match="separate"):
        prepare_candidates(raw, reviewed, reviewed / "replacement")


@pytest.mark.parametrize("frames", [18000, 18001])
def test_zero_policy_preserves_entire_short_raw_and_all_metadata(tmp_path, frames):
    raw, reviewed, output = fixture(tmp_path, short_frames=frames)
    result = prepare_candidates(raw, reviewed, output, short_policy="zero")
    assert result["files"] == 7
    assert result["continuous_candidates"] == 6
    assert result["zero_padded_originals"] == 1
    assert result["pending_short_originals"] == 0
    assert result["short_source_policy"] == "zero"
    assert (output / "metadata.csv").read_bytes() == (reviewed / "metadata.csv").read_bytes()
    row = next(r for r in read_rows(output / "preprocessing_recipe.csv")
               if r["status"] == "zero_padded_original")
    assert (int(row["crop_start_sample"]), int(row["crop_end_sample"])) == (0, frames)
    left, right = int(row["pad_left_samples"]), int(row["pad_right_samples"])
    assert left + right == 24000 - frames
    assert right - left in (0, 1)
    original, _ = sf.read(raw / row["path"], dtype="int16")
    signal, _ = sf.read(output / row["path"], dtype="int16")
    np.testing.assert_array_equal(signal[left:left + frames], original)
    assert not np.any(signal[:left])
    assert not np.any(signal[left + frames:])
    assert len(read_rows(output / "pending_short_sources.csv")) == 0
    assert digest(output / "metadata.csv") == result["metadata_sha256"]
    assert digest(output / "preprocessing_recipe.csv") == result["recipe_sha256"]


def test_complete_recipe_replays_bit_identical_audio_with_existing_replayer(tmp_path):
    from preprocessing.clip_audio import replay_dataset

    raw, reviewed, output = fixture(tmp_path)
    prepare_candidates(raw, reviewed, output, short_policy="zero")
    result = replay_dataset(raw, output / "preprocessing_recipe.csv", tmp_path / "replayed")
    assert result["files"] == 7
    assert result["all_output_hashes_match"]
