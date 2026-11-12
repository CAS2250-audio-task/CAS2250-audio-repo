"""Pinned MS-SNSD speech-noise files, excluding its renamed DEMAND recordings."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
import soundfile as sf
import torch
import torchaudio

from src.engine import file_hash, write_json

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "runs/conformer-noise-focus-v1"
REPORT = ROOT / "reports/conformer-noise-focus-v1"
REVISION = "fe61c4ba0d9ac8dd7e23d719cc79f8947e1dc742"
PREFIXES = ("Babble_", "NeighborSpeaking_", "AirportAnnouncement")


def prepare(entry):
    relative = entry["path"]
    name = Path(relative).name
    family = "babble" if name.startswith("Babble_") else "neighbor_speech" if name.startswith("NeighborSpeaking_") else "announcements"
    original = STUDY / "sources/originals" / name
    original.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://raw.githubusercontent.com/microsoft/MS-SNSD/{REVISION}/{relative}"
    if not original.exists():
        data = urllib.request.urlopen(url, timeout=90).read()
        assert hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest() == entry["sha"]
        original.write_bytes(data)
    data = original.read_bytes()
    assert hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest() == entry["sha"]
    audio, rate = sf.read(original, dtype="float32", always_2d=True)
    wave = torch.from_numpy(audio.mean(1))
    if rate != 16000:
        wave = torchaudio.functional.resample(wave, rate, 16000)
    assert len(wave) >= 24000 and torch.isfinite(wave).all() and wave.square().mean() > 1e-8
    noise_id = "speech_"+original.stem.lower()
    target = STUDY / "sources" / (noise_id+".wav")
    sf.write(target, wave.numpy(), 16000, subtype="FLOAT")
    return dict(noise_id=noise_id, category="crowd", family=family, original_filename=name,
        source_repository="microsoft/MS-SNSD", repository_revision=REVISION, repository_path=relative,
        git_blob_sha1=entry["sha"], download_url=url, original_sha256=file_hash(original),
        original_sample_rate=rate, original_channels=audio.shape[1], prepared_path=str(target.relative_to(ROOT)),
        prepared_sha256=file_hash(target), pcm_sha256=hashlib.sha256(wave.numpy().tobytes()).hexdigest(),
        samples=len(wave), seconds=len(wave)/16000, usage="training speech-interference augmentation only",
        provenance="MS-SNSD train Babble/NeighborSpeaking/AirportAnnouncement subset; underlying Freesound IDs are not supplied",
        license="MS-SNSD README states Freesound noise subset is CC0; DEMAND-named subset excluded")


def main():
    torch.set_num_threads(2)
    STUDY.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    candidates = STUDY / "candidate_sources.json"
    if not candidates.exists():
        tree = json.load(urllib.request.urlopen(f"https://api.github.com/repos/microsoft/MS-SNSD/git/trees/{REVISION}?recursive=1"))
        files = [e for e in tree["tree"] if e["path"].startswith("noise_train/") and e["path"].endswith(".wav")
                 and Path(e["path"]).name.startswith(PREFIXES)]
        write_json(candidates, dict(repository="microsoft/MS-SNSD", revision=REVISION, files=files))
    meta = json.loads(candidates.read_text())
    assert meta["revision"] == REVISION and len(meta["files"]) == 37
    assert all(e["path"].startswith("noise_train/") and Path(e["path"]).name.startswith(PREFIXES) for e in meta["files"])
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        prepared = list(pool.map(prepare, meta["files"]))
    kept, duplicates, seen = [], [], {}
    for source in prepared:
        if source["pcm_sha256"] in seen:
            duplicates.append(dict(excluded=source["noise_id"], identical_to=seen[source["pcm_sha256"]]))
        else:
            seen[source["pcm_sha256"]] = source["noise_id"]
            kept.append(source)
    original = json.loads((ROOT / "reports/conformer-aug-v1/training_noise_sources.json").read_text())
    evaluation = json.loads((ROOT / "reports/noise-stress-v1/noise_sources.json").read_text())
    for key in ("original_sha256", "prepared_sha256", "original_filename"):
        assert not {s[key] for s in kept} & {s[key] for s in original+evaluation}
    write_json(REPORT / "added_speech_sources.json", kept)
    write_json(REPORT / "source_preparation_audit.json", dict(candidate_files=37, unique_pcm_files=len(kept),
        exact_pcm_duplicates_removed=duplicates, original_training_sources=6, evaluation_sources=8,
        shared_filename_original_hash_prepared_hash=0, test_audio_read=False,
        limitation="Different numbered files may share underlying recordings; unique PCM files are not claimed to be independent original sessions"))
    print("ADDED_SPEECH_SOURCES_READY", len(kept), "seconds", sum(s["seconds"] for s in kept), flush=True)


if __name__ == "__main__":
    main()
