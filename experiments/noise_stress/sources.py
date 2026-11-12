"""Download original long recordings, verifying publisher checksums."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.parse
import urllib.request
import zipfile

import numpy as np
import soundfile as sf
import torch
import torchaudio

from src.engine import file_hash, write_json

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "runs/noise-stress-v1"
REPORT = ROOT / "reports/noise-stress-v1"
SOURCES = [
    ("traffic", "road", "1227121", "STRAFFIC_16k.zip", "2efa87262f272bbf9ba578088e81939c"),
    ("car", "road", "1227121", "TCAR_16k.zip", "4d930012796bd298932245a26189f973"),
    ("cafeteria", "crowd", "1227121", "PCAFETER_16k.zip", "99927d148128254141a9417d051510bb"),
    ("restaurant", "crowd", "1227121", "PRESTO_16k.zip", "b98d2e6854eeebb397f29a8ad7457092"),
    ("station", "crowd", "1227121", "PSTATION_16k.zip", "d7448009f6c2aeb6ba570375df1750a3"),
    ("washing", "machine", "1227121", "DWASHING_16k.zip", "7e5ee9437ce9409c5f9a779b6212a240"),
    ("compressor", "machine", "7493820", "Air_Compressor_with_initial_silence_2022_09_10_14_57_53_pyaudio.wav", "b0a416eeac0a972ec2dc20d48f44eca9"),
    ("fan", "machine", "7493820", "Exhaust Fan 2022_07_21_11_43_40_pyaudio.wav", "2fda509599250495165096249ab4dc8f"),
]


def md5(path):
    digest = hashlib.md5()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url, target, expected):
    if target.exists() and md5(target) == expected:
        return
    partial = target.with_suffix(target.suffix + ".partial")
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "audio-research/1.0"})
            with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
            if md5(partial) != expected:
                raise ValueError("Publisher MD5 mismatch: " + target.name)
            partial.replace(target)
            return
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))


def prepare(entry):
    noise_id, category, record, filename, checksum = entry
    directory = STUDY / "sources"
    directory.mkdir(parents=True, exist_ok=True)
    url = f"https://zenodo.org/records/{record}/files/{urllib.parse.quote(filename)}?download=1"
    original = directory / filename
    download(url, original, checksum)
    channel = "mono"
    if filename.endswith(".zip"):
        with zipfile.ZipFile(original) as archive:
            members = [name for name in archive.namelist() if name.lower().endswith("ch01.wav")]
            if len(members) != 1:
                raise ValueError(f"Expected exactly one original microphone ch01: {members}")
            channel = members[0]
            with archive.open(channel) as stream:
                audio, rate = sf.read(stream, dtype="float32", always_2d=True)
    else:
        audio, rate = sf.read(original, dtype="float32", always_2d=True)
    original_shape = audio.shape
    wave = torch.from_numpy(audio.mean(axis=1))
    if rate != 16000:
        wave = torchaudio.functional.resample(wave, rate, 16000)
    if not torch.isfinite(wave).all() or wave.square().mean() <= 0:
        raise ValueError("Invalid recording: " + filename)
    output = directory / (noise_id + "_mono16k.wav")
    sf.write(output, wave.numpy(), 16000, subtype="FLOAT")
    result = dict(noise_id=noise_id, category=category, record_url=f"https://zenodo.org/records/{record}",
                  download_url=url, original_filename=filename, publisher_md5=checksum,
                  original_sha256=file_hash(original), channel=channel,
                  original_sample_rate=rate, original_channels=original_shape[1],
                  original_samples=original_shape[0], sample_rate=16000, samples=len(wave),
                  seconds=len(wave)/16000, prepared_path=str(output.relative_to(ROOT)),
                  prepared_sha256=file_hash(output), usage="development stress evaluation only; reserve entire original recording",
                  api_license="CC BY 4.0", description_license="CC BY-SA 3.0" if record=="1227121" else "not specified in description")
    print("SOURCE_READY", noise_id, result["seconds"], flush=True)
    return result


def main():
    torch.set_num_threads(2)
    REPORT.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(prepare, SOURCES))
    write_json(REPORT / "noise_sources.json", results)
    print("ALL_SOURCES_READY", len(results), flush=True)


if __name__ == "__main__":
    main()
