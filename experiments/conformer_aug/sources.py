"""Training noises reserved by original recording, disjoint from stress sources."""
import concurrent.futures
import json
from pathlib import Path
import urllib.parse
import zipfile

import numpy as np
import soundfile as sf
import torch
import torchaudio

from src.engine import file_hash, write_json
from experiments.noise_stress.sources import download

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/"runs/conformer-aug-v1"
REPORT=ROOT/"reports/conformer-aug-v1"
SOURCES=(
    ("train_traffic","road","7493820","Traffic_2022_12_26_19_20_15_pyaudio.wav","3f651be707438398b47f6d2b94702608"),
    ("train_bus","road","1227121","TBUS_16k.zip","706b11b0d8504f9f3b3f3211e91b3863"),
    ("train_square","crowd","1227121","SPSQUARE_16k.zip","205d0e7b8fe74504a2f8d252fc414b9e"),
    ("train_canteen","crowd","7493820","Canteen Noise 2022_08_18_12_27_56_pyaudio.wav","f89e0c5752c1718379d3b865e5a9ad5f"),
    ("train_mixer","machine","7493820","Concrete Mixture_2022_07_29_14_49_01_pyaudio.wav","38e5f6d0cae31bfc5a7f19c14d27e193"),
    ("train_pump","machine","7493820","Monoblock Pump_2022_12_29_08_28_28_pyaudio.wav","55bde93eb75696f817b850a4b76c4851"),
)


def prepare(entry):
    noise_id,category,record,filename,checksum=entry
    directory=STUDY/"noise_sources"
    directory.mkdir(parents=True,exist_ok=True)
    original=directory/filename
    url=f"https://zenodo.org/records/{record}/files/{urllib.parse.quote(filename)}?download=1"
    download(url,original,checksum)
    channel="mono"
    if filename.endswith(".zip"):
        with zipfile.ZipFile(original) as archive:
            members=[name for name in archive.namelist() if name.lower().endswith("ch01.wav")]
            assert len(members)==1
            channel=members[0]
            with archive.open(channel) as stream:
                audio,rate=sf.read(stream,dtype="float32",always_2d=True)
    else:
        audio,rate=sf.read(original,dtype="float32",always_2d=True)
    wave=torch.from_numpy(audio.mean(axis=1))
    if rate!=16000:wave=torchaudio.functional.resample(wave,rate,16000)
    assert torch.isfinite(wave).all() and wave.square().mean()>0
    target=directory/(noise_id+".wav")
    sf.write(target,wave.numpy(),16000,subtype="FLOAT")
    result=dict(noise_id=noise_id,category=category,original_filename=filename,
                record_url=f"https://zenodo.org/records/{record}",download_url=url,
                publisher_md5=checksum,original_sha256=file_hash(original),channel=channel,
                original_sample_rate=rate,original_channels=audio.shape[1],
                prepared_path=str(target.relative_to(ROOT)),prepared_sha256=file_hash(target),
                samples=len(wave),seconds=len(wave)/16000,usage="training augmentation only",
                api_license="CC BY 4.0",description_license="CC BY-SA 3.0" if record=="1227121" else "not specified")
    print("TRAIN_NOISE_READY",noise_id,result["seconds"],flush=True)
    return result


def main():
    torch.set_num_threads(2)
    REPORT.mkdir(parents=True,exist_ok=True)
    evaluation=json.loads((ROOT/"reports/noise-stress-v1/noise_sources.json").read_text())
    assert not {s[3] for s in SOURCES}&{s["original_filename"] for s in evaluation}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        sources=list(pool.map(prepare,SOURCES))
    assert not {s["original_sha256"] for s in sources}&{s["original_sha256"] for s in evaluation}
    assert not {s["prepared_sha256"] for s in sources}&{s["prepared_sha256"] for s in evaluation}
    write_json(REPORT/"training_noise_sources.json",sources)
    print("TRAIN_SOURCES_READY",len(sources),flush=True)


if __name__=="__main__":main()
