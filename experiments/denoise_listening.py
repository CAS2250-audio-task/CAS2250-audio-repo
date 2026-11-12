"""Export exact evaluated waves, with predictions and explicit example selection."""
import hashlib
import json

import numpy as np
import pandas as pd
import soundfile as sf

from src import engine
from experiments.denoise_eval.filters import ROOT, STUDY, REPORT, FILTERS


def main():
    p=pd.read_csv(REPORT/"predictions.csv.gz")
    folder=STUDY/"listening"
    folder.mkdir(exist_ok=True)
    speaker="spk02"
    cases=pd.read_csv(STUDY/f"{speaker}_cases.csv.gz")
    waves={name:np.load(STUDY/"filtered"/name/f"{speaker}.npy",mmap_mode="r") for name in FILTERS}
    original_path="audio/출발/spk02_052.wav"
    clean_row=cases[cases.kind.eq("clean")&cases.path.eq(original_path)].iloc[0]
    original=waves["none"][int(clean_row.case_index)]
    sf.write(folder/"original.wav",original,16000,subtype="FLOAT")
    labels=["출발","정지","전진","후진","왼쪽","오른쪽","시작","복귀","대기","재시작","silence"]
    manifest=[]
    lines=["# 잡음 제거 전후 듣기", "", "기존에 제공한 held-out spk02의 `출발/spk02_052.wav`를 유지했다. "
        "0dB에서 traffic/restaurant/compressor, draw0의 고정 예시다. 오류 여부를 보고 다시 고른 예시가 아니다.", "",
        "모든 WAV는 실제 평가된16kHz·1.5초 float32 파형이다. 개별 볼륨/피크 정규화하지 않았다. "
        "이어듣기는 원본→혼합→DFN3→DNS64→NSNet2, 중간0.4초 간격이다. OGG는 편의용 손실 압축본이다.", "",
        "[잡음 추가 전 원본](original.wav)", ""]
    for noise_id in ("traffic","restaurant","compressor"):
        row=cases[cases.kind.eq("mixed")&cases.noise_id.eq(noise_id)&cases.draw.eq(0)&cases.path.eq(original_path)].iloc[0]
        index=int(row.case_index)
        comparison=[original,np.zeros(6400,np.float32)]
        lines += [f"## {noise_id} · 0dB", "", "| 입력 | 예측 | 실제 입력 WAV |", "| --- | --- | --- |"]
        for name in FILTERS:
            wave=np.array(waves[name][index])
            info=p[p["filter"].eq(name)&p.heldout_speaker.eq(speaker)&p.case_index.eq(index)].iloc[0]
            digest=hashlib.sha256(wave.tobytes()).hexdigest()
            assert digest==info.output_sha256
            filename=f"{noise_id}_{name}.wav"
            sf.write(folder/filename,wave,16000,subtype="FLOAT")
            comparison.extend([wave,np.zeros(6400,np.float32)])
            manifest.append(dict(noise_id=noise_id,path=original_path,filter=name,case_index=index,
                                 input_sha256=row.input_sha256,output_sha256=digest,file=filename,
                                 true_label="출발",pred_label=labels[int(info.pred_idx)]))
            lines.append(f"| {name} | {labels[int(info.pred_idx)]} | [{filename}]({filename}) |")
        joined=np.concatenate(comparison[:-1])
        sf.write(folder/f"{noise_id}_comparison.wav",joined,16000,subtype="FLOAT")
        sf.write(folder/f"{noise_id}_comparison.ogg",joined,16000,format="OGG",subtype="VORBIS")
        lines += ["",f"[이어듣기 WAV]({noise_id}_comparison.wav) · [이어듣기 OGG]({noise_id}_comparison.ogg)",""]
    (folder/"README.md").write_text("\n".join(lines),encoding="utf-8")
    engine.write_json(folder/"manifest.json",manifest)
    print("LISTENING_EXPORTED",len(manifest),folder,flush=True)


if __name__=="__main__":main()
