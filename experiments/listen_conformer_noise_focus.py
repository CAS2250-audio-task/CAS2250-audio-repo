"""Exact new training mixtures and fixed validation inputs with all model predictions."""
import hashlib
import json

import numpy as np
import pandas as pd
import soundfile as sf

from src import engine
from src.config import load_config
from src.dataset import read_metadata
from experiments.conformer_sanity import speaker_fold
from experiments.conformer_noise_focus.augment import FocusTrainingClips
from experiments.conformer_noise_focus.sources import ROOT,STUDY,REPORT
from experiments.conformer_noise_focus.run import CONDITIONS


def main():
    folder=STUDY/"listening";folder.mkdir(exist_ok=True)
    training=folder/"training";training.mkdir(exist_ok=True)
    validation=folder/"validation";validation.mkdir(exist_ok=True)
    cfg=load_config(ROOT/"configs/task2_real.json")
    metadata=read_metadata(cfg)
    protocol=json.loads((REPORT/"protocol.json").read_text())
    traces=pd.read_csv(REPORT/"training_draws.csv.gz",low_memory=False)
    source_map={s["noise_id"]:s for s in protocol["original_noise_sources"]+protocol["added_speech_sources"]}
    recordings={s["noise_id"]:sf.read(ROOT/s["prepared_path"],dtype="float32")[0] for s in source_map.values()}
    fold=speaker_fold(metadata,"spk05",cfg)
    data=FocusTrainingClips(fold[fold.split.eq("train")],cfg,"spk05",protocol["original_noise_sources"],
                           protocol["added_speech_sources"],recordings,**CONDITIONS["speech_focus"])
    samples=traces[traces.training_condition.eq("speech_focus")&traces.heldout_speaker.eq("spk05")
                   &traces.added_speech_selected&traces.near_zero_selected&traces.path.str.startswith("audio/출발/")]
    manifests=[]
    lines=["# 이번 증강·DFN3 비교의 듣기 샘플", "", "## 실제 훈련 입력", "",
        "결합 조건의 spk05-fold 훈련 draw 중 출발 파일에서 각 유형의 SNR이0dB에 가장 가까운 사례를 골랐다. "
        "모델 오류 여부로 고른 예시는 아니다. 실제1.5초 입력을 해시까지 재현했으며 개별 볼륨/피크 정규화는 없다.", ""]
    for family in ("babble","neighbor_speech","announcements"):
        candidates=samples[samples.source_family.eq(family)]
        row=candidates.loc[candidates.snr_db.abs().idxmin()]
        data.set_epoch(int(row.epoch));data.draw=int(row.draw)
        index=int(np.flatnonzero(data.frame.path.eq(row.path))[0])
        wave,_,_=data[index];wave=wave.numpy()
        assert hashlib.sha256(wave.tobytes()).hexdigest()==row.input_sha256
        original=data.wave[index]
        source=recordings[row.noise_id][int(row.start_sample):int(row.start_sample)+24000].astype(np.float64)
        source-=source.mean()
        noise=(source*row.noise_gain*row.joint_attenuation).astype(np.float32)
        for name,output in (("original",original),("noise_component",noise),("mixed",wave)):
            file=training/f"{family}_{name}.wav"
            sf.write(file,output,16000,subtype="FLOAT")
            manifests.append(dict(kind="training",family=family,file=str(file.relative_to(folder)),
                output_sha256=hashlib.sha256(output.tobytes()).hexdigest(),path=row.path,
                heldout_speaker="spk05",epoch=int(row.epoch),draw=int(row.draw),noise_id=row.noise_id,snr_db=float(row.snr_db)))
        comparison=np.concatenate([original,np.zeros(6400,np.float32),noise,np.zeros(6400,np.float32),wave])
        sf.write(training/f"{family}_comparison.wav",comparison,16000,subtype="FLOAT")
        lines += [f"- **{family} / {row.snr_db:+.2f}dB**: [실제 혼합](training/{family}_mixed.wav), "
                  f"[추가 잡음 성분](training/{family}_noise_component.wav), [원본→잡음→혼합](training/{family}_comparison.wav)"]
    p=pd.read_csv(REPORT/"predictions.csv.gz")
    labels=cfg.labels
    lines += ["", "## 동일 검증 음성의 DFN3 전후", "",
        "이전부터 사용한 spk02_052 출발 파일의 traffic/restaurant/compressor 0dB draw0다. "
        "학습 조건이 달라도 입력 WAV는 같고 예측이 달라진다. 이어듣기는 원본→혼합→DFN3이며0.4초 간격이다.", ""]
    cases=pd.read_csv(ROOT/"runs/denoise-eval-v1/spk02_cases.csv.gz")
    raw=np.load(ROOT/"runs/denoise-eval-v1/spk02_inputs.npy",mmap_mode="r")
    filtered=np.load(ROOT/"runs/denoise-eval-v1/filtered/deepfilternet3/spk02.npy",mmap_mode="r")
    path="audio/출발/spk02_052.wav"
    original=np.array(raw[int(cases[cases.kind.eq("clean")&cases.path.eq(path)].iloc[0].case_index)])
    pred_rows=[]
    for noise_id in ("traffic","restaurant","compressor"):
        row=cases[cases.kind.eq("mixed")&cases.noise_id.eq(noise_id)&cases.draw.eq(0)&cases.path.eq(path)].iloc[0]
        i=int(row.case_index)
        audit=json.loads((STUDY/"evaluation_audits"/f"spk02_mixed_{noise_id}_0_0.json").read_text())
        j=audit["paths"].index(path)
        for denoiser,wave in (("off",np.array(raw[i])),("dfn3",np.array(filtered[i]))):
            expected=audit["input_sha256" if denoiser=="off" else "dfn3_output_sha256"][j]
            assert hashlib.sha256(wave.tobytes()).hexdigest()==expected
            file=validation/f"{noise_id}_{denoiser}.wav"
            sf.write(file,wave,16000,subtype="FLOAT")
            manifests.append(dict(kind="validation",noise_id=noise_id,denoiser=denoiser,
                file=str(file.relative_to(folder)),output_sha256=expected,path=path,snr_db=0,draw=0))
        joined=np.concatenate([original,np.zeros(6400,np.float32),raw[i],np.zeros(6400,np.float32),filtered[i]])
        sf.write(validation/f"{noise_id}_comparison.wav",joined,16000,subtype="FLOAT")
        lines += [f"### {noise_id}", "", f"[원본→소음 혼합→DFN3](validation/{noise_id}_comparison.wav)", "",
                  "| 학습 조건 | DFN3 끔 | DFN3 켬 |", "| --- | --- | --- |"]
        chosen=p[p.heldout_speaker.eq("spk02")&p.kind.eq("mixed")&p.noise_id.eq(noise_id)&p.snr_db.eq(0)&p.draw.eq(0)&p.path.eq(path)]
        for condition in ("baseline",*CONDITIONS):
            predictions={r.denoiser:labels[int(r.pred_idx)] for r in chosen[chosen.training_condition.eq(condition)].itertuples()}
            lines.append(f"| {condition} | {predictions['off']} | {predictions['dfn3']} |")
            pred_rows.append(dict(noise_id=noise_id,training_condition=condition,**predictions))
        lines.append("")
    pd.DataFrame(pred_rows).to_csv(folder/"validation_predictions.csv",index=False)
    (folder/"README.md").write_text("\n".join(lines),encoding="utf-8")
    engine.write_json(folder/"manifest.json",manifests)
    print("FOCUS_LISTENING_EXPORTED",len(manifests),flush=True)


if __name__=="__main__":main()
