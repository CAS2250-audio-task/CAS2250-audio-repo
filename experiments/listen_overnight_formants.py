"""Export actual formant-only training examples from the frozen waveform bank."""
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from src import engine

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/"runs/overnight-v1"
REPORT=ROOT/"reports/overnight-v1/listening/formants"


def digest(wave):
    return hashlib.sha256(np.asarray(wave,dtype=np.float32).tobytes()).hexdigest()


def main():
    REPORT.mkdir(parents=True,exist_ok=True)
    data_root=ROOT/"datasets/task2/fixed-1p5-v1"
    metadata=pd.read_csv(data_root/"metadata.csv")
    words=metadata[metadata.group.ne("speaker:spk01")&metadata.label.ne("silence")].reset_index(drop=True)
    indices={p:i for i,p in enumerate(words.path)}
    records=json.loads((STUDY/"waveforms/rows.json").read_text())
    bank=np.load(STUDY/"waveforms/waves.npy",mmap_mode="r")
    grouped={}
    for record in records:
        if record["flags"]==4:grouped.setdefault(record["path"],[]).append(record)
    manifest=[];sections=[]
    lines=["# 실제 포먼트 변형 듣기 샘플", "",
        "각 이어듣기는 원본 → 포먼트 낮춤 → 포먼트 높임 순서이며 사이에0.4초 간격이 있다.",
        "이번 B조건에서 사용한 실제 파형 캐시의 flags4(포먼트만)이다. 피치·속도 변형·추가 소음 없이 음색 차이를 비교한다.",
        "샘플은 두 변형의 비율이 서로 반대이고 차이가3% 이상인 출발 파일 중 경로순 첫 사례다. 예측 정오 여부는 선택에 쓰지 않았다.",
        "별도 볼륨 정규화 없이16kHz PCM16으로 저장했다. 변형 전 float32 파형 해시는 캐시와 실제 training trace 모두 대조했다.",
        "화자2는02검증fold의학습에포함되지않는다. 이화자의증강이훈련에쓰인03fold의입력을예시로내보냈다.", ""]
    for speaker,heldout in (("spk02","spk03"),("spk03","spk02")):
        chosen=None
        for path,pair in sorted(grouped.items()):
            if not path.startswith(f"audio/출발/{speaker}_"):continue
            low=[r for r in pair if r["details"]["formant_ratio"]<.97]
            high=[r for r in pair if r["details"]["formant_ratio"]>1.03]
            if low and high:
                chosen=(path,low[0],high[0]);break
        assert chosen is not None
        path,low,high=chosen
        events={}
        for p in sorted((STUDY/"candidates/formant_B"/heldout/"epochs").glob("augmentation_*.csv.gz")):
            frame=pd.read_csv(p)
            q=frame[frame.path.eq(path)&frame.formant_selected&~frame.noise_selected]
            for row in q.to_dict("records"):
                events.setdefault(int(row["formant_variant"]),row)
            if all(r["variant"] in events for r in (low,high)):break
        assert all(r["variant"] in events for r in (low,high))
        original,rate=sf.read(data_root/path,dtype="float32")
        assert rate==16000 and original.shape==(24000,)
        folder=REPORT/speaker;folder.mkdir(exist_ok=True)
        waves=[("original",original,1.,None)]
        for name,record in (("formants_lower",low),("formants_higher",high)):
            wave=np.array(bank[indices[path],3,record["variant"]])
            sha=digest(wave)
            assert sha==record["output_sha256"]==events[record["variant"]]["input_sha256"]
            waves.append((name,wave,record["details"]["formant_ratio"],events[record["variant"]]))
        for name,wave,ratio,event in waves:
            target=folder/f"{name}.wav";sf.write(target,wave,16000,subtype="PCM_16")
            reread,sr=sf.read(target,dtype="float32")
            assert sr==16000 and reread.shape==(24000,) and np.max(np.abs(reread-wave))<=1/32768+1e-7
            manifest.append(dict(speaker=speaker,source=path,file=str(target.relative_to(REPORT)),formant_ratio=ratio,
                source_float32_sha256=digest(wave),export_file_sha256=engine.file_hash(target),
                training_event=None if event is None else dict(heldout_speaker=heldout,epoch=int(event["epoch"]),draw=int(event["draw"]),input_sha256=event["input_sha256"])))
        gap=np.zeros(6400,np.float32)
        comparison=np.concatenate([original,gap,waves[1][1],gap,waves[2][1]])
        sf.write(folder/"comparison.wav",comparison,16000,subtype="PCM_16")
        ratios=[r[2] for r in waves]
        lines += [f"## {speaker} · 출발", "", f"원본 `{path}`. 포먼트 비율 **{ratios[1]:.4f}배 → {ratios[2]:.4f}배**.", "",
            f"[원본→낮춤→높임 이어듣기]({speaker}/comparison.wav)", "",
            f"[원본]({speaker}/original.wav) · [포먼트 낮춤]({speaker}/formants_lower.wav) · [포먼트 높임]({speaker}/formants_higher.wav)", ""]
        players=[]
        for label,file in (("원본 → 포먼트 낮춤 → 포먼트 높임","comparison"),("원본","original"),(f"포먼트 {ratios[1]:.4f}배","formants_lower"),(f"포먼트 {ratios[2]:.4f}배","formants_higher")):
            players.append(f'<p>{html.escape(label)}<br><audio controls preload="none" src="{speaker}/{file}.wav"></audio></p>')
        sections.append(f'<section><h2>{speaker} · 출발</h2><p>{html.escape(path)}</p>{"".join(players)}</section>')
    (REPORT/"README.md").write_text("\n".join(lines).rstrip()+"\n")
    (REPORT/"index.html").write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>포먼트 듣기 비교</title><style>body{font:17px system-ui;max-width:760px;margin:30px auto;padding:0 20px}audio{width:100%}section{margin:35px 0}</style><h1>실제 포먼트 변형 샘플</h1><p>각 비교는 원본 → 포먼트 낮춤 → 포먼트 높임입니다. 피치는 유지하고 추가 소음은 넣지 않았습니다.</p>'+''.join(sections)+'</html>\n')
    engine.write_json(REPORT/"manifest.json",manifest)
    print(json.dumps([dict(speaker=m['speaker'],ratio=m['formant_ratio'],file=m['file']) for m in manifest],ensure_ascii=False))


if __name__=="__main__":main()
