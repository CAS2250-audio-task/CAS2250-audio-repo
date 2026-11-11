"""One-time, recorded pre-training cache repair; no model had been trained."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .features import FeatureConfig, NAMES, apply_support, fit_scaler
from .data import render_job
from src.engine import file_hash, write_json


def repair(study):
    study = Path(study)
    if list((study/"runs").glob("*/best.pt")):
        raise ValueError("This preflight repair is forbidden after training starts")
    fc = FeatureConfig()
    repaired = []
    for directory in sorted((study/"cache").iterdir()):
        mp = directory/"manifest.json"
        if not mp.exists():
            continue
        manifest=json.loads(mp.read_text())
        if manifest["feature_config"] == fc.to_dict():
            continue
        archive=directory/"inputs.npz"
        before=file_hash(archive)
        if before != manifest["inputs_sha256"]:
            raise ValueError("Changed preflight cache")
        with np.load(archive,allow_pickle=False) as a:
            arrays={k:a[k].copy() for k in a.files}
        for i,job in enumerate(manifest["jobs"]):
            wav,_=render_job(job)
            arrays["raw"][i],arrays["valid"][i]=apply_support(wav,arrays["raw"][i],fc)
        temp=directory/"support-repair.partial.npz"
        np.savez(temp,**arrays)
        temp.replace(archive)
        manifest["preflight_repair"]=dict(previous_inputs_sha256=before,
            reason="LPC finite roots in digitally empty frames; correct support masks and derived slopes before any training")
        manifest["inputs_sha256"]=file_hash(archive)
        manifest["feature_config"]=fc.to_dict()
        write_json(mp,manifest)
        pd.DataFrame([dict(feature=n,frames=int(arrays["valid"][:,j].size),
            unavailable_frames=int((~arrays["valid"][:,j]).sum()),
            unavailable_fraction=float((~arrays["valid"][:,j]).mean()))
            for j,n in enumerate(NAMES)]).to_csv(directory/"availability.csv",index=False)
        repaired.append(dict(condition=directory.name,before=before,after=manifest["inputs_sha256"]))
    frame=pd.read_csv(study/"cache/original/metadata.csv")
    with np.load(study/"cache/original/inputs.npz") as a:
        train=frame.split.to_numpy()=="train"
        write_json(study/"scaler.json",fit_scaler(a["raw"][train],a["valid"][train]))
    p=json.loads((study/"protocol.json").read_text())
    p["feature_config"]=fc.to_dict()
    write_json(study/"protocol.json",p)
    write_json(study/"preflight_support_repair.json",dict(before_training=True,conditions=repaired))
    print(f"Repaired {len(repaired)} caches before training")


if __name__=="__main__":
    repair("runs/auxiliary-v1")
