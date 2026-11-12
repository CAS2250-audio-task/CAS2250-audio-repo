"""Check speech-noise probes for re-encoded copies of our evaluation recordings."""
import json

import numpy as np
import pandas as pd
from scipy import signal
import soundfile as sf

from src.engine import write_json
from .sources import ROOT, REPORT


def low_rate(source):
    wave,rate=sf.read(ROOT/source["prepared_path"],dtype="float32")
    assert rate==16000
    return signal.resample_poly(wave,1,80).astype(np.float64)


def main():
    added=json.loads((REPORT/"added_speech_sources.json").read_text())
    evaluation=json.loads((ROOT/"reports/noise-stress-v1/noise_sources.json").read_text())
    references={s["noise_id"]:low_rate(s) for s in evaluation}
    denominators={}
    rows=[]
    for source in added:
        wave=low_rate(source)
        length=min(1600,len(wave))
        for start in sorted({0,(len(wave)-length)//2,len(wave)-length}):
            template=wave[start:start+length].copy();template-=template.mean()
            scale=np.linalg.norm(template)
            if scale<1e-10:continue
            for noise_id,reference in references.items():
                key=(noise_id,length)
                if key not in denominators:
                    a=np.r_[0,np.cumsum(reference)]
                    b=np.r_[0,np.cumsum(reference**2)]
                    local_power=b[length:]-b[:-length]-(a[length:]-a[:-length])**2/length
                    denominators[key]=np.sqrt(np.maximum(local_power,1e-20))
                correlation=signal.correlate(reference,template,mode="valid",method="fft")/(denominators[key]*scale)
                best=int(np.abs(correlation).argmax())
                rows.append(dict(added_noise_id=source["noise_id"],eval_noise_id=noise_id,
                    probe_start_seconds=start/200,probe_seconds=length/200,
                    max_abs_correlation=float(abs(correlation[best])),eval_start_seconds=best/200))
        print("SOURCE_OVERLAP_PROBED",source["noise_id"],flush=True)
    frame=pd.DataFrame(rows)
    frame.to_csv(REPORT/"source_overlap_probes.csv",index=False)
    suspicious=frame[frame.max_abs_correlation>.95]
    write_json(REPORT/"source_overlap_audit.json",dict(added_files=len(added),evaluation_recordings=len(evaluation),
        probe_pairs=len(frame),downsample_rate=200,probe_seconds=8,probe_positions="first/middle/last",
        maximum_abs_correlation=float(frame.max_abs_correlation.max()),threshold=.95,
        suspicious_probe_pairs=len(suspicious),test_audio_read=False,
        limitation="Probes reject obvious shifted/rescaled/re-encoded copies; they cannot prove complete provenance or absence of all partial overlap"))
    assert suspicious.empty,suspicious.to_string()
    print("SOURCE_OVERLAP_AUDIT_OK",frame.max_abs_correlation.max(),flush=True)


if __name__=="__main__":main()
