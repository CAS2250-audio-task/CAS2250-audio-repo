"""Disposable actual-audio timing pilot; its weights are never used by the study."""
import json,time
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.nn import functional as F
from .run import DATA,OUT,Frontend,features,predict
from .models import matched_models

def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    frame=pd.read_csv(DATA/'metadata.csv')
    frame=frame[frame.speaker_id.isin(['spk03','spk04','spk05'])].reset_index(drop=True)
    waves=np.stack([sf.read(DATA/p,dtype='float32')[0] for p in frame.path])
    frontend=Frontend().cuda()
    start=time.monotonic();x=features(frontend,waves);torch.cuda.synchronize()
    frontend_seconds=time.monotonic()-start
    x=(x-x.mean((0,2),keepdim=True))/x.std((0,2),keepdim=True).clamp_min(1e-4)
    rows=[]
    for name,model in matched_models().items():
        model=model.cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        y=torch.zeros(len(x),device='cuda',dtype=torch.long)
        # Warm up kernels; do not use these weights outside this timing process.
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True);F.cross_entropy(model(x[:32]),y[:32]).backward();optimizer.step()
        torch.cuda.synchronize();start=time.monotonic()
        for i in range(0,len(x),32):
            optimizer.zero_grad(set_to_none=True);loss=F.cross_entropy(model(x[i:i+32]),y[i:i+32]);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
        torch.cuda.synchronize();training_seconds=time.monotonic()-start
        start=time.monotonic()
        for _ in range(3):predict(model,x)
        torch.cuda.synchronize();evaluation_seconds=time.monotonic()-start
        rows.append(dict(model=name,training_609_seconds=training_seconds,evaluation_1827_seconds=evaluation_seconds))
    value=dict(frontend_609_seconds=frontend_seconds,models=rows,
               estimated_200_epochs_minutes=sum(r['training_609_seconds']+r['evaluation_1827_seconds'] for r in rows)*200/60,
               excludes='online waveform mixing, checkpoint/audit I/O, source restoration, cache creation and final evaluation')
    (OUT/'timing_pilot.json').write_text(json.dumps(value,indent=2))
    print(json.dumps(value,indent=2),flush=True)

if __name__=='__main__':main()
