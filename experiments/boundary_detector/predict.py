"""Extract predicted intervals; no default extra margin, no input loudness normalization."""
import argparse,json
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from .data import SR,load_audio
from .model import Detector,Frontend
from .segments import decode


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--audio',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu');p.add_argument('--margin-ms',type=float,default=0)
    a=p.parse_args()
    if a.margin_ms<0:p.error('Negative margin')
    if a.output.exists():raise FileExistsError('Use a fresh output directory')
    torch.set_num_threads(2);s=torch.load(a.checkpoint,map_location=a.device,weights_only=False);c=s['config']
    model=Detector(c['width']).to(a.device);model.load_state_dict(s['model']);model.eval();frontend=Frontend(c['hop']).to(a.device)
    wave=load_audio(a.audio)
    with torch.no_grad():prob=model((frontend(torch.as_tensor(wave,device=a.device)[None])-s['mean'])/s['std']).sigmoid()[0].cpu().numpy()
    segments=decode(prob,c['hop']/SR,len(wave)/SR,c['on'],c['off'],c['on_ms'],c['off_ms'])
    a.output.mkdir(parents=True)
    for i,(start,end) in enumerate(segments):
        lo=max(0,round((start-a.margin_ms/1000)*SR));hi=min(len(wave),round((end+a.margin_ms/1000)*SR))
        sf.write(a.output/f'segment_{i:03d}.wav',wave[lo:hi],SR,subtype='FLOAT')
    payload=dict(audio=str(a.audio),sample_rate=SR,segments=segments,margin_ms=a.margin_ms,checkpoint_epoch=s['epoch'],note='Timestamps are model predictions, not certified ground truth')
    (a.output/'segments.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');np.save(a.output/'probabilities.npy',prob)
    print(json.dumps(payload,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
