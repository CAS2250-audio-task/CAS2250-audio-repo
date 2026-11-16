"""Inference-only analysis on S02 validation; no training or holdout access.

Rebuild exact detector evaluation mixtures. Feed those SAME mixed samples to
historical classifiers inside a gold-centered 1.5s window (oracle localization).
This compares error patterns, not task-equivalent accuracy. Sweep +5/+10dB
without checkpoint/threshold changes to isolate additional-noise severity.
"""
import hashlib,json,os,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from experiments.boundary_detector.data import load_audio,read_manifest,SR,sha
from experiments.boundary_detector.train import mix
from experiments.boundary_detector.model import Frontend,make_detector
from experiments.boundary_detector.segments import decode,asymmetric_cost
from experiments.acoustic_local.features import Extractor
from experiments.acoustic_local.models import acoustic_models
from experiments.compact_pooling.run import Frontend as ClassifierFrontend

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/boundary-detector-v1/error-types'
BASE=ROOT/'runs/boundary-300-v1'
OLD=Path('/root/CAS2250-compact-local-acoustic-aux')
LABELS=['출발','정지','전진','후진','왼쪽','오른쪽','시작','복귀','대기','재시작','silence']

def crop(wave,a,b):
    start=round((a+b)/2*SR)-12000;out=np.zeros(24000,np.float32)
    lo=max(0,start);hi=min(len(wave),start+24000)
    if hi>lo:out[lo-start:hi-start]=wave[lo:hi]
    assert b-a<1.5
    return out

def classify_errors(gold,pred):
    a,b=gold[0];ov=[(max(a,c),min(b,d)) for c,d in pred if min(b,d)>max(a,c)]
    lead=ov[0][0]-a if ov else b-a;trail=b-ov[-1][1] if ov else b-a
    gaps=sum(max(0,ov[i][0]-ov[i-1][1]) for i in range(1,len(ov)))
    extra=sum(d-c for c,d in pred)-sum(d-c for c,d in ov)
    ious=[max(0,min(b,d)-max(a,c))/(max(b,d)-min(a,c)) for c,d in pred]
    late=bool(ov and lead>.02);early=bool(ov and trail>.02)
    spurious=sum(min(b,d)<=max(a,c) for c,d in pred)
    return dict(event_hit=max(ious,default=0)>=.5,complete_miss=not ov,
                internal_split=len(ov)>1,extra_events=spurious>0,
                late_start=late,early_end=early,overextension=extra>.05,
                leading_missing_ms=lead*1000,trailing_missing_ms=trail*1000,internal_gap_ms=gaps*1000,
                extra_ms=extra*1000,predicted_events=len(pred),
                strict30=len(pred)==1 and abs(pred[0][0]-a)<=.03 and abs(pred[0][1]-b)<=.03,
                cost_ms=asymmetric_cost(gold,pred)['cost_seconds']*1000)

def boundary_snr(wave,noise,segments,key,snr):
    rng=np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little'))
    offset=int(rng.integers(len(noise)-len(wave)+1));n=noise[offset:offset+len(wave)].astype(np.float64);n-=n.mean()
    a,b=segments[0];lo,hi=round(a*SR),round(b*SR)
    power=np.mean(wave[lo:hi].astype(np.float64)**2);gain=np.sqrt(power/(np.mean(n[lo:hi]**2)*10**(snr/10)))
    def ratio(l,h):return 10*np.log10(max(np.mean(wave[l:h].astype(np.float64)**2),1e-20)/max(np.mean((gain*n[l:h])**2),1e-20))
    return dict(onset_snr_db=ratio(lo,min(hi,lo+800)),offset_snr_db=ratio(max(lo,hi-800),hi))

@torch.inference_mode()
def main():
    OUT.mkdir(exist_ok=True,parents=True);torch.set_num_threads(2)
    records=read_manifest(ROOT/'runs/manual-clipped-v1-import/manifest.jsonl',['S02'])
    originals=[load_audio(r['audio_path']) for r in records]
    sources=json.loads((ROOT/'reports/compact-acoustic-local-spk03-e1000/protocol.json').read_text())['sources']
    noise_names=[r['noise_id'] for r in sources['evaluation']]
    noise_paths=[Path(r['path']) for r in json.loads((BASE/'noise.json').read_text())['validation']]
    noises=[load_audio(p) for p in noise_paths]
    scaler=json.loads((ROOT/'reports/compact-acoustic-local-spk03-e1000/auxiliary_scaler.json').read_text())
    classifiers=acoustic_models(42,scaler);class_states={}
    for name,m in classifiers.items():
        s=torch.load(OLD/f'runs/compact-acoustic-local-spk03-e1000/spk03/{name}_best.pt',map_location='cuda',weights_only=False)
        m.to('cuda').load_state_dict(s['model']);m.eval();class_states[name]=s
    detectors={};det_states={}
    for arch in ['tcn','cnn_gru']:
        s=torch.load(BASE/arch/'best.pt',map_location='cuda',weights_only=False)
        m=make_detector(arch,s['config']['width']).cuda();m.load_state_dict(s['model']);m.eval();detectors[arch]=m;det_states[arch]=s
    frontend=Frontend(80).cuda().eval();cf=ClassifierFrontend().cuda().eval()
    ext=Extractor(4);rows=[];classrows=[];sampleplots=[];tic=time.monotonic()
    try:
        for snr in [None,0,5,10]:
            cases=[];cropped=[]
            for i,(r,w) in enumerate(zip(records,originals)):
                a,b=r['segments'][0];label=Path(r['audio_path']).parent.name
                for n in ([None] if snr is None else range(len(noises))):
                    key=f'{r["source_id"]}|{n+1 if n is not None else 0}|42'
                    rng=np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little'))
                    x=w if snr is None else mix(w,noises[n],r['segments'],rng,snr,1.)
                    c=dict(source_id=r['source_id'],label=label,noise_id='original' if snr is None else noise_names[n],snr_db=99 if snr is None else snr,duration_seconds=len(w)/SR)
                    if snr is not None:c.update(boundary_snr(w,noises[n],r['segments'],key,snr))
                    c['gold']=r['segments'];c['wave']=x;cases.append(c);cropped.append(crop(x,a,b))
            cw=np.stack(cropped);print('extract',snr,len(cases),flush=True)
            # Auxiliary extraction from noisy oracle-window itself, never clean features.
            aux=torch.from_numpy(ext(cw)).cuda()
            logits_by_model={name:[] for name in classifiers}
            for j in range(0,len(cw),32):
                mel=cf(torch.as_tensor(cw[j:j+32],device='cuda'))
                for name,m in classifiers.items():
                    s=class_states[name];x=torch.cat([(mel-s['mean'])/s['std'],aux[j:j+32]],1)
                    logits_by_model[name].extend(m(x).argmax(1).cpu().tolist())
            for name,pred in logits_by_model.items():
                for c,y in zip(cases,pred):classrows.append(dict(source_id=c['source_id'],label=c['label'],noise_id=c['noise_id'],snr_db=c['snr_db'],model=name,predicted=LABELS[y],correct=LABELS[y]==c['label']))
            for j in range(0,len(cases),32):
                batch=cases[j:j+32];ml=[frontend(torch.as_tensor(c['wave'],device='cuda')[None])[0] for c in batch];length=max(x.shape[-1] for x in ml)
                for arch,m in detectors.items():
                    s=det_states[arch];x=torch.stack([F.pad((z-s['mean'][0])/s['std'][0],(0,length-z.shape[-1])) for z in ml]);probs=m(x).sigmoid().cpu().numpy()
                    for c,z,p in zip(batch,ml,probs):
                        p=p[:z.shape[-1]];seg=decode(p,.005,c['duration_seconds'],.6,.4,20,50)
                        info={k:v for k,v in c.items() if k not in ['wave','gold']};info.update(model=arch,**classify_errors(c['gold'],seg),predicted=json.dumps(seg),gold=json.dumps(c['gold']))
                        rows.append(info)
                        if snr==0 and arch=='cnn_gru':sampleplots.append((info,p.copy(),c['wave'].copy()))
            pd.DataFrame(rows).to_csv(OUT/'detector_cases.csv.gz',index=False)
            pd.DataFrame(classrows).to_csv(OUT/'classifier_cases.csv.gz',index=False)
            print('done',snr,'elapsed_seconds',round(time.monotonic()-tic,1),flush=True)
    finally:ext.close()
    # Exact replay audit: 0dB/original predictions match stored best checkpoint predictions.
    replay_differences=[]
    for arch in detectors:
        prior=json.loads((BASE/arch/'best_predictions.json').read_text());lookup={(r['source_id'],r['condition']):r['predicted'] for r in prior}
        for r in rows:
            if r['model']==arch and r['snr_db'] in [0,99]:
                condition='original' if r['snr_db']==99 else 'noise'+str(noise_names.index(r['noise_id']))
                actual=json.loads(r['predicted']);expected=lookup[(r['source_id'],condition)]
                if actual!=expected:
                    assert np.asarray(actual).shape==np.asarray(expected).shape
                    maximum=float(np.abs(np.asarray(actual)-np.asarray(expected)).max())
                    assert maximum<=.010001,(arch,r['source_id'],condition,maximum)
                    replay_differences.append(dict(model=arch,source_id=r['source_id'],condition=condition,max_endpoint_difference_ms=1000*maximum))
    df=pd.DataFrame(rows);cl=pd.DataFrame(classrows)
    nums=['event_hit','complete_miss','internal_split','extra_events','late_start','early_end','overextension','strict30','cost_ms','leading_missing_ms','trailing_missing_ms','extra_ms']
    df.groupby(['model','snr_db'])[nums].mean().to_csv(OUT/'detector_snr.csv')
    df[df.snr_db==0].groupby(['model','noise_id'])[nums].mean().to_csv(OUT/'detector_noise.csv')
    df[df.snr_db==0].groupby(['model','label'])[nums].mean().to_csv(OUT/'detector_commands.csv')
    cl.groupby(['model','snr_db']).correct.mean().to_csv(OUT/'classifier_snr.csv')
    cl[cl.snr_db==0].groupby(['model','noise_id']).correct.mean().to_csv(OUT/'classifier_noise.csv')
    cl[cl.snr_db==0].groupby(['model','label']).correct.mean().to_csv(OUT/'classifier_commands.csv')
    paired=df.merge(cl,on=['source_id','label','noise_id','snr_db'],suffixes=('_detector','_classifier'))
    paired[(paired.snr_db==0)].groupby(['model_detector','model_classifier','event_hit']).correct.agg(['mean','count']).to_csv(OUT/'paired_event_vs_classification.csv')
    for model in classifiers:
        pred=cl[(cl.snr_db==0)&(cl.model==model)];ct=pred[pred.label!=pred.predicted].groupby(['label','predicted']).size().sort_values(ascending=False)
        ct.to_csv(OUT/f'{model}_confusions.csv')
    # Diagnostic examples: one of each mechanism; include exact mixtures and probability tracks.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    chosen=[]
    for category in ['late_start','early_end','internal_split','extra_events','complete_miss']:
        candidates=[z for z in sampleplots if z[0][category]]
        candidates.sort(key=lambda z:z[0]['cost_ms'],reverse=True)
        chosen.append((category,candidates[len(candidates)//2]))
    fig,axes=plt.subplots(len(chosen),2,figsize=(13,12))
    examples=[]
    for index,(category,(r,p,w)) in enumerate(chosen):
        a,b=json.loads(r['gold'])[0];t=np.arange(len(w))/SR
        ax=axes[index,0];ax.plot(t,w,lw=.35,color='gray');ax.axvspan(a,b,alpha=.2,color='green');ax.set_title(f'{category}: {r["source_id"]}, {r["noise_id"]}')
        ax=axes[index,1];ax.plot(np.arange(len(p))*.005,p);ax.axvspan(a,b,alpha=.15,color='green');ax.axhline(.6,ls='--',color='gray');ax.axhline(.4,ls='--',color='gray')
        for c,d in json.loads(r['predicted']):ax.axvspan(c,d,alpha=.13,color='red')
        ax.set_ylim(-.02,1.02);examples.append(dict(category=category,**r))
    fig.tight_layout();fig.savefig(OUT/'diagnostic_probability_tracks.png',dpi=150);plt.close(fig)
    (OUT/'diagnostic_examples.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    audit=dict(validation_speaker='S02 / 노정민',holdout_read=False,training=False,thresholds_changed=False,levels=[0,5,10],cases=len(rows),classification_cases=len(classrows),replay_within10ms=True,replay_differences=replay_differences,
               oracle_crop='1.5s window centered on manual speech midpoint, zero padding outside original, SAME noisy waveform, no extra loudness normalization',
               limitations=['classifiers evaluated on new recordings and gold-localized windows, not historical fixed1.5 dataset','SNR sweep is validation diagnosis, no checkpoint selection','Original speech itself may contain noise','Noise categories not manually listened/audited'],
               hashes={str(p):sha(p) for p in noise_paths},elapsed_seconds=time.monotonic()-tic)
    (OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('COMPLETE',round(time.monotonic()-tic,1),flush=True)

if __name__=='__main__':main()
