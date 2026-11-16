"""Frozen-checkpoint audit: EOF artifacts vs remaining noise robustness failures.

No training, threshold/margin adjustment, best-epoch reselection or S01 access.
Historical results stay intact. Primary finite-record correction holds the last
observed mel frame beyond EOF. Acoustic silence is a separate control, not a
chosen best policy. Long-noise continuation provides an observed-future control.
"""
import hashlib,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from experiments.boundary_detector.data import load_audio,read_manifest,sha
from experiments.boundary_detector.model import Frontend,CNNGRU
from experiments.boundary_detector.train import write,load_records,mix
from experiments.boundary_detector.segments import decode,boundary_metrics,asymmetric_cost
from experiments.detector_diversity.models import make_model
from experiments.detector_diversity.inference import infer_logits
from experiments.detector_diversity.report import cache
from experiments.detector_diversity.train import region_metrics
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/detector-diversity-v2';OUT=ROOT/'reports/detector-diversity-v2/eof-audit'

@torch.inference_mode()
def score(model,items,args,policy,mean,std):
    details=[];endpoint=[];prefix_diffs=[]
    silence=(-10-mean[0])/std[0]
    for offset in range(0,len(items),32):
        batch=items[offset:offset+32];lengths=[m.shape[-1] for _,_,_,m in batch];t=max(lengths)
        x=torch.stack([F.pad(m,(0,t-m.shape[-1])) for _,_,_,m in batch])
        logits=infer_logits(model,x,lengths,policy,silence).sigmoid().cpu().numpy()
        if policy=='hold' and getattr(model,'lookahead_frames',0):
            old=model(x).sigmoid().cpu().numpy();k=model.lookahead_frames
            prefix_diffs.extend(float(np.abs(a[:n-k]-b[:n-k]).max()) for a,b,n in zip(logits,old,lengths) if n>k)
        for (r,duration,condition,mel),prob in zip(batch,logits):
            intervals=decode(prob[:mel.shape[-1]],.005,duration,.6,.4,20,50)
            details.append(dict(source_id=r['source_id'],speaker=r['speaker'],condition=condition,duration_seconds=duration,gold=r['segments'],predicted=intervals,annotation_source=r['annotation_source']))
            endpoint.append(boundary_metrics(r['segments'],intervals))
    metrics=region_metrics(details);metrics['asymmetric_cost_ms']=1000*np.mean([asymmetric_cost(r['gold'],r['predicted'])['cost_seconds'] for r in details]);groups={}
    for condition in ['original','noisy','noise_only']:
        rs=[r for r in details if ('noisy' if r['condition'].startswith('noise') and r['condition']!='noise_only' else r['condition'])==condition]
        if not rs:continue
        bs=[boundary_metrics(r['gold'],r['predicted']) for r in rs];matched=sum(b['matched'] for b in bs);gold=sum(b['gold'] for b in bs);pred=sum(b['pred'] for b in bs)
        ss=[v for b in bs for v in b['start_errors']];ee=[v for b in bs for v in b['end_errors']];events=[(a,b,r['duration_seconds']) for r in rs for a,b in r['predicted']]
        groups[condition]=dict(cases=len(rs),event_f1=2*matched/max(gold+pred,1),matched=matched,missed_events=gold-matched,false_events=pred-matched,fp_files=sum(bool(r['predicted']) for r in rs) if condition=='noise_only' else None,events_starting_last80ms=sum(a>=duration-.08 for a,b,duration in events),events_touching_end=sum(b>=duration-.005 for a,b,duration in events),start_mae_ms=1000*float(np.mean(ss)) if ss else None,end_mae_ms=1000*float(np.mean(ee)) if ee else None)
    metrics['by_condition']=groups;metrics['max_prefix_probability_change']=max(prefix_diffs,default=0.)
    if prefix_diffs:assert max(prefix_diffs)<.0001,'EOF correction altered past predictions materially'
    return metrics,details

@torch.inference_mode()
def real_future(frontend,noises,mean,std,reference):
    result=[];skipped=[];extra=1600
    for n,noise in enumerate(noises):
        for j in range(10):
            seed=42+10000+n*100+j;rng=np.random.default_rng(seed)
            offset=int(rng.integers(len(noise)-24000+1))
            if offset+24000+extra>len(noise):skipped.append([n,j]);continue
            crop=noise[offset:offset+24000].astype(np.float64);dc=crop.mean();crop-=dc;power=np.mean(crop*crop)
            if power<1e-12:skipped.append([n,j]);continue
            gain=np.sqrt(reference/power);atten=min(1.,.99/max(np.max(np.abs(gain*crop)),1e-12))
            wave=((noise[offset:offset+24000+extra].astype(np.float64)-dc)*gain*atten).astype(np.float32)
            old=mix(np.zeros(24000,np.float32),noise,[],np.random.default_rng(seed),0,reference)
            np.testing.assert_array_equal(wave[:24000],old)
            mel=(frontend(torch.as_tensor(wave,device='cuda')[None])-mean)/std
            r=dict(source_id=f'negative_{n}_{j}',speaker='external',segments=[],annotation_source='external_noise')
            # Full future is supplied; score only original 1.5s, never mask tail frames.
            result.append((r,1.5,'noise_only',mel[0]))
    return result,skipped

@torch.inference_mode()
def main():
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);tic=time.monotonic();names=['baseline','lookahead','boundary','spectral','combined'];states={};models={}
    for name in names:
        s=torch.load(RUN/name/'best.pt',map_location='cuda',weights_only=False);c=s['config'];m=make_model(c['variant'],c['width'],c['hop'],c.get('enabled')).cuda();m.load_state_dict(s['model']);m.eval();states[name]=s;models[name]=m
    oldpath=ROOT.parent/'CAS2250-speech-boundary-detector/runs/boundary-300-v1/cnn_gru/best.pt';s=torch.load(oldpath,map_location='cuda',weights_only=False);m=CNNGRU(64).cuda();m.load_state_dict(s['model']);m.eval();states['v1_legacy_noise']=s;models['v1_legacy_noise']=m
    for name in ['lookahead','combined']:
        s=torch.load(RUN/name/'last.pt',map_location='cuda',weights_only=False);c=s['config'];m=make_model(c['variant'],c['width'],c['hop'],c.get('enabled')).cuda();m.load_state_dict(s['model']);m.eval();states[name+'_last']=s;models[name+'_last']=m
    base=states['baseline'];mean=base['mean'];std=base['std'];ref=base['reference_power'];frontend=Frontend(80).cuda().eval();records=load_records(read_manifest(RUN/'manifest.jsonl',['S02']))
    metadata=json.loads((RUN/'noise.json').read_text());noises=[load_audio(r['path']) for r in metadata['validation']];audit=json.loads((RUN/'noise_audit.json').read_text())['audit'];new_noises=[load_audio(r['path']) for r in audit]
    args=SimpleNamespace(seed=42,device='cuda',hop=80,batch=32,on=.6,off=.4,on_ms=20.,off_ms=50.,eval_snr=0)
    historical={(r['name'],r['suite']):r for r in json.loads((ROOT/'reports/detector-diversity-v2/final_metrics.json').read_text())};results=[];replay=[]
    def run(name,items,suite,policy):
        s=states[name];own=items if torch.equal(s['mean'],mean) and torch.equal(s['std'],std) else [(r,d,c,(mel*std[0]+mean[0]-s['mean'][0])/s['std'][0]) for r,d,c,mel in items]
        metrics,details=score(models[name],own,args,policy,s['mean'],s['std']);row=dict(model=name,suite=suite,policy=policy,epoch=s['epoch'],**metrics);results.append(row)
        write(OUT/f'{name}_{suite}_{policy}_predictions.json',details)
        if policy=='legacy_zero' and name in names:
            prior=historical[(name,suite)]
            replay.append(dict(model=name,suite=suite,cost_difference_ms=metrics['asymmetric_cost_ms']-prior['asymmetric_cost_ms']))
            assert abs(metrics['asymmetric_cost_ms']-prior['asymmetric_cost_ms'])<.1
        write(OUT/'metrics.json',results)
        print(json.dumps(dict(model=name,suite=suite,policy=policy,original=metrics['by_condition'].get('original'),noisy=metrics['by_condition'].get('noisy'),noise_only=metrics['by_condition'].get('noise_only'))),flush=True)
        return row
    for suite,snr,bank in [('legacy_0db',0,noises),('legacy_5db',5,noises),('legacy_10db',10,noises),('new_environment_0db',0,new_noises)]:
        args.eval_snr=snr;items,retries=cache(frontend,records,bank,ref,mean,std,args)
        for name in names+['v1_legacy_noise']:run(name,items,suite,'hold')
        if snr==0:
            for name in ['lookahead','combined']:
                run(name,items,suite,'legacy_zero');run(name,items,suite,'silence');run(name+'_last',items,suite,'hold')
        del items
    for suite,bank in [('legacy_noise_observed_future',noises),('new_noise_observed_future',new_noises)]:
        items,skipped=real_future(frontend,bank,mean,std,ref)
        for name in names+['v1_legacy_noise']:run(name,items,suite,'hold')
        write(OUT/(suite+'_context.json'),dict(cases=len(items),skipped_no_available_future=skipped,original1p5_waveform_identical=True,observed_future_ms=100,labels_scored_duration_seconds=1.5))
    # Compare event changes, retaining all original scoring frames.
    for name in ['lookahead','combined']:
        for suite in ['legacy_0db','new_environment_0db']:
            old=json.loads((OUT/f'{name}_{suite}_legacy_zero_predictions.json').read_text());held=json.loads((OUT/f'{name}_{suite}_hold_predictions.json').read_text());changes=[]
            for a,b in zip(old,held):
                assert a['source_id']==b['source_id'] and a['condition']==b['condition']
                if a['predicted']!=b['predicted']:changes.append(dict(source_id=a['source_id'],condition=a['condition'],gold=a['gold'],old=a['predicted'],held=b['predicted'],duration_seconds=a['duration_seconds']))
            write(OUT/f'{name}_{suite}_changed_cases.json',changes)
    write(OUT/'audit.json',dict(training=False,checkpoint_reselection=False,thresholds_changed=False,margin_ms=0,tail_scoring_frames_removed=False,negative_labels_all_zero=True,validation_speaker='S02',S01_evaluated=False,primary_policy='hold last observed frame only for unavailable future beyond each true sequence end',silence_control='Acoustic zero waveform has logmel -10, standardized per checkpoint',observed_future_control='Same exact first1.5s noise waveform/gain, actual next100ms supplied; score original1.5s only',legacy_replay=replay,checkpoint_sha256={n:sha(RUN/n/'best.pt') for n in names},elapsed_seconds=time.monotonic()-tic))
    print('COMPLETE',time.monotonic()-tic,flush=True)

if __name__=='__main__':main()
