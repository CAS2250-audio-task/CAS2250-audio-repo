"""Evaluate frozen selected checkpoints: crop retention, over-inclusion, errors and GPU latency."""
import hashlib,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from experiments.boundary_detector.data import load_audio,read_manifest,SR
from experiments.boundary_detector.model import Frontend,CNNGRU
from experiments.boundary_detector.train import write,load_records,reference_power,evaluate,mix
from .models import make_model
from .train import region_metrics
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/detector-diversity-v2';REPORT=ROOT/'reports/detector-diversity-v2'

def cache(frontend,records,noises,reference,mean,std,args):
    result=[];retries=0
    for record,wave in records:
        for n,noise in enumerate([None]+noises):
            key=f'{record["source_id"]}|{n}|{args.seed}';rng=np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little'))
            x=wave
            if noise is not None:
                for retry in range(100):
                    try:x=mix(wave,noise,record['segments'],rng,args.eval_snr,reference);break
                    except ValueError as exc:
                        if 'silent noise crop' not in str(exc):raise
                else:raise ValueError('No audible deterministic audit crop')
                retries+=retry
            mel=(frontend(torch.as_tensor(x,device=args.device)[None])-mean)/std
            result.append((record,len(x)/SR,'original' if noise is None else f'noise{n-1}',mel[0]))
    for n,noise in enumerate(noises):
        for j in range(10):
            rng=np.random.default_rng(args.seed+10000+n*100+j)
            for retry in range(100):
                try:x=mix(np.zeros(24000,np.float32),noise,[],rng,args.eval_snr,reference);break
                except ValueError as exc:
                    if 'silent noise crop' not in str(exc):raise
            else:raise ValueError('No audible noise-only audit crop')
            retries+=retry;record=dict(source_id=f'negative_{n}_{j}',speaker='external',segments=[],annotation_source='external_noise')
            mel=(frontend(torch.as_tensor(x,device=args.device)[None])-mean)/std
            result.append((record,1.5,'noise_only',mel[0]))
    return result,retries

@torch.inference_mode()
def latency(model,frontend,mean,std):
    wave=torch.randn(1,24000,device='cuda')*.01
    for _ in range(20):model((frontend(wave)-mean)/std)
    torch.cuda.synchronize();times=[]
    for _ in range(100):
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record();model((frontend(wave)-mean)/std);end.record();end.synchronize();times.append(start.elapsed_time(end))
    return dict(gpu_frontend_model_median_ms=float(np.median(times)),gpu_frontend_model_p95_ms=float(np.percentile(times,95)),input_seconds=1.5,batch=1,lookahead_ms=getattr(model,'lookahead_frames',0)*5,frontend_lookahead_ms=12.5,note='Whole-clip GPU inference plus mel, excluding audio IO; algorithmic waiting reported separately')

def phase_comparison(names=None):
    REPORT.mkdir(exist_ok=True,parents=True);names=names or ['baseline','lookahead','boundary','spectral'];rows=[]
    for name in names:
        p=OUT/name;hist=json.loads((p/'history.json').read_text());best=min(hist,key=lambda r:r['asymmetric_cost_ms']);inputs=json.loads((p/'inputs.json').read_text())
        rows.append(dict(name=name,parameters=inputs['parameters'],epoch=best['epoch'],cost_ms=best['asymmetric_cost_ms'],speech_missing_pct=best['speech_missing_pct'],original=best['region_by_condition']['original'],noisy=best['region_by_condition']['noisy'],noise_only=best['region_by_condition']['noise_only'],original_event_f1=best['by_condition']['original']['event_f1'],noisy_event_f1=best['by_condition']['noisy']['event_f1'],last_cost_ms=hist[-1]['asymmetric_cost_ms']))
    write(REPORT/'phase1_comparison.json',rows);return rows

@torch.inference_mode()
def final_report():
    REPORT.mkdir(exist_ok=True,parents=True);torch.set_num_threads(2);names=['baseline','lookahead','boundary','spectral']
    if (OUT/'combined/best.pt').exists():names+=['combined']
    states={name:torch.load(OUT/name/'best.pt',map_location='cuda',weights_only=False) for name in names};refstate=states['baseline'];mean=refstate['mean'];std=refstate['std'];ref=refstate['reference_power']
    models={}
    for name,state in states.items():
        torch.testing.assert_close(state['mean'],mean,rtol=1e-5,atol=1e-5);torch.testing.assert_close(state['std'],std,rtol=1e-5,atol=1e-5)
        c=state['config'];m=make_model(c['variant'],c['width'],c['hop'],c.get('enabled')).cuda();m.load_state_dict(state['model']);m.eval();models[name]=m
    old_path=ROOT.parent/'CAS2250-speech-boundary-detector/runs/boundary-300-v1/cnn_gru/best.pt'
    if old_path.exists():
        old=torch.load(old_path,map_location='cuda',weights_only=False);torch.testing.assert_close(old['mean'],mean,rtol=0,atol=0);torch.testing.assert_close(old['std'],std,rtol=0,atol=0)
        m=CNNGRU(64).cuda();m.load_state_dict(old['model']);m.eval();models['v1_legacy_noise']=m;states['v1_legacy_noise']=old
    records=load_records(read_manifest(OUT/'manifest.jsonl',['S02']));noise_metadata=json.loads((OUT/'noise.json').read_text());noises=[load_audio(r['path']) for r in noise_metadata['validation']];audit_meta=json.loads((OUT/'noise_audit.json').read_text())['audit'];audit_noises=[load_audio(r['path']) for r in audit_meta]
    args=SimpleNamespace(seed=42,device='cuda',hop=80,batch=32,on=.6,off=.4,on_ms=20.,off_ms=50.,eval_snr=0)
    frontend=Frontend(80).cuda().eval();rows=json.loads((REPORT/'final_metrics.json').read_text()) if (REPORT/'final_metrics.json').exists() else []
    existing={(r['name'],r['suite']) for r in rows}
    for suite,snr,bank in [('legacy_0db',0,noises),('legacy_5db',5,noises),('legacy_10db',10,noises),('new_environment_0db',0,audit_noises)]:
        todo={name:model for name,model in models.items() if (name,suite) not in existing}
        if not todo:continue
        args.eval_snr=snr;validation,retries=cache(frontend,records,bank,ref,mean,std,args)
        for name,model in todo.items():
            own=states[name]
            model_cache=validation if torch.equal(own['mean'],mean) and torch.equal(own['std'],std) else [(r,d,c,(mel*std[0]+mean[0]-own['mean'][0])/own['std'][0]) for r,d,c,mel in validation]
            metrics,details=evaluate(model,model_cache,args);metrics.update(region_metrics(details));row=dict(name=name,suite=suite,epoch=states[name]['epoch'],noise_crop_retries=retries,**metrics);rows.append(row)
            write(REPORT/f'{name}_{suite}_predictions.json',details)
            import pandas as pd
            from experiments.analyze_boundary_errors import classify_errors
            case_rows=[]
            for detail in details:
                if not detail['gold']:continue
                source=next(r for r,_ in records if r['source_id']==detail['source_id'])
                case_rows.append(dict(source_id=detail['source_id'],label=Path(source['audio_path']).parent.name,condition=detail['condition'],**classify_errors(detail['gold'],detail['predicted'])))
            cases=pd.DataFrame(case_rows);cases.to_csv(REPORT/f'{name}_{suite}_errors.csv.gz',index=False)
            cases.groupby('label')[['complete_miss','internal_split','extra_events','late_start','early_end','leading_missing_ms','trailing_missing_ms']].mean().to_csv(REPORT/f'{name}_{suite}_commands.csv')
            print(json.dumps(dict(stage='final_evaluation',name=name,suite=suite,missing_pct=metrics['region_by_condition']['noisy']['missing_pct'],cost_ms=metrics['asymmetric_cost_ms'])),flush=True)
        del validation
    timings=json.loads((REPORT/'latency.json').read_text()) if (REPORT/'latency.json').exists() else {}
    timings.update({name:latency(model,frontend,states[name]['mean'],states[name]['std']) for name,model in models.items() if name not in timings});write(REPORT/'final_metrics.json',rows);write(REPORT/'latency.json',timings)
    comparison=phase_comparison(names);write(REPORT/'completion.json',dict(stage='complete',models=list(models),epochs_per_phase=300,heldout_speech_S01_evaluated=False,noise_audit_used_for_selection=False,all_common_waveform_augmentations=True))
    lines=['# Detector diversity v2 results','','Selected by the same legacy validation asymmetric duration cost; no crop margins. New environmental sources evaluated only after checkpoint selection.','',
           '| Model | Params | Best epoch | Original event F1 | Legacy0dB event F1 | Noisy speech missing | Extra ms/case | Noise-only FP files |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in comparison:
        lines.append(f"| {r['name']} | {r['parameters']} | {r['epoch']} | {100*r['original_event_f1']:.2f}% | {100*r['noisy_event_f1']:.2f}% | {r['noisy']['missing_pct']:.2f}% | {r['noisy']['extra_ms_per_case']:.1f} | {100*r['noise_only']['noise_fp_rate']:.1f}% |")
    lines+=['','## Fixed-checkpoint noise severity and new-environment audit','','| Model | Suite | Event F1 (noisy) | Speech missing (noisy) | Extra ms/case (noisy) |','|---|---|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['name']} | {r['suite']} | {100*r['by_condition']['noisy']['event_f1']:.2f}% | {r['region_by_condition']['noisy']['missing_pct']:.2f}% | {r['region_by_condition']['noisy']['extra_ms_per_case']:.1f} |")
    lines+=['','## GPU latency','','| Model | Median ms | p95 ms | Added lookahead ms |','|---|---:|---:|---:|']
    for name,t in timings.items():lines.append(f"| {name} | {t['gpu_frontend_model_median_ms']:.3f} | {t['gpu_frontend_model_p95_ms']:.3f} | {t['lookahead_ms']:.1f} |")
    lines+=['','Frontend adds nominal12.5ms analysis lookahead to all models. GPU timings include mel+model on1.5s audio, batch1, excluding IO; not per-frame streaming timing.','',
            'Speech missing counts only uncovered gold time; extra/background time is reported separately. Event F1 and speech retention can rank models differently. Thresholds.6/.4, confirmation20/50ms backdated are held fixed. Boundary heads are training auxiliary heads, not a changed decoder.','',
            'Limits: one validation speaker/one primary seed; original may already contain noise; categorical noise selection is not a listening audit; no intentional speech/babble augmentation. New22sources are disjoint original recordings but largely same broad categories, not necessarily novel semantic noise classes. No S01 speech holdout evaluation.']
    (REPORT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':final_report()
