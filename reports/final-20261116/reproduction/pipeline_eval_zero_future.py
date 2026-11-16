"""Same classifier + exact same D3 waves: OFF / oracle / actual crop."""
import argparse,json,time,math
from pathlib import Path
import os,sys
CLASSIFIER_WORK=Path(os.environ.get("CLASSIFIER_WORK",Path.cwd())).resolve()
sys.path.insert(0,str(CLASSIFIER_WORK))
import numpy as np
import torch
from study.models import make_model
from study.features import Features,extract_aux,collate
from study.train import metrics
from study.contract import write_json
from study.vendor.detector_model import Frontend as DetectorFrontend
from study.vendor.diversity_models import make_model as detector_model
from study.vendor.inference import infer_logits
from study.vendor.segments import decode
from final_shared import reconstruct,crop_wave,file_sha,digest_wave

ROOT=CLASSIFIER_WORK


class AuxiliaryDetector(torch.nn.Module):
    """Exact detector_frontend wrapper, retaining checkpoint normalization buffers."""
    def __init__(self,width=64,hop=80):
        super().__init__()
        self.detector=detector_model('lookahead',width,hop)
        self.lookahead_frames=self.detector.lookahead_frames
        self.auxiliary=True
        self.projection=torch.nn.Conv1d(3,64,1,bias=False)
        torch.nn.init.zeros_(self.projection.weight)
        self.register_buffer('mean',torch.zeros(1,64,1))
        self.register_buffer('std',torch.ones(1,64,1))
        lo=2595*math.log10(1+20/700);hi=2595*math.log10(1+7800/700)
        hz=700*(10**(torch.linspace(lo,hi,66)[1:-1]/2595)-1)
        self.register_buffer('high_band',hz>=3000)
    def descriptors(self,x):
        delta=torch.cat([torch.zeros_like(x[:,:,:1]),x[:,:,1:]-x[:,:,:-1]],dim=2)
        flux=delta.relu().mean(dim=1)
        high_flux=delta[:,self.high_band].relu().mean(dim=1)
        power_fraction=torch.softmax((x*self.std+self.mean)*math.log(10),dim=1)
        high_ratio=power_fraction[:,self.high_band].sum(dim=1)
        return torch.stack([flux,high_flux,high_ratio],dim=1)
    def forward_heads(self,x):
        return self.detector.forward_heads(x+self.projection(self.descriptors(x)))
    def forward(self,x):return self.forward_heads(x)[0]


class CropDetector:
    def __init__(self,path):
        s=torch.load(path,map_location='cuda',weights_only=False);c=s['config']
        assert c['hop']==80 and c['variant'] in ['lookahead','baseline']
        state=s['model']
        if c.get('auxiliary',False):
            self.model=AuxiliaryDetector(c.get('width',64),80).cuda().eval()
            self.model.load_state_dict(state,strict=True)
            torch.testing.assert_close(self.model.mean,s['mean'],rtol=0,atol=0)
            torch.testing.assert_close(self.model.std,s['std'],rtol=0,atol=0)
        else:
            if any(k.startswith('detector.') for k in state):state={k[len('detector.'):]:v for k,v in state.items() if k.startswith('detector.')}
            self.model=detector_model(c['variant'],c.get('width',64),80).cuda().eval();self.model.load_state_dict(state,strict=True)
        self.frontend=DetectorFrontend(80).cuda().eval();self.mean=s['mean'];self.std=s['std']
        self.sha=file_sha(path)
    @torch.inference_mode()
    def crop(self,wave):
        x=(self.frontend(torch.as_tensor(wave,device='cuda')[None])-self.mean)/self.std
        prob=infer_logits(self.model,x,[x.shape[-1]],policy='hold').sigmoid()[0].cpu().numpy()
        segments=decode(prob,.005,len(wave)/16000,.6,.4,20,50)
        if not segments:return None
        return wave[max(0,round((segments[0][0]-.05)*16000)):min(len(wave),round((segments[-1][1]+.03)*16000))].copy()


@torch.inference_mode()
def evaluate(checkpoint,condition,raw_root,output,detector_checkpoint=None,verify=False,actual_only=False):
    torch.set_num_threads(2);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    s=torch.load(checkpoint,map_location='cuda',weights_only=False)
    aux=condition=='mean_aux';model=make_model('basic','mean',aux,s.get('aux_scaler'),42).cuda().eval();model.load_state_dict(s['model'])
    front=Features('standard');scaler=s['mel_scaler']
    metadata=json.loads((ROOT/'bank/metadata.json').read_text());rows,waves,sources=reconstruct(raw_root,metadata,True)
    assert len(rows)==1880
    detector=CropDetector(detector_checkpoint) if detector_checkpoint else None
    timing_detector=detector or CropDetector(ROOT/'reference_detector.pt')
    results=[]
    for mode in (['actual'] if actual_only else ['off','oracle','actual']):
        tic=time.monotonic();crops=[];encoded=[];pred=[]
        for r,w in zip(rows,waves):
            crop=detector.crop(w) if mode=='actual' and detector else crop_wave(w,r,mode,sources)
            crops.append(crop)
            encoded.append(None if crop is None else front.encode(crop,scaler,auxiliary=aux))
        for start in range(0,len(rows),64):
            items=encoded[start:start+64];valid=[i for i,x in enumerate(items) if x is not None]
            outputs=[10]*len(items)
            if valid:
                x,lengths,_=collate([items[i] for i in valid],'cuda')
                guesses=model(x,lengths).argmax(1).cpu().tolist()
                for i,y in zip(valid,guesses):outputs[i]=y
            pred+=outputs
        by_noise={}
        for i in range(8):
            indices=[j for j,r in enumerate(rows) if r['view']=='noisy' and r['noise_id']==f'noise{i}']
            assert len(indices)==200
            by_noise[f'noise{i}']=metrics([rows[j]['label'] for j in indices],[pred[j] for j in indices])
        original=[j for j,r in enumerate(rows) if r['view']=='reference'];negative=[j for j,r in enumerate(rows) if r['view']=='noise_only']
        record=dict(mode=mode,condition=condition,classifier_checkpoint_sha256=file_sha(checkpoint),
            classifier_epoch=s['epoch'],detector_checkpoint_sha256=detector.sha if detector and mode=='actual' else metadata['detector_checkpoint_sha256'] if mode=='actual' else None,
            zero_command10_f1=float(np.mean([v['command10_f1'] for v in by_noise.values()])),
            original=metrics([rows[j]['label'] for j in original],[pred[j] for j in original]),
            noise_fp_files=sum(pred[j]!=10 for j in negative),forced_silence_files=sum(x is None for x in crops),
            same_full_waveforms_verified=1880,scaler_refit=False,by_noise=by_noise,
            note='Oracle negative gate is defined from labels, not learned rejection; oracle never a deployable candidate')
        if verify and mode=='actual' and detector is None:
            target=ROOT/('frozen200/mean_aux/best_predictions.json' if aux else 'frozen200/mean/best_predictions.json')
            old=json.loads(target.read_text());assert len(old)==1880
            assert pred==[r['prediction'] for r in old], 'Original classifier/crop replay mismatch'
        # Resident CPU->features->H2D->classifier->D2H latency on30 distributed inputs.
        times=[];gpu_times=[];parameter_bytes=sum(x.numel()*x.element_size() for x in model.state_dict().values())
        for j in np.linspace(0,len(rows)-1,30,dtype=int):
            wave=waves[j];row=rows[j]
            torch.cuda.synchronize();begin=time.perf_counter()
            crop=timing_detector.crop(wave) if mode=='actual' else crop_wave(wave,row,mode,sources)
            if crop is not None:
                x=front.encode(crop,scaler,auxiliary=aux);batch,lengths,_=collate([x],'cuda')
                a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
                a.record();guess=model(batch,lengths).argmax(1);b.record();guess.cpu();b.synchronize();gpu_times.append(a.elapsed_time(b))
            torch.cuda.synchronize();times.append((time.perf_counter()-begin)*1000)
        record['latency']=dict(resident_pipeline_p50_ms=float(np.median(times)),resident_pipeline_p95_ms=float(np.percentile(times,95)),
            classifier_gpu_p50_ms=float(np.median(gpu_times)) if gpu_times else None,
            includes='native waveform crop/gate, CPU mel, Burg if auxiliary, H2D, classifier, D2H; actual mode always includes the runtime GPU detector, including the frozen reference detector',
            excludes='file IO, model loading, audio collection waiting; oracle annotation cost not measured',
            actual_detector_executed=mode=='actual',accuracy_detection_source='runtime model' if detector else 'verified original stored crop contract',latency_detection_source='runtime GPU model' if mode=='actual' else 'none',
            lookahead_ms=getattr(timing_detector.model,'lookahead_frames',0)*5 if mode=='actual' else 0,parameters=sum(p.numel() for p in model.parameters()),state_tensor_bytes=parameter_bytes,
            device=torch.cuda.get_device_name(),samples=30,oracle_diagnostic_only=mode=='oracle')
        record['evaluation_seconds']=time.monotonic()-tic;results.append(record)
        write_json(out/f'{condition}_{mode}_predictions.json',[dict(source_id=r['source_id'],view=r['view'],noise_id=r.get('noise_id'),truth=r['label'],prediction=p) for r,p in zip(rows,pred)])
        write_json(out/f'{condition}_threeway.json',results)
        print('PIPELINE_RESULT',json.dumps({k:v for k,v in record.items() if k not in ['by_noise','original']},ensure_ascii=False),flush=True)
    return results


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--condition',choices=['mean','mean_aux'],required=True)
    p.add_argument('--raw-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--detector-checkpoint',type=Path);p.add_argument('--verify-original',action='store_true');p.add_argument('--actual-only',action='store_true');a=p.parse_args()
    evaluate(a.checkpoint,a.condition,a.raw_root,a.output,a.detector_checkpoint,a.verify_original,a.actual_only)
