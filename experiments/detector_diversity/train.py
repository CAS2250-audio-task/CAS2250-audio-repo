"""Four matched 300-epoch noise-diverse boundary detector ablations."""
import argparse,hashlib,json,math,random,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from experiments.boundary_detector.data import read_manifest,noise_manifest,load_audio,sha
from experiments.boundary_detector.model import Frontend
from experiments.boundary_detector.train import write,load_records,reference_power,validation_cache,evaluate
from experiments.boundary_detector.segments import asymmetric_cost
from .augment import Windows
from .models import make_model
ROOT=Path(__file__).resolve().parents[2]

def region_metrics(details):
    sums=dict(speech=0.,leading=0.,trailing=0.,internal=0.,missed=0.,extra=0.);fractions=[];groups={}
    for r in details:
        c=asymmetric_cost(r['gold'],r['predicted']);speech=sum(b-a for a,b in r['gold'])
        vals=dict(speech=speech,leading=c['leading_missing_seconds'],trailing=c['trailing_missing_seconds'],internal=c['internal_missing_seconds'],missed=c['missed_speech_seconds'],extra=c['extra_seconds'])
        condition='noisy' if r['condition'].startswith('noise') and r['condition']!='noise_only' else r['condition']
        g=groups.setdefault(condition,dict(speech=0.,missing=0.,extra=0.,files=0,fp_files=0))
        for k,v in vals.items():sums[k]+=v
        missing=vals['leading']+vals['trailing']+vals['internal']+vals['missed'];g['speech']+=speech;g['missing']+=missing;g['extra']+=vals['extra'];g['files']+=1;g['fp_files']+=bool(r['predicted'])
        if speech:fractions.append(missing/speech)
    missing=sums['leading']+sums['trailing']+sums['internal']+sums['missed']
    return dict(speech_missing_pct=100*missing/max(sums['speech'],1e-12),speech_missing_median_pct=100*float(np.median(fractions)),extra_ms_per_case=1000*sums['extra']/len(details),region_by_condition={k:dict(missing_pct=100*g['missing']/g['speech'] if g['speech'] else None,extra_ms_per_case=1000*g['extra']/g['files'],noise_fp_rate=g['fp_files']/g['files'] if k=='noise_only' else None) for k,g in groups.items()})

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant',choices=['baseline','lookahead','boundary','spectral','combined'],required=True);p.add_argument('--enabled',nargs='*')
    p.add_argument('--manifest',type=Path,default=ROOT/'runs/detector-diversity-v2/manifest.jsonl');p.add_argument('--noise-manifest',type=Path,default=ROOT/'runs/detector-diversity-v2/noise.json');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=300);p.add_argument('--batch',type=int,default=32);p.add_argument('--seed',type=int,default=42);p.add_argument('--hop',type=int,default=80);p.add_argument('--width',type=int,default=64);p.add_argument('--device',default='cuda');p.add_argument('--resume',action='store_true')
    args=p.parse_args();args.eval_snr=0;args.on=.6;args.off=.4;args.on_ms=20.;args.off_ms=50.
    if args.epochs<1 or args.batch<1:p.error('Invalid training budget')
    if args.enabled is not None and not set(args.enabled)<= {'lookahead','boundary','spectral'}:p.error('Unknown combined change')
    args.output.mkdir(parents=True,exist_ok=True)
    sources={str(p.relative_to(ROOT)):sha(p) for folder in ['experiments/detector_diversity','experiments/boundary_detector'] for p in (ROOT/folder).glob('*.py')}
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k!='resume'}
    config.update(source_sha256=sources,manifest_sha256=sha(args.manifest),noise_manifest_sha256=sha(args.noise_manifest),selection='minimum asymmetric duration cost; leading3/trailing5/internal5/complete5/extra1',augmentation='source-family-balanced-v2, no speaker transforms, onset/offset labels unchanged',training=dict(lr=.001,lr_min=.00001,weight_decay=.0001,window_seconds=1.5,training_extra_future_ms=80,boundary_aux_weight=.2,boundary_sigma_ms=10,boundary_positive_weight=20))
    cfg=args.output/'config.json'
    if cfg.exists():
        if not args.resume or json.loads(cfg.read_text(encoding='utf-8'))!=config:raise ValueError('Existing or mismatched config')
    else:write(cfg,config)
    write(args.output/'progress.json',dict(stage='initializing',epoch=0,epochs=args.epochs))
    torch.set_num_threads(2);torch.manual_seed(args.seed);random.seed(args.seed);np.random.seed(args.seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True,warn_only=True)
    model=make_model(args.variant,args.width,args.hop,args.enabled).to(args.device);frontend=Frontend(args.hop).to(args.device).eval()
    tr=read_manifest(args.manifest,['S03','S04','S05']);va=read_manifest(args.manifest,['S02'])
    assert not {r['audio_sha256'] for r in tr}&{r['audio_sha256'] for r in va}
    records=load_records(tr);validation=load_records(va);ref=reference_power(records)
    paths=noise_manifest(args.noise_manifest);noises={k:[load_audio(p) for p in ps] for k,ps in paths.items()}
    metadata=json.loads(args.noise_manifest.read_text(encoding='utf-8'));families=[r['family'] for r in metadata['train']]
    data=Windows(records,noises['train'],families,ref,args.hop,1.5,args.seed)
    with torch.no_grad():
        total=torch.zeros(1,64,1,device=args.device);square=total.clone();count=0
        for _,wave in records:
            x=frontend(torch.as_tensor(wave,device=args.device)[None]);valid=torch.arange(x.shape[-1],device=args.device)*args.hop<len(wave);x=x[:,:,valid]
            total+=x.sum(2,keepdim=True);square+=(x*x).sum(2,keepdim=True);count+=x.shape[-1]
        mean=total/count;std=(square/count-mean.square()).clamp_min(1e-8).sqrt()
    data.epoch=1;trace=hashlib.sha256()
    for i in range(min(64,len(data))):
        for tensor in data[i]:trace.update(tensor.numpy().tobytes())
    write(args.output/'inputs.json',dict(train_ids=[r['source_id'] for r in tr],validation_ids=[r['source_id'] for r in va],noise_hashes={k:[sha(p) for p in ps] for k,ps in paths.items()},parameters=sum(p.numel() for p in model.parameters()),epoch1_first64_augmented_sha256=trace.hexdigest(),normalization_sha256=hashlib.sha256(mean.cpu().numpy().tobytes()+std.cpu().numpy().tobytes()).hexdigest(),device=torch.cuda.get_device_name() if str(args.device).startswith('cuda') else args.device,lookahead_ms=model.lookahead_frames*args.hop/16,frontend_lookahead_ms=12.5))
    cache=validation_cache(frontend,validation,noises['validation'],ref,mean,std,args)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);history=[];best=float('inf');start=1
    if args.resume:
        s=torch.load(args.output/'last.pt',map_location=args.device,weights_only=False);model.load_state_dict(s['model']);optimizer.load_state_dict(s['optimizer']);history=s['history'];best=s['best'];start=s['epoch']+1
    tick=time.monotonic()
    for epoch in range(start,args.epochs+1):
        model.train();data.epoch=epoch;lr=.00001+(.001-.00001)*(1+math.cos(math.pi*(epoch-1)/max(1,args.epochs-1)))/2
        for group in optimizer.param_groups:group['lr']=lr
        loader=DataLoader(data,batch_size=args.batch,shuffle=True,generator=torch.Generator().manual_seed(args.seed+epoch),num_workers=0)
        losses=[];boundary_losses=[]
        for wave,y,valid,weight,target in loader:
            wave=wave.to(args.device);y=y.to(args.device);valid=valid.to(args.device);weight=weight.to(args.device);target=target.to(args.device)
            with torch.no_grad():x=(frontend(wave)-mean)/std
            logits,boundaries=model.forward_heads(x);logits=logits[:,:y.shape[-1]]
            if boundaries is not None:boundaries=boundaries[:,:,:y.shape[-1]]
            main=(F.binary_cross_entropy_with_logits(logits[valid],y[valid],reduction='none')*weight[valid]).mean();loss=main
            if boundaries is not None:
                # Supervise locations only; keep the SAME speech decoder across all ablations.
                aux=F.binary_cross_entropy_with_logits(boundaries,target,pos_weight=torch.full((1,2,1),20.,device=args.device),reduction='none')
                aux=(aux*torch.tensor([3/4,5/4],device=args.device)[None,:,None])[:, :, :]
                aux=aux.masked_select(valid[:,None].expand_as(aux)).mean();loss=loss+.2*aux;boundary_losses.append(float(aux))
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();losses.append(float(main))
        metrics,details=evaluate(model,cache,args);metrics.update(region_metrics(details))
        row=dict(epoch=epoch,lr=lr,train_weighted_bce=float(np.mean(losses)),train_boundary_loss=float(np.mean(boundary_losses)) if boundary_losses else None,**metrics);history.append(row)
        improved=metrics['asymmetric_cost_ms']<best
        if improved:best=metrics['asymmetric_cost_ms']
        state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),mean=mean,std=std,reference_power=ref,config=config,epoch=epoch,history=history,best=best)
        torch.save(state,args.output/'last.pt')
        if improved:torch.save(state,args.output/'best.pt');write(args.output/'best_predictions.json',details)
        write(args.output/'history.json',history)
        progress=dict(stage='training',variant=args.variant,epoch=epoch,epochs=args.epochs,estimated_training_minutes_remaining=(args.epochs-epoch)*(time.monotonic()-tick)/(epoch-start+1)/60,metrics=row)
        write(args.output/'progress.json',progress);print(json.dumps(progress),flush=True)
    write(args.output/'progress.json',dict(stage='complete',variant=args.variant,epoch=args.epochs,epochs=args.epochs,best_asymmetric_cost_ms=best))

if __name__=='__main__':main()
