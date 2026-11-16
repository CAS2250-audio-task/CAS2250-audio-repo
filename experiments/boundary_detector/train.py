"""Supervised detector training; real data must be provided explicitly."""
import argparse,hashlib,json,math,random,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from .data import SR,sha,read_manifest,load_audio,frame_labels,noise_manifest
from .model import make_detector,Frontend
from .segments import decode,boundary_metrics,asymmetric_cost


def write(path,value):
    p=Path(path);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(p)


def mix(wave,noise,segments,rng,snr,reference_power):
    noise=np.asarray(noise)
    if len(noise)<len(wave):noise=np.tile(noise,math.ceil(len(wave)/len(noise)))
    offset=int(rng.integers(len(noise)-len(wave)+1));noise=noise[offset:offset+len(wave)].astype(np.float64,copy=True);noise-=noise.mean()
    mask=np.zeros(len(wave),bool)
    for a,b in segments:mask[max(0,round(a*SR)):min(len(wave),round(b*SR))]=True
    power=float(np.mean(wave[mask].astype(np.float64)**2)) if mask.any() else reference_power
    npower=float(np.mean(noise[mask]**2)) if mask.any() else float(np.mean(noise**2))
    if power<=0 or npower<1e-12:raise ValueError('Zero speech/reference power or silent noise crop')
    gain=np.sqrt(power/(npower*10**(snr/10)))
    result=wave.astype(np.float64)+gain*noise
    result*=min(1.,.99/max(np.max(np.abs(result)),1e-12))
    return result.astype(np.float32)


def load_records(records):
    result=[]
    for record in records:
        if sha(record['audio_path'])!=record['audio_sha256']:raise ValueError(f'Audio changed after import: {record["source_id"]}')
        wave=load_audio(record['audio_path']);duration=len(wave)/SR
        if any(b>duration+1/SR for a,b in record['segments']):raise ValueError('Annotation exceeds decoded audio')
        result.append((record,wave))
    return result


def reference_power(records):
    powers=[]
    for r,w in records:
        for a,b in r['segments']:
            x=w[round(a*SR):round(b*SR)]
            if len(x) and np.mean(x.astype(np.float64)**2)>0:powers.append(np.mean(x.astype(np.float64)**2))
    if not powers:raise ValueError('Training set needs nonzero positive speech')
    return float(np.median(powers))


class Windows(Dataset):
    def __init__(self,records,noises,reference,hop=160,seconds=1.5,seed=42,noise_probability=.8,snr=(-5,20)):
        self.records=records;self.noises=noises;self.reference=reference;self.hop=hop;self.samples=round(seconds*SR);self.seed=seed;self.prob=noise_probability;self.snr=snr;self.epoch=0
    def __len__(self):return 4*len(self.records)
    def __getitem__(self,index):
        r,original=self.records[index//4];kind=index%4;rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch,index]))
        segments=r['segments'];wave=original
        if kind==3 and self.noises:
            # Real external noise, not zero padding, provides extra negative examples.
            wave=np.zeros(max(len(original),self.samples),np.float32);segments=[]
            wave=mix(wave,self.noises[int(rng.integers(len(self.noises)))],[],rng,float(rng.uniform(*self.snr)),self.reference)
        elif self.noises and rng.random()<self.prob:
            # Mix on the entire original; scale using annotated foreground region.
            wave=mix(original,self.noises[int(rng.integers(len(self.noises)))],segments,rng,float(rng.uniform(*self.snr)),self.reference)
        if segments and kind in [0,1]:
            s=segments[int(rng.integers(len(segments)))];anchor=round(s[kind]*SR)
            start=anchor-int(rng.uniform(.2,.8)*self.samples)
        else:start=int(rng.integers(min(0,len(wave)-self.samples),max(0,len(wave)-self.samples)+1))
        x=np.zeros(self.samples,np.float32);lo=max(0,start);hi=min(len(wave),start+self.samples)
        if hi>lo:x[lo-start:hi-start]=wave[lo:hi]
        gain=10**(float(rng.uniform(-6,3))/20);x*=min(gain,.99/max(np.max(np.abs(x)),1e-12))
        centers=(start+np.arange(self.samples//self.hop+1)*self.hop)/SR
        labels=frame_labels(dict(segments=segments),centers)
        valid=(centers>=0)&(centers<len(wave)/SR)
        weights=np.ones(len(centers),np.float32)
        for a,b in segments:
            weights[(centers>=a)&(centers<min(b,a+.05))]=3.
            weights[(centers>=max(a,b-.05))&(centers<b)]=5.
        return torch.from_numpy(x),torch.from_numpy(labels),torch.from_numpy(valid),torch.from_numpy(weights)


@torch.no_grad()
def validation_cache(frontend,records,noises,reference,mean,std,args):
    cache=[]
    for record,wave in records:
        for n,noise in enumerate([None]+noises):
            key=f'{record["source_id"]}|{n}|{args.seed}'
            rng=np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little'))
            x=wave if noise is None else mix(wave,noise,record['segments'],rng,args.eval_snr,reference)
            mel=(frontend(torch.as_tensor(x,device=args.device)[None])-mean)/std
            cache.append((record,len(x)/SR,'original' if noise is None else f'noise{n-1}',mel[0]))
    # Independent noise-only validation windows: fixed offsets and amplitude,
    # scaled to training median speech power, never to validation speech.
    for n,noise in enumerate(noises):
        for j in range(10):
            rng=np.random.default_rng(args.seed+10000+n*100+j)
            x=mix(np.zeros(round(1.5*SR),np.float32),noise,[],rng,args.eval_snr,reference)
            record=dict(source_id=f'negative_{n}_{j}',speaker='external',segments=[],annotation_source='external_noise')
            mel=(frontend(torch.as_tensor(x,device=args.device)[None])-mean)/std
            cache.append((record,len(x)/SR,'noise_only',mel[0]))
    return cache


@torch.no_grad()
def evaluate(model,cache,args):
    model.eval();tot_loss=0.;frames=0;tp=fp=fn=0;details=[];boundary=[];condition_totals={};costs=[]
    for offset in range(0,len(cache),args.batch):
        batch=cache[offset:offset+args.batch]
        length=max(m.shape[-1] for _,_,_,m in batch)
        x=torch.stack([F.pad(m,(0,length-m.shape[-1])) for _,_,_,m in batch])
        outputs=model(x)
        for (record,duration,condition,mel),output in zip(batch,outputs):
            logits=output[:mel.shape[-1]]
            centers=np.arange(len(logits))*args.hop/SR;valid=centers<duration
            y=torch.tensor(frame_labels(record,centers),device=args.device);mask=torch.tensor(valid,device=args.device)
            tot_loss+=float(F.binary_cross_entropy_with_logits(logits[mask],y[mask],reduction='sum'));frames+=int(mask.sum())
            probabilities=logits.sigmoid().cpu().numpy();prediction=probabilities[valid]>=.5;truth=y.cpu().numpy()[valid]>0
            tp+=int((prediction&truth).sum());fp+=int((prediction&~truth).sum());fn+=int((~prediction&truth).sum())
            intervals=decode(probabilities,args.hop/SR,duration,args.on,args.off,args.on_ms,args.off_ms)
            b=boundary_metrics(record['segments'],intervals);boundary.append(b)
            costs.append(asymmetric_cost(record['segments'],intervals))
            key='noisy' if condition.startswith('noise') and condition!='noise_only' else condition
            c=condition_totals.setdefault(key,dict(tp=0,fp=0,fn=0,gold=0,pred=0,matched=0))
            for name,value in dict(tp=int((prediction&truth).sum()),fp=int((prediction&~truth).sum()),fn=int((~prediction&truth).sum()),gold=b['gold'],pred=b['pred'],matched=b['matched']).items():c[name]+=value
            details.append(dict(source_id=record['source_id'],speaker=record['speaker'],condition=condition,duration_seconds=duration,gold=record['segments'],predicted=intervals,annotation_source=record['annotation_source']))
    matched=sum(b['matched'] for b in boundary);ng=sum(b['gold'] for b in boundary);npred=sum(b['pred'] for b in boundary)
    starts=[x for b in boundary for x in b['start_errors']];ends=[x for b in boundary for x in b['end_errors']]
    metrics=dict(bce=tot_loss/max(frames,1),frame_f1=2*tp/max(2*tp+fp+fn,1),event_f1=2*matched/max(ng+npred,1),matched_events=matched,missed_events=ng-matched,false_positive_events=npred-matched,
        start_mae_ms=1000*float(np.mean(starts)) if starts else None,end_mae_ms=1000*float(np.mean(ends)) if ends else None,
        start_p95_ms=1000*float(np.percentile(starts,95)) if starts else None,end_p95_ms=1000*float(np.percentile(ends,95)) if ends else None,
        start_cut_events_over10ms=sum(b['start_cuts'] for b in boundary),end_cut_events_over10ms=sum(b['end_cuts'] for b in boundary))
    metrics['by_condition']={k:dict(frame_f1=2*c['tp']/max(2*c['tp']+c['fp']+c['fn'],1),event_f1=2*c['matched']/max(c['gold']+c['pred'],1),missed_events=c['gold']-c['matched'],false_positive_events=c['pred']-c['matched']) for k,c in condition_totals.items()}
    metrics['asymmetric_cost_ms']=1000*sum(x['cost_seconds'] for x in costs)/len(costs)
    metrics['cost_components_ms']={k:1000*sum(x[k] for x in costs)/len(costs) for k in costs[0] if k!='cost_seconds'}
    return metrics,details


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--train-speakers',nargs='+',required=True);p.add_argument('--val-speakers',nargs='+',required=True)
    p.add_argument('--noise-manifest',type=Path);p.add_argument('--allow-noise-free',action='store_true');p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true')
    p.add_argument('--architecture',choices=['tcn','cnn_gru'],default='tcn')
    p.add_argument('--width',type=int,choices=[32,64],default=64);p.add_argument('--hop',type=int,choices=[80,160],default=160);p.add_argument('--window-seconds',type=float,default=1.5)
    p.add_argument('--epochs',type=int,default=100);p.add_argument('--batch',type=int,default=32);p.add_argument('--lr',type=float,default=.001);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--noise-probability',type=float,default=.8);p.add_argument('--snr-min',type=float,default=-5);p.add_argument('--snr-max',type=float,default=20);p.add_argument('--eval-snr',type=float,default=0)
    p.add_argument('--on',type=float,default=.6);p.add_argument('--off',type=float,default=.4);p.add_argument('--on-ms',type=float,default=20);p.add_argument('--off-ms',type=float,default=50);p.add_argument('--device',default='cuda')
    args=p.parse_args()
    if set(args.train_speakers)&set(args.val_speakers):p.error('Train/validation speakers overlap')
    if not args.noise_manifest and not args.allow_noise_free:p.error('Provide noise-manifest; --allow-noise-free is for explicit controls/smoke checks')
    if args.epochs<1 or args.batch<1 or args.window_seconds<=0 or args.lr<=0 or not 0<=args.noise_probability<=1 or args.snr_min>args.snr_max:p.error('Invalid training settings')
    if not 0<=args.off<=args.on<=1 or min(args.on_ms,args.off_ms)<0:p.error('Invalid boundary postprocessing')
    train=read_manifest(args.manifest,args.train_speakers);val=read_manifest(args.manifest,args.val_speakers)
    if {r['audio_sha256'] for r in train}&{r['audio_sha256'] for r in val}:p.error('Original audio hashes overlap across folds')
    args.output.mkdir(parents=True,exist_ok=True)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k!='resume'}
    config.update(manifest_sha256=sha(args.manifest),noise_manifest_sha256=sha(args.noise_manifest) if args.noise_manifest else None,
        source_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},label_note='Annotations may be automatic/weak; boundary accuracy requires audited reference labels',selection='minimum mean asymmetric duration cost across original, fixed noisy and noise-only validation',crop_margin_ms=0,boundary_weights=dict(start=3,end=5,extra=1,training_boundary_ms=50))
    cfg=args.output/'config.json'
    if cfg.exists():
        if not args.resume:raise FileExistsError('Existing run; use --resume with identical config or a new output')
        if json.loads(cfg.read_text(encoding='utf-8'))!=config:raise ValueError('Run configuration/source/data changed')
    else:write(cfg,config)
    torch.set_num_threads(2);torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=True)
    frontend=Frontend(args.hop).to(args.device).eval();model=make_detector(args.architecture,args.width).to(args.device)
    records=load_records(train);validation=load_records(val);reference=reference_power(records)
    noises=noise_manifest(args.noise_manifest) if args.noise_manifest else {'train':[],'validation':[]}
    train_noises=[load_audio(x) for x in noises['train']];val_noises=[load_audio(x) for x in noises['validation']]
    if any(not len(x) or np.mean(x.astype(np.float64)**2)<1e-12 for x in train_noises+val_noises):raise ValueError('Empty/silent noise file')
    write(args.output/'inputs.json',dict(train_ids=[r['source_id'] for r in train],validation_ids=[r['source_id'] for r in val],noise_hashes={k:{str(p):sha(p) for p in v} for k,v in noises.items()},torch=str(torch.__version__),device=args.device,parameters=sum(x.numel() for x in model.parameters())))
    with torch.no_grad():
        total=torch.zeros(1,64,1,device=args.device);square=total.clone();count=0
        for _,wave in records:
            x=frontend(torch.as_tensor(wave,device=args.device)[None]);valid=torch.arange(x.shape[-1],device=args.device)*args.hop<len(wave);x=x[:,:,valid]
            total+=x.sum(2,keepdim=True);square+=(x*x).sum(2,keepdim=True);count+=x.shape[-1]
        mean=total/count;std=(square/count-mean.square()).clamp_min(1e-8).sqrt()
    cache=validation_cache(frontend,validation,val_noises,reference,mean,std,args)
    data=Windows(records,train_noises,reference,args.hop,args.window_seconds,args.seed,args.noise_probability,(args.snr_min,args.snr_max))
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4);history=[];best=float('inf');start=1
    checkpoint=args.output/'last.pt'
    if args.resume:
        state=torch.load(checkpoint,map_location=args.device,weights_only=False)
        if state['config']!=config:raise ValueError('Checkpoint protocol mismatch')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);mean=state['mean'];std=state['std'];history=state['history'];best=state['best'];start=state['epoch']+1
    tick=time.monotonic()
    for epoch in range(start,args.epochs+1):
        model.train();data.epoch=epoch
        loader=DataLoader(data,batch_size=args.batch,shuffle=True,generator=torch.Generator().manual_seed(args.seed+epoch),num_workers=0)
        lr=1e-5+(args.lr-1e-5)*(1+math.cos(math.pi*(epoch-1)/max(1,args.epochs-1)))/2
        for group in optimizer.param_groups:group['lr']=lr
        loss_sum=frames=0
        for wave,y,valid,weights in loader:
            wave=wave.to(args.device);y=y.to(args.device);valid=valid.to(args.device)
            with torch.no_grad():x=(frontend(wave)-mean)/std
            logits=model(x);weights=weights.to(args.device);loss=(F.binary_cross_entropy_with_logits(logits[valid],y[valid],reduction="none")*weights[valid]).mean()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            n=int(valid.sum());loss_sum+=float(loss)*n;frames+=n
        metrics,details=evaluate(model,cache,args)
        row=dict(epoch=epoch,lr=lr,train_weighted_bce=loss_sum/frames,**metrics);history.append(row)
        improved=metrics['asymmetric_cost_ms']<best
        if improved:best=metrics['asymmetric_cost_ms']
        state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),mean=mean,std=std,config=config,epoch=epoch,history=history,best=best)
        torch.save(state,args.output/'last.pt')
        if improved:
            torch.save(state,args.output/'best.pt');write(args.output/'best_predictions.json',details)
        write(args.output/'history.json',history)
        progress=dict(stage='training',epoch=epoch,epochs=args.epochs,estimated_training_minutes_remaining=(args.epochs-epoch)*(time.monotonic()-tick)/(epoch-start+1)/60,metrics=row)
        write(args.output/'progress.json',progress);print(json.dumps(progress),flush=True)
    write(args.output/'progress.json',dict(stage='complete',epoch=args.epochs,epochs=args.epochs,best_validation_asymmetric_cost_ms=best))


if __name__=='__main__':main()
