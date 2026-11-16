"""Isolated label-smoothing ablation; unchanged feature bank and input contract."""
import argparse,collections,copy,json,math,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from experiments.compact_pooling.objectives import supervised_contrastive
from experiments.frequency_local_subset import canonical_projectors
from .models import make_model,HEADS,NORMALIZATION
from .features import collate
from .contract import sha,write_json

ROOT=Path(__file__).resolve().parents[1]
SPEAKER_IDENTITIES={
 'S01':dict(name='강재훈',legacy_id='spk01'),
 'S02':dict(name='노정민',legacy_id='spk03'),
 'S03':dict(name='곽민채',legacy_id='spk02'),
 'S04':dict(name='최환준',legacy_id='spk05'),
 'S05':dict(name='이서준',legacy_id='spk04')}
CONDITIONS={
 'base':('basic','standard','pooled',False),
 'axis':('axis','standard','pooled',False),
 'band2':('frequency','standard','pooled',False),
 'band2_attn':('frequency_attn','standard','pooled',False),
 'band2_attn_low':('frequency_attn','lowfocus','pooled',False),
 'base_aux':('basic','standard','pooled',True),
 'mean':('basic','standard','mean',False),
 'mean_aux':('basic','standard','mean',True)}

def metrics(true,pred):
 cm=np.bincount(np.asarray(true)*11+np.asarray(pred),minlength=121).reshape(11,11)
 den=cm.sum(0)+cm.sum(1);f=np.divide(2*cm.diagonal(),den,out=np.zeros(11),where=den>0)
 return dict(command10_f1=float(f[:10].mean()),macro11_f1=float(f.mean()),silence_f1=float(f[10]),
             accuracy=float(np.trace(cm)/max(1,cm.sum())),confusion=cm.tolist())

def smoothing_nll(probabilities, labels, epsilon=0.):
 """CE with11-class uniform smoothing; preserve the legacy probability clamp."""
 logprob=probabilities.clamp_min(1e-8).log()
 nll=F.nll_loss(logprob,labels)
 return (1.-epsilon)*nll-epsilon*logprob.mean()

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--condition',choices=CONDITIONS,required=True);p.add_argument('--bank',type=Path,default=ROOT/'bank')
 p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=200)
 p.add_argument('--schedule-epochs',type=int,default=1000);p.add_argument('--batch',type=int,default=32)
 p.add_argument('--seed',type=int,default=42);p.add_argument('--device',default='cuda')
 p.add_argument('--stop-after',type=int,help='Runtime stop budget; fixed protocol can resume to later stages')
 p.add_argument('--profile',action='store_true');p.add_argument('--resume',action='store_true')
 p.add_argument('--label-smoothing',type=float,default=0.,help='Uniform smoothing over all11 classes;0 preserves legacy CE')
 p.add_argument('--input-mode',choices=['actual','off'],default='actual',help='Explicit crop or native full-source bank contract')
 args=p.parse_args();assert 1<=args.epochs<=args.schedule_epochs
 if not math.isfinite(args.label_smoothing) or not 0<=args.label_smoothing<=1:p.error('--label-smoothing must be finite and between0 and1')
 run_until=args.stop_after or args.epochs;assert 1<=run_until<=args.epochs
 torch.set_num_threads(2);torch.manual_seed(args.seed);np.random.seed(args.seed)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 assert json.loads((args.bank/'COMPLETE.json').read_text())['stage']=='complete'
 meta=json.loads((args.bank/'metadata.json').read_text());rows=meta['rows'];pairs=meta['pairs']
 if args.input_mode=='actual':
  assert meta['margin']==[50,30] and meta['selection_policy']=='first_start_last_end' and meta['no_detection_policy']=='force_silence'
 else:
  assert meta['margin']==[0,0] and meta['selection_policy']=='full_source_no_detector_no_gold' and meta['no_detection_policy']=='classifier_decides_all_inputs'
 arch,mode,head,aux=CONDITIONS[args.condition]
 encoded=torch.load(args.bank/(mode+'.pt'),map_location='cpu',weights_only=False)
 aux_scaler=json.loads((args.bank/'auxiliary_scaler.json').read_text()) if aux else None
 if aux:
  raw=torch.load(args.bank/'auxiliary.pt',map_location='cpu',weights_only=False)
  encoded=[None if x is None else torch.cat([x,torch.as_tensor(a)]) for x,a in zip(encoded,raw)]
 model=make_model(arch,head,aux,aux_scaler,args.seed).to(args.device)
 projector=copy.deepcopy(canonical_projectors()[HEADS[head]]).to(args.device)
 initial=torch.load(args.bank/'initialization.pt',map_location='cpu',weights_only=False)
 model.load_state_dict(initial['models'][args.condition],strict=True)
 projector.load_state_dict(initial['projectors'][HEADS[head]],strict=True)
 params=list(model.parameters())+list(projector.parameters())
 optimizer=torch.optim.AdamW(params,lr=.001,weight_decay=.0001)
 val=[i for i,r in enumerate(rows) if r['split']=='validation']
 def forward(indices):
  valid=[j for j,i in enumerate(indices) if encoded[i] is not None]
  # User decision: no detector output => silence, not deleted or zero-padded crop.
  probabilities=torch.zeros(len(indices),11,device=args.device);probabilities[:,10]=1.
  embedding=torch.zeros(len(indices),32,device=args.device)
  if valid:
   selected=[encoded[indices[j]] for j in valid]
   x,lengths,_=collate(selected,args.device);z=model.embedding(x,lengths)
   probs=model.classifier(z).softmax(1)
   index=torch.tensor(valid,device=args.device)
   probabilities=probabilities.index_copy(0,index,probs);embedding=embedding.index_copy(0,index,z)
  return probabilities,embedding
 def train_batch(batch):
  indices=[pairs[i][0] for i in batch]+[pairs[i][1] for i in batch]
  y=torch.tensor([rows[i]['label'] for i in indices],device=args.device)
  prob,z=forward(indices)
  ce=smoothing_nll(prob,y,args.label_smoothing)
  good=[encoded[pairs[i][0]] is not None and encoded[pairs[i][1]] is not None for i in batch]
  valid=torch.tensor(good+good,device=args.device)
  contrast=supervised_contrastive(projector(z[valid]),y[valid],.1,'same_command')
  loss=ce+.1*contrast
  if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
  optimizer.zero_grad(set_to_none=True)
  if loss.requires_grad:
   loss.backward();torch.nn.utils.clip_grad_norm_(params,5.);optimizer.step()
  return float(loss.detach()),int(prob.argmax(1).eq(y).sum())
 def sync():
  if args.device.startswith('cuda'):torch.cuda.synchronize()
 @torch.inference_mode()
 def evaluate():
  model.eval();pred=[]
  for offset in range(0,len(val),64):pred.extend(forward(val[offset:offset+64])[0].argmax(1).cpu().tolist())
  results=[]
  for i,v in zip(val,pred):results.append(dict(index=i,source_id=rows[i]['source_id'],view=rows[i]['view'],noise_id=rows[i].get('noise_id'),truth=rows[i]['label'],prediction=v,forced_silence=encoded[i] is None))
  units={}
  for r in results:
   if r['view']=='noisy':units.setdefault(r['noise_id'],[]).append(r)
  by_noise={k:metrics([r['truth'] for r in group],[r['prediction'] for r in group]) for k,group in units.items()}
  original=[r for r in results if r['view']=='reference'];negative=[r for r in results if r['view']=='noise_only']
  expected=collections.Counter(rows[i]['view'] for i in val)
  assert len(original)==expected['reference'] and len(negative)==expected['noise_only']
  assert sum(len(v) for v in units.values())==expected['noisy']
  speakers=sorted({rows[i]['speaker'] for i in val if rows[i]['label']!=10})
  if meta.get('validation_speakers'):assert set(speakers)==set(meta['validation_speakers'])
  by_speaker={}
  for speaker in speakers:
   selected=[r for r in results if rows[r['index']]['speaker']==speaker]
   reference=[r for r in selected if r['view']=='reference']
   noisy=[r for r in selected if r['view']=='noisy']
   noise_groups={k:[r for r in noisy if r['noise_id']==k] for k in sorted({r['noise_id'] for r in noisy})}
   per_noise={k:metrics([r['truth'] for r in group],[r['prediction'] for r in group]) for k,group in noise_groups.items()}
   by_speaker[speaker]=dict(original=metrics([r['truth'] for r in reference],[r['prediction'] for r in reference]),
    zero_command10_f1=float(np.mean([r['command10_f1'] for r in per_noise.values()])) if per_noise else None,
    by_noise=per_noise,reference_files=len(reference),noisy_files=len(noisy))
  score=dict(zero_command10_f1=float(np.mean([r['command10_f1'] for r in by_noise.values()])),
   noise_only_fpr=float(np.mean([r['prediction']!=10 for r in negative])) if negative else None,
   by_speaker=by_speaker,validation_counts=dict(expected),
   original=metrics([r['truth'] for r in original],[r['prediction'] for r in original]),by_noise=by_noise)
  return score,results
 if args.profile:
  args.output.mkdir(parents=True,exist_ok=False);model.train();projector.train();times=[]
  for i in range(5):
   sync();tick=time.perf_counter();train_batch(list(range(i*args.batch,(i+1)*args.batch)));sync()
   if i>=2:times.append(time.perf_counter()-tick)
  tick=time.perf_counter();evaluate();sync();seconds=time.perf_counter()-tick
  value=dict(condition=args.condition,median_train_step_seconds=float(np.median(times)),validation_seconds=seconds,
    pair_batches=math.ceil(len(pairs)/args.batch),estimated_epoch_seconds=float(np.median(times))*math.ceil(len(pairs)/args.batch)+seconds,
    parameters=sum(p.numel() for p in model.parameters()),input_bank_sha256=sha(args.bank/'metadata.json'),device=args.device,
    profile_only=True,optimizer_updates_discarded=True)
  write_json(args.output/'profile.json',value);print(json.dumps(value),flush=True);return
 critical=list((ROOT/'experiments').rglob('*.py'))+[Path(__file__),ROOT/'study/models.py',ROOT/'study/features.py',ROOT/'study/frontend.py',ROOT/'study/masked.py']
 protocol=dict(condition=args.condition,seed=args.seed,epochs=args.epochs,schedule_epochs=args.schedule_epochs,
  batch_pairs=args.batch,objective='paired uniform-label-smoothed CE +0.1 same_command SupCon;temp0.1;silence CE only',
  label_smoothing=args.label_smoothing,smoothing_classes=11,feature_policy='unchanged input bank; no frontend or detector changes',
  input_mode=args.input_mode,
  speaker_id_namespace='D3 manual-clipped-v1 Sxx; differs from legacy spkXX',
  speaker_identities={sid:SPEAKER_IDENTITIES.get(sid,dict(name='unmapped',legacy_id=None))
   for sid in sorted(set(meta.get('train_speakers',[])+meta.get('validation_speakers',[])))},
  bank_counts=dict(train_pairs=len(pairs),train_rows=sum(r['split']=='train' for r in rows),
   validation_rows=len(val),validation_by_view=dict(collections.Counter(rows[i]['view'] for i in val)),
   train_speakers=meta.get('train_speakers'),validation_speakers=meta.get('validation_speakers')),
  optimizer='AdamW1e-3 cosine1e-5 over1000;WD1e-4;clip5',selection='mean command10 F1 on fixed validation8noise0dB draw0',
  bank_sha256=sha(args.bank/'metadata.json'),feature_sha256=sha(args.bank/(mode+'.pt')),
  aux_sha256=sha(args.bank/'auxiliary.pt') if aux else None,normalization=NORMALIZATION,
  initialization_sha256=sha(args.bank/'initialization.pt'),
  detector=meta['detector'],data_version=meta['data_version'],margin=meta['margin'],
  selection_policy=meta['selection_policy'],no_detection_policy=meta['no_detection_policy'],
  source_sha256={str(f.relative_to(ROOT)):sha(f) for f in critical},limitations=meta['limitations'])
 args.output.mkdir(parents=True,exist_ok=args.resume);protocol_path=args.output/'protocol.json'
 history=[];best=-1.;start=1
 if args.resume:
  assert json.loads(protocol_path.read_text())==protocol,'Resume inputs/code/protocol changed'
  s=torch.load(args.output/'last.pt',map_location=args.device,weights_only=False)
  model.load_state_dict(s['model']);projector.load_state_dict(s['projector']);optimizer.load_state_dict(s['optimizer'])
  history=s['history'];best=s['best'];start=s['epoch']+1
 else:write_json(protocol_path,protocol)
 for epoch in range(start,run_until+1):
  sync();tick=time.perf_counter();model.train();projector.train()
  lr=1e-5+(.001-1e-5)*(1+math.cos(math.pi*(epoch-1)/max(1,args.schedule_epochs-1)))/2
  for g in optimizer.param_groups:g['lr']=lr
  order=np.random.default_rng(np.random.SeedSequence([args.seed,epoch])).permutation(len(pairs));total=0.;correct=0
  for offset in range(0,len(order),args.batch):
   batch=order[offset:offset+args.batch].tolist();loss,right=train_batch(batch);total+=loss*len(batch);correct+=right
  scores,predictions=evaluate();sync();elapsed=time.perf_counter()-tick
  improved=scores['zero_command10_f1']>best
  if improved:best=scores['zero_command10_f1']
  row=dict(epoch=epoch,lr=lr,train_loss=total/len(pairs),train_accuracy=correct/(2*len(pairs)),epoch_seconds=elapsed,**scores)
  history.append(row)
  state=dict(epoch=epoch,model=model.state_dict(),projector=projector.state_dict(),optimizer=optimizer.state_dict(),
   history=history,best=best,protocol_sha256=sha(protocol_path),mel_scaler=json.loads((args.bank/(mode+'_scaler.json')).read_text()),aux_scaler=aux_scaler)
  tmp=args.output/'last.partial';torch.save(state,tmp);tmp.replace(args.output/'last.pt')
  if improved:torch.save(state,args.output/'best.pt');write_json(args.output/'best_predictions.json',predictions)
  write_json(args.output/'history.json',history)
  status=dict(stage='complete' if epoch==run_until else 'training',condition=args.condition,epoch=epoch,epochs=run_until,planned_epochs=args.epochs,
   epoch_seconds=elapsed,best_zero_command10_f1=best,parameters=sum(p.numel() for p in model.parameters()),
   eta_seconds=(run_until-epoch)*float(np.median([r['epoch_seconds'] for r in history[-5:]])))
  write_json(args.output/'status.json',status);print(json.dumps(status),flush=True)

if __name__=='__main__':main()
