"""Continue all three spk03 classifiers from epoch100 LAST to total epoch1000."""
import argparse
import fcntl
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.nn import functional as F

from experiments.compact_pooling import run as b
from experiments.compact_pooling.models import matched_models

ROOT=b.ROOT
PARENT=ROOT/'runs/compact-pooling-v1/spk03'
PARENT_REPORT=ROOT/'reports/compact-pooling-v1'
NAMES=b.NAMES


def final_evaluation(models,units,frame,waves,targets,val_idx,noises,frontend,mean,std,reference,folder,protocol):
    rows=[]
    baseline=json.loads((PARENT/'summary.json').read_text())
    for name,model in models.items():
        last,last_details=b.evaluate(model,units)
        state=torch.load(folder/(name+'_best.pt'),map_location='cuda',weights_only=True)
        model.load_state_dict(state['model'])
        selected,details=b.evaluate(model,units)
        predictions=[]
        for unit in units:
            pred=b.predict(model,unit['x']).argmax(1).cpu().tolist()
            for index,value in zip(val_idx,pred):
                predictions.append(dict(path=frame.iloc[index].path,true_idx=int(targets[index]),pred_idx=value,
                    kind=unit['kind'],noise_id=unit['noise_id'],draw=unit['draw']))
        pd.DataFrame(predictions).to_csv(folder/(name+'_best_predictions.csv.gz'),index=False)
        false_positives=0;count=0
        for noise_id,recording in noises:
            noise_waves=[]
            for draw in range(3):
                for i in range(20):
                    noise,_=b.crop_noise(recording,f'noise_only_{i}',noise_id,draw)
                    noise_waves.append(b.mix_at_snr(np.zeros(24000,np.float32),noise,0.,reference)[0])
            noise_x=b.features(frontend,np.stack(noise_waves),mean,std)
            false_positives+=int(b.predict(model,noise_x).argmax(1).ne(10).sum());count+=len(noise_x)
        old=next(r for r in baseline if r['model']==name)
        row=dict(speaker='spk03',model=name,parameters=sum(p.numel() for p in model.parameters()),
            best_epoch=state['epoch'],best=selected,last=last,noise_only_fpr=false_positives/count,
            baseline_100=old,delta_best_zero_f1=selected['zero_f1']-old['best']['zero_f1'])
        b.write(folder/(name+'_final_metrics.json'),dict(summary=row,best_units=details,last_units=last_details))
        rows.append(row)
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,default=1000)
    parser.add_argument('--output',type=Path,default=ROOT/'runs/compact-pooling-spk03-1000-v1')
    parser.add_argument('--resume-lr',type=float,default=1e-4)
    parser.add_argument('--minimum-lr',type=float,default=1e-5)
    args=parser.parse_args()
    assert args.epochs>100 and args.resume_lr>=args.minimum_lr>0
    b.OUT=args.output.resolve();b.REPORT=ROOT/'reports'/b.OUT.name
    assert b.OUT!=PARENT.parent
    b.OUT.mkdir(parents=True,exist_ok=True);b.REPORT.mkdir(parents=True,exist_ok=True)
    lock=(b.OUT/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    assert torch.cuda.is_available()
    old=json.loads((PARENT_REPORT/'protocol.json').read_text())
    for path,expected in old['source_sha256'].items():assert b.sha(ROOT/path)==expected,path
    for path,expected in old['audio_sha256'].items():assert b.sha(b.DATA/path)==expected,path
    start=torch.load(PARENT/'last.pt',map_location='cuda',weights_only=True)
    assert start['epoch']==100 and start['protocol_sha256']==b.sha(PARENT_REPORT/'protocol.json')
    protocol=dict(study=b.OUT.name,heldout_speaker='spk03',training_speakers=['spk02','spk04','spk05'],
        total_epochs=args.epochs,additional_epochs=args.epochs-100,
        initialization='All three exact epoch100 LAST model and AdamW states, not selected-best states',
        inherited_protocol=old,inherited_protocol_sha256=b.sha(PARENT_REPORT/'protocol.json'),
        initial_checkpoint_sha256=b.sha(PARENT/'last.pt'),
        normalization='Exact mean/std restored from epoch100 checkpoint; recomputed only for verification',
        lr=dict(epoch101=args.resume_lr,final=args.minimum_lr,
                schedule='Single cosine over epochs101..total; reset LR only, retain AdamW moments'),
        exposures='609 shared augmented clips/epoch, independently updated models; absolute epoch seeded noise and permutation',
        selection='Earliest strict maximum mean per-noise 11-class macroF1 on 0dB draw0; includes old 100-epoch best checkpoints',
        final='Selected-best and final-last: original and all three 0dB crops; selected-best noise-only false positives',
        milestones=[v for v in [200,500,1000] if v<=args.epochs],
        limitations=['One training seed and one development speaker; no independent test',
            'This is continuation with an LR restart, not 1000 epochs of the original scratch cosine schedule',
            'Validation draw0 is reused in checkpoint selection and final evaluation'],
        continuation_source_sha256={str(p.relative_to(ROOT)):b.sha(p) for p in Path(__file__).parent.glob('*.py')})
    protocol_path=b.REPORT/'protocol.json'
    if protocol_path.exists():assert json.loads(protocol_path.read_text())==protocol,'Continuation protocol changed'
    else:b.write(protocol_path,protocol)
    if (b.REPORT/'summary.json').exists():
        b.progress(stage='complete',report=str(b.REPORT/'README.md'));return
    frame=pd.read_csv(b.DATA/'metadata.csv')
    frame=frame[frame.speaker_id.isin(['spk02','spk03','spk04','spk05'])].reset_index(drop=True)
    assert len(frame)==812 and not frame.path.str.contains('spk01').any()
    waves=np.stack([sf.read(b.DATA/p,dtype='float32')[0] for p in frame.path])
    targets=np.array([b.LABELS.index(s) for s in frame.label])
    train_idx=np.flatnonzero(frame.speaker_id.ne('spk03'));val_idx=np.flatnonzero(frame.speaker_id.eq('spk03'))
    assert len(train_idx)==609 and len(val_idx)==203
    bank=b.formant_bank(frame,waves)
    manifest=json.loads((ROOT/'runs/compact-pooling-v1/sources.json').read_text())
    assert manifest==old['sources']
    noises={}
    for kind,records in manifest.items():
        noises[kind]=[]
        for r in records:
            path=ROOT/r['prepared_path'];assert b.sha(path)==r['prepared_sha256']
            noise,rate=sf.read(path,dtype='float32');assert rate==16000
            noises[kind].append((r['noise_id'],noise))
    frontend=b.Frontend().cuda().eval();mean=start['mean'];std=start['std']
    original=b.features(frontend,waves[train_idx])
    assert torch.allclose(mean,original.mean((0,2),keepdim=True),atol=1e-6,rtol=1e-6)
    assert torch.allclose(std,original.std((0,2),keepdim=True).clamp_min(1e-4),atol=1e-6,rtol=1e-6)
    del original
    reference=float(np.median(np.sqrt(np.mean(waves[train_idx[targets[train_idx]!=10]].astype(np.float64)**2,axis=1)))**2)
    units=b.validation_units(frontend,frame,waves,targets,val_idx,noises['evaluation'],mean,std,reference,1)
    models={k:v.cuda() for k,v in matched_models(42).items()}
    optimizers={k:torch.optim.AdamW(v.parameters(),lr=args.resume_lr,weight_decay=1e-4) for k,v in models.items()}
    folder=b.OUT/'spk03';folder.mkdir(exist_ok=True)
    checkpoint=folder/'last.pt'
    if checkpoint.exists():
        start=torch.load(checkpoint,map_location='cuda',weights_only=True)
        assert start['protocol_sha256']==b.sha(protocol_path)
    first_epoch=start['epoch']+1
    best=start['best']
    for name in NAMES:
        models[name].load_state_dict(start['models'][name]);optimizers[name].load_state_dict(start['optimizers'][name])
    if first_epoch==101:
        history=json.loads((PARENT/'history.json').read_text())
        for name in NAMES:
            expected=next(r for r in history if r['model']==name and r['epoch']==100)
            score,_=b.evaluate(models[name],units)
            assert abs(score['zero_f1']-expected['zero_f1'])<1e-6,name
            state=torch.load(PARENT/(name+'_best.pt'),map_location='cuda',weights_only=True)
            state['parent_protocol_sha256']=state['protocol_sha256'];state['protocol_sha256']=b.sha(protocol_path)
            b.save(folder/(name+'_best.pt'),state)
        b.write(folder/'history.json',history)
    else:
        history=[r for r in json.loads((folder/'history.json').read_text()) if r['epoch']<first_epoch]
    b.write(folder/'initialization_verified.json',dict(parent_epoch=100,epoch100_validation_replayed=True,
        parent_checkpoint_sha256=b.sha(PARENT/'last.pt'),restored_optimizer=True,normalization_verified=True))
    b.progress(stage='initialization_verified',speaker='spk03',epoch=first_epoch-1,epochs=args.epochs)
    tick=time.monotonic()
    for epoch in range(first_epoch,args.epochs+1):
        since=time.monotonic()
        rng=np.random.default_rng(np.random.SeedSequence([42,3,epoch,100]))
        indices=rng.permutation(train_idx)
        augmented,trace=b.augment(waves,indices,targets,bank,noises['train'],'spk03',epoch,reference)
        x=b.features(frontend,augmented,mean,std);y=torch.tensor(targets[indices],device='cuda')
        lr=args.minimum_lr+(args.resume_lr-args.minimum_lr)*(1+math.cos(math.pi*(epoch-101)/max(1,args.epochs-101)))/2
        loss_sum={name:0. for name in NAMES};correct={name:0 for name in NAMES}
        for name in NAMES:
            models[name].train()
            for group in optimizers[name].param_groups:group['lr']=lr
        for offset in range(0,len(x),32):
            batch=x[offset:offset+32];label=y[offset:offset+32]
            for name in NAMES:
                opt=optimizers[name];opt.zero_grad(set_to_none=True)
                logits=models[name](batch);loss=F.cross_entropy(logits,label);assert torch.isfinite(loss)
                loss.backward();torch.nn.utils.clip_grad_norm_(models[name].parameters(),5.);opt.step()
                loss_sum[name]+=float(loss)*len(batch);correct[name]+=int(logits.argmax(1).eq(label).sum())
        epoch_rows=[]
        for name in NAMES:
            score,details=b.evaluate(models[name],units)
            row=dict(epoch=epoch,model=name,lr=lr,train_loss=loss_sum[name]/609,train_accuracy=correct[name]/609,**score)
            if score['zero_f1']>best[name]['score']:
                best[name]=dict(epoch=epoch,score=score['zero_f1'])
                b.save(folder/(name+'_best.pt'),dict(model=models[name].state_dict(),epoch=epoch,score=score,
                    mean=mean,std=std,protocol_sha256=b.sha(protocol_path)))
                b.write(folder/(name+'_best_validation.json'),details)
            epoch_rows.append(row)
        history.extend(epoch_rows);b.write(folder/'history.json',history)
        b.write(folder/f'augmentation_{epoch:04d}.json',trace)
        if epoch%10==0 or epoch==args.epochs:
            snapshot=dict(epoch=epoch,models={k:v.state_dict() for k,v in models.items()},
                optimizers={k:v.state_dict() for k,v in optimizers.items()},best=best,mean=mean,std=std,
                protocol_sha256=b.sha(protocol_path))
            b.save(checkpoint,snapshot)
            if epoch in protocol['milestones']:b.save(folder/f'epoch_{epoch:04d}.pt',snapshot)
        torch.cuda.synchronize()
        b.progress(stage='training',speaker='spk03',epoch=epoch,epochs=args.epochs,epoch_seconds=time.monotonic()-since,
            estimated_training_minutes_remaining=(args.epochs-epoch)*(time.monotonic()-tick)/(epoch-first_epoch+1)/60,results=epoch_rows)
    del units
    units=b.validation_units(frontend,frame,waves,targets,val_idx,noises['evaluation'],mean,std,reference,3)
    rows=final_evaluation(models,units,frame,waves,targets,val_idx,noises['evaluation'],frontend,mean,std,reference,folder,protocol)
    b.write(b.REPORT/'summary.json',rows)
    lines=['# 화자03 · 공통 CNN 3종 · 100→1000epoch 연장','',
        f'총 {args.epochs}epoch. 이전100epoch LAST 모델·AdamW 상태를 복원했다. 학습률은101epoch에서 {args.resume_lr:g}로 재시작하고 cosine으로 {args.minimum_lr:g}까지 낮췄다.',
        '학습 화자02·04·05 (609개), 검증 화자03 (203개). 기존 소음·포먼트 캐시와 평가 입력을 그대로 사용했다.','',
        '| 모델 | 100epoch 선택 0dB F1 | 연장 선택 epoch | 연장 선택 원본 F1 | 연장 선택 0dB F1 | 증감 | 최종 LAST 0dB F1 | 잡음 단독 오반응률 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['model']} | {r['baseline_100']['best']['zero_f1']*100:.2f}% | {r['best_epoch']} | {r['best']['original_f1']*100:.2f}% | {r['best']['zero_f1']*100:.2f}% | {r['delta_best_zero_f1']*100:+.2f}pp | {r['last']['zero_f1']*100:.2f}% | {r['noise_only_fpr']*100:.2f}% |")
    lines+=['','0dB draw0로 체크포인트를 선택하고 최종3draw로 평가했다. 과거100epoch의 선택 체크포인트도 후보에 포함했다. 개발 화자1명·seed1개 결과이며 독립 test 성능이 아니다. 학습률 재시작을 포함한 연장 효과다.']
    (b.REPORT/'README.md').write_text('\n'.join(lines)+'\n')
    b.progress(stage='complete',report=str(b.REPORT/'README.md'),results=rows)

if __name__=='__main__':main()
