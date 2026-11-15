"""Matched small-CNN readout experiment; no detector, pretrained weights or test01 audio."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch import nn
from torch.nn import functional as F
import torchaudio

from .models import matched_models
from .objectives import ProjectionHead,paired_loss
from experiments.conformer_aug.augment import draw_rng
from experiments.noise_stress.mixing import crop_noise, mix_at_snr
from experiments.overnight.data import shift_formants
from experiments.overnight.waveforms import settings

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'runs/compact-pooling-v1'
REPORT = ROOT/'reports/compact-pooling-v1'
DATA = ROOT/'datasets/task2/fixed-1p5-v1'
LABELS = json.loads((ROOT/'configs/task2_real.json').read_text())['labels']
NAMES = ('cnn_mean', 'cnn_pooled_query', 'cnn_self_cls')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def save(path, value):
    temporary = path.with_suffix('.partial')
    torch.save(value, temporary)
    temporary.replace(path)


def progress(**kwargs):
    value = dict(updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()), pid=os.getpid(), **kwargs)
    write(OUT/'progress.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


class Frontend(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,n_fft=512,win_length=400,hop_length=160,
            f_min=20.,f_max=7800.,n_mels=64,power=2.,center=True)

    def forward(self, wave):
        return self.mel(wave).clamp_min(1e-10).log10()


@torch.no_grad()
def features(frontend, waves, mean=None, std=None):
    chunks=[]
    for offset in range(0,len(waves),64):
        x=frontend(torch.as_tensor(np.array(waves[offset:offset+64]),device='cuda'))
        if mean is not None:
            x=(x-mean)/std
        chunks.append(x)
    return torch.cat(chunks)


def metrics(logits, labels):
    pred=logits.argmax(1)
    cm=torch.bincount(labels*11+pred,minlength=121).reshape(11,11).float()
    denom=cm.sum(0)+cm.sum(1)
    score=torch.where(denom>0,2*cm.diag()/denom,torch.zeros_like(denom))
    return dict(macro_f1=float(score.mean()),accuracy=float((pred==labels).float().mean()),
                loss=float(F.cross_entropy(logits,labels)))


@torch.inference_mode()
def predict(model, x):
    model.eval()
    return torch.cat([model(x[i:i+32]) for i in range(0,len(x),32)])


def formant_worker():
    torch.set_num_threads(1)


def formant_task(task):
    index,path,wave=task
    result=[]; rows=[]
    for variant in range(2):
        ratio=settings(path,variant)[2]
        transformed=shift_formants(wave,ratio)
        result.append(transformed)
        rows.append(dict(index=index,path=path,variant=variant,ratio=ratio,
                         waveform_sha256=hashlib.sha256(transformed.tobytes()).hexdigest()))
    return index,np.stack(result),rows


def formant_bank(frame, waves):
    cache=ROOT/'runs/compact-pooling-v1'
    cache.mkdir(parents=True,exist_ok=True)
    target=cache/'formants.npy'
    pins={p:sha(DATA/p) for p in frame.path}
    identity=dict(paths=frame.path.tolist(),audio_sha256=pins,
                  transform_source_sha256=sha(ROOT/'experiments/overnight/data.py'),
                  settings_source_sha256=sha(ROOT/'experiments/overnight/waveforms.py'),
                  cache_version='weak-formants-process-isolated-v1')
    manifest=cache/'formants.json'
    if target.exists() and manifest.exists():
        old=json.loads(manifest.read_text())
        assert old['identity']==identity and old['sha256']==sha(target)
        return np.load(target,mmap_mode='r')
    bank=np.lib.format.open_memmap(target.with_suffix('.partial.npy'),mode='w+',dtype='float32',
                                  shape=(len(frame),2,24000))
    bank[:]=waves[:,None,:]
    tasks=[(i,row.path,waves[i]) for i,row in enumerate(frame.itertuples()) if row.label!='silence']
    traces=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=formant_worker) as pool:
        for count,(i,transformed,rows) in enumerate(pool.map(formant_task,tasks),1):
            bank[i]=transformed
            traces.extend(rows)
            if count%100==0:progress(stage='formant_cache',completed=count,total=len(tasks))
    bank.flush()
    del bank
    target.with_suffix('.partial.npy').replace(target)
    write(manifest,dict(identity=identity,sha256=sha(target),rows=traces))
    return np.load(target,mmap_mode='r')


def augment(waves, indices, targets, bank, train_noises, speaker, epoch, reference, noise_probability=.5):
    augmented=[]; trace=[]
    for draw,index in enumerate(indices):
        word=targets[index]!=10
        active=bool(word and draw_rng(42,speaker,epoch,draw,6).random()<.5)
        variant=int(draw_rng(42,speaker,epoch,draw,8).integers(2))
        wave=np.array(bank[index,variant] if active else waves[index])
        rng=draw_rng(42,speaker,epoch,draw,2)
        noisy=bool(rng.random()<noise_probability)
        row=dict(index=int(index),formant=active,variant=variant,noise=noisy)
        if noisy:
            noise_id,recording=train_noises[int(rng.integers(len(train_noises)))]
            for retry in range(1000):
                start=int(rng.integers(0,len(recording)-24000+1))
                noise=recording[start:start+24000].astype(np.float64)
                noise-=noise.mean()
                if np.mean(noise**2)>=1e-8:break
            else:raise ValueError('No energetic noise crop')
            snr=float(rng.uniform(-5,20))
            power=reference if not word else float(np.mean(wave.astype(np.float64)**2))
            wave,_=mix_at_snr(wave,noise,snr,power)
            row.update(noise_id=noise_id,start=start,snr=snr)
        augmented.append(wave)
        trace.append(row)
    return np.stack(augmented),trace


def validation_units(frontend, frame, waves, targets, indices, eval_noises, mean, std, reference, draws):
    clean=features(frontend,waves[indices],mean,std)
    labels=torch.tensor(targets[indices],device='cuda')
    units=[dict(kind='original',noise_id='none',draw=-1,x=clean,y=labels)]
    for noise_id,recording in eval_noises:
        for draw in range(draws):
            mixed=[]
            for index in indices:
                noise,_=crop_noise(recording,frame.iloc[index].path,noise_id,draw)
                wave=waves[index]
                power=reference if targets[index]==10 else float(np.mean(wave.astype(np.float64)**2))
                mixed.append(mix_at_snr(wave,noise,0.,power)[0])
            x=features(frontend,np.stack(mixed),mean,std)
            units.append(dict(kind='mixed_0db',noise_id=noise_id,draw=draw,x=x,y=labels))
    return units


def evaluate(model, units):
    rows=[]
    for unit in units:
        rows.append(dict(kind=unit['kind'],noise_id=unit['noise_id'],draw=unit['draw'],
                         **metrics(predict(model,unit['x']),unit['y'])))
    zero=[r for r in rows if r['kind']=='mixed_0db']
    return dict(zero_f1=float(np.mean([r['macro_f1'] for r in zero])),
                zero_loss=float(np.mean([r['loss'] for r in zero])),
                original_f1=rows[0]['macro_f1']),rows


def main():
    global OUT,REPORT,NAMES
    parser=argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--folds',nargs='+',default=['spk02','spk03'])
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--asset-manifest',type=Path,default=ROOT/'runs/compact-pooling-v1/sources.json')
    parser.add_argument('--objective',choices=['ce','paired_ce','paired_supcon'],default='ce')
    parser.add_argument('--models',nargs='+',choices=NAMES,default=list(NAMES))
    parser.add_argument('--supcon-weight',type=float,default=.1)
    parser.add_argument('--temperature',type=float,default=.1)
    args=parser.parse_args()
    assert args.epochs>0 and args.temperature>0 and args.supcon_weight>=0
    assert len(set(args.models))==len(args.models)
    assert args.objective=='ce' or args.output.resolve()!=OUT.resolve(), 'Paired runs require a separate --output folder'
    NAMES=tuple(args.models)
    OUT=args.output.resolve()
    REPORT=ROOT/'reports'/OUT.name
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    assert torch.cuda.is_available()
    frame=pd.read_csv(DATA/'metadata.csv')
    frame=frame[frame.speaker_id.isin(['spk02','spk03','spk04','spk05'])].reset_index(drop=True)
    assert len(frame)==812 and not frame.path.str.contains('spk01').any()
    targets=np.array([LABELS.index(s) for s in frame.label])
    waves=[]
    for p in frame.path:
        wave,rate=sf.read(DATA/p,dtype='float32')
        assert rate==16000 and wave.shape==(24000,) and np.isfinite(wave).all()
        waves.append(wave)
    waves=np.stack(waves)
    source_manifest=json.loads(args.asset_manifest.read_text())
    noises={}
    for kind,rows in source_manifest.items():
        noises[kind]=[]
        for row in rows:
            path=ROOT/row['prepared_path']
            assert sha(path)==row['prepared_sha256']
            wave,rate=sf.read(path,dtype='float32')
            assert rate==16000
            noises[kind].append((row['noise_id'],wave))
    protocol=dict(study=OUT.name,seed=42,folds=args.folds,epochs=args.epochs,
        split='Separate leave-one-speaker-out: each fold trains on the other three dev speakers; 609 train / 203 validation.',
        input='Full 1.5s; no detector, VAD probabilities or speech masks',
        architecture='Matchbox-inspired common CNN + projection32; mean / single-head ST-inspired pooled query / one transformer block with CLS (d32 heads2 FFN64). Not exact published model reproductions.',
        initialization='Random; identical CNN, projection and classifier initial tensors; independently trained models',
        frontend='64 log10 mel bands, 20..7800Hz, FFT512 win400 hop160 center=True; 151 frames; per-band mean/std fitted on original training-fold audio only',
        objective=dict(name=args.objective,supcon_weight=args.supcon_weight if args.objective=='paired_supcon' else 0.,temperature=args.temperature,
            paired_reference='Same formant-transformed original before additional noise; not guaranteed clean',
            positives='Same command, including different speakers and paired views; exclude silence from contrastive only',
            projection='32->64->32, training only; discarded at inference' if args.objective=='paired_supcon' else None),
        augmentation=dict(noise_probability=.5 if args.objective=='ce' else 1.,snr_db=[-5,20],formant_probability=.5,
            formant_range=[1/1.1,1.1],fixed_variants=2,pitch_preserved=True,
            transform='Historical Praat Change gender settings and waveform-first-bank-v1 formant seeds',
            order='formants then noise',speed=False,pitch=False,specaugment=False),
        environment=dict(torch=torch.__version__,torchaudio=torchaudio.__version__,gpu=torch.cuda.get_device_name(0)),
        optimizer=dict(name='AdamW',lr=.001,weight_decay=.0001,batch=32,gradient_clip_norm=5.,
                       schedule='cosine to 1e-5 over fixed epoch budget'),
        exposures='609 unique clips/epoch; same permutation, transformed waveforms and features for all three independently updated models',
        monitor=f'Original + {len(noises["evaluation"])} reserved noise sources at 0dB fixed draw0 every epoch',
        selection='Per model/fold earliest strict maximum mean of per-noise 11-class macroF1 values at 0dB draw0',
        final='Selected best + final last: original and all three 0dB noise crops; noise-only false positives on selected best',
        limitations=['One seed and two dev folds; not independent test results',
                    'The validation draw0 used for checkpoint selection is included in final three-draw evaluation',
                    'Self attention adds a transformer block, positional encoding and CLS, not only a different weighted mean',
                    'The common CNN has a large temporal receptive field; mean pooling does not imply absence of temporal context',
                    'Additional unseen-noise suite and learned detector are not part of this run'],
        test_audio_read=False,test_evaluated=False,labels=LABELS,sources=source_manifest,
        audio_sha256={p:sha(DATA/p) for p in frame.path},
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')})
    protocol_path=REPORT/'protocol.json'
    if protocol_path.exists():assert json.loads(protocol_path.read_text())==protocol,'Run protocol changed'
    else:write(protocol_path,protocol)
    progress(stage='preparing_formants')
    bank=formant_bank(frame,waves)
    frontend=Frontend().cuda().eval()
    summaries=[]
    for speaker in args.folds:
        folder=OUT/speaker;folder.mkdir(exist_ok=True)
        if (folder/'summary.json').exists():
            summaries.extend(json.loads((folder/'summary.json').read_text()));continue
        train_idx=np.flatnonzero(frame.speaker_id.ne(speaker))
        val_idx=np.flatnonzero(frame.speaker_id.eq(speaker))
        assert len(train_idx)==609 and len(val_idx)==203
        assert not set(frame.iloc[train_idx].path)&set(frame.iloc[val_idx].path)
        original_train=features(frontend,waves[train_idx])
        mean=original_train.mean((0,2),keepdim=True)
        std=original_train.std((0,2),keepdim=True).clamp_min(1e-4)
        del original_train
        reference=float(np.median(np.sqrt(np.mean(waves[train_idx[targets[train_idx]!=10]].astype(np.float64)**2,axis=1)))**2)
        units=validation_units(frontend,frame,waves,targets,val_idx,noises['evaluation'],mean,std,reference,1)
        models={k:v.cuda() for k,v in matched_models(42).items() if k in NAMES}
        # The classifier also starts identically; only readout-specific tensors differ.
        for name in NAMES[1:]:models[name].classifier.load_state_dict(models[NAMES[0]].classifier.state_dict())
        torch.manual_seed(4242)
        projectors={name:ProjectionHead().cuda() for name in NAMES} if args.objective=='paired_supcon' else {}
        optimizers={name:torch.optim.AdamW(list(model.parameters())+(list(projectors[name].parameters()) if name in projectors else []),lr=.001,weight_decay=.0001) for name,model in models.items()}
        best={name:dict(score=-1.,epoch=0) for name in NAMES}
        start_epoch=1
        last=folder/'last.pt'
        if last.exists():
            restored=torch.load(last,map_location='cuda',weights_only=False)
            assert restored['protocol_sha256']==sha(protocol_path)
            for name in NAMES:
                models[name].load_state_dict(restored['models'][name]);optimizers[name].load_state_dict(restored['optimizers'][name])
            for name,projector in projectors.items():projector.load_state_dict(restored['projectors'][name])
            best=restored['best'];start_epoch=restored['epoch']+1
        history=[]
        if (folder/'history.json').exists():
            history=[r for r in json.loads((folder/'history.json').read_text()) if r['epoch']<start_epoch]
        tick=time.monotonic()
        for epoch in range(start_epoch,args.epochs+1):
            since=time.monotonic()
            rng=np.random.default_rng(np.random.SeedSequence([42,int(speaker[-2:]),epoch,100]))
            indices=rng.permutation(train_idx)
            augmented,trace=augment(waves,indices,targets,bank,noises['train'],speaker,epoch,reference,
                                    noise_probability=.5 if args.objective=='ce' else 1.)
            x=features(frontend,augmented,mean,std)
            paired_x=None
            if args.objective!='ce':
                reference_waves=np.stack([bank[r['index'],r['variant']] if r['formant'] else waves[r['index']] for r in trace])
                paired_x=features(frontend,reference_waves,mean,std)
            y=torch.tensor(targets[indices],device='cuda')
            lr=1e-5+(.001-1e-5)*(1+math.cos(math.pi*(epoch-1)/max(1,args.epochs-1)))/2
            losses={name:0. for name in NAMES};correct={name:0 for name in NAMES}
            for name in NAMES:
                models[name].train()
                for group in optimizers[name].param_groups:group['lr']=lr
            for offset in range(0,len(x),32):
                batch=x[offset:offset+32];label=y[offset:offset+32]
                for name in NAMES:
                    optimizer=optimizers[name];optimizer.zero_grad(set_to_none=True)
                    if paired_x is None:
                        logits=models[name](batch);loss=F.cross_entropy(logits,label)
                    else:
                        loss,logits,_,_=paired_loss(models[name],projectors.get(name),paired_x[offset:offset+32],batch,label,
                                                  weight=args.supcon_weight,temperature=args.temperature)
                    assert torch.isfinite(loss)
                    loss.backward();torch.nn.utils.clip_grad_norm_(list(models[name].parameters())+(list(projectors[name].parameters()) if name in projectors else []),5.)
                    optimizer.step()
                    losses[name]+=float(loss)*len(batch)
                    correct[name]+=int((logits.argmax(1)==label).sum())
            epoch_rows=[]
            for name in NAMES:
                score,details=evaluate(models[name],units)
                row=dict(epoch=epoch,model=name,lr=lr,train_loss=losses[name]/609,
                         train_accuracy=correct[name]/609,**score)
                if score['zero_f1']>best[name]['score']:
                    best[name]=dict(score=score['zero_f1'],epoch=epoch)
                    save(folder/(name+'_best.pt'),dict(model=models[name].state_dict(),epoch=epoch,
                        score=score,mean=mean,std=std,protocol_sha256=sha(protocol_path),
                        projector=projectors[name].state_dict() if name in projectors else None))
                    write(folder/(name+'_best_validation.json'),details)
                epoch_rows.append(row)
            history.extend(epoch_rows)
            write(folder/'history.json',history)
            # Store an auditable shared augmentation stream once, not once per model.
            write(folder/f'augmentation_{epoch:03d}.json',trace)
            if epoch%10==0 or epoch==args.epochs:
                save(last,dict(epoch=epoch,models={k:v.state_dict() for k,v in models.items()},
                    optimizers={k:v.state_dict() for k,v in optimizers.items()},projectors={k:v.state_dict() for k,v in projectors.items()},best=best,
                    mean=mean,std=std,protocol_sha256=sha(protocol_path)))
            torch.cuda.synchronize()
            elapsed=time.monotonic()-since
            remaining=(args.epochs-epoch)+(len(args.folds)-args.folds.index(speaker)-1)*args.epochs
            progress(stage='training',speaker=speaker,epoch=epoch,epochs=args.epochs,epoch_seconds=elapsed,
                     estimated_training_minutes_remaining=remaining*((time.monotonic()-tick)/(epoch-start_epoch+1))/60,
                     results=epoch_rows)
        del units
        torch.cuda.empty_cache()
        units=validation_units(frontend,frame,waves,targets,val_idx,noises['evaluation'],mean,std,reference,3)
        fold_rows=[]
        for name in NAMES:
            model=models[name]
            last_score,last_details=evaluate(model,units)
            state=torch.load(folder/(name+'_best.pt'),map_location='cuda',weights_only=True)
            model.load_state_dict(state['model'])
            score,details=evaluate(model,units)
            predictions=[]
            for unit in units:
                preds=predict(model,unit['x']).argmax(1).cpu().tolist()
                for index,prediction in zip(val_idx,preds):
                    predictions.append(dict(path=frame.iloc[index].path,true_idx=int(targets[index]),pred_idx=prediction,
                        kind=unit['kind'],noise_id=unit['noise_id'],draw=unit['draw']))
            pd.DataFrame(predictions).to_csv(folder/(name+'_best_predictions.csv.gz'),index=False)
            false_positives=0;count=0
            for noise_id,recording in noises['evaluation']:
                noise_waves=[]
                for draw in range(3):
                    for i in range(20):
                        noise,_=crop_noise(recording,f'noise_only_{i}',noise_id,draw)
                        noise_waves.append(mix_at_snr(np.zeros(24000,np.float32),noise,0.,reference)[0])
                noise_x=features(frontend,np.stack(noise_waves),mean,std)
                false_positives+=int(predict(model,noise_x).argmax(1).ne(10).sum());count+=len(noise_x)
            row=dict(speaker=speaker,model=name,parameters=sum(p.numel() for p in model.parameters()),
                     best_epoch=state['epoch'],best=score,last=last_score,noise_only_fpr=false_positives/count,
                     objective=args.objective,training_only_projection_parameters=sum(p.numel() for p in projectors[name].parameters()) if name in projectors else 0)
            fold_rows.append(row)
            write(folder/(name+'_final_metrics.json'),dict(summary=row,best_units=details,last_units=last_details))
        write(folder/'summary.json',fold_rows)
        summaries.extend(fold_rows)
        del units,models,optimizers,projectors
        torch.cuda.empty_cache()
    write(REPORT/'summary.json',summaries)
    lines=['# 공통 CNN · 집계 방식 비교 · 증강 적용','',f'학습 목적: {args.objective}', '',
           '검출기/발화확률/사전학습 없이 전체 1.5초를 입력한다. 화자02·03 각각 홀드아웃하는 두 개발 분할이다.',
           f'소음 p={0.5 if args.objective=="ce" else 1.0}, SNR [-5,20]dB + 약한 포먼트 p=0.5. 모델은 같은 CNN·분류기 초기값과 같은 증강 입력을 사용하고 독립 학습한다.','',
           '| 모델 | 파라미터 | 화자 | 선택 epoch | 원본 F1 | 0dB F1 (3draw) | 마지막 epoch 0dB F1 | 잡음 단독 오반응률 |',
           '|---|---:|---|---:|---:|---:|---:|---:|']
    for row in summaries:
        lines.append(f"| {row['model']} | {row['parameters']:,} | {row['speaker']} | {row['best_epoch']} | {row['best']['original_f1']*100:.2f}% | {row['best']['zero_f1']*100:.2f}% | {row['last']['zero_f1']*100:.2f}% | {row['noise_only_fpr']*100:.2f}% |")
    lines+=['','한 seed·개발 2분할의 선별 결과다. 0dB draw0로 체크포인트를 선택했고, 최종 3draw에도 draw0가 포함된다. 독립 test 성능이나 검출기 효과를 의미하지 않는다.',
            '기존 4분할 Conformer 결과와는 모델·학습 초기화·전처리·검증 분할이 달라 직접적인 개선 수치로 해석할 수 없다.','',
            f'상세 설정: protocol.json. 전체 epoch 기록·증강 추적·선택/최종 체크포인트·예측값은 {OUT.relative_to(ROOT)} 아래에 보존한다.']
    (REPORT/'README.md').write_text('\n'.join(lines)+'\n')
    progress(stage='complete',report=str(REPORT/'README.md'),results=summaries)


if __name__=='__main__':
    main()
