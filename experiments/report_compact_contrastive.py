"""Audit and summarize the completed100/1000epoch contrastive study, no fitting."""
import copy
import datetime
import hashlib
import json
import shutil
from pathlib import Path
import time

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.compact_pooling.models import matched_models
from experiments.compact_pooling.run import Frontend

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports/compact-contrastive-launch'
NAMES=('cnn_mean','cnn_pooled_query','cnn_self_cls')
DISPLAY={'cnn_mean':'Mean pooling','cnn_pooled_query':'Pooled query','cnn_self_cls':'Self-attention + CLS'}
CONDITIONS=('noise','noise-speaker')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def f1(frame):
    cm=np.bincount(frame.true_idx.to_numpy()*11+frame.pred_idx.to_numpy(),minlength=121).reshape(11,11)
    den=cm.sum(0)+cm.sum(1)
    return float(np.divide(2.*cm.diagonal(),den,out=np.zeros(11),where=den!=0).mean())


def benchmark():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    data=ROOT/'datasets/task2/fixed-1p5-v1'
    metadata=pd.read_csv(data/'metadata.csv')
    row=metadata[(metadata.speaker_id=='spk03')&(metadata.label!='silence')].iloc[0]
    wave,sr=sf.read(data/row.path,dtype='float32');assert sr==16000
    wave=torch.tensor(wave[None])
    frontend=Frontend().cuda().eval();cpu_frontend=Frontend().eval()
    def measure(function,cuda):
        for _ in range(20):function()
        result=[]
        for _ in range(100):
            if cuda:torch.cuda.synchronize()
            tick=time.perf_counter();function()
            if cuda:torch.cuda.synchronize()
            result.append((time.perf_counter()-tick)*1000)
        return dict(median_ms=float(np.median(result)),p95_ms=float(np.percentile(result,95)))
    result=[]
    with torch.inference_mode():
        for name,model in matched_models().items():
            state=torch.load(ROOT/f'runs/compact-noise-speaker-spk03-e1000/spk03/{name}_best.pt',map_location='cpu',weights_only=False)
            model.load_state_dict(state['model']);model=model.cuda().eval()
            mean=state['mean'].cuda();std=state['std'].cuda()
            x=(frontend(wave.cuda())-mean)/std
            gpu=measure(lambda:model(x),True)
            e2e=measure(lambda:model((frontend(wave.cuda())-mean)/std).argmax(-1).cpu(),True)
            cpu_model=copy.deepcopy(model).cpu().eval();cpu_mean=mean.cpu();cpu_std=std.cpu()
            cpu=measure(lambda:cpu_model((cpu_frontend(wave)-cpu_mean)/cpu_std).argmax(-1),False)
            result.append(dict(model=name,gpu_model=gpu,gpu_frontend_model_transfer=e2e,cpu_frontend_model=cpu))
    value=dict(results=result,batch=1,cpu_threads=2,gpu=torch.cuda.get_device_name(),warmup=20,measurements=100,
        input_path=row.path,input_seconds=1.5,weights='noise-speaker/spk03 selected best checkpoints, projection head excluded',
        excludes='WAV decode, model loading, 1.5s audio acquisition, endpoint detection and communications',
        note='GPU end-to-end starts with decoded CPU waveform, includes GPU frontend+normalization+classifier and CPU class output; CPU starts with decoded waveform.')
    write(REPORT/'inference_latency.json',value)
    return value


def main():
    summary=pd.read_csv(REPORT/'comparison.csv')
    audit=dict(history_rows=0,prediction_sets_recomputed=0,shared_augmentation_files_verified=0,
               protocol_code_hashes_verified=True,all_folds_completed=True)
    histories={};prefix_rows=[];finish=[];export_history=[]
    for speaker,epochs in [('spk03',1000),('spk02',100)]:
        for condition in CONDITIONS:
            name=f'compact-{condition}-{speaker}-e{epochs}'
            run=ROOT/'runs'/name
            history=pd.DataFrame(json.loads((run/speaker/'history.json').read_text()))
            protocol=json.loads((ROOT/'reports'/name/'protocol.json').read_text())
            for path,expected in protocol['source_sha256'].items():assert digest(ROOT/path)==expected,path
            assert not protocol['test_audio_read'] and not protocol['test_evaluated']
            assert protocol['folds']==[speaker] and protocol['epochs']==epochs
            assert len(history)==3*epochs
            progress=json.loads((run/'progress.json').read_text());assert progress['stage']=='complete'
            finish.append(progress['updated_utc'])
            audit['history_rows']+=len(history)
            export_history.append(history.assign(condition=condition,speaker=speaker,budget=epochs))
            for file in (run/speaker).glob('*_best_predictions.csv.gz'):
                shutil.copyfile(file,ROOT/'reports'/name/file.name)
            for file in (run/speaker).glob('*_final_metrics.json'):
                shutil.copyfile(file,ROOT/'reports'/name/file.name)
            for model in NAMES:
                part=history[history.model==model].sort_values('epoch').reset_index(drop=True)
                assert part.epoch.tolist()==list(range(1,epochs+1))
                histories[(condition,speaker,model)]=part
                chosen=summary[(summary.condition==condition)&(summary.speaker==speaker)&(summary.model==model)].iloc[0]
                assert int(part.iloc[part.zero_f1.argmax()].epoch)==int(chosen.best_epoch)
                preds=pd.read_csv(run/speaker/(model+'_best_predictions.csv.gz'))
                assert not preds.path.str.contains('spk01').any()
                clean=f1(preds[preds.kind=='original'])
                noisy=np.mean([f1(p) for _,p in preds[preds.kind=='mixed_0db'].groupby(['noise_id','draw'])])
                assert abs(clean-chosen.original_f1)<1e-6 and abs(noisy-chosen.zero_f1)<1e-6
                audit['prediction_sets_recomputed']+=1
                prefix=part[part.epoch<=100]
                prefix_rows.append(dict(condition=condition,speaker=speaker,model=model,
                    maximum_first100_draw0=float(prefix.zero_f1.max()),maximum_full_draw0=float(part.zero_f1.max()),
                    first100_best_epoch=int(prefix.iloc[prefix.zero_f1.argmax()].epoch),full_best_epoch=int(chosen.best_epoch),
                    epoch100_draw0=float(part.iloc[99].zero_f1),final_draw0=float(part.iloc[-1].zero_f1),
                    final_train_accuracy=float(part.iloc[-1].train_accuracy)))
        a=ROOT/f'runs/compact-noise-{speaker}-e{epochs}'/speaker
        b=ROOT/f'runs/compact-noise-speaker-{speaker}-e{epochs}'/speaker
        for epoch in range(1,epochs+1):
            assert digest(a/f'augmentation_{epoch:03d}.json')==digest(b/f'augmentation_{epoch:03d}.json'),(speaker,epoch)
            audit['shared_augmentation_files_verified']+=1
    assert audit['history_rows']==6600 and audit['prediction_sets_recomputed']==12
    pd.concat(export_history,ignore_index=True).to_csv(REPORT/'epoch_histories.csv.gz',index=False)
    pd.DataFrame(prefix_rows).to_csv(REPORT/'epoch_budget_diagnostics.csv',index=False)
    fig,axes=plt.subplots(2,3,figsize=(15,7),sharex='row',sharey='row')
    for i,speaker in enumerate(['spk03','spk02']):
        for j,model in enumerate(NAMES):
            ax=axes[i,j]
            for condition in CONDITIONS:
                h=histories[(condition,speaker,model)]
                # Raw curves retained faintly; rolling means are only for readability.
                color='tab:blue' if condition=='noise' else 'tab:orange'
                ax.plot(h.epoch,h.zero_f1*100,color=color,alpha=.16,linewidth=.6)
                ax.plot(h.epoch,h.zero_f1.rolling(10,min_periods=1).mean()*100,color=color,label=condition)
            ax.set(title=speaker+' / '+DISPLAY[model],xlabel='Epoch',ylabel='0 dB macro F1, draw0 (%)')
            ax.grid(alpha=.2)
    axes[0,0].legend();fig.tight_layout();fig.savefig(REPORT/'learning_curves.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    x=np.arange(3);width=.36
    for ax,speaker in zip(axes,['spk03','spk02']):
        for offset,condition in [(-width/2,'noise'),(width/2,'noise-speaker')]:
            vals=[float(summary[(summary.condition==condition)&(summary.speaker==speaker)&(summary.model==model)].iloc[0].zero_f1)*100 for model in NAMES]
            bars=ax.bar(x+offset,vals,width,label=condition)
            ax.bar_label(bars,fmt='%.2f',fontsize=8)
        ax.set(title=speaker,xticks=x,xticklabels=['Mean','Pooled query','Self + CLS'],ylabel='Selected best 0 dB macro F1 (%)',ylim=(0,100))
        ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(REPORT/'best_comparison.png',dpi=160);plt.close(fig)
    timings=benchmark()
    state=json.loads((ROOT/'runs/compact-contrastive-launch/status.json').read_text())
    start=datetime.datetime.fromisoformat(state['started_utc'])
    end=max(datetime.datetime.fromisoformat(s.replace('Z','+00:00')) for s in finish)
    zone=datetime.timezone(datetime.timedelta(hours=9))
    minutes=(end-start).total_seconds()/60
    audit.update(started_kst=start.astimezone(zone).isoformat(),finished_kst=end.astimezone(zone).isoformat(),wall_minutes=minutes)
    write(REPORT/'summary_audit.json',audit)
    lines=['# Compact CNN 소음·화자 대조학습 결과 정리','',
        '**화자03에서는 평균 풀링+화자 대조가 최고, 화자02에서는 self-attention+화자 대조가 최고였다. 화자 대조 효과와 구조 순위는 화자별로 달랐다.**', '',
        f'완료: 한국시간 {start.astimezone(zone):%Y-%m-%d %H:%M:%S}~{end.astimezone(zone):%H:%M:%S}, 실제 경과 {minutes:.2f}분. 3모델×2조건×(1000+100) = **6,600 model-epochs**. 두 조건을 같은 RTX3090에서 병렬 실행했다.', '',
        '## 어떤 실험인가','',
        '| 항목 | 설정 |','|---|---|',
        '| 모델 | 공통 작은 CNN 뒤 Mean pooling / 대표쿼리 Pooled attention / 작은 Self-attention+CLS |',
        '| 추론 파라미터 | Mean77,835 / Pooled78,859 / Self86,315 |',
        '| 초기화 | 모두 무작위. 사전학습 없음. 같은 CNN·분류기 초기값, 이후 독립 학습 |',
        '| 검증 | spk03은1000epoch, spk02는100epoch. 각 분할은 나머지 개발화자3명609개 학습 / 검증203개 |',
        '| 입력·증강 | 전체1.5초,64-bin logmel. 약한 포먼트p0.5, 원본/추가소음2뷰, SNR U[-5,20]dB |',
        '| 소음만 대조 | 같은 녹음의 원본·소음 버전만positive. 다른 녹음의 같은 명령어는negative로 쓰지 않고 제외 |',
        '| 화자까지 대조 | 같은 명령어의 다른 녹음·학습화자도positive. 두 조건 모두 silence는CE만 적용 |',
        '| 손실 | CE +0.1 SupCon, temperature0.1,32→64→32 학습 전용 투영부는 추론 시 제거 |',
        '| 최고epoch 선택 | 고정0dB draw0의8소음별 macro F1 평균 최고, 동률은먼저 나온epoch |',
        '| 최종점수 | 선택checkpoint의 원본 / 8소음×3crop 0dB 평균. 종료점도같은조건 평가 |', '',
        '검출기/VAD 없이 전체 음성을 사용했다. 학습/평가 소음원본은 분리했고, 두 대조조건의 모든 학습 파형 변형·소음 crop·SNR trace 파일1,100쌍이 정확히 일치한다. 동일 명령어를 다른 화자가 발음한 경우까지 가깝게 만드는 것이며, 화자ID를 입력 특징으로 주거나 검증 화자를 학습한 것이 아니다.', '',
        '## 1. 최고checkpoint 비교 — 0dB macro F1','',
        '아래 최고checkpoint는 선택용 draw0 최고에서의 최종3draw 점수다. 매epoch 최종3draw를 평가해 골랐다는 뜻이 아니다. 원본F1 최고를 별도로 골라 합치지도 않았다.','']
    for speaker,epochs in [('spk03',1000),('spk02',100)]:
        lines += [f'### {speaker}, 최대{epochs}epoch','',
            '| 모델 | 소음만 | 소음+화자 | 화자추가 차이 | 선택epoch 소음만 / 화자포함 |',
            '|---|---:|---:|---:|---|']
        for model in NAMES:
            a=summary[(summary.condition=='noise')&(summary.speaker==speaker)&(summary.model==model)].iloc[0]
            b=summary[(summary.condition=='noise-speaker')&(summary.speaker==speaker)&(summary.model==model)].iloc[0]
            lines.append(f'| {DISPLAY[model]} | {a.zero_f1*100:.2f}% | {b.zero_f1*100:.2f}% | {(b.zero_f1-a.zero_f1)*100:+.2f}%p | {a.best_epoch} / {b.best_epoch} |')
        lines.append('')
    lines += ['![최고점 비교](best_comparison.png)','',
        '화자03: Mean은 화자 포함85.44%로 관측 최고이며 원본95.89%, 잡음단독 오반응4.38%다. 화자02: Self+CLS는 화자 포함50.81%로 관측 최고지만 잡음단독 오반응35.42%가 남았다. 고점만으로 실제 운용 구성을 확정하지 않는다.', '',
        '## 2. 최고점의 원본·잡음·오반응과 종료점','',
        '| 화자 | 모델 | 조건 | 최고점 원본F1 | 최고점0dB F1 | 종료점0dB F1 | 최고점 잡음단독 오반응 |',
        '|---|---|---|---:|---:|---:|---:|']
    for r in summary.to_dict('records'):
        lines.append(f'| {r["speaker"]} | {DISPLAY[r["model"]]} | {r["condition"]} | {r["original_f1"]*100:.2f}% | {r["zero_f1"]*100:.2f}% | {r["last_zero_f1"]*100:.2f}% | {r["noise_only_fpr"]*100:.2f}% |')
    lines += ['',
        '모든 실행에서 종료점0dB F1은 선택checkpoint의 최종0dB F1보다 낮았다. 학습을 길게 했다고 마지막 가중치가 최선은 아니다. 이 표만으로 모든 후반 악화를 과적합으로 확정하지 않는다. epoch별 학습·검증 loss와 F1은 원본 history에 보존했다.', '',
        '## 3. 1000epoch를 진행한 화자03의 변화','',
        '아래는 **같은 cosine1000 학습 경로**의 epoch100과1000을 고정 draw0로 비교했다. cosine100으로 별도로 학습한 모델과의 비교가 아니다. 앞 표의 최종3draw와 집계가 다르다.', '',
        '| 모델 | 조건 | epoch100 0dB F1 | epoch1000 0dB F1 | 변화 |',
        '|---|---|---:|---:|---:|']
    for r in prefix_rows:
        if r['speaker']!='spk03':continue
        lines.append(f'| {DISPLAY[r["model"]]} | {r["condition"]} | {r["epoch100_draw0"]*100:.2f}% | {r["final_draw0"]*100:.2f}% | {(r["final_draw0"]-r["epoch100_draw0"])*100:+.2f}%p |')
    lines += ['',
        'Mean/Pooled은 이 경로에서 100epoch 이후에도 개선됐다. Self+CLS 화자 대조는80.10→80.30%로 종료점 차이가 작았고, 최고epoch는331이었다.', '',
        '![학습곡선](learning_curves.png)', '',
        '곡선의 얇은 선은 원본epoch 기록, 굵은 선은 가독성을 위한10epoch 이동평균이다. 점수 선택에는 이동평균을 사용하지 않았다.', '',
        '## 4. 추론 속도 — 현재 서버 실측','',
        'batch1,1.5초 음성,CPU2threads,20회워밍업 후100회. 화자03/화자포함 조건의 선택checkpoint를 사용했다. 모든 대조학습 projection은 제거했다.', '',
        '| 모델 | GPU 모델만 중앙값 | CPU파형→GPU frontend·모델→CPU예측 중앙값 | CPU frontend+모델 중앙값 |',
        '|---|---:|---:|---:|']
    for r in timings['results']:
        lines.append(f'| {DISPLAY[r["model"]]} | {r["gpu_model"]["median_ms"]:.2f}ms | {r["gpu_frontend_model_transfer"]["median_ms"]:.2f}ms | {r["cpu_frontend_model"]["median_ms"]:.2f}ms |')
    lines += ['',
        'WAV읽기·모델로딩·1.5초음성을모으는시간·발화종료판단·통신은 제외했다. 실제 서비스 응답지연 전체가 이 수치라는 뜻은 아니다. GPU타이밍은 동기화했고 p95는 inference_latency.json에 있다.', '',
        '## 5. 해석과 남은 비교','',
        '- 화자 대조가 모든 모델·화자에서 개선된 것은 아니다. spk03 Mean/Pooled은 개선됐고, spk02 Mean/Pooled은 악화·Self만 개선됐다. 한 seed에서의 관측이다.',
        '- spk02와03은 검증화자와 학습화자 구성·학습예산이 모두 다르다. 두 결과를 같은 예산의4-fold CV 평균으로 합치지 않는다. spk02의 낮은 성능 원인을 모델 용량이나 발음 중 하나로 확정하지 않는다.',
        '- 이 서버에서 진행한 것은 대조학습2조건이다. 동일 입력/학습량의paired-CE 대조군을100/1000epoch로 실행하지 않았다. 따라서 여기서 대조학습 자체가 CE보다 얼마나 좋아졌는지는 확정하지 않는다.',
        '- 다른 서버의기본CE는100epoch·소음p0.5이고 여기서는두뷰·추가소음p1.0이다. torch/CUDA버전도달라 원격CE와의차이를 순수로스효과로 주장하지 않는다.',
        '- 최고epoch를고른draw0가최종3draw에포함돼있다. 개발화자2명·seed42의선별결과이며 독립test01은읽지않았다. 새소음유형·방해말소리·실서비스거절성능은미측정이다.', '',
        '## 재현·감사','',
        f'모든6600개epoch행과12개선택checkpoint의저장예측을확인했다. 원본/혼합예측에서F1을독립재집계했고, 학습trace1100쌍이같음을검증했다. 실행중소스의해시도프로토콜과일치한다.', '',
        '[화자·모델별CSV](comparison.csv) · [전체epoch기록](epoch_histories.csv.gz) · [학습예산진단](epoch_budget_diagnostics.csv) · [전체감사](summary_audit.json) · [추론시간](inference_latency.json) · [기존예측검증](result_checks.json)', '',
        '각실행의 protocol.json 및 summary.json은 reports/compact-{noise,noise-speaker}-{spk03-e1000,spk02-e100}/ 아래에 있다. 체크포인트·전체학습기록·증강trace·예측은동일이름의 runs/에 보관한다. 이번정리에서추가학습은없었다.','']
    (REPORT/'README.md').write_text('\n'.join(lines))
    print(json.dumps(audit,ensure_ascii=False))
    print(json.dumps(timings,ensure_ascii=False))


if __name__=='__main__':main()
