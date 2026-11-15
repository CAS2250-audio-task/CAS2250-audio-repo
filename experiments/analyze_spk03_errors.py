"""Paired file-level analysis of all three compact models and all three objectives."""
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/compact-contrastive-launch/spk03-errors'
MODELS=('cnn_mean','cnn_pooled_query','cnn_self_cls')
CONDITIONS=('ce','noise','noise-speaker')
DISPLAY={'cnn_mean':'Mean','cnn_pooled_query':'Pooled','cnn_self_cls':'Self+CLS'}
KEYS=['path','kind','noise_id','draw']


def metric(frame):
    cm=np.bincount(frame.true_idx.to_numpy()*11+frame.pred_idx.to_numpy(),minlength=121).reshape(11,11)
    den=cm.sum(0)+cm.sum(1)
    return float(np.divide(2.*cm.diagonal(),den,out=np.zeros(11),where=den!=0).mean())


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    labels=json.loads((ROOT/'configs/task2_real.json').read_text())['labels']
    label_map=dict(enumerate(labels));frames={};summaries=[];source_hashes={}
    wrongs=[];confusions=[];class_rows=[];noise_rows=[];file_rows=[];original_rows=[]
    for condition in CONDITIONS:
        if condition=='ce':
            summary=json.loads((OUT/'ce_reference/summary.json').read_text())
        else:
            summary=json.loads((ROOT/f'reports/compact-{condition}-spk03-e1000/summary.json').read_text())
        for model in MODELS:
            source=OUT/'ce_reference'/f'{model}_best_predictions.csv.gz' if condition=='ce' else ROOT/f'reports/compact-{condition}-spk03-e1000/{model}_best_predictions.csv.gz'
            source_hashes[f'{condition}/{model}']=hashlib.sha256(source.read_bytes()).hexdigest()
            p=pd.read_csv(source)
            assert len(p)==5075 and not p.duplicated(KEYS).any()
            assert len(p[p.kind=='original'])==203 and len(p[p.kind=='mixed_0db'])==4872
            assert p.path.str.contains('spk03_').all()
            p['true_label']=p.true_idx.map(label_map);p['pred_label']=p.pred_idx.map(label_map)
            p['wrong']=p.true_idx.ne(p.pred_idx)
            p['condition']=condition;p['model']=model
            frames[(condition,model)]=p
            original=p[p.kind=='original'];noisy=p[p.kind=='mixed_0db'];bad=noisy[noisy.wrong]
            r=next(v for v in summary if v['model']==model)
            original_f1=metric(original)
            noise_f1=float(np.mean([metric(g) for _,g in noisy.groupby(['noise_id','draw'])]))
            assert abs(original_f1-r['best']['original_f1'])<1e-6
            assert abs(noise_f1-r['best']['zero_f1'])<1e-6
            summaries.append(dict(condition=condition,model=model,best_epoch=r['best_epoch'],original_f1=original_f1,
                original_errors=int(original.wrong.sum()),original_total=203,noise_f1=noise_f1,
                noisy_errors=int(noisy.wrong.sum()),noisy_total=4872,noise_only_fpr=r['noise_only_fpr']))
            wrongs.append(p[p.wrong]);original_rows.append(original)
            for (true,pred),g in bad.groupby(['true_label','pred_label']):
                confusions.append(dict(condition=condition,model=model,true_label=true,pred_label=pred,errors=len(g)))
            for true,g in noisy.groupby('true_label'):
                class_rows.append(dict(condition=condition,model=model,true_label=true,total=len(g),errors=int(g.wrong.sum()),error_rate=float(g.wrong.mean())))
            for noise,g in noisy.groupby('noise_id'):
                score=float(np.mean([metric(d) for _,d in g.groupby('draw')]))
                noise_rows.append(dict(condition=condition,model=model,noise_id=noise,total=len(g),errors=int(g.wrong.sum()),error_rate=float(g.wrong.mean()),macro_f1=score))
            for path,g in noisy.groupby('path'):
                counts=g[g.wrong].pred_label.value_counts()
                file_rows.append(dict(condition=condition,model=model,path=path,true_label=g.true_label.iloc[0],
                   wrong_conditions=int(g.wrong.sum()),conditions=24,
                   original_pred=original[original.path==path].pred_label.iloc[0],
                   wrong_prediction_counts=json.dumps(counts.to_dict(),ensure_ascii=False)))
    summary=pd.DataFrame(summaries);conf=pd.DataFrame(confusions);classes=pd.DataFrame(class_rows);noises=pd.DataFrame(noise_rows);files=pd.DataFrame(file_rows)
    summary.to_csv(OUT/'summary.csv',index=False);conf.to_csv(OUT/'confusions.csv',index=False)
    classes.to_csv(OUT/'class_errors.csv',index=False);noises.to_csv(OUT/'noise_errors.csv',index=False)
    files.sort_values('wrong_conditions',ascending=False).to_csv(OUT/'file_robustness.csv',index=False)
    pd.concat(wrongs).to_csv(OUT/'all_wrong_predictions.csv.gz',index=False)
    original=pd.concat(original_rows)
    original[original.wrong].to_csv(OUT/'original_wrong_predictions.csv',index=False)
    original.pivot(index=['path','true_label'],columns=['model','condition'],values='pred_label').to_csv(OUT/'original_all_nine_predictions.csv')
    paired=[];pair_details=[]
    for model in MODELS:
        for before,after in [('ce','noise'),('ce','noise-speaker'),('noise','noise-speaker')]:
            p=frames[(before,model)].merge(frames[(after,model)],on=KEYS,validate='one_to_one',suffixes=('_before','_after'))
            assert len(p)==5075 and (p.true_idx_before==p.true_idx_after).all()
            p['change']=np.select([p.wrong_before & ~p.wrong_after,~p.wrong_before & p.wrong_after,p.wrong_before & p.wrong_after],['fixed','broken','both_wrong'],default='both_correct')
            p['comparison']=before+' -> '+after
            pair_details.append(p[p.change.isin(['fixed','broken'])][KEYS+['model_before','comparison','true_label_before','pred_label_before','pred_label_after','change']])
            for kind,g in p.groupby('kind'):
                fixed=int(g.change.eq('fixed').sum());broken=int(g.change.eq('broken').sum())
                paired.append(dict(model=model,before=before,after=after,kind=kind,fixed=fixed,broken=broken,net_correct_gain=fixed-broken))
    pd.DataFrame(paired).to_csv(OUT/'paired_changes.csv',index=False)
    pd.concat(pair_details).to_csv(OUT/'paired_changed_predictions.csv.gz',index=False)
    common_original=set.intersection(*[set(p[(p.kind=='original')&p.wrong].path) for p in frames.values()])
    fig,axes=plt.subplots(1,3,figsize=(14,4.4),sharey=True)
    for ax,model in zip(axes,MODELS):
        for i,condition in enumerate(CONDITIONS):
            g=noises[(noises.model==model)&(noises.condition==condition)].set_index('noise_id').reindex(sorted(noises.noise_id.unique()))
            ax.bar(np.arange(len(g))+(i-1)*.24,g.error_rate*100,.24,label=condition)
        ax.set(title=DISPLAY[model],ylabel='Error rate (%)',xticks=np.arange(8),xticklabels=sorted(noises.noise_id.unique()),ylim=(0,45))
        ax.tick_params(axis='x',rotation=55);ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'noise_breakdown.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4.4),sharey=True)
    for ax,model in zip(axes,MODELS):
        p=frames[('noise-speaker',model)].query('kind=="mixed_0db"')
        cm=np.bincount(p.true_idx.to_numpy()*11+p.pred_idx.to_numpy(),minlength=121).reshape(11,11)
        cm=cm/cm.sum(1,keepdims=True)
        ax.imshow(cm,vmin=0,vmax=1,cmap='Blues')
        # English labels keep plot portable; Korean labels are in the CSV and report.
        words=['go','stop','forward','backward','left','right','start','return','wait','restart','silence']
        ax.set(title=DISPLAY[model]+' / noise-speaker',xticks=np.arange(11),xticklabels=words,yticks=np.arange(11),yticklabels=words,xlabel='Prediction',ylabel='True')
        ax.tick_params(axis='x',rotation=70)
    fig.tight_layout();fig.savefig(OUT/'speaker_confusions.png',dpi=160);plt.close(fig)
    lines=['# 화자03: 세 모델의 오답 파일·명령어·소음 분석','',
      '**Mean / Pooled / Self+CLS 각각의 CE·소음대조·화자포함대조, 총9조합을 모두 같은 파일·소음 조건으로 비교했다.** CE는 새1000epoch 릴리스의 예측이며 모든 모델의 선택checkpoint 결과다.','',
      '원본은203개다. 소음 평가는 같은203개에8소음×3crop을 반복한4,872조건이며, 서로 다른발화4,872개라는 뜻은 아니다. 표의오답수는조건별건수, F1은소음별·crop별macro F1의평균이다.','',
      '## 전체9조합','', '| 모델 | 학습조건 | 원본오답 /203 | 0dB오답 /4872 | 0dB F1 | 잡음단독 오반응 |', '|---|---|---:|---:|---:|---:|']
    for model in MODELS:
        for condition in CONDITIONS:
            row=summary[(summary.model==model)&(summary.condition==condition)].iloc[0]
            lines.append(f'| {DISPLAY[model]} | {condition} | {row.original_errors} | {row.noisy_errors} | {row.noise_f1*100:.2f}% | {row.noise_only_fpr*100:.2f}% |')
    lines += ['',
      '## 모델별 주요 혼동','', '| 모델 | 조건 | 자주 틀린 정답→예측, 건수 |','|---|---|---|']
    for model in MODELS:
        for condition in CONDITIONS:
            top=conf[(conf.model==model)&(conf.condition==condition)].sort_values('errors',ascending=False).head(4)
            text=', '.join(f'{r.true_label}→{r.pred_label} {r.errors}' for r in top.itertuples())
            lines.append(f'| {DISPLAY[model]} | {condition} | {text} |')
    lines += ['',
      '- **Mean:** 전체 소음 오답은908→745→679로 줄었다. 다만 재시작→시작은111→158→127로 CE보다 여전히 많다. 화자 포함에서는 재시작·전진이 주요 취약점이다.',
      '- **Pooled:** 전체 소음 오답1003→800→718과 재시작→시작131→64→43이 모두 개선됐다. 전진→정지118→107→87은 남았고, 화자 포함의 원본 오답은13→16으로 늘었다.',
      '- **Self+CLS:** 소음만 대조는749오답으로 화자 포함770보다 적다. 화자 포함은 재시작→시작131→38을 줄였지만, 오른쪽→왼쪽35→103과 정지→전진50→98이 늘었다. 명령어별 개선과 악화가 함께 관찰됐다.', '',
      '## 어느 소음에서 틀렸나','',
      '화자포함 조건의오답: Mean은restaurant148/609,cafeteria118/609,compressor115/609. Pooled는restaurant158/609,compressor135/609,cafeteria121/609. Self는restaurant202/609,compressor133/609,cafeteria128/609다.', '',
      '식당·구내식당·압축기 녹음에서세모델모두어려웠다. 이평가는소음유형마다원본녹음하나이므로해당유형전체에대한성능으로일반화하지않는다.', '',
      '![소음별 오답률](noise_breakdown.png)', '',
      '## 자주 틀리는 구체적인 파일','',
      '각파일은24가지소음조건에서평가했다. 아래는화자포함조건의모델별상위5개다. 전체9조합은 file_robustness.csv에있다.', '',
      '| 모델 | 파일 | 정답 | 원본예측 | 소음오답 /24 | 오답예측 분포 |','|---|---|---|---|---:|---|']
    for model in MODELS:
        top=files[(files.condition=='noise-speaker')&(files.model==model)].sort_values(['wrong_conditions','path'],ascending=[False,True]).head(5)
        for row in top.itertuples():
            lines.append(f'| {DISPLAY[model]} | {row.path} | {row.true_label} | {row.original_pred} | {row.wrong_conditions} | {row.wrong_prediction_counts} |')
    lines += ['',
      '## 원본부터 틀린 파일','',
      '대표 최고점인Mean+화자포함은원본9개를틀렸다. 다른모델을포함한모든원본오답은 original_wrong_predictions.csv에있다.', '',
      '| 파일 | 정답 | 예측 |','|---|---|---|']
    for row in frames[('noise-speaker','cnn_mean')].query('kind=="original" and wrong').itertuples():
        lines.append(f'| {row.path} | {row.true_label} | {row.pred_label} |')
    lines += ['',
      f'9조합모두원본부터틀리는공통파일은{len(common_original)}개다. 특정파일의녹음·라벨이잘못됐다는결론은예측만으로낼수없다.', '',
      '## 같은 파일에서 개선과 악화','',
      '| 모델 | 전→후 | 원본 수정 / 새오답 | 소음 수정 / 새오답 |','|---|---|---|---|']
    pairs=pd.DataFrame(paired)
    for model in MODELS:
        for before,after in [('ce','noise'),('ce','noise-speaker'),('noise','noise-speaker')]:
            g=pairs[(pairs.model==model)&(pairs.before==before)&(pairs.after==after)].set_index('kind')
            lines.append(f'| {DISPLAY[model]} | {before}→{after} | {g.loc["original","fixed"]} / {g.loc["original","broken"]} | {g.loc["mixed_0db","fixed"]} / {g.loc["mixed_0db","broken"]} |')
    lines += ['',
      '예를들어Mean의CE→화자포함은소음조건460개를고쳤고, 맞던231개를새로틀려순229개정답이늘었다. 전체점수향상과개별명령어악화가동시에발생한다.', '',
      '## 원인에 대해 말할 수 있는 범위','',
      '- 재시작/시작은문자열상접두부분이달라, 약한초반음절의단서를놓치는지확인할가설이있다. 그러나현재예측만으로재음절이실제로잡음에가려졌거나클리핑됐다고확정하지않는다.',
      '- 전진/정지, 오른쪽/왼쪽의혼동은중요하지만, attention이실제로어느구간에집중했는지·확신도가얼마였는지는저장예측CSV에없다. 주의지도나소음제거기능을이표에서추론하지않는다.',
      '- 원본오답과소음에서만생긴오답을분리했다. 소음조건반복과단일화자·seed1개의한계를유지한다. 전후비교는동일파일조건이며, CE와대조학습은입력노출량·LR스케줄·torch환경이다르므로로스하나의인과효과는아니다.', '',
      '![화자포함 조건 혼동행렬](speaker_confusions.png)', '',
      '## 자료','',
      '[9조합 요약](summary.csv) · [모든 혼동쌍](confusions.csv) · [명령어별 오류](class_errors.csv) · [소음별 오류](noise_errors.csv) · [파일별24조건 오류](file_robustness.csv)',
      '[모든 오답](all_wrong_predictions.csv.gz) · [원본 오답](original_wrong_predictions.csv) · [원본9조합 예측](original_all_nine_predictions.csv) · [수정/새오답 집계](paired_changes.csv) · [바뀐 개별 예측](paired_changed_predictions.csv.gz)',
      '[CE 예측 출처·해시](ce_reference/provenance.json) · [분석 검사](analysis_audit.json)', '',
      '추가학습·새test평가는하지않고, 저장된선택checkpoint예측을재분석했다. 음성을청취완료했다고주장하지않는다.']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    audit=dict(models=3,objectives=3,combinations=9,unique_original_clips=203,original_predictions=1827,
               noisy_predictions=43848,paired_comparisons=9,one_to_one_pairing_verified=True,
               nine_original_and_noise_f1_scores_recomputed=True,source_prediction_sha256=source_hashes,
               common_original_wrong_all9=sorted(common_original),new_training=False,new_test=False)
    (OUT/'analysis_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(summary.to_string(index=False))


if __name__=='__main__':main()
