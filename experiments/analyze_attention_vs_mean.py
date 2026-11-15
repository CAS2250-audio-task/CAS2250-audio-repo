"""Pair local-route model predictions to describe attention gains and regressions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'reports/compact-local-local-spk03-e1000'
OUT=ROOT/'reports/compact-local-features/attention-vs-mean'
LABELS=['출발','정지','전진','후진','왼쪽','오른쪽','시작','복귀','대기','재시작','silence']
MODELS=['cnn_mean','cnn_pooled_query','cnn_self_cls']
KEYS=['path','kind','noise_id','draw','true_idx']


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    frames={m:pd.read_csv(SOURCE/f'{m}_best_predictions.csv.gz') for m in MODELS}
    merged=frames[MODELS[0]].rename(columns={'pred_idx':MODELS[0]})
    for m in MODELS[1:]:
        merged=merged.merge(frames[m].rename(columns={'pred_idx':m}),on=KEYS,validate='one_to_one')
    assert len(merged)==5075 and not merged.duplicated(KEYS).any()
    merged['label']=merged.true_idx.map(dict(enumerate(LABELS)))
    for m in MODELS:merged[m+'_label']=merged[m].map(dict(enumerate(LABELS)))
    changes=[];classes=[];noises=[];files=[];confusions=[];totals=[]
    for m in MODELS[1:]:
        q=merged.copy();q['attention']=m
        mc=q.cnn_mean.eq(q.true_idx);ac=q[m].eq(q.true_idx)
        q['outcome']=np.select([~mc & ac,mc & ~ac,mc & ac],['fixed','new_error','both_correct'],default='both_wrong')
        q['mean_correct']=mc;q['attention_correct']=ac
        changes.append(q)
        for kind,g in q.groupby('kind'):
            fixed=int(g.outcome.eq('fixed').sum());broke=int(g.outcome.eq('new_error').sum())
            totals.append(dict(attention=m,kind=kind,n=len(g),mean_errors=int((~g.mean_correct).sum()),attention_errors=int((~g.attention_correct).sum()),fixed=fixed,new_errors=broke,net_fewer_errors=fixed-broke,mean_accuracy=g.mean_correct.mean(),attention_accuracy=g.attention_correct.mean()))
            for label,c in g.groupby('label',sort=False):
                classes.append(dict(attention=m,kind=kind,label=label,n=len(c),mean_correct=int(c.mean_correct.sum()),attention_correct=int(c.attention_correct.sum()),fixed=int(c.outcome.eq('fixed').sum()),new_errors=int(c.outcome.eq('new_error').sum()),accuracy_delta_pp=100*(c.attention_correct.mean()-c.mean_correct.mean())))
        noisy=q[q.kind=='mixed_0db']
        for noise,g in noisy.groupby('noise_id'):
            noises.append(dict(attention=m,noise_id=noise,n=len(g),mean_accuracy=g.mean_correct.mean(),attention_accuracy=g.attention_correct.mean(),accuracy_delta_pp=100*(g.attention_correct.mean()-g.mean_correct.mean()),fixed=int(g.outcome.eq('fixed').sum()),new_errors=int(g.outcome.eq('new_error').sum())))
        for path,g in noisy.groupby('path'):
            files.append(dict(attention=m,path=path,label=g.label.iloc[0],n=len(g),mean_correct=int(g.mean_correct.sum()),attention_correct=int(g.attention_correct.sum()),fixed=int(g.outcome.eq('fixed').sum()),new_errors=int(g.outcome.eq('new_error').sum()),net_correct=int(g.attention_correct.sum()-g.mean_correct.sum())))
        for model in ['cnn_mean',m]:
            wrong=noisy[noisy[model]!=noisy.true_idx]
            for (true,pred),n in wrong.groupby(['true_idx',model]).size().items():
                confusions.append(dict(attention_comparison=m,model=model,true_label=LABELS[true],pred_label=LABELS[pred],count=int(n)))
    tables=dict(summary=pd.DataFrame(totals),class_changes=pd.DataFrame(classes),noise_changes=pd.DataFrame(noises),file_changes=pd.DataFrame(files),confusions=pd.DataFrame(confusions))
    for name,table in tables.items():table.to_csv(OUT/(name+'.csv'),index=False)
    cases=pd.concat(changes,ignore_index=True)
    cases.to_csv(OUT/'all_paired_predictions.csv.gz',index=False)
    cases[(cases.kind=='original') & cases.outcome.isin(['fixed','new_error'])].to_csv(OUT/'original_changed_cases.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,table,key,title in [(axes[0],tables['class_changes'].query("kind=='mixed_0db'"),'label','By command: accuracy change (pp)'),(axes[1],tables['noise_changes'],'noise_id','By noise: accuracy change (pp)')]:
        pivot=table.pivot(index=key,columns='attention',values='accuracy_delta_pp')
        if key=='label':pivot=pivot.reindex(LABELS)
        pivot.columns=['pooled','self'];pivot.index=([str(i)+': '+name for i,name in enumerate(['go','stop','forward','backward','left','right','start','return','wait','restart','silence'])] if key=='label' else pivot.index)
        image=ax.imshow(pivot.values,cmap='RdBu',vmin=-20,vmax=20,aspect='auto')
        ax.set_xticks(range(2),pivot.columns);ax.set_yticks(range(len(pivot)),pivot.index);ax.set_title(title)
        for i in range(len(pivot)):
            for j in range(2):ax.text(j,i,f'{pivot.iloc[i,j]:+.1f}',ha='center',va='center',fontsize=9)
    fig.colorbar(image,ax=axes.tolist(),label='Attention accuracy minus mean (percentage points)',shrink=.7)
    fig.savefig(OUT/'accuracy_changes.png',dpi=160,bbox_inches='tight');plt.close(fig)
    lines=['# 중간 특징 전달 모델: 어텐션 추가의 정답/오답 변화','',
        '화자03, local 구조, 소음+화자 SupCon, seed42. Mean806/Pooled565/Self508epoch의 소음 검증 최고 체크포인트를 비교했다. 각 입력의 예측을1:1로 짝지었다.', '',
        '주의: 같은 CNN 구조·초기값에서 시작했지만 각 모델의 CNN도 독립 학습됐고 checkpoint epoch도 다르다. 아래는 완성 모델 간 차이이며 어텐션만 교체한 인과효과가 아니다. Self는 위치인코딩·CLS·FFN도 추가한다. 입력 길이1.5초, 추가잡음0dB만 분석했으므로 다른SNR/음성길이 일반화는 알 수 없다.', '',
        '## 정답률과 교체 수','', '| 모델 | 조건 | 평균 오답 | 어텐션 오답 | 기존 오답 수정 | 새 오답 | 평균 정답률 | 어텐션 정답률 |','|---|---|---:|---:|---:|---:|---:|---:|']
    for row in totals:
        lines.append(f'| {row["attention"]} | {row["kind"]} | {row["mean_errors"]} | {row["attention_errors"]} | {row["fixed"]} | {row["new_errors"]} | {row["mean_accuracy"]*100:.2f}% | {row["attention_accuracy"]*100:.2f}% |')
    lines+=['','정답률(accuracy)과 기존 보고서의 클래스별 macro F1은 서로 다른 지표다. 소음4872조건은203녹음×8소음×3draw이며 독립4872녹음이 아니다.', '', '## 명령어별 소음 정답률 변화','', '| 명령어 | Mean 정답률 | Pooled 변화 | Self 변화 |','|---|---:|---:|---:|']
    for label in LABELS:
        a=tables['class_changes'];a=a[(a.kind=='mixed_0db')&(a.label==label)].set_index('attention')
        first=a.iloc[0]
        lines.append(f'| {label} | {100*first.mean_correct/first.n:.2f}% | {a.loc[MODELS[1],"accuracy_delta_pp"]:+.2f}%p | {a.loc[MODELS[2],"accuracy_delta_pp"]:+.2f}%p |')
    lines+=['','## 소음별 정답률 변화','', '| 소음 | Pooled 변화 | Self 변화 |','|---|---:|---:|']
    for noise in sorted(tables['noise_changes'].noise_id.unique()):
        a=tables['noise_changes'].query('noise_id==@noise').set_index('attention')
        lines.append(f'| {noise} | {a.loc[MODELS[1],"accuracy_delta_pp"]:+.2f}%p | {a.loc[MODELS[2],"accuracy_delta_pp"]:+.2f}%p |')
    lines+=['','## 반복적으로 개선/악화되는 녹음','', '| 모델 | 구분 | 파일 | Mean 정답/24 | Attention 정답/24 |','|---|---|---|---:|---:|']
    for m in MODELS[1:]:
        f=tables['file_changes'].query('attention==@m')
        for kind,part in [('개선',f.sort_values(['net_correct','path'],ascending=[False,True]).head(3)),('악화',f.sort_values(['net_correct','path']).head(3))]:
            for _,row in part.iterrows():lines.append(f'| {m} | {kind} | {row.path} | {row.mean_correct} | {row.attention_correct} |')
    lines+=['','## 해석 범위','',
        '어텐션은 시간별 비중을 학습할 수 있지만 항상 유효 발화에 집중한다는 보장은 없다. 짧은 구별 구간을 선택하는 이득과 소음/다른 발음 단서에 비중을 주는 손실이라는 설명은 가능한 가설이며, 해당 구간 청취·정렬로 증명한 결과는 아니다. CNN 특징은 이미 넓은 문맥을 담아 최종 평균도 시간 정보를 간접 활용한다.', '',
        '잡음만 입력했을 때 오반응은Mean4.58%,Pooled11.88%,Self12.08%였다. 아래 혼합음성 정답률과 별도 조건이며, 어텐션이 잡음을 자동으로 더 잘 무시한다고 해석하면 안 된다.', '',
        '확인 실험이 필요하면 같은 학습된 CNN을 고정하고 head만 학습하는 비교로 범위를 좁힐 수 있다. 현재는 추가 학습하지 않았다. 동일 개발검증의 결과이며test는 아니다.', '',
        '파일: summary.csv, class_changes.csv, noise_changes.csv, file_changes.csv, confusions.csv, original_changed_cases.csv, all_paired_predictions.csv.gz. 그림의 명령어는 표와 같은 순서의 영어 label이다.', '', '![정답률 변화](accuracy_changes.png)']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))

def metric_tradeoff():
    q=pd.read_csv(OUT/'all_paired_predictions.csv.gz')
    q=q[(q.attention=='cnn_pooled_query')&(q.kind=='mixed_0db')]
    records=[];trade=[]
    for model in MODELS:
        units=[]
        for _,g in q.groupby(['noise_id','draw']):
            cm=np.bincount(g.true_idx.to_numpy()*11+g[model].to_numpy(),minlength=121).reshape(11,11)
            def div(a,b):return np.divide(a,b,out=np.zeros(11),where=b!=0)
            units.append(np.array([div(cm.diagonal(),cm.sum(0)),div(cm.diagonal(),cm.sum(1)),div(2*cm.diagonal(),cm.sum(0)+cm.sum(1))]))
        avg=np.mean(units,axis=0)
        for i,label in enumerate(LABELS):records.append(dict(model=model,label=label,precision=avg[0,i],recall=avg[1,i],f1=avg[2,i]))
        trade.append(dict(model=model,command_to_silence=int(((q.true_idx!=10)&(q[model]==10)).sum()),silence_to_command=int(((q.true_idx==10)&(q[model]!=10)).sum()),command_only_macro_f1_11class_context=float(avg[2,:10].mean()),silence_f1=float(avg[2,10])))
        expected=next(r for r in json.loads((SOURCE/'summary.json').read_text()) if r['model']==model)['best']['zero_f1']
        assert abs(avg[2].mean()-expected)<1e-6
    pd.DataFrame(records).to_csv(OUT/'per_class_precision_recall_f1.csv',index=False)
    pd.DataFrame(trade).to_csv(OUT/'silence_tradeoff.csv',index=False)
    lines=['','## 정답률과 macro F1: silence 구분의 영향','',
        '0dB 정답률은Mean87.46%,Pooled86.64%,Self86.15%이나 macro F1은85.44%,86.77%,85.55%다. 정답률은 총정답 비율이고 macro F1은 클래스별 precision/recall을 반영한다. 다음 표는 같은24개noise/draw에서 계산한 클래스별F1을 평균했다.', '',
        '| 모델 | 명령어→silence 오답/4800 | silence→명령어 오답/72 | 명령어10종 평균F1 | silence F1 |',
        '|---|---:|---:|---:|---:|']
    for row in trade:lines.append(f'| {row["model"]} | {row["command_to_silence"]} | {row["silence_to_command"]} | {100*row["command_only_macro_f1_11class_context"]:.2f}% | {100*row["silence_f1"]:.2f}% |')
    lines+=['','명령어10종 평균F1은 동일11클래스 혼동행렬에서silence 항목만 평균에서 제외한 값이며 silence 예측으로 발생한 명령어 오답을 제거하지 않는다. 순수10클래스 재학습/재평가가 아니다.', '',
        '소음에서 어텐션 모델의 전체macro F1 개선은 silence F1 상승에 크게 의존한다. 명령어 구별의 전반적 개선으로 해석하면 안 된다. 어텐션 모델은 명령어를silence로 잘못 보내는 횟수가 감소했지만, 잡음만 입력한 별도 평가에서는 오반응이 증가했다. 클래스별precision/recall/F1은 per_class_precision_recall_f1.csv, 혼동은silence_tradeoff.csv에 저장했다.']
    with (OUT/'README.md').open('a') as f:f.write('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
    metric_tradeoff()
