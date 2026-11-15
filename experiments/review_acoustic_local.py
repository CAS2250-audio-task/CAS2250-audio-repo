"""Recompute auxiliary comparison, class tradeoffs, paired outcomes and source/trace audit."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/compact-acoustic-local-v1'
NAMES=['cnn_mean','cnn_pooled_query','cnn_self_cls']
LABELS=['출발','정지','전진','후진','왼쪽','오른쪽','시작','복귀','대기','재시작','silence']


def main():
    rows=[];classes=[];paired=[];histories=[]
    for condition,name in [('local','compact-local-local-spk03-e1000'),('aux','compact-acoustic-local-spk03-e1000')]:
        report=ROOT/'reports'/name
        for model in NAMES:
            p=pd.read_csv(report/(model+'_best_predictions.csv.gz'));assert len(p)==5075
            noisy=p[p.kind=='mixed_0db'];original=p[p.kind=='original'];values=[]
            for _,g in noisy.groupby(['noise_id','draw']):
                cm=np.bincount(g.true_idx.to_numpy()*11+g.pred_idx.to_numpy(),minlength=121).reshape(11,11)
                den=cm.sum(0)+cm.sum(1);values.append(np.divide(2*cm.diagonal(),den,out=np.zeros(11),where=den!=0))
            f1=np.mean(values,axis=0)
            record=next(r for r in json.loads((report/'summary.json').read_text()) if r['model']==model)
            assert abs(f1.mean()-record['best']['zero_f1'])<1e-6
            def confusion(a,b):return int(((noisy.true_idx==a)&(noisy.pred_idx==b)).sum())
            rows.append(dict(condition=condition,model=model,best_epoch=record['best_epoch'],original_f1=record['best']['original_f1'],zero_f1=f1.mean(),command_f1=f1[:10].mean(),silence_f1=f1[10],original_accuracy=original.true_idx.eq(original.pred_idx).mean(),noise_accuracy=noisy.true_idx.eq(noisy.pred_idx).mean(),noise_only_fpr=record['noise_only_fpr'],forward_to_stop=confusion(2,1),stop_to_forward=confusion(1,2),restart_to_start=confusion(9,6),command_to_silence=int(((noisy.true_idx!=10)&(noisy.pred_idx==10)).sum())))
            for i,label in enumerate(LABELS):
                q=noisy[noisy.true_idx==i];classes.append(dict(condition=condition,model=model,label=label,f1=f1[i],recall=q.pred_idx.eq(i).mean(),errors=int(q.pred_idx.ne(i).sum())))
            if condition=='aux':
                base=pd.read_csv(ROOT/'reports/compact-local-local-spk03-e1000'/(model+'_best_predictions.csv.gz'))
                both=base.merge(p,on=['path','kind','noise_id','draw','true_idx'],suffixes=('_base','_aux'),validate='one_to_one')
                for kind,g in both.groupby('kind'):
                    bc=g.pred_idx_base.eq(g.true_idx);ac=g.pred_idx_aux.eq(g.true_idx)
                    paired.append(dict(model=model,kind=kind,fixed=int((~bc&ac).sum()),new_errors=int((bc&~ac).sum())))
    pd.DataFrame(rows).to_csv(OUT/'review_metrics.csv',index=False);pd.DataFrame(classes).to_csv(OUT/'review_classes.csv',index=False);pd.DataFrame(paired).to_csv(OUT/'review_paired_changes.csv',index=False)
    folder=ROOT/'runs/compact-acoustic-local-spk03-e1000/spk03'
    h=pd.DataFrame(json.loads((folder/'history.json').read_text()));assert len(h)==3000 and np.isfinite(h[['train_loss','zero_f1']]).all().all()
    for model in NAMES:
        g=h[h.model==model];record=next(r for r in rows if r['condition']=='aux' and r['model']==model)
        assert len(g)==1000 and int(g.loc[g.zero_f1.idxmax(),'epoch'])==record['best_epoch']
    proto=json.loads((ROOT/'reports/compact-acoustic-local-spk03-e1000/protocol.json').read_text())
    for name,digest in proto['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    baseline=Path('/root/CAS2250-compact-local-features/runs/compact-local-local-spk03-e1000/spk03')
    for i in range(1,1001):
        name=f'augmentation_{i:03d}.json';assert hashlib.sha256((folder/name).read_bytes()).digest()==hashlib.sha256((baseline/name).read_bytes()).digest()
    (OUT/'review_audit.json').write_text(json.dumps(dict(histories_checked=3,model_epochs=3000,prediction_sets_recomputed=6,paired_rows=15225,source_hashes_matched=len(proto['source_sha256']),augmentation_epochs_matched=1000),indent=2)+'\n')
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(pd.DataFrame(classes).query("label in ['정지','전진','후진','재시작']").round(4).to_string(index=False))
    print(pd.DataFrame(paired).to_string(index=False))


if __name__=='__main__':main()
