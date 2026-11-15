import json
from pathlib import Path
import numpy as np,pandas as pd
root=Path.cwd();out=root/'reports/frequency-local-v1';table=pd.read_csv(out/'all18_comparison.csv');labels=json.loads((root/'configs/task2_real.json').read_text())['labels']
classes=[];noises=[];confusions=[];pairs=[];lastrows=[];frames={}
for row in table.itertuples():
 p=root/'runs'/row.study/'spk03'/(row.model+'_best_predictions.csv.gz')
 if not p.exists():p=root/'reports'/row.study/p.name
 f=pd.read_csv(p);f=f[f.kind=='mixed_0db'].copy();frames[(row.cnn,row.vertical,row.model)]=f
 for name,g in f.groupby('true_idx'):
  classes.append(dict(cnn=row.cnn,vertical=row.vertical,model=row.model,command=labels[name],errors=int(g.pred_idx.ne(g.true_idx).sum()),count=len(g)))
 for noise,g in f.groupby('noise_id'):
  noises.append(dict(cnn=row.cnn,vertical=row.vertical,model=row.model,noise=noise,error_rate=g.true_idx.ne(g.pred_idx).mean()))
 for (a,b),n in f[f.true_idx!=f.pred_idx].groupby(['true_idx','pred_idx']).size().items():
  confusions.append(dict(cnn=row.cnn,vertical=row.vertical,model=row.model,true=labels[a],predicted=labels[b],errors=int(n)))
 s=next(r for r in json.loads((root/'reports'/row.study/'summary.json').read_text()) if r['model']==row.model)
 lastrows.append(dict(cnn=row.cnn,vertical=row.vertical,model=row.model,best_epoch=row.best_epoch,best_zero_f1=row.zero_f1,last_zero_f1=s['last']['zero_f1'],gap_pp=100*(s['last']['zero_f1']-row.zero_f1)))
for row in table.itertuples():
 f=frames[(row.cnn,row.vertical,row.model)];ref=frames[('basic',False,row.model)]
 p=f.merge(ref,on=['path','kind','noise_id','draw','true_idx'],suffixes=('_new','_base'),validate='one_to_one');assert len(p)==4872
 base=p.pred_idx_base.eq(p.true_idx);new=p.pred_idx_new.eq(p.true_idx)
 pairs.append(dict(cnn=row.cnn,vertical=row.vertical,model=row.model,fixed=int((~base&new).sum()),new_errors=int((base&~new).sum()),both_wrong=int((~base&~new).sum()),net_fixed=int(new.sum()-base.sum())))
for name,rows in [('class_errors',classes),('noise_errors',noises),('confusions',confusions),('paired_changes',pairs),('last_vs_selected',lastrows)]:pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False)
print('PAIRS\n',pd.DataFrame(pairs).to_string(index=False));print('LAST\n',pd.DataFrame(lastrows).to_string(index=False))
print('BASE POOLED NOISE\n',pd.DataFrame(noises).query("cnn=='basic' and vertical==False and model=='cnn_pooled_query'").sort_values('error_rate',ascending=False).to_string(index=False))
print('BASE POOLED CONFUSIONS\n',pd.DataFrame(confusions).query("cnn=='basic' and vertical==False and model=='cnn_pooled_query'").sort_values('errors',ascending=False).head(10).to_string(index=False))
print('BASE MEAN CONFUSIONS\n',pd.DataFrame(confusions).query("cnn=='basic' and vertical==False and model=='cnn_mean'").sort_values('errors',ascending=False).head(8).to_string(index=False))
print('BASE SELF VERTICAL CONFUSIONS\n',pd.DataFrame(confusions).query("cnn=='basic' and vertical==True and model=='cnn_self_cls'").sort_values('errors',ascending=False).head(8).to_string(index=False))
print('POOLED NOISE ALL\n',pd.DataFrame(noises).query("model=='cnn_pooled_query'").pivot(index='noise',columns=['cnn','vertical'],values='error_rate').round(3).to_string())
print('WORSE_LAST_COUNT',sum(r['gap_pp']<0 for r in lastrows),'MEDIAN_GAP',np.median([r['gap_pp'] for r in lastrows]))
