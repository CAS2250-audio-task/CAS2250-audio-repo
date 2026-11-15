"""Posthoc learning curves. Does not alter training code, checkpoints or selection."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
MODELS={'cnn_mean':('Mean pooling','#167d9a'),'cnn_pooled_query':('Pooled-query attention','#db8a21'),'cnn_self_cls':('Self-attention + CLS','#9b4bc0')}
summary=json.loads((OUT/'summary.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,4,figsize=(17,8),sharex=True)
statistics=[]
for row,speaker in enumerate(['spk02','spk03']):
    history=pd.DataFrame(json.loads((ROOT/'runs/compact-pooling-v1'/speaker/'history.json').read_text()))
    for name,(label,color) in MODELS.items():
        data=history[history.model.eq(name)].sort_values('epoch')
        chosen=next(x for x in summary if x['speaker']==speaker and x['model']==name)
        for col,(key,title,scale) in enumerate([
            ('train_loss','Training loss (augmented inputs)',1),
            ('train_accuracy','Training accuracy (augmented inputs)',100),
            ('original_f1','Validation F1: original recordings',100),
            ('zero_f1','Validation F1: 0 dB, fixed draw 0',100)]):
            ax=axes[row,col]
            value=data[key]*scale
            ax.plot(data.epoch,value,color=color,alpha=.18,linewidth=.8)
            ax.plot(data.epoch,value.rolling(5,min_periods=1).mean(),color=color,linewidth=2,label=label)
            if col>=2:
                point=data[data.epoch.eq(chosen['best_epoch'])].iloc[0]
                ax.scatter([point.epoch],[point[key]*scale],s=60,marker='*',color=color,zorder=5,edgecolors='white',linewidths=.4)
            ax.set_title(title)
            ax.grid(alpha=.15)
            ax.set_xlim(1,100)
            if col>=1:ax.set_ylim(0,101)
            if row==1:ax.set_xlabel('Epoch')
        best_original=data.loc[data.original_f1.idxmax()]
        early=data[data.epoch.between(61,80)]
        late=data[data.epoch.between(81,100)]
        statistics.append(dict(speaker=speaker,model=name,selected_epoch=chosen['best_epoch'],
            selected_original_f1=chosen['best']['original_f1'],
            maximum_original_f1=float(best_original.original_f1),maximum_original_epoch=int(best_original.epoch),
            mean_train_accuracy_last20=float(late.train_accuracy.mean()),
            zero_f1_last20=float(late.zero_f1.mean()),zero_f1_previous20=float(early.zero_f1.mean()),
            original_f1_last20=float(late.original_f1.mean()),original_f1_previous20=float(early.original_f1.mean())))
    axes[row,0].set_ylabel(speaker+' holdout\nCross-entropy')
    for col in [1,2,3]:axes[row,col].set_ylabel('%')
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.955),ncol=3,frameon=False)
fig.suptitle('Shared CNN, random initialization, noise + weak formant augmentation',fontsize=16,y=.995)
fig.text(.5,.012,'Bold: trailing 5-epoch mean. Faint: raw values. Stars: checkpoint selected using 0 dB validation.\nOriginal recordings have no added evaluation noise; they are not guaranteed noise-free. Separate speaker holdouts; development results.',ha='center',fontsize=10)
fig.tight_layout(rect=[0,.07,1,.92])
for extension in ['png','pdf']:
    fig.savefig(OUT/('learning_curves.'+extension),dpi=170,bbox_inches='tight')
(OUT/'learning_curve_analysis.json').write_text(json.dumps(statistics,ensure_ascii=False,indent=2)+'\n')
plt.close(fig)
# A larger focused image for readable inline display of the requested original/0dB curves.
fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True)
for row,speaker in enumerate(['spk02','spk03']):
    history=pd.DataFrame(json.loads((ROOT/'runs/compact-pooling-v1'/speaker/'history.json').read_text()))
    for name,(label,color) in MODELS.items():
        data=history[history.model.eq(name)].sort_values('epoch')
        for col,key in enumerate(['original_f1','zero_f1']):
            ax=axes[row,col]
            ax.plot(data.epoch,data[key]*100,color=color,alpha=.17,lw=.7)
            ax.plot(data.epoch,data[key].rolling(5,min_periods=1).mean()*100,color=color,lw=2,label=label)
            ax.grid(alpha=.15);ax.set_ylim(0,100);ax.set_xlim(1,100)
            ax.set_title(speaker+(' | Original: no added noise' if col==0 else ' | 0 dB: fixed draw 0'))
            if col==0:ax.set_ylabel('Validation macro F1 (%)')
            if row==1:ax.set_xlabel('Epoch')
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.955),ncol=3,frameon=False)
fig.suptitle('Validation learning curves: original recordings vs added noise',y=.995,fontsize=14)
fig.text(.5,.012,'Bold: trailing 5-epoch mean. Faint: raw values. One training seed; two development speaker holdouts.',ha='center',fontsize=10)
fig.tight_layout(rect=[0,.035,1,.92])
for extension in ['png','pdf']:
    fig.savefig(OUT/('validation_curves.'+extension),dpi=170,bbox_inches='tight')
print(json.dumps(statistics,indent=2))
