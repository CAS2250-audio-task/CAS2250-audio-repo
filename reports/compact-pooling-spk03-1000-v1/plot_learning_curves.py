"""Posthoc full1000-epoch curves; training and checkpoint selection are unchanged."""
import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
RUN=ROOT/'runs'/OUT.name
COLORS={'cnn_mean':('Mean pooling','#167d9a'),'cnn_pooled_query':('Pooled-query attention','#db8a21'),'cnn_self_cls':('Self-attention + CLS','#9b4bc0')}

def main():
    rows=pd.DataFrame(json.loads((RUN/'spk03/history.json').read_text()))
    summary=json.loads((OUT/'summary.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(3,2,figsize=(13,11),sharex=True)
    specs=[('train_loss','Training cross-entropy',1),('train_accuracy','Training accuracy on augmented inputs (%)',100),
           ('original_f1','Original validation macro F1 (%)',100),('zero_f1','0 dB validation macro F1: draw0 (%)',100),
           ('zero_loss','0 dB validation cross-entropy: draw0',1),('lr','Learning rate (log scale)',1)]
    for name,(label,color) in COLORS.items():
        r=rows[rows.model.eq(name)].sort_values('epoch')
        chosen=next(s for s in summary if s['model']==name)
        for ax,(key,title,scale) in zip(axes.flat,specs):
            ax.plot(r.epoch,r[key]*scale,color=color,alpha=.18,lw=.65)
            values=r[key] if key=='lr' else r[key].rolling(20,min_periods=1).mean()
            ax.plot(r.epoch,values*scale,color=color,lw=1.8,label=label)
            ax.axvline(100,color='#555555',ls='--',lw=1,alpha=.7)
            ax.set_title(title);ax.grid(alpha=.15);ax.set_xlim(1,1000)
            if key in ['original_f1','zero_f1']:
                point=r[r.epoch.eq(chosen['best_epoch'])].iloc[0]
                ax.scatter([point.epoch],[point[key]*100],marker='*',s=65,color=color,zorder=5,edgecolor='white',linewidth=.4)
            if key.endswith('f1') or key=='train_accuracy':ax.set_ylim(0,101)
            if key=='lr':ax.set_yscale('log')
    for ax in axes[-1]:ax.set_xlabel('Total epoch')
    fig.suptitle('spk03 holdout: common CNN, continuation from epoch100 LAST',fontsize=15,y=.995)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.965),ncol=3,frameon=False)
    fig.text(.5,.012,'Bold: trailing 20-epoch mean (except LR). Faint: raw epochs. Dashed: continuation starts after epoch100.\nStars: selected checkpoint. The 0 dB curves use draw0; final scores average three draws. Development results, one speaker and one seed.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.05,1,.94])
    for ext in ['png','pdf']:fig.savefig(OUT/('learning_curves.'+ext),dpi=170,bbox_inches='tight')
    statistics=[]
    for name in COLORS:
        r=rows[rows.model.eq(name)]
        windows={}
        for low,high in [(81,100),(181,200),(481,500),(781,800),(981,1000)]:
            w=r[r.epoch.between(low,high)]
            windows[f'{low}_{high}']={key:float(w[key].mean()) for key in ['train_loss','train_accuracy','original_f1','zero_f1','zero_loss']}
        statistics.append(dict(model=name,windows=windows))
    (OUT/'learning_curve_analysis.json').write_text(json.dumps(statistics,indent=2)+'\n')
    print('CURVES_COMPLETE')

if __name__=='__main__':main()
