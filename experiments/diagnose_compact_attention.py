"""Diagnostic clean/noisy weight swaps. Clean references are NOT available at deployment."""
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.nn import functional as F
from experiments.compact_pooling.models import matched_models,position_encoding
from experiments.compact_pooling.run import Frontend,validation_units

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/compact-contrastive-launch/spk03-errors/mechanisms'


def self_data(model,h):
    readout=model.readout
    assert len(readout.blocks)==1
    x=torch.cat((readout.cls.expand(len(h),-1,-1),h),dim=1)
    x=x+position_encoding(x.shape[1],x.shape[2],x.device,x.dtype)
    block=readout.blocks[0];att=block.attention
    def split(layer):return layer(x).view(len(x),x.shape[1],att.heads,att.head_width).transpose(1,2)
    q,k,v=split(att.q),split(att.k),split(att.v)
    weights=torch.softmax((q@k.transpose(-2,-1))/math.sqrt(att.head_width),dim=-1)
    return x,v,weights,q[:,:,0]


def self_output(model,x,v,weights):
    block=model.readout.blocks[0];att=block.attention
    mixed=(weights@v).transpose(1,2).reshape(len(x),x.shape[1],x.shape[2])
    y=block.norm1(x+att.output(mixed))
    y=block.norm2(y+block.ffn(y))
    return y[:,0]


def js(a,b):
    a=a.clamp_min(1e-12);b=b.clamp_min(1e-12);middle=(a+b)/2
    return (.5*(a*(a.log()-middle.log())).sum(-1)+.5*(b*(b.log()-middle.log())).sum(-1)).mean(-1)


def entropy(a):
    return -(a*a.clamp_min(1e-12).log()).sum(-1).mean(-1)/math.log(a.shape[-1])


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    data=ROOT/'datasets/task2/fixed-1p5-v1'
    frame=pd.read_csv(data/'metadata.csv');frame=frame[frame.speaker_id.isin(['spk02','spk03','spk04','spk05'])].reset_index(drop=True)
    assert len(frame)==812 and not frame.path.str.contains('spk01').any()
    labels=json.loads((ROOT/'configs/task2_real.json').read_text())['labels']
    targets=np.array([labels.index(v) for v in frame.label])
    waves=np.stack([sf.read(data/p,dtype='float32')[0] for p in frame.path])
    train_idx=np.flatnonzero(frame.speaker_id.ne('spk03'));val_idx=np.flatnonzero(frame.speaker_id.eq('spk03'))
    reference=float(np.median(np.sqrt(np.mean(waves[train_idx[targets[train_idx]!=10]].astype(np.float64)**2,axis=1)))**2)
    sources=json.loads((ROOT/'runs/compact-pooling-v1/sources.json').read_text())
    noises=[(row['noise_id'],sf.read(ROOT/row['prepared_path'],dtype='float32')[0]) for row in sources['evaluation']]
    models=matched_models();records=[];replay=0;query_equal=True
    frontend=Frontend().cuda().eval()
    with torch.inference_mode():
        for name in ['cnn_pooled_query','cnn_self_cls']:
            model=models[name].cuda().eval()
            saved=torch.load(ROOT/f'runs/compact-noise-speaker-spk03-e1000/spk03/{name}_best.pt',map_location='cuda',weights_only=False)
            model.load_state_dict(saved['model'])
            units=validation_units(frontend,frame,waves,targets,val_idx,noises,saved['mean'],saved['std'],reference,3)
            clean_x=units[0]['x'];clean_h=torch.cat([model.backbone(x) for x in clean_x.split(32)])
            clean_logits=torch.cat([model.classifier(model.readout(h)) for h in clean_h.split(32)])
            stored=pd.read_csv(ROOT/f'reports/compact-noise-speaker-spk03-e1000/{name}_best_predictions.csv.gz')
            clean_pred=clean_logits.argmax(-1).cpu().numpy()
            expected=stored[stored.kind=='original'].set_index('path').loc[frame.iloc[val_idx].path].pred_idx.to_numpy()
            assert np.array_equal(clean_pred,expected)
            replay+=203
            for unit in units[1:]:
                expected=stored[(stored.kind=='mixed_0db')&(stored.noise_id==unit['noise_id'])&(stored.draw==unit['draw'])].set_index('path').loc[frame.iloc[val_idx].path].pred_idx.to_numpy()
                for offset in range(0,203,32):
                    count=min(32,203-offset);h_c=clean_h[offset:offset+count];h_n=model.backbone(unit['x'][offset:offset+count])
                    standard=model.classifier(model.readout(h_n))
                    if name=='cnn_pooled_query':
                        w=model.readout.shared
                        kc,kn=w(h_c),w(h_n);qc,qn=w(h_c.mean(1))[:,None],w(h_n.mean(1))[:,None]
                        ac=torch.softmax((qc@kc.transpose(-2,-1))/math.sqrt(32),-1)
                        an=torch.softmax((qn@kn.transpose(-2,-1))/math.sqrt(32),-1)
                        actual=model.classifier((an@kn)[:,0])
                        swapped=model.classifier((ac@kn)[:,0])
                        cleanvalue=model.classifier((an@kc)[:,0])
                        aq=torch.softmax((qc@kn.transpose(-2,-1))/math.sqrt(32),-1)
                        cleanquery=model.classifier((aq@kn)[:,0])
                        cosine=F.cosine_similarity(qc[:,0],qn[:,0],dim=-1)
                        a_c,a_n=ac,an;self_mass=torch.zeros(count,device='cuda')
                    else:
                        xc,vc,ac,qc=self_data(model,h_c);xn,vn,an,qn=self_data(model,h_n)
                        actual=model.classifier(self_output(model,xn,vn,an))
                        swapped=model.classifier(self_output(model,xn,vn,ac))
                        cleanvalue=model.classifier(self_output(model,xc,vc,an))
                        cleanquery=actual
                        cosine=F.cosine_similarity(qc,qn,dim=-1).mean(-1)
                        assert torch.equal(qc,qn),'Single-block CLS query must be input-independent'
                        a_c,a_n=ac[:,:,0,:],an[:,:,0,:];self_mass=an[:,:,0,0].mean(-1)
                    assert torch.allclose(actual,standard,atol=2e-4,rtol=2e-4)
                    pred=standard.argmax(-1).cpu().numpy();assert np.array_equal(pred,expected[offset:offset+count]),(name,unit['noise_id'],unit['draw'])
                    replay+=count
                    ps=swapped.argmax(-1).cpu().tolist();pv=cleanvalue.argmax(-1).cpu().tolist();pq=cleanquery.argmax(-1).cpu().tolist()
                    divergence=js(a_c,a_n).cpu().tolist();en=entropy(a_n).cpu().tolist();ec=entropy(a_c).cpu().tolist();cos=cosine.cpu().tolist();mass=self_mass.cpu().tolist()
                    for i in range(count):
                        index=val_idx[offset+i];target=int(targets[index])
                        records.append(dict(model=name,path=frame.iloc[index].path,true_idx=target,true_label=labels[target],noise_id=unit['noise_id'],draw=unit['draw'],
                            clean_pred=int(clean_pred[offset+i]),actual_pred=int(pred[i]),clean_weights_noisy_values_pred=ps[i],noisy_weights_clean_values_pred=pv[i],clean_query_noisy_kv_pred=pq[i],
                            attention_js=divergence[i],clean_attention_normalized_entropy=ec[i],noisy_attention_normalized_entropy=en[i],query_cosine=cos[i],cls_self_attention_mass=mass[i]))
                print(name,unit['noise_id'],unit['draw'],'checked',flush=True)
            del units,model
    records=pd.DataFrame(records);records.to_csv(OUT/'diagnostic_cases.csv.gz',index=False)
    result=[]
    for model,g in records.groupby('model'):
        bad=g.actual_pred.ne(g.true_idx);clean_good=g.clean_pred.eq(g.true_idx);counter_good=g.clean_weights_noisy_values_pred.eq(g.true_idx)
        q_good=g.clean_query_noisy_kv_pred.eq(g.true_idx)
        result.append(dict(model=model,noisy_cases=len(g),actual_errors=int(bad.sum()),
            clean_weights_noisy_values_errors=int((~counter_good).sum()),wrong_fixed_by_clean_weights=int((bad&counter_good).sum()),correct_broken_by_clean_weights=int((~bad&~counter_good).sum()),
            clean_correct_noisy_wrong=int((clean_good&bad).sum()),clean_correct_wrong_fixed_by_clean_weights=int((clean_good&bad&counter_good).sum()),
            clean_query_noisy_kv_errors=int((~q_good).sum()),wrong_fixed_by_clean_query=int((bad&q_good).sum()),correct_broken_by_clean_query=int((~bad&~q_good).sum()),
            noisy_weights_clean_values_errors=int(g.noisy_weights_clean_values_pred.ne(g.true_idx).sum()),
            attention_js_correct_median=float(g.loc[~bad,'attention_js'].median()),attention_js_wrong_median=float(g.loc[bad,'attention_js'].median()),
            attention_entropy_correct_median=float(g.loc[~bad,'noisy_attention_normalized_entropy'].median()),attention_entropy_wrong_median=float(g.loc[bad,'noisy_attention_normalized_entropy'].median()),
            query_cosine_wrong_median=float(g.loc[bad,'query_cosine'].median()),mean_cls_self_mass=float(g.cls_self_attention_mass.mean())))
    pd.DataFrame(result).to_csv(OUT/'summary.csv',index=False)
    (OUT/'audit.json').write_text(json.dumps(dict(stored_predictions_replayed=replay,expected_replays=10150,
        actual_manual_readout_matches_native=True,reference_only_diagnostic_not_deployable=True,
        pretrained_weights=False,new_training=False,test01_audio_read=False),indent=2)+'\n')
    assert replay==10150
    print('RESULT',json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
