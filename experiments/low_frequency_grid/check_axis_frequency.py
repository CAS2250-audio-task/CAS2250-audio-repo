"""Real D1 paired-batch forward/backward check, without updating training models."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from experiments.compact_pooling.objectives import paired_loss
from experiments.frequency_local.models import AxisBlock, FrequencyAttention
from experiments.frequency_local_subset import canonical_projectors
from experiments.quick_frequency40.models import BandRoutes, Frontend
from .frontend import Frontend as MelFrontend
from .models import models
from . import trainer


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',type=Path,default=trainer.ROOT/'reports/axis-frequency-followup/preflight.json')
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    assert torch.cuda.is_available()
    frame=pd.read_csv(trainer.DATA/'metadata.csv')
    frame=frame[frame.speaker_id.isin(['spk02','spk03','spk04','spk05'])].reset_index(drop=True)
    targets=np.array([trainer.LABELS.index(label) for label in frame.label])
    waves=[]
    for path in frame.path:
        wave,rate=sf.read(trainer.DATA/path,dtype='float32')
        assert rate==16000 and wave.shape==(24000,)
        waves.append(wave)
    waves=np.stack(waves)
    train_idx=np.flatnonzero(frame.speaker_id.ne('spk03'))
    mel=MelFrontend('standard').cuda().eval()
    original=trainer.features(mel,waves[train_idx])
    mean=original.mean((0,2),keepdim=True)
    std=original.std((0,2),keepdim=True).clamp_min(1e-4)
    reference=float(np.median(np.sqrt(np.mean(waves[train_idx[targets[train_idx]!=10]].astype(np.float64)**2,axis=1)))**2)
    source=json.loads((trainer.ROOT/'runs/compact-pooling-v1/sources.json').read_text())
    noises=[]
    for row in source['train']:
        path=trainer.ROOT/row['prepared_path']
        assert trainer.sha(path)==row['prepared_sha256']
        wave,rate=sf.read(path,dtype='float32')
        assert rate==16000
        noises.append((row['noise_id'],wave))
    bank=trainer.formant_bank(frame,waves)
    rng=np.random.default_rng(np.random.SeedSequence([42,3,1,100]))
    indices=rng.permutation(train_idx)[:32]
    augmented,trace=trainer.augment(waves,indices,targets,bank,noises,'spk03',1,reference,noise_probability=1.)
    reference_waves=np.stack([bank[r['index'],r['variant']] if r['formant'] else waves[r['index']] for r in trace])
    x=trainer.features(mel,augmented,mean,std)
    paired=trainer.features(mel,reference_waves,mean,std)
    labels=torch.tensor(targets[indices],device='cuda')
    variants=('axis_frequency','axis_frequency_attn')
    candidates={key:models(42,key)['cnn_pooled_query'].cuda() for key in variants}
    baseline=models(42,'frequency')['cnn_pooled_query'].cuda().eval()
    base_state=baseline.state_dict()
    for model in candidates.values():
        state=model.state_dict()
        assert all(torch.equal(state[key],value) for key,value in base_state.items())
        assert isinstance(model.backbone[0],Frontend)
        assert isinstance(model.backbone[2],BandRoutes)
        assert len(model.backbone[2].blocks)==5
        assert all(isinstance(block,AxisBlock) for block in model.backbone[2].blocks)
        model.eval()
        with torch.no_grad():
            assert torch.equal(model(x),baseline(x)), 'Residual modules must preserve initial outputs'
    # Verify the two contiguous frequency regions and their concatenation order.
    probe=torch.arange(8.,device='cuda').view(1,1,8,1).expand(1,2,8,3)
    expected=torch.tensor([1.5,1.5,5.5,5.5],device='cuda').expand(1,3,4)
    assert torch.equal(BandRoutes.pool(probe),expected)
    results=[]
    for key,model in candidates.items():
        assert sum(isinstance(m,FrequencyAttention) for m in model.modules())==int(key.endswith('_attn'))
        projector=copy.deepcopy(canonical_projectors()['cnn_pooled_query']).cuda()
        model.train()
        loss,logits,ce,contrast=paired_loss(model,projector,paired,x,labels,weight=.1,temperature=.1,positive_mode='same_command')
        assert logits.shape==(32,11) and torch.isfinite(logits).all() and torch.isfinite(loss)
        loss.backward()
        parameters=list(model.named_parameters())+list(projector.named_parameters(prefix='projector'))
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for _,p in parameters)
        assert model.backbone[0].output.weight.grad.abs().sum()>0
        assert model.backbone[2].local_projection.weight.grad.abs().sum()>0
        if key.endswith('_attn'):
            assert model.backbone[1].output.weight.grad.abs().sum()>0
        results.append(dict(architecture=key,input_shape=list(x.shape),logits_shape=list(logits.shape),
            loss=float(loss),ce=float(ce),supcon=float(contrast),parameters=sum(p.numel() for p in model.parameters()),
            all_gradients_finite=True,initial_output_equal_to_band2=True,
            frontend_output_gradient=float(model.backbone[0].output.weight.grad.norm())))
    trainer.write(args.output,dict(status='passed',seed=42,fold='spk03',batch_paths=frame.iloc[indices].path.tolist(),
        augmentation_trace=trace,source_sha256={str(p.relative_to(trainer.ROOT)):trainer.sha(p) for p in Path(__file__).parent.glob('*.py')},
        two_contiguous_bands_verified=True,results=results))
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
