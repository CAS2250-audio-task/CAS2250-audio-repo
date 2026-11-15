"""Meaningful ablation checks: initialization, support, shape, and local-path gradients."""
import hashlib,json,torch
from pathlib import Path
from torch import nn
from torch.nn import functional as F
from experiments.compact_pooling.models import matched_models
from experiments.local_features.models import variant_models

ROOT=Path(__file__).resolve().parents[1]


def receptive(encoder,stop=None):
    field=1;jump=1
    for i,block in enumerate(encoder):
        for layer in block.main:
            if isinstance(layer,nn.Conv1d):
                field+=(layer.kernel_size[0]-1)*layer.dilation[0]*jump
                jump*=layer.stride[0]
        if stop==i:break
    return field


def main():
    torch.set_num_threads(2);torch.manual_seed(4242)
    x=torch.randn(3,64,151)
    reference=matched_models(42)
    variants={v:variant_models(42,v) for v in ['baseline','short','local','both']}
    counts={}
    for v,models in variants.items():
        counts[v]={}
        for name,model in models.items():
            model.eval();reference[name].eval()
            assert receptive(model.backbone.encoder)==(61 if v in ['short','both'] else 109)
            assert receptive(model.backbone.encoder,1)==23
            a=reference[name].state_dict();b=model.state_dict()
            for key,value in a.items():
                if v in ['short','both'] and key=='backbone.encoder.4.main.0.weight':continue
                assert torch.equal(value,b[key]),(v,name,key)
            with torch.no_grad():
                result=model(x);assert result.shape==(3,11) and torch.isfinite(result).all()
                assert model.backbone(x).shape==(3,151,32)
                if v in ['baseline','local']:assert torch.equal(result,reference[name](x))
            counts[v][name]=sum(p.numel() for p in model.parameters())
            if v in ['local','both']:
                model.train();F.cross_entropy(model(x),torch.tensor([0,1,2])).backward()
                assert model.backbone.local_projection.weight.grad.abs().sum()>0
                assert torch.isfinite(model.backbone.local_projection.weight.grad).all()
    # Fresh models: the gradient checks above intentionally update train-mode BN statistics.
    fresh_short=variant_models(42,'short');fresh_both=variant_models(42,'both')
    for name in reference:
        with torch.no_grad():
            fresh_short[name].eval();fresh_both[name].eval()
            assert torch.equal(fresh_short[name](x),fresh_both[name](x))
    old=Path('/root/CAS2250-compact-contrastive-100/reports/compact-noise-speaker-spk03-e1000/protocol.json')
    prior=json.loads(old.read_text())
    for path in ['experiments/compact_pooling/models.py','experiments/compact_pooling/objectives.py']:
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==prior['source_sha256'][path]
    identity=json.loads((ROOT/'runs/compact-pooling-v1/formants.json').read_text())['identity']
    assert hashlib.sha256((ROOT/'experiments/overnight/data.py').read_bytes()).hexdigest()==identity['transform_source_sha256']
    assert hashlib.sha256((ROOT/'experiments/overnight/waveforms.py').read_bytes()).hexdigest()==identity['settings_source_sha256']
    value=dict(global_rf_frames={'baseline':109,'short':61,'local':109,'both':61},local_rf_frames=23,
        shared_initial_tensors_equal=True,local_initial_output_equals_baseline=True,both_initial_output_equals_short=True,
        local_projection_has_nonzero_finite_gradient=True,output_shape_3_151_32=True,parameter_counts=counts,
        original_model_objective_sources_unchanged=True)
    p=ROOT/'reports/compact-local-features/preflight.json';p.parent.mkdir(exist_ok=True,parents=True)
    p.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))


if __name__=='__main__':main()
