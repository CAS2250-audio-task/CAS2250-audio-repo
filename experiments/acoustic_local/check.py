"""Check initialization equivalence, auxiliary gradients, legacy extraction and alignment."""
import json,time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from experiments.local_features.models import variant_models
from .models import acoustic_models
from .features import extract,NAMES


def main():
    torch.set_num_threads(2)
    scaler=dict(mean=[1000.]*6,std=[200.]*6)
    base=variant_models(42,'local');models=acoustic_models(42,scaler)
    x=torch.randn(3,64,151);a=torch.randn(3,6,151)*200+1000;valid=torch.ones_like(a);valid[0,:,:20]=0
    combined=torch.cat([x,a,valid],dim=1)
    counts={}
    for name,model in models.items():
        model.eval();base[name].eval()
        with torch.no_grad():assert torch.equal(model(combined),base[name](x))
        model.train();torch.nn.functional.cross_entropy(model(combined),torch.tensor([0,1,2])).backward()
        grad=model.backbone.auxiliary[-1].weight.grad
        assert torch.isfinite(grad).all() and grad.abs().sum()>0
        counts[name]=sum(p.numel() for p in model.parameters())
    root=Path(__file__).resolve().parents[2]
    wave,rate=sf.read(root/'datasets/task2/fixed-1p5-v1/audio/후진/spk03_184.wav',dtype='float32');assert rate==16000
    t=time.monotonic();encoded=extract(wave);elapsed=time.monotonic()-t
    assert encoded.shape==(12,151) and np.isfinite(encoded).all()
    assert not extract(np.zeros(24000,np.float32)).any()
    # Compatibility aliases for the historical librosa0.8 extractor, check only.
    for name,value in [("complex",complex),("float",float),("int",int)]:
        if name not in np.__dict__:setattr(np,name,value)
    from experiments.overnight.features import extract as legacy,NAMES as legacy_names
    raw,observed=legacy(wave)
    indexes=[legacy_names.index(n) for n in NAMES]
    assert np.array_equal(encoded[6:],observed[indexes].astype(np.float32))
    np.testing.assert_allclose(encoded[:6],np.where(observed[indexes],raw[indexes],0),rtol=0,atol=0)
    output=dict(initial_output_equals_local=True,finite_nonzero_auxiliary_projection_gradients=True,
        shape=[12,151],silent_availability_zero=True,legacy_set3_values_and_masks_exact=True,
        parameters=counts,extra_parameters={k:counts[k]-sum(p.numel() for p in base[k].parameters()) for k in counts},
        extraction_seconds_single_clip=elapsed)
    p=root/'reports/compact-acoustic-local-v1/preflight.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))


if __name__=='__main__':main()
