"""Check future-context limits, matched initialization, annotation alignment and splits."""
import json,unittest
from pathlib import Path
import numpy as np
import torch
from .models import make_model
from .augment import Windows
from experiments.boundary_detector.data import noise_manifest,sha
ROOT=Path(__file__).resolve().parents[2]
class Checks(unittest.TestCase):
    def test_shapes_gradient_and_lookahead_limit(self):
        torch.set_num_threads(2)
        for variant in ['baseline','lookahead','boundary','spectral']:
            torch.manual_seed(42);model=make_model(variant);x=torch.randn(2,64,80);z=x.clone();k=16 if variant=='lookahead' else 0;z[:,:,40+k:]+=10
            y,b=model.forward_heads(x);torch.testing.assert_close(y[:,:40],model(z)[:,:40],rtol=0,atol=0)
            self.assertEqual(tuple(y.shape),(2,80));loss=y.square().mean()
            if b is not None:self.assertEqual(tuple(b.shape),(2,2,80));loss+=b.square().mean()
            loss.backward();self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
    def test_matched_common_initialization(self):
        states=[]
        for variant in ['baseline','lookahead','boundary','spectral']:
            torch.manual_seed(42);states.append(make_model(variant).state_dict())
        for state in states[1:]:
            for name,value in states[0].items():
                if name.startswith('core.gru.') or name.startswith('core.local.') or name.startswith('core.head.'):torch.testing.assert_close(value,state[name],rtol=0,atol=0)
    def test_window_alignment_and_seed(self):
        wave=np.zeros(16000,np.float32);wave[3200:11200]=np.sin(np.arange(8000)*.2)*.1
        r=dict(segments=[[.2,.7]]);noise=np.random.default_rng(2).normal(0,.03,40000).astype(np.float32)
        data=Windows([(r,wave)],[noise],['test'],.01)
        for i in range(4):
            tensors=data[i]
            for x,y in zip(tensors,data[i]):self.assertTrue(torch.equal(x,y))
            x,y,valid,w,b=tensors
            self.assertEqual(x.shape,(25280,));self.assertEqual(y.shape,(301,));self.assertEqual(b.shape,(2,301));self.assertTrue(torch.isfinite(x).all());self.assertLessEqual(float(x.abs().max()),.99001)
            if i==3:self.assertEqual(float(y.sum()),0);self.assertEqual(float(b.sum()),0)
            self.assertTrue(set(w.tolist())<={1.,3.,5.})
    def test_real_noise_sources_disjoint(self):
        root=ROOT/'runs/detector-diversity-v2';n=noise_manifest(root/'noise.json');prov=json.loads((root/'noise_provenance.json').read_text())
        train=[r for r in prov['records'] if r['split']=='train'];audit=[r for r in prov['records'] if r['split']=='audit']
        self.assertEqual(len(n['train']),94);self.assertEqual(len(train),88);self.assertEqual(len(audit),22)
        self.assertFalse({r['original_group'] for r in train}&{r['original_group'] for r in audit})
        self.assertFalse({sha(p) for p in n['train']}&{r['prepared_sha256'] for r in audit})
        self.assertEqual(len({r['original_group'] for r in train+audit}),110)

if __name__=='__main__':unittest.main(verbosity=2)
