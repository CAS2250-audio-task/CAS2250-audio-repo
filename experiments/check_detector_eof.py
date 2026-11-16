import unittest
import torch
from experiments.detector_diversity.models import make_model
from experiments.detector_diversity.inference import infer_logits
class Checks(unittest.TestCase):
 def setUp(self):torch.set_num_threads(2);torch.manual_seed(42)
 def test_causal_unchanged(self):
  m=make_model('baseline').eval();x=torch.randn(2,64,80)
  torch.testing.assert_close(m(x),infer_logits(m,x),rtol=0,atol=0)
 def test_prefix_and_length_aware_batch(self):
  m=make_model('lookahead').eval();x=torch.randn(2,64,80);x[1,:,55:]=0
  with torch.no_grad():
   legacy=m(x);held=infer_logits(m,x,[80,55])
   for i,n in enumerate([80,55]):
    solo=infer_logits(m,x[i:i+1,:,:n])
    torch.testing.assert_close(held[i,:n],solo[0],rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(held[i,:n-16],legacy[i,:n-16],rtol=1e-5,atol=1e-6)
 def test_silence_policy_and_shape(self):
  m=make_model('lookahead').eval();x=torch.randn(1,64,40)
  y=infer_logits(m,x,policy='silence',silence=torch.full((64,1),-6.))
  self.assertEqual(y.shape,(1,40));self.assertTrue(torch.isfinite(y).all())
if __name__=='__main__':unittest.main(verbosity=2)
