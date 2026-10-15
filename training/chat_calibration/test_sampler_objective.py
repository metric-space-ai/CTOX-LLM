"""The training TV must equal the independently implemented measurement TV."""
import math,unittest
import numpy as np
import torch
from fit import sampler_tv_loss
from measure import distribution,tv
from analyze import tool_positions
class SamplerObjective(unittest.TestCase):
    def test_tool_span_boundaries_and_role_reset(self):
        # Opening/closing markers count; a new role ends an unfinished span.
        self.assertEqual(tool_positions([9,4,5,7,6,8,9,0,4,5,3,0,2],[4,5],[6,8],0),{1,2,3,4,5,8,9,10})
    def test_value_and_gradient(self):
        a=np.full(24,-1000.,dtype=np.float32);a[:2]=np.log([.51,.49])
        b=np.full(24,-1000.,dtype=np.float32);b[:2]=np.log([.49,.51])
        reference=torch.zeros((1,24),dtype=torch.float64)
        for i,p in distribution(b):reference[0,i]=p
        x=torch.tensor(a.reshape(1,-1),requires_grad=True)
        loss=sampler_tv_loss(x,reference,torch)
        self.assertAlmostEqual(float(loss.detach()),tv(distribution(a),distribution(b)),places=7)
        loss.backward();self.assertGreater(float(x.grad[0,0]),0);self.assertLess(float(x.grad[0,1]),0)
    def test_truncation_and_ties_match_measurement(self):
        for a in [np.arange(24,dtype=np.float32)*.1,np.zeros(24,dtype=np.float32),np.array([0.]+[-100.]*23,dtype=np.float32)]:
            reference=torch.zeros((1,24),dtype=torch.float64)
            for i,p in distribution(a):reference[0,i]=p
            self.assertLess(float(sampler_tv_loss(torch.tensor(a.reshape(1,-1)),reference,torch)),1e-8)
if __name__=='__main__':unittest.main()
