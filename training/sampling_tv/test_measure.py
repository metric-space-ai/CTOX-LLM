import importlib.util
import math
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('measure',Path(__file__).with_name('measure.py'))
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
class MetricContract(unittest.TestCase):
    def test_shifted_mass_is_two_percent_not_top1_flip(self):
        p=[[0,.51],[1,.49]]; q=[[0,.49],[1,.51]]
        self.assertAlmostEqual(m.tv(p,q),.02)
        self.assertEqual(m.tv(p,p),0)
        self.assertEqual(m.tv([[0,1]],[[1,1]]),1)
        with self.assertRaises(ValueError): m.tv([[0,.5]],q)
        with self.assertRaises(ValueError): m.tv([[0,.5],[0,.5]],q)
    def test_truncation_is_exact_post_sampler_zero(self):
        x=[math.log(.51),math.log(.49)]+[-1000.]*22
        p=m.distribution(x); self.assertEqual(len(p),2); self.assertAlmostEqual(p[0][1],.51)
        self.assertEqual(m.distribution([math.log(.96),math.log(.04)]+[-1000.]*22),[[0,1.]])
        self.assertEqual([i for i,p in m.distribution([0.]*24)],list(range(19)))
        with self.assertRaises(ValueError): m.distribution([float('nan')]*24)
if __name__=='__main__': unittest.main()
