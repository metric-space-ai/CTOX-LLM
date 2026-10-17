import unittest,numpy as np
from analyze import fit_gain
from replay import patched_manifest
class Contracts(unittest.TestCase):
    def test_known_scale_recovery_and_trust_bound(self):
        x=np.array([[1.,2.],[2.,1.],[3.,4.]]);r=np.ones_like(x)
        np.testing.assert_allclose(fit_gain(x,r,r+1.05*x,0),[1.05,1.05],atol=1e-10)
        np.testing.assert_allclose(fit_gain(x,r,r+5*x,0),[1.1,1.1],atol=1e-10)
    def test_patch_preserves_weight_payload_contract(self):
        name='model.language_model.layers.0.mlp.down_proj.weight.s_out'
        original={'tensors':[{'name':'model.weight','dtype':'q2_b64','offset':0,'length':18,'sha256':'a'*64,'shape':[64]},
                              {'name':name,'dtype':'f16','offset':64,'length':4,'sha256':'b'*64,'shape':[2]}]}
        updated,changes=patched_manifest(original,{name:np.array([1.05,1.],dtype='f2')})
        self.assertEqual(updated['tensors'][0],original['tensors'][0]);self.assertEqual(set(changes),{64})
        self.assertEqual(original['tensors'][1]['sha256'],'b'*64)
        with self.assertRaises(ValueError):patched_manifest(original,{'model.weight':np.ones(64)})
        with self.assertRaises(ValueError):patched_manifest(original,{name:np.array([np.nan,1.])})
if __name__=='__main__':unittest.main()
