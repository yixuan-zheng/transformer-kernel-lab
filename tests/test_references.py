import unittest
import torch
from baselines.torch_ops import rmsnorm, residual_rmsnorm, swiglu

class ReferenceTests(unittest.TestCase):
    def test_normalization(self):
        for width in (7, 1024, 1537):
            x=torch.randn(3,width)
            w=torch.randn(width)
            expected=(x.double()/torch.sqrt(x.double().square().mean(-1,keepdim=True)+1e-6)*w.double()).float()
            torch.testing.assert_close(rmsnorm(x,w),expected)
            torch.testing.assert_close(rmsnorm(torch.zeros_like(x),w),torch.zeros_like(x))

    def test_residual_contract(self):
        x=torch.randn(3,7,dtype=torch.float16); r=torch.randn_like(x); w=torch.ones(7,dtype=x.dtype)
        before=x.clone(); rb=r.clone()
        y,updated=residual_rmsnorm(x,r,w)
        torch.testing.assert_close(updated,x+r)
        torch.testing.assert_close(y,rmsnorm(x+r,w))
        self.assertTrue(torch.equal(x,before) and torch.equal(r,rb))

    def test_swiglu(self):
        g=torch.tensor([-20.,-1.,0.,1.,20.]); up=torch.arange(5.)
        expected=(g.double()/(1+torch.exp(-g.double()))*up.double()).float()
        torch.testing.assert_close(swiglu(g,up),expected)

if __name__=='__main__': unittest.main()
