"""GPU correctness matrix; run as python -m tests.validate_kernels."""
import json
from pathlib import Path
import torch
from kernels import rmsnorm, residual_rmsnorm, swiglu
from baselines import torch_ops as ref

def main():
    torch.manual_seed(42)
    results=[]
    for dtype in (torch.float16, torch.bfloat16):
      for m,n in [(1,1),(1,7),(8,1024),(32,1537),(128,4096),(512,8192),(2048,1024),(2,16384),(0,7)]:
       for scale in (0.,1e-4,1.,100.):
        x=torch.randn(m,n,device='cuda',dtype=dtype)*scale
        r=torch.randn_like(x)*scale; w=torch.randn(n,device='cuda',dtype=dtype)
        before=x.clone(); rb=r.clone()
        for name,fn,rf,args in [('rmsnorm',rmsnorm,ref.rmsnorm,(x,w)),('residual_rmsnorm',residual_rmsnorm,ref.residual_rmsnorm,(x,r,w)),('swiglu',swiglu,ref.swiglu,(x,r))]:
         got=fn(*args); expected=rf(*args)
         pairs=zip(got,expected) if isinstance(got,tuple) else [(got,expected)]
         for index,(a,b) in enumerate(pairs):
          tolerance=2e-3 if dtype==torch.float16 else 2e-2
          torch.testing.assert_close(a,b,rtol=tolerance,atol=tolerance)
          assert a.dtype==dtype and a.shape==b.shape
          finite=torch.isfinite(a) & torch.isfinite(b)
          err=(a.float()[finite]-b.float()[finite]).abs()
          results.append(dict(operator=name,output=index,dtype=str(dtype),rows=m,width=n,scale=scale,nonfinite_count=int((~finite).sum().item()),max_abs_error=err.max().item() if err.numel() else 0.,rms_error=err.square().mean().sqrt().item() if err.numel() else 0.))
        assert torch.equal(x,before) and torch.equal(r,rb)
    x=torch.randn(3,7,device='cuda',dtype=torch.float16); w=torch.ones(7,device='cuda',dtype=x.dtype)
    for bad in [lambda:rmsnorm(x.t(),w),lambda:rmsnorm(x,w,0),lambda:rmsnorm(x,w[:3]),lambda:swiglu(x,x.float()),lambda:rmsnorm(x.requires_grad_(),w)]:
        try: bad()
        except ValueError: pass
        else: raise AssertionError('unsupported input accepted')
    torch.cuda.synchronize()
    Path('results').mkdir(exist_ok=True)
    Path('results/correctness.json').write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,cases=results,passed=len(results)),indent=2,allow_nan=False))
    print('CORRECTNESS_PASS',len(results),torch.cuda.get_device_name(),flush=True)

if __name__=='__main__': main()
