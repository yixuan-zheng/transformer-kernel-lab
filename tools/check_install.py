"""Smoke-test installed wheel outside the source tree, within a GPU allocation."""
import json
import torch
import kernels
from baselines import torch_ops

def main():
 torch.manual_seed(7)
 for dtype in (torch.float16,torch.bfloat16):
  x=torch.randn(8,1537,device='cuda',dtype=dtype);r=torch.randn_like(x);w=torch.randn(1537,device='cuda',dtype=dtype)
  for name,args in [('rmsnorm',(x,w)),('residual_rmsnorm',(x,r,w)),('swiglu',(x,r))]:
   got=getattr(kernels,name)(*args);ref=getattr(torch_ops,name)(*args)
   pairs=zip(got,ref) if isinstance(got,tuple) else [(got,ref)]
   for a,b in pairs:torch.testing.assert_close(a,b,rtol=.002 if dtype==torch.float16 else .02,atol=.002 if dtype==torch.float16 else .02)
 torch.cuda.synchronize()
 print(json.dumps(dict(status='INSTALL_GPU_PASS',installed_package='/site-packages/' in kernels.__file__,torch=torch.__version__,gpu=torch.cuda.get_device_name(),dtypes=['FP16','BF16'],operators=3)),flush=True)

if __name__=='__main__':main()
