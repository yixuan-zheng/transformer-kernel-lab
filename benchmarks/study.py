"""Bounded, reproducible L40S performance study. Run from the project root."""
import argparse,csv,json,random,statistics,subprocess,time
from pathlib import Path
import torch,triton
import triton.language as tl
from kernels.ops import _norm,_swiglu
from baselines import torch_ops as ref

@triton.jit
def _add(X,R,U,N:tl.constexpr,B:tl.constexpr):
 i=tl.program_id(0)*B+tl.arange(0,B)
 tl.store(U+i,tl.load(X+i,i<N,0).to(tl.float32)+tl.load(R+i,i<N,0).to(tl.float32),i<N)

@triton.jit
def _silu(G,T,N:tl.constexpr,B:tl.constexpr):
 i=tl.program_id(0)*B+tl.arange(0,B);g=tl.load(G+i,i<N,0).to(tl.float32)
 tl.store(T+i,g/(1.+tl.exp(-g)),i<N)

@triton.jit
def _mul(T,U,Y,N:tl.constexpr,B:tl.constexpr):
 i=tl.program_id(0)*B+tl.arange(0,B)
 tl.store(Y+i,tl.load(T+i,i<N,0)*tl.load(U+i,i<N,0).to(tl.float32),i<N)

def make(op,x,r,w,config=None,unfused=False):
 m,n=x.shape
 def run():
  y=torch.empty_like(x)
  if op=='swiglu':
   b=config or 256
   if unfused:
    tmp=torch.empty_like(x,dtype=torch.float32)
    _silu[(triton.cdiv(x.numel(),b),)](x,tmp,x.numel(),b,enable_fp_fusion=False)
    _mul[(triton.cdiv(x.numel(),b),)](tmp,r,y,x.numel(),b,enable_fp_fusion=False)
   else: _swiglu[(triton.cdiv(x.numel(),b),)](x,r,y,x.numel(),b,enable_fp_fusion=False)
   return y
  fused=op=='residual_rmsnorm'
  updated=torch.empty_like(x) if fused else y
  source=x
  if fused and unfused:
   _add[(triton.cdiv(x.numel(),256),)](x,r,updated,x.numel(),256)
   source=updated
  _norm[(m,)](source,r,w,y,updated,n,1e-6,fused and not unfused,triton.next_power_of_2(n),num_warps=config or (4 if n<=4096 else 8),enable_fp_fusion=False)
  return (y,updated) if fused else y
 return run

def bench(fn,mode):
 if mode=='graph':
  s=torch.cuda.Stream();s.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(s): ms=triton.testing.do_bench_cudagraph(fn,rep=15)
  s.synchronize();return ms*1000
 a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
 a.record()
 for _ in range(30): fn()
 b.record();b.synchronize();return a.elapsed_time(b)/30*1000

def savecsv(path,rows):
 with path.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def tensors(m,n,dtype):
 return (torch.randn(m,n,device='cuda',dtype=dtype),torch.randn(m,n,device='cuda',dtype=dtype),torch.randn(n,device='cuda',dtype=dtype))

def check(fn,expected,dtype):
 got=fn()
 pairs=zip(got,expected) if isinstance(got,tuple) else [(got,expected)]
 for a,b in pairs: torch.testing.assert_close(a,b,rtol=.002 if dtype==torch.float16 else .02,atol=.002 if dtype==torch.float16 else .02)

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--ncu',action='store_true');a=p.parse_args()
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 if 'L40S' not in torch.cuda.get_device_name():raise RuntimeError('The published study targets L40S')
 torch.manual_seed(2026);rng=random.Random(2026)
 if a.ncu:
  x,r,w=tensors(2048,8192,torch.float16)
  for unfused in (True,False):
   f=make('residual_rmsnorm',x,r,w,unfused=unfused)
   f();torch.cuda.synchronize()
  return
 metadata=dict(gpu=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability(),torch=torch.__version__,triton=triton.__version__,cuda=torch.version.cuda,seed=2026,started=time.time(),device=str(torch.cuda.get_device_properties(0)))
 metadata['nvidia_smi']=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,clocks.sm,clocks.mem,power.limit','--format=csv'],text=True)
 (out/'environment.json').write_text(json.dumps(metadata,indent=2))
 ops=['rmsnorm','residual_rmsnorm','swiglu'];tuning=[];selected={}
 # A fixed global configuration per operator selected using four development shapes.
 for op in ops:
  candidates=[128,256,512,1024] if op=='swiglu' else [4,8,16]
  scores={c:[] for c in candidates}
  for m,n in [(32,1024),(32,8192),(512,1024),(512,8192)]:
   x,r,w=tensors(m,n,torch.float16)
   args=(x,r) if op=='swiglu' else ((x,r,w) if op=='residual_rmsnorm' else (x,w))
   expected=getattr(ref,op)(*args)
   for c in candidates:check(make(op,x,r,w,c),expected,x.dtype)
   for batch in range(3):
    order=candidates[:];rng.shuffle(order)
    for c in order:
     us=bench(make(op,x,r,w,c),'graph');scores[c].append(us)
     tuning.append(dict(operator=op,rows=m,width=n,config=c,batch=batch,latency_us=us))
  # Geometric mean latency gives each shape comparable influence.
  import math
  selected[op]=min(candidates,key=lambda c:statistics.mean(math.log(v) for v in scores[c]))
 savecsv(out/'tuning.csv',tuning);(out/'selected.json').write_text(json.dumps(selected,indent=2))
 compiled={op:torch.compile(getattr(ref,op)) for op in ops}
 raw=[];unsupported=[]
 for op in ops:
  widths=[1024,2048,4096,8192]+([11008,14336,28672] if op=='swiglu' else [])
  shapes=[(m,n,d) for m in [1,8,32,128,512,2048] for n in widths for d in [torch.float16,torch.bfloat16]]
  rng.shuffle(shapes)
  for m,n,d in shapes:
   x,r,w=tensors(m,n,d)
   args=(x,r) if op=='swiglu' else ((x,r,w) if op=='residual_rmsnorm' else (x,w))
   expected=getattr(ref,op)(*args)
   funcs={'eager':lambda:getattr(ref,op)(*args),'compile':lambda:compiled[op](*args),'triton_initial':make(op,x,r,w),'triton_tuned':make(op,x,r,w,selected[op])}
   if op!='rmsnorm':funcs['triton_unfused']=make(op,x,r,w,selected[op],True)
   for name,fn in funcs.items():
    check(fn,expected,d)
    for _ in range(5):fn()
   torch.cuda.synchronize()
   for batch in range(5):
    order=[(name,mode) for name in funcs for mode in ['events','graph']];rng.shuffle(order)
    for name,mode in order:
     us=bench(funcs[name],mode)
     raw.append(dict(operator=op,rows=m,width=n,dtype=str(d),baseline=name,mode=mode,batch=batch,latency_us=us,region='development' if d==torch.float16 and (m,n) in [(32,1024),(32,8192),(512,1024),(512,8192)] else 'validation',width_role='intermediate' if op=='swiglu' and n>8192 else 'supplementary' if op=='swiglu' else 'hidden'))
   savecsv(out/'raw.csv',raw)
   print('BENCH',op,m,n,d,flush=True)
 # Collect traces after all timing, so profiler instrumentation cannot bias measurements.
 profiles=[]
 (out/'traces').mkdir(exist_ok=True)
 for op,m,n in [('residual_rmsnorm',1,4096),('residual_rmsnorm',2048,8192),('swiglu',32,14336)]:
  x,r,w=tensors(m,n,torch.float16)
  args=(x,r) if op=='swiglu' else (x,r,w)
  funcs={'eager':lambda:getattr(ref,op)(*args),'compile':lambda:compiled[op](*args),'triton_tuned':make(op,x,r,w,selected[op]),'triton_unfused':make(op,x,r,w,selected[op],True)}
  for name,fn in funcs.items():
   fn();torch.cuda.synchronize()
   with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],profile_memory=True) as prof:
    with torch.profiler.record_function('measured_operator'):fn();torch.cuda.synchronize()
   prof.export_chrome_trace(str(out/'traces'/f'trace-{op}-{m}-{n}-{name}.json'))
   events=[e for e in prof.events() if e.device_type==torch.autograd.DeviceType.CUDA and e.name not in ['[memory]']]
   profiles.append(dict(operator=op,rows=m,width=n,baseline=name,cuda_events=len(events),names=';'.join(e.name for e in events),logical_bytes_saved_vs_unfused=(2*m*n if op=='residual_rmsnorm' else 8*m*n) if name=='triton_tuned' else 0))
 savecsv(out/'profiles.csv',profiles)
 (out/'unsupported.json').write_text(json.dumps(dict(failures=unsupported,limits=['CUDA L40S only','FP16/BF16 contiguous inputs','normalization width <=16384','no gradients','no measured DRAM bandwidth; logical traffic only']),indent=2))
 print('BENCH_PASS',len(raw),selected,flush=True)

if __name__=='__main__':main()
