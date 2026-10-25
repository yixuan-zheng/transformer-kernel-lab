"""Audit saved samples and generate benchmark summaries and plots; CPU-only."""
import csv,json,math,statistics,itertools,argparse
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def read(p):return list(csv.DictReader(p.open()))
def save(p,rows):
 with p.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
 p=argparse.ArgumentParser();p.add_argument('directory');a=p.parse_args();out=Path(a.directory)
 raw=read(out/'raw.csv');groups=defaultdict(list)
 keys=['operator','rows','width','dtype','baseline','mode']
 for r in raw:
  assert math.isfinite(float(r['latency_us'])) and float(r['latency_us'])>0
  groups[tuple(r[k] for k in keys)].append(r)
 expected=set()
 for op in ['rmsnorm','residual_rmsnorm','swiglu']:
  widths=[1024,2048,4096,8192]+([11008,14336,28672] if op=='swiglu' else [])
  names=['eager','compile','triton_initial','triton_tuned']+(['triton_unfused'] if op!='rmsnorm' else [])
  for m,n,d,b,mode in itertools.product([1,8,32,128,512,2048],widths,['torch.float16','torch.bfloat16'],names,['events','graph']):
   expected.add((op,str(m),str(n),d,b,mode))
 assert set(groups)==expected
 summaries=[];lookup={}
 for key,rows in groups.items():
  assert sorted(int(r['batch']) for r in rows)==list(range(5))
  values=[float(r['latency_us']) for r in rows];q1,q3=np.percentile(values,[25,75])
  s=dict(zip(keys,key));s.update(median_us=statistics.median(values),q25_us=float(q1),q75_us=float(q3),min_us=min(values),max_us=max(values),samples=5)
  op,m,n,_,base,_=key;m=int(m);n=int(n)
  logical_bytes=(3*m*n*2 if op=='swiglu' else ((4 if op=='residual_rmsnorm' else 2)*m*n+n)*2)
  if base=='triton_unfused':logical_bytes+=8*m*n if op=='swiglu' else 2*m*n
  s.update(logical_minimum_io_bytes=logical_bytes,logical_minimum_io_GBps=logical_bytes/s['median_us']/1000)
  summaries.append(s);lookup[key]=s
 save(out/'summary.csv',summaries)
 comparisons=[]
 for key,s in lookup.items():
  if key[4]!='triton_tuned':continue
  for base in ['eager','compile','triton_initial']+(['triton_unfused'] if key[0]!='rmsnorm' else []):
   b=lookup[(*key[:4],base,key[5])]
   comparisons.append(dict(operator=key[0],rows=key[1],width=key[2],dtype=key[3],mode=key[5],comparison=base,speedup=b['median_us']/s['median_us'],reference_us=b['median_us'],tuned_us=s['median_us'],iqr_overlaps=not(b['q75_us']<s['q25_us'] or s['q75_us']<b['q25_us'])))
 save(out/'comparisons.csv',comparisons)
 plots=out/'plots';plots.mkdir(exist_ok=True)
 plt.rcParams.update({'font.size':10,'figure.dpi':140})
 for mode in ['graph','events']:
  fig,axes=plt.subplots(2,3,figsize=(17,8),constrained_layout=True)
  for row,dtype in enumerate(['torch.float16','torch.bfloat16']):
   for col,op in enumerate(['rmsnorm','residual_rmsnorm','swiglu']):
    ax=axes[row,col];widths=[1024,2048,4096,8192]+([11008,14336,28672] if op=='swiglu' else [])
    data=np.array([[next(r['speedup'] for r in comparisons if r['operator']==op and int(r['rows'])==m and int(r['width'])==n and r['dtype']==dtype and r['mode']==mode and r['comparison']=='compile') for n in widths] for m in [1,8,32,128,512,2048]])
    im=ax.imshow(np.log2(data),vmin=-2,vmax=2,cmap='RdBu',aspect='auto')
    for i in range(6):
     for j in range(len(widths)):ax.text(j,i,f'{data[i,j]:.2f}',ha='center',va='center',fontsize=7,color='white' if abs(np.log2(data[i,j]))>1.2 else 'black')
    ax.set_xticks(range(len(widths)),widths,rotation=45);ax.set_yticks(range(6),[1,8,32,128,512,2048]);ax.set_title(op+' / '+dtype.split('.')[-1]);ax.set_xlabel('Width');ax.set_ylabel('Rows')
  fig.colorbar(im,ax=axes,label='log2 speedup vs torch.compile (positive favors Triton)')
  fig.suptitle('L40S: median compiled / tuned Triton latency — '+mode+' timing')
  fig.savefig(plots/f'speedup-vs-compile-{mode}.png');plt.close(fig)
 # Tuning outcome on validation points, including losses.
 fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
 for ax,op in zip(axes,['rmsnorm','residual_rmsnorm','swiglu']):
  vals=[r['speedup'] for r in comparisons if r['operator']==op and r['mode']=='graph' and r['comparison']=='triton_initial' and not(r['dtype']=='torch.float16' and (int(r['rows']),int(r['width'])) in [(32,1024),(32,8192),(512,1024),(512,8192)])]
  ax.hist(vals,bins=15);ax.axvline(1,color='black',ls='--');ax.set_title(op);ax.set_xlabel('Initial / tuned median latency');ax.set_ylabel('Validation shape/dtype cases')
 fig.savefig(plots/'optimization-validation.png');plt.close(fig)
 # Profiled cases: latency and IQR, with modes kept on separate axes.
 fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
 for col,(op,m,n) in enumerate([('residual_rmsnorm',1,4096),('residual_rmsnorm',2048,8192),('swiglu',32,14336)]):
  names=['eager','compile','triton_initial','triton_tuned','triton_unfused']
  for row,mode in enumerate(['graph','events']):
   ax=axes[row,col];ss=[lookup[(op,str(m),str(n),'torch.float16',b,mode)] for b in names]
   ax.bar(range(5),[s['median_us'] for s in ss]);ax.errorbar(range(5),[s['median_us'] for s in ss],yerr=[[s['median_us']-s['q25_us'] for s in ss],[s['q75_us']-s['median_us'] for s in ss]],fmt='none',ecolor='black',capsize=3)
   ax.set_xticks(range(5),names,rotation=45,ha='right');ax.set_ylabel('Latency (µs), median and IQR');ax.set_title(f'{op} [{m}, {n}] FP16 / {mode}');ax.set_yscale('log')
 fig.savefig(plots/'representative-latencies.png');plt.close(fig)
 aggregates=[]
 for op,mode,base in itertools.product(['rmsnorm','residual_rmsnorm','swiglu'],['graph','events'],['eager','compile','triton_initial','triton_unfused']):
  rr=[r for r in comparisons if r['operator']==op and r['mode']==mode and r['comparison']==base]
  if not rr:continue
  vals=[r['speedup'] for r in rr]
  aggregates.append(dict(operator=op,mode=mode,comparison=base,cases=len(rr),geomean=math.exp(statistics.mean(math.log(v) for v in vals)),minimum=min(vals),maximum=max(vals),wins_gt_5pct=sum(v>1.05 for v in vals),losses_gt_5pct=sum(v<1/1.05 for v in vals),iqr_overlaps=sum(r['iqr_overlaps'] for r in rr)))
 save(out/'aggregate.csv',aggregates)
 counts=[]
 for f in sorted((out/'traces').glob('trace-*.json')):
  trace=json.loads(f.read_text())['traceEvents']
  op,m,n,base=f.stem.removeprefix('trace-').split('-',3)
  kernels=[e for e in trace if e.get('cat')=='kernel']
  count=len(kernels)
  assert count==({'eager':5 if op=='swiglu' else 10,'compile':1,'triton_tuned':1,'triton_unfused':2}[base])
  counts.append(dict(operator=op,rows=m,width=n,baseline=base,kernel_launches=count,gpu_annotation_markers=sum(e.get('cat')=='gpu_user_annotation' for e in trace),kernel_names=';'.join(e['name'] for e in kernels),trace_file='traces/'+f.name))
 assert len(counts)==12
 save(out/'kernel-counts.csv',counts)
 audit=dict(raw_rows=len(raw),summary_rows=len(summaries),unique_groups=len(groups),all_groups_have_five_batches=True,all_values_finite_positive=True,full_expected_coverage=True)
 audit.update(inspected_traces=len(counts),trace_kernel_counts_verified=True)
 (out/'audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit))

if __name__=='__main__':main()
