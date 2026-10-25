"""Generate the Performance report from audited saved measurements (stdlib only)."""
import csv,io,json,math,statistics,argparse
from pathlib import Path

def read(p):return list(csv.DictReader(p.open()))
def gm(rr):return math.exp(statistics.mean(math.log(float(x['speedup'])) for x in rr))
def table(header,rows):return '| '+' | '.join(header)+' |\n| '+' | '.join(['---']*len(header))+' |\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows)

def main():
 p=argparse.ArgumentParser();p.add_argument('directory');a=p.parse_args();out=Path(a.directory)
 comparisons=read(out/'comparisons.csv');summary=read(out/'summary.csv');audit=json.loads((out/'audit.json').read_text())
 assert audit['raw_rows']==8520 and audit['trace_kernel_counts_verified']
 selected=json.loads((out/'selected.json').read_text())
 def subset(op,mode,base,primary=True):
  return [r for r in comparisons if r['operator']==op and r['mode']==mode and r['comparison']==base and (not primary or op!='swiglu' or int(r['width'])>8192)]
 overall=[]
 for mode in ['graph','events']:
  for op in ['rmsnorm','residual_rmsnorm','swiglu']:
   rr=subset(op,mode,'compile');v=[float(r['speedup']) for r in rr]
   overall.append([op,mode,len(rr),f'{gm(rr):.3f}×',sum(x>1.05 for x in v),sum(x<1/1.05 for x in v),len(v)-sum(x>1.05 or x<1/1.05 for x in v)])
 tuning=[]
 for op in ['rmsnorm','residual_rmsnorm','swiglu']:
  rr=[r for r in subset(op,'graph','triton_initial',False) if not(r['dtype']=='torch.float16' and (int(r['rows']),int(r['width'])) in [(32,1024),(32,8192),(512,1024),(512,8192)]) and (op=='swiglu' or int(r['width'])<=4096)]
  v=[float(r['speedup']) for r in rr]
  tuning.append([op,selected[op],len(rr),f'{gm(rr):.3f}×',f'{min(v):.3f}–{max(v):.3f}×',sum(x>1.05 for x in v),sum(x<1/1.05 for x in v)])
 case_rows=[]
 for op,m,n in [('residual_rmsnorm',1,4096),('residual_rmsnorm',2048,8192),('swiglu',32,14336)]:
  for base in ['eager','compile','triton_initial','triton_tuned','triton_unfused']:
   r=next(r for r in summary if r['operator']==op and int(r['rows'])==m and int(r['width'])==n and r['dtype']=='torch.float16' and r['mode']=='graph' and r['baseline']==base)
   case_rows.append([f'{op} [{m}, {n}]',base,f"{float(r['median_us']):.3f}",f"{float(r['q25_us']):.3f}–{float(r['q75_us']):.3f}"])
 counts=read(out/'kernel-counts.csv');launchrows=[]
 for op,m,n in [('residual_rmsnorm','1','4096'),('residual_rmsnorm','2048','8192'),('swiglu','32','14336')]:
  launchrows.append([f'{op} [{m}, {n}]']+[next(r['kernel_launches'] for r in counts if (r['operator'],r['rows'],r['width'],r['baseline'])==(op,m,n,b)) for b in ['eager','compile','triton_tuned','triton_unfused']])
 text=(out/'nsight-metrics.csv').read_text();lines=text.splitlines();start=next(i for i,s in enumerate(lines) if s.startswith('"ID"') or s.startswith('ID,'))
 metrics=list(csv.DictReader(io.StringIO('\n'.join(lines[start:]))))
 assert len(metrics)==9
 assert {r['Kernel Name'] for r in metrics}=={'_add','_norm'}
 byid={i:{r['Metric Name']:int(r['Metric Value'].replace(',','')) for r in metrics if r['ID']==str(i)} for i in range(3)}
 counterrows=[];counters=[]
 for label,ids in [('Unfused add + RMSNorm',[0,1]),('Fused residual RMSNorm',[2])]:
  dramread=sum(byid[i]['dram__bytes_read.sum'] for i in ids);dramwrite=sum(byid[i]['dram__bytes_write.sum'] for i in ids);l2=sum(byid[i]['lts__t_bytes.sum'] for i in ids)
  counters.append(dict(path=label,dram_read_bytes=dramread,dram_write_bytes=dramwrite,l2_bytes=l2))
  counterrows.append([label,f'{dramread/2**20:.3f}',f'{dramwrite/2**20:.3f}',f'{l2/2**20:.3f}'])
 with (out/'nsight-summary.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(counters[0]));w.writeheader();w.writerows(counters)
 l2saved=(counters[0]['l2_bytes']-counters[1]['l2_bytes'])/2**20
 report='''# Performance study

Three forward-only Transformer operators were measured on NVIDIA L40S.

## Result

Triton improves on the measured default compiled baseline in many cases, but fusion is not uniformly beneficial. The clearest loss is residual RMSNorm at [2048, 8192]: its fused implementation is slower even though it launches fewer kernels and reduces recorded L2 traffic. Compiled PyTorch already provides equivalent one-kernel fusion in every profiled residual/SwiGLU case; launch elimination is therefore not an explanation for Triton-versus-compile differences there.

## Run and contracts

L40S compute capability 8.9, driver 615.71.09, PyTorch 2.4.1+cu121, Triton 3.0.0, CUDA runtime 12.1. The initial clock snapshot was idle; clocks were not locked.

Rows: 1, 8, 32, 128, 512, 2048. Normalization widths: 1024, 2048, 4096, 8192. FP16 and BF16 throughout. SwiGLU primary widths: **11008, 14336, 28672**, representing intermediate MLP dimensions; the four supplementary widths remain supplementary. These choices correspond to published Llama configurations in the [AWS engineering reference](https://awslabs.github.io/ai-on-sagemaker-hyperpod/docs/slurm-orchestration/training-and-fine-tuning/fsdp/fully-sharded-data-parallel). These are operator workloads, not full decode/prefill or model benchmarks.

All baselines use identical tensor shapes and dtypes. RMSNorm uses FP32 accumulation and weight multiplication before casting. Residual addition rounds to input dtype before normalization and returns **both** normalized output and updated residual; nothing is mutated. Unfused Triton writes that residual in an add kernel and reads it in the same normalization kernel/configuration used for the fused comparison. SwiGLU's two-kernel control writes a **FP32** SiLU intermediate, preserving the FP32 activation/multiply contract. All benchmark implementations passed reference comparisons before timing with tolerances (FP16 atol/rtol 0.002; BF16 0.02).

Comparisons: eager PyTorch, default `torch.compile`, initial Triton, tuned Triton, and unfused Triton for the two fused operators. No standalone unfused comparator is needed for RMSNorm. FlashAttention, vLLM and FlashInfer were absent; an external serving stack was not installed. The compiler baseline uses its default adaptive shape policy, not an exhaustive search over compiler settings.

## Timing and main comparisons

Five batches per case, randomized order of baseline/timing-mode pairs, randomized shape order, seed 2026. Every implementation is checked and warmed before measurement; JIT/compilation is excluded. Event batches average 30 invocations. Graph batches use 15 ms replay measurements on a side stream. Tables report the median of five batch averages; IQR and min/max are in `summary.csv`. IQR is within-run variability, not a confidence interval or independent-session reproducibility claim.

The following table uses all normalization points and only the **36 realistic-width SwiGLU points**. Ratios are geometric means of per-case **compiled median / tuned median**. Wins >1.05× and losses <1/1.05× are descriptive thresholds, not statistical significance tests.

'''+table(['Operator','Timing','Cases','Geo. speedup vs compile','Wins','Losses','Within band'],overall)+'''

Detailed per-case values, spread and IQR overlap flags are preserved in `summary.csv` and `comparisons.csv`. Supplementary SwiGLU widths are included in the complete CSVs and heatmaps, rather than pooled into the primary table. Across all widths, graph speedups over eager have geometric means 10.491× RMSNorm, 8.822× residual RMSNorm, and 8.208× SwiGLU; these eager comparisons are not the project's strongest baseline.

Graph timings repeatedly reuse captured buffers and can be cache resident; they suppress much of host submission overhead. Event intervals can include GPU idle gaps caused by Python/dispatch/allocation work and shared-node CPU scheduling. They are not pure kernel timings or synchronized end-to-end wall time. Allocations are included at the callable level, but graph replay removes their normal host costs. Triton benchmark callables use direct launches rather than the validating public wrappers, as the reference callables also omit wrapper validation. Do not combine graph and event ratios or interpret either as model-serving speedup.

## Bounded optimization

Only four FP16 development shapes were used: rows 32/512 × widths 1024/8192. Three graph batches per candidate, 120 tuning samples total. Norm candidates: 4, 8, 16 warps; SwiGLU candidates: 128, 256, 512, 1024 elements per block, four warps. One fixed configuration per operator was selected by geometric-mean development latency; no per-shape dispatch or extensive autotuning was added.

The table below excludes development points and, for normalization, excludes width 8192 where initial and selected settings are both eight warps. Ratios are **initial median / tuned median**, graph mode.

'''+table(['Operator','Selected config','Changed validation cases','Geo. speedup','Range','Wins >5%','Losses >5%'],tuning)+'''

Normalization tuning has a modest measured effect. The width-8192 repeated measurements are effectively a noise control, not changed code. SwiGLU's primary intermediate-width points show a 1.341× geometric-mean initial/tuned ratio, while supplementary small cases include regressions. At [32,14336] FP16, changing 256 to 1024 elements reduces the launch grid from 1792 to 448 blocks and lowers graph latency from 2.189 to 1.637 µs (1.337×). Both versions are already one kernel and have the same logical I/O; this is a launch-configuration result, not additional fusion. Selected configurations live in `selected.json` and the benchmark launcher; the public defaults are documented in the kernel contracts.

## Representative measurements

Graph-mode FP16 latencies, microseconds; each entry includes its IQR. The plots include a separate event-mode panel.

'''+table(['Case','Baseline','Median µs','IQR µs'],case_rows)+'''

Fusion helps the small residual case: 2.298→1.530 µs versus unfused Triton, with two kernels reduced to one. At the large residual case, fusion instead increases 158.519→181.293 µs; compiled PyTorch is 156.796 µs. BF16 at the same large shape also loses against compile (182.313 versus 134.201 µs). The measurements establish this throughput tradeoff but do not establish its occupancy/register/vectorization root cause; those counters were not collected. Fewer launches and fewer bytes alone are insufficient predictors of latency here.

SwiGLU is also not universally better than compile: at [2048,14336] BF16 it measures 268.092 versus 252.858 µs, despite its overall primary-width advantage. All losses and near-equalities remain in the CSVs and heatmaps.

## Profiler evidence

Twelve Chrome traces in `traces/` were inspected. Kernel counts below come from events with category `kernel`, not every CUDA-device event. `profiles.csv` records device-event totals, which include one GPU annotation marker per trace; `kernel-counts.csv` is the audited launch-count table.

'''+table(['FP16 case','Eager','Compile','Triton tuned','Triton unfused'],launchrows)+'''

Logical traffic savings relative to the equivalent unfused Triton controls: residual fusion eliminates one input-dtype residual reread, **2×rows×width bytes** (8 KiB for [1,4096], 32 MiB for [2048,8192]). The residual output write is required in both paths and is **not** counted as saved. SwiGLU eliminates an FP32 activation write and read, **8×rows×width bytes** (3.5 MiB for [32,14336]). These are algorithmic counts, not DRAM counter readings.

Nsight Compute successfully captured `_add`, standalone `_norm`, and fused `_norm` at [2048,8192] FP16, using the eight-warp normalization configuration. Raw counters are in `nsight-metrics.csv`; parsed sums are in `nsight-summary.csv`.

'''+table(['Path','DRAM reads MiB','DRAM writes MiB','L2 traffic MiB'],counterrows)+f'''

The recorded L2 reduction is **{l2saved:.3f} MiB**, close to the logical 32 MiB residual reread eliminated. Counter collection is a separate replay experiment with profiler cache/replay behavior, not the randomized benchmark. Stores can remain in cache beyond a kernel's counter window (including the recorded zero DRAM writes for standalone norm); therefore these DRAM figures are not a complete application writeback balance or steady-state bandwidth claim. Lower L2 traffic did not translate to lower benchmark latency in this case.

`summary.csv` also includes a logical-minimum-I/O/latency estimate: norm counts weights once, residual counts both outputs, and unfused controls include their intermediate I/O. Eager/compiled values use the ideal fused lower bound and do not model eager intermediates. These estimates are explicitly not measured DRAM bandwidth and can reflect cache reuse.

## Artifacts, audit and limits

- `raw.csv`: 8,520 samples; `summary.csv`: 1,704 median/IQR/min/max rows; `comparisons.csv` and `aggregate.csv`: derived ratios.
- `tuning.csv`: 120 bounded-sweep samples; `selected.json`: fixed selections.
- `trace-*.json`, `kernel-counts.csv`, `nsight-metrics.csv`, `nsight-summary.csv`: profiler evidence.
- `plots/speedup-vs-compile-graph.png`, `plots/speedup-vs-compile-events.png`, `plots/optimization-validation.png`, `plots/representative-latencies.png`: inspected plots. Heatmap labels are speedup ratios; colors are log2 ratios, saturating at 0.25×/4×.
- `audit.json`: complete expected coverage, five unique batches per group, finite positive timings, and all 12 trace counts verified. No requested timing case was skipped or failed correctness. Environment metadata and measurement data are included with the results.

Unsupported public inputs remain CPU, noncontiguous layouts, mixed shapes/devices/dtypes, gradient requests and normalization widths above 16384. BF16/FP16 are supported; other precisions, backward passes and other architectures are outside scope. Performance coverage stops at norm width 8192 while correctness coverage includes 16384. No assertions are made about all possible NaN/Inf inputs. One GPU architecture and one timing run limit portability and repeatability claims. Clock state was recorded before warmup, not locked or continuously sampled. No exhaustive production-library or compiler-setting comparison was performed.

Run `tkl-bench --output results/new-run` on an allocated L40S with a fresh output directory. CPU analysis requires Matplotlib and NumPy: `python benchmarks/analyze.py RESULTS_DIRECTORY`, then `python benchmarks/report.py RESULTS_DIRECTORY`. The latter also requires the saved Nsight CSV for this report. No rerun is needed to inspect the saved evidence.

The measurements cover these three operators on L40S; no model-level speedup is claimed.
'''
 (out/'REPORT.md').write_text(report)
 analysis='''# Performance analysis — L40S

The strongest result is a workload-dependent comparison rather than a universal fusion claim. In graph replay, tuned Triton has geometric-mean speedups over the measured default compiled baseline of **1.537× RMSNorm**, **1.565× fused residual RMSNorm**, and **1.235× SwiGLU at realistic intermediate widths**. These are equal-weight per-case ratios across both FP16/BF16; they are not model-level speedups.

The bounded launch sweep changed little for normalization: on changed, non-development points, initial/tuned geometric means are 1.017× and 1.024×. SwiGLU benefits more: 1.341× on primary widths, with regressions in some supplementary small cases. At [32,14336] FP16 the larger tile reduces blocks fourfold and improves graph latency 2.189→1.637 µs. This does not save another launch or additional logical bytes.

Fusion helps small residual workloads: [1,4096] FP16 falls from 2.298 µs unfused to 1.530 µs fused, removing one launch and an 8 KiB residual reread. But at [2048,8192], fused FP16 is **181.293 µs**, slower than unfused **158.519 µs** and compile **156.796 µs**. Nsight records lower L2 traffic for fusion, so traffic reduction alone cannot explain performance. The available counters do not isolate the remaining throughput bottleneck.

Compiled PyTorch already emits one kernel in all three profiled cases, matching fused Triton's launch count. Any advantage over compile in those cases comes from differences beyond simply fusing eager operations. SwiGLU [2048,14336] BF16 loses to compile, 268.092 versus 252.858 µs. The heatmaps retain these losses.

Five randomized batches expose within-run variability. Read medians together with IQR/min/max in summary.csv; event measurements include host-induced GPU gaps, and graph replay can be cache resident. Neither mode measures end-to-end inference. Hardware-counter collection is a separate profiler replay, not a bandwidth measurement of the benchmark loop. See REPORT.md for all contracts, caveats, counts and supporting artifacts.
'''
 (out/'PERFORMANCE.md').write_text(analysis)
 print('REPORT_PASS',len(report.split()),'words')

if __name__=='__main__':main()
