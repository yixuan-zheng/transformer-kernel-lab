# Performance analysis — L40S

The strongest result is a workload-dependent comparison rather than a universal fusion claim. In graph replay, tuned Triton has geometric-mean speedups over the measured default compiled baseline of **1.537× RMSNorm**, **1.565× fused residual RMSNorm**, and **1.235× SwiGLU at realistic intermediate widths**. These are equal-weight per-case ratios across both FP16/BF16; they are not model-level speedups.

The bounded launch sweep changed little for normalization: on changed, non-development points, initial/tuned geometric means are 1.017× and 1.024×. SwiGLU benefits more: 1.341× on primary widths, with regressions in some supplementary small cases. At [32,14336] FP16 the larger tile reduces blocks fourfold and improves graph latency 2.189→1.637 µs. This does not save another launch or additional logical bytes.

Fusion helps small residual workloads: [1,4096] FP16 falls from 2.298 µs unfused to 1.530 µs fused, removing one launch and an 8 KiB residual reread. But at [2048,8192], fused FP16 is **181.293 µs**, slower than unfused **158.519 µs** and compile **156.796 µs**. Nsight records lower L2 traffic for fusion, so traffic reduction alone cannot explain performance. The available counters do not isolate the remaining throughput bottleneck.

Compiled PyTorch already emits one kernel in all three profiled cases, matching fused Triton's launch count. Any advantage over compile in those cases comes from differences beyond simply fusing eager operations. SwiGLU [2048,14336] BF16 loses to compile, 268.092 versus 252.858 µs. The heatmaps retain these losses.

Five randomized batches expose within-run variability. Read medians together with IQR/min/max in summary.csv; event measurements include host-induced GPU gaps, and graph replay can be cache resident. Neither mode measures end-to-end inference. Hardware-counter collection is a separate profiler replay, not a bandwidth measurement of the benchmark loop. See REPORT.md for all contracts, caveats, counts and supporting artifacts.
