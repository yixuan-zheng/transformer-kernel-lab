# Architecture and numerical contracts

`kernels/ops.py` contains the implementation; `kernels/__init__.py` exposes the three functions. Wrappers validate shape, dtype, contiguity, device and gradient state, allocate outputs, and launch within the input CUDA device context.

## Normalization

For each row, load the input with masked lanes, convert to FP32, reduce the squared values, divide by the true width, add epsilon and apply reciprocal square root. Multiply the FP32 normalized values by FP32 weights, then cast the output to the input dtype. Epsilon defaults to 1e-6 and must be positive and finite. Padding contributes zero to the reduction. Empty row batches produce empty outputs without launching.

The initial setting is four warps through width 4096, eight above it. Supported public width is at most 16384; the measured performance matrix stops at 8192. The bounded sweep selected eight warps globally for its tuned comparisons. It did not change the public API defaults because improvements were small and workload dependent.

## Residual outputs

Compute `updated = (x.float() + residual.float()).to(x.dtype)` before normalization. Return `(normalized, updated)` without mutation. The fused kernel stores updated and computes normalization from the same rounded value. The two-launch control writes updated in `_add`, then reads it in `_norm` with identical launch settings for the normalization step.

The fused and unfused paths both require the updated-residual output write. Fusion saves only its extra read: `rows * width * 2` bytes for FP16/BF16. Counting an eliminated write here would compare different output contracts.

## SwiGLU

Compute `silu(gate.float()) * up.float()` then cast once. The fused launch flattens matching contiguous tensors. The two-launch control stores the SiLU result in FP32 before multiplying; storing that intermediate in FP16/BF16 would change the computation. Fusion removes a four-byte write and read per element. The initial tile is 256 elements/four warps; The bounded sweep selected 1024 elements/four warps, with measured small-case regressions.

## Measurement boundary

Benchmark callables allocate outputs and directly launch kernels, without the public wrapper validation overhead; PyTorch reference callables likewise omit validation. Graph replay removes normal host allocation/dispatch costs; CUDA-event intervals over repeated calls can include host-induced GPU idle time. Compilation, correctness, warmup and profiling occur outside the timed sample intervals.

`summary.csv` includes a logical minimum I/O model; weights are counted once and unfused intermediate traffic is added explicitly. This is an ideal traffic model, not actual transaction counting, and does not model every eager intermediate. Separate Nsight L2/DRAM counters are preserved rather than substituted into that estimate.
