# References and attribution

The operators and fusion patterns are established techniques. This project implements them to study their numerical contracts, launch configurations and measured performance. No source code was copied from the repositories below; their licenses do not license this project's independently written code. If future changes copy upstream code, preserve its license and required notices at that time.

- [Official Triton LayerNorm tutorial](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html): row reductions, masking and warp/block choices; conceptual engineering reference.
- [Unsloth normalization kernels](https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/rms_layernorm.py): reference for precision choices; not vendored.
- [FlashAttention normalization](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/triton/layer_norm.py): residual fusion and output-contract reference; not vendored or installed as a benchmark dependency.
- [vLLM normalization operations](https://github.com/vllm-project/vllm/blob/main/vllm/ir/ops/layernorm.py): production semantics reference; not installed as a comparator.
- [FlashInfer benchmark](https://github.com/flashinfer-ai/flashinfer/blob/main/benchmarks/bench_fused_add_rmsnorm.py): workload/benchmark reference; not run here.
- [Triton CUDA Graph timing API](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench_cudagraph.html): used through the pinned Triton 3.0 API, requiring side-stream capture.
- [Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html): interpretation of collected memory counters and replay behavior.
- [AWS Llama configuration table](https://awslabs.github.io/ai-on-sagemaker-hyperpod/docs/slurm-orchestration/training-and-fine-tuning/fsdp/fully-sharded-data-parallel): primary intermediate dimension choices 11008, 14336 and 28672.

Measured data and conclusions are in results/l40s. The README does not claim fastest kernels, new algorithms, production-library superiority or full-model speedups.
