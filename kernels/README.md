# Kernel contracts

Import `rmsnorm`, `residual_rmsnorm`, and `swiglu` from `kernels`.

Inputs must be contiguous CUDA tensors in FP16 or BF16, on the same device, with matching dtypes. Gradients are unsupported and `requires_grad=True` is rejected. Wrappers allocate fresh outputs and never mutate inputs.

- `rmsnorm(x, weight, eps=1e-6)`: x is [rows, width], weight is [width]. Width is 1 through 16,384; zero rows are supported. Accumulation, reciprocal square root, and weight multiplication use FP32, followed by an input-dtype output cast. Epsilon must be finite and positive.
- `residual_rmsnorm(x, residual, weight, eps=1e-6)`: returns `(normalized, updated_residual)`. Addition rounds to input dtype before normalization, matching the reference. The updated residual is stored as an output in the same kernel launch. Its required write is not counted as an eliminated intermediate write.
- `swiglu(gate, up)`: matching contiguous shapes of any rank, including empty tensors. Computes `silu(gate) * up` using FP32 intermediates, then casts once. NaNs, infinities and overflow are not sanitized.

Normalization uses one Triton program per row, masked power-of-two blocks, four warps through width 4096 and eight for larger widths. SwiGLU uses flattened 256-element blocks. These are initial configurations, not claimed optimal settings. Implicit FP fusion is disabled to keep operation ordering comparable to the reference.

Run GPU validation from the project root: `python -m tests.validate_kernels`. Tests compare all returned outputs, check input immutability, and reject unsupported layouts, dtypes, epsilon values, weight shapes and gradient requests. The performance study includes tuning, equivalent unfused comparisons and profiling.

The implementation uses standard Triton reduction and masking patterns described in the official LayerNorm tutorial: https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html . No source code was copied from third-party inference libraries.
