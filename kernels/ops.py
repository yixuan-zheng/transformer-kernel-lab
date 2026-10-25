"""Contiguous, forward-only Transformer operators for FP16/BF16 CUDA tensors."""
import math
import torch
import triton
import triton.language as tl

@triton.jit
def _norm(X, R, W, Y, U, N: tl.constexpr, EPS: tl.constexpr,
          RESIDUAL: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    offset = row * N + col
    x = tl.load(X + offset, col < N, 0).to(tl.float32)
    if RESIDUAL:
        r = tl.load(R + offset, col < N, 0).to(tl.float32)
        # Match the reference's input-dtype residual rounding before reduction.
        x = (x + r).to(X.dtype.element_ty).to(tl.float32)
        tl.store(U + offset, x, col < N)
    variance = tl.sum(x * x, axis=0) / N
    w = tl.load(W + col, col < N, 0).to(tl.float32)
    y = (x * tl.rsqrt(variance + EPS)) * w
    tl.store(Y + offset, y, col < N)

@triton.jit
def _swiglu(G, U, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    g = tl.load(G + i, i < N, 0).to(tl.float32)
    u = tl.load(U + i, i < N, 0).to(tl.float32)
    y = (g / (1.0 + tl.exp(-g))) * u
    tl.store(Y + i, y, i < N)

def _input(x):
    if not isinstance(x, torch.Tensor):
        raise TypeError('expected a torch.Tensor')
    if not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError('expected CUDA FP16 or BF16 tensor')
    if x.requires_grad:
        raise ValueError('forward-only operators do not support requires_grad')
    if not x.is_contiguous():
        raise ValueError('only contiguous tensors are supported')

def _pair(x, other):
    _input(other)
    if x.shape != other.shape or x.device != other.device or x.dtype != other.dtype:
        raise ValueError('inputs must have identical shape, device and dtype')

def _norm_args(x, weight, eps):
    _input(x); _input(weight)
    if x.ndim != 2 or x.shape[1] == 0 or x.shape[1] > 16384:
        raise ValueError('normalization expects [rows, width] with 1 <= width <= 16384')
    if weight.shape != (x.shape[1],) or weight.dtype != x.dtype or weight.device != x.device:
        raise ValueError('weight must have shape [width] and match input device/dtype')
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError('epsilon must be finite and positive')

def _launch_norm(x, weight, eps, residual=None):
    _norm_args(x, weight, eps)
    if residual is not None:
        _pair(x, residual)
    y = torch.empty_like(x)
    updated = torch.empty_like(x) if residual is not None else y
    if x.shape[0]:
        block = triton.next_power_of_2(x.shape[1])
        with torch.cuda.device(x.device):
            _norm[(x.shape[0],)](x, residual if residual is not None else x,
                                weight, y, updated, x.shape[1], eps,
                                residual is not None, block,
                                num_warps=4 if block <= 4096 else 8,
                                enable_fp_fusion=False)
    return (y, updated) if residual is not None else y

def rmsnorm(x, weight, eps=1e-6):
    return _launch_norm(x, weight, eps)

def residual_rmsnorm(x, residual, weight, eps=1e-6):
    return _launch_norm(x, weight, eps, residual)

def swiglu(gate, up):
    _input(gate); _pair(gate, up)
    y = torch.empty_like(gate)
    if gate.numel():
        with torch.cuda.device(gate.device):
            _swiglu[(triton.cdiv(gate.numel(), 256),)](
                gate, up, y, gate.numel(), 256, enable_fp_fusion=False)
    return y
