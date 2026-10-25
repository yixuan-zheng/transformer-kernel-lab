import torch

def rmsnorm(x, weight, eps=1e-6):
    xf = x.float()
    return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps) * weight.float()).to(x.dtype)

def residual_rmsnorm(x, residual, weight, eps=1e-6):
    updated = x + residual
    return rmsnorm(updated, weight, eps), updated

def swiglu(gate, up):
    return (torch.nn.functional.silu(gate.float()) * up.float()).to(gate.dtype)
