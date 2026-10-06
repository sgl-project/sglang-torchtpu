import torch
import torch.nn as nn


class RMSNormGated(nn.Module):
    """Pure-torch gated RMSNorm for TPU (``norm_before_gate=True``).

    Computes ``rmsnorm(x) * silu(z)`` over the last dim, matching FLA's
    ``RMSNormGated`` (whose Triton kernel is CUDA-only). The learnable
    ``weight`` param name matches so checkpoint loading is unchanged.
    """

    def __init__(self, hidden_size, eps=1e-6, activation="silu", dtype=None):
        super().__init__()
        # Qwen3.8 calls the same SiLU gate "swish", as accepted by FLA.
        # Guard other activations rather than silently applying no gate — that
        # would be a subtle correctness bug.
        if activation not in ("silu", "swish"):
            raise NotImplementedError(
                f"RMSNormGated gate activation {activation!r} not supported "
                "(only 'silu' or 'swish')"
            )
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.eps = eps

    def forward(self, x, z=None):
        in_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(-1, keepdim=True)
        out = x * torch.rsqrt(var + self.eps) * self.weight.float()
        if z is not None:
            out = out * torch.nn.functional.silu(z.float())
        return out.to(in_dtype)
