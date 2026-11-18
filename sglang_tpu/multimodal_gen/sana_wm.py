# SPDX-License-Identifier: Apache-2.0
"""SANA-WM on TPU workers: real-valued RoPE tables, float32 noise draws,
compiled streaming blocks.

Runtime patch, tracked debt:
- Why: SANA-WM's DiT keeps its 3D RoPE table as a complex128 buffer and
  rotates queries, keys and the camera branch by float64 complex products;
  torch_tpu cannot move complex128 tensors to the device (MOVA failed at that
  move, CDK j-bc9336da). Separately, the SANA-WM stages draw their starting,
  refiner and realtime noise in bfloat16 from a device generator, and torch_tpu
  builds a bfloat16 normal from 7 random bits (128 values). The streaming
  stage calls ``SanaWMBlock.forward_long``, which bypasses the compiled
  ``__call__`` of a regionally compiled block: stage 1 runs eagerly.
- How: :func:`install` (1) keeps ``WanRotaryPosEmbed``'s table as float32
  ``(cos, sin)`` pairs of the same float64 frequencies, in a plain attribute
  that ``Module.to`` neither moves nor casts, and returns per-token tables as
  ``[2, 1, 1, N, D/2]`` (cos, then sin, in the complex table's layout), so the
  DiT's slicing of the last two dims is unchanged; (2) rebinds the GDN,
  softmax and camera rotations to a float32 rotation (the pair-swap form of
  ``sglang_tpu.multimodal_gen.rope`` for bfloat16 inputs) that rounds once to
  the input dtype; (3) has the methods holding SANA-WM's noise draws draw
  bfloat16 and float16 normals on a TPU device in float32 and round them;
  (4) routes a regionally compiled block's ``forward_long`` through its
  compiled ``__call__``, with the streaming scans' backward pass by reverse
  indexing (torch_tpu fails to compile a view of the flipped copies, j-a59c2834)
  and the fused camera weights cached outside the graph. A compiled path
  whose call raises runs eagerly from then on.
- Upstream PR or issue: none.
- Remove when: SANA-WM builds real RoPE tables on every platform, torch_tpu
  draws low-precision normals from float32 samples, and the streaming stage
  calls compiled modules.
"""

from __future__ import annotations

import functools

import torch
from torch.utils._device import _device_constructors

from sglang.kernels.ops.diffusion.common.fallback_torch import (
    apply_rotary_embedding_native,
)
from sglang.multimodal_gen.runtime.models.dits import sana_wm_components as components
from sglang.multimodal_gen.runtime.models.dits.sana_wm import SanaWMBlock
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm import (
    base,
    realtime_chain,
    refiner,
    streaming_refiner,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang_tpu.multimodal_gen.rope import apply_rotary_embedding_pair_swap

logger = init_logger(__name__)

# Devices whose low-precision normals are drawn in float32.
_DEVICE_TYPES = ("tpu",)
_LOW_PRECISION = (torch.bfloat16, torch.float16)

_STOCK_INIT_FREQS = components.WanRotaryPosEmbed._init_freqs_buffer
_STOCK_ROTARY_EMB_DN = components._apply_rotary_emb_dn
_STOCK_ROTARY_EMB_BHND = components._apply_rotary_emb_bhnd
_STOCK_COMPLEX_ROPE = components._apply_complex_rope
_STOCK_REVERSE_SCAN = components._sana_wm_reverse_scan
_STOCK_CAM_QKV_PARAMS = (
    components.BidirectionalGDNUCPESinglePathLiteLA._get_cam_qkv_params
)
_STOCK_BLOCK_FORWARD = SanaWMBlock.forward
_STOCK_BLOCK_FORWARD_LONG = SanaWMBlock.forward_long
# Compiled paths whose call raised once: they run eagerly from then on.
_EAGER_PATHS = set()

# The methods that hold SANA-WM's random draws.
NOISE_METHODS = (
    (base.SanaWMBeforeDenoisingStage, "_prepare_noise_latents"),
    (refiner.SanaWMLTX2RefinerStage, "_refine_one"),
    (streaming_refiner.RefinerChunkRunner, "refine_block"),
    (realtime_chain.SanaWMRealtimeLatentPrepStage, "forward"),
)


def init_freqs_buffer(self) -> None:
    """The stock table as a float32 ``[2, max_seq_len, D/2]`` attribute (cos, sin)."""
    self.__dict__.pop("_freqs", None)
    _STOCK_INIT_FREQS(self)
    freqs = self._buffers.pop("_freqs")
    self._freqs = torch.view_as_real(freqs).float().movedim(-1, 0).contiguous()


def rope_forward(self, fhw, device, frame_index=None) -> torch.Tensor:
    """``WanRotaryPosEmbed.forward`` on the real table: ``[2, 1, 1, N, D/2]``."""
    fspec, pph, ppw = fhw
    freqs = self._freqs.to(device)
    d = self.attention_head_dim
    t_size = d // 2 - 2 * (d // 6)
    ft, fh, fw = freqs.split_with_sizes([t_size, d // 6, d // 6], dim=-1)
    if frame_index is not None:
        f_pos = frame_index.to(device=device, dtype=torch.long)
    elif isinstance(fspec, (tuple, list)):
        f_pos = torch.arange(
            int(fspec[0]), int(fspec[1]), device=device, dtype=torch.long
        )
    else:
        f_pos = torch.arange(int(fspec), device=device, dtype=torch.long)
    ppf = int(f_pos.shape[0])
    shape = (2, ppf, pph, ppw, -1)
    out = torch.cat(
        [
            ft[:, f_pos].view(2, ppf, 1, 1, -1).expand(shape),
            fh[:, :pph].view(2, 1, pph, 1, -1).expand(shape),
            fw[:, :ppw].view(2, 1, 1, ppw, -1).expand(shape),
        ],
        dim=-1,
    )
    return out.reshape(2, 1, 1, ppf * pph * ppw, -1)


def _rotate(x: torch.Tensor, freqs: torch.Tensor, inverse: bool = False):
    """Rotate the pairs of ``x`` ``[B, N, H, D]`` by a real table in float32."""
    cos, sin = freqs[0, 0, 0], freqs[1, 0, 0]
    if inverse:
        sin = -sin
    if x.dtype == torch.bfloat16:
        return apply_rotary_embedding_pair_swap(x, cos, sin, interleaved=True)
    # The native form runs in its input dtype: rotate float32, round once.
    return apply_rotary_embedding_native(x.float(), cos, sin, interleaved=True).to(
        x.dtype
    )


def apply_rotary_emb_dn(hidden_states: torch.Tensor, freqs: torch.Tensor):
    """``_apply_rotary_emb_dn`` (``(B, H, D, N)``); real tables rotate in float32."""
    if freqs.is_complex():
        return _STOCK_ROTARY_EMB_DN(hidden_states, freqs)
    return _rotate(hidden_states.permute(0, 3, 1, 2), freqs).permute(0, 2, 3, 1)


def apply_rotary_emb_bhnd(hidden_states: torch.Tensor, freqs: torch.Tensor):
    """``_apply_rotary_emb_bhnd`` (``(B, H, N, D)``); real tables rotate in float32."""
    if freqs.is_complex():
        return _STOCK_ROTARY_EMB_BHND(hidden_states, freqs)
    return _rotate(hidden_states.transpose(1, 2), freqs).transpose(1, 2)


def apply_complex_rope(
    hidden_states: torch.Tensor, freqs: torch.Tensor, inverse: bool = False
):
    """``_apply_complex_rope`` (``(B, H, N, D)``); the inverse negates the sines."""
    if freqs.is_complex():
        return _STOCK_COMPLEX_ROPE(hidden_states, freqs, inverse)
    return _rotate(hidden_states.transpose(1, 2), freqs, inverse).transpose(1, 2)


def _compiled_or_eager(path, compiled, eager, *args, **kwargs):
    """Run ``compiled``; once a call raises, ``path`` runs ``eager`` from then on."""
    if path not in _EAGER_PATHS:
        try:
            return compiled(*args, **kwargs)
        except Exception:
            logger.warning(
                "SANA-WM compiled %s failed; it runs eagerly from now on.",
                path,
                exc_info=True,
            )
            _EAGER_PATHS.add(path)
    return eager(*args, **kwargs)


def block_forward(self, *args, kv_cache=None, save_kv_cache=False, **kwargs):
    """``SanaWMBlock.forward``; given a stream cache, the streaming ``forward_long``."""
    if kv_cache is None:
        return _STOCK_BLOCK_FORWARD(self, *args, **kwargs)
    return _STOCK_BLOCK_FORWARD_LONG(
        self, *args, kv_cache=kv_cache, save_kv_cache=save_kv_cache
    )


def block_forward_long(self, x, *args, kv_cache, save_kv_cache):
    """``SanaWMBlock.forward_long`` through a regionally compiled block's
    ``__call__`` for TPU tensors."""
    stock = functools.partial(_STOCK_BLOCK_FORWARD_LONG, self)
    if self._compiled_call_impl is None or x.device.type not in _DEVICE_TYPES:
        return stock(x, *args, kv_cache=kv_cache, save_kv_cache=save_kv_cache)
    self.attn._get_cam_qkv_params()  # fills the fused camera weights' cache
    # Block 0 takes the patch embedding's transposed output: one input layout.
    return _compiled_or_eager(
        "stage-1 block",
        self,
        stock,
        x.contiguous(),
        *args,
        kv_cache=kv_cache,
        save_kv_cache=save_kv_cache,
    )


def get_cam_qkv_params(self):
    """``_get_cam_qkv_params``; compiled blocks take the fused weights the eager
    caller cached (the cache key reads data pointers, a graph break)."""
    cached = self._cam_qkv_params_cache
    if cached is not None and torch.compiler.is_compiling():
        return cached[1]
    return _STOCK_CAM_QKV_PARAMS(self)


def reverse_scan(q_rot, k_rot, v, beta, decay, *, reference, q=None, k=None):
    """``_sana_wm_reverse_scan``: TPU tensors take the reverse-indexed scan."""
    if q_rot.device.type not in _DEVICE_TYPES:
        return _STOCK_REVERSE_SCAN(
            q_rot, k_rot, v, beta, decay, reference=reference, q=q, k=k
        )
    return components._sana_wm_reverse_scan_impl(q_rot, k_rot, v, beta, decay, q, k)


def _float32_randn(stock):
    @functools.wraps(stock)
    def randn(*size, dtype=None, device=None, **kwargs):
        if (
            dtype in _LOW_PRECISION
            and device is not None
            and torch.device(device).type in _DEVICE_TYPES
            and kwargs.get("out") is None
        ):
            return stock(*size, dtype=torch.float32, device=device, **kwargs).to(dtype)
        return stock(*size, dtype=dtype, device=device, **kwargs)

    return randn


def _draws_in_float32(method):
    """Run ``method`` with ``torch.randn`` drawing low-precision normals in float32."""

    @functools.wraps(method)
    def wrapper(*args, **kwargs):
        stock = torch.randn
        torch.randn = _float32_randn(stock)
        try:
            return method(*args, **kwargs)
        finally:
            torch.randn = stock

    wrapper.__wrapped_for_tpu__ = True
    return wrapper


def install() -> None:
    rope_cls = components.WanRotaryPosEmbed
    rope_cls._init_freqs_buffer = init_freqs_buffer
    rope_cls.forward = rope_forward
    components._apply_rotary_emb_dn = apply_rotary_emb_dn
    components._apply_rotary_emb_bhnd = apply_rotary_emb_bhnd
    components._apply_complex_rope = apply_complex_rope
    components._sana_wm_reverse_scan = reverse_scan
    components.BidirectionalGDNUCPESinglePathLiteLA._get_cam_qkv_params = (
        get_cam_qkv_params
    )
    SanaWMBlock.forward = block_forward
    SanaWMBlock.forward_long = block_forward_long
    # ``with torch.device(...)`` routes the constructors in this cached set; fill
    # it first so it holds the stock torch.randn the draws swap out.
    _device_constructors()
    for cls, name in NOISE_METHODS:
        method = getattr(cls, name)
        if not getattr(method, "__wrapped_for_tpu__", False):
            setattr(cls, name, _draws_in_float32(method))
