# SPDX-License-Identifier: Apache-2.0
"""VDN-H3 hybrid attention on TPU: compiled per block and video size.

``--attention-backend hybrid_window_attn_h3`` resolves to
:class:`TpuHybridWindowAttentionH3Backend` on TPU workers. Its impl hands the
VDN hybrid attention a compile-safe core and builds request metadata that holds
only what is fixed for a video size:

- The packed rows are ``[text | cond | audio | video | pad]``. The one
  per-prompt input is ``used = cu_seqlens[1]``, read on device as the
  live-length kernel reads it: the video starts at ``used - F * tpf`` and the
  text length is that start minus the condition and audio rows. Prompts of one
  video size and 512-row bucket share one compiled block.
- The window softmax is exact, built from the ring step kernel
  (``tpu_ring_attention_step``) with its float32 carry: the dense rows
  (text, condition, audio and the anchor frames) attend all keys; the window
  rows attend the global rows and anchor frames, then resume on their chunk's
  window frames, equal-shape chunks in one batched call.
- Every Ulysses exchange is a lane-dense all-to-all (the frame sums and the
  gate hidden too), and each rank's linear-branch heads are cut and cloned
  once per process, so all ranks run the same program.
- The linear branch runs between the compiled graphs of a block, one graph
  per stage: torch_tpu compiles it whole to wrong values (see
  ``_linear_branch``). The core's parts before and after it keep their
  collectives in one program each.
- The delta rule inverts I + A by a Newton-Schulz iteration (matmuls only);
  ``--attention-backend-config '{"vdn_spd_inverse": "cholesky"}'`` keeps the
  core's Cholesky chain.

The token refiner resolves this backend too and keeps the splash backend's
host-bounds SDPA path.
"""

from __future__ import annotations

import dataclasses
import functools

import torch

from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_ulysses_ctx,
    get_ulysses_parallel_world_size,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.hybrid_window_attn_h3 import (
    HybridWindowAttentionH3Metadata,
    HybridWindowAttentionH3MetadataBuilder,
    _chunk_groups,
    window_mask_frames,
)
from sglang.multimodal_gen.runtime.layers.usp import (
    _usp_all_to_all_single,
    _usp_input_all_to_all_packed_qkv,
    _usp_output_all_to_all,
)
from sglang.multimodal_gen.runtime.managers.forward_context import get_forward_context
from sglang.multimodal_gen.runtime.models.dits import minimax_h3_vdn as branch_ops
from sglang.multimodal_gen.runtime.models.dits.minimax_h3_vdn import (
    VDNH3Layout,
    cholesky_spd_inverse,
)
from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum,
    current_platform,
)
from sglang.multimodal_gen.runtime.utils.torch_compile import (
    build_torch_compile_kwargs,
)
from sglang_tpu.multimodal_gen.tpu_splash_attn import (
    TpuSplashAttentionBackend,
    TpuSplashAttentionImpl,
    _ring_kernel_fallback_reason,
    tpu_ring_attention_step,
)

# X <- X (2I - M X) from X = 2 / (2 + |M - I|_F) I squares the residual each
# step; 20 steps reach float32 rounding for |A|_F up to ~1e5 (|A|_F is at most
# the rows behind A, 1008 per frame at 1344x768).
NEWTON_SCHULZ_STEPS = 20


def newton_schulz_spd_inverse(m: torch.Tensor) -> torch.Tensor:
    """Batched inverse of SPD ``m`` with eigenvalues >= 1 (I + A, A PSD)."""
    eye = torch.eye(m.shape[-1], device=m.device, dtype=m.dtype)
    norm = (m - eye).square().sum((-2, -1), keepdim=True).sqrt()
    x = (2.0 / (2.0 + norm)) * eye
    for _ in range(NEWTON_SCHULZ_STEPS):
        x = 2.0 * x - x @ (m @ x)
    return 0.5 * (x + x.transpose(-1, -2))


_SPD_INVERSES = {
    "newton_schulz": newton_schulz_spd_inverse,
    "cholesky": cholesky_spd_inverse,
}


@dataclasses.dataclass
class TpuHybridWindowAttentionH3Metadata(HybridWindowAttentionH3Metadata):
    # Fixed per video size: compiled blocks read only these, not ``layout``.
    video_layout: VDNH3Layout | None = None  # the video rows alone
    cond_audio_rows: int = 0  # rows between the text and the video
    row_offset: torch.Tensor | None = None  # int64 [1], this rank's first row


def _prepare_branches(model, inverse: str) -> None:
    """Set each linear branch's inverse; under Ulysses, cut its per-head
    parameters to this rank's heads once (cloned: no storage offsets)."""
    if inverse not in _SPD_INVERSES:
        raise ValueError(
            f"vdn_spd_inverse={inverse!r}; expected one of {sorted(_SPD_INVERSES)}"
        )
    ws, rank = get_ulysses_ctx()
    for block in model.blocks:
        branch = block.attn.hybrid.linear_attention
        branch.spd_inverse = _SPD_INVERSES[inverse]
        if ws == 1 or getattr(branch, "tpu_head_shard", None) is not None:
            continue
        heads, dim = branch.local_heads // ws, branch.head_dim
        rows = slice(rank * heads * dim, (rank + 1) * heads * dim)
        params = [
            (branch.alpha.up.weight, rows),
            (branch.alpha.dt_bias, rows),
            (branch.alpha.A_log, slice(rank * heads, (rank + 1) * heads)),
            (branch.output_gate.up.weight, rows),
            (branch.output_gate.up.bias, rows),
        ]
        conv = branch.short_conv
        for name in conv.targets if conv is not None else ():
            params += [
                (getattr(conv, f"{name}_{part}")["weight"], rows)
                for part in ("sp", "tm")
            ]
        for param, part in params:
            if param is not None:
                param.data = param.data[part].clone()
        branch.local_heads = branch.alpha.local_heads = heads
        branch.output_gate.local_heads = heads
        branch.tpu_head_shard = (rank, ws)


class TpuHybridWindowAttentionH3MetadataBuilder(HybridWindowAttentionH3MetadataBuilder):
    def build(  # type: ignore[override]
        self,
        *,
        layout: VDNH3Layout,
        hybrid,
        device: torch.device,
        rope_cache_full=None,
        current_timestep: int = 0,
        model=None,
        server_args=None,
        **kwargs,
    ) -> TpuHybridWindowAttentionH3Metadata:
        if layout.video_end != layout.used:
            raise ValueError(
                "VDN-H3 on TPU needs the video rows last before the padding "
                f"(video ends at {layout.video_end}, used={layout.used})"
            )
        config = getattr(server_args, "attention_backend_config", None) or {}
        if model is not None:
            _prepare_branches(
                model, str(config.get("vdn_spd_inverse", "newton_schulz"))
            )
        ws, rank = get_ulysses_ctx()
        frames, tpf = layout.num_frames, layout.tokens_per_frame
        return TpuHybridWindowAttentionH3Metadata(
            current_timestep=current_timestep,
            layout=layout,
            full_cover=hybrid.full_cover(frames),
            rope_cache_full=rope_cache_full,
            video_layout=VDNH3Layout(
                seq_len=frames * tpf,
                used=frames * tpf,
                text_len=0,
                video_start=0,
                num_frames=frames,
                tokens_per_frame=tpf,
                frame_height=layout.frame_height,
                frame_width=layout.frame_width,
            ),
            cond_audio_rows=layout.video_start - layout.text_len,
            row_offset=torch.tensor([rank * (layout.seq_len // ws)], device=device),
        )


def _window_softmax(q, k, v, used, *, hybrid, frames, tpf, scale):
    """Heads-major [h, S, d] q/k/v -> [h, S, d]: rows [0, G) as in q, then the
    video rows frame by frame (G = S - frames * tpf; ``used`` is on device)."""
    seq_len = q.shape[1]
    globals_ = seq_len - frames * tpf
    video_start = used - frames * tpf
    video = video_start + torch.arange(frames * tpf, device=q.device)
    qv, kv, vv = (t.index_select(1, video) for t in (q, k, v))
    _, dense_rows, dense_cols = window_mask_frames(hybrid, frames)
    zero = torch.zeros((), dtype=torch.int32, device=q.device)
    paths: set[bool] = set()

    def span(t, lo, hi):  # video frames [lo, hi)
        return t[..., lo * tpf : hi * tpf, :]

    def step(qs, ks, vs, real, carry=None):
        # One kernel path per chain: the kernel and the reference carry differ.
        paths.add(_ring_kernel_fallback_reason(qs, ks, vs) is None)
        if carry is not None and len(paths) > 1:
            raise ValueError(
                f"VDN-H3 window chain mixes kernel paths at {tuple(qs.shape)}"
            )
        info = torch.stack([zero, (zero + real).to(torch.int32)])
        return tpu_ring_attention_step(qs, ks, vs, info, carry, scale)

    # Dense rows (rows [0, G) and the anchor frames) attend every live key.
    anchors = sorted(dense_rows)
    q_dense = torch.cat([q[:, :globals_]] + [span(qv, f, f + 1) for f in anchors], 1)
    dense, _ = step(q_dense[None], k[None], v[None], used)
    paths.clear()
    pieces = {f: span(dense[0, :, globals_:], i, i + 1) for i, f in enumerate(anchors)}
    window = [f for f in range(frames) if f not in dense_rows]
    if window:
        # The window rows first attend the anchor columns and rows [0, video start).
        cols = sorted(dense_cols)
        kb, vb = (
            torch.cat([span(t, f, f + 1) for f in cols] + [x[:, :globals_]], 1)[None]
            for t, x in ((kv, k), (vv, v))
        )
        first = window[0]
        out_b, carry = step(
            span(qv, first, window[-1] + 1)[None], kb, vb, len(cols) * tpf + video_start
        )
        bounds, _, _ = window_mask_frames(hybrid, frames)
        by_shape: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for group in _chunk_groups(hybrid.window_bounds(frames), dense_rows):
            lo, hi = bounds[group[0]]
            lo, hi = lo + (lo in dense_cols), hi - (hi in dense_cols)
            if lo > hi:  # nothing left in the window beyond the anchors
                pieces[group[0]] = span(
                    out_b[0], group[0] - first, group[-1] + 1 - first
                )
                continue
            by_shape.setdefault((len(group), hi - lo + 1), []).append((group[0], lo))
        # Then each chunk resumes on its window frames, one call per shape.
        for (nq, nk), items in by_shape.items():
            qs = torch.stack([span(qv, f, f + nq) for f, _ in items])
            ks = torch.stack([span(kv, lo, lo + nk) for _, lo in items])
            vs = torch.stack([span(vv, lo, lo + nk) for _, lo in items])
            # the carry rows of these frames: out [1, h, Q, d], lse [..., Q]
            rows = [slice((f - first) * tpf, (f - first + nq) * tpf) for f, _ in items]
            resume = (
                torch.cat([carry[0][:, :, r] for r in rows]),
                torch.cat([carry[1][..., r] for r in rows]),
            )
            out, _ = step(qs, ks, vs, nk * tpf, resume)
            pieces.update({f: out[i] for i, (f, _) in enumerate(items)})
    return torch.cat([dense[0, :, :globals_]] + [pieces[f] for f in sorted(pieces)], 1)


# The branch's stages, compiled one graph each: torch_tpu compiles each alone
# to the eager values, but not chained (the features into the frame statistics:
# rel_l2 0.44 at 1344x768). linear_features stays eager: compiled at the video
# shape it moves from the eager values (rel_l2 0.0028), which the (I + A)
# inverse then grows to 0.0058 in the branch output (CDK j-aa3608d0).
BRANCH_STAGES = (
    "frame_statistics",
    "delta_factor_apply",
    "run_boundary_scans",
    "gather_linear_state",
    "linear_epilogue",
)


@functools.cache
def _compiled_branch_stages() -> dict:
    kwargs = build_torch_compile_kwargs(mode=None)
    return {n: torch.compile(getattr(branch_ops, n), **kwargs) for n in BRANCH_STAGES}


@torch.compiler.disable
@torch._dynamo.config.patch(recompile_limit=64)
def _linear_branch(branch, layout, staged, *inputs):
    """``branch`` on its video rows between compiled block graphs, its stages
    compiled one graph each when ``staged`` (the block is compiled).

    torch_tpu (0.2.1.dev20261028) compiles the whole branch to wrong values:
    rel_l2 0.56 against float32 at 1344x768 (block 0, both inverses), 3e-3
    eager. The output is a fresh tensor.
    """
    stages = _compiled_branch_stages() if staged else {}
    eager = {name: getattr(branch_ops, name) for name in stages}
    vars(branch_ops).update(stages)
    try:
        return _run_branch(branch, layout, *inputs)
    finally:
        vars(branch_ops).update(eager)


def _run_branch(branch, layout, q, k, v, beta, gate, frame_mean, *text):
    text_k, text_v, text_beta = text
    return branch(
        q_raw=q,
        k_raw=k,
        v_raw=v,
        beta=beta,
        gate=gate,
        frame_mean=frame_mean,
        layout=layout,
        text_k_raw=text_k,
        text_v_raw=text_v,
        text_beta=text_beta,
    )


def _exchange(t: torch.Tensor, ws: int) -> torch.Tensor:
    # dim-0 chunk i goes to Ulysses rank i; chunk i of the result came from rank i
    return t if ws == 1 else _usp_all_to_all_single(t)


def _rows_to_window(
    attention,
    meta,
    x,
    q,
    k,
    v,
    softmax_gate,
    beta,
    gate_hidden,
    rope_cache,
    cu_seqlens,
    ws,
):
    """Compiled part 1: exchanges, the window softmax, the branch inputs.
    -> (softmax [h, S, d], row weights [n * h, S], gather index or None,
    branch inputs or None)."""
    from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import _apply_rope_qk

    branch = attention.hybrid.linear_attention
    frames, tpf = meta.video_layout.num_frames, meta.video_layout.tokens_per_frame
    local_rows, heads, dim = x.shape[0], q.shape[1] // ws, q.shape[2]
    seq_len, video_rows = local_rows * ws, frames * tpf
    globals_ = seq_len - video_rows
    used = cu_seqlens[1].long()
    video_start = used - video_rows
    rows = torch.arange(seq_len, device=x.device)
    video = video_start + rows[:video_rows]

    # Rows to heads: q/k/v, then the per-head scalars heads-major [n, h, S].
    q, k, v = _usp_input_all_to_all_packed_qkv(q, k, v)
    scalars = torch.stack([beta] if softmax_gate is None else [beta, softmax_gate])
    count = scalars.shape[0]
    scalars = _exchange(
        scalars.permute(2, 0, 1).reshape(ws, heads, count, local_rows), ws
    )
    scalars = scalars.permute(2, 1, 0, 3).reshape(count, heads, seq_len)
    cos_sin, positions = meta.rope_cache_full if ws > 1 else rope_cache
    # out of place: the branch reads the raw q/k
    q_sm, k_sm = _apply_rope_qk(
        attention.q_norm(q),
        attention.k_norm(k),
        cos_sin,
        positions,
        rotary_emb=attention.rotary_emb,
    )
    q_sm, k_sm, v_h = (t.transpose(0, 1).contiguous() for t in (q_sm, k_sm, v))
    scale = attention.softmax_scale
    live = (rows < used).to(q.dtype)
    gate = scalars[1] if softmax_gate is not None else torch.ones_like(scalars[0])
    if meta.full_cover:
        info = cu_seqlens[:2].to(torch.int32)
        out, _ = tpu_ring_attention_step(
            q_sm[None], k_sm[None], v_h[None], info, None, scale
        )
        return out[0], gate * live, None, None
    if branch.local_heads != heads:
        raise RuntimeError("VDN-H3 branch heads were not split for this Ulysses rank")
    # fp32 frame sums of this rank's rows (one-hot over device row ids), summed over ranks
    frame = torch.div(
        meta.row_offset + rows[:local_rows] - video_start, tpf, rounding_mode="floor"
    )
    onehot = (frame[None] == torch.arange(frames, device=x.device)[:, None]).float()
    sums = (onehot @ x.float()).expand(ws, -1, -1).contiguous()
    frame_mean = _exchange(sums, ws).sum(0) / tpf
    gate_rows = _exchange(gate_hidden.expand(ws, -1, -1).contiguous(), ws)
    gate_rows = gate_rows.reshape(seq_len, -1).index_select(0, video)
    beta_rows = scalars[0].t()
    text_mask = (rows[:globals_] < video_start - meta.cond_audio_rows).to(beta.dtype)
    softmax = _window_softmax(
        q_sm, k_sm, v_h, used, hybrid=branch.hybrid, frames=frames, tpf=tpf, scale=scale
    )
    # [rows before the video | video, frame by frame] back to packed row order
    src = torch.where(rows < video_start, rows, rows + seq_len - used)
    weight = torch.cat([gate * live, live.expand(heads, -1)])
    branch_inputs = (
        q.index_select(0, video),
        k.index_select(0, video),
        v.index_select(0, video),
        beta_rows.index_select(0, video),
        branch.output_gate.up_gate(gate_rows),
        frame_mean,
        k[:globals_],
        v[:globals_],
        beta_rows[:globals_] * text_mask[:, None],
    )
    return softmax, weight, src.clamp(max=seq_len - 1), branch_inputs


def _window_to_rows(softmax, weight, src, linear, ws):
    """Compiled part 2: [softmax | linear] heads back to rows in one exchange
    -> (softmax [L, H, d], linear [L, H * d] or None), fresh tensors."""
    heads, seq_len, dim = softmax.shape
    out = softmax
    if linear is not None:
        linear = linear.view(-1, heads, dim).transpose(0, 1)
        pad = linear.new_zeros(heads, seq_len - linear.shape[1], dim)
        out = torch.cat([out, torch.cat([pad, linear], 1)]).index_select(1, src)
    out = out * weight[..., None].to(out.dtype)
    out = _usp_output_all_to_all(out.transpose(0, 1)[None], head_dim=2)[0]
    out = out.reshape(seq_len // ws, ws, -1, heads, dim)
    # the caller's rest compiles as a graph of its own (see the eager branch),
    # and torch_tpu guards each compiled input's storage offset
    parts = [
        out[:, :, i].clone(memory_format=torch.contiguous_format)
        for i in range(out.shape[2])
    ]
    softmax_rows = parts[0].view(seq_len // ws, ws * heads, dim)
    if linear is None:
        return softmax_rows, None
    return softmax_rows, parts[1].view(seq_len // ws, -1)


def tpu_hybrid_attention_core(
    attention,
    x: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_gate: torch.Tensor | None,
    beta: torch.Tensor,
    gate_hidden: torch.Tensor,
    *,
    rope_cache,
    cu_seqlens: torch.Tensor,
    cu_seqlens_host=None,
    max_seqlen: int = 0,
    ulysses_active: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``MiniMaxH3VDNHybridAttention``'s core: row-shard q/k/v [L, H, d] and
    per-head scalars [L, H] -> (softmax [L, H, d], linear [L, H * d] or None)."""
    meta = get_forward_context().attn_metadata
    if not isinstance(meta, TpuHybridWindowAttentionH3Metadata):
        raise RuntimeError(
            "the TPU hybrid_window_attn_h3 core needs its request metadata in the "
            f"forward context (got {type(meta).__name__})"
        )
    ws = get_ulysses_parallel_world_size() if ulysses_active else 1
    # Under torch.compile the parts around the branch are graphs of their
    # own; spmd_parts keep each one's collectives in one torch_tpu program.
    parts = attention._attention_impl.spmd_parts
    staged = parts is not None and torch.compiler.is_compiling()
    if not staged:
        parts = (_rows_to_window, _window_to_rows)
    softmax, weight, src, branch_inputs = parts[0](
        attention, meta, x, q, k, v, softmax_gate, beta, gate_hidden,
        rope_cache, cu_seqlens, ws,
    )  # fmt: skip
    linear = None
    if branch_inputs is not None:
        branch = attention.hybrid.linear_attention
        linear = _linear_branch(branch, meta.video_layout, staged, *branch_inputs)
    return parts[1](softmax, weight, src, linear, ws)


class TpuHybridWindowAttentionH3Impl(TpuSplashAttentionImpl):
    """The splash impl (token refiner: host bounds, SDPA) plus the VDN seams."""

    hybrid_attention_core = staticmethod(tpu_hybrid_attention_core)
    hybrid_metadata_builder_cls = TpuHybridWindowAttentionH3MetadataBuilder

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # the core's compiled parts under torch_tpu's spmd_safe (TPU only)
        region = current_platform.prepare_spmd_region_wrapper()
        self.spmd_parts = None
        if region is not None:
            self.spmd_parts = (region(_rows_to_window), region(_window_to_rows))


class TpuHybridWindowAttentionH3Backend(TpuSplashAttentionBackend):
    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.HYBRID_WINDOW_ATTN_H3

    @classmethod
    def supports_ring_rotation(cls) -> bool:
        # VDN-H3 runs Ulysses only.
        return False

    @staticmethod
    def get_impl_cls() -> type[TpuHybridWindowAttentionH3Impl]:
        return TpuHybridWindowAttentionH3Impl

    @staticmethod
    def get_metadata_cls() -> type[TpuHybridWindowAttentionH3Metadata]:
        return TpuHybridWindowAttentionH3Metadata

    @staticmethod
    def get_builder_cls() -> type[TpuHybridWindowAttentionH3MetadataBuilder]:
        return TpuHybridWindowAttentionH3MetadataBuilder
