"""On-device test: the fused RoPE + all-layer KV write kernel matches eager torch.

Requires a TPU + torch_tpu + a jax with Pallas. NOT a CI unit test. Run with:

    PYTHONPATH=/path/to/sglang-torchtpu/python \
        python3 test/manual/tpu/test_tpu_kv_fused_rope_write.py

Three orthogonal properties:

- Packing and addressing: fused_rope_write_kv against fused_rope_write_kv_ref,
  which spells the same steps out per token in eager torch. Divergence means
  the kernel drifted from what the fused KV pool expects.
- Span decomposition: get_contiguous_slot_spans must reproduce the original
  slots exactly, on both branches. Slots are consecutive only within a page, so
  a span that outlives its page writes over a neighbour's KV -- and runs off the
  pool when it sits in the top page.
- Rotation angle: _rope_tables' cos/sin rows fed through that rotation must
  equal the draft's real apply_k_rope. A wrong angle does not crash -- it
  silently degrades draft quality and costs accept length.

Geometry mirrors the gpt-oss-120b DFlash draft @ TP8 (8 layers, 1 KV head,
head_dim 64, page 128) with the page count scaled down.
"""

import torch

from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.rotary_embedding.yarn import YaRNScalingRotaryEmbedding
from sglang_tpu.dflash_draft_kv_write import (
    get_contiguous_slot_spans,
)
from sglang_tpu.kv_fused_rope_write import (
    fused_rope_write_kv,
    fused_rope_write_kv_ref,
)

DEVICE = "tpu:0"
LAYERS = 8
HEADS = 1
HEAD_DIM = 64
NUM_PAGES = 32
PAGE_SIZE = 128
MAX_NUM_REQS = 8
# Trailing dims of a fused layer buffer: kv_heads_packed, kv_packing,
# padded_head_dim. Full buffer shape is [NUM_PAGES, PAGE_SIZE, *TAIL].
TAIL = (1, 4, 128)
# gpt-oss DFlash draft rope, verbatim from the checkpoint config. yarn matters
# here: it folds mscale into cos_sin_cache, so a gather-based path stays correct
# only because the scaling never appears outside the cache.
ROPE_THETA = 150000
MAX_POSITION = 131072
ROPE_SCALING = {
    "rope_type": "yarn",
    "factor": 32.0,
    "beta_fast": 32.0,
    "beta_slow": 1.0,
    "original_max_position_embeddings": 4096,
    "truncate": False,
}


def _buffers(dtype: torch.dtype, device: str = DEVICE) -> list[torch.Tensor]:
    """Zeroed fused layer buffers, each [NUM_PAGES, PAGE_SIZE, *TAIL]."""
    return [
        torch.zeros((NUM_PAGES, PAGE_SIZE, *TAIL), dtype=dtype, device=device)
        for _ in range(LAYERS)
    ]


def _inputs(
    *, num_tokens: int, dtype: torch.dtype, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random kernel inputs on device.

    Args:
      num_tokens: Tokens to write.
      dtype: Dtype for all four tensors.
      seed: Generator seed.

    Returns:
      Tuple of k, v, cos, sin. k and v are [LAYERS, num_tokens, HEADS,
      HEAD_DIM]; cos and sin are [num_tokens, HEAD_DIM // 2].
    """
    gen = torch.Generator().manual_seed(seed)
    shape = (LAYERS, num_tokens, HEADS, HEAD_DIM)

    def to_dev(t: torch.Tensor) -> torch.Tensor:
        return t.to(dtype).to(DEVICE)

    return (
        to_dev(torch.randn(shape, generator=gen)),
        to_dev(torch.randn(shape, generator=gen)),
        to_dev(torch.randn((num_tokens, HEAD_DIM // 2), generator=gen)),
        to_dev(torch.randn((num_tokens, HEAD_DIM // 2), generator=gen)),
    )


def _compare(
    *,
    spec: list[tuple[int, int, int]],
    max_span_len: int,
    num_tokens: int,
    seed: int,
    what: str,
    loc: torch.Tensor | None = None,
) -> None:
    """Assert the kernel and the eager reference land identical buffers.

    Args:
      spec: Span table as (destination slot, source token, length) triples.
        Source offsets are explicit because verify strides them by block_size
        while extend packs them back to back.
      max_span_len: Span length bound handed to the kernel.
      num_tokens: Source token count.
      seed: Input generator seed.
      what: Case name for the failure message.
      loc: Flat slots for the reference, when it must differ from the span
        table's -- verify masks uncommitted tokens onto slot 0, which the span
        kernel skips entirely.
    """
    dtype = torch.bfloat16
    to_dev = lambda xs: torch.tensor(xs, dtype=torch.int32, device=DEVICE)  # noqa: E731
    dst_starts = to_dev([d for d, _, _ in spec])
    src_starts = to_dev([s for _, s, _ in spec])
    span_lens = to_dev([n for _, _, n in spec])
    if loc is None:
        flat: list[int] = []
        for dst, _, length in spec:
            flat.extend(range(dst, dst + length))
        loc = torch.tensor(flat, dtype=torch.int64, device=DEVICE)
        masked = False
    else:
        masked = True
    k, v, cos, sin = _inputs(num_tokens=num_tokens, dtype=dtype, seed=seed)

    want, got = _buffers(dtype), _buffers(dtype)
    fused_rope_write_kv_ref(loc=loc, k=k, v=v, cos=cos, sin=sin, kv_bufs=want)
    fused_rope_write_kv(
        dst_starts=dst_starts,
        src_starts=src_starts,
        span_lens=span_lens,
        k=k,
        v=v,
        cos=cos,
        sin=sin,
        kv_bufs=got,
        max_span_len=max_span_len,
    )
    torch.accelerator.synchronize()

    bad = []
    for layer in range(LAYERS):
        a, b = want[layer].cpu(), got[layer].cpu()
        if masked:
            # Slot 0 is the reference's dump for masked tokens.
            a[0, 0], b[0, 0] = 0, 0
        if not torch.allclose(a, b, atol=3e-2, rtol=3e-2):
            bad.append(layer)
    print(f"  {what}: {LAYERS - len(bad)}/{LAYERS} layers match ({len(spec)} spans)")
    assert not bad, f"{what}: kernel diverges from eager reference on layers {bad}"


def test_matches_eager_reference() -> None:
    """The kernel's packing and index math match eager torch on device."""
    print("== fused rope + KV write vs eager reference ==")
    # Extend over scattered pages, page-aligned; the trailing empty span is what
    # the aligned derivation always emits.
    _compare(
        spec=[(5 * PAGE_SIZE, 0, 128), (9 * PAGE_SIZE, 128, 128), (0, 256, 0)],
        max_span_len=PAGE_SIZE,
        num_tokens=256,
        seed=0,
        what="extend aligned + empty span",
    )
    # Extend resuming mid-page: short first span, then whole pages.
    _compare(
        spec=[
            (3 * PAGE_SIZE + 91, 0, 37),
            (7 * PAGE_SIZE, 37, 128),
            (2 * PAGE_SIZE, 165, 59),
        ],
        max_span_len=PAGE_SIZE,
        num_tokens=224,
        seed=1,
        what="extend unaligned head",
    )
    # Verify: one span per request, length = accepted count, so uncommitted
    # slots go unwritten. Source is strided by block_size, not packed. The
    # reference still dumps the masked tokens on slot 0.
    block = 10
    commits = [10, 0, 3, 7]
    starts = [1000, 1500, 2100, 300]
    off = torch.arange(block)
    loc2d = torch.tensor(starts)[:, None] + off[None, :]
    valid = off[None, :] < torch.tensor(commits)[:, None]
    _compare(
        spec=[(s, b * block, c) for b, (s, c) in enumerate(zip(starts, commits))],
        max_span_len=block,
        num_tokens=len(starts) * block,
        seed=2,
        what="verify per-request commit lens",
        loc=torch.where(valid, loc2d, torch.zeros_like(loc2d)).reshape(-1).to(DEVICE),
    )
    # Verify spilling across a page: the request's tail lives in the page the
    # allocator handed out next, not the physically adjacent one.
    rows = [_paged_slots([6, 13], 122, block), _paged_slots([1, 20], 0, block)]
    commits = [10, 4]
    _compare(
        spec=[
            (rows[0][0], 0, 6),
            (rows[0][6], 6, 4),
            (rows[1][0], block, 4),
            (rows[1][4], block + 4, 0),
        ],
        max_span_len=block,
        num_tokens=len(rows) * block,
        seed=3,
        what="verify commit spilling across a page",
        loc=torch.tensor(
            [
                s if i < c else 0
                for row, c in zip(rows, commits)
                for i, s in enumerate(row)
            ],
            dtype=torch.int64,
            device=DEVICE,
        ),
    )


def _paged_slots(pages: list[int], offset: int, count: int) -> list[int]:
    """Slots one request occupies, laid out the way the paged allocator does.

    Args:
      pages: Page ids the request holds, in order.
      offset: Page offset the request resumes at.
      count: Tokens to place.

    Returns:
      count slot indices, consecutive within a page and jumping between pages.
    """
    slots: list[int] = []
    page = 0
    while len(slots) < count:
        take = min(PAGE_SIZE - offset, count - len(slots))
        base = pages[page] * PAGE_SIZE + offset
        slots.extend(range(base, base + take))
        offset, page = 0, page + 1
    return slots


def _addressed(
    dst_starts: torch.Tensor, src_starts: torch.Tensor, span_lens: torch.Tensor
) -> tuple[list[int], list[int]]:
    """Flatten a span table back into the (slot, source) pairs it writes."""
    dsts: list[int] = []
    srcs: list[int] = []
    for d, s, n in zip(
        dst_starts.cpu().tolist(), src_starts.cpu().tolist(), span_lens.cpu().tolist()
    ):
        assert n <= PAGE_SIZE, f"span of {n} tokens cannot stay inside a page"
        dsts.extend(range(d, d + n))
        srcs.extend(range(s, s + n))
    return dsts, srcs


def test_extend_spans_reconstruct_slots() -> None:
    """get_contiguous_slot_spans' extend branch reproduces the flat slots it decomposes.

    Covers the multi-request batch a PD prefill server actually sends: each
    request resumes at its own page offset, so a decomposition derived from the
    first slot's alignment writes later requests' tokens into the first
    request's pages. Padding is included because the caller zero-pads to a
    compile bucket and those tokens must go unwritten.
    """
    print("== extend span decomposition ==")
    cases = [
        ([(([4, 1, 9, 6, 2, 11]), 0, 256)], 0),
        ([(([4, 1, 9, 6, 2, 11]), 91, 300)], 0),
        ([(([4, 1]), 127, 129)], 0),
        ([(([4]), 5, 7)], 0),
        # Three requests, none page-aligned, plus bucket padding.
        (
            [([3, 8], 0, 100), ([12, 5], 44, 150), ([17], 120, 8)],
            54,
        ),
    ]
    for reqs, pad in cases:
        slots: list[int] = []
        for pages, offset, count in reqs:
            slots.extend(_paged_slots(pages, offset, count))
        num_real = len(slots)
        # On device: the extend branch is compiled in production, and the TPU
        # backend rejects op patterns that CPU tracing never sees.
        cache_loc = torch.tensor(slots + [0] * pad, dtype=torch.int64).to(DEVICE)

        dst_starts, src_starts, span_lens, max_span_len = get_contiguous_slot_spans(
            page_size=PAGE_SIZE,
            cache_loc=cache_loc,
            cache_loc_2d=None,
            commit_lens=None,
            max_num_reqs=MAX_NUM_REQS,
        )
        assert max_span_len == PAGE_SIZE, max_span_len
        dsts, srcs = _addressed(dst_starts, src_starts, span_lens)
        print(
            f"  reqs={len(reqs)} tokens={num_real:>4} pad={pad:>3}: "
            f"{len(span_lens)} spans, {int(span_lens.sum())} written"
        )
        assert dsts == slots, f"reqs={reqs} pad={pad}: spans address the wrong slots"
        assert srcs == list(range(num_real)), (
            f"reqs={reqs} pad={pad}: spans skip or repeat sources"
        )


def test_verify_spans_reconstruct_slots() -> None:
    """get_contiguous_slot_spans' verify branch reproduces each committed prefix.

    A commit block starting near the end of a page runs off it, and the tokens
    past the boundary live in whatever page the allocator handed out next. A
    single span per request writes them into the physically adjacent page
    instead -- corrupting a neighbour, and running off the pool entirely when
    the block sits in the top page.
    """
    print("== verify span decomposition ==")
    block = 10
    # Offsets 0 and 44 stay inside a page; 120 and 127 spill across.
    reqs = [([3, 8], 0, 10), ([12, 5], 44, 3), ([17, 2], 120, 10), ([9, 6], 127, 7)]
    rows = [_paged_slots(pages, offset, block) for pages, offset, _ in reqs]
    commits = [c for _, _, c in reqs]
    cache_loc_2d = torch.tensor(rows, dtype=torch.int64).to(DEVICE)

    dst_starts, src_starts, span_lens, max_span_len = get_contiguous_slot_spans(
        page_size=PAGE_SIZE,
        cache_loc=torch.zeros(len(reqs) * block, dtype=torch.int64, device=DEVICE),
        cache_loc_2d=cache_loc_2d,
        commit_lens=torch.tensor(commits, dtype=torch.int32, device=DEVICE),
        max_num_reqs=MAX_NUM_REQS,
    )
    assert max_span_len == block, max_span_len
    dsts, srcs = _addressed(dst_starts, src_starts, span_lens)

    want_dsts: list[int] = []
    want_srcs: list[int] = []
    for i, (row, commit) in enumerate(zip(rows, commits)):
        want_dsts.extend(row[:commit])
        want_srcs.extend(range(i * block, i * block + commit))
    print(f"  reqs={len(reqs)} commits={commits}: {len(span_lens)} spans")
    assert sorted(zip(srcs, dsts)) == sorted(zip(want_srcs, want_dsts)), (
        "verify spans address the wrong slots"
    )


def _real_rope(device: str) -> YaRNScalingRotaryEmbedding:
    """Build the draft's actual rotary embedding with its cache on device.

    RotaryEmbedding.__init__ reads the exec config namespace, which fails closed
    until a ServerArgs is published -- hence the publish. get_rope builds the
    cache on torch's default device, which torch_tpu sets to TPU.

    Args:
      device: Device for cos_sin_cache, matching the tensors it will multiply.

    Returns:
      The draft's rotary embedding, cache relocated.
    """
    from sglang.srt.runtime_context import publish
    from sglang.srt.server_args import ServerArgs

    publish(ServerArgs(model_path="dummy"), role="test")
    rope = get_rope(
        HEAD_DIM,
        rotary_dim=HEAD_DIM,
        max_position=MAX_POSITION,
        base=ROPE_THETA,
        rope_scaling=ROPE_SCALING,
        is_neox_style=True,
    )
    assert isinstance(rope, YaRNScalingRotaryEmbedding), (
        f"expected yarn scaling from the draft config, got {type(rope).__name__}"
    )
    rope.cos_sin_cache = rope.cos_sin_cache.to(device)
    return rope


def test_matches_real_apply_k_rope() -> None:
    """_rope_tables plus the fused rotation equal the draft's apply_k_rope.

    Compares against the eager reference on CPU in fp32, not the kernel: the
    question is whether the angle is right, and CPU answers it to 1e-6 rather
    than bf16's 3e-2. test_matches_eager_reference pins the kernel to that same
    reference on device, so the two together cover kernel vs apply_k_rope.
    """
    print("== fused rope math vs the draft's real apply_k_rope (cpu, fp32) ==")
    rope = _real_rope("cpu")

    # Spans the yarn regimes: below original_max_position_embeddings (4096),
    # across it, and far beyond, where interpolation dominates.
    positions = torch.tensor(
        [0, 1, 7, 4095, 4096, 4097, 30000, 131071], dtype=torch.int64
    )
    num_tokens = positions.numel()
    gen = torch.Generator().manual_seed(7)
    k = torch.randn((LAYERS, num_tokens, HEADS, HEAD_DIM), generator=gen)
    v = torch.zeros_like(k)

    # apply_k_rope passes a K-shaped dummy Q so every backend's head-count check
    # passes; mirror it exactly.
    dummy_q = k[0].new_empty(k[0].shape)
    _, want_k = rope(positions, dummy_q, k[0].clone())

    # _rope_tables, verbatim. cos and sin are [num_tokens, HEAD_DIM // 2].
    cos_sin = rope.cos_sin_cache.index_select(0, positions).to(k.dtype)
    cos, sin = cos_sin.chunk(2, dim=-1)

    loc = torch.arange(num_tokens, dtype=torch.int64)
    bufs = _buffers(torch.float32, device="cpu")
    fused_rope_write_kv_ref(loc=loc, k=k, v=v, cos=cos, sin=sin, kv_bufs=bufs)

    # A slot is [kv_heads_packed, kv_packing, padded_head_dim] with K and V
    # adjacent; row 0 of the single KV head is the rotated K.
    got_k = bufs[0].flatten(0, 1)[loc][:, 0, 0, :HEAD_DIM].reshape(want_k.shape)
    max_err = (want_k - got_k).abs().max().item()
    print(f"  positions={positions.tolist()}")
    print(f"  rope={type(rope).__name__} mscale={rope.mscale}")
    print(f"  max abs err={max_err:.4g} (K max abs={want_k.abs().max().item():.4g})")
    assert max_err < 1e-6, f"fused rope diverges from apply_k_rope: {max_err:.4g}"


if __name__ == "__main__":
    test_matches_real_apply_k_rope()
    print(f"TPU chips: {torch.accelerator.device_count()}")
    # A device allocation brings the pjrt client up; .to(DEVICE) alone does not.
    torch.zeros(1, device=DEVICE)
    test_extend_spans_reconstruct_slots()
    test_verify_spans_reconstruct_slots()
    test_matches_eager_reference()
    print("\nFUSED ROPE + KV WRITE TEST PASSED")
