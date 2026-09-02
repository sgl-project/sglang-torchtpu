"""On-device test: the fused RoPE + all-layer KV write kernel matches eager torch.

Requires a TPU + torch_tpu + a jax with Pallas. NOT a CI unit test. Run with:

    PYTHONPATH=/path/to/sglang-torchtpu/python \
        python3 test/manual/tpu/test_tpu_kv_fused_rope_write.py

Three orthogonal properties, one per test:

- Packing and addressing: fused_rope_write_kv against fused_rope_write_kv_ref,
  which spells the same steps out per token in eager torch. Divergence means
  the kernel drifted from what the fused KV pool expects.
- Span decomposition: get_contiguous_slot_spans' boundary math must reproduce
  the original slots exactly, including a mid-page start and the trailing empty
  span a page-aligned write leaves behind.
- Rotation angle: _rope_tables' cos/sin rows fed through that rotation must
  equal the draft's real apply_k_rope. A wrong angle does not crash -- it
  silently degrades draft quality and costs accept length.

Geometry mirrors the gpt-oss-120b DFlash draft @ TP8 (8 layers, 1 KV head,
head_dim 64, page 128) with the page count scaled down.
"""

import torch

from sglang.srt.hardware_backend.tpu.dflash_draft_kv_write import (
    get_contiguous_slot_spans,
)
from sglang.srt.hardware_backend.tpu.kv_fused_rope_write import (
    fused_rope_write_kv,
    fused_rope_write_kv_ref,
)
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.rotary_embedding.yarn import YaRNScalingRotaryEmbedding

DEVICE = "tpu:0"
LAYERS = 8
HEADS = 1
HEAD_DIM = 64
NUM_PAGES = 32
PAGE_SIZE = 128
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
    torch.tpu.synchronize()

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


def _spans_for(cache_loc: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Drive the real get_contiguous_slot_spans extend branch."""
    return get_contiguous_slot_spans(
        page_size=PAGE_SIZE, cache_loc=cache_loc, cache_loc_2d=None, commit_lens=None
    )


def test_extend_spans_reconstruct_slots() -> None:
    """get_contiguous_slot_spans' extend branch reproduces the flat slots it decomposes."""
    print("== extend span decomposition ==")
    for head_offset, num_tokens in ((0, 256), (91, 300), (127, 129), (5, 7)):
        pages = [4, 1, 9, 6, 2, 11]
        loc = []
        offset, page = head_offset, 0
        while len(loc) < num_tokens:
            take = min(PAGE_SIZE - offset, num_tokens - len(loc))
            base = pages[page] * PAGE_SIZE + offset
            loc.extend(range(base, base + take))
            offset, page = 0, page + 1
        cache_loc = torch.tensor(loc, dtype=torch.int64)

        dst_starts, src_starts, span_lens, max_span_len = _spans_for(cache_loc)
        assert max_span_len == PAGE_SIZE, max_span_len
        rebuilt = torch.cat(
            [
                torch.arange(int(d), int(d) + int(n))
                for d, n in zip(dst_starts, span_lens)
                if int(n) > 0
            ]
        )
        srcs = torch.cat(
            [
                torch.arange(int(s), int(s) + int(n))
                for s, n in zip(src_starts, span_lens)
                if int(n) > 0
            ]
        )
        print(
            f"  head={head_offset:>3} tokens={num_tokens:>4}: "
            f"{len(span_lens)} spans, {int(span_lens.sum())} written"
        )
        assert torch.equal(
            rebuilt, cache_loc
        ), f"head={head_offset} tokens={num_tokens}: spans address the wrong slots"
        assert torch.equal(
            srcs, torch.arange(num_tokens)
        ), f"head={head_offset} tokens={num_tokens}: spans skip or repeat sources"


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
    assert isinstance(
        rope, YaRNScalingRotaryEmbedding
    ), f"expected yarn scaling from the draft config, got {type(rope).__name__}"
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
    test_extend_spans_reconstruct_slots()
    print(f"TPU chips: {torch.tpu.device_count()}")
    test_matches_eager_reference()
    print("\nFUSED ROPE + KV WRITE TEST PASSED")
