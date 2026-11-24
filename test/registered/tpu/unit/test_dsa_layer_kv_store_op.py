# SPDX-License-Identifier: Apache-2.0
"""The backend's cache-only store leaves its caches as attention + indexer do.

``TpuPallasSparseMlaAttnBackend.store_layer_kv`` (plain DSA) and
``store_layer_kv_pooled`` (pooled indexer) take what ``_forward_mla`` and the
indexer op take, and run one registered op that updates the layer's caches in
place. Each case drives both routes through the backend's own methods and the
real ops on TPU tensors, from the same stale caches, and compares every cache
byte for byte. That covers the store JAX, the op's registration and in-place
update, its operand order, and the latent store form and scale; both index-key
writers run through the op. Cases run the attention through the per-query op
and through the q-block prefill op. One plain and one pooled case also run the
store op inside a fullgraph torch.compile, where its in-place update must
survive functionalization.

Needs a TPU (torch_tpu); skipped otherwise. The attention / indexer token
splits need several chips, and the pooled indexer's raw ring is off by
default; neither is covered here.

Usage:
    python test/registered/tpu/unit/test_dsa_layer_kv_store_op.py
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=120, suite="stage-a-test-tpu-unit", chips=1)

import functools
import importlib.util
import itertools
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.test_utils import CustomTestCase

_DEVICE = "tpu"
_PAGE_SIZE = 128
_NOPE, _ROPE = 512, 64
_INDEX_HEADS, _INDEX_DIM = 32, 128
_ATTN_HEADS = 8
_TOPK = 256
_KPOOL = 4
_FP8 = torch.float8_e4m3fn
_K_SCALE = 0.5


def _no_tpu() -> bool:
    return importlib.util.find_spec("torch_tpu") is None or not hasattr(torch, "tpu")


def _step(lens, *, token_bucket, is_decode, padded_rows, zero_rope, kpool):
    """One padded step: metadata, layer inputs and stale caches.

    ``lens`` is ``[(q_len, kv_len)]`` per real request; ``kv_len`` counts the
    step's tokens. Padded rows carry one token each at position 0 of request
    0's first page, a slot no real token of these cases writes, so a padded
    row that stored anything would change it.
    """
    from sglang_tpu.attention.pallas.dsa_index_topk import index_key_cache_shape
    from sglang_tpu.attention.pallas.dsa_kpool_topk import pooled_record_bytes

    gen = torch.Generator().manual_seed(len(lens) * 7 + token_bucket + kpool)
    num_active = len(lens)
    assert all(kv > q for q, kv in lens[:1]), "request 0 must have a cached prefix"
    pages_needed = [-(-kv // _PAGE_SIZE) for _, kv in lens]
    pages_per_seq = max(pages_needed) + 1
    num_pages = 1 + sum(pages_needed) + 4
    pool = 1 + torch.randperm(num_pages - 1, generator=gen)
    table = torch.zeros(num_active + padded_rows, pages_per_seq, dtype=torch.int32)
    used = 0
    for i, n in enumerate(pages_needed):
        table[i, :n] = pool[used : used + n]
        used += n
    table[num_active:] = table[0]
    q_lens = [q for q, _ in lens] + [1] * padded_rows
    kv_lens = [kv for _, kv in lens] + [1] * padded_rows
    assert sum(q_lens) <= token_bucket
    positions = torch.cat(
        [torch.arange(kv - q, kv) for q, kv in zip(q_lens, kv_lens)]
        + [torch.zeros(token_bucket - sum(q_lens), dtype=torch.int64)]
    )
    # Attention rows list each query's causal positions, -1 padded to whole lanes.
    width = _TOPK if kpool == 1 else -(-(_TOPK + kpool - 1) // 128) * 128
    topk = torch.full((token_bucket, width), -1, dtype=torch.int32)
    for row, pos in zip(topk, positions.tolist()):
        n = min(pos + 1, width)
        row[:n] = torch.arange(n)

    def normal(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, generator=gen).to(dtype).to(_DEVICE)

    def stale(*shape, dtype=torch.uint8):
        if dtype == torch.uint8:
            return torch.randint(0, 256, shape, generator=gen, dtype=torch.uint8)
        return torch.randn(*shape, generator=gen).to(dtype)

    caches = {
        "nope": stale(num_pages, _PAGE_SIZE, 4, 128),
        "rope": stale(1, 1, 4, 128)
        if zero_rope
        else stale(num_pages, _PAGE_SIZE // 4, 4, 128),
    }
    if kpool == 1:
        record = index_key_cache_shape(num_pages, _PAGE_SIZE, _INDEX_DIM)[-1]
        caches["index_k"] = stale(num_pages, _PAGE_SIZE // 4, 4, record)
    else:
        caches["raw"] = stale(
            num_pages, _PAGE_SIZE, 2 * _INDEX_DIM, dtype=torch.bfloat16
        )
        caches["index_k"] = stale(
            num_pages, _PAGE_SIZE // kpool // 4, 4, pooled_record_bytes(_INDEX_DIM)
        )
    return SimpleNamespace(
        kpool=kpool,
        zero_rope=zero_rope,
        aliased_page=int(table[0, 0]),
        metadata=SimpleNamespace(
            kv_lens=torch.tensor(kv_lens, dtype=torch.int32).to(_DEVICE),
            page_indices=table.reshape(-1).to(_DEVICE),
            cu_q_lens=torch.tensor(
                [0, *itertools.accumulate(q_lens)], dtype=torch.int32
            ).to(_DEVICE),
            cu_kv_lens=torch.zeros(len(q_lens) + 1, dtype=torch.int32).to(_DEVICE),
            distribution=torch.tensor(
                [num_active] * 3 if is_decode else [0, 0, num_active],
                dtype=torch.int32,
            ).to(_DEVICE),
            decode_query_size=1,
            segments="decode" if is_decode else "mixed",
        ),
        forward_mode=ForwardMode.DECODE if is_decode else ForwardMode.EXTEND,
        q=normal(token_bucket, _ATTN_HEADS, _NOPE),
        q_rope=normal(token_bucket, _ATTN_HEADS, _ROPE),
        k=normal(token_bucket, 1, _NOPE),
        k_rope=(
            torch.zeros(token_bucket, 1, _ROPE, dtype=torch.bfloat16).to(_DEVICE)
            if zero_rope
            else normal(token_bucket, 1, _ROPE)
        ),
        topk=topk.to(_DEVICE),
        index_query=normal(token_bucket, _INDEX_HEADS, _INDEX_DIM),
        index_keys=normal(token_bucket, _INDEX_DIM),
        gates=normal(token_bucket, _INDEX_DIM),
        ape=normal(kpool, _INDEX_DIM, dtype=torch.float32),
        head_weights=normal(token_bucket, _INDEX_HEADS, dtype=torch.float32),
        caches=caches,
    )


_ATTENTION_LAYER, _INDEX_LAYER = 3, 5


def _backend(step, *, qblock_prefill):
    """A sparse backend reduced to its ops, its metadata and a one-layer pool."""
    from sglang_tpu.attention.pallas.mla_sparse_fused import (
        QBLOCK_GATHER_CHUNK_TOKENS,
    )
    from sglang_tpu.attention_pallas_mla_sparse import (
        TpuPallasSparseMlaAttnBackend,
        _build_pallas_sparse_mla_kernel_op,
    )
    from sglang_tpu.dsa_indexer_tpu import (
        build_dsa_index_topk_op,
        build_dsa_kpool_topk_op,
    )

    caches = {name: c.to(_DEVICE) for name, c in step.caches.items()}

    def cache(name, layer_id, *, expected):
        assert layer_id == expected, (name, layer_id, expected)
        return caches[name]

    pooled = step.kpool > 1
    if not qblock_prefill:
        prefill_op = None
    elif pooled:
        prefill_op = _build_pallas_sparse_mla_kernel_op(
            step.kpool, QBLOCK_GATHER_CHUNK_TOKENS, zero_rope=step.zero_rope
        )
    else:
        prefill_op = _build_pallas_sparse_mla_kernel_op(1)
    backend = SimpleNamespace(
        kv_cache_quant_dtype=_FP8,
        forward_metadata=step.metadata,
        token_to_kv_pool=SimpleNamespace(
            get_nope_cache=functools.partial(cache, "nope", expected=_ATTENTION_LAYER),
            get_rope_cache=functools.partial(cache, "rope", expected=_ATTENTION_LAYER),
            get_index_key_cache=functools.partial(
                cache, "index_k", expected=_INDEX_LAYER
            ),
            get_kpool_raw_cache=functools.partial(cache, "raw", expected=_INDEX_LAYER),
        ),
        _op=_build_pallas_sparse_mla_kernel_op(zero_rope=step.zero_rope),
        _prefill_op=prefill_op,
        _index_topk_op=build_dsa_kpool_topk_op()
        if pooled
        else build_dsa_index_topk_op(),
        _kpool=step.kpool,
        _kpool_width=step.topk.shape[1],
        _zero_rope=step.zero_rope,
        _topk=_TOPK,
        _attn_split_world=0,
        _dcp=1,
        _dcp_gathers_prefill=False,
        raw_ring_pages_cnt=0,
    )
    for name in (
        "_mla_kernel_args",
        "_kv_cache_buffers",
        "_kv_lens",
        "_attention_split_rows",
        "_raw_ring_args",
        "_latents_for_store",
        "_dcp_gathered",
    ):
        setattr(
            backend,
            name,
            functools.partial(getattr(TpuPallasSparseMlaAttnBackend, name), backend),
        )
    # One chip: no indexer token split.
    backend._indexer_split_rows = lambda forward_batch, num_tokens: 0
    return backend, caches


def _store_compiled(backend, step, *, layer, key_store):
    """The store op of ``backend``'s route, called inside a fullgraph compile."""
    from sglang_tpu.attention_pallas_mla_sparse import _kpool_query_bound
    from sglang_tpu.dsa_layer_kv_store_op import (
        call_op,
        plain_store_op_name,
        pooled_store_op_name,
    )

    new_kv_c, new_k_pe = backend._latents_for_store(
        layer=layer, k=step.k, k_rope=step.k_rope
    )
    md = backend.forward_metadata
    pool = backend.token_to_kv_pool
    latent_caches = (
        pool.get_nope_cache(_ATTENTION_LAYER),
        pool.get_rope_cache(_ATTENTION_LAYER),
    )
    metadata = (md.kv_lens, md.page_indices, md.cu_q_lens, md.distribution)
    write_rope = not step.zero_rope
    if step.kpool > 1:
        bound = _kpool_query_bound(md, num_tokens=step.index_keys.shape[0])

        def run(*args):
            name = pooled_store_op_name(step.kpool, bound, 0, write_rope=write_rope)
            call_op(name, *args)

        args = (
            new_kv_c,
            new_k_pe,
            step.index_keys,
            step.gates,
            step.ape,
            *latent_caches,
            pool.get_kpool_raw_cache(_INDEX_LAYER),
            pool.get_index_key_cache(_INDEX_LAYER),
            *metadata,
            md.kv_lens,  # raw_page_base stand-in without a raw ring
        )
    else:

        def run(*args):
            call_op(plain_store_op_name(key_store, write_rope=write_rope), *args)

        args = (
            new_kv_c,
            new_k_pe,
            step.index_keys,
            *latent_caches,
            pool.get_index_key_cache(_INDEX_LAYER),
            *metadata,
        )
    torch.compile(run, backend="tpu", fullgraph=True, dynamic=False)(*args)
    torch.accelerator.synchronize()


@unittest.skipIf(_no_tpu(), "needs torch_tpu and a TPU")
class TestStoreLayerKv(CustomTestCase):
    def _assert_same_caches(
        self, step, *, key_store="xla", qblock_prefill=False, compiled=False
    ):
        from sglang_tpu.attention_pallas_mla_sparse import (
            TpuPallasSparseMlaAttnBackend as Backend,
        )

        layer = SimpleNamespace(
            v_head_dim=_NOPE,
            qk_head_dim=_NOPE + (0 if step.zero_rope else _ROPE),
            tp_q_head_num=_ATTN_HEADS,
            scaling=float((128 + _ROPE) ** -0.5),
            layer_id=_ATTENTION_LAYER,
            k_scale_float=_K_SCALE,
        )
        forward_batch = SimpleNamespace(spec_info=None, forward_mode=step.forward_mode)
        index_args = dict(
            forward_batch=forward_batch,
            layer_id=_INDEX_LAYER,
            query=step.index_query,
            head_weights=step.head_weights,
            weight_scale=float(_INDEX_DIM**-0.5 * _INDEX_HEADS**-0.5),
        )

        full, full_caches = _backend(step, qblock_prefill=qblock_prefill)
        if step.kpool > 1:
            Backend.index_topk_kpool(
                full,
                **index_args,
                key=step.index_keys,
                gate=step.gates,
                ape=step.ape,
                kpool=step.kpool,
            )
        else:
            Backend.index_topk(
                full, **index_args, key=step.index_keys, key_store=key_store
            )
        Backend._forward_mla(
            full,
            step.q,
            step.k,
            layer,
            forward_batch,
            q_rope=step.q_rope,
            k_rope=step.k_rope,
            topk_indices=step.topk,
        )

        only, only_caches = _backend(step, qblock_prefill=qblock_prefill)
        store_args = dict(
            forward_batch=forward_batch,
            layer=layer,
            index_layer_id=_INDEX_LAYER,
            k=step.k,
            k_rope=step.k_rope,
        )
        if compiled:
            _store_compiled(only, step, layer=layer, key_store=key_store)
        elif step.kpool > 1:
            Backend.store_layer_kv_pooled(
                only, **store_args, keys=step.index_keys, gates=step.gates, ape=step.ape
            )
        else:
            Backend.store_layer_kv(
                only, **store_args, index_keys=step.index_keys, key_store=key_store
            )

        for name, before in step.caches.items():
            want, got = full_caches[name].cpu(), only_caches[name].cpu()
            self.assertTrue(torch.equal(got, want), name)
            if step.zero_rope and name == "rope":
                self.assertTrue(torch.equal(got, before), name)
                continue
            # Positive control: the full layer wrote this cache in place.
            self.assertFalse(torch.equal(want, before), name)
        # Padded rows stored nothing: request 0's slot 0 keeps its stale bytes.
        page = step.aliased_page
        slot0 = {"nope": (page, 0), "rope": (page, 0, 0), "raw": (page, 0)}
        if step.kpool == 1:
            slot0["index_k"] = (page, 0, 0)
        if step.zero_rope:
            del slot0["rope"]
        if step.kpool == 1:
            del slot0["raw"]
        for name, index in slot0.items():
            self.assertTrue(
                torch.equal(only_caches[name].cpu()[index], step.caches[name][index]),
                f"padded row wrote {name}",
            )

    def test_plain_extend_per_query_attention(self):
        # A prefix ending mid-page whose new tokens cross into the next page, a
        # short cached prefix, a fresh prompt, and a padded row.
        step = _step(
            [(40, 150), (40, 120), (24, 24)],
            token_bucket=128,
            is_decode=False,
            padded_rows=1,
            zero_rope=False,
            kpool=1,
        )
        self._assert_same_caches(step)

    def test_plain_extend_qblock_prefill(self):
        # Large enough for the q-block prefill op (512 tokens, a multiple of 128).
        step = _step(
            [(200, 330), (200, 200), (100, 100)],
            token_bucket=512,
            is_decode=False,
            padded_rows=1,
            zero_rope=False,
            kpool=1,
        )
        self._assert_same_caches(step, qblock_prefill=True)

    def test_plain_decode_on_either_key_writer(self):
        step = _step(
            [(1, 300), (1, 37)],
            token_bucket=4,
            is_decode=True,
            padded_rows=2,
            zero_rope=False,
            kpool=1,
        )
        for key_store in ("xla", "pallas"):
            with self.subTest(key_store=key_store):
                self._assert_same_caches(step, key_store=key_store)

    def test_pooled_indexer_without_rope(self):
        # GLM-5.3-Flash's draft layer: pooled indexer (kpool 4), no rope lane.
        # Prefixes end mid-pool and the new tokens close pools and cross pages.
        for qblock_prefill, lens, bucket in (
            (False, [(40, 150), (40, 121), (24, 24)], 128),
            (True, [(200, 330), (200, 203), (100, 100)], 512),
        ):
            with self.subTest(qblock_prefill=qblock_prefill):
                step = _step(
                    lens,
                    token_bucket=bucket,
                    is_decode=False,
                    padded_rows=1,
                    zero_rope=True,
                    kpool=_KPOOL,
                )
                self._assert_same_caches(step, qblock_prefill=qblock_prefill)

    def test_compiled_store_op_updates_the_callers_caches(self):
        for kpool, zero_rope in ((1, False), (_KPOOL, True)):
            with self.subTest(kpool=kpool):
                step = _step(
                    [(40, 150), (40, 121), (24, 24)],
                    token_bucket=128,
                    is_decode=False,
                    padded_rows=1,
                    zero_rope=zero_rope,
                    kpool=kpool,
                )
                self._assert_same_caches(step, compiled=True)

    def test_pooled_indexer_decode(self):
        step = _step(
            [(1, 300), (1, 39)],
            token_bucket=4,
            is_decode=True,
            padded_rows=2,
            zero_rope=True,
            kpool=_KPOOL,
        )
        self._assert_same_caches(step)


if __name__ == "__main__":
    unittest.main()
