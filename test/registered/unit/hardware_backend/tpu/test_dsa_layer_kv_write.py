"""CPU guards for the cache-only write of a DSA layer.

A writer that fills a layer's latent and index-key caches without running the
layer must store the bytes the full layer stores. These cases pin the inputs it
builds against the inputs the full path hands its ops: core's
``forward_absorb_prepare`` for the latent, ``indexer_forward`` for the index
key. Rope runs through the TPU rope tables (built on CPU tensors).
"""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.layers.attention.dsa.dsa_indexer_kpool import IndexerKPool
from sglang.srt.layers.layer_boundary.adapters.attention import AttentionInputs
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.rotary_embedding import base as rope_base
from sglang.srt.layers.rotary_embedding import rope_variant
from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
from sglang.srt.layers.rotary_embedding.rope_variant import (
    DeepseekScalingRotaryEmbedding,
)
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA
from sglang.srt.runtime_context import get_forward, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.attention_pallas_mla_sparse import TpuPallasSparseMlaAttnBackend
from sglang_tpu.dsa_indexer_tpu import (
    index_key,
    indexer_forward,
    key_store_mode,
    kpool_indexer_forward,
    pooled_index_key_and_gate,
)
from sglang_tpu.dsa_layer_kv_write import (
    dsa_layer_kv_unsupported_reason,
    mla_cache_latents,
    write_dsa_layer_kv,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_T, _HIDDEN, _HEADS = 6, 48, 2


def _tpu_rope(rotary_dim: int, *, neox: bool) -> DeepseekScalingRotaryEmbedding:
    """DeepSeek YaRN rope, run through its TPU tables."""
    with ExitStack() as stack:
        for module in (rope_base, rope_variant):
            stack.enter_context(patch.object(module, "_is_tpu", True))
        return DeepseekScalingRotaryEmbedding(
            head_size=rotary_dim,
            rotary_dim=rotary_dim,
            max_position_embeddings=256,
            base=10000,
            is_neox_style=neox,
            scaling_factor=4.0,
            dtype=torch.bfloat16,
            device="cpu",
        )


def _plain_rope(rotary_dim: int, *, neox: bool) -> RotaryEmbedding:
    """Unscaled rope, as GLM-5.x builds it for MLA and the indexer (gptj)."""
    with ExitStack() as stack:
        for module in (rope_base, rope_variant):
            stack.enter_context(patch.object(module, "_is_tpu", True))
        return RotaryEmbedding(
            head_size=rotary_dim,
            rotary_dim=rotary_dim,
            max_position_embeddings=256,
            base=10000,
            is_neox_style=neox,
            dtype=torch.bfloat16,
        )


def _linear(in_features: int, out_features: int):
    """A ReplicatedLinear stand-in: returns ``(output, bias)``."""
    weight = torch.randn(out_features, in_features, dtype=torch.bfloat16) * 0.1
    return lambda x: (x @ weight.T, None)


def _mla_attention(*, rope_dim: int, rope) -> DeepseekV2AttentionMLA:
    """The parts of a DSA layer's MLA attention ``forward_absorb_prepare`` reads."""
    q_lora_rank, kv_lora_rank, nope_dim = 16, 32, 12
    attn = DeepseekV2AttentionMLA.__new__(DeepseekV2AttentionMLA)
    nn.Module.__init__(attn)
    attn.q_lora_rank = q_lora_rank
    attn.kv_lora_rank = kv_lora_rank
    attn.qk_rope_head_dim = rope_dim
    attn.qk_nope_head_dim = nope_dim
    attn.qk_head_dim = nope_dim + rope_dim
    attn.num_local_heads = _HEADS
    attn.dcp_size = 1
    attn.q_a_layernorm = RMSNorm(q_lora_rank)
    attn.kv_a_layernorm = RMSNorm(kv_lora_rank)
    attn.kv_a_layernorm.weight.data.uniform_(0.5, 1.5)
    # None with a nonzero rope_dim is skip_rope: the rope lane is stored unrotated.
    attn.rotary_emb = rope
    attn.q_b_proj = _linear(q_lora_rank, _HEADS * attn.qk_head_dim)
    attn._use_min_latency_q_b_gemm = False
    attn.w_kc = torch.randn(_HEADS, nope_dim, kv_lora_rank, dtype=torch.bfloat16)
    attn.use_deep_gemm_bmm = False
    attn.w_kc_qrep = attn.q_b_proj_qrep_weight = None
    attn.alt_stream = None
    attn.current_attention_backend = "pallas_tpu_mla_sparse"
    # A DSA layer that runs its own indexer between the latent norm and rope;
    # its rows do not reach the latent.
    attn.use_dsa = True
    attn.skip_topk = False
    attn.layer_id = 0
    attn.indexer = lambda **_kwargs: "rows"
    return attn


class TestMlaCacheLatents(CustomTestCase):
    def test_latents_are_what_absorb_prepare_hands_the_backend(self):
        """The cache-only latents equal the ``k`` / ``k_rope`` core's absorb
        path passes to attention, bit for bit, for each rope a TPU model uses."""
        cases = {
            "yarn_gptj": (8, lambda: _tpu_rope(8, neox=False)),
            "yarn_neox": (8, lambda: _tpu_rope(8, neox=True)),
            "plain_gptj": (8, lambda: _plain_rope(8, neox=False)),
            "skip_rope": (8, lambda: None),
            "zero_rope": (0, lambda: None),
        }
        for name, (rope_dim, make_rope) in cases.items():
            with self.subTest(name), ExitStack() as stack:
                for module in (rope_base, rope_variant):
                    stack.enter_context(patch.object(module, "_is_tpu", True))
                torch.manual_seed(0)
                attn = _mla_attention(rope_dim=rope_dim, rope=make_rope())
                width = attn.q_lora_rank + attn.kv_lora_rank + rope_dim
                qkv_a = _linear(_HIDDEN, width)
                hidden = torch.randn(_T, _HIDDEN, dtype=torch.bfloat16)
                positions = torch.tensor([0, 1, 2, 7, 8, 30])
                forward_batch = SimpleNamespace(
                    forward_mode=ForwardMode.EXTEND, attn_cp_metadata=None
                )
                attn_inputs = AttentionInputs(
                    hidden, forward_batch, lambda h, _fb: qkv_a(h)[0]
                )
                stack.enter_context(
                    get_parallel().override(
                        dcp_replicate_q_proj=False, attn_cp_size=1, dcp_enabled=False
                    )
                )
                stack.enter_context(get_forward().scoped(attn_inputs=attn_inputs))
                _, k_pe, _, k_nope = attn.forward_absorb_prepare(
                    positions, hidden, forward_batch, zero_allocator=None
                )[:4]

                got_nope, got_pe = mla_cache_latents(
                    attn, qkv_latent=qkv_a(hidden)[0], positions=positions
                )

                self.assertEqual(got_nope.dtype, k_nope.dtype)
                self.assertTrue(torch.equal(got_nope, k_nope))
                self.assertEqual(got_pe.shape, k_pe.shape)
                self.assertEqual(got_pe.dtype, k_pe.dtype)
                self.assertTrue(torch.equal(got_pe, k_pe))


def _indexer(*, rope_dim: int, neox: bool = True):
    """The parts of a DSA ``Indexer`` that ``indexer_forward`` reads."""
    head_dim, n_heads, q_lora_rank = 32, 4, 16
    return SimpleNamespace(
        n_heads=n_heads,
        head_dim=head_dim,
        wq_b=_linear(q_lora_rank, n_heads * head_dim),
        wk=_linear(_HIDDEN, head_dim),
        k_norm=nn.LayerNorm(head_dim, dtype=torch.bfloat16),
        weights_proj=_linear(_HIDDEN, n_heads),
        skip_rope=rope_dim == 0,
        rope_head_dim=rope_dim,
        rotary_emb=_plain_rope(rope_dim, neox=neox) if rope_dim else None,
        softmax_scale=head_dim**-0.5,
        bf16_head_gate=False,
    )


class TestIndexKey(CustomTestCase):
    def test_key_is_what_indexer_forward_hands_index_topk(self):
        """The cache-only index key equals the key the full indexer stores, bit
        for bit, with rope on the leading columns and for a NoPE indexer."""
        for rope_dim, neox in ((16, True), (16, False), (0, True)):
            with self.subTest(rope_dim=rope_dim, neox=neox), ExitStack() as stack:
                for module in (rope_base, rope_variant):
                    stack.enter_context(patch.object(module, "_is_tpu", True))
                torch.manual_seed(0)
                indexer = _indexer(rope_dim=rope_dim, neox=neox)
                x = torch.randn(_T, _HIDDEN, dtype=torch.bfloat16)
                q_lora = torch.randn(_T, 16, dtype=torch.bfloat16)
                positions = torch.tensor([3, 4, 5, 0, 1, 2])
                stored = {}

                def index_topk(*, key, **_kwargs):
                    stored["key"] = key
                    return None

                indexer_forward(
                    indexer,
                    backend=SimpleNamespace(index_topk=index_topk),
                    x=x,
                    q_lora=q_lora,
                    positions=positions,
                    forward_batch=None,
                    layer_id=0,
                )

                got = index_key(indexer, x=x, positions=positions)

                self.assertEqual(got.dtype, stored["key"].dtype)
                self.assertTrue(torch.equal(got, stored["key"]))


class TestPooledIndexKeyAndGate(CustomTestCase):
    def test_key_and_gate_are_what_kpool_indexer_forward_stores(self):
        """The pooled indexer's raw record inputs equal what the full pooled
        indexer hands index_topk_kpool, bit for bit."""
        torch.manual_seed(0)
        head_dim = 32
        indexer = SimpleNamespace(
            **vars(_indexer(rope_dim=0)),
            index_kpool_compress_gate=torch.randn(
                head_dim, _HIDDEN, dtype=torch.bfloat16
            ),
            index_kpool_compress_ape=torch.randn(4, head_dim),
            index_kpool=4,
        )
        # The pooled indexer projects its head weights from f32 activations.
        weights = torch.randn(indexer.n_heads, _HIDDEN)
        indexer.weights_proj = lambda x: (x @ weights.T, None)
        x = torch.randn(_T, _HIDDEN, dtype=torch.bfloat16)
        stored = {}

        def index_topk_kpool(*, key, gate, **_kwargs):
            stored.update(key=key, gate=gate)
            return None

        kpool_indexer_forward(
            indexer,
            backend=SimpleNamespace(index_topk_kpool=index_topk_kpool),
            x=x,
            q_lora=torch.randn(_T, 16, dtype=torch.bfloat16),
            forward_batch=None,
            layer_id=0,
        )

        key, gate = pooled_index_key_and_gate(indexer, x)

        self.assertTrue(torch.equal(key, stored["key"]))
        self.assertTrue(torch.equal(gate, stored["gate"]))


def _sparse_backend(*, kpool: int) -> TpuPallasSparseMlaAttnBackend:
    backend = TpuPallasSparseMlaAttnBackend.__new__(TpuPallasSparseMlaAttnBackend)
    backend._kpool = kpool
    return backend


class TestUnsupportedReason(CustomTestCase):
    def test_refuses_each_path_where_the_full_layer_stores_otherwise(self):
        """None only for core's DSA MLA layer with its own indexer on the sparse
        backend, without context parallelism; otherwise a reason naming the
        path. The context-parallel cases guard a future platform that allows
        them."""
        layer = dict(use_dsa=True, q_lora_rank=16, skip_topk=True, is_nextn=True)
        parallel = dict(dcp_enabled=False, attn_cp_size=1)

        def attention(**changes):
            attn = _mla_attention(rope_dim=0, rope=None)
            for name, value in {**layer, **changes}.items():
                setattr(attn, name, value)
            return attn

        class _OwnAbsorb(DeepseekV2AttentionMLA):
            def forward_absorb_prepare(self, *args, **kwargs):
                raise NotImplementedError

        own_absorb = _OwnAbsorb.__new__(_OwnAbsorb)
        cases = [
            (attention(), {}, None),
            (attention(use_dsa=False), {}, "not a DSA layer"),
            (attention(q_lora_rank=None), {}, "q_lora_rank"),
            (attention(is_nextn=False), {}, "reuses another layer's top-k"),
            (attention(), dict(dcp_enabled=True), "decode context parallelism"),
            (attention(), dict(attn_cp_size=2), "prefill context parallelism"),
            (SimpleNamespace(**layer), {}, "not DeepseekV2AttentionMLA"),
            (own_absorb, {}, "overrides forward_absorb_prepare"),
        ]
        for attn, parallel_change, expected in cases:
            with self.subTest(expected=expected, parallel=parallel_change):
                with get_parallel().override(**{**parallel, **parallel_change}):
                    reason = dsa_layer_kv_unsupported_reason(
                        attn=attn, backend=_sparse_backend(kpool=1)
                    )
                if expected is None:
                    self.assertIsNone(reason)
                else:
                    self.assertIn(expected, reason)
        with get_parallel().override(**parallel):
            reason = dsa_layer_kv_unsupported_reason(
                attn=attention(), backend=SimpleNamespace()
            )
        self.assertIn("not pallas_tpu_mla_sparse", reason)

    def test_each_store_refuses_the_other_indexer_before_writing(self):
        args = dict(
            forward_batch=None, layer=None, index_layer_id=0, k=None, k_rope=None
        )
        with self.assertRaisesRegex(NotImplementedError, "store_layer_kv_pooled"):
            TpuPallasSparseMlaAttnBackend.store_layer_kv(
                _sparse_backend(kpool=4), **args, index_keys=None, key_store="xla"
            )
        with self.assertRaisesRegex(NotImplementedError, "store_layer_kv"):
            TpuPallasSparseMlaAttnBackend.store_layer_kv_pooled(
                _sparse_backend(kpool=1), **args, keys=None, gates=None, ape=None
            )


class _RecordingBackend:
    def __init__(self):
        self.calls = []

    def store_layer_kv(self, **kwargs):
        self.calls.append(("plain", kwargs))

    def store_layer_kv_pooled(self, **kwargs):
        self.calls.append(("pooled", kwargs))


class TestWriteDsaLayerKv(CustomTestCase):
    def _attention(self, indexer):
        attn = _mla_attention(rope_dim=8, rope=_plain_rope(8, neox=False))
        attn.layer_id = 7
        attn.attn_mqa = SimpleNamespace(layer_id=7)
        attn.indexer = indexer
        return attn

    def test_hands_each_store_the_layers_ids_latents_and_index_inputs(self):
        """Plain DSA gets the index key and the indexer's writer choice; the
        pooled indexer gets its key, gate and ape; both get the attention's
        RadixAttention and the indexer's layer id."""
        with ExitStack() as stack:
            for module in (rope_base, rope_variant):
                stack.enter_context(patch.object(module, "_is_tpu", True))
            torch.manual_seed(0)
            positions = torch.arange(_T)
            hidden = torch.randn(_T, _HIDDEN, dtype=torch.bfloat16)
            pooled = IndexerKPool.__new__(IndexerKPool)
            nn.Module.__init__(pooled)
            for name, value in vars(_indexer(rope_dim=0)).items():
                object.__setattr__(pooled, name, value)
            object.__setattr__(
                pooled,
                "index_kpool_compress_gate",
                torch.randn(32, _HIDDEN, dtype=torch.bfloat16),
            )
            object.__setattr__(pooled, "index_kpool_compress_ape", torch.randn(4, 32))
            for indexer, kind in ((_indexer(rope_dim=16), "plain"), (pooled, "pooled")):
                with self.subTest(kind):
                    attn = self._attention(indexer)
                    qkv_latent = torch.randn(_T, 16 + 32 + 8, dtype=torch.bfloat16)
                    backend = _RecordingBackend()
                    write_dsa_layer_kv(
                        attn,
                        backend=backend,
                        forward_batch="fb",
                        qkv_latent=qkv_latent,
                        hidden_states=hidden,
                        positions=positions,
                    )
                    [(got_kind, kwargs)] = backend.calls
                    self.assertEqual(got_kind, kind)
                    self.assertEqual(kwargs["forward_batch"], "fb")
                    self.assertIs(kwargs["layer"], attn.attn_mqa)
                    self.assertEqual(kwargs["index_layer_id"], 7)
                    k, k_rope = mla_cache_latents(
                        attn, qkv_latent=qkv_latent, positions=positions
                    )
                    self.assertTrue(torch.equal(kwargs["k"], k))
                    self.assertTrue(torch.equal(kwargs["k_rope"], k_rope))
                    if kind == "plain":
                        want = index_key(indexer, x=hidden, positions=positions)
                        self.assertTrue(torch.equal(kwargs["index_keys"], want))
                        self.assertEqual(
                            kwargs["key_store"], key_store_mode(indexer, _T)
                        )
                    else:
                        keys, gates = pooled_index_key_and_gate(indexer, hidden)
                        self.assertTrue(torch.equal(kwargs["keys"], keys))
                        self.assertTrue(torch.equal(kwargs["gates"], gates))
                        self.assertIs(kwargs["ape"], indexer.index_kpool_compress_ape)


if __name__ == "__main__":
    unittest.main()
