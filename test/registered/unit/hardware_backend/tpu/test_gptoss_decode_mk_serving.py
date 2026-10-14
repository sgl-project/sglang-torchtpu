"""CPU checks of the GPT-OSS decode megakernel serving integration (T21 Phase 2).

Covers what can run without a TPU or a full sglang install:
  * checkpoint (HF MXFP4) -> per-rank kernel weights vs the kernel's own ``pack_weights``
    (TP and EP MoE), bit-exact after dequantization;
  * sglang decode metadata (kv_lens, rectangular page tables, full->SWA mapping, rope cache,
    bucket padding) -> kernel operands, against out_cache_loc semantics;
  * the per-layer in-place KV write kernel;
  * one end-to-end decode step at tiny dims: repacked weights + per-layer (split) pools +
    the adapter glue through the megakernel in Pallas interpret mode on 8 CPU devices, vs
    the pure-JAX reference, including the KV writes.

The sglang model / torch_tpu custom ops are not instantiated here (no torch_tpu on CPU).

Run (8 virtual CPU devices, jax + torch CPU):
  XLA_FLAGS=--xla_force_host_platform_device_count=8 JAX_PLATFORMS=cpu \
    python -m unittest test.registered.unit.hardware_backend.tpu.test_gptoss_decode_mk_serving
"""

import dataclasses
import importlib.util
import os
import unittest

import numpy as np
import torch

from sglang.srt.hardware_backend.tpu.kernels.gptoss_decode_megakernel import config as C
from sglang.test.ci.ci_register import register_cpu_ci, register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.gptoss_decode_mk import weights as W

register_cpu_ci(est_time=25, suite="base-a-test-cpu")
register_tpu_ci(est_time=25, suite="stage-a-test-tpu-unit")

_HAS_JAX = importlib.util.find_spec("jax") is not None
if _HAS_JAX:
    os.environ["JAX_PLATFORMS"] = "cpu"
    if "xla_force_host_platform_device_count" not in os.environ.get("XLA_FLAGS", ""):
        os.environ["XLA_FLAGS"] = (
            os.environ.get("XLA_FLAGS", "")
            + " --xla_force_host_platform_device_count=8"
        )

    import jax  # noqa: E402
    import jax.numpy as jnp  # noqa: E402
    from jax.sharding import Mesh  # noqa: E402
    from jax.sharding import PartitionSpec as P  # noqa: E402

    from sglang.srt.hardware_backend.tpu.kernels.gptoss_decode_megakernel import (  # noqa: E402
        megakernel as MK,
    )
    from sglang.srt.hardware_backend.tpu.kernels.gptoss_decode_megakernel import (
        reference as REF,
    )
    from sglang_tpu.gptoss_decode_mk import adapter as A  # noqa: E402

LUT = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], np.float32)


def _cfg(B=4, moe_mode="tp", **kw):
    # H=224: a multiple of 32 (MXFP4 K blocks, as the real 2880) that still exercises the
    # hidden padding (Hp=256); I=384 -> 12 scale blocks over 8 ranks (2,2,2,2,1,1,1,1).
    return C.tiny(B=B, H=224, moe_mode=moe_mode, w_dtype="mxfp4", **kw)


def _dequant(blocks, scales):
    """HF MXFP4 [..., N, K/32, 16] u8 + [..., N, K/32] u8 -> [..., N, K] f32
    (low nibble = even element, as MXFP4QuantizeUtil.dequantize)."""
    b = np.asarray(blocks)
    codes = np.stack([b & 0x0F, b >> 4], -1).reshape(b.shape[:-1] + (32,))
    v = LUT[codes & 7] * np.where(codes & 8, -1.0, 1.0)
    v = v * np.exp2(np.asarray(scales, np.float32) - 127.0)[..., None]
    return v.reshape(v.shape[:-2] + (-1,)).astype(np.float32)


def _checkpoint(c, seed=0):
    """Random HF-layout GPT-OSS checkpoint tensors (torch) for c.L layers."""
    g = torch.Generator().manual_seed(seed)
    H, NQ, NKV, D, E, I = c.H, c.NQ, c.NKV, c.D, c.E, c.I

    def n(*shape, s=1.0):
        return (torch.randn(*shape, generator=g) * s).to(torch.bfloat16)

    def u8(*shape, lo=0, hi=256):
        return torch.randint(lo, hi, shape, generator=g, dtype=torch.int32).to(
            torch.uint8
        )

    ck = {}
    for l in range(c.L):
        p = f"model.layers.{l}."
        ck.update(
            {
                p + "self_attn.q_proj.weight": n(NQ * D, H, s=H**-0.5),
                p + "self_attn.q_proj.bias": n(NQ * D, s=0.1),
                p + "self_attn.k_proj.weight": n(NKV * D, H, s=H**-0.5),
                p + "self_attn.k_proj.bias": n(NKV * D, s=0.1),
                p + "self_attn.v_proj.weight": n(NKV * D, H, s=H**-0.5),
                p + "self_attn.v_proj.bias": n(NKV * D, s=0.1),
                p + "self_attn.o_proj.weight": n(H, NQ * D, s=(NQ * D) ** -0.5),
                p + "self_attn.o_proj.bias": n(H, s=0.1),
                p + "self_attn.sinks": n(NQ),
                p + "input_layernorm.weight": (
                    1 + 0.1 * torch.randn(H, generator=g)
                ).to(torch.bfloat16),
                p + "post_attention_layernorm.weight": (
                    1 + 0.1 * torch.randn(H, generator=g)
                ).to(torch.bfloat16),
                p + "mlp.router.weight": n(E, H, s=H**-0.5),
                p + "mlp.router.bias": n(E, s=0.1),
                p + "mlp.experts.gate_up_proj_blocks": u8(E, 2 * I, H // 32, 16),
                p + "mlp.experts.gate_up_proj_scales": u8(
                    E, 2 * I, H // 32, lo=119, hi=123
                ),
                p + "mlp.experts.gate_up_proj_bias": n(E, 2 * I, s=0.1),
                p + "mlp.experts.down_proj_blocks": u8(E, H, I // 32, 16),
                p + "mlp.experts.down_proj_scales": u8(E, H, I // 32, lo=119, hi=123),
                p + "mlp.experts.down_proj_bias": n(E, H, s=0.1),
            }
        )
    # non-layer tensors the builder must ignore
    ck["model.embed_tokens.weight"] = n(8, H)
    ck["model.norm.weight"] = n(H)
    return ck


def _global_layers(c, ck):
    """Checkpoint -> the reference's global model-layout LayerWeights (exact MXFP4 values)."""

    def f(t):
        return np.asarray(t.float().numpy())

    def bf(x):
        return jnp.asarray(np.asarray(x, np.float32)).astype(jnp.bfloat16)

    layers = []
    for l in range(c.L):
        p = f"model.layers.{l}."
        gk = lambda k: ck[p + k]  # noqa: E731
        gu = _dequant(
            gk("mlp.experts.gate_up_proj_blocks"), gk("mlp.experts.gate_up_proj_scales")
        )
        dn = _dequant(
            gk("mlp.experts.down_proj_blocks"), gk("mlp.experts.down_proj_scales")
        )
        gub = f(gk("mlp.experts.gate_up_proj_bias"))
        layers.append(
            REF.LayerWeights(
                ln1=bf(f(gk("input_layernorm.weight"))),
                wqkv=bf(
                    np.concatenate(
                        [f(gk(f"self_attn.{x}_proj.weight")).T for x in "qkv"], -1
                    )
                ),
                bqkv=bf(
                    np.concatenate([f(gk(f"self_attn.{x}_proj.bias")) for x in "qkv"])
                ),
                sinks=bf(f(gk("self_attn.sinks"))),
                wo=bf(f(gk("self_attn.o_proj.weight")).T),
                bo=bf(f(gk("self_attn.o_proj.bias"))),
                ln2=bf(f(gk("post_attention_layernorm.weight"))),
                wr=bf(f(gk("mlp.router.weight")).T),
                br=bf(f(gk("mlp.router.bias"))),
                # interleaved rows (gate_i = 2i, up_i = 2i+1) -> [E, H, gate I | up I]
                w13=bf(
                    np.concatenate(
                        [
                            gu[:, 0::2].transpose(0, 2, 1),
                            gu[:, 1::2].transpose(0, 2, 1),
                        ],
                        -1,
                    )
                ),
                b13=bf(np.concatenate([gub[:, 0::2], gub[:, 1::2]], -1)),
                w2=bf(dn.transpose(0, 2, 1)),
                b2=bf(f(gk("mlp.experts.down_proj_bias"))),
            )
        )
    return layers


def _build_rank(c, ck, r, seed=1):
    b = W.MkWeightBuilder(c, r)
    names = list(ck)
    np.random.default_rng(seed + r).shuffle(names)  # stream order must not matter
    used = sum(b.consume(nm, ck[nm]) for nm in names)
    assert used == 19 * c.L, used
    return b.finalize()


def _to_jax(t):
    if t.dtype == torch.bfloat16:
        return jnp.asarray(t.float().numpy()).astype(jnp.bfloat16)
    if t.dtype == torch.float8_e4m3fn:  # bit-exact through the raw bytes
        return jax.lax.bitcast_convert_type(
            jnp.asarray(t.view(torch.uint8).numpy()), jnp.float8_e4m3fn
        )
    return jnp.asarray(t.numpy())


def _unpack(words, scales):
    words = jnp.asarray(words)
    if words.dtype == jnp.int32:
        words = jax.lax.bitcast_convert_type(words, jnp.uint32)
    return np.asarray(MK.mxfp4_unpack_xla(words, jnp.asarray(scales)))


class TestWeightRepack(CustomTestCase):
    def test_e8m0_to_bf16_exact(self):
        e = torch.arange(0, 256, dtype=torch.int32).to(torch.uint8)
        got = W.e8m0_to_bf16(e).float().numpy().astype(np.float64)
        want = np.exp2(np.clip(np.arange(256), 0, 254) - 127.0)
        np.testing.assert_array_equal(got, want)

    def _check_mode(self, moe_mode):
        c = _cfg(moe_mode=moe_mode)
        ck = _checkpoint(c)
        ref = MK.pack_weights(c, _global_layers(c, ck))
        for r in range(c.R):
            mine = _build_rank(c, ck, r)
            for k in ("wqkv", "wo", "wr", "vec", "sinks", "b13", "b2"):
                np.testing.assert_array_equal(
                    np.asarray(_to_jax(mine[k]).astype(jnp.float32)),
                    np.asarray(ref[k][r].astype(jnp.float32)),
                    err_msg=f"{moe_mode} r{r} {k}",
                )
            for q, s in (("w13q", "w13s"), ("w2q", "w2s")):
                np.testing.assert_array_equal(
                    _unpack(_to_jax(mine[q]), _to_jax(mine[s])),
                    _unpack(ref[q][r], ref[s][r]),
                    err_msg=f"{moe_mode} r{r} {q}",
                )
            for k, (shape, _) in W.per_rank_specs(c).items():
                self.assertEqual(tuple(mine[k].shape), shape)
                self.assertEqual(
                    tuple(mine[k].shape), tuple(MK.packed_specs(c)[k].shape[1:])
                )

    @unittest.skipUnless(_HAS_JAX, "requires JAX")
    def test_tp_matches_pack_weights(self):
        self._check_mode("tp")

    @unittest.skipUnless(_HAS_JAX, "requires JAX")
    def test_ep_matches_pack_weights(self):
        self._check_mode("ep")

    @unittest.skipUnless(_HAS_JAX, "requires JAX")
    def test_fp8_matches_pack_weights(self):
        # MXFP4 stream -> fp8 experts == the kernel's own fp8 pack_weights, bit for bit
        for moe_mode in ("tp", "ep"):
            c = _cfg(moe_mode=moe_mode)
            ck = _checkpoint(c, seed=6)
            c8 = dataclasses.replace(c, w_dtype="fp8")
            ref = MK.pack_weights(c8, _global_layers(c, ck))
            for r in (0, 3, 7):
                mine, bad = W.mxfp4_to_fp8(c, _build_rank(c, ck, r))
                self.assertEqual(bad, 0)
                self.assertEqual(tuple(mine), MK.w_names(c8))
                for k in MK.w_names(c8):
                    a = (
                        np.asarray(_to_jax(mine[k]).astype(jnp.float32))
                        if mine[k].dtype != torch.float8_e4m3fn
                        else mine[k].float().numpy()
                    )
                    np.testing.assert_array_equal(
                        a,
                        np.asarray(ref[k][r].astype(jnp.float32)),
                        err_msg=f"{moe_mode} r{r} {k}",
                    )
                    self.assertEqual(
                        tuple(mine[k].shape), tuple(MK.packed_specs(c8)[k].shape[1:])
                    )

    def test_fp8_inexact_is_counted(self):
        w = torch.tensor([[2.0**-17], [1.0]])  # below e4m3's subnormal range: inexact
        _, _, bad = W.fp8_pack_kn(w)
        self.assertEqual(bad, 1)

    @unittest.skipUnless(_HAS_JAX, "requires JAX")
    def test_quantizer_matches_reference(self):
        rng = np.random.default_rng(2)
        w = (
            rng.normal(size=(3, 96, 40))
            * np.exp2(rng.integers(-6, 3, size=(3, 96, 40)))
        ).astype(np.float32)
        w[0, :32, 0] = 0.0  # an all-zero block
        words, scales = W.quantize_mxfp4_kn(torch.from_numpy(w))
        code, scale = REF.mxfp4_quantize(jnp.asarray(w))
        np.testing.assert_array_equal(
            _unpack(_to_jax(words), _to_jax(scales)), REF.mxfp4_dequantize(code, scale)
        )

    @unittest.skipUnless(_HAS_JAX, "requires JAX")
    def test_bf16_checkpoint_packs_identically(self):
        # a bf16 (HF dequantized) checkpoint of the same weights -> the same kernel weights
        for moe_mode in ("tp", "ep"):
            c = _cfg(moe_mode=moe_mode)
            ck = _checkpoint(c, seed=4)
            ck16 = {}
            for k, v in ck.items():
                if k.endswith("gate_up_proj_blocks"):
                    p = k[: -len("_blocks")]
                    gu = _dequant(v, ck[p + "_scales"])  # [E, 2I, H] interleaved rows
                    ck16[p] = torch.from_numpy(gu.transpose(0, 2, 1).copy()).to(
                        torch.bfloat16
                    )
                elif k.endswith("down_proj_blocks"):
                    p = k[: -len("_blocks")]
                    dn = _dequant(v, ck[p + "_scales"])  # [E, H, I]
                    ck16[p] = torch.from_numpy(dn.transpose(0, 2, 1).copy()).to(
                        torch.bfloat16
                    )
                elif not k.endswith("_scales"):
                    ck16[k] = v
            for r in (0, 5):
                a, b = _build_rank(c, ck, r), W.MkWeightBuilder(c, r)
                for k, v in ck16.items():
                    b.consume(k, v)
                b = b.finalize()
                for k in ("wqkv", "wo", "wr", "vec", "sinks", "b13", "b2"):
                    self.assertTrue(torch.equal(a[k], b[k]), f"{moe_mode} r{r} {k}")
                for q, s in (("w13q", "w13s"), ("w2q", "w2s")):
                    np.testing.assert_array_equal(
                        _unpack(_to_jax(a[q]), _to_jax(a[s])),
                        _unpack(_to_jax(b[q]), _to_jax(b[s])),
                        err_msg=f"{moe_mode} r{r} {q}",
                    )

    def test_missing_tensor_is_reported(self):
        c = _cfg()
        ck = _checkpoint(c)
        ck.pop("model.layers.1.mlp.experts.down_proj_scales")
        b = W.MkWeightBuilder(c, 0)
        for k, v in ck.items():
            b.consume(k, v)
        self.assertEqual(b.missing(), [(1, "dn_scales")])
        with self.assertRaises(ValueError):
            b.finalize()

    def test_full_size_budget(self):
        # per-rank HBM of the megakernel weights at the real GPT-OSS-120B geometry
        for mode in ("tp", "ep"):
            for wd in ("mxfp4", "fp8"):
                c = C.full(1, w_dtype=wd, moe_mode=mode)
                gib = W.per_rank_bytes(c) / 2**30
                print(
                    f"GPT-OSS-120B megakernel weights per rank ({mode} MoE, {wd}): {gib:.2f} GiB"
                )
                self.assertLess(gib, 17.0)


def _alloc(rng, n_pages_total, used):
    free = [p for p in rng.permutation(np.arange(1, n_pages_total)) if p not in used]
    return free


@unittest.skipUnless(_HAS_JAX, "requires JAX")
class TestDecodeMapping(CustomTestCase):
    """sglang decode metadata -> kernel operands, against out_cache_loc semantics."""

    def test_page_tables_positions_and_write_slots(self):
        page, pps, nb_full, nb_swa = 64, 8, 64, 40
        rng = np.random.default_rng(0)
        seq_lens = [1, 64, 65, 200, 129, 3]  # includes page boundaries and window edges
        n_req_slots = 10
        req_to_token = np.zeros((n_req_slots, pps * page), np.int64)
        full_to_swa = np.zeros(nb_full * page, np.int64)
        full_free = list(rng.permutation(np.arange(1, nb_full)))
        swa_free = list(rng.permutation(np.arange(1, nb_swa)))
        req_pool = rng.choice(
            np.arange(1, n_req_slots), size=len(seq_lens), replace=False
        )
        for rq, L in zip(req_pool, seq_lens):
            for pg_i in range((L + page - 1) // page):
                fp = full_free.pop()
                req_to_token[rq, pg_i * page : (pg_i + 1) * page] = (
                    fp * page + np.arange(page)
                )
                # SWA keeps only pages that still hold window tokens (the rest were freed)
                if (pg_i + 1) * page > L - 1 - 127:
                    sp = swa_free.pop()
                    full_to_swa[fp * page : (fp + 1) * page] = sp * page + np.arange(
                        page
                    )
        B = 8  # bucket; 2 padding rows (seq_lens 0, req_pool_indices 0)
        seq = np.array(seq_lens + [0, 0], np.int32)
        rpi = np.array(list(req_pool) + [0, 0], np.int64)
        # exactly what TpuPallasRpaV3AttnBackend._build_metadata_host builds
        cols = np.arange(0, pps * page, page)
        page_indices = (req_to_token[rpi[:, None], cols[None, :]] // page).astype(
            np.int32
        )
        swa_page_indices = (
            full_to_swa[page_indices.astype(np.int64) * page] // page
        ).astype(np.int32)
        cos_sin = jnp.asarray(rng.normal(size=(pps * page, 64)), jnp.bfloat16)
        x = jnp.zeros((B, 224), jnp.bfloat16)

        c = A.trace_config(
            _cfg(B=1),
            B=B,
            pps=pps,
            kv_dtype=jnp.float8_e4m3fn,
            nb_swa=nb_swa,
            nb_full=nb_full,
            page=page,
        )
        s = jax.jit(lambda *a: A.step_inputs(c, *a))(
            jnp.asarray(seq),
            jnp.asarray(page_indices.reshape(-1)),
            jnp.asarray(swa_page_indices.reshape(-1)),
            cos_sin,
            x,
        )
        s = {k: np.asarray(v) for k, v in s.items()}

        for b in range(B):
            live = seq[b] > 0
            pos = seq[b] - 1 if live else 0
            self.assertEqual(s["pos"][b], pos)
            if not live:
                self.assertEqual(
                    (s["newpg"][b], s["newpg"][B + b], s["off"][b]), (0, 0, 0)
                )
                continue
            loc = req_to_token[rpi[b], pos]  # = out_cache_loc of this decode step
            self.assertEqual(
                s["newpg"][B + b] * page + s["off"][b], loc, f"full slot b{b}"
            )
            self.assertEqual(
                s["newpg"][b] * page + s["off"][b], full_to_swa[loc], f"swa slot b{b}"
            )
            # sliding layers read exactly the SWA pages holding window tokens [pos-127, pos)
            lo = max(pos - 127, 0)
            want = (
                [
                    full_to_swa[req_to_token[rpi[b], p * page]] // page
                    for p in range(lo // page, (max(pos - 1, 0)) // page + 1)
                ]
                if pos > 0
                else []
            )
            got = list(s["ptw"][b * c.NWP : b * c.NWP + len(want)])
            self.assertEqual(got, want, f"window pages b{b}")
        np.testing.assert_array_equal(s["ptf"], page_indices.reshape(-1))
        # rope: kernel tables are the stock cache rows at pos, neox halves, 2 heads / 128 lanes
        cs = np.asarray(cos_sin.astype(jnp.float32))[s["pos"]]
        np.testing.assert_array_equal(s["cos"][:B, :32], cs[:, :32])
        np.testing.assert_array_equal(s["cos"][:B, 96:128], cs[:, :32])
        np.testing.assert_array_equal(s["sin"][:B, :32], -cs[:, 32:])
        np.testing.assert_array_equal(s["sin"][:B, 32:64], cs[:, 32:])
        self.assertTrue(np.all(s["cos"][B:] == 0))

    def test_kv_write_in_place(self):
        nb, page, pk = 6, 64, 4
        rng = np.random.default_rng(1)
        pool = jnp.asarray(rng.normal(size=(nb, page, 1, pk, 128)), jnp.float8_e4m3fn)
        kvn = jnp.asarray(rng.normal(size=(16, 1, pk, 128)), jnp.float8_e4m3fn)
        pg = jnp.asarray([3, 1, 5, 0], jnp.int32)
        off = jnp.asarray([0, 63, 17, 0], jnp.int32)
        out = jax.jit(lambda *a: A.kv_write(*a, interpret=True))(pg, off, kvn, pool)
        want = np.asarray(pool.astype(jnp.float32)).copy()
        for b in range(4):
            want[int(pg[b]), int(off[b])] = np.asarray(kvn[b].astype(jnp.float32))
        np.testing.assert_array_equal(np.asarray(out.astype(jnp.float32)), want)

    def test_kv_write_layers_matches_per_layer(self):
        L, B, page, pk = 4, 3, 64, 4
        rng = np.random.default_rng(2)
        c = A.trace_config(
            _cfg(B=1, L=L),
            B=B,
            pps=4,
            kv_dtype=jnp.float8_e4m3fn,
            nb_swa=5,
            nb_full=7,
            page=page,
        )
        pools = [
            jnp.asarray(
                rng.normal(size=((5 if i < L // 2 else 7), page, 1, pk, 128)),
                jnp.float8_e4m3fn,
            )
            for i in range(L)
        ]
        kvn = jnp.asarray(rng.normal(size=(L, 16, 1, pk, 128)), jnp.float8_e4m3fn)
        newpg = jnp.asarray([1, 4, 0, 6, 2, 0], jnp.int32)  # SWA pages | full pages
        off = jnp.asarray([5, 63, 0], jnp.int32)
        got = jax.jit(
            lambda n, o, k, *p: A.kv_write_layers(n, o, k, *p, interpret=True)
        )(newpg, off, kvn, *pools)
        want = A.kv_write_all(c, newpg, off, kvn, pools, interpret=True)
        for g, w in zip(got, want):
            np.testing.assert_array_equal(
                np.asarray(g.astype(jnp.float32)), np.asarray(w.astype(jnp.float32))
            )
        # an unsharded side-effecting custom call breaks torch_tpu's SPMD compile
        for f, args in (
            (A.kv_write_layers, (newpg, off, kvn, *pools)),
            (A.kv_write, (newpg[:B], off, kvn[0], pools[0])),
        ):
            hlo = jax.jit(f).trace(*args).lower(lowering_platforms=("tpu",)).as_text()
            self.assertIn("tpu_custom_call", hlo)
            self.assertNotIn("has_side_effect = true", hlo)

    def test_fp8_step_lowering(self):
        c = A.trace_config(
            dataclasses.replace(_cfg(B=2), w_dtype="fp8"),
            B=2,
            pps=4,
            kv_dtype=jnp.float8_e4m3fn,
            nb_swa=24,
            nb_full=24,
            page=64,
        )
        mesh = Mesh(np.array(jax.devices("cpu")[:8]), ("tp",))
        sds = jax.ShapeDtypeStruct
        tmap = {
            torch.bfloat16: jnp.bfloat16,
            torch.float32: jnp.float32,
            torch.float8_e4m3fn: jnp.float8_e4m3fn,
        }
        ws = [sds(sh, tmap[dt]) for sh, dt in W.per_rank_specs(c).values()]
        pools = [sds((24, 64, 1, 4, 128), jnp.float8_e4m3fn) for _ in range(c.L)]
        args = (
            [
                sds((2, c.H), jnp.bfloat16),
                sds((2,), jnp.int32),
                sds((8,), jnp.int32),
                sds((8,), jnp.int32),
                sds((256, 64), jnp.bfloat16),
                sds((1,), jnp.int32),
            ]
            + ws
            + pools
        )

        def per_rank(x, kl, pi, spi, cs, rank, *rest):
            return A.per_rank_step(
                c,
                rank,
                x,
                kl,
                pi,
                spi,
                cs,
                list(rest[: len(ws)]),
                list(rest[len(ws) :]),
            )

        f = jax.shard_map(
            per_rank,
            mesh=mesh,
            in_specs=(P(),) * len(args),
            out_specs=(P(),) * 4,
            check_vma=False,
        )
        with _tpu7x_info():
            txt = jax.jit(f).trace(*args).lower(lowering_platforms=("tpu",)).as_text()
        self.assertEqual(txt.count("stablehlo.bitcast_convert"), 0)
        self.assertEqual(txt.count("tpu_custom_call"), 1)
        self.assertNotIn("has_side_effect = true", txt)

    def test_uint32_weights_and_noop_step_lowering(self):
        # serving stores code words as uint32: the step must not bitcast (copy) them; the
        # int32 fallback shows the 2 weight-sized bitcast_converts seen in the j-b120d683 run
        c = A.trace_config(
            _cfg(B=2),
            B=2,
            pps=4,
            kv_dtype=jnp.float8_e4m3fn,
            nb_swa=24,
            nb_full=24,
            page=64,
        )
        mesh = Mesh(np.array(jax.devices("cpu")[:8]), ("tp",))
        sds = jax.ShapeDtypeStruct
        specs = W.per_rank_specs(c)
        tmap = {torch.bfloat16: jnp.bfloat16, torch.float32: jnp.float32}
        pools = [sds((24, 64, 1, 4, 128), jnp.float8_e4m3fn) for _ in range(c.L)]
        for wdt, want_bitcasts in ((jnp.uint32, 0), (jnp.int32, 2)):
            ws = [sds(specs[k][0], tmap.get(specs[k][1], wdt)) for k in W.W_NAMES]
            args = (
                [
                    sds((2, c.H), jnp.bfloat16),
                    sds((2,), jnp.int32),
                    sds((8,), jnp.int32),
                    sds((8,), jnp.int32),
                    sds((256, 64), jnp.bfloat16),
                    sds((1,), jnp.int32),
                ]
                + ws
                + pools
            )
            for fn in (A.per_rank_step, A.per_rank_noop):

                def per_rank(x, kl, pi, spi, cs, rank, *rest, fn=fn):
                    return fn(
                        c,
                        rank,
                        x,
                        kl,
                        pi,
                        spi,
                        cs,
                        list(rest[: len(ws)]),
                        list(rest[len(ws) :]),
                    )

                f = jax.shard_map(
                    per_rank,
                    mesh=mesh,
                    in_specs=(P(),) * len(args),
                    out_specs=(P(),) * 4,
                    check_vma=False,
                )
                with _tpu7x_info():
                    txt = (
                        jax.jit(f)
                        .trace(*args)
                        .lower(lowering_platforms=("tpu",))
                        .as_text()
                    )
                self.assertEqual(txt.count("stablehlo.bitcast_convert"), want_bitcasts)
                self.assertEqual(txt.count("tpu_custom_call"), 1)
                self.assertNotIn("has_side_effect = true", txt)

    def test_noop_step_interpret(self):
        c = A.trace_config(
            _cfg(B=2),
            B=2,
            pps=4,
            kv_dtype=jnp.float8_e4m3fn,
            nb_swa=24,
            nb_full=24,
            page=64,
        )
        specs = W.per_rank_specs(c)
        tmap = {torch.bfloat16: jnp.bfloat16, torch.float32: jnp.float32}
        ws = [
            jnp.zeros(specs[k][0], tmap.get(specs[k][1], jnp.uint32)) for k in W.W_NAMES
        ]
        pools = [jnp.zeros((24, 64, 1, 4, 128), jnp.float8_e4m3fn) for _ in range(c.L)]
        x = jnp.ones((2, c.H), jnp.bfloat16)
        out, kvn, newpg, off = jax.jit(
            lambda *a: A.per_rank_noop(
                c,
                a[0],
                a[1],
                a[2],
                a[3],
                a[4],
                a[5],
                list(a[6 : 6 + len(ws)]),
                list(a[6 + len(ws) :]),
                interpret=True,
            )
        )(
            jnp.zeros((1,), jnp.int32),
            x,
            jnp.asarray([3, 0], jnp.int32),
            jnp.ones((8,), jnp.int32),
            jnp.ones((8,), jnp.int32),
            jnp.zeros((256, 64), jnp.bfloat16),
            *ws,
            *pools,
        )
        np.testing.assert_array_equal(np.asarray(out.astype(jnp.float32)), 1.0)
        self.assertEqual(kvn.shape, (c.L, c.BR, 1, 4, 128))
        self.assertEqual(list(np.asarray(off)), [2, 0])


class _tpu7x_info:
    """Pallas TPU lowering needs chip info; pretend TPU7x (as tpu_info_for_cpu_export)."""

    def __enter__(self):
        from jax._src.pallas.mosaic import tpu_info
        from jax.experimental.pallas import tpu as pltpu

        self.saved = (
            pltpu.get_tpu_info,
            tpu_info.get_tpu_info,
            getattr(tpu_info, "get_device_kind", None),
            getattr(tpu_info, "is_tpu_device", None),
        )
        info = tpu_info.get_tpu_info_for_chip(
            tpu_info.chip_version_from_device_kind("TPU7x"), 1
        )
        pltpu.get_tpu_info = tpu_info.get_tpu_info = lambda: info
        tpu_info.get_device_kind = lambda: "TPU7x"
        tpu_info.is_tpu_device = lambda: True

    def __exit__(self, *exc):
        from jax._src.pallas.mosaic import tpu_info
        from jax.experimental.pallas import tpu as pltpu

        pltpu.get_tpu_info, tpu_info.get_tpu_info, dk, it = self.saved
        if dk is not None:
            tpu_info.get_device_kind = dk
        if it is not None:
            tpu_info.is_tpu_device = it


def _rel(a, b):
    a = np.asarray(jnp.asarray(a).astype(jnp.float32))
    b = np.asarray(jnp.asarray(b).astype(jnp.float32))
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-6))


@unittest.skipUnless(_HAS_JAX, "requires JAX")
class TestEndToEndTinyStep(CustomTestCase):
    """Repacked weights + split pools + adapter glue through the megakernel (interpret)."""

    def _run(self, moe_mode, kv_dtype, w_dtype="mxfp4"):
        B = 4
        c0 = _cfg(B=B, moe_mode=moe_mode, kv_dtype=kv_dtype)
        ck = _checkpoint(c0, seed=3)
        layers = _global_layers(c0, ck)
        ranks = [_build_rank(c0, ck, r) for r in range(c0.R)]
        if (
            w_dtype == "fp8"
        ):  # the serving default: MXFP4 converted to e4m3 x column scale
            conv = [W.mxfp4_to_fp8(c0, h) for h in ranks]
            self.assertEqual(sum(b for _, b in conv), 0)
            ranks = [h for h, _ in conv]
            c0 = dataclasses.replace(c0, w_dtype="fp8")
        names = MK.w_names(c0)
        Wst = [jnp.stack([_to_jax(ranks[r][k]) for r in range(c0.R)]) for k in names]

        # the reference step's KV history, scattered into the paged pools
        rng = np.random.default_rng(5)
        pos_live = np.array([130, 63, 0, 200], np.int32)
        kv_lens = pos_live + 1
        kv_lens[3] = 0  # bucket padding row
        pos_ref = np.maximum(kv_lens - 1, 0)
        T = c0.max_pages * c0.page
        kvdt = REF.kv_jdtype(c0)
        keys = jax.random.split(jax.random.PRNGKey(7), 3)
        k_hist = (jax.random.normal(keys[0], (c0.L, B, T, c0.NKV, c0.D)) * 2.0).astype(
            kvdt
        )
        v_hist = jax.random.normal(keys[1], (c0.L, B, T, c0.NKV, c0.D)).astype(kvdt)
        pt_swa = (1 + rng.permutation(c0.nb_swa - 1)[: B * c0.max_pages]).reshape(
            B, c0.max_pages
        )
        pt_full = (1 + rng.permutation(c0.nb_full - 1)[: B * c0.max_pages]).reshape(
            B, c0.max_pages
        )
        garbage = [
            rng.normal(size=sp.shape).astype(np.float32) for sp in MK.pool_specs(c0)
        ]
        swa, full = MK.build_pools(
            c0, k_hist, v_hist, pos_ref, pt_swa, pt_full, base=garbage
        )
        pools = [swa[:, p] for p in range(c0.L // 2)] + [
            full[:, p] for p in range(c0.L // 2)
        ]

        x = jax.random.normal(keys[2], (B, c0.H), jnp.float32).astype(jnp.bfloat16)
        cos_t, sin_t = REF.yarn_tables(c0, jnp.arange(T))
        cos_sin = jnp.concatenate(
            [cos_t[:, : c0.D // 2], sin_t[:, : c0.D // 2]], -1
        )  # stock layout

        c = A.trace_config(
            c0,
            B=B,
            pps=c0.max_pages,
            kv_dtype=kvdt,
            nb_swa=c0.nb_swa,
            nb_full=c0.nb_full,
            page=c0.page,
        )
        mesh = Mesh(np.array(jax.devices("cpu")[: c.R]), ("tp",))
        nw, npl = len(names), len(pools)

        def body(rank, x, kvl, pi, spi, cs, *rest):
            w = [t[0] for t in rest[:nw]]
            pl_ = [t[0] for t in rest[nw:]]
            out, kvn, newpg, off = A.per_rank_step(
                c, rank[0], x, kvl, pi, spi, cs, w, pl_, interpret=True
            )
            if moe_mode == "tp":  # serving default: all layers in one call
                new_pools = A.kv_write_layers(newpg, off, kvn, *pl_, interpret=True)
            else:
                new_pools = A.kv_write_all(c, newpg, off, kvn, pl_, interpret=True)
            return (out, *[p[None] for p in new_pools])

        f = jax.jit(
            jax.shard_map(
                body,
                mesh=mesh,
                in_specs=(P("tp"), P(), P(), P(), P(), P()) + (P("tp"),) * (nw + npl),
                out_specs=(P(),) + (P("tp"),) * npl,
                check_vma=False,
            )
        )
        res = f(
            jnp.arange(c.R, dtype=jnp.int32),
            x,
            jnp.asarray(kv_lens),
            jnp.asarray(pt_full.reshape(-1), jnp.int32),
            jnp.asarray(pt_swa.reshape(-1), jnp.int32),
            cos_sin,
            *Wst,
            *pools,
        )
        out, new_pools = res[0], res[1:]

        ref, kn_ref, vn_ref = REF.step(
            c0, x, jnp.asarray(pos_ref), k_hist, v_hist, layers
        )
        rel = _rel(out[:3], ref[:3])
        print(
            f"serving step {moe_mode}/{kv_dtype}/{c0.w_dtype}: megakernel(repacked, split pools) vs "
            f"reference rel {rel:.2e}"
        )
        self.assertLess(rel, 1e-2)

        # KV writes: exactly the live rows' new-token slots changed, holding the reference K/V
        # (up to the same rounding drift as the hidden state; fp8/bf16 roundings may flip)
        for i, (old, new) in enumerate(zip(pools, new_pools)):
            layer = A.pool_layer_order(c.L)[i]
            pt = pt_swa if c.sliding[layer] else pt_full
            old = np.asarray(old.astype(jnp.float32)).copy()
            new = np.asarray(new.astype(jnp.float32))
            got, want = [], []
            for b in range(3):
                p = int(pos_ref[b])
                pg, o = pt[b, p // c.page], p % c.page
                for r in range(c.R):
                    got += [new[r, pg, o, 0, 0, : c.D], new[r, pg, o, 0, 1, : c.D]]
                    want += [
                        np.asarray(kn_ref[layer, b, r].astype(jnp.float32)),
                        np.asarray(vn_ref[layer, b, r].astype(jnp.float32)),
                    ]
                old[:, pg, o] = new[:, pg, o]
            kv_rel = _rel(np.stack(got), np.stack(want))
            self.assertLess(kv_rel, 2e-2, f"pool {i} (layer {layer})")
            old[:, 0, 0] = new[:, 0, 0]  # the padding row writes the reserved page 0
            np.testing.assert_array_equal(
                new, old, err_msg=f"pool {i} touched elsewhere"
            )

    def test_tp_moe_fp8_kv(self):
        self._run("tp", "fp8", w_dtype="fp8")

    def test_ep_moe_bf16_kv(self):
        self._run("ep", "bf16")


class TestRuntimeGating(CustomTestCase):
    """The torch-side choices that do not need torch_tpu."""

    def setUp(self):
        from types import SimpleNamespace as NS

        from sglang_tpu.gptoss_decode_mk import runtime as RT

        self.RT, self.NS = RT, NS
        self.hf = NS(
            num_hidden_layers=36,
            num_key_value_heads=8,
            hidden_size=2880,
            num_attention_heads=64,
            head_dim=64,
            layer_types=["sliding_attention", "full_attention"] * 18,
        )
        self.par = NS(
            pp_size=1, attn_dp_size=1, attn_cp_size=1, attn_dcp_size=1, attn_tp_size=8
        )

    def test_static_disable_reasons(self):
        RT, NS, hf, par = self.RT, self.NS, self.hf, self.par
        self.assertIsNone(RT.static_disable_reason(hf, par, None))
        self.assertIn("speculative", RT.static_disable_reason(hf, par, "EAGLE3"))
        self.assertIn(
            "pp_size",
            RT.static_disable_reason(hf, NS(**{**vars(par), "pp_size": 2}), None),
        )
        self.assertIn(
            "attn_tp_size",
            RT.static_disable_reason(hf, NS(**{**vars(par), "attn_tp_size": 4}), None),
        )
        self.assertIn(
            "dp/cp",
            RT.static_disable_reason(hf, NS(**{**vars(par), "attn_dp_size": 2}), None),
        )
        bad = NS(**{**vars(hf), "layer_types": ["full_attention"] * 36})
        self.assertIn("layer_types", RT.static_disable_reason(bad, par, None))

    def test_applies_only_to_plain_decode_buckets(self):
        RT, NS = self.RT, self.NS

        def fb(decode=True, spec=None, bs=8):
            return NS(
                forward_mode=NS(is_decode=lambda: decode),
                spec_info=spec,
                seq_lens=torch.zeros(bs),
            )

        ids = torch.zeros(8, dtype=torch.int64)
        A_ = RT.GptOssDecodeMK.applies
        self.assertTrue(A_(fb(), ids, None, None, []))
        self.assertFalse(A_(fb(decode=False), ids, None, None, []))  # extend/mixed/idle
        self.assertFalse(A_(fb(spec=object()), ids, None, None, []))  # spec decode
        self.assertFalse(A_(fb(), ids, torch.zeros(8, 4), None, []))  # input_embeds
        self.assertFalse(A_(fb(), ids, None, None, [3]))  # aux hidden capture
        big = torch.zeros(128, dtype=torch.int64)
        self.assertFalse(
            A_(fb(bs=128), big, None, None, [])
        )  # above the compiled buckets
        self.assertFalse(A_(fb(bs=4), ids, None, None, []))  # not 1 token per request

    def test_enable_env(self):
        RT = self.RT
        old = os.environ.get(RT.ENV_ENABLE)
        try:
            os.environ[RT.ENV_ENABLE] = "1"
            self.assertTrue(RT.enabled_by_env())
            os.environ[RT.ENV_ENABLE] = "0"
            self.assertFalse(RT.enabled_by_env())
        finally:
            if old is None:
                os.environ.pop(RT.ENV_ENABLE, None)
            else:
                os.environ[RT.ENV_ENABLE] = old

    def test_kv_write_layers_op_schema(self):
        names = ["newpg", "off", "kvn"] + [f"kv_{i}" for i in range(36)]
        f = self.RT._mutating_op_fn(names, lambda *a: None)
        schema = torch.library.infer_schema(f, mutates_args=tuple(names[3:]))
        self.assertIn("Tensor newpg, Tensor off, Tensor kvn, Tensor(a3!) kv_0", schema)
        self.assertIn("Tensor(a38!) kv_35) -> ()", schema)

    @unittest.skipUnless(_HAS_JAX, "requires JAX")
    def test_generated_op_signature(self):
        import inspect

        names = ["x", "kv_lens"] + [f"kv_{i}" for i in range(36)]
        f = self.RT._signature_fn(names, 4, lambda *a: a)
        sig = inspect.signature(f)
        self.assertEqual(list(sig.parameters), names)
        self.assertTrue(all(p.annotation is jax.Array for p in sig.parameters.values()))
        self.assertEqual(f(*range(len(names))), tuple(range(len(names))))


# CPU stand-ins for the torch_tpu ops (same call signatures as the real ones)
_FAKE_L, _FAKE_PAGES, _FAKE_PAGE = 4, 6, 8


@torch.library.custom_op("mk_cpu_test::step", mutates_args=())
def _fake_step_impl(
    x: torch.Tensor,
    kv_lens: torch.Tensor,
    page_indices: torch.Tensor,
    swa_page_indices: torch.Tensor,
    cos_sin: torch.Tensor,
    rank: torch.Tensor,
    rest: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B = x.shape[0]
    hidden = x * 2
    kvn = torch.ones(_FAKE_L, 2, B, 128, dtype=x.dtype)
    newpg = torch.cat([swa_page_indices[:B], page_indices[:B]]).to(torch.int32)
    off = (kv_lens.clamp(min=1) - 1).remainder(_FAKE_PAGE).to(torch.int32)
    return hidden, kvn, newpg, off


@_fake_step_impl.register_fake
def _(x, kv_lens, page_indices, swa_page_indices, cos_sin, rank, rest):
    B = x.shape[0]
    return (
        torch.empty_like(x),
        x.new_empty(_FAKE_L, 2, B, 128),
        torch.empty(2 * B, dtype=torch.int32),
        torch.empty(B, dtype=torch.int32),
    )


@torch.library.custom_op("mk_cpu_test::kv_write_layers", mutates_args=("pools",))
def _fake_kv_write_layers_impl(
    newpg: torch.Tensor, off: torch.Tensor, kvn: torch.Tensor, pools: list[torch.Tensor]
) -> None:
    for p in pools:
        p[0, 0] += 1


@torch.library.custom_op("mk_cpu_test::kv_write", mutates_args=("pool",))
def _fake_kv_write_impl(
    pg: torch.Tensor, off: torch.Tensor, kvn: torch.Tensor, pool: torch.Tensor
) -> None:
    pool[0, 0] += 1


def _fake_step(x, kv_lens, pi, spi, cs, rank, *rest):
    return _fake_step_impl(x, kv_lens, pi, spi, cs, rank, list(rest))


def _fake_kv_write_layers(newpg, off, kvn, *pools):
    _fake_kv_write_layers_impl(newpg, off, kvn, list(pools))


class TestRunUnderTorchCompile(CustomTestCase):
    """``GptOssDecodeMK.run`` (the code inside the decode graph) traces under
    ``torch.compile(fullgraph=True)`` for every diagnostics / ablation / KV-write setting,
    with CPU stand-ins for the torch_tpu ops (catches e.g. context managers, locks or host
    timers inside the traced region, which made e2e3 fail in decode precompile)."""

    def _model(self):
        from types import SimpleNamespace as NS

        L = _FAKE_L

        class Attn(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = NS(k_scale_float=None, v_scale_float=1.0)
                self.rotary_emb = NS(cos_sin_cache=torch.randn(64, 64))

        class Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = Attn()

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = torch.nn.Embedding(32, 128)
                self.norm = torch.nn.LayerNorm(128)
                self.layers = torch.nn.ModuleList([Layer() for _ in range(L)])

        class Pool:
            def __init__(self):
                self.layers_mapping = {l: (l // 2, l % 2 == 0) for l in range(L)}
                self.bufs = [
                    torch.zeros(
                        _FAKE_PAGES, _FAKE_PAGE, 1, 2, 128, dtype=torch.bfloat16
                    )
                    for _ in range(L)
                ]

            def get_fused_kv_buffer(self, l):
                return self.bufs[l]

        class Backend:
            def __init__(self, B):
                self.token_to_kv_pool = Pool()
                self.forward_metadata = NS(
                    kv_lens=torch.arange(1, B + 1, dtype=torch.int32),
                    page_indices=torch.arange(B * 2, dtype=torch.int32),
                    swa_page_indices=torch.arange(B * 2, dtype=torch.int32),
                    decode_query_size=1,
                )
                self._dcp = 1

        return Model().eval(), Backend

    def _check(self, env: dict, B: int = 4):
        import sys
        import types
        from unittest import mock

        from sglang_tpu.gptoss_decode_mk import runtime as RT

        model, Backend = self._model()
        backend = Backend(B)
        fc = types.ModuleType("sglang.srt.model_executor.forward_context")
        fc.get_attn_backend = lambda: backend
        c = dataclasses.replace(_cfg(B=B), L=_FAKE_L, w_dtype="fp8")
        host = {k: torch.zeros(2, dtype=torch.float32) for k in W.w_names(c)}
        full_env = {RT.ENV_TIMING: "0", RT.ENV_ABLATE: "", RT.ENV_KV_WRITE: "layers"}
        full_env.update(env)
        # (not patch.dict(sys.modules): restoring it would drop torch._dynamo & co.
        # imported during the test, whose op registrations cannot be re-run)
        old_fc = sys.modules.get(fc.__name__)
        sys.modules[fc.__name__] = fc
        self.addCleanup(
            lambda: (
                sys.modules.pop(fc.__name__, None)
                if old_fc is None
                else sys.modules.__setitem__(fc.__name__, old_fc)
            )
        )
        with (
            mock.patch.dict(os.environ, full_env),
            mock.patch.object(RT, "get_step_op", lambda *a, **k: _fake_step),
            mock.patch.object(
                RT, "get_kv_write_layers_op", lambda L: _fake_kv_write_layers
            ),
            mock.patch.object(RT, "get_kv_write_op", lambda: _fake_kv_write_impl),
        ):
            mk = RT.GptOssDecodeMK(c, 0, 8, host, "cpu")

            class FB:  # ForwardBatch is a plain (data)class
                residual_stream = object()

            fb = FB()
            ids = torch.arange(B)

            def fwd(ids):
                return mk.run(model, ids, fb)

            torch._dynamo.reset()
            with torch.no_grad():
                eager = fwd(ids)  # eager path: diagnostics active when TIMING=1
                compiled = torch.compile(fwd, fullgraph=True, backend="eager")(ids)
            torch.testing.assert_close(compiled, eager)
            self.assertIsNone(fb.residual_stream)
            if mk.diag is not None:  # only the eager run is counted
                self.assertEqual(mk.diag.eager, 1)
        return backend

    def test_timing_and_ablations_fullgraph(self):
        from sglang_tpu.gptoss_decode_mk import runtime as RT

        for timing in ("0", "1"):
            for ablate in RT.ABLATIONS:
                for kvw in ("layers", "per_layer"):
                    with self.subTest(timing=timing, ablate=ablate, kv_write=kvw):
                        backend = self._check(
                            {
                                RT.ENV_TIMING: timing,
                                RT.ENV_ABLATE: ablate,
                                RT.ENV_KV_WRITE: kvw,
                            }
                        )
                        writes = float(
                            backend.token_to_kv_pool.bufs[0][0, 0].float().max()
                        )
                        # eager + compiled run each write once unless writes are ablated
                        want = 0.0 if ablate in ("no_kv_write", "no_mk") else 2.0
                        self.assertEqual(writes, want)


if __name__ == "__main__":
    unittest.main()
