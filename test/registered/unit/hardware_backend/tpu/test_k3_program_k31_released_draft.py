"""Kimi-K3.1-Pro with its released DSpark draft (Inferact/Kimi-K3.1-Pro-DSpark,
a Qwen3-style speculator: GQA with a 2048-token sliding window, drafting from a
ring) on the K3 programs: the speculator config checks, the target's embedding
and LM head in the pre-shard, the fused draft's parts, the prefill ring's aux
streams, the decoder's verify-then-draft path and the draft's pinned revision.
Host logic over stubs; the fused kernel itself is checked on TPU, the recipe's
draft selection in test_tpu_kimi_k3_1_program_recipe.py."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import jax.numpy as jnp
import numpy as np
import torch

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.k3_programs import sglang_colocated, sglang_program
from sglang_tpu.k3_programs.kimi_k3 import draft as draft_mod
from sglang_tpu.k3_programs.kimi_k3 import draft_ring, prefill
from sglang_tpu.k3_programs.kimi_k3.decode import KimiDecoder
from sglang_tpu.k3_programs.kimi_k3.decode_ops import GREEDY, Control
from sglang_tpu.k3_programs.kimi_k3.draft_mla_prefill import stream_source
from sglang_tpu.k3_programs.kimi_k3.kernels import dspark
from sglang_tpu.k3_programs.program import RequestInput, SamplingSpec
from sglang_tpu.k3_programs.tools import preshard

# Needs jax and torch_tpu, which the CPU runners do not install; the TPU unit
# stage's time budget is spent, so test_k3_programs_nightly.py runs it in the
# TPU nightly.
register_tpu_ci(
    est_time=60,
    suite="nightly-tpu-4-v7",
    nightly=True,
    disabled="runs in the TPU nightly through test_k3_programs_nightly.py",
)

K31_AUX = (76, 80, 84, 88, 92)


def released_config(**changes):
    """Inferact/Kimi-K3.1-Pro-DSpark's config.json (e4e6e626), the fields read."""
    layer = {
        "head_dim": 64,
        "hidden_size": 7168,
        "intermediate_size": 14336,
        "layer_types": ["sliding_attention"] * 5,
        "model_type": "qwen3",
        "num_attention_heads": 96,
        "num_hidden_layers": 5,
        "num_key_value_heads": 16,
        "rms_norm_eps": 1e-05,
        "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
        "sliding_window": 2048,
        "use_sliding_window": True,
        "vocab_size": 163840,
    }
    raw = {
        "architectures": ["DSparkDraftModel"],
        "aux_hidden_state_layer_ids": list(K31_AUX),
        "block_size": 7,
        "draft_vocab_size": 163840,
        "markov_head_type": "vanilla",
        "markov_rank": 512,
        "mask_token_id": 163839,
        "sample_from_anchor": True,
        "sliding_window_non_causal": False,
        "speculators_model_type": "dspark",
        "target_hidden_size": None,
        "transformer_layer_config": layer,
    }
    for key, value in changes.items():
        if key.startswith("layer_"):
            layer[key[len("layer_") :]] = value
        else:
            raw[key] = value
    return raw


def write_config(directory: Path, raw: dict) -> Path:
    (directory / "config.json").write_text(json.dumps(raw))
    return directory


class TestSpeculatorConfig(CustomTestCase):
    def test_released_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = dspark.DraftConfig.from_checkpoint(
                write_config(Path(tmp), released_config())
            )
        self.assertEqual(c.aux_layers, K31_AUX)
        self.assertEqual((c.markov_rank, c.mask_token, c.window), (512, 163839, 2048))
        self.assertEqual((c.heads, c.kv_heads, c.head_dim), (96, 16, 64))

    def test_kimi_k3_speculator_config(self):
        raw = released_config(
            aux_hidden_state_layer_ids=[24, 48, 72, 88, 92],
            block_size=8,
            markov_rank=256,
            mask_token_id=163837,
        )
        dspark.check_speculator_config(raw)

    def test_refuses_what_the_kernel_does_not_run(self):
        for changes in (
            {"sliding_window_non_causal": True},
            {"layer_rope_parameters": {"rope_type": "yarn", "rope_theta": 1e4}},
            {"layer_layer_types": ["full_attention"] * 5},
            {"layer_num_key_value_heads": 8},
            {"block_size": 5},
            {"markov_rank": 128},
            {"sample_from_anchor": False},
            {"aux_hidden_state_layer_ids": [76, 80, 84, 88]},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                dspark.check_speculator_config(released_config(**changes))


TINY = dspark.DraftConfig(
    hidden=8, layers=1, heads=2, kv_heads=1, head_dim=4, intermediate=8, vocab=16,
    aux_layers=(1, 2), markov_rank=2,
)  # fmt: skip


class TestSharedEmbedding(CustomTestCase):
    """A speculator without its own embedding and LM head takes the target's."""

    def _checkpoint(self, tmp: Path) -> dict:
        from safetensors.numpy import save_file

        rng = np.random.default_rng(0)
        own = {
            name: rng.standard_normal(shape).astype(np.float32)
            for name, shape in dspark.weight_shapes(TINY).items()
            if name not in dspark.SHARED_WITH_TARGET
        }
        save_file(own, str(tmp / "model.safetensors"))
        return own

    def test_shared_tensors(self):
        with tempfile.TemporaryDirectory() as tmp:
            own = self._checkpoint(Path(tmp))
            self.assertEqual(
                dspark.missing_shared(Path(tmp)),
                ("embed_tokens.weight", "lm_head.weight"),
            )
            with self.assertRaises(ValueError):
                dspark.load_checkpoint_weights(Path(tmp), TINY)
            shared = {
                name: np.full((TINY.vocab, TINY.hidden), i + 1, np.float32)
                for i, name in enumerate(dspark.SHARED_WITH_TARGET)
            }
            weights = dspark.load_checkpoint_weights(Path(tmp), TINY, shared)
        self.assertEqual(float(weights["embed_tokens.weight"][3, 2]), 1.0)
        self.assertEqual(float(weights["lm_head.weight"][5, 1]), 2.0)
        np.testing.assert_array_equal(
            weights["fc.weight"].astype(np.float32),
            own["fc.weight"].astype(weights["fc.weight"].dtype).astype(np.float32),
        )

    def test_preshard_reads_the_target(self):
        reads, loaded = [], {}

        class Target:
            def __init__(self, path):
                self.path = path

            def read(self, key):
                reads.append((self.path, key))
                return key

        def load_weights(directory, config, shared=None):
            loaded["shared"] = shared
            return {}

        stub = SimpleNamespace(
            DraftConfig=SimpleNamespace(from_checkpoint=lambda d: TINY),
            missing_shared=lambda d: ("embed_tokens.weight", "lm_head.weight"),
            SHARED_WITH_TARGET=dspark.SHARED_WITH_TARGET,
            load_checkpoint_weights=load_weights,
            kernel_weight_shards=lambda c, w, max_position: "shards",
        )
        mla = SimpleNamespace(is_mla_draft=lambda d: False)
        with (
            mock.patch.object(preshard, "_mla_module", return_value=mla),
            mock.patch.object(
                preshard,
                "_modules",
                return_value=(SimpleNamespace(Checkpoint=Target), stub),
            ),
        ):
            with self.assertRaises(ValueError):
                preshard.draft_shards(stub, "/draft", 16384)
            out = preshard.draft_shards(stub, "/draft", 16384, "/k31")
        self.assertEqual(out, "shards")
        self.assertEqual(
            reads,
            [
                ("/k31", "language_model.model.embed_tokens.weight"),
                ("/k31", "language_model.lm_head.weight"),
            ],
        )
        self.assertEqual(
            sorted(loaded["shared"]), ["embed_tokens.weight", "lm_head.weight"]
        )


class TestRopeTables(CustomTestCase):
    """KernelConfig.draft_rope_bf16: BF16 tables hold exactly what the kernel
    rounds the FP32 rows to, and each window covers a block's 8 positions."""

    def test_bf16_tables(self):
        import dataclasses

        import ml_dtypes

        c = dspark.DraftConfig()
        f32 = dspark.rope_tables(c, 5000)
        bf16 = dspark.rope_tables(dataclasses.replace(c, rope_bf16=True), 5000)
        for a, b in zip(f32, bf16):
            self.assertEqual(b.dtype, ml_dtypes.bfloat16)
            np.testing.assert_array_equal(a.astype(ml_dtypes.bfloat16), b)
        for rope_bf16 in (False, True):
            align, rows, _ = dspark._rope_window(
                dataclasses.replace(c, rope_bf16=rope_bf16)
            )
            self.assertEqual((align, rows), (16, 32) if rope_bf16 else (8, 16))
            for position in range(100):
                start = position // align * align
                self.assertLess(position + dspark.ROWS - 1, start + rows)


def _draft(markov_rank):
    d = object.__new__(draft_mod.DsparkDraft)
    d.config = SimpleNamespace(markov_rank=markov_rank)
    return d


class TestParts(CustomTestCase):
    def test_rank_512_fits_vmem(self):
        # Fused calls of at most two requests in the kernel's whole-weight
        # mode (up to four requests) and at most eight in its streamed mode.
        d = _draft(512)
        self.assertEqual(d.parts(1, 8), [(0, 1)])
        self.assertEqual(d.parts(2, 8), [(0, 2)])
        self.assertEqual(d.parts(4, 8), [(0, 2), (2, 2)])
        self.assertEqual(d.parts(4, 4), [(0, 2), (2, 2)])
        self.assertEqual(d.parts(8, 4), [(0, 8)])
        self.assertEqual(d.parts(16, 4), [(0, 8), (8, 8)])
        for verified in (1, 2, 3, 4, 8, 16):
            for rows in (4, 8):
                for _, count in d.parts(verified, rows):
                    self.assertTrue(count <= 2 or 5 <= count <= 8, (verified, rows))

    def test_rank_256_unchanged(self):
        d = _draft(256)
        self.assertEqual(d.parts(16, 4), [(0, 16)])
        self.assertEqual(d.parts(8, 8), [(0, 8)])
        self.assertEqual(d.parts(16, 8), [(0, 8), (8, 8)])


def _stub_fused(weights, caches, positions, aux, logits, vectors, bases, count, anchor):
    out = jnp.zeros((bases.shape[0], 128), jnp.int32)
    out = out.at[:, dspark.OUT_COUNT].set(count).at[:, dspark.OUT_BONUS].set(anchor)
    # Outputs that depend on each request's own verify rows.
    out = out.at[:, 40].set(aux[0, :: dspark.ROWS, 0].astype(jnp.int32))
    out = out.at[:, 41].set(logits[0, :, 0, 0].astype(jnp.int32))
    return out, caches + 1, positions + bases[None, :, None]


class TestDraftingInParts(CustomTestCase):
    """The capped parts draft exactly what one fused call over the whole
    verify would (emitted tokens and control state)."""

    def test_parts_match_one_call(self):
        for verified, rows_per in ((4, 8), (4, 4), (16, 4)):
            with self.subTest(verified=verified, rows_per=rows_per):
                self._check(verified, rows_per)

    def _check(self, verified, rows_per):
        rng = np.random.default_rng(verified * rows_per)
        pages = 3
        control = Control(slots=65, rows=64, pages=pages)
        slots = rng.permutation(np.arange(1, 65))[:verified]
        slots[[0, verified - 1]] = 0  # padding requests share the null slot
        entries = [
            (int(s), i % 3 == 0, int(rng.integers(5, 999)), int(rng.integers(9, 9000)),
             (0.0, 1.0, 0, 0))
            for i, s in enumerate(slots)
        ]  # fmt: skip
        plan = control.plan(entries, [np.arange(pages, dtype=np.int32)] * verified)
        state = np.concatenate([rng.integers(0, 50, 3 * control.slots), plan])
        state = jnp.asarray(state.astype(np.int32))
        total = verified * rows_per
        ids = jnp.asarray(
            rng.integers(0, 1000, 4 * total + verified * pages), jnp.int32
        )
        aux = jnp.asarray(rng.integers(0, 99, (5, total, 16)), jnp.bfloat16)
        logits = jnp.asarray(rng.integers(0, 99, (32, total, 32)), jnp.float32)
        decisions = jnp.asarray(
            np.stack(
                [rng.integers(0, rows_per, verified), rng.integers(0, 999, verified)], 1
            ),
            jnp.int32,
        )

        def draft(requests, first, state):
            fn = draft_mod._batched_fn(
                dspark, _stub_fused, control, (), requests, rows_per, first, verified
            )
            return fn(
                jnp.zeros((1, requests, 2, 4, 8), jnp.bfloat16),
                jnp.zeros((1, requests, 4), jnp.int32),
                jnp.zeros((1, requests, 128), jnp.int32),
                aux, logits, ids, state, decisions,
            )  # fmt: skip

        whole = draft(verified, 0, state)
        parts, part_state = [], state
        for first, requests in _draft(512).parts(verified, rows_per):
            out = draft(requests, first, part_state)
            parts.append(out)
            part_state = out[4]
        np.testing.assert_array_equal(
            np.concatenate([p[0] for p in parts], axis=1), whole[0]
        )
        np.testing.assert_array_equal(np.concatenate([p[3] for p in parts]), whole[3])
        np.testing.assert_array_equal(part_state, whole[4])


def _ring(aux_layers):
    ring = object.__new__(draft_ring.PrefillRing)
    ring.config = SimpleNamespace(aux_layers=tuple(aux_layers))
    ring.sources = {layer: stream_source(layer) for layer in aux_layers}
    return ring


class TestPrefillRingStreams(CustomTestCase):
    def test_k31_taps(self):
        ring = _ring(K31_AUX)
        # 84 is a block snapshot; 76 is pair 9's input (kept), 80 / 88 pair
        # 9's / 10's middles, 92 the final op's input.
        self.assertEqual(ring.entering_layers, (76, 80, 88, 92))
        rows, hidden = 3, 4
        blocks = jnp.arange(8 * rows * hidden, dtype=jnp.float32).reshape(
            8, rows, hidden
        )
        taps = {
            layer: jnp.full((rows, hidden), layer, jnp.float32)
            for layer in ring.entering_layers
        }
        aux = draft_ring.aux_streams(K31_AUX, blocks, taps)
        self.assertEqual(aux.shape, (5, rows, hidden))
        for i, layer in enumerate(K31_AUX):
            expected = blocks[7] if layer == 84 else taps[layer]
            np.testing.assert_array_equal(aux[i], expected)

    def test_kimi_k3_taps_as_before(self):
        # The vendored stack: blocks 2, 4, 6, the pair-10 middle, the prefix.
        ring = _ring((24, 48, 72, 88, 92))
        self.assertEqual(ring.entering_layers, (88, 92))
        blocks = jnp.arange(8 * 2 * 2, dtype=jnp.float32).reshape(8, 2, 2)
        middle, prefix = jnp.full((2, 2), -1.0), jnp.full((2, 2), -2.0)
        aux = draft_ring.aux_streams(
            (24, 48, 72, 88, 92), blocks, {88: middle, 92: prefix}
        )
        old = jnp.stack((blocks[2], blocks[4], blocks[6], middle, prefix))
        np.testing.assert_array_equal(aux, old)

    def test_prefill_keeps_the_ring_streams(self):
        prefiller = object.__new__(prefill.KimiPrefiller)
        prefiller.draft_latents, prefiller.ring = None, _ring(K31_AUX)
        self.assertEqual(prefiller._draft_keep(), (76, 80, 88, 92))
        prefiller.ring = None
        self.assertEqual(prefiller._draft_keep(), ())


class RingDraft:
    """A ring draft's interface (``DsparkDraft``): no fused verify op, no
    verify packets, a ring imported per request."""

    aux_layers = K31_AUX
    imports_ring = True

    def __init__(self):
        self.config = SimpleNamespace(vocab=32 * 4)
        self.calls = []

    def from_ring(self, cache, positions, anchor):
        return SimpleNamespace(vector=("ring", cache))

    def new_state(self, anchor):
        return SimpleNamespace(vector=("cold", anchor))

    def _synthetic_rates(self, drafts):
        return (1.0, 1.0, 1.0, 0.16, 0.0, 0.0, 0.0)

    def step_single(self, state, aux, logits, ids, control, decisions, layout):
        n = len(self.calls)
        self.calls.append(dict(state=state, ids=ids, decisions=decisions))
        return (
            SimpleNamespace(vector=("drafted", n)),
            ("emitted", n),
            ("following", n),
            control,
        )


class TestDecoderWithRingDraft(CustomTestCase):
    def _decoder(self):
        dec = KimiDecoder.__new__(KimiDecoder)
        dec.layout = Control(5, 4, 4)
        dec.control = torch.zeros(dec.layout.size, dtype=torch.int32)
        dec.device_plan, dec.first_aux = None, None
        dec.stats = dict.fromkeys(
            ("plans", "enters", "chained", "packets", "plain", "batched", "predrafts"),
            0,
        )
        dec.draft, dec.verifies = RingDraft(), []

        def verify_packed(ids, aux_layers, layout, control, mode, rates, packet=None):
            dec.verifies.append(dict(ids=ids, mode=mode, packet=packet, rates=rates))
            return ("sharded",), ("aux",), ("decisions",)

        dec.ops = SimpleNamespace(verify_packed=verify_packed)
        dec.spec_req = dec.draft_state = dec.next_ids = dec.next_packet = None
        dec._drafts, dec._fresh = {}, {}
        dec._offered = {"a": ("ring_a", "positions_a")}
        dec._zero = torch.zeros(1, dtype=torch.int32)
        dec._modes = {0: "mode0", 1: "mode1"}
        dec._sync_plan = lambda entries, tables: False
        dec._enter = lambda: lambda control, vector, keep: (("entered", keep),)
        return dec

    def test_verify_then_draft(self):
        dec = self._decoder()
        table = np.array([3, 5, 0, 0], np.int32)
        request = RequestInput(
            req_id="a", computed=10, scheduled=8,
            token_ids=np.arange(11, dtype=np.int32), num_tokens=11, num_prompt_tokens=10,
            block_tables=(table,), sampling=SamplingSpec(),
        )  # fmt: skip
        entry = (1, True, 10, 10, GREEDY)
        first = dec._speculate(request, [entry], [table], False)
        # The first step drafts from the imported ring; placeholder rows (mode 0).
        self.assertEqual(dec.draft.calls[0]["state"].vector, ("ring", "ring_a"))
        self.assertEqual(dec.verifies[0]["mode"], "mode0")
        self.assertEqual(dec.verifies[0]["rates"], dec.draft._synthetic_rates(7))
        self.assertIsNone(dec.next_packet)
        self.assertEqual(first, ("emitted", 0))
        # The next step chains: the draft op's operand, no packet.
        second = dec._speculate(request, [(1, False, 0, 11, GREEDY)], [table], False)
        self.assertEqual(dec.verifies[1]["ids"], ("following", 0))
        self.assertEqual(dec.verifies[1]["mode"], "mode1")
        self.assertIsNone(dec.verifies[1]["packet"])
        self.assertEqual(second, ("emitted", 1))
        self.assertEqual(dec.stats["packets"], 0)

    def test_no_first_step_drafts(self):
        prefiller = SimpleNamespace(draft_latents=None, ring=object())
        decoder = SimpleNamespace(draft=RingDraft())
        self.assertFalse(sglang_colocated.share_first_aux(prefiller, decoder))


class TestDraftRevision(CustomTestCase):
    def test_revision_pinned_checkpoint(self):
        with mock.patch(
            "huggingface_hub.snapshot_download", return_value="/snap"
        ) as snap:
            sglang_program._local_checkpoint("Inferact/Kimi-K3.1-Pro-DSpark@e4e6e626")
            sglang_program._local_checkpoint("Inferact/Kimi-K3.1-Pro-DSpark-preview")
            sglang_program._local_checkpoint(
                "Inferact/Kimi-K3.1-Pro-DSpark-preview", "6a2ef20a"
            )
            # Two different pins of one draft.
            with self.assertRaisesRegex(ValueError, "revision"):
                sglang_program._local_checkpoint(
                    "Inferact/Kimi-K3.1-Pro-DSpark@e4e6e626", "6a2ef20a"
                )
        self.assertEqual(
            snap.call_args_list,
            [
                mock.call(
                    "Inferact/Kimi-K3.1-Pro-DSpark",
                    revision="e4e6e626",
                    local_files_only=True,
                ),
                mock.call(
                    "Inferact/Kimi-K3.1-Pro-DSpark-preview",
                    revision=None,
                    local_files_only=True,
                ),
                mock.call(
                    "Inferact/Kimi-K3.1-Pro-DSpark-preview",
                    revision="6a2ef20a",
                    local_files_only=True,
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main()
