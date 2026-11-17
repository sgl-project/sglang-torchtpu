"""First-step drafts on the colocated K3 programs
(SGLANG_TPU_K3_PROGRAM_OPT_FIRST_STEP_DRAFTS): the prefill keeps each
finishing prompt's last-position aux streams, and the decoder drafts from them
before the request's first verify, which then verifies real drafts (mode 1,
chained) instead of placeholder rows. Host logic over stub ops, and the aux
row export on a CPU mesh."""

import os
import subprocess
import sys
import textwrap
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.k3_programs import sglang_colocated
from sglang_tpu.k3_programs.kimi_k3 import prefill as prefill_mod
from sglang_tpu.k3_programs.kimi_k3.decode import KimiDecoder
from sglang_tpu.k3_programs.kimi_k3.decode_ops import GREEDY, Control
from sglang_tpu.k3_programs.program import RequestInput, SamplingSpec

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

PAGES = 4
SLOT = 1
PROMPT = 10
FIRST = 7  # the prefill's token


class StubDraft:
    """Records ``step_single`` calls; returns tagged outputs."""

    aux_layers = (3, 24, 48, 72, 90)
    imports_ring = False

    def __init__(self):
        self.config = SimpleNamespace(vocab=32 * 4, hidden=16)
        self.calls = []

    def new_state(self, anchor):
        return SimpleNamespace(vector=("new", anchor))

    def _synthetic_rates(self, drafts):
        return (1.0, 1.0, 0.69, 0.0, 0.0, 0.0, 0.0)

    def step_single(self, state, aux, logits, ids, control, decisions, layout):
        n = len(self.calls)
        self.calls.append(
            dict(state=state, aux=aux, logits=logits, ids=ids, decisions=decisions)
        )
        return (
            SimpleNamespace(vector=("drafted", n)),
            ("emitted", n),
            ("following", n),
            control,
            ("packet", n),
        )


class StubOps:
    def __init__(self):
        self.verifies = []

    def verify_packed(self, ids, aux_layers, layout, control, mode, rates, packet=None):
        self.verifies.append(dict(ids=ids, mode=mode, packet=packet))
        return ("sharded",), ("aux",), ("decisions",)


def _decoder():
    dec = KimiDecoder.__new__(KimiDecoder)
    dec.layout = Control(5, 4, PAGES)
    dec.control = torch.zeros(dec.layout.size, dtype=torch.int32)
    dec.device_plan = None
    dec.stats = dict.fromkeys(
        ("plans", "enters", "chained", "packets", "plain", "batched", "predrafts"), 0
    )
    dec.draft, dec.ops = StubDraft(), StubOps()
    dec.spec_req = dec.draft_state = dec.next_ids = dec.next_packet = None
    dec._drafts, dec._offered, dec._fresh = {}, {}, {}
    dec._zero = torch.zeros(1, dtype=torch.int32)
    dec._modes = {0: "mode0", 1: "mode1"}
    dec._predraft_logits = None
    dec.first_aux = {}
    dec.slot_map = SimpleNamespace(release=lambda rids: None)
    dec.tracked = set()
    dec._group = {}
    dec.plans = []
    dec.entered = []

    def sync_plan(entries, tables):
        dec.plans.append((entries, tables))
        return True

    def enter():
        def op(control, vector, keep):
            dec.entered.append(keep)
            return (("entered", len(dec.entered)),)

        return op

    dec._sync_plan = sync_plan
    dec._enter = enter
    return dec


def _request(scheduled=8, outputs=(FIRST,), computed=PROMPT):
    prompt = np.arange(100, 100 + PROMPT, dtype=np.int32)
    token_ids = np.concatenate((prompt, np.asarray(outputs, np.int32)))
    table = np.array([3, 5, 0, 0], np.int32)
    return RequestInput(
        req_id="a",
        computed=computed,
        scheduled=scheduled,
        token_ids=token_ids,
        num_tokens=len(token_ids),
        num_prompt_tokens=PROMPT,
        block_tables=(table,),
        sampling=SamplingSpec(),
    ), table


def _first_step(dec, request, table):
    known = min(request.end, request.num_tokens)
    entry = (SLOT, True, int(request.token_ids[known - 1]), known - 1, GREEDY)
    return dec._speculate(request, [entry], [table], False)


class TestFirstStepSpeculation(CustomTestCase):
    def test_predrafted_first_verify(self):
        dec = _decoder()
        dec.first_aux["a"] = "aux_a"
        request, table = _request()
        emitted = _first_step(dec, request, table)
        predraft, after = dec.draft.calls
        # The draft op on the prompt's last position: none accepted, the
        # prefill's token as the bonus, the last prompt token as the input.
        self.assertEqual(predraft["aux"], "aux_a")
        self.assertEqual(predraft["decisions"].tolist(), [[0, FIRST]])
        ids = predraft["ids"].numpy()
        self.assertEqual(ids[0], 100 + PROMPT - 1)
        self.assertEqual(ids[1:8].tolist(), [FIRST] * 7)
        self.assertEqual(ids[8:16].tolist(), list(range(PROMPT - 1, PROMPT + 7)))
        self.assertEqual(ids[16:24].tolist(), [SLOT] * 8)
        self.assertEqual(ids[24], 0)
        self.assertEqual(ids[25:].reshape(8, PAGES).tolist(), [table.tolist()] * 8)
        self.assertEqual(tuple(predraft["logits"].shape), (1, 8, 4))
        # The first verify takes the op's drafts and packet (mode 1, chained,
        # no operand rebuilt from the control).
        (verify,) = dec.ops.verifies
        self.assertEqual(verify["ids"], ("following", 0))
        self.assertEqual(verify["packet"], ("packet", 0))
        self.assertEqual(verify["mode"], "mode1")
        self.assertEqual(dec.entered, [])
        self.assertEqual(after["state"].vector, ("drafted", 0))
        self.assertEqual(emitted, ("emitted", 1))
        self.assertEqual(dec.next_ids, ("following", 1))
        self.assertEqual(dec.stats["predrafts"], 1)
        self.assertEqual(dec.stats["chained"], 1)
        self.assertNotIn("a", dec.first_aux)

    def test_without_aux_the_first_verify_rejects_placeholders(self):
        dec = _decoder()
        request, table = _request()
        _first_step(dec, request, table)
        (verify,) = dec.ops.verifies
        self.assertEqual(verify["mode"], "mode0")
        self.assertEqual(verify["packet"], None)
        self.assertEqual(dec.entered, ["mode0"])
        self.assertEqual(len(dec.draft.calls), 1)
        self.assertEqual(dec.stats["predrafts"], 0)

    def test_falls_back_and_drops_the_aux(self):
        cases = (
            dict(scheduled=4),  # not every verify row scheduled
            dict(outputs=(FIRST, 9)),  # a token to replay first
            dict(computed=PROMPT - 1, outputs=()),  # the prompt's end not computed
        )
        for kwargs in cases:
            with self.subTest(**kwargs):
                dec = _decoder()
                dec.first_aux["a"] = "aux_a"
                request, table = _request(**kwargs)
                _first_step(dec, request, table)
                self.assertEqual(dec.ops.verifies[0]["mode"], "mode0")
                self.assertEqual(dec.stats["predrafts"], 0)
                self.assertNotIn("a", dec.first_aux)

    def test_continuing_step_does_not_predraft(self):
        dec = _decoder()
        dec.first_aux["a"] = "aux_a"
        request, table = _request()
        _first_step(dec, request, table)
        dec.first_aux["a"] = "stale"
        entry = (SLOT, False, 0, 0, GREEDY)
        dec._sync_plan = lambda entries, tables: False
        dec._speculate(request, [entry], [table], False)
        self.assertEqual(len(dec.draft.calls), 3)  # one draft op, after the verify
        self.assertEqual(dec.ops.verifies[-1]["ids"], ("following", 1))
        self.assertEqual(dec.first_aux["a"], "stale")

    def test_forget_drops_the_aux(self):
        dec = _decoder()
        dec.first_aux.update(a="aux_a", b="aux_b")
        dec.forget({"a"})
        self.assertEqual(list(dec.first_aux), ["b"])


class TestLastPromptRows(CustomTestCase):
    def test_finishing_pieces_pick_their_last_row(self):
        Piece = prefill_mod.Piece
        rows = [
            SimpleNamespace(num_prompt_tokens=600),
            SimpleNamespace(num_prompt_tokens=5000),
        ]
        # Window rows 0..1023, held reversed (row w at ownership index 1023 - w).
        order = np.arange(1024)[::-1].copy()
        members = [
            (Piece(row=0, start=512, end=600), 0),  # finishes: last row 87
            (Piece(row=1, start=0, end=512), 256),  # continues
        ]
        picked = prefill_mod.last_prompt_rows(rows, members, order, 4)
        self.assertEqual(picked.tolist(), [1023 - 87, -1, -1, -1])
        self.assertIsNone(prefill_mod.last_prompt_rows(rows, members[1:], order, 4))

    def test_kept_aux_is_capped(self):
        prefiller = prefill_mod.KimiPrefiller.__new__(prefill_mod.KimiPrefiller)
        prefiller.first_aux = {}
        for i in range(prefill_mod.FIRST_AUX_KEPT + 3):
            prefiller._keep_first_aux(f"r{i}", i)
        self.assertEqual(len(prefiller.first_aux), prefill_mod.FIRST_AUX_KEPT)
        self.assertEqual(next(iter(prefiller.first_aux)), "r3")


class TestShareFirstAux(CustomTestCase):
    def test_on_with_the_mla_draft_only(self):
        mla = SimpleNamespace(imports_ring=False)
        prefiller = SimpleNamespace(draft_latents=object(), first_aux=None)
        decoder = SimpleNamespace(draft=mla, first_aux=None)
        self.assertTrue(sglang_colocated.share_first_aux(prefiller, decoder))
        self.assertIs(prefiller.first_aux, decoder.first_aux)
        ring = SimpleNamespace(draft_latents=None, first_aux=None)
        self.assertFalse(
            sglang_colocated.share_first_aux(
                ring, SimpleNamespace(draft=SimpleNamespace(), first_aux=None)
            )
        )
        decoder = SimpleNamespace(draft=SimpleNamespace(), first_aux=None)
        self.assertFalse(sglang_colocated.share_first_aux(prefiller, decoder))
        self.assertIsNone(decoder.first_aux)


class TestFirstAuxRowsOnMesh(CustomTestCase):
    """``draft_mla_prefill.first_aux_fn`` under shard_map on four host
    devices: each picked row of every aux stream (AttnRes block snapshots and
    entering taps), exactly, repeated over the verify rows; zeros for -1."""

    SCRIPT = """
        import os
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
        os.environ["JAX_PLATFORMS"] = "cpu"
        import numpy as np, jax, jax.numpy as jnp
        from jax.sharding import Mesh, PartitionSpec as P
        from sglang_tpu.k3_programs.kimi_k3.draft_mla_prefill import first_aux_fn

        R, LOCAL, H, ROWS = 4, 8, 16, 8
        layers = (3, 24, 48, 72, 90)
        rng = np.random.default_rng(0)
        blocks = rng.standard_normal((R, 8, LOCAL, H)).astype(jnp.bfloat16)
        tap3 = rng.standard_normal((R, LOCAL, H)).astype(jnp.bfloat16)
        tap90 = rng.standard_normal((R, LOCAL, H)).astype(jnp.bfloat16)
        picked = np.array([5, -1, 30], np.int32)
        fn = first_aux_fn(layers, "tp", len(picked), ROWS)
        mesh = Mesh(np.asarray(jax.devices()), ("tp",))
        f = jax.jit(jax.shard_map(fn, mesh=mesh, in_specs=(P("tp"),) * 3 + (P(),),
                                  out_specs=(P(),) * len(picked), check_vma=False))
        outs = [np.asarray(x) for x in f(blocks, tap3, tap90, picked)]
        def stream(layer, rank, row):
            if layer == 3:
                return tap3[rank, row]
            if layer == 90:
                return tap90[rank, row]
            return blocks[rank, layer // 12, row]
        for s, j in enumerate(picked):
            assert outs[s].shape == (len(layers), ROWS, H), outs[s].shape
            for a, layer in enumerate(layers):
                want = np.zeros(H, np.float32) if j < 0 else np.asarray(
                    stream(layer, j // LOCAL, j % LOCAL), np.float32)
                for r in range(ROWS):
                    np.testing.assert_array_equal(np.asarray(outs[s][a, r], np.float32), want)
        print("OK")
    """

    def test_rows_match_the_owning_rank(self):
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(self.SCRIPT)],
            capture_output=True,
            text=True,
            env=env,
            timeout=600,
        )
        self.assertIn("OK", result.stdout, result.stderr[-3000:])


if __name__ == "__main__":
    unittest.main()
