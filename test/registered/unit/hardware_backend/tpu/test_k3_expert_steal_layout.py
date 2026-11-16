"""``KernelConfig.expert_steal``: EP8/TP4 routed experts in the sibling-steal
layout (kernels/expert_steal.py), the two cores of a chip holding different
expert groups with the same TP4 quarter.

Kernel rank ``r`` must hold what stock rank ``d = expert_directory(r)`` holds:
the experts the rank loader reads (decode and EP8 prefill, from pre-shard
directory ``d``), the routes the decode router gives it (bit for bit) and the
EP8 prefill's expert offset, while its dense shards stay its own. The decode
MoE's per-rank work then equals stock rank ``d``'s (the expert stream reads no
rank), and only the routed reduction changes: lane ``l`` of each host carries
stock lane ``LANE_ORDER[l]``'s partial, so each host's eight-lane FP32 sum adds
the same BF16 partials in that order."""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.k3_programs.kimi_k3.config import KernelConfig
from sglang_tpu.k3_programs.kimi_k3.kernels import expert_steal as es
from sglang_tpu.k3_programs.kimi_k3.kernels import rank_loader

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

RANKS, LOCAL, LAYERS = 32, 112, 3
# Kernel rank 8h + l holds stock rank 8h + LANE_ORDER[l]'s experts.
LANE_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)


class TestStealLayout(CustomTestCase):
    def test_chip_cores_hold_different_groups(self):
        quarters = {}
        for rank in range(RANKS):
            group, quarter = es.expert_group(rank), es.expert_quarter(rank)
            directory = es.expert_directory(rank)
            # Pre-shard directory d holds group d // 4, quarter d % 4.
            self.assertEqual((directory // 4, directory % 4), (group, quarter))
            self.assertEqual(directory, 8 * (rank // 8) + LANE_ORDER[rank % 8])
            # The chip's other core: the paired group, the same quarter.
            self.assertEqual(es.expert_group(rank ^ 1), group ^ 1)
            self.assertEqual(es.expert_quarter(rank ^ 1), quarter)
            quarters.setdefault(group, []).append(quarter)
            self.assertEqual(es.expert_group(rank, False), rank // 4)
            self.assertEqual(es.expert_quarter(rank, False), rank % 4)
        self.assertEqual(sorted(quarters), list(range(8)))
        for held in quarters.values():
            self.assertEqual(sorted(held), [0, 1, 2, 3])

    def test_owned_expert_group(self):
        for rank in range(RANKS):
            self.assertEqual(
                es.owned_expert_group(rank, 4, True), es.expert_group(rank)
            )
            self.assertEqual(es.owned_expert_group(rank, 4, False), rank // 4)
            for expert_tensor_parallel in (1, 2):  # EP32, EP16: no remap
                for steal in (False, True):
                    self.assertEqual(
                        es.owned_expert_group(rank, expert_tensor_parallel, steal),
                        rank // expert_tensor_parallel,
                    )

    def test_config(self):
        self.assertFalse(KernelConfig().expert_steal)
        self.assertTrue(KernelConfig().replace(expert_steal="1").expert_steal)
        with self.assertRaises(ValueError):
            KernelConfig(expert_steal=True, decode_expert_parallel=16)


DENSE = (
    rank_loader.PREFILL_IO
    + rank_loader.PREFILL_COMMON
    + rank_loader.PREFILL_KDA
    + rank_loader.PREFILL_MLA
    + rank_loader.PREFILL_MOE_DENSE
)
# [MoE layers, local experts, ...], dtype, files.
EXPERTS = {
    "expert_gu": ((LAYERS - 1, LOCAL, 2, 8), np.uint32, 2),
    "expert_gus": ((LAYERS - 1, LOCAL, 1, 8), np.uint8, 1),
    "expert_down": ((LAYERS - 1, LOCAL, 1, 8), np.uint32, 2),
    "expert_ds": ((LAYERS - 1, LOCAL, 1, 8), np.uint8, 1),
}
CACHE = (
    "latent_down",
    "latent_up",
    "shared_gu_0",
    "shared_gu_1",
    "shared_down_0",
    "shared_down_1",
)


def _tags(group, quarter):
    """[MoE layers, local experts] uint32: global expert id, TP4 quarter and
    MoE layer, packed."""
    ids = np.arange(LOCAL, dtype=np.uint32) + LOCAL * group
    layers = np.arange(LAYERS - 1, dtype=np.uint32)[:, None]
    return ids[None] << 8 | np.uint32(quarter) << 4 | layers


def _write_preshard(root):
    """32 pre-shard rank directories (directory d: experts of group d // 4,
    quarter d % 4; dense values naming d) and a projection cache."""
    rng = np.random.default_rng(0)
    for directory in range(RANKS):
        path = root / f"rank{directory:02d}"
        path.mkdir()
        families, offset = {}, 0
        with open(path / "dense.bin", "wb") as dense:
            for i, name in enumerate(DENSE):
                width = 768 if name in ("k_o", "m_o") else 4
                value = np.full((2, width), 1000 * directory + i, np.float32)
                dense.write(value.tobytes())
                families[name] = {
                    "file": "dense.bin",
                    "offset": offset,
                    "shape": list(value.shape),
                    "dtype": "float32",
                }
                offset += value.nbytes
        for name, (shape, dtype, files) in EXPERTS.items():
            if dtype == np.uint32:
                tags = _tags(directory // 4, directory % 4)
                value = np.broadcast_to(tags[:, :, None, None], shape) + (
                    name == "expert_down"
                )
            else:
                value = rng.integers(0, 256, shape, dtype=np.uint8)
            spans = [
                (k * shape[0] // files, (k + 1) * shape[0] // files)
                for k in range(files)
            ]
            names = (
                [f"{name}_{k}.bin" for k in range(files)]
                if files > 1
                else [f"{name}.bin"]
            )
            for filename, (lo, hi) in zip(names, spans):
                np.ascontiguousarray(value[lo:hi]).tofile(path / filename)
            families[name] = {
                "files": names,
                "layer_ranges": [list(span) for span in spans],
                "shape": list(shape),
                "dtype": np.dtype(dtype).name,
            }
        (path / "index.json").write_text(
            json.dumps({"rank": directory, "layers": LAYERS, "families": families})
        )
    cache = root / "prefill-sp-cache"
    cache.mkdir()
    layout = {}
    for i, name in enumerate(CACHE):
        np.full((2, 4), i, np.float32).tofile(cache / f"{name}.bin")
        layout[name] = {"file": f"{name}.bin", "shape": [2, 4], "dtype": "float32"}
    (cache / "layout.json").write_text(
        json.dumps({"layers": LAYERS, "families": layout})
    )


def _arrays(iterator):
    """``{(group, family): array}`` from a rank-loader iterator, per-file
    expert parts (``family@index/count``) joined in layer order."""
    arrays, parts = {}, {}
    for group, name, array in iterator:
        family, _, part = name.partition("@")
        if part:
            parts.setdefault((group, family), {})[int(part.split("/")[0])] = array
        else:
            arrays[(group, family)] = np.asarray(array)
    for key, pieces in parts.items():
        arrays[key] = np.concatenate([pieces[i] for i in sorted(pieces)])
    return arrays


class TestLoaderAgreesWithOwnership(CustomTestCase):
    """The rank loader under ``expert_steal``, decode and EP8 prefill, on a
    small pre-shard: every rank's experts are stock rank
    ``expert_directory(rank)``'s (byte for byte), the group the decode router
    and the prefill assign the rank and its quarter; its other families are
    its own, as without the layout."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        _write_preshard(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _load(self, rank, steal, split, prefill):
        config = KernelConfig(expert_steal=steal, prefill_expert_parallel=8)
        with mock.patch.object(rank_loader, "_config", lambda: config):
            if prefill:
                iterator = rank_loader.iter_prefill_rank(
                    self.root,
                    self.root / "prefill-sp-cache",
                    rank,
                    layers=LAYERS,
                    split_experts=split,
                )
            else:
                iterator = rank_loader.iter_decode_rank(
                    self.root, rank, layers=LAYERS, split_experts=split
                )
            return _arrays(iterator)

    def test_loader_and_ownership_agree_for_every_rank(self):
        for prefill in (False, True):
            group_name = "moe" if prefill else "decode"
            for split in (False, True):
                stock = [self._load(r, False, split, prefill) for r in range(RANKS)]
                for rank in range(RANKS):
                    case = dict(rank=rank, prefill=prefill, split=split)
                    moved = self._load(rank, True, split, prefill)
                    directory = es.expert_directory(rank)
                    self.assertEqual(moved.keys(), stock[rank].keys(), case)
                    for key, array in moved.items():
                        expert = key[1] in rank_loader.EXPERT_FAMILIES
                        want = stock[directory if expert else rank][key]
                        self.assertEqual(array.dtype, want.dtype, (case, key))
                        np.testing.assert_array_equal(array, want, str((case, key)))
                    # The experts of the group the decode router gives the
                    # rank, from the EP8 prefill's first expert on.
                    group = es.owned_expert_group(rank, 4, True)
                    tags = _tags(group, es.expert_quarter(rank))
                    gate_up = moved[(group_name, "expert_gu")]
                    down = moved[(group_name, "expert_down")]
                    for family, tagged in ((gate_up, tags), (down, tags + 1)):
                        np.testing.assert_array_equal(
                            family,
                            np.broadcast_to(tagged[..., None, None], family.shape),
                        )
                    first_expert = (
                        es.owned_expert_group(rank, 32 * LOCAL // 896, True) * LOCAL
                    )
                    self.assertEqual(int(gate_up[0, 0, 0, 0]) >> 8, first_expert, case)
                    if rank != directory:
                        self.assertNotEqual(
                            stock[rank][(group_name, "expert_gu")][0, 0, 0, 0],
                            gate_up[0, 0, 0, 0],
                        )


class TestRouterOwnership(CustomTestCase):
    """The decode router (``_route_hidden_states``, both router halves on
    every rank) on 32 host devices, Pallas interpret mode, B1 / B8 / B16 (null
    rows at B16): with the steal layout (installed config) kernel rank ``r``'s
    local routes, probabilities, route counts and probability totals are stock
    rank ``expert_directory(r)``'s, bit for bit; in both layouts each live
    row's top-16 experts land once per TP4 quarter over the 32 ranks. The
    sharded router (``_route_owned_rows``) is traced: its ownership test asks
    ``owned_expert_group`` with the installed layout every selection."""

    SCRIPT = """
        import os
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=32"
        os.environ["JAX_PLATFORMS"] = "cpu"
        import ml_dtypes, numpy as np, jax, jax.numpy as jnp
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu
        from jax.sharding import Mesh, PartitionSpec as P
        from sglang_tpu.k3_programs.kimi_k3 import config as kc
        kc.install(kc.KernelConfig(expert_steal=True))
        from sglang_tpu.k3_programs.kimi_k3.kernels import decode_megakernel as dmk
        from sglang_tpu.k3_programs.kimi_k3.kernels import expert_steal as es

        assert dmk.EXPERT_STEAL is True
        H, E, K, LOCAL = 256, 896, 16, 112
        mesh = Mesh(np.asarray(jax.devices()), ("tp",))
        NAMES = ("experts", "probabilities", "counts", "totals")

        def run(hidden, router, bias, live):
            B = hidden.shape[0]

            def kernel(live_ref, hidden_ref, bias_ref, router_ref, experts_out,
                       probabilities_out, counts_out, totals_out, scores_vmem, sends,
                       receives):
                live_rows = jnp.stack([live_ref[b] for b in range(B)]) != 0
                experts, probabilities, counts, totals = dmk._route_hidden_states(
                    hidden_ref[...], router_ref, bias_ref, scores_vmem, sends.at[0],
                    receives.at[0], selected_expert_count=K, local_expert_count=LOCAL,
                    expert_tensor_parallel=4, on_local_selection=lambda *unused: None,
                    after_selection=lambda unused: None, live_rows=live_rows,
                    wait_for_projection_half=lambda slot: None)
                experts_out[...] = experts
                probabilities_out[...] = probabilities
                counts_out[...] = jnp.broadcast_to(counts[:, None], counts_out.shape)
                totals_out[...] = jnp.broadcast_to(totals[:, None], totals_out.shape)

            vmem = pltpu.VMEM
            shapes = ((B, K, jnp.int32), (B, K, jnp.float32), (B, 128, jnp.int32),
                      (B, 128, jnp.float32))

            def body(live_rows, hidden, bias, router):
                # Both router halves in VMEM, this rank's lane's first (the
                # decode_router_both_halves arena layout).
                lane = jax.lax.axis_index("tp") % 2
                halves = jnp.pad(router, ((0, 0), (0, 1024 - E)))
                halves = halves.reshape(H, 2, 512).transpose(1, 0, 2)
                own = jnp.where(lane == 0, halves[0], halves[1])
                peer = jnp.where(lane == 0, halves[1], halves[0])
                outputs = pl.pallas_call(
                    kernel,
                    out_shape=tuple(jax.ShapeDtypeStruct(s[:2], s[2]) for s in shapes),
                    grid_spec=pltpu.PrefetchScalarGridSpec(
                        num_scalar_prefetch=1,
                        in_specs=[pl.BlockSpec(memory_space=vmem)] * 3,
                        out_specs=[pl.BlockSpec(memory_space=vmem)] * len(shapes),
                        scratch_shapes=[vmem((B, 1024), jnp.float32),
                                        pltpu.SemaphoreType.DMA((1,)),
                                        pltpu.SemaphoreType.DMA((1,))]),
                    interpret=True,
                )(live_rows, hidden, bias, jnp.stack([own, peer]))
                return tuple(x[None] for x in outputs)

            f = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(P(),) * 4,
                                      out_specs=(P("tp"),) * len(shapes),
                                      check_vma=False))
            return [np.asarray(x) for x in f(live.astype(np.int32), hidden, bias, router)]

        rng = np.random.default_rng(0)
        for B in (1, 8, 16):
            hidden = rng.standard_normal((B, H)).astype(ml_dtypes.bfloat16)
            router = (rng.standard_normal((H, E)) * H**-0.5).astype(ml_dtypes.bfloat16)
            bias = (0.1 * rng.standard_normal((1, E))).astype(np.float32)
            live = (np.arange(B) % 3 != 1) if B == 16 else np.ones(B, bool)
            steal = run(hidden, router, bias, live)
            dmk.EXPERT_STEAL = False
            stock = run(hidden, router, bias, live)
            dmk.EXPERT_STEAL = True
            for rank in range(32):
                directory = es.expert_directory(rank)
                for name, moved, kept in zip(NAMES, steal, stock):
                    assert moved[rank].tobytes() == kept[directory].tobytes(), (
                        B, rank, name)
            for layout, outputs in ((True, steal), (False, stock)):
                experts, counts = outputs[0], outputs[2][..., 0]
                for b in range(B):
                    held = sorted(
                        (LOCAL * es.expert_group(r, layout) + int(experts[r, b, s]),
                         es.expert_quarter(r, layout))
                        for r in range(32) for s in range(counts[r, b]))
                    ids = sorted({expert for expert, _ in held})
                    assert len(ids) == (K if live[b] else 0), (B, layout, b, ids)
                    assert held == [(e, q) for e in ids for q in range(4)], (B, b)
            print(f"B{B}: steal rank r == stock rank expert_directory(r)", flush=True)

        # The sharded router (_route_owned_rows; remote copies, so traced here,
        # not run) takes its ownership from the same function.
        calls = []
        owned_expert_group = es.owned_expert_group

        def recording(rank, expert_tensor_parallel, steal_layout):
            calls.append((expert_tensor_parallel, steal_layout))
            return owned_expert_group(rank, expert_tensor_parallel, steal_layout)

        es.owned_expert_group = recording
        B, ROWS = 8, 2

        def sharded_kernel(owned_ref, bias_ref, router_ref, experts_out, scores_vmem,
                           source_vmem, selections_vmem, sends, receives):
            experts_out[...] = dmk._route_owned_rows(
                owned_ref, B, router_ref, bias_ref, scores_vmem, source_vmem,
                selections_vmem, sends, receives, selected_expert_count=K,
                local_expert_count=LOCAL, expert_tensor_parallel=4,
                on_local_selection=lambda *unused: None,
                after_selection=lambda unused: None)[0]

        def sharded_body(owned, bias, router):
            vmem = pltpu.VMEM
            return pl.pallas_call(
                sharded_kernel,
                out_shape=jax.ShapeDtypeStruct((B, K), jnp.int32),
                in_specs=[pl.BlockSpec(memory_space=vmem)] * 3,
                out_specs=pl.BlockSpec(memory_space=vmem),
                scratch_shapes=[vmem((B, 1024), jnp.float32), vmem((ROWS, 128), jnp.float32),
                                vmem((B, 128), jnp.float32), pltpu.SemaphoreType.DMA((5,)),
                                pltpu.SemaphoreType.DMA((5,))],
                interpret=True,
            )(owned, bias, router)[None]

        jax.make_jaxpr(jax.shard_map(sharded_body, mesh=mesh, in_specs=(P(),) * 3,
                                     out_specs=P("tp"), check_vma=False))(
            jnp.zeros((ROWS, H), jnp.bfloat16), jnp.zeros((1, E), jnp.float32),
            jnp.zeros((1, H, 512), jnp.bfloat16))
        assert calls == [(4, True)] * K, calls
        print("OK")
    """

    def test_router_matches_stock_ranks(self):
        try:
            from jax.experimental.pallas import tpu as pltpu  # noqa: F401
        except ImportError:
            self.skipTest("no Pallas TPU")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(self.SCRIPT)],
            capture_output=True,
            text=True,
            env=env,
            timeout=900,
        )
        self.assertIn(
            "OK",
            result.stdout.splitlines()[-1:] or [""],
            result.stdout[-2000:] + result.stderr[-3000:],
        )


if __name__ == "__main__":
    unittest.main()
