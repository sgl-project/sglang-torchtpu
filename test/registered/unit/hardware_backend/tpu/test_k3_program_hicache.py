"""The Kimi-K3 programs' host tier (sglang_hicache) over CPU unified pools and a
row-copying stand-in for the TPU DMA engine: KV pages (every pool, the draft's
included) and KDA checkpoints round-trip, the stack strategy wins for K3 pools,
the controller offloads the checkpoint pool, and the program's worker waits for
a reload before it reads the prefix."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.managers.cache_controller import BatchDoneCounter
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as assembler
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu import tpu_dma_hybrid_cache_controller as tpu_controller
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.k3_programs import sglang_bridge as bridge
from sglang_tpu.k3_programs import sglang_hicache as hicache
from sglang_tpu.k3_programs import sglang_pools
from sglang_tpu.k3_programs import sglang_worker as worker
from sglang_tpu.k3_programs.kimi_k3.layout import KDA, MLA
from sglang_tpu.k3_programs.program import StepOutput
from sglang_tpu.tpu_dma_hicache import NOOP_EVENT, register_tpu_dma_hicache_backend

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

ENGINE = "k3_row_engine"
STEP = 8192  # bf16 logical block


class RowEngine:
    """Copies rows of every pool between the device and host blocks."""

    built = []

    def __init__(self, device_pool, num_host_pages):
        self.pools = [device_pool.get_fused_kv_buffer(i) for i in range(device_pool.layer_num)]
        self.host = [torch.zeros((num_host_pages, *p.shape[1:]), dtype=p.dtype) for p in self.pools]
        self.calls = []
        RowEngine.built.append(self)

    def backup(self, device_pages, host_pages):
        self.calls.append(("backup", list(device_pages), list(host_pages)))
        for pool, host in zip(self.pools, self.host):
            host[torch.tensor(host_pages)] = pool[torch.tensor(device_pages)]
        return NOOP_EVENT

    def load(self, host_pages, device_pages):
        self.calls.append(("load", list(host_pages), list(device_pages)))
        for pool, host in zip(self.pools, self.host):
            pool.index_copy_(0, torch.tensor(device_pages), host[torch.tensor(host_pages)])
        return NOOP_EVENT


def setUpModule():
    register_tpu_dma_hicache_backend(ENGINE, RowEngine)


def _pools(slots=4, pages=10):
    with mock.patch(
        "sglang.srt.runtime_context.get_model",
        return_value=SimpleNamespace(kv_cache_dtype="auto"),
    ):
        state = sglang_pools.K3StatePool(size=slots, mamba_layer_ids=KDA, device="cpu")
        kv = sglang_pools.K3UnifiedKVPool(
            size=pages * STEP, page_size=STEP, dtype=torch.bfloat16, layer_num=len(MLA), device="cpu"
        )
    for i, pool in enumerate(kv.raw_pools()):
        pool.copy_(torch.randn(pool.shape).to(pool.dtype) + i)
    return state, kv


def _tokens(pages):
    return torch.cat([torch.arange(p * STEP, (p + 1) * STEP) for p in pages])


class TestHostPools(CustomTestCase):
    def setUp(self):
        self.state, self.kv = _pools()
        self.view = hicache.K3PoolView(self.kv.raw_pools(), STEP, "cpu")
        self.engine = RowEngine(self.view, 8 + 3 * 2)
        self.kv_host = hicache.K3HostKVPool(self.kv, self.view, self.engine, 8, STEP)
        self.state_host = hicache.K3HostStatePool(self.state, self.engine, 2, 8, self.view.block_bytes)

    def test_view_and_sizes(self):
        self.assertEqual(self.view.layer_num, len(MLA) + bridge.DSPARK_DRAFT_LAYERS)
        self.assertEqual(self.kv_host.size, 8 * STEP)
        self.assertEqual(self.kv_host.size_per_token, self.view.block_bytes // STEP)
        self.assertEqual(self.state_host.size_per_token, 3 * self.view.block_bytes)
        self.assertEqual(hicache.host_blocks(0, 2.0, 10, 4, 1000), (20, 4))
        self.assertEqual(hicache.host_blocks(1e-5, 0, 10, 1, 1000), (7, 1))
        with self.assertRaisesRegex(ValueError, "holds no KV page"):
            hicache.host_blocks(1e-6, 0, 10, 4, 1000)

    def test_kv_pages_round_trip_in_every_pool(self):
        before = [p[[3, 5]].clone() for p in self.kv.raw_pools()]
        host = self.kv_host.alloc(2 * STEP)
        self.kv_host.backup_from_device_all_layer(self.kv, host, _tokens([3, 5]), ENGINE)
        for pool in self.kv.raw_pools():
            pool[[3, 5]] = 0
        self.kv_host.load_all_layer(self.kv, host, _tokens([6, 7]))
        for pool, rows in zip(self.kv.raw_pools(), before):
            torch.testing.assert_close(pool[[6, 7]], rows)
        self.assertEqual(self.engine.calls[0], ("backup", [3, 5], [0, 1]))
        self.assertEqual(self.kv_host.free(host), 2 * STEP)
        self.assertEqual(self.kv_host.available_size(), 8 * STEP)

    def test_checkpoints_round_trip_beside_the_pages(self):
        plan = self.kv.plan
        rows = [plan.state_row(2) + g for g in range(3)]
        before = [p[rows].clone() for p in self.kv.raw_pools()]
        host = self.state_host.alloc(1)
        self.state_host.backup_from_device_all_layer(self.state, host, torch.tensor([2]), ENGINE)
        self.assertEqual(self.engine.calls[-1], ("backup", rows, [8, 9, 10]))
        # Loaded into another slot, as a prefix hit's copy into a request's slot.
        self.state_host.load_all_layer(self.state, host, torch.tensor([[3]]))
        target = [plan.state_row(3) + g for g in range(3)]
        for pool, rows_before in zip(self.kv.raw_pools(), before):
            torch.testing.assert_close(pool[target], rows_before)
        self.assertIsNone(self.state_host.alloc(2))
        self.assertEqual(self.state_host.free(host), 1)
        self.assertEqual(self.state_host.available_size(), 2)


class TestStrategyAndController(CustomTestCase):
    def _kvcache(self, kv):
        cache = HybridLinearKVPool.__new__(HybridLinearKVPool)
        cache.full_kv_pool = kv
        cache.full_attention_layer_id_mapping = {layer: m for m, layer in enumerate(MLA)}
        cache.start_layer = 0
        return cache

    def test_k3_strategy_wins_for_the_unified_pools(self):
        _, kv = _pools()
        kvcache = self._kvcache(kv)
        components = {ComponentType.FULL, ComponentType.MAMBA}
        hicache.install()
        hicache.install()  # idempotent
        self.assertIsInstance(assembler._select_strategy(kvcache, components), hicache.K3Strategy)
        other = HybridLinearKVPool.__new__(HybridLinearKVPool)
        other.full_kv_pool = object()
        self.assertNotIsInstance(
            assembler._select_strategy(other, components), hicache.K3Strategy
        )

    def test_controller_offloads_checkpoints(self):
        tpu_controller._check_kv_page_pool(
            PoolName.MAMBA, hicache.K3HybridCacheController.offloaded_pools
        )
        with self.assertRaisesRegex(NotImplementedError, "mamba"):
            tpu_controller._check_kv_page_pool(PoolName.MAMBA, frozenset({PoolName.KV}))

    def _build(self, state, kv, kv_allocator=None):
        kvcache = self._kvcache(kv)
        allocator = SimpleNamespace(alloc=mock.Mock(), free=mock.Mock())
        kv_allocator = kv_allocator or SimpleNamespace(kv_free_fence=None)
        kv_allocator.get_kvcache = lambda: kvcache
        params = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(
                mamba_pool=state,
                mamba_map={layer: k for k, layer in enumerate(KDA)},
                mamba_allocator=allocator,
            ),
            token_to_kv_pool_allocator=kv_allocator,
            page_size=STEP,
            tp_cache_group=None,
            attn_cp_cache_group=None,
            attn_tp_cache_group=None,
            pp_cache_group=None,
        )
        memory = SimpleNamespace(
            hicache_size=0,
            hicache_ratio=2.0,
            hicache_io_backend=ENGINE,
            hicache_write_policy="write_through",
            hicache_host_memory_mode="cache",
        )
        RowEngine.built.clear()
        with (
            mock.patch("sglang.srt.runtime_context.get_memory", return_value=memory),
            mock.patch(
                "sglang.srt.mem_cache.pool_host.base.host_memory_budget_bytes",
                side_effect=lambda requested: requested,
            ),
            tpu_envs.SGLANG_TPU_K3_PROGRAM_HICACHE_CHECKPOINTS.override(3),
        ):
            result = hicache.K3Strategy().build(
                cache=SimpleNamespace(), kvcache=kvcache, params=params, server_args=None,
                load_cache_event=None,
            )
        (engine,) = RowEngine.built
        return result, engine

    def test_build_assembles_both_pools_on_one_engine(self):
        state, kv = _pools()
        result, engine = self._build(state, kv)
        kv_host = result.component_host_pools[ComponentType.FULL]
        state_host = result.component_host_pools[ComponentType.MAMBA]
        self.assertEqual(kv_host.page_num, 2 * kv.plan.kv_blocks)
        self.assertEqual((state_host.size, state_host.block_base), (3, kv_host.page_num))
        self.assertEqual(engine.host[0].shape[0], kv_host.page_num + 3 * 3)
        self.assertIs(kv_host.transfer, state_host.transfer)
        self.assertIsInstance(result.cache_controller, hicache.K3HybridCacheController)
        self.assertEqual(
            {e.name for e in result.host_pool_group.entries}, {PoolName.KV, PoolName.MAMBA}
        )
        self.assertIsInstance(kv.layer_transfer_counter, BatchDoneCounter)

    def test_controller_backs_up_and_reloads_a_prefix_with_its_checkpoint(self):
        state, kv = _pools()
        kv_allocator = SimpleNamespace(
            kv_free_fence=None, alloc=mock.Mock(return_value=_tokens([6, 7]))
        )
        result, _ = self._build(state, kv, kv_allocator)
        controller = result.cache_controller
        plan = kv.plan
        pages = [p[[3, 5]].clone() for p in kv.raw_pools()]
        rows = [plan.state_row(2) + g for g in range(3)]
        states = [p[rows].clone() for p in kv.raw_pools()]
        backup = PoolTransfer(name=PoolName.MAMBA, device_indices=torch.tensor([2]))
        host = controller.write(_tokens([3, 5]), extra_pools=[backup])
        for ack in controller.ack_write_queue:
            ack.finish_event.synchronize()
        for pool in kv.raw_pools():
            pool[[3, 5]] = 0
            pool[rows] = 0
        restore = PoolTransfer(
            name=PoolName.MAMBA, host_indices=backup.host_indices, device_indices=torch.tensor([3])
        )
        device = controller.load(host, extra_pools=[restore])
        self.assertEqual(device.tolist(), _tokens([6, 7]).tolist())
        self.assertGreaterEqual(controller.start_loading(), 0)
        target = [plan.state_row(3) + g for g in range(3)]
        for pool, page_rows, state_rows in zip(kv.raw_pools(), pages, states):
            torch.testing.assert_close(pool[[6, 7]], page_rows)
            torch.testing.assert_close(pool[target], state_rows)


class TestPlatformAllowsTheK3HostTier(CustomTestCase):
    def test_hierarchical_cache_needs_a_dma_backend(self):
        from sglang_tpu.tpu_platform import TpuSRTPlatform

        view = SimpleNamespace(
            enable_mixed_chunk=False,
            disable_radix_cache=False,
            mamba_radix_cache_strategy="extra_buffer",
            enable_hierarchical_cache=True,
            enable_int8_mamba_checkpoint=False,
            hicache_io_backend="kernel",
            speculative_algorithm="K3_PROGRAM",
        )
        with self.assertRaisesRegex(ValueError, "hicache-io-backend"):
            TpuSRTPlatform().validate_mamba_prefix_cache(view)
        view.hicache_io_backend = "raiden"  # the K3 host tier's engine
        TpuSRTPlatform().validate_mamba_prefix_cache(view)
        view.enable_int8_mamba_checkpoint = True  # still refused
        with self.assertRaisesRegex(ValueError, "int8-mamba-checkpoint"):
            TpuSRTPlatform().validate_mamba_prefix_cache(view)


class _StubPrefill:
    def __init__(self, order):
        self.order = order

    def step(self, step_input):
        self.order.append("step")
        return StepOutput(torch.full((len(step_input.requests), 1), 50, dtype=torch.int32))


def _req(rid, prompt, pool_idx, slot):
    return SimpleNamespace(
        rid=rid, grammar=None, is_retracted=False, finished=lambda: False,
        origin_input_ids=list(range(prompt)), output_ids=[],
        sampling_params=SimpleNamespace(temperature=1.0, top_p=1.0, top_k=1, sampling_seed=None),
        kv=SimpleNamespace(kv_committed_len=0, req_pool_idx=pool_idx, mamba_pool_idx=torch.tensor([slot])),
    )


class TestWorkerWaitsForReloads(CustomTestCase):
    def test_reload_lands_before_the_state_copy_and_the_step(self):
        order = []
        counter = BatchDoneCounter()
        counter.set_event(0, SimpleNamespace(synchronize=lambda: order.append("wait")))
        plan = bridge.UnifiedPoolPlan(40, 4, sglang_pools.cache_layout("auto"))
        r2t = torch.zeros((3, 4 * STEP), dtype=torch.int32)
        r2t[1, :STEP] = torch.arange(STEP, 2 * STEP)
        state_pool = SimpleNamespace(copy_from=lambda src, dst: order.append("copy"))
        target = SimpleNamespace(
            model_runner=None,
            hicache_layer_transfer_counter=counter,
            get_memory_pool=lambda: (
                SimpleNamespace(req_to_token=r2t, mamba_pool=state_pool), None
            ),
        )
        worker.K3ProgramWorker.build_steps = lambda target_worker: worker.PrefillSteps(
            _StubPrefill(order), plan, table_width=4
        )
        self.addCleanup(setattr, worker.K3ProgramWorker, "build_steps", None)
        w = worker.K3ProgramWorker(None, 0, 0, target)
        w.alloc_memory_pool()
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            reqs=[_req("a", prompt=100, pool_idx=1, slot=1)],
            prefix_lens=[0],
            extend_lens=[100],
            seq_lens=torch.tensor([100]),
            return_logprob=False,
            hicache_consumer_index=0,
            mamba_cow_src_indices=torch.tensor([2]),
            mamba_cow_dst_indices=torch.tensor([1]),
        )
        w.forward_batch_generation(batch)
        self.assertEqual(order, ["wait", "copy", "step"])
        self.assertEqual(counter.consumer_index, 0)
        batch.hicache_consumer_index = -1  # nothing loaded: no wait
        order.clear()
        w.forward_batch_generation(batch)
        self.assertEqual(order, ["copy", "step"])


if __name__ == "__main__":
    unittest.main()
