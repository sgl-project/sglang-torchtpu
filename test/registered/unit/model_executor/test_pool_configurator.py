"""CPU-only tests for pool_configurator.py: available_bytes -> MemoryPoolConfig."""

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.runtime_context import (
    get_memory,
    get_parallel,
    get_schedule,
    get_server_args,
    get_spec,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


@contextlib.contextmanager
def mock_cpu_env(kv_size=2, tp_size=1, swa_eviction_interval=4):
    """Mock GPU-dependent state; swa_eviction_interval is kept small so the
    chunk-cap formula stays hand-computable."""
    from sglang.srt.environ import envs

    with (
        patch("torch._utils._element_size", return_value=kv_size),
        get_parallel().override(
            tp_size=tp_size,
            attn_tp_size=tp_size,
            attn_dp_size=1,
            attn_cp_size=1,
            moe_ep_size=1,
            moe_ep_group=None,
            moe_dp_size=1,
            moe_tp_size=tp_size,
        ),
        envs.SGLANG_SWA_EVICTION_INTERVAL.override(swa_eviction_interval),
    ):
        yield


def _publish_config(testcase, **fields):
    """Restored on the case's cleanup, so a failed case cannot leak a partial
    publish into a later file of a monolithic run."""
    from sglang.srt.runtime_context import get_context

    override = get_context().override_server_args(**fields)
    override.install()
    testcase.addCleanup(override.restore)


def _make_model_runner(
    testcase,
    *,
    num_kv_heads=4,
    head_dim=64,
    v_head_dim=64,
    num_layers=32,
    use_mla_backend=False,
    is_hybrid_swa=False,
    full_attention_layer_ids=None,
    swa_attention_layer_ids=None,
    swa_num_kv_heads=None,
    swa_head_dim=None,
    swa_v_head_dim=None,
    swa_full_tokens_ratio=0.5,
    page_size=1,
    mambaish_config=None,
    disable_radix_cache=False,
    chunked_prefill_size=None,
    disable_overlap_schedule=False,
    sliding_window_size=None,
    speculative_num_draft_tokens=None,
    speculative_algorithm=None,
    speculative_num_steps=None,
    speculative_eagle_topk=None,
    disaggregation_mode="null",
    max_running_requests=None,
    pp_size=1,
    pp_async_batch_depth=0,
    disaggregation_decode_extra_slots=0,
    enable_unified_memory=False,
    kv_lora_rank=512,
    qk_rope_head_dim=64,
    swa_kv_lora_rank=128,
    swa_qk_rope_head_dim=32,
):
    """Create a mock ModelRunner with the fields configurators need."""
    mr = MagicMock()

    mr.use_mla_backend = use_mla_backend
    mr.is_draft_worker = False
    mr.num_effective_layers = num_layers
    mr.start_layer = 0
    mr.end_layer = num_layers
    mr.dp_size = 1
    mr.page_size = page_size
    mr.pool_page_size = page_size
    mr.kv_full_layout = None
    mr.mambaish_config = mambaish_config
    mr.is_hybrid_swa = is_hybrid_swa
    mr.sliding_window_size = sliding_window_size

    mc = SimpleNamespace()
    mc.head_dim = head_dim
    mc.v_head_dim = v_head_dim
    mc.kv_lora_rank = kv_lora_rank
    mc.qk_rope_head_dim = qk_rope_head_dim
    mc.swa_kv_lora_rank = swa_kv_lora_rank
    mc.swa_qk_rope_head_dim = swa_qk_rope_head_dim
    mc.attention_arch = AttentionArch.MLA if use_mla_backend else AttentionArch.MHA
    mc.is_hybrid_swa = is_hybrid_swa
    mc.full_attention_layer_ids = (
        full_attention_layer_ids
        if full_attention_layer_ids is not None
        else list(range(num_layers))
    )
    mc.swa_attention_layer_ids = (
        swa_attention_layer_ids if swa_attention_layer_ids is not None else []
    )
    mc.swa_head_dim = swa_head_dim or head_dim
    mc.swa_v_head_dim = swa_v_head_dim or v_head_dim
    mc.get_num_kv_heads = lambda tp_size, dcp_size=1: num_kv_heads
    mc.get_swa_num_kv_heads = lambda tp_size, dcp_size=1: (
        swa_num_kv_heads or num_kv_heads
    )
    mc.hf_config = SimpleNamespace(architectures=["LlamaForCausalLM"])
    mc.hf_config.model_type = "llama"
    mc.hf_config.get_text_config = lambda: mc.hf_config
    mc.linear_attn_registry_result = None
    mc.context_len = 8192
    mc.is_draft_model = False
    mr.model_config = mc
    mr.kv_cache_dtype = "fake_bf16"
    mr.kv_cache_dtype_str = "auto"
    mr.speculative_draft_attention_backend = None

    # The configurator reads the published bags; mr.server_args still backs its
    # whole-object hand-offs.
    _publish_config(
        testcase,
        max_total_tokens=None,
        swa_full_tokens_ratio=swa_full_tokens_ratio,
        page_size=page_size,
        disable_radix_cache=disable_radix_cache,
        chunked_prefill_size=chunked_prefill_size,
        disable_overlap_schedule=disable_overlap_schedule,
        speculative_num_draft_tokens=speculative_num_draft_tokens,
        speculative_algorithm=speculative_algorithm,
        speculative_num_steps=speculative_num_steps,
        speculative_eagle_topk=speculative_eagle_topk,
        disaggregation_mode=disaggregation_mode,
        max_running_requests=max_running_requests,
        pp_size=pp_size,
        pp_async_batch_depth=pp_async_batch_depth,
        disaggregation_decode_extra_slots=disaggregation_decode_extra_slots,
        enable_unified_memory=enable_unified_memory,
        enable_hisparse=False,
        enable_hierarchical_cache=False,
        enable_dsa_cache_layer_split=False,
        kv_cache_dtype="auto",
    )
    mr.server_args = get_server_args()

    spec = MagicMock()
    spec.is_eagle.return_value = False
    spec.is_standalone.return_value = False
    spec.is_dflash.return_value = False
    spec.is_dflash_family.return_value = False
    spec.is_none.return_value = True
    mr.spec_algorithm = spec

    # Single stage: the runner's local hybrid-SWA view is the whole model's split.
    mr.layer_info = SimpleNamespace(
        start_layer=0,
        end_layer=num_layers,
        num_effective_layers=num_layers,
        swa_attention_layer_ids=list(mc.swa_attention_layer_ids),
        full_attention_layer_ids=list(mc.full_attention_layer_ids),
        is_hybrid_swa_mtp_draft=False,
    )
    mr.attn_dp_size = 1
    mr.pp_size = 1
    mr.pp_group = SimpleNamespace(rank_in_group=0)
    mr.spec_aux_config = SimpleNamespace(
        eagle_draft_num_layers=None,
        eagle_draft_swa_num_layers=None,
        dflash_draft_num_layers=None,
    )

    return mr


def _configure_dsa_model(model_runner):
    hf_config = model_runner.model_config.hf_config
    hf_config.architectures = ["GlmMoeDsaForCausalLM"]
    hf_config.index_topk = 2048
    hf_config.index_head_dim = 128


KV_SIZE = 2  # bf16


def _full_per_token(mr):
    mc = mr.model_config
    return mc.get_num_kv_heads(1) * (mc.head_dim + mc.v_head_dim) * KV_SIZE


def _swa_per_token(mr):
    mc = mr.model_config
    return mc.get_swa_num_kv_heads(1) * (mc.swa_head_dim + mc.swa_v_head_dim) * KV_SIZE


def _actual_memory_used(mr, config):
    """Compute actual memory consumed by the pool sizes in config."""
    mc = mr.model_config
    full_pt = _full_per_token(mr)
    swa_pt = _swa_per_token(mr)
    nf = len(mc.full_attention_layer_ids)
    ns = len(mc.swa_attention_layer_ids)

    if mr.is_hybrid_swa:
        full = config.full_max_total_num_tokens or 0
        swa = config.swa_max_total_num_tokens or 0
        return full * full_pt * nf + swa * swa_pt * ns
    else:
        return config.max_total_num_tokens * full_pt * (nf + ns)


class TestDefaultConfigurator(CustomTestCase):
    """Default (MHA): available_bytes -> tokens, memory invariant holds."""

    def _run(self, available_bytes, page_size=1, **kwargs):
        mr = _make_model_runner(self, page_size=page_size, **kwargs)
        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
            config = cfg.calculate_pool_sizes(available_bytes, page_size)
        return mr, cfg, config

    def test_memory_utilization(self):
        """Memory used should be <= available and within 1% of available."""
        available = 10_000_000
        mr, cfg, config = self._run(available)
        used = _actual_memory_used(mr, config)
        self.assertLessEqual(used, available)
        self.assertGreater(used, available * 0.99)

    def test_page_alignment(self):
        available = 10_000_000
        _, _, config = self._run(available, page_size=128)
        self.assertEqual(config.max_total_num_tokens % 128, 0)

    def test_constraint_respected(self):
        """calculate_pool_sizes_from_max_tokens respects the limit."""
        mr, cfg, config = self._run(10_000_000)
        with mock_cpu_env():
            constrained = cfg.calculate_pool_sizes_from_max_tokens(100, page_size=1)
        self.assertEqual(constrained.max_total_num_tokens, 100)

    def test_constraint_page_aligned(self):
        mr, cfg, _ = self._run(10_000_000, page_size=128)
        with mock_cpu_env():
            constrained = cfg.calculate_pool_sizes_from_max_tokens(1000, page_size=128)
        self.assertEqual(constrained.max_total_num_tokens, 896)  # 1000 // 128 * 128

    def test_no_swa_fields(self):
        _, _, config = self._run(10_000_000)
        self.assertIsNone(config.full_max_total_num_tokens)
        self.assertIsNone(config.swa_max_total_num_tokens)

    def test_padding_platform_budget_covers_the_padded_pool(self):
        """Bug regression: sizing priced heads * (head_dim + v_head_dim) while
        the built pool padded, so the allocation outgrew the HBM budget."""
        available = 10_000_000
        pad_to = 128

        def padded_bytes(*, num_kv_heads, head_dim, v_head_dim, kv_cache_dtype):
            del kv_cache_dtype
            del v_head_dim  # the fused layout gives V a slot of K's padded width
            return num_kv_heads * 2 * max(head_dim, pad_to) * KV_SIZE

        with patch(
            "sglang.srt.model_executor.pool_configurator.current_platform",
            SimpleNamespace(get_mha_kv_bytes_per_token=padded_bytes),
        ):
            mr, _, config = self._run(available)

        mc = mr.model_config
        used = (
            config.max_total_num_tokens
            * mr.layer_info.num_effective_layers
            * padded_bytes(
                num_kv_heads=mc.get_num_kv_heads(1),
                head_dim=mc.head_dim,
                v_head_dim=mc.v_head_dim,
                kv_cache_dtype=None,
            )
        )
        self.assertLessEqual(used, available)
        self.assertGreater(used, available * 0.99)

    @patch(
        "sglang.srt.mem_cache.kv_cache_configurator.calculate_mla_kv_cache_dim",
        side_effect=(576, 656),
    )
    def test_dsa_mla_cell_size_uses_backend_kv_layout(
        self,
        mock_calculate_mla_kv_cache_dim,
    ):
        num_layers = 2
        raw = _make_model_runner(
            self,
            num_layers=num_layers,
            use_mla_backend=True,
        )
        packed = _make_model_runner(
            self,
            num_layers=num_layers,
            use_mla_backend=True,
        )
        _configure_dsa_model(raw)
        _configure_dsa_model(packed)

        with mock_cpu_env(kv_size=1):
            from sglang.srt.model_executor.pool_configurator import (
                DefaultPoolConfigurator,
            )

            raw_configurator = DefaultPoolConfigurator(raw)
            packed_configurator = DefaultPoolConfigurator(packed)

        # The DSA indexer adds 128 FP8 values and one FP32 scale (4 bytes).
        self.assertEqual(raw_configurator._cell_size, (576 + 132) * num_layers)
        self.assertEqual(packed_configurator._cell_size, (656 + 132) * num_layers)
        self.assertEqual(mock_calculate_mla_kv_cache_dim.call_count, 2)


class TestHybridSWAConfigurator(CustomTestCase):
    """Hybrid SWA: full/swa split, ratio, memory invariant."""

    def _make_swa_runner(
        self,
        full_layers=16,
        swa_layers=16,
        ratio=0.5,
        page_size=1,
        enable_unified_memory=False,
    ):
        return _make_model_runner(
            self,
            is_hybrid_swa=True,
            full_attention_layer_ids=list(range(full_layers)),
            swa_attention_layer_ids=list(range(full_layers, full_layers + swa_layers)),
            swa_num_kv_heads=4,
            page_size=page_size,
            swa_full_tokens_ratio=ratio,
            enable_unified_memory=enable_unified_memory,
        )

    def _run(self, available_bytes, **kwargs):
        mr = self._make_swa_runner(**kwargs)
        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
            config = cfg.calculate_pool_sizes(available_bytes, get_schedule().page_size)
        return mr, cfg, config

    def test_memory_utilization(self):
        """Memory used should be <= available and within 1% of available."""
        available = 10_000_000
        mr, _, config = self._run(available)
        used = _actual_memory_used(mr, config)
        self.assertLessEqual(used, available)
        self.assertGreater(used, available * 0.99)

    def test_dcp_leaves_non_tpu_geometry_unchanged(self):
        """Only TPU widens the hybrid-SWA pool under DCP; another platform keeps
        its per-rank KV heads and the physical page of the SWA admission floor."""
        from sglang.srt.model_executor import pool_configurator

        def geometry(dcp_size):
            mr = self._make_swa_runner(page_size=16)
            mc = mr.model_config
            mc.get_num_kv_heads = lambda tp_size, dcp_size=1: 8 // (tp_size // dcp_size)
            mc.get_swa_num_kv_heads = mc.get_num_kv_heads
            with (
                mock_cpu_env(tp_size=8),
                get_parallel().override(attn_dcp_size=dcp_size),
                patch.object(
                    pool_configurator.current_platform,
                    "dcp_shards_every_kv_pool",
                    return_value=False,
                ),
                patch.object(
                    pool_configurator.MemoryPoolConfigurator, "validate_swa_pool_size"
                ) as floor,
            ):
                cfg = pool_configurator.create_memory_pool_configurator(mr)
                cfg.calculate_pool_sizes(10_000_000, 16)
            return cfg._full_per_token, cfg._swa_per_token, floor.call_args

        self.assertEqual(geometry(2), geometry(1))

    def test_draft_does_not_inherit_target_shared_byte_budget(self):
        """A separate draft pool must not allocate the target's byte envelope again."""
        from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator

        mr, _, config = self._run(1 << 20, enable_unified_memory=True)
        self.assertIsNotNone(config.unified_memory_pool_bytes)
        configurator = object.__new__(KVCacheConfigurator)
        configurator.device = "cpu"
        configurator.model_config = mr.model_config
        configurator.is_hybrid_swa = True
        configurator.is_draft_worker = False
        target = configurator._derive_pool_sizes(config=config)
        configurator.is_draft_worker = True
        draft = configurator._derive_pool_sizes(config=config)
        self.assertEqual(
            target.unified_memory_pool_bytes, config.unified_memory_pool_bytes
        )
        self.assertIsNone(draft.unified_memory_pool_bytes)
        self.assertEqual(
            draft.full_max_total_num_tokens, config.full_max_total_num_tokens
        )
        self.assertEqual(
            draft.swa_max_total_num_tokens, config.swa_max_total_num_tokens
        )

    def test_unified_capacity_is_maximal_with_draft_pool(self):
        page_size = 8
        full_layers = 2
        swa_layers = 1
        draft_layers = 2
        draft_swa_layers = 1
        ratio = 0.5
        mr = _make_model_runner(
            self,
            is_hybrid_swa=True,
            full_attention_layer_ids=list(range(full_layers)),
            swa_attention_layer_ids=list(range(full_layers, full_layers + swa_layers)),
            swa_num_kv_heads=4,
            swa_full_tokens_ratio=ratio,
            page_size=page_size,
            enable_unified_memory=True,
            speculative_algorithm="EAGLE",
        )
        mr.spec_algorithm.is_eagle.return_value = True
        mr.spec_algorithm.is_none.return_value = False
        mr.spec_aux_config.eagle_draft_num_layers = draft_layers
        mr.spec_aux_config.eagle_draft_swa_num_layers = draft_swa_layers

        full_bytes_per_token = _full_per_token(mr)
        swa_bytes_per_token = _swa_per_token(mr)
        target_full_bytes_per_token = full_bytes_per_token * full_layers
        draft_bytes_per_token = (
            full_bytes_per_token * (draft_layers - draft_swa_layers)
            + swa_bytes_per_token * draft_swa_layers
        )

        def allocation_bytes(full_tokens, *, include_reserved_draft_page=True):
            swa_tokens = int(full_tokens * ratio) // page_size * page_size
            target_bytes = (
                full_tokens * target_full_bytes_per_token
                + swa_tokens * swa_bytes_per_token * swa_layers
            )
            virtual_span = max(target_bytes // target_full_bytes_per_token - 1, 0)
            draft_tokens = (virtual_span + page_size - 1) // page_size * page_size
            if include_reserved_draft_page:
                draft_tokens += page_size
            return target_bytes + draft_tokens * draft_bytes_per_token

        expected_full_tokens = 10 * page_size
        available = allocation_bytes(
            expected_full_tokens + page_size,
            include_reserved_draft_page=False,
        )
        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
            config = cfg.calculate_pool_sizes(available, page_size)

        full_tokens = config.full_max_total_num_tokens
        self.assertEqual(full_tokens % page_size, 0)
        self.assertEqual(full_tokens, expected_full_tokens)
        self.assertLessEqual(allocation_bytes(full_tokens), available)
        self.assertGreater(allocation_bytes(full_tokens + page_size), available)

    @patch(
        "sglang.srt.mem_cache.kv_cache_configurator.calculate_mla_kv_cache_dim",
        return_value=576,
    )
    def test_mla_uses_full_and_swa_latent_geometry(
        self,
        mock_calculate_mla_kv_cache_dim,
    ):
        """Hybrid MLA pools must not be sized from MHA head geometry."""
        available = 10_000_000
        full_layers = 2
        swa_layers = 3
        swa_kv_lora_rank = 128
        swa_qk_rope_head_dim = 32
        mr = _make_model_runner(
            self,
            num_kv_heads=32,
            head_dim=256,
            v_head_dim=256,
            use_mla_backend=True,
            is_hybrid_swa=True,
            full_attention_layer_ids=list(range(full_layers)),
            swa_attention_layer_ids=list(range(full_layers, full_layers + swa_layers)),
            swa_num_kv_heads=16,
            swa_head_dim=128,
            swa_v_head_dim=128,
            swa_kv_lora_rank=swa_kv_lora_rank,
            swa_qk_rope_head_dim=swa_qk_rope_head_dim,
            swa_full_tokens_ratio=0.5,
        )

        with mock_cpu_env(kv_size=2):
            from sglang.srt.model_executor.pool_configurator import (
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
            config = cfg.calculate_pool_sizes(available, page_size=1)

        expected_full_per_token = 576 * 2
        expected_swa_per_token = (swa_kv_lora_rank + swa_qk_rope_head_dim) * 2
        self.assertEqual(cfg._full_per_token, expected_full_per_token)
        self.assertEqual(cfg._swa_per_token, expected_swa_per_token)
        mock_calculate_mla_kv_cache_dim.assert_called_once()

        used = (
            config.full_max_total_num_tokens * expected_full_per_token * full_layers
            + config.swa_max_total_num_tokens * expected_swa_per_token * swa_layers
        )
        self.assertLessEqual(used, available)
        self.assertGreater(used, available * 0.99)

    def test_ratio_respected(self):
        """swa_tokens ~= full_tokens * ratio (within page alignment)"""
        available = 10_000_000
        for ratio in [0.25, 0.5, 0.75, 1.0]:
            mr, _, config = self._run(available, ratio=ratio, page_size=1)
            full = config.full_max_total_num_tokens
            swa = config.swa_max_total_num_tokens
            self.assertEqual(swa, int(full * ratio), f"ratio={ratio}")

    def test_ratio_with_page_alignment(self):
        """With page alignment, swa_tokens = align(full_tokens * ratio)"""
        available = 10_000_000
        mr, _, config = self._run(available, ratio=0.5, page_size=128)
        full = config.full_max_total_num_tokens
        swa = config.swa_max_total_num_tokens
        self.assertEqual(full % 128, 0)
        self.assertEqual(swa % 128, 0)
        self.assertEqual(swa, (int(full * 0.5) // 128) * 128)

    def test_max_total_equals_full(self):
        """For hybrid, max_total_num_tokens = full_max_total_num_tokens"""
        _, _, config = self._run(10_000_000)
        self.assertEqual(config.max_total_num_tokens, config.full_max_total_num_tokens)

    def test_constraint_respected(self):
        """full_tokens = constrained value after re-run"""
        mr, cfg, _ = self._run(10_000_000, page_size=1)
        with mock_cpu_env():
            config = cfg.calculate_pool_sizes_from_max_tokens(200, page_size=1)
        self.assertEqual(config.full_max_total_num_tokens, 200)
        self.assertEqual(config.swa_max_total_num_tokens, 100)

    def test_constraint_memory_within_budget(self):
        """After constraint, memory <= original budget (but less than profiled due to constraint)."""
        available = 10_000_000
        mr, cfg, original = self._run(available, page_size=1)
        user_limit = original.full_max_total_num_tokens // 2
        with mock_cpu_env():
            config = cfg.calculate_pool_sizes_from_max_tokens(
                user_limit, get_schedule().page_size
            )
        used = _actual_memory_used(mr, config)
        self.assertLessEqual(used, available)
        # constrained should use roughly half the memory
        original_used = _actual_memory_used(mr, original)
        self.assertAlmostEqual(used / original_used, 0.5, delta=0.01)

    def test_different_layer_counts(self):
        """Asymmetric full/swa layer counts"""
        available = 10_000_000
        mr, _, config = self._run(available, full_layers=24, swa_layers=8, ratio=0.5)
        used = _actual_memory_used(mr, config)
        self.assertLessEqual(used, available)
        self.assertEqual(
            config.swa_max_total_num_tokens,
            int(config.full_max_total_num_tokens * 0.5),
        )

    def test_padding_platform_budget_covers_the_padded_pool(self):
        """Bug regression: sizing priced heads * (head_dim + v_head_dim) while
        the built pool padded, so the allocation outgrew the HBM budget."""
        available = 10_000_000
        pad_to = 128

        def padded_bytes(*, num_kv_heads, head_dim, v_head_dim, kv_cache_dtype):
            del kv_cache_dtype
            del v_head_dim  # the fused layout gives V a slot of K's padded width
            return num_kv_heads * 2 * max(head_dim, pad_to) * KV_SIZE

        mr = self._make_swa_runner(page_size=1)
        with (
            mock_cpu_env(),
            patch(
                "sglang.srt.model_executor.pool_configurator.current_platform",
                SimpleNamespace(
                    get_mha_kv_bytes_per_token=padded_bytes,
                    dcp_shards_every_kv_pool=lambda: False,
                ),
            ),
        ):
            from sglang.srt.model_executor.pool_configurator import (
                create_memory_pool_configurator,
            )

            config = create_memory_pool_configurator(mr).calculate_pool_sizes(
                available, get_schedule().page_size
            )

        mc = mr.model_config
        used = config.full_max_total_num_tokens * len(
            mc.full_attention_layer_ids
        ) * padded_bytes(
            num_kv_heads=mc.get_num_kv_heads(1),
            head_dim=mc.head_dim,
            v_head_dim=mc.v_head_dim,
            kv_cache_dtype=None,
        ) + config.swa_max_total_num_tokens * len(
            mc.swa_attention_layer_ids
        ) * padded_bytes(
            num_kv_heads=mc.get_swa_num_kv_heads(1),
            head_dim=mc.swa_head_dim,
            v_head_dim=mc.swa_v_head_dim,
            kv_cache_dtype=None,
        )
        self.assertLessEqual(used, available)
        self.assertGreater(used, available * 0.99)

    def test_chunk_cache_cap_accounts_for_spec_topk_page_rounding(self):
        available = 1_000_000
        mr = _make_model_runner(
            self,
            is_hybrid_swa=True,
            full_attention_layer_ids=[0],
            swa_attention_layer_ids=[1],
            swa_num_kv_heads=4,
            swa_full_tokens_ratio=0.5,
            disable_radix_cache=True,
            chunked_prefill_size=4,
            sliding_window_size=8,
            page_size=4,
            max_running_requests=2,
            speculative_algorithm="EAGLE",
            speculative_num_steps=3,
            speculative_eagle_topk=2,
            speculative_num_draft_tokens=5,
            disable_overlap_schedule=True,  # spec-v1: no double allocation
        )
        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
            config = cfg.calculate_pool_sizes(available, page_size=4)

        # spec-v1 (overlap off): decode_alloc = max(ceil_align(3+4,4)*2,
        # ceil_align(5,4)) = 16. trailing = 8 + 20 + page(4) = 32; per req =
        # 32 + 16 = 48. Global prefill = 1*chunk(4) + page(4) = 8.
        # cap = 48 * 2 + 8 = 104 -> ceil_align(104, 4) = 104.
        self.assertEqual(config.swa_max_total_num_tokens, 104)
        self.assertLessEqual(_actual_memory_used(mr, config), available)

    def test_chunk_cache_cap_doubles_decode_alloc_for_spec_v2_overlap(self):
        # Overlap on -> spec-v2: decode_alloc = 2 * get_alloc_len_per_decode =
        # 2 * max(steps*topk, max_draft) = 2 * max(6, 5) = 12 (page=1, since the
        # v2 allocator does not support page>1 & topk>1). trailing = 8 + 20 +
        # page(1) = 29; per req = 29 + 12 = 41. Global prefill =
        # 2*chunk(4) + page(1) = 9; cap = 41 * 2 + 9 = 91.
        available = 1_000_000
        mr = _make_model_runner(
            self,
            is_hybrid_swa=True,
            full_attention_layer_ids=[0],
            swa_attention_layer_ids=[1],
            swa_num_kv_heads=4,
            swa_full_tokens_ratio=0.5,
            disable_radix_cache=True,
            chunked_prefill_size=4,
            sliding_window_size=8,
            page_size=1,
            max_running_requests=2,
            speculative_algorithm="EAGLE",
            speculative_num_steps=3,
            speculative_eagle_topk=2,
            speculative_num_draft_tokens=5,
            disable_overlap_schedule=False,  # spec-v2: 2 * get_alloc_len_per_decode
        )
        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
            config = cfg.calculate_pool_sizes(available, page_size=1)

        self.assertEqual(config.swa_max_total_num_tokens, 91)
        self.assertLessEqual(_actual_memory_used(mr, config), available)

    def test_chunk_cache_cap_accounts_for_draft_swa_layers(self):
        """Draft SWA tensors consume the same fixed-capacity pool as target SWA."""
        available = 1_000_000
        mr = _make_model_runner(
            self,
            is_hybrid_swa=True,
            full_attention_layer_ids=[0],
            swa_attention_layer_ids=[1],
            swa_num_kv_heads=4,
            disable_radix_cache=True,
            chunked_prefill_size=4,
            sliding_window_size=8,
            page_size=1,
            max_running_requests=2,
        )
        mr.spec_algorithm.is_eagle.return_value = True
        mr.spec_algorithm.is_none.return_value = False
        mr.spec_aux_config.eagle_draft_num_layers = 1
        mr.spec_aux_config.eagle_draft_swa_num_layers = 1

        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
            config = cfg.calculate_pool_sizes(available, page_size=1)

        full_tokens = config.full_max_total_num_tokens
        swa_tokens = config.swa_max_total_num_tokens
        used = full_tokens * _full_per_token(mr) + swa_tokens * _swa_per_token(mr) * 2
        self.assertLessEqual(used, available)
        self.assertGreater(used, available * 0.99)

    def test_chunk_cache_cap_drops_prefill_for_disagg_decode(self):
        available = 1_000_000
        mr = _make_model_runner(
            self,
            is_hybrid_swa=True,
            full_attention_layer_ids=[0],
            swa_attention_layer_ids=[1],
            swa_num_kv_heads=4,
            swa_full_tokens_ratio=0.5,
            disable_radix_cache=True,
            chunked_prefill_size=1000,
            sliding_window_size=4,
            page_size=1,
            max_running_requests=10,
            disaggregation_mode="decode",
        )
        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
            config = cfg.calculate_pool_sizes(available, page_size=1)

        # disagg decode drops the prefill term: per req = 4 + 1 + 4 + 1 = 10.
        self.assertEqual(config.swa_max_total_num_tokens, 100)
        self.assertLessEqual(_actual_memory_used(mr, config), available)

    def test_chunk_cache_cap_prefill_holds_window_plus_chunk(self):
        available = 1_000_000
        mr = _make_model_runner(
            self,
            is_hybrid_swa=True,
            full_attention_layer_ids=[0],
            swa_attention_layer_ids=[1],
            swa_num_kv_heads=4,
            swa_full_tokens_ratio=0.5,
            disable_radix_cache=True,
            chunked_prefill_size=16,
            sliding_window_size=8,
            page_size=4,
            max_running_requests=2,
            disaggregation_mode="prefill",
            disable_overlap_schedule=False,  # overlap -> 2 chunks in flight
        )
        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
            config = cfg.calculate_pool_sizes(available, page_size=4)

        # per req = trailing(window(8) + eviction(4) + page(4)) + decode_alloc(4)
        # = 20. Global prefill = 2*chunk(16) + page(4) = 36.
        # cap = 20 * max_running_requests(2) + 36 = 76.
        self.assertEqual(config.swa_max_total_num_tokens, 76)
        self.assertLessEqual(_actual_memory_used(mr, config), available)

    def test_chunk_cache_cap_holds_a_chunk_per_pipeline_slot(self):
        # Under pipeline parallelism a stage's scheduler has one micro-batch per
        # pipeline slot (pp_size + async depth) in flight, each holding a
        # prefill chunk in the SWA pool until it leaves the pipeline; with the
        # single-engine budget of one or two chunks, eight stages admitted only
        # three 16k prefills at a time (gpt-oss-120b PP8, 16k prompts).
        available = 1_000_000
        mr = _make_model_runner(
            self,
            is_hybrid_swa=True,
            full_attention_layer_ids=[0],
            swa_attention_layer_ids=[1],
            swa_num_kv_heads=4,
            swa_full_tokens_ratio=0.5,
            disable_radix_cache=True,
            chunked_prefill_size=16,
            sliding_window_size=8,
            page_size=4,
            max_running_requests=2,
            disaggregation_mode="prefill",
            disable_overlap_schedule=True,  # one chunk per slot
            pp_size=8,
            pp_async_batch_depth=1,
        )
        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
            config = cfg.calculate_pool_sizes(available, page_size=4)

        # per req = 20 as above. Global prefill = 9 slots * chunk(16) + page(4)
        # = 148. cap = 20 * 2 + 148 = 188.
        self.assertEqual(config.swa_max_total_num_tokens, 188)
        self.assertLessEqual(_actual_memory_used(mr, config), available)

    def test_ratio_pool_holds_a_chunk_per_pipeline_slot(self):
        # With the radix cache on, the SWA pool was sized by the ratio alone, so
        # the chunks a pipeline keeps in flight evicted cached prefixes' SWA
        # tails and prefix hits vanished under load (gpt-oss-120b PP8: hits at
        # c8, none at c32). One chunk per pipeline slot now comes on top of
        # the ratio share, out of the same memory budget.
        available = 1_000_000
        mr = _make_model_runner(
            self,
            is_hybrid_swa=True,
            full_attention_layer_ids=[0],
            swa_attention_layer_ids=[1],
            swa_num_kv_heads=4,
            swa_full_tokens_ratio=0.5,
            chunked_prefill_size=16,
            sliding_window_size=8,
            page_size=4,
            disable_overlap_schedule=True,  # one chunk per slot
            pp_size=8,
            pp_async_batch_depth=1,
        )
        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
            config = cfg.calculate_pool_sizes(available, page_size=4)
            constrained = cfg.calculate_pool_sizes_from_max_tokens(1000, page_size=4)

        # Ratio share = half the full pool (page aligned); pipeline share =
        # 9 slots * chunk(16) = 144.
        full = config.full_max_total_num_tokens
        self.assertEqual(config.swa_max_total_num_tokens, (full // 2) // 4 * 4 + 144)
        self.assertLessEqual(_actual_memory_used(mr, config), available)
        # An explicit token budget keeps the full pool and grows SWA instead.
        self.assertEqual(constrained.full_max_total_num_tokens, 1000)
        self.assertEqual(constrained.swa_max_total_num_tokens, 500 + 144)

    def test_chunk_cache_cap_disagg_decode_pre_alloc(self):
        # decode adds disaggregation_decode_extra_slots in-transfer slots to the
        # request count (num_reserved_decode_tokens is a full-pool concern, not SWA).
        available = 2_000_000
        mr = _make_model_runner(
            self,
            is_hybrid_swa=True,
            full_attention_layer_ids=[0],
            swa_attention_layer_ids=[1],
            swa_num_kv_heads=4,
            swa_full_tokens_ratio=0.5,
            disable_radix_cache=True,
            chunked_prefill_size=1000,
            sliding_window_size=4,
            page_size=1,
            max_running_requests=10,
            disaggregation_mode="decode",
            disaggregation_decode_extra_slots=2,
        )
        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
            config = cfg.calculate_pool_sizes(available, page_size=1)

        # active per req = 4 + 1 + 4 + 1 = 10 for the 10 running requests; the 2
        # in-transfer extra slots hold only window + page = 4 + 1 = 5 each.
        # cap = 10 * 10 + 5 * 2 = 110.
        self.assertEqual(config.swa_max_total_num_tokens, 110)
        self.assertLessEqual(_actual_memory_used(mr, config), available)


class TestAllSWAConfigurator(CustomTestCase):
    """All-SWA (full_layers=0): special case."""

    def _run(self, available_bytes, ratio=0.5, page_size=1, **kwargs):
        mr = _make_model_runner(
            self,
            is_hybrid_swa=True,
            full_attention_layer_ids=[],
            swa_attention_layer_ids=list(range(32)),
            swa_num_kv_heads=4,
            swa_full_tokens_ratio=ratio,
            page_size=page_size,
            **kwargs,
        )
        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
            config = cfg.calculate_pool_sizes(available_bytes, page_size)
        return mr, cfg, config

    def test_full_max_is_zero(self):
        _, _, config = self._run(10_000_000)
        self.assertEqual(config.full_max_total_num_tokens, 0)

    def test_max_total_equals_swa(self):
        _, _, config = self._run(10_000_000)
        self.assertEqual(config.max_total_num_tokens, config.swa_max_total_num_tokens)

    def test_memory_utilization(self):
        """Memory used should be <= available and within 1% of available."""
        available = 10_000_000
        mr, _, config = self._run(available)
        swa_pt = _swa_per_token(mr)
        ns = len(mr.model_config.swa_attention_layer_ids)
        used = config.swa_max_total_num_tokens * swa_pt * ns
        self.assertLessEqual(used, available)
        self.assertGreater(used, available * 0.99)

    def test_constraint_respected(self):
        mr, cfg, _ = self._run(10_000_000, page_size=1)
        with mock_cpu_env():
            config = cfg.calculate_pool_sizes_from_max_tokens(500, page_size=1)
        self.assertEqual(config.max_total_num_tokens, 500)
        self.assertEqual(config.swa_max_total_num_tokens, 500)


class TestEagleConfigurator(CustomTestCase):
    """EAGLE: draft KV cache must be accounted for so total allocation fits in budget."""

    def test_eagle_does_not_exceed_budget(self):
        """Total memory (target + draft KV cache) must not exceed available."""
        available = 10_000_000
        num_layers = 32
        eagle_draft_num_layers = 4

        mr = _make_model_runner(self, num_layers=num_layers)
        mr.spec_algorithm.is_eagle.return_value = True
        mr.spec_algorithm.is_standalone.return_value = False
        mr.spec_algorithm.is_none.return_value = False
        mr.spec_aux_config.eagle_draft_num_layers = eagle_draft_num_layers

        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
            config = cfg.calculate_pool_sizes(available, 1)

        full_pt = _full_per_token(mr)
        total_layers = num_layers + eagle_draft_num_layers
        used = config.max_total_num_tokens * full_pt * total_layers
        self.assertLessEqual(used, available)

    @patch(
        "sglang.srt.mem_cache.kv_cache_configurator.calculate_mla_kv_cache_dim",
        return_value=576,
    )
    def test_dsa_draft_full_indexer_cost_does_not_exceed_budget(
        self,
        _mock_calculate_mla_kv_cache_dim,
    ):
        """A sharing-enabled target must not discount the draft's full index-K."""
        available = 10_000_000
        num_layers = 78
        draft_num_layers = 1
        active_indexer_layers = 21
        indexer_bytes_per_token = 132

        mr = _make_model_runner(self, num_layers=num_layers, use_mla_backend=True)
        _configure_dsa_model(mr)
        mr.model_config.hf_config.index_topk_freq = 4
        mr.model_config.hf_config.index_skip_topk_offset = 3
        mr.spec_algorithm.is_eagle.return_value = True
        mr.spec_algorithm.is_none.return_value = False
        mr.spec_aux_config.eagle_draft_num_layers = draft_num_layers

        with mock_cpu_env(kv_size=1):
            from sglang.srt.model_executor.pool_configurator import (
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
            config = cfg.calculate_pool_sizes(available, page_size=1)

        actual_bytes_per_token = (
            576 * num_layers
            + indexer_bytes_per_token * active_indexer_layers
            + (576 + indexer_bytes_per_token) * draft_num_layers
        )
        self.assertEqual(cfg._cell_size, actual_bytes_per_token)
        self.assertLessEqual(
            config.max_total_num_tokens * actual_bytes_per_token,
            available,
        )

    def test_hybrid_swa_draft_uses_swa_geometry_and_capacity(self):
        """SWA draft layers use SWA KV geometry and capacity."""
        available = 10_000_000
        ratio = 0.25
        mr = _make_model_runner(
            self,
            num_kv_heads=8,
            head_dim=64,
            v_head_dim=64,
            num_layers=4,
            is_hybrid_swa=True,
            full_attention_layer_ids=[0, 1],
            swa_attention_layer_ids=[2, 3],
            swa_num_kv_heads=2,
            swa_head_dim=32,
            swa_v_head_dim=32,
            swa_full_tokens_ratio=ratio,
        )
        mr.spec_algorithm.is_eagle.return_value = True
        mr.spec_algorithm.is_none.return_value = False
        mr.spec_aux_config.eagle_draft_num_layers = 1
        mr.spec_aux_config.eagle_draft_swa_num_layers = 1

        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
            config = cfg.calculate_pool_sizes(available, page_size=1)

        full_tokens = config.full_max_total_num_tokens
        swa_tokens = config.swa_max_total_num_tokens
        full_pt = _full_per_token(mr)
        swa_pt = _swa_per_token(mr)
        used = full_tokens * full_pt * 2 + swa_tokens * swa_pt * 3
        self.assertLessEqual(used, available)
        self.assertGreater(used, available * 0.99)


class TestDSAIndexerAllocationPolicy(CustomTestCase):
    @patch(
        "sglang.srt.mem_cache.kv_cache_configurator.calculate_mla_kv_cache_dim",
        return_value=576,
    )
    def test_resolved_hicache_override_prices_every_indexer_layer(
        self,
        _mock_calculate_mla_kv_cache_dim,
    ):
        """Post-publish HiCache overrides must keep sizing and allocation aligned."""
        num_layers = 6
        mr = _make_model_runner(self, num_layers=num_layers, use_mla_backend=True)
        _configure_dsa_model(mr)
        mr.model_config.hf_config.index_topk_freq = 4
        mr.model_config.hf_config.index_skip_topk_offset = 3

        with (
            get_memory().override(enable_hierarchical_cache=True),
            mock_cpu_env(kv_size=1),
        ):
            from sglang.srt.model_executor.pool_configurator import (
                DefaultPoolConfigurator,
            )

            cfg = DefaultPoolConfigurator(mr)

        self.assertEqual(cfg._cell_size, (576 + 132) * num_layers)

    @patch(
        "sglang.srt.mem_cache.kv_cache_configurator.calculate_mla_kv_cache_dim",
        return_value=576,
    )
    def test_pd_prices_every_indexer_layer(
        self,
        _mock_calculate_mla_kv_cache_dim,
    ):
        """PD must retain dense index-K metadata until transports support sparsity."""
        num_layers = 6
        mr = _make_model_runner(
            self,
            num_layers=num_layers,
            use_mla_backend=True,
            disaggregation_mode="prefill",
        )
        _configure_dsa_model(mr)
        mr.model_config.hf_config.index_topk_freq = 4
        mr.model_config.hf_config.index_skip_topk_offset = 3

        with mock_cpu_env(kv_size=1):
            from sglang.srt.model_executor.pool_configurator import (
                DefaultPoolConfigurator,
            )

            cfg = DefaultPoolConfigurator(mr)

        self.assertEqual(cfg._cell_size, (576 + 132) * num_layers)


class TestFactory(CustomTestCase):
    def test_default_for_non_swa(self):
        mr = _make_model_runner(self, is_hybrid_swa=False)
        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                DefaultPoolConfigurator,
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
        self.assertIsInstance(cfg, DefaultPoolConfigurator)

    def test_swa_for_hybrid(self):
        mr = _make_model_runner(
            self,
            is_hybrid_swa=True,
            full_attention_layer_ids=list(range(16)),
            swa_attention_layer_ids=list(range(16, 32)),
            swa_num_kv_heads=4,
        )
        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                HybridSWAPoolConfigurator,
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
        self.assertIsInstance(cfg, HybridSWAPoolConfigurator)

    def test_chunk_cap_configurator_selection(self):
        # SWAChunkCapPoolConfigurator is selected only when max_running_requests is set.
        def _cfg(max_running_requests):
            mr = _make_model_runner(
                self,
                is_hybrid_swa=True,
                full_attention_layer_ids=[0],
                swa_attention_layer_ids=[1],
                swa_num_kv_heads=4,
                disable_radix_cache=True,
                chunked_prefill_size=4,
                sliding_window_size=8,
                max_running_requests=max_running_requests,
            )
            with mock_cpu_env():
                from sglang.srt.model_executor.pool_configurator import (
                    create_memory_pool_configurator,
                )

                return create_memory_pool_configurator(mr)

        from sglang.srt.model_executor.pool_configurator import (
            SWAChunkCapPoolConfigurator,
        )

        self.assertIsInstance(_cfg(2), SWAChunkCapPoolConfigurator)
        self.assertNotIsInstance(_cfg(None), SWAChunkCapPoolConfigurator)


class TestDflashDraftKvBudget(CustomTestCase):
    """DFLASH draft KV pool as a flat bytes/token term on the target's budget."""

    def test_bytes_per_token_from_draft_geometry(self):
        import torch

        from sglang.srt.speculative.dflash_utils import (
            dflash_draft_cell_size_per_token,
        )

        draft = SimpleNamespace(
            get_num_kv_heads=lambda tp: 4, head_dim=128, v_head_dim=128
        )
        # 4 kv heads * (128 + 128) dims * 5 layers * 2 bytes
        self.assertEqual(
            dflash_draft_cell_size_per_token(
                draft_model_config=draft,
                draft_num_layers=5,
                draft_kv_cache_dtype=torch.bfloat16,
                tp_size=1,
            ),
            10240,
        )
        self.assertEqual(
            dflash_draft_cell_size_per_token(
                draft_model_config=draft,
                draft_num_layers=0,
                draft_kv_cache_dtype=torch.bfloat16,
                tp_size=1,
            ),
            0,
        )

    def test_dcp_replication_scales_draft_budget(self):
        """The replicated draft pool spans every DCP virtual location."""
        draft_kv_per_token = 10_240
        mr = _make_model_runner(self)
        mr.spec_algorithm.is_dflash_family.return_value = True
        mr.spec_aux_config = SimpleNamespace(
            eagle_draft_num_layers=None,
            dflash_draft_num_layers=5,
            dflash_draft_cell_size_per_token=draft_kv_per_token,
        )

        target_kv_per_token = 4 * (64 + 64) * 32 * KV_SIZE
        for dcp_size in (1, 8):
            with self.subTest(dcp_size=dcp_size):
                # TP=8 makes both topologies valid. The mock deliberately keeps
                # target geometry fixed so this assertion isolates the draft term.
                with (
                    mock_cpu_env(tp_size=8),
                    get_parallel().override(attn_dcp_size=dcp_size),
                ):
                    from sglang.srt.model_executor.pool_configurator import (
                        create_memory_pool_configurator,
                    )

                    cfg = create_memory_pool_configurator(mr)

                self.assertEqual(
                    cfg._cell_size,
                    target_kv_per_token + draft_kv_per_token * dcp_size,
                )

    def test_hybrid_swa_budget_shrinks_by_draft_pool(self):
        """The HybridSWA budget must charge the DFLASH draft pool."""
        available = 10_000_000

        def _tokens(draft_kv_per_token):
            mr = _make_model_runner(
                self,
                is_hybrid_swa=True,
                full_attention_layer_ids=list(range(16)),
                swa_attention_layer_ids=list(range(16, 32)),
                swa_num_kv_heads=4,
            )
            mr.spec_algorithm.is_dflash_family.return_value = True
            mr.spec_aux_config = SimpleNamespace(
                eagle_draft_num_layers=None,
                dflash_draft_num_layers=5,
                dflash_draft_cell_size_per_token=draft_kv_per_token,
            )
            with mock_cpu_env():
                from sglang.srt.model_executor.pool_configurator import (
                    create_memory_pool_configurator,
                )

                cfg = create_memory_pool_configurator(mr)
                config = cfg.calculate_pool_sizes(available, get_schedule().page_size)
            return config.full_max_total_num_tokens

        self.assertLess(_tokens(10240), _tokens(None))


class TestMixedKVFormats(CustomTestCase):
    def test_mxint8_draft_layout_and_replication(self):
        from sglang.srt.model_executor.pool_configurator import (
            _dflash_draft_cell_size,
            _mxint8_draft_per_token,
        )

        mr = _make_model_runner(self)
        mr.server_args = SimpleNamespace(
            tpu_mxint8_group_size=32, tpu_mxint8_full_kv_layout="blocked"
        )
        mr.spec_algorithm.is_dflash_family.return_value = True
        mr.spec_aux_config.dflash_draft_num_layers = 2
        mr.spec_aux_config.dflash_draft_cell_size_per_token = 768
        mr.spec_aux_config.dflash_draft_kv_geometry = (3, 64)
        # H3D64 has 384 payload rows and 12 real scale rows. Blocked pages
        # reserve 64 exponent rows across 1/2/4 token blocks; merged uses 32.
        for page_size, dcp_size, expected in (
            (128, 1, 448),
            (256, 1, 416),
            (512, 1, 400),
            (128, 2, 416),
        ):
            with self.subTest(page_size=page_size, dcp_size=dcp_size):
                mr.pool_page_size = page_size
                with (
                    get_parallel().override(
                        tp_size=2,
                        attn_tp_size=2,
                        attn_dcp_size=dcp_size,
                        moe_tp_size=2,
                    ),
                    get_spec().override(speculative_draft_kv_cache_dtype="mxint8"),
                ):
                    self.assertEqual(_mxint8_draft_per_token(mr, 3, 64), expected)
                    self.assertEqual(
                        _mxint8_draft_per_token(mr, 3, 64, is_swa=True), 416
                    )
                    self.assertEqual(
                        _dflash_draft_cell_size(mr), 2 * expected * dcp_size
                    )
        mr.spec_aux_config.dflash_draft_kv_geometry = None
        with get_spec().override(speculative_draft_kv_cache_dtype="mxint8"):
            with self.assertRaisesRegex(
                ValueError, "resolved DFLASH draft KV geometry"
            ):
                _dflash_draft_cell_size(mr)

    def test_dflash_resolver_preserves_draft_geometry(self):
        from sglang.srt.model_executor.model_runner_components import (
            spec_aux_hidden_state as aux,
        )

        _make_model_runner(self)
        draft = SimpleNamespace(
            hf_config=SimpleNamespace(),
            get_num_kv_heads=MagicMock(return_value=3),
            head_dim=64,
        )
        parsed = SimpleNamespace(
            require_num_layers=lambda: 2,
            num_target_layers=8,
            resolve_target_layer_ids=lambda **_: [1, 5],
        )
        algorithm = SimpleNamespace(
            is_dflash_family=lambda: True, is_dspark=lambda: False
        )
        config = aux.SpecAuxHiddenStateConfig()
        with (
            patch.object(aux.ModelConfig, "from_server_args", return_value=draft),
            patch.object(aux, "_resolve_dflash_draft_cell_size", return_value=768),
            patch(
                "sglang.srt.speculative.dflash_utils.parse_dflash_draft_config",
                return_value=parsed,
            ),
            get_parallel().override(tp_size=2, attn_tp_size=2, moe_tp_size=2),
        ):
            aux._resolve_dflash_aux_hidden_state(
                config=config,
                server_args=SimpleNamespace(),
                model_config=SimpleNamespace(
                    hf_config=SimpleNamespace(),
                    hf_text_config=SimpleNamespace(num_hidden_layers=8),
                ),
                spec_algorithm=algorithm,
                is_draft_worker=False,
            )
        self.assertEqual(config.dflash_draft_kv_geometry, (3, 64))
        self.assertEqual(config.dflash_draft_cell_size_per_token, 768)
        draft.get_num_kv_heads.assert_called_once_with(2)

    def test_ordinary_formats_preserve_per_token_accounting(self):
        for dtype_str, dtype, per_token in (
            ("auto", torch.bfloat16, 1024),
            ("bf16", torch.bfloat16, 1024),
            ("fp8_e4m3", torch.float8_e4m3fn, 512),
            ("mxfp8", torch.float8_e4m3fn, 528),
        ):
            with self.subTest(dtype=dtype_str):
                mr = _make_model_runner(self, num_layers=2, page_size=4)
                mr.kv_cache_dtype_str = dtype_str
                mr.kv_cache_dtype = dtype
                with get_parallel().override(attn_tp_size=1):
                    from sglang.srt.model_executor.pool_configurator import (
                        create_memory_pool_configurator,
                    )

                    cfg = create_memory_pool_configurator(mr)
                    config = cfg.calculate_pool_sizes(1_000_000, 4)
                self.assertEqual(
                    config.max_total_num_tokens, (1_000_000 // (2 * per_token)) // 4 * 4
                )

    def test_mixed_formats_reserve_dummy_pages_and_align_capacity(self):
        from sglang.srt.model_executor.pool_configurator import (
            create_memory_pool_configurator,
        )
        from sglang_tpu.kv_mxint8 import cache_bytes_per_token

        available, page_size = 10_000_000, 128
        # 512 INT8 payload bytes plus 16 exponent rows aligned to 32.
        mxint8_cost = cache_bytes_per_token(4, 64, page_size, group_size=32)
        self.assertEqual(mxint8_cost, 544)
        for layout in ("default", "hybrid", "all_swa", "chunk_cap"):
            for target_format, draft_format, target_cost, draft_cost in (
                ("mxint8", "bf16", mxint8_cost, 1024),
                ("mxint8", "auto", mxint8_cost, 1024),
                ("bf16", "mxint8", 1024, mxint8_cost),
            ):
                with self.subTest(
                    layout=layout, target=target_format, draft=draft_format
                ):
                    hybrid = layout != "default"
                    chunk_cap = layout == "chunk_cap"
                    mr = _make_model_runner(
                        self,
                        num_layers=2,
                        page_size=page_size,
                        is_hybrid_swa=hybrid,
                        full_attention_layer_ids=[] if layout == "all_swa" else [0],
                        swa_attention_layer_ids=[0, 1] if layout == "all_swa" else [1],
                        disable_radix_cache=chunk_cap,
                        chunked_prefill_size=4 if chunk_cap else None,
                        sliding_window_size=8 if chunk_cap else None,
                        max_running_requests=2 if chunk_cap else None,
                    )
                    mr.kv_cache_dtype_str = target_format
                    mr.kv_cache_dtype = (
                        torch.int8 if target_format == "mxint8" else torch.bfloat16
                    )
                    mr.pool_page_size = page_size
                    mr.kv_group_size = 32
                    mr.kv_full_layout = "merged"
                    mr.kv_swa_layout = "merged"
                    mr.spec_algorithm.is_eagle.return_value = True
                    mr.spec_aux_config.eagle_draft_num_layers = 1
                    mr.spec_aux_config.eagle_draft_swa_num_layers = int(hybrid)
                    with (
                        mock_cpu_env(),
                        get_spec().override(
                            speculative_draft_kv_cache_dtype=draft_format
                        ),
                    ):
                        cfg = create_memory_pool_configurator(mr)
                        config = cfg.calculate_pool_sizes(available, page_size)
                        constrained = cfg.calculate_pool_sizes_from_max_tokens(
                            1025, page_size
                        )

                    dummy_bytes = page_size * (2 * target_cost + draft_cost)
                    budget = available - dummy_bytes
                    if layout in ("default", "all_swa"):
                        expected = (
                            budget
                            // (2 * target_cost + draft_cost)
                            // page_size
                            * page_size
                        )
                        used = (config.max_total_num_tokens + page_size) * (
                            2 * target_cost + draft_cost
                        )
                    else:
                        if chunk_cap:
                            # Per request: window 8 + eviction 4 + page 128 +
                            # decode 128. Two requests plus two chunks of 4 and
                            # one page: 672, rounded up to 768 tokens.
                            self.assertEqual(config.swa_max_total_num_tokens, 768)
                            self.assertEqual(constrained.swa_max_total_num_tokens, 768)
                            expected = (
                                (budget - 768 * (target_cost + draft_cost))
                                // target_cost
                                // page_size
                                * page_size
                            )
                        else:
                            expected = (
                                int(
                                    budget
                                    // (target_cost + 0.5 * (target_cost + draft_cost))
                                )
                                // page_size
                                * page_size
                            )
                        used = (
                            config.full_max_total_num_tokens * target_cost
                            + config.swa_max_total_num_tokens
                            * (target_cost + draft_cost)
                            + dummy_bytes
                        )
                        self.assertEqual(config.swa_max_total_num_tokens % page_size, 0)
                    self.assertEqual(config.max_total_num_tokens, expected)
                    self.assertEqual(constrained.max_total_num_tokens, 1024)
                    if layout == "hybrid":
                        self.assertEqual(constrained.swa_max_total_num_tokens, 512)
                    elif layout == "all_swa":
                        self.assertEqual(constrained.swa_max_total_num_tokens, 1024)
                        self.assertEqual(config.full_max_total_num_tokens, 0)
                    self.assertLessEqual(used, available)
                    self.assertGreater(
                        used, available - page_size * (2 * target_cost + draft_cost)
                    )

    def test_mxint8_groups_preserve_asymmetric_full_swa_geometry(self):
        from sglang.srt.model_executor.pool_configurator import (
            create_memory_pool_configurator,
        )
        from sglang_tpu.kv_mxint8 import cache_bytes_per_token

        available, page_size = 10_000_000, 128
        # Full: 2048 payload bytes plus 128/64/32 exponent rows.
        # SWA: 256 payload bytes plus one 32-row aligned exponent buffer.
        for group_size, full_cost in ((16, 2176), (32, 2112), (64, 2080)):
            with self.subTest(group_size=group_size):
                self.assertEqual(
                    cache_bytes_per_token(8, 128, page_size, group_size=group_size),
                    full_cost,
                )
                swa_cost = 288
                self.assertEqual(
                    cache_bytes_per_token(2, 64, page_size, group_size=group_size),
                    swa_cost,
                )
                mr = _make_model_runner(
                    self,
                    num_kv_heads=8,
                    head_dim=128,
                    v_head_dim=128,
                    num_layers=3,
                    page_size=page_size,
                    is_hybrid_swa=True,
                    full_attention_layer_ids=[0],
                    swa_attention_layer_ids=[1, 2],
                    swa_num_kv_heads=2,
                    swa_head_dim=64,
                    swa_v_head_dim=64,
                )
                mr.kv_cache_dtype_str = "mxint8"
                mr.kv_cache_dtype = torch.int8
                mr.pool_page_size = page_size
                mr.kv_group_size = group_size
                mr.kv_full_layout = "merged"
                mr.kv_swa_layout = "merged"
                mr.spec_algorithm.is_eagle.return_value = True
                mr.spec_aux_config.eagle_draft_num_layers = 1
                mr.spec_aux_config.eagle_draft_swa_num_layers = 1
                with (
                    mock_cpu_env(),
                    get_spec().override(speculative_draft_kv_cache_dtype="bf16"),
                ):
                    cfg = create_memory_pool_configurator(mr)
                    config = cfg.calculate_pool_sizes(available, page_size)

                draft_swa_cost = 2 * 2 * 64 * 2
                swa_cell_size = 2 * swa_cost + draft_swa_cost
                dummy_bytes = page_size * (full_cost + swa_cell_size)
                self.assertEqual(cfg._full_per_token, full_cost)
                self.assertEqual(cfg._swa_per_token, swa_cost)
                self.assertEqual(cfg._draft_swa_per_token, draft_swa_cost)
                self.assertEqual(
                    config.full_max_total_num_tokens,
                    int((available - dummy_bytes) // (full_cost + 0.5 * swa_cell_size))
                    // page_size
                    * page_size,
                )
                self.assertEqual(
                    config.swa_max_total_num_tokens,
                    (config.full_max_total_num_tokens // 2) // page_size * page_size,
                )
                used = (
                    config.full_max_total_num_tokens * full_cost
                    + config.swa_max_total_num_tokens * swa_cell_size
                    + dummy_bytes
                )
                self.assertLessEqual(used, available)
                self.assertGreater(
                    used, available - page_size * (full_cost + swa_cell_size)
                )

    def test_dflash_budget_uses_runner_local_backend(self):
        from sglang.srt.model_executor.pool_configurator import (
            create_memory_pool_configurator,
        )

        available, page_size = 1_000_000, 128
        # TPU adds 32 aligned exponent rows to the 1024-byte draft payload.
        for local_backend, global_backend, draft_cost, dummy_tokens in (
            ("fa4", "tpu", 1024, 0),
            ("tpu", "fa4", 1056, page_size),
            (None, "fa4", 1024, 0),
            (None, "tpu", 1056, page_size),
        ):
            with self.subTest(
                local_backend=local_backend, global_backend=global_backend
            ):
                mr = _make_model_runner(self, num_layers=2, page_size=page_size)
                mr.kv_cache_dtype = torch.bfloat16
                mr.kv_cache_dtype_str = "bf16"
                mr.speculative_draft_attention_backend = local_backend
                mr.spec_algorithm.is_dflash_family.return_value = True
                mr.spec_aux_config.dflash_draft_num_layers = 1
                mr.spec_aux_config.dflash_draft_cell_size_per_token = 1024
                mr.spec_aux_config.dflash_draft_kv_geometry = (4, 128)
                with (
                    mock_cpu_env(),
                    get_spec().override(
                        speculative_draft_kv_cache_dtype="mxint8",
                        speculative_draft_attention_backend=global_backend,
                    ),
                ):
                    cfg = create_memory_pool_configurator(mr)
                    config = cfg.calculate_pool_sizes(available, page_size)
                self.assertEqual(cfg._cell_size, 2048 + draft_cost)
                self.assertEqual(
                    config.max_total_num_tokens,
                    (available // (2048 + draft_cost) - dummy_tokens)
                    // page_size
                    * page_size,
                )
                self.assertLessEqual(
                    (config.max_total_num_tokens + dummy_tokens) * (2048 + draft_cost),
                    available,
                )


class TestSWAPoolFloor(CustomTestCase):
    """An SWA pool below the prefill admission floor must fail here, not livelock
    the scheduler at warmup."""

    def _hybrid_swa_from_max_tokens(self, max_tokens, ratio, page_size, window):
        mr = _make_model_runner(
            self,
            is_hybrid_swa=True,
            full_attention_layer_ids=list(range(16)),
            swa_attention_layer_ids=list(range(16, 32)),
            swa_num_kv_heads=4,
            page_size=page_size,
            swa_full_tokens_ratio=ratio,
            sliding_window_size=window,
        )
        with mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
            return cfg.calculate_pool_sizes_from_max_tokens(max_tokens, page_size)

    def test_hybrid_swa_rejects_single_page_pool(self):
        with self.assertRaisesRegex(ValueError, "cannot hold even one request"):
            self._hybrid_swa_from_max_tokens(
                max_tokens=4096, ratio=0.1, page_size=256, window=128
            )

    def test_hybrid_swa_accepts_pool_above_floor(self):
        config = self._hybrid_swa_from_max_tokens(
            max_tokens=32768, ratio=0.1, page_size=256, window=128
        )
        self.assertEqual(config.swa_max_total_num_tokens, 3072)

    def _dsv4_sizes(self, max_tokens, page_size, unified=False):
        """Exercise the DSV4 size arithmetic without a full V4 model fixture:
        _compute_dsv4_sizes reads only these attributes."""
        from sglang.srt.model_executor.pool_configurator import DSV4PoolConfigurator

        cfg = object.__new__(DSV4PoolConfigurator)
        cfg.swa_ratio = 0.1
        cfg.sliding_window_size = 128
        cfg.ring_sizes = {4: 8}
        cfg.c4_shrink_factor = 1
        cfg._unified = unified
        cfg.operator_swa_ratio = None
        cfg.swa_cap_tokens = None
        cfg.swa_prefix_tails = 0
        return cfg._compute_dsv4_sizes(max_tokens, page_size)

    def test_dsv4_rejects_single_page_pool(self):
        # DeepSeek-V4-Flash defaults (page 256, ratio 0.1): int(4096 * 0.1) aligns
        # down to one page, below the 128 + 256 floor.
        with self.assertRaisesRegex(ValueError, "cannot hold even one request"):
            self._dsv4_sizes(max_tokens=4096, page_size=256)

    def test_dsv4_accepts_pool_above_floor(self):
        sizes = self._dsv4_sizes(max_tokens=32768, page_size=256)
        self.assertEqual(sizes.full_max_total_num_tokens, 32768)
        self.assertEqual(sizes.swa_max_total_num_tokens, 3072)
        # Non-unified: the c4 state pool scales with the paged SWA pool, and
        # is sized by the page the ring is addressed by (not the window).
        self.assertEqual(sizes.c4_state_pool_size, 3072 // 256 * 8)

    def test_dsv4_token_cap_never_grows_total_footprint(self):
        """Capping the budget-derived token count must not grow the total footprint."""
        cfg = self._dsv4_configurator_for_budget()
        page_size = 128
        budget = 256 * (1 << 30)
        base = cfg.calculate_pool_sizes(budget, page_size)
        base_bytes = self._dsv4_total_bytes(cfg, base.max_total_num_tokens)
        self.assertLessEqual(base_bytes, budget)
        for numerator in (999, 900, 500, 100, 1):
            capped_tokens = (
                base.max_total_num_tokens * numerator // 1000 // page_size * page_size
            )
            if capped_tokens <= 0:
                continue
            capped = cfg.calculate_pool_sizes_from_max_tokens(capped_tokens, page_size)
            capped_bytes = self._dsv4_total_bytes(cfg, capped.max_total_num_tokens)
            with self.subTest(numerator=numerator):
                self.assertLessEqual(capped_bytes, base_bytes)

    # White-box 671B-class shape: the byte arithmetic runs without a model fixture.
    def _dsv4_configurator_for_budget(self):
        from sglang.srt.model_executor.pool_configurator import DSV4PoolConfigurator

        cfg = object.__new__(DSV4PoolConfigurator)
        cfg.qk_nope_head_dim, cfg.qk_rope_head_dim = 128, 64
        cfg.attn_head_dim = 192
        cfg.indexer_head_dim = 128
        cfg.num_layers_total = 61
        # Every layer owns both a c4 and a c128 store: an upper bound, not a model.
        cfg.stage_owner_layers = {4: list(range(61)), 128: list(range(61))}
        cfg.ring_sizes = {4: 8, 128: 128}
        cfg._swa_ring_size = 128
        cfg._spec_infl = 1.0
        cfg.context_len = 65536
        cfg.bytes_per_full_token = 576.0
        cfg.requested_max_running_requests_per_worker = None
        cfg.swa_ratio = 0.1
        cfg.sliding_window_size = 4096
        cfg.page_size = 128
        cfg.c4_shrink_factor = 1
        cfg.online_c128_mtp_max_draft_tokens = 0
        cfg.disaggregation_mode = None
        cfg.disaggregation_decode_extra_slots = 0
        cfg._unified = True
        cfg.operator_swa_ratio = None
        cfg.swa_cap_tokens = None
        cfg.swa_prefix_tails = 0
        cfg.request_window_bytes = 0
        cfg.bytes_per_swa_token = 0.0
        cfg._unified_fp8 = False
        cfg._dspark_draft_on_bf16 = False
        # object.__new__ skips __init__; bf16 unified row is 2B * latent
        cfg._unified_row_bytes = cfg.attn_head_dim * 2
        return cfg

    # Token pool plus the request-scoped fixed pools, sized from the
    # concurrency resolve_max_num_reqs derives from this token count.
    def _dsv4_total_bytes(self, cfg, tokens):
        estimated = max(min(int(tokens / cfg.context_len * 512), 4096), 2048)
        max_running_requests = min(estimated, tokens // 2)
        return int(
            tokens * cfg.bytes_per_full_token
            + cfg._fixed_swa_bytes(max_running_requests)
            + cfg._fixed_c4_state_bytes(max_running_requests)
            + cfg._get_c128_state_fixed_bytes(max_running_requests)
            + cfg._get_c2_state_fixed_bytes(max_running_requests)
        )

    def test_dsv4_c2_state_fixed_bytes_mirror_pool(self):
        """Ratio-2 kv_source layers keep an fp32 (kv, score) ring per request
        slot; the budget must charge the rows CompressStatePool allocates."""
        from sglang.srt.model_executor.pool_configurator import DSV4PoolConfigurator

        cfg = object.__new__(DSV4PoolConfigurator)
        cfg.attn_head_dim = 512
        cfg.stage_owner_layers = {2: [2, 8], 1: [20]}
        cfg.ring_sizes = {2: 2}
        cfg.disaggregation_mode = None
        cfg.disaggregation_decode_extra_slots = 0
        slots = 10 + 1
        # size + ring + 1 rows, padded to the ratio (2); 2 * head_dim fp32 per row.
        rows = (slots * 2 + 2 + 1 + 1) // 2 * 2
        self.assertEqual(cfg._get_c2_state_fixed_bytes(10), rows * 2 * 512 * 4 * 2)
        cfg.stage_owner_layers = {1: [20]}
        self.assertEqual(cfg._get_c2_state_fixed_bytes(10), 0)

    def test_dsv4_owner_layers_follow_the_pool_rule(self):
        """The configurator budgets the layers the pool allocates: every ratio-4
        / 128 layer, only the kv_source layers of ratios 1 / 2, all PP-local."""
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            collect_sources_by_ratio,
        )

        ratios = [0, 0, 2, 2, 1, 1, 4, 128, 4, 2]
        owners = collect_sources_by_ratio(ratios, [2, 4, 9], range(0, 8))
        self.assertEqual(owners, {4: [6], 128: [7], 1: [4], 2: [2]})
        owners = collect_sources_by_ratio(ratios, [2, 4, 9], range(8, 10))
        self.assertEqual(owners, {4: [8], 2: [9]})

    def test_dsv4_paged_kv_budget(self):
        from sglang.srt.environ import envs

        cfg = self._dsv4_configurator_for_budget()
        cfg._unified = False
        cfg.encoder_replay = False
        cfg.page_size = 256
        cfg.kv_bytes = 584
        cfg.attn_head_dim = 512
        cfg.num_layers_total = 3
        cfg.stage_owner_layers = {4: [1], 128: [2]}
        cfg.indexer_bytes_per_token = 132
        _publish_config(self, dsv4_attn_backend="flashmla")
        with (
            envs.SGLANG_DSV4_KV_LAYOUT.override("v4"),
            envs.SGLANG_DSV4_COMPRESS_STATE_DTYPE.override("float32"),
        ):
            cfg.bytes_per_swa_token = cfg._get_bytes_per_swa_token()
            self.assertEqual(cfg.bytes_per_swa_token, 3 * 585 + 320)
            self.assertEqual(
                cfg._get_bytes_per_full_token(),
                0.1 * cfg.bytes_per_swa_token + 585 / 4 + 864 / 128 + 132 / 4,
            )

    def test_dsv4_paged_dspark_budget_reserves_window_and_draft_layers(self):
        from sglang.srt.model_executor.pool_configurator import DSV4PoolConfigurator

        _publish_config(
            self,
            enable_encoder_swa_bounded_replay=True,
            speculative_algorithm="DSPARK",
            speculative_num_draft_tokens=6,
            speculative_dspark_block_size=5,
            page_size=256,
            max_running_requests=2,
            chunked_prefill_size=256,
        )
        cfg = SimpleNamespace(
            qk_nope_head_dim=448,
            qk_rope_head_dim=64,
            index_head_dim=128,
            context_len=131072,
            compress_ratios=[0, 0] + [2] * 18 + [1] * 20,
            window_size=128,
            hf_config=SimpleNamespace(kv_source_layer_ids=[2, 8, 14, 20]),
        )
        spec = SimpleNamespace(is_dspark=lambda: True, is_none=lambda: False)
        kvc = SimpleNamespace(
            kv_cache_dtype_str="fp8_e4m3",
            model_config=cfg,
            layer_info=SimpleNamespace(start_layer=0, end_layer=40),
            pp_size=1,
            attn_dp_size=1,
            sliding_window_size=128,
            page_size=256,
            spec_algorithm=spec,
            spec_aux_config=SimpleNamespace(dflash_draft_num_layers=3),
        )
        planner = DSV4PoolConfigurator(kvc)
        self.assertEqual(planner.bytes_per_swa_token, 3 * 584)
        budget = 256 * 1024 * 1024
        sizes = planner.calculate_pool_sizes(budget, 256)
        self.assertEqual(sizes.swa_max_total_num_tokens, planner.swa_cap_tokens)
        self.assertLessEqual(
            sizes.full_max_total_num_tokens * planner.bytes_per_full_token
            + planner._get_swa_fixed_bytes(),
            budget,
        )

    def test_dsv4_trtllm_kv_bytes(self):
        from sglang.srt.model_executor.pool_configurator import DSV4PoolConfigurator

        kvc = _make_model_runner(self, num_layers=1, page_size=256)
        _publish_config(self, dsv4_attn_backend="trtllm", page_size=256)
        kvc.kv_cache_dtype_str = "fp8_e4m3"
        cfg = kvc.model_config
        cfg.qk_nope_head_dim = 448
        cfg.index_head_dim = 128
        cfg.compress_ratios = [0]
        cfg.window_size = 128
        cfg.hf_config.kv_source_layer_ids = []
        planner = DSV4PoolConfigurator(kvc)
        self.assertEqual(planner.kv_bytes, 512)
        self.assertEqual(planner.bytes_per_swa_token, 512)

    def test_dsv4_tpu_budget_accounting(self):
        from sglang.srt.model_executor.pool_configurator import DSV4PoolConfigurator

        kvc = _make_model_runner(self, num_layers=44, page_size=256)
        _publish_config(self, dsv4_attn_backend="tpu_dsv4", page_size=256)
        kvc.kv_cache_dtype_str = "bf16"
        cfg = kvc.model_config
        cfg.qk_nope_head_dim = 448
        cfg.qk_rope_head_dim = 64
        cfg.index_head_dim = 128
        cfg.compress_ratios = [0, 0] + [128, 4] * 21
        cfg.window_size = 512
        cfg.hf_config.kv_source_layer_ids = []
        planner = DSV4PoolConfigurator(kvc)
        self.assertTrue(planner._is_tpu_dsv4)
        self.assertEqual(planner.bytes_per_swa_token, 44 * 1024.0)
        expected_full = (
            0.1 * (44 * 1024.0)
            + (1 / 4) * 640 * 21
            + (1 / 128) * 1024 * 21
            + (1 / 4) * 256 * 21
        )
        self.assertEqual(planner.bytes_per_full_token, expected_full)
        mrr = 64
        num_req_slots = mrr + 1
        state_pages = 1 + num_req_slots * 32 + 2048
        c4_rows = state_pages * 4
        self.assertEqual(
            planner._fixed_c4_state_bytes(mrr),
            c4_rows * (4 * 512 * 4 + 4 * 128 * 4) * 21,
        )
        c128_rows = state_pages * 8
        self.assertEqual(
            planner._get_c128_state_fixed_bytes(mrr),
            c128_rows * (2 * 512 * 4) * 21,
        )

    def test_dsv4_pro_tpu_budget_accounting(self):
        from sglang.srt.model_executor.pool_configurator import DSV4PoolConfigurator

        kvc = _make_model_runner(self, num_layers=61, page_size=256)
        _publish_config(self, dsv4_attn_backend="tpu_dsv4", page_size=256)
        kvc.kv_cache_dtype_str = "bf16"
        cfg = kvc.model_config
        cfg.qk_nope_head_dim = 448
        cfg.qk_rope_head_dim = 64
        cfg.index_head_dim = 128
        # DeepSeek-V4-Pro config.json has 64 entries for 61 target layers
        # (31 HCA, 30 CSA, and 3 trailing MTP 0s).
        cfg.compress_ratios = [128] + [128, 4] * 30 + [0, 0, 0]
        cfg.window_size = 128
        cfg.hf_config.kv_source_layer_ids = []
        planner = DSV4PoolConfigurator(kvc)
        self.assertTrue(planner._is_tpu_dsv4)
        self.assertEqual(planner.num_layers_total, 61)
        self.assertEqual(planner.num_layers(4), 30)
        self.assertEqual(planner.num_layers(128), 31)
        self.assertEqual(planner.bytes_per_swa_token, 61 * 1024.0)
        expected_full = (
            0.1 * (61 * 1024.0)
            + (1 / 4) * 640 * 30
            + (1 / 128) * 1024 * 31
            + (1 / 4) * 256 * 30
        )
        self.assertEqual(planner.bytes_per_full_token, expected_full)

    def test_dsv4_unified_c4_state_not_token_scaled(self):
        # Unified-KV sizes the c4 state ring from max_running_requests in
        # finalize_with_max_running_requests, so it must not scale here.
        sizes = self._dsv4_sizes(max_tokens=32768, page_size=256, unified=True)
        self.assertEqual(sizes.full_max_total_num_tokens, 32768)
        self.assertEqual(sizes.swa_max_total_num_tokens, 3072)
        self.assertEqual(sizes.c4_state_pool_size, 0)

    def test_dsv4_fp8_dspark_swa_ring_includes_bf16_draft(self):
        """Target fp8 ring + one-layer bf16 draft ring, not (T+1)/T * fp8."""
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import dsv4_unified_row_bytes

        cfg = self._dsv4_configurator_for_budget()
        cfg._unified_fp8 = True
        cfg._unified_row_bytes = dsv4_unified_row_bytes(448, 64, fp8=True)
        cfg.qk_nope_head_dim, cfg.qk_rope_head_dim = 448, 64
        cfg._dspark_draft_on_bf16 = True
        cfg._spec_infl = (cfg.num_layers_total + 1) / cfg.num_layers_total
        mrr = 32
        slots = cfg._get_num_req_slots(mrr)
        got = cfg._fixed_swa_bytes(mrr)
        target = (
            slots * cfg._swa_ring_size * cfg._unified_row_bytes * cfg.num_layers_total
        )
        draft = slots * cfg._swa_ring_size * dsv4_unified_row_bytes(448, 64, False)
        self.assertEqual(got, target + draft)
        mtp_formula = int(target * cfg._spec_infl)
        self.assertGreater(got, mtp_formula)

    def test_dsv4_fp8_mtp_swa_ring_keeps_spec_inflation(self):
        cfg = self._dsv4_configurator_for_budget()
        cfg._unified_fp8 = True
        cfg._unified_row_bytes = 640
        cfg._dspark_draft_on_bf16 = False
        cfg._spec_infl = (cfg.num_layers_total + 1) / cfg.num_layers_total
        mrr = 32
        slots = cfg._get_num_req_slots(mrr)
        target = slots * cfg._swa_ring_size * 640 * cfg.num_layers_total
        self.assertEqual(cfg._fixed_swa_bytes(mrr), int(target * cfg._spec_infl))

    def test_dsv4_fp8_pd_refuses_pp_and_hisparse(self):
        from sglang.srt.model_executor.pool_configurator import (
            check_dsv4_unified_fp8_pd_supported,
        )

        base = dict(
            unified_fp8=True,
            disaggregation_mode="prefill",
            pp_size=1,
            enable_hisparse=False,
        )
        check_dsv4_unified_fp8_pd_supported(**base)
        # bf16 PD keeps both
        check_dsv4_unified_fp8_pd_supported(
            **{**base, "unified_fp8": False, "pp_size": 2, "enable_hisparse": True}
        )
        # fp8 without PD keeps both
        check_dsv4_unified_fp8_pd_supported(
            **{
                **base,
                "disaggregation_mode": "null",
                "pp_size": 2,
                "enable_hisparse": True,
            }
        )
        for mode in ("prefill", "decode"):
            for key, value, message in (
                ("pp_size", 2, "pp_size=2"),
                ("enable_hisparse", True, "enable-hisparse"),
            ):
                with self.subTest(disaggregation_mode=mode, refused=key):
                    with self.assertRaisesRegex(ValueError, message):
                        check_dsv4_unified_fp8_pd_supported(
                            **{**base, "disaggregation_mode": mode, key: value}
                        )


class TestOutOfTreeHybridSwaRouting(CustomTestCase):
    """Under an OOT platform a hybrid-SWA model must get the full+SWA sub-pools
    and the generic SWA allocator, not a flat pool and a paged allocator."""

    def _fake_configurator(self):
        _publish_config(self, page_size=64, disaggregation_mode="null")
        return SimpleNamespace(
            device="cpu",
            kv_cache_dtype=torch.bfloat16,
            kv_cache_dtype_str="auto",
            is_draft_worker=False,
            is_hybrid_swa=True,
            use_mla_backend=False,
            mambaish_config=None,
            model_config=SimpleNamespace(hf_config=SimpleNamespace()),
            _build_tpu_swa_kv_pool=MagicMock(name="swa_builder"),
            _build_oot_mha_kv_pool=MagicMock(name="flat_builder"),
        )

    def _sizes(self):
        from sglang.srt.mem_cache.kv_cache_configurator import _PoolSizes

        return _PoolSizes(
            max_total_num_tokens=256,
            max_running_requests=4,
            full_max_total_num_tokens=256,
            swa_max_total_num_tokens=128,
            c4_max_total_num_tokens=0,
            c128_max_total_num_tokens=0,
            c4_state_pool_size=0,
            c128_state_pool_size=0,
            c4_state_dtype=None,
            c128_state_dtype=None,
        )

    def test_oot_hybrid_swa_builds_full_and_swa_sub_pools(self):
        from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator

        fake = self._fake_configurator()
        with patch(
            "sglang.srt.mem_cache.kv_cache_configurator.current_platform"
        ) as platform:
            platform.is_out_of_tree.return_value = True
            platform.is_tpu.return_value = False
            KVCacheConfigurator._build_token_to_kv_pool(
                fake,
                sizes=self._sizes(),
                is_dsa_model=False,
                is_dsv4_model=False,
                req_to_token_pool=MagicMock(),
            )
        fake._build_tpu_swa_kv_pool.assert_called_once_with(
            full_max_total_num_tokens=256, swa_max_total_num_tokens=128
        )
        fake._build_oot_mha_kv_pool.assert_not_called()

    def test_oot_hybrid_swa_uses_generic_swa_allocator(self):
        from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator

        fake = self._fake_configurator()
        with (
            patch(
                "sglang.srt.mem_cache.kv_cache_configurator.current_platform"
            ) as platform,
            patch(
                "sglang.srt.mem_cache.kv_cache_configurator.SWATokenToKVPoolAllocator"
            ) as swa_allocator,
        ):
            platform.is_out_of_tree.return_value = True
            platform.is_tpu.return_value = False
            KVCacheConfigurator._build_token_to_kv_pool_allocator(
                fake,
                sizes=self._sizes(),
                token_to_kv_pool=MagicMock(),
                is_dsv4_model=False,
                req_to_token_pool=MagicMock(),
                token_to_kv_pool_allocator=None,
            )
        platform.get_paged_allocator_cls.assert_not_called()
        swa_allocator.assert_called_once()


class TestTpuDsaRouting(CustomTestCase):
    """On TPU a DSA model must get the TPU DSA pool. The generic DSA builder
    allocates the GPU layout, which the TPU kernels cannot read."""

    _fake_configurator = TestOutOfTreeHybridSwaRouting._fake_configurator
    _sizes = TestOutOfTreeHybridSwaRouting._sizes

    def test_tpu_dsa_model_builds_the_tpu_dsa_pool(self):
        from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator

        fake = self._fake_configurator()
        fake.is_hybrid_swa = False
        fake.use_mla_backend = True
        fake._build_tpu_dsa_kv_pool = MagicMock(name="tpu_dsa_builder")
        fake._build_dsa_kv_pool = MagicMock(name="generic_dsa_builder")
        with patch(
            "sglang.srt.mem_cache.kv_cache_configurator.current_platform"
        ) as platform:
            platform.is_out_of_tree.return_value = False
            platform.is_tpu.return_value = True
            KVCacheConfigurator._build_token_to_kv_pool(
                fake,
                sizes=self._sizes(),
                is_dsa_model=True,
                is_dsv4_model=False,
                req_to_token_pool=MagicMock(),
            )
        fake._build_tpu_dsa_kv_pool.assert_called_once_with(max_total_num_tokens=256)
        fake._build_dsa_kv_pool.assert_not_called()


class TestOutOfTreeMlaPoolConstruction(CustomTestCase):
    """The core OOT MLA builder passes index_head_dim, which no TPU MLA pool took,
    so every MLA model failed to build its KV pool under SGLANG_PLATFORM=tpu_oot."""

    def test_oot_mla_builder_kwargs_reach_the_tpu_mla_pools(self):
        from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
        from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
        from sglang_tpu.kv_pool import TpuFusedMLATokenToKVPool
        from sglang_tpu.platform import TpuOOTPlatform
        from sglang_tpu.tpu_platform import TpuSRTPlatform

        _publish_config(self, page_size=64, disaggregation_mode="null")
        fake = SimpleNamespace(
            device="cpu",
            pool_page_size=64,
            kv_cache_dtype=torch.bfloat16,
            model_config=SimpleNamespace(
                kv_lora_rank=512, qk_rope_head_dim=64, index_head_dim=None
            ),
            layer_info=SimpleNamespace(
                num_effective_layers=2, start_layer=0, end_layer=2
            ),
        )
        for pool_cls in (TpuFusedMLATokenToKVPool, MLATokenToKVPool):
            with (
                self.subTest(pool=pool_cls.__name__),
                patch.object(
                    TpuSRTPlatform, "get_mla_kv_pool_cls", lambda _, c=pool_cls: c
                ),
                # autospec raises TypeError on a keyword the real __init__ lacks.
                patch.object(pool_cls, "__init__", autospec=True, return_value=None),
                patch(
                    "sglang.srt.mem_cache.kv_cache_configurator.current_platform",
                    TpuOOTPlatform(),
                ),
            ):
                pool = KVCacheConfigurator._build_oot_mla_kv_pool(
                    fake, max_total_num_tokens=256, is_dsa_model=False
                )
                self.assertIsInstance(pool, pool_cls)


if __name__ == "__main__":
    unittest.main()
