"""CPU-only unit tests for the TPU hybrid-SWA 8-page alignment floor.

The two-pool hybrid-SWA KV cache miscompiles under torch.compile on TPU unless
the SWA sub-pool's page count is a multiple of the sublane tiling factor (8);
_derive_pool_sizes rounds down to enforce that. The round-down used to be
guarded by `aligned > 0`, meant to stop a small pool being zeroed -- but its
effect was that a pool of fewer than 8 pages skipped the alignment entirely and
was allocated at its unaligned size, which is exactly the state the round-down
exists to prevent. Nothing logged, and the failure surfaced much later as a
serving hang rather than a startup error. Such a pool is now refused at startup.

That floor is reachable from ordinary flags: `--max-total-tokens 32768` with
the TPU recipes' `--swa-full-tokens-ratio 0.10` and page_size 512 gives a
3072-token SWA pool -- 6 pages -- and the server then hangs on the first
request large enough to touch the upper pages.

These tests pin the three regimes: above the floor the alignment fires, at an
exact multiple it is a no-op, and below the floor startup is refused with a
message naming the 8-page minimum.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

PAGE_SIZE = 512


def _swa_pool_tokens(swa_tokens, *, page_multiple=8, page_size=PAGE_SIZE):
    """Run _derive_pool_sizes over a hybrid-SWA config and report the SWA size.

    ``page_multiple`` is the platform's answer to ``swa_pool_page_multiple()``
    (8 on TPU, 1 elsewhere)."""
    from sglang.srt import runtime_context as rc
    from sglang.srt.mem_cache import kv_cache_configurator
    from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
    from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig

    fake = SimpleNamespace(
        device="cpu",
        is_hybrid_swa=True,
        is_draft_worker=False,
        loc_space_scale=1,
        pool_page_size=page_size,
        model_config=SimpleNamespace(hf_config=SimpleNamespace()),
    )
    config = MemoryPoolConfig(
        max_total_num_tokens=32768,
        max_running_requests=64,
        full_max_total_num_tokens=32768,
        swa_max_total_num_tokens=swa_tokens,
    )
    with (
        rc.get_context().override_server_args(page_size=page_size),
        patch.object(
            kv_cache_configurator.current_platform,
            "swa_pool_page_multiple",
            return_value=page_multiple,
        ),
    ):
        sizes = KVCacheConfigurator._derive_pool_sizes(fake, config=config)
    return sizes.swa_max_total_num_tokens


class TestSWAPageAlignment(unittest.TestCase):
    def test_unaligned_pool_above_floor_is_rounded_down(self):
        """34 pages -> 32. The documented behaviour, and the observed one."""
        self.assertEqual(_swa_pool_tokens(34 * PAGE_SIZE), 32 * PAGE_SIZE)

    def test_already_aligned_pool_is_untouched(self):
        for pages in (8, 32, 64):
            with self.subTest(pages=pages):
                self.assertEqual(_swa_pool_tokens(pages * PAGE_SIZE), pages * PAGE_SIZE)

    def test_platform_without_a_page_multiple_is_untouched(self):
        """A platform answering 1 keeps the pool size as computed."""
        self.assertEqual(
            _swa_pool_tokens(34 * PAGE_SIZE, page_multiple=1), 34 * PAGE_SIZE
        )

    def test_pool_below_the_floor_is_refused_at_startup(self):
        """A sub-8-page pool cannot satisfy the invariant, so refuse rather than serve.

        6 pages is what `--max-total-tokens 32768 --swa-full-tokens-ratio 0.10`
        produces at page_size 512. Rounding down yields 0, so before this change
        the `aligned > 0` guard declined to act and the pool was allocated at 6
        pages, and the miscompile surfaced as a permanent serving hang.

        Match on the message, not just the type. The assertion this replaces
        accepted either a raised ValueError or a returned size with
        `pages % 8 == 0`. A guard that returned a zero-token pool raises
        nothing, so it fell through to the modulus arm -- where `0 % 8 == 0`
        passed it. Matching the message pins the branch actually intended.
        """
        with self.assertRaisesRegex(ValueError, "below the 8-page minimum"):
            _swa_pool_tokens(6 * PAGE_SIZE)


if __name__ == "__main__":
    unittest.main()
