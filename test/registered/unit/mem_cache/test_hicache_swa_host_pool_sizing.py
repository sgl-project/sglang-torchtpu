"""HiCache L2 on a hybrid-SWA model: the SWA host pool bounds the full pool too.

Both host pools take the same token inflow, so an SWA pool at
``swa_full_tokens_ratio`` R turns over 1/R times as fast. Retention then follows
the smaller pool, while the sizing knobs and the exposed gauge both describe the
larger one -- so a prefix whose full half is comfortably host-resident still
returns ``host_hit_length`` 0 once its SWA half has churned out.

The mechanism, measured rather than assumed. An undersized SWA pool **refuses
the host allocation** -- 31 refusals at ratio 0.2 here, none at parity. A node
that cannot be backed then holds neither device nor host data, and
``_iteratively_delete_tombstone_ancestors`` reaps it, cascading to childless
ancestors: 31 refusals delete 62 nodes, leaving 41 in the tree against 103 at
parity. The re-match then returns 0 because the prefix is **gone**, not because
it is unreachable -- and its full half was freed with it.

Note this is *not* the conjunctive component match. Flipping ``_all_valid`` in
``unified_tree_core`` from ``all()`` to ``any()`` leaves every result in this
file unchanged, because by then there is no node left to validate. A failure
here means the sizing relationship changed, not that the validators were
decoupled.

The TPU e2e offload tests cover hybrid-SWA end to end, but none varies the ratio;
the tests that set a low ratio enable no HiCache. This crosses the two.

No accelerator: tree bookkeeping only, about a second.
"""

import contextlib
import os
import sys
import unittest
from array import array
from unittest import mock

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _fixtures():
    """Shared unified-cache fixtures.

    A sibling test module rather than a package, so it only imports once this
    directory is on ``sys.path``. Kept in a function because that ordering
    cannot survive import sorting at module scope.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)

    import test_unified_radix_cache_unittest as fixtures

    from sglang.srt.mem_cache.pool_host import mha
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

    return fixtures, mha, ComponentType


# Sized so the churn below lands between the two host pools.
SWA_RATIO_PARITY = 1.0
SWA_RATIO_LOW = 0.2
TARGET_TOKENS = 40
# Above the low-ratio SWA host pool (~819 tokens) so the target's SWA half is
# evicted, below the full host pool (~4097) so its full half survives in both.
CHURN_TOKENS = 2000


@contextlib.contextmanager
def _on_cpu(T):
    """build_fixture asks the platform for a device; there is none on this lane."""
    import sglang.srt.utils as _u
    import sglang.srt.utils.common as _c

    with contextlib.ExitStack() as stack:
        for module in (T, _u, _c):
            if hasattr(module, "get_device"):
                stack.enter_context(
                    mock.patch.object(module, "get_device", lambda *a, **k: "cpu")
                )
        yield


def _measure(swa_ratio: float, churn_tokens: int = CHURN_TOKENS) -> dict:
    """Back a prefix to host, drop it from device, churn, then re-match.

    ``churn_tokens`` is the working set pushed through after the target is
    backed up. It is a parameter because the collapse is set by the working set
    relative to the SWA pool, not by the ratio alone -- see the band test.
    """
    T, _mha, ComponentType = _fixtures()
    measured: dict = {}

    orig_pool = T.SWAKVPool
    orig_alloc = T.SWATokenToKVPoolAllocator
    orig_backup = _mha.MHATokenToKVPoolHost.backup_from_device_all_layer

    def _sized(factory):
        def build(*args, **kwargs):
            kwargs["size_swa"] = max(8, int(kwargs["size"] * swa_ratio))
            return factory(*args, **kwargs)

        return build

    T.SWAKVPool = _sized(orig_pool)
    T.SWATokenToKVPoolAllocator = _sized(orig_alloc)
    _mha.MHATokenToKVPoolHost.backup_from_device_all_layer = lambda self, *a, **k: None

    class _Case(T._InsertWalkSuite):
        cfg = T.CacheConfig(
            page_size=1,
            components=(ComponentType.FULL, ComponentType.SWA),
            sliding_window_size=4,
            kv_size=2048,
            max_context_len=2048,
            max_num_reqs=64,
        )
        _backup_node = T.UnifiedRadixCacheSuite._backup_node
        _path_chain = T.UnifiedRadixCacheSuite._path_chain

        def _init_hicache(self, cache):
            """Own init: ``UnifiedRadixCacheSuite._init_hicache`` patches
            ``assembler.get_mha_host_pool_cls``, which this tree no longer defines.
            Patch the current hook instead, for the same purpose -- force
            ``pin_memory=False`` so the host pool allocates without a device.
            """
            from sglang.srt.mem_cache.pool_host import base as pool_host_base
            from sglang.srt.platforms import current_platform

            # The host pool refuses to build unless psutil reports headroom, minus
            # a fixed reserve, divided across the ranks on the host. A CPU CI
            # runner can compute that as negative, which rejects even the ~0.01 GB
            # pools here. The budget is a production sizing guard and is not what
            # this test exercises, so give it a fixed generous answer.
            budget = mock.patch.object(
                pool_host_base, "host_memory_budget_bytes", return_value=1 << 30
            )
            budget.start()
            self.addCleanup(budget.stop)

            orig = current_platform.get_mha_kv_pool_host_cls

            def _unpinned(device_pool):
                host_cls = orig(device_pool)

                def factory(*args, **kwargs):
                    kwargs["pin_memory"] = False
                    return host_cls(*args, **kwargs)

                return factory

            patcher = mock.patch.object(
                type(current_platform),
                "get_mha_kv_pool_host_cls",
                side_effect=lambda dp: _unpinned(dp),
            )
            patcher.start()
            self.addCleanup(patcher.stop)

            server_args = T.ServerArgs(
                model_path="dummy",
                page_size=self.cfg.page_size,
                hicache_io_backend="direct",
                hicache_mem_layout="page_first_direct",
                hicache_write_policy="write_through",
            )
            T.set_global_server_args_for_scheduler(server_args)
            cache.init_hicache(server_args, cache.cache_init_params)
            cache.write_through_threshold = 1 << 30
            cache.load_back_threshold = 0

        def _match(self, cache, seq):
            return cache.match_prefix(
                T.MatchPrefixParams(key=T.RadixKey(array("q", seq)))
            )

        def _node(self, cache, seq):
            # Newer tree cores address nodes by opaque handle and expose
            # root_node_handle(); older ones pass node objects. Pick by
            # capability -- the wrong one fails deep inside the arena.
            node = self._match(cache, seq).last_device_node
            if hasattr(cache, "root_node_handle"):
                return node
            resolve = getattr(cache, "resolve_node_handle", None)
            return resolve(node) if resolve and isinstance(node, int) else node

        def runTest(self):
            cache, allocator, rtp = T.build_fixture(self.cfg)
            self._init_hicache(cache)

            target = list(range(1, TARGET_TOKENS + 1))
            self._insert(cache, allocator, rtp, target)
            self._backup_node(cache, self._node(cache, target))
            cache.evict(T.EvictParams(num_tokens=len(target)))
            before = self._match(cache, target).host_hit_length

            churn = 0
            while churn < churn_tokens:
                base = 10_000 + churn * 10
                seq = list(range(base, base + TARGET_TOKENS))
                self._insert(cache, allocator, rtp, seq)
                self._backup_node(cache, self._node(cache, seq))
                cache.evict(T.EvictParams(num_tokens=len(seq)))
                churn += len(seq)

            measured.update(
                {
                    "full_host": cache.full_kv_pool_host.size,
                    "swa_host": cache.swa_kv_pool_host.size,
                    "before": before,
                    "churn": churn,
                    "after": self._match(cache, target).host_hit_length,
                    # handed back so a test can inspect or alter the surviving
                    # tree; the suppression arm needs to drop one component's
                    # host half without disturbing the other.
                    "rematch": lambda: self._match(cache, target).host_hit_length,
                    "core": cache.tree_core,
                }
            )

    try:
        with _on_cpu(T):
            case = _Case()
            case.runTest()
            case.doCleanups()
        return measured
    finally:
        T.SWAKVPool = orig_pool
        T.SWATokenToKVPoolAllocator = orig_alloc
        _mha.MHATokenToKVPoolHost.backup_from_device_all_layer = orig_backup


class TestHiCacheSwaHostPoolSizing(unittest.TestCase):
    """The churn is chosen to sit between the two pools, so these arms isolate
    SWA pool size. They do not model a workload and imply no hit rate for one."""

    def test_parity_keeps_the_prefix_host_resident(self):
        """Equal pools: a host-backed prefix survives churn the full pool absorbs."""
        r = _measure(SWA_RATIO_PARITY)
        self.assertEqual(r["swa_host"], r["full_host"], r)
        self.assertEqual(r["before"], TARGET_TOKENS, r)
        self.assertEqual(
            r["after"],
            TARGET_TOKENS,
            f"host-backed prefix lost to churn the full pool absorbs: {r}",
        )

    def test_a_higher_ratio_narrows_the_band_but_does_not_close_it(self):
        """The two arms above invite "low ratios are bad, so 0.5 is fine". Wrong.

        Read together, parity-survives and low-collapses suggest a cutoff
        somewhere between, and that a comfortable ratio is therefore safe. It is
        not a cutoff. Retention follows the SWA pool, so the collapse begins
        wherever the working set outgrows it -- and the SWA pool is
        ``ratio x full pool``. Every ratio below 1.0 therefore leaves a band

            ratio x full_host  <  working set  <  full_host

        in which the SWA half is gone from host, the full half is still there,
        and the conjunctive match discards both. Raising the ratio narrows that
        band by (1 - ratio); only 1.0 closes it.

        So this asserts the rule rather than a threshold: for a mid ratio, a
        churn under its SWA pool retains the prefix and a churn over it does
        not. A fixed-churn sweep across ratios would show an apparent cutoff and
        reproduce the very misreading this guards against.
        """
        mid = 0.5
        probe = _measure(mid, churn_tokens=1)
        swa_host, full_host = probe["swa_host"], probe["full_host"]
        self.assertLess(swa_host, full_host, probe)

        # Under the SWA pool: the prefix is still there.
        under = _measure(mid, churn_tokens=max(TARGET_TOKENS * 2, swa_host // 3))
        self.assertEqual(
            under["after"],
            TARGET_TOKENS,
            f"churn inside the SWA pool should retain the prefix: {under}",
        )

        # Over it, but still inside the full pool, so the SWA half is the only
        # possible reason -- the same isolation the two arms above rely on.
        # Midpoint of the band rather than a multiple of swa_host: at ratio 0.5
        # doubling it lands past full_host and the arm would no longer isolate.
        over = _measure(mid, churn_tokens=(swa_host + full_host) // 2)
        self.assertLess(over["churn"], full_host, over)
        self.assertEqual(
            over["after"],
            0,
            f"a ratio of {mid} is not immune: churn past its SWA pool still "
            f"zeroes the host hit for the full half too: {over}",
        )

    def test_losing_only_the_swa_host_half_also_zeroes_the_hit(self):
        """The other route to a zero hit: the node survives and the match stops.

        The three arms above are all ratio-driven, and a low ratio takes one
        specific route -- the SWA host allocation is refused at backup time, the
        node never gains a host presence, and the tombstone reaper deletes it
        along with its full half. The validators never run.

        There is a second route, reached when an SWA half leaves host *after* a
        successful backup: the node survives (the reaper keeps anything whose
        Full half is on host), the conjunctive component match rejects it, and
        the walk halts before the full half -- still resident -- is counted.
        This arm constructs that state directly from a healthy parity tree, and
        it is the one that leaves usable KV stranded rather than freed.

        Unlike the reaping arms, this one *is* sensitive to the combinator:
        the same state returns the full hit under ``any()``.
        """
        r = _measure(SWA_RATIO_PARITY)
        self.assertEqual(r["after"], TARGET_TOKENS, r)

        core, rematch = r["core"], r["rematch"]
        _, _, ComponentType = _fixtures()

        dropped = 0
        stack = [core.root_node]
        while stack:
            node = stack.pop()
            try:
                data = node.component(ComponentType.SWA)
            except Exception:  # noqa: BLE001 - node carries no SWA component
                data = None
            if data is not None and getattr(data, "host_value", None) is not None:
                data.host_value = None
                dropped += 1
            stack.extend(node.children.values())

        self.assertGreater(dropped, 0, "no SWA host halves to drop; fixture changed")
        self.assertEqual(
            rematch(),
            0,
            "an absent SWA host half should suppress the hit on the full half, "
            f"which is still resident (dropped {dropped} SWA halves)",
        )

    def test_small_swa_pool_bounds_the_full_host_hit(self):
        """A smaller SWA pool zeroes the host hit for the full half as well.

        Same target, same churn, same full pool as the parity case -- only the
        SWA pool differs. Documents today's behaviour: if this starts failing,
        retention no longer follows the SWA pool, so delete this test and
        revisit swa_full_tokens_ratio guidance. (It will not catch the
        components being decoupled -- see the module docstring.)
        """
        r = _measure(SWA_RATIO_LOW)
        self.assertLess(r["swa_host"], r["full_host"], r)
        self.assertEqual(r["before"], TARGET_TOKENS, r)
        self.assertEqual(
            r["after"],
            0,
            f"expected the smaller SWA pool to bound the host hit: {r}",
        )


if __name__ == "__main__":
    unittest.main()
