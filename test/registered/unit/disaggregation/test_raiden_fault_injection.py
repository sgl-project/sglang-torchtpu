"""Raiden fault-injection presets turn a share of requests into tpu-sync rules."""

import sys
import unittest
from types import ModuleType
from unittest.mock import patch

from sglang.srt.disaggregation.raiden import fault_injection
from sglang.srt.disaggregation.raiden.fault_injection import (
    RaidenFaultInjector,
    build_rules,
    expected_per_request,
    load_config,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

CONFIG = load_config()
DELAY_5_SEC = "delay_5_percent_req_max_5_sec"
DELAY_60_SEC = "delay_5_percent_req_max_60_sec"
FAIL_DELAY_5_SEC = "fail_5_percent_req_delay_5_percent_req_max_5_sec"
ROLES = ("prefill", "decode")
RANKS = 8
READS_PER_REQUEST = 3.5
LAYERS_PER_READ = 18
PER_LAYER_HOOKS = {site.hook for site in CONFIG.sites if site.per_layer}
TIME_LIMITED_HOOKS = {site.hook for site in CONFIG.sites if site.limit_ms is not None}


class _FakeNative:
    """Stands in for tpu_sync.api.torch.fault_injection."""

    def __init__(self):
        self.installs = []

    def inject_faults(self, rules):
        self.installs.append(list(rules))

    def get_hit_count(self):
        return 0


def _rules(preset, role):
    return build_rules(
        sites=CONFIG.sites,
        preset=CONFIG.presets[preset],
        role=role,
        ranks=RANKS,
        reads_per_request=READS_PER_REQUEST,
        layers_per_request=READS_PER_REQUEST * LAYERS_PER_READ,
    )


def _injector(native, preset=FAIL_DELAY_5_SEC, role="decode"):
    return RaidenFaultInjector(
        native=native,
        name=preset,
        preset=CONFIG.presets[preset],
        sites=CONFIG.sites,
        role=role,
        ranks=RANKS,
    )


def _serve(injector, *, requests, reads, first_room=0):
    for room in range(first_room, first_room + requests):
        for _ in range(reads):
            injector.note_read(room=room, layers=LAYERS_PER_READ)


def _probability(rules, hook, action):
    (rule,) = [r for r in rules if r["hook"] == hook and r["action"] == action]
    return rule["probability"]


class TestBuildRules(unittest.TestCase):
    def _injections_per_request(self, preset, action):
        """Expected injections per request, summed over both roles and all ranks."""
        total = 0.0
        for role in ROLES:
            for rule in _rules(preset, role):
                if rule["action"] != action:
                    continue
                calls = READS_PER_REQUEST
                if rule["hook"] in PER_LAYER_HOOKS:
                    calls *= LAYERS_PER_READ
                total += rule["probability"] * calls * RANKS
        return total

    def test_all_processes_together_inject_the_preset_share(self):
        for name, preset in CONFIG.presets.items():
            with self.subTest(preset=name):
                self.assertAlmostEqual(
                    self._injections_per_request(name, "fail"),
                    preset.fails_per_request,
                )
                self.assertAlmostEqual(
                    self._injections_per_request(name, "delay"),
                    preset.delays_per_request,
                )

    def test_expected_count_of_a_process_matches_its_rules(self):
        for name, preset in CONFIG.presets.items():
            with self.subTest(preset=name):
                total = sum(
                    expected_per_request(
                        sites=CONFIG.sites, preset=preset, role=role, ranks=RANKS
                    )
                    * RANKS
                    for role in ROLES
                )
                self.assertAlmostEqual(
                    total, preset.fails_per_request + preset.delays_per_request
                )

    def test_a_long_delay_stays_out_of_time_limited_steps(self):
        """A delay past raiden's limit on a step fails the read, so a preset
        with a longer delay must not delay that step."""
        hooks = {
            preset: {rule["hook"] for role in ROLES for rule in _rules(preset, role)}
            for preset in (DELAY_5_SEC, DELAY_60_SEC)
        }
        self.assertTrue(TIME_LIMITED_HOOKS)
        self.assertFalse(hooks[DELAY_60_SEC] & TIME_LIMITED_HOOKS)
        self.assertTrue(TIME_LIMITED_HOOKS <= hooks[DELAY_5_SEC])

    def test_delay_presets_install_no_failure(self):
        for preset in (DELAY_5_SEC, DELAY_60_SEC):
            for role in ROLES:
                rules = _rules(preset, role)
                self.assertEqual({r["action"] for r in rules}, {"delay"})
                self.assertEqual(
                    {r["max_delay_ms"] for r in rules},
                    {CONFIG.presets[preset].max_delay_ms},
                )


class TestInjector(unittest.TestCase):
    def test_nothing_is_armed_before_the_warmup_requests(self):
        native = _FakeNative()
        injector = _injector(native)
        _serve(injector, requests=fault_injection._WARMUP_REQUESTS - 1, reads=4)
        self.assertEqual(native.installs, [])
        _serve(injector, requests=1, reads=1, first_room=10_000)
        self.assertEqual(len(native.installs), 1)

    def test_a_request_with_many_reads_counts_once(self):
        """More reads per request must lower the per-call probability, not
        raise the number of requests."""
        few, many = _FakeNative(), _FakeNative()
        _serve(_injector(few), requests=fault_injection._WARMUP_REQUESTS, reads=1)
        _serve(_injector(many), requests=fault_injection._WARMUP_REQUESTS, reads=14)
        hook = "transfer_recv_session.pull.request"
        ratio = _probability(few.installs[0], hook, "fail") / _probability(
            many.installs[0], hook, "fail"
        )
        # The arming read is the last request's first one.
        expected = ((fault_injection._WARMUP_REQUESTS - 1) * 14 + 1) / (
            fault_injection._WARMUP_REQUESTS
        )
        self.assertAlmostEqual(ratio, expected)

    def test_probabilities_follow_a_change_in_request_size(self):
        native = _FakeNative()
        injector = _injector(native)
        _serve(injector, requests=fault_injection._WARMUP_REQUESTS, reads=1)
        _serve(
            injector,
            requests=fault_injection._REFRESH_REQUESTS,
            reads=14,
            first_room=10_000,
        )
        self.assertEqual(len(native.installs), 2)
        hook = "block_transport.recv.layer_done"
        self.assertLess(
            _probability(native.installs[1], hook, "delay"),
            _probability(native.installs[0], hook, "delay"),
        )

    def test_a_rule_set_tpu_sync_rejects_does_not_fail_the_read(self):
        native = _FakeNative()
        native.inject_faults = lambda rules: (_ for _ in ()).throw(
            ValueError("unknown hook")
        )
        _serve(_injector(native), requests=fault_injection._WARMUP_REQUESTS, reads=1)


class TestStart(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(fault_injection, "_injector", None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_unset_selects_nothing(self):
        with envs.SGLANG_TEST_RAIDEN_FAULT_INJECTION.override(None):
            fault_injection.start(role="decode", ranks=RANKS)
        self.assertFalse(fault_injection.active())
        fault_injection.note_read(room=1, layers=LAYERS_PER_READ)

    def test_unknown_preset_is_refused(self):
        with envs.SGLANG_TEST_RAIDEN_FAULT_INJECTION.override("delay_5_percent"):
            with self.assertRaises(ValueError):
                fault_injection.start(role="decode", ranks=RANKS)

    def test_preset_arms_this_process(self):
        native = _FakeNative()
        module = ModuleType("tpu_sync.api.torch.fault_injection")
        module.inject_faults = native.inject_faults
        module.get_hit_count = native.get_hit_count
        package = ModuleType("tpu_sync.api.torch")
        package.fault_injection = module
        modules = {
            "tpu_sync": ModuleType("tpu_sync"),
            "tpu_sync.api": ModuleType("tpu_sync.api"),
            "tpu_sync.api.torch": package,
            "tpu_sync.api.torch.fault_injection": module,
        }
        with patch.dict(sys.modules, modules):
            with envs.SGLANG_TEST_RAIDEN_FAULT_INJECTION.override(DELAY_5_SEC):
                fault_injection.start(role="prefill", ranks=RANKS)
        self.assertTrue(fault_injection.active())
        for room in range(fault_injection._WARMUP_REQUESTS):
            fault_injection.note_read(room=room, layers=LAYERS_PER_READ)
        self.assertEqual(len(native.installs), 1)
        self.assertEqual({r["action"] for r in native.installs[0]}, {"delay"})


if __name__ == "__main__":
    unittest.main()
