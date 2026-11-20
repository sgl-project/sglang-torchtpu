"""CPU checks of the plugin hook that admits a fitting encoder-replay hit whole."""

import pkgutil
import unittest
from contextlib import ExitStack

from test_prefill_adder import TestPrefillAdder

from sglang.srt.plugins.hook_registry import HookRegistry, _propagate_patch
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.hooks import HOOKS
from sglang_tpu.hooks.replay_chunk_fit import hook_specs

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _apply(stack: ExitStack, hooks) -> None:
    """Apply through the real registry, restoring every binding afterwards."""
    for target, hook, kind in hooks:
        owner_path, name = target.rsplit(".", 1)
        owner = pkgutil.resolve_name(owner_path)
        raw = vars(owner).get(name)
        original = getattr(owner, name)
        HookRegistry._apply_target(target, [(kind, hook, None)])
        wrapped = getattr(owner, name)

        def restore(
            owner=owner, name=name, raw=raw, original=original, wrapped=wrapped
        ):
            setattr(owner, name, raw)
            _propagate_patch(wrapped, original, owner)

        stack.callback(restore)


class TestReplayChunkFitHook(TestPrefillAdder):
    """Reuses TestPrefillAdder's fixtures; its tests are dropped below."""

    def setUp(self):
        super().setUp()
        override = get_context().override_server_args(
            enable_encoder_swa_bounded_replay=True
        )
        override.install()
        self.addCleanup(override.restore)
        self.mock_tree_cache.supports_mamba.return_value = False
        self.mock_tree_cache.supports_prefix_sharing.return_value = False
        self.mock_token_allocator.available_size.return_value = 32768

    def _admit(
        self,
        new_tokens,
        *,
        prefix_len=4096,
        hooked=True,
        rem_chunk_tokens=1024,
        chunk_tokens_cap=None,
    ):
        with ExitStack() as stack:
            if hooked:
                _apply(stack, hook_specs())
            adder = self.create_adder(
                self.create_running_batch(),
                page_size=256,
                rem_chunk_tokens=rem_chunk_tokens,
            )
            req = self.create_shared_req(f"hit-{new_tokens}-{prefix_len}")
            req.prefix_len = prefix_len
            req.full_untruncated_fill_ids = list(range(prefix_len + new_tokens))
            adder.add_one_req(
                req,
                has_chunked_req=False,
                truncation_align_size=None,
                chunk_tokens_cap=chunk_tokens_cap,
            )
            return adder, req

    def test_registered(self):
        for spec in hook_specs():
            self.assertIn(spec, HOOKS)

    def test_replay_hit_that_fits_runs_in_one_chunk(self):
        """The replay rows leave the chunk limit off a page boundary, so core's
        page-ceiled fit check splits a fitting hit into a page-floored chunk
        plus a second step."""
        for new_tokens, extend_len, rem_after in (
            (800, 800, 96),
            (896, 896, 0),  # replay 128 + 896 fills the 1024 chunk exactly
            (897, 768, 128),  # does not fit: page-floored chunk, as in core
        ):
            with self.subTest(new_tokens=new_tokens):
                adder, req = self._admit(new_tokens)
                self.assertEqual((req.extend_end - req.prefix_len), extend_len)
                self.assertEqual(adder.new_chunked_req is req, extend_len < new_tokens)
                self.assertEqual(adder.rem_chunk_tokens, rem_after)
                self.assertFalse(adder.exact_chunk_fill)

    def test_off_page_request_cap_counts_as_the_limit(self):
        """A per-request cap below the batch budget is the limit the hit must fit."""
        for new_tokens, extend_len in ((800, 800), (897, 768)):
            with self.subTest(new_tokens=new_tokens):
                adder, req = self._admit(
                    new_tokens, rem_chunk_tokens=2048, chunk_tokens_cap=896
                )
                self.assertEqual((req.extend_end - req.prefix_len), extend_len)
                self.assertEqual(adder.new_chunked_req is req, extend_len < new_tokens)

    def test_core_alone_splits_the_fitting_hit(self):
        adder, req = self._admit(800, hooked=False)
        self.assertEqual((req.extend_end - req.prefix_len), 768)
        self.assertIs(adder.new_chunked_req, req)

    def test_without_replay_the_hooks_change_nothing(self):
        for new_tokens in (800, 1024, 1100):
            with self.subTest(new_tokens=new_tokens):
                core, core_req = self._admit(new_tokens, prefix_len=0, hooked=False)
                hooked, hooked_req = self._admit(new_tokens, prefix_len=0)
                self.assertEqual(
                    hooked_req.extend_end - hooked_req.prefix_len,
                    core_req.extend_end - core_req.prefix_len,
                )
                self.assertEqual(hooked.rem_chunk_tokens, core.rem_chunk_tokens)
                self.assertEqual(hooked.rem_input_tokens, core.rem_input_tokens)


# Keep the inherited TestPrefillAdder tests from running a second time here.
for _name in list(vars(TestPrefillAdder)):
    if _name.startswith("test_") and _name not in vars(TestReplayChunkFitHook):
        setattr(TestReplayChunkFitHook, _name, None)
del TestPrefillAdder


if __name__ == "__main__":
    unittest.main()
