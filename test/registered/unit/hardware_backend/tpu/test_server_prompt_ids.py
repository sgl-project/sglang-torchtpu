"""Server-rendered prompt ids skip the vocabulary walk; every other list is still checked."""

import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.hooks import server_prompt_ids as h

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _render(ids, client_ids=None):
    """Run the _process_messages hook around a stub that returns ``ids``."""
    request = SimpleNamespace(input_ids=client_ids)
    return h.process_messages(
        lambda self, req, *a, **k: SimpleNamespace(prompt_ids=ids), None, request, False
    )


class TestServerPromptIds(CustomTestCase):
    def setUp(self):
        h._rendered.clear()

    def _validate_with(self, ids):
        calls = []
        h.validate_input_ids_in_vocab(
            lambda _self, obj: calls.append(obj.input_ids),
            None,
            SimpleNamespace(input_ids=ids),
        )
        return bool(calls)

    def test_rendered_list_skips_once(self):
        with tpu_envs.SGLANG_TPU_SKIP_SERVER_PROMPT_VOCAB_CHECK.override(True):
            ids = [1, 2, 3]
            _render(ids)
            self.assertFalse(self._validate_with(ids))
            # Forgotten once checked: a second validation of the same list runs.
            self.assertTrue(self._validate_with(ids))

    def test_client_supplied_and_modified_lists_are_validated(self):
        with tpu_envs.SGLANG_TPU_SKIP_SERVER_PROMPT_VOCAB_CHECK.override(True):
            client = [5, 6]
            _render(client, client_ids=client)  # client sent input_ids: never trusted
            self.assertTrue(self._validate_with(client))
            extended = [1, 2]
            _render(extended)
            extended.append(10**9)  # changed length after rendering
            self.assertTrue(self._validate_with(extended))
            rendered = [1, 2]
            _render(rendered)
            self.assertTrue(self._validate_with(list(rendered)))  # a copy

    def test_off_by_default(self):
        ids = [1, 2, 3]
        _render(ids)
        self.assertTrue(self._validate_with(ids))
        self.assertEqual(len(h._rendered), 0)

    def test_bounded(self):
        with tpu_envs.SGLANG_TPU_SKIP_SERVER_PROMPT_VOCAB_CHECK.override(True):
            lists = [[i] for i in range(h._MAX_REMEMBERED + 10)]
            for ids in lists:
                _render(ids)
            self.assertEqual(len(h._rendered), h._MAX_REMEMBERED)
            self.assertTrue(self._validate_with(lists[0]))  # oldest dropped
            self.assertFalse(self._validate_with(lists[-1]))


if __name__ == "__main__":
    unittest.main()
