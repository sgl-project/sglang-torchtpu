"""--session-id-header: a request's session_id from an HTTP header (off by default)."""

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

import argparse
import unittest
from unittest.mock import AsyncMock, Mock

from fastapi import Request
from starlette.datastructures import Headers

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
)
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.entrypoints.openai.serving_completions import OpenAIServingCompletion
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _raw_request(headers):
    raw = Mock(spec=Request)
    raw.headers = Headers(headers)
    return raw


class _TemplateManager:
    chat_template_name = None
    jinja_template_content_format = None
    completion_template_name = None
    reasoning_config = None
    force_reasoning = False
    jinja_template_may_reorder_tool_results = False


class TestSessionIdHeaderCli(CustomTestCase):
    def _parse(self, args_list):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        args = parser.parse_args(["--model", "dummy"] + args_list)
        server_args = ServerArgs.from_cli_args(args)
        server_args.resolve_once()
        return server_args

    def test_off_by_default(self):
        self.assertIsNone(self._parse([]).session_id_header)

    def test_flag_alone_selects_x_correlation_id(self):
        args = self._parse(["--session-id-header"])
        self.assertEqual(args.session_id_header, "X-Correlation-ID")

    def test_header_name_is_configurable(self):
        args = self._parse(["--session-id-header", "X-Session-ID"])
        self.assertEqual(args.session_id_header, "X-Session-ID")

    def test_session_leaf_flags(self):
        off = self._parse([])
        self.assertFalse(off.session_leaf_at_prompt_end)
        self.assertFalse(off.session_release_evicted_leaf)
        on = self._parse(
            ["--session-leaf-at-prompt-end", "--session-release-evicted-leaf"]
        )
        self.assertTrue(on.session_leaf_at_prompt_end)
        self.assertTrue(on.session_release_evicted_leaf)
        self.assertFalse(off.session_coverage_host_only)
        self.assertTrue(
            self._parse(["--session-coverage-host-only"]).session_coverage_host_only
        )


class TestSessionIdFromHeader(CustomTestCase):
    def _completion(self, header, **req_kwargs):
        reset_context()
        self.addCleanup(reset_context)
        publish(
            ServerArgs(model_path="dummy", session_id_header=header), role="tokenizer"
        )
        tm = Mock(spec=TokenizerManager)
        tm.tokenizer = Mock()
        tm.tokenizer.encode.return_value = [1, 2, 3]
        tm.model_config = Mock(is_multimodal=False)
        tm.server_args = Mock(enable_cache_report=False)
        tm.generate_request = AsyncMock()
        return OpenAIServingCompletion(tm, _TemplateManager()), CompletionRequest(
            model="x", prompt=[1, 2, 3], max_tokens=4, **req_kwargs
        )

    def test_off_ignores_the_header(self):
        sc, req = self._completion(None)
        raw = _raw_request({"X-Correlation-ID": "conv-7"})
        internal, _ = sc._convert_to_internal_request(req, raw)
        self.assertIsNone(internal.session_id)

    def test_completions_take_the_session_from_the_header(self):
        sc, req = self._completion("X-Correlation-ID")
        # Header names are case-insensitive, as on the wire.
        raw = _raw_request({"x-correlation-id": "conv-7"})
        internal, _ = sc._convert_to_internal_request(req, raw)
        self.assertEqual(internal.session_id, "conv-7")

    def test_body_session_id_wins(self):
        sc, req = self._completion("X-Correlation-ID", session_id="body-1")
        raw = _raw_request({"X-Correlation-ID": "conv-7"})
        internal, _ = sc._convert_to_internal_request(req, raw)
        self.assertEqual(internal.session_id, "body-1")

    def test_session_params_body_keeps_its_explicit_session(self):
        sc, req = self._completion("X-Correlation-ID", session_params={"id": "s"})
        raw = _raw_request({"X-Correlation-ID": "conv-7"})
        internal, _ = sc._convert_to_internal_request(req, raw)
        self.assertIsNone(internal.session_id)

    def test_missing_or_empty_header_gives_no_session(self):
        sc, req = self._completion("X-Correlation-ID")
        for headers in ({}, {"X-Correlation-ID": ""}):
            internal, _ = sc._convert_to_internal_request(req, _raw_request(headers))
            self.assertIsNone(internal.session_id)

    def test_configured_name_only(self):
        sc, req = self._completion("X-Session-ID")
        raw = _raw_request({"X-Correlation-ID": "conv-7", "X-Session-ID": "s-9"})
        internal, _ = sc._convert_to_internal_request(req, raw)
        self.assertEqual(internal.session_id, "s-9")

    def test_chat_takes_the_session_from_the_header(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(
            ServerArgs(
                model_path="dummy",
                session_id_header="X-Correlation-ID",
                default_chat_template_kwargs=None,
            ),
            role="tokenizer",
        )
        chat = OpenAIServingChat.__new__(OpenAIServingChat)
        req = ChatCompletionRequest(
            model="x", messages=[{"role": "user", "content": "Hi?"}], max_tokens=4
        )
        raw = _raw_request({"X-Correlation-ID": "conv-7"})
        self.assertEqual(chat.extract_session_id(req, raw), "conv-7")
        # The chat conversion fills GenerateReqInput.session_id through the helper.
        code = OpenAIServingChat._convert_to_internal_request.__code__
        self.assertIn("extract_session_id", code.co_names)


if __name__ == "__main__":
    unittest.main()
