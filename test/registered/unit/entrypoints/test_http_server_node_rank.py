import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.entrypoints import http_server
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestHttpServerNodeRank(unittest.TestCase):
    def test_serving_uses_rank_resolved_by_engine(self):
        for initial_rank in (0, 1):
            for final_rank in (0, 1):
                for rust in (False, True):
                    for custom_engine in (False, True):
                        with self.subTest(
                            initial=initial_rank,
                            final=final_rank,
                            rust=rust,
                            custom_engine=custom_engine,
                        ):
                            self.check_launch(
                                initial_rank, final_rank, rust, custom_engine
                            )

    def check_launch(self, initial_rank, final_rank, rust, custom_engine):
        server_args = SimpleNamespace(node_rank=initial_rank)
        resolved = SimpleNamespace(node_rank=initial_rank)
        scheduler = Mock(scheduler_infos=[{}])
        tokenizer, template, ports, watchdog = (object() for _ in range(4))

        def launch(**kwargs):
            self.assertIs(kwargs["server_args"], server_args)
            resolved.node_rank = final_rank
            return tokenizer, template, ports, scheduler, watchdog, []

        engine = SimpleNamespace(_launch_subprocesses=Mock(side_effect=launch))
        callback = Mock()
        with (
            patch.object(
                http_server.Engine, "_launch_subprocesses", side_effect=launch
            ) as default,
            patch.object(http_server, "get_parallel", return_value=resolved),
            patch.object(
                http_server,
                "get_serving",
                return_value=SimpleNamespace(skip_server_warmup=False),
            ),
            patch.object(http_server.envs.SGLANG_RUST_SERVER, "get", return_value=rust),
            patch.object(http_server, "_setup_and_run_http_server") as serve,
            patch.object(http_server, "_execute_server_warmup") as warmup,
        ):
            kwargs = {"engine_class": engine} if custom_engine else {}
            http_server.launch_server(server_args, launch_callback=callback, **kwargs)
            (
                engine._launch_subprocesses if custom_engine else default
            ).assert_called_once()
            (
                default if custom_engine else engine._launch_subprocesses
            ).assert_not_called()
        # Raw arguments deliberately retain the old rank: the published result
        # of Engine normalization controls serving, not the input field.
        self.assertEqual(server_args.node_rank, initial_rank)
        if final_rank:
            serve.assert_not_called()
            warmup.assert_not_called()
            callback.assert_not_called()
            scheduler.block_until_scheduler_exits.assert_not_called()
        elif rust:
            # Each Rust listener runs its own startup warmup.
            serve.assert_not_called()
            warmup.assert_not_called()
            callback.assert_called_once_with()
            scheduler.block_until_scheduler_exits.assert_called_once_with()
        else:
            serve.assert_called_once_with(
                server_args,
                tokenizer,
                template,
                ports,
                scheduler.scheduler_infos,
                watchdog,
                execute_warmup_func=http_server._execute_server_warmup,
                launch_callback=callback,
            )
            warmup.assert_not_called()
            callback.assert_not_called()


if __name__ == "__main__":
    unittest.main()
