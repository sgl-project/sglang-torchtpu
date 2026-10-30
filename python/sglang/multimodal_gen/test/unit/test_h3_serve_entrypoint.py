"""Public H3 serve routing and config forwarding, without TPU execution."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from sglang.cli.main import main as public_main
from sglang.multimodal_gen.configs.pipeline_configs.base import PipelineConfig
from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
    MiniMaxH3PipelineConfig,
)
from sglang.multimodal_gen.runtime.entrypoints.cli import serve as diffusion_serve
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.platforms.tpu import TpuPlatform
from sglang.multimodal_gen.runtime.server_args import ServerArgs


@pytest.fixture
def server_config():
    return (
        Path(__file__).resolve().parents[5]
        / "scripts/torchtpu/h3_eager_baseline/server.json"
    )


def resolve_public_cli(config=None, node=0, extra=(), auto=False, rendezvous=True):
    argv = [
        "sglang",
        "serve",
        "--model-path",
        "MiniMax/MiniMax-H3",
    ]
    if config is not None:
        argv += ["--config", str(config)]
    if not auto:
        argv += ["--model-type", "diffusion"]
    if node is not None:
        argv += ["--node-rank", str(node)]
    if rendezvous:
        argv += ["--dist-init-addr", "127.0.0.1:10031"]
    argv += list(extra)
    env = {
        "TPU_WORKER_HOSTNAMES": "host0",
        "TPU_WORKER_ID": str(node if node is not None else 0),
        "TORCH_TPU_DEVICES_PER_HOST": "8",
        "TPU_TOPOLOGY": "2x4x1",
    }
    # Exercise the real public CLI and ServerArgs normalization. Only model
    # resolution, device bootstrap, port probes and service launch are mocked.
    with (
        patch.object(sys, "argv", argv),
        patch.dict(os.environ, env),
        patch.object(
            PipelineConfig, "from_kwargs", return_value=MiniMaxH3PipelineConfig()
        ),
        patch.object(current_platform, "is_tpu", return_value=True),
        patch.object(current_platform, "is_cpu", return_value=False),
        patch.object(
            current_platform,
            "apply_server_args_defaults",
            side_effect=TpuPlatform().apply_server_args_defaults,
        ),
        patch.object(ServerArgs, "_require_port"),
        patch("sglang_tpu.patches._prepare_tpu_env") as prepare,
        patch("sglang.srt.plugins.load_plugins"),
        patch("sglang.cli.serve.kill_process_tree"),
        patch.object(diffusion_serve, "dispatch_launch") as dispatch,
    ):
        public_main()
        dispatch.assert_called_once()
        prepare.assert_called_once_with(
            world_size=8, resolve_single_host_topology=False
        )
        return dispatch.call_args.args[0]


def test_public_serve_resolves_single_replica_defaults(server_config):
    args = resolve_public_cli(server_config)
    assert (args.nnodes, args.node_rank, args.num_gpus) == (1, 0, 8)
    assert args.dist_init_addr == "127.0.0.1:10031"
    assert args.master_port == 10032
    assert args.scheduler_port == 5650
    assert args.scheduler_endpoints == ["tcp://0.0.0.0:5650"]
    assert (args.dp_size, args.sp_degree, args.ulysses_degree) == (1, 8, 8)
    assert (args.hsdp_replicate_dim, args.hsdp_shard_dim) == (1, 8)
    assert args.encoder_parallel == "fold"
    assert args.warmup_mode == "off"
    assert args.enable_torch_compile is False
    assert args.regional_compile is False
    assert args.enable_breakable_cuda_graph is False
    assert args.dit_cpu_offload is False
    assert args.dit_layerwise_offload is False


def test_public_serve_accepts_user_parallelism_and_cli_overrides(server_config):
    args = resolve_public_cli(
        server_config,
        extra=[
            "--dp-size",
            "2",
            "--sp-degree",
            "4",
            "--ulysses-degree",
            "4",
            "--hsdp-replicate-dim",
            "2",
            "--hsdp-shard-dim",
            "4",
            "--encoder-parallel",
            "replicate",
            "--host",
            "127.0.0.1",
            "--dist-init-addr",
            "custom-head:12000",
            "--text-encoder-cpu-offload",
            "false",
        ],
    )
    assert (args.dp_size, args.sp_degree, args.ulysses_degree) == (2, 4, 4)
    assert (args.hsdp_replicate_dim, args.hsdp_shard_dim) == (2, 4)
    assert args.encoder_parallel == "replicate"
    assert args.host == "127.0.0.1"
    assert args.dist_init_addr == "custom-head:12000"
    assert args.text_encoder_cpu_offload is False
    assert args.scheduler_endpoints == ["tcp://127.0.0.1:5650", "tcp://127.0.0.1:5651"]
    assert args.scheduler_port == 5650


def test_public_serve_auto_routes_registered_h3(server_config):
    assert (
        resolve_public_cli(server_config, auto=True).pipeline_class_name
        == "MiniMaxH3Pipeline"
    )


def test_public_serve_accepts_regional_compile(server_config):
    args = resolve_public_cli(
        server_config,
        extra=[
            "--enable-torch-compile",
            "true",
            "--regional-compile",
            "true",
        ],
    )
    assert args.enable_torch_compile is True
    assert args.regional_compile is True
    assert args.warmup_mode == "off"


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ([], (True, True)),
        (["--enable-torch-compile", "false"], (False, False)),
        (["--regional-compile", "false"], (True, False)),
    ],
)
def test_public_serve_compile_defaults_without_recipe(overrides, expected):
    args = resolve_public_cli(
        extra=[
            "--num-gpus",
            "8",
            "--performance-mode",
            "manual",
            "--warmup-mode",
            "off",
            "--sp-degree",
            "8",
            "--ulysses-degree",
            "8",
            "--attention-backend",
            "torch_sdpa",
            *overrides,
        ],
    )
    assert (args.enable_torch_compile, args.regional_compile) == expected


def test_tpu_compile_defaults_leave_other_models_unchanged():
    from types import SimpleNamespace

    from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
        FastH3PipelineConfig,
    )

    for config in (PipelineConfig(), FastH3PipelineConfig()):
        args = SimpleNamespace(
            pipeline_config=config,
            enable_torch_compile=False,
            regional_compile=False,
        )
        TpuPlatform().apply_server_args_defaults(args)
        assert (args.enable_torch_compile, args.regional_compile) == (False, False)


def test_yaml_config_is_forwarded_to_native_parser(server_config, tmp_path):
    import yaml

    config = json.loads(server_config.read_text())
    config["text_encoder_cpu_offload"] = False
    config["dist_init_addr"] = "yaml-head:12001"
    path = tmp_path / "server.yaml"
    path.write_text(yaml.safe_dump(config))
    args = resolve_public_cli(path, rendezvous=False)
    assert args.text_encoder_cpu_offload is False
    assert args.dist_init_addr == "yaml-head:12001"


def test_public_serve_loads_compiled_recipe(server_config):
    config = server_config.parent.parent / "h3_compiled_baseline/server.json"
    args = resolve_public_cli(config)
    assert args.enable_torch_compile is True
    assert args.regional_compile is True
    assert args.warmup_mode == "off"
    eager_config = json.loads(server_config.read_text())
    compiled_config = json.loads(config.read_text())
    for key in ("enable_torch_compile", "regional_compile"):
        eager_config.pop(key, None)
        compiled_config.pop(key, None)
    assert compiled_config == eager_config


def test_public_serve_can_disable_compile_in_compiled_recipe(server_config):
    config = server_config.parent.parent / "h3_compiled_baseline/server.json"
    args = resolve_public_cli(config, extra=["--enable-torch-compile", "false"])
    assert args.enable_torch_compile is False


def test_native_serve_infers_worker_rank_from_deployment(server_config):
    args = resolve_public_cli(server_config, node=None)
    assert args.node_rank == 0
