#!/usr/bin/env python3
"""Parse actual staged CLI grammar without constructing model/runtime/device."""
import argparse
import json
from pathlib import Path

from benchmark.tpu.recipe.configs.config_eval import server_args_to_cli
from sglang.srt.server_args import ServerArgs


def main():
    path = Path('benchmark/tpu/glm52_epic/torch_staged_loading_candidate.json')
    cfg = json.loads(path.read_text())
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    # argparse alone accepts unknown keys whose False values get omitted by
    # config_eval. Check every supplied key against actual grammar destinations.
    destinations = {a.dest for a in parser._actions}
    unknown = set(cfg['server_args']) - destinations
    if unknown:
        raise ValueError(f'unknown staged config keys: {sorted(unknown)}')
    parsed = parser.parse_args(server_args_to_cli(cfg['server_args']))
    assert parsed.weight_loader_disable_mmap is True
    assert parsed.weight_loader_prefetch_checkpoints is False
    extra = parsed.model_loader_extra_config
    if isinstance(extra, str): extra = json.loads(extra)
    assert extra['num_threads'] == 1 and extra['enable_multithread_load'] is True
    assert parsed.tp_size == parsed.ep_size == 16 and parsed.nnodes == 2
    print('STAGED_NATIVE_CLI_PARSE_PASS '+json.dumps(dict(config=str(path),unknown_keys=[],
          loader_extra=extra,scope='native argparse only; no ServerArgs resolution or model/device startup')),flush=True)


if __name__ == '__main__':
    main()
