#!/usr/bin/env bash
# UNSUBMITTED experimental native-startup smoke; NOT a performance benchmark.
# Run identically on both Falcon hosts. No installs, model writes or remote APIs.
set -euo pipefail
ulimit -c 0
exec python3 -u - "$@" <<'PY'
import glob
import hashlib
import http.server
import importlib.metadata
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request

START = time.monotonic()
# Reserve 40 seconds for native launcher's child cleanup inside the 7200s bound.
DEADLINE = START + 7160
RANK = int(os.environ.get('FALCON_RANK', '-1'))
OUT = Path('/tmp/tpu_logs/glm52-torch') / f'rank-{RANK}'
OUT.mkdir(parents=True, exist_ok=True)
result = {'status': 'FAILED', 'scope': 'startup and one token-ID smoke only; no performance claim',
          'falcon_rank': RANK, 'source_commit': None}
launcher = None
server_log = None
control = None
memory_stop = threading.Event()
memory_thread = None
state = {'status': 'RUNNING', 'peer_ack': False}


def save(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def require(test, message):
    if not test:
        raise RuntimeError(message)


def remaining(cap=30):
    value = min(cap, DEADLINE - time.monotonic())
    if value <= 0:
        raise TimeoutError('7160s work deadline exhausted (40s cleanup reserved)')
    return value


def run(argv, name, timeout=120):
    proc = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, timeout=remaining(timeout), check=False)
    (OUT / name).write_text(proc.stdout)
    require(proc.returncode == 0, f'{argv!r} failed: exit {proc.returncode}; see {name}')
    return proc.stdout.strip()


def check_child():
    remaining()
    if launcher is not None and launcher.poll() is not None:
        raise RuntimeError(f'native launcher exited unexpectedly: {launcher.returncode}')


def request(url, payload=None, timeout=5):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=remaining(timeout)) as response:
        body = response.read()
        return response.status, body


def interrupted(signum, _frame):
    raise SystemExit(128 + signum)


signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)
signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('work deadline')))
signal.alarm(7160)

try:
    # Fail closed BEFORE imports that might initialize the TPU or any model load.
    image = os.environ.get('GLM52_IMAGE', '')
    if os.environ.get('GLM52_BOOTSTRAP_PROVENANCE'):
        # Benchmark-only alternative identity: a tagged BASE is not a runtime image.
        # The separately delivered, hashed bootstrap records the exact source/wheel
        # and pins; the installed runtime is still checked independently below.
        provenance = json.loads(Path(os.environ['GLM52_BOOTSTRAP_PROVENANCE']).read_text())
        bootstrap_sha = os.environ.get('GLM52_BOOTSTRAP_SHA256', '')
        require(re.fullmatch('[0-9a-f]{64}', bootstrap_sha), 'missing bootstrap SHA256')
        require(image == 'us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.9.0-rev1',
                'bootstrap mode accepts only documented peer base tag')
        require(provenance.get('schema_version') == 1 and all(provenance.get(k) == v for k, v in {
            'base_image': image, 'source_commit': os.environ.get('SOURCE_COMMIT'),
            'source_repo': 'https://github.com/sgl-project/sglang-torchtpu.git',
            'bootstrap_sha256': bootstrap_sha,
            'wheel': 'torch_tpu-0.2.1.dev20261028201649-cp312-cp312-manylinux_2_31_x86_64.whl',
            'wheel_sha256': '6e12816e92f9a5d9649358541f332f06583ccdfee5b6a74434dbf2141757fd82',
            'peer_experiment': 'exp-bsvzoblh0d', 'peer_task_status': 'succeeded',
        }.items()), 'bootstrap provenance does not match documented peer identity')
        require(provenance.get('peer_setup_sha256') ==
                'e13506f849a5f7fef901cce7730a0bb1fdabe125b3eba4173475f59dfbff7a00',
                'extracted peer bootstrap provenance mismatch')
        save('bootstrap-provenance.json', provenance)
        result['bootstrap_provenance'] = provenance
    else:
        require(re.fullmatch(r'[^\s<>]+@sha256:[0-9a-f]{64}', image),
                'GLM52_IMAGE must be a user-provided immutable image digest; no default image')
    require('lmsysorg/sglang:v0.1.0-tpu' not in image, 'known mismatched public image rejected')
    require(RANK in (0, 1) and os.environ.get('FALCON_WORLD_SIZE') == '2',
            'expected exactly two Falcon hosts, ranks 0 and 1 (not device ranks)')
    expected = os.environ.get('SOURCE_COMMIT', '')
    require(re.fullmatch('[0-9a-f]{40}', expected), 'SOURCE_COMMIT must be a full SHA')
    actual = run(['git', 'rev-parse', 'HEAD'], 'source-sha.txt')
    result.update(source_commit=actual, image=image)
    require(actual == expected, 'checkout SHA differs from requested exact source')
    require(not run(['git', 'status', '--porcelain', '--untracked-files=no'], 'source-status.txt'),
            'tracked source modifications detected')
    mode = os.environ.get('GLM52_BASELINE_MODE', '')
    require(mode in ('', 'eager', 'compile-smoke'), 'unsupported baseline mode')
    candidate = Path('benchmark/tpu/glm52_epic/' + (
        'torch_runnable_baseline.json' if mode == 'eager' else
        'torch_compile_smoke.json' if mode == 'compile-smoke' else
        'torch_prefill_candidate.json'))
    require(candidate.is_file(), 'run from exact source checkout root')
    candidate_bytes = candidate.read_bytes()
    (OUT / candidate.name).write_bytes(candidate_bytes)
    cfg = json.loads(candidate_bytes)
    args = cfg['server_args']
    runtime_env = cfg.get('runtime_env', cfg.get('runtime_envs', {}))
    if mode in ('eager', 'compile-smoke'):
        require(runtime_env.get('SGLANG_TPU_ENABLE_QBLOCK_SPARSE_MLA_PREFILL') == '1',
                'short baseline must enable DSA qblock prefill')
        require(runtime_env.get('SGLANG_TPU_ASYNC_COPY_WORKER') == '0',
                'short baseline must disable async copy worker')
        require('xla_tpu_buffer_assignment_enable_fast_merge_fallback=false' in
                runtime_env.get('LIBTPU_INIT_ARGS', ''),
                'short baseline must disable libtpu fast-merge fallback')
        require('xla_tpu_offload_gather_to_sparsecore=true' in
                runtime_env.get('LIBTPU_INIT_ARGS', ''),
                'short baseline must offload gather to SparseCore')
    require(all(args.get(k) == v for k, v in {
        'tp_size': 16, 'attn_dp_size': 16, 'dp_size': 1, 'ep_size': 16,
        'nnodes': 2, 'device': 'tpu', 'model_path': '/models/GLM5.2-fp8-channel-wise'
        }.items()), 'candidate mesh/model contract changed')
    if mode in ('eager', 'compile-smoke'):
        require(args.get('page_size') == 128,
                'short baseline must use the qblock-compatible page size')
    contract = ({'enable_torch_compile': False, 'context_length': 4096,
                 'max_total_tokens': 8192, 'chunked_prefill_size': 4096,
                 'max_prefill_tokens': 4096, 'page_size': 128} if mode == 'eager' else
                {'enable_torch_compile': True, 'context_length': 4096,
                 'max_total_tokens': 8192, 'chunked_prefill_size': 4096,
                 'max_prefill_tokens': 4096, 'page_size': 128, 'torch_compile_max_bs': 1,
                 'tpu_compile_bs': [1], 'tpu_compile_tokens': [2048]} if mode == 'compile-smoke' else {
                 'chunked_prefill_size': 32768, 'max_prefill_tokens': 32768,
                 'tpu_compile_tokens': [2048]})
    require(all(args.get(k) == v for k, v in contract.items()), 'baseline shape contract changed')
    if mode in ('eager', 'compile-smoke'):
        require(not os.environ.get('GLM52_DIAGNOSTIC_REQUESTS'),
                'baseline smoke cannot run matched-workload diagnostic')
        result['scope'] = f'full-checkpoint short-input {mode} smoke; NOT JAX performance'
    result['candidate_sha256'] = hashlib.sha256(candidate_bytes).hexdigest()
    pins = {'torch': '2.11.0+cpu', 'jax': '0.11.2', 'jaxlib': '0.11.2',
            'libtpu': '0.0.50', 'torch_tpu': '0.2.1.dev20261028201649',
            'tpu-sync-torch': '0.0.1.dev20261031081414'}
    versions = {key: importlib.metadata.version(key) for key in pins}
    save('versions.json', versions)
    result['versions'] = versions
    # The checkout's torch==2.11.0 pin resolves to its +cpu build for PrivateUse1.
    require(versions['torch'] in ('2.11.0', '2.11.0+cpu'), 'torch must be pinned 2.11.0 CPU ABI')
    require(all(versions[k] == v for k, v in pins.items() if k != 'torch'),
            f'installed runtime mismatch: {versions}; no runtime repair/install is allowed')
    result['versions'] = versions
    save('versions.json', versions)
    require(os.environ.get('TPU_TOPOLOGY') == '2x2x2', 'expected physical 2x2x2 chip slice')
    hosts = os.environ.get('TPU_WORKER_HOSTNAMES', '').split(',')
    require(len(hosts) == 2 and len(set(hosts)) == 2 and all(
        h and not h.startswith('USER_') and ':' not in h and not any(c.isspace() for c in h) for h in hosts),
        'require ordered TPU_WORKER_HOSTNAMES with exactly two distinct bare hostnames')
    if 'TPU_WORKER_ID' in os.environ:
        require(os.environ['TPU_WORKER_ID'] == str(RANK), 'Falcon and input TPU host ranks disagree')
    for key, value in [('FALCON_JAX_PROCESS_COUNT', '2'), ('FALCON_JAX_PROCESS_ID', str(RANK))]:
        if key in os.environ:
            require(os.environ[key] == value, f'{key} conflicts with two-host Falcon mapping')
    for key in ('TPU_VISIBLE_CHIPS', 'FORCE_SINGLE_HOST', 'SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE'):
        require(not os.environ.get(key), f'{key} is incompatible with full native slice')
    # TPU runtimes may expose VFIO rather than legacy /dev/accelN nodes.
    # Device-file counts are not a physical topology API; native coordinate
    # discovery below must succeed before the server can initialize its mesh.
    devices = sorted(glob.glob('/dev/accel[0-9]*') + glob.glob('/dev/accel/accel[0-9]*')
                     + glob.glob('/dev/vfio/[0-9]*'))
    os.environ.update(NODE_RANK=str(RANK), RANK0_HOST=hosts[0], DIST_INIT_ADDR=hosts[0]+':10011',
        TPU_WORKER_ID=str(RANK), TORCH_TPU_TOPOLOGY='2,2,2,2', TORCH_TPU_DEVICES_PER_HOST='8',
        TORCH_TPU_BASE_PORT='8070', SGLANG_PLATFORM='tpu', SGLANG_USE_RAY='0',
        HOME='/tmp/tpu_logs/home', XDG_CACHE_HOME='/tmp/tpu_logs/cache',
        TMPDIR='/tmp/tpu_logs/tmp', PIP_CACHE_DIR='/tmp/tpu_logs/pip-cache',
        UV_CACHE_DIR='/tmp/tpu_logs/uv-cache', JAX_COMPILATION_CACHE_DIR='/tmp/tpu_logs/jax-cache',
        RUN_DIR=str(OUT), TORCH_COMPILE_DEBUG_DIR=str(OUT / 'compile-debug'),
        # Disable implicit /dev/shm tier2; all cache writes stay under /tmp.
        ENABLE_COMPILE_CACHE='0', TORCH_TPU_TIER2_COMPILATION_CACHE='disabled',
        TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT='/tmp/tpu_logs/torch-cache',
        PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
    for name in ('HOME', 'TMPDIR', 'XDG_CACHE_HOME', 'PIP_CACHE_DIR', 'UV_CACHE_DIR',
                 'JAX_COMPILATION_CACHE_DIR'):
        Path(os.environ[name]).mkdir(parents=True, exist_ok=True)
    # Probe only native physical coordinates in an isolated process, no JAX client
    # or PJRT distributed world survives to contend with the real server.
    coords = run([sys.executable, '-m', 'sglang_tpu.node_normalization',
                  '--print_local_device_coords'], 'native-coordinate-probe.log', 180)
    result['coordinate_probe'] = coords
    result['topology'] = {'physical_chips': 8, 'hosts': hosts, 'physical_chip_nodes': devices,
                          'devices_per_host': 8, 'total_devices': 16,
                          'chip_topology': '2x2x2', 'native_topology': '2,2,2,2'}
    save('preflight.json', result)
    # Native Engine startup performs the cross-host coordinate exchange, verifies
    # complete physical assignment and normalizes pod-index -> physical host rank.
    # Its input leader forwards HTTP if the physical rank-zero host changes.
    run([sys.executable, '-m', 'benchmark.tpu.launch_server', '--config', '@'+str(candidate),
         '--dry-run'], 'native-dry-run.json')

    class Control(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != '/status':
                self.send_error(404)
                return
            payload = json.dumps(state).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):
            if self.path != '/ack':
                self.send_error(404)
                return
            state['peer_ack'] = True
            self.send_response(200)
            self.end_headers()

        def log_message(self, *_):
            pass

    if RANK == 0:
        control = http.server.ThreadingHTTPServer(('0.0.0.0', 31001), Control)
        threading.Thread(target=control.serve_forever, daemon=True).start()
    from benchmark.tpu.glm52_epic.startup_memory import collect as collect_memory

    def sample_memory():
        with (OUT / 'host-memory.jsonl').open('a', buffering=1) as stream:
            while not memory_stop.is_set():
                try:
                    data = collect_memory()
                    stream.write(json.dumps(data) + '\n')
                    print('GLM52_HOST_MEMORY ' + json.dumps(data), flush=True)
                except Exception as error:
                    # Diagnostic sampling must never terminate model execution.
                    print(f'GLM52_HOST_MEMORY_ERROR {error!r}', flush=True)
                memory_stop.wait(30)

    memory_thread = threading.Thread(target=sample_memory, daemon=True)
    memory_thread.start()
    server_log = (OUT / 'server.log').open('wb', buffering=0)
    command = [sys.executable, '-u', '-m', 'benchmark.tpu.launch_server', '--config',
               '@'+str(candidate), '--shutdown-grace', '25']
    save('command.json', command)
    launcher = subprocess.Popen(command, stdout=server_log, stderr=subprocess.STDOUT,
                                start_new_session=True)
    if RANK == 0:
        while True:
            check_child()
            try:
                code, body = request('http://127.0.0.1:30000/health')
                if code == 200:
                    (OUT / 'health.body').write_bytes(body)
                    break
            except (OSError, urllib.error.URLError):
                pass
            time.sleep(1)
        # Plain-DSA q-block prefill is selected only for >=512 tokens. A four-token
        # probe silently falls back to the old per-query sparse kernel and cannot
        # validate the peer recipe we are testing here. The long candidate keeps
        # the historical short probe because it is not the q-block smoke.
        smoke_tokens = 512 if mode in ('eager', 'compile-smoke') else 4
        # One output comes from prefill logits; require a second output to
        # exercise a decode forward, which never uses the q-block prefill op.
        smoke_outputs = 2 if mode in ('eager', 'compile-smoke') else 1
        request_body = {'input_ids': [1] * smoke_tokens, 'sampling_params': {
            'temperature': 0, 'max_new_tokens': smoke_outputs, 'ignore_eos': True}, 'stream': False}
        save('generate-request.json', request_body)
        code, body = request('http://127.0.0.1:30000/generate', request_body, timeout=600)
        (OUT / 'generate-response.json').write_bytes(body)
        response = json.loads(body)
        require(code == 200 and isinstance(response, dict), '/generate failed')
        meta = response.get('meta_info', {})
        require(meta.get('completion_tokens') == smoke_outputs,
                f'smoke did not produce exactly {smoke_outputs} tokens')
        require(meta.get('finish_reason', {}).get('type') != 'abort', 'generation aborted')
        ids = response.get('output_ids')
        # Current native response includes IDs; never silently treat absence as success.
        require(isinstance(ids, list) and len(ids) == smoke_outputs
                and all(type(token) is int for token in ids),
                f'current native /generate must return {smoke_outputs} output token IDs')
        result.update(health_status=200, generate_status=code, output_ids=ids,
                      request_success_count=1, generate_input_tokens=len(request_body['input_ids']),
                      expected_attention_path=(
                          'plain_dsa_qblock_prefill'
                          if mode in ('eager', 'compile-smoke')
                          else 'candidate_default_attention_path'))
        check_child()
        # Opt-in only: prepared requests must reference frozen data + its manifest
        # available locally (e.g. supplemental exact JAX source /workspace/sglang-jax).
        # No observer is required/enabled here; no claim of matched JAX cadence.
        diagnostic_requests = os.environ.get('GLM52_DIAGNOSTIC_REQUESTS', '')
        if diagnostic_requests:
            client_out = OUT / 'client'
            client_out.mkdir(parents=True, exist_ok=True)
            require(Path(diagnostic_requests).is_file(), 'diagnostic requests not mounted locally')
            run([sys.executable, 'benchmark/tpu/glm52_epic/torch_cache_hit_client.py',
                 'run', '--requests', diagnostic_requests,
                 '--base-url', 'http://127.0.0.1:30000',
                 '--allow-unmatched-admission', '--exclusive-server',
                 '--timeout', str(remaining(3600)),
                 '--output', str(client_out / 'result.json')], 'client/client.log', 7000)
            diagnostic = json.loads((client_out / 'result.json').read_text())
            require(diagnostic.get('status') == 'http_diagnostic_validated_UNMATCHED',
                    'client did not validate HTTP diagnostic')
            check_child()
            result['http_diagnostic_status'] = diagnostic['status']
            state['http_diagnostic_pass'] = True
        state['status'] = 'SUCCEEDED'
        while not state['peer_ack']:
            check_child()
            time.sleep(0.2)
    else:
        while True:
            check_child()
            try:
                _, body = request('http://'+hosts[0]+':31001/status')
                remote = json.loads(body)
                require(remote['status'] != 'FAILED', 'input rank0 reported smoke failure')
                if remote['status'] == 'SUCCEEDED':
                    check_child()
                    request('http://'+hosts[0]+':31001/ack', {})
                    result['rank0_smoke_acknowledged'] = True
                    if remote.get('http_diagnostic_pass'):
                        state['http_diagnostic_pass'] = True
                    break
            except (OSError, urllib.error.URLError):
                pass
            time.sleep(1)
    result['status'] = ('SMOKE_AND_HTTP_DIAGNOSTIC_PASS' if state.get('http_diagnostic_pass')
                        else 'SUCCEEDED')
    if state.get('http_diagnostic_pass'):
        result['scope'] = 'smoke and UNMATCHED HTTP diagnostic; NOT JAX cadence'
except BaseException as error:
    state['status'] = 'FAILED'
    result['error'] = f'{type(error).__name__}: {error}'
    traceback.print_exc()
finally:
    signal.alarm(0)
    # The native launcher owns separate server process groups. SIGTERM it first
    # so its finally block reaps workers; capture descendant groups as fallback.
    if launcher is not None:
        descendants = {launcher.pid}
        for _ in range(32):
            before = len(descendants)
            for proc in Path('/proc').glob('[0-9]*/stat'):
                try:
                    fields = proc.read_text().rsplit(')', 1)[1].split()
                    if int(fields[1]) in descendants:
                        descendants.add(int(proc.parent.name))
                except (OSError, ValueError, IndexError):
                    pass
            if len(descendants) == before:
                break
        groups = set()
        for pid in descendants:
            try:
                groups.add(os.getpgid(pid))
            except ProcessLookupError:
                pass
        try:
            launcher.send_signal(signal.SIGTERM)
            launcher.wait(timeout=32)
        except subprocess.TimeoutExpired:
            result.update(status='FAILED', cleanup_error='native launcher did not stop within 32s')
        for group in groups:
            try:
                os.killpg(group, signal.SIGKILL)
            except ProcessLookupError:
                pass
        launcher.wait(timeout=3)
        result['launcher_shutdown_exit_code'] = launcher.returncode
    memory_stop.set()
    if memory_thread is not None:
        memory_thread.join(timeout=2)
    if server_log:
        server_log.close()
    if control:
        control.shutdown()
        control.server_close()
    result['elapsed_seconds'] = round(time.monotonic() - START, 3)
    save('result.json', result)
    print(json.dumps(result, sort_keys=True), flush=True)
    # Replay WHOLE native server log to Falcon's durable stdout sink, including
    # on failure; the untruncated /tmp copy also remains available for exp cp.
    if (OUT / 'server.log').exists():
        print(f'===== BEGIN WHOLE SERVER LOG rank={RANK} =====', flush=True)
        with (OUT / 'server.log').open('rb') as stream:
            while chunk := stream.read(1024 * 1024):
                sys.stdout.buffer.write(chunk)
        sys.stdout.buffer.flush()
        print(f'===== END WHOLE SERVER LOG rank={RANK} =====', flush=True)
sys.exit(0 if result['status'] in ('SUCCEEDED', 'SMOKE_AND_HTTP_DIAGNOSTIC_PASS') else 1)
PY
