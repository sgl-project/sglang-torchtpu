#!/usr/bin/env bash
# UNSUBMITTED benchmark-only adapter of successful exp-bsvzoblh0d's setup_env.sh.
# Peer extracted script SHA256 is recorded in the manifest/provenance; no giant bundle.
# Installs ONLY in a fresh local venv; never changes production images or model mounts.
# Caller must bound the complete bootstrap+runner with GNU timeout (manifest: 9600s).
set -euo pipefail
ulimit -c 0
: "${SOURCE_REPO:?}" "${SOURCE_SHA:?}" "${TORCH_TPU_WHEEL:?}" "${TORCH_TPU_WHEEL_SHA256:?}"
: "${ARTIFACT_LOCAL_DIR:?must be shared between the two hosts}" "${GLM52_RENDEZVOUS_ID:?unique per allocation}"
: "${GLM52_IMAGE:?}" "${GLM52_BOOTSTRAP_SHA256:?}" "${GLM52_PEER_SETUP_SHA256:?}"
[[ "$SOURCE_SHA" =~ ^[0-9a-f]{40}$ && "$GLM52_BOOTSTRAP_SHA256" =~ ^[0-9a-f]{64}$ ]]
[[ "$TORCH_TPU_WHEEL_SHA256" == 6e12816e92f9a5d9649358541f332f06583ccdfee5b6a74434dbf2141757fd82 ]]
[[ "$GLM52_IMAGE" == us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.9.0-rev1 ]]
[[ "${FALCON_WORLD_SIZE:-}" == 2 && "${FALCON_RANK:-}" =~ ^[01]$ ]]
[[ "${TPU_TOPOLOGY:-}" == 2x2x2 ]]
[[ "$GLM52_RENDEZVOUS_ID" =~ ^[A-Za-z0-9_-]+$ ]]
export TMPDIR=/tmp/tpu_logs/tmp PIP_CACHE_DIR=/tmp/tpu_logs/pip-cache PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
# Do not conceal input topology overrides: runner must reject them.
ROOT=$(mktemp -d /tmp/glm52-peer-rank"$FALCON_RANK".XXXXXX)
export SRC="$ROOT/src" VENV="$ROOT/venv"
export ART="$ARTIFACT_LOCAL_DIR/peer-bootstrap/rank-$FALCON_RANK"
mkdir -p "$ART" "$TMPDIR"
exec > >(tee -a "$ART/driver.log") 2>&1
printf '%s  %s\n' "$GLM52_BOOTSTRAP_SHA256" "$0" | sha256sum -c -
restore_pyproject() {
    if [[ -f "$ROOT/pyproject.original" && -d "$SRC/.git" ]]; then
        cp "$ROOT/pyproject.original" "$SRC/python/pyproject.toml"
    fi
}
trap restore_pyproject EXIT
# Exact peer clone procedure: no branch fallback, local source mutation, or remote submit.
git init -q "$SRC"
git -C "$SRC" remote add origin "$SOURCE_REPO"
git -C "$SRC" fetch -q --depth=1 origin "$SOURCE_SHA"
git -C "$SRC" checkout -q --detach FETCH_HEAD
[[ "$(git -C "$SRC" rev-parse HEAD)" == "$SOURCE_SHA" ]]
git -C "$SRC" log -1 --format='%H %s' > "$ART/source.txt"
cp "$SRC/python/pyproject.toml" "$ROOT/pyproject.original"
cp "$SRC/python/pyproject_tpu.toml" "$SRC/python/pyproject.toml"
python3 -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
python3 -m venv "$VENV"
export PATH="$VENV/bin:$PATH"
python -m pip install -q --upgrade pip setuptools setuptools-scm wheel > "$ART/pip-base.log" 2>&1
WHEEL=/wheels/torch-tpu/$TORCH_TPU_WHEEL
printf '%s  %s\n' "$TORCH_TPU_WHEEL_SHA256" "$WHEEL" | sha256sum -c - > "$ART/torch-tpu-sha.txt"
# Preserve the source's full dependency pins, substituting only the verified private wheel.
python - "$SRC/python/pyproject_tpu.toml" "$WHEEL" > "$ROOT/req.txt" <<'PY'
import pathlib, re, sys, tomllib
project = tomllib.loads(pathlib.Path(sys.argv[1]).read_text())['project']
for dep in project['dependencies']:
    if re.match(r'torch[_-]tpu\b', dep):
        pin = dep.split('==')[1]
        assert f'-{pin}-' in pathlib.Path(sys.argv[2]).name, (dep, sys.argv[2])
        print(sys.argv[2])
    else:
        print(dep)
PY
# Peer prepared artifact supplies find-links, not an assertion of offline dependency closure.
test -d /prepared/wheelhouse
python -m pip install --find-links=/prepared/wheelhouse --extra-index-url https://download.pytorch.org/whl/cpu \
    -r "$ROOT/req.txt" > "$ART/pip-req.log" 2>&1
python -m pip install --no-deps --no-build-isolation -e "$SRC/python" -e "$SRC" > "$ART/pip-editable.log" 2>&1
# Editable metadata is installed; restore the ORIGINAL tracked file before runner clean check.
restore_pyproject
[[ -z "$(git -C "$SRC" status --porcelain --untracked-files=no)" ]]
python -m pip freeze > "$ART/pip-freeze.txt"
# Same peer prerequisite: jax-ai base lacks libibverbs for tpu-sync raw transfer.
if ! ldconfig -p | awk '/libibverbs.so.1/ {found=1} END {exit !found}'; then
    apt-get update -qq > "$ART/apt.log" 2>&1
    apt-get install -y -qq libibverbs1 librdmacm1 >> "$ART/apt.log" 2>&1
fi
export TPU_LIBRARY_PATH
TPU_LIBRARY_PATH=$(python -c 'import libtpu; print(libtpu.get_library_path())')
export SOURCE_COMMIT="$SOURCE_SHA"
export GLM52_BOOTSTRAP_PROVENANCE="$ART/bootstrap-provenance.json"
python - <<'PY'
import hashlib, importlib.metadata as m, json, os
from pathlib import Path
import tomllib
pins = {}
for dep in tomllib.loads((Path(os.environ['SRC'])/'python/pyproject_tpu.toml').read_text())['project']['dependencies']:
    name, _, version = dep.partition('==')
    if name in ('torch_tpu', 'torch', 'torchvision', 'jax', 'jaxlib', 'libtpu', 'tpu-sync-torch'):
        pins[name] = version
versions = {name: m.version(name) for name in pins}
assert all(versions[n].split('+')[0] == v for n, v in pins.items()), (pins, versions)
p = dict(schema_version=1, scope='benchmark-only bootstrap identity; base tag NOT immutable image identity',
         peer_experiment='exp-bsvzoblh0d', peer_task_status='succeeded',
         peer_setup_sha256=os.environ['GLM52_PEER_SETUP_SHA256'],
         bootstrap_sha256=os.environ['GLM52_BOOTSTRAP_SHA256'], base_image=os.environ['GLM52_IMAGE'],
         source_repo=os.environ['SOURCE_REPO'], source_commit=os.environ['SOURCE_SHA'],
         wheel=os.environ['TORCH_TPU_WHEEL'], wheel_sha256=os.environ['TORCH_TPU_WHEEL_SHA256'],
         wheel_path='/wheels/torch-tpu/'+os.environ['TORCH_TPU_WHEEL'],
         prepared_wheelhouse='/prepared/wheelhouse', want=pins, got=versions,
         candidate_sha256=hashlib.sha256((Path(os.environ['SRC'])/'benchmark/tpu/glm52_epic/torch_prefill_candidate.json').read_bytes()).hexdigest())
Path(os.environ['GLM52_BOOTSTRAP_PROVENANCE']).write_text(json.dumps(p, indent=2)+'\n')
print(json.dumps(p, sort_keys=True), flush=True)
PY
# Shared artifact rendezvous: atomic rank JSON records, explicitly ordered by Falcon rank.
# A unique allocation ID prevents an old job's records from becoming new peer addresses.
export TPU_WORKER_HOSTNAMES
TPU_WORKER_HOSTNAMES=$(python - <<'PY'
import ipaddress, json, os, socket, time
from pathlib import Path
rank = int(os.environ['FALCON_RANK'])
root = Path(os.environ['ARTIFACT_LOCAL_DIR'])/'peer-rendezvous'/os.environ['GLM52_RENDEZVOUS_ID']
root.mkdir(parents=True, exist_ok=True)
hostname = socket.gethostname()
ips = sorted({a[4][0] for a in socket.getaddrinfo(hostname, None, socket.AF_INET)
              if not ipaddress.ip_address(a[4][0]).is_loopback})
assert ips, f'no routable IPv4 for {hostname}'
record = dict(rank=rank, hostname=hostname, ip=ips[0], ips=ips,
              source_commit=os.environ['SOURCE_SHA'], bootstrap_sha256=os.environ['GLM52_BOOTSTRAP_SHA256'],
              rendezvous_id=os.environ['GLM52_RENDEZVOUS_ID'])
target = root/f'rank{rank}.json'
tmp = root/f'.rank{rank}-{os.getpid()}.tmp'
tmp.write_text(json.dumps(record)+'\n')
os.replace(tmp, target)
deadline = time.monotonic()+300
while True:
    try:
        peers = [json.loads((root/f'rank{i}.json').read_text()) for i in range(2)]
        for i, peer in enumerate(peers):
            assert peer['rank'] == i
            for key in ('source_commit', 'bootstrap_sha256', 'rendezvous_id'):
                assert peer[key] == record[key], (key, peer)
            assert not ipaddress.ip_address(peer['ip']).is_loopback
        assert peers[0]['ip'] != peers[1]['ip'], peers
        assert peers[0]['hostname'] != peers[1]['hostname'], peers
        Path(os.environ['ART'], 'ordered-hosts.json').write_text(json.dumps(peers, indent=2)+'\n')
        print(','.join(peer['ip'] for peer in peers))
        break
    except (FileNotFoundError, json.JSONDecodeError):
        if time.monotonic() >= deadline:
            raise TimeoutError('two-host shared ARTIFACT_LOCAL_DIR rendezvous exceeded 300s')
        time.sleep(1)
PY
)
cd "$SRC"
# Gate intentionally remains strict in the existing runner. Do not invent an image digest.
# Parent must explicitly approve benchmark provenance acceptance before this can run on the tag.
RUNNER=${GLM52_RUNNER_PATH:-$SRC/benchmark/tpu/glm52_epic/torch_8chip_runner.sh}
exec timeout --signal=TERM --kill-after=5s 7195s bash "$RUNNER"
