#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
SRC=/workspace/sglang-torchtpu
RUN_DIR=/tmp/glm52-baseline-runtime
ART="$ARTIFACT_LOCAL_DIR/glm52-e2e/rank-$FALCON_RANK"
mkdir -p "$ART" "$RUN_DIR" /tmp/tpu_logs/tmp
rm -f "$RUN_DIR/SETUP_READY"
exec > >(tee -a "$ART/driver.log") 2>&1
export TMPDIR="$RUN_DIR/tmp" PIP_NO_CACHE_DIR=1
mkdir -p "$TMPDIR"
export PATH="$RUN_DIR/venv/bin:$PATH"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
report_failure() {
  code=${1:-$?}
  if (( code != 0 )); then
    for log in "$ART/pip-requirements.log" "$ART/pip-editable.log" "$ART/apt.log"; do
      if [[ -f "$log" ]]; then echo "SETUP_FAILURE_LOG $log"; tail -n 100 "$log"; fi
    done
  fi
  exit "$code"
}
trap report_failure EXIT
cd "$SRC"
test "$(git rev-parse HEAD)" = "$SOURCE_COMMIT"
python3 -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
python3 -m venv "$RUN_DIR/venv"
PY="$RUN_DIR/venv/bin/python"
"$PY" -m pip install -q --upgrade pip setuptools setuptools-scm wheel
WHEEL=/wheels/torch-tpu/torch_tpu-0.2.1.dev20261028201649-cp312-cp312-manylinux_2_31_x86_64.whl
printf '%s  %s\n' 6e12816e92f9a5d9649358541f332f06583ccdfee5b6a74434dbf2141757fd82 "$WHEEL" | sha256sum -c -
"$PY" - "$WHEEL" <<'PY' > "$RUN_DIR/requirements.txt"
import pathlib, re, sys, tomllib
for dep in tomllib.loads(pathlib.Path('python/pyproject_tpu.toml').read_text())['project']['dependencies']:
    if re.match(r'torch[_-]tpu\b', dep):
        assert '-'+dep.split('==')[1]+'-' in pathlib.Path(sys.argv[1]).name
        print(sys.argv[1])
    else:
        print(dep)
PY
"$PY" -m pip install --find-links=/prepared/wheelhouse --extra-index-url https://download.pytorch.org/whl/cpu -r "$RUN_DIR/requirements.txt" > "$ART/pip-requirements.log" 2>&1
cp python/pyproject.toml "$RUN_DIR/pyproject.original"
restore_pyproject() { cp "$RUN_DIR/pyproject.original" "$SRC/python/pyproject.toml"; }
trap 'code=$?; restore_pyproject; report_failure "$code"' EXIT
cp python/pyproject_tpu.toml python/pyproject.toml
"$PY" -m pip install --no-deps --no-build-isolation -e python -e . > "$ART/pip-editable.log" 2>&1
restore_pyproject
if ! ldconfig -p | awk '/libibverbs.so.1/ {found=1} END {exit !found}'; then
    apt-get update -qq > "$ART/apt.log" 2>&1
    apt-get install -y -qq libibverbs1 librdmacm1 >> "$ART/apt.log" 2>&1
fi
export TPU_LIBRARY_PATH
TPU_LIBRARY_PATH=$("$PY" -c 'import libtpu; print(libtpu.get_library_path())')
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:-} --xla_tpu_buffer_assignment_enable_fast_merge_fallback=false"
export SGLANG_PLATFORM=tpu
export GLM52_ART="$ART"
"$PY" - <<'PY'
import hashlib, importlib.metadata as m, json, os, pathlib, socket, sys, tomllib
pins={}
for dep in tomllib.loads(pathlib.Path('python/pyproject_tpu.toml').read_text())['project']['dependencies']:
    name,_,version=dep.partition('==')
    if name in ('torch','torch_tpu','torchvision','jax','jaxlib','libtpu','tpu-sync-torch'): pins[name]=version
actual={name:m.version(name) for name in pins}
assert all(actual[n].split('+')[0]==v for n,v in pins.items()),(pins,actual)
a=pathlib.Path(os.environ['GLM52_ART'])
e={'executable':sys.executable,'python':sys.version,'declared':pins,'actual':actual,'hostname':socket.gethostname(),'ip':socket.gethostbyname(socket.gethostname()),'env':{k:v for k,v in os.environ.items() if k.startswith(('TPU_','FALCON_JAX','MEGASCALE','LIBTPU','XLA_'))}}
(a/'setup-runtime.json').write_text(json.dumps(e,indent=2))
p=pathlib.Path('/models/GLM5.2-fp8-channel-wise')
meta={}
for name in ('config.json','model.safetensors.index.json','tokenizer_config.json'):
 raw=(p/name).read_bytes();meta[name]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
 (a/name).write_bytes(raw)
expected={
 'config.json':'0735624bbd5666283a1f135c59ff4c15e8b51c56fc50e022181525253f29463b',
 'model.safetensors.index.json':'0ee9d61ce4e0ca912c2a5b8220f57b1060fb95ec09304da77c85a43d56aeb2a0',
}
assert all(meta[n]['sha256']==v for n,v in expected.items()),meta
(a/'checkpoint-identity.json').write_text(json.dumps(meta,indent=2))
print('SETUP_READY',json.dumps(e),flush=True)
PY
cat > "$RUN_DIR/setup.sh" <<EOF
export SRC='$SRC' PY='$PY' GLM52_ART='$ART'
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export TMPDIR='$RUN_DIR/tmp' XLA_PYTHON_CLIENT_PREALLOCATE=false
export TPU_LIBRARY_PATH='$TPU_LIBRARY_PATH' LIBTPU_INIT_ARGS='$LIBTPU_INIT_ARGS' SGLANG_PLATFORM=tpu
cd '$SRC'
EOF
touch "$RUN_DIR/SETUP_READY"
echo BOOTSTRAP_PASS
