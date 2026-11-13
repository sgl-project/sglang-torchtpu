# Recover exact TorchTPU runtime using successful Falcon tasks

## Source of executable configuration

Inspected through `falcon exp list` and `falcon exp get`, without provider/API bypass:

- `exp-bsvzoblh0d`: SUCCEEDED, sglang-torchtpu source `56522deaae06e6308e9f70775f4273faf43e2eef`; runtime bootstrap uses JAX AI base and a fresh Python3.12 venv. Its serving log ends with200/200 GSM8K completions,198correct,zero failed,`E2E_DONE rc=0`. This validates a runtime/bootstrap route, NOT GLM52 correctness or performance.
- `exp-h8vu1379oa`: SUCCEEDED, staged exact torch_tpu wheel and verified its hash in Falcon artifact.
- `exp-8tku39alke`: SUCCEEDED, prepared dependency wheelhouse mounted read-only by the serving task.
- `exp-d9kadleozc`: SUCCEEDED Qwen export task also inspected; it uses older public release image and syncs JAX/JAXLIB selectively. We choose the first task's exact full stack, not that partial-sync route.

## No private pip index needed

Base image is `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.9.0-rev1`; this is a bootstrap base, NOT the runtime version. Install all exact project pins in a fresh venv.

Read-only Falcon mounts copied from successful task:

- `/wheels/torch-tpu`: bucket `tpu-for-training-falcon-logs`, prefix `experiments/exp-h8vu1379oa/artifacts/art-6jnw46ypty/wheels/cp312-linux-x86_64`.
- `/prepared`: same bucket, prefix `experiments/exp-8tku39alke/artifacts/art-hakzf5m8u1/glm53-prepared`; pip uses `/prepared/wheelhouse` as find-links, plus public PyPI and PyTorch CPU index.

Wheel `torch_tpu-0.2.1.dev20261028201649-cp312-cp312-manylinux_2_31_x86_64.whl` must hash to `6e12816e92f9a5d9649358541f332f06583ccdfee5b6a74434dbf2141757fd82` before install.

Required stack remains torch2.11.0 CPU ABI, torch_tpu0.2.1.dev20261028201649, tpu-sync-torch0.0.1.dev20261031081414, JAX/JAXLIB0.11.2, libtpu0.0.50. `tpu-sync-torch` is available publicly. Install libibverbs1/librdmacm1 as the peer task does.

## New actual verification

`exp-ldtcncqheb` / `art-oxwfsxmg4o`: submitted CPU-only bootstrap verification using the above mounts and peer setup, epic source `dd312270841887966b5480be2f451f6c19de1496`, then the nine actual CPU loader tests. It progressed to RUNNING/INSTALL. Actual pip log confirms mounted torch_tpu wheel processing, public JAX/JAXLIB0.11.2 downloads, and installation of the complete dependency set without private-index401/403. Final runtime/test status still requires collection; no pass claimed here.

Previous private-index401 remains a fact about that failed install route, **not a blocker of the whole Torch baseline objective**. The successful peer route supersedes that conclusion.

## Metric bookkeeping correction

Recorded JAX result contains **8** qualified intervals,4per wave across two DP16 waves for C32. Earlier prose said4total; the JSON samples and median1242.394121ms were correct. This does not turn a single run into multi-run stability acceptance.
