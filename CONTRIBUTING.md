# Contributing to SGLang-torchtpu

Welcome. This page gets you from a clone to a merged pull request. SGLang's [contribution guide](https://docs.sglang.io/developer_guide/contribution_guide.html) applies here too; this page covers what is different on TPU.

Questions at any point: open a [GitHub Discussion](<discussions url placeholder>) or ask in the SGLang Slack channel `<channel placeholder>`.

## Set up

Clone the repository and install both distributions from source:

```bash
git clone https://github.com/sgl-project/sglang-torchtpu.git && cd sglang-torchtpu
pip install -e python/          # sglang (the engine)
pip install --no-deps -e .      # sglang-tpu (the TPU plugin)
pip install pre-commit && pre-commit install
```

On a TPU host, the TPU wheels (`torch_tpu`, `libtpu`) come from the pins in `python/pyproject_tpu.toml`. On a laptop, skip them: everything in the next section marked CPU runs without a TPU.

If those pins require a separate package index, configure it outside the repository:

```bash
export UV_INDEX="torch-tpu-registry=https://YOUR_INDEX_HOST/simple/"
# For an authenticated index, use uv's credential provider or set:
export UV_INDEX_TORCH_TPU_REGISTRY_USERNAME="YOUR_INDEX_USERNAME"
# Set UV_INDEX_TORCH_TPU_REGISTRY_PASSWORD securely in your environment.
```

See [uv's package index documentation](https://docs.astral.sh/uv/concepts/indexes/)
and [TPU CI deployment configuration](.github/tpu-ci/README.md#deployment-configuration).


## Run tests

| What | Command | Needs |
| --- | --- | --- |
| CPU unit tests (platform logic, compile policies, spec utils, offload bookkeeping) | `python3 test/run_suite.py --hw cpu --suite base-a-test-cpu` | CPU |
| One test file, the way CI runs it | `python3 test/registered/unit/hardware_backend/tpu/<file>.py` | CPU or TPU, per file |
| Kernel correctness suite | `python3 test/run_suite.py --hw tpu --suite stage-a-test-tpu-unit` | TPU |
| Single-host end-to-end (the PR gate) | `python3 test/run_suite.py --hw tpu --suite stage-b-test-tpu-4-v7` | 4-chip TPU7x host |
| Launch a server and benchmark it | see [benchmark/tpu/README.md](benchmark/tpu/README.md) | TPU |

Every TPU test file starts with a `register_tpu_ci(...)` marker that tells CI where it runs and tells the support matrix what it proves:

```python
register_tpu_ci(
    est_time=1800,
    suite="stage-b-test-tpu-4-v7",
    labels=["singlehost", "gpt-oss-20b", "spec", "dpa"],   # topology, model, features
)
```

Labels are required and come from `scripts/ci/tpu/support_matrix_vocab.json`; add new ones there in the same PR. `SUPPORT_MATRIX.md` is generated from these markers, so a feature counts as supported only when a registered test backs it. Conventions for test layout and `CustomTestCase` are in [test/registered/README.md](test/registered/README.md).

## Send a pull request

1. Most contributors do not have push access, so [fork the repository](https://github.com/sgl-project/sglang-torchtpu/fork), add it as a remote, and push your branch there:

   ```bash
   git remote add fork https://github.com/<your_user_name>/sglang-torchtpu.git
   git checkout -b my-change && git push -u fork my-change
   ```

   Then open the PR against `sgl-project/sglang-torchtpu:main`. One concern per PR.
2. Run `pre-commit run --all-files`. It runs ruff, isort, codespell, and the TPU debt ratchet (see Architecture below).
3. Review your own diff against [AGENTS.md](AGENTS.md). It is the checklist maintainers and their coding agents apply, including the plugin-migration rules; pointing your own agent at it before you open the PR catches placement problems earlier than CI does.
4. Title with an area prefix: `[TPU]` backend code, `[TPU/CI]` CI and tooling, `[Docs]`, `[Upstream]` for a change also open against sgl-project/sglang (link it).
5. Fill in the PR template as [REVIEW.md](REVIEW.md) describes. The description becomes the squash-merge commit message, so keep it accurate.
6. Lint and CPU tests run automatically on every PR, from forks included, with no secrets involved.
7. **TPU tests run on request**, because TPU capacity is limited and fork PRs cannot reach the TPU runners on their own. An authorized user adds the `run-ci` label after reading the diff and comments `/tag-and-rerun-ci` on the PR; that runs the TPU suite once for the current commit. A new push needs a new comment. `/rerun-failed-ci` reruns what failed. Only users listed in `.github/CI_PERMISSIONS.json` can run CI commands. A PR author can also rerun failed CI on their own PR with `/rerun-failed-ci`. Ask for the run in a PR comment once the CPU checks are green.
8. CODEOWNERS assigns reviewers for TPU paths. One approval plus code-owner review, all threads resolved, then a maintainer squash-merges.

A new kernel ships with a correctness test against a reference. A new feature ships with a registered e2e test. Disabling a test is a registration change, `disabled="<reason, with an issue link>"`, and the test stays visible in the matrix as Disabled.

## File an issue

Use the [bug report template](https://github.com/sgl-project/sglang-torchtpu/issues/new/choose). It asks for what we need to reproduce on TPU:

- chip generation and topology (single-host, 2 hosts, 2P+2D);
- `torch_tpu` and `libtpu` versions and the sglang-torchtpu commit;
- the full launch command and the model;
- the error, and whether the same command works on a smaller model such as gpt-oss-20b or Qwen3-0.6B.

Engine bugs that reproduce on GPU belong in [sgl-project/sglang](https://github.com/sgl-project/sglang/issues). Security issues go through [SECURITY.md](SECURITY.md), never a public issue. Feature requests are welcome as issues; check [SUPPORT.md](SUPPORT.md) first, since the feature may already exist at an Experimental tier.

## Community

- Slack: `<channel placeholder>` in the SGLang workspace, for questions and quick help.
- [GitHub Discussions](<discussions url placeholder>) for design threads and longer questions.
- Issues labeled `good first issue` are scoped for newcomers and come with a maintainer willing to pair.

## Architecture

### Layout

| Path | What lives there |
| --- | --- |
| `sglang_tpu/` | The TPU backend: platform classes, Pallas attention and MoE kernels, KV pools, raiden and native PD transports, compile policies, graph runners, hooks. **All new TPU code goes here.** |
| `python/` | A fork of sgl-project/sglang, synced weekly from upstream. Carries the TPU branches that core still needs. |
| `test/registered/tpu/`, `test/registered/unit/hardware_backend/tpu/` | TPU tests. |
| `benchmark/tpu/` | Launch recipes, benchmark matrix, per-model configs. |
| `scripts/ci/tpu/`, `.github/tpu-ci/`, `tools/plugin/` | CI plumbing, the support-matrix generator, migration tooling. |

### The plugin and the migration

SGLang discovers hardware backends through the `sglang.srt.platforms` entry point. `sglang_tpu/` is that plugin: `pip install sglang sglang-tpu` is the target install, with nothing TPU-specific inside the `sglang` wheel.

Today the backend still runs on the fork in `python/`, because some TPU behaviour lives as `is_tpu()` branches inside shared engine code. The migration moves each of those branches out, either into the plugin through an extension point upstream already has, or by adding a small extension point ("seam") upstream first. `tools/plugin/scoreboard.py` counts the remaining branches, imports, patches, and hooks, and the daily TPU Plugin Scoreboard workflow publishes it. **Those counts may only go down.** When they reach zero, `python/` is replaced by a pinned upstream release.

What this means for a change:

- **TPU-only code** (kernels, pools, backends, recipes, tests) goes under `sglang_tpu/`. Never under `python/sglang/srt/`, and never into one of the old re-export shim files there.
- **Generic engine changes** go to upstream first and arrive here through the sync. If you cannot wait, open the upstream PR, then the same change here with the `[Upstream]` prefix.
- **A core change that TPU needs** uses the first of these that fits: an existing `SRTPlatform` method or capability flag; a registry filled from `init_backend()` (`register_attention_backend`, `register_oot_forward`, `get_quantization_config`); subclass-and-override through a factory; a seam PR upstream, carried here as a `sglang_tpu/hooks/` entry until it merges. Seam PRs stay under about 300 lines, carry no TPU code, and ship with a CPU unit test. The hook header links the seam PR, and a cleanup PR deletes the hook after the sync brings the seam in.
- **The ratchet** (`tools/plugin/check_tpu_branches.py`, run by pre-commit) rejects any PR that raises a per-file count of `is_tpu` branches or `sglang_tpu` imports in core. Raising a baseline needs a linked upstream seam issue or PR and a maintainer's approval. Lowering one is always welcome: run it with `--write-baseline` and commit the JSON.

Details, mode A versus mode B, and the list of known divergences: [sglang_tpu/README.md](sglang_tpu/README.md).

Code style follows upstream's [code style guidance](https://docs.sglang.io/developer_guide/contribution_guide.html#code-style-guidance); the review rules in [AGENTS.md](AGENTS.md) are what a PR is checked against.
