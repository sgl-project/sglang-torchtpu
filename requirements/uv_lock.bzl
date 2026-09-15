"""Regenerate the uv requirements lock as a Bazel action.

The compile is a real build action: the uv binary comes from the hermetic
multitool toolchain (pinned by requirements/uv.lock.json in MODULE.bazel) and
Bazel stages it like any other tool, so the resolve is identical on every
machine. The lock itself is checked in, so `bazel run
//requirements:regenerate_lock` runs a small runner that copies the compiled
lock back into the workspace.
"""

# The runner is only ever executed by `bazel run`, which sets the working
# directory to this target's runfiles workspace dir -- so the compiled lock,
# a runfile of this target, sits at exactly its runfiles-relative path, and
# BUILD_WORKSPACE_DIRECTORY points back at the source checkout.
_RUNNER = """\
#!/usr/bin/env bash
set -euo pipefail

cp {lock_runfiles_path} "$BUILD_WORKSPACE_DIRECTORY/{out_workspace}"
echo "wrote $BUILD_WORKSPACE_DIRECTORY/{out_workspace}"
"""


def _uv_lock_impl(ctx):
    # `uv pip compile` insists each input is named exactly `pyproject.toml`,
    # so stage inputs under that name in per-input scratch dirs.
    staged = []
    for i, src in enumerate(ctx.files.pyprojects):
        out = ctx.actions.declare_file("%s.scratch/%d/pyproject.toml" % (ctx.label.name, i))
        if i == 0:
            ctx.actions.symlink(output = out, target_file = src)
        else:
            # --extra applies to every input pyproject. Only the primary
            # project's test suite is under test here; dependency projects'
            # optional suites can impose unrelated, conflicting version pins.
            # Hide their optional-dependencies table from uv in the staged copy.
            ctx.actions.run_shell(
                inputs = [src],
                outputs = [out],
                arguments = [src.path, out.path],
                command = "sed 's/^\\([[:space:]]*\\)\\[project\\.optional-dependencies\\]/\\1[tool.bazel.unused-optional-dependencies]/' \"$1\" > \"$2\"",
                mnemonic = "StageRuntimePyproject",
            )
        staged.append(out)

    lock = ctx.actions.declare_file(ctx.label.name + ".lock")
    cache = ctx.actions.declare_directory(ctx.label.name + ".uv_cache")
    args = ctx.actions.args()
    args.add("pip")
    args.add("compile")
    args.add_all([s.path for s in staged])
    for extra in ctx.attr.extras:
        args.add("--extra")
        args.add(extra)
    args.add("--excludes")
    args.add(ctx.file.excludes.path)
    args.add("--python-version")
    args.add(ctx.attr.python_version)
    # Resolve against the pinned Bazel interpreter, not whatever is on PATH.
    python = ctx.file._python
    args.add("--python")
    args.add(python.path)
    args.add("--index-url")
    args.add("https://pypi.org/simple")
    args.add("--extra-index-url")
    args.add("https://download.pytorch.org/whl/cpu")
    args.add("--index-strategy")
    args.add("unsafe-best-match")
    args.add("--output-file")
    args.add(lock.path)
    ctx.actions.run(
        executable = ctx.executable._uv,
        arguments = [args],
        inputs = staged + [ctx.file.excludes, python],
        outputs = [lock, cache],
        # HOME is not writable in the sandbox; keep uv's cache and managed
        # downloads inside it.
        env = {
            "UV_CACHE_DIR": cache.path,
            "UV_PYTHON_INSTALL_DIR": cache.path + "/python",
        },
        mnemonic = "UvPipCompile",
        progress_message = "uv pip compile %s" % ctx.files.pyprojects[0].short_path,
    )

    runner = ctx.actions.declare_file(ctx.label.name + ".sh")
    ctx.actions.write(
        output = runner,
        content = _RUNNER.format(
            # Runfiles path of the lock, relative to the workspace dir that
            # `bazel run` cd's into: generated files land under their package.
            lock_runfiles_path = ctx.label.package + "/" + lock.basename,
            out_workspace = ctx.attr.out,
        ),
        is_executable = True,
    )

    return [DefaultInfo(
        executable = runner,
        runfiles = ctx.runfiles(files = [lock]),
    )]


uv_lock = rule(
    implementation = _uv_lock_impl,
    attrs = {
        "pyprojects": attr.label_list(
            allow_files = True,
            mandatory = True,
            doc = """The pyproject files to resolve together. The first is the
primary project; additional ones (e.g. @torch_tpu//:pyproject) add their
runtime dependency sets to the same resolution so the root gets one mediated lock.""",
        ),
        "excludes": attr.label(
            allow_single_file = True,
            mandatory = True,
            doc = "Package excludes file passed to `uv pip compile --excludes`.",
        ),
        "out": attr.string(
            mandatory = True,
            doc = "Workspace-relative path `bazel run` writes the compiled lock to.",
        ),
        "extras": attr.string_list(
            default = ["test"],
            doc = "Extras to resolve from the first pyproject only (`--extra`).",
        ),
        "python_version": attr.string(
            default = "3.12",
            doc = "--python-version for the resolve.",
        ),
        "_uv": attr.label(
            default = Label("@multitool//tools/uv"),
            executable = True,
            cfg = "exec",
        ),
        "_python": attr.label(
            # The interpreter uv resolves against: the same pinned 3.12 the
            # whole build uses. A single file (the toolchain python3).
            default = Label("@python_3_12//:python3"),
            allow_single_file = True,
            cfg = "exec",
        ),
    },

    executable = True,
)
