#!/usr/bin/env bash
# Lint the fork's TPU workflows with actionlint and zizmor.
#
# Scope: workflows this fork owns (TPU CI, the upstream sync bot, the radixark
# image). Upstream-owned workflows are left to upstream.
#
# Fails on actionlint errors, shellcheck warnings and zizmor findings of high
# severity. Lower-severity zizmor findings are printed but do not fail.
#
# Usage: scripts/ci/tpu/check_workflows.sh   (needs actionlint, shellcheck, zizmor)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

files=()
for f in .github/workflows/*.yml; do
  case "$(basename "$f")" in
    *tpu*|bot-sync-from-upstream.yml|release-docker-radixark.yml) files+=("$f") ;;
  esac
done
echo "Checking ${#files[@]} workflows:"
printf '  %s\n' "${files[@]}"

rc=0

echo "::group::actionlint"
SHELLCHECK_OPTS="--severity=warning" actionlint "${files[@]}" || rc=1
echo "::endgroup::"

echo "::group::zizmor (all findings, informational)"
zizmor --config .github/zizmor.yml --min-severity low "${files[@]}" || true
echo "::endgroup::"

echo "zizmor (high severity, blocking)"
zizmor --config .github/zizmor.yml --min-severity high "${files[@]}" || rc=1

exit "$rc"
