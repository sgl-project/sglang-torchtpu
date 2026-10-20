#!/bin/bash
set -euo pipefail

# Merge upstream sgl-project/sglang `main` into this fork and open a PR.
#
# Unlike scripts/code_sync/copy_from_oss.py (which rsyncs/overwrites specific
# folders), this preserves upstream git history by doing a real `git merge`.
#
# Intended to run from a GitHub Actions job on the `main` branch with full
# history (actions/checkout fetch-depth: 0).
#
# Requirements:
#   - git and gh (GitHub CLI) on PATH
#   - GH_TOKEN env var: a PAT allowed to push branches and open PRs
#     (we pass secrets.GH_PAT_FOR_PULL_REQUEST in the workflow)

UPSTREAM_URL="https://github.com/sgl-project/sglang.git"
UPSTREAM_BRANCH="main"
BASE_BRANCH="main"
REPO="sgl-project-dev/sglang-torchtpu"

git config user.name "sglang-bot"
git config user.email "sglang-bot@users.noreply.github.com"

# (Re)create the upstream remote and fetch its main branch.
git remote remove upstream 2>/dev/null || true
git remote add upstream "$UPSTREAM_URL"
git fetch upstream "$UPSTREAM_BRANCH" --no-tags

git checkout "$BASE_BRANCH"

# Bail out early if there is nothing new upstream. After a previous sync PR is
# merged, those commits live in main's history, so they stop being counted.
AHEAD=$(git rev-list --count "${BASE_BRANCH}..upstream/${UPSTREAM_BRANCH}")
if [ "$AHEAD" -eq 0 ]; then
  echo "Already up to date with upstream/${UPSTREAM_BRANCH}. Nothing to do."
  [ -n "${GITHUB_STEP_SUMMARY:-}" ] && \
    echo "Already up to date with upstream/${UPSTREAM_BRANCH}." >> "$GITHUB_STEP_SUMMARY"
  exit 0
fi

UP_SHA=$(git rev-parse --short "upstream/${UPSTREAM_BRANCH}")
DATE=$(date +%Y%m%d)
# Use a deterministic branch name per upstream commit to prevent duplicate PRs.
BRANCH="bot/sync-from-upstream-${UP_SHA}"

echo "${AHEAD} new upstream commit(s); upstream/${UPSTREAM_BRANCH} is at ${UP_SHA}."

# Skip if an open PR for this exact branch already exists (re-run safety).
if [ -n "$(gh pr list --repo "$REPO" --head "$BRANCH" --state open --json number --jq '.[].number')" ]; then
  echo "An open PR for $BRANCH already exists. Skipping."
  exit 0
fi

git checkout -B "$BRANCH" "$BASE_BRANCH"

# The CI allowlist is repository policy, not an upstream contributor ranking.
# Restore it from the local base before committing even when upstream's change
# would merge cleanly. Other merge conflicts keep the existing review workflow.
if ! git merge --no-commit --no-ff "upstream/${UPSTREAM_BRANCH}"; then
  git rev-parse --verify MERGE_HEAD >/dev/null || exit 1
fi
git restore --source="$BASE_BRANCH" --staged --worktree -- .github/CI_PERMISSIONS.json
CONFLICTS=$(git diff --name-only --diff-filter=U || true)
if [ -n "$CONFLICTS" ]; then
  echo "Merge has conflicts in:"
  echo "$CONFLICTS"
  git add -A
else
  echo "Clean merge of upstream/${UPSTREAM_BRANCH} (${UP_SHA}); local CI allowlist preserved."
fi
git commit --no-edit

git push origin "$BRANCH" --force

# Build the PR body.
COMMIT_RANGE_URL="https://github.com/sgl-project/sglang/commits/${UPSTREAM_BRANCH}"
if [ -n "$CONFLICTS" ]; then
  TITLE="[Automated][NEEDS CONFLICT RESOLUTION] Merge upstream sglang ${UP_SHA} ($DATE)"
  CONFLICT_SECTION=$(printf '\n## Conflicts to resolve\nThis merge had conflicts. The branch was committed **with conflict markers**. Check out `%s`, resolve them, and push before merging:\n\n```\n%s\n```\n' "$BRANCH" "$CONFLICTS")
else
  TITLE="[Automated] Merge upstream sglang ${UP_SHA} ($DATE)"
  CONFLICT_SECTION=""
fi

# Operating instructions shown on every sync PR so reviewers merge it correctly.
HOWTO_SECTION=$(printf '\n## How to merge this PR\n- **Use "Create a merge commit"** — do **not** squash or rebase. This is a merge-based sync that must preserve upstream git history; squashing collapses the upstream commits into one and rebasing rewrites them, either of which breaks future incremental syncs.\n- **Add the `run-ci` label** to trigger PR CI (after resolving any conflicts), then **fix any new test failures** before merging.\n')

BODY=$(printf 'Merges %s new commit(s) from [sgl-project/sglang@%s](https://github.com/sgl-project/sglang/commit/%s) into `%s`.\n%s%s\nUpstream commits: %s\n\n---\n*Automated by `scripts/code_sync/merge_from_upstream.sh` (.github/workflows/bot-sync-from-upstream.yml).*' \
  "$AHEAD" "$UP_SHA" "$UP_SHA" "$BASE_BRANCH" "$CONFLICT_SECTION" "$HOWTO_SECTION" "$COMMIT_RANGE_URL")

PR_URL=$(gh pr create \
  --repo "$REPO" \
  --base "$BASE_BRANCH" \
  --head "$BRANCH" \
  --title "$TITLE" \
  --body "$BODY")

echo "Opened PR: $PR_URL"
if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
  {
    echo "## Upstream sync PR opened"
    echo "$PR_URL"
    [ -n "$CONFLICTS" ] && echo -e "\n**Has conflicts** - manual resolution needed."
  } >> "$GITHUB_STEP_SUMMARY"
fi
