#!/usr/bin/env bash
#
# One-shot setup for the selective CD in .github/workflows/cd.yml.
#
# It automates every mechanical step so the only things left to you are the two that
# CAN'T be automated: an interactive `gh auth login`, and one `sudo` to install the
# runner as a background service.
#
# What it does, in order:
#   1. Ensure the GitHub CLI (`gh`) is installed and you're authenticated.
#   2. Ensure this repo exists on GitHub and the code is pushed (creates it if needed).
#   3. Clone a DEDICATED deploy checkout (DEPLOY_DIR) and set the repo variable DEPLOY_DIR
#      that cd.yml reads. (Dedicated on purpose: the deploy step runs `git checkout --force`.)
#   4. Download, register and start a self-hosted runner (label `self-hosted`) on THIS host.
#
# Idempotent-ish: re-running skips steps already done. Override any default via env vars, e.g.
#   DEPLOY_DIR=/srv/delivery-risk-mlops RUNNER_DIR=$HOME/actions-runner ./scripts/setup-self-hosted-runner.sh
#
set -euo pipefail

# ── Config (override via env) ──────────────────────────────────────────────────────────
DEPLOY_DIR="${DEPLOY_DIR:-$HOME/deploy/delivery-risk-mlops}"
RUNNER_DIR="${RUNNER_DIR:-$HOME/actions-runner}"
RUNNER_LABELS="${RUNNER_LABELS:-self-hosted}"
REPO_SLUG="${REPO_SLUG:-}"          # owner/name; auto-detected or you're prompted to create it
REPO_VISIBILITY="${REPO_VISIBILITY:-private}"

say()  { printf '\n\033[1;36m>> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m!! %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31mxx %s\033[0m\n' "$*" >&2; exit 1; }

# ── 1. gh installed + authenticated ────────────────────────────────────────────────────
say "Step 1/4 — GitHub CLI"
if ! command -v gh >/dev/null 2>&1; then
  warn "gh not found — installing via apt (needs sudo)."
  command -v apt-get >/dev/null || die "Non-apt system: install gh manually (https://github.com/cli/cli#installation), then re-run."
  sudo mkdir -p -m 755 /etc/apt/keyrings
  wget -qO- https://cli.github.com/packages/githubcli-archive-keyring.gpg \
    | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg >/dev/null
  sudo chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
    | sudo tee /etc/apt/sources.list.d/github-cli.list >/dev/null
  sudo apt-get update -qq && sudo apt-get install -y gh
fi
if ! gh auth status >/dev/null 2>&1; then
  warn "Not authenticated. Launching interactive login (this is one of the two manual steps)."
  gh auth login
fi
gh auth status >/dev/null 2>&1 || die "gh is still not authenticated."
echo "gh ready as: $(gh api user -q .login)"

# ── 2. GitHub repo exists + code pushed ────────────────────────────────────────────────
say "Step 2/4 — GitHub repository"
if [ -z "$REPO_SLUG" ]; then
  REPO_SLUG="$(gh repo view --json nameWithOwner -q .nameWithOwner 2>/dev/null || true)"
fi
if [ -z "$REPO_SLUG" ]; then
  warn "No GitHub repo is linked (your origin is a local bundle)."
  DEFAULT_NAME="$(basename "$(git rev-parse --show-toplevel)")"
  read -r -p "Create a new $REPO_VISIBILITY GitHub repo named [$DEFAULT_NAME]? (name / Enter to accept / Ctrl-C to abort): " NAME
  NAME="${NAME:-$DEFAULT_NAME}"
  # Creates the repo, adds it as remote `github`, and pushes the current branch.
  gh repo create "$NAME" "--$REPO_VISIBILITY" --source=. --remote=github --push
  REPO_SLUG="$(gh repo view --json nameWithOwner -q .nameWithOwner)"
else
  echo "Using existing repo: $REPO_SLUG"
  # Make sure what's on this machine is actually pushed, or cd.yml validates stale code.
  warn "Ensure your latest commits are pushed to $REPO_SLUG before relying on CD."
fi
REPO_URL="https://github.com/$REPO_SLUG"
echo "Repo: $REPO_URL"

# ── 3. DEPLOY_DIR clone + repo variable ────────────────────────────────────────────────
say "Step 3/4 — deploy checkout + DEPLOY_DIR variable"
if [ ! -d "$DEPLOY_DIR/.git" ]; then
  mkdir -p "$(dirname "$DEPLOY_DIR")"
  git clone "$REPO_URL" "$DEPLOY_DIR"
else
  echo "Deploy checkout already present at $DEPLOY_DIR"
fi
gh variable set DEPLOY_DIR --repo "$REPO_SLUG" --body "$DEPLOY_DIR"
echo "Set repo variable DEPLOY_DIR=$DEPLOY_DIR"

# ── 4. Self-hosted runner ──────────────────────────────────────────────────────────────
say "Step 4/4 — self-hosted runner"
if [ -f "$RUNNER_DIR/.runner" ]; then
  echo "A runner is already configured in $RUNNER_DIR — skipping registration."
else
  mkdir -p "$RUNNER_DIR"; cd "$RUNNER_DIR"
  RUNNER_VER="$(gh api repos/actions/runner/releases/latest -q .tag_name | sed 's/^v//')"
  TARBALL="actions-runner-linux-x64-${RUNNER_VER}.tar.gz"
  if [ ! -f "$TARBALL" ]; then
    echo "Downloading runner v${RUNNER_VER}..."
    curl -fsSL -o "$TARBALL" \
      "https://github.com/actions/runner/releases/download/v${RUNNER_VER}/${TARBALL}"
  fi
  tar xzf "$TARBALL"
  # Short-lived registration token minted via the API (needs admin on the repo).
  REG_TOKEN="$(gh api -X POST "repos/$REPO_SLUG/actions/runners/registration-token" -q .token)"
  ./config.sh --url "$REPO_URL" --token "$REG_TOKEN" \
    --labels "$RUNNER_LABELS" --name "$(hostname)-cd" --unattended --replace
  warn "Installing the runner as a service (the second manual step — needs sudo)."
  sudo ./svc.sh install "$(whoami)"
  sudo ./svc.sh start
fi

say "Done."
cat <<EOF
Self-hosted CD is set up:
  repo         : $REPO_URL
  runner       : $RUNNER_DIR  (label: $RUNNER_LABELS, service running)
  DEPLOY_DIR   : $DEPLOY_DIR  (repo variable set)

Verify:
  - Runner shows 'Idle': $REPO_URL/settings/actions/runners
  - Variable present    : gh variable list --repo $REPO_SLUG
  - Trigger it: push a change under app/**, pipeline/**, or grafana/** to main and watch
    $REPO_URL/actions
EOF
