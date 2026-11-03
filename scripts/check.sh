#!/usr/bin/env bash
# Every gate CI runs, in the same order. Run before pushing; CI runs exactly this.
# Usage: scripts/check.sh           (lint, format check, typecheck, tests)
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

echo "▶ python: ruff lint";   uv run --locked ruff check .
echo "▶ python: ruff format"; uv run --locked ruff format --check .
echo "▶ python: pyright";     uv run --locked pyright
echo "▶ python: pytest";     uv run --locked pytest

# `npm ci` needs the lockfile, so the lockfile marks the frontend as present (CI keys on it too).
if [[ -f frontend/package-lock.json ]]; then
  echo "▶ frontend: install";   npm --prefix frontend ci --no-audit --no-fund
  echo "▶ frontend: typecheck"; npm --prefix frontend run typecheck
  echo "▶ frontend: lint";      npm --prefix frontend run lint
  echo "▶ frontend: format";    npm --prefix frontend run format:check
  echo "▶ frontend: test";      npm --prefix frontend run test --if-present
  echo "▶ frontend: build";     npm --prefix frontend run build
fi
echo "✔ all checks passed"
