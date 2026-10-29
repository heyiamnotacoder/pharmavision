#!/usr/bin/env bash
# SessionStart hook (see .claude/settings.json). Cloud sessions only: bring the clone's
# environment in line with the lockfiles. The environment setup script has already warmed the
# package caches, so this mostly links files and takes seconds.
[[ "${CLAUDE_CODE_REMOTE:-}" == "true" ]] || exit 0
cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0

uv sync --locked --quiet || echo "session-start: uv sync failed; run 'uv sync --locked' to see why" >&2

if [[ -f frontend/package-lock.json ]]; then
  stamp=frontend/node_modules/.session-start.sha256
  if ! sha256sum --check --status "$stamp" 2>/dev/null; then
    rm -f "$stamp"
    npm --prefix frontend ci --no-audit --no-fund --silent \
      && sha256sum frontend/package.json frontend/package-lock.json > "$stamp" \
      || echo "session-start: npm ci failed; run it by hand to see why" >&2
  fi
fi
exit 0
