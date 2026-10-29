#!/bin/bash
# PharmaVision: cloud environment setup script.
#
# Paste this whole file into claude.ai/code → your environment → Setup script (or "Choose file").
# It runs as root on Ubuntu 24.04 before Claude Code starts. The result is snapshotted and reused
# for about 7 days, so later sessions start with everything on disk. Rules from Anthropic's docs:
# finish in under ~5 minutes or the snapshot isn't kept, and always exit 0 or the session fails.
# Environment variables you set on the environment are NOT available yet when this runs.
#
# What it does:
#   1. apt: ffmpeg (ingest and clip cutting), Python 3.12 if the image lacks it
#   2. Warms uv's cache with the project's Python dependencies and pyright's node server, so the
#      SessionStart `uv sync --locked` only links files
#   3. Writes ~/.claude/CLAUDE.md with your personal rules (your laptop's ~/.claude doesn't reach
#      cloud sessions)
#
# Keep the dependency list in step 2 roughly in line with pyproject.toml. Drift only costs time:
# `uv sync` downloads whatever the cache lacks.
set -u
log() { echo "[pharmavision-setup] $*"; }

# 1. System packages (one apt run; parallel apt runs fight over the lock)
export DEBIAN_FRONTEND=noninteractive
apt_pkgs=(ffmpeg)
command -v python3.12 >/dev/null || apt_pkgs+=(python3.12 python3.12-venv)
if apt-get update -qq && apt-get install -y -qq --no-install-recommends "${apt_pkgs[@]}" >/dev/null; then
  log "apt ok: ${apt_pkgs[*]}"
else
  log "apt install failed (${apt_pkgs[*]}); ask Claude to install by hand"
fi

# 2. Python dependency cache
warm_python() {
  command -v uv >/dev/null || pip install -q uv || return 1
  local py
  py="$(command -v python3.12 || true)"
  [[ -n "$py" ]] || { uv python install 3.12 && py=3.12; } || return 1
  local tmp
  tmp="$(mktemp -d)"
  uv venv -q --python "$py" "$tmp/.venv" || return 1
  uv pip install -q --python "$tmp/.venv/bin/python" \
    "pydantic>=2.9" "pyyaml>=6.0" "numpy>=2.1" "pandas>=2.2" "pyarrow>=18.0" \
    "shapely>=2.0" "opencv-python-headless>=4.10" "pycocotools>=2.0.8" \
    "scipy>=1.14" "statsmodels>=0.14" \
    "openai>=1.50" "meta-sam-parser>=0.1.0" "httpx>=0.27" "python-dotenv>=1.0" \
    "fastapi>=0.115" "uvicorn[standard]>=0.32" "python-multipart>=0.0.12" \
    "pytest>=8.3" "hypothesis>=6.115" "ruff>=0.7" "pyright>=1.1.390" "hatchling>=1.25" || return 1
  # pyright fetches its node server on first run; do it now so it lands in the snapshot
  "$tmp/.venv/bin/pyright" --version >/dev/null 2>&1 || true
  rm -rf "$tmp"
}
if warm_python; then log "python cache warm"; else log "python cache warm-up failed; uv sync will download"; fi

# 3. Personal instructions for every cloud session
mkdir -p ~/.claude
cat > ~/.claude/CLAUDE.md <<'EOF'
## Global rules (copied from the laptop's ~/.claude/CLAUDE.md)
- User: Dhruvik, Junior Resident, MD Pharmacology, AIIMS Jodhpur.
- Before deleting any infrastructure, ask whether the deletion can be undone. If not, ask the user first.
- File search: use ripgrep. `rg --files` lists files; `rg PATTERN` searches contents. Don't use find or grep for those jobs.
- Write or design tests first, then code. Don't spam tests: few end-to-end tests, more small focused ones. Use type checks and linters.
- For every change: create a PR → run code review on it → fix → merge to main.
- Highlight what matters in every reply: `inline code` for key terms, **bold** for key phrases.
- Explain with schematic diagrams, flowcharts and tables rather than long prose.
EOF
log "wrote ~/.claude/CLAUDE.md"

log "done"
exit 0
