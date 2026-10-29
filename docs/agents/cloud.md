# Cloud sessions and CI

How a change gets from a cloud session to `main`, and how to set up the cloud environment.
Facts checked against code.claude.com/docs (cloud sessions, cloud environments, settings, skills)
on Oct 29, 2026.

## Where each kind of work runs

| Work | Where | Why |
|---|---|---|
| Building stages, tests, frontend | Cloud session | Long-running, no secrets needed |
| SAM 3.1 / Muse Spark spike | Laptop | Needs `MODEL_API_KEY`; cloud's Trusted network can't reach `api.meta.ai` |
| `eval/` on the Sturman dataset | Laptop | Needs the key and 2.5 GB of video; GPL labels stay in `eval/data/` |
| CI | GitHub Actions | Runs `scripts/check.sh` on every PR and on `main` |

## The path of one change

```
cloud session                         GitHub
─────────────                         ──────
SessionStart hook: uv sync --locked
branch feat/<stage>
write tests → code
scripts/check.sh ──── must pass ────▶ push branch
                                      open PR (REST, not `gh pr`)
                                      CI: scripts/check.sh ──▶ ✔ / ✘
code-review skill vs origin/main
  └─ findings ──────────────────────▶ PR comment
fix → check.sh → push ──────────────▶ CI again
                                      CI green + findings answered
                                      squash-merge ──▶ main
```

## One-time setup

1. **GitHub access.** Install the Claude GitHub App on `heyiamnotacoder/pharmavision`
   (github.com/apps/claude). The repo is private, so sessions can only clone it this way (or after
   `/web-setup` in a laptop terminal). The app also enables Auto-fix on PRs.
2. **Environment.** At claude.ai/code, create an environment named `pharmavision`:
   - **Network access:** Trusted (PyPI, npm, GitHub, Ubuntu archives are on the default list).
   - **Setup script:** the contents of `scripts/cloud/environment-setup.sh`. It installs `ffmpeg`,
     warms the Python package cache and writes your personal rules to `~/.claude/CLAUDE.md`.
     It must finish in about 5 minutes to be cached; tested in an Ubuntu 24.04 container.
   - **Environment variables:** none needed. Don't put `MODEL_API_KEY` here: anyone using the
     environment can read its variables, and cloud work doesn't call the Meta API.
3. **Session-start hook.** `.claude/settings.json` runs `scripts/cloud/session-start.sh` at every
   cloud session start (it exits at once on the laptop). It also stops Claude's Read tool from
   opening `.env`. A shell command could still read the file, so keep the real key on the laptop
   only.
4. **First cloud session.** Ask Claude to run `scripts/check.sh` and `/context`. Expect all checks
   green, and `/root/.claude/CLAUDE.md` under Memory files.

## What reaches a cloud session

| Item | Reaches cloud? | How it gets there |
|---|---|---|
| `CLAUDE.md`, `GLOSSARY.md`, `docs/` | Yes | Committed |
| Matt Pocock skills | Yes | Vendored in `.claude/skills/` (`scripts/sync-matt-pocock-skills.sh`) |
| `.claude/settings.json` hooks and permissions | Yes | Committed |
| Your laptop `~/.claude/CLAUDE.md`, skills, plugins, MCP servers | No | The setup script writes a copy of the rules |
| `gh pr` / `gh issue` | No (GraphQL blocked) | Use REST, see `issue-tracker.md` |

## Limits to know

- **Branch protection is off.** GitHub refuses it on a private repo without GitHub Pro. Until the
  repo goes public (it must be public by Nov 4 anyway), "green CI and a review before merge" is
  enforced by the workflow rule only, not by GitHub. Once it's public, require the
  `lint · typecheck · test` check on `main`.
- **Cloud VM:** 4 vCPU, 16 GB RAM, 30 GB disk; foreground commands time out at 2 minutes by default
  (`BASH_DEFAULT_TIMEOUT_MS` on the environment raises it).
- **Setup script changes** take effect when the cache rebuilds (on edit, or after about 7 days).
