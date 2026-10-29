#!/usr/bin/env bash
# Vendor Matt Pocock's engineering skills into .claude/skills/ so cloud sessions get them.
#
# Cloud sessions don't read ~/.claude/skills on the laptop; they load the repo's .claude/skills/.
# Locally nothing is duplicated: a personal skill with the same name wins over the project copy.
#
# The pin matches the copies installed on the laptop (~/.agents/skills, updated Oct 20, 2026).
# The laptop copies have their `disable-model-invocation` lines removed so Claude can invoke the
# skills on its own; this script strips the same lines from the vendored copies.
#
# Usage: scripts/sync-matt-pocock-skills.sh [commit]   (run from anywhere, then commit the result)
set -euo pipefail

PIN="${1:-5beb35b040e3de97e54fa9e386d7e42d15c295c2}"
SKILLS=(
  ask-matt code-review codebase-design diagnosing-bugs domain-modeling grill-me grill-with-docs
  grilling handoff implement implement-spec improve-codebase-architecture pr prototype research
  retro setup-matt-pocock-skills tdd teach to-questionnaire to-spec to-tickets triage wait-what
  wayfinder wizard writing-for-agents
)

repo_root="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
dest="$repo_root/.claude/skills"
src="$(mktemp -d)"
trap 'rm -rf "$src"' EXIT

git clone --quiet --filter=blob:none https://github.com/mattpocock/skills.git "$src"
git -C "$src" checkout --quiet "$PIN"

mkdir -p "$dest"
for name in "${SKILLS[@]}"; do
  # Skills live under skills/<category>/<name>/; the category isn't part of the skill's identity.
  skill_dir="$(find "$src/skills" -mindepth 2 -maxdepth 2 -type d -name "$name" -print -quit)"
  if [[ -z "$skill_dir" ]]; then
    echo "missing upstream at $PIN: $name" >&2
    exit 1
  fi
  rm -rf "${dest:?}/$name"
  cp -R "$skill_dir" "$dest/$name"
  sed -i.bak '/^disable-model-invocation:/d' "$dest/$name/SKILL.md" && rm "$dest/$name/SKILL.md.bak"
done

cp "$src/LICENSE" "$dest/LICENSE-mattpocock-skills"
printf 'mattpocock/skills @ %s\n' "$PIN" > "$dest/.mattpocock-skills-pin"
echo "Vendored ${#SKILLS[@]} skills from mattpocock/skills@${PIN:0:7} into .claude/skills/"
