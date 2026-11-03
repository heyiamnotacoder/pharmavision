#!/usr/bin/env bash
# Pack a finished study and every run its study.yaml names into a tarball for the deploy's
# DEMO_URL (see Dockerfile). Laptop only: runs/ and studies/ are git-ignored.
#
#   scripts/pack-demo.sh STUDY_ID [OUT.tar.gz]
#
# Run folders hold pipeline artifacts and the normalised Sturman video (CC BY 4.0), never the
# GPL-3.0 human labels, which live in eval/data/ only.
set -euo pipefail

study_id="${1:?usage: scripts/pack-demo.sh STUDY_ID [OUT.tar.gz]}"
out="${2:-demo.tar.gz}"
study="studies/${study_id}"
[[ -f "${study}/report.json" ]] || { echo "${study}: no report.json yet" >&2; exit 1; }

run_ids=$(uv run --no-sync python - "${study}/study.yaml" <<'EOF'
import sys
from pathlib import Path

import yaml

from pharmavision.artifacts import StudyDesign

design = StudyDesign.model_validate(yaml.safe_load(Path(sys.argv[1]).read_text()))
print("\n".join(t.run_id for t in design.trials))
EOF
)

staging=$(mktemp -d)
trap 'rm -rf "${staging}"' EXIT
mkdir -p "${staging}/runs" "${staging}/studies"
cp -R "${study}" "${staging}/studies/"
while read -r run_id; do
    [[ -f "runs/${run_id}/qc.json" ]] || { echo "runs/${run_id}: not finished (no qc.json)" >&2; exit 1; }
    cp -R "runs/${run_id}" "${staging}/runs/"
done <<< "${run_ids}"

tar -czf "${out}" -C "${staging}" runs studies
echo "${out}: $(wc -l <<< "${run_ids}" | tr -d ' ') runs and study ${study_id}, $(du -h "${out}" | cut -f1)"
