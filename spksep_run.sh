#!/usr/bin/env bash
# Execute TSE with an already allocated node. Submission uses run.sh.
set -euo pipefail
: "${SLURM_JOB_ID:?Run on an allocated compute node}"
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_root"
. ./path.sh
exec "${PYTHON_BIN:-python3}" -u train.py "$@"
