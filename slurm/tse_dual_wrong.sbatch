#!/usr/bin/env bash
# Submission resources and node configuration are frozen by submit_tse_dual_wrong.py.
set -euo pipefail
trap 'exit 143' TERM
cd "${SLURM_SUBMIT_DIR:?Submit from the project directory}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
. ./path.sh
"${2:?Python executable required}" -B -u tools/submit_training.py \
    --run-submission "${1:?Frozen submission directory required}"
