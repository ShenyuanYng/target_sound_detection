#!/usr/bin/env bash
# Resources and the predecessor dependency are set by submit_continuation.py.
set -euo pipefail
# A wall-time SIGTERM must not be mistaken for an ordinary training error.
trap 'exit 143' TERM
cd "${SLURM_SUBMIT_DIR:?Submit from the project directory}"
run_dir="${1:?Existing experiment directory required}"
python_bin="${2:?Python executable required}"
chain_dir="${3:?Chain record directory required}"

if [[ -f "$chain_dir/failed" ]]; then
    echo "A previous continuation failed; inspect $chain_dir/failed. Training skipped."
    exit 1
fi
export PYTHONUNBUFFERED=1
. ./path.sh
mkdir -p "$run_dir/logs"
# Each job gets its own log; never overwrite the running job's training.log.
log_path="$run_dir/logs/training_job${SLURM_JOB_ID}.log"
exec >> "$log_path" 2>&1
printf 'job_id=%s hostname=%s run_dir=%s\n' "$SLURM_JOB_ID" "$HOSTNAME" "$run_dir"
# A scheduler timeout kills the process; it does not create a failure marker.
# Ordinary Python errors stop subsequent jobs from retrying the same error.
resume_command=("$python_bin" resume_training.py --run-dir "$run_dir")
"${resume_command[@]}" || {
    result=$?
    if [[ "$result" == 137 || "$result" == 143 ]]; then
        exit "$result"
    fi
    printf 'Job %s failed with exit code %s. See %s\n' "$SLURM_JOB_ID" "$result" "$log_path" > "$chain_dir/failed"
    exit "$result"
}
