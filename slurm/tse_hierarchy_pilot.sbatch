#!/usr/bin/env bash
# The L40 partition on ailab01 contains both L40 and RTX 5880 Ada GPUs.
# An untyped GPU request permits either model; Slurm sets CUDA_VISIBLE_DEVICES.
#SBATCH --job-name=tse-pilot-10pct
#SBATCH --partition=L40
#SBATCH --nodelist=ailab01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --chdir=/home/syy12632604/projects/TSE
#SBATCH --output=/home/syy12632604/projects/TSE/experiments/tse_hierarchy_v2/slurm_logs/%x-%j.log
#SBATCH --open-mode=append

set -euo pipefail
: "${SLURM_JOB_ID:?Submit this file with sbatch}"
[[ "${SLURMD_NODENAME:-}" == ailab01 ]] || {
    echo 'This configuration uses the source audio paths on ailab01.' >&2
    exit 1
}

project=/home/syy12632604/projects/TSE
cd "$project"
. ./path.sh
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2

pilot_name="pilot_10pct_bf16_e5_job${SLURM_JOB_ID}"
pilot_root="$project/experiments/tse_hierarchy_v2/$pilot_name"
mkdir -p "$(dirname -- "$pilot_root")"
mkdir "$pilot_root"
printf 'Experiment: %s\nView after initialization: %s/data/%s\n' "$pilot_root" "$project" "$pilot_name"
# Keep the new run empty except for run.log until prepare_run reserves it.
exec >> "$pilot_root/run.log" 2>&1
printf 'job=%s partition=%s node=%s allocated_gpus=%s\n' \
    "$SLURM_JOB_ID" "${SLURM_JOB_PARTITION:-}" "${SLURMD_NODENAME:-}" "${CUDA_VISIBLE_DEVICES:-Slurm-managed}"
sha256sum slurm/tse_hierarchy_pilot.sbatch exp_defs/tse_hierarchy_v2_pilot_10pct_bf16.yaml

srun --ntasks=1 /home/syy12632604/anaconda3/envs/audioset/bin/python3.9 -B -u train.py \
    --exp-def exp_defs/tse_hierarchy_v2_pilot_10pct_bf16.yaml \
    --save-name "$pilot_name"
