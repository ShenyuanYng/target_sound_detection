#!/usr/bin/env bash
# 解析运行方式；训练参数交给统一提交器校验和冻结。
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_root"
. "$project_root/path.sh"

stage=0
use_slurm=1
python_bin="${PYTHON_BIN:-}"
arguments=(--node "${SLURMD_NODENAME:-ailab01}")
show_help=0
while (( $# )); do
    option="${1%%=*}"
    option="${option//_/-}"
    case "$option" in
        --stage|--use-slurm|--python-bin)
            if [[ "$1" == *=* ]]; then
                value="${1#*=}"
                shift
            else
                if (( $# < 2 )) || [[ "$2" == --* ]]; then
                    printf 'Missing value for %s\n' "$1" >&2
                    exit 2
                fi
                value="$2"
                shift 2
            fi
            case "$option" in
                --stage) stage="$value" ;;
                --use-slurm) use_slurm="$value" ;;
                --python-bin) python_bin="$value" ;;
            esac
            ;;
        --submit)
            [[ "$1" == --submit ]] || {
                echo '--submit takes no value; use --stage 0 for preview.' >&2; exit 2;
            }
            stage=1; use_slurm=1; shift
            ;;
        --help|-h) show_help=1; shift ;;
        --execute|--run-submission)
            printf '%s is an internal submitter option; use --stage / --use-slurm.\n' "$option" >&2
            exit 2
            ;;
        *)
            # Normalize option names only; never alter values containing underscores.
            if [[ "$1" == --* ]]; then
                if [[ "$1" == *=* ]]; then
                    arguments+=("$option=${1#*=}")
                else
                    arguments+=("$option")
                fi
            else
                arguments+=("$1")
            fi
            shift
            ;;
    esac
done

[[ "$stage" == 0 || "$stage" == 1 ]] || {
    echo 'stage must be 0 (preview) or 1 (train).' >&2; exit 2;
}
[[ "$use_slurm" == 0 || "$use_slurm" == 1 ]] || {
    echo 'use_slurm must be 0 (allocated node) or 1 (submit Slurm job).' >&2; exit 2;
}
if [[ -z "$python_bin" ]]; then
    python_bin=/home/syy12632604/anaconda3/envs/audioset/bin/python
    [[ -x "$python_bin" ]] || python_bin=python3
fi
if (( show_help )); then
    cat <<'HELP'
TSE shell launcher
  --stage 0|1         0: preview (default); 1: start a fresh experiment
  --use-slurm 0|1     1: submit from login node; 0: execute in an allocated node
  --submit            Shortcut for --stage 1 --use-slurm 1
  --python-bin PATH   Interpreter used by the launcher and training jobs

Other options below select the YAML, training overrides, resources and run tags.
Resume an existing run on its allocated node with:
  python resume_training.py --run-dir /home/syy12632604/projects/TSE/experiments/...
HELP
    exec "$python_bin" -B tools/submit_training.py --help
fi
if [[ "$stage" == 1 ]]; then
    if [[ "$use_slurm" == 1 ]]; then
        arguments+=(--submit)
    else
        : "${SLURM_JOB_ID:?use_slurm=0 requires an allocated Slurm compute node}"
        arguments+=(--execute)
    fi
fi
exec "$python_bin" -B -u tools/submit_training.py \
    --python-bin "$python_bin" "${arguments[@]}"
