#!/usr/bin/env bash
# 从脚本位置解析项目根目录，避免在其他目录启动时加载错误的源码。
_tse_project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$_tse_project_root/tools:$_tse_project_root:$PATH"
export LC_ALL=C
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONPATH="$_tse_project_root${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
unset _tse_project_root
