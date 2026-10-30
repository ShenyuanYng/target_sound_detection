#!/usr/bin/env bash
# Preview by default; add --submit to schedule on the shared compute server.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_root"
exec "${PYTHON_BIN:-python3}" tools/submit_training.py "$@"
