#!/bin/bash
# Lightweight syntax checks only; no Torch/CUDA imports or training.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if git rev-parse --git-dir >/dev/null 2>&1; then
    git diff --check
    git diff --cached --check
fi
"${CHECK_PYTHON:-python3}" - <<'PY'
import ast
from pathlib import Path
import subprocess

files = [str(p) for p in Path('.').rglob('*') if p.is_file()
         and not any(part in {'.git', '.venv', '__pycache__', 'experiments', 'data', 'outputs', 'output'} for part in p.parts)]
checked = 0
for name in sorted(set(files)):
    path = Path(name)
    if not path.is_file():
        continue
    if path.suffix == '.py':
        ast.parse(path.read_bytes(), filename=name)
    elif path.suffix in ('.sh', '.sbatch'):
        subprocess.run(['bash', '-n', str(path)], check=True)
    else:
        continue
    checked += 1
print(f'Local checks passed: {checked} Python/shell files; Git whitespace checks passed.')
PY
