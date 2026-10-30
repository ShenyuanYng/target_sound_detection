"""Freeze configuration and source provenance without copying project code."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

import yaml

from utils.project_paths import PROJECT, training_path


def record_run(hparams, resume_run_dir=None):
    config = dict(hparams) if isinstance(hparams, dict) else vars(hparams).copy()
    parent = training_path(config['save_path'])
    name = Path(config['save_name'])
    if name.is_absolute() or '..' in name.parts or str(name) in ('', '.'):
        raise ValueError('save_name must be a nonempty relative run name without ..')
    run = training_path(resume_run_dir) if resume_run_dir else training_path(parent / name)
    marker = run / '.tse_run_created'
    if resume_run_dir:
        if not marker.is_file():
            raise ValueError('Resume requires a run created by this standalone TSE project')
    else:
        run.mkdir(parents=True, exist_ok=True)
        if (run / 'mylast.ckpt').exists() or any(run.glob('version_*')):
            raise ValueError('Choose a fresh TSE run directory; existing checkpoints/logs cannot be replaced')
        with marker.open('x') as handle:
            json.dump(dict(project='TSE', created_unix=time.time()), handle)

    sources = []
    for directory in ('creators', 'datasets', 'nets', 'loss_functions', 'utils', 'tools', 'slurm', 'exp_defs', 'metadata'):
        sources.extend(p for p in (PROJECT / directory).rglob('*')
                       if p.is_file() and p.suffix in ('.py', '.sh', '.sbatch', '.yaml', '.json', '.csv'))
    sources.extend(p for p in PROJECT.iterdir() if p.is_file() and p.suffix in ('.py', '.sh', '.txt'))
    hashes = {str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(set(sources))}
    revision = subprocess.run(['git', 'rev-parse', '--verify', 'HEAD'], cwd=PROJECT,
                              capture_output=True, text=True, timeout=10)
    status = subprocess.run(['git', 'status', '--porcelain'], cwd=PROJECT,
                            capture_output=True, text=True, timeout=10)
    record = dict(project='TSE', code_directory=str(PROJECT), run_directory=str(run),
                  git_revision=revision.stdout.strip() if revision.returncode == 0 else None,
                  git_status=status.stdout if status.returncode == 0 else None,
                  source_sha256=hashes, effective_config=config, code_copied=False,
                  created_unix=time.time(), host=socket.gethostname(), pid=os.getpid(),
                  slurm_job_id=os.environ.get('SLURM_JOB_ID'))
    if resume_run_dir:
        destination = run/'continuation'/f'provenance_{time.time_ns()}_{os.getpid()}.json'
        destination.parent.mkdir(exist_ok=True)
    else:
        destination = run/'source_provenance.json'
        effective = run/'effective_config.yaml'
        if effective.exists():
            frozen = run/'submitted_config.yaml'
            if not frozen.exists():
                frozen.write_bytes(effective.read_bytes())
        effective.write_text(yaml.safe_dump(config, sort_keys=False))
    with destination.open('x') as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
        handle.write('\n')
    return run
