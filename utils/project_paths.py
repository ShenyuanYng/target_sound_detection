"""Code and output roots for standalone TSE; corpus/cache paths are separate."""
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
SERVER_PROJECT = Path('/home/syy12632604/projects/TSE')
EXPERIMENTS = SERVER_PROJECT / 'experiments'


def training_path(value):
    path = Path(value)
    path = (path if path.is_absolute() else PROJECT / path).resolve()
    roots = {EXPERIMENTS.resolve(), (PROJECT / 'experiments').resolve()}
    if not any(path == root or root in path.parents for root in roots):
        raise ValueError(f'TSE training artifacts must be under {EXPERIMENTS}: {path}')
    if Path('/data') in path.parents:
        raise ValueError('Training artifacts cannot be placed on node-local /data')
    return path
