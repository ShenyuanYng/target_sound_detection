"""Explicit execution overrides for an existing experiment-3 run."""
import json
from pathlib import Path


RUNTIME_DEFAULTS = dict(tse_gradient_checkpointing=False,
    tse_train_condition_mode='all', tse_condition_backward='combined')
OVERRIDE_FILENAME = 'tse_runtime_overrides.json'


def validate_runtime_overrides(settings):
    if not isinstance(settings, dict) or set(settings) - set(RUNTIME_DEFAULTS):
        raise ValueError('Only checkpointing, condition sampling and backward execution may be overridden')
    if ('tse_gradient_checkpointing' in settings and
            type(settings['tse_gradient_checkpointing']) is not bool):
        raise ValueError('tse_gradient_checkpointing must be a boolean')
    if settings.get('tse_train_condition_mode', 'all') not in ('all', 'sampled_correct_wrong', 'sampled_one'):
        raise ValueError('Unknown TSE condition mode')
    if settings.get('tse_condition_backward', 'combined') not in ('combined', 'sequential'):
        raise ValueError('Unknown TSE backward execution')


def read_runtime_overrides(run_dir):
    path = Path(run_dir) / OVERRIDE_FILENAME
    if not path.exists():
        return {}
    record = json.loads(path.read_text())
    if record.get('schema') != 1:
        raise ValueError('Unknown TSE runtime override schema')
    settings = record['settings']
    validate_runtime_overrides(settings)
    return settings
