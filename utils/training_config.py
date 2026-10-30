"""Validate the standalone extraction task."""
from utils.tse_config import validate_tse_config

def validate_training_data_config(hparams):
    if hparams.dataset not in ('audioset_strong_fsd50k_tse',
            'audioset_strong_dual_fsd_wrong_tse', 'audioset_strong_dual_fsd_hierarchy_tse_v2'):
        raise ValueError(f'Unsupported TSE dataset: {hparams.dataset}')
    return validate_tse_config(hparams)
