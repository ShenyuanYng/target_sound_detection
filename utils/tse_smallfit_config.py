"""Validate the bounded fixed-mixture experiment before allocating a GPU."""
import math


def validate_smallfit_config(hp):
    if not getattr(hp, 'tse_smallfit', False):
        return
    expected = dict(dataset='audioset_strong_dual_fsd_hierarchy_tse_v2',
        tse_pilot_artifacts=False, train_percent_check=1.0, val_percent_check=1.0,
        accumulate_grad_batches=1, tse_condition_backward='combined',
        num_sanity_val_steps=-1,
        tse_no_pair_policy='exclude')
    for key, value in expected.items():
        if getattr(hp, key, None) != value:
            raise ValueError(f'Fixed-mixture smallfit requires {key}={value!r}')
    if getattr(hp, 'tse_train_condition_mode', None) not in ('sampled_one', 'sampled_correct'):
        raise ValueError('Smallfit samples one condition per mixture: sampled_one or sampled_correct')
    reference = getattr(hp, 'tse_smallfit_reference_run', None)
    if reference is not None and (not isinstance(reference, str) or not reference.strip()):
        raise ValueError('tse_smallfit_reference_run must name an existing standalone smallfit run')
    for key in ('train_percent_check', 'val_percent_check'):
        if type(getattr(hp, key)) is not float:
            raise ValueError(f'{key} must be float 1.0, not integer 1 batch')
    mids = hp.tse_smallfit_mids
    if not isinstance(mids, list) or len(mids) != 8 or len(set(mids)) != 8:
        raise ValueError('Choose eight distinct smallfit MIDs; keep the full model vocabulary')
    for key in ('max_steps', 'tse_smallfit_eval_every', 'tse_smallfit_train_mixtures',
                'tse_smallfit_val_mixtures', 'tse_smallfit_examples',
                'tse_smallfit_background_candidates', 'tse_smallfit_min_train_sources',
                'tse_smallfit_min_val_sources'):
        if type(getattr(hp, key, None)) is not int or getattr(hp, key) < 1:
            raise ValueError(f'{key} must be a positive integer')
    if hp.max_steps % hp.tse_smallfit_eval_every:
        raise ValueError('max_steps must end on a smallfit evaluation boundary')
    for key in ('tse_smallfit_train_mixtures', 'tse_smallfit_val_mixtures'):
        size = getattr(hp, key)
        if size % (2 * len(mids)):
            raise ValueError(f'{key} must be divisible by 16 for balanced classes and overlap')
        if size // 2 > hp.tse_smallfit_background_candidates:
            raise ValueError('Increase tse_smallfit_background_candidates')
        if hp.tse_smallfit_examples > size:
            raise ValueError('Too many fixed listening examples')
    if hp.tse_smallfit_examples != 8:
        raise ValueError('Use eight listening examples to cover the eight target classes')
    required_epochs = math.ceil(hp.max_steps / math.ceil(hp.tse_smallfit_train_mixtures / hp.batch_size))
    if hp.epochs < required_epochs:
        raise ValueError(f'epochs must be at least {required_epochs} to reach max_steps')
