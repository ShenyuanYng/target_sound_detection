"""Validation isolated from historical SED configuration requirements."""
import math


def validate_tse_config(hp):
    from datasets.tse_hierarchy_labels import MODE as HIERARCHY_MODE
    dual = hp.dataset in ('audioset_strong_dual_fsd_wrong_tse', HIERARCHY_MODE)
    if hp.dataset == HIERARCHY_MODE:
        from utils.tse_hierarchy_config import validate_hierarchy_config
        validate_hierarchy_config(hp)
    expected = dict(task_type='tse', model_name='TSEDPRNNTF', loss_type='ratiol1' if dual else 'tse',
        feature_type='waveform', collate_type='sequence', samplerate=16000,
        train_on_chunk=10., rms_norm=0, mix_normalization='peak',
        mix_fsd_rms_mode='active', pad_back_or_front=0, test_after_fit=False,
        tse_hop=128)
    for name, value in expected.items():
        if getattr(hp, name, None) != value:
            raise ValueError(f'TSE v1 requires {name}={value!r}')
    if hp.tse_window not in (256, 512):
        raise ValueError('TSE supports tse_window=256 or 512, with tse_hop=128')
    if hp.devices != 1:
        raise ValueError('TSE v1 supports one device; distributed CV sampler deduplication is not enabled')
    if not isinstance(getattr(hp, 'tse_gradient_checkpointing', False), bool):
        raise ValueError('tse_gradient_checkpointing must be a boolean')
    if getattr(hp, 'init_model_checkpoint', None):
        raise ValueError('TSE v1 starts from random weights; use a TSE checkpoint for resumption')
    if hp.mix_background_split_path == hp.mix_parent_background_split_path:
        raise ValueError('Use the existing Strong-filtered split, not the full parent split')
    if hp.tse_no_pair_policy not in ('error', 'exclude'):
        raise ValueError('tse_no_pair_policy must be error or explicit exclude')
    if hp.batch_size < 1 or hp.num_classes != 527:
        raise ValueError('TSE requires positive batch_size and the existing 527-class MID map')
    if hp.tse_emb_dim % 4 or min(hp.tse_emb_dim, hp.tse_hidden, hp.tse_layers, hp.tse_condition_dim) < 1:
        raise ValueError('Invalid DPRNNTF dimensions; embedding channels must be divisible by four')
    if hp.tse_seqmodel_type not in (3860, 3861) or not 0 < hp.tse_emb_hs <= hp.tse_emb_ks or hp.tse_emb_ks % hp.tse_emb_hs:
        raise ValueError('Invalid teacher DPRNNTF block configuration')
    loss_parameters = (hp.tse_silence_rms,) if dual else (hp.tse_mrstft_weight, hp.tse_silence_rms)
    if not all(math.isfinite(v) and v > 0 for v in loss_parameters):
        raise ValueError('Invalid TSE loss weight/threshold')
    if dual:
        from utils.tse_runtime import RUNTIME_DEFAULTS, validate_runtime_overrides
        validate_runtime_overrides({name: getattr(hp, name, default)
                                    for name, default in RUNTIME_DEFAULTS.items()})
        for name, default in (('tse_log_batch_loss', False), ('tse_record_overlap_loss', True)):
            if type(getattr(hp, name, default)) is not bool:
                raise ValueError(f'{name} must be a boolean')
        if (getattr(hp, 'tse_condition_backward', 'combined') == 'sequential' and
                getattr(hp, 'accumulate_grad_batches', 1) != 1):
            raise ValueError('Sequential condition backward requires accumulate_grad_batches=1')
        flags = dict(rix2_flag=1, mag_flag=1, permutation_invariant=0, mc_flag=0,
                     tse_return_ri=True, mix_strong_annotation_complete=True)
        if any(getattr(hp, key, None) != value for key, value in flags.items()):
            raise ValueError('Experiment 3 requires RI+Mag L1, raw RI, no PIT/MC and Strong labels')
        for name in ('tse_dual_max_pair_attempts', 'tse_dual_max_crop_attempts'):
            if type(getattr(hp, name, None)) is not int or getattr(hp, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if not getattr(hp, 'tse_ontology_path', None):
            raise ValueError('Experiment 3 requires an explicit ontology JSON')
    from datasets.active_rms import validate_rms_settings
    from datasets.mix_utils import validate_snr_db_range
    validate_rms_settings(hp.mix_fsd_rms_mode, hp.mix_active_threshold_dbfs, hp.mix_active_frame_ms)
    validate_snr_db_range(hp.mix_snr_db_range)
    if hp.mix_snr_db is not None and not math.isfinite(hp.mix_snr_db):
        raise ValueError('Nonfinite mix_snr_db')
