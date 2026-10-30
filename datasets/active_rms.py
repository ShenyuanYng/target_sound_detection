"""Active foreground measurement and mixing shared by all FSD mainlines.

dBFS uses amplitude full scale 1. Frames are rectangular, non-overlapping,
anchored at crop sample zero; the last frame has its actual length (no padding).
The pre-gain active mask is fixed for all subsequent RMS/SNR measurements.
"""
from __future__ import annotations

import numpy as np

from datasets.mix_utils import (
    SAMPLE_RATE, TARGET_NUM_SAMPLES, _waveform,
    pad_segment_to_target_length, normalize_rms, compute_rms,
    validate_background, scale_foreground_to_snr,
    eligible_target_mids, prepare_fsd_crop, sample_rng, waveform_statistics,
)

#输入模式，阈值，帧长   主要是检验作用
def validate_rms_settings(mode, threshold_dbfs, frame_ms):
    """Validate the persisted active-only configuration, not dispatch RMS modes."""
    if mode != 'active':
        raise ValueError('FSD mixing requires active RMS')
    if not np.isfinite(threshold_dbfs) or not -120 <= threshold_dbfs <= 0:
        raise ValueError('Active RMS threshold must be finite and in [-120, 0] dBFS')
    if not np.isfinite(frame_ms) or not 0 < frame_ms <= 1000:
        raise ValueError('Active RMS frame_ms must be in (0, 1000]')
    samples = SAMPLE_RATE * float(frame_ms) / 1000
    if samples < 1 or abs(samples - round(samples)) > 1e-8:
        raise ValueError('Active RMS frame_ms must represent an integer number of samples')

#输入波形，阈值，帧长
def active_rms(waveform, *, threshold_dbfs=-40., frame_ms=20.):
    """Return reference RMS, original-sample mask, and JSON-safe diagnostics.

    分帧计算零均值 RMS，选超阈值帧，再对活动样本并集计算一次 RMS。
    无活动帧时以 max(crop RMS, threshold amplitude) 为增益分母。
    Mask 只用于统计，不删除或清零低能量采样点；尾帧按实际长度计算。
    """
    validate_rms_settings('active', threshold_dbfs, frame_ms)
    value = _waveform(waveform).astype(np.float64)
    size = round(SAMPLE_RATE * float(frame_ms) / 1000)
    starts = np.arange(0, len(value), size)
    counts = np.minimum(size, len(value) - starts)
    frame_rms = np.array([compute_rms(value[start:start + count])
                          for start, count in zip(starts, counts)])
    threshold = float(10 ** (float(threshold_dbfs) / 20))
    selected = frame_rms > threshold
    mask = np.repeat(selected, counts)
    active_samples = int(counts[selected].sum())
    measured = compute_rms(value[mask]) if active_samples else None
    crop_rms = compute_rms(value)
    reference = measured if active_samples else max(crop_rms, threshold)
    return reference, mask, dict(
        active_threshold_dbfs=float(threshold_dbfs), active_frame_ms=float(frame_ms),
        active_frame_samples=size, active_frame_count=int(selected.sum()),
        total_frame_count=len(starts), active_sample_count=active_samples,
        active_sample_fraction=active_samples / len(value),
        target_active_rms_raw=measured, target_crop_rms_raw=crop_rms,
        rms_fallback='none' if active_samples else 'crop_rms_with_threshold_floor',
        target_rms_gain_denominator=reference,
    )

#计算SNR
def _snr(target_rms, background_rms):
    return (float(20 * np.log10(target_rms / background_rms))
            if target_rms > 0 and background_rms > 0 else None)

#用活动帧RMS基准缩放crop->补零->混合->峰值归一化
def mix_rms(background, crop, *, threshold_dbfs=-40., frame_ms=20.,
            snr_db=0., normalization='peak', rng=0):
    """唯一的单源混音流程：Active RMS → 增益 → 随机补零 → 共同峰值缩放。

    Input crop is already cleaned/cropped, without insertion padding. Background
    reference is its full 10-second RMS. Returns (background, target, mixture,
    metadata), with float32 stems sharing the final mixture divisor.
    Dual FSD uses normalization='none' before its own joint two-source placement.
    Keep floating-point operations and RNG order unchanged for fixed CV replay.
    """
    validate_rms_settings('active', threshold_dbfs, frame_ms)
    background, crop = validate_background(background), _waveform(crop)
    if len(crop) > TARGET_NUM_SAMPLES:
        raise ValueError('Retained crop must be at most 10 s')
    if not np.isfinite(snr_db) or normalization not in ('peak', 'none'):
        raise ValueError('Invalid SNR or normalization')
    reference, mask, diagnostics = active_rms(crop, threshold_dbfs=threshold_dbfs, frame_ms=frame_ms)
    # Active selection was done before gain. Never re-threshold normalized audio.
    fallback = not mask.any()
    measure_mask = mask if not fallback else np.ones(len(crop), dtype=bool)
    bg, bg_before = normalize_rms(background)
    bg_rms = compute_rms(bg)
    normalized = (crop.astype(np.float64) / reference).astype(np.float32)
    normalized_rms = compute_rms(normalized[measure_mask])
    # Preserve the original normalize -> SNR scale -> pad -> peak order.
    # On fallback, do not cancel the denominator floor by re-normalizing silence.
    scaled, scale = scale_foreground_to_snr(normalized, snr_db=snr_db,
        background_rms=bg_rms, reference_mask=measure_mask,
        reference_rms=1. if fallback else None)
    gain = scale / reference
    measured = compute_rms(scaled[measure_mask])
    actual = _snr(measured, bg_rms)
    if not fallback and (actual is None or abs(actual - snr_db) > 1e-4):
        raise ValueError('Active SNR mismatch')
    padded, padding = pad_segment_to_target_length(scaled, rng=rng)
    """相加与共同peak归一化
    divisor：除数
    """
    mixture = _waveform(bg + padded)
    peak = float(np.max(np.abs(mixture)))
    triggered = normalization == 'peak' and peak > 1
    divisor = peak if triggered else 1.
    exported_bg = (bg / divisor).astype(np.float32)
    exported_target = (padded / divisor).astype(np.float32)
    mixture = (mixture / divisor).astype(np.float32)
    start, end = padding['target_start_sample'], padding['target_end_sample']
    exported_reference = compute_rms(exported_target[start:end][measure_mask])
    exported_snr = _snr(exported_reference, compute_rms(exported_bg))
    if not fallback and (exported_snr is None or abs(exported_snr - snr_db) > 1e-4):
        raise ValueError('Exported active SNR mismatch')
    return exported_bg, exported_target, mixture, dict(
        **padding, **diagnostics, fsd_rms_mode='active', snr_definition='active',
        snr_scaling='active_power_ratio_foreground_gain', zero_mean=True,
        background_rms_before_norm=bg_before, target_rms_before_norm=reference,
        background_rms_after_source_norm=bg_rms,
        target_rms_after_source_norm=normalized_rms, target_scale=scale,
        background_rms_before_final_norm=bg_rms, target_rms_before_final_norm=measured,
        target_rms_10s_before_final_norm=compute_rms(padded),
        snr_db=float(snr_db), snr_db_requested=float(snr_db),
        snr_db_actual_before_final_peak_norm=actual,
        snr_error_db=None if actual is None else actual - snr_db,
        snr_target_met=actual is not None and abs(actual - snr_db) < 1e-4,
        target_gain=gain, mixture_peak_before_final_norm=peak,
        final_peak_normalized=bool(triggered), final_normalization_divisor=divisor,
        normalization=normalization, mixture_peak=float(np.max(np.abs(mixture))),
        background_rms_exported=compute_rms(exported_bg), target_rms_exported=exported_reference,
        target_rms_10s_exported=compute_rms(exported_target), snr_db_exported=exported_snr,
        rms_scope='FSD pre-gain active frames (fallback: crop); AudioSet full 10 seconds',
        export_components='both multiplied by the same final mixture gain',
    )


def make_snr_mixture(background, fsd_waveform, *, background_id, fsd_id, target_mid, seed,
                     background_weak_mids, fsd_mids, snr_db=0.0, normalization='peak',
                     threshold_dbfs=-40., frame_ms=20.):
    """从 FSD 源生成一条有身份信息的 Active RMS 混音。

    校验目标 MID → 去零/重复/裁剪（固定 crop/pad seed）→ mix_rms。
    返回 (background, target, mixture, metadata)；三条波形都是 160000 点。
    此层只组织源预处理与记录制作履历，不另实现 RMS 缩放或混音算法。
    """
    if target_mid not in eligible_target_mids(background_weak_mids, fsd_mids):
        raise ValueError('target_mid must belong to FSD mids and be absent from background weak_mids')
    crop, metadata, pad_seed = prepare_fsd_crop(fsd_waveform,
        rng=sample_rng(background_id, fsd_id, target_mid, seed))
    bg, tg, mixture, mixing = mix_rms(background, crop, snr_db=snr_db,
        normalization=normalization, rng=pad_seed,
        threshold_dbfs=threshold_dbfs, frame_ms=frame_ms)
    #合并metadata
    mixing.pop('truncated_num_samples', None)  # retain pre-padding truncation metadata
    metadata.update(mixing)
    start, end = metadata['target_start_sample'], metadata['target_end_sample']
    crop_rms, local_bg_rms = compute_rms(tg[start:end]), compute_rms(bg[start:end])
    full_target_rms, full_bg_rms = compute_rms(tg), compute_rms(bg)
    metadata.update({'background_id': str(background_id), 'fsd_id': str(fsd_id),
                     'target_mid': target_mid, 'seed': int(seed), 'sample_rate': SAMPLE_RATE,
                     'background_peak_raw': waveform_statistics(background)['peak'],
                     'target_peak_raw': waveform_statistics(crop)['peak'],
                     'target_rms_crop_exported': crop_rms,
                     'target_rms_crop_before_final_norm': crop_rms * mixing['final_normalization_divisor'],
                     'snr_db_full_10s_exported': (float(20*np.log10(full_target_rms/full_bg_rms))
                                                if full_target_rms > 0 and full_bg_rms > 0 else None),
                     # Legacy name: measures the insertion interval, not active-frame RMS.
                     'snr_db_active_region_exported': (float(20*np.log10(crop_rms/local_bg_rms))
                                                       if crop_rms > 0 and local_bg_rms > 0 else None)})
    return bg, tg, mixture, metadata
