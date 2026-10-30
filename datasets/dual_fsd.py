"""
FSD双condition：一条背景混入两个FSD，产出两条条件监督
Two independent Active RMS foregrounds with legacy crop/energy semantics."""
import numpy as np

from datasets.active_rms import mix_rms
from datasets.mix_utils import TARGET_NUM_SAMPLES, prepare_fsd_crop
from datasets.dual_fsd_crop import conditional_pair_crops, POLICY

"""
尝试为两个FSD各裁剪一段能塞进10s的crop；每个前景独立用mix_rms(normalization='none') 按整 10s 背景 RMS 缩放；
撤销临时补零后把两段联合定位（随机排序、随机分配前后中空隙），相加后统一峰值归一化
输出：(bg, [target0, target1], mixture, meta)
"""
def make_dual_mixture(background, waveforms, *, seed, gap_samples=512,
                      max_crop_attempts=1024, threshold_dbfs=-40., frame_ms=20.,
                      snr_db=0., normalization='peak'):
    """Reject crops that cannot fit; then sample the conditional law directly.

    Successful legacy draws remain unchanged. Exhaustion uses a finite sampler
    over the same feasible crop pairs, without shortening or replacing sources.

    Each foreground is independently scaled against full-10s background RMS.
    mix_rms(normalization='none') supplies the existing gains/fallback exactly;
    its temporary padding is undone, then both crops are jointly positioned.
    """
    if len(waveforms) != 2 or not 512 <= gap_samples < TARGET_NUM_SAMPLES - 16000:
        raise ValueError('Need two sources and a gap >= one 512-sample GT window')
    if max_crop_attempts < 1 or normalization != 'peak':
        raise ValueError('Need positive crop attempts and joint peak normalization')
    rng = np.random.default_rng(seed)
    crop_sampling = 'legacy_rejection'
    for attempt in range(max_crop_attempts):
        seeds = [int(x) for x in rng.integers(0, np.iinfo(np.int64).max, size=2)]
        prepared = [prepare_fsd_crop(w, rng=s) for w, s in zip(waveforms, seeds)]
        lengths = [len(item[0]) for item in prepared]
        free = TARGET_NUM_SAMPLES - sum(lengths) - gap_samples
        if free >= 0:
            break
    else:
        prepared = conditional_pair_crops(waveforms,
            budget=TARGET_NUM_SAMPLES-gap_samples, rng=rng)
        lengths = [len(item[0]) for item in prepared]
        free = TARGET_NUM_SAMPLES - sum(lengths) - gap_samples
        seeds = [None, None]  # Fallback replays from explicit geometry / pair seed.
        crop_sampling = POLICY
    # Uniform weak compositions into leading/middle/trailing gaps, random order.
    cuts = np.sort(rng.choice(free + 2, size=2, replace=False))
    leading, middle = int(cuts[0]), int(cuts[1] - cuts[0] - 1)
    order = [int(x) for x in rng.permutation(2)]
    starts = [0, 0]
    starts[order[0]] = leading
    starts[order[1]] = leading + lengths[order[0]] + gap_samples + middle
    targets, events = [], []
    for (crop, geometry, pad_seed), crop_seed, start in zip(prepared, seeds, starts):
        bg, temporary, _, scaling = mix_rms(background, crop,
            threshold_dbfs=threshold_dbfs, frame_ms=frame_ms, snr_db=snr_db,
            normalization='none', rng=pad_seed)
        scaled = temporary[scaling['target_start_sample']:scaling['target_end_sample']]
        target = np.zeros(TARGET_NUM_SAMPLES, dtype=np.float32)
        target[start:start + len(crop)] = scaled
        targets.append(target)
        events.append(dict(crop_seed=crop_seed, crop_sampling=crop_sampling, **geometry,
            target_start_sample=start, target_end_sample=start + len(crop),
            pad_left=start, pad_right=TARGET_NUM_SAMPLES-start-len(crop),
            rms_fallback=scaling['rms_fallback'], target_gain=scaling['target_gain'],
            active_sample_fraction=scaling['active_sample_fraction'],
            snr_target_met=scaling['snr_target_met'],
            snr_db=float(snr_db),
            snr_db_before_peak=scaling['snr_db_exported']))
    mixture = bg + targets[0] + targets[1]
    peak = float(np.max(np.abs(mixture)))
    divisor = peak if peak > 1 else 1.

    #返回的meta双 FSD 混音过程的"完整描述元数据"字典
    return (bg / divisor, [t / divisor for t in targets], mixture / divisor,
            dict(seed=int(seed), events=events, order=order, crop_attempts=attempt+1,
                 crop_sampling=crop_sampling,
                 gap_samples=gap_samples, snr_db=float(snr_db), snr_db_requested=float(snr_db),
                 snr_definition='each FSD active RMS / full 10s AudioSet RMS',
                 snr_scaling='active_power_ratio_foreground_gain', zero_mean=True,
                 mixture_peak_before_final_norm=peak, final_normalization_divisor=divisor))
