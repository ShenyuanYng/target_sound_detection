"""通用波形工具：读取、校验、裁剪、补零、零均值 RMS 与 SNR 增益。

Array operations preserve inputs and caller-owned RNG state. Sample ends are
exclusive. FSD resampling lives in fsd50k_cache; Active RMS mixing lives in
active_rms. This module does not import either pipeline.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
TARGET_NUM_SAMPLES = 160000
DEFAULT_SNR_DB_RANGE = (-5.0, 5.0)

GEOMETRY_KEYS = (
    'num_nonzero_samples', 'num_repeats', 'repeated_num_samples',
    'sampled_crop_start_sample', 'sampled_crop_end_sample',
    'crop_start_sample', 'crop_end_sample', 'crop_num_samples',
    'truncated_num_samples', 'pad_left', 'pad_right',
    'target_start_sample', 'target_end_sample',
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


#内部校验器：把输入任意波形转为float32的1D数组，要求[非空，有限，单声道]，每个元素是一个采样点在某一时刻的幅度值（不是频谱、不是特征，是原始波形采样）
def _waveform(waveform):
    value = np.asarray(waveform, dtype=np.float32)
    if value.ndim != 1 or not len(value) or not np.isfinite(value).all():
        raise ValueError("Expected a nonempty, finite, mono waveform")
    return value

# Random Number Generator（随机数生成器）
def _rng(rng):
    if isinstance(rng, np.random.Generator):
        return copy.deepcopy(rng)
    if isinstance(rng, (int, np.integer)) and not isinstance(rng, bool):
        return np.random.default_rng(int(rng))
    raise TypeError("rng must be an explicit integer seed or numpy Generator")


def sample_rng(background_id, fsd_id, target_mid, seed):
    """输入：background_id、fsd_id、target_mid、seed
输出：np.random.Generator
作用：用 sha256(json.dumps([...])) 前 16 字节构造种子，与进程、迭代顺序、Python hash salt 无关。
因此同一 (background_id, fsd_id, mid, seed) 在任何机器/worker 上都会得到同样的裁剪与补齐随机序列——CV 可复现的基础。"""
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    key = json.dumps([str(background_id), str(fsd_id), str(target_mid), int(seed)],
                     ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:16], "big"))

#输入弱标签集合，FSD类别集合，合法目标=FSD类别-背景弱标签
def eligible_target_mids(weak_mids, fsd_mids):
    """Exact MID exclusion only; no display-name or ontology matching."""
    return sorted(set(fsd_mids) - set(weak_mids))

#输入音频路径，读背景：要求正好16k，拒绝而非重采样；多通道取平均转单声道
def read_mono_waveform(path):
    """Read mono float32 at exactly 16 kHz. Reject, never resample, other rates."""
    info = sf.info(str(path))
    if info.samplerate != SAMPLE_RATE:
        raise ValueError(f"Sample rate {info.samplerate} != {SAMPLE_RATE}: {path}; no resampling performed")
    waveform, rate = sf.read(str(path), dtype="float32", always_2d=True)
    if rate != SAMPLE_RATE:
        raise ValueError(f"Sample rate changed while reading {path}")
    mono = waveform.mean(axis=1, dtype=np.float32)
    return _waveform(mono).copy(), {"sample_rate": int(rate), "channels": int(info.channels),
                                    "original_num_samples": int(len(mono))}

# 校验背景正好 160000 点（16 kHz、10 秒），不裁剪或补零。
def validate_background(waveform):
    value = _waveform(waveform)
    if len(value) != TARGET_NUM_SAMPLES:
        raise ValueError(f"AudioSet background has {len(value)} samples, expected 160000; no crop/pad performed")
    return value.copy()

#删除所有严格为0的采样点，输出去0后的波形
def remove_zero_energy_samples(waveform):
    """Remove strictly zero mono samples only, including negative zero."""
    value = _waveform(waveform)
    result = value[value != 0].copy()
    if not len(result):
        raise ValueError("FSD target is entirely zero after mono conversion")
    return result

#输入波形，短音频重复到至少2s，输出重复后波形，重复份数
def repeat_to_min_duration(waveform, min_duration=2.0):
    """Return (repeated waveform, total copy count); count=1 means no repeat."""
    value = _waveform(waveform)
    if not math.isfinite(min_duration) or min_duration <= 0:
        raise ValueError("min_duration must be positive and finite")
    minimum = math.ceil(min_duration * SAMPLE_RATE)
    copies = max(1, (minimum + len(value) - 1) // len(value))
    return np.tile(value, copies), int(copies)

#输入波形，rng，最短时长，随机裁剪跨越音频中点的片段，输出（片段，裁剪元数据）
def sample_middle_crossing_segment(waveform, *, rng, min_duration=0.5):
    """Draw start < N/2 and exclusive end > N/2, with length >= minimum.

    Start is uniform over feasible first-half sample indices; end is uniform
    over feasible second-half boundaries conditional on that start.
    """
    value = _waveform(waveform)
    if not math.isfinite(min_duration) or min_duration <= 0:
        raise ValueError("min_duration must be positive and finite")
    minimum = math.ceil(min_duration * SAMPLE_RATE)
    n = len(value)
    if n < minimum:
        raise ValueError("Waveform is shorter than the minimum crop duration")
    generator = _rng(rng)
    start = int(generator.integers(0, min((n - 1) // 2, n - minimum) + 1))
    end = int(generator.integers(max(n // 2 + 1, start + minimum), n + 1))
    return value[start:end].copy(), {"crop_start_sample": start, "crop_end_sample": end,
                                     "crop_num_samples": end - start}

#输入片段，rng，目标长度，超长截断保留前段，否则随机左右补零到10s
def pad_segment_to_target_length(segment, *, rng, target_length=TARGET_NUM_SAMPLES):
    """Keep a long segment's first target_length samples; otherwise randomly pad."""
    value = _waveform(segment)
    if not isinstance(target_length, (int, np.integer)) or target_length < 1:
        raise ValueError("target_length must be a positive sample count")
    retained = value[:target_length]
    gap = target_length - len(retained)
    left = int(_rng(rng).integers(0, gap + 1)) if gap else 0
    right = gap - left
    result = np.pad(retained, (left, right)).astype(np.float32, copy=False)
    return result, {"pad_left": left, "pad_right": right, "crop_num_samples": len(retained),
                    "target_start_sample": left, "target_end_sample": left + len(retained),
                    "truncated_num_samples": len(value) - len(retained)}

#输入原始FSD波形，rng，串起上述预处理，并限制最长10s，并把RNG拆成crop_seed和pad_seed
def prepare_fsd_crop(waveform, *, rng):
    """
    输入：原始 FSD 波形 (N,)（N 不定）、上述稳定 RNG
    输出：(crop, metadata, pad_seed)
    crop：np.float32，shape (min(M, 160000),)，未补齐的目标裁剪
    metadata：裁剪统计（原始长度、非零数、重复份数、裁剪起止等）
    pad_seed：int，从同一个 RNG 派生，留给后续"随机左右补零"用
    """
    original = _waveform(waveform)
    nonzero = remove_zero_energy_samples(original)  # 删除所有严格为零的点，含内部零点。
    repeated, copies = repeat_to_min_duration(nonzero)  # 短片段重复到至少两秒。
    generator = _rng(rng)
    crop_seed, pad_seed = generator.integers(0, np.iinfo(np.int64).max, size=2)
    segment, sampled = sample_middle_crossing_segment(repeated, rng=int(crop_seed))
    crop = segment[:TARGET_NUM_SAMPLES].copy()  # 保留跨中点抽样段的前 10 秒。
    metadata = {
        "fsd_original_num_samples": len(original), "num_nonzero_samples": len(nonzero),
        "num_repeats": copies, "additional_repeats": copies - 1,
        "repeated_num_samples": len(repeated),
        "sampled_crop_start_sample": sampled["crop_start_sample"],
        "sampled_crop_end_sample": sampled["crop_end_sample"],
        "sampled_crop_num_samples": sampled["crop_num_samples"],
        "crop_start_sample": sampled["crop_start_sample"],
        "crop_end_sample": sampled["crop_start_sample"] + len(crop),
        "crop_num_samples": len(crop), "truncated_num_samples": len(segment) - len(crop),
    }
    return crop, metadata, int(pad_seed)


#输入波形，统计采样点数，峰值，超出[-1,1]的采样点数
def waveform_statistics(waveform):
    value = _waveform(waveform)
    return {"num_samples": len(value), "peak": float(np.max(np.abs(value))),
            "samples_outside_unit_range": int(np.count_nonzero(np.abs(value) > 1))}


def compute_rms(waveform):
    """Float64 zero-mean RMS; empty, silent and constant inputs return zero.

    Only the measurement is centered; the input waveform is never modified.
    No energy floor is added. Callers must guard zero before dividing by RMS.
    """
    value = np.asarray(waveform, dtype=np.float64)
    if value.ndim != 1 or not np.isfinite(value).all():
        raise ValueError('Expected a finite mono waveform')
    if not value.size:
        return 0.0
    if np.all(value == value[0]):
        return 0.0
    value = value - np.mean(value)  # 统计时减均值，不平移原波形。
    return float(np.sqrt(np.mean(value * value)))


#输入波形，按zero-mean RMS缩放；只调整增益，不平移波形,对Audioset背景调用
def normalize_rms(waveform):
    """Scale to unit zero-mean RMS, preserving padding zeros and input arrays."""
    value = _waveform(waveform)
    rms = compute_rms(value)
    if rms == 0:
        raise ValueError('Cannot RMS-normalize a waveform with zero zero-mean RMS (silence or DC)')
    normalized = (value.astype(np.float64) / rms).astype(np.float32)
    return _waveform(normalized), rms


def validate_snr_db_range(snr_db_range):
    """Return two finite, ordered dB bounds without drawing random numbers."""
    bounds = np.asarray(snr_db_range, dtype=np.float64)
    if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] > bounds[1]:
        raise ValueError('snr_db_range must contain two finite bounds with low <= high')
    return tuple(float(x) for x in bounds)


def sample_snr_db(snr_db_range=DEFAULT_SNR_DB_RANGE, *, sample_id=None, seed=0, rng=None):
    """Uniform dB draw; sample_id + seed gives an order/worker-stable draw.

    Deterministic draws use an independent RNG and never advance crop/placement
    RNGs. Without sample_id, use the caller RNG or the seeded NumPy train stream.
    """
    low, high = validate_snr_db_range(snr_db_range)
    if sample_id is not None:
        if rng is not None:
            raise ValueError('Use sample_id + seed or rng, not both')
        if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
            raise TypeError('SNR seed must be an integer')
        key = json.dumps(['snr_db_v1', sample_id, int(seed)],
                         sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:16], 'big'))
    elif rng is None:
        rng = np.random
    return float(rng.uniform(low, high)) if low < high else low


def scale_foreground_to_snr(foreground, *, snr_db, background_rms,
                            reference_mask=None, reference_rms=None):
    """
    RMS 归一化：先把两条声音调整到可比较的基准。
    SNR 缩放：再决定目标声音相对于背景应该有多响。
    reference_mask:FSD那些采样点参与RMS计算
    reference_rms：指定用于snr缩放的前景RMS数值，不是又做了一次归一化，而是获得 SNR 公式中分母 R_f 的数值
    """
    value = _waveform(foreground)
    if not np.isfinite(snr_db) or not np.isfinite(background_rms) or background_rms <= 0:
        raise ValueError('SNR scaling requires finite dB and positive background RMS')
    if reference_rms is None:
        if reference_mask is not None:
            reference_mask = np.asarray(reference_mask, dtype=bool)
            if reference_mask.shape != value.shape:
                raise ValueError('SNR reference mask must match foreground shape')
        reference_rms = compute_rms(value if reference_mask is None else value[reference_mask])
    if not np.isfinite(reference_rms) or reference_rms <= 0:
        raise ValueError('SNR scaling requires positive foreground reference RMS')
    with np.errstate(over='ignore', under='ignore'):
        
        gain = float(np.sqrt(np.power(10.0, float(snr_db) / 10.0))
                     * background_rms / reference_rms)
        scaled = (value.astype(np.float64) * gain).astype(np.float32)
    if not np.isfinite(gain):
        raise ValueError('SNR gain is not finite')
    return _waveform(scaled), gain
