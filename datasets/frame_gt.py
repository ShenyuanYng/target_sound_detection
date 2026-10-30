"""Target-only frame pseudo-strong GT, independent of Dataset/model/loss.

Matches AudioSetLogMelDataset's centered frame geometry, not spectral power:
raw frames are NOT Hann-weighted and no FFT is used, following paper 5.4.3.
Threshold is mandatory; +40/-40 are experiments, not a training default.
根据FSD目标波形的逐帧能量生成伪强标签
"""
from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000
NUM_SAMPLES = 160000
N_FFT = WIN_LENGTH = 512
HOP_LENGTH = 256
NUM_FRAMES = 626
POWER_REFERENCES = ('paper', 'waveform_mean')


def framing_description():
    return {'sample_rate': SAMPLE_RATE, 'n_fft': N_FFT, 'win_length': WIN_LENGTH,
            'hop_length': HOP_LENGTH, 'center': True, 'pad_mode': 'constant',
            'pad_samples_each_side': N_FFT//2, 'num_samples': NUM_SAMPLES,
            'num_frames': NUM_FRAMES, 'power_window': 'raw unweighted samples; no FFT'}

#输入16k采样目标波形，居中0填充后滑动窗口切帧，得到[626，512]
def frame_target_waveform(target_waveform):
    """Return raw [626,512] frames, centered at sample 0,256,...,160000.

    Ends are exclusive; supports extend 256 samples beyond either audio edge.
    The final frame centered at exactly 10 s is retained and may be positive.
    """
    value = np.asarray(target_waveform, dtype=np.float64)
    if value.ndim != 1 or len(value) != NUM_SAMPLES or not np.isfinite(value).all():
        raise ValueError('Target must be finite mono 160000 samples at 16 kHz')
    padded = np.pad(value, (N_FFT//2, N_FFT//2), mode='constant')
    frames = np.lib.stride_tricks.sliding_window_view(padded, WIN_LENGTH)[::HOP_LENGTH].copy()
    if frames.shape != (NUM_FRAMES, WIN_LENGTH):
        raise RuntimeError('Frame geometry must match the 626-frame Log-Mel input')
    return frames

#输入目标波形，功率基准，计算每帧的能量，非0采样数，P_part、P_all、能量比 ER(dB)（全零帧 P_part=0, ER=-inf）
def analyze_target_power(target_waveform, *, power_reference='paper'):
    """Nonzero-only powers. Silence is P_part=0 and ER=-inf, without epsilon.

    paper: literal Eq.(12), sum of overlapping frame energies / number of
    nonzero ORIGINAL waveform samples. At 50% overlap this is twice the usual
    waveform mean power. The paper's denominator interpretation is ambiguous;
    waveform_mean provides the explicit conventional comparison, never silently
    substitutes it. Neither reference includes added zero padding in its count.
    """
    if power_reference not in POWER_REFERENCES:
        raise ValueError('power_reference must be paper or waveform_mean')
    frames = frame_target_waveform(target_waveform)
    value = np.asarray(target_waveform, dtype=np.float64)
    counts = np.count_nonzero(frames, axis=1)
    total_nonzero = int(np.count_nonzero(value))
    with np.errstate(over='raise', invalid='raise'):
        try:
            frame_energy = np.sum(frames * frames, axis=1, dtype=np.float64)
            waveform_energy = float(np.sum(value * value, dtype=np.float64))
        except FloatingPointError as error:
            raise ValueError('Target magnitude overflows float64 power calculation') from error
    p_part = np.divide(frame_energy, counts, out=np.zeros(NUM_FRAMES,dtype=np.float64), where=counts>0)
    numerator = float(frame_energy.sum()) if power_reference == 'paper' else waveform_energy
    p_all = numerator/total_nonzero if total_nonzero else 0.0
    er = np.full(NUM_FRAMES, -np.inf, dtype=np.float64)
    valid = (counts > 0) & (p_part > 0)
    if p_all > 0:
        er[valid] = 10*(np.log10(p_part[valid])-np.log10(p_all))
    centers = np.arange(NUM_FRAMES,dtype=np.int64)*HOP_LENGTH
    return {'p_part':p_part, 'p_all':float(p_all), 'er_db':er,
            'frame_nonzero_counts':counts, 'waveform_nonzero_count':total_nonzero,
            'framed_nonzero_count':int(counts.sum()),
            'waveform_mean_power':waveform_energy/total_nonzero if total_nonzero else 0.0,
            'frame_center_samples':centers, 'frame_start_samples':centers-N_FFT//2,
            'frame_end_samples':centers+N_FFT//2, 'power_reference':power_reference,
            'framing':framing_description()}

#输入分析结果，阈值  严格ER>threshold且帧非0->1，否则0
def threshold_frame_power(analysis, *, threshold_db):
    """Strict ER > threshold, with all-zero frames always zero."""
    if not np.isscalar(threshold_db) or not np.isfinite(threshold_db):
        raise ValueError('threshold_db must be a finite scalar')
    return ((analysis['frame_nonzero_counts']>0) & np.isfinite(analysis['er_db'])
            & (analysis['er_db']>float(threshold_db))).astype(np.uint8)


def generate_frame_gt(target_waveform, *, threshold_db, power_reference='paper'):
    analysis = analyze_target_power(target_waveform,power_reference=power_reference)
    return {**analysis, 'gt':threshold_frame_power(analysis,threshold_db=threshold_db),
            'threshold_db':float(threshold_db)}
