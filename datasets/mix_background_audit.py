"""Read-only waveform checks for mix backgrounds; never change split membership.
检查AudioSet背景音是否合法（是否全0/非16k等）
"""
from __future__ import annotations

import numpy as np
import soundfile as sf

from datasets.fsd50k_cache import file_sha256
from datasets.mix_utils import read_mono_waveform, validate_background, compute_rms

#输入背景条目（含id，audio_path），读单声道->检验16k，算RMS，
# 全零判 zero_mono（区分原始全零 vs 通道抵消），异常判 invalid
def inspect_background(entry):
    record = {'background_id': entry['id'], 'audio_path': entry['audio_path']}
    try:
        waveform, info = read_mono_waveform(entry['audio_path'])
        validate_background(waveform)
        record.update(info, rms=compute_rms(waveform))
        if not np.any(waveform):
            original, _ = sf.read(entry['audio_path'], dtype='float32', always_2d=True)
            record.update(status='zero_mono',
                reason='zero_samples' if not np.any(original) else 'channel_cancellation',
                source_sha256=file_sha256(entry['audio_path']))
        else:
            record['status'] = 'ok'
    except (ValueError, RuntimeError, OSError) as error:
        record.update(status='invalid', reason=str(error))
    return record

#输入背景列表、进度回调，遍历审计收集非ok记录
def audit_backgrounds(entries, *, progress=None):
    issues = []
    for index, entry in enumerate(entries, 1):
        record = inspect_background(entry)
        if record['status'] != 'ok':
            issues.append(record)
        if progress is not None and (index % 1000 == 0 or index == len(entries)):
            progress(index, len(entries), len(issues))
    zeros = sum(r['status'] == 'zero_mono' for r in issues)
    return dict(source_count=len(entries), valid_count=len(entries)-len(issues),
                zero_mono_count=zeros, invalid_count=len(issues)-zeros, issues=issues)
