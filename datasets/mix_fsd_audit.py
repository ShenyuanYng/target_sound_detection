"""Read-only audit of one official FSD pool before any target sampling.
检查FSD音频池，排除转为单声道后完全为0的音频
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import soundfile as sf

from datasets.fsd50k_cache import file_sha256
from datasets.mix_utils import compute_rms


def inspect_fsd(entry):
    record = dict(fsd_id=entry['fsd_id'], audio_path=entry['audio_path'],
                  split=entry['split'])
    try:
        before = file_sha256(entry['audio_path'])
        record['source_sha256'] = before
        samples, rate = sf.read(entry['audio_path'], dtype='float32', always_2d=True)
        record.update(sample_rate=int(rate), channels=int(samples.shape[1]),
                      original_num_samples=len(samples), mono_num_samples=len(samples))
        if rate not in (16000, 44100):
            raise ValueError(f'Unsupported FSD sample rate {rate}')
        if not len(samples) or not np.isfinite(samples).all():
            raise ValueError('Expected nonempty finite FSD audio')
        # Exactly the float32 channel mean used by the existing 16 kHz cache.
        mono = samples.mean(axis=1, dtype=np.float32)
        rms = compute_rms(mono)  # float64 accumulation; no silence threshold.
        if file_sha256(entry['audio_path']) != before:
            raise ValueError('FSD source changed during audit')
        record['rms'] = rms
        if not np.any(mono) or rms == 0:
            record.update(status='zero_mono',
                reason='zero_samples' if not np.any(samples) else 'channel_cancellation')
        else:
            record['status'] = 'ok'
    except (ValueError, RuntimeError, OSError) as error:
        record.update(status='invalid', reason=str(error))
    return record


def audit_fsd_pool(entries, *, split, progress=None):
    """Scan the complete pool, excluding only exact mono zeros within this split."""
    if split not in ('train', 'val') or any(e['split'] != split for e in entries):
        raise ValueError('FSD pool contains the wrong official partition')
    ids = [e['fsd_id'] for e in entries]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate FSD IDs')
    excluded, invalid = [], []
    inventory = hashlib.sha256()
    for index, entry in enumerate(sorted(entries, key=lambda e: int(e['fsd_id'])), 1):
        record = inspect_fsd(entry)
        inventory.update(json.dumps(record, sort_keys=True, allow_nan=False).encode() + b'\n')
        if record['status'] == 'zero_mono':
            excluded.append(record)
        elif record['status'] == 'invalid':
            invalid.append(record)
        if progress is not None and (index % 1000 == 0 or index == len(entries)):
            progress(index, len(entries), len(excluded) + len(invalid))
    return dict(split=split, source_count=len(entries),
        effective_count=len(entries)-len(excluded)-len(invalid), scanned=True,
        zero_mono_count=len(excluded), invalid_count=len(invalid), excluded=excluded,
        invalid=invalid, source_inventory_sha256=inventory.hexdigest())
