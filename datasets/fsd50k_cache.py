"""Persistent mono float32 FSD cache; original files are opened read-only.
读取FSD波形，通道平均转单声道，44.1kHz重采样的16k，缓存不修改原始文件
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import fcntl

import numpy as np
import scipy
from scipy.signal import resample_poly
import soundfile as sf

from datasets.mix_utils import SAMPLE_RATE, _waveform

#输入文件路径，分块计算SHA256，输出十六进制哈希字符串
def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

#输入FSD音频路径，缓存目录计算源哈希→校验采样率（仅 44100/16000）→构造 identity→
# 按 key 加文件锁读缓存；未命中则重采样并原子发布（wav+json manifest，manifest 最后发布）；校验缓存完整性
#输出(16k单声道波形, identity/cache_hit等字典)
def read_cached_fsd(path, cache_dir):
    """Cache deterministic 44100->16000 polyphase resampling (160/441).

    16 kHz inputs pass through into the same cache without resampling. Other
    rates are rejected. The key includes source content and software versions.
    Per-key file locks and atomic publication prevent partial concurrent reads.
    """
    source, root = Path(path).resolve(strict=True), Path(cache_dir).resolve()
    if root == source.parent or source.parent in root.parents:
        raise ValueError('Cache must be outside the original audio directory')
    digest = file_sha256(source)
    info = sf.info(str(source))
    if info.samplerate not in (44100, SAMPLE_RATE):
        raise ValueError(f'Unsupported FSD sample rate {info.samplerate}: {source}')
    identity = {'source_path': str(source), 'source_sha256': digest,
                'source_sample_rate': info.samplerate, 'source_num_samples': info.frames,
                'source_channels': info.channels, 'output_sample_rate': SAMPLE_RATE,
                'algorithm': 'mono-mean-float32_resample_poly_kaiser5_constant_v1',
                'scipy_version': scipy.__version__, 'numpy_version': np.__version__,
                'soundfile_version': sf.__version__}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    root.mkdir(parents=True, exist_ok=True)
    wav, manifest = root / (key + '.wav'), root / (key + '.json')
    with (root / (key + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        hit = manifest.exists()
        if hit:
            meta = json.loads(manifest.read_text())
            if meta['identity'] != identity or file_sha256(wav) != meta['cache_sha256']:
                raise ValueError(f'Cache integrity check failed: {wav}')
        else:
            value, rate = sf.read(str(source), dtype='float32', always_2d=True)
            if rate != info.samplerate:
                raise ValueError('FSD source rate changed during read')
            mono = _waveform(value.mean(axis=1, dtype=np.float32))
            converted = (resample_poly(mono, 160, 441, window=('kaiser', 5.0), padtype='constant')
                         if rate == 44100 else mono.copy())
            converted = _waveform(converted)
            if file_sha256(source) != digest:
                raise ValueError('FSD source changed during resampling')
            with tempfile.NamedTemporaryFile(dir=root, suffix='.wav', delete=False) as temporary:
                temp_wav = Path(temporary.name)
            sf.write(str(temp_wav), converted, SAMPLE_RATE, subtype='FLOAT')
            meta = {'identity': identity, 'cache_num_samples': len(converted),
                    'cache_sha256': file_sha256(temp_wav)}
            with tempfile.NamedTemporaryFile(mode='w', dir=root, suffix='.json', delete=False) as temporary:
                json.dump(meta, temporary, indent=2)
                temp_manifest = Path(temporary.name)
            os.replace(temp_wav, wav)
            os.replace(temp_manifest, manifest)  # manifest published last under lock
        waveform, rate = sf.read(str(wav), dtype='float32')
        expected = (info.frames * SAMPLE_RATE + info.samplerate - 1) // info.samplerate
        if rate != SAMPLE_RATE or len(waveform) != expected:
            raise ValueError(f'Invalid cached waveform format: {wav}')
        waveform = _waveform(waveform)
        return waveform, {**identity, 'cache_path': str(wav), 'cache_hit': hit,
                          'resampled': info.samplerate != SAMPLE_RATE,
                          'resampled_num_samples': len(waveform)}
