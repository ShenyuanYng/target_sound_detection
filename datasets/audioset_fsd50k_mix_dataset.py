"""One dynamic train or immutable CV mixture per AudioSet background.

CV manifests pin source hashes, processing code/versions and explicit geometry.
Changed inputs/settings fail closed; choose a new manifest for a new experiment.
基础混音数据集：Aduioset背景+FSD前景
组织一次训练样本：选择FSD，调用缓存以及混音函数，生成标签，提取特征；管理动态训练混音和固定验证混音
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile
from urllib.parse import quote

import numpy as np
import scipy
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset

from datasets.audioset_logmel_dataset import AudioSetLogMelDataset
from datasets.active_rms import make_snr_mixture, validate_rms_settings
from datasets.frame_gt import generate_frame_gt, POWER_REFERENCES
from datasets.mix_background_audit import audit_backgrounds
from datasets.mix_fsd_audit import audit_fsd_pool, save_fsd_audit
from datasets.fsd50k_cache import file_sha256, read_cached_fsd
from datasets.mix_utils import (eligible_target_mids,
    read_mono_waveform, sample_snr_db, validate_snr_db_range, DEFAULT_SNR_DB_RANGE)


GEOMETRY_KEYS = (
    'num_nonzero_samples', 'num_repeats', 'repeated_num_samples',
    'sampled_crop_start_sample', 'sampled_crop_end_sample',
    'crop_start_sample', 'crop_end_sample', 'crop_num_samples',
    'truncated_num_samples', 'pad_left', 'pad_right',
    'target_start_sample', 'target_end_sample',
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()

"""
输入背景列表，FSD池，类别映射，分区等
校验参数->背景审计->FSD审计（剔除全0）->构建self.features（复用 AudioSetLogMelDataset 的特征管线，拿到 mid_to_index）
->构建合法(FSD,MID)对self.pairs->cv是加载/生成，manifest
"""
class AudioSetFSD50KMixDataset(Dataset):
    def __init__(self, path_list, fsd_pool, class_map_path, *, partition,
                 cache_dir, cv_manifest_path=None, seed=7, snr_db=None,
                 snr_db_range=DEFAULT_SNR_DB_RANGE,
                 power_reference='paper', threshold_db=-40.0,
                 normalization='peak', num_classes=527, sr=16000,
                 window_length=512, hop_length=256, fft_size=None,
                 window_type='hann', n_mels=64, rms_norm=2,
                 zero_background_policy='error', zero_fsd_policy='exclude',
                 fsd_audit_path=None, fsd_rms_mode='active',
                 active_threshold_dbfs=-40., active_frame_ms=20.):
        if partition not in ('tr', 'cv'):
            raise ValueError('Mix dataset supports tr/cv only')
        if (sr, window_length, hop_length, fft_size or window_length) != (16000, 512, 256, 512):
            raise ValueError('Phase 3 GT requires 16 kHz / window=FFT=512 / hop=256')
        if power_reference not in POWER_REFERENCES or not np.isfinite(threshold_db):
            raise ValueError('Invalid frame GT configuration')
        if ((snr_db is not None and not np.isfinite(snr_db)) or
                normalization not in ('peak', 'none') or rms_norm not in (0, 2)):
            raise ValueError('Invalid mix normalization/SNR configuration')
        self.partition, self.cache_dir = partition, str(cache_dir)
        validate_rms_settings(fsd_rms_mode, active_threshold_dbfs, active_frame_ms)
        self.rms_settings = dict(mode=fsd_rms_mode, threshold_dbfs=float(active_threshold_dbfs),
                                 frame_ms=float(active_frame_ms))
        self.seed = int(seed)
        self.snr_db = None if snr_db is None else float(snr_db)
        self.snr_db_range = (validate_snr_db_range(snr_db_range) if snr_db is None else None)
        self.power_reference, self.threshold_db = power_reference, float(threshold_db)
        self.normalization, self.rms_norm = normalization, rms_norm
        self.path_list = sorted(path_list, key=lambda e: e['id'])
        if zero_background_policy not in ('error', 'exclude'):
            raise ValueError('zero_background_policy must be error or exclude')
        self.zero_background_policy = zero_background_policy
        source_backgrounds = digest(self.path_list)
        source_ids = [e['id'] for e in self.path_list]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError('Duplicate background IDs')
        self.background_audit = dict(source_count=len(self.path_list),
            effective_count=len(self.path_list), policy=zero_background_policy,
            scanned=False, excluded=[])
        if zero_background_policy == 'exclude':
            logger = logging.getLogger(__name__)
            audit = audit_backgrounds(self.path_list,
                progress=lambda done, total, issues: logger.info(
                    '%s background audit: %d/%d, issues=%d', partition, done, total, issues))
            invalid = [r for r in audit['issues'] if r['status'] == 'invalid']
            if invalid:
                raise ValueError(f'{partition}: non-silence background errors cannot be excluded: '
                                 f'{json.dumps(invalid)}')
            excluded_ids = {r['background_id'] for r in audit['issues']}
            self.path_list = [e for e in self.path_list if e['id'] not in excluded_ids]
            self.background_audit.update(scanned=True, effective_count=len(self.path_list),
                excluded=audit['issues'], zero_mono_count=audit['zero_mono_count'],
                invalid_count=audit['invalid_count'])
            if excluded_ids:
                logger.warning('%s: excluded %d zero-mono backgrounds; %d -> %d',
                    partition, len(excluded_ids), len(source_ids), len(self.path_list))
        self.fsd_pool = sorted(fsd_pool, key=lambda e: int(e['fsd_id']))
        if not self.path_list or not self.fsd_pool:
            raise ValueError('Background and FSD pools must be nonempty')
        self.fsd_by_id = {e['fsd_id']: e for e in self.fsd_pool}
        if len(self.fsd_by_id) != len(self.fsd_pool):
            raise ValueError('Duplicate FSD IDs')
        if any(e['split'] != {'tr': 'train', 'cv': 'val'}[partition] for e in self.fsd_pool):
            raise ValueError('FSD pool contains the wrong official partition')
        if zero_fsd_policy not in ('error', 'exclude'):
            raise ValueError('zero_fsd_policy must be error or exclude')
        self.zero_fsd_policy = zero_fsd_policy
        logger = logging.getLogger(__name__)
        source_pool = self.fsd_pool
        self.fsd_audit = audit_fsd_pool(source_pool,
            split={'tr': 'train', 'cv': 'val'}[partition],
            progress=lambda done, total, issues: logger.info(
                '%s FSD audit: %d/%d, issues=%d', partition, done, total, issues))
        self.fsd_audit['policy'] = zero_fsd_policy
        if fsd_audit_path is not None:
            save_fsd_audit(fsd_audit_path, self.fsd_audit, source_pool)
        if self.fsd_audit['invalid_count']:
            raise ValueError(f'{partition}: non-silence FSD errors cannot be excluded: '
                             f'{json.dumps(self.fsd_audit["invalid"])}')
        if self.fsd_audit['excluded'] and zero_fsd_policy == 'error':
            raise ValueError(f'{partition}: zero-mono FSD targets: '
                             f'{json.dumps(self.fsd_audit["excluded"])}')
        excluded_fsd_ids = {r['fsd_id'] for r in self.fsd_audit['excluded']}
        self.fsd_pool = [e for e in source_pool if e['fsd_id'] not in excluded_fsd_ids]
        self.fsd_by_id = {e['fsd_id']: e for e in self.fsd_pool}
        logger.info('%s FSD pool: source=%d, effective=%d, excluded=%d', partition,
                    len(source_pool), len(self.fsd_pool), len(excluded_fsd_ids))
        if not self.fsd_pool:
            raise ValueError(f'{partition}: FSD pool must be nonempty after zero-mono exclusion')
        # Reuse the exact baseline feature implementation without modifying it.
        self.features = AudioSetLogMelDataset(
            [{'weak_mids': self.fsd_pool[0]['mids']}], class_map_path,
            num_classes=num_classes, sr=sr, window_length=window_length,
            hop_length=hop_length, fft_size=fft_size, window_type=window_type,
            n_mels=n_mels, rms_norm=rms_norm)
        self.mid_to_index = self.features.mid_to_index
        self.num_classes = num_classes
        mids = {m for e in self.fsd_pool for m in e['mids']}
        if not mids <= self.mid_to_index.keys():
            raise ValueError('Unmapped FSD MID')
        if any(not eligible_target_mids(e['weak_mids'], mids) for e in self.path_list):
            raise ValueError('Background has no legal FSD target')
        # Sample uniformly over legal (audio, MID) pairs using rejection sampling.
        self.pairs = [(e['fsd_id'], m) for e in self.fsd_pool for m in sorted(set(e['mids']))]
        self.manifest = None
        if partition == 'cv':
            if cv_manifest_path is None:
                raise ValueError('CV requires a persistent manifest path')
            settings = dict(seed=self.seed, snr_db=self.snr_db, snr_definition='active',
                rms_settings={**self.rms_settings, 'zero_mean': True,
                              'fallback': 'crop_rms_with_threshold_floor'},
                snr_scaling='active_power_ratio_foreground_gain',
                snr_db_range=list(self.snr_db_range) if self.snr_db_range is not None else None,
                snr_sampling='sample_id_seed_uniform_v1' if self.snr_db is None else 'fixed',
                power_reference=power_reference, threshold_db=self.threshold_db,
                normalization=normalization, rms_norm=rms_norm, n_mels=n_mels,
                source_backgrounds=source_backgrounds,
                background_audit=self.background_audit,
                fsd_audit=self.fsd_audit,
                backgrounds=digest(self.path_list), fsd_pool=digest(self.fsd_pool),
                class_map_sha256=file_sha256(class_map_path),
                versions={k: str(v) for k, v in dict(numpy=np.__version__, scipy=scipy.__version__,
                    soundfile=sf.__version__, torch=torch.__version__, torchaudio=torchaudio.__version__).items()},
                code={name: file_sha256(Path(__file__).with_name(name + '.py')) for name in
                      ('audioset_fsd50k_mix_dataset', 'mix_background_audit', 'mix_fsd_audit', 'mix_utils',
                       'active_rms', 'frame_gt', 'fsd50k_cache', 'audioset_logmel_dataset')})
            self.manifest = self._load_or_create_manifest(Path(cv_manifest_path), settings)
#返回背景数量
    def __len__(self):
        return len(self.path_list)
#输入背景条目，rng  拒绝采样：随机抽（fsd_id,mid），若mid不在背景weak_mids中则返回
#输出（fsd条目，mid）
    def _select(self, background, rng):
        excluded = set(background['weak_mids'])
        while True:
            fsd_id, mid = self.pairs[int(rng.integers(len(self.pairs)))]
            if mid not in excluded:
                return self.fsd_by_id[fsd_id], mid
#输入背景、FSD、目标 mid、seed，读背景→读 FSD 缓存→make_snr_mixture 混音；异常包装成带上下文的 ValueError
#输出(target, mixture, meta, cache)
    def _sample_snr(self, background_id, seed):
        """Train draw seeds advance; CV draw seeds and stored SNRs are fixed."""
        if self.snr_db is not None:
            return self.snr_db
        return sample_snr_db(self.snr_db_range, sample_id=background_id, seed=seed)

    def _mix(self, background, fsd, mid, seed, *, snr_db=None):
        snr_db = self._sample_snr(background['id'], seed) if snr_db is None else snr_db
        try:
            bg, _ = read_mono_waveform(background['audio_path'])
            if not np.any(bg):
                raise ValueError('Background is entirely zero after mono conversion; '
                                 'finite crop-based SNR is undefined. '
                                 'Run tools/audit_mix_dataset.py to audit tr/cv.')
            target, cache = read_cached_fsd(fsd['audio_path'], self.cache_dir)
            _, target, mixture, meta = make_snr_mixture(bg, target,
                background_id=background['id'], fsd_id=fsd['fsd_id'], target_mid=mid,
                seed=seed, background_weak_mids=background['weak_mids'], fsd_mids=fsd['mids'],
                snr_db=snr_db, normalization=self.normalization,
                threshold_dbfs=self.rms_settings['threshold_dbfs'],
                frame_ms=self.rms_settings['frame_ms'])
        except (ValueError, RuntimeError, OSError) as error:
            raise ValueError(f'Mix sample failed: partition={self.partition}, '
                f'background_id={background["id"]}, background_path={background["audio_path"]}, '
                f'fsd_id={fsd["fsd_id"]}, fsd_path={fsd["audio_path"]}, '
                f'target_mid={mid}, seed={seed}: {error}') from error
        return target, mixture, meta, cache
#上个月人员清单路径、设置  文件锁+原子写；已存在则校验 schema/settings/行哈希/背景顺序，变化则报错
#输出manifest字典  manifest 是 cv（验证）分区专属的持久化 JSON 文件
    def _load_or_create_manifest(self, path, settings):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.with_suffix(path.suffix + '.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if path.exists():
                saved = json.loads(path.read_text())
                if saved.get('schema') != 1 or saved.get('settings') != settings:
                    raise ValueError('CV manifest settings/inventory changed; use a new manifest path')
                if saved.get('rows_sha256') != digest(saved['rows']):
                    raise ValueError('CV manifest rows checksum mismatch')
                if [r['background_id'] for r in saved['rows']] != [e['id'] for e in self.path_list]:
                    raise ValueError('CV manifest background order/inventory mismatch')
                if any('snr_db' not in r or not isinstance(r['snr_db'], (int, float)) or
                       not np.isfinite(r['snr_db']) for r in saved['rows']):
                    raise ValueError('CV manifest requires a finite snr_db for every sample')
                return saved
            saved = self._generate_manifest(settings)
            with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as handle:
                json.dump(saved, handle, indent=2, allow_nan=False)
                temporary = handle.name
            os.replace(temporary, path)
            return saved
#输入：设置   对每个背景用确定 RNG 选 FSD/mid/seed 并混音，记录源哈希与几何，输出manifest字典
    def _generate_manifest(self, settings):
        rows = []
        for background in self.path_list:
            # Independent of traversal order and process hash salt.
            rng = np.random.default_rng(int(digest([background['id'], self.seed]), 16))
            fsd, mid = self._select(background, rng)
            seed = int(rng.integers(0, np.iinfo(np.int64).max))
            _, _, meta, cache = self._mix(background, fsd, mid, seed)
            rows.append(dict(background_id=background['id'], background_path=background['audio_path'],
                fsd_id=fsd['fsd_id'], fsd_path=fsd['audio_path'], target_mid=mid,
                seed=seed, snr_db=meta['snr_db'], snr_definition=meta['snr_definition'],
                geometry={key: meta[key] for key in GEOMETRY_KEYS},
                background_sha256=file_sha256(background['audio_path']),
                fsd_sha256=cache['source_sha256']))
        saved = dict(schema=1, settings=settings, rows=rows, rows_sha256=digest(rows))
        return saved

    def __getitem__(self, index):
        return self.get_with_metadata(index)[0]

    """
    输入索引，cv从manifest取固定fsd/mid/seed（并校验哈希/几何）
    tr用全局numpy rng动态抽样；_mix→generate_frame_gt 得强标签→
    可选 std 归一化→_logmel_features 提特征→_frame_centers 算帧中心→
    半开 valid mask→one-hot 条件向量→组装样本
    输出((inputs, target), meta)

    样本结构：inputs = {'logmel', 'weak_labels'(one-hot条件)}；
    target = {'strong_labels', 'strong_mask', 'frame_times', 'class_index', 'utt_id', 'audio_path'}
    """
    def get_with_metadata(self, index):
        """Diagnostic access uses the same path and draw as __getitem__."""
        background = self.path_list[index]
        if self.partition == 'cv':
            row = self.manifest['rows'][index]
            fsd, mid, seed = self.fsd_by_id[row['fsd_id']], row['target_mid'], row['seed']
            if file_sha256(background['audio_path']) != row['background_sha256']:
                raise ValueError('CV background content changed')
        else:
            # Global numpy state is seeded by the existing DataLoader worker init;
            # draws advance also with persistent workers and num_workers=0.
            seed = int(np.random.randint(0, np.iinfo(np.int64).max))
            fsd, mid = self._select(background, np.random.default_rng(seed))
        snr_db = row['snr_db'] if self.partition == 'cv' else None
        target, mixture, meta, cache = self._mix(background, fsd, mid, seed, snr_db=snr_db)
        if self.partition == 'cv':
            if cache['source_sha256'] != row['fsd_sha256']:
                raise ValueError('CV FSD content changed')
            if {key: meta[key] for key in GEOMETRY_KEYS} != row['geometry']:
                raise ValueError('CV crop/padding replay differs from persisted manifest')
        labels = generate_frame_gt(target, threshold_db=self.threshold_db,
                                   power_reference=self.power_reference)['gt'].astype(np.float32)
        if self.rms_norm == 2:
            scale = float(np.std(mixture))
            if scale > 0:
                mixture = mixture / scale
        logmel = self.features._logmel_features(mixture)
        centers = self.features._frame_centers(len(logmel), 0)
        # Baseline's half-open valid centers: final 10.000 s frame is masked.
        mask = (centers < len(mixture) / self.features.sr).astype(np.float32)
        condition = np.zeros(self.num_classes, dtype=np.float32)
        class_index = self.mid_to_index[mid]
        condition[class_index] = 1
        identity = [self.partition, background['id'], fsd['fsd_id'], mid, str(seed)]
        utt_id = '|'.join(quote(value, safe='') for value in identity)
        sample = ({'logmel': logmel, 'weak_labels': condition}, dict(
            strong_labels=labels, strong_mask=mask, frame_times=centers,
            class_index=class_index, utt_id=utt_id, audio_path=background['audio_path']))
        return sample, {**meta, 'fsd_split': fsd['split'], 'cache_hit': cache['cache_hit']}
