"""Single FSD stem extraction with exact-MID conflict filtering and fixed CV.

Only mixture generation is reused from SED. No logmel, frame labels, condition
expansion, or mixture-only standard-deviation normalization is performed.
"""
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np
import scipy
import soundfile as sf
from torch.utils.data import Dataset

from datasets.audioset_fsd50k_mix_dataset import AudioSetFSD50KMixDataset, digest, GEOMETRY_KEYS
from datasets.fsd50k_cache import file_sha256, read_cached_fsd
from datasets.mix_background_audit import audit_backgrounds
from datasets.mix_fsd_audit import audit_fsd_pool
from datasets.active_rms import make_snr_mixture
from datasets.mix_utils import (read_mono_waveform, sample_snr_db,
                                validate_snr_db_range)
from datasets.paths.audioset_paths import load_class_map, resolve_path
from datasets.paths.fsd50k_paths import read_fsd_dev_index
from datasets.strong_mix import build_strong_mix_paths, exclude_declared_train_entries

MODE = 'audioset_strong_fsd50k_tse'


def forbidden_mids(background):
    return (set(background.get('original_weak_mids', background.get('weak_mids', [])))
            | set(background.get('strong_positive_mids', []))
            | {e['mid'] for e in background.get('events', []) if e['present']})


def waveform_hash(value):
    import hashlib
    return hashlib.sha256(np.asarray(value, dtype='<f4').tobytes()).hexdigest()


def is_multilabel(entry):
    # Count all source annotations, including any MID outside the model's map.
    return len(set(entry.get('original_mids', entry['mids']))) > 1


class NoLegalPairError(ValueError):
    def __init__(self, report):
        self.report = report
        super().__init__('TSE backgrounds have no legal FSD/MID pairing: '
                         + json.dumps(report['no_legal_pair_background_ids']))


class TSEMixDataset(Dataset):
    def __init__(self, path_list, fsd_pool, class_map_path, *, partition,
                 cache_dir, cv_manifest_path=None, seed=7, snr_db=None,
                 snr_db_range=(-5., 5.), active_threshold_dbfs=-40., active_frame_ms=20.,
                 no_pair_policy='error', zero_background_policy='exclude',
                 zero_fsd_policy='exclude', num_classes=527):
        if partition not in ('tr', 'cv') or no_pair_policy not in ('error', 'exclude'):
            raise ValueError('Invalid TSE partition/no-pair policy')
        if zero_background_policy not in ('error', 'exclude') or zero_fsd_policy not in ('error', 'exclude'):
            raise ValueError('Invalid zero-waveform policy')
        if snr_db is not None and not np.isfinite(snr_db):
            raise ValueError('SNR must be finite')
        from datasets.active_rms import validate_rms_settings
        validate_rms_settings('active', active_threshold_dbfs, active_frame_ms)
        self.partition, self.cache_dir, self.seed = partition, str(cache_dir), int(seed)
        self.snr_db = snr_db
        self.snr_db_range = validate_snr_db_range(snr_db_range)
        self.threshold, self.frame_ms = active_threshold_dbfs, active_frame_ms
        self.mid_to_index, names = load_class_map(class_map_path)
        if len(names) != num_classes:
            raise ValueError('Class map does not match num_classes')
        self.num_classes = num_classes
        paths = sorted(path_list, key=lambda e: e['id'])
        pool = sorted(fsd_pool, key=lambda e: int(e['fsd_id']))
        if not paths or not pool or len({e['id'] for e in paths}) != len(paths):
            raise ValueError('TSE needs nonempty unique backgrounds and FSD pool')
        if len({e['fsd_id'] for e in pool}) != len(pool):
            raise ValueError('Duplicate FSD IDs')
        if any(not e['mids'] or not set(e['mids']) <= self.mid_to_index.keys() for e in pool):
            raise ValueError('FSD must contain mapped MID labels')
        self.background_audit = audit_backgrounds(paths)
        self.fsd_audit = audit_fsd_pool(pool, split={'tr': 'train', 'cv': 'val'}[partition])
        if self.fsd_audit['invalid_count'] or any(e['status'] == 'invalid' for e in self.background_audit['issues']):
            raise ValueError('Invalid source audio: ' + json.dumps(
                dict(backgrounds=self.background_audit, fsd=self.fsd_audit)))
        bad_bg = {e['background_id'] for e in self.background_audit['issues']}
        bad_fsd = {e['fsd_id'] for e in self.fsd_audit['excluded']}
        if (bad_bg and zero_background_policy == 'error') or (bad_fsd and zero_fsd_policy == 'error'):
            raise ValueError(f'Zero-energy sources: backgrounds={sorted(bad_bg)}, FSD={sorted(bad_fsd)}')
        self.fsd_pool = [e for e in pool if e['fsd_id'] not in bad_fsd]
        self.fsd_by_id = {e['fsd_id']: e for e in self.fsd_pool}
        self.by_mid = defaultdict(list)
        for entry in self.fsd_pool:
            for mid in sorted(set(entry['mids'])):
                self.by_mid[mid].append(entry)
        self.mid_weights = {m: np.array([1. / len(set(e['mids'])) for e in entries])
                            for m, entries in self.by_mid.items()}
        self.mid_mass = {m: float(w.sum()) for m, w in self.mid_weights.items()}
        self.allowed = {}
        blocked, retained = [], []
        for entry in paths:
            if entry['id'] in bad_bg:
                continue
            mids = sorted(self.by_mid.keys() - forbidden_mids(entry))
            if not mids:
                blocked.append(entry['id'])
            else:
                retained.append(entry)
                self.allowed[entry['id']] = mids
        self.path_list = retained
        coverage = Counter(m for mids in self.allowed.values() for m in mids)
        multi = sum(is_multilabel(e) for e in self.fsd_pool)
        reachable_fsd = {e['fsd_id'] for mid in coverage for e in self.by_mid[mid]}
        legal_pairs = sum(len(self.by_mid[m]) * count for m, count in coverage.items())
        all_pairs = sum(len(set(e['mids'])) for e in self.fsd_pool) * (len(paths) - len(bad_bg))
        multi_mass = {m: sum(1. / len(set(e['mids'])) for e in entries if is_multilabel(e))
                      for m, entries in self.by_mid.items()}
        expected_multi = [sum(multi_mass[m] for m in mids) / sum(self.mid_mass[m] for m in mids)
                          for mids in self.allowed.values()]
        self.report = dict(
            partition=partition, source_background_count=len(paths),
            effective_background_count=len(retained), samples_per_epoch=len(retained),
            zero_background_ids=sorted(bad_bg), no_legal_pair_policy=no_pair_policy,
            no_legal_pair_background_ids=blocked, source_fsd_count=len(pool),
            effective_fsd_count=len(self.fsd_pool), zero_fsd_ids=sorted(bad_fsd),
            reachable_fsd_count=len(reachable_fsd),
            unreachable_fsd_ids=sorted(self.fsd_by_id.keys() - reachable_fsd),
            legal_background_fsd_mid_pair_count=legal_pairs,
            rejected_background_fsd_mid_pair_count=all_pairs - legal_pairs,
            eligible_condition_class_count=len(coverage),
            eligible_condition_background_counts=dict(sorted(coverage.items())),
            unreachable_fsd_mids=sorted(self.by_mid.keys() - coverage.keys()),
            multilabel_fsd_count=multi, multilabel_fsd_fraction=multi / max(1, len(self.fsd_pool)),
            fsd_with_unmapped_labels_count=sum(bool(set(e.get('original_mids', e['mids']))
                - self.mid_to_index.keys()) for e in self.fsd_pool),
            expected_sample_multilabel_fraction=float(np.mean(expected_multi)) if expected_multi else None,
            condition_policy='FSD MID only; exclude exact original Weak/positive Strong MID',
            sampling_policy='uniform FSD then uniform MID, conditioned on legal pairing',
            target_semantics='entire processed FSD stem, including co-occurring sounds',
            unresolved_ambiguity='weak labels do not guarantee target presence in the crop or background absence')
        if blocked and no_pair_policy == 'error':
            raise NoLegalPairError(self.report)
        if not retained or not self.fsd_pool:
            raise ValueError('No usable TSE backgrounds/FSD remain: ' + json.dumps(self.report))
        self.manifest = None
        if partition == 'cv':
            if cv_manifest_path is None:
                raise ValueError('TSE CV requires its own persistent manifest')
            settings = dict(task=MODE, version=1, seed=self.seed, snr_db=snr_db,
                snr_db_range=list(self.snr_db_range), threshold=active_threshold_dbfs,
                frame_ms=active_frame_ms, normalization='joint_peak_only',
                no_pair_policy=no_pair_policy, zero_background_policy=zero_background_policy,
                zero_fsd_policy=zero_fsd_policy,
                path_inventory=digest(paths), fsd_inventory=digest(pool),
                effective_paths=digest(self.path_list), effective_fsd=digest(self.fsd_pool),
                class_map_sha256=file_sha256(class_map_path),
                source_audits=dict(backgrounds=self.background_audit, fsd=self.fsd_audit),
                versions=dict(numpy=np.__version__, scipy=scipy.__version__, soundfile=sf.__version__),
                code={name: file_sha256(Path(__file__).with_name(name + '.py')) for name in
                      ('tse_dataset', 'mix_utils', 'active_rms', 'fsd50k_cache',
                       'mix_background_audit', 'mix_fsd_audit', 'audioset_fsd50k_mix_dataset')})
            # Reuse locking, atomic publication, setting and row-integrity checks.
            self.manifest = AudioSetFSD50KMixDataset._load_or_create_manifest(
                self, Path(cv_manifest_path), settings)
            counts = Counter(row['target_mid'] for row in self.manifest['rows'])
            observed_multi = sum(is_multilabel(self.fsd_by_id[r['fsd_id']])
                                 for r in self.manifest['rows'])
            self.report.update(cv_condition_class_count=len(counts), cv_condition_counts=dict(counts),
                cv_multilabel_sample_count=observed_multi,
                cv_multilabel_sample_fraction=observed_multi / len(self.path_list),
                cv_manifest_rows_sha256=self.manifest['rows_sha256'])

    def __len__(self):
        return len(self.path_list)

    def _select(self, background, rng):
        # Finite direct sampling, equivalent to rejecting conflicts in the old
        # uniform-audio -> uniform-MID law. Never loops until a pair is found.
        mids = self.allowed[background['id']]
        masses = np.array([self.mid_mass[m] for m in mids])
        mid = str(rng.choice(mids, p=masses / masses.sum()))
        weights = self.mid_weights[mid]
        entry = self.by_mid[mid][int(rng.choice(len(weights), p=weights / weights.sum()))]
        return entry, mid
    """
    mix只负责"读文件+身份/标签检验"，所有的波形变换（清理，重复，裁剪，缩放，补齐，相加，峰值归一化）
    都收敛到make_snr_mixture这个单一入口，保证训练/CV/审计走同一条路径
    """
    def _mix(self, background, fsd, mid, seed, snr_db):
        if mid in forbidden_mids(background) or mid not in fsd['mids']:
            raise ValueError(f'TSE label conflict: {background["id"]}, {fsd["fsd_id"]}, {mid}')
        bg, _ = read_mono_waveform(background['audio_path'])#读取Audioset背景
        #读取FSD，cache_dir是缓存目录
        raw_target, cache = read_cached_fsd(fsd['audio_path'], self.cache_dir)
        bg, target, mixture, meta = make_snr_mixture(bg, raw_target,
            background_id=background['id'], fsd_id=fsd['fsd_id'], target_mid=mid, seed=seed,
            background_weak_mids=sorted(forbidden_mids(background)), fsd_mids=fsd['mids'],
            snr_db=snr_db, normalization='peak',
            threshold_dbfs=self.threshold, frame_ms=self.frame_ms)
        if any(x.shape != (160000,) or not np.isfinite(x).all() for x in (bg, target, mixture)):
            raise ValueError('Invalid 10-second stem')
        error = float(np.max(np.abs(mixture - (bg + target))))
        if error > 1e-6:
            raise ValueError(f'Mixture/stem inconsistency: {error}')
        return bg, target, mixture, {**meta, 'fsd_sha256': cache['source_sha256'],
            'stem_sum_max_error': error, 'fsd_mids': sorted(set(fsd.get('original_mids', fsd['mids']))),
            'mapped_fsd_mids': sorted(set(fsd['mids'])),
            'multilabel_fsd': is_multilabel(fsd), 'fsd_split': fsd['split']}

    def _snr(self, background, seed):
        return (float(self.snr_db) if self.snr_db is not None else
                sample_snr_db(self.snr_db_range, sample_id=background['id'], seed=seed))

    def _row(self, background, fsd, mid, seed, snr, stems, meta):
        return dict(background_id=background['id'], fsd_id=fsd['fsd_id'], target_mid=mid,
            background_path=background['audio_path'], fsd_path=fsd['audio_path'],
            seed=seed, snr_db=snr, background_sha256=file_sha256(background['audio_path']),
            fsd_sha256=meta['fsd_sha256'], geometry={k: meta[k] for k in GEOMETRY_KEYS},
            target_gain=meta['target_gain'], final_normalization_divisor=meta['final_normalization_divisor'],
            waveform_sha256={k: waveform_hash(v) for k, v in zip(('background', 'target', 'mixture'), stems)})

    def _generate_manifest(self, settings):
        rows = []
        for background in self.path_list:
            rng = np.random.default_rng(int(digest([MODE, background['id'], self.seed]), 16))
            fsd, mid = self._select(background, rng)
            seed = int(rng.integers(0, np.iinfo(np.int64).max))
            snr = self._snr(background, seed)
            *stems, meta = self._mix(background, fsd, mid, seed, snr)
            rows.append(self._row(background, fsd, mid, seed, snr, stems, meta))
        return dict(schema=1, settings=settings, rows=rows, rows_sha256=digest(rows))

    def get_with_metadata(self, index):
        #background是dict，含'id'、'audio_path'、'weak_mids'、'strong_positive_mids'、'events' 等字段
        background = self.path_list[index]
        if self.partition == 'cv':
            row = self.manifest['rows'][index]
            fsd, mid, seed, snr = self.fsd_by_id[row['fsd_id']], row['target_mid'], row['seed'], row['snr_db']
        else:
            seed = int(np.random.randint(0, np.iinfo(np.int64).max))
            """
            输入background，rng
            输出：fsd（FSD pool 里的一个 dict，含 fsd_id/audio_path/mids/split/original_mids）+ mid（str，目标 MID，如 "/m/0bt9lr"）
            """
            fsd, mid = self._select(background, np.random.default_rng(seed))
            snr = self._snr(background, seed)#输入 background，seed 输出SNR (dB,float类型)

            """
            输出三个numpy数组和一个meta dict bg：背景波形shape (160000,)target：目标 stem 波形，shape (160000,)
            mixture：混音波形，shape (160000,)，且严格满足 mixture ≈ bg + target

            作用：混音，读背景单声道波形，读FSD波形，调make_snr_mixture 做 SNR 混音 + 峰值归一化 → 校验 10 秒长度与有限性 → 返回四元组。
            """
        bg, target, mixture, meta = self._mix(background, fsd, mid, seed, snr)
        
        #cv重放校验
        if self.partition == 'cv' and self._row(background, fsd, mid, seed, snr, (bg, target, mixture), meta) != row:
            raise ValueError('TSE CV source content, geometry, gain or waveform replay changed')
        
        #condition生成，one-hot
        condition = np.zeros(self.num_classes, dtype=np.float32)
        condition[self.mid_to_index[mid]] = 1.
        #构造sample是一个 二元组 (inputs, targets)，两个元素都是 dict：
        sample = (dict(mixture_waveform=mixture, condition=condition), dict(
            target_waveform=target, background_waveform=bg, mixture_waveform=mixture,
            class_index=self.mid_to_index[mid], condition_mid=mid,
            multilabel_fsd=int(meta['multilabel_fsd']),
            utt_id=f'{self.partition}|{background["id"]}|{fsd["fsd_id"]}|{mid}|{seed}'))
        return sample, meta

    def __getitem__(self, index):
        """
        单条样本的生产者，决定一条样本里有哪些字段，每个字段的shape
        get_with_metadata会返回sample和meta，但是只需要sample

        """
        return self.get_with_metadata(index)[0]


def tse_paths(hp, partition):
    if partition not in ('tr', 'cv'):
        raise ValueError('TSE v1 supports Train/CV only')
    # Requires and preserves the existing split. No new random partition.
    return build_strong_mix_paths(hp)[0][partition]


def create_tse_dataset(hp, partition, path_list=None, is_training=None):
    if partition not in ('tr', 'cv') or (is_training is not None and bool(is_training) != (partition == 'tr')):
        raise ValueError('TSE sampling mode is determined by partition')
    if '{node}' in hp.mix_cv_manifest_path:
        raise ValueError('Resolve {node} via spksep_run.sh or pass an explicit --mix-cv-manifest-path')
    resolve = lambda p: resolve_path(p, hp.data_path)
    pools, _ = read_fsd_dev_index(resolve(hp.mix_fsd_dev_audio),
        resolve(hp.mix_fsd_ground_truth), resolve(hp.class_map_path))
    # The shared SED index keeps only mapped labels. Preserve original annotation
    # multiplicity for TSE ambiguity reports without changing that index's API.
    with (Path(resolve(hp.mix_fsd_ground_truth)) / 'dev.csv').open(newline='', encoding='utf-8-sig') as handle:
        original_mids = {row['fname'].strip(): sorted(set(m.strip() for m in row['mids'].split(',')))
                         for row in csv.DictReader(handle)}
    for pool in pools.values():
        for entry in pool:
            entry['original_mids'] = original_mids[entry['fsd_id']]
    saved = json.loads(Path(resolve(hp.mix_fsd_split_path)).read_text())
    if saved.get('strategy') != 'official_dev_train_val' or any(
            saved[p] != [e['fsd_id'] for e in pools[p]] for p in ('train', 'val')):
        raise ValueError('FSD official Train/CV inventory differs from saved split')
    pools['train'], exclusions = exclude_declared_train_entries(pools['train'],
        hp.mix_content_duplicate_exclusions['fsd50k'], id_key='fsd_id', source='fsd50k')
    dataset = TSEMixDataset(tse_paths(hp, partition) if path_list is None else path_list,
        pools['train' if partition == 'tr' else 'val'], resolve(hp.class_map_path),
        partition=partition, cache_dir=resolve(hp.mix_cache_dir),
        cv_manifest_path=resolve(hp.mix_cv_manifest_path), seed=hp.mix_seed,
        snr_db=hp.mix_snr_db, snr_db_range=hp.mix_snr_db_range,
        active_threshold_dbfs=hp.mix_active_threshold_dbfs, active_frame_ms=hp.mix_active_frame_ms,
        no_pair_policy=hp.tse_no_pair_policy, zero_background_policy=hp.mix_zero_background_policy,
        zero_fsd_policy=hp.mix_zero_fsd_policy, num_classes=hp.num_classes)
    dataset.report['declared_train_fsd_duplicate_exclusions'] = exclusions
    return dataset
