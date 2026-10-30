"""Independent Strong+FSD supervision around the unchanged waveform mixer.

FSD placement is a SOURCE-level pseudo label, not human semantic timing.
The declared full GT supervises every valid frame; only padding is masked.
Strong+FSD，Strong 实验的 FSD 抽样、condition 展开，audioset强标签与FSD放置区间的联合GT；复用底层混音
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from datasets.audioset_fsd50k_mix_dataset import AudioSetFSD50KMixDataset, digest
from datasets.fsd50k_cache import file_sha256
from datasets.paths.audioset_paths import (load_class_map, read_strong_labels,
    read_weak_labels, resolve_path, resolve_paths)
from datasets.paths.audioset_weak_paths import build_audioset_weak_index
from datasets.train_split import split_train, video_id

MODE = 'audioset_strong_fsd50k_mix'
VERSION = 'joint-placement-v4-full-gt'
GT_POLICY = 'full_joint_positive_union'

#输入路径和值，安全的保存JSON数据（原子写入+文件锁）
def _persist(path, value):
    """Reuse the mix manifest's lock/atomic write convention."""
    import fcntl
    import os
    import tempfile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            if json.loads(path.read_text()) != value:
                raise ValueError(f'Strong mix inventory/settings changed; use a new path: {path}')
        else:
            with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as handle:
                json.dump(value, handle, indent=2, allow_nan=False)
                temporary = handle.name
            os.replace(temporary, path)

#输入原始强标签事件字典，清洗和校验Audioset强标签，raw_events是字典，键是音频id，值是事件列表
def _validate_strong_intervals(raw_events):
    """Exclude zero-duration point annotations; reject malformed intervals."""
    events = {}
    zero_duration = []
    retained = 0
    for clip, rows in raw_events.items():
        valid_rows = []
        for event in rows:
            onset, offset = event['onset'], event['offset']
            if (not np.isfinite([onset, offset]).all()
                    or not 0 <= onset <= offset <= 10):#是否在0-10s范围内
                raise ValueError(f'Invalid clip-relative Strong interval: {clip}: {event}')
            if onset == offset:#剔除0时长事件
                zero_duration.append((clip, event))
                continue
            valid_rows.append(event)
            retained += 1
        if valid_rows:
            events[clip] = valid_rows
    report = dict(
        raw_event_count=sum(len(rows) for rows in raw_events.values()),
        retained_interval_count=retained,
        retained_clip_count=len(events),
        zero_duration_event_count=len(zero_duration),
        zero_duration_clip_count=len({clip for clip, _ in zero_duration}),
        zero_duration_present_count=sum(event['present'] for _, event in zero_duration),
        zero_duration_not_present_count=sum(not event['present'] for _, event in zero_duration),
        policy='exclude onset == offset because [onset, offset) covers no frame',
    )
    return events, report

"""
输入训练列表，待剔除ID列表，ID键，来源名
功能：从训练集中剔除显式声明的条目（防数据泄漏/重复）。
核心逻辑：
把 train 列表转成以 id 为键的字典，检查是否有重复 ID。
检查要剔除的 identifiers 是否都在训练集中。
过滤掉这些 ID，并返回过滤后的列表和一份“剔除报告”（记录了剔除了哪些、原因是什么）。
"""
def exclude_declared_train_entries(train, identifiers, *, id_key, source):
    """Remove explicitly declared entries from this experiment's Train pool."""
    if not isinstance(identifiers, list):
        raise ValueError(f'{source} Train exclusions must be a list')
    train_by_id = {str(entry[id_key]): entry for entry in train}
    if len(train_by_id) != len(train):
        raise ValueError(f'{source} duplicate Train IDs prevent exclusion')
    excluded_ids = {str(identity) for identity in identifiers}
    if len(excluded_ids) != len(identifiers) or not excluded_ids <= train_by_id.keys():
        raise ValueError(f'Unknown or repeated {source} Train exclusions: {identifiers!r}')
    excluded = [dict(train_id=identity,
        train_path=train_by_id[identity]['audio_path'], reason='cross_partition_content_duplicate')
        for identity in sorted(excluded_ids)]
    effective = [entry for entry in train if str(entry[id_key]) not in excluded_ids]
    return effective, dict(source=source, policy='exclude_declared_train_ids',
        source_train_count=len(train), effective_train_count=len(effective),
        excluded_train_count=len(excluded), excluded=excluded)

"""
输入超参hp
功能：构建混合数据集的核心划分与配置
1、加载数据：加载Audioset的弱标签索引，类别映射，强标签
2、求交集：找出同时拥有弱标签和强标签的音频片段（只是为了统计，不用于选数据）
3、复用父划分：读取已有的平衡划分（Train/cv）
4、过滤与展开：
计算625个帧中心，遍历每个划分中的背景音，为每个背景音提取出真实存在的正类MID，并构建positive和negative掩码矩阵
保留至少一个正类的背景音
5、数据泄露检查：确保Train和CV中没有相同的video_id或音频路径
6.持久化：保存所有设置，哈希值和划分结果
7。报告
"""
def build_strong_mix_paths(hp):
    resolve = lambda p: resolve_path(p, hp.data_path)
    index = build_audioset_weak_index(hp.mix_audio_dirs, hp.mix_weak_csvs,
        hp.class_map_path, data_root=hp.data_path, strict=hp.mix_strict_paths)
    mapping, _ = load_class_map(resolve(hp.class_map_path))
    raw_events = read_strong_labels(resolve(hp.mix_strong_tsv))
    events, interval_report = _validate_strong_intervals(raw_events)
    all_weak = {e['id'] for p in resolve_paths(hp.mix_weak_csvs, hp.data_path)
                for e in read_weak_labels(p)}
    available = {e['id'] for e in index['entries']}
    intersection = all_weak & events.keys()
    scope = sorted({e['mid'] for rows in events.values() for e in rows
                    if e['present'] and e['mid'] in mapping})
    # Preserve the existing balanced split, then filter WITHIN each partition.
    # Repartitioning the intersection would break baseline train/CV isolation.
    parent = Path(resolve(hp.mix_parent_background_split_path))
    if not parent.is_file():
        raise FileNotFoundError(f'Existing balanced split required: {parent}')
    #①用”可用弱标签条目“做平衡划分
    original = split_train(index['entries'], hp.data_path, str(parent),
                          hp.train_val_fraction, hp.train_split_seed)
    centers = np.arange(625, dtype=np.float32) * 256 / 16000
    splits = {}
    for partition, entries in original.items():
        selected = []
        for entry in entries:
            rows = events.get(entry['id'], [])#②选取这个clip的强标签
            valid = []
            for mid in sorted({e['mid'] for e in rows if e['present'] and e['mid'] in mapping}):
                positive = np.zeros(len(centers), dtype=bool)
                negative = np.zeros(len(centers), dtype=bool)
                for event in rows:
                    if event['mid'] == mid:
                        active = (centers >= event['onset']) & (centers < event['offset'])
                        if event['present']:
                            positive |= active
                        else:
                            negative |= active
                if positive.any():
                    valid.append(mid)
            if valid: #③有至少一个正类mid才保留
                selected.append({**entry, 'events': rows, 'strong_positive_mids': valid,
                                 'strong_scope': scope})
        splits[partition] = selected
        if not selected:
            raise ValueError(f'No usable balanced/Strong backgrounds in {partition}')
    declarations = hp.mix_content_duplicate_exclusions
    if not isinstance(declarations, dict) or set(declarations) != {'audioset', 'fsd50k'}:
        raise ValueError('mix_content_duplicate_exclusions requires audioset and fsd50k lists')
    splits['tr'], duplicate_report = exclude_declared_train_entries(
        splits['tr'], declarations['audioset'], id_key='id', source='audioset')
    if not splits['tr']:
        raise ValueError('No Train backgrounds remain after content duplicate exclusion')
    if {video_id(e) for e in splits['tr']} & {video_id(e) for e in splits['cv']}:
        raise ValueError('AudioSet video leakage')
    if {e['audio_path'] for e in splits['tr']} & {e['audio_path'] for e in splits['cv']}:
        raise ValueError('AudioSet audio path leakage')
    settings = dict(version=VERSION, parent_split_sha256=file_sha256(parent),
        strong_sha256=file_sha256(resolve(hp.mix_strong_tsv)),
        class_map_sha256=file_sha256(resolve(hp.class_map_path)),
        zero_duration_interval_policy='exclude',
        content_duplicate_exclusions=duplicate_report['excluded'],
        entries_sha256=digest(splits), **{p: [e['id'] for e in entries] for p, entries in splits.items()})
    _persist(resolve(hp.mix_background_split_path), settings)
    def coverage(ids):
        counts = Counter(e['mid'] for clip in ids for e in events[clip] if e['present'])
        return dict(mapped_class_count=len(counts.keys() & mapping.keys()),
                    mapped_event_counts={m: n for m, n in sorted(counts.items()) if m in mapping},
                    unmapped_event_counts={m: n for m, n in sorted(counts.items()) if m not in mapping})
    report = dict(version=VERSION, balanced_metadata_clips=len(all_weak),
        balanced_available_clips=len(available), strong_tsv_clips=len(raw_events),
        strong_usable_interval_clips=len(events), strong_interval_filter=interval_report,
        content_duplicate_exclusions=duplicate_report,
        metadata_intersection_clips=len(intersection),
        available_intersection_clips=len(available & events.keys()),
        missing_intersection_audio=len(intersection - available),
        intersection_coverage=coverage(intersection),
        available_coverage=coverage(available & events.keys()),
        partitions={p: dict(backgrounds=len(entries), videos=len({video_id(e) for e in entries}),
                    **coverage({e['id'] for e in entries})) for p, entries in splits.items()},
        balanced_index_report=index['report'], split=settings)
    return splits, report

"""
初始化标签策略，要求 complete 必须为 True，且 verified_absence 不能有值
"""
class LabelPolicy:
    """Full, exact-MID positive union. Other valid frames are negative."""
    def __init__(self, mapping, *, complete=True, verified_absence=None):
        if not complete or verified_absence:
            raise ValueError('Full GT requires complete labels and no absence overrides')
        self.mapping = mapping
    """
    输入背景、FSD、目标 mid、混音 meta、帧中心、条件 mid
    生成单个MID对应的强标签
    1.初始化全0的labels数组，valid掩码限定在0-10s
    2.背景音监督：遍历背景音时间，如果事件MID等于当前要生成的mid且present=True，把对应的时间段置为 1。
    3.FSD放置监督：如果当前 mid 是 FSD 目标，把 FSD 放入的时间段（target_start_sample 到 target_end_sample）也置为 1
    4.正类并集：如果有重叠，正类优先
    """
    def labels(self, background, fsd, target_mid, meta, centers, mid):
        if mid not in self.mapping:
            raise ValueError(f'Unmapped Condition MID: {mid}')
        valid = np.isfinite(centers) & (centers >= 0) & (centers < 10.0)
        labels = np.zeros(len(centers), dtype=bool)
        for event in background['events']:
            if event['mid'] == mid and event['present']:
                labels |= (centers >= event['onset']) & (centers < event['offset'])
        if mid == target_mid:
            start = meta['target_start_sample'] / 16000
            end = meta['target_end_sample'] / 16000
            labels |= (centers >= start) & (centers < end)
        # Positive union wins over any overlapping event, including NOT_PRESENT.
        return (labels & valid).astype(np.float32), valid.astype(np.float32)


class StrongMixtureCore(AudioSetFSD50KMixDataset):
    """No changes to selection, audio preprocessing, SNR, placement or features."""
    def __init__(self, path_list, *args, supervision_settings, **kwargs):
        self.supervision_settings = supervision_settings
        # The legacy mixer uses weak_mids only to forbid same-MID targets.
        # Clear that exclusion input in private copies; preserve the raw metadata.
        # No change to the shared mixer, its fingerprints, or other Dataset modes.
        paths = [{**entry,
                  'original_weak_mids': list(entry.get('original_weak_mids', entry['weak_mids'])),
                  'weak_mids': []} for entry in path_list]
        super().__init__(paths, *args, **kwargs)
    #随机选一个FSD音频，在从中选一个MID作为目标
    def _select(self, background, rng):
        fsd = self.fsd_pool[int(rng.integers(len(self.fsd_pool)))]
        mids = sorted(set(fsd['mids']))
        return fsd, mids[int(rng.integers(len(mids)))]
    #在父类 manifest 基础上，追加 joint_supervision 设置和当前代码文件的 SHA256 哈希值，确保实验可复现
    def _load_or_create_manifest(self, path, settings):
        return super()._load_or_create_manifest(path, {
            **settings, 'joint_supervision': self.supervision_settings,
            'joint_code_sha256': file_sha256(__file__)})

#包装 Core，供 PyTorch DataLoader 使用，核心是条件展开
class StrongFSD50KMixDataset(Dataset):
    """
    构建self.conditions:对于每个背景音，必定生成一个 ('fsd', target_mid) 条件。
 然后遍历该背景音里的所有 strong_positive_mids，为每一个 MID 生成一个 ('audioset', mid) 条件。
如果是 CV 集，还要记录真实的 target_mid。
    """
    def __init__(self, core, policy):
        self.dataset, self.policy = core, policy
        self.conditions = []
        for i, background in enumerate(core.path_list):
            target_mid = core.manifest['rows'][i]['target_mid'] if core.partition == 'cv' else None
            self.conditions.append((i, target_mid, 'fsd'))
            self.conditions.extend((i, mid, 'audioset')
                for mid in background['strong_positive_mids'])
        self.cv_conditions = self.conditions if core.partition == 'cv' else []

    def __getattr__(self, name):
        dataset = self.__dict__.get('dataset')
        if dataset is None:
            raise AttributeError(name)
        return getattr(dataset, name)

    def __len__(self):
        return len(self.conditions)
    """
    1.生成one-hot的条件向量condition
    2.调用self。policy.label是生成强标签label和掩码mask
    3.组装并返回字典，包含：原始输入，条件向量，强标签，掩码，类别索引，来源等
    """
    def condition_sample(self, sample, meta, background_index, mid, source):
        inputs, target = sample
        background = self.path_list[background_index]
        fsd = self.fsd_by_id[meta['fsd_id']]
        labels, mask = self.policy.labels(background, fsd, meta['target_mid'], meta,
                                         target['frame_times'], mid)
        category = self.mid_to_index[mid]
        condition = np.zeros(self.num_classes, dtype=np.float32)
        condition[category] = 1
        return ({**inputs, 'weak_labels': condition}, {**target,
            'strong_labels': labels, 'strong_mask': mask, 'class_index': category,
            'condition_source': source, 'condition_mid': mid,
            'mixture_id': target['utt_id'],
            'utt_id': f'{target["utt_id"]}|{source}|{category}',
            'shared_mid': int(mid == meta['target_mid'] and mid in background['strong_positive_mids'])})
    """
    取条件→调 core 的 get_with_metadata 取一次混音→
    若 source=='fsd' 用 meta 里真实 target_mid→condition_sample→
    校验至少一个有效正帧
    """
    def get_with_metadata(self, index):
        background_index, mid, source = self.conditions[index]
        sample, meta = self.dataset.get_with_metadata(background_index)
        if source == 'fsd':
            mid = meta['target_mid']
        result = self.condition_sample(sample, meta, background_index, mid, source)
        if not (result[1]['strong_labels'] * result[1]['strong_mask']).any():
            raise ValueError('Condition has no valid positive frames')
        return result, {**meta, 'background_index': background_index,
                       'condition_mid': mid, 'condition_source': source}

    def __getitem__(self, index):
        return self.get_with_metadata(index)[0]
    #仅评估用：一次混音共享给所有条件（fsd + 所有 audioset 正类 + 随机 wrong 类）
    def condition_bundle(self, background_index, wrong_count=0):
        """One mix draw shared by every condition; evaluation only."""
        sample, meta = self.dataset.get_with_metadata(background_index)
        background = self.path_list[background_index]
        fsd = self.fsd_by_id[meta['fsd_id']]
        pairs = [('fsd', meta['target_mid'])]
        pairs += [('audioset', m) for m in background['strong_positive_mids']]
        excluded = set(fsd['mids']) | set(background['weak_mids'])
        excluded |= {e['mid'] for e in background['events'] if e['present']}
        # Limit wrongs to classes listed in the official mapped FSD training pool,
        # which is supplied independently of the CV pool by the factory.
        candidates = sorted(set(self.supervision_settings['wrong_pool']) - excluded)
        rng = np.random.default_rng(np.random.SeedSequence([meta['seed'], 0x57524F]))
        wrongs = rng.choice(candidates, min(wrong_count, len(candidates)), replace=False)
        pairs += [('wrong_candidate', str(m)) for m in wrongs]
        samples = [self.condition_sample(sample, meta, background_index, m, s) for s, m in pairs]
        return samples, {**meta, 'wrong_requested': wrong_count, 'wrong_available': len(wrongs)}


def supervision(hp, pools):
    resolve = lambda p: resolve_path(p, hp.data_path)
    absent_path = resolve(hp.mix_fsd_verified_absence_path)
    absent = json.loads(Path(absent_path).read_text()) if absent_path else None
    mapping, _ = load_class_map(resolve(hp.class_map_path))
    if absent is not None:
        all_ids = {e['fsd_id'] for p in pools.values() for e in p}
        if not isinstance(absent, dict) or not absent.keys() <= all_ids:
            raise ValueError('Verified absence must map official FSD IDs to MID lists')
        for mids in absent.values():
            if not isinstance(mids, list) or not set(mids) <= mapping.keys():
                raise ValueError('Invalid verified-absence MID list')
    settings = dict(version=VERSION, fsd_gt='placement',
        annotation_complete=hp.mix_strong_annotation_complete,
        gt_policy=hp.mix_gt_policy, fsd_selection='uniform_audio_then_uniform_mid',
        train_conditions=hp.mix_train_condition_policy,
        label_relation_policy='exact_mid_only',
        verified_absence_sha256=file_sha256(absent_path) if absent_path else None,
        strong_tsv_sha256=file_sha256(resolve(hp.mix_strong_tsv)),
        split_sha256=file_sha256(resolve(hp.mix_background_split_path)),
        wrong_pool=sorted({m for e in pools['train'] for m in e['mids']}))
    return LabelPolicy(mapping, complete=hp.mix_strong_annotation_complete,
                       verified_absence=absent), settings
