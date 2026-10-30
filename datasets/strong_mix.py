"""Build TSE backgrounds from the existing AudioSet Train/CV split.

Strong annotations only filter backgrounds and identify conflicting MID labels.
This module does not generate SED frame targets or expand training conditions.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np

from datasets.mix_utils import digest
from datasets.fsd50k_cache import file_sha256
from datasets.paths.audioset_paths import (load_class_map, read_strong_labels,
    read_weak_labels, resolve_path, resolve_paths)
from datasets.paths.audioset_weak_paths import build_audioset_weak_index
from datasets.train_split import split_train, video_id

VERSION = 'joint-placement-v4-full-gt'

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
