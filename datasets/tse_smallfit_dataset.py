"""Build and replay a balanced, immutable subset of the hierarchy-v2 task.

The official source splits and 188-class vocabulary remain authoritative.
Only the eight requested classes enter this diagnostic train/validation set.
Waveforms are held in RAM; reusable audio caches remain on the compute node.
"""
from collections import Counter
import json
from pathlib import Path

import numpy as np
from loguru import logger
from torch.utils.data import Dataset

from datasets.fsd50k_cache import file_sha256
from datasets.mix_fsd_audit import audit_fsd_pool
from datasets.mix_utils import digest, sample_snr_db
from datasets.paths.audioset_paths import load_class_map, resolve_path
from datasets.strong_mix import _persist
from datasets.tse_dual_mix import PlacementError
from datasets.tse_hierarchy_dataset import (TSEHierarchyDataset, load_pools, dual_paths,
    project_path, _audit_train_pool, _condition_vocabulary)
from datasets.tse_hierarchy_labels import AncestorReducer, positive_pool
from tools.audit_common import leakage_report


class FixedMixtures(Dataset):
    """Preload verified manifest rows; repeated access never draws new audio."""

    def __init__(self, base, manifest, partition):
        self.partition = partition
        self.manifest = manifest
        backgrounds = {entry['id']: entry for entry in base.path_list}
        base.path_list = [backgrounds[row['background_id']] for row in manifest['rows'][::2]]
        base.manifest, base.partition = manifest, 'cv'
        self.samples = []
        for index in range(len(manifest['rows'])):
            (inputs, targets), _ = base.get_with_metadata(index)
            targets['utt_id'] = targets['utt_id'].replace('cv|', partition + '|', 1)
            self.samples.append((inputs, targets))
        self.path_list = base.path_list
        used = {event['fsd_id'] for row in manifest['rows'] for event in row['events']}
        self.fsd_pool = [entry for entry in base.fsd_pool if entry['fsd_id'] in used]
        self.report = dict(partition=partition, mixtures=len(self), frozen=True,
            backgrounds=len(self.path_list), source_count=len(self.fsd_pool),
            overlap_count=sum(row['overlap'] for row in manifest['rows']),
            condition_counts={name: dict(Counter(
                row['events'][i]['target_mid'] for row in manifest['rows']))
                for i, name in enumerate(('a', 'b'))},
            wrong_counts=dict(Counter(row['wrong_mid'] for row in manifest['rows'])),
            rows_sha256=manifest['rows_sha256'], source_audit=base.report,
            reference=manifest.get('reference'))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def draw_pair(base, background, mids, seed, overlap):
    """Use the existing ontology, placement, SNR and normalization checks."""
    rng = np.random.default_rng(seed)
    candidates, banned = base.allowed[background['id']]
    first = [i for i in candidates if base.groups[i]['conditions'] == [mids[0]]]
    for _ in range(base.max_pair_attempts):
        if not first:
            break
        a = int(rng.choice(first))
        partners = [i for i in base._partners(a, candidates, banned)
                    if base.groups[i]['conditions'] == [mids[1]]]
        if not partners:
            continue
        b = int(rng.choice(partners))
        sources = [base.groups[i]['entries'][int(rng.integers(len(base.groups[i]['entries'])))]
                   for i in (a, b)]
        wrong = str(rng.choice(sorted(base.wrong_pool - banned - base.group_bans[a] - base.group_bans[b])))
        mix_seed = int(rng.integers(0, 2**63 - 1))
        snr = (float(base.snr_db) if base.snr_db is not None else
               sample_snr_db(base.snr_range, sample_id=background['id'], seed=mix_seed))
        try:
            return base._mix(background, sources, mids, wrong, mix_seed, overlap, snr)[-1]
        except PlacementError:
            continue
    raise PlacementError(f'No legal audio placement for {background["id"]}, {mids}')


def make_rows(base, mids, count, seed):
    """Each background contributes one nonoverlap and one overlap example.

    Within blocks of eight backgrounds every class appears once as A and B.
    Pair permutations obey the full-label ontology (e.g. Bark and Meow can
    conflict through an explicitly annotated Animal ancestor).
    """
    rng = np.random.default_rng(seed)
    available = list(rng.permutation(len(base.path_list)))
    legal = set()
    for first, group in enumerate(base.groups):
        for second in base._partners(first, list(range(len(base.groups))), set()):
            legal.add((group['conditions'][0], base.groups[second]['conditions'][0]))
    rows = []
    for slot in range(count // 2):
        if slot % len(mids) == 0:
            for _ in range(10000):
                partners = list(rng.permutation(mids))
                if all((a, b) in legal for a, b in zip(mids, partners)):
                    break
            else:
                raise ValueError('Selected classes cannot form balanced legal A/B pairs')
        pair = [mids[slot % len(mids)], str(partners[slot % len(mids)])]
        for index in list(available):
            background = base.path_list[index]
            try:
                drawn = [draw_pair(base, background, pair,
                    int(digest([seed, slot, background['id'], overlap]), 16) % (2**63 - 1), overlap)
                    for overlap in (False, True)]
            except PlacementError:
                continue
            rows.extend(drawn)
            available.remove(index)
            break
        else:
            raise ValueError(f'Cannot fill fixed pair {pair}; increase background candidates or change MIDs')
        if (slot + 1) % 16 == 0:
            logger.info('SMALLFIT_DATA partition={} backgrounds={}/{}', base.partition, slot + 1, count // 2)
    return rows


def read_reference_manifest(path, settings):
    """Reuse only sample rows; retain current settings and verify every waveform on replay."""
    saved = json.loads(path.read_text())
    previous = saved.get('settings', {})
    if (saved.get('schema') != 1 or digest(saved['rows']) != saved['rows_sha256']
            or len(saved['rows']) != settings['mixtures']):
        raise ValueError(f'Invalid reference smallfit manifest: {path}')
    if {k: v for k, v in previous.items() if k != 'code'} != {
            k: v for k, v in settings.items() if k != 'code'}:
        raise ValueError(f'Reference smallfit data/settings differ: {path}')
    # This module adds reference loading. The source readers, label rules and
    # mixing implementations must still match the original run exactly.
    mixing_code = lambda code: {k: v for k, v in code.items() if k != 'tse_smallfit_dataset.py'}
    if mixing_code(previous.get('code', {})) != mixing_code(settings['code']):
        raise ValueError(f'Reference smallfit mixing code changed: {path}')
    return dict(schema=1, settings=settings, rows=saved['rows'], rows_sha256=saved['rows_sha256'],
        reference=dict(manifest=str(path.resolve()), manifest_sha256=file_sha256(path),
                       rows_sha256=saved['rows_sha256']))


def build_fixed_datasets(hp, run):
    root = Path(run) / 'fixed_data'
    root.mkdir(parents=True, exist_ok=True)
    reference = getattr(hp, 'tse_smallfit_reference_run', None)
    if reference:
        reference = Path(reference).resolve(strict=True)
        if reference == Path(run).resolve():
            raise ValueError('Reference samples must come from a different run')
        if json.loads((reference/'.tse_run_created').read_text()).get('project') != 'TSE':
            raise ValueError('Reference must be a standalone TSE run')
        for partition in ('tr', 'cv'):
            if not (reference/'fixed_data'/f'{partition}.json').is_file():
                raise FileNotFoundError(f'Missing reference smallfit {partition} manifest: {reference}')
    pools = load_pools(hp)
    train_audit = _audit_train_pool(json.dumps(pools['train'], sort_keys=True))
    vocabulary = _condition_vocabulary(hp, pools['train'], train_audit)
    mids = list(hp.tse_smallfit_mids)
    if not set(mids) <= set(vocabulary):
        raise ValueError('Smallfit MIDs must belong to the original 188-class training vocabulary')
    reducer = AncestorReducer(json.loads(project_path(hp.tse_ontology_path).read_text()))
    class_map = resolve_path(hp.class_map_path, hp.data_path)
    indices, names = load_class_map(class_map)
    datasets = {}
    for partition, pool_name, size, minimum in (
            ('tr', 'train', hp.tse_smallfit_train_mixtures, hp.tse_smallfit_min_train_sources),
            ('cv', 'val', hp.tse_smallfit_val_mixtures, hp.tse_smallfit_min_val_sources)):
        positive, _ = positive_pool(pools[pool_name], reducer, indices,
            {entry['fsd_id'] for entry in train_audit['excluded']} if partition == 'tr' else set())
        selected = [entry for entry in positive if entry['positive_mid'] in mids]
        source_audit = audit_fsd_pool(selected, split=pool_name)
        bad = {entry['fsd_id'] for entry in source_audit['excluded']}
        counts = Counter(entry['positive_mid'] for entry in selected if entry['fsd_id'] not in bad)
        if any(counts[mid] < minimum for mid in mids):
            raise ValueError(f'Insufficient valid {partition} sources for selected MIDs: {dict(counts)}')
        candidates = sorted(dual_paths(hp, partition), key=lambda entry: entry['id'])
        rng = np.random.default_rng(hp.mix_seed + (partition == 'cv'))
        paths = [candidates[i] for i in rng.permutation(len(candidates))[:hp.tse_smallfit_background_candidates]]
        # Full source annotations remain authoritative, even after label reduction.
        base = TSEHierarchyDataset(paths, selected, class_map, partition=partition,
            ontology_path=project_path(hp.tse_ontology_path), train_condition_mids=mids,
            cache_dir=hp.mix_cache_dir, cv_manifest_path=root/f'{partition}.json',
            seed=hp.mix_seed, snr_db=hp.mix_snr_db, snr_db_range=hp.mix_snr_db_range,
            threshold_dbfs=hp.mix_active_threshold_dbfs, frame_ms=hp.mix_active_frame_ms,
            max_pair_attempts=hp.tse_dual_max_pair_attempts, max_crop_attempts=hp.tse_dual_max_crop_attempts,
            num_classes=hp.num_classes, zero_fsd_policy=hp.mix_zero_fsd_policy,
            zero_background_policy=hp.mix_zero_background_policy, no_pair_policy=hp.tse_no_pair_policy,
            fsd_audit=source_audit, prepare_manifest=False)
        settings = dict(partition=partition, mids=mids, mixtures=size, seed=hp.mix_seed,
            candidates=digest(paths), sources=digest(selected), snr_db=hp.mix_snr_db,
            snr_range=hp.mix_snr_db_range, threshold=hp.mix_active_threshold_dbfs,
            frame_ms=hp.mix_active_frame_ms, max_pair_attempts=hp.tse_dual_max_pair_attempts,
            max_crop_attempts=hp.tse_dual_max_crop_attempts,
            no_pair_policy=hp.tse_no_pair_policy, zero_fsd_policy=hp.mix_zero_fsd_policy,
            zero_background_policy=hp.mix_zero_background_policy,
            class_map_sha256=file_sha256(class_map), ontology_sha256=base.ontology.sha256,
            code={name: file_sha256(Path(__file__).parent/name) for name in
                  ('tse_smallfit_dataset.py', 'tse_hierarchy_dataset.py', 'tse_dual_mix.py',
                   'tse_ontology.py', 'tse_hierarchy_labels.py', 'mix_utils.py', 'active_rms.py',
                   'fsd50k_cache.py', 'tse_dataset.py')})
        path = root/f'{partition}.json'
        replay = read_reference_manifest(reference/'fixed_data'/f'{partition}.json', settings) if reference else None
        if path.exists():
            manifest = json.loads(path.read_text())
            if (manifest.get('settings') != settings or
                    digest(manifest['rows']) != manifest['rows_sha256'] or len(manifest['rows']) != size):
                raise ValueError(f'Fixed data changed: {path}; create a new run')
            if replay is not None and (manifest.get('reference') != replay['reference']
                                      or manifest['rows_sha256'] != replay['rows_sha256']):
                raise ValueError(f'Fixed data reference changed: {path}; create a new run')
        elif replay is not None:
            manifest = replay
            _persist(path, manifest)
        else:
            rows = make_rows(base, mids, size, hp.mix_seed + (partition == 'cv'))
            manifest = dict(schema=1, settings=settings, rows=rows, rows_sha256=digest(rows))
            _persist(path, manifest)
        rows = manifest['rows']
        if (manifest.get('schema') != 1 or
                len({row['background_id'] for row in rows}) != size // 2 or
                any(row['overlap'] != bool(i % 2) for i, row in enumerate(rows)) or
                any(rows[i]['background_id'] != rows[i + 1]['background_id'] for i in range(0, size, 2)) or
                any(row['wrong_mid'] not in mids for row in rows)):
            raise ValueError('Fixed manifest lost its paired-background/overlap structure')
        for condition in (0, 1):
            counts = Counter(row['events'][condition]['target_mid'] for row in rows)
            if counts != Counter({mid: size // len(mids) for mid in mids}):
                raise ValueError('Fixed manifest no longer has balanced target classes')
        datasets[partition] = FixedMixtures(base, manifest, partition)
    report = leakage_report(datasets['tr'], datasets['cv'], full=True)
    if report['status'] != 'passed':
        raise ValueError(f'Smallfit source leakage: {report}')
    _persist(root/'source_leakage.json', report)
    _persist(root/'summary.json', dict(classes=[dict(mid=mid, name=names[indices[mid]]) for mid in mids],
        **{partition: dataset.report for partition, dataset in datasets.items()}))
    return datasets
