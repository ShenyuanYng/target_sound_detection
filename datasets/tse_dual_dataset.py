"""AudioSet Strong + dual FSD + one absent condition, with immutable CV.

Each background has one nonoverlap slot and one partial-overlap slot. Both
slots resample sources/crops/positions in Train; CV freezes the complete draw.
All derived files live under project paths, separate from source corpora.
"""
from argparse import Namespace
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np
import scipy
import soundfile as sf
from torch.utils.data import Dataset

from datasets.audioset_fsd50k_mix_dataset import digest
from datasets.fsd50k_cache import file_sha256, read_cached_fsd
from datasets.mix_background_audit import audit_backgrounds
from datasets.mix_fsd_audit import audit_fsd_pool
from datasets.mix_utils import read_mono_waveform, sample_snr_db, validate_snr_db_range
from datasets.paths.audioset_paths import load_class_map, resolve_path
from datasets.paths.fsd50k_paths import read_fsd_dev_index
from datasets.strong_mix import build_strong_mix_paths, exclude_declared_train_entries, _persist
from datasets.tse_dataset import forbidden_mids, waveform_hash
from datasets.tse_dual_mix import make_tse_dual_mixture, PlacementError
from datasets.tse_ontology import OntologyPolicy, source_mids

MODE = 'audioset_strong_dual_fsd_wrong_tse'
PROJECT = Path(__file__).resolve().parents[1]


def project_path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT / path).resolve()


def validate_artifact_paths(hp, extra_paths=()):
    roots = [Path(hp.data_path).resolve()] + [Path(resolve_path(p, hp.data_path)).resolve() for p in (
        hp.mix_fsd_dev_audio, hp.mix_fsd_ground_truth, *hp.mix_audio_dirs)]
    outputs = [project_path(getattr(hp, key)) for key in (
        'mix_cache_dir', 'mix_cv_manifest_path', 'mix_background_split_path', 'save_path')]
    for path in [*outputs, *map(Path, extra_paths)]:
        if any(path == root or root in path.parents for root in roots):
            raise ValueError(f'Derived files must be outside source corpus directories: {path}')


def dual_paths(hp, partition):
    if partition not in ('tr', 'cv'):
        raise ValueError('Experiment 3 supports Train/CV only')
    validate_artifact_paths(hp)
    config = Namespace(**(dict(hp) if isinstance(hp, dict) else vars(hp)))
    config.mix_background_split_path = str(project_path(hp.mix_background_split_path))
    return build_strong_mix_paths(config)[0][partition]


def load_pools(hp):
    resolve = lambda p: resolve_path(p, hp.data_path)
    pools, _ = read_fsd_dev_index(resolve(hp.mix_fsd_dev_audio),
        resolve(hp.mix_fsd_ground_truth), resolve(hp.class_map_path))
    with (Path(resolve(hp.mix_fsd_ground_truth)) / 'dev.csv').open(newline='', encoding='utf-8-sig') as handle:
        originals = {row['fname'].strip(): sorted(set(m.strip() for m in row['mids'].split(',')))
                     for row in csv.DictReader(handle)}
    for pool in pools.values():
        for entry in pool:
            entry['original_mids'] = originals[entry['fsd_id']]
    saved = json.loads(Path(resolve(hp.mix_fsd_split_path)).read_text())
    if saved.get('strategy') != 'official_dev_train_val' or any(
            saved[p] != [e['fsd_id'] for e in pools[p]] for p in ('train', 'val')):
        raise ValueError('FSD official Train/Val inventory differs from the saved split')
    pools['train'], _ = exclude_declared_train_entries(pools['train'],
        hp.mix_content_duplicate_exclusions['fsd50k'], id_key='fsd_id', source='fsd50k')
    return pools


class TSEDualWrongDataset(Dataset):
    def __init__(self, path_list, fsd_pool, class_map_path, *, partition, ontology_path,
                 train_condition_mids, cache_dir, cv_manifest_path, seed=7,
                 snr_db=None, snr_db_range=(-5., 5.), threshold_dbfs=-40., frame_ms=20.,
                 max_pair_attempts=128, max_crop_attempts=128, num_classes=527,
                 zero_fsd_policy='exclude', zero_background_policy='exclude',
                 no_pair_policy='error'):
        if partition not in ('tr', 'cv') or no_pair_policy not in ('error', 'exclude'):
            raise ValueError('Invalid partition or no-pair policy')
        if min(max_pair_attempts, max_crop_attempts) < 1:
            raise ValueError('Resampling limits must be positive')
        if snr_db is not None and not np.isfinite(snr_db):
            raise ValueError('Nonfinite SNR')
        self.partition, self.seed = partition, int(seed)
        self.cache_dir = str(cache_dir)
        self.ontology = OntologyPolicy(ontology_path)
        self.mid_to_index, names = load_class_map(class_map_path)
        if len(names) != num_classes:
            raise ValueError('Class map size mismatch')
        self.num_classes = num_classes
        self.threshold, self.frame_ms = float(threshold_dbfs), float(frame_ms)
        self.snr_db, self.snr_range = snr_db, validate_snr_db_range(snr_db_range)
        self.max_pair_attempts, self.max_crop_attempts = max_pair_attempts, max_crop_attempts
        paths = sorted(path_list, key=lambda e: e['id'])
        pool = sorted(fsd_pool, key=lambda e: int(e['fsd_id']))
        if not paths or not pool or len({e['id'] for e in paths}) != len(paths):
            raise ValueError('Need nonempty unique backgrounds and FSD pool')
        self.background_audit = audit_backgrounds(paths)
        self.fsd_audit = audit_fsd_pool(pool, split='train' if partition == 'tr' else 'val')
        if self.background_audit['invalid_count'] or self.fsd_audit['invalid_count']:
            raise ValueError('Invalid source audio: ' + json.dumps(
                dict(background=self.background_audit, fsd=self.fsd_audit)))
        bad_bg = {e['background_id'] for e in self.background_audit['issues']}
        bad_fsd = {e['fsd_id'] for e in self.fsd_audit['excluded']}
        if zero_fsd_policy not in ('error', 'exclude') or zero_background_policy not in ('error', 'exclude'):
            raise ValueError('Invalid zero-energy policy')
        if (bad_bg and zero_background_policy == 'error') or (bad_fsd and zero_fsd_policy == 'error'):
            raise ValueError('Zero-energy source found')
        self.fsd_pool = [e for e in pool if e['fsd_id'] not in bad_fsd
                         and self.ontology.known(source_mids(e))]
        self.fsd_by_id = {e['fsd_id']: e for e in self.fsd_pool}
        # Only labels with positive FSD training support can become negative queries.
        self.wrong_pool = (set(train_condition_mids) & self.mid_to_index.keys()
                           & self.ontology.nodes.keys()) - self.ontology.excluded_conditions
        if not self.wrong_pool:
            raise ValueError('No train-supported wrong-condition vocabulary')
        self.wrong_bits = {mid: 1 << i for i, mid in enumerate(sorted(self.wrong_pool))}
        self.all_wrong_bits = (1 << len(self.wrong_bits)) - 1
        groups = defaultdict(list)
        for entry in self.fsd_pool:
            if set(entry['mids']) - self.ontology.excluded_conditions:
                groups[tuple(sorted(source_mids(entry)))].append(entry)
        self.groups = [dict(mids=frozenset(labels), entries=entries,
                            conditions=sorted(set(labels) & self.mid_to_index.keys()
                                              - self.ontology.excluded_conditions))
                       for labels, entries in sorted(groups.items())]
        self.group_bans = [self.ontology.forbidden(g['mids']) for g in self.groups]
        self.group_wrong_bits = [self._wrong_mask(banned) for banned in self.group_bans]
        self.allowed, excluded = {}, []
        coverage = Counter()
        self.path_list = []
        for bg in paths:
            mids = forbidden_mids(bg)
            if bg['id'] in bad_bg or not self.ontology.known(mids):
                excluded.append(bg['id'])
                continue
            banned = self.ontology.forbidden(mids)
            candidates = [i for i, g in enumerate(self.groups) if not g['mids'] & banned]
            # Finite label-level feasibility check, independent of audio crop sampling.
            feasible = any(self._partners(i, candidates, banned) for i in candidates)
            if not feasible:
                excluded.append(bg['id'])
                if no_pair_policy == 'error':
                    raise ValueError(f'No legal dual-FSD/wrong-MID combination for {bg["id"]}')
                continue
            self.path_list.append(bg)
            self.allowed[bg['id']] = (candidates, banned)
            coverage.update({mid for i in candidates for mid in self.groups[i]['conditions']})
        if not self.path_list:
            raise ValueError('No usable backgrounds after ontology/pair filtering')
        self.report = dict(partition=partition, effective_background_count=len(self.path_list),
            samples_per_epoch=2*len(self.path_list), excluded_background_ids=excluded,
            effective_fsd_count=len(self.fsd_pool), unknown_ontology_fsd_ids=[e['fsd_id'] for e in pool
                if not self.ontology.known(source_mids(e))],
            eligible_condition_class_count=len(coverage),
            eligible_condition_background_counts=dict(coverage),
            multilabel_fsd_fraction=sum(len(source_mids(e)) > 1 for e in self.fsd_pool)/max(1, len(self.fsd_pool)),
            ontology_sha256=self.ontology.sha256, no_pair_policy=no_pair_policy,
            label_policy='full-label exact/ancestor/descendant exclusion; prefer single-label pairs',
            absence_caveat='Absence inferred from annotations and ontology; unlabelled sounds remain possible',
            target_semantics='entire processed FSD source stem; multi-label fallback includes co-occurrences',
            placement_policy='one nonoverlap and one 25%-50% shorter-interval overlap slot per background',
            wrong_condition_mids=sorted(self.wrong_pool))
        self.manifest = None
        if partition == 'cv':
            code_files = ('datasets/tse_dual_dataset.py', 'datasets/tse_dual_mix.py',
                'datasets/tse_ontology.py', 'datasets/mix_utils.py', 'datasets/active_rms.py',
                'datasets/fsd50k_cache.py', 'datasets/tse_dataset.py')
            settings = dict(task=MODE, version=1, seed=self.seed, paths=digest(paths), pool=digest(pool),
                effective_backgrounds=digest(self.path_list), effective_pool=digest(self.fsd_pool),
                train_condition_mids=sorted(self.wrong_pool), class_map_sha256=file_sha256(class_map_path),
                ontology_sha256=self.ontology.sha256, snr_db=snr_db, snr_range=list(self.snr_range),
                threshold=self.threshold, frame_ms=self.frame_ms, max_pair_attempts=max_pair_attempts,
                max_crop_attempts=max_crop_attempts, no_pair_policy=no_pair_policy,
                zero_fsd_policy=zero_fsd_policy, zero_background_policy=zero_background_policy,
                fsd_audit=self.fsd_audit, background_audit=self.background_audit,
                versions=dict(numpy=np.__version__, scipy=scipy.__version__, soundfile=sf.__version__),
                code={name: file_sha256(PROJECT/name) for name in code_files})
            path = Path(cv_manifest_path)
            if path.exists():
                saved = json.loads(path.read_text())
                if saved.get('schema') != 1 or saved.get('settings') != settings:
                    raise ValueError('Experiment 3 CV settings/inventory changed; choose a new manifest path')
            else:
                rows = [self._generate_row(i) for i in range(len(self))]
                saved = dict(schema=1, settings=settings, rows=rows, rows_sha256=digest(rows))
                _persist(path, saved)
            if (saved['rows_sha256'] != digest(saved['rows']) or len(saved['rows']) != len(self)
                    or [(r['background_id'], r['overlap']) for r in saved['rows']] != [
                        (self.path_list[i//2]['id'], bool(i % 2)) for i in range(len(self))]):
                raise ValueError('Experiment 3 CV rows changed')
            self.manifest = saved
            self.report['cv_single_label_stem_fraction'] = sum(
                len(e['fsd_mids']) == 1 for row in saved['rows'] for e in row['events']) / (2*len(self))

    def __len__(self):
        return 2 * len(self.path_list)

    def _wrong_mask(self, mids):
        return sum(self.wrong_bits.get(mid, 0) for mid in mids)

    def _partners(self, first, candidates, bg_banned):
        remaining = self.all_wrong_bits & ~self._wrong_mask(bg_banned) & ~self.group_wrong_bits[first]
        return [i for i in candidates if not self.groups[i]['mids'] & self.group_bans[first]
                and remaining & ~self.group_wrong_bits[i]]

    def _select(self, background, rng):
        candidates, bg_banned = self.allowed[background['id']]
        # Exhaust feasible single-label pairs before considering multi-label fallbacks.
        single = [i for i in candidates if len(self.groups[i]['mids']) == 1]
        for eligible in (single, candidates):
            order = [int(i) for i in rng.permutation(eligible)]
            order.sort(key=lambda i: len(self.groups[i]['mids']) != 1)
            for first in order:
                partners = self._partners(first, eligible, bg_banned)
                if not partners:
                    continue
                singles = [i for i in partners if len(self.groups[i]['mids']) == 1]
                second = int(rng.choice(singles or partners))
                indices = [first, second]
                # Randomize A/B identity, so the preference does not bias condition position.
                rng.shuffle(indices)
                sources, mids = [], []
                for i in indices:
                    group = self.groups[i]
                    sources.append(group['entries'][int(rng.integers(len(group['entries'])))])
                    mids.append(str(rng.choice(group['conditions'])))
                wrong = str(rng.choice(sorted(self.wrong_pool - bg_banned
                                             - self.group_bans[first] - self.group_bans[second])))
                return sources, mids, wrong
        raise ValueError(f'No legal label combination for {background["id"]}')

    def _mix(self, background, sources, mids, wrong, seed, overlap, snr):
        all_labels = [source_mids(e) for e in sources]
        bg_labels = forbidden_mids(background)
        if (mids[0] == mids[1] or sources[0]['fsd_id'] == sources[1]['fsd_id']
                or any(mid not in labels for mid, labels in zip(mids, all_labels))
                or self.ontology.conflicts(*all_labels)
                or any(self.ontology.conflicts(labels, bg_labels) for labels in all_labels)
                or wrong not in self.wrong_pool
                or self.ontology.conflicts({wrong}, set.union(set(bg_labels), *map(set, all_labels)))):
            raise ValueError('Illegal dual-FSD or wrong-condition labels')
        bg = read_mono_waveform(background['audio_path'])[0]
        loaded = [read_cached_fsd(e['audio_path'], self.cache_dir) for e in sources]
        if loaded[0][1]['source_sha256'] == loaded[1][1]['source_sha256']:
            raise PlacementError('Identical source files cannot be mixed as different targets')
        bg, targets, mix, meta = make_tse_dual_mixture(bg, [v[0] for v in loaded],
            seed=seed, overlap=overlap, snr_db=snr, threshold_dbfs=self.threshold,
            frame_ms=self.frame_ms, max_attempts=self.max_crop_attempts)
        for e, source, mid, (_, cache) in zip(meta['events'], sources, mids, loaded):
            e.update(fsd_id=source['fsd_id'], fsd_path=source['audio_path'], target_mid=mid,
                     fsd_sha256=cache['source_sha256'], fsd_mids=sorted(source_mids(source)))
        meta.update(background_id=background['id'], background_path=background['audio_path'],
            background_sha256=file_sha256(background['audio_path']), wrong_mid=wrong,
            waveform_sha256={k: waveform_hash(v) for k, v in zip(
                ('background', 'a', 'b', 'mixture'), (bg, *targets, mix))})
        return bg, targets, mix, meta

    def _draw(self, index, seed):
        background, overlap = self.path_list[index//2], bool(index % 2)
        rng = np.random.default_rng(seed)
        for _ in range(self.max_pair_attempts):
            sources, mids, wrong = self._select(background, rng)
            mix_seed = int(rng.integers(0, 2**63-1))
            snr = (float(self.snr_db) if self.snr_db is not None else
                   sample_snr_db(self.snr_range, sample_id=background['id'], seed=mix_seed))
            try:
                return self._mix(background, sources, mids, wrong, mix_seed, overlap, snr)
            except PlacementError:
                continue
        raise ValueError(f'Cannot construct sample: {background["id"]}, overlap={overlap}; '
                         f'exhausted {self.max_pair_attempts} source-pair attempts')

    def _generate_row(self, index):
        seed = int(digest([MODE, self.path_list[index//2]['id'], index % 2, self.seed]), 16)
        return self._draw(index, seed)[-1]

    def get_with_metadata(self, index):
        if self.partition == 'tr':
            bg, targets, mix, meta = self._draw(index, int(np.random.randint(0, 2**63-1)))
        else:
            row = self.manifest['rows'][index]
            sources = [self.fsd_by_id[e['fsd_id']] for e in row['events']]
            mids = [e['target_mid'] for e in row['events']]
            bg, targets, mix, meta = self._mix(self.path_list[index//2], sources, mids,
                row['wrong_mid'], row['seed'], row['overlap'], row['snr_db'])
            if meta != row:
                raise ValueError('Experiment 3 CV source/geometry/gain/waveform replay changed')
        mids = [e['target_mid'] for e in meta['events']] + [meta['wrong_mid']]
        inputs = dict(mixture_waveform=mix)
        result = dict(mixture_waveform=mix, background_waveform=bg,
                      overlap=meta['overlap'], overlap_fraction=meta['overlap_fraction'],
                      utt_id=f'{self.partition}|{meta["background_id"]}|{index % 2}|{meta["seed"]}')
        for name, mid, waveform in zip(('a', 'b', 'wrong'), mids, (*targets, np.zeros_like(mix))):
            condition = np.zeros(self.num_classes, dtype=np.float32)
            condition[self.mid_to_index[mid]] = 1.
            inputs[f'condition_{name}'] = condition
            result[f'target_{name}'] = waveform
            result[f'mid_{name}'] = mid
        result['multilabel_fsd'] = int(any(len(e['fsd_mids']) > 1 for e in meta['events']))
        return (inputs, result), meta

    def __getitem__(self, index):
        return self.get_with_metadata(index)[0]


def create_tse_dual_dataset(hp, partition, path_list=None, is_training=None):
    if partition not in ('tr', 'cv') or (is_training is not None and is_training != (partition == 'tr')):
        raise ValueError('Sampling mode is determined by partition')
    validate_artifact_paths(hp)
    pools = load_pools(hp)
    train_mids = {m for e in pools['train'] for m in e['mids']}
    return TSEDualWrongDataset(dual_paths(hp, partition) if path_list is None else path_list,
        pools['train' if partition == 'tr' else 'val'], resolve_path(hp.class_map_path, hp.data_path),
        partition=partition, ontology_path=project_path(hp.tse_ontology_path),
        train_condition_mids=train_mids, cache_dir=project_path(hp.mix_cache_dir),
        cv_manifest_path=project_path(hp.mix_cv_manifest_path), seed=hp.mix_seed,
        snr_db=hp.mix_snr_db, snr_db_range=hp.mix_snr_db_range,
        threshold_dbfs=hp.mix_active_threshold_dbfs, frame_ms=hp.mix_active_frame_ms,
        max_pair_attempts=hp.tse_dual_max_pair_attempts, max_crop_attempts=hp.tse_dual_max_crop_attempts,
        num_classes=hp.num_classes, zero_fsd_policy=hp.mix_zero_fsd_policy,
        zero_background_policy=hp.mix_zero_background_policy, no_pair_policy=hp.tse_no_pair_policy)
