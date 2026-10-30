"""Shared deterministic replay and source-leakage audits for current datasets."""
from collections import defaultdict
import numpy as np
from datasets.fsd50k_cache import file_sha256
from datasets.train_split import video_id


def equal_samples(left, right):
    for a, b in zip(left, right):
        assert a.keys() == b.keys()
        for key in a:
            if isinstance(a[key], np.ndarray):
                assert np.array_equal(a[key], b[key]), key
            else:
                assert a[key] == b[key], key


def _identity_leakage_report(train, cv):
    def values(dataset, attribute, key):
        return {str(entry[key]) for entry in getattr(dataset, attribute)}
    audio = dict(
        clip_id_overlap=sorted(values(train,'path_list','id') & values(cv,'path_list','id')),
        youtube_video_id_overlap=sorted({video_id(e) for e in train.path_list} &
                                       {video_id(e) for e in cv.path_list}),
        audio_path_overlap=sorted(values(train,'path_list','audio_path') & values(cv,'path_list','audio_path')))
    fsd = dict(
        fsd_id_overlap=sorted(values(train,'fsd_pool','fsd_id') & values(cv,'fsd_pool','fsd_id')),
        audio_path_overlap=sorted(values(train,'fsd_pool','audio_path') & values(cv,'fsd_pool','audio_path')))
    if any(audio.values()) or any(fsd.values()):
        raise AssertionError(f'Train/CV identity leakage: {audio}, {fsd}')
    return dict(audioset=audio, fsd50k=fsd, content_hashes=None, status='passed')

def leakage_report(train, cv, *, full):
    # Identity checks are shared by the Strong and dual-FSD audits.
    report = _identity_leakage_report(train, cv)
    if not full:
        return report
    inventories = {}
    for source, attribute, id_key in (('audioset', 'path_list', 'id'),
                                      ('fsd50k', 'fsd_pool', 'fsd_id')):
        inventories[source] = {}
        for partition, dataset in (('tr', train), ('cv', cv)):
            grouped = defaultdict(list)
            for entry in getattr(dataset, attribute):
                grouped[file_sha256(entry['audio_path'])].append(
                    dict(id=str(entry[id_key]), audio_path=entry['audio_path']))
            inventories[source][partition] = grouped
    details, hashes, suggestions = {}, {}, {}
    for source, parts in inventories.items():
        overlap = sorted(parts['tr'].keys() & parts['cv'].keys())
        hashes[source + '_content_hash_overlap'] = overlap
        details[source] = [dict(sha256=h, tr=parts['tr'][h], cv=parts['cv'][h]) for h in overlap]
        suggestions[source] = sorted({e['id'] for h in overlap for e in parts['tr'][h]})
    audio_hashes = set().union(*(part.keys() for part in inventories['audioset'].values()))
    fsd_hashes = set().union(*(part.keys() for part in inventories['fsd50k'].values()))
    cross = sorted(audio_hashes & fsd_hashes)
    hashes['cross_source_content_hash_overlap'] = cross
    details['cross_source'] = [dict(sha256=h, sources={
        source: {p: entries.get(h, []) for p, entries in parts.items()}
        for source, parts in inventories.items()}) for h in cross]
    report.update(content_hashes=hashes, content_duplicates=details,
                  suggested_train_exclusions=suggestions,
                  status='failed' if any(hashes.values()) else 'passed')
    return report
