"""Metadata-only label reduction for TSE hierarchy v2 (no audio/ML imports)."""
from collections import Counter
from functools import lru_cache


MODE = 'audioset_strong_dual_fsd_hierarchy_tse_v2'
LABEL_POLICY = 'annotated-most-specific-singleton-v2'


class AncestorReducer:
    """Keep annotated descendants, including all branches of a multi-parent DAG."""

    def __init__(self, rows):
        self.nodes = {row['id']: row for row in rows}
        if not self.nodes or len(self.nodes) != len(rows):
            raise ValueError('Empty ontology or duplicate MID')
        parents = {mid: set() for mid in self.nodes}
        for mid, row in self.nodes.items():
            for child in row['child_ids']:
                if child not in parents:
                    raise ValueError(f'Unknown ontology child: {child}')
                parents[child].add(mid)
        visiting = set()

        @lru_cache(None)
        def ancestors(mid):
            if mid in visiting:
                raise ValueError('Ontology contains a cycle')
            visiting.add(mid)
            result = set(parents[mid])
            for parent in parents[mid]:
                result.update(ancestors(parent))
            visiting.remove(mid)
            return frozenset(result)

        self.ancestors = {mid: ancestors(mid) for mid in self.nodes}
        self.excluded = {mid for mid, row in self.nodes.items()
                         if row['name'].lower() == 'silence'}

    def reduce(self, mids):
        original = frozenset(mids)
        if not original or original - self.nodes.keys():
            raise ValueError('Need nonempty, known original labels')
        redundant = set().union(*(self.ancestors[mid] for mid in original))
        return original - redundant


def positive_pool(entries, reducer, mapped_mids, excluded_ids=()):
    """Return copies with full labels intact; reduce BEFORE applying the class map.

    A multi-label remainder, unknown MID, unmapped most-specific MID, or Silence
    never becomes a target. In particular, unmapped children do not become their
    mapped parents. The caller supplies audited zero/invalid source exclusions.
    """
    excluded_ids, mapped_mids = set(excluded_ids), set(mapped_mids)
    result, rejected, counts = [], Counter(), Counter()
    for entry in entries:
        original = frozenset(entry.get('original_mids', entry['mids']))
        if entry['fsd_id'] in excluded_ids:
            rejected['audited_exclusion'] += 1
            continue
        if not original or original - reducer.nodes.keys():
            rejected['unknown_or_empty_labels'] += 1
            continue
        specific = reducer.reduce(original)
        if len(specific) != 1:
            rejected['multiple_specific_labels'] += 1
            continue
        mid = next(iter(specific))
        if mid not in mapped_mids or mid in reducer.excluded:
            rejected['unmapped_or_silence'] += 1
            continue
        result.append({**entry, 'original_mids': sorted(original),
                       'most_specific_mids': [mid], 'positive_mid': mid})
        counts[mid] += 1
    return result, dict(input_count=len(entries), positive_count=len(result),
                        positive_class_count=len(counts), positive_mid_counts=dict(sorted(counts.items())),
                        rejected_counts=dict(sorted(rejected.items())), label_policy=LABEL_POLICY)


def training_vocabulary(entries, reducer, mapped_mids, excluded_ids=(), expected_count=None):
    if any(entry['split'] != 'train' for entry in entries):
        raise ValueError('Condition vocabulary must use the official training split only')
    pool, report = positive_pool(entries, reducer, mapped_mids, excluded_ids)
    mids = frozenset(report['positive_mid_counts'])
    if not mids or (expected_count is not None and len(mids) != expected_count):
        raise ValueError(f'Expected {expected_count} positive training classes, found {len(mids)}')
    return mids, pool, report
