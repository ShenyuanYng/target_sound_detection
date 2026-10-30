"""Conservative AudioSet label filtering for conditional source extraction.

Ontology ancestry detects semantic containment, not arbitrary co-occurrence.
Unknown labels fail closed. Missing annotations still cannot prove absence.
"""
import json
from functools import lru_cache
from pathlib import Path

from datasets.fsd50k_cache import file_sha256


class OntologyPolicy:
    def __init__(self, path):
        self.path = str(Path(path).resolve(strict=True))
        rows = json.loads(Path(self.path).read_text())
        self.nodes = {row['id']: row for row in rows}
        if not self.nodes or len(self.nodes) != len(rows):
            raise ValueError('Empty ontology or duplicate MID')
        parents = {mid: set() for mid in self.nodes}
        for mid, row in self.nodes.items():
            for child in row['child_ids']:
                if child not in self.nodes:
                    raise ValueError(f'Ontology child is missing: {child}')
                parents[child].add(mid)
        visiting = set()

        @lru_cache(None)
        def ancestors(mid):
            if mid in visiting:
                raise ValueError('Ontology contains a cycle')
            visiting.add(mid)
            result = {mid}
            for parent in parents[mid]:
                result.update(ancestors(parent))
            visiting.remove(mid)
            return frozenset(result)

        self.related = {mid: set(ancestors(mid)) for mid in self.nodes}
        for child in self.nodes:
            for parent in ancestors(child):
                self.related[parent].add(child)
        self.sha256 = file_sha256(self.path)
        self.excluded_conditions = {mid for mid, row in self.nodes.items()
                                    if row['name'].lower() == 'silence'}

    def known(self, mids):
        return bool(mids) and set(mids) <= self.nodes.keys()

    def forbidden(self, mids):
        mids = set(mids)
        unknown = mids - self.nodes.keys()
        if unknown:
            raise ValueError(f'Cannot infer absence for unknown ontology labels: {sorted(unknown)}')
        return set().union(*(self.related[mid] for mid in mids))

    def conflicts(self, left, right):
        return bool(self.forbidden(left) & set(right))


def source_mids(entry):
    return frozenset(entry.get('original_mids', entry['mids']))
