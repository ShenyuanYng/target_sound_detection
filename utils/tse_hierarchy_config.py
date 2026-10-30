"""Keep v2 outputs separate from source corpora and every v1 artifact."""
from pathlib import Path

from datasets.tse_hierarchy_labels import MODE, LABEL_POLICY


def validate_hierarchy_paths(hp, extra_paths=()):
    from utils.project_paths import training_path
    root = training_path(hp.tse_artifact_root)
    cache = Path(hp.mix_cache_dir).resolve()
    manifests = [Path(getattr(hp, key)) for key in (
        'mix_cv_manifest_path', 'mix_background_split_path', 'tse_condition_vocabulary_path')]
    for path in manifests:
        if not path.is_absolute() or root not in path.resolve().parents:
            raise ValueError(f'Hierarchy manifests must be inside {root}: {path}')
    output = training_path(hp.save_path)
    for path in extra_paths:
        training_path(path)
    resolved = [cache, output, *[p.resolve() for p in manifests]]
    if len(set(resolved)) != len(resolved):
        raise ValueError('Hierarchy artifact paths must be distinct')
    for path in resolved[2:]:
        if cache in path.parents or output in path.parents:
            raise ValueError('Put hierarchy manifests outside cache and training directories')


def validate_hierarchy_config(hp):
    if hp.dataset != MODE or hp.tse_label_policy != LABEL_POLICY:
        raise ValueError('Expected explicit hierarchy v2 dataset and label policy')
    if type(hp.tse_expected_condition_classes) is not int or hp.tse_expected_condition_classes != 188:
        raise ValueError('This version requires the 188-class positive training vocabulary')
    validate_hierarchy_paths(hp)
