"""Dataset factories for the three retained TSE variants."""

def datapath_creator(hparams, partition):
    """Join strong annotations, weak labels, and available audio files."""
    if hparams.dataset == 'audioset_strong_dual_fsd_hierarchy_tse_v2':
        from datasets.tse_hierarchy_dataset import dual_paths
        return dual_paths(hparams, partition)
    if hparams.dataset == 'audioset_strong_dual_fsd_wrong_tse':
        from datasets.tse_dual_dataset import dual_paths
        return dual_paths(hparams, partition)
    if hparams.dataset == 'audioset_strong_fsd50k_tse':
        from datasets.tse_dataset import tse_paths
        return tse_paths(hparams, partition)
    raise ValueError(f'Unsupported TSE dataset: {hparams.dataset}')


def dataset_creator(hparams, partition, path_list=None, is_training=None):
    """Create one AudioSet Strong split without importing unrelated tasks."""
    if hparams.dataset == 'audioset_strong_dual_fsd_hierarchy_tse_v2':
        from datasets.tse_hierarchy_dataset import create_tse_hierarchy_dataset
        return create_tse_hierarchy_dataset(hparams, partition, path_list, is_training)
    if hparams.dataset == 'audioset_strong_dual_fsd_wrong_tse':
        from datasets.tse_dual_dataset import create_tse_dual_dataset
        return create_tse_dual_dataset(hparams, partition, path_list, is_training)
    if hparams.dataset == 'audioset_strong_fsd50k_tse':
        from datasets.tse_dataset import create_tse_dataset
        return create_tse_dataset(hparams, partition, path_list, is_training)
    raise ValueError(f'Unsupported TSE dataset: {hparams.dataset}')

