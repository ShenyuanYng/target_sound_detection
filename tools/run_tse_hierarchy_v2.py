"""Hierarchy v2 entry point. Default: print a plan without creating data or models.

Training requires --mode train. This tool does not submit Slurm jobs or modify v1.
"""
import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from utils.config import yaml_to_parser
from utils.tse_hierarchy_config import validate_hierarchy_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exp-def', default=str(PROJECT / 'exp_defs/tse_strong_dual_fsd_hierarchy_v2.yaml'))
    parser.add_argument('--mode', choices=('plan', 'train'), default='plan')
    parser.add_argument('--resume-run-dir', help='Only a checkpoint produced by this v2 configuration')
    args, overrides = parser.parse_known_args()
    if args.resume_run_dir:
        if args.mode != 'train' or overrides:
            raise ValueError('Resume uses the frozen run configuration; use --mode train without overrides')
        from resume_training import continue_run
        continue_run(args.resume_run_dir)
        return
    hp = yaml_to_parser(args.exp_def).parse_args(overrides)
    validate_hierarchy_config(hp)
    if args.mode == 'plan':
        if args.resume_run_dir:
            raise ValueError('Resume requires explicit --mode train')
        print(json.dumps(dict(mode='plan', dataset=hp.dataset, label_policy=hp.tse_label_policy,
            expected_training_classes=hp.tse_expected_condition_classes,
            condition_mode=hp.tse_train_condition_mode, artifacts={key: getattr(hp, key) for key in (
                'save_path', 'mix_cache_dir', 'mix_background_split_path', 'mix_cv_manifest_path',
                'tse_condition_vocabulary_path')},
            status='configuration only; no audio reads, model creation or job submission'), indent=2))
        return
    if args.resume_run_dir:
        from utils.tse_hierarchy_config import validate_hierarchy_paths
        validate_hierarchy_paths(hp, [args.resume_run_dir])
    # Import training dependencies only after an explicit training request.
    from train import main as train
    train(hp, resume_run_dir=args.resume_run_dir)


if __name__ == '__main__':
    main()
