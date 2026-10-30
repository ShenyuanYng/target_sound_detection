"""Independent experiment-3 entry point. Default action is a data check.

Use --mode train explicitly for a full run; --mode smoke is bounded to one
epoch and a small number of optimizer steps. No Slurm/SSH/synchronization.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml

from datasets.tse_dual_dataset import MODE, PROJECT, project_path
from datasets.fsd50k_cache import file_sha256
from tse_dual_separator import TSEDualSourceSeparator
from utils.config import yaml_to_parser
from utils.tse_config import validate_tse_config


def check_data(model, count):
    model.setup('fit')
    model._build_data_loader('tr')
    model._build_data_loader('cv')
    summaries = {}
    for partition, dataset in model.datasets.items():
        rows = []
        for index in range(min(count, len(dataset))):
            sample, meta = dataset.get_with_metadata(index)
            inputs, targets = sample
            residual = inputs['mixture_waveform'] - (
                targets['background_waveform'] + targets['target_a'] + targets['target_b'])
            if float(np.abs(residual).max()) > 1e-6 or np.any(targets['target_wrong']):
                raise AssertionError('Invalid waveform supervision')
            if bool(meta['overlap']) != bool(index % 2):
                raise AssertionError('Overlap quota changed')
            if meta['overlap'] and not .25 <= meta['overlap_fraction'] <= .5:
                raise AssertionError('Partial overlap outside [25%, 50%]')
            if partition == 'cv':
                from tools.audit_common import equal_samples
                repeated, replay_meta = dataset.get_with_metadata(index)
                equal_samples(sample, repeated)
                if meta != replay_meta:
                    raise AssertionError('CV metadata replay changed')
            rows.append(dict(background_id=meta['background_id'], overlap=meta['overlap'],
                overlap_fraction=meta['overlap_fraction'], seed=meta['seed'],
                fsd_ids=[e['fsd_id'] for e in meta['events']],
                mids=[e['target_mid'] for e in meta['events']], wrong_mid=meta['wrong_mid']))
        summaries[partition] = dict(dataset_report=dataset.report, checked=len(rows), samples=rows)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exp-def', default=str(PROJECT/'exp_defs/tse_strong_dual_fsd_wrong.yaml'))
    parser.add_argument('--mode', choices=('check', 'smoke', 'validate', 'train'), default='check')
    parser.add_argument('--samples', type=int, default=4)
    parser.add_argument('--train-batches', type=int, default=1)
    parser.add_argument('--val-batches', type=int,
                        help='CV batch limit; default 2 in smoke, full CV in validate')
    parser.add_argument('--output', default='/home/syy12632604/projects/TSE/experiments/diagnostics/dual_check')
    parser.add_argument('--checkpoint')
    parser.add_argument('--resume-run-dir')
    args, overrides = parser.parse_known_args()
    if args.resume_run_dir:
        if args.mode != 'train' or overrides:
            raise ValueError('Resume uses the frozen run configuration; use --mode train without overrides')
        from resume_training import continue_run
        continue_run(args.resume_run_dir)
        return
    hp = yaml_to_parser(args.exp_def).parse_args(overrides)
    validate_tse_config(hp)
    if (hp.dataset != MODE or min(args.samples, args.train_batches) < 1
            or (args.val_batches is not None and args.val_batches < 1)):
        raise ValueError('Expected experiment-3 config and positive check limits')
    if args.resume_run_dir and args.mode != 'train':
        raise ValueError('Resume is only supported for explicit training')
    if args.checkpoint and args.mode != 'validate':
        raise ValueError('Checkpoint is only supported for validation')
    if args.mode == 'train':
        from train import main as train
        hp.save_path = str(project_path(hp.save_path))
        train(hp, resume_run_dir=args.resume_run_dir)
        return
    output = project_path(args.output)
    # Refuse to replace previous reports or collide with any source corpus.
    from datasets.tse_dual_dataset import validate_artifact_paths
    validate_artifact_paths(hp, extra_paths=[output])
    output.mkdir(parents=True, exist_ok=False)
    hp.save_path = str(output)
    if args.mode == 'smoke':
        hp.accelerator, hp.devices, hp.num_workers, hp.precision = 'cpu', 1, 0, '32-true'
        hp.epochs, hp.accumulate_grad_batches = 1, 1
    import pytorch_lightning as pl
    pl.seed_everything(hp.seed, workers=True)
    torch.set_num_threads(2)
    model = (TSEDualSourceSeparator.load_from_checkpoint(args.checkpoint, hparams=hp)
             if args.mode == 'validate' and args.checkpoint else TSEDualSourceSeparator(hp))
    if args.mode == 'validate' and not args.checkpoint:
        raise ValueError('Validation requires an explicit checkpoint')
    val_batches = args.val_batches if args.val_batches is not None else (2 if args.mode == 'smoke' else 1.0)
    report = dict(mode=args.mode, config=vars(hp), source_config_sha256=file_sha256(args.exp_def),
                  limits=dict(checked_samples_per_partition=args.samples,
                              train_batches=args.train_batches, val_batches=val_batches))
    report['data'] = check_data(model, args.samples)
    if args.mode in ('smoke', 'validate'):
        trainer = pl.Trainer(accelerator='cpu' if args.mode == 'smoke' else hp.accelerator,
            devices=1, max_epochs=1, max_steps=args.train_batches if args.mode == 'smoke' else -1,
            limit_train_batches=args.train_batches, limit_val_batches=val_batches,
            num_sanity_val_steps=0, logger=False, enable_checkpointing=False,
            enable_progress_bar=False, enable_model_summary=False,
            gradient_clip_val=hp.gradient_clip_val)
        if args.mode == 'smoke':
            before = {name: p.detach().clone() for name, p in model.model.named_parameters()}
            trainer.fit(model)
            report['optimizer_steps'] = trainer.global_step
            report['parameters_updated'] = any(not torch.equal(before[n], p.detach())
                                               for n, p in model.model.named_parameters())
            if trainer.global_step != args.train_batches or not report['parameters_updated']:
                raise AssertionError('Smoke check did not update parameters as requested')
        # Explicit bounded CV also runs when max_steps ended fit before epoch validation.
        trainer.validate(model, verbose=False)
        report['metrics'] = json.loads((output/'val_tse_dual_metrics.json').read_text())
    report['status'] = 'passed'
    (output/'check_report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    (output/'effective_config.yaml').write_text(yaml.safe_dump(vars(hp), sort_keys=False))
    print(json.dumps(dict(status='passed', mode=args.mode, output=str(output)), indent=2))


if __name__ == '__main__':
    main()
