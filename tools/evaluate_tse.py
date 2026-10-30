"""Audit fixed TSE CV, optionally evaluate a checkpoint and export aligned audio.

Without a checkpoint this only audits data; it never trains a model.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import soundfile as sf
import torch

from creators.dataset import dataset_creator
from datasets.paths.audioset_paths import resolve_path
from loss_functions.tse_loss import si_sdr
from tools.audit_common import leakage_report
from utils.collate import collate_seq
from utils.config import yaml_to_parser
from utils.tse_config import validate_tse_config


def missing_inputs(hp):
    names = ('class_map_path', 'mix_strong_tsv', 'mix_parent_background_split_path',
             'mix_fsd_split_path', 'mix_fsd_dev_audio', 'mix_fsd_ground_truth')
    paths = [resolve_path(getattr(hp, name), hp.data_path) for name in names]
    for name in ('mix_audio_dirs', 'mix_weak_csvs'):
        value = getattr(hp, name)
        paths.extend(resolve_path(p, hp.data_path) for p in ([value] if isinstance(value, str) else value))
    return [p for p in paths if not Path(p).exists()]


def run(hp, output, checkpoint=None, max_samples=0, export_count=5):
    validate_tse_config(hp)
    absent = missing_inputs(hp)
    if absent:
        return dict(status='unavailable', reason='Configured corpus/metadata are not present locally',
                    missing_paths=absent, filtered_counts=None, class_coverage=None)
    datasets = {p: dataset_creator(hp, p) for p in ('tr', 'cv')}
    for dataset in datasets.values():
        if any(Path(e['audio_path']).resolve().parent == output.resolve() or
               Path(e['audio_path']).resolve().parent in output.resolve().parents
               for e in dataset.path_list + dataset.fsd_pool):
            raise ValueError('Exports must be outside original audio directories')
    leakage = leakage_report(datasets['tr'], datasets['cv'], full=True)
    if leakage['status'] != 'passed':
        raise ValueError('Source leakage: ' + json.dumps(leakage))
    cv = datasets['cv']
    cv.report['cv_mids_without_legal_train_support'] = sorted(
        cv.report['eligible_condition_background_counts'].keys()
        - datasets['tr'].report['eligible_condition_background_counts'].keys())
    model = None
    if checkpoint:
        from tse_separator import TSESourceSeparator
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model = TSESourceSeparator(hp)
        model.on_load_checkpoint(saved)
        model.load_state_dict(saved['state_dict'], strict=True)
        model.eval()
    count = min(len(cv), max_samples) if max_samples else len(cv)
    rows, worst_error = [], 0.
    for i in range(count):
        sample, meta = cv.get_with_metadata(i)
        repeated, _ = cv.get_with_metadata(i)
        inputs, targets = sample
        for key in ('target_waveform', 'background_waveform', 'mixture_waveform'):
            np.testing.assert_array_equal(targets[key], repeated[1][key])
        worst_error = max(worst_error, meta['stem_sum_max_error'])
        prediction = None
        row = dict(index=i, utt_id=targets['utt_id'], condition_mid=targets['condition_mid'], **meta)
        if model is not None:
            batch = collate_seq([sample])
            with torch.no_grad():
                predicted = model(batch[0])
                losses = model.loss(predicted, batch[1])
                score, valid = si_sdr(predicted['target_waveform'], batch[1]['target_waveform'][0], hp.tse_silence_rms)
                baseline, _ = si_sdr(batch[0]['mixture_waveform'][0], batch[1]['target_waveform'][0], hp.tse_silence_rms)
            row.update(loss=float(losses['loss']), mrstft=float(losses['mrstft']),
                si_sdr=float(score[0]) if valid[0] else None,
                si_sdri=float(score[0] - baseline[0]) if valid[0] else None)
            prediction = predicted['target_waveform'][0].cpu().numpy()
        if i < export_count:
            directory = output / f'sample_{i:04d}'
            directory.mkdir(parents=True, exist_ok=True)
            waves = dict(mixture=inputs['mixture_waveform'], target=targets['target_waveform'],
                         background=targets['background_waveform'])
            if prediction is not None:
                waves['prediction'] = prediction
            # Float WAV avoids clipping/independent gains; preserve relative amplitudes.
            for name, waveform in waves.items():
                sf.write(str(directory / f'{name}.wav'), waveform, 16000, subtype='FLOAT')
            (directory / 'metadata.json').write_text(json.dumps(row, indent=2, allow_nan=False))
        rows.append(row)
    report = dict(status='passed', partitions={p: d.report for p, d in datasets.items()},
        leakage=leakage, evaluated_cv_samples=count, full_cv=count == len(cv),
        stem_sum_max_error=worst_error, replay_exact=True, checkpoint=checkpoint, samples=rows)
    if model is not None:
        for key in ('si_sdr', 'si_sdri', 'loss', 'mrstft'):
            values = [r[key] for r in rows if r[key] is not None]
            report[key] = float(np.mean(values)) if values else None
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exp-def', default='exp_defs/tse_strong_fsd50k.yaml')
    parser.add_argument('--output', required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--max-samples', type=int, default=0, help='0 audits/evaluates all fixed CV mixtures')
    parser.add_argument('--export-count', type=int, default=5)
    args, overrides = parser.parse_known_args()
    if args.max_samples < 0 or args.export_count < 0:
        parser.error('Sample/export counts must be nonnegative')
    hp = yaml_to_parser(args.exp_def).parse_args(overrides)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    try:
        report = run(hp, output, args.checkpoint, args.max_samples, args.export_count)
    except (ValueError, OSError, AssertionError) as error:
        report = dict(status='failed', error=str(error), dataset_report=getattr(error, 'report', None))
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('samples', 'partitions', 'leakage')}, indent=2))
    return 0 if report['status'] == 'passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
