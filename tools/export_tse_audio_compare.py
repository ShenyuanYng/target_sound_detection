"""Replay fixed TSE CV with its recorded best checkpoint and export listening WAVs.

SDR columns mean the existing training SI-SDR, not BSS Eval SDR. No training,
new mixtures, rescaling, clipping, alignment correction or audio postprocessing.
"""
import argparse
import csv
import json
from argparse import Namespace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import soundfile as sf
import torch

from datasets.fsd50k_cache import file_sha256
from datasets.paths.audioset_paths import load_class_map, resolve_path
from loss_functions.tse_loss import si_sdr


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def choose_samples(rows, seed):
    """Six seeded, MID-diverse random rows, then distinct high/low tail rows.

    Extremes come from the upper/lower decile of the entire valid CV. Prefer
    unused MIDs within that tail, falling back to repeats when necessary.
    """
    valid = [r for r in rows if r['SDRi'] is not None]
    if len(valid) < 10:
        raise ValueError('Need at least 10 non-silent CV references')
    rng = np.random.default_rng(seed)
    selected, used_indices, used_mids = [], set(), set()

    def take(pool, count, method):
        for _ in range(count):
            candidates = [r for r in pool if r['cv_index'] not in used_indices]
            if not candidates:
                raise ValueError('Not enough distinct samples in ' + method)
            diverse = [r for r in candidates if r['MID'] not in used_mids]
            row = (diverse or candidates)[0]
            selected.append(dict(row, sampling_method=method))
            used_indices.add(row['cv_index'])
            used_mids.add(row['MID'])

    take([valid[i] for i in rng.permutation(len(valid))], 6, 'random_mid_diverse')
    ranked = sorted(valid, key=lambda r: (r['SDRi'], r['cv_index']))
    # At least 10 entries ensures the six random selections cannot exhaust a tail.
    tail = max(10, int(np.ceil(len(valid) * .1)))
    take(list(reversed(ranked[-tail:])), 2, 'high_sdri_top_decile')
    take(ranked[:tail], 2, 'low_sdri_bottom_decile')
    return selected


def check_wave(wave):
    if wave.dtype != np.float32 or wave.shape != (160000,) or not np.isfinite(wave).all():
        raise ValueError('Expected finite float32 mono 160000-sample waveform')


def export_wav(path, wave):
    check_wave(wave)
    sf.write(str(path), wave, 16000, subtype='FLOAT')
    restored, rate = sf.read(str(path), dtype='float32')
    info = sf.info(str(path))
    if rate != 16000 or info.channels != 1 or info.subtype != 'FLOAT':
        raise ValueError('Incorrect exported WAV format')
    check_wave(restored)
    np.testing.assert_array_equal(restored, wave)
    return float(np.max(np.abs(restored - wave)))


def best_checkpoint(run, expected_epoch=None):
    last = torch.load(str(run / 'mylast.ckpt'), map_location='cpu', weights_only=False)
    callbacks = [v for v in last.get('callbacks', {}).values()
                 if isinstance(v, dict) and v.get('monitor') == 'val_loss'
                 and v.get('best_model_path')]
    if len(callbacks) != 1:
        raise ValueError('Cannot unambiguously resolve recorded best val_loss checkpoint')
    record = callbacks[0]
    recorded = Path(record['best_model_path'])
    candidates = list(run.rglob(recorded.name))
    if len(candidates) != 1:
        raise ValueError('Recorded best checkpoint is missing or ambiguous')
    checkpoint = candidates[0]
    saved = torch.load(str(checkpoint), map_location='cpu', weights_only=False)
    if expected_epoch is not None and int(saved['epoch']) + 1 != expected_epoch:
        raise ValueError(f'Best checkpoint epoch is {int(saved["epoch"]) + 1}, expected {expected_epoch}')
    return checkpoint, saved, float(record['best_model_score'])


def run(args):
    from creators.dataset import dataset_creator
    from tse_separator import TSESourceSeparator

    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError('Output already exists; choose a new directory to avoid overwriting')
    checkpoint, saved, best_loss = best_checkpoint(Path(args.run_dir).resolve(), args.expected_epoch)
    epoch = int(saved['epoch']) + 1
    hp = Namespace(**saved['hyper_parameters'])
    manifest_path = Path(resolve_path(hp.mix_cv_manifest_path, hp.data_path))
    if not manifest_path.is_file():
        raise ValueError('Existing CV manifest required; will not generate a new manifest')
    manifest_hash = file_sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    # Fail before dataset construction if local replay code differs from training.
    import datasets.tse_dataset as dataset_module
    for name, expected in manifest['settings']['code'].items():
        source = Path(dataset_module.__file__).with_name(name + '.py')
        if file_sha256(source) != expected:
            raise ValueError(f'CV replay source differs from manifest: {source}; use original training code')
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0' if args.device == 'cuda' else 'cpu')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable; run inside a GPU allocation')
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device)
    print(f'Loading epoch {epoch}, CV Loss={best_loss:.6f}: {checkpoint}', flush=True)
    model = TSESourceSeparator(hp)
    model.on_load_checkpoint(saved)
    model.load_state_dict(saved['state_dict'], strict=True)
    del saved
    model.to(device).eval()
    if any(m.training for m in model.modules()):
        raise ValueError('Model did not enter eval mode')
    cv = dataset_creator(hp, 'cv')
    if file_sha256(manifest_path) != manifest_hash:
        raise ValueError('CV manifest changed during dataset construction')
    mid_to_index, names = load_class_map(resolve_path(hp.class_map_path, hp.data_path))

    @torch.inference_mode()
    def infer(index):
        sample, meta = cv.get_with_metadata(index)  # validates original hashes/gains/geometry
        inputs, targets = sample
        mid = targets['condition_mid']
        expected = np.zeros(hp.num_classes, dtype=np.float32)
        expected[mid_to_index[mid]] = 1.
        np.testing.assert_array_equal(inputs['condition'], expected)
        if mid != manifest['rows'][index]['target_mid']:
            raise ValueError('Condition does not match CV manifest')
        mixture, target = inputs['mixture_waveform'], targets['target_waveform']
        check_wave(mixture)
        check_wave(target)
        x = torch.from_numpy(mixture).unsqueeze(0).to(device)
        y = torch.from_numpy(target).unsqueeze(0).to(device)
        condition = torch.from_numpy(expected).unsqueeze(0).to(device)
        predicted = model(dict(mixture_waveform=x, condition=condition))['target_waveform']
        prediction = predicted[0].cpu().numpy()
        check_wave(prediction)
        score, valid = si_sdr(predicted, y, hp.tse_silence_rms)
        baseline, _ = si_sdr(x, y, hp.tse_silence_rms)
        row = dict(cv_index=index, utt_id=targets['utt_id'], MID=mid,
                   class_name=names[mid_to_index[mid]], Condition=mid,
                   condition_index=mid_to_index[mid],
                   SDR=float(score[0]) if valid[0] else None,
                   SDRi=float(score[0] - baseline[0]) if valid[0] else None,
                   Mixture_SDR=float(baseline[0]) if valid[0] else None,
                   fsd_id=manifest['rows'][index]['fsd_id'],
                   multilabel_fsd=bool(meta['multilabel_fsd']),
                   fsd_mids=json.dumps(meta['fsd_mids']))
        return row, dict(mixture=mixture, prediction=prediction, target_gt=target)

    rows = []
    for i in range(len(cv)):
        row, _ = infer(i)
        rows.append(row)
        if (i + 1) % 50 == 0 or i + 1 == len(cv):
            print(f'Fixed CV inference: {i + 1}/{len(cv)}', flush=True)
    selected = choose_samples(rows, args.seed)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / 'cv_scores.csv', rows)
    summary, max_export_error = [], 0.
    for i, row in enumerate(selected, 1):
        repeated, waves = infer(row['cv_index'])
        for key in ('SDR', 'SDRi', 'Mixture_SDR'):
            np.testing.assert_allclose(repeated[key], row[key], rtol=0, atol=1e-4)
        directory = output / f'sample_{i:03d}'
        directory.mkdir()
        for name, wave in waves.items():
            max_export_error = max(max_export_error, export_wav(directory / f'{name}.wav', wave))
        summary.append(dict(Sample_ID=directory.name, **row))
    if file_sha256(manifest_path) != manifest_hash:
        raise ValueError('CV manifest changed during inference')
    write_csv(output / 'summary.csv', summary)
    valid = [r for r in rows if r['SDR'] is not None]
    report = dict(status='passed', checkpoint=str(checkpoint), checkpoint_sha256=file_sha256(checkpoint),
                  epoch=epoch, recorded_best_cv_loss=best_loss,
                  cv_manifest=str(manifest_path), cv_manifest_sha256=manifest_hash,
                  seed=args.seed, device=str(device), evaluated_cv_samples=len(rows),
                  gpu_memory_fraction=args.gpu_memory_fraction,
                  peak_allocated_gib=(torch.cuda.max_memory_allocated(device) / 1024**3
                                      if device.type == 'cuda' else None),
                  valid_cv_samples=len(valid), silent_cv_samples=len(rows) - len(valid),
                  cv_si_sdr=float(np.mean([r['SDR'] for r in valid])),
                  cv_si_sdri=float(np.mean([r['SDRi'] for r in valid])),
                  selected_mid_count=len({r['MID'] for r in selected}),
                  sample_count=len(summary), wav_count=3 * len(summary),
                  wav_max_roundtrip_error=max_export_error,
                  audio='16000 Hz, mono, 160000 samples, FLOAT WAV; no independent normalization',
                  metric='Training zero-mean SI-SDR; SDRi = prediction SI-SDR - mixture SI-SDR',
                  sampling='6 seeded random rows preferring unused MIDs; 2 highest/2 lowest within '
                           'top/bottom decile (minimum 10 candidates), preferring unused MIDs; no repeats',
                  target_semantics='Whole FSD stem, including co-occurring sounds; not a pure MID source')
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output', default='experiments/inference/tse_audio_compare')
    parser.add_argument('--expected-epoch', type=int, help='Optional one-based epoch assertion; defaults to recorded best')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--gpu-memory-fraction', type=float, default=1.,
                        help='Upper bound for this process PyTorch GPU allocator, in (0,1]')
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error('--cpu-threads must be positive')
    if not 0 < args.gpu_memory_fraction <= 1:
        parser.error('--gpu-memory-fraction must be in (0,1]')
    run(args)


if __name__ == '__main__':
    main()
