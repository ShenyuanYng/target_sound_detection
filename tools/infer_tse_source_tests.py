"""Two source-generalization listening tests; no training or separation scores.

Experiment 1 uses unchanged AudioSet CV waveforms. Experiment 2 adds a same-MID
FSD Val stem using the existing mixer. The same AudioSet/MID pairs are used in
both experiments for direct comparison. New selection metadata is independent
of the training CV manifest, which is only read, never rewritten.
"""
import argparse
from argparse import Namespace
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from datasets.active_rms import make_snr_mixture
from datasets.fsd50k_cache import file_sha256
from datasets.mix_utils import compute_rms, read_mono_waveform, sample_snr_db
from datasets.paths.audioset_paths import load_class_map, read_strong_labels, resolve_path
from datasets.paths.fsd50k_paths import read_fsd_dev_index
from tools.export_tse_audio_compare import best_checkpoint, check_wave, export_wav, write_csv


def wave_hash(wave):
    return hashlib.sha256(np.asarray(wave, dtype='<f4').tobytes()).hexdigest()


def positive_intervals(events, mid):
    """Merge positive clip-relative intervals; do not treat missing labels as GT."""
    intervals = []
    for e in events:
        if e['mid'] != mid or not e['present']:
            continue
        a, b = float(e['onset']), float(e['offset'])
        if not np.isfinite([a, b]).all() or not 0 <= a <= b <= 10:
            raise ValueError('Invalid Strong interval')
        if a < b:
            intervals.append((a, b))
    merged = []
    for a, b in sorted(intervals):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(b, merged[-1][1])
        else:
            merged.append([a, b])
    return merged


def overlap_seconds(start, end, intervals, margin=0.):
    # Merge expanded intervals to avoid double counting.
    expanded = positive_intervals([
        dict(mid='x', present=True, onset=max(0., a-margin), offset=min(10., b+margin))
        for a, b in intervals], 'x')
    return sum(max(0., min(end, b)-max(start, a)) for a, b in expanded)


def read_fsd(path):
    """Same conversion as fsd50k_cache, but read-only: do not populate caches."""
    wave, rate = sf.read(str(path), dtype='float32', always_2d=True)
    wave = wave.mean(axis=1, dtype=np.float32)
    if rate == 44100:
        wave = resample_poly(wave, 160, 441, window=('kaiser', 5.0), padtype='constant')
    elif rate != 16000:
        raise ValueError(f'Unsupported FSD sample rate: {rate}')
    if not len(wave) or not np.isfinite(wave).all():
        raise ValueError('Invalid FSD waveform')
    return wave


def json_write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def load_inventory(hp, run, epoch):
    resolve = lambda p: Path(resolve_path(p, hp.data_path))
    paths = dict(cv_manifest=resolve(hp.mix_cv_manifest_path),
                 background_split=resolve(hp.mix_background_split_path),
                 fsd_split=resolve(hp.mix_fsd_split_path),
                 strong_labels=resolve(hp.mix_strong_tsv), class_map=resolve(hp.class_map_path),
                 fsd_labels=resolve(hp.mix_fsd_ground_truth)/'dev.csv')
    hashes = {k: dict(path=str(p), sha256=file_sha256(p)) for k, p in paths.items()}
    manifest = json.loads(paths['cv_manifest'].read_text())
    split = json.loads(paths['background_split'].read_text())
    fsd_split = json.loads(paths['fsd_split'].read_text())
    mapping, names = load_class_map(str(paths['class_map']))
    if len(names) != hp.num_classes:
        raise ValueError('Checkpoint/class map mismatch')
    if hashes['class_map']['sha256'] != manifest['settings']['class_map_sha256']:
        raise ValueError('Training class map changed')
    if hashes['strong_labels']['sha256'] != split['strong_sha256']:
        raise ValueError('Strong labels changed from saved split')
    if set(split['tr']) & set(split['cv']):
        raise ValueError('AudioSet Train/CV overlap')
    if {i.rsplit('_', 1)[0] for i in split['tr']} & {i.rsplit('_', 1)[0] for i in split['cv']}:
        raise ValueError('AudioSet Train/CV video overlap')
    bg = {r['background_id']: r for r in manifest['rows']}
    if len(bg) != len(manifest['rows']) or not set(bg) <= set(split['cv']):
        raise ValueError('Existing CV manifest contains duplicate/non-CV backgrounds')
    pools, _ = read_fsd_dev_index(str(resolve(hp.mix_fsd_dev_audio)),
                                str(resolve(hp.mix_fsd_ground_truth)), str(paths['class_map']))
    if fsd_split.get('strategy') != 'official_dev_train_val' or any(
            fsd_split[p] != [e['fsd_id'] for e in pools[p]] for p in ('train', 'val')):
        raise ValueError('FSD official split differs from training inventory')
    if set(fsd_split['train']) & set(fsd_split['val']):
        raise ValueError('FSD Train/Val overlap')
    # Use observed training counts from epochs already included in this checkpoint.
    # Later epoch reports cannot establish which conditions the selected checkpoint saw.
    observed = set()
    evidence = []
    for p in sorted(run.rglob('train_tse_metrics.json')):
        report = json.loads(p.read_text())
        if 0 <= int(report['epoch']) < epoch:
            observed.update(int(i) for i, n in report['sampled_class_index_counts'].items() if n > 0)
            evidence.append(dict(path=str(p), sha256=file_sha256(p),
                                 epoch_one_based=int(report['epoch'])+1))
    if not evidence:
        raise ValueError('No observed training Condition report at/before checkpoint epoch')
    exclusions = set(hp.mix_content_duplicate_exclusions['fsd50k'])
    train_mids = {mid for e in pools['train'] if e['fsd_id'] not in exclusions for mid in e['mids']}
    allowed = {mid for mid, i in mapping.items() if i in observed and mid in train_mids}
    # Preserve all original FSD labels for co-occurrence reporting.
    import csv
    with paths['fsd_labels'].open(encoding='utf-8-sig', newline='') as f:
        original = {r['fname']: r['mids'].split(',') for r in csv.DictReader(f)}
    by_mid = defaultdict(list)
    for entry in pools['val']:
        entry['original_mids'] = sorted(set(m.strip() for m in original[entry['fsd_id']]))
        for mid in sorted(set(entry['mids']) & allowed):
            by_mid[mid].append(entry)
    events = read_strong_labels(str(paths['strong_labels']))
    candidates = defaultdict(list)
    for identity, entry in sorted(bg.items()):
        for mid in sorted({e['mid'] for e in events.get(identity, []) if e['present']} & allowed & by_mid.keys()):
            intervals = positive_intervals(events[identity], mid)
            if intervals:
                candidates[mid].append((entry, intervals))
    return mapping, names, candidates, by_mid, dict(
        inputs=hashes, observed_training_evidence=evidence,
        condition_policy='Only indices observed in saved training reports at/before checkpoint epoch',
        observed_supported_mid_count=len(allowed), candidate_mid_count=len(candidates),
        candidate_audioset_count=len({r['background_id'] for v in candidates.values() for r, _ in v}),
        candidate_mids=sorted(candidates))


def select_pairs(hp, candidates, fsd_by_mid, args):
    rng = np.random.default_rng(args.seed)
    mids = [str(m) for m in rng.permutation(sorted(candidates))]
    chosen, used_bg, used_fsd = [], set(), set()
    skipped = Counter()
    audio_cache, fsd_cache = {}, {}

    def attempt(mid):
        backgrounds = [candidates[mid][i] for i in rng.permutation(len(candidates[mid]))]
        sources = [fsd_by_mid[mid][i] for i in rng.permutation(len(fsd_by_mid[mid]))]
        # Keep co-occurrence ambiguity low without inventing cleaner GT.
        sources.sort(key=lambda f: len(f['original_mids']) > 1)
        best, best_overlap = None, float('inf')
        for trial in range(args.max_attempts_per_mid):
            entry, intervals = backgrounds[trial % len(backgrounds)]
            fsd = sources[(trial // len(backgrounds)) % len(sources)]
            # Also change source when there are many available backgrounds.
            if len(backgrounds) >= args.max_attempts_per_mid:
                fsd = sources[trial % len(sources)]
            identity, fsd_id = entry['background_id'], fsd['fsd_id']
            if identity in used_bg or fsd_id in used_fsd:
                skipped['already_used_source'] += 1
                continue
            if identity not in audio_cache:
                if file_sha256(entry['background_path']) != entry['background_sha256']:
                    raise ValueError('AudioSet source differs from existing CV manifest')
                wave, _ = read_mono_waveform(entry['background_path'])
                check_wave(wave)
                audio_cache[identity] = wave
            if fsd_id not in fsd_cache:
                fsd_cache[fsd_id] = read_fsd(fsd['audio_path'])
            original, raw_fsd = audio_cache[identity], fsd_cache[fsd_id]
            if compute_rms(raw_fsd) == 0:
                skipped['zero_fsd_source'] += 1
                continue
            seed = int(rng.integers(0, 2**63-1))
            snr = (float(hp.mix_snr_db) if hp.mix_snr_db is not None else
                   sample_snr_db(hp.mix_snr_db_range, sample_id=identity, seed=seed))
            # Same-MID presence is deliberate in this inference-only experiment.
            # Clearing the exclusion input does NOT change training or its labels.
            bg, target, mix, meta = make_snr_mixture(original, raw_fsd,
                background_id=identity, fsd_id=fsd_id, target_mid=mid, seed=seed,
                background_weak_mids=[], fsd_mids=fsd['mids'], snr_db=snr,
                normalization='peak', threshold_dbfs=hp.mix_active_threshold_dbfs,
                frame_ms=hp.mix_active_frame_ms)
            active_seconds = meta['active_sample_count']/16000
            if (meta['rms_fallback'] != 'none' or active_seconds < args.min_active_seconds
                    or compute_rms(target) < args.min_target_rms):
                skipped['inaudible_or_insufficient_active_fsd_crop'] += 1
                continue
            a, b = meta['target_start_sample']/16000, meta['target_end_sample']/16000
            overlap = overlap_seconds(a, b, intervals, args.margin_seconds)
            if overlap < best_overlap:
                check_wave(bg); check_wave(target); check_wave(mix)
                error = float(np.max(np.abs(mix-(bg+target))))
                if error > 1e-6:
                    raise ValueError('Mixture/stem mismatch')
                row = dict(MID=mid, audioset_id=identity, audioset_path=entry['background_path'],
                    audioset_sha256=entry['background_sha256'], strong_intervals=intervals,
                    fsd_id=fsd_id, fsd_path=fsd['audio_path'], fsd_split='val',
                    fsd_original_mids=fsd['original_mids'], fsd_multilabel=len(fsd['original_mids']) > 1,
                    mixing_seed=seed, mixing=meta,
                    fsd_start_seconds=a, fsd_end_seconds=b,
                    strong_overlap_seconds=overlap_seconds(a, b, intervals),
                    margin_overlap_seconds=overlap, stem_sum_max_error=error)
                waves = dict(original=original, audioset_background=bg, fsd_gt=target, mixture=mix)
                row['waveform_sha256'] = {k: wave_hash(w) for k, w in waves.items()}
                best, best_overlap = (row, waves), overlap
            if overlap == 0:
                break
        return best

    # One sample per MID first; repeat classes only if diversity is exhausted.
    for cycle in range(args.count):
        for mid in mids:
            if len(chosen) == args.count:
                break
            result = attempt(mid)
            if result is not None:
                row, waves = result
                row['fsd_sha256'] = file_sha256(row['fsd_path'])
                chosen.append((row, waves))
                used_bg.add(row['audioset_id']); used_fsd.add(row['fsd_id'])
                print(f"Selected {len(chosen)}/{args.count}: {mid}, overlap={row['strong_overlap_seconds']:.3f}s", flush=True)
        if len(chosen) == args.count:
            break
        if cycle == 0:
            print('Distinct MID pool exhausted; trying unused recordings from repeated MIDs', flush=True)
    if len(chosen) != args.count:
        raise ValueError(f'Only {len(chosen)}/{args.count} valid pairs found; exclusions={dict(skipped)}')
    return chosen, dict(skipped)


def summary_row(i, row, mapping, names, experiment):
    common = dict(Sample_ID=f'sample_{i:03d}', MID=row['MID'],
        class_name=names[mapping[row['MID']]], Condition=row['MID'],
        condition_index=mapping[row['MID']], audioset_id=row['audioset_id'],
        strong_intervals_seconds=json.dumps(row['strong_intervals']),
        audio_sample_rate=16000, audio_seconds=10, audioset_split='cv')
    if experiment == 2:
        common.update(fsd_id=row['fsd_id'], fsd_split='val',
            fsd_start_seconds=row['fsd_start_seconds'], fsd_end_seconds=row['fsd_end_seconds'],
            overlap_seconds=row['strong_overlap_seconds'],
            placement_nonoverlap=row['strong_overlap_seconds'] == 0,
            fsd_multilabel=row['fsd_multilabel'], fsd_mids=json.dumps(row['fsd_original_mids']),
            mixing_seed=row['mixing_seed'], requested_snr_db=row['mixing']['snr_db_requested'],
            actual_active_snr_db=row['mixing']['snr_db_exported'],
            full_10s_energy_ratio_db=row['mixing']['snr_db_full_10s_exported'],
            final_common_divisor=row['mixing']['final_normalization_divisor'],
            audioset_gain=(1/row['mixing']['background_rms_before_norm']
                           /row['mixing']['final_normalization_divisor']),
            fsd_gain=row['mixing']['target_gain']/row['mixing']['final_normalization_divisor'])
    return common


def run(args):
    from tse_separator import TSESourceSeparator
    run_dir, output = Path(args.run_dir).resolve(), Path(args.output).resolve()
    if output.exists():
        raise ValueError('Output exists; choose another --output, never overwrite results')
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint, saved, best_loss = best_checkpoint(run_dir)
    epoch = int(saved['epoch']) + 1
    hp = Namespace(**saved['hyper_parameters'])
    print(f'Using recorded best checkpoint: epoch {epoch}, loss={best_loss:.6f}', flush=True)
    mapping, names, candidates, fsd_by_mid, inventory = load_inventory(hp, run_dir, epoch)
    selected, skipped = select_pairs(hp, candidates, fsd_by_mid, args)
    device = torch.device('cuda:0' if args.device == 'cuda' else 'cpu')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA unavailable; run with a GPU allocation or --device cpu')
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device)
    model = TSESourceSeparator(hp)
    model.on_load_checkpoint(saved)
    model.load_state_dict(saved['state_dict'], strict=True)
    del saved
    model.to(device).eval()
    if any(m.training for m in model.modules()):
        raise ValueError('Model must be in eval mode')

    @torch.inference_mode()
    def predict(wave, mid):
        condition = torch.zeros(1, hp.num_classes, device=device)
        condition[0, mapping[mid]] = 1
        assert condition.sum().item() == 1
        result = model(dict(mixture_waveform=torch.from_numpy(wave).unsqueeze(0).to(device),
                            condition=condition))['target_waveform']
        if result.requires_grad:
            raise ValueError('Unexpected autograd graph')
        wave = result[0].cpu().numpy()
        check_wave(wave)
        return wave

    output.mkdir(parents=True, exist_ok=False)
    metadata = dict(status='running', checkpoint=str(checkpoint), checkpoint_sha256=file_sha256(checkpoint),
        epoch=epoch, seed=args.seed, arguments=vars(args), inventory=inventory,
        skipped_candidates=skipped, samples=[r for r, _ in selected],
        notes=['No separation score: AudioSet has no isolated target waveform.',
               'FSD GT is the whole processed FSD stem, not the sum of same-class sources.',
               'FSD insertion interval is source placement, not semantic event timing.',
               'Experiment 1 keeps original amplitude; experiment 2 uses training mixer gains.',
               'Selection filters near-silent crops and prefers single-label FSD; not a random benchmark.',
               'Same AudioSet/MID pairs in both experiments; no independent export normalization.'])
    json_write(output/'manifest.json', metadata)
    for experiment in (1, 2):
        folder = output/f'experiment_{experiment}'
        folder.mkdir()
        summaries = []
        for i, (row, waves) in enumerate(selected, 1):
            inputs = waves['original' if experiment == 1 else 'mixture']
            prediction = predict(inputs, row['MID'])
            directory = folder/f'sample_{i:03d}'
            directory.mkdir()
            exports = ({'original': waves['original']} if experiment == 1 else
                       {k: waves[k] for k in ('audioset_background', 'fsd_gt', 'mixture')})
            exports['prediction'] = prediction
            for name, wave in exports.items():
                export_wav(directory/f'{name}.wav', wave)
            summaries.append(summary_row(i, row, mapping, names, experiment))
            print(f'Experiment {experiment}: {i}/{args.count} exported', flush=True)
        write_csv(folder/'summary.csv', summaries)
    for item in inventory['inputs'].values():
        if file_sha256(item['path']) != item['sha256']:
            raise ValueError('Source metadata changed during inference')
    for row, _ in selected:
        for prefix in ('audioset', 'fsd'):
            if file_sha256(row[prefix+'_path']) != row[prefix+'_sha256']:
                raise ValueError('Selected source changed during inference')
    report = dict(status='passed', epoch=epoch, samples_per_experiment=args.count,
        distinct_mids=len({r['MID'] for r, _ in selected}), wav_count=6*args.count,
        nonoverlap_pairs=sum(r['strong_overlap_seconds'] == 0 for r, _ in selected),
        full_margin_nonoverlap_pairs=sum(r['margin_overlap_seconds'] == 0 for r, _ in selected),
        fsd_multilabel_pairs=sum(r['fsd_multilabel'] for r, _ in selected),
        max_stem_sum_error=max(r['stem_sum_max_error'] for r, _ in selected),
        wav_roundtrip_max_error=0.,
        peak_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3 if device.type == 'cuda' else None)
    metadata['status'] = 'passed'
    json_write(output/'manifest.json', metadata)
    json_write(output/'report.json', report)
    print(json.dumps(report, indent=2), flush=True)


def self_test():
    events = [dict(mid='x', present=True, onset=a, offset=b) for a, b in [(1, 2), (1.5, 3), (5, 6)]]
    intervals = positive_intervals(events, 'x')
    assert intervals == [[1., 3.], [5., 6.]]
    assert overlap_seconds(3., 5., intervals) == 0
    assert abs(overlap_seconds(3., 5., intervals, .1)-.2) < 1e-8
    assert overlap_seconds(0., 10., intervals) == 3
    t = np.arange(160000)/16000
    background = (.1*np.sin(2*np.pi*271*t)).astype('float32')
    fsd = (.2*np.sin(2*np.pi*713*t[:32000])).astype('float32')
    kwargs = dict(background_id='test_0', fsd_id='1', target_mid='x', seed=7,
                  background_weak_mids=[], fsd_mids=['x'], snr_db=-2)
    a = make_snr_mixture(background, fsd, **kwargs)
    b = make_snr_mixture(background, fsd, **kwargs)
    for x, y in zip(a[:3], b[:3]):
        np.testing.assert_array_equal(x, y)
    assert np.max(np.abs(a[2]-a[0]-a[1])) < 1e-6
    assert abs(a[3]['snr_db_exported']+2) < 1e-4
    quiet = make_snr_mixture(background, fsd*1e-6, **kwargs)
    assert quiet[3]['rms_fallback'] != 'none'
    with tempfile.TemporaryDirectory() as directory:
        for i, wave in enumerate(a[:3]):
            assert export_wav(Path(directory)/f'{i}.wav', wave) == 0
    print('SELF_TEST_PASSED: interval union/overlap, deterministic mixer, stem sum, SNR, quiet fallback, FLOAT WAV roundtrip')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir')
    parser.add_argument('--output', default='experiments/inference/tse_inference_tests')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--gpu-memory-fraction', type=float, default=.18)
    parser.add_argument('--max-attempts-per-mid', type=int, default=128)
    parser.add_argument('--margin-seconds', type=float, default=.1)
    parser.add_argument('--min-active-seconds', type=float, default=.1)
    parser.add_argument('--min-target-rms', type=float, default=1e-4)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.run_dir:
        parser.error('--run-dir is required')
    if min(args.count, args.cpu_threads, args.max_attempts_per_mid) < 1:
        parser.error('Counts must be positive')
    if not 0 < args.gpu_memory_fraction <= 1:
        parser.error('GPU memory fraction must be in (0,1]')
    if not all(np.isfinite(v) and v >= 0 for v in
               (args.margin_seconds, args.min_active_seconds, args.min_target_rms)):
        parser.error('Thresholds must be finite and nonnegative')
    run(args)


if __name__ == '__main__':
    main()
