"""Recorded-best TSE listening tests: same/different MID, nonoverlap then overlap.

Reuses make_dual_mixture unchanged. Overlap replays the SAME crops/gains through
mix_rms, changes only placement, and jointly normalizes background and both stems.
SDR in this experiment means the training zero-mean SI-SDR (= SI-SNR), not BSS Eval.
"""
import argparse
from argparse import Namespace
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from datasets.active_rms import active_rms, mix_rms
from datasets.dual_fsd import make_dual_mixture
from datasets.dual_fsd_crop import POLICY, replay_conditional_crop
from datasets.fsd50k_cache import file_sha256
from datasets.mix_utils import compute_rms, prepare_fsd_crop, read_mono_waveform, sample_snr_db
from datasets.paths.audioset_paths import read_strong_labels, read_weak_labels, resolve_paths
from loss_functions.tse_loss import si_sdr
from tools.export_tse_audio_compare import best_checkpoint, check_wave, export_wav, write_csv
from tools.infer_tse_source_tests import json_write, load_inventory, read_fsd, wave_hash

N, SR = 160000, 16000


def render(background, crops, starts):
    stems = []
    for crop, start in zip(crops, starts):
        if not 0 <= start <= N-len(crop):
            raise ValueError('Invalid placement')
        stem = np.zeros(N, dtype=np.float32)
        stem[start:start+len(crop)] = crop
        stems.append(stem)
    mixed = background + stems[0] + stems[1]
    divisor = max(1., float(np.abs(mixed).max()))
    waves = dict(audioset_background=background/divisor, fsd_a_gt=stems[0]/divisor,
                 fsd_b_gt=stems[1]/divisor, mixture=mixed/divisor)
    waves['sum_gt'] = waves['fsd_a_gt'] + waves['fsd_b_gt']
    for wave in waves.values():
        check_wave(wave)
    error = float(np.abs(waves['mixture']-(waves['audioset_background']+waves['sum_gt'])).max())
    if error > 1e-6:
        raise ValueError('Joint mixture/stem inconsistency')
    return waves, divisor, error


def replay_sources(background, raw, meta, hp):
    crops, masks, diagnostics = [], [], []
    for wave, event in zip(raw, meta['events']):
        if event['crop_sampling'] == POLICY:
            crop, geometry = replay_conditional_crop(wave, event)
        else:
            crop, geometry, _ = prepare_fsd_crop(wave, rng=event['crop_seed'])
        if any(event[k] != v for k, v in geometry.items()):
            raise ValueError('Original dual-crop geometry changed')
        _, mask, diag = active_rms(crop, threshold_dbfs=hp.mix_active_threshold_dbfs,
                                  frame_ms=hp.mix_active_frame_ms)
        bg, padded, _, scale = mix_rms(background, crop, normalization='none', rng=0,
            snr_db=meta['snr_db'], threshold_dbfs=hp.mix_active_threshold_dbfs,
            frame_ms=hp.mix_active_frame_ms)
        if scale['target_gain'] != event['target_gain']:
            raise ValueError('Replayed source gain changed')
        crops.append(padded[scale['target_start_sample']:scale['target_end_sample']].copy())
        masks.append(mask)
        diagnostics.append(diag)
    return bg, crops, masks, diagnostics


def placement_stats(starts, lengths, masks):
    regions = [[int(s), int(s+l)] for s, l in zip(starts, lengths)]
    overlap = max(0, min(r[1] for r in regions)-max(r[0] for r in regions))
    active = []
    for s, mask in zip(starts, masks):
        full = np.zeros(N, dtype=bool)
        full[s:s+len(mask)] = mask
        active.append(full)
    return dict(intervals_seconds=[[a/SR, b/SR] for a, b in regions],
                starts_samples=[int(s) for s in starts],
                overlap_seconds=overlap/SR,
                overlap_fraction_shorter=overlap/min(lengths),
                simultaneous_active_seconds=int(np.count_nonzero(active[0] & active[1]))/SR)


def overlap_placement(lengths, masks, seed, minimum_active=.05):
    """At least half of the shorter crop overlaps; prefer real active overlap."""
    rng = np.random.default_rng(seed)
    required = max(1, int(np.ceil(min(lengths)*.5)))
    best = None
    for trial in range(128):
        delta = (lengths[0]-required if trial == 0 else
                 int(rng.integers(required-lengths[1], lengths[0]-required+1)))
        left, right = min(0, delta), max(lengths[0], delta+lengths[1])
        leading = int(rng.integers(0, N-(right-left)+1))
        starts = [leading-left, leading+delta-left]
        stats = placement_stats(starts, lengths, masks)
        if best is None or stats['simultaneous_active_seconds'] > best[1]['simultaneous_active_seconds']:
            best = starts, stats
        if stats['simultaneous_active_seconds'] >= minimum_active:
            return starts, stats
    return best if best[1]['simultaneous_active_seconds'] >= minimum_active else None


def variants(background, raw, seed, snr, hp):
    bg0, stems0, mixture0, meta = make_dual_mixture(background, raw, seed=seed,
        gap_samples=512, max_crop_attempts=32, snr_db=snr,
        threshold_dbfs=hp.mix_active_threshold_dbfs, frame_ms=hp.mix_active_frame_ms)
    bg, crops, masks, diagnostics = replay_sources(background, raw, meta, hp)
    if any(d['rms_fallback'] != 'none' or d['active_sample_count']/SR < .1 for d in diagnostics):
        return None, 'near_silent_crop'
    starts = [e['target_start_sample'] for e in meta['events']]
    non, divisor, error = render(bg, crops, starts)
    # Exact equality checks the full original mixing order, not just stem addition.
    for expected, actual in zip([bg0, *stems0, mixture0],
                               [non[k] for k in ('audioset_background', 'fsd_a_gt', 'fsd_b_gt', 'mixture')]):
        np.testing.assert_array_equal(actual, expected)
    if divisor != meta['final_normalization_divisor']:
        raise ValueError('Original joint divisor changed')
    lengths = [len(c) for c in crops]
    non_stats = placement_stats(starts, lengths, masks)
    if non_stats['overlap_seconds'] != 0:
        raise ValueError('Legacy dual mixer unexpectedly overlaps')
    placement = overlap_placement(lengths, masks, seed)
    if placement is None:
        return None, 'no_meaningful_active_overlap'
    over_starts, over_stats = placement
    over, over_divisor, over_error = render(bg, crops, over_starts)
    result = {}
    for name, waves, div, err, stats in [('nonoverlap', non, divisor, error, non_stats),
                                       ('overlap', over, over_divisor, over_error, over_stats)]:
        if any(compute_rms(waves[k]) < 1e-4 for k in ('fsd_a_gt', 'fsd_b_gt', 'sum_gt')):
            return None, 'near_silent_export'
        actual_snr = []
        for k, start, mask in zip(('fsd_a_gt', 'fsd_b_gt'), stats['starts_samples'], masks):
            measured = 20*np.log10(compute_rms(waves[k][start:start+len(mask)][mask])
                                   /compute_rms(waves['audioset_background']))
            if abs(measured-snr) > 1e-4:
                raise ValueError('Per-source active SNR changed')
            actual_snr.append(float(measured))
        result[name] = (waves, dict(**stats, joint_divisor=div, stem_sum_max_error=err,
            actual_active_snr_db=actual_snr, waveform_sha256={k: wave_hash(w) for k, w in waves.items()}))
    return (result, meta), None


def candidate_inventory(hp, run, epoch):
    mapping, names, _, by_mid, inventory = load_inventory(hp, run, epoch)
    cv = json.loads(Path(inventory['inputs']['cv_manifest']['path']).read_text())
    strong = read_strong_labels(inventory['inputs']['strong_labels']['path'])
    weak = defaultdict(set)
    for i, p in enumerate(resolve_paths(hp.mix_weak_csvs, hp.data_path)):
        inventory['inputs'][f'weak_labels_{i}'] = dict(path=p, sha256=file_sha256(p))
        for row in read_weak_labels(p):
            weak[row['id']].update(row['weak_mids'])
    backgrounds = []
    for row in cv['rows']:
        identity = row['background_id']
        if identity not in weak:
            raise ValueError('CV background has no weak-label metadata')
        banned = weak[identity] | {e['mid'] for e in strong.get(identity, []) if e['present']}
        backgrounds.append(dict(**row, forbidden_mids=sorted(banned)))
    return mapping, names, by_mid, backgrounds, inventory


def select_experiments(hp, by_mid, backgrounds, count, seed):
    rng = np.random.default_rng(seed)
    used_fsd, used_bg = set(), set()
    wave_cache, bg_cache, source_hashes = {}, {}, {}
    excluded = Counter()
    selected = {}
    for experiment in ('A', 'B'):
        rows, used_mids = [], set()
        mids = [str(m) for m in rng.permutation(sorted(by_mid))]
        # Prioritize classes for which single-label pairs/individuals exist.
        mids.sort(key=lambda m: sum(len(s['original_mids']) == 1 for s in by_mid[m]) < (2 if experiment == 'A' else 1))
        for mid_a in mids:
            if len(rows) == count:
                break
            if mid_a in used_mids:
                continue
            for attempt in range(48):
                others = [m for m in mids if m != mid_a]
                others.sort(key=lambda m: m in used_mids)
                if experiment == 'A':
                    mid_b = mid_a
                else:
                    mid_b = others[attempt % len(others)]
                pools = []
                for mid in (mid_a, mid_b):
                    available = [s for s in by_mid[mid] if s['fsd_id'] not in used_fsd]
                    single = [s for s in available if len(s['original_mids']) == 1]
                    pools.append(single if len(single) >= (2 if experiment == 'A' else 1) else available)
                if not all(pools):
                    continue
                a = pools[0][int(rng.integers(len(pools[0])))]
                legal = [b for b in pools[1] if b['fsd_id'] != a['fsd_id'] and (
                    experiment == 'A' or not set(a['original_mids']) & set(b['original_mids']))]
                if not legal:
                    excluded['no_legal_fsd_pair'] += 1
                    continue
                b = legal[int(rng.integers(len(legal)))]
                bg_pool = [e for e in backgrounds if e['background_id'] not in used_bg
                           and not {mid_a, mid_b} & set(e['forbidden_mids'])]
                if not bg_pool:
                    excluded['no_legal_background'] += 1
                    continue
                bg_entry = bg_pool[int(rng.integers(len(bg_pool)))]
                identity = bg_entry['background_id']
                if identity not in bg_cache:
                    if file_sha256(bg_entry['background_path']) != bg_entry['background_sha256']:
                        raise ValueError('CV background audio changed')
                    bg_cache[identity] = read_mono_waveform(bg_entry['background_path'])[0]
                    check_wave(bg_cache[identity])
                raw = []
                for entry in (a, b):
                    if entry['fsd_id'] not in wave_cache:
                        source_hashes[entry['fsd_id']] = file_sha256(entry['audio_path'])
                        wave_cache[entry['fsd_id']] = read_fsd(entry['audio_path'])
                    raw.append(wave_cache[entry['fsd_id']])
                if (source_hashes[a['fsd_id']] == source_hashes[b['fsd_id']]
                        or (raw[0].shape == raw[1].shape and np.array_equal(raw[0], raw[1]))):
                    excluded['duplicate_fsd_content'] += 1
                    continue
                if any(compute_rms(w) == 0 for w in raw):
                    excluded['zero_fsd_source'] += 1
                    continue
                pair_seed = int(rng.integers(0, 2**63-1))
                snr = (float(hp.mix_snr_db) if hp.mix_snr_db is not None else
                       sample_snr_db(hp.mix_snr_db_range, sample_id=identity, seed=pair_seed))
                prepared, reason = variants(bg_cache[identity], raw, pair_seed, snr, hp)
                if prepared is None:
                    excluded[reason] += 1
                    continue
                versions, meta = prepared
                record = dict(sample_id=f'sample_{len(rows)+1:03d}', experiment=experiment,
                    mid_a=mid_a, mid_b=mid_b, background=bg_entry,
                    sources=[dict(**s, sha256=source_hashes[s['fsd_id']]) for s in (a, b)],
                    pair_seed=pair_seed, mixing=meta,
                    placements={k: v[1] for k, v in versions.items()})
                rows.append((record, versions))
                used_bg.add(identity); used_fsd.update([a['fsd_id'], b['fsd_id']])
                used_mids.update([mid_a, mid_b])
                print(f'Selected {experiment} {len(rows)}/{count}: {mid_a}, {mid_b}', flush=True)
                break
        if len(rows) != count:
            raise ValueError(f'{experiment}: only {len(rows)}/{count} distinct-MID groups found; {dict(excluded)}')
        selected[experiment] = rows
    return selected, dict(excluded)


def score(prediction, target, silence_rms):
    value, valid = si_sdr(torch.from_numpy(prediction).unsqueeze(0),
                         torch.from_numpy(target).unsqueeze(0), silence_rms)
    if not bool(valid[0]) or not torch.isfinite(value).all():
        raise ValueError('Invalid SI-SNR reference/score')
    return float(value[0])


def metrics(experiment, mix, a, b, predictions, silence_rms):
    measure = lambda x, y: score(x, y, silence_rms)
    if experiment == 'A':
        target = a+b
        baseline, result = measure(mix, target), measure(predictions[0], target)
        return dict(SI_SNR_sum_dB=result, mixture_SI_SNR_sum_dB=baseline,
                    SI_SNRi_sum_dB=result-baseline,
                    prediction_vs_A_dB=measure(predictions[0], a),
                    prediction_vs_B_dB=measure(predictions[0], b))
    pa, pb = predictions
    aa, ab, ba, bb = measure(pa, a), measure(pa, b), measure(pb, a), measure(pb, b)
    ma, mb = measure(mix, a), measure(mix, b)
    return dict(SI_SNR_A_dB=aa, SI_SNR_B_dB=bb, SI_SNRi_A_dB=aa-ma, SI_SNRi_B_dB=bb-mb,
                mixture_SI_SNR_A_dB=ma, mixture_SI_SNR_B_dB=mb,
                cross_SI_SNR_predA_gtB_dB=ab, cross_SI_SNR_predB_gtA_dB=ba,
                selectivity_A_dB=aa-ab, selectivity_B_dB=bb-ba,
                prediction_difference_rms=float(np.sqrt(np.mean((pa.astype('float64')-pb)**2))),
                predictions_identical=bool(np.array_equal(pa, pb)))


def run(args):
    from tse_separator import TSESourceSeparator
    output, run_dir = Path(args.output).resolve(), Path(args.run_dir).resolve()
    if output.exists():
        raise ValueError('Output exists; choose a new directory to avoid overwrite')
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint, saved, best_loss = best_checkpoint(run_dir)
    epoch = int(saved['epoch']) + 1
    hp = Namespace(**saved['hyper_parameters'])
    print(f'Epoch {epoch} best checkpoint loaded, CV loss={best_loss:.6f}', flush=True)
    mapping, names, by_mid, backgrounds, inventory = candidate_inventory(hp, run_dir, epoch)
    selected, excluded = select_experiments(hp, by_mid, backgrounds, args.count, args.seed)
    device = torch.device('cuda:0' if args.device == 'cuda' else 'cpu')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA unavailable')
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device)
    model = TSESourceSeparator(hp)
    model.on_load_checkpoint(saved)
    model.load_state_dict(saved['state_dict'], strict=True)
    del saved
    model.to(device).eval()
    if any(m.training for m in model.modules()):
        raise ValueError('Model must be in eval mode')

    @torch.inference_mode()
    def predict(x, mid):
        condition = torch.zeros(1, hp.num_classes, device=device)
        condition[0, mapping[mid]] = 1.
        result = model(dict(mixture_waveform=x, condition=condition))['target_waveform']
        if result.requires_grad:
            raise ValueError('Unexpected inference gradient graph')
        wave = result[0].cpu().numpy()
        check_wave(wave)
        return wave

    output.mkdir(parents=True, exist_ok=False)
    metadata = dict(status='running', epoch=epoch, checkpoint=str(checkpoint),
        checkpoint_sha256=file_sha256(checkpoint), seed=args.seed, arguments=vars(args),
        inventory=inventory, excluded_candidates=excluded,
        samples={k: [r for r, _ in v] for k, v in selected.items()},
        metric='Training zero-mean SI-SDR / SI-SNR, dB; SDRi uses the same mixture and GT.',
        notes=['Same source IDs, crops, and pre-peak gains across placement variants.',
               'Known exact target MIDs absent from background Weak and positive Strong labels.',
               'Experiment B additionally requires disjoint original FSD label sets.',
               'Prefer single-label FSD; whole FSD stems remain the waveform GT.',
               'Reject near-silent crops; overlap requires >=50% shorter crop and >=50 ms active overlap.',
               'No source or prediction is independently normalized; no PIT or source reassignment.',
               'A individual-stem scores are diagnostic, not its joint-target objective.'])
    source_files = [Path(__file__), Path('datasets/dual_fsd.py'), Path('datasets/dual_fsd_crop.py'),
                    Path('datasets/active_rms.py'), Path('datasets/mix_utils.py'), Path('loss_functions/tse_loss.py')]
    root = Path(__file__).resolve().parents[1]
    metadata['code_sha256'] = {str(p): file_sha256(p if p.is_absolute() else root/p) for p in source_files}
    json_write(output/'manifest.json', metadata)
    results = {}
    # Finish all nonoverlap inference before starting overlap inference.
    for placement in ('nonoverlap', 'overlap'):
        for experiment in ('A', 'B'):
            folder = output/placement/f'experiment_{experiment}'
            folder.mkdir(parents=True)
            rows = []
            for record, versions in selected[experiment]:
                waves, stats = versions[placement]
                mix = waves['mixture']
                # Both B calls receive the very same tensor; only Condition changes.
                x = torch.from_numpy(mix).unsqueeze(0).to(device)
                original_input = x.clone()
                predictions = [predict(x, record['mid_a'])]
                if experiment == 'B':
                    predictions.append(predict(x, record['mid_b']))
                if not torch.equal(x, original_input) or wave_hash(mix) != stats['waveform_sha256']['mixture']:
                    raise ValueError('Condition comparison changed the mixture')
                calculated = metrics(experiment, mix, waves['fsd_a_gt'], waves['fsd_b_gt'],
                                     predictions, hp.tse_silence_rms)
                exports = {k: waves[k] for k in ('audioset_background', 'mixture', 'fsd_a_gt', 'fsd_b_gt')}
                if experiment == 'A':
                    exports.update(sum_gt=waves['sum_gt'], prediction=predictions[0])
                else:
                    exports.update(prediction_condition_a=predictions[0], prediction_condition_b=predictions[1])
                sample = folder/record['sample_id']
                sample.mkdir()
                for name, wave in exports.items():
                    export_wav(sample/f'{name}.wav', wave)
                row = dict(Sample_ID=record['sample_id'], placement=placement,
                    audioset_id=record['background']['background_id'],
                    MID_A=record['mid_a'], class_A=names[mapping[record['mid_a']]],
                    MID_B=record['mid_b'], class_B=names[mapping[record['mid_b']]],
                    condition_index_A=mapping[record['mid_a']], condition_index_B=mapping[record['mid_b']],
                    fsd_A_id=record['sources'][0]['fsd_id'], fsd_B_id=record['sources'][1]['fsd_id'],
                    fsd_A_mids=json.dumps(record['sources'][0]['original_mids']),
                    fsd_B_mids=json.dumps(record['sources'][1]['original_mids']),
                    fsd_A_single_label=len(record['sources'][0]['original_mids']) == 1,
                    fsd_B_single_label=len(record['sources'][1]['original_mids']) == 1,
                    A_start_seconds=stats['intervals_seconds'][0][0], A_end_seconds=stats['intervals_seconds'][0][1],
                    B_start_seconds=stats['intervals_seconds'][1][0], B_end_seconds=stats['intervals_seconds'][1][1],
                    overlap_seconds=stats['overlap_seconds'], simultaneous_active_seconds=stats['simultaneous_active_seconds'],
                    requested_active_snr_db=record['mixing']['snr_db'],
                    actual_active_snr_A_db=stats['actual_active_snr_db'][0],
                    actual_active_snr_B_db=stats['actual_active_snr_db'][1],
                    common_peak_divisor=stats['joint_divisor'], background_known_mid_conflict=False,
                    mixture_sha256=wave_hash(mix), **calculated)
                write_csv(sample/'metrics.csv', [row])
                rows.append(row)
                print(f'{placement} {experiment} {len(rows)}/{args.count} exported', flush=True)
            write_csv(folder/'summary.csv', rows)
            keys = ['SI_SNR_sum_dB', 'SI_SNRi_sum_dB'] if experiment == 'A' else [
                'SI_SNR_A_dB', 'SI_SNR_B_dB', 'SI_SNRi_A_dB', 'SI_SNRi_B_dB',
                'cross_SI_SNR_predA_gtB_dB', 'cross_SI_SNR_predB_gtA_dB', 'selectivity_A_dB', 'selectivity_B_dB']
            results[f'{placement}/{experiment}'] = dict(samples=len(rows),
                distinct_mids=len({r[k] for r in rows for k in ('MID_A', 'MID_B')}),
                single_label_sources=sum(r['fsd_A_single_label']+r['fsd_B_single_label'] for r in rows),
                mean_metrics={k: float(np.mean([r[k] for r in rows])) for k in keys})
            if experiment == 'B':
                results[f'{placement}/{experiment}']['both_conditions_prefer_correct_gt'] = sum(
                    r['selectivity_A_dB'] > 0 and r['selectivity_B_dB'] > 0 for r in rows)
    for item in inventory['inputs'].values():
        if file_sha256(item['path']) != item['sha256']:
            raise ValueError('Inventory changed during inference')
    for records in selected.values():
        for r, _ in records:
            if file_sha256(r['background']['background_path']) != r['background']['background_sha256']:
                raise ValueError('AudioSet source changed')
            for source in r['sources']:
                if file_sha256(source['audio_path']) != source['sha256']:
                    raise ValueError('FSD source changed')
    metadata['status'] = 'passed'
    json_write(output/'manifest.json', metadata)
    report = dict(status='passed', epoch=epoch, results=results, wav_count=args.count*24,
        wav_roundtrip_max_error=0., max_stem_sum_error=max(
            v[1]['stem_sum_max_error'] for records in selected.values() for _, versions in records for v in versions.values()),
        peak_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3 if device.type == 'cuda' else None)
    json_write(output/'report.json', report)
    print(json.dumps(report, indent=2), flush=True)


def self_test():
    hp = Namespace(mix_active_threshold_dbfs=-40., mix_active_frame_ms=20.)
    t = np.arange(N)/SR
    bg = (.1*np.sin(2*np.pi*237*t)).astype('float32')
    raw = [(.2*np.sin(2*np.pi*f*t[:length])).astype('float32')
           for f, length in [(503, 32000), (907, 40000)]]
    first, reason = variants(bg, raw, 7, -2., hp)
    assert reason is None
    second, _ = variants(bg, raw, 7, -2., hp)
    for placement in ('nonoverlap', 'overlap'):
        waves, stats = first[0][placement]
        for k, wave in waves.items():
            np.testing.assert_array_equal(wave, second[0][placement][0][k])
        assert stats['overlap_seconds'] == 0 if placement == 'nonoverlap' else stats['simultaneous_active_seconds'] >= .05
        a, b = waves['fsd_a_gt'], waves['fsd_b_gt']
        correct = metrics('B', waves['mixture'], a, b, [a, b], 1e-5)
        wrong = metrics('B', waves['mixture'], a, b, [b, a], 1e-5)
        assert correct['selectivity_A_dB'] > 0 and correct['selectivity_B_dB'] > 0
        assert wrong['selectivity_A_dB'] < 0 and wrong['selectivity_B_dB'] < 0
        assert metrics('A', waves['mixture'], a, b, [a+b], 1e-5)['SI_SNR_sum_dB'] > 70
        with tempfile.TemporaryDirectory() as tmp:
            for k, wave in waves.items():
                assert export_wav(Path(tmp)/(k+'.wav'), wave) == 0
    # Force the finite conditional crop sampler and verify replay through it too.
    long = [np.tile(w, 16) for w in raw]
    _, _, _, meta = make_dual_mixture(bg, long, seed=7, max_crop_attempts=1)
    assert meta['crop_sampling'] == POLICY
    rbg, crops, masks, _ = replay_sources(bg, long, meta, hp)
    replayed, _, _ = render(rbg, crops, [e['target_start_sample'] for e in meta['events']])
    np.testing.assert_array_equal(replayed['mixture'], make_dual_mixture(bg, long, seed=7, max_crop_attempts=1)[2])
    print('SELF_TEST_PASSED: legacy/conditional replay, deterministic placement, active overlap, joint gains, WAV roundtrip, correct/swapped conditions, sum GT')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir')
    parser.add_argument('--output', default='experiments/inference/tse_dual_fsd_tests')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--gpu-memory-fraction', type=float, default=.18)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.run_dir:
        parser.error('--run-dir is required')
    if min(args.count, args.cpu_threads) < 1 or not 0 < args.gpu_memory_fraction <= 1:
        parser.error('Invalid count, thread count or GPU memory fraction')
    run(args)


if __name__ == '__main__':
    main()
