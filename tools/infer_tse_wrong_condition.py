"""Fixed-CV correct vs three wrong MID conditions; no training or ontology file.

Replays original mixing seeds and validates every stem's recorded sample hash.
Metrics use the training si_sdr implementation (zero-mean SI-SNR terminology).
Near-silent predictions retain raw scores for audit but are excluded from usable
SI-SNR/difference/correlation aggregates; actual output energy is always reported.
"""
import argparse
from argparse import Namespace
from collections import Counter
import json
from pathlib import Path
import re
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from datasets.active_rms import make_snr_mixture
from datasets.audioset_fsd50k_mix_dataset import GEOMETRY_KEYS, digest
from datasets.fsd50k_cache import file_sha256
from datasets.mix_utils import compute_rms, read_mono_waveform
from datasets.paths.audioset_paths import read_strong_labels, read_weak_labels, resolve_paths
from loss_functions.tse_loss import si_sdr
from tools.export_tse_audio_compare import best_checkpoint, check_wave, export_wav, write_csv
from tools.infer_tse_source_tests import json_write, load_inventory, read_fsd, wave_hash


# Conservative name heuristics only: not a taxonomy or proof of semantic distance.
NAME_HINTS = {
    'voice_body': 'speech speak talk whisper shout scream yell sing vocal voice laugh chuckle giggle cry sob baby child cough sneeze snore snort breath burp eructation hiccup fart chew mastication human',
    'animal': 'animal bird chirp tweet squawk crow rooster chicken cluck duck goose owl hoot pigeon dove coo dog bark woof howl growl whimper cat meow purr horse neigh cow moo sheep goat bleat pig oink frog croak insect cricket mosquito buzz bee fly lion roar',
    'music': 'music instrument guitar piano keyboard organ violin fiddle cello flute clarinet saxophone trumpet trombone brass bass drum percussion cymbal gong bell chime marimba xylophone accordion harmonica harp pluck strum mallet',
    'vehicle_machine': 'vehicle car truck bus train railway rail motor engine aircraft airplane helicopter propeller boat ship traffic motorcycle race skid brake gear machine mechanism printer power tool drill saw chainsaw lawnmower ratchet jackhammer electric shaver',
    'water_nature': 'water liquid stream river ocean sea wave splash splatter pour drip trickle gurgle bubble sink wash flush rain thunder storm wind fire flame crackle',
    'impact_surface': 'impact knock tap thump thud bang slam slap smack pop gunshot gun firearm explosion burst crack crushing crush crumple crinkle rustle tear rip roll scrape scratch rub sand cut chop bounce clatter rattle glass shatter smash break wood hammer contact',
    'electronic': 'electronic beep bleep alarm siren telephone phone ringtone dial buzzer computer electric static',
    'noise_texture': 'noise hiss static hum buzz squeak squeal whoosh swoosh swish rumble whirr whiz',
}


def name_features(name):
    tokens = set(re.findall(r'[a-z]+', name.lower())) - {'sound', 'sounds', 'and', 'or', 'of', 'with', 'other', 'the', 'a', 'an'}
    families = {family for family, hints in NAME_HINTS.items()
                if any(token.startswith(hint) for hint in hints.split() for token in tokens)}
    return tokens, families


def choose_wrong(allowed, true_mids, background_mids, mapping, names, rng):
    banned = set(true_mids) | set(background_mids)
    tokens, families = set(), set()
    for mid in true_mids:
        if mid in mapping:
            t, f = name_features(names[mapping[mid]])
            tokens |= t; families |= f
    pool = []
    for mid in sorted(set(allowed)-banned):
        t, f = name_features(names[mapping[mid]])
        if not tokens & t and not families & f:
            pool.append(mid)
    if len(pool) < 3:
        return None
    wrong = [str(m) for m in rng.choice(pool, size=3, replace=False)]
    return wrong, dict(eligible_wrong_count=len(pool), excluded_name_families=sorted(families),
                       semantic_policy='disjoint name tokens and broad keyword families; heuristic only; no ontology')


def replay(row, source, background_mids, hp):
    for path, expected in [(row['background_path'], row['background_sha256']),
                           (row['fsd_path'], row['fsd_sha256'])]:
        if file_sha256(path) != expected:
            raise ValueError('CV audio content changed: '+path)
    background = read_mono_waveform(row['background_path'])[0]
    raw = read_fsd(row['fsd_path'])
    bg, target, mixture, meta = make_snr_mixture(background, raw,
        background_id=row['background_id'], fsd_id=row['fsd_id'], target_mid=row['target_mid'],
        seed=row['seed'], background_weak_mids=background_mids, fsd_mids=source['mids'],
        snr_db=row['snr_db'], normalization='peak', threshold_dbfs=hp.mix_active_threshold_dbfs,
        frame_ms=hp.mix_active_frame_ms)
    if {k: meta[k] for k in GEOMETRY_KEYS} != row['geometry']:
        raise ValueError('Fixed CV crop/placement changed')
    for key in ('target_gain', 'final_normalization_divisor'):
        if meta[key] != row[key]:
            raise ValueError('Fixed CV gain changed')
    for name, wave in zip(('background', 'target', 'mixture'), (bg, target, mixture)):
        check_wave(wave)
        if wave_hash(wave) != row['waveform_sha256'][name]:
            raise ValueError('Fixed CV waveform does not match recorded hash: '+name)
    if np.max(np.abs(mixture-bg-target)) > 1e-6:
        raise ValueError('Mixture/stem inconsistency')
    return mixture, target


def select_samples(hp, run_dir, count, seed, epoch):
    mapping, names, _, by_mid, inventory = load_inventory(hp, run_dir, epoch)
    manifest = json.loads(Path(inventory['inputs']['cv_manifest']['path']).read_text())
    if digest(manifest['rows']) != manifest['rows_sha256']:
        raise ValueError('Fixed CV manifest integrity check failed')
    by_id = {e['fsd_id']: e for pool in by_mid.values() for e in pool}
    # Only the original single-FSD CV rows are used. No dual-FSD code is called.
    strong = read_strong_labels(inventory['inputs']['strong_labels']['path'])
    background_mids = {}
    for i, path in enumerate(resolve_paths(hp.mix_weak_csvs, hp.data_path)):
        inventory['inputs'][f'weak_labels_{i}'] = dict(path=path, sha256=file_sha256(path))
        for entry in read_weak_labels(path):
            background_mids.setdefault(entry['id'], set()).update(entry['weak_mids'])
    for identity, mids in background_mids.items():
        mids.update(e['mid'] for e in strong.get(identity, []) if e['present'])
    rng = np.random.default_rng(seed)
    selected, used_mids, skipped = [], set(), Counter()
    for index in rng.permutation(len(manifest['rows'])):
        row = manifest['rows'][int(index)]
        mid = row['target_mid']
        if mid in used_mids or mid not in by_mid:
            continue
        source = by_id[row['fsd_id']]
        banned_bg = sorted(background_mids[row['background_id']])
        choice = choose_wrong(by_mid.keys(), source['original_mids'], banned_bg, mapping, names, rng)
        if choice is None:
            skipped['insufficient_semantically_distant_wrong_conditions'] += 1
            continue
        mixture, target = replay(row, source, banned_bg, hp)
        if compute_rms(target) <= hp.tse_silence_rms:
            skipped['invalid_silent_reference'] += 1
            continue
        wrong, policy = choice
        record = dict(sample_id=f'sample_{len(selected)+1:03d}', cv_index=int(index), cv_row=row,
            correct_mid=mid, correct_class=names[mapping[mid]], wrong_mids=wrong,
            wrong_classes=[names[mapping[m]] for m in wrong], fsd_original_mids=source['original_mids'],
            background_known_mids=banned_bg, selection_policy=policy,
            target_rms=compute_rms(target), reference_near_silent=compute_rms(target) < 1e-4)
        selected.append((record, mixture, target))
        used_mids.add(mid)
        print(f"{record['sample_id']}: {record['correct_class']} -> {', '.join(record['wrong_classes'])}", flush=True)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f'Only {len(selected)}/{count} unique correct MID samples found')
    return selected, mapping, names, inventory, dict(skipped)


def energy(wave, threshold):
    x = wave.astype(np.float64)
    mean_square = float(np.mean(x*x))
    rms = float(np.sqrt(mean_square))
    ac_rms = float(np.std(x))
    return dict(mean_square=mean_square, rms=rms, zero_mean_rms=ac_rms,
                rms_dbfs=float(20*np.log10(max(rms, 1e-12))), peak=float(np.abs(x).max()),
                near_silent=bool(ac_rms <= threshold), exact_zero=bool(not np.any(x)),
                dc_mean=float(x.mean()))


def measured_score(wave, target, reference_threshold, prediction_threshold):
    value, valid = si_sdr(torch.from_numpy(wave).unsqueeze(0),
                         torch.from_numpy(target).unsqueeze(0), reference_threshold)
    if not bool(valid[0]) or not torch.isfinite(value).all():
        raise ValueError('Invalid target/score')
    e = energy(wave, prediction_threshold)
    raw = float(value[0])
    return dict(SI_SNR_raw_dB=raw, SI_SNR_dB=None if e['near_silent'] else raw, **e)


def similarity(correct, wrong, ec, ew):
    c, w = correct.astype(np.float64), wrong.astype(np.float64)
    c0, w0 = c-c.mean(), w-w.mean()
    both_audible = not (ec['near_silent'] or ew['near_silent'])
    corr = float(np.clip(np.dot(c0, w0)/(np.linalg.norm(c0)*np.linalg.norm(w0)), -1, 1)) if both_audible else None
    return dict(pearson_correlation=corr, absolute_correlation=abs(corr) if corr is not None else None,
        waveform_difference_rms=float(np.sqrt(np.mean((c-w)**2))),
        relative_waveform_rmse=float(np.sqrt(np.mean((c-w)**2))/ec['rms']) if ec['rms'] > 1e-12 else None,
        wrong_to_correct_rms_db=float(20*np.log10(max(ew['rms'], 1e-12)/ec['rms'])) if ec['rms'] > 1e-12 else None,
        wrong_projection_gain=float(np.dot(c0, w0)/np.dot(c0, c0)) if not ec['near_silent'] else None,
        waveforms_identical=bool(np.array_equal(correct, wrong)))


def run(args):
    from tse_separator import TSESourceSeparator
    output, run_dir = Path(args.output).resolve(), Path(args.run_dir).resolve()
    if output.exists() and not args.preview_selection:
        raise ValueError('Output exists; choose another --output to avoid overwrite')
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint, saved, best_loss = best_checkpoint(run_dir)
    epoch = int(saved['epoch']) + 1
    hp = Namespace(**saved['hyper_parameters'])
    print(f'Best checkpoint: epoch {epoch}, CV loss={best_loss:.6f}', flush=True)
    selected, mapping, names, inventory, skipped = select_samples(hp, run_dir, args.count, args.seed, epoch)
    if args.preview_selection:
        print('SELECTION_AND_FIXED_CV_REPLAY_PASSED', flush=True)
        return
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
    def infer(x, mid):
        condition = torch.zeros(1, hp.num_classes, device=device)
        condition[0, mapping[mid]] = 1.
        pred = model(dict(mixture_waveform=x, condition=condition))['target_waveform']
        if pred.requires_grad:
            raise ValueError('Unexpected autograd graph')
        wave = pred[0].cpu().numpy()
        check_wave(wave)
        return wave

    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(status='running', epoch=epoch, checkpoint=str(checkpoint),
        checkpoint_sha256=file_sha256(checkpoint), arguments=vars(args), inventory=inventory,
        samples=[r for r, _, _ in selected], skipped_samples=skipped,
        metrics='Training zero-mean SI-SDR (= SI-SNR); all scores reference original FSD target.',
        near_silence_policy='Prediction zero-mean RMS <= configured threshold: usable score/difference/correlation null; raw training score retained. Energy always reported.',
        correlation_policy='Zero-lag signed Pearson correlation, without alignment or gain normalization of exported audio.',
        energy_policy='Raw waveform mean-square/RMS; dBFS floor -240 dB. Near-silent flag uses AC RMS; DC and total RMS are also reported.',
        semantic_policy='Name-token/keyword heuristic only; no ontology files.',
        no_absence_guarantee='Weak/Strong labels may omit sounds; wrong-class absence is not proven.')
    root = Path(__file__).resolve().parents[1]
    manifest['code_sha256'] = {str(p): file_sha256(root/p) for p in (
        'tools/infer_tse_wrong_condition.py', 'datasets/active_rms.py', 'datasets/mix_utils.py',
        'nets/tse_dprnntf.py', 'loss_functions/tse_loss.py')}
    json_write(output/'manifest.json', manifest)
    summaries, correct_scores = [], []
    for record, mixture, target in selected:
        directory = output/record['sample_id']
        directory.mkdir()
        x = torch.from_numpy(mixture).unsqueeze(0).to(device)
        original = x.clone()
        correct = infer(x, record['correct_mid'])
        c = measured_score(correct, target, hp.tse_silence_rms, args.prediction_silence_rms)
        correct_scores.append(c)
        for name, wave in [('mixture', mixture), ('target_gt', target), ('prediction_correct', correct)]:
            export_wav(directory/(name+'.wav'), wave)
        rows = []
        for i, wrong_mid in enumerate(record['wrong_mids'], 1):
            wrong = infer(x, wrong_mid)
            w = measured_score(wrong, target, hp.tse_silence_rms, args.prediction_silence_rms)
            compare = similarity(correct, wrong, c, w)
            difference = c['SI_SNR_dB']-w['SI_SNR_dB'] if c['SI_SNR_dB'] is not None and w['SI_SNR_dB'] is not None else None
            row = dict(Sample_ID=record['sample_id'], cv_index=record['cv_index'],
                audioset_id=record['cv_row']['background_id'], fsd_id=record['cv_row']['fsd_id'],
                correct_MID=record['correct_mid'], correct_class=record['correct_class'],
                wrong_number=i, wrong_MID=wrong_mid, wrong_class=names[mapping[wrong_mid]],
                fsd_all_mids=json.dumps(record['fsd_original_mids']), target_rms=record['target_rms'],
                reference_near_silent=record['reference_near_silent'],
                target_start_seconds=record['cv_row']['geometry']['target_start_sample']/16000,
                target_end_seconds=record['cv_row']['geometry']['target_end_sample']/16000,
                SI_SNR_difference_dB=difference,
                SI_SNR_difference_raw_dB=c['SI_SNR_raw_dB']-w['SI_SNR_raw_dB'],
                **{'correct_'+k: v for k, v in c.items()}, **{'wrong_'+k: v for k, v in w.items()},
                **compare, mixture_sha256=record['cv_row']['waveform_sha256']['mixture'])
            export_wav(directory/f'prediction_wrong_{i}.wav', wrong)
            rows.append(row)
        if not torch.equal(x, original) or wave_hash(mixture) != record['cv_row']['waveform_sha256']['mixture']:
            raise ValueError('Condition switch changed input mixture')
        write_csv(directory/'metrics.csv', rows)
        summaries.extend(rows)
        print(f"{record['sample_id']} exported: correct={c['SI_SNR_dB']}, wrong={[r['wrong_SI_SNR_dB'] for r in rows]}", flush=True)
    write_csv(output/'summary.csv', summaries)
    for item in inventory['inputs'].values():
        if file_sha256(item['path']) != item['sha256']:
            raise ValueError('Source metadata changed during inference')
    for record, _, _ in selected:
        r = record['cv_row']
        for prefix in ('background', 'fsd'):
            if file_sha256(r[prefix+'_path']) != r[prefix+'_sha256']:
                raise ValueError('Source audio changed during inference')
    def mean(values):
        values = [v for v in values if v is not None]
        return float(np.mean(values)) if values else None
    valid = [r for r in summaries if r['SI_SNR_difference_dB'] is not None]
    correlations = [r['pearson_correlation'] for r in summaries if r['pearson_correlation'] is not None]
    report = dict(status='passed', epoch=epoch, seed=args.seed, samples=len(selected), wrong_comparisons=len(summaries),
        wav_count=len(selected)*6, wav_roundtrip_max_error=0., fixed_cv_waveform_hashes_match=True,
        correct_mean_SI_SNR_dB=mean([r['SI_SNR_dB'] for r in correct_scores]),
        wrong_mean_SI_SNR_dB=mean([r['wrong_SI_SNR_dB'] for r in summaries]),
        mean_difference_dB=mean([r['SI_SNR_difference_dB'] for r in valid]),
        valid_score_comparisons=len(valid), correct_near_silent=sum(r['near_silent'] for r in correct_scores),
        wrong_near_silent=sum(r['wrong_near_silent'] for r in summaries),
        valid_correlation_comparisons=len(correlations), correlation_above_098=sum(v > .98 for v in correlations),
        mean_correlation=mean(correlations),
        mean_wrong_to_correct_rms_db=mean([r['wrong_to_correct_rms_db'] for r in summaries]),
        absolute_difference_below_1db=sum(abs(r['SI_SNR_difference_dB']) < 1 for r in valid),
        peak_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3 if device.type == 'cuda' else None)
    manifest['status'] = 'passed'
    json_write(output/'manifest.json', manifest)
    json_write(output/'report.json', report)
    print(json.dumps(report, indent=2), flush=True)


def self_test():
    target = np.sin(np.arange(160000)*.17).astype('float32')*.1
    correct = measured_score(target, target, 1e-5, 1e-5)
    quiet = measured_score(target*1e-7, target, 1e-5, 1e-5)
    zero = measured_score(np.zeros_like(target), target, 1e-5, 1e-5)
    assert correct['SI_SNR_dB'] > 70 and quiet['SI_SNR_dB'] is None and zero['SI_SNR_dB'] is None
    assert np.isfinite(quiet['SI_SNR_raw_dB'])  # Training's relative floor also affects tiny outputs.
    low_target = target*.01
    faint_match = measured_score(low_target*.001, low_target, 1e-5, 1e-5)
    assert faint_match['SI_SNR_raw_dB'] > 10 and faint_match['SI_SNR_dB'] is None
    assert similarity(target, target*1e-7, correct, quiet)['pearson_correlation'] is None
    half = measured_score(target*.5, target, 1e-5, 1e-5)
    comparison = similarity(target, target*.5, correct, half)
    assert abs(comparison['pearson_correlation']-1) < 1e-10
    assert abs(comparison['relative_waveform_rmse']-.5) < 1e-8
    mapping = dict(zip(['a','b','c','d','e','f','g','h'], range(8)))
    names = ['Dog', 'Bark', 'Cat', 'Speech', 'Piano', 'Rain', 'Engine', 'Knock']
    choice = choose_wrong(mapping, ['a','e'], ['g'], mapping, names, np.random.default_rng(7))
    assert choice is not None and set(choice[0]) == {'d','f','h'}
    with tempfile.TemporaryDirectory() as tmp:
        export_wav(Path(tmp)/'quiet.wav', target*1e-7)
    json.dumps([correct, quiet, zero, comparison], allow_nan=False)
    print('SELF_TEST_PASSED: exact/near silence, stable raw vs usable scores, correlation vs gain, label exclusions, FLOAT WAV roundtrip')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir')
    parser.add_argument('--output', default='experiments/inference/tse_wrong_condition')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--gpu-memory-fraction', type=float, default=.18)
    parser.add_argument('--prediction-silence-rms', type=float, default=1e-5)
    parser.add_argument('--preview-selection', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.run_dir:
        parser.error('--run-dir is required')
    if (min(args.count,args.cpu_threads) < 1 or not 0 < args.gpu_memory_fraction <= 1
            or not np.isfinite(args.prediction_silence_rms) or args.prediction_silence_rms <= 0):
        parser.error('Invalid count/thread/memory/silence setting')
    run(args)


if __name__ == '__main__':
    main()
