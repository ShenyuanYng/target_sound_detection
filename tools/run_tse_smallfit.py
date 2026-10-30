"""Bounded, isolated fitting check. Never calls the full dataset factory or Trainer."""
import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import random
import re
import socket
import sys
import subprocess
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import numpy as np
import soundfile as sf
import torch

from datasets.paths.audioset_paths import load_class_map, read_weak_labels, _resolve_audio, resolve_path
from datasets.tse_hierarchy_labels import AncestorReducer, training_vocabulary
from datasets.tse_hierarchy_dataset import TSEHierarchyDataset
from datasets.fsd50k_cache import read_cached_fsd, file_sha256
from datasets.mix_utils import read_mono_waveform
from loss_functions.tse_ri_loss import TSERILoss
from nets.tse_dprnntf import create_tse_model
from tools.tse_smallfit_checks import measure, metric_checks, loss_checks, write_json
from utils.config import yaml_to_parser
from utils.tse_config import validate_tse_config


def label_name(name):
    return re.sub(r'[^A-Za-z0-9_-]+', '_', name).strip('_')[:55]


def emit(event, **values):
    print(json.dumps(dict(event=event, **values), ensure_ascii=False, allow_nan=False), flush=True)


def prepare(hp, root, count, seed, background_split):
    if (root/'prepared_manifest.json').exists():
        raise ValueError('Use existing prepared data with audit/benchmark/fit; do not regenerate')
    mapped, names = load_class_map(resolve_path(hp.class_map_path, hp.data_path))
    reducer = AncestorReducer(json.loads((PROJECT/hp.tse_ontology_path).read_text()))
    pool = []
    with (Path(hp.mix_fsd_ground_truth)/'dev.csv').open(newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            if row['split'] != 'train':
                continue
            original = sorted(set(row['mids'].split(',')))
            mids = sorted(set(original)&mapped.keys())
            if mids:
                pool.append(dict(fsd_id=row['fname'], split='train', mids=mids, original_mids=original,
                    audio_path=str(Path(hp.mix_fsd_dev_audio)/(row['fname']+'.wav'))))
    pool.sort(key=lambda e:int(e['fsd_id']))
    official = json.loads(Path(resolve_path(hp.mix_fsd_split_path, hp.data_path)).read_text())
    assert [e['fsd_id'] for e in pool] == official['train']
    audit_path = Path(hp.data_path)/'mix_train_cv_seed7/fsd_audit_nonzero_v1/fsd_tr_audit.json'
    audit = json.loads(audit_path.read_text())
    assert audit['scanned'] and not audit['invalid_count'] and audit['source_count'] == len(pool)
    excluded = set(hp.mix_content_duplicate_exclusions['fsd50k']) | {e['fsd_id'] for e in audit['excluded']}
    vocabulary, positive, label_report = training_vocabulary(pool, reducer, mapped, excluded, 188)
    groups = defaultdict(list)
    for entry in positive:
        groups[entry['positive_mid']].append(entry)
    # A small, class-diverse TRAIN subset; no validation audio enters fitting.
    rng = np.random.default_rng(seed)
    mids = [m for m, entries in sorted(groups.items()) if len(entries) >= 20]
    rng.shuffle(mids)
    selected = []
    for mid in mids[:16]:
        selected.extend(groups[mid][int(i)] for i in rng.choice(len(groups[mid]), 2, replace=False))
    saved_split = json.loads(Path(background_split).read_text())
    background_ids = list(saved_split['tr'])
    rng.shuffle(background_ids)
    selected_ids = set(background_ids[:count//2+4])
    weak = read_weak_labels(resolve_path(hp.mix_weak_csvs[0], hp.data_path), keep_ids=selected_ids)
    events = defaultdict(list)
    with open(resolve_path(hp.mix_strong_tsv, hp.data_path), newline='') as f:
        for row in csv.reader(f, delimiter='\t'):
            if len(row)<4 or row[0] not in selected_ids:
                continue
            onset, offset = float(row[1]), float(row[2])
            assert 0 <= onset <= offset <= 10
            if onset == offset:
                continue
            present = len(row)<5 or row[4].strip().upper()=='PRESENT'
            events[row[0]].append(dict(onset=onset, offset=offset, mid=row[3].strip(), present=present))
    paths = []
    for entry in weak:
        audio = _resolve_audio(entry, hp.mix_audio_dirs)
        if audio is None:
            continue
        paths.append({**entry, 'audio_path':audio, 'original_weak_mids':entry['weak_mids'],
                      'events':events[entry['id']], 'strong_positive_mids':sorted(
                          {e['mid'] for e in events[entry['id']] if e['present']})})
    dataset = TSEHierarchyDataset(paths, selected, resolve_path(hp.class_map_path, hp.data_path),
        partition='tr', ontology_path=PROJECT/hp.tse_ontology_path, train_condition_mids=vocabulary,
        cache_dir=Path(hp.mix_cache_dir), cv_manifest_path=root/'unused_cv.json', seed=seed,
        snr_db=hp.mix_snr_db, snr_db_range=hp.mix_snr_db_range,
        threshold_dbfs=hp.mix_active_threshold_dbfs, frame_ms=hp.mix_active_frame_ms,
        no_pair_policy='exclude', max_pair_attempts=128, max_crop_attempts=128)
    assert len(dataset) >= count, 'Not enough legal small-set backgrounds'
    samples, gt_checks = [], []
    for index in range(count):
        bg, targets, mixture, meta = dataset._draw(index, seed*10000+index)
        sample_id = f'sample_{index:02d}'
        directory = root/'audio'/sample_id
        directory.mkdir(parents=True, exist_ok=False)
        mids = [e['target_mid'] for e in meta['events']]+[meta['wrong_mid']]
        labels = {key:dict(mid=mid, index=mapped[mid], name=names[mapped[mid]])
                  for key,mid in zip(('a','b','wrong'), mids)}
        files = {'mix':'mixture.wav', 'background':'background.wav'}
        files.update({k:f'gt_{k.upper()}_{label_name(labels[k]["name"])}.wav' for k in labels})
        arrays = dict(mix=mixture, background=bg, a=targets[0], b=targets[1], wrong=np.zeros_like(mixture))
        for key, waveform in arrays.items():
            sf.write(str(directory/files[key]), waveform, 16000, subtype='FLOAT')
        # Reconstruct target stems independently from recorded crop coordinates and gains.
        errors, source_zero_fractions = [], []
        for event, target in zip(meta['events'], targets):
            original, _ = read_cached_fsd(event['fsd_path'], Path(hp.mix_cache_dir))
            clean = original[original != 0]
            repeated = np.tile(clean, event['num_repeats'])
            crop = repeated[event['crop_start_sample']:event['crop_end_sample']]
            expected = np.zeros_like(mixture)
            start, end = event['target_start_sample'], event['target_end_sample']
            assert len(crop) == end-start
            expected[start:end] = (crop.astype(np.float64)*event['target_gain']/meta['final_normalization_divisor']).astype(np.float32)
            errors.append(float(np.max(np.abs(expected-target))))
            source_zero_fractions.append(float(np.mean(original==0)))
            assert np.all(target[:start]==0) and np.all(target[end:]==0)
        raw_bg, _ = read_mono_waveform(meta['background_path'])
        bg_expected = raw_bg.astype(np.float64)/np.std(raw_bg.astype(np.float64))/meta['final_normalization_divisor']
        bg_error = float(np.max(np.abs(bg_expected-bg)))
        sum_error = float(np.max(np.abs(mixture-(bg+targets[0]+targets[1]))))
        assert max(errors+[bg_error,sum_error]) < 2e-6
        assert all(np.std(t)>1e-5 for t in targets)
        assert not np.any(arrays['wrong'])
        # Exact deterministic replay also guards source IDs, placement and common gain.
        replay = dataset._draw(index, seed*10000+index)
        assert replay[-1]==meta and np.array_equal(replay[2], mixture)
        assert all(np.array_equal(x,y) for x,y in zip(replay[1],targets))
        item = dict(id=sample_id, folder=str(directory.relative_to(root)), files=files,
                    conditions=labels, metadata=meta)
        write_json(directory/'labels_and_geometry.json', item)
        samples.append(item)
        gt_checks.append(dict(id=sample_id, target_reconstruction_errors=errors,
            background_reconstruction_error=bg_error, stem_sum_error=sum_error,
            exact_zero_fractions_in_original_sources=source_zero_fractions, replay='exact'))
        emit('prepared', sample=sample_id, labels={k:v['name'] for k,v in labels.items()}, overlap=meta['overlap'])
    manifest = dict(schema=1, scope='fixed TRAIN samples only; no generalization claim', seed=seed,
        git_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip(),
        source_config=vars(hp), samples=samples, condition_vocabulary=sorted(vocabulary),
        positive_pool_report=label_report, selected_pool_report=dataset.report,
        source_audits=dict(background=dataset.background_audit, fsd=dataset.fsd_audit),
        code_sha256={f:file_sha256(PROJECT/f) for f in ('datasets/tse_hierarchy_dataset.py',
            'datasets/tse_hierarchy_labels.py','datasets/tse_dual_mix.py','nets/tse_dprnntf.py',
            'loss_functions/tse_ri_loss.py','loss_functions/tse_loss.py')})
    write_json(root/'prepared_manifest.json', manifest)
    write_json(root/'groundtruth_checks.json', dict(status='passed', cases=gt_checks,
        limitation='Waveform construction/alignment verified; weak labels do not prove acoustic purity'))
    (root/'README.txt').write_text('固定训练样本的小规模拟合检查，不是泛化评测。\n'
        'audio/sample_XX/ 下：mixture=混音，background=背景，gt_A/B=目标，gt_WRONG=全零。\n'
        'pred_A/B/WRONG 为 latest_evaluation.json 标明步数的模型输出。benchmark_ 开头的文件仅是短基准结果。\n'
        '所有 WAV 均为 16 kHz、10 秒、float32，同一幅度尺度；没有分别归一化或裁掉静音。\n'
        'labels_and_geometry.json 记录类别、原始标签、原始文件和时间区间。\n'
        '音频和报告为用户要求的结果，予以保留；只清理临时传输文件。\n')


def load_samples(root):
    manifest = json.loads((root/'prepared_manifest.json').read_text())
    result = []
    for item in manifest['samples']:
        arrays = {k:sf.read(str(root/item['folder']/f), dtype='float32')[0] for k,f in item['files'].items()}
        assert all(len(v)==160000 and np.isfinite(v).all() for v in arrays.values())
        result.append({**item, **arrays})
    return manifest, result


def audit(hp, root):
    _, samples = load_samples(root)
    report = dict(metrics=metric_checks(), loss=loss_checks(hp, samples))
    write_json(root/'loss_metric_checks.json', report)
    emit('audits_complete', loss=report['loss']['status'],
         legacy_metric_scale_issue=bool(report['metrics']['legacy_scale_dependence_detected']),
         zero_output_loss=report['loss']['all_zero_objective'])


def predict(model, loss_fn, sample, name, device):
    mixture = torch.from_numpy(sample['mix'][None]).to(device)
    target = torch.from_numpy(sample[name][None]).to(device)
    condition = torch.zeros(1,527,device=device)
    condition[0,sample['conditions'][name]['index']] = 1.
    prediction = model(dict(mixture_waveform=mixture, condition=condition))
    losses = loss_fn(prediction, dict(mixture_waveform=mixture,target_waveform=target))
    return prediction, losses


@torch.no_grad()
def evaluate(model, loss_fn, samples, root, device, step, prefix='pred', limit=None):
    model.eval()
    cases = []
    for sample in samples[:limit]:
        for name in ('a','b','wrong'):
            prediction, losses = predict(model, loss_fn, sample, name, device)
            value = prediction['target_waveform'][0].cpu().numpy()
            stats = measure(value, sample[name], sample['mix'], wrong=name=='wrong')
            stats.update(sample=sample['id'], condition=name, loss=float(losses['loss']))
            cases.append(stats)
            filename = f'{prefix}_{name.upper()}_{label_name(sample["conditions"][name]["name"])}.wav'
            sf.write(str(root/sample['folder']/filename), value, 16000, subtype='FLOAT')
    correct = [r for r in cases if r['condition']!='wrong']
    wrong = [r for r in cases if r['condition']=='wrong']
    report = dict(step=step, evaluated_mixtures=len(correct)//2, cases=cases,
        loss=float(np.mean([r['loss'] for r in cases])),
        positive_si_sdri_db=float(np.mean([r['si_sdri_db'] for r in correct])),
        positive_improvement_fraction=float(np.mean([r['si_sdri_db']>0 for r in correct])),
        wrong_energy_ratio=float(np.mean([r['output_to_mixture_energy'] for r in wrong])))
    write_json(root/(prefix+'_evaluation.json'), report)
    if prefix=='pred':
        write_json(root/'latest_evaluation.json', report)
    emit('evaluation', **{k:v for k,v in report.items() if k!='cases'})
    return report


def fit(hp, root, args):
    if not (root/'loss_metric_checks.json').exists():
        raise ValueError('Run preparation and numerical audits before fitting')
    manifest, samples = load_samples(root)
    if manifest['source_config'] != vars(hp):
        raise ValueError('Prepared-data configuration changed')
    device = torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('No allocated/usable CUDA device')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = create_tse_model(hp).to(device)
    loss_fn = TSERILoss(hp).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.learning_rate)
    order = [(i,n) for i in range(len(samples)) for n in ('a','b','wrong')]
    rng = random.Random(args.seed)
    start_step = 0
    checkpoint = root/'smallfit_last.pt'
    if args.resume:
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        if saved['source_config'] != vars(hp):
            raise ValueError('Checkpoint configuration differs')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        start_step = int(saved['step'])
        rng.setstate(saved['rng_state'])
        order = saved['order']
    emit('fit_started', device=str(device), host=socket.gethostname(), threads=torch.get_num_threads(),
        parameter_count=sum(p.numel() for p in model.parameters()), start_step=start_step,
        max_steps=args.steps, max_seconds=args.max_seconds,
        gpu=torch.cuda.get_device_name() if device.type=='cuda' else None)
    started = time.monotonic()
    durations = []
    step = start_step
    for step in range(start_step+1,args.steps+1):
        if (step-1)%len(order)==0:
            rng.shuffle(order)
        index,name = order[(step-1)%len(order)]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        tick = time.monotonic()
        prediction, losses = predict(model, loss_fn, samples[index], name, device)
        loss = losses['loss']
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite small-fit loss')
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), hp.gradient_clip_val, error_if_nonfinite=True)
        optimizer.step()
        if device.type=='cuda':
            torch.cuda.synchronize()
        duration = time.monotonic()-tick
        durations.append(duration)
        record = dict(step=step, sample=samples[index]['id'], condition=name, loss=float(loss.detach()),
            gradient_norm=float(gradient), seconds=duration, elapsed=time.monotonic()-started)
        del prediction, losses, loss
        with (root/'fit_history.jsonl').open('a') as handle:
            handle.write(json.dumps(record)+'\n')
        emit('fit_step', **record)
        stop = step==args.steps or time.monotonic()-started>=args.max_seconds
        if step%args.eval_every==0 or stop:
            temp = checkpoint.with_suffix('.tmp')
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,
                source_config=vars(hp),rng_state=rng.getstate(),order=order),temp)
            temp.replace(checkpoint)
            if args.stage=='benchmark':
                evaluate(model,loss_fn,samples,root,device,step,prefix='benchmark',limit=1)
            else:
                evaluate(model,loss_fn,samples,root,device,step)
        if stop:
            break
    write_json(root/(args.stage+'_run.json'),dict(status='completed_bounded_run',device=str(device),step=step,
        seconds_per_step=float(np.mean(durations)) if durations else None,
        elapsed_seconds=time.monotonic()-started, full_model=True, waveform_seconds=10,
        note='Completion of a bounded run is not evidence that fitting succeeded; inspect evaluation metrics'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exp-def',default=str(PROJECT/'exp_defs/tse_strong_dual_fsd_hierarchy_v2.yaml'))
    parser.add_argument('--output',required=True)
    parser.add_argument('--stage',choices=('prepare','audit','benchmark','fit'),required=True)
    parser.add_argument('--background-split',default='/home/syy12632604/projects/TSE/experiments/manifests/tse_hierarchy_v2/ailab01/background_split.json')
    parser.add_argument('--samples',type=int,default=16)
    parser.add_argument('--seed',type=int,default=71)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--steps',type=int,default=384)
    parser.add_argument('--eval-every',type=int,default=96)
    parser.add_argument('--max-seconds',type=int,default=3600)
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    if args.samples<2 or args.samples%2 or min(args.threads,args.steps,args.eval_every,args.max_seconds)<1:
        raise ValueError('Need positive bounds and an even sample count')
    root = Path(args.output).resolve()
    allowed = Path('/home/syy12632604/projects/TSE/experiments/diagnostics').resolve()
    if allowed not in root.parents:
        raise ValueError('Small-fit artifacts must be inside the TSE/experiments/diagnostics directory')
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Run bounded audio/model work inside a Slurm allocation')
    root.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    hp = yaml_to_parser(args.exp_def).parse_args([])
    validate_tse_config(hp)
    emit('stage_started',stage=args.stage,job=os.environ['SLURM_JOB_ID'],output=str(root))
    if args.stage=='prepare':
        prepare(hp,root,args.samples,args.seed,args.background_split)
    elif args.stage=='audit':
        audit(hp,root)
    else:
        fit(hp,root,args)


if __name__=='__main__':
    main()
