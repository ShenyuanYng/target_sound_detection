"""Export a bounded, score-independent listening preview from immutable dual CV.

Run from the shared project on an allocated compute node. Original WAVs retain
float32 samples; browser WAVs only convert to PCM16, without gain normalization.
"""
import argparse
from argparse import Namespace
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import soundfile as sf
import torch

from datasets.audioset_fsd50k_mix_dataset import digest
from datasets.fsd50k_cache import file_sha256
from datasets.mix_utils import read_mono_waveform
from datasets.paths.audioset_paths import load_class_map, resolve_path
from datasets.tse_dataset import waveform_hash
from datasets.tse_dual_mix import make_tse_dual_mixture
from loss_functions.tse_loss import si_sdr
from tools.export_tse_audio_compare import export_wav
from tools.infer_tse_source_tests import read_fsd
from tse_dual_separator import TSEDualSourceSeparator


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def replay(row, hp):
    sources = [(row['background_path'], row['background_sha256'])] + [
        (event['fsd_path'], event['fsd_sha256']) for event in row['events']]
    for path, expected in sources:
        if file_sha256(path) != expected:
            raise ValueError(f'Source audio changed: {path}')
    background = read_mono_waveform(row['background_path'])[0]
    raw = [read_fsd(event['fsd_path']) for event in row['events']]
    bg, targets, mixture, meta = make_tse_dual_mixture(background, raw,
        seed=row['seed'], overlap=row['overlap'], snr_db=row['snr_db'],
        threshold_dbfs=hp.mix_active_threshold_dbfs, frame_ms=hp.mix_active_frame_ms,
        max_attempts=hp.tse_dual_max_crop_attempts)
    for key, value in meta.items():
        if key == 'events':
            for actual, recorded in zip(value, row[key]):
                if any(recorded[k] != v for k, v in actual.items()):
                    raise ValueError('Recorded crop/gain/placement changed')
        elif row[key] != value:
            raise ValueError(f'Recorded mixing metadata changed: {key}')
    waves = dict(background=bg, target_a=targets[0], target_b=targets[1], mixture=mixture)
    for key, manifest_key in [('background', 'background'), ('target_a', 'a'),
                              ('target_b', 'b'), ('mixture', 'mixture')]:
        if waveform_hash(waves[key]) != row['waveform_sha256'][manifest_key]:
            raise ValueError(f'CV waveform hash mismatch: {key}')
    for path, expected in sources:
        if file_sha256(path) != expected:
            raise ValueError(f'Source changed while replaying: {path}')
    return waves


def energy(wave):
    rms = float(np.sqrt(np.mean(wave.astype(np.float64)**2)))
    return dict(rms=rms, rms_dbfs=float(20*np.log10(max(rms, 1e-12))),
                peak=float(np.abs(wave).max()))


def score(prediction, target, mixture, silence_rms):
    p, t, m = [torch.from_numpy(w).unsqueeze(0) for w in (prediction, target, mixture)]
    value, valid = si_sdr(p, t, silence_rms)
    baseline, _ = si_sdr(m, t, silence_rms)
    if not bool(valid[0]):
        raise ValueError('Unexpected silent reference')
    return dict(si_snr_db=float(value[0]), mixture_si_snr_db=float(baseline[0]),
                si_snri_db=float(value[0]-baseline[0]),
                output_to_target_rms_db=energy(prediction)['rms_dbfs']-energy(target)['rms_dbfs'])


def plot_waves(directory, waves, names):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.signal import stft
    order = ['mixture', 'target_a', 'prediction_a', 'target_b', 'prediction_b', 'prediction_wrong']
    labels = ['Mixture', 'A reference: '+names[0], 'A prediction',
              'B reference: '+names[1], 'B prediction', 'Absent query: '+names[2]]
    fig, axes = plt.subplots(6, 2, figsize=(13, 12), constrained_layout=True)
    peak = max(float(np.abs(waves[k]).max()) for k in order) * 1.05
    # Fixed shared waveform and STFT scales expose attenuation instead of hiding it.
    for i, (key, label) in enumerate(zip(order, labels)):
        w = waves[key]
        blocks = w.reshape(800, -1)
        times = np.arange(800) * (10/800)
        axes[i, 0].fill_between(times, blocks.min(1), blocks.max(1), color='#26718b')
        axes[i, 0].set(xlim=(0, 10), ylim=(-peak, peak), ylabel='Amplitude', title=label)
        f, t, z = stft(w, fs=16000, nperseg=512, noverlap=384)
        spec = axes[i, 1].pcolormesh(t, f/1000, 20*np.log10(np.maximum(np.abs(z), 1e-8)),
                                   vmin=-100, vmax=0, shading='auto', cmap='magma')
        axes[i, 1].set(xlim=(0, 10), ylim=(0, 8), ylabel='Frequency (kHz)')
    for ax in axes[-1]:
        ax.set_xlabel('Time (s)')
    fig.colorbar(spec, ax=axes[:, 1], label='STFT magnitude (dB re 1)')
    fig.savefig(directory/'comparison.png', dpi=130)
    plt.close(fig)


def render_report(output, report):
    esc = html.escape
    sections = []
    track_names = [('mixture', '原始混音'), ('background', '背景真值'),
                   ('target_a', 'A 目标真值'), ('prediction_a', 'A 分离结果'),
                   ('target_b', 'B 目标真值'), ('prediction_b', 'B 分离结果'),
                   ('prediction_wrong', '错误条件输出（期望静音）')]
    for sample in report['samples']:
        directory = sample['directory']
        tracks = ''.join(f'<div><h3>{label}</h3><audio controls preload="metadata" '
            f'src="{directory}/{key}_listen.wav"></audio><p>RMS '
            f'{sample["energy"][key]["rms_dbfs"]:.1f} dBFS · '
            f'<a href="{directory}/{key}.wav">原始 FLOAT WAV</a></p></div>'
            for key, label in track_names)
        metrics = ''.join(f'<tr><td>{label.upper()} · {esc(sample["classes"][i])}</td>'
            f'<td>{sample["scores"][label]["si_snr_db"]:.2f}</td>'
            f'<td>{sample["scores"][label]["si_snri_db"]:+.2f}</td>'
            f'<td>{sample["scores"][label]["output_to_target_rms_db"]:+.1f}</td></tr>'
            for i, label in enumerate(('a', 'b')))
        sections.append(f'<section><h2>CV {sample["cv_index"]} · '
            f'{"重叠" if sample["row"]["overlap"] else "不重叠"}</h2>'
            f'<p>A：{esc(sample["classes"][0])}　B：{esc(sample["classes"][1])}　'
            f'错误条件：{esc(sample["classes"][2])}</p><div class="tracks">{tracks}</div>'
            f'<table><thead><tr><th>目标</th><th>SI-SNR (dB)</th><th>改善 (dB)</th>'
            f'<th>输出 / 真值 RMS (dB)</th></tr></thead><tbody>{metrics}</tbody></table>'
            f'<img src="{directory}/comparison.png" alt="共享刻度的波形和频谱对比"></section>')
    output.joinpath('index.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>双 FSD 分离试听</title>
<style>body{font:16px/1.6 system-ui,sans-serif;color:#202a32;background:#fafbfc;max-width:1180px;margin:32px auto;padding:0 20px}h1,h2,h3{font-weight:600}h3{font-size:16px;margin:0 0 8px}section{border-top:1px solid #cdd6dc;margin-top:32px;padding-top:20px}.tracks{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:24px}audio{width:100%}p{overflow-wrap:anywhere}.tracks p{font-size:13px;color:#475767}table{border-collapse:collapse;width:100%;margin:20px 0}th,td{text-align:left;padding:8px;border-bottom:1px solid #d9e0e5}img{width:100%;height:auto}a{color:#146888}code{overflow-wrap:anywhere}@media(max-width:500px){body{padding:0 12px}table{font-size:12px}th,td{padding:5px}}</style>
<h1>双 FSD 分离试听 · Job 39546</h1>''' +
        f'<p>Checkpoint：第 {report["epoch_one_based"]} 轮，step {report["global_step"]}。'
        f'固定 CV 按 seed={report["seed"]} 分层随机抽样；重叠、不重叠各 '
        f'{len(report["samples"])//2} 条，未按结果筛选。此小样本仅用于诊断。</p>'
        '<p>所有音频保留原始音量和对齐；试听文件只转换为 PCM16，未逐条放大。'
        '波形共用幅度刻度，频谱共用 −100 至 0 dB 刻度。SI-SNR 改善为分离结果减去混音基线。</p>' +
        ''.join(sections) + '<p><a href="report.json">完整指标与复现记录</a></p>'
        '<script>document.addEventListener("play",e=>{if(e.target.tagName==="AUDIO")'
        'document.querySelectorAll("audio").forEach(a=>{if(a!==e.target)a.pause()})},true)</script></html>')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--per-group', type=int, default=2)
    parser.add_argument('--seed', type=int, default=39546)
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args()
    if args.per_group < 1 or args.cpu_threads < 1:
        parser.error('Counts must be positive')
    output, checkpoint = Path(args.output).resolve(), Path(args.checkpoint).resolve()
    if output.exists():
        raise ValueError('Choose a new output directory')
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    checkpoint_hash = file_sha256(checkpoint)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if file_sha256(checkpoint) != checkpoint_hash:
        raise ValueError('Checkpoint changed while loading; retry')
    hp = Namespace(**saved['hyper_parameters'])
    if hp.dataset != 'audioset_strong_dual_fsd_wrong_tse':
        raise ValueError('Expected dual-FSD/wrong-condition checkpoint')
    manifest_path = Path(hp.mix_cv_manifest_path)
    if not manifest_path.is_absolute():
        manifest_path = ROOT/manifest_path
    manifest_hash = file_sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest['rows_sha256'] != digest(manifest['rows']):
        raise ValueError('CV manifest row hash mismatch')
    for name, expected in manifest['settings']['code'].items():
        if file_sha256(ROOT/name) != expected:
            raise ValueError(f'CV source changed: {name}')
    class_path = resolve_path(hp.class_map_path, hp.data_path)
    if file_sha256(class_path) != manifest['settings']['class_map_sha256']:
        raise ValueError('Class map changed')
    mapping, names = load_class_map(class_path)
    rng = np.random.default_rng(args.seed)
    selected = []
    for overlap in (False, True):
        pool = [i for i, row in enumerate(manifest['rows']) if row['overlap'] == overlap]
        selected.extend(int(i) for i in rng.choice(pool, args.per_group, replace=False))
    model = TSEDualSourceSeparator(hp)
    model.on_load_checkpoint(saved)
    model.load_state_dict(saved['state_dict'], strict=True)
    model.eval()
    report = dict(status='running', checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash,
        epoch_one_based=int(saved['epoch'])+1, global_step=int(saved['global_step']),
        effective_config=vars(hp), manifest=str(manifest_path), manifest_sha256=manifest_hash,
        seed=args.seed, indices=selected, selection='seeded random, equal overlap/nonoverlap; no score selection',
        git_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source_hashes={}, samples=[], device='cpu', slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        created_utc=datetime.now(timezone.utc).isoformat(),
        audio_policy='raw FLOAT WAV; browser PCM16 conversion with no gain change or clipping',
        metric='training zero-mean SI-SDR / SI-SNR; improvement relative to mixture')
    del saved
    for name in subprocess.check_output(['git', 'ls-files', '-z', '*.py'], cwd=ROOT).decode().split('\0'):
        if name and (ROOT/name).is_file():
            report['source_hashes'][name] = file_sha256(ROOT/name)
    report['source_hashes']['tools/export_tse_dual_preview.py'] = file_sha256(__file__)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'report.json', report)
    print(f'Checkpoint epoch={report["epoch_one_based"]}; selected={selected}', flush=True)
    for index in selected:
        row = manifest['rows'][index]
        waves = replay(row, hp)
        mids = [event['target_mid'] for event in row['events']] + [row['wrong_mid']]
        x = torch.from_numpy(waves['mixture']).unsqueeze(0)
        scores = {}
        for label, mid in zip(('a', 'b', 'wrong'), mids):
            condition = torch.zeros(1, hp.num_classes)
            condition[0, mapping[mid]] = 1.
            with torch.inference_mode():
                pred = model(dict(mixture_waveform=x, condition=condition))['target_waveform']
            waves[f'prediction_{label}'] = pred[0].cpu().numpy().copy()
            if label != 'wrong':
                scores[label] = score(waves[f'prediction_{label}'], waves[f'target_{label}'],
                                      waves['mixture'], hp.tse_silence_rms)
            print(f'CV {index} condition {label} complete', flush=True)
        directory = output/f'cv_{index:04d}'
        directory.mkdir()
        for key, wave in waves.items():
            export_wav(directory/f'{key}.wav', wave)
            if float(np.abs(wave).max()) > 1.:
                raise ValueError('PCM16 preview would clip')
            sf.write(str(directory/f'{key}_listen.wav'), wave, 16000, subtype='PCM_16')
            decoded, rate = sf.read(str(directory/f'{key}_listen.wav'), dtype='float32')
            if rate != 16000 or decoded.shape != wave.shape or np.abs(decoded-wave).max() > 1/32768+1e-7:
                raise ValueError('PCM16 preview conversion error')
        sample = dict(cv_index=index, directory=directory.name, row=row, mids=mids,
            classes=[names[mapping[mid]] for mid in mids], scores=scores,
            energy={key: energy(wave) for key, wave in waves.items()})
        plot_waves(directory, waves, sample['classes'])
        write_json(directory/'metadata.json', sample)
        report['samples'].append(sample)
        write_json(output/'report.json', report)
        print(json.dumps(dict(cv_index=index, scores=scores)), flush=True)
    if file_sha256(manifest_path) != manifest_hash:
        raise ValueError('CV manifest changed during inference')
    if any(file_sha256(ROOT/name) != expected for name, expected in report['source_hashes'].items()):
        raise ValueError('Project source changed during inference')
    report['status'] = 'complete'
    report['audio_sha256'] = {str(p.relative_to(output)): file_sha256(p) for p in output.rglob('*.wav')}
    write_json(output/'report.json', report)
    render_report(output, report)
    print(f'COMPLETE {output}', flush=True)


if __name__ == '__main__':
    main()
