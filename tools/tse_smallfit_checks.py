"""Independent numerical checks for a bounded TSE fitting experiment."""
import json
from pathlib import Path

import numpy as np
import torch

from loss_functions.tse_loss import si_sdr as legacy_si_sdr
from loss_functions.tse_ri_loss import TSERILoss
from nets.dprnntf_original.audio_utils_torch import do_stft, do_istft


def reference_sisdr(estimate, target):
    """Float64, separately normalized vectors; cap only the final correlation.

    Exact zero output is undefined mathematically: report a -120 dB penalty AND
    an explicit flag. Never evaluate a silent reference with this metric.
    """
    x, y = np.asarray(estimate, dtype=np.float64), np.asarray(target, dtype=np.float64)
    x, y = x-x.mean(), y-y.mean()
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    if ny / np.sqrt(len(y)) <= 1e-5:
        return None, 'silent_reference'
    if nx == 0:
        return -120., 'zero_output'
    x, y = x/nx, y/ny
    projection = np.dot(x, y)*y
    signal = float(np.dot(projection, projection))
    error = float(np.sum((x-projection)**2))
    return float(10*np.log10(max(signal, 1e-12)/max(error, 1e-12))), 'valid'


def measure(estimate, target, mixture, wrong=False):
    x, y, mix = [np.asarray(v, dtype=np.float64) for v in (estimate, target, mixture)]
    power, reference_power = np.mean(x*x), np.mean(y*y)
    result = dict(output_rms=float(np.sqrt(power)), target_rms=float(np.sqrt(reference_power)),
        output_to_mixture_energy=float(power/max(np.mean(mix*mix), 1e-30)),
        peak=float(np.abs(x).max()))
    if not wrong:
        score, status = reference_sisdr(x, y)
        baseline, _ = reference_sisdr(mix, y)
        old, _ = legacy_si_sdr(torch.tensor(x[None], dtype=torch.float32),
                              torch.tensor(y[None], dtype=torch.float32))
        result.update(si_sdr_db=score, metric_status=status,
            mixture_si_sdr_db=baseline, si_sdri_db=score-baseline if score is not None else None,
            legacy_si_sdr_db=float(old.item()), output_to_target_energy=float(power/max(reference_power, 1e-30)))
    return result


def metric_checks():
    rng = np.random.default_rng(71)
    target = rng.normal(size=16000).astype(np.float32)
    estimate = target + .5*rng.normal(size=len(target)).astype(np.float32)
    rows = []
    for scale in (1., .01, 1e-4, 1e-6):
        prediction = estimate*scale
        old, _ = legacy_si_sdr(torch.tensor(prediction[None]), torch.tensor(target[None]))
        ref, _ = reference_sisdr(prediction, target)
        rows.append(dict(scale=scale, legacy_si_sdr_db=float(old), reference_si_sdr_db=ref))
    reference_spread = float(np.ptp([r['reference_si_sdr_db'] for r in rows]))
    assert reference_spread < 1e-5, rows
    perfect, _ = reference_sisdr(target, target)
    zero, flag = reference_sisdr(np.zeros_like(target), target)
    assert perfect > 100 and zero < -100 and flag == 'zero_output'
    return dict(reference_checks='passed', scale_sweep=rows,
        legacy_scale_dependence_detected=bool(np.ptp([r['legacy_si_sdr_db'] for r in rows]) > .01),
        zero_output_convention='undefined; explicit flag and -120 dB penalty',
        note='Independent diagnostic metric only; running experiments are unchanged')


def loss_checks(hp, samples):
    loss_fn = TSERILoss(hp)
    rows = []
    for sample in samples:
        mix = torch.tensor(sample['mix'][None])
        mix_ri = do_stft(mix, hp.tse_window, hp.tse_hop)
        for name in ('a', 'b', 'wrong'):
            target = torch.tensor(sample[name][None])
            ri = do_stft(target, hp.tse_window, hp.tse_hop)
            targets = dict(mixture_waveform=mix, target_waveform=target)
            perfect = loss_fn(dict(target_ri=ri), targets)
            assert float(perfect['loss']) == 0.
            reconstructed = do_istft(ri, hp.tse_window, hp.tse_window, hp.tse_hop)[..., :target.shape[-1]]
            reconstruction_error = float((reconstructed-target).abs().max())
            assert reconstruction_error < 2e-6, reconstruction_error
            observed = loss_fn(dict(target_ri=mix_ri), targets)
            # An independent NumPy implementation uses native [B,T,F,RI] layout.
            m, t = mix_ri.double().numpy(), ri.double().numpy()
            denom = np.sqrt(m[...,0]**2+m[...,1]**2+1e-5).sum()
            expected = (np.abs(m[...,0]-t[...,0]).sum()+np.abs(m[...,1]-t[...,1]).sum()
                +np.abs(np.sqrt(m[...,0]**2+m[...,1]**2+1e-5)
                       -np.sqrt(t[...,0]**2+t[...,1]**2+1e-5)).sum())/denom
            error = abs(float(observed['loss'])-expected)
            assert error < 2e-6, error
            zero = loss_fn(dict(target_ri=torch.zeros_like(ri)), targets)
            rows.append(dict(sample=sample['id'], condition=name, perfect_loss=float(perfect['loss']),
                mixture_prediction_loss=float(observed['loss']), independent_loss_error=error,
                zero_prediction_loss=float(zero['loss']), stft_roundtrip_max_error=reconstruction_error))
    # A finite gradient through the actual loss adapter, without any network.
    probe = mix_ri.detach().clone().requires_grad_()
    value = loss_fn(dict(target_ri=probe), dict(mixture_waveform=mix, target_waveform=torch.zeros_like(mix)))['loss']
    value.backward()
    assert torch.isfinite(probe.grad).all() and probe.grad.abs().sum() > 0
    return dict(status='passed', cases=rows, gradient_check='passed',
                all_zero_objective=float(np.mean([r['zero_prediction_loss'] for r in rows])))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temporary.replace(path)
