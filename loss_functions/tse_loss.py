"""Single-target SI-SDR and multi-resolution STFT; no permutation search."""
import torch
from torch import nn


def unpack_waveform(value):
    tensor = value[0] if isinstance(value, (tuple, list)) else value
    if not torch.is_tensor(tensor) or tensor.ndim != 2:
        raise ValueError('Expected waveform [B,L]')
    if isinstance(value, (tuple, list)) and not (value[1] == tensor.shape[-1]).all():
        raise ValueError('TSE v1 requires equal full waveform lengths, without batch padding')
    return tensor


def si_sdr(estimate, target, silence_rms=1e-5, eps=1e-8):
    """Return per-example zero-mean SI-SDR (dB), and valid-reference mask.

    Invalid references produce a finite placeholder, excluded from metric means.
    A relative power floor stabilizes perfect reconstruction; zero output is -80 dB.
    """
    if estimate.shape != target.shape or estimate.ndim != 2:
        raise ValueError('SI-SDR waveform shapes must match [B,L]')
    estimate, target = estimate.float(), target.float()
    estimate = estimate - estimate.mean(-1, keepdim=True)
    target = target - target.mean(-1, keepdim=True)
    power = target.square().mean(-1, keepdim=True)
    valid = power.squeeze(-1) > silence_rms ** 2
    alpha = (estimate * target).mean(-1, keepdim=True) / power.clamp_min(1e-20)
    projected = alpha * target
    floor = (power * eps).clamp_min(1e-20)
    signal = projected.square().mean(-1, keepdim=True).clamp_min(floor)
    error = (estimate - projected).square().mean(-1, keepdim=True).clamp_min(floor)
    score = (10 * torch.log10(signal / error)).squeeze(-1)
    score = torch.where(estimate.square().mean(-1) <= 1e-20,
                        torch.full_like(score, -80.), score)
    return torch.where(valid, score, torch.zeros_like(score)), valid


class TSELoss(nn.Module):
    def __init__(self, mrstft_weight=1., resolutions=((256, 64), (512, 128), (1024, 256)),
                 silence_rms=1e-5):
        super().__init__()
        if not mrstft_weight > 0 or not silence_rms > 0 or not resolutions:
            raise ValueError('TSE loss needs positive weights/threshold and STFT resolutions')
        self.mrstft_weight, self.silence_rms = float(mrstft_weight), float(silence_rms)
        self.resolutions = tuple(tuple(r) for r in resolutions)
        for i, (size, hop) in enumerate(self.resolutions):
            if not (isinstance(size, int) and isinstance(hop, int) and 0 < hop <= size):
                raise ValueError('MR-STFT resolutions must be (FFT size, hop) integer pairs')
            self.register_buffer(f'window_{i}', torch.hann_window(size), persistent=False)

    def spectral_loss(self, estimate, target):
        # All samples, including inserted silence, constrain the reconstruction.
        losses = []
        for i, (size, hop) in enumerate(self.resolutions):
            window = getattr(self, f'window_{i}').to(estimate)
            def magnitude(waveform):
                spectrum = torch.stft(waveform, size, hop_length=hop, window=window,
                    center=True, pad_mode='constant', return_complex=True)
                return spectrum.abs()
            predicted, reference = magnitude(estimate), magnitude(target)
            difference = (predicted - reference).flatten(1)
            reference_norm = torch.linalg.vector_norm(reference.flatten(1), dim=-1)
            convergence = torch.linalg.vector_norm(difference, dim=-1) / reference_norm.clamp_min(1e-5)
            log_magnitude = (predicted.clamp_min(1e-5).log()
                             - reference.clamp_min(1e-5).log()).abs().mean(dim=(-2, -1))
            losses.append(convergence + log_magnitude)
        return torch.stack(losses).mean(0)

    def forward(self, predictions, targets, **kwargs):
        estimate = unpack_waveform(predictions['target_waveform'])
        target = unpack_waveform(targets['target_waveform']).to(estimate)
        if estimate.shape != target.shape:
            raise ValueError('Prediction and GT must have identical sample alignment/length')
        with torch.autocast(device_type=estimate.device.type, enabled=False):
            estimate, target = estimate.float(), target.float()
            scores, valid = si_sdr(estimate, target, self.silence_rms)
            if valid.any():
                sdr_loss = -scores[valid].mean()
                spectral = self.spectral_loss(estimate[valid], target[valid]).mean()
            else:
                sdr_loss = spectral = estimate.sum() * 0.
            # SI-SDR is undefined for silent references; explicitly suppress output.
            silence = estimate[~valid].square().mean() if (~valid).any() else estimate.sum() * 0.
            loss = sdr_loss + self.mrstft_weight * spectral + silence
        return dict(loss=loss, neg_si_sdr=sdr_loss.detach(), mrstft=spectral.detach(),
                    silence_loss=silence.detach())


def create_tse_loss(hp):
    return TSELoss(hp.tse_mrstft_weight, hp.tse_mrstft_resolutions, hp.tse_silence_rms)
