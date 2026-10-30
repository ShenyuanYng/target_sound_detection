"""Per-condition, per-placement metrics; silent queries never enter SI-SNR."""
import numpy as np
import torch

from loss_functions.tse_loss import si_sdr


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
        old, _ = si_sdr(torch.tensor(x[None], dtype=torch.float32),
                         torch.tensor(y[None], dtype=torch.float32))
        result.update(si_sdr_db=score, metric_status=status,
            mixture_si_sdr_db=baseline, si_sdri_db=score-baseline if score is not None else None,
            legacy_si_sdr_db=float(old.item()), output_to_target_energy=float(power/max(reference_power, 1e-30)))
    return result


class TSEDualMetrics:
    def __init__(self, silence_rms=1e-5, record_overlap_loss=True):
        self.silence_rms = silence_rms
        self.record_overlap_loss = record_overlap_loss
        self.sums = {group: {name: dict(count=0, loss=0., valid=0,
            si_snr_db=0., si_snri_db=0., output_rms=0., energy_ratio=0.)
            for name in ('a', 'b', 'wrong')} for group in ('all', 'nonoverlap', 'overlap')}
        self.objectives = {group: dict(count=0, loss=0.) for group in self.sums}

    @torch.no_grad()
    def update_objective(self, losses, overlap):
        """Accumulate the actual per-mixture estimator, including sampled training."""
        for group, item in self.objectives.items():
            mask = (torch.ones_like(overlap) if group == 'all' else
                    overlap if group == 'overlap' else ~overlap)
            item['count'] += int(mask.sum())
            if group == 'all' or self.record_overlap_loss:
                item['loss'] += float(losses.detach()[mask].double().sum())

    @torch.no_grad()
    def update(self, name, prediction, target, mixture, losses, overlap):
        prediction, target, mixture = prediction.detach().float(), target.float(), mixture.float()
        if name == 'wrong':
            power = prediction.square().mean(-1)
            rms = power.sqrt()
            ratio = power / mixture.square().mean(-1).clamp_min(1e-12)
        else:
            score, valid = si_sdr(prediction, target, self.silence_rms)
            baseline, _ = si_sdr(mixture, target, self.silence_rms)
        for group in self.sums:
            mask = (torch.ones_like(overlap) if group == 'all' else
                    overlap if group == 'overlap' else ~overlap)
            item = self.sums[group][name]
            item['count'] += int(mask.sum())
            if group == 'all' or self.record_overlap_loss:
                item['loss'] += float(losses.detach()[mask].double().sum())
            if name == 'wrong':
                item['output_rms'] += float(rms[mask].double().sum())
                item['energy_ratio'] += float(ratio[mask].double().sum())
            else:
                active = mask & valid
                item['valid'] += int(active.sum())
                item['si_snr_db'] += float(score[active].double().sum())
                item['si_snri_db'] += float((score-baseline)[active].double().sum())

    def compute(self):
        result = {}
        for group, conditions in self.sums.items():
            result[group] = {}
            for name, sums in conditions.items():
                count = sums['count']
                item = dict(count=count)
                if group == 'all' or self.record_overlap_loss:
                    item['loss'] = sums['loss']/count if count else None
                if name == 'wrong':
                    item.update({key: sums[key]/count if count else None
                                 for key in ('output_rms', 'energy_ratio')})
                else:
                    item.update(valid_references=sums['valid'],
                        **{key: sums[key]/sums['valid'] if sums['valid'] else None
                           for key in ('si_snr_db', 'si_snri_db')})
                result[group][name] = item
            objective = self.objectives[group]
            result[group]['sample_count'] = objective['count']
            if group == 'all' or self.record_overlap_loss:
                losses = [result[group][n]['loss'] for n in ('a', 'b', 'wrong')]
                result[group]['loss'] = (objective['loss']/objective['count'] if objective['count'] else
                    sum(losses)/3 if all(v is not None for v in losses) else None)
        return result
