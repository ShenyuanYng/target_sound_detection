"""Fixed-mixture diagnostic training with exhaustive Train/CV evaluation."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.tse_smallfit_dataset import build_fixed_datasets
from loss_functions.tse_loss import unpack_waveform
from nets.dprnntf_original.audio_utils_torch import do_stft, do_istft
from tse_dual_separator import TSEDualSourceSeparator
from utils.my_optimizer import ReduceLROnPlateau_verbose
from utils.tse_dual_metrics import measure
from utils.tse_smallfit_callback import summarize_scores


class TSESmallfitSeparator(TSEDualSourceSeparator):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.run = Path(self.hparams.save_path).parent

    def setup(self, stage=None):
        if stage not in (None, 'fit', 'validate'):
            raise ValueError('Smallfit supports fit/validate only')
        if not self.datasets:
            self.datasets = build_fixed_datasets(self.hparams, self.run)

    def _loader(self, partition, training=False):
        # Separate RNGs keep evaluation from changing training order/conditions.
        generator = torch.Generator().manual_seed(self.hparams.seed + (not training))
        return DataLoader(self.datasets[partition], batch_size=self.hparams.batch_size,
            shuffle=training and bool(self.hparams.shuffle), collate_fn=self.collate_fn, generator=generator,
            num_workers=self.hparams.num_workers if training else 0,
            persistent_workers=training and self.hparams.num_workers > 0,
            worker_init_fn=self._worker_init_fn)

    def train_dataloader(self):
        return self._loader('tr', training=True)

    def val_dataloader(self):
        return [self._loader('tr'), self._loader('cv')]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        # Epochs have only 128 updates. Step the scheduler on complete 500-step
        # evaluations, never on a stale validation value at intervening epochs.
        self.smallfit_scheduler = ReduceLROnPlateau_verbose(optimizer,
            patience=self.hparams.patience, factor=self.hparams.decay_factor)
        return optimizer

    def on_validation_epoch_start(self):
        self.eval_rows = {phase: [] for phase in ('train_eval', 'val')}
        self.eval_audio = {phase: {} for phase in self.eval_rows}

    def _loss_waveforms(self, mixture, target):
        if self.hparams.rms_norm == 0:
            return mixture, target, torch.ones_like(mixture[:, :1])
        # Teacher rms_norm=2: per-mixture np.std(..., ddof=0), without subtracting
        # the mean from either waveform. Both GTs use this same mixture scale.
        with torch.autocast(device_type=mixture.device.type, enabled=False):
            mix_std = mixture.float().std(dim=-1, keepdim=True, correction=0)
            mix_std = torch.where(mix_std == 0, torch.ones_like(mix_std), mix_std)
            return mixture.float() / mix_std, target.float() / mix_std, mix_std

    def _condition_prediction(self, mixture, condition, target):
        if self.hparams.rms_norm == 0:
            return super()._condition_prediction(mixture, condition, target)
        loss_mix, loss_target, mix_std = self._loss_waveforms(mixture, target)
        waveform, losses = super()._condition_prediction(loss_mix, condition, loss_target)
        # RI loss stays in the normalized domain. Only the waveform used for
        # metrics/listening returns to the original mixture/GT amplitude scale.
        return waveform * mix_std, losses

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        phase = ('train_eval', 'val')[dataloader_idx]
        inputs, targets = batch
        mixture = unpack_waveform(inputs['mixture_waveform'])
        mix_cpu = mixture.float().cpu().numpy()
        count = len(mix_cpu)
        selected = {2 * index + index % 2 for index in range(self.hparams.tse_smallfit_examples)}
        estimates = {}
        for name in ('a', 'b', 'wrong'):
            target = unpack_waveform(targets[f'target_{name}'])
            waveform, losses = self._condition_prediction(mixture,
                unpack_waveform(inputs[f'condition_{name}']), target)
            if not all(torch.isfinite(value).all() for value in losses.values()):
                raise FloatingPointError(f'Nonfinite smallfit {phase}/{name} loss')
            waves, references = waveform.float().cpu().numpy(), target.float().cpu().numpy()
            estimates[name] = waves
            baselines = {}
            if self.trainer.sanity_checking:
                loss_mix, loss_target, mix_std = self._loss_waveforms(mixture, target)
                with torch.autocast(device_type=mixture.device.type, enabled=False):
                    oracle = do_stft(loss_target.float(), self.hparams.tse_window, self.hparams.tse_hop)
                    restored = do_istft(oracle, self.hparams.tse_window,
                                       self.hparams.tse_window, self.hparams.tse_hop)[..., :target.shape[-1]] * mix_std
                    if not torch.allclose(restored, target.float(), atol=2e-6, rtol=1e-5):
                        raise ValueError('STFT/iSTFT oracle reconstruction failed')
                    for label, ri in (('oracle', oracle), ('zero', torch.zeros_like(oracle)),
                            ('mixture', do_stft(loss_mix.float(), self.hparams.tse_window, self.hparams.tse_hop))):
                        result = self.loss(dict(target_ri=ri),
                                           dict(target_waveform=loss_target, mixture_waveform=loss_mix))
                        baselines[label] = result['per_example_loss'].cpu().tolist()
                    if max(baselines['oracle']) > 1e-6:
                        raise ValueError('Ground-truth RI must have zero loss')
            for row in range(count):
                index = batch_idx * self.hparams.batch_size + row
                metrics = measure(waves[row], references[row], mix_cpu[row], wrong=name == 'wrong')
                if not np.isfinite(waves[row]).all():
                    raise FloatingPointError('Nonfinite smallfit waveform')
                metrics.update(index=index, condition=name, mid=targets[f'mid_{name}'][row],
                    utt_id=targets['utt_id'][row], overlap=bool(targets['overlap'][row]),
                    loss=float(losses['per_example_loss'][row]),
                    **{key: float(losses[key]) for key in ('sR', 'sI', 'sM')},
                    **{f'{key}_loss': values[row] for key, values in baselines.items()})
                self.eval_rows[phase].append(metrics)
                if index in selected:
                    self.eval_audio[phase].setdefault(index, {})[name] = waves[row].copy()
            del waveform, losses
        for row in range(count):
            a, b = (estimates[name][row].astype(np.float64) for name in ('a', 'b'))
            a, b = a - a.mean(), b - b.mean()
            correlation = float(np.clip(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30), -1., 1.))
            for item in self.eval_rows[phase][-3 * count:]:
                if item['index'] == batch_idx * self.hparams.batch_size + row:
                    item['a_b_output_correlation'] = correlation

    def on_validation_epoch_end(self):
        conditions = (('a', 'b') if self.hparams.tse_train_condition_mode == 'sampled_correct'
                      else ('a', 'b', 'wrong'))
        self.eval_summary = {phase: summarize_scores(rows, loss_conditions=conditions)
                             for phase, rows in self.eval_rows.items()}
        for phase, summary in self.eval_summary.items():
            self.log(f'{phase}_loss', summary['loss'], add_dataloader_idx=False)
        if not self.trainer.sanity_checking:
            self.smallfit_scheduler.step(self.eval_summary['val']['loss'])

    def on_save_checkpoint(self, checkpoint):
        checkpoint['smallfit_scheduler'] = self.smallfit_scheduler.state_dict()

    def on_load_checkpoint(self, checkpoint):
        raise ValueError('Fixed-mixture smallfit uses one fresh allocation; do not resume a diagnostic run')
