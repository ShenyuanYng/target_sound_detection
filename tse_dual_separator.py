"""Full or unbiased sampled-condition training; exhaustive three-condition CV."""
import torch
from loguru import logger

from loss_functions.tse_loss import unpack_waveform
from tse_separator import TSESourceSeparator
from utils.tse_dual_metrics import TSEDualMetrics


class TSEDualSourceSeparator(TSESourceSeparator):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.automatic_optimization = self.hparams.get('tse_condition_backward', 'combined') != 'sequential'

    def _reset_metrics(self, phase):
        self._metric_sums[phase] = TSEDualMetrics(self.hparams.tse_silence_rms,
            record_overlap_loss=self.hparams.get('tse_record_overlap_loss', True))

    def configure_optimizers(self):
        config = super().configure_optimizers()
        if not self.automatic_optimization:
            config['lr_scheduler'].pop('monitor', None)
        return config

    def _condition_mode(self, phase):
        return self.hparams.get('tse_train_condition_mode', 'all') if phase == 'train' else 'all'

    def on_load_checkpoint(self, checkpoint):
        super().on_load_checkpoint(checkpoint)
        saved_manual = checkpoint['hyper_parameters'].get('tse_condition_backward', 'combined') == 'sequential'
        if saved_manual == (not self.automatic_optimization) or 'loops' not in checkpoint:
            return
        # Lightning keeps separate optimizer-step counters for the two loop types.
        # Preserve global_step when switching execution without resetting Adam.
        fit = checkpoint['loops']['fit_loop']
        steps = int(checkpoint['global_step'])
        if not self.automatic_optimization:
            progress = fit['epoch_loop.manual_optimization.optim_step_progress']
            progress['total'].update(ready=steps, completed=steps)
        else:
            progress = fit['epoch_loop.automatic_optimization.optim_progress']
            progress['optimizer']['step']['total'].update(ready=steps, completed=steps)

    def training_step(self, batch, batch_idx):
        if self.automatic_optimization:
            return self._waveform_step(batch, 'train')
        optimizer = self.optimizers()
        optimizer.zero_grad()
        loss = self._waveform_step(batch, 'train')
        self.clip_gradients(optimizer, gradient_clip_val=self.hparams.gradient_clip_val,
                            gradient_clip_algorithm='norm')
        optimizer.step()
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.global_rank != 0:
            return
        if self.hparams.get('tse_log_batch_loss', False):
            stats = self._train_batch_stats
            value = lambda x: 'n/a' if x is None else f'{float(x):.6f}'
            logger.info('TSE_BATCH epoch={}/{} phase=train batch={}/{} global_step={} '
                        'loss={} sR={} sI={} sM={} loss_a={} loss_b={} loss_wrong={} '
                        'count_a={} count_b={} count_wrong={} condition_mode={}',
                        self.current_epoch + 1, self.hparams.epochs, batch_idx + 1,
                        self.trainer.num_training_batches, self.global_step,
                        value(stats['loss']), value(stats['sR']), value(stats['sI']), value(stats['sM']),
                        value(stats['a']['loss']), value(stats['b']['loss']), value(stats['wrong']['loss']),
                        stats['a']['count'], stats['b']['count'], stats['wrong']['count'],
                        self._condition_mode('train'))
        if batch_idx == 0 and self.global_rank == 0:
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs
            allocated = torch.cuda.max_memory_allocated(self.device)/2**30 if self.device.type == 'cuda' else 0.
            reserved = torch.cuda.max_memory_reserved(self.device)/2**30 if self.device.type == 'cuda' else 0.
            logger.info('TSE_TRAIN_READY epoch={}/{} global_step={} condition_mode={} '
                        'checkpointing={} backward={} loss={:.6f} peak_allocated_gib={:.3f} '
                        'peak_reserved_gib={:.3f}', self.current_epoch + 1, self.hparams.epochs,
                        self.global_step, self._condition_mode('train'),
                        self.model.gradient_checkpointing,
                        self.hparams.get('tse_condition_backward', 'combined'),
                        float(loss.detach()), allocated, reserved)

    def _condition_prediction(self, mixture, condition, target):
        """Return a waveform in the input amplitude scale and the training losses."""
        predictions = self.forward(dict(mixture_waveform=mixture, condition=condition))
        losses = self.loss(predictions, dict(target_waveform=target, mixture_waveform=mixture))
        return predictions['target_waveform'], losses

    def _condition_step(self, mixture, condition, target, name, weight, overlap, phase, selection):
        waveform, losses = self._condition_prediction(mixture, condition, target)
        if not all(torch.isfinite(v).all() for v in losses.values()):
            raise FloatingPointError(f'Nonfinite {name} RI loss')
        self.log_dict({f'{phase}_loss_{name}': losses['loss'].detach(),
                       **{f'{phase}_{name}_{key}': losses[key] for key in ('sR', 'sI', 'sM')}},
                      on_step=phase == 'train', on_epoch=True, sync_dist=False,
                      batch_size=mixture.shape[0])
        if name in ('correct', 'selected'):
            labels = ('a', 'b') if name == 'correct' else ('a', 'b', 'wrong')
            masks = [(label, selection == index) for index, label in enumerate(labels)]
        else:
            masks = [(name, torch.ones_like(overlap))]
        for label, mask in masks:
            count = int(mask.sum())
            if count:
                per_condition = losses['per_example_loss'][mask]
                self._metric_sums[phase].update(label, waveform[mask],
                    target[mask], mixture[mask], per_condition, overlap[mask])
                mean = per_condition.mean().detach()
                if name in ('correct', 'selected'):
                    self.log(f'{phase}_loss_{label}', mean,
                             on_step=True, on_epoch=True, sync_dist=False, batch_size=count)
                if phase == 'train' and self.hparams.get('tse_log_batch_loss', False):
                    self._train_batch_stats[label] = dict(loss=mean, count=count)
        if phase == 'train' and self.hparams.get('tse_log_batch_loss', False):
            for key in ('sR', 'sI', 'sM'):
                self._train_batch_stats[key] += weight * losses[key].detach()
        objective = weight * losses['per_example_loss'].detach()
        loss = weight * losses['loss']
        if phase == 'train' and not self.automatic_optimization:
            self.manual_backward(loss)
            loss = loss.detach()
        # The whole graph for this condition goes out of scope after its backward.
        return loss, objective

    def _waveform_step(self, batch, phase):
        inputs, targets = batch
        mixture = unpack_waveform(inputs['mixture_waveform'])
        overlap = torch.as_tensor(targets['overlap'], dtype=torch.bool, device=mixture.device)
        if phase not in self._metric_sums:
            self._reset_metrics(phase)
        mode = self._condition_mode(phase)
        selection = None
        if mode == 'sampled_correct':
            selection = torch.randint(2, (mixture.shape[0],), device=mixture.device)
            names = ('correct',)
        elif mode == 'sampled_correct_wrong':
            selection = (torch.rand(mixture.shape[0], device=mixture.device) >= 0.5).long()
            names = ('correct', 'wrong')
        elif mode == 'sampled_one':
            selection = torch.randint(3, (mixture.shape[0],), device=mixture.device)
            names = ('selected',)
        else:
            names = ('a', 'b', 'wrong')
        if phase == 'train' and self.hparams.get('tse_log_batch_loss', False):
            self._train_batch_stats = {name: dict(count=0, loss=None) for name in ('a', 'b', 'wrong')}
            self._train_batch_stats.update(sR=0., sI=0., sM=0.)
        total, objectives = [], []
        for name in names:
            if name in ('correct', 'selected'):
                labels = ('a', 'b') if name == 'correct' else ('a', 'b', 'wrong')
                rows = torch.arange(mixture.shape[0], device=mixture.device)
                condition = torch.stack([unpack_waveform(inputs[f'condition_{label}'])
                                         for label in labels], dim=1)[rows, selection]
                target = torch.stack([unpack_waveform(targets[f'target_{label}'])
                                      for label in labels], dim=1)[rows, selection]
                weight = 2 / 3 if mode == 'sampled_correct_wrong' else 1.
            else:
                condition = unpack_waveform(inputs[f'condition_{name}'])
                target = unpack_waveform(targets[f'target_{name}'])
                weight = 1 / 3
            loss, objective = self._condition_step(mixture, condition, target, name, weight,
                                                   overlap, phase, selection)
            total.append(loss)
            objectives.append(objective)
        loss = torch.stack(total).sum()
        self._metric_sums[phase].update_objective(torch.stack(objectives).sum(0), overlap)
        self.log(f'{phase}_loss', loss.detach(), on_step=phase == 'train', on_epoch=True,
                 sync_dist=False, batch_size=mixture.shape[0])
        if phase == 'train' and self.hparams.get('tse_log_batch_loss', False):
            self._train_batch_stats['loss'] = loss.detach()
        return loss

    def on_validation_epoch_end(self):
        report = self._finish_metrics('val')
        if (not self.automatic_optimization and not self.trainer.sanity_checking
                and self.trainer.state.fn == 'fit' and report is not None):
            # Lightning does not automatically step schedulers in manual mode.
            self.lr_schedulers().step(report['groups']['all']['loss'])

    def _finish_metrics(self, phase):
        metric = self._metric_sums.get(phase)
        if metric is None:
            return
        mode = self._condition_mode(phase)
        report = dict(epoch=self.current_epoch, groups=metric.compute(), condition_mode=mode,
            objective=('RI_L1_Mag_L1_selected_correct; P(A)=P(B)=1/2; no wrong training'
                       if mode == 'sampled_correct' else
                       'RI_L1_Mag_L1_selected_condition; P(A)=P(B)=P(wrong)=1/3'
                       if mode == 'sampled_one' else
                       '(2 * RI_L1_Mag_L1_selected_correct + RI_L1_Mag_L1_wrong) / 3'
                       if mode == 'sampled_correct_wrong' else
                       '(RI_L1_Mag_L1_A + RI_L1_Mag_L1_B + RI_L1_Mag_L1_wrong) / 3'),
            energy_ratio_definition='mean(output**2) / mean(mixture**2); arithmetic mean over samples',
            si_snr_definition='existing zero-mean SI-SDR; SI-SNRi relative to the same mixture/GT')
        if not self.trainer.sanity_checking:
            self._write_report(f'{phase}_tse_dual_metrics.json', report)
            if self.global_rank == 0:
                groups = report['groups']
                value = lambda x: 'n/a' if x is None else f'{x:.6f}'
                logger.info(
                    'TSE_EPOCH epoch={}/{} phase={} loss={} loss_a={} loss_b={} '
                    'loss_wrong={} samples={} '
                    'count_a={} count_b={} count_wrong={} condition_mode={}',
                    self.current_epoch + 1, self.hparams.epochs, phase,
                    value(groups['all']['loss']), value(groups['all']['a']['loss']),
                    value(groups['all']['b']['loss']), value(groups['all']['wrong']['loss']),
                    groups['all']['sample_count'], groups['all']['a']['count'],
                    groups['all']['b']['count'], groups['all']['wrong']['count'], mode)
            for group, conditions in report['groups'].items():
                for name in ('a', 'b', 'wrong'):
                    for key, value in conditions[name].items():
                        if value is not None:
                            self.log(f'{phase}_{group}_{name}_{key}', value, sync_dist=False)
        self._reset_metrics(phase)
        return report
