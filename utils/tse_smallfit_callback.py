"""Step-based metrics, fixed-gain listening files and comparable spectrograms."""
import csv
import json
from pathlib import Path
import time

from loguru import logger
import numpy as np
import soundfile as sf
from pytorch_lightning import Callback


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def summarize_scores(rows, loss_conditions=('a', 'b', 'wrong')):
    if not rows:
        raise ValueError('Empty fixed-mixture evaluation')
    if tuple(loss_conditions) not in (('a', 'b'), ('a', 'b', 'wrong')):
        raise ValueError('Smallfit loss must average A/B or A/B/wrong')
    mean = lambda items, key: float(np.mean([row[key] for row in items]))
    objective = [row for row in rows if row['condition'] in loss_conditions]
    correct = [row for row in rows if row['condition'] in ('a', 'b')]
    result = dict(mixtures=len(rows) // 3, queries=len(rows), loss=mean(objective, 'loss'),
        loss_conditions=list(loss_conditions), loss_queries=len(objective),
        correct_loss=mean(correct, 'loss'), all_conditions_loss=mean(rows, 'loss'), conditions={})
    for key in ('oracle_loss', 'zero_loss', 'mixture_loss'):
        if key in objective[0]:
            result[key] = mean(objective, key)
    for name in ('a', 'b', 'wrong'):
        items = [row for row in rows if row['condition'] == name]
        values = dict(count=len(items), **{key: mean(items, key) for key in
            ('loss', 'sR', 'sI', 'sM', 'output_rms', 'output_to_mixture_energy')})
        if name != 'wrong':
            valid = [row for row in items if row['si_sdri_db'] is not None]
            values.update(valid_references=len(valid),
                si_sdri_db=mean(valid, 'si_sdri_db') if valid else None,
                si_sdri_median_db=float(np.median([row['si_sdri_db'] for row in valid])) if valid else None,
                improved_fraction=mean([dict(improved=row['si_sdri_db'] > 0) for row in valid], 'improved') if valid else None,
                output_to_target_energy=mean(items, 'output_to_target_energy'),
                zero_outputs=sum(row['metric_status'] == 'zero_output' for row in items))
        for key in ('oracle_loss', 'zero_loss', 'mixture_loss'):
            if key in items[0]:
                values[key] = mean(items, key)
        result['conditions'][name] = values
    result['groups'] = {}
    for overlap in (False, True):
        items = [row for row in rows if row['overlap'] == overlap]
        selected = [row for row in items if row['condition'] in loss_conditions]
        positive = [row for row in items if row['condition'] != 'wrong' and row['si_sdri_db'] is not None]
        result['groups']['overlap' if overlap else 'nonoverlap'] = dict(queries=len(items),
            loss=mean(selected, 'loss') if selected else None,
            all_conditions_loss=mean(items, 'loss') if items else None,
            si_sdri_db=mean(positive, 'si_sdri_db') if positive else None)
    result['a_b_output_correlation'] = mean(rows, 'a_b_output_correlation')
    return result


def spectrogram(path, waves, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.signal import stft

    figure, axes = plt.subplots(3, 2, figsize=(14, 8), constrained_layout=True)
    for axis, (name, wave) in zip(axes.flat, waves.items()):
        frequency, seconds, spectrum = stft(wave, fs=16000, nperseg=512, noverlap=384)
        db = 20 * np.log10(np.maximum(np.abs(spectrum), 1e-5))
        plot = axis.pcolormesh(seconds, frequency / 1000, db, vmin=-100, vmax=0,
                               cmap='magma', shading='auto', rasterized=True)
        axis.set(title=name, xlabel='Time (s)', ylabel='Frequency (kHz)', xlim=(0, len(wave) / 16000), ylim=(0, 8))
    figure.suptitle(title + ' | fixed amplitude scale, no per-file normalization')
    figure.colorbar(plot, ax=axes.ravel().tolist(), label='Magnitude (dB re 1)', shrink=.8)
    figure.savefig(path, dpi=110)
    plt.close(figure)


class TSESmallfitCallback(Callback):
    def __init__(self, root):
        self.root = Path(root)

    def status(self, trainer, status, **extra):
        write_json(self.root/'status.json', dict(status=status, global_step=int(trainer.global_step),
            max_steps=trainer.max_steps, updated_unix=time.time(), **extra))

    def on_fit_start(self, trainer, pl_module):
        # Check plotting dependencies before spending the training budget.
        import matplotlib
        matplotlib.use('Agg')
        summary = json.loads((self.root/'fixed_data/summary.json').read_text())
        self.class_names = {row['mid']: row['name'] for row in summary['classes']}
        self.status(trainer, 'preparing')

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % 50 == 0:
            self.status(trainer, 'training')

    def on_validation_start(self, trainer, pl_module):
        self.status(trainer, 'evaluating')

    def on_validation_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        step = int(trainer.global_step)
        directory = self.root/'evaluations'/f'step_{step:08d}'
        directory.mkdir(parents=True, exist_ok=False)
        report = dict(global_step=step, epoch=int(trainer.current_epoch), initial=trainer.sanity_checking,
            learning_rate=trainer.optimizers[0].param_groups[0]['lr'],
            inspection_point=step in (1000, 2000), **pl_module.eval_summary)
        write_json(directory/'metrics.json', report)
        with (self.root/'step_metrics.jsonl').open('a') as handle:
            handle.write(json.dumps(report, allow_nan=False) + '\n')
        for phase, rows in pl_module.eval_rows.items():
            fields = list(dict.fromkeys(key for row in rows for key in row))
            with (directory/f'{phase}_scores.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            dataset = pl_module.datasets['tr' if phase == 'train_eval' else 'cv']
            for index, predictions in pl_module.eval_audio[phase].items():
                inputs, targets = dataset[index]
                shared = self.root/'listening'/phase/f'sample_{index:03d}'
                shared.mkdir(parents=True, exist_ok=True)
                sources = dict(mixture=inputs['mixture_waveform'], gt_a=targets['target_a'],
                               gt_b=targets['target_b'], gt_wrong=targets['target_wrong'],
                               background=targets['background_waveform'])
                if step == 0:
                    for name, wave in sources.items():
                        sf.write(str(shared/f'{name}.wav'), wave, 16000, subtype='FLOAT')
                    write_json(shared/'sample.json', dict(index=index, utt_id=targets['utt_id'],
                        overlap=bool(targets['overlap']), mids={name: targets[f'mid_{name}'] for name in ('a', 'b', 'wrong')}))
                output = shared/f'step_{step:08d}'
                output.mkdir()
                for name, wave in predictions.items():
                    sf.write(str(output/f'pred_{name}.wav'), wave, 16000, subtype='FLOAT')
                name_a = self.class_names[targets['mid_a']]
                name_b = self.class_names[targets['mid_b']]
                name_wrong = self.class_names[targets['mid_wrong']]
                waves = {'Mixture': sources['mixture'], f'Target A: {name_a}': sources['gt_a'],
                    f'Prediction A: {name_a}': predictions['a'], f'Target B: {name_b}': sources['gt_b'],
                    f'Prediction B: {name_b}': predictions['b'], f'Absent query: {name_wrong}': predictions['wrong']}
                spectrogram(output/'spectrogram.png', waves, f'{phase} sample {index} step {step}')
        if not trainer.sanity_checking:
            checkpoints = self.root/'checkpoints'
            checkpoints.mkdir(exist_ok=True)
            trainer.save_checkpoint(checkpoints/f'step_{step:08d}.ckpt')
        scalars = {'smallfit/learning_rate': report['learning_rate']}
        for phase, summary in pl_module.eval_summary.items():
            for key in ('loss', 'correct_loss', 'all_conditions_loss', 'oracle_loss', 'zero_loss', 'mixture_loss'):
                if key in summary:
                    scalars[f'smallfit/{phase}/{key}'] = summary[key]
            for name, values in summary['conditions'].items():
                for key, value in values.items():
                    if value is not None:
                        scalars[f'smallfit/{phase}/{name}/{key}'] = value
        trainer.logger.log_metrics(scalars, step=step)
        logger.info('SMALLFIT_EVAL {}', json.dumps(report, allow_nan=False))
        self.status(trainer, 'training', last_evaluation_step=step,
                    inspection_point=report['inspection_point'])

    def on_fit_end(self, trainer, pl_module):
        complete = trainer.global_step == trainer.max_steps and not trainer.interrupted
        self.status(trainer, 'completed' if complete else 'stopped')

    def on_exception(self, trainer, pl_module, exception):
        self.status(trainer, 'failed', error=f'{type(exception).__name__}: {exception}')
