"""Small, fixed-CV audio exports and complete epoch component logs for a pilot."""
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time

from loguru import logger
import numpy as np
from pytorch_lightning import Callback
import soundfile as sf
import torch

from utils.tse_dual_metrics import measure


def _tensor(value):
    return value[0] if isinstance(value, (tuple, list)) else value


def _scalar(value):
    return float(value.detach().float().cpu()) if isinstance(value, torch.Tensor) else float(value)


def _json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temporary.replace(path)


class TSEPilotCallback(Callback):
    """Append through ``train.main(..., extra_callbacks=[callback])``.

    ``run_root`` is the explicitly chosen experiment directory, independent of
    TensorBoard's version subdirectory. Construction performs no I/O or inference.
    Training component aggregation requires ``tse_log_batch_loss: true``.
    """

    def __init__(self, run_root, view_dir=None, max_examples=2, export_initial=True):
        super().__init__()
        self.root = Path(run_root).resolve()
        self.view = Path(view_dir).absolute() if view_dir else None
        self.project = Path(__file__).resolve().parents[1]
        if self.project/'experiments' not in self.root.parents:
            raise ValueError('Pilot artifacts must be inside shared PROJECT/experiments')
        if max_examples not in (1, 2):
            raise ValueError('Export only one or two fixed CV mixtures')
        self.max_examples, self.export_initial = max_examples, export_initial
        self.examples, self.example_metadata = [], []
        self.train_sums, self.train_count = {}, 0
        self.val_count, self.val_metrics = 0, None
        self.initial_exported = False
        self.run_prepared = False

    def prepare_run(self, hparams):
        """Reserve a fresh run before TensorBoard creates its version directory."""
        if self.root.exists():
            if not self.root.is_dir():
                raise ValueError('Pilot run root must be a directory')
            existing = list(self.root.iterdir())
            if any(path.name != 'run.log' for path in existing):
                raise ValueError('Pilot output already contains artifacts; choose a fresh save_name')
            if existing and (existing[0].is_symlink() or not existing[0].is_file()):
                raise ValueError('An existing run.log must be a real file, not a symlink')
        config = dict(hparams) if isinstance(hparams, dict) else vars(hparams).copy()
        self.root.mkdir(parents=True, exist_ok=True)
        process = dict(pid=os.getpid(), host=socket.gethostname(),
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            slurm_job_id=os.environ.get('SLURM_JOB_ID'), created_unix=time.time())
        # Exclusive creation closes the race between two manual starts. Keep the
        # marker after completion so a previous run can never be silently reused.
        with (self.root/'.pilot_created').open('x') as handle:
            handle.write(json.dumps(process, allow_nan=False)+'\n')
        try:
            self._prepare()
            _json(self.root/'effective_config.json', config)
            files = (
                'train.py', 'tools/submit_training.py', 'utils/tse_pilot_callback.py',
                'utils/tse_config.py', 'utils/tse_hierarchy_config.py', 'utils/training_config.py',
                'utils/tse_dual_metrics.py', 'utils/backtracking_model_checkpoint.py',
                'utils/mytensorboardlogger.py', 'utils/collate.py',
                'creators/dataset.py', 'creators/model.py', 'creators/task.py', 'creators/loss.py',
                'tse_separator.py', 'tse_dual_separator.py', 'source_separator.py',
                'loss_functions/tse_ri_loss.py', 'loss_functions/tse_loss.py',
                'nets/tse_dprnntf.py', 'nets/dprnntf_original/tfgridnet.py',
                'nets/dprnntf_original/audio_utils_torch.py', 'nets/dprnntf_original/normalization.py',
                'nets/dprnntf_original/conv_block.py', 'nets/dprnntf_original/activation.py',
                'datasets/tse_hierarchy_dataset.py', 'datasets/tse_hierarchy_labels.py',
                'datasets/tse_dual_mix.py', 'datasets/tse_ontology.py', 'datasets/mix_utils.py',
                'datasets/active_rms.py', 'datasets/fsd50k_cache.py', 'datasets/tse_dataset.py',
                'requirements.txt')
            hashes = {name: hashlib.sha256((self.project/name).read_bytes()).hexdigest() for name in files}
            revision = subprocess.run(['git', 'rev-parse', '--verify', 'HEAD'], cwd=self.project,
                capture_output=True, text=True, timeout=10)
            _json(self.root/'provenance.json', dict(project=str(self.project), output=str(self.root),
                git_revision=revision.stdout.strip() if revision.returncode == 0 else None,
                git_revision_error=revision.stderr.strip() if revision.returncode else None,
                source_sha256=hashes, effective_config_sha256=hashlib.sha256(
                    (self.root/'effective_config.json').read_bytes()).hexdigest(),
                process=process, code_copied=False))
            _json(self.root/'status.json', dict(status='prepared', stage='prepared',
                output=str(self.root), epochs=int(config['epochs']), completed_epochs=0,
                global_step=0, updated_unix=time.time(), **process))
            self.run_prepared = True
        except BaseException as error:
            _json(self.root/'status.json', dict(status='failed', stage='preparation',
                error_type=type(error).__name__, error=str(error), updated_unix=time.time(), **process))
            raise
        return self

    def _prepare(self):
        self.root.mkdir(parents=True, exist_ok=True)
        audio = self.root/'audio'
        if audio.is_symlink():
            raise ValueError('Experiment audio output must be a real directory')
        audio.mkdir(exist_ok=True)
        if self.view is None:
            return
        if self.view.parent != self.project/'data' or self.view.is_symlink():
            raise ValueError('The data view must be a real directory directly under PROJECT/data')
        self.view.mkdir(parents=True, exist_ok=True)
        if set(path.name for path in self.view.iterdir()) - {'run.log', 'audio'}:
            raise ValueError('The data view may contain only run.log and audio links')
        for name, target in (('run.log', self.root/'run.log'), ('audio', audio)):
            link = self.view/name
            if link.is_symlink():
                if link.resolve() != target.resolve():
                    raise ValueError(f'Unexpected {name} target in the data view')
            elif link.exists():
                raise ValueError(f'The data view {name} must be a symlink')
            else:
                link.symlink_to(target, target_is_directory=name == 'audio')

    def on_fit_start(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        if not self.run_prepared:
            raise ValueError('Call pilot_callback.prepare_run(hparams) before creating the logger')
        self._prepare()
        if not pl_module.hparams.get('tse_log_batch_loss', False):
            raise ValueError('Pilot component logging requires tse_log_batch_loss=true')

    def _progress(self, trainer, **changes):
        path = self.root/'status.json'
        state = json.loads(path.read_text()) if path.exists() else {}
        state.update(status='running', stage='training', global_step=int(trainer.global_step),
                     updated_unix=time.time())
        state.update(changes)
        _json(path, state)

    def on_fit_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        path = self.root/'status.json'
        if path.exists() and json.loads(path.read_text()).get('status') == 'failed':
            return
        complete = int(trainer.current_epoch) >= int(pl_module.hparams.epochs) and not trainer.interrupted
        self._progress(trainer, status='completed' if complete else 'stopped', stage='finished',
            completed_epochs=int(trainer.current_epoch),
            note='Completion describes the bounded run, not a claim that separation has converged')

    def on_exception(self, trainer, pl_module, exception):
        if not trainer.is_global_zero:
            return
        interrupted = isinstance(exception, (KeyboardInterrupt, SystemExit))
        self._progress(trainer, status='stopped' if interrupted else 'failed',
            stage='interrupted' if interrupted else 'failed',
            error_type=type(exception).__name__, error=str(exception))

    def _save_wave(self, filename, wave):
        path = self.root/'audio'/filename
        values = np.asarray(wave, dtype=np.float32)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError('Expected a finite mono waveform for the pilot audio export')
        if path.exists() or path.is_symlink():
            previous, rate = sf.read(str(path), dtype='float32')
            if path.is_symlink() or rate != 16000 or not np.array_equal(previous, values):
                raise ValueError(f'Refusing to overwrite existing pilot audio: {filename}')
            return
        sf.write(str(path), values, 16000, subtype='FLOAT')

    def _capture(self, batch):
        inputs, targets = batch
        count = min(self.max_examples, _tensor(inputs['mixture_waveform']).shape[0])
        examples, metadata = [], []
        for row in range(count):
            copy = lambda value: _tensor(value)[row].detach().float().cpu().clone()
            example = dict(mixture=copy(inputs['mixture_waveform']),
                conditions={name: copy(inputs[f'condition_{name}']) for name in ('a', 'b', 'wrong')},
                targets={name: copy(targets[f'target_{name}']) for name in ('a', 'b', 'wrong')})
            if 'background_waveform' in targets:
                example['background'] = copy(targets['background_waveform'])
            mids = {name: targets[f'mid_{name}'][row] for name in ('a', 'b', 'wrong')}
            arrays = dict(mixture=example['mixture'], **example['targets'])
            if 'background' in example:
                arrays['background'] = example['background']
            hashes = {name: hashlib.sha256(value.numpy().tobytes()).hexdigest()
                      for name, value in arrays.items()}
            item = dict(example_index=row, cv_batch_index=0, cv_batch_row=row,
                utt_id=targets['utt_id'][row], mids=mids, waveform_sha256=hashes,
                condition_indices={name: int(value.argmax()) for name, value in example['conditions'].items()})
            example['metadata'] = item
            examples.append(example)
            metadata.append(item)
        if self.examples:
            if metadata != self.example_metadata:
                raise ValueError('The fixed CV examples changed between validation passes')
            return
        manifest = dict(scope='First CV batch only; these examples are not the full validation set',
            selection='First one or two rows; adjacent rows can share a background with different overlap',
            samplerate=16000, amplitude='Original shared mixture/GT scale; no per-file normalization',
            examples=metadata)
        path = self.root/'fixed_validation_examples.json'
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError('Existing fixed validation example manifest differs')
        self.examples, self.example_metadata = examples, metadata
        _json(path, manifest)
        for index, example in enumerate(examples):
            self._save_wave(f'sample_{index:02d}_mixture.wav', example['mixture'].numpy())
            if 'background' in example:
                self._save_wave(f'sample_{index:02d}_background.wav', example['background'].numpy())
            for name in ('a', 'b', 'wrong'):
                mid = re.sub(r'[^A-Za-z0-9_-]', '_', example['metadata']['mids'][name])
                self._save_wave(f'sample_{index:02d}_gt_{name.upper()}_{mid}.wav', example['targets'][name].numpy())

    def on_validation_epoch_start(self, trainer, pl_module):
        self.val_count = 0

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not trainer.is_global_zero or dataloader_idx != 0:
            return
        if batch_idx == 0:
            self._capture(batch)
        self.val_count += int(_tensor(batch[0]['mixture_waveform']).shape[0])

    def _export(self, trainer, pl_module, epoch, initial=False):
        if not self.examples:
            return
        step = int(trainer.global_step)
        tag = f'epoch_{epoch:03d}_step_{step:08d}'+('_initial' if initial else '')
        journal = self.root/'audio_metrics.jsonl'
        if journal.exists():
            for line in journal.read_text().splitlines():
                if json.loads(line)['tag'] == tag:
                    return
        was_training = pl_module.training
        pl_module.eval()
        device = pl_module.device
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
        cases = []
        try:
            with torch.random.fork_rng(devices=devices), torch.inference_mode(), trainer.precision_plugin.forward_context():
                for index, example in enumerate(self.examples):
                    mixture = example['mixture'][None].to(device)
                    for name in ('a', 'b', 'wrong'):
                        prediction = pl_module(dict(mixture_waveform=mixture,
                            condition=example['conditions'][name][None].to(device)))
                        losses = pl_module.loss(prediction, dict(mixture_waveform=mixture,
                            target_waveform=example['targets'][name][None].to(device)))
                        wave = prediction['target_waveform'][0].detach().float().cpu().numpy()
                        row = measure(wave, example['targets'][name].numpy(),
                            example['mixture'].numpy(), wrong=name == 'wrong')
                        row.update(example_index=index, utt_id=example['metadata']['utt_id'], condition=name,
                            **{key: _scalar(losses[key]) for key in ('loss', 'sR', 'sI', 'sM')})
                        cases.append(row)
                        mid = re.sub(r'[^A-Za-z0-9_-]', '_', example['metadata']['mids'][name])
                        self._save_wave(f'sample_{index:02d}_{tag}_pred_{name.upper()}_{mid}.wav', wave)
                        del prediction, losses
        finally:
            pl_module.train(was_training)
        correct, wrong = [row for row in cases if row['condition'] != 'wrong'], [row for row in cases if row['condition'] == 'wrong']
        valid_scores = [row['si_sdri_db'] for row in correct if row['si_sdri_db'] is not None]
        summary = dict(mixtures=len(self.examples), queries=len(cases),
            **{key: float(np.mean([row[key] for row in cases])) for key in ('loss', 'sR', 'sI', 'sM')},
            positive_valid_queries=len(valid_scores),
            positive_si_sdri_db=float(np.mean(valid_scores)) if valid_scores else None,
            wrong_output_to_mixture_energy=float(np.mean([row['output_to_mixture_energy'] for row in wrong])))
        report = dict(tag=tag, epoch=epoch, global_step=step, initial=initial,
            scope='Fixed one/two CV examples only, not aggregate validation metrics',
            summary=summary, cases=cases)
        with journal.open('a') as handle:
            handle.write(json.dumps(report, ensure_ascii=False, allow_nan=False)+'\n')
        logger.info('PILOT_AUDIO_EVAL {}', json.dumps({key: value for key, value in report.items() if key != 'cases'}, ensure_ascii=False))

    def on_sanity_check_end(self, trainer, pl_module):
        if trainer.is_global_zero and self.export_initial and trainer.global_step == 0 and self.examples:
            self._export(trainer, pl_module, epoch=0, initial=True)
            self.initial_exported = True

    def on_train_start(self, trainer, pl_module):
        if trainer.is_global_zero:
            self._progress(trainer, completed_epochs=0)
        if trainer.is_global_zero and self.export_initial and trainer.global_step == 0 and not self.initial_exported:
            logger.info('PILOT_AUDIO_BASELINE skipped: no fixed examples were obtained from sanity validation')

    def on_train_epoch_start(self, trainer, pl_module):
        self.train_sums = {key: 0. for key in ('loss', 'sR', 'sI', 'sM')}
        self.train_count, self.val_metrics = 0, None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not trainer.is_global_zero:
            return
        count = int(_tensor(batch[0]['mixture_waveform']).shape[0])
        stats = pl_module._train_batch_stats
        for key in self.train_sums:
            self.train_sums[key] += _scalar(stats[key])*count
        self.train_count += count

    def on_validation_end(self, trainer, pl_module):
        if not trainer.is_global_zero or trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        values = dict(loss=_scalar(metrics['val_loss']), mixtures=self.val_count)
        for key in ('sR', 'sI', 'sM'):
            values[key] = sum(_scalar(metrics[f'val_{name}_{key}']) for name in ('a', 'b', 'wrong'))/3
        values['conditions'] = {}
        for name in ('a', 'b', 'wrong'):
            condition = {key: _scalar(metrics[f'val_{name}_{key}']) for key in ('sR', 'sI', 'sM')}
            for key in ('loss', 'count', 'valid_references', 'si_snr_db', 'si_snri_db',
                        'output_rms', 'energy_ratio'):
                metric_name = f'val_all_{name}_{key}'
                if metric_name in metrics:
                    condition[key] = _scalar(metrics[metric_name])
            values['conditions'][name] = condition
        self.val_metrics = dict(epoch=int(trainer.current_epoch)+1, **values)
        self._export(trainer, pl_module, epoch=int(trainer.current_epoch)+1)

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        epoch = int(trainer.current_epoch)+1
        if not self.train_count or self.val_metrics is None or self.val_metrics['epoch'] != epoch:
            raise ValueError('Missing current-epoch train/validation statistics for pilot summary')
        record = dict(epoch=epoch, global_step=int(trainer.global_step),
            train=dict(mixtures=self.train_count, **{key: value/self.train_count for key, value in self.train_sums.items()}),
            val={key: value for key, value in self.val_metrics.items() if key != 'epoch'},
            scope='All train/validation batches processed this epoch, subject to configured batch limits')
        with (self.root/'epoch_metrics.jsonl').open('a') as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
        self._progress(trainer, completed_epochs=epoch)
        logger.info('PILOT_EPOCH {}', json.dumps(record, ensure_ascii=False))
