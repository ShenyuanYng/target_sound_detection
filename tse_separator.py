"""TSE Lightning task using the project's optimizer, loader and forward adapters.

SourceSeparator contains local optimizer and loader primitives only.
This task owns all waveform losses, metrics and lifecycle hooks.
"""
from argparse import Namespace
from pathlib import Path
import json

import pytorch_lightning as pl
import torch
from loguru import logger

from source_separator import SourceSeparator
from creators.collate import collate_creator
from creators.dataset import datapath_creator, dataset_creator
from creators.loss import loss_creator
from creators.model import model_creator
from creators.task import task_creator
from datasets.train_split import video_id
from loss_functions.tse_loss import si_sdr, unpack_waveform
from utils.tse_config import validate_tse_config


class TSESourceSeparator(SourceSeparator):
    def __init__(self, hparams):
        # Reuse shared methods without instantiating any SED metrics or models.
        pl.LightningModule.__init__(self)
        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)
        validate_tse_config(hparams) #检查当前参数组合是否合法
        self.save_hyperparameters(hparams)#保存配置供训练和Checkpoint使用
        self.model = model_creator(hparams)
        self.loss = loss_creator(hparams)
        self.task = task_creator(hparams)
        self.collate_fn = collate_creator(hparams)#创建dataloader使用的组批函数
        self.split_paths, self.datasets = None, {}
        self._metric_sums = {}
        self._condition_counts = {}

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, map_location='cpu', strict=True,
                             weights_only=False, hparams=None):
        # The shared project saves a flat Namespace, whereas Lightning's automatic
        # constructor inference expects a named hparams argument for this class.
        saved = torch.load(checkpoint_path, map_location=map_location, weights_only=weights_only)
        model = cls(saved['hyper_parameters'] if hparams is None else hparams)
        model.on_load_checkpoint(saved)
        model.load_state_dict(saved['state_dict'], strict=strict)
        return model

    def setup(self, stage=None):
        if stage not in (None, 'fit', 'validate'):
            raise ValueError('Single-target TSE supports fit/validate only')
        if self.split_paths is None:
            self.split_paths = {p: datapath_creator(self.hparams, p) for p in ('tr', 'cv')}
            videos = {p: {video_id(e) for e in entries} for p, entries in self.split_paths.items()}
            if videos['tr'] & videos['cv']:
                raise ValueError('TSE Train/CV share background videos')

    def _build_data_loader(self, partition):
        from torch.utils.data import DataLoader
        if partition not in self.datasets:
            from datasets.tse_dataset import NoLegalPairError
            try:
                self.datasets[partition] = dataset_creator(self.hparams, partition,
                    path_list=self.split_paths[partition])
            except NoLegalPairError as error:
                self._write_report(f'dataset_{partition}_failed.json', error.report)
                raise
        dataset = self.datasets[partition]
        if len(self.datasets) == 2:
            from tools.audit_common import leakage_report
            # Includes source IDs, paths, and byte-identical cross-split/corpus audio.
            if not getattr(self, '_leakage_checked', False):
                report = leakage_report(self.datasets['tr'], self.datasets['cv'], full=True)
                if report['status'] != 'passed':
                    raise ValueError('TSE source leakage: ' + json.dumps(report))
                self._leakage_checked = True
                self._write_report('source_leakage.json', report)
            missing = (self.datasets['cv'].report['eligible_condition_background_counts'].keys()
                       - self.datasets['tr'].report['eligible_condition_background_counts'].keys())
            self.datasets['cv'].report['cv_mids_without_legal_train_support'] = sorted(missing)
            self._write_report('dataset_cv.json', self.datasets['cv'].report)
        self._write_report(f'dataset_{partition}.json', dataset.report)
        logger.info('TSE {}: {} backgrounds, {} eligible classes, multilabel pool fraction={:.3f}',
                    partition, len(dataset.path_list), dataset.report['eligible_condition_class_count'],
                    dataset.report['multilabel_fsd_fraction'])
        #dataloader
        return DataLoader(dataset, batch_size=self.hparams.batch_size,
            collate_fn=self.collate_fn, shuffle=bool(self.hparams.shuffle) if partition == 'tr' else False,
            num_workers=self.hparams.num_workers, worker_init_fn=self._worker_init_fn,
            persistent_workers=self.hparams.num_workers > 0)

    def _write_report(self, name, report):
        if self.global_rank == 0:
            root = Path(self.hparams.save_path)
            root.mkdir(parents=True, exist_ok=True)
            (root / name).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')

    def on_train_epoch_start(self):
        self._reset_metrics('train')

    def on_validation_epoch_start(self):
        self._reset_metrics('val')

    def _reset_metrics(self, phase):
        self._metric_sums[phase] = torch.zeros(6, device=self.device, dtype=torch.float64)
        self._condition_counts[phase] = torch.zeros(self.hparams.num_classes, device=self.device, dtype=torch.long)

    def _waveform_step(self, batch, phase):
        """
        _waveform_step是训练循环的最小单元，接受collate组好的（input，targets），
        调用父类forward得到prediction，调用loss_creator创建的loss得到losses，用
        losses驱动反传，用 losses 的其余项 + si_sdr 在 no_grad 下累计可解释指标；
        只把标量 loss 交还 Lightning，指标交给 _finish_metrics 在 epoch 末落盘。
        
        
        解包batch，之后将输入前向传播得到模型的预测结果，之后进入loss
        """
        inputs, targets = batch
        predictions = self.forward(inputs)
        losses = self.loss(predictions, targets)
        #数值健康检查
        if not all(torch.isfinite(v).all() for v in losses.values()):
            raise FloatingPointError(f'Nonfinite TSE loss: {targets["utt_id"]}')
        #把损失写进lightning日志
        count = unpack_waveform(targets['target_waveform']).shape[0]
        self.log_dict({f'{phase}_{k}': v for k, v in losses.items()},
            on_step=phase == 'train', on_epoch=True, sync_dist=True, batch_size=count)
        #无梯度累积指标
        with torch.no_grad():
            target = unpack_waveform(targets['target_waveform'])
            #每个样本一个SI-SDR的值，valid 标出非静音、可评估的样本。
            score, valid = si_sdr(predictions['target_waveform'], target, self.hparams.tse_silence_rms)
            #不提取、直接拿混音当预测的SI-SDR,作为下届
            baseline, _ = si_sdr(unpack_waveform(inputs['mixture_waveform']), target,
                                 self.hparams.tse_silence_rms)
            if phase not in self._metric_sums:
                self._reset_metrics(phase)
            self._metric_sums[phase] += torch.stack((score[valid].double().sum(),
                (score - baseline)[valid].double().sum(), valid.double().sum(), (~valid).double().sum(),
                torch.tensor(sum(targets['multilabel_fsd']), device=self.device, dtype=torch.float64),
                torch.tensor(count, device=self.device, dtype=torch.float64)))
            condition = inputs['condition'][0] if isinstance(inputs['condition'], (tuple, list)) else inputs['condition']
            self._condition_counts[phase] += torch.bincount(condition.argmax(-1), minlength=self.hparams.num_classes)
        #返回给lightning的唯一标量
        return losses['loss']
    """
    返回给lightning之后lightning干了什么：
    1、training_strp返回标量loss
    2、lightning调用loss.backward（）进行反向传播
    3.触发钩子函数on_after_backward()
    4、optimizer.step（）+zero_grad（）优化器
    5.self.log_dict(),把train loss等值挂到lighting的logger上，epoch结束由on_train_epoch_end()->finish_metrics汇总写报告
    """

    def training_step(self, batch, batch_idx):
        return self._waveform_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._waveform_step(batch, 'val')

    def on_after_backward(self):
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.parameters()):
            raise FloatingPointError('Nonfinite TSE gradient')

    def _finish_metrics(self, phase):
        values = self._metric_sums.get(phase)
        if values is None:
            return
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(values)
            torch.distributed.all_reduce(self._condition_counts[phase])
        score, improvement, valid, silent, multilabel, total = values.tolist()
        report = dict(epoch=self.current_epoch, valid_samples=int(valid), silent_samples=int(silent),
            si_sdr=score / valid if valid else None, si_sdri=improvement / valid if valid else None,
            sampled_multilabel_fraction=multilabel / total if total else None,
            sampled_class_index_counts={str(i): int(n) for i, n in enumerate(
                self._condition_counts[phase].tolist()) if n})
        if not self.trainer.sanity_checking:
            if valid:
                self.log(f'{phase}_si_sdr', report['si_sdr'], sync_dist=False)
                self.log(f'{phase}_si_sdri', report['si_sdri'], sync_dist=False)
            self.log(f'{phase}_silent_samples', silent, sync_dist=False)
            self._write_report(f'{phase}_tse_metrics.json', report)
        self._reset_metrics(phase)

    def on_train_epoch_end(self):
        self._finish_metrics('train')

    def on_validation_epoch_end(self):
        self._finish_metrics('val')

    def on_load_checkpoint(self, checkpoint):
        saved = checkpoint['hyper_parameters']
        if saved.get('task_type') != 'tse':
            raise ValueError('Cannot resume TSE from an SED checkpoint')
        names = {k for k in set(saved) | set(self.hparams) if k.startswith(('mix_', 'tse_'))}
        names |= {'dataset', 'task_type', 'model_name', 'loss_type', 'feature_type', 'samplerate',
                  'num_classes', 'class_map_path', 'data_path', 'train_val_fraction',
                  'train_split_seed', 'rms_norm', 'train_on_chunk'}
        if self.hparams.dataset == 'audioset_strong_dual_fsd_wrong_tse':
            from utils.tse_runtime import RUNTIME_DEFAULTS
            # These change execution / the unbiased estimator, not model weights,
            # data, the RI objective or the exhaustive validation protocol.
            for name, default in RUNTIME_DEFAULTS.items():
                before, after = saved.get(name, default), self.hparams.get(name, default)
                if before != after:
                    logger.info('TSE resume override: {}={} -> {}', name, before, after)
            names -= RUNTIME_DEFAULTS.keys()
        for name in names:
            if saved.get(name) != self.hparams.get(name):
                raise ValueError(f'TSE checkpoint setting changed: {name}')
