"""Train standalone target sound extraction models."""

import argparse
import os.path
from pathlib import Path
import random

import loguru
import numpy as np
import pytorch_lightning as pl
import torch
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer

from utils.backtracking_model_checkpoint import BacktrackingModelCheckpointUpgrade
from utils.config import yaml_to_parser
from utils.mytensorboardlogger import MyTensorBoardLogger
from utils.training_config import validate_training_data_config
from utils.project_paths import training_path
from utils.run_artifacts import record_run


def trainer_device_config(hparams):
    """Return portable Lightning accelerator, device, and strategy settings."""
    accelerator = getattr(hparams, "accelerator", "auto")
    requested_devices = getattr(hparams, "devices", 1)
    if accelerator == "auto":
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    if accelerator == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("accelerator=gpu was requested, but CUDA is unavailable")

    if accelerator == "cpu":
        devices = 1 if requested_devices in (None, 0, "auto") else requested_devices
        strategy = "auto"
    else:
        devices = requested_devices
        if isinstance(devices, int) and devices <= 0:
            devices = 1
        strategy = getattr(hparams, "distributed_backend", "auto")
        if strategy in {"ddp", "ddp_find_unused_parameters_true"} and devices == 1:
            strategy = "auto"
    return accelerator, devices, strategy



#checkpoint保存和恢复
def _checkpoint_callback(save_path):
    return BacktrackingModelCheckpointUpgrade(
        dirpath=save_path,
        filename="_ckpt_epoch_{epoch}",
        auto_insert_metric_name=False,
        monitor="val_loss",
        save_last=True,
        save_top_k=1,
        save_weights_only=False,
        mode="min",
        every_n_epochs=1,
        verbose=False,
    )


def _resume_checkpoint(callback, run_dir):
    checkpoint_path = os.path.join(run_dir, "mylast.ckpt")
    if not os.path.exists(checkpoint_path):
        return None

    # Lightning restores model, optimizer, scheduler, loops and callback states
    # directly from this checkpoint. The historical .pt sidecar may be stale.
    loguru.logger.info("loading mylast.ckpt from {}", checkpoint_path)
    return checkpoint_path


def main(hparams, resume_run_dir=None, checkpoint_dir=None, extra_callbacks=()):
    hparams.save_path = str(training_path(hparams.save_path))
    validate_training_data_config(hparams)
    smallfit = getattr(hparams, 'tse_smallfit', False)
    if smallfit and resume_run_dir:
        raise ValueError('Fixed-mixture smallfit requires a fresh, single-allocation run')
    if hparams.seed is None:
        hparams.seed = random.getrandbits(16)
    hparams.cudnn_deterministic = bool(hparams.cudnn_deterministic)
    hparams.cudnn_benchmark = bool(hparams.cudnn_benchmark)
    cudnn.deterministic = hparams.cudnn_deterministic
    cudnn.benchmark = hparams.cudnn_benchmark

    #固定随机状态，让模型初始化，数据随机裁剪等行为尽量可重复
    random.seed(hparams.seed)
    np.random.seed(hparams.seed)
    torch.manual_seed(hparams.seed)
    torch.set_float32_matmul_precision("high")

    # Optional diagnostics for the full-pool pilot; the normal fit path is unchanged.
    if getattr(hparams, 'tse_pilot_artifacts', False):
        if hparams.dataset != 'audioset_strong_dual_fsd_hierarchy_tse_v2' or resume_run_dir:
            raise ValueError('Pilot artifacts require a fresh hierarchy-v2 run')
        from utils.tse_pilot_callback import TSEPilotCallback
        pilot_root = Path(hparams.save_path).resolve()/hparams.save_name
        view = Path(__file__).resolve().parent/'data'/hparams.save_name
        pilot_callback = TSEPilotCallback(pilot_root, view_dir=view,
                                         max_examples=2, export_initial=True)
        pilot_callback.prepare_run(hparams)
        extra_callbacks = (*extra_callbacks, pilot_callback)

    recorded_run = record_run(hparams, resume_run_dir=resume_run_dir)
    if smallfit:
        from utils.tse_smallfit_callback import TSESmallfitCallback
        extra_callbacks = (*extra_callbacks, TSESmallfitCallback(recorded_run))

    tensorboard = MyTensorBoardLogger(
        save_dir=hparams.save_path,
        name=hparams.save_name,
    )

    #创建日志目录
    hparams.save_path = os.path.join(
        hparams.save_path,
        tensorboard.name,
        f"version_{tensorboard.version}",
    )

    #检查是否续训
    run_dir = os.path.dirname(hparams.save_path)
    if resume_run_dir is not None:
        run_dir = resume_run_dir
    checkpoint = _checkpoint_callback(checkpoint_dir or hparams.save_path)
    resume_from = _resume_checkpoint(checkpoint, run_dir)
    if resume_from and resume_run_dir is None:
        raise ValueError('TSE new runs need a fresh output directory; use explicit resume for a TSE checkpoint')
    if resume_run_dir is not None and resume_from is None:
        raise FileNotFoundError(f"Resume checkpoint missing in {run_dir}")
    accelerator, devices, strategy = trainer_device_config(hparams)
    sequential_conditions = (hparams.dataset in ('audioset_strong_dual_fsd_wrong_tse',
                               'audioset_strong_dual_fsd_hierarchy_tse_v2') and
                             getattr(hparams, 'tse_condition_backward', 'combined') == 'sequential')


    #创建lightning trainer
    trainer = Trainer(
        min_epochs=0 if smallfit else hparams.epochs,
        max_epochs=hparams.epochs,
        max_steps=hparams.max_steps if smallfit else -1,
        logger=tensorboard,
        limit_train_batches=hparams.train_percent_check,
        limit_val_batches=hparams.val_percent_check,
        enable_checkpointing=True,
        callbacks=[checkpoint, pl.callbacks.ModelSummary(max_depth=5), *extra_callbacks],
        default_root_dir=hparams.save_path,
        devices=devices,
        accelerator=accelerator,
        strategy=strategy,
        # Manual optimization clips once after all condition gradients accumulate.
        gradient_clip_val=None if sequential_conditions else hparams.gradient_clip_val,
        enable_progress_bar=False,
        benchmark=hparams.cudnn_benchmark,
        deterministic=hparams.cudnn_deterministic,
        num_sanity_val_steps=-1 if smallfit else getattr(hparams, 'num_sanity_val_steps', 10),
        **({'val_check_interval': hparams.tse_smallfit_eval_every,
            'check_val_every_n_epoch': None} if smallfit else {}),
        log_every_n_steps=getattr(hparams, 'log_every_n_steps', 50),
        **({'precision': getattr(hparams, 'precision', '32-true'),
            'accumulate_grad_batches': getattr(hparams, 'accumulate_grad_batches', 1)}),
    )

    if smallfit:
        from tse_smallfit_separator import TSESmallfitSeparator as TSESourceSeparator
    elif hparams.dataset in ('audioset_strong_dual_fsd_wrong_tse',
                           'audioset_strong_dual_fsd_hierarchy_tse_v2'):
        from tse_dual_separator import TSEDualSourceSeparator as TSESourceSeparator
    else:
        from tse_separator import TSESourceSeparator
    model = TSESourceSeparator(hparams)

    trainer.fit(model, ckpt_path=resume_from)
    if hparams.test_after_fit:
        trainer.test(model, ckpt_path="best")
    return trainer


#程序入口
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-def",
        type=str,
        default=str(Path(__file__).resolve().parent / "exp_defs/tse_strong_dual_fsd_hierarchy_v2.yaml"),
    )
    """
    取参数，只把--exp-def取出来，知道应该读取那个YAML,剩余参数放入other_options
    """
    args, other_options = parser.parse_known_args()
    hyperparams = yaml_to_parser(args.exp_def).parse_args(other_options)
    main(hyperparams)
