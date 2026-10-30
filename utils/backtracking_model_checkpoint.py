"""Checkpoint callback that keeps a resumable checkpoint beside each run."""

import os.path

import torch
from pytorch_lightning.callbacks import ModelCheckpoint


class BacktrackingModelCheckpointUpgrade(ModelCheckpoint):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.CHECKPOINT_NAME_LAST = "mylast"
        self.FILE_EXTENSION = ".ckpt"
        self.sanity_check_complete = False

    def _save_last_checkpoint(self, trainer, monitor_candidates):
        if not self.save_last:
            return

        filepath = self._format_checkpoint_name(
            self.CHECKPOINT_NAME_LAST,
            monitor_candidates,
        )
        filepath = os.path.join(
            os.path.dirname(self.dirpath),
            f"{filepath}{self.FILE_EXTENSION}",
        )
        previous, self.last_model_path = self.last_model_path, filepath
        self._save_checkpoint(trainer, filepath)
        if previous and previous != filepath:
            self._remove_checkpoint(trainer, previous)

    def on_validation_end(self, trainer, pl_module):
        super().on_validation_end(trainer, pl_module)
        if self.sanity_check_complete and trainer.is_global_zero:
            last_ckpt_dir = os.path.dirname(self.dirpath)
            torch.save(
                [
                    self._last_global_step_saved,
                    self.current_score,
                    self.best_k_models,
                    self.kth_best_model_path,
                    self.best_model_score,
                    self.best_model_path,
                    self.last_model_path,
                    trainer.current_epoch,
                ],
                os.path.join(last_ckpt_dir, "mylast_best_info.pt"),
            )
        self.sanity_check_complete = True
