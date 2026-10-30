"""Local training primitives shared by the TSE variants."""
import random
import numpy as np
import pytorch_lightning as pl
import torch
from torch import optim
from utils.my_optimizer import ReduceLROnPlateau_verbose

class SourceSeparator(pl.LightningModule):
    """Optimizer, forward adapter and loaders; no SED task dependencies."""

    def forward(self, model_input, target_dict=None):
        target_dict = {} if target_dict is None else target_dict
        #进入任务适配器
        return self.task.forward(
            self.model,#又会进入model
            model_input,
            target_dict,
            current_epoch=self.current_epoch,
        )

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        #学习率调度器
        scheduler = ReduceLROnPlateau_verbose(
            optimizer,
            patience=self.hparams.patience,
            factor=self.hparams.decay_factor,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    def _worker_init_fn(self, worker_id):
        worker_seed = (
            self.hparams.seed
            + worker_id
            + self.current_epoch * 101
            + self.global_rank * 100003
        )
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    def train_dataloader(self):
        return self._build_data_loader("tr")

    def val_dataloader(self):
        return self._build_data_loader("cv")
