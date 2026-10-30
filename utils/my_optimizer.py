import torch
from torch import optim
from loguru import logger

class ReduceLROnPlateau_verbose(optim.lr_scheduler.ReduceLROnPlateau):
    # based on https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/lr_scheduler.py#L1232
    def _reduce_lr(self, epoch):
        if len(self.optimizer.param_groups) != len(self.min_lrs):
            if self.default_min_lr is None:
                raise RuntimeError(
                    "The number of param groups in the `optimizer` "
                    f"({len(self.optimizer.param_groups)}) differs "
                    f"from when `ReduceLROnPlateau` was initialized "
                    f"({len(self.min_lrs)}), usually due to a new "
                    "param group being added to the optimizer. Please "
                    "modify the `min_lrs` field to match the length "
                    "of the `optimizer` param groups."
                )
            else:
                self.min_lrs = [self.default_min_lr] * len(self.optimizer.param_groups)

        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group["lr"])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group["lr"] = new_lr

            epoch_str = ("%.2f" if isinstance(epoch, float) else "%.5d") % epoch
            logger.info('Epoch {}: reducing learning rate of group {} to {}.'.format(epoch_str, i, new_lr))
