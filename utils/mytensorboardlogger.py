from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from argparse import Namespace
from loguru import logger
import os


class MyTensorBoardLogger(TensorBoardLogger):
    def __init__(self, save_dir, name="lightning_logs", version=None, **kwargs):
        if version is None:
            version = self.slurm_job_id
        super().__init__(save_dir, name=name, version=version, **kwargs)
        self.experiment
        self.testing = False

    @property
    def slurm_job_id(self):
        try:
            job_id = os.environ['SLURM_JOB_ID']
            job_id = int(job_id)
        except Exception:
            job_id = None
        return job_id

    @rank_zero_only
    def log_hyperparams(self, params):
        if self.testing:
            return

        if params is None:
            return

        if isinstance(params, Namespace):
            params = vars(params)
        params = dict(params)

        for k, v in params.items():
            if v is None:
                params[k] = 'None'

        logger.info('\nParameters:')
        for k, v in params.items():
            logger.info(f'{k} = {v}')

        logger.info(f'Tensorboard logs: {self.experiment.log_dir}')

        super().log_hyperparams(params)
