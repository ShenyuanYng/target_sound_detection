"""Target sound extraction loss factory."""
def loss_creator(hparams):
    if hparams.loss_type == 'ratiol1' and hparams.task_type == 'tse':
        from loss_functions.tse_ri_loss import TSERILoss
        return TSERILoss(hparams)
    if hparams.loss_type == 'tse':
        from loss_functions.tse_loss import create_tse_loss
        return create_tse_loss(hparams)
    raise ValueError(f'Unsupported TSE loss: {hparams.loss_type}')
