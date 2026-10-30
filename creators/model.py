"""Target sound extraction model factory."""
def model_creator(hparams):
    if hparams.model_name != 'TSEDPRNNTF':
        raise ValueError(f'Unsupported TSE model: {hparams.model_name}')
    from nets.tse_dprnntf import create_tse_model
    return create_tse_model(hparams)
