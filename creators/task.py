"""Waveform extraction adapter; ground truth never enters inference."""
class TSETask:
    def forward(self, model, model_input, target_dict, current_epoch):
        return model(model_input, current_epoch=current_epoch)

def task_creator(hparams):
    if hparams.task_type != 'tse':
        raise ValueError(f'Unsupported task: {hparams.task_type}')
    return TSETask()
