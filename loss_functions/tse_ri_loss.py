"""Fixed-order RI L1 + magnitude L1 from the supplied riinf_enh_loss.

Supported branch: ratiol1, RIx2=1, mag=1, PIT=0, MC=0, mag_sf=1,
rix2_weight=1, no iSTFT/STFT projection, no SI-SNR training loss.
The 1e-5 magnitude stabilizer and utterance-wise mixture normalization match
the supplied function. Per-example terms additionally support grouped metrics.
"""
import torch
from torch import nn

from loss_functions.tse_loss import unpack_waveform
from nets.dprnntf_original.audio_utils_torch import do_stft


def riinf_enh_loss(est_ri, target_dict, *, loss_type='ratiol1', rix2_flag=1,
                   mag_flag=1, permutation_invariant=0, mc_flag=0,
                   sisnrse_flag=0, stft_istft_proj_flag=0, mag_sf=1., rix2_weight=1.):
    """est_ri是模型预测的目标复数频谱RI，target_dict是包含mixture和GT的频谱字典"""
    if (loss_type, rix2_flag, mag_flag, permutation_invariant, mc_flag,
            sisnrse_flag, stft_istft_proj_flag, mag_sf, rix2_weight) != (
            'ratiol1', 1, 1, 0, 0, 0, 0, 1., 1.):
        raise ValueError('Experiment 3 requires fixed RI L1 + magnitude L1 flags')
    """从字典中取复数频谱，[B,T,1,1,2,F]其中的两个1是单目标单通道相关的保留维度"""
    mixture, target = target_dict['mix_ri'][0], target_dict['srcs_ri'][0]
    if (est_ri.ndim != 6 or est_ri.shape[2:5] != (1, 1, 2)
            or est_ri.shape != target.shape or mixture.shape != target.shape):
        raise ValueError('RI tensors must all have shape [B,T,1,1,2,F]')
    with torch.autocast(device_type=est_ri.device.type, enabled=False):
        estimate, target, mixture = est_ri.float(), target.float(), mixture.float()
        """取实虚部"""
        er, ei = estimate[..., 0, :], estimate[..., 1, :]
        tr, ti = target[..., 0, :], target[..., 1, :]
        mr, mi = mixture[..., 0, :], mixture[..., 1, :]
        """先平方，相加，开根号，假设mr的形状是[B,T,1,1,F],flatten(1)会把第一维及以后的所有维度展平->[B,F*T]
        接着sum（-1）对每个样本的所有时间帧和频率点求和，结果是形状[B]的tensor"""
        weight = torch.sqrt(mr.square() + mi.square() + 1e-5).flatten(1).sum(-1)
        weight = weight.clamp_min(torch.finfo(torch.float32).tiny)#确保weight不会低于这个值，避免除0
        """计算实部，虚部loss，.flatten(1).sum(-1)将每个样本所有的时间帧、频率点的绝对误差加起来
        这里实际上计算的还是预测以及target之间的loss，weight只是mix的归一化参考"""
        real = (er-tr).abs().flatten(1).sum(-1) / weight
        imag = (ei-ti).abs().flatten(1).sum(-1) / weight
        magnitude = (torch.sqrt(er.square()+ei.square()+1e-5)
                     - torch.sqrt(tr.square()+ti.square()+1e-5)).abs().flatten(1).sum(-1) / weight
        per_example = real + imag + magnitude#组合每个样本的总loss
    return dict(loss=per_example.mean(), per_example_loss=per_example,
                sR=real.mean().detach(), sI=imag.mean().detach(), sM=magnitude.mean().detach())


class TSERILoss(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.window, self.hop = hp.tse_window, hp.tse_hop
        self.flags = {k: getattr(hp, k) for k in (
            'loss_type', 'rix2_flag', 'mag_flag', 'permutation_invariant', 'mc_flag')}

    def forward(self, predictions, targets):
        #获取预测频谱
        estimate = predictions['target_ri']
        mixture = unpack_waveform(targets['mixture_waveform'])
        target = unpack_waveform(targets['target_waveform'])
        if target.shape != mixture.shape:
            raise ValueError('Mixture and GT must be sample-aligned')
        with torch.autocast(device_type=estimate.device.type, enabled=False):
            mix_ri = do_stft(mixture.float(), self.window, self.hop)
            src_ri = do_stft(target.float(), self.window, self.hop)
            adapt = lambda x: x.permute(0, 1, 3, 2)[:, :, None, None, :, :]#调整张量格式
            return riinf_enh_loss(adapt(estimate),
                dict(mix_ri=(adapt(mix_ri),), srcs_ri=(adapt(src_ri),)), **self.flags)
