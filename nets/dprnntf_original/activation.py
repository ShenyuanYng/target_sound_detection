import torch.nn as nn
import torch
from .normalization import Identity


def get_activation_layer(act_type='elu'):
    if act_type == 'elu':#正数部分保持不变，复数部分不会全部变成0，输出下限接近-1
        return nn.ELU()
    elif act_type == 'relu':#负数变为0
        return nn.ReLU()
    elif act_type == 'None':
        return Identity() #原样返回输入
    elif act_type == 'prelu':
        return nn.PReLU(num_parameters=1, init=0.25)#负数保留一部分，而且保留比例可以训练
    else:
        raise
"""
混合波形
[1,32000]
    ↓ STFT
[1,503,129,2]
    ↓ 输入整理
[1,2,503,129]
    ↓ 卷积和激活
[1,16,503,129]
    ↓ TFGridNet 主体
[1,16,503,129]
    ↓ 无激活的输出反卷积
[1,4,503,129]
    ↓ reshape
两个说话人的 RI 频谱
[1,503,2,1,2,129]
"""