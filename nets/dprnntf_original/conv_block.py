import torch.nn as nn
import torch
import torch.utils.checkpoint as cp
from .normalization import get_norm_layer
from .activation import get_activation_layer

"""
定义了一个通用卷积模块ConvBlock，包括以下操作
卷积或转置卷积
归一化
激活函数
因果卷积的尾部裁剪
可选的梯度检查点
"""

def get_padding_size(kernel_size, dilation=1, use_causal=0): #卷积核大小，膨胀系数，因果卷积
    if use_causal == 0:
        assert kernel_size%2 == 1
        padding = kernel_size//2 * dilation
    elif use_causal > 0:
        padding = (kernel_size-1) * dilation 
    else:
        raise
    return padding


class Chomp(nn.Module):#裁掉因果卷积右侧输出
    def __init__(self, chomp_size):
        super().__init__()
        assert chomp_size >= 0
        self.chomp_size = chomp_size

    def forward(self, h):
        # [B,C,T,F] or [B,C,T]
        return h[:,:,:-self.chomp_size] if self.chomp_size else h


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
            stride=1, padding=(0,0), dilation=1,
            act_type='elu', norm_type='GlobalInstanceNormalization', convnormact_type=0,
            use_deconv=0, mem_eff=0, n_dims=4,
            use_causal=0, n_freqs=-1, groups=1,
            use_bias=True):
        super().__init__()
        
        assert groups == 1
        if n_dims == 4:#[B,C,T,F]
            module_name = nn.ConvTranspose2d if use_deconv else nn.Conv2d
        elif n_dims == 3:#[B,C,T]
            module_name = nn.ConvTranspose1d if use_deconv else nn.Conv1d
        else:
            raise

        #创建卷积层
        self.conv = module_name(in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, dilation=dilation,
                bias=use_bias)
        if use_bias:
            self.conv.bias.data[:] = 0.0

        #归一化  以CF为结尾的归一化
        if norm_type.endswith('CF'):
            if convnormact_type in [1,2,]:
                self.norm = get_norm_layer((out_channels,n_freqs), norm_type, n_dims, use_causal=use_causal)
            else:
                raise
        else:#普通归一化
            if convnormact_type == 0:  # 归一化-激活-卷积
                self.norm = get_norm_layer(in_channels, norm_type, n_dims, use_causal=use_causal)
            else:#卷积-归一化-激活
                self.norm = get_norm_layer(out_channels, norm_type, n_dims, use_causal=use_causal)

        #创建激活函数
        self.actfn = get_activation_layer(act_type)
        self.convnormact_type = convnormact_type
        self.mem_eff = mem_eff

        if use_causal:
            assert len(padding) == 2
            chomp_size = padding[0]
        else:
            chomp_size = 0
        self.chomp = Chomp(chomp_size)

    def forward(self, *h):
        def func(*h_):
            h_ = torch.cat(h_, dim=1) if len(h_) > 1 else h_[0]  #tensor的拼接
            if self.convnormact_type == 2:
                #convactnorm
                return self.norm(self.actfn(self.chomp(self.conv(h_))))
            elif self.convnormact_type == 1:
                #convnormact
                return self.actfn(self.norm(self.chomp(self.conv(h_))))
            else:
                #normactconv
                return self.chomp(self.conv(self.actfn(self.norm(h_))))
        return cp.checkpoint(func, *h) if self.mem_eff else func(*h)
