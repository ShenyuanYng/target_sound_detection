import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init
from loguru import logger

"""
各种归一化方法，归一化的目标是，把某组特征调整到相对稳定的尺度，使模型更容易训练
标准化结果 = (x - 均值) / sqrt(方差 + eps)
最终结果   = 标准化结果 × gamma + beta
均值、方差：从张量的某些维度上计算
eps：防止除以零，当前通常是 1e-5
gamma：可训练的缩放参数
beta：可训练的平移参数
"""

"""
[1,16,503,129],会用这个样本中全部16 × 503 × 129个数计算一个均值和方差,形状不变
[1,16,503,129]
    ↓ GlobalLayerNormalization
[1,16,503,129]
"""

"""
GlobalLayerNormalization	每个样本	          C,T,F	[B,1,1,1]
GlobalInstanceNormalization	每个样本、每个通道	    T,F	[B,C,1,1]
LayerNormalization4D	    每个样本、每个时频点	C	[B,1,T,F]
LayerNormalization4DCF	    每个样本、每个时间帧	C,F	[B,1,T,1]
AllHeadLayerNormalization4DCF	每个样本、每头、每个时间帧	E,F	[B,H,1,T,1]
BatchNorm2d	                每个通道	          B,T,F	每通道一组
"""
def GlobalLayerNormalization(input_dimension):  #输入张量的特征通道数C
    """We can achieve the TASNet global layer normalization with the
    built in GroupNormalization, assuming input is an array of shape
    (n_batch, n_features, n_time)"""
    return nn.GroupNorm(1, input_dimension, eps=1e-5) #真正输入groupnorm的数据是forward时传进去的tensor



def GlobalGroupNormalization(input_dimension,group_size=4):
    #把通道分成若干组，每组包含group_size个通道
    # Actually GlobalGroupLayerNormalization
    if input_dimension % group_size == 0:
        num_groups = input_dimension // group_size
    else:#不能整除 → 所有通道合成一组
        print(input_dimension, group_size)
        num_groups = 1 # layer norm
    return nn.GroupNorm(num_groups, input_dimension, eps=1e-5)

def GlobalGroupInsNormalization(input_dimension,group_size=4):
    if input_dimension % group_size == 0:
        num_groups = input_dimension // group_size
    else:#不能整除 → 每个通道单独一组
        print(input_dimension, group_size)
        num_groups = input_dimension # instance norm
    return nn.GroupNorm(num_groups, input_dimension, eps=1e-5)

def GlobalInstanceNormalization(input_dimension):#每个通道分别在时间和频率上统计
    return nn.GroupNorm(input_dimension, input_dimension, eps=1e-5)

class LayerNormalization(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super(LayerNormalization, self).__init__()
        self.layer = nn.LayerNorm(input_dimension, eps=eps)

    def forward(self, x):
        return self.layer(x)

class LayerNormalization4D(nn.Module):#专门处理通道优先格式
    def __init__(self, input_dimension, n_dims=4, eps=1e-5, affine_dim=1):
        super(LayerNormalization4D, self).__init__()
        if n_dims == 4:
            # [B,C,T,F]
            if affine_dim == 1:
                param_size = [1,input_dimension,1,1]
            else:
                raise
        else:
            raise
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x):
        if x.ndim == 4: 
            _,C,_,_ = x.shape
            stat_dim = (1,)
        else:
            raise
        mu_ = x.mean(dim=stat_dim,keepdim=True) # [B,1,T,F]
        #mu_sq_ = (x**2).mean(dim=stat_dim,keepdim=True) # [B,1,T,F]
        #std_ = torch.sqrt(mu_sq_-mu_**2+self.eps) # [B,1,T,F], catched on Aug. 2, this may be accidentally nan!
        std_ = torch.sqrt(x.var(dim=stat_dim,unbiased=False,keepdim=True)+self.eps) # [B,1,T,F]
        x_hat = ((x-mu_)/std_)*self.gamma + self.beta
        return x_hat


class LayerNormalization4DCF(nn.Module):#同时考虑通道和频率每个时间帧对 C×F 个值归一化，统计维度：C、F
    def __init__(self, input_dimension, n_dims=4, eps=1e-5):
        super(LayerNormalization4DCF, self).__init__()
        assert len(input_dimension) == 2
        if n_dims == 4:
            # [B,C,T,F]
            param_size = [1,input_dimension[0],1,input_dimension[1]]
        else:
            raise
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x):
        if x.ndim == 4: 
            stat_dim = (1,3)
        else:
            raise
        mu_ = x.mean(dim=stat_dim,keepdim=True) # [B,1,T,1]
        std_ = torch.sqrt(x.var(dim=stat_dim,unbiased=False,keepdim=True)+self.eps) # [B,1,T,F]
        x_hat = ((x-mu_)/std_)*self.gamma + self.beta
        return x_hat


class AllHeadLayerNormalization4DCF(nn.Module):#为多头注意力的QKV的归一化
    def __init__(self, input_dimension, n_dims=4, eps=1e-5):
        super(AllHeadLayerNormalization4DCF, self).__init__()
        assert len(input_dimension) == 3
        H, E, n_freqs = input_dimension  #H注意力头数
        if n_dims == 4:
            # [B,C,T,F]
            param_size = [1,H,E,1,n_freqs]
        else:
            raise
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.act = nn.PReLU(num_parameters=H, init=0.25)
        self.eps = eps
        self.H = H
        self.E = E
        self.n_freqs = n_freqs

    def forward(self, x):
        assert x.ndim == 4
        B, _, T, _ = x.shape
        x = x.view([B, self.H, self.E, T, self.n_freqs])
        x = self.act(x) # [B,H,E,T,F]
        stat_dim = (2,4)
        mu_ = x.mean(dim=stat_dim,keepdim=True) # [B,H,1,T,1]
        std_ = torch.sqrt(x.var(dim=stat_dim,unbiased=False,keepdim=True)+self.eps) # [B,H,1,T,1]
        x = ((x-mu_)/std_) * self.gamma + self.beta # [B,H,E,T,F]
        return x


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()
    def forward(self, x):
        return x


#统一调度归一化
def get_norm_layer(input_dimension, norm_type='BatchNorm', n_dims=3, use_causal=0):
    if norm_type == 'BatchNorm':
        if n_dims == 3:#三维输入[B,C,T]
            return nn.BatchNorm1d(input_dimension, momentum=0.005, eps=1e-5)
        elif n_dims == 4:#四维输入[B,C,T,F]
            return nn.BatchNorm2d(input_dimension, momentum=0.005, eps=1e-5)
        else:
            raise
    elif norm_type == 'InsNorm':
        if n_dims == 3:
            return nn.InstanceNorm1d(input_dimension, momentum=0.005,
                    eps=1e-5, affine=True, track_running_stats=True)
        elif n_dims == 4:
            return nn.InstanceNorm2d(input_dimension, momentum=0.005,
                    eps=1e-5, affine=True, track_running_stats=True)
        else:
            raise
    elif norm_type == 'GlobalLayerNormalization':
        assert use_causal == 0
        return GlobalLayerNormalization(input_dimension)
    elif norm_type == 'GlobalInstanceNormalization':
        #assert use_causal == 0 # danger
        return GlobalInstanceNormalization(input_dimension)
    elif norm_type.startswith('GlobalGroupNormalization'):
        assert use_causal == 0
        group_size = int(norm_type.split('GroupNormalization')[1])
        return GlobalGroupNormalization(input_dimension,group_size=group_size)
    elif norm_type.startswith('GlobalGroupInsNormalization'):
        assert use_causal == 0
        group_size = int(norm_type.split('GroupInsNormalization')[1])
        return GlobalGroupInsNormalization(input_dimension,group_size=group_size)

    elif norm_type == 'LayerNormalization':
        if n_dims == 3:
            return LayerNormalization(input_dimension) # [..., C]
        elif n_dims == 4:
            return LayerNormalization4D(input_dimension) # [B, C, T, F]
        else:
            raise
    elif norm_type == 'LayerNormalization4DCF':
        if n_dims == 4:
            return LayerNormalization4DCF(input_dimension) # [B, C, T, F]
        else:
            raise
    elif norm_type == 'AllHeadLayerNormalization4DCF':
        if n_dims == 4:
            return AllHeadLayerNormalization4DCF(input_dimension) # [B, C, T, F]
        else:
            raise

    elif norm_type == 'None':
        return Identity()
    else:
        raise
