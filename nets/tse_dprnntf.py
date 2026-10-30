"""MID-conditioned waveform adapter around the teacher's unchanged TF blocks."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from nets.dprnntf_original.tfgridnet import DPRNNTF
from nets.dprnntf_original.audio_utils_torch import do_stft, do_istft


def unpack(value):#使模型拿到真正的tenor  unpack((mixture_tensor, mixture_lengths))
    return value[0] if isinstance(value, (tuple, list)) else value

"""
film的作用是：把MID标识转换成每个音频特征通道的缩放量和偏移量，让同一个音频主干能够根据目标类别调整处理方式
condition_dim是传入类别向量的维度128
channels：需要被调制的音频特征通道数 128

FiLM 参数：两层 Linear 的 weight、bias，在训练中更新，同一个 FiLM 面对所有 MID 共用这些参数。
调制系数 γ、β：根据当前 MID 向量计算出来的结果，因此不同 MID 可以得到不同的系数。
"""
class FiLM(nn.Module):
    def __init__(self, condition_dim, channels):
        super().__init__()
        """第一层线性层[B,128]->silu->第二层线性层[B,256]
        第一层：重新组合 MID 向量中的信息，学习怎样把类别表示转换成适合调制的中间表示。维数仍是 128，但数值和表达方式改变了。
        第二层：生成每个音频通道需要的 γ、β。因为有 128 个通道，每个需要两个系数，所以输出 128×2=256
        """
        self.projection = nn.Sequential(nn.Linear(condition_dim, condition_dim),
                                        nn.SiLU(), nn.Linear(condition_dim, 2 * channels))
        """把最后一层初始化为0，也就是第二层线性层的w和b"""
        nn.init.zeros_(self.projection[-1].weight)
        nn.init.zeros_(self.projection[-1].bias)

    """condition是调用者传入的context[B,128]，已经不是原来的[B,527],调用位置是features = film(features, context)，features[2,128,1251,129]"""
    def forward(self, features, condition):
        """先通过projection得到[B,256],  然后沿最后一维，把向量拆成两份
            前128个->gama:[2,128],后128-> beta:[2,128]       
        """
        gamma, beta = self.projection(condition).chunk(2, dim=-1)
        """两个none各插入一个长度为1的维度，gamma：[2,128,1,1]beta： [2,128,1,1]，使得他们可以与[B,C,T,F]的特征相乘相加
        """
        return features * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]






class TSEDPRNNTF(nn.Module):
    """创建并保存层及其参数
        TSESourceSeparator.__init__(hparams)
            → model_creator(hparams)
              → create_tse_model(hparams)
                 → TSEDPRNNTF.__init__(...)
                     → 创建 DPRNNTF backbone
                     → 创建 class_embedding
                     → 创建四个 FiLM
    """
    
    def __init__(self, num_classes=527, condition_dim=128, emb_dim=128,#emb_dim是表述音频的特征通道数
                 hidden=200, layers=4, emb_ks=1, emb_hs=1, seqmodel_type=3861,
                 window=512, hop=128, gradient_checkpointing=False, return_ri=False):
        super().__init__()
        if window not in (256, 512) or hop != 128:
            raise ValueError('TSE supports sqrt-Hann STFT windows 256/512 with hop=128')
        if num_classes < 1 or condition_dim < 1 or layers < 1:
            raise ValueError('Class count, embedding dimension and layer count must be positive')
        self.window, self.hop, self.num_classes = window, hop, num_classes
        self.gradient_checkpointing = gradient_checkpointing
        self.return_ri = return_ri
        """self.backbone 是一个真正的 DPRNNTF 模型对象
        backbone
        ├── conv0-0       输入卷积模块
        ├── dual_rnn      四个完整的时频处理 Block
        └── deconv0       输出转置卷积模块  
        """
        self.backbone = DPRNNTF(
            n_srcs=1, n_freqs=window // 2 + 1, n_imics=1, n_omics=1,#单通道混合音频，单通道目标音频
            seqmodel_type=seqmodel_type, n_layers=layers, num_hid_units=hidden,
            emb_dim=emb_dim, emb_ks=emb_ks, emb_hs=emb_hs,
            masking_or_mapping=1, mask_nonlinearity='linear', rmax=5.,
            act_type='prelu', norm_type='LayerNormalization', convnormact_type=2,
            n_extra_feats=0, n_pred_components=2, first_conv_use_act=1, mag_sf=1.)
        """创建一张可训练的表，类别 0   → 128 维向量 ...
            把离散的类别身份转为后续FILM能使用的连续向量
        """
        self.class_embedding = nn.Embedding(num_classes, condition_dim)
        self.films = nn.ModuleList([FiLM(condition_dim, emb_dim) for _ in range(layers)])

    def forward(self, model_input, target_dict=None, current_epoch=None):
        # target_dict is never read: inference needs only waveform and MID.
        mixture = unpack(model_input['mixture_waveform'])#[2,160000]
        condition = unpack(model_input['condition'])#[2,527]
        # 检查输出
        if mixture.ndim != 2 or mixture.shape[-1] < 1 or not torch.isfinite(mixture).all():
            raise ValueError('Expected finite mixture_waveform [B,L]')
        if condition.shape != (mixture.shape[0], self.num_classes):
            raise ValueError('Expected one-hot condition [B,num_classes]')
        if not (((condition == 0) | (condition == 1)).all() and (condition.sum(-1) == 1).all()):
            raise ValueError('Each TSE condition must contain exactly one MID')
        # one-hot转类别索引，再查Embedding
        context = self.class_embedding(condition.argmax(-1))
        # STFT and overlap-add stay float32, including under mixed precision.
        with torch.autocast(device_type=mixture.device.type, enabled=False):
            ri = do_stft(mixture.float(), self.window, self.hop)
        """ RI调整维度并进入Conv2D [B,T,2,F] -> [B,2,T,F] layout.
        这里getattr(..., 'conv0-0') 取出的是tfgridnet中构造函数中注册的输入卷积模块
        Conv2D：2 → 128 通道，kernel=3×3，stride=1，padding=1
                → GlobalLayerNormalization
                → PReLU
        [2,2,1251,129]
       │ ConvBlock
       ▼
        [2,128,1251,129]
        """
        features = getattr(self.backbone, 'conv0-0')(ri.permute(0, 1, 3, 2).contiguous().transpose(1, 2))
        

        """
        输入卷积输出
        → FiLM 1 → Block 1
        → FiLM 2 → Block 2
        → FiLM 3 → Block 3
        → FiLM 4 → Block 4
        每个block：
        输入：[B,C,T,F]
         → 综合处理频率、时间和全局关系
        输出：[B,C,T,F]
        """
        for film, block in zip(self.films, self.backbone.dual_rnn):
            """
            film 根据每条样本的类别向量，调整该样本的音频特征：
            film(features, context)     features：[2,128,1251,129]context： [2,128]
                     ↓
            新的 features：[2,128,1251,129]
            """
            features = film(features, context)
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                # Recompute each teacher block during backward instead of retaining
                # all recurrent/attention activations. Pass the block directly so
                # backward cannot accidentally capture another loop iteration.
                features = checkpoint(block, features, use_reentrant=False)
            else:
                features = block(features)
        
        #四轮block之后通过deconv0 得到目标RI，features[2,128,1251,129]经过deconv之后[2,2,1251,129].再经过permute[2,1251,129,2]
        estimate_ri = self.backbone.deconv0(features).permute(0, 2, 3, 1)
        
        # iSTFT 重建波形
        with torch.autocast(device_type=mixture.device.type, enabled=False):
            waveform = do_istft(estimate_ri.float(), self.window, self.window, self.hop)
        if waveform.shape[-1] < mixture.shape[-1]:
            raise RuntimeError('iSTFT returned fewer samples than the input')
        """返回的是字典：{
    'target_waveform': Tensor[2,160000]
}
        """
        result = {'target_waveform': waveform[..., :mixture.shape[-1]]}
        if self.return_ri:
            result['target_ri'] = estimate_ri.float()
        return result


def create_tse_model(hp):
    return TSEDPRNNTF(num_classes=hp.num_classes, condition_dim=hp.tse_condition_dim,
        emb_dim=hp.tse_emb_dim, hidden=hp.tse_hidden, layers=hp.tse_layers,
        emb_ks=hp.tse_emb_ks, emb_hs=hp.tse_emb_hs, seqmodel_type=hp.tse_seqmodel_type,
        window=hp.tse_window, hop=hp.tse_hop,
        gradient_checkpointing=getattr(hp, 'tse_gradient_checkpointing', False),
        return_ri=getattr(hp, 'tse_return_ri', False))
