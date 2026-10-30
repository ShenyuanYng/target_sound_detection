import torch
import torch.nn as nn
from .normalization import get_norm_layer
from .conv_block import ConvBlock
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init
import math
from .activation import get_activation_layer
from functools import partial
import random
from .audio_utils_torch import do_istft, do_stft, complex_multiply, scaling_magnitude



"""
模型不断变换由 mix_ri 得到的特征，最后预测出一个结果。
训练过程中再通过 Loss 反向传播修改模型权重，使得下一次预测的结果越来越接近真实语音

模型一以混合语音的Real/Imag时频表示为输入，先将他转成高维特征。通过Conv，Rnn，Attention等模块不断变换这些特征，
最终预测目标语音的复数Mask或直接预测目标RI。训练时通过不断比较预测结果和真实目标语音，反向传播不断修改模型权重，
使得模型最终输出的目标语音RI越来越接近真实语音
"""

#一个“频率 LSTM + 时间 LSTM + 时间注意力”的残差处理块
class FrameSeqAttn3U3NmUfdDualRNNBlockTFEmbViTLNFast(nn.Module):
    """
    Implementation of the intra-RNN and the inter-RNN
    input:
    """
    #def __getitem__(self, key):
    #    return getattr(self, key)

    def __init__(self,
            #内部特征通道数，分块长度，分块步长，频点数，LSTM单方向隐藏维度
            emb_dim, emb_ks, emb_hs, n_freqs, hidden_channels,
            #注意力头数  approx_qk_dim：控制每个注意力头的 Q/K 特征大小
            n_head=4, attn_mode=0, approx_qk_dim=512,
            rnn_type='LSTM',
            act_type='elu', norm_type='LayerNormalization', convnormact_type=1,
            dropout=0., use_causal=0):
        super(FrameSeqAttn3U3NmUfdDualRNNBlockTFEmbViTLNFast, self).__init__()
        assert use_causal == 0
        assert emb_ks % emb_hs == 0
        assert norm_type == 'LayerNormalization'


        """
        频率LSTM
        """
        #计算每个 LSTM 输入的特征维度
        in_channels = emb_dim * emb_ks
        #创建频率方向的 LSTM
        bidirectional = (use_causal == 0)
        #创建频率LSTM前的归一化层
        self.intra_norm = get_norm_layer(emb_dim, norm_type, 3, use_causal=use_causal)
        """
        按字符串从torch.nn中取类
        输入维度：128
        单方向隐藏维度：128
        层数：1
        batch 在第一维
        双向：True
        """
        self.intra_rnn = getattr(nn, rnn_type)(in_channels, hidden_channels, 1, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        #n_tmp 是一个中间变量，它代表的是经过频率方向 LSTM 处理后，每个时间步（帧）的特征维度
        n_tmp = hidden_channels*2 if bidirectional else hidden_channels
        #创建频率分块还原层
        if emb_ks == emb_hs:
            self.intra_linear = nn.Linear(n_tmp, in_channels)
        else:
            self.intra_linear = nn.ConvTranspose1d(n_tmp, emb_dim, emb_ks, stride=emb_hs, padding=0)  




        """
        创建时间方向的LSTM
        """
        self.inter_norm = get_norm_layer(emb_dim, norm_type, 3, use_causal=use_causal)
        self.inter_rnn = getattr(nn, rnn_type)(in_channels, hidden_channels, 1, batch_first=True, dropout=dropout, bidirectional=bidirectional)

        if emb_ks == emb_hs:
            self.inter_linear = nn.Linear(n_tmp, in_channels)
        else:
            self.inter_linear = nn.ConvTranspose1d(n_tmp, emb_dim, emb_ks, stride=emb_hs, padding=0)

        cb_kwargs = {'norm_type': norm_type, 'act_type': act_type, 'convnormact_type': convnormact_type}
        self.convnormact_type = convnormact_type
        self.act_type = act_type
        self.norm_type = norm_type
        padding = 0

        """
        准备注意力模块
        """
        #E 是每个注意力头在每个频率点上生成的 Q、K 特征数。
        E = math.ceil(approx_qk_dim*1.0/n_freqs) # approx_qk_dim is only approximate
        #限制支持的频率点数
        if n_freqs in [513,129,257,65,33,]:
            if n_freqs == 129:
                assert E in [4,8,]
            if n_freqs == 513:
                assert E in [2,]
            if n_freqs == 257:
                assert E in [2,4,]
            if n_freqs == 65:
                assert E in [8,]
            if n_freqs == 33:
                assert E in [16,]
            self.n_head = n_head
            assert emb_dim % n_head == 0


            """
            创建以后用来生成Q.K.V的卷积层和归一化层
            """
            if attn_mode == 4: #多头注意力机制，四个头，每个头都有一套独立的Q K V
                for ii in range(n_head):
                    #模块名，卷积层中（输入通道，输出通道，卷积核大小，stride，填充）
                    self.add_module('attn_conv_Q_%d'%ii, ConvBlock(emb_dim, E, (1,1), stride=(1,1), padding=padding,
                                act_type=act_type, convnormact_type=convnormact_type, norm_type='LayerNormalization4DCF', n_freqs=n_freqs))
                    self.add_module('attn_conv_K_%d'%ii, ConvBlock(emb_dim, E, (1,1), stride=(1,1), padding=padding,
                                act_type=act_type, convnormact_type=convnormact_type, norm_type='LayerNormalization4DCF', n_freqs=n_freqs))
                    self.add_module('attn_conv_V_%d'%ii, ConvBlock(emb_dim, emb_dim//n_head, (1,1), stride=(1,1), padding=padding,
                                act_type=act_type, convnormact_type=convnormact_type, norm_type='LayerNormalization4DCF', n_freqs=n_freqs))
                self.add_module('attn_concat_proj', ConvBlock(emb_dim, emb_dim, (1,1), stride=(1,1), padding=padding,
                            act_type=act_type, convnormact_type=convnormact_type, norm_type='LayerNormalization4DCF', n_freqs=n_freqs))
            elif attn_mode == 41:#用一个大卷积，一次生成所有头
                assert act_type == 'prelu' and  convnormact_type == 2
                self.add_module('attn_conv_Q', nn.Conv2d(emb_dim, n_head*E, 1, stride=1, padding=0))
                #self['attn_conv_Q'].bias.data[:] = 0.0
                self.attn_conv_Q.bias.data[:] = 0.0
                self.add_module('attn_norm_Q', get_norm_layer((n_head, E, n_freqs), 'AllHeadLayerNormalization4DCF', 4))

                self.add_module('attn_conv_K', nn.Conv2d(emb_dim, n_head*E, 1, stride=1, padding=0))
                #self['attn_conv_K'].bias.data[:] = 0.0
                self.attn_conv_K.bias.data[:] = 0.0
                self.add_module('attn_norm_K', get_norm_layer((n_head, E, n_freqs), 'AllHeadLayerNormalization4DCF', 4))

                self.add_module('attn_conv_V', nn.Conv2d(emb_dim, n_head*emb_dim//n_head, 1, stride=1, padding=0))
                #self['attn_conv_V'].bias.data[:] = 0.0
                self.attn_conv_V.bias.data[:] = 0.0
                self.add_module('attn_norm_V', get_norm_layer((n_head, emb_dim//n_head, n_freqs), 'AllHeadLayerNormalization4DCF', 4))

                self.add_module('attn_concat_proj', ConvBlock(emb_dim, emb_dim, (1,1), stride=(1,1), padding=padding,
                            act_type=act_type, convnormact_type=convnormact_type, norm_type='LayerNormalization4DCF', n_freqs=n_freqs))
            else:
                raise
        else:
            raise

#保存分块参数
        self.emb_dim = emb_dim
        self.emb_ks = emb_ks
        self.emb_hs = emb_hs



    """
    时频处理块的前向计算
    """    
    def forward(self, x):
        """
        x: [B, C, T, Q]
        out: [B, C, T, Q]
        """
        B, C, old_T, old_Q = x.shape
        olp = self.emb_ks - self.emb_hs
        #补齐长度
        T = math.ceil((old_T+2*olp-self.emb_ks)*1./self.emb_hs)*self.emb_hs + self.emb_ks
        Q = math.ceil((old_Q+2*olp-self.emb_ks)*1./self.emb_hs)*self.emb_hs + self.emb_ks
        #重排维度
        x = x.permute(0,2,3,1) # [B, old_T, old_Q, C]
        x = F.pad(x, (0,0,olp,Q-old_Q-olp,olp,T-old_T-olp)) # [B, T, Q, C]


        #沿频率轴的
        # intra RNN
        input_ = x
        intra_rnn = self.intra_norm(input_) # [B, T, Q, C]
        if self.emb_ks == self.emb_hs:
            intra_rnn = intra_rnn.view([B*T, -1, self.emb_ks*C]) # [BT, Q//I, I*C]
            intra_rnn, _ = self.intra_rnn(intra_rnn) # [BT, Q//I, H]
            intra_rnn = self.intra_linear(intra_rnn) # [BT, Q//I, I*C]
            intra_rnn = intra_rnn.view([B, T, Q, C])
        else:
            intra_rnn = intra_rnn.view([B*T, Q, C]) # [BT, Q, C]
            intra_rnn = intra_rnn.transpose(1,2) # [BT, C, Q]
            #none会增加一个长度为1的维度
            intra_rnn = F.unfold(intra_rnn[...,None], (self.emb_ks,1), stride=(self.emb_hs,1)) # [BT, C*I, -1]
            intra_rnn = intra_rnn.transpose(1,2) # [BT, -1, C*I]
            #频率双向LSTM
            intra_rnn, _ = self.intra_rnn(intra_rnn) # [BT, -1, H]


            #重建完整频率轴
            intra_rnn = intra_rnn.transpose(1,2) # [BT, H, -1]
            intra_rnn = self.intra_linear(intra_rnn) # [BT, C, Q]
            intra_rnn = intra_rnn.view([B, T, C, Q])
            intra_rnn = intra_rnn.transpose(-2,-1) # [B, T, Q, C]
        intra_rnn = intra_rnn + input_ # [B, T, Q, C]

        intra_rnn = intra_rnn.transpose(1,2) # [B, Q, T, C]



        # inter RNN
        input_ = intra_rnn
        inter_rnn = self.inter_norm(input_) # [B, Q, T, C]
        if self.emb_ks == self.emb_hs:
            inter_rnn = inter_rnn.view([B*Q, -1, self.emb_ks*C]) # [BQ, T//I, I*C]
            inter_rnn, _ = self.inter_rnn(inter_rnn) # [BQ, T//I, H]
            inter_rnn = self.inter_linear(inter_rnn) # [BQ, T//I, I*C]
            inter_rnn = inter_rnn.view([B, Q, T, C])
        else:
            inter_rnn = inter_rnn.view(B*Q, T, C) # [BQ, T, C]
            inter_rnn = inter_rnn.transpose(1,2) # [BQ, C, T]
            inter_rnn = F.unfold(inter_rnn[...,None], (self.emb_ks,1), stride=(self.emb_hs,1)) # [BQ, C*I, -1]
            inter_rnn = inter_rnn.transpose(1,2) # [BQ, -1, C*I] 

            inter_rnn, _ = self.inter_rnn(inter_rnn) # [BQ, -1, H]

            inter_rnn = inter_rnn.transpose(1,2) # [BQ, H, -1]
            inter_rnn = self.inter_linear(inter_rnn) # [BQ, C, T]
            inter_rnn = inter_rnn.view([B, Q, C, T])
            inter_rnn = inter_rnn.transpose(-2,-1) # [B, Q, T, C] 
        inter_rnn = inter_rnn + input_ # [B, Q, T, C]

        inter_rnn = inter_rnn.permute(0,3,2,1) # [B, C, T, Q]

        inter_rnn = inter_rnn[...,olp:olp+old_T,olp:olp+old_Q]
        batch = inter_rnn




        #多头注意力机制
        if hasattr(self, 'attn_conv_Q'):
            #Q = self['attn_norm_Q'](self['attn_conv_Q'](batch)) # [B, n_head, C, T, Q]
            #K = self['attn_norm_K'](self['attn_conv_K'](batch)) # [B, n_head, C, T, Q]
            #V = self['attn_norm_V'](self['attn_conv_V'](batch)) # [B, n_head, C, T, Q]
            Q = self.attn_norm_Q(self.attn_conv_Q(batch)) # [B, n_head, C, T, Q]
            K = self.attn_norm_K(self.attn_conv_K(batch)) # [B, n_head, C, T, Q]
            V = self.attn_norm_V(self.attn_conv_V(batch)) # [B, n_head, C, T, Q]
        else:
            raise
            all_Q, all_K, all_V = [], [], []
            for ii in range(self.n_head):
                #all_Q.append(self['attn_conv_Q_%d'%ii](batch)) # [B, C, T, Q]
                #all_K.append(self['attn_conv_K_%d'%ii](batch)) # [B, C, T, Q]
                #all_V.append(self['attn_conv_V_%d'%ii](batch)) # [B, C, T, Q]
                all_Q.append(getattr(self,'attn_conv_Q_%d'%ii)(batch)) # [B, C, T, Q]
                all_K.append(getattr(self,'attn_conv_K_%d'%ii)(batch)) # [B, C, T, Q]
                all_V.append(getattr(self,'attn_conv_V_%d'%ii)(batch)) # [B, C, T, Q]
            Q = torch.stack(all_Q, dim=1) # [B, n_head, C, T, Q]
            K = torch.stack(all_K, dim=1) # [B, n_head, C, T, Q]
            V = torch.stack(all_V, dim=1) # [B, n_head, C, T, Q]
        Q = Q.view(-1, *Q.shape[2:]) # [B*n_head, C, T, Q]
        K = K.view(-1, *K.shape[2:]) # [B*n_head, C, T, Q]
        V = V.view(-1, *V.shape[2:]) # [B*n_head, C, T, Q]

        """
        if callable(getattr(F, 'scaled_dot_product_attention', None)):
            Q = Q.transpose(1,2)
            Q = Q.flatten(start_dim=2) # [B', T, C*Q]

            K = K.transpose(1,2)
            K = K.flatten(start_dim=2) # [B', T, C*Q]

            V = V.transpose(1,2) # [B', T, C, Q]
            old_shape = V.shape
            V = V.flatten(start_dim=2) # [B', T, C*Q]
     
            V = F.scaled_dot_product_attention(Q, K, V) # [B', T, C*Q]
        else:
        """
        Q = Q.transpose(1,2)
        Q = Q.flatten(start_dim=2) # [B', T, C*Q]
        #
        K = K.transpose(2,3)
        K = K.contiguous().view([B*self.n_head, -1, old_T]) # [B', C*Q, T]
        #
        V = V.transpose(1,2) # [B', T, C, Q]
        old_shape = V.shape
        V = V.flatten(start_dim=2) # [B', T, C*Q]
        emb_dim = Q.shape[-1]
        #
        #attn_mat = torch.matmul(Q, K.transpose(1,2)) / (emb_dim**0.5) # [B', T, T]
        attn_mat = torch.matmul(Q, K) / (emb_dim**0.5) # [B', T, T]
        attn_mat = F.softmax(attn_mat, dim=2) # [B', T, T]
        V = torch.matmul(attn_mat, V) # [B', T, C*Q]


        V = V.reshape(old_shape) # [B', T, C, Q]
        V = V.transpose(1,2) # [B', C, T, Q]
        emb_dim = V.shape[1]

        batch = V.contiguous().view([B,self.n_head*emb_dim,old_T,old_Q]) # [B, C, T, Q])
        #batch = self['attn_concat_proj'](batch) # [B, C, T, Q])
        batch = self.attn_concat_proj(batch) # [B, C, T, Q])

        out = batch + inter_rnn
        return out




#完整分离模型 
class DPRNNTF(nn.Module):

    #def __getitem__(self, key):
    #    return getattr(self, key)

    def __init__(self,
            n_srcs, n_freqs,
            n_imics, n_omics,
            seqmodel_type, n_layers, num_hid_units, emb_dim, emb_ks, emb_hs,
            masking_or_mapping, mask_nonlinearity, rmax,
            act_type='elu', norm_type='GlobalInstanceNormalization', convnormact_type=2,
            feature_type='ri',
            n_extra_feats=0,
            n_pred_components=2,
            first_conv_use_act=0,
            mag_sf=1.0,
            ):
        super(DPRNNTF, self).__init__()

        assert n_imics >= n_omics
        in_channels_list = [n_imics*n_pred_components] + [n_pred_components]*n_extra_feats


        #创建输入卷积
        t_ksize = 3
        ks, padding = (t_ksize,3), (t_ksize//2,1)
        for idx, in_channels in enumerate(in_channels_list):
            self.add_module('conv0-%d'%idx,
                    ConvBlock(in_channels,emb_dim,ks,padding=padding,
                        norm_type='GlobalLayerNormalization',act_type='None' if first_conv_use_act == 0 else act_type,convnormact_type=1))

        self.dual_rnn = nn.ModuleList([])
        if seqmodel_type in [3860, 3861,]:
            # fmseqattn34U3NmUfdLSTMViTLNallAllHeadFast
            for ii in range(n_layers):
                self.dual_rnn.append(
                        FrameSeqAttn3U3NmUfdDualRNNBlockTFEmbViTLNFast(emb_dim, emb_ks, emb_hs, n_freqs, num_hid_units,
                            n_head=4, attn_mode=41, approx_qk_dim=512 if seqmodel_type == 3860 else 1024,
                            rnn_type='LSTM', act_type=act_type, norm_type=norm_type, convnormact_type=convnormact_type))
        else:
            raise


        #创建输出反卷积
        self.add_module('deconv0',
                ConvBlock(emb_dim,(n_srcs)*n_omics*n_pred_components,ks,
                    padding=padding,use_deconv=1,norm_type='None',act_type='None',convnormact_type=1))

        self.n_srcs = n_srcs
        self.n_freqs = n_freqs
        self.n_imics = n_imics
        self.n_omics = n_omics
        self.n_pred_components = n_pred_components
        self.seqmodel_type = seqmodel_type
        self.n_layers = n_layers
        self.num_hid_units = num_hid_units
        self.masking_or_mapping = masking_or_mapping
        self.mask_nonlinearity = mask_nonlinearity
        self.rmax = rmax
        self.t_ksize = t_ksize
        self.act_type = act_type
        self.norm_type = norm_type
        self.convnormact_type = convnormact_type
        self.n_extra_feats = n_extra_feats

        self.mag_sf = mag_sf

    """
    拿到混合音频的STFT的实虚部→整理形状→卷积升维→多层dualRNN建模→反卷积输出→生成mask或直接映射→得到各个生源的复数时频谱估计
    """
    def forward(self, batch, target_dict={}, current_epoch=None):
        # batch is [n_batch, n_pad_frames, 1, n_imics+n_extra_feats, n_pred_components, n_freqs]
        #拿混合音频
        mix_ri = target_dict['mix_ri'][0] # [n_batch, n_pad_frames, 1, n_imics, n_pred_components, n_freqs]

        #限制并选择麦克风
        assert self.n_imics in [1, 8]
        mix_ri = mix_ri[:,:,:,:self.n_imics]
        n_batch, N = mix_ri.shape[:2]
        batch = mix_ri.contiguous()
        #对输入STFT的幅度进行数值尺度调整，让网络更容易训练
        if hasattr(self, 'mag_sf') and self.mag_sf != 1.0:
            batch_cs = torch.complex(batch[...,0,:], batch[...,1,:])
            batch_r, batch_i, _ = scaling_magnitude(batch_cs, self.mag_sf, eps=1e-5)
            batch = torch.stack([batch_r, batch_i], dim=-2)

        #合并多余维度
        batch = batch.view([n_batch,N,-1,self.n_pred_components*self.n_freqs])

        if self.n_extra_feats == 0:
            batch = batch.view([n_batch,N,-1,self.n_freqs])
            batch = batch.transpose(1,2) # [n_batch, -1, N, n_freqs]
            #batch = self['conv0-0'](batch)
            batch = getattr(self,'conv0-0')(batch)
        else: #如果还有额外特征
            feats = [ batch ] + target_dict['extra_feats'] # [n_batch, N, -1, n_pred_components*n_freqs], and each [n_batch, N, 1, n_srcs, n_pred_components, n_freqs]
            batch = []
            for idx, feat in enumerate(feats): #每一种特征都reshape，调整顺序，全部都投影到相同的emb_dim
                feat = feat.reshape([n_batch,N,-1,self.n_freqs])
                feat = feat.transpose(1,2) # [n_batch, -1, N, n_freqs]
                batch.append(getattr(self,'conv0-%d'%idx)(feat))
            batch = torch.sum(torch.stack(batch,dim=0),dim=0) #把不同来源的embedding加起来


        """
        核心：经过多个Dual-RNN block  一个block就包括上面的 intra Rnn  inter Rnn  attention
        [B,emb_dim,N,F]里面的特征值在不断改变，feature越来越包含“那个声源属于谁”的信息
        """
        for i in range(self.n_layers):
            batch = self.dual_rnn[i](batch)


        #把高维转换成最终输出
        batch = self.deconv0(batch)
        # batch is [n_batch, -1, N, n_freqs]

        batch = batch.transpose(1,2) # [n_batch, N, -1, n_freqs]
        batch = batch.view([n_batch, N, self.n_srcs, self.n_omics, -1, self.n_freqs]) # -1 = self.n_pred_components + (0 or 1)

        

        """
        Masking:模型预测 M,然后目标声源s≈M✖️混合音频
        X = mixture STFT
        M = 网络预测的 mask，这里的mask其实就是生成的real和imag，但是是混合语音的
        Ŝ = 估计声源 STFT
        """
        """
        mapping是直接生成目标语音的real和imag
        """
        masking_or_mapping = self.masking_or_mapping
        if masking_or_mapping == 0:
            # masking
            if self.mask_nonlinearity == 'linear':
                batch = torch.clamp(batch,-self.rmax,self.rmax)
            elif self.mask_nonlinearity == 'tanh':
                batch = torch.tanh(batch)*self.rmax
            elif self.mask_nonlinearity == 'linearNoT':
                pass    
            else:
                raise
        elif masking_or_mapping == 1:
            # mapping
            if self.mask_nonlinearity == 'linear':
                pass
            else:
                raise
        else:
            raise

        est_ri = batch
        # est_ri.shape is [n_batch, N, n_srcs, n_omics, n_pred_components, n_freqs]
        # mix_ri.shape is [n_batch, N,      1, n_imics, n_pred_components, n_freqs]
        if masking_or_mapping == 0:
            assert self.n_omics == 1
            #把复数Mask拆成实部和虚部
            est_r, est_i = est_ri[...,0,:], est_ri[...,1,:]         # each [n_batch, n_frames, n_srcs, 1, n_freqs]
            mix_r, mix_i = mix_ri[...,:1,0,:], mix_ri[...,:1,1,:]   # each [n_batch, n_frames,      1, 1, n_freqs]
            #复数乘法
            est_r, est_i = complex_multiply((est_r,est_i), (mix_r,mix_i))
            est_ri = torch.stack([est_r, est_i], dim=-2)            #      [n_batch, n_frames, n_srcs, 1, 2, n_freqs]
        elif masking_or_mapping == 1:
            pass
        else:
            raise

        return est_ri
    """
    ① 输入准备
mix_ri
[B,N,1,Mic,2,F]

        ↓

② Feature Embedding
reshape
transpose
ConvBlock

        ↓

[B,emb_dim,N,F]

        ↓

③ TF-GridNet 主干
Dual-RNN × n_layers

Intra RNN → 频率关系
Inter RNN → 时间关系
Attention → 全局关系

        ↓

④ 输出映射
deconv0

        ↓

[B,N,n_srcs,1,2,F]

        ↓

⑤ 声源重建
如果 masking：

预测 Complex Mask
       ×
Mixture STFT
       ↓
Estimated Source STFT

        ↓

return est_ri
    """
