import collections.abc as container_abcs
import torch
import torch.nn.utils.rnn as rnn
import numpy as np

"""
将dataset返回的多个单样本递归组合成一个batch，并对变长Numpy数组补0，同时保存每条数据的原始长度
"""

#pad_back_or_front=0 默认在序列尾部补0
def collate_seq(batch, pad_back_or_front=0):
    elem = batch[0]
    elem_type = type(elem)
    """
    根据数据类型分别处理：
    字符串：整理成字符串列表。
    NumPy 数组：转成 float32 Tensor、补零，并返回 (padded_tensor, lengths)。
    元组或列表：逐位置递归处理。
    字典：逐字段递归处理。
    其他数据：直接保留为列表
    model_input = {
    "logmel":      (Tensor[B, T_max, 64], lengths[B]),
    "weak_labels": (Tensor[B, 527],       lengths[B]),
}

target_dict = {
    "strong_labels": (Tensor[B, T_max], lengths[B]),
    "strong_mask":   (Tensor[B, T_max], lengths[B]),
    "frame_times":   (Tensor[B, T_max], lengths[B]),
    "class_index":   [类别编号, ...],
    "utt_id":        [音频ID, ...],
    "audio_path":    [音频路径, ...],
}

[
    {
        'mixture_waveform': (mixture_tensor, mixture_lengths),
        'condition': (condition_tensor, condition_lengths),
    },
    {
        'target_waveform': (target_tensor, target_lengths),
        # 其他字段
    }
]
    """
    if elem_type.__name__ == 'str':
        return [b for b in batch]
    elif elem_type.__name__ == 'ndarray':
        lengths = torch.tensor([len(b) for b in batch])
        if pad_back_or_front == 0 or len(batch) == 1:
            # padding back
            pad_features = rnn.pad_sequence([torch.tensor(b, dtype=torch.float32) for b in batch],
                                            batch_first=True)
        elif pad_back_or_front == 1:
            # padding front
            pad_features = torch.flip(
                    rnn.pad_sequence(
                        [torch.flip(torch.tensor(b, dtype=torch.float32),dims=[0,]) for b in batch],
                        batch_first=True),
                    dims=[1,],
                    )
        else:
            raise
        return pad_features, lengths
    elif isinstance(elem, container_abcs.Sequence):
        transposed = zip(*batch)
        return [
            collate_seq(samples, pad_back_or_front=pad_back_or_front)
            for samples in transposed
        ]
    elif isinstance(elem, container_abcs.Mapping):
        return {
            key: collate_seq(
                [d[key] for d in batch],
                pad_back_or_front=pad_back_or_front,
            )
            for key in elem
        }
    else:
        # not collate
        return [b for b in batch]
